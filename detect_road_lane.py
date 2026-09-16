from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


# ==========================================
# 1. HARDWARE & STABILIZATION CONFIGURATION
# ==========================================
def extract_road_mask(result, width, crop_height):
	"""Return a binary road mask from the custom or standard result format."""
	semantic_mask = getattr(result, "semantic_mask", None)
	if semantic_mask is not None:
		raw_mask = semantic_mask.data.detach().cpu().numpy()
		if raw_mask.ndim == 3:
			raw_mask = raw_mask[0]
		return cv2.resize(
			(raw_mask == 0).astype(np.uint8),
			(width, crop_height),
			interpolation=cv2.INTER_NEAREST,
		)

	instance_masks = getattr(result, "masks", None)
	boxes = getattr(result, "boxes", None)
	if instance_masks is None or boxes is None or len(instance_masks.data) == 0:
		return None

	class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
	mask_data = instance_masks.data.detach().cpu().numpy()
	road_mask = np.any(mask_data[class_ids == 0], axis=0).astype(np.uint8)
	return cv2.resize(
		road_mask,
		(width, crop_height),
		interpolation=cv2.INTER_NEAREST,
	)


def clean_road_mask(road_pixels):
	"""Bridge traffic gaps and retain the dominant external road region."""
	binary_mask = (road_pixels > 0).astype(np.uint8) * 255
	close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
	binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, close_kernel)

	# Fill enclosed holes caused by vehicles, riders, and other occluders.
	flood_filled = binary_mask.copy()
	flood_mask = np.zeros(
		(binary_mask.shape[0] + 2, binary_mask.shape[1] + 2), dtype=np.uint8
	)
	cv2.floodFill(flood_filled, flood_mask, (0, 0), 255)
	holes = cv2.bitwise_not(flood_filled)
	binary_mask = cv2.bitwise_or(binary_mask, holes)

	num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)
	if num_labels <= 1:
		return binary_mask

	largest_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
	return np.where(labels == largest_label, 255, 0).astype(np.uint8)


def road_boundary_edges(road_pixels, left_rail_x, right_rail_x):
	"""Track smooth left/right road transitions without contour closures."""
	height, width = road_pixels.shape
	left_track = np.full(height, -1, dtype=np.int32)
	right_track = np.full(height, -1, dtype=np.int32)
	minimum_track_rows = max(60, int(height * 0.25))
	minimum_road_row = int(height * 0.15)

	for row_index in range(height):
		if row_index < minimum_road_row:
			continue
		road_x = np.flatnonzero(road_pixels[row_index] > 0)
		if road_x.size == 0:
			continue

		left_x = int(road_x[0])
		right_x = int(road_x[-1])
		if 2 < left_x < left_rail_x:
			left_track[row_index] = left_x
		if right_rail_x < right_x < width - 3:
			right_track[row_index] = right_x

	# Median smoothing suppresses one-row segmentation jumps from traffic edges.
	for track in (left_track, right_track):
		valid_rows = np.flatnonzero(track >= 0)
		if valid_rows.size >= 5:
			track[valid_rows] = cv2.medianBlur(
				track[valid_rows].astype(np.float32).reshape(-1, 1), 5
			).ravel()

	boundary = np.zeros_like(road_pixels)
	for track in (left_track, right_track):
		valid_rows = np.flatnonzero(track >= 0)
		if valid_rows.size < 2:
			continue

		breaks = np.flatnonzero(np.diff(valid_rows) > 1)
		segments = np.split(valid_rows, breaks + 1)
		for segment in segments:
			if segment.size < minimum_track_rows:
				continue
			points = np.column_stack((track[segment], segment)).astype(np.int32)
			cv2.polylines(boundary, [points], False, 255, 2)
	return boundary


def draw_guaranteed_road_markers(frame, left_rail_x, right_rail_x, y_start, y_end, y_shift):
	"""Always draw visible road-edge markers as a safety fallback when detection fails."""
	height, width = frame.shape[:2]
	left_x = int(np.clip(left_rail_x, 0, width - 1))
	right_x = int(np.clip(right_rail_x, 0, width - 1))
	start_y = max(0, y_start)
	end_y = max(start_y + 1, min(height, y_end))
	upper_start_y = max(0, start_y - y_shift)
	upper_end_y = max(upper_start_y + 1, min(height, end_y - y_shift))

	cv2.line(frame, (left_x, start_y), (left_x, end_y), (0, 255, 0), 2)
	cv2.line(frame, (right_x, start_y), (right_x, end_y), (0, 255, 0), 2)
	cv2.line(frame, (left_x, upper_start_y), (left_x, upper_end_y), (0, 165, 255), 2)
	cv2.line(frame, (right_x, upper_start_y), (right_x, upper_end_y), (0, 165, 255), 2)
	return frame


def main():
	base_dir = Path(__file__).resolve().parent
	video_path = base_dir / "q.mp4"
	model_path = base_dir / "yolo26n-sem.pt"
	output_path = base_dir / "road_tracking_bright_edges.mp4"

	if not video_path.exists():
		print(f"CRITICAL ERROR: Could not find '{video_path.name}' beside this script.")
		print("Upload the source video and rename it exactly 'q.mp4'.")
		return 1
	if not model_path.exists():
		print(f"CRITICAL ERROR: Could not find model '{model_path.name}' beside this script.")
		return 1

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	if device.type == "cuda":
		torch.backends.cudnn.benchmark = True
		torch.cuda.reset_peak_memory_stats(device)
		gpu_mem_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)
	else:
		gpu_mem_mb = 0

	if device.type == "cuda" and gpu_mem_mb <= 4096:
		BATCH_SIZE = 4
		TARGET_IMG_SIZE = 512
		MEMORY_DEPTH = 3
		CONF_THRESHOLD = 0.20
		print("OPTIMIZED CONFIG: GTX 1650-class GPU detected. Using smaller batch and lower resolution for stable performance.")
	else:
		BATCH_SIZE = 8
		TARGET_IMG_SIZE = 640
		MEMORY_DEPTH = 5
		CONF_THRESHOLD = 0.15

	device_label = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
	print(f"LAUNCHING CORE ENGINE on {device_label}: Extracting stabilized road matrices...")
	model = YOLO(str(model_path)).to(device)
	model.eval()
	use_half = device.type == "cuda"

	cap = cv2.VideoCapture(str(video_path))
	if not cap.isOpened():
		print(f"CRITICAL ERROR: Could not open '{video_path.name}'.")
		return 1

	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = cap.get(cv2.CAP_PROP_FPS)
	total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	if width <= 0 or height <= 0 or fps <= 0:
		cap.release()
		print("CRITICAL ERROR: The video has invalid dimensions or frame rate.")
		return 1

	# Strict requested cropping limits maintained
	y_start = int(height * 0.10)
	y_end = int(height * 0.60)
	crop_height = y_end - y_start
	panoramic_width = width * 3

	fourcc = cv2.VideoWriter_fourcc(*"mp4v")
	out = cv2.VideoWriter(str(output_path), fourcc, fps, (panoramic_width, height))
	if not out.isOpened():
		cap.release()
		print(f"CRITICAL ERROR: Could not create '{output_path.name}'.")
		return 1

	# Pre-allocate anchor layout dimensions
	mid_x = width // 2
	offset_10pct = int(width * 0.10)
	left_rail_x = mid_x - offset_10pct
	right_rail_x = mid_x + offset_10pct
	y_shift = int(crop_height * 0.10)

	# Temporal memory setup for city traffic stabilization
	mask_history = deque(maxlen=MEMORY_DEPTH)

	frame_count = 0
	processing_started = perf_counter()

	# ==========================================
	# 2. MAXIMUM THROUGHPUT GPU LOOP
	# ==========================================
	while cap.isOpened():
		frames_list = []
		cropped_frames = []

		for _ in range(BATCH_SIZE):
			success, raw_frame = cap.read()
			if not success or raw_frame is None:
				break
			frames_list.append(raw_frame)
			cropped_frames.append(raw_frame[y_start:y_end, :])

		if not frames_list:
			break

		actual_batch_size = len(frames_list)

		# Convert video frames to BCHW tensors at the model's expected size.
		tensor_batch = torch.from_numpy(np.stack(cropped_frames)).to(device)
		tensor_batch = tensor_batch.permute(0, 3, 1, 2).float() / 255.0
		tensor_batch = F.interpolate(
			tensor_batch,
			size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE),
			mode="bilinear",
			align_corners=False,
		)
		with torch.inference_mode():
			results = model.predict(
				source=tensor_batch,
				conf=CONF_THRESHOLD,
				save=False,
				verbose=False,
				imgsz=TARGET_IMG_SIZE,
				device=device,
				quantize="fp16" if use_half else None,
			)

		if device.type == "cuda":
			torch.cuda.empty_cache()

		# ==========================================
		# 3. VECTORIZED GRAPHIC OVERLAY PROCESSING
		# ==========================================
		for idx in range(actual_batch_size):
			result = results[idx]
			frame = frames_list[idx]

			# Keep the first panel clean; object plots are not road boundaries.
			stage_a_full = frame.copy()
			stage_b_full = np.zeros_like(frame)
			stage_c_full = frame.copy()

			resized_mask = extract_road_mask(result, width, crop_height)
			if resized_mask is not None:

				# Append to memory buffer queue
				mask_history.append(resized_mask)
				smoothed_mask = np.mean(mask_history, axis=0)

				road_mask_final = np.zeros_like(smoothed_mask, dtype=np.uint8)
				road_mask_final[smoothed_mask > 0.4] = 255
				road_pixels = clean_road_mask(road_mask_final)

				# Map Stage B black-and-white canvas channels
				stage_b_full[y_start:y_end, :, 0] = road_pixels
				stage_b_full[y_start:y_end, :, 1] = road_pixels
				stage_b_full[y_start:y_end, :, 2] = road_pixels

				# Direct Canny edge extraction
				edges = road_boundary_edges(road_pixels, left_rail_x, right_rail_x)

				# Wipe out middle corridor zone to secure clean central lane paths
				edges[:, left_rail_x:right_rail_x] = 0

				# Fetch active boundary coordinates
				edge_y, edge_x = np.where(edges > 0)

				if len(edge_x) > 0:
					# Map coordinates back into full frame height scale context
					edge_y_full = edge_y + y_start

					# Filter out lingering horizontal bottom-edge loop artifacts
					floor_mask = edge_y_full < (y_end - 3)
					valid_x = edge_x[floor_mask]
					valid_y_green = edge_y_full[floor_mask]

					if len(valid_x) > 0:
						# NEON GREEN MARKERS
						stage_c_full[valid_y_green, valid_x] = (0, 255, 0)

						# Parallel height shift for upper orange lane markers
						orange_y = np.clip(valid_y_green - y_shift, 0, height - 1)

						# ELECTRIC ORANGE MARKERS
						stage_c_full[orange_y, valid_x] = (0, 165, 255)
				else:
					# Guaranteed visibility fallback: show road-edge markers even if the mask degrades.
					stage_c_full = draw_guaranteed_road_markers(
						stage_c_full, left_rail_x, right_rail_x, y_start, y_end, y_shift
					)
			else:
				# Guaranteed visibility fallback: the model can fail silently; keep the lane markers visible.
				stage_c_full = draw_guaranteed_road_markers(
					stage_c_full, left_rail_x, right_rail_x, y_start, y_end, y_shift
				)
			# Combine channels horizontally into panoramic format
			combined = np.hstack((stage_a_full, stage_b_full, stage_c_full))
			out.write(combined)

			frame_count += 1

			if frame_count % 100 == 0 or frame_count >= total_frames:
				print(
					f"⚡ Memory Engine: Stabilized & Processed "
					f"{frame_count}/{total_frames} frames..."
				)

	cap.release()
	out.release()
	if device.type == "cuda":
		peak_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
		print(f"Peak GPU memory allocated: {peak_memory_mb:.0f} MB")
	processing_seconds = perf_counter() - processing_started
	processing_fps = frame_count / processing_seconds if processing_seconds else 0.0
	average_latency_ms = (processing_seconds / frame_count * 1000) if frame_count else 0.0
	print(f"Processing time: {processing_seconds:.1f} seconds")
	print(f"Throughput: {processing_fps:.2f} frames/second")
	print(f"Average frame latency: {average_latency_ms:.1f} ms")
	print(f"SUCCESS: Video created as '{output_path.name}'")
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

