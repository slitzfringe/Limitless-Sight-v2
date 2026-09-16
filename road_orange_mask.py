from collections import deque
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO


# ---------------------------------------------------------------------------
# Stable base road detector.
# This is the clean baseline version. Experimental visual styles should live in
# separate files so each improvement can be reviewed and approved separately.
# ---------------------------------------------------------------------------


def extract_road_mask(result, width, crop_height):
    """Return binary road mask from semantic or instance segmentation output."""
    semantic_mask = getattr(result, "semantic_mask", None)
    if semantic_mask is not None:
        raw_mask = semantic_mask.data.detach().cpu().numpy()
        if raw_mask.ndim == 3:
            raw_mask = raw_mask[0]
        road_mask = (raw_mask == 0).astype(np.uint8)
        return cv2.resize(road_mask, (width, crop_height), interpolation=cv2.INTER_NEAREST)

    instance_masks = getattr(result, "masks", None)
    boxes = getattr(result, "boxes", None)
    if instance_masks is None or boxes is None or len(instance_masks.data) == 0:
        return None

    class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
    mask_data = instance_masks.data.detach().cpu().numpy()
    road_mask = np.any(mask_data[class_ids == 0], axis=0).astype(np.uint8)
    return cv2.resize(road_mask, (width, crop_height), interpolation=cv2.INTER_NEAREST)


def clean_road_mask(mask_img):
    """Keep the dominant road mask and clear small gaps caused by occlusions."""
    binary = (mask_img > 0).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    flood = binary.copy()
    flood_mask = np.zeros((binary.shape[0] + 2, binary.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    binary = cv2.bitwise_or(binary, holes)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    if num_labels <= 1:
        return binary

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == largest, 255, 0).astype(np.uint8)


def draw_road_fill(frame, road_mask, y_start, y_end, alpha=0.22):
    """Apply a gentle green tint over the detected road region."""
    output = frame.copy()
    mask_resized = cv2.resize(road_mask, (frame.shape[1], y_end - y_start), interpolation=cv2.INTER_NEAREST)
    mask_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    mask_3ch = mask_3ch.astype(np.float32) / 255.0

    roi = output[y_start:y_end, :, :].astype(np.float32)
    green_overlay = np.zeros_like(roi)
    green_overlay[:, :, 0] = 0
    green_overlay[:, :, 1] = 180
    green_overlay[:, :, 2] = 0

    masked = roi * (1.0 - alpha * mask_3ch) + green_overlay * (alpha * mask_3ch)
    output[y_start:y_end, :, :] = masked.astype(np.uint8)
    return output


def build_comparison_frame(original, masked, final):
    """Build a three-panel output: original, segmented mask, final result."""
    height, width = original.shape[:2]
    panel_width = width
    combined = np.zeros((height, panel_width * 3, 3), dtype=np.uint8)
    combined[:, :panel_width] = original
    combined[:, panel_width:panel_width * 2] = masked
    combined[:, panel_width * 2:] = final
    return combined


def main():
    base_dir = Path(__file__).resolve().parent
    video_path = base_dir / "q.mp4"
    model_path = base_dir / "yolo26n-sem.pt"
    output_path = base_dir / "road_orange_mask.mp4"

    if not video_path.exists():
        print(f"CRITICAL ERROR: missing {video_path.name}")
        return 1
    if not model_path.exists():
        print(f"CRITICAL ERROR: missing {model_path.name}")
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats(device)

    model = YOLO(str(model_path)).to(device)
    model.eval()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"CRITICAL ERROR: could not open {video_path.name}")
        return 1

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if width <= 0 or height <= 0 or fps <= 0:
        cap.release()
        print("CRITICAL ERROR: invalid video dimensions or fps")
        return 1

    y_start = int(height * 0.12)
    y_end = height
    crop_height = y_end - y_start
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width * 3, height))
    if not out.isOpened():
        cap.release()
        print(f"CRITICAL ERROR: could not create {output_path.name}")
        return 1

    batch_size = 4
    target_size = 512
    memory_depth = 3
    conf = 0.2

    mask_history = deque(maxlen=memory_depth)
    frame_count = 0
    start = perf_counter()

    while cap.isOpened():
        frames = []
        cropped = []

        for _ in range(batch_size):
            success, raw = cap.read()
            if not success or raw is None:
                break
            frames.append(raw)
            cropped.append(raw[y_start:y_end, :])

        if not frames:
            break

        tensor_batch = np.stack(cropped)
        tensor_batch = torch.from_numpy(tensor_batch).to(device).permute(0, 3, 1, 2).float() / 255.0
        tensor_batch = F.interpolate(
            tensor_batch,
            size=(target_size, target_size),
            mode="bilinear",
            align_corners=False,
        )

        with torch.inference_mode():
            results = model.predict(
                source=tensor_batch,
                conf=conf,
                save=False,
                verbose=False,
                imgsz=target_size,
                device=device,
                quantize="fp16" if device.type == "cuda" else None,
            )

        for idx in range(len(frames)):
            result = results[idx]
            frame = frames[idx]
            output_frame = frame.copy()

            road_mask = extract_road_mask(result, width, crop_height)
            if road_mask is not None:
                mask_history.append(road_mask)
                smoothed_mask = np.mean(mask_history, axis=0) if len(mask_history) > 1 else road_mask.astype(np.float32)
                road_threshold = np.zeros_like(smoothed_mask, dtype=np.uint8)
                road_threshold[smoothed_mask > 0.35] = 255
                road_mask_clean = clean_road_mask(road_threshold)

                mask_preview = frame.copy()
                mask_roi = mask_preview[y_start:y_end, :].copy()
                mask_roi[road_mask_clean > 0] = [0, 255, 0]
                mask_preview[y_start:y_end, :] = mask_roi

                output_frame = draw_road_fill(output_frame, road_mask_clean, y_start, y_end, alpha=0.22)

            comparison_frame = build_comparison_frame(frame, mask_preview, output_frame)
            out.write(comparison_frame)
            frame_count += 1

            if frame_count % 100 == 0 or frame_count >= total_frames:
                print(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    out.release()
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Peak GPU memory allocated: {peak_mb:.0f} MB")

    elapsed = perf_counter() - start
    print(f"Processing time: {elapsed:.1f}s")
    print(f"SUCCESS: {output_path.name} created")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
