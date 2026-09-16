import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from ultralytics import YOLO


VIDEO_PATH = 'q.mp4'
OUTPUT_NAME = 'road_tracking_stabilized.mp4'


def draw_guaranteed_markers(frame, left_rail_x, right_rail_x, y_start, y_end, y_shift):
    """Always draw visible lane markers as a safety fallback."""
    height, width = frame.shape[:2]
    left_x = int(np.clip(left_rail_x, 0, width - 1))
    right_x = int(np.clip(right_rail_x, 0, width - 1))
    lower_y0 = max(0, y_start)
    lower_y1 = min(height, y_end)
    upper_y0 = max(0, lower_y0 - y_shift)
    upper_y1 = min(height, lower_y1 - y_shift)

    cv2.line(frame, (left_x, lower_y0), (left_x, lower_y1), (0, 255, 0), 2)
    cv2.line(frame, (right_x, lower_y0), (right_x, lower_y1), (0, 255, 0), 2)
    cv2.line(frame, (left_x, upper_y0), (left_x, upper_y1), (0, 165, 255), 2)
    cv2.line(frame, (right_x, upper_y0), (right_x, upper_y1), (0, 165, 255), 2)
    return frame


if not os.path.exists(VIDEO_PATH):
    print(f"❌ CRITICAL ERROR: Could not find '{VIDEO_PATH}' in your workspace.")
else:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.set_grad_enabled(False)

    gpu_mem_mb = 0
    if device.type == 'cuda':
        gpu_mem_mb = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)

    if device.type == 'cuda' and gpu_mem_mb <= 4096:
        BATCH_SIZE = 4
        TARGET_IMG_SIZE = 512
        MEMORY_DEPTH = 3
        CONF_THRESHOLD = 0.20
        print('⚙️ GTX 1650-class hardware detected: using a smaller batch and lower inference size for stability.')
    else:
        BATCH_SIZE = 8
        TARGET_IMG_SIZE = 640
        MEMORY_DEPTH = 5
        CONF_THRESHOLD = 0.15

    print('🔥 LAUNCHING STABILIZED CITY-TRAFFIC ENGINE...')
    model = YOLO('yolo26n-sem.pt').to(device)
    model.eval()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"❌ CRITICAL ERROR: Could not open '{VIDEO_PATH}'.")
    else:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if width <= 0 or height <= 0 or fps <= 0:
            cap.release()
            print('❌ CRITICAL ERROR: Invalid video dimensions or frame rate.')
        else:
            y_start = int(height * 0.10)
            y_end = int(height * 0.60)
            crop_height = y_end - y_start
            panoramic_width = width * 3

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(OUTPUT_NAME, fourcc, fps, (panoramic_width, height))
            if not out.isOpened():
                cap.release()
                print(f"❌ CRITICAL ERROR: Could not create '{OUTPUT_NAME}'.")
            else:
                mid_x = width // 2
                offset_10pct = int(width * 0.10)
                left_rail_x = mid_x - offset_10pct
                right_rail_x = mid_x + offset_10pct
                y_shift = int(crop_height * 0.10)

                mask_history = deque(maxlen=MEMORY_DEPTH)
                frame_count = 0

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

                    tensor_batch = np.stack(cropped_frames)
                    tensor_batch = torch.from_numpy(tensor_batch).to(device).permute(0, 3, 1, 2).float() / 255.0
                    tensor_batch = F.interpolate(
                        tensor_batch,
                        size=(TARGET_IMG_SIZE, TARGET_IMG_SIZE),
                        mode='bilinear',
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
                            quantize='fp16' if device.type == 'cuda' else None,
                        )

                    for idx in range(len(frames_list)):
                        result = results[idx]
                        frame = frames_list[idx]

                        stage_a_full = frame.copy()
                        stage_b_full = np.zeros_like(frame)
                        stage_c_full = frame.copy()

                        resized_mask = None
                        if hasattr(result, 'semantic_mask') and result.semantic_mask is not None:
                            raw_mask = result.semantic_mask.data.cpu().numpy()
                            if raw_mask.ndim == 3:
                                raw_mask = raw_mask[0]
                            resized_mask = cv2.resize(
                                (raw_mask == 0).astype(np.uint8),
                                (width, crop_height),
                                interpolation=cv2.INTER_NEAREST,
                            )
                        elif hasattr(result, 'masks') and result.masks is not None and len(result.masks.data) > 0:
                            try:
                                boxes = getattr(result, 'boxes', None)
                                if boxes is not None:
                                    class_ids = boxes.cls.detach().cpu().numpy().astype(np.int64)
                                    mask_data = result.masks.data.detach().cpu().numpy()
                                    road_mask = np.any(mask_data[class_ids == 0], axis=0).astype(np.uint8)
                                    resized_mask = cv2.resize(
                                        road_mask,
                                        (width, crop_height),
                                        interpolation=cv2.INTER_NEAREST,
                                    )
                            except Exception:
                                resized_mask = None

                        if resized_mask is not None:
                            mask_history.append(resized_mask)
                            smoothed_mask = np.mean(mask_history, axis=0)
                            road_mask_final = np.zeros_like(smoothed_mask, dtype=np.uint8)
                            road_mask_final[smoothed_mask > 0.4] = 255

                            road_pixels = road_mask_final
                            stage_b_full[y_start:y_end, :, 0] = road_pixels
                            stage_b_full[y_start:y_end, :, 1] = road_pixels
                            stage_b_full[y_start:y_end, :, 2] = road_pixels

                            edges = cv2.Canny(road_pixels, 100, 200)
                            edges[:, left_rail_x:right_rail_x] = 0
                            edge_y, edge_x = np.where(edges > 0)

                            if len(edge_x) > 0:
                                edge_y_full = edge_y + y_start
                                floor_mask = edge_y_full < (y_end - 3)
                                valid_x = edge_x[floor_mask]
                                valid_y = edge_y_full[floor_mask]
                                if len(valid_x) > 0:
                                    stage_c_full[valid_y, valid_x] = (0, 255, 0)
                                    orange_y = np.clip(valid_y - y_shift, 0, height - 1)
                                    stage_c_full[orange_y, valid_x] = (0, 165, 255)
                            else:
                                stage_c_full = draw_guaranteed_markers(
                                    stage_c_full, left_rail_x, right_rail_x, y_start, y_end, y_shift
                                )
                        else:
                            stage_c_full = draw_guaranteed_markers(
                                stage_c_full, left_rail_x, right_rail_x, y_start, y_end, y_shift
                            )

                        combined = np.hstack((stage_a_full, stage_b_full, stage_c_full))
                        out.write(combined)
                        frame_count += 1

                    if device.type == 'cuda':
                        torch.cuda.empty_cache()

                    if frame_count % 100 == 0 or frame_count >= total_frames:
                        print(f'⚡ Memory Engine: Stabilized & Saved {frame_count}/{total_frames} frames...')

                cap.release()
                out.release()
                print(f"\n🎉 SUCCESS! Stabilized city-traffic video built as: '{OUTPUT_NAME}'")
