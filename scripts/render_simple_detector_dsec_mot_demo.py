#!/usr/bin/env python3
"""Render a DSEC-MOT video demo with SimpleDenseDetector predictions.

By default, detections and ground truth are drawn only in the event-camera
coordinate system. Use ``--draw-scaled-frame-boxes`` for a simple visual
640x480 scaling, or ``--draw-calibrated-frame-boxes`` to use the sequence
calibration with an assumed object depth. The calibrated overlay is still an
approximation because a 2D event-camera box has no per-object depth.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.data.pretrained_detection_dataset import ErosSnapshotStore
from src.data.representations import REPRESENTATION_CHOICES, BenchmarkRepresentation
from src.data.representations import representation_components
from src.evaluation.detection_export import (
    CLASS_NAMES,
    load_annotations,
    load_event_file,
    load_image_timestamps,
    read_events,
)
from src.utils.io import load_config
from src.models.simple_detector import (
    SimpleDenseDetector,
    SimpleDetectorConfig,
    decode_dense_detections,
    normalise_representation_tensor,
)


TOP_BAR = 56
GAP = 16
PRED_COLOR = (0, 165, 255)
GT_COLOR = (0, 255, 0)


def group_annotations_by_timestamp(annotations) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.timestamp, []).append(annotation)
    return grouped


def filter_annotations_by_class(annotations, class_ids: list[int] | None):
    if class_ids is None:
        return annotations
    allowed = set(class_ids)
    return [annotation for annotation in annotations if annotation.class_id in allowed]


def build_timestamp_to_png(seq_dir: Path, sequence: str) -> dict[int, Path]:
    timestamps = load_image_timestamps(seq_dir / f"{sequence}_image_timestamps.txt")
    png_paths = sorted((seq_dir / "images_rectified_left").glob("*.png"))
    if len(timestamps) != len(png_paths):
        raise SystemExit(
            f"Mismatch between timestamps ({len(timestamps)}) and PNGs ({len(png_paths)}) "
            f"in {seq_dir / 'images_rectified_left'}"
        )
    return dict(zip(timestamps, png_paths))


def events_to_bgr(events: np.ndarray) -> np.ndarray:
    frame = np.zeros((EVENT_HEIGHT, EVENT_WIDTH, 3), dtype=np.uint8)
    if events.size == 0:
        return frame
    x = events["x"].astype(np.int32)
    y = events["y"].astype(np.int32)
    p = events["p"].astype(bool)
    valid = (x >= 0) & (x < EVENT_WIDTH) & (y >= 0) & (y < EVENT_HEIGHT)
    x = x[valid]
    y = y[valid]
    p = p[valid]
    frame[y[p], x[p], 1] = 255
    frame[y[~p], x[~p], 2] = 255
    return frame


def eros_to_bgr(surface: np.ndarray | None) -> np.ndarray:
    if surface is None:
        return np.zeros((EVENT_HEIGHT, EVENT_WIDTH, 3), dtype=np.uint8)
    surface = np.asarray(surface)
    if surface.ndim == 3:
        surface = surface[0]
    max_value = float(surface.max()) if surface.size else 0.0
    if max_value <= 1.0:
        surface = surface * 255.0
    gray = np.clip(surface, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[SimpleDenseDetector, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleDenseDetector(SimpleDetectorConfig(**checkpoint["model_config"])).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, checkpoint


def camera_matrix(values: list[float]) -> np.ndarray:
    fx, fy, cx, cy = values
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def load_event_to_frame_projection(seq_dir: Path) -> dict[str, np.ndarray]:
    calibration = load_config(seq_dir / "calibration" / "cam_to_cam.yaml")
    intrinsics = calibration["intrinsics"]
    extrinsics = calibration["extrinsics"]
    t_10 = np.asarray(extrinsics["T_10"], dtype=np.float64)
    return {
        "event_k": camera_matrix(intrinsics["camRect0"]["camera_matrix"]),
        "frame_k": camera_matrix(intrinsics["camRect1"]["camera_matrix"]),
        "event_rect_r": np.asarray(extrinsics["R_rect0"], dtype=np.float64),
        "frame_rect_r": np.asarray(extrinsics["R_rect1"], dtype=np.float64),
        "r_10": t_10[:3, :3],
        "t_10": t_10[:3, 3],
    }


def project_event_points_to_frame(
    points_xy: np.ndarray,
    projection: dict[str, np.ndarray],
    depth_m: float,
) -> np.ndarray:
    homogeneous = np.column_stack([points_xy, np.ones(len(points_xy), dtype=np.float64)])
    event_rect = (np.linalg.inv(projection["event_k"]) @ homogeneous.T) * depth_m
    event_raw = projection["event_rect_r"].T @ event_rect
    frame_raw = projection["r_10"] @ event_raw + projection["t_10"][:, None]
    frame_rect = projection["frame_rect_r"] @ frame_raw
    frame_pixels = projection["frame_k"] @ frame_rect
    return (frame_pixels[:2] / np.maximum(frame_pixels[2:], 1e-6)).T


def project_event_box_to_frame(
    left: float,
    top: float,
    width: float,
    height: float,
    projection: dict[str, np.ndarray],
    depth_m: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int] | None:
    points = np.array(
        [
            [left, top],
            [left + width, top],
            [left + width, top + height],
            [left, top + height],
        ],
        dtype=np.float64,
    )
    projected = project_event_points_to_frame(points, projection, depth_m)
    if not np.isfinite(projected).all():
        return None
    x1, y1 = projected.min(axis=0)
    x2, y2 = projected.max(axis=0)
    x1_i = max(0, int(round(x1)))
    y1_i = max(0, int(round(y1)))
    x2_i = min(image_width - 1, int(round(x2)))
    y2_i = min(image_height - 1, int(round(y2)))
    if x2_i <= x1_i or y2_i <= y1_i:
        return None
    return x1_i, y1_i, x2_i, y2_i


def draw_ground_truth(frame_bgr: np.ndarray, annotations) -> np.ndarray:
    output = frame_bgr.copy()
    for annotation in annotations:
        left = max(0, int(round(annotation.left)))
        top = max(0, int(round(annotation.top)))
        right = min(EVENT_WIDTH - 1, int(round(annotation.left + annotation.width)))
        bottom = min(EVENT_HEIGHT - 1, int(round(annotation.top + annotation.height)))
        cv2.rectangle(output, (left, top), (right, bottom), GT_COLOR, 2)
        class_name = CLASS_NAMES.get(annotation.class_id, str(annotation.class_id))
        text = f"gt:{class_name} id={annotation.track_id}"
        text_y = max(14, top - 6)
        cv2.putText(
            output, text, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, GT_COLOR, 1, cv2.LINE_AA
        )
    return output


def draw_projected_ground_truth(
    frame_bgr: np.ndarray,
    annotations,
    projection: dict[str, np.ndarray],
    depth_m: float,
) -> np.ndarray:
    output = frame_bgr.copy()
    image_height, image_width = output.shape[:2]
    for annotation in annotations:
        box = project_event_box_to_frame(
            left=annotation.left,
            top=annotation.top,
            width=annotation.width,
            height=annotation.height,
            projection=projection,
            depth_m=depth_m,
            image_width=image_width,
            image_height=image_height,
        )
        if box is None:
            continue
        left, top, right, bottom = box
        cv2.rectangle(output, (left, top), (right, bottom), GT_COLOR, 4)
        class_name = CLASS_NAMES.get(annotation.class_id, str(annotation.class_id))
        text = f"gt:{class_name} id={annotation.track_id}"
        text_y = max(28, top - 10)
        cv2.putText(
            output, text, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, GT_COLOR, 2, cv2.LINE_AA
        )
    return output


def draw_predictions(frame_bgr: np.ndarray, detections) -> np.ndarray:
    output = frame_bgr.copy()
    for detection in detections:
        left = max(0, int(round(detection.bbox_left)))
        top = max(0, int(round(detection.bbox_top)))
        right = min(EVENT_WIDTH - 1, int(round(detection.bbox_left + detection.bbox_width)))
        bottom = min(EVENT_HEIGHT - 1, int(round(detection.bbox_top + detection.bbox_height)))
        cv2.rectangle(output, (left, top), (right, bottom), PRED_COLOR, 2)
        class_name = CLASS_NAMES.get(detection.class_id, str(detection.class_id))
        text = f"pred:{class_name} {detection.score:.2f}"
        text_y = min(EVENT_HEIGHT - 8, bottom + 14) if top < 16 else top - 6
        cv2.putText(
            output, text, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRED_COLOR, 1, cv2.LINE_AA
        )
    return output


def draw_projected_predictions(
    frame_bgr: np.ndarray,
    detections,
    projection: dict[str, np.ndarray],
    depth_m: float,
) -> np.ndarray:
    output = frame_bgr.copy()
    image_height, image_width = output.shape[:2]
    for detection in detections:
        box = project_event_box_to_frame(
            left=detection.bbox_left,
            top=detection.bbox_top,
            width=detection.bbox_width,
            height=detection.bbox_height,
            projection=projection,
            depth_m=depth_m,
            image_width=image_width,
            image_height=image_height,
        )
        if box is None:
            continue
        left, top, right, bottom = box
        cv2.rectangle(output, (left, top), (right, bottom), PRED_COLOR, 4)
        class_name = CLASS_NAMES.get(detection.class_id, str(detection.class_id))
        text = f"pred:{class_name} {detection.score:.2f}"
        text_y = min(image_height - 16, bottom + 30) if top < 32 else top - 10
        cv2.putText(
            output, text, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, PRED_COLOR, 2, cv2.LINE_AA
        )
    return output


def compose_frame(
    event_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    timestamp_us: int,
    pred_count: int,
    gt_count: int,
    label: str,
    frame_label: str,
) -> np.ndarray:
    right = cv2.resize(rectified_bgr, (EVENT_WIDTH, EVENT_HEIGHT), interpolation=cv2.INTER_AREA)
    canvas = np.full((EVENT_HEIGHT + TOP_BAR, EVENT_WIDTH * 2 + GAP, 3), 24, dtype=np.uint8)
    canvas[TOP_BAR:, :EVENT_WIDTH] = event_bgr
    canvas[TOP_BAR:, EVENT_WIDTH + GAP :] = right
    cv2.putText(
        canvas,
        "Event camera | ground truth: green | prediction: orange",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        frame_label,
        (EVENT_WIDTH + GAP + 8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"{label} | t={timestamp_us} us | pred={pred_count} | gt={gt_count}",
        (8, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequence", default="interlaken_00_d")
    parser.add_argument("--representation", choices=REPRESENTATION_CHOICES, default=None)
    parser.add_argument("--num-bins", type=int, default=None)
    parser.add_argument("--time-window-us", type=int, default=None)
    parser.add_argument(
        "--eros-cache-root",
        type=Path,
        default=None,
        help=(
            "Directory with precomputed EROS snapshots. Defaults to the path saved "
            "in the checkpoint, or data/cache/dsec_mot_eros."
        ),
    )
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-detections", type=int, default=50)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=200)
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--without-ground-truth", action="store_true")
    parser.add_argument(
        "--draw-scaled-frame-boxes",
        action="store_true",
        help=(
            "Also draw boxes on the rectified frame-camera panel after simple "
            "scaling to 640x480. This is an approximate visual overlay, not a "
            "calibrated projection."
        ),
    )
    parser.add_argument(
        "--draw-calibrated-frame-boxes",
        action="store_true",
        help=(
            "Also draw boxes on the rectified frame-camera panel using cam_to_cam.yaml "
            "and an assumed object depth. This is a calibration-aware approximation."
        ),
    )
    parser.add_argument(
        "--frame-box-depth-m",
        type=float,
        default=20.0,
        help="Assumed depth in meters for --draw-calibrated-frame-boxes.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but torch.cuda.is_available() is False.")
    if args.draw_scaled_frame_boxes and args.draw_calibrated_frame_boxes:
        raise SystemExit(
            "Use only one of --draw-scaled-frame-boxes or --draw-calibrated-frame-boxes."
        )
    if args.frame_box_depth_m <= 0:
        raise SystemExit("--frame-box-depth-m must be positive.")

    model, checkpoint = load_model(args.checkpoint, device)
    benchmark_config = checkpoint.get("benchmark_config", {})
    representation = args.representation or benchmark_config.get("representation", "voxel_grid")
    num_bins = (
        args.num_bins if args.num_bins is not None else int(benchmark_config.get("num_bins", 5))
    )
    time_window_us = (
        args.time_window_us
        if args.time_window_us is not None
        else int(benchmark_config.get("time_window_us", 50_000))
    )
    class_ids = benchmark_config.get("class_ids")
    class_ids = [int(class_id) for class_id in class_ids] if class_ids is not None else None
    eros_cache_root = args.eros_cache_root or Path(
        benchmark_config.get("eros_cache_root", "data/cache/dsec_mot_eros")
    )
    needs_eros = "eros" in representation_components(representation)
    eros_store = ErosSnapshotStore(eros_cache_root) if needs_eros else None
    transform = BenchmarkRepresentation(representation=representation, num_bins=num_bins)

    seq_dir = args.root / args.split / args.sequence
    annotation_path = args.root / "annotations" / args.split / f"{args.sequence}.txt"
    events_h5 = seq_dir / "events_left" / "events.h5"
    if not seq_dir.exists():
        raise SystemExit(f"Sequence directory does not exist: {seq_dir}")
    if not annotation_path.exists():
        raise SystemExit(f"Annotation file does not exist: {annotation_path}")
    if not events_h5.exists():
        raise SystemExit(f"Missing events file: {events_h5}")

    timestamps = load_image_timestamps(seq_dir / f"{args.sequence}_image_timestamps.txt")
    frame_entries = list(enumerate(timestamps))
    if args.start_frame:
        frame_entries = frame_entries[args.start_frame :]
    if args.max_frames > 0:
        frame_entries = frame_entries[: args.max_frames]
    if not frame_entries:
        raise SystemExit("No frames selected for rendering.")

    grouped = group_annotations_by_timestamp(load_annotations(annotation_path))
    png_map = build_timestamp_to_png(seq_dir, args.sequence)
    frame_projection = (
        load_event_to_frame_projection(seq_dir) if args.draw_calibrated_frame_boxes else None
    )
    output_path = args.output or (
        Path("data/processed/simple_detector_demos")
        / args.split
        / f"{args.sequence}_{representation}_thr{args.score_threshold:.2f}.avi"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (EVENT_WIDTH * 2 + GAP, EVENT_HEIGHT + TOP_BAR),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {output_path}")

    fusion_mode = checkpoint["model_config"].get("fusion_mode", "single")
    label = f"{fusion_mode} | {representation} | threshold={args.score_threshold:.2f}"
    handle, x, y, p, t, ms_to_idx, t_offset, _ = load_event_file(events_h5)
    try:
        total = len(frame_entries)
        for position, (frame_index, timestamp_us) in enumerate(frame_entries, start=1):
            events = read_events(x, y, p, t, ms_to_idx, t_offset, timestamp_us, time_window_us)
            eros = (
                eros_store.get(args.split, args.sequence, frame_index, timestamp_us)
                if eros_store is not None
                else None
            )
            dense = transform(events, eros=eros)
            tensor = torch.from_numpy(dense).float().unsqueeze(0).to(device, non_blocking=True)
            tensor = normalise_representation_tensor(tensor, model.config.component_channels)
            with torch.inference_mode():
                detections = decode_dense_detections(
                    outputs=model(tensor),
                    frame_index=frame_index,
                    timestamp=timestamp_us,
                    score_threshold=args.score_threshold,
                    nms_iou_threshold=args.nms_iou_threshold,
                    max_detections=args.max_detections,
                    feature_stride=model.config.feature_stride,
                )

            event_bgr = eros_to_bgr(eros) if representation == "eros" else events_to_bgr(events)
            annotations = filter_annotations_by_class(grouped.get(timestamp_us, []), class_ids)
            if not args.without_ground_truth:
                event_bgr = draw_ground_truth(event_bgr, annotations)
            event_bgr = draw_predictions(event_bgr, detections)

            rectified_path = png_map.get(timestamp_us)
            if rectified_path is None:
                raise SystemExit(f"No PNG found for timestamp {timestamp_us}")
            rectified_bgr = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
            if rectified_bgr is None:
                raise SystemExit(f"Could not read {rectified_path}")
            frame_label = "Rectified frame camera | unannotated reference"
            if args.draw_scaled_frame_boxes:
                rectified_bgr = cv2.resize(
                    rectified_bgr, (EVENT_WIDTH, EVENT_HEIGHT), interpolation=cv2.INTER_AREA
                )
                if not args.without_ground_truth:
                    rectified_bgr = draw_ground_truth(rectified_bgr, annotations)
                rectified_bgr = draw_predictions(rectified_bgr, detections)
                frame_label = "Rectified frame camera | scaled boxes (not calibrated)"
            elif args.draw_calibrated_frame_boxes:
                if frame_projection is None:
                    raise AssertionError(
                        "frame_projection must be loaded for calibrated frame boxes"
                    )
                if not args.without_ground_truth:
                    rectified_bgr = draw_projected_ground_truth(
                        rectified_bgr,
                        annotations,
                        projection=frame_projection,
                        depth_m=args.frame_box_depth_m,
                    )
                rectified_bgr = draw_projected_predictions(
                    rectified_bgr,
                    detections,
                    projection=frame_projection,
                    depth_m=args.frame_box_depth_m,
                )
                frame_label = (
                    f"Rectified frame camera | calibrated approx z={args.frame_box_depth_m:g}m"
                )
            writer.write(
                compose_frame(
                    event_bgr=event_bgr,
                    rectified_bgr=rectified_bgr,
                    timestamp_us=timestamp_us,
                    pred_count=len(detections),
                    gt_count=len(annotations),
                    label=label,
                    frame_label=frame_label,
                )
            )
            if position == 1 or position == total or position % 25 == 0:
                print(
                    f"{position}/{total}: {timestamp_us} | preds={len(detections)} | gt={len(annotations)}"
                )
    finally:
        handle.close()
        writer.release()

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
