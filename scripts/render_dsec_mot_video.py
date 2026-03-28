#!/usr/bin/env python3
"""Render a DSEC-MOT sequence as a side-by-side video."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from visualize_dsec_mot_samples import (
    CLASS_NAMES,
    EVENT_HEIGHT,
    EVENT_WIDTH,
    build_event_frame,
    build_timestamp_to_png,
    group_annotations_by_timestamp,
    import_or_die,
    load_annotations,
    load_event_file,
    load_image_timestamps,
)


TOP_BAR = 30
GAP = 16


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/datasets/dsec_mot"),
        help="Path to the local DSEC-MOT root.",
    )
    parser.add_argument("--split", choices=("train", "test"), required=True, help="Dataset split.")
    parser.add_argument("--sequence", required=True, help="Sequence name, for example interlaken_00_a.")
    parser.add_argument(
        "--window-ms",
        type=float,
        default=50.0,
        help="Event accumulation window in milliseconds before each timestamp.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=20.0,
        help="Output video frames per second.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Start from this image/frame index.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Maximum number of frames to render. 0 means all frames.",
    )
    parser.add_argument(
        "--annotated-only",
        action="store_true",
        help="Render only frames that have at least one annotation.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output video path. Defaults to data/processed/dsec_mot_videos/<split>/<sequence>.avi",
    )
    return parser.parse_args()


def draw_boxes_bgr(frame: np.ndarray, annotations) -> np.ndarray:
    output = frame.copy()
    for annotation in annotations:
        left = max(0, int(round(annotation.left)))
        top = max(0, int(round(annotation.top)))
        right = min(EVENT_WIDTH - 1, int(round(annotation.left + annotation.width)))
        bottom = min(EVENT_HEIGHT - 1, int(round(annotation.top + annotation.height)))
        cv2.rectangle(output, (left, top), (right, bottom), (0, 255, 0), 2)
        class_name = CLASS_NAMES.get(annotation.class_id, str(annotation.class_id))
        label = f"{class_name} id={annotation.track_id}"
        text_y = max(14, top - 6)
        cv2.putText(output, label, (left, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return output


def compose_frame(event_rgb: np.ndarray, rectified_bgr: np.ndarray, timestamp_us: int, boxes_count: int) -> np.ndarray:
    event_bgr = cv2.cvtColor(event_rgb, cv2.COLOR_RGB2BGR)
    right = cv2.resize(rectified_bgr, (EVENT_WIDTH, EVENT_HEIGHT), interpolation=cv2.INTER_AREA)

    canvas = np.full((EVENT_HEIGHT + TOP_BAR, EVENT_WIDTH * 2 + GAP, 3), 24, dtype=np.uint8)
    canvas[TOP_BAR:, :EVENT_WIDTH] = event_bgr
    canvas[TOP_BAR:, EVENT_WIDTH + GAP:] = right

    cv2.putText(
        canvas,
        f"Event view with DSEC-MOT boxes | t={timestamp_us} us | boxes={boxes_count}",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Rectified left frame | reference only",
        (EVENT_WIDTH + GAP + 8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> int:
    args = parse_args()
    h5py, np_h5, _image, _image_draw = import_or_die()

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
    grouped = group_annotations_by_timestamp(load_annotations(annotation_path))
    png_map = build_timestamp_to_png(seq_dir, args.sequence)

    if args.annotated_only:
        timestamps = [timestamp for timestamp in timestamps if timestamp in grouped]

    if args.start_frame:
        timestamps = timestamps[args.start_frame :]
    if args.max_frames > 0:
        timestamps = timestamps[: args.max_frames]
    if not timestamps:
        raise SystemExit("No frames selected for rendering.")

    output_path = args.output
    if output_path is None:
        output_path = Path("data/processed/dsec_mot_videos") / args.split / f"{args.sequence}.avi"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        args.fps,
        (EVENT_WIDTH * 2 + GAP, EVENT_HEIGHT + TOP_BAR),
    )
    if not writer.isOpened():
        raise SystemExit(f"Could not open video writer for {output_path}")

    window_us = int(args.window_ms * 1000)
    handle, x, y, p, t, ms_to_idx, t_offset = load_event_file({"events_h5": events_h5}, h5py)
    try:
        total = len(timestamps)
        for index, timestamp_us in enumerate(timestamps, start=1):
            event_rgb = build_event_frame(x, y, p, t, ms_to_idx, t_offset, timestamp_us, window_us, np_h5)
            event_rgb = draw_boxes_bgr(cv2.cvtColor(event_rgb, cv2.COLOR_RGB2BGR), grouped.get(timestamp_us, []))
            event_rgb = cv2.cvtColor(event_rgb, cv2.COLOR_BGR2RGB)

            rectified_path = png_map.get(timestamp_us)
            if rectified_path is None:
                raise SystemExit(f"No PNG found for timestamp {timestamp_us}")
            rectified_bgr = cv2.imread(str(rectified_path), cv2.IMREAD_COLOR)
            if rectified_bgr is None:
                raise SystemExit(f"Could not read {rectified_path}")

            frame = compose_frame(
                event_rgb=event_rgb,
                rectified_bgr=rectified_bgr,
                timestamp_us=timestamp_us,
                boxes_count=len(grouped.get(timestamp_us, [])),
            )
            writer.write(frame)

            if index == 1 or index == total or index % 100 == 0:
                print(f"{index}/{total}: {timestamp_us}")
    finally:
        handle.close()
        writer.release()

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
