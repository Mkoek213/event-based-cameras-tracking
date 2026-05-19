#!/usr/bin/env python3
"""Convert local DSEC-MOT into the NPZ frame layout expected by EvRT-DETR."""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.evaluation.detection_export import load_annotations, load_event_file, load_image_timestamps
from src.evaluation.evrtdetr_runtime import build_stacked_histogram


CLASS_NAMES = [
    "car",
    "pedestrian",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "train",
]
DEFAULT_VAL_SEQUENCES = ["zurich_city_09_c"]
DEFAULT_TEST_SEQUENCES = ["interlaken_00_d", "zurich_city_00_b"]

try:  # pragma: no cover - optional runtime dependency
    from psee_adt.io.box_loading import BBOX_DTYPE as PSEE_BBOX_DTYPE
except ImportError:  # pragma: no cover - fallback for conversion only
    PSEE_BBOX_DTYPE = np.dtype(
        [
            ("t", np.int64),
            ("x", np.float32),
            ("y", np.float32),
            ("w", np.float32),
            ("h", np.float32),
            ("class_id", np.uint32),
            ("track_id", np.uint32),
            ("class_confidence", np.float32),
        ]
    )


@dataclass(frozen=True)
class SequenceJob:
    source_split: str
    source_sequence: str
    target_split: str
    video_index: int


def parse_sequence_list(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/dsec_mot_evrtdetr_npz"),
        help="Destination root for EvRT-DETR-compatible NPZ files.",
    )
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--val-sequences",
        default=",".join(DEFAULT_VAL_SEQUENCES),
        help="Comma-separated train sequences to hold out under the output val split.",
    )
    parser.add_argument(
        "--test-sequences",
        default=",".join(DEFAULT_TEST_SEQUENCES),
        help="Comma-separated sequences to place under the output test split.",
    )
    parser.add_argument(
        "--max-frames-per-sequence",
        type=int,
        default=0,
        help="For smoke tests only. 0 converts every frame.",
    )
    return parser.parse_args()


def collect_jobs(root: Path, val_sequences: list[str], test_sequences: list[str]) -> list[SequenceJob]:
    train_sequences = sorted(path.name for path in (root / "train").iterdir() if path.is_dir())
    test_split_sequences = sorted(path.name for path in (root / "test").iterdir() if path.is_dir())

    unknown_val = sorted(set(val_sequences) - set(train_sequences))
    if unknown_val:
        raise SystemExit(f"Validation sequences not found under {root / 'train'}: {unknown_val}")

    unknown_test = sorted(set(test_sequences) - set(test_split_sequences))
    if unknown_test:
        raise SystemExit(f"Test sequences not found under {root / 'test'}: {unknown_test}")

    jobs: list[SequenceJob] = []
    ordered: list[tuple[str, str, str]] = []
    ordered += [("train", seq, "train") for seq in train_sequences if seq not in set(val_sequences)]
    ordered += [("train", seq, "val") for seq in val_sequences]
    ordered += [("test", seq, "test") for seq in test_sequences]

    for video_index, (source_split, source_sequence, target_split) in enumerate(ordered):
        jobs.append(
            SequenceJob(
                source_split=source_split,
                source_sequence=source_sequence,
                target_split=target_split,
                video_index=video_index,
            )
        )

    return jobs


def annotations_to_structured(annotations: list, timestamp_us: int) -> np.ndarray:
    result = np.zeros((len(annotations),), dtype=PSEE_BBOX_DTYPE)
    if len(annotations) == 0:
        return result

    result["t"] = np.int64(timestamp_us)
    result["x"] = np.array([ann.left for ann in annotations], dtype=np.float32)
    result["y"] = np.array([ann.top for ann in annotations], dtype=np.float32)
    result["w"] = np.array([ann.width for ann in annotations], dtype=np.float32)
    result["h"] = np.array([ann.height for ann in annotations], dtype=np.float32)
    result["class_id"] = np.array([ann.class_id for ann in annotations], dtype=np.uint32)
    result["track_id"] = np.array([ann.track_id for ann in annotations], dtype=np.uint32)
    result["class_confidence"] = np.ones((len(annotations),), dtype=np.float32)
    return result


def labels_to_dict(annotations: list, timestamp_us: int, image_id: int) -> dict | None:
    if not annotations:
        return None

    boxes = np.zeros((len(annotations), 4), dtype=np.float32)
    labels = np.zeros((len(annotations),), dtype=np.int32)
    areas = np.zeros((len(annotations),), dtype=np.float32)

    for index, ann in enumerate(annotations):
        left = float(np.clip(ann.left, 0.0, EVENT_WIDTH - 1))
        top = float(np.clip(ann.top, 0.0, EVENT_HEIGHT - 1))
        right = float(np.clip(ann.left + ann.width, 0.0, EVENT_WIDTH - 1))
        bottom = float(np.clip(ann.top + ann.height, 0.0, EVENT_HEIGHT - 1))
        boxes[index] = [left, top, right, bottom]
        labels[index] = int(ann.class_id)
        areas[index] = max(0.0, (right - left) * (bottom - top))

    psee_labels = annotations_to_structured(annotations, timestamp_us)
    return {
        "boxes": boxes,
        "labels": labels,
        "image_id": np.full((len(annotations),), fill_value=image_id, dtype=np.uint64),
        "area": areas,
        "iscrowd": np.zeros((len(annotations),), dtype=np.uint8),
        "time": np.full((len(annotations),), fill_value=timestamp_us, dtype=np.int64),
        "psee_labels": psee_labels,
        "dsec_labels_raw": psee_labels,
    }


def process_sequence(
    root: Path,
    output: Path,
    window_us: int,
    n_bins: int,
    max_frames_per_sequence: int,
    job: SequenceJob,
) -> dict:
    seq_dir = root / job.source_split / job.source_sequence
    ann_path = root / "annotations" / job.source_split / f"{job.source_sequence}.txt"
    timestamps_path = seq_dir / f"{job.source_sequence}_image_timestamps.txt"
    events_path = seq_dir / "events_left" / "events.h5"

    timestamps = load_image_timestamps(timestamps_path)
    if max_frames_per_sequence > 0:
        timestamps = timestamps[:max_frames_per_sequence]

    annotations = load_annotations(ann_path)
    grouped: dict[int, list] = {}
    for ann in annotations:
        grouped.setdefault(ann.timestamp, []).append(ann)

    target_dir = output / job.target_split / job.source_sequence
    target_dir.mkdir(parents=True, exist_ok=True)

    handle, x, y, p, t, ms_to_idx, t_offset, np_h5 = load_event_file(events_path)
    labeled_count = 0
    try:
        for frame_index, timestamp_us in enumerate(timestamps):
            hist = build_stacked_histogram(
                x=x,
                y=y,
                p=p,
                t=t,
                ms_to_idx=ms_to_idx,
                t_offset=t_offset,
                timestamp_us=timestamp_us,
                window_us=window_us,
                n_bins=n_bins,
                np_mod=np_h5,
            ).astype(np.uint16, copy=False)

            image_id = (job.video_index << 32) + frame_index
            labels_dict = labels_to_dict(grouped.get(timestamp_us, []), timestamp_us, image_id)

            np.savez_compressed(
                target_dir / f"data_{frame_index:05d}.npz",
                frame=hist,
                time=np.int64(timestamp_us),
            )
            if labels_dict is not None:
                labeled_count += 1
                np.savez_compressed(target_dir / f"labels_{frame_index:05d}.npz", **labels_dict)
    finally:
        handle.close()

    return {
        "source_split": job.source_split,
        "source_sequence": job.source_sequence,
        "target_split": job.target_split,
        "frame_count": len(timestamps),
        "labeled_frame_count": labeled_count,
    }


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    val_sequences = parse_sequence_list(args.val_sequences)
    test_sequences = parse_sequence_list(args.test_sequences)
    jobs = collect_jobs(root, val_sequences=val_sequences, test_sequences=test_sequences)
    window_us = int(round(args.window_ms * 1000.0))

    results: list[dict]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [
                pool.submit(
                    process_sequence,
                    root,
                    output,
                    window_us,
                    args.n_bins,
                    args.max_frames_per_sequence,
                    job,
                )
                for job in jobs
            ]
            results = [future.result() for future in futures]
    else:
        results = [
            process_sequence(
                root=root,
                output=output,
                window_us=window_us,
                n_bins=args.n_bins,
                max_frames_per_sequence=args.max_frames_per_sequence,
                job=job,
            )
            for job in jobs
        ]

    config = {
        "source_root": str(root),
        "output_root": str(output),
        "image_size": [EVENT_HEIGHT, EVENT_WIDTH],
        "time_window_us": window_us,
        "n_bins": args.n_bins,
        "classes": CLASS_NAMES,
        "val_sequences": val_sequences,
        "test_sequences": test_sequences,
        "jobs": [asdict(job) for job in jobs],
        "results": results,
    }
    (output / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    for item in results:
        print(
            f"{item['target_split']}/{item['source_sequence']}: "
            f"frames={item['frame_count']} labeled_frames={item['labeled_frame_count']}"
        )

    print(f"Wrote EvRT-DETR NPZ dataset to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
