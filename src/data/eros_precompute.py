"""Precompute persistent EROS snapshots for DSEC-MOT sequences."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH, _timestamp_window_indices
from src.evaluation.detection_export import load_event_file, load_image_timestamps

try:
    from numba import njit
except ImportError:
    njit = None


def _update_eros_python(surface, xs, ys, radius: int, decay: float) -> None:
    """Reference Python implementation of local EROS updates."""

    height, width = surface.shape
    for x, y in zip(xs, ys):
        x0, x1 = max(int(x) - radius, 0), min(int(x) + radius + 1, width)
        y0, y1 = max(int(y) - radius, 0), min(int(y) + radius + 1, height)
        surface[y0:y1, x0:x1] *= decay
        surface[int(y), int(x)] = 255.0


if njit is not None:

    @njit(cache=True)
    def _update_eros_numba(surface, xs, ys, radius: int, decay: float) -> None:
        height, width = surface.shape
        for index in range(len(xs)):
            x, y = int(xs[index]), int(ys[index])
            x0, x1 = max(x - radius, 0), min(x + radius + 1, width)
            y0, y1 = max(y - radius, 0), min(y + radius + 1, height)
            for yy in range(y0, y1):
                for xx in range(x0, x1):
                    surface[yy, xx] *= decay
            surface[y, x] = 255.0


def update_eros(surface, xs, ys, radius: int, decay: float) -> None:
    """Update an EROS surface with a batch of events."""

    if njit is None:
        _update_eros_python(surface, xs, ys, radius, decay)
    else:
        _update_eros_numba(surface, xs, ys, radius, decay)


def precompute_sequence(
    root: Path,
    output_root: Path,
    split: str,
    sequence: str,
    radius: int,
    decay: float,
) -> None:
    """Precompute EROS snapshots at image timestamps for one sequence."""

    seq_dir = root / split / sequence
    timestamps = load_image_timestamps(seq_dir / f"{sequence}_image_timestamps.txt")
    output_dir = output_root / split / sequence
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots = np.lib.format.open_memmap(
        output_dir / "snapshots.npy",
        mode="w+",
        dtype=np.uint8,
        shape=(len(timestamps), EVENT_HEIGHT, EVENT_WIDTH),
    )
    surface = np.zeros((EVENT_HEIGHT, EVENT_WIDTH), dtype=np.float32)
    handle, x, y, _, t, ms_to_idx, t_offset, _ = load_event_file(
        seq_dir / "events_left" / "events.h5"
    )
    previous_timestamp = int(t[0]) + int(t_offset) - 1
    started = time.perf_counter()
    try:
        for frame_index, timestamp in enumerate(timestamps):
            start_idx, end_idx = _timestamp_window_indices(
                ms_to_idx,
                t,
                t_offset,
                previous_timestamp + 1,
                int(timestamp),
            )
            if end_idx > start_idx:
                xs = x[start_idx:end_idx].astype(np.int32)
                ys = y[start_idx:end_idx].astype(np.int32)
                valid = (xs >= 0) & (xs < EVENT_WIDTH) & (ys >= 0) & (ys < EVENT_HEIGHT)
                update_eros(surface, xs[valid], ys[valid], radius, decay)
            snapshots[frame_index] = np.clip(surface, 0, 255).astype(np.uint8)
            previous_timestamp = int(timestamp)
            if (
                frame_index == 0
                or (frame_index + 1) % 250 == 0
                or frame_index + 1 == len(timestamps)
            ):
                print(f"[{split}/{sequence}] {frame_index + 1}/{len(timestamps)}", flush=True)
    finally:
        handle.close()
        snapshots.flush()

    metadata = {
        "split": split,
        "sequence": sequence,
        "radius": radius,
        "decay": decay,
        "timestamps": timestamps,
        "elapsed_s": time.perf_counter() - started,
        "implementation": "numba" if njit is not None else "python",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--output-root", type=Path, default=Path("data/cache/dsec_mot_eros"))
    parser.add_argument("--splits", nargs="+", default=["train", "test"])
    parser.add_argument("--sequences", nargs="+", default=None)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument(
        "--decay",
        type=float,
        default=None,
        help="Per-event local decay. Defaults to 0.3 ** (1 / radius), following PUCK Algorithm 1.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def selected_sequences(root: Path, split: str, requested: list[str] | None) -> list[str]:
    """Return requested sequence names or discover every sequence in a split."""

    if requested is not None:
        return requested
    return sorted(path.name for path in (root / split).iterdir() if path.is_dir())


def main() -> int:
    args = parse_args()
    if args.radius <= 0:
        raise SystemExit("--radius must be positive.")
    decay = args.decay if args.decay is not None else 0.3 ** (1.0 / args.radius)
    if not 0.0 < decay < 1.0:
        raise SystemExit("--decay must be between 0 and 1.")
    if njit is None:
        print("WARNING: numba is not installed. Exact EROS precomputation will be very slow.")

    for split in args.splits:
        for sequence in selected_sequences(args.root, split, args.sequences):
            output_path = args.output_root / split / sequence / "snapshots.npy"
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing cache: {output_path}")
                continue
            precompute_sequence(args.root, args.output_root, split, sequence, args.radius, decay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
