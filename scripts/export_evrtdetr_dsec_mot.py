#!/usr/bin/env python3
"""Export external EvRT-DETR detections for DSEC-MOT to the thesis JSON schema."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evrtdetr_export import export_evrtdetr_detections_for_sequence
from src.evaluation.evrtdetr_runtime import DEFAULT_PUBLIC_MODEL_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_PUBLIC_MODEL_ROOT)
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--n-bins", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Defaults to results/detections_evrtdetr/<split>/<sequence>.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output or Path("results/detections_evrtdetr") / args.split / f"{args.sequence}.json"

    def progress(position: int, total: int, timestamp_us: int, pred_count: int) -> None:
        if position == 1 or position == total or position % 100 == 0:
            print(f"{position}/{total}: t={timestamp_us} preds={pred_count}")

    payload = export_evrtdetr_detections_for_sequence(
        model_dir=args.model_dir,
        root=args.root,
        split=args.split,
        sequence=args.sequence,
        output_path=output,
        score_threshold=args.score_threshold,
        device=args.device,
        window_ms=args.window_ms,
        n_bins=args.n_bins,
        start_frame=args.start_frame,
        max_frames=args.max_frames,
        progress_callback=progress,
    )
    print(f"Saved {len(payload['detections'])} detections to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
