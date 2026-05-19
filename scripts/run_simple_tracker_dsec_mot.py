#!/usr/bin/env python3
"""Run the simple class-aware IoU tracker on exported DSEC-MOT detections."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.simple_tracker import track_detections


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, required=True, help="Detection export JSON.")
    parser.add_argument("--output", type=Path, required=True, help="Tracker output .txt file.")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--max-missed-frames", type=int, default=2)
    parser.add_argument("--min-hits", type=int, default=1)
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional tracker run summary JSON. Defaults to <output>.summary.json",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = track_detections(
        detection_export_path=args.detections,
        output_path=args.output,
        iou_threshold=args.iou_threshold,
        max_missed_frames=args.max_missed_frames,
        min_hits=args.min_hits,
    )
    summary_output = args.summary_output or args.output.with_suffix(args.output.suffix + ".summary.json")
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary_output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved tracker output to {args.output}")
    print(f"Saved tracker summary to {summary_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
