#!/usr/bin/env python3
"""Evaluate an external EvRT-DETR detector on DSEC-MOT with the thesis tracker and TrackEval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.evrtdetr_export import export_evrtdetr_detections_for_sequence
from src.evaluation.evrtdetr_runtime import DEFAULT_PUBLIC_MODEL_ROOT
from src.evaluation.simple_tracker import track_detections
from src.evaluation.trackeval_adapter import (
    export_trackeval_bundle,
    run_trackeval,
    summarise_trackeval_results,
    write_summary_csv,
    write_summary_json,
)


DEFAULT_TEST_SEQUENCES = ["interlaken_00_d", "zurich_city_00_b"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_PUBLIC_MODEL_ROOT)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--sequences",
        nargs="+",
        default=None,
        help="Defaults to interlaken_00_d and zurich_city_00_b for the test split.",
    )
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--score-threshold", type=float, default=0.35)
    parser.add_argument("--track-iou-threshold", type=float, default=0.5)
    parser.add_argument("--track-max-missed-frames", type=int, default=2)
    parser.add_argument("--track-min-hits", type=int, default=1)
    parser.add_argument("--eval-iou-threshold", type=float, default=0.5)
    parser.add_argument("--window-ms", type=float, default=50.0)
    parser.add_argument("--n-bins", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--tracker-name", default="simple_iou_tracker")
    parser.add_argument("--trackeval-root", type=Path, default=Path("external/TrackEval"))
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda or cpu.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/dsec_mot_trackeval_evrtdetr"),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Defaults to <model_dir_name>_<split>.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sequences = args.sequences or (DEFAULT_TEST_SEQUENCES if args.split == "test" else [])
    if not sequences:
        raise SystemExit("No sequences selected. Pass --sequences explicitly for non-test splits.")

    run_name = args.run_name or f"{args.model_dir.resolve().name}_{args.split}"
    output_root = args.output_root / run_name
    detections_dir = output_root / "detections"
    tracks_dir = output_root / "tracks"
    trackeval_dir = output_root / "trackeval"
    output_root.mkdir(parents=True, exist_ok=True)

    detection_runs: dict[str, dict] = {}
    tracker_runs: dict[str, dict] = {}

    def progress(sequence: str):
        def callback(position: int, total: int, timestamp_us: int, pred_count: int) -> None:
            if position == 1 or position == total or position % 100 == 0:
                print(f"[{sequence}] {position}/{total}: t={timestamp_us} preds={pred_count}")

        return callback

    for sequence in sequences:
        detection_path = detections_dir / f"{sequence}.json"
        print(f"Exporting detections for {sequence} -> {detection_path}")
        detection_runs[sequence] = export_evrtdetr_detections_for_sequence(
            model_dir=args.model_dir,
            root=args.root,
            split=args.split,
            sequence=sequence,
            output_path=detection_path,
            score_threshold=args.score_threshold,
            device=args.device,
            window_ms=args.window_ms,
            n_bins=args.n_bins,
            start_frame=args.start_frame,
            max_frames=args.max_frames,
            progress_callback=progress(sequence),
        )

        track_path = tracks_dir / f"{sequence}.txt"
        print(f"Tracking {sequence} -> {track_path}")
        tracker_runs[sequence] = track_detections(
            detection_export_path=detection_path,
            output_path=track_path,
            iou_threshold=args.track_iou_threshold,
            max_missed_frames=args.track_max_missed_frames,
            min_hits=args.track_min_hits,
        )

    (output_root / "detection_runs.json").write_text(json.dumps(detection_runs, indent=2), encoding="utf-8")
    (output_root / "tracker_runs.json").write_text(json.dumps(tracker_runs, indent=2), encoding="utf-8")

    print(f"Preparing TrackEval bundle in {trackeval_dir}")
    bundle = export_trackeval_bundle(
        dataset_root=args.root,
        split=args.split,
        sequences=sequences,
        tracker_name=args.tracker_name,
        tracker_results_dir=tracks_dir,
        output_root=trackeval_dir,
    )

    print("Running TrackEval")
    raw_results, raw_messages = run_trackeval(
        bundle=bundle,
        output_root=trackeval_dir / "reports",
        trackeval_root=args.trackeval_root,
        eval_iou_threshold=args.eval_iou_threshold,
    )

    summary = summarise_trackeval_results(
        results=raw_results,
        tracker_name=args.tracker_name,
        eval_iou_threshold=args.eval_iou_threshold,
        trackeval_root=args.trackeval_root,
    )
    messages_path = output_root / "trackeval_messages.json"
    messages_path.write_text(json.dumps(raw_messages, indent=2), encoding="utf-8")

    summary_json = output_root / "metrics_summary.json"
    summary_csv = output_root / "metrics_summary.csv"
    write_summary_json(summary, summary_json)
    write_summary_csv(summary, summary_csv)

    print("Per-sequence results:")
    for sequence, sequence_summary in summary["per_sequence"].items():
        metrics = sequence_summary["metrics"]
        print(
            f"  {sequence}: MOTA={metrics['MOTA']:.4f} IDF1={metrics['IDF1']:.4f} IDS={metrics['IDS']}"
        )
    aggregate = summary["aggregate"]
    print(f"  COMBINED: MOTA={aggregate['MOTA']:.4f} IDF1={aggregate['IDF1']:.4f} IDS={aggregate['IDS']}")
    print(f"Saved summary JSON to {summary_json}")
    print(f"Saved summary CSV to {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
