#!/usr/bin/env python3
"""Evaluate one SimpleDenseDetector checkpoint with multiple MOT trackers."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

from src.experiments.common import (
    CommandRunner,
    EvalTarget,
    require_checkpoint,
    threshold_label,
)
from src.experiments.common import simple_detector_eval_command as build_eval_command

DEFAULT_TRACKERS = (
    "iou",
    "sort",
    "bytetrack",
    "boxmot_bytetrack",
    "boxmot_ocsort",
    "boxmot_botsort",
)
DEFAULT_EVAL_TARGETS = (
    EvalTarget("train", "zurich_city_01_d", "val"),
    EvalTarget("test", ("interlaken_00_d", "zurich_city_00_b"), "test"),
)
RUN_NAME_RE = re.compile(r"^(?P<base_tracker>.+)_(?P<split>val|test)_thr(?P<threshold>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/datasets/dsec_mot"))
    parser.add_argument("--trackers", nargs="+", default=list(DEFAULT_TRACKERS))
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.10, 0.25, 0.50, 0.70, 0.90, 0.95],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--track-iou-threshold", type=float, default=0.5)
    parser.add_argument("--track-max-missed-frames", type=int, default=30)
    parser.add_argument("--track-min-hits", type=int, default=1)
    parser.add_argument("--track-high-threshold", type=float, default=0.6)
    parser.add_argument("--track-low-threshold", type=float, default=0.1)
    parser.add_argument("--classes-to-eval", nargs="+", default=None)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_tracker_benchmark"),
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/tracker_benchmark_logs"))
    parser.add_argument("--run-prefix", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not require_checkpoint(args.checkpoint, args.dry_run):
        return 1

    python = sys.executable
    runner = CommandRunner(dry_run=args.dry_run)
    run_prefix = args.run_prefix or args.checkpoint.parent.name

    for target in DEFAULT_EVAL_TARGETS:
        for tracker in args.trackers:
            for threshold in args.thresholds:
                run_name = f"{run_prefix}_{tracker}_{target.label}_thr{threshold_label(threshold)}"
                summary = args.results_root / run_name / "metrics_summary.json"
                if summary.exists() and not args.overwrite:
                    print(f"Skipping eval, summary exists: {summary}")
                    continue

                command = build_eval_command(
                    python=python,
                    checkpoint=args.checkpoint,
                    root=args.root,
                    target=target,
                    threshold=threshold,
                    device=args.device,
                    max_detections=args.max_detections,
                    output_root=args.results_root,
                    run_name=run_name,
                    classes_to_eval=args.classes_to_eval,
                    tracker_backend=tracker,
                    tracker_name=tracker,
                    track_iou_threshold=args.track_iou_threshold,
                    track_max_missed_frames=args.track_max_missed_frames,
                    track_min_hits=args.track_min_hits,
                    track_high_threshold=args.track_high_threshold,
                    track_low_threshold=args.track_low_threshold,
                )
                code = runner.run(command, args.log_dir / f"eval_{run_name}.log")
                if code != 0:
                    return code

    if not args.dry_run:
        write_summary_tables(args.results_root)
    print("Tracker benchmark completed.")
    return 0


def write_summary_tables(results_root: Path) -> None:
    rows = collect_summary_rows(results_root)
    all_path = results_root / "tracker_metrics.csv"
    selected_path = results_root / "tracker_metrics_val_selected.csv"
    write_csv(all_path, rows)
    write_csv(selected_path, build_val_selected_rows(rows))
    print(f"Saved tracker metrics to {all_path}")
    print(f"Saved validation-selected tracker metrics to {selected_path}")


def collect_summary_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(results_root.glob("*/metrics_summary.json")):
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        match = RUN_NAME_RE.match(summary_path.parent.name)
        if not match:
            continue
        tracker = payload.get("tracker_name", "")
        base_tracker = match.group("base_tracker")
        tracker_suffix = f"_{tracker}"
        base = (
            base_tracker[: -len(tracker_suffix)]
            if tracker and base_tracker.endswith(tracker_suffix)
            else base_tracker
        )
        metrics = payload.get("aggregate", {})
        rows.append(
            {
                "run": summary_path.parent.name,
                "base": base,
                "tracker": tracker,
                "split": match.group("split"),
                "threshold": match.group("threshold"),
                "HOTA": metrics.get("HOTA", ""),
                "MOTA": metrics.get("MOTA", ""),
                "IDF1": metrics.get("IDF1", ""),
                "IDS": metrics.get("IDS", ""),
                "FP": metrics.get("FP", ""),
                "FN": metrics.get("FN", ""),
            }
        )
    return rows


def build_val_selected_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]] = {}
    for row in rows:
        split = str(row["split"])
        threshold = str(row["threshold"])
        key = (str(row["base"]), str(row["tracker"]))
        grouped.setdefault(key, {}).setdefault(split, {})[threshold] = row

    selected: list[dict[str, object]] = []
    for (base, tracker), by_split in sorted(grouped.items()):
        val_rows = by_split.get("val", {})
        test_rows = by_split.get("test", {})
        if not val_rows or not test_rows:
            continue
        threshold, val_row = max(
            val_rows.items(),
            key=lambda item: float(item[1].get("HOTA") or -1),
        )
        test_row = test_rows.get(threshold)
        if test_row is None:
            continue
        selected.append(
            {
                "base": base,
                "tracker": tracker,
                "selected_threshold": threshold,
                "val_HOTA": val_row.get("HOTA", ""),
                "val_MOTA": val_row.get("MOTA", ""),
                "val_IDF1": val_row.get("IDF1", ""),
                "test_HOTA": test_row.get("HOTA", ""),
                "test_MOTA": test_row.get("MOTA", ""),
                "test_IDF1": test_row.get("IDF1", ""),
                "test_IDS": test_row.get("IDS", ""),
                "test_FP": test_row.get("FP", ""),
                "test_FN": test_row.get("FN", ""),
            }
        )
    return selected


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
