#!/usr/bin/env python3
"""Select each EROS variant's threshold by validation HOTA and report its test metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("results/dsec_mot_trackeval_simple_detector_eros")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/dsec_mot_trackeval_simple_detector_eros/summary.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    best: dict[str, tuple[str, dict]] = {}
    for path in args.results_root.glob("*_val_thr*/metrics_summary.json"):
        label, threshold = path.parent.name.rsplit("_val_thr", 1)
        metrics = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
        current = best.get(label)
        if current is None or metrics["HOTA"] > current[1]["HOTA"]:
            best[label] = (threshold, metrics)

    rows: list[dict] = []
    for label, (threshold, val) in sorted(best.items()):
        test_path = args.results_root / f"{label}_test_thr{threshold}" / "metrics_summary.json"
        if not test_path.exists():
            print(f"Missing corresponding test result: {test_path}")
            continue
        test = json.loads(test_path.read_text(encoding="utf-8"))["aggregate"]
        rows.append(
            {
                "variant": label,
                "threshold": int(threshold) / 100,
                **{f"val_{key}": val[key] for key in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN")},
                **{f"test_{key}": test[key] for key in ("HOTA", "MOTA", "IDF1", "IDS", "FP", "FN")},
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        with args.output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(rows, indent=2))
    print(f"Saved summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
