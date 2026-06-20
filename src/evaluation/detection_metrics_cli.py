"""CLI implementation for detection-only metric evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.evaluation.detection_metrics import evaluate_detection_export


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detections", type=Path, nargs="*", default=[])
    parser.add_argument(
        "--results-root",
        type=Path,
        default=None,
        help="When provided, evaluates every detections/*.json file below this root.",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-csv", type=Path, default=None)
    return parser.parse_args()


def collect_detection_paths(detections: list[Path], results_root: Path | None) -> list[Path]:
    """Collect explicit detection exports and optional exports below a results root."""

    paths = list(detections)
    if results_root is not None:
        paths.extend(sorted(results_root.glob("*/detections/*.json")))
    return paths


def evaluate_paths(paths: list[Path], iou_threshold: float) -> list[dict[str, object]]:
    """Evaluate detection exports, save per-run JSON summaries and return CSV rows."""

    rows: list[dict[str, object]] = []
    for path in paths:
        summary = evaluate_detection_export(path, iou_threshold=iou_threshold)
        output_path = path.parent.parent / "detection_metrics.json"
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        aggregate = summary["aggregate"]
        rows.append(
            {
                "run": path.parent.parent.name,
                "split": summary["split"],
                "sequence": summary["sequence"],
                "score_threshold": summary["score_threshold"],
                "mAP50": aggregate["mAP50"],
                "precision": aggregate["precision"],
                "recall": aggregate["recall"],
                "f1": aggregate["f1"],
                "tp": aggregate["tp"],
                "fp": aggregate["fp"],
                "fn": aggregate["fn"],
                "gt": aggregate["gt"],
                "detections": aggregate["detections"],
            }
        )
        print(
            f"{path.parent.parent.name}: AP50={aggregate['mAP50']:.4f} "
            f"P={aggregate['precision']:.4f} R={aggregate['recall']:.4f} "
            f"F1={aggregate['f1']:.4f}"
        )
    return rows


def write_csv(rows: list[dict[str, object]], output_csv: Path) -> None:
    """Write detection metric rows to CSV."""

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved CSV to {output_csv}")


def main() -> int:
    args = parse_args()
    paths = collect_detection_paths(args.detections, args.results_root)
    if not paths:
        raise SystemExit("Pass --detections or --results-root.")

    rows = evaluate_paths(paths, iou_threshold=args.iou_threshold)
    if args.output_csv is not None:
        write_csv(rows, args.output_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
