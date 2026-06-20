#!/usr/bin/env python3
"""Collect car-only MOT and detection metrics from existing TrackEval runs."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

RUN_SUFFIX_RE = re.compile(r"^(?P<base>.+)_(?P<split>val|test)_thr(?P<threshold>\d+)$")
MOT_FIELDS = (
    "HOTA",
    "DetA",
    "AssA",
    "MOTA",
    "IDF1",
    "IDSW",
    "CLR_TP",
    "CLR_FN",
    "CLR_FP",
    "Dets",
    "GT_Dets",
)
DET_FIELDS = ("AP50", "precision", "recall", "f1", "tp", "fp", "fn", "gt", "detections")


def parse_car_summary(path: Path) -> dict[str, float]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError(f"Invalid summary file: {path}")
    keys = lines[0].split()
    values = lines[1].split()
    if len(keys) != len(values):
        raise ValueError(f"Header/value length mismatch in {path}")
    result: dict[str, float] = {}
    for key, value in zip(keys, values):
        parsed = float(value)
        result[key] = int(parsed) if parsed.is_integer() else parsed
    return result


def parse_run_name(run_name: str) -> tuple[str, str, str]:
    match = RUN_SUFFIX_RE.match(run_name)
    if not match:
        return run_name, "", ""
    return match.group("base"), match.group("split"), match.group("threshold")


def read_car_detection_metrics(run_dir: Path) -> dict[str, float | int | str]:
    metrics_path = run_dir / "detection_metrics.json"
    if not metrics_path.exists():
        return {}
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return payload.get("per_class", {}).get("car", {})


def collect_rows(results_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for summary_path in sorted(results_root.glob("**/trackeval/reports/*/car_summary.txt")):
        run_dir = summary_path.parents[3]
        tracker = summary_path.parents[0].name
        base, split, threshold = parse_run_name(run_dir.name)
        mot = parse_car_summary(summary_path)
        det = read_car_detection_metrics(run_dir)

        row: dict[str, object] = {
            "results_root": str(run_dir.parent),
            "run": run_dir.name,
            "base": base,
            "split": split,
            "threshold": threshold,
            "tracker": tracker,
        }
        row.update({f"car_{field}": mot.get(field, "") for field in MOT_FIELDS})
        row.update({f"car_det_{field}": det.get(field, "") for field in DET_FIELDS})
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_val_selected_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_config: dict[tuple[str, str], dict[str, dict[str, dict[str, object]]]] = {}
    for row in rows:
        split = str(row["split"])
        threshold = str(row["threshold"])
        if split not in {"val", "test"} or not threshold:
            continue
        key = (str(row["results_root"]), str(row["base"]))
        by_config.setdefault(key, {}).setdefault(split, {})[threshold] = row

    selected: list[dict[str, object]] = []
    for (results_root, base), splits in sorted(by_config.items()):
        val_rows = splits.get("val", {})
        test_rows = splits.get("test", {})
        if not val_rows or not test_rows:
            continue
        best_threshold, best_val = max(
            val_rows.items(),
            key=lambda item: float(item[1].get("car_HOTA") or -1),
        )
        test = test_rows.get(best_threshold)
        if test is None:
            continue
        selected.append(
            {
                "results_root": results_root,
                "base": base,
                "selected_threshold": best_threshold,
                "val_car_HOTA": best_val.get("car_HOTA", ""),
                "val_car_MOTA": best_val.get("car_MOTA", ""),
                "val_car_IDF1": best_val.get("car_IDF1", ""),
                "val_car_IDSW": best_val.get("car_IDSW", ""),
                "val_car_CLR_FP": best_val.get("car_CLR_FP", ""),
                "val_car_CLR_FN": best_val.get("car_CLR_FN", ""),
                "test_car_HOTA": test.get("car_HOTA", ""),
                "test_car_MOTA": test.get("car_MOTA", ""),
                "test_car_IDF1": test.get("car_IDF1", ""),
                "test_car_IDSW": test.get("car_IDSW", ""),
                "test_car_CLR_FP": test.get("car_CLR_FP", ""),
                "test_car_CLR_FN": test.get("car_CLR_FN", ""),
                "test_car_det_AP50": test.get("car_det_AP50", ""),
                "test_car_det_precision": test.get("car_det_precision", ""),
                "test_car_det_recall": test.get("car_det_recall", ""),
                "test_car_det_f1": test.get("car_det_f1", ""),
            }
        )
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/car_only_metrics.csv"),
    )
    parser.add_argument(
        "--selected-output-csv",
        type=Path,
        default=Path("results/car_only_val_selected_metrics.csv"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = collect_rows(args.results_root)
    fieldnames = [
        "results_root",
        "run",
        "base",
        "split",
        "threshold",
        "tracker",
        *[f"car_{field}" for field in MOT_FIELDS],
        *[f"car_det_{field}" for field in DET_FIELDS],
    ]
    write_csv(args.output_csv, rows, fieldnames)

    selected_rows = build_val_selected_rows(rows)
    selected_fieldnames = [
        "results_root",
        "base",
        "selected_threshold",
        "val_car_HOTA",
        "val_car_MOTA",
        "val_car_IDF1",
        "val_car_IDSW",
        "val_car_CLR_FP",
        "val_car_CLR_FN",
        "test_car_HOTA",
        "test_car_MOTA",
        "test_car_IDF1",
        "test_car_IDSW",
        "test_car_CLR_FP",
        "test_car_CLR_FN",
        "test_car_det_AP50",
        "test_car_det_precision",
        "test_car_det_recall",
        "test_car_det_f1",
    ]
    write_csv(args.selected_output_csv, selected_rows, selected_fieldnames)

    print(f"Saved {len(rows)} car-only rows to {args.output_csv}")
    print(f"Saved {len(selected_rows)} val-selected rows to {args.selected_output_csv}")
    for row in sorted(selected_rows, key=lambda item: float(item["test_car_HOTA"]), reverse=True)[
        :12
    ]:
        print(
            f"{row['base']}: thr={row['selected_threshold']} "
            f"test car HOTA={float(row['test_car_HOTA']):.4f} "
            f"MOTA={float(row['test_car_MOTA']):.4f} "
            f"IDF1={float(row['test_car_IDF1']):.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
