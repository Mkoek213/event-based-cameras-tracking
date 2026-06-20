#!/usr/bin/env python3
"""Select thresholds by validation HOTA and print corresponding test results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("results/dsec_mot_trackeval_pretrained_detector"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    validation: dict[str, list[tuple[float, str, dict]]] = {}
    pattern = re.compile(r"^(?P<label>.+)_val_thr(?P<threshold>\d+)$")
    for path in args.results_root.glob("*_val_thr*/metrics_summary.json"):
        match = pattern.match(path.parent.name)
        if not match:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))["aggregate"]
        validation.setdefault(match.group("label"), []).append(
            (float(metrics["HOTA"]), match.group("threshold"), metrics)
        )

    header = [
        "variant",
        "threshold",
        "val_HOTA",
        "test_HOTA",
        "test_MOTA",
        "test_IDF1",
        "IDS",
        "FP",
        "FN",
    ]
    print("\t".join(header))
    for label, candidates in sorted(validation.items()):
        _, threshold, val = max(candidates)
        test_path = args.results_root / f"{label}_test_thr{threshold}" / "metrics_summary.json"
        if not test_path.exists():
            print(f"{label}\t{threshold}\t{val['HOTA']:.4f}\tMISSING_TEST")
            continue
        test = json.loads(test_path.read_text(encoding="utf-8"))["aggregate"]
        print(
            f"{label}\t{threshold}\t{val['HOTA']:.4f}\t{test['HOTA']:.4f}\t"
            f"{test['MOTA']:.4f}\t{test['IDF1']:.4f}\t{test['IDS']}\t{test['FP']}\t{test['FN']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
