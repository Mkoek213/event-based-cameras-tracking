"""Tests for the multi-tracker benchmark runner."""

import sys
from pathlib import Path

from src.experiments.simple_detector_tracker_benchmark import (
    build_val_selected_rows,
    main,
)


def metric_row(
    *,
    split: str,
    threshold: str,
    hota: float,
    idf1: float,
) -> dict[str, object]:
    return {
        "run": f"example_iou_{split}_thr{threshold}",
        "base": "example",
        "tracker": "iou",
        "split": split,
        "threshold": threshold,
        "HOTA": hota,
        "MOTA": 0.1,
        "IDF1": idf1,
        "IDS": 1,
        "FP": 2,
        "FN": 3,
    }


def test_validation_selection_uses_matching_test_threshold() -> None:
    rows = [
        metric_row(split="val", threshold="050", hota=0.2, idf1=0.3),
        metric_row(split="val", threshold="090", hota=0.4, idf1=0.5),
        metric_row(split="test", threshold="050", hota=0.7, idf1=0.8),
        metric_row(split="test", threshold="090", hota=0.6, idf1=0.7),
    ]

    selected = build_val_selected_rows(rows)

    assert len(selected) == 1
    assert selected[0]["selected_threshold"] == "090"
    assert selected[0]["val_HOTA"] == 0.4
    assert selected[0]["test_HOTA"] == 0.6


def test_dry_run_covers_trackers_thresholds_and_both_splits(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "simple_detector_tracker_benchmark",
            "--checkpoint",
            str(tmp_path / "missing.pt"),
            "--trackers",
            "iou",
            "boxmot_botsort",
            "--thresholds",
            "0.5",
            "0.9",
            "--device",
            "cpu",
            "--results-root",
            str(tmp_path / "results"),
            "--log-dir",
            str(tmp_path / "logs"),
            "--dry-run",
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out

    assert output.count("-m src.evaluation.simple_detector_trackeval_cli") == 8
    assert "--sequences zurich_city_01_d" in output
    assert "--sequences interlaken_00_d zurich_city_00_b" in output
    assert "--tracker-backend boxmot_botsort" in output
