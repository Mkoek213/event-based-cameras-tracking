"""Tests for the event-ReID benchmark experiment matrix and selection policy."""

import argparse
import json
import sys
from pathlib import Path

import torch

from src.experiments.event_reid_embedding_benchmark import (
    SCOPES,
    EvaluationSpec,
    build_training_command,
    evaluation_command,
    main,
    select_validation_configuration,
    selected_scopes,
    validation_matrix,
    write_aggregate_outputs,
)


def runner_args(**overrides: object) -> argparse.Namespace:
    values = {
        "epochs": 30,
        "batch_size": 2,
        "grad_accum_steps": 4,
        "num_workers": 0,
        "device": "cuda",
        "resume": False,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_training_matrix_has_four_exact_r1_r2_commands(tmp_path: Path) -> None:
    commands = [
        build_training_command(
            "python",
            Path("data/datasets/dsec_mot"),
            tmp_path / "runs",
            scope,
            variant,
            runner_args(),
        )
        for scope in selected_scopes("both")
        for variant in ("R1", "R2")
    ]

    assert len(commands) == 4
    assert all(
        command[:3] == ["python", "-m", "src.training.recurrent_embedding_detector"]
        for command in commands
    )
    assert {command[command.index("--run-name") + 1] for command in commands} == {
        "event_frame_voxel_grid_bins3_w32_gated_two_branch_r1_non_recurrent",
        "event_frame_voxel_grid_bins3_w32_gated_two_branch_r2_recurrent",
    }
    assert sum("--no-recurrent-embedding" in command for command in commands) == 2
    assert sum("--recurrent-embedding" in command for command in commands) == 2
    assert all(command[command.index("--embedding-dim") + 1] == "256" for command in commands)
    assert all(command[command.index("--roi-size") + 1] == "7" for command in commands)
    car_commands = [command for command in commands if "--class-ids" in command]
    assert len(car_commands) == 2
    assert all(command[command.index("--class-ids") + 1] == "0" for command in car_commands)


def test_validation_matrix_covers_both_scopes_variants_modes_and_full_grids() -> None:
    matrix = validation_matrix(selected_scopes("both"))

    assert len(matrix) == 390
    assert {spec.scope for spec in matrix} == {"all_classes", "car_only"}
    assert {spec.variant for spec in matrix} == {"R0", "R1", "R2"}
    assert {spec.mode for spec in matrix} == {"motion", "reid"}
    assert sum(spec.scope == "all_classes" for spec in matrix) == 234
    assert sum(spec.scope == "car_only" for spec in matrix) == 156
    assert sum(spec.variant == "R0" and spec.with_reid for spec in matrix) == 0
    assert all(
        "score" in spec.run_name and "appearance" in spec.run_name and "proximity" in spec.run_name
        for spec in matrix
    )


def test_test_command_combines_both_sequences_and_passes_reid_thresholds(
    tmp_path: Path,
) -> None:
    spec = EvaluationSpec(
        scope="all_classes",
        variant="R2",
        mode="reid",
        split="test",
        score_threshold=0.5,
        appearance_threshold=0.2,
        proximity_threshold=0.7,
        with_reid=True,
    )
    command = evaluation_command(
        "python",
        Path("data/datasets/dsec_mot"),
        tmp_path,
        Path("best.pt"),
        SCOPES["all_classes"],
        spec,
        "cuda",
        100,
    )

    sequence_index = command.index("--sequences")
    assert command[sequence_index + 1 : sequence_index + 3] == [
        "interlaken_00_d",
        "zurich_city_00_b",
    ]
    assert "--track-with-reid" in command
    assert command[command.index("--track-appearance-thresh") + 1] == "0.2"
    assert command[command.index("--track-proximity-thresh") + 1] == "0.7"


def test_validation_selection_uses_hota_then_assa_then_idf1(tmp_path: Path) -> None:
    candidates = [
        EvaluationSpec("all_classes", "R1", "motion", "val", score, 0.25, 0.5, False)
        for score in (0.1, 0.2, 0.3)
    ]
    metrics = [
        {"HOTA": 0.5, "AssA": 0.8, "IDF1": 0.9},
        {"HOTA": 0.6, "AssA": 0.1, "IDF1": 0.1},
        {"HOTA": 0.6, "AssA": 0.2, "IDF1": 0.0},
    ]
    for spec, aggregate in zip(candidates, metrics):
        directory = tmp_path / spec.run_name
        directory.mkdir()
        (directory / "metrics_summary.json").write_text(
            json.dumps({"aggregate": aggregate}),
            encoding="utf-8",
        )

    selected, _ = select_validation_configuration(tmp_path, candidates)

    assert selected.score_threshold == 0.3


def test_runner_dry_run_prints_four_train_commands_and_full_matrix(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "event_reid_embedding_benchmark",
            "--dry-run",
            "--device",
            "cpu",
            "--runs-root",
            str(tmp_path / "runs"),
            "--results-root",
            str(tmp_path / "results"),
            "--log-dir",
            str(tmp_path / "logs"),
        ],
    )

    assert main() == 0
    output = capsys.readouterr().out

    assert output.count("-m src.training.recurrent_embedding_detector") == 4
    assert "Validation matrix contains 390 runs." in output
    assert output.count("-m src.evaluation.simple_detector_trackeval_cli") == 390
    assert "Test stage requires completed validation summaries" in output


def test_aggregate_outputs_include_required_combined_and_per_sequence_fields(
    tmp_path: Path,
) -> None:
    runs_root = tmp_path / "runs"
    results_root = tmp_path / "results"
    checkpoint = (
        runs_root
        / "all_classes"
        / "event_frame_voxel_grid_bins3_w32_gated_two_branch_r1_non_recurrent"
        / "best.pt"
    )
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {
            "selected_epoch": 7,
            "val_retrieval_map": 0.55,
            "val_retrieval_rank1": 0.75,
        },
        checkpoint,
    )
    validation = EvaluationSpec("all_classes", "R1", "reid", "val", 0.5, 0.2, 0.7, True)
    test = EvaluationSpec("all_classes", "R1", "reid", "test", 0.5, 0.2, 0.7, True)
    test_directory = results_root / test.run_name
    test_directory.mkdir(parents=True)
    combined = {
        "HOTA": 0.4,
        "AssA": 0.5,
        "DetA": 0.3,
        "MOTA": 0.2,
        "IDF1": 0.6,
        "IDS": 4,
        "FP": 5,
        "FN": 6,
    }
    (test_directory / "metrics_summary.json").write_text(
        json.dumps(
            {
                "aggregate": combined,
                "per_sequence": {
                    "interlaken_00_d": {"metrics": combined},
                    "zurich_city_00_b": {"metrics": combined},
                },
            }
        ),
        encoding="utf-8",
    )
    selections = {
        ("all_classes", "R1", "reid"): (
            validation,
            {"aggregate": {"HOTA": 0.45, "AssA": 0.52, "IDF1": 0.61}},
        )
    }

    write_aggregate_outputs(results_root, runs_root, selections)

    payload = json.loads((results_root / "benchmark_summary.json").read_text(encoding="utf-8"))
    record = payload["records"][0]
    assert record["selected_epoch"] == 7
    assert record["retrieval_map"] == 0.55
    assert record["reid_enabled"] is True
    assert record["combined_test"]["DetA"] == 0.3
    assert set(record["per_sequence_test"]) == {
        "interlaken_00_d",
        "zurich_city_00_b",
    }
    csv_lines = (results_root / "benchmark_summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(csv_lines) == 4
    assert "COMBINED" in csv_lines[-1]
