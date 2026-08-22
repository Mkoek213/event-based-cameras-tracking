"""Tests for the staged recurrent-placement ablation runner."""

from argparse import Namespace
from pathlib import Path

from src.experiments.recurrent_placement_ablation import (
    CONTROL_VARIANT,
    PLACEMENTS,
    SCOPES,
    VARIANTS,
    evaluate_command,
    training_command,
)


def runner_args(tmp_path: Path) -> Namespace:
    return Namespace(
        root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        results_root=tmp_path / "results",
        epochs=30,
        batch_size=2,
        grad_accum_steps=4,
        num_workers=0,
        device="cuda",
        max_train_clips=0,
        max_val_clips=0,
        max_eval_frames=0,
        max_detections=100,
        overwrite=False,
    )


def test_matrix_contains_three_residual_cells_and_direct_gru() -> None:
    assert len(VARIANTS) == 24
    assert len({variant.name for variant in VARIANTS}) == 24

    residual = [variant for variant in VARIANTS if variant.mode == "residual"]
    direct = [variant for variant in VARIANTS if variant.mode == "direct"]

    assert len(residual) == 3 * len(PLACEMENTS)
    assert {variant.cell_type for variant in residual} == {
        "convgru",
        "convlstm",
        "convrnn",
    }
    assert len(direct) == len(PLACEMENTS)
    assert {variant.cell_type for variant in direct} == {"convgru"}
    assert {variant.placement for variant in VARIANTS} == set(PLACEMENTS)


def test_training_command_preserves_small_gated_detector_and_freezes_r0(
    tmp_path: Path,
) -> None:
    variant = next(item for item in VARIANTS if item.name == "lstm_res_all")
    command = training_command(
        "python",
        runner_args(tmp_path),
        SCOPES["all_classes"],
        variant,
    )

    assert command[:3] == (
        "python",
        "-m",
        "src.training.recurrent_embedding_detector",
    )
    assert command[command.index("--fusion-mode") + 1] == "gated_two_branch"
    assert command[command.index("--model-width") + 1] == "32"
    assert command[command.index("--epochs") + 1] == "30"
    assert command[command.index("--temporal-recurrence-type") + 1] == "convlstm"
    assert command[command.index("--temporal-recurrence-mode") + 1] == "residual"
    assert command[command.index("--temporal-recurrence-locations") + 1] == ",".join(
        PLACEMENTS["all"]
    )
    assert "--freeze-detector" in command
    assert "--train-temporal-adapters" in command
    assert "--save-every-epoch" in command
    assert "--resume" in command


def test_matched_control_trains_embedding_head_without_temporal_adapter(
    tmp_path: Path,
) -> None:
    command = training_command(
        "python",
        runner_args(tmp_path),
        SCOPES["all_classes"],
        CONTROL_VARIANT,
    )

    assert command[command.index("--run-name") + 1] == "control"
    assert "--initial-model-checkpoint" in command
    assert "--freeze-detector" in command
    assert "--temporal-recurrence-locations" not in command
    assert "--train-temporal-adapters" not in command


def test_car_only_command_keeps_identical_protocol_but_filters_class(
    tmp_path: Path,
) -> None:
    variant = next(item for item in VARIANTS if item.name == "gru_direct_neck")
    command = training_command(
        "python",
        runner_args(tmp_path),
        SCOPES["car_only"],
        variant,
    )

    assert command[command.index("--class-ids") + 1] == "0"
    assert command[command.index("--num-classes") + 1] == "1"
    assert command[command.index("--clip-length") + 1] == "8"
    assert command[command.index("--clip-stride") + 1] == "8"


def test_evaluation_cache_is_scoped_to_exact_checkpoint_and_score(tmp_path: Path) -> None:
    checkpoint = tmp_path / "epoch_001.pt"
    checkpoint.write_bytes(b"checkpoint")

    command, _ = evaluate_command(
        runner_args(tmp_path),
        SCOPES["all_classes"],
        "gru_res_neck",
        checkpoint,
        "epoch_001",
        "reid",
        "val",
        0.9,
        0.25,
        0.5,
    )

    cache_root = command[command.index("--detection-cache-root") + 1]
    assert "epoch_001_val_score090" in cache_root
    assert str(checkpoint.stat().st_size) in cache_root
    assert "--detection-cache-score-threshold" not in command
