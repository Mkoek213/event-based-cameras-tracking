"""Tests for contribution and temporal-memory benchmark commands."""

from argparse import Namespace
from pathlib import Path

from src.experiments.novelty_contribution_benchmark import (
    SCOPES,
    contribution_variants,
    evaluate_command,
    memory_variants,
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
        memory_effective_frames=64,
        num_workers=0,
        device="cuda",
        max_train_clips=0,
        max_val_clips=0,
        max_eval_frames=0,
        max_detections=100,
        max_embedding_pairs=200_000,
        recurrence_variant="gru_direct_neck",
        overwrite=False,
    )


def command_value(command: tuple[str, ...] | list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def test_contribution_matrix_isolates_three_modules_and_combines_them() -> None:
    variants = {variant.name: variant for variant in contribution_variants()}

    assert set(variants) == {
        "baseline",
        "gating_only",
        "embedding_only",
        "recurrence_only",
        "all_modules",
    }
    assert variants["baseline"].static_checkpoint
    assert variants["baseline"].fusion_mode == "two_branch"
    assert variants["gating_only"].fusion_mode == "gated_two_branch"
    assert variants["embedding_only"].embedding and not variants["embedding_only"].recurrence
    assert variants["recurrence_only"].recurrence and not variants["recurrence_only"].embedding
    assert variants["all_modules"].embedding and variants["all_modules"].recurrence
    assert variants["all_modules"].fusion_mode == "gated_two_branch"


def test_isolated_training_commands_do_not_mix_embedding_and_recurrence(tmp_path: Path) -> None:
    variants = {variant.name: variant for variant in contribution_variants()}
    args = runner_args(tmp_path)

    embedding = training_command(args, SCOPES["all_classes"], variants["embedding_only"])
    recurrence = training_command(args, SCOPES["all_classes"], variants["recurrence_only"])
    combined = training_command(args, SCOPES["all_classes"], variants["all_modules"])

    assert command_value(embedding, "--fusion-mode") == "two_branch"
    assert command_value(embedding, "--embedding-dim") == "128"
    assert "--temporal-recurrence-locations" not in embedding
    assert command_value(recurrence, "--embedding-dim") == "0"
    assert command_value(recurrence, "--identity-ce-weight") == "0.0"
    assert command_value(recurrence, "--triplet-weight") == "0.0"
    assert command_value(recurrence, "--temporal-recurrence-locations") == "neck"
    assert command_value(recurrence, "--temporal-recurrence-mode") == "direct"
    assert command_value(combined, "--fusion-mode") == "gated_two_branch"
    assert command_value(combined, "--embedding-dim") == "128"
    assert "--train-temporal-adapters" in combined
    assert "--freeze-detector" in embedding
    assert "--freeze-detector" in recurrence
    assert "--freeze-detector" in combined


def test_memory_matrix_has_three_horizons_reset_carry_and_burn_in() -> None:
    variants = {variant.name: variant for variant in memory_variants()}

    assert len(variants) == 8
    for length in (8, 16, 32):
        assert variants[f"clip{length}_reset"].clip_length == length
        assert not variants[f"clip{length}_reset"].carry_state
        assert variants[f"clip{length}_carry"].carry_state
    assert variants["clip16_reset_burn4"].burn_in_frames == 4
    assert variants["clip16_carry_burn4"].carry_state
    assert variants["clip16_carry_burn4"].burn_in_frames == 4


def test_memory_commands_hold_frames_per_optimizer_step_constant(tmp_path: Path) -> None:
    args = runner_args(tmp_path)
    variants = {variant.name: variant for variant in memory_variants()}

    for length, expected_accumulation in ((8, "8"), (16, "4"), (32, "2")):
        command = training_command(args, SCOPES["car_only"], variants[f"clip{length}_carry"])
        assert command_value(command, "--batch-size") == "1"
        assert command_value(command, "--grad-accum-steps") == expected_accumulation
        assert command_value(command, "--clip-length") == str(length)
        assert command_value(command, "--clip-stride") == str(length)
        assert "--ordered-clips" in command
        assert "--carry-state-between-clips" in command
        assert command_value(command, "--class-ids") == "0"


def test_evaluation_uses_component_normalisation_and_exact_checkpoint_cache(
    tmp_path: Path,
) -> None:
    variant = next(item for item in contribution_variants() if item.name == "all_modules")
    checkpoint = tmp_path / "epoch_001.pt"
    checkpoint.write_bytes(b"checkpoint")

    command, _ = evaluate_command(
        runner_args(tmp_path),
        SCOPES["all_classes"],
        variant,
        checkpoint,
        "epoch_001",
        "reid",
        "val",
        0.9,
        0.25,
        0.5,
    )

    assert command_value(command, "--input-normalisation") == "component"
    assert "--track-with-reid" in command
    cache_root = command_value(command, "--detection-cache-root")
    assert "epoch_001_val_score090" in cache_root
    assert str(checkpoint.stat().st_size) in cache_root
