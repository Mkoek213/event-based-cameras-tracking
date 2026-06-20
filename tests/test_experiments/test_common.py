from pathlib import Path

from src.experiments.common import (
    EvalTarget,
    VariantSpec,
    simple_detector_eval_command,
    simple_detector_train_command,
    sweep_label,
    threshold_label,
    variant_label,
)


def test_variant_and_sweep_labels_are_stable() -> None:
    assert threshold_label(0.95) == "095"
    assert sweep_label(5, 50_000) == "bins5_win50ms"
    assert variant_label("event_frame", "single") == "event_frame"
    assert (
        variant_label("event_frame_voxel_grid", "gated_two_branch")
        == "event_frame_voxel_grid_gated_two_branch"
    )


def test_variant_checkpoint_name_matches_training_layout() -> None:
    assert VariantSpec("event_frame").checkpoint_name(num_bins=5, width=32) == (
        "event_frame_bins5_w32"
    )
    assert (
        VariantSpec("event_frame_voxel_grid", "two_branch").checkpoint_name(
            num_bins=5,
            width=32,
        )
        == "event_frame_voxel_grid_bins5_w32_two_branch"
    )


def test_simple_detector_train_command_includes_optional_arguments() -> None:
    command = simple_detector_train_command(
        python="python",
        root=Path("data/datasets/dsec_mot"),
        representation="event_frame",
        fusion_mode="single",
        num_bins=5,
        time_window_us=50_000,
        epochs=30,
        batch_size=16,
        num_workers=4,
        width=32,
        device="cuda",
        output_dir=Path("runs/out"),
        eros_cache_root=Path("data/cache/dsec_mot_eros"),
        class_ids=[0],
        num_classes=1,
    )

    assert command[:3] == ["python", "-m", "src.training.simple_detector"]
    assert command[command.index("--representation") + 1] == "event_frame"
    assert command[command.index("--class-ids") + 1] == "0"
    assert command[command.index("--num-classes") + 1] == "1"
    assert command[command.index("--eros-cache-root") + 1] == "data/cache/dsec_mot_eros"


def test_simple_detector_eval_command_includes_target_and_classes() -> None:
    command = simple_detector_eval_command(
        python="python",
        checkpoint=Path("runs/model/best.pt"),
        root=Path("data/datasets/dsec_mot"),
        target=EvalTarget("test", "interlaken_00_d", "test"),
        threshold=0.95,
        device="cuda",
        max_detections=50,
        output_root=Path("results/out"),
        run_name="example",
        classes_to_eval=["car"],
    )

    assert command[:3] == ["python", "-m", "src.evaluation.simple_detector_trackeval_cli"]
    assert command[command.index("--split") + 1] == "test"
    assert command[command.index("--sequences") + 1] == "interlaken_00_d"
    assert command[command.index("--classes-to-eval") + 1] == "car"
