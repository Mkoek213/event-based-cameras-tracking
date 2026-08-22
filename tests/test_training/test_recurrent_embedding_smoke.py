"""Loss, retrieval, and CPU smoke tests for object-level association training."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.sequence_dataset import collate_clip_batch
from src.models.simple_detector import SimpleDenseDetector, SimpleDetectorConfig
from src.training.recurrent_embedding_detector import (
    batch_hard_cosine_triplet_loss,
    class_aware_retrieval_metrics,
    detach_recurrent_state,
    identity_loss,
    is_better_checkpoint,
    load_compatible_embedding_head,
    run_clip_epoch,
)

CLIP_LENGTH = 3
IMAGE_SIZE = 64
GRID_SIZE = IMAGE_SIZE // 8


def make_clip_item(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    events = rng.random((CLIP_LENGTH, 8, IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    cls = np.zeros((CLIP_LENGTH, GRID_SIZE, GRID_SIZE), dtype=np.int64)
    bbox = np.zeros((CLIP_LENGTH, 4, GRID_SIZE, GRID_SIZE), dtype=np.float32)
    pos_mask = np.zeros((CLIP_LENGTH, GRID_SIZE, GRID_SIZE), dtype=bool)
    identity = np.full((CLIP_LENGTH, GRID_SIZE, GRID_SIZE), -1, dtype=np.int64)
    roi_boxes = []
    roi_identities = []
    roi_tracks = []
    roi_classes = []
    for time_index in range(CLIP_LENGTH):
        cls[time_index, 2, 2] = 1
        cls[time_index, 5, 5] = 1
        bbox[time_index, :, 2, 2] = 1.0
        bbox[time_index, :, 5, 5] = 1.0
        pos_mask[time_index, 2, 2] = True
        pos_mask[time_index, 5, 5] = True
        identity[time_index, 2, 2] = 0
        identity[time_index, 5, 5] = 1
        roi_boxes.append(
            np.asarray(
                [[4.0, 4.0, 28.0, 28.0], [36.0, 36.0, 60.0, 60.0]],
                dtype=np.float32,
            )
        )
        roi_identities.append(np.asarray([0, 1], dtype=np.int64))
        roi_tracks.append(np.asarray([10, 20], dtype=np.int64))
        roi_classes.append(np.asarray([0, 0], dtype=np.int64))
    return {
        "events": events,
        "cls": cls,
        "bbox": bbox,
        "pos_mask": pos_mask,
        "identity": identity,
        "roi_boxes": roi_boxes,
        "roi_identity_targets": roi_identities,
        "roi_track_ids": roi_tracks,
        "roi_class_ids": roi_classes,
        "meta": [
            {"sequence": "seq", "frame_index": index, "timestamp": index}
            for index in range(CLIP_LENGTH)
        ],
    }


def make_model(recurrent: bool, head_type: str = "roi") -> SimpleDenseDetector:
    return SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=8,
            width=8,
            fusion_mode="gated_two_branch",
            component_channels=(2, 6),
            embedding_dim=16,
            embedding_hidden_dim=12,
            embedding_head_type=head_type,
            embedding_roi_size=7,
            embedding_recurrent=recurrent,
        )
    )


def test_batch_hard_triplet_selects_hardest_same_class_pairs() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.8, 0.6], [0.6, 0.8]],
        requires_grad=True,
    )
    identities = torch.tensor([0, 0, 1, 1])
    classes = torch.zeros(4, dtype=torch.long)

    loss, anchors = batch_hard_cosine_triplet_loss(embeddings, identities, classes, margin=0.3)

    assert anchors == 4
    assert float(loss.detach()) == pytest.approx(0.62, abs=1e-6)


def test_triplet_never_uses_cross_class_negatives_and_skips_invalid_anchors() -> None:
    embeddings = torch.randn(4, 8, requires_grad=True)
    identities = torch.tensor([0, 0, 1, 1])
    classes = torch.tensor([0, 0, 1, 1])

    loss, anchors = batch_hard_cosine_triplet_loss(embeddings, identities, classes)

    assert anchors == 0
    assert float(loss.detach()) == 0.0
    loss.backward()
    assert embeddings.grad is not None


def test_identity_loss_ignores_unknown_identities() -> None:
    classifier = nn.Linear(8, 2)
    embeddings = torch.randn(2, 8)
    targets = torch.tensor([-1, -1])

    loss, known = identity_loss(embeddings, targets, classifier)
    assert known == 0
    assert float(loss) == 0.0

    loss, known = identity_loss(embeddings, torch.tensor([1, -1]), classifier)
    assert known == 1
    assert torch.isfinite(loss)


def test_class_aware_retrieval_metrics_on_known_example() -> None:
    embeddings = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [-1.0, 0.0],
            [-0.99, 0.01],
            [1.0, 0.0],
        ]
    )
    classes = torch.tensor([0, 0, 0, 0, 1])
    sequences = ["seq"] * 5
    tracks = torch.tensor([10, 10, 20, 20, 99])

    metrics = class_aware_retrieval_metrics(embeddings, classes, sequences, tracks)

    assert metrics["valid_queries"] == 4
    assert metrics["retrieval_map"] == pytest.approx(1.0)
    assert metrics["retrieval_rank1"] == pytest.approx(1.0)


def test_checkpoint_ties_use_rank1_then_lower_detection_loss() -> None:
    incumbent = {
        "retrieval_map": 0.5,
        "retrieval_rank1": 0.6,
        "detection_loss": 2.0,
    }

    assert is_better_checkpoint(
        {"retrieval_map": 0.6, "retrieval_rank1": 0.0, "detection_loss": 9.0},
        incumbent,
    )
    assert is_better_checkpoint(
        {"retrieval_map": 0.5, "retrieval_rank1": 0.7, "detection_loss": 9.0},
        incumbent,
    )
    assert is_better_checkpoint(
        {"retrieval_map": 0.5, "retrieval_rank1": 0.6, "detection_loss": 1.0},
        incumbent,
    )
    assert not is_better_checkpoint(
        {"retrieval_map": 0.5, "retrieval_rank1": 0.6, "detection_loss": 3.0},
        incumbent,
    )


@pytest.mark.parametrize(
    "head_type,recurrent",
    [("roi", False), ("roi", True), ("dense", False), ("dense", True)],
    ids=["R1", "R2", "D1", "D2"],
)
def test_cpu_smoke_epoch_has_finite_losses_updates_weights_and_retrieves(
    head_type: str,
    recurrent: bool,
) -> None:
    torch.manual_seed(0)
    model = make_model(recurrent, head_type)
    classifier = nn.Linear(16, 2)
    loader = DataLoader(
        [make_clip_item(0)],
        batch_size=1,
        collate_fn=collate_clip_batch,
    )
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)
    before = model.embedding_head.weight.detach().clone()

    train = run_clip_epoch(
        model=model,
        classifier=classifier,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        use_amp=False,
        background_weight=0.05,
        bbox_weight=1.0,
        identity_ce_weight=1.0,
        triplet_weight=1.0,
        triplet_margin=0.3,
        grad_clip_norm=5.0,
        grad_accum_steps=1,
        log_every=0,
        epoch=1,
        phase="train",
        component_splits=(2, 6),
        compute_retrieval=False,
    )
    with torch.inference_mode():
        validation = run_clip_epoch(
            model=model,
            classifier=classifier,
            loader=loader,
            device=torch.device("cpu"),
            optimizer=None,
            use_amp=False,
            background_weight=0.05,
            bbox_weight=1.0,
            identity_ce_weight=1.0,
            triplet_weight=1.0,
            triplet_margin=0.3,
            grad_clip_norm=0.0,
            grad_accum_steps=1,
            log_every=0,
            epoch=1,
            phase="val",
            component_splits=(2, 6),
            compute_retrieval=True,
        )

    for key in ("total_loss", "detection_loss", "identity_loss", "triplet_loss"):
        assert math.isfinite(float(train[key]))
    assert train["valid_triplet_anchors"] > 0
    assert validation["valid_queries"] == 6
    assert math.isfinite(float(validation["retrieval_map"]))
    assert math.isfinite(float(validation["retrieval_rank1"]))
    assert not torch.equal(before, model.embedding_head.weight.detach())


def test_frozen_detector_weights_and_batch_norm_stats_do_not_change() -> None:
    torch.manual_seed(3)
    model = make_model(recurrent=False)
    model.set_detector_trainable(False)
    classifier = nn.Linear(16, 2)
    loader = DataLoader(
        [make_clip_item(1)],
        batch_size=1,
        collate_fn=collate_clip_batch,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in list(model.parameters()) + list(classifier.parameters())
            if parameter.requires_grad
        ],
        lr=1e-3,
    )
    detector_before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("embedding_")
    }
    embedding_before = model.embedding_head.weight.detach().clone()

    run_clip_epoch(
        model=model,
        classifier=classifier,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        use_amp=False,
        background_weight=0.05,
        bbox_weight=1.0,
        identity_ce_weight=1.0,
        triplet_weight=1.0,
        triplet_margin=0.3,
        grad_clip_norm=5.0,
        grad_accum_steps=1,
        log_every=0,
        epoch=1,
        phase="train",
        component_splits=(2, 6),
        compute_retrieval=False,
    )

    detector_after = model.state_dict()
    assert all(torch.equal(value, detector_after[key]) for key, value in detector_before.items())
    assert not torch.equal(embedding_before, model.embedding_head.weight.detach())


@pytest.mark.parametrize("recurrent", [False, True], ids=["D1", "D2"])
def test_initial_dense_head_load_does_not_replace_detector(
    tmp_path: Path,
    recurrent: bool,
) -> None:
    torch.manual_seed(7)
    source = make_model(recurrent, "dense")
    target = make_model(recurrent, "dense")
    with torch.no_grad():
        for name, parameter in source.named_parameters():
            if name.startswith("embedding_"):
                parameter.fill_(0.125)
    checkpoint = tmp_path / "dense_head.pt"
    torch.save({"model_state": source.state_dict()}, checkpoint)
    detector_before = {
        key: value.detach().clone()
        for key, value in target.state_dict().items()
        if not key.startswith("embedding_")
    }

    loaded, skipped = load_compatible_embedding_head(
        target,
        checkpoint,
        torch.device("cpu"),
    )

    assert loaded > 0
    assert skipped == 0
    assert all(
        torch.equal(value, target.state_dict()[key]) for key, value in detector_before.items()
    )
    assert torch.allclose(
        target.embedding_head.weight,
        torch.full_like(target.embedding_head.weight, 0.125),
    )


def test_temporal_convlstm_smoke_updates_only_adapter_and_embedding_head() -> None:
    torch.manual_seed(19)
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=8,
            width=8,
            fusion_mode="gated_two_branch",
            component_channels=(2, 6),
            embedding_dim=16,
            embedding_hidden_dim=12,
            embedding_head_type="dense",
            embedding_recurrent=False,
            temporal_recurrence_locations=("neck",),
            temporal_recurrence_type="convlstm",
            temporal_recurrence_mode="residual",
        )
    )
    model.set_detector_trainable(False)
    model.set_temporal_trainable(True)
    classifier = nn.Linear(16, 2)
    loader = DataLoader(
        [make_clip_item(4)],
        batch_size=1,
        collate_fn=collate_clip_batch,
    )
    optimizer = torch.optim.AdamW(
        [
            parameter
            for parameter in list(model.parameters()) + list(classifier.parameters())
            if parameter.requires_grad
        ],
        lr=1e-3,
    )
    base_before = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if not key.startswith("embedding_") and not key.startswith("temporal_adapters.")
    }
    gain_before = model.temporal_adapters["neck"].residual_gain.detach().clone()

    stats = run_clip_epoch(
        model=model,
        classifier=classifier,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        use_amp=False,
        background_weight=0.05,
        bbox_weight=1.0,
        identity_ce_weight=1.0,
        triplet_weight=1.0,
        triplet_margin=0.3,
        grad_clip_norm=5.0,
        grad_accum_steps=1,
        log_every=0,
        epoch=1,
        phase="train",
        component_splits=(2, 6),
        compute_retrieval=False,
    )

    assert math.isfinite(float(stats["total_loss"]))
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in base_before.items())
    assert not torch.equal(
        gain_before,
        model.temporal_adapters["neck"].residual_gain.detach(),
    )


def make_detector_only_temporal_model() -> SimpleDenseDetector:
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=8,
            width=8,
            fusion_mode="gated_two_branch",
            component_channels=(2, 6),
            embedding_dim=0,
            temporal_recurrence_locations=("neck",),
            temporal_recurrence_type="convgru",
            temporal_recurrence_mode="direct",
        )
    )
    model.set_detector_trainable(False)
    model.set_temporal_trainable(True)
    return model


def test_detach_recurrent_state_handles_lstm_tuple_and_location_dictionary() -> None:
    tensor = torch.randn(1, 2, 3, 3, requires_grad=True)

    detached = detach_recurrent_state({"neck": (tensor, tensor + 1.0)})

    assert isinstance(detached, dict)
    assert isinstance(detached["neck"], tuple)
    assert all(not item.requires_grad for item in detached["neck"])


def test_detector_only_recurrence_carries_detached_state_between_contiguous_clips() -> None:
    first = make_clip_item(21)
    second = make_clip_item(22)
    second["meta"] = [
        {"sequence": "seq", "frame_index": index + CLIP_LENGTH, "timestamp": index}
        for index in range(CLIP_LENGTH)
    ]
    model = make_detector_only_temporal_model()
    loader = DataLoader([first, second], batch_size=1, collate_fn=collate_clip_batch)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=1e-3
    )

    stats = run_clip_epoch(
        model=model,
        classifier=None,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        use_amp=False,
        background_weight=0.05,
        bbox_weight=1.0,
        identity_ce_weight=0.0,
        triplet_weight=0.0,
        triplet_margin=0.3,
        grad_clip_norm=5.0,
        grad_accum_steps=1,
        log_every=0,
        epoch=1,
        phase="train",
        component_splits=(2, 6),
        compute_retrieval=False,
        carry_state_between_clips=True,
        burn_in_frames=1,
    )

    assert stats["state_resets"] == 1
    assert stats["state_carries"] == 1
    assert stats["supervised_frames"] == 5
    assert stats["identity_loss"] == 0.0
    assert stats["triplet_loss"] == 0.0
    assert math.isfinite(float(stats["detection_loss"]))


def test_reset_memory_applies_burn_in_to_every_clip() -> None:
    model = make_detector_only_temporal_model()
    loader = DataLoader(
        [make_clip_item(31), make_clip_item(32)],
        batch_size=1,
        collate_fn=collate_clip_batch,
    )
    with torch.inference_mode():
        stats = run_clip_epoch(
            model=model,
            classifier=None,
            loader=loader,
            device=torch.device("cpu"),
            optimizer=None,
            use_amp=False,
            background_weight=0.05,
            bbox_weight=1.0,
            identity_ce_weight=0.0,
            triplet_weight=0.0,
            triplet_margin=0.3,
            grad_clip_norm=0.0,
            grad_accum_steps=1,
            log_every=0,
            epoch=1,
            phase="val",
            component_splits=(2, 6),
            compute_retrieval=False,
            carry_state_between_clips=False,
            burn_in_frames=2,
        )

    assert stats["state_resets"] == 2
    assert stats["state_carries"] == 0
    assert stats["supervised_frames"] == 2
