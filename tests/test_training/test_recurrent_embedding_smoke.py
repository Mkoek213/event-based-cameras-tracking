"""Loss, retrieval, and CPU smoke tests for object-level association training."""

import math

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
    identity_loss,
    is_better_checkpoint,
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


def make_model(recurrent: bool) -> SimpleDenseDetector:
    return SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=8,
            width=8,
            fusion_mode="gated_two_branch",
            component_channels=(2, 6),
            embedding_dim=16,
            embedding_hidden_dim=12,
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


@pytest.mark.parametrize("recurrent", [False, True], ids=["R1", "R2"])
def test_cpu_smoke_epoch_has_finite_losses_updates_weights_and_retrieves(
    recurrent: bool,
) -> None:
    torch.manual_seed(0)
    model = make_model(recurrent)
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
