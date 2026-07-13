"""Smoke tests for the recurrent embedding trainer loop."""

import math

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.data.sequence_dataset import collate_clip_batch
from src.models.simple_detector import SimpleDenseDetector, SimpleDetectorConfig
from src.training.recurrent_embedding_detector import identity_loss, run_clip_epoch

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
    for t in range(CLIP_LENGTH):
        cls[t, 2, 3] = 1
        bbox[t, :, 2, 3] = [1.0, 1.0, 1.0, 1.0]
        pos_mask[t, 2, 3] = True
        identity[t, 2, 3] = t % 2
    return {
        "events": events,
        "cls": cls,
        "bbox": bbox,
        "pos_mask": pos_mask,
        "identity": identity,
        "meta": [{"sequence": "seq", "frame_index": t, "timestamp": t} for t in range(CLIP_LENGTH)],
    }


def make_model() -> SimpleDenseDetector:
    return SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=8,
            width=8,
            fusion_mode="gated_two_branch",
            component_channels=(2, 6),
            embedding_dim=8,
            embedding_recurrent=True,
        )
    )


def test_run_clip_epoch_computes_finite_loss_and_updates_weights():
    torch.manual_seed(0)
    model = make_model()
    classifier = nn.Linear(8, 2)
    loader = DataLoader(
        [make_clip_item(seed) for seed in range(2)],
        batch_size=1,
        collate_fn=collate_clip_batch,
    )
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(classifier.parameters()), lr=1e-3)
    before = model.embedding_head.weight.detach().clone()

    stats = run_clip_epoch(
        model=model,
        classifier=classifier,
        loader=loader,
        device=torch.device("cpu"),
        optimizer=optimizer,
        use_amp=False,
        background_weight=0.05,
        bbox_weight=1.0,
        embedding_loss_weight=1.0,
        grad_clip_norm=5.0,
        grad_accum_steps=1,
        log_every=0,
        epoch=1,
        phase="train",
        component_splits=(2, 6),
    )

    assert math.isfinite(stats["loss"])
    assert math.isfinite(stats["det_loss"])
    assert math.isfinite(stats["id_loss"])
    assert stats["id_loss"] > 0.0
    assert stats["identity_cells"] > 0
    assert not torch.equal(before, model.embedding_head.weight.detach())


def test_identity_loss_ignores_unknown_identities():
    classifier = nn.Linear(8, 2)
    embeddings = torch.randn(1, 8, 4, 4)
    targets = torch.full((1, 4, 4), -1, dtype=torch.long)

    loss, known_cells = identity_loss(embeddings, targets, classifier)

    assert known_cells == 0
    assert float(loss) == 0.0

    targets[0, 1, 1] = 1
    loss, known_cells = identity_loss(embeddings, targets, classifier)
    assert known_cells == 1
    assert torch.isfinite(loss)
