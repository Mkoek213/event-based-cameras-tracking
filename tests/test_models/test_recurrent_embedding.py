"""Tests for the recurrent embedding head on the lightweight dense detector."""

import torch

from src.models.simple_detector import (
    ConvGRUCell,
    SimpleDenseDetector,
    SimpleDetectorConfig,
    decode_dense_detections,
)

GATED_CONFIG = dict(
    in_channels=8,
    width=8,
    fusion_mode="gated_two_branch",
    component_channels=(2, 6),
)


def test_conv_gru_cell_initialises_and_updates_state():
    cell = ConvGRUCell(4, 6)
    x = torch.randn(2, 4, 10, 12)

    first = cell(x)
    second = cell(x, first)

    assert first.shape == (2, 6, 10, 12)
    assert second.shape == (2, 6, 10, 12)
    assert not torch.equal(first, second)


def test_embedding_head_outputs_normalised_embeddings_and_state():
    model = SimpleDenseDetector(
        SimpleDetectorConfig(**GATED_CONFIG, embedding_dim=16, embedding_recurrent=True)
    )
    outputs = model(torch.randn(2, 8, 96, 128))

    assert set(outputs) == {"cls_logits", "bbox_raw", "embeddings", "embedding_state"}
    assert outputs["embeddings"].shape == (2, 16, 12, 16)
    assert outputs["embedding_state"].shape == (2, 16, 12, 16)
    norms = outputs["embeddings"].norm(dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-4)

    next_outputs = model(torch.randn(2, 8, 96, 128), outputs["embedding_state"])
    assert next_outputs["embedding_state"].shape == (2, 16, 12, 16)


def test_non_recurrent_embedding_head_keeps_state_none():
    model = SimpleDenseDetector(
        SimpleDetectorConfig(**GATED_CONFIG, embedding_dim=16, embedding_recurrent=False)
    )
    outputs = model(torch.randn(1, 8, 96, 128))

    assert outputs["embeddings"].shape == (1, 16, 12, 16)
    assert outputs["embedding_state"] is None
    assert model.embedding_recurrent_cell is None


def test_disabled_embedding_head_returns_only_detection_outputs():
    model = SimpleDenseDetector(SimpleDetectorConfig(**GATED_CONFIG))
    outputs = model(torch.zeros(1, 8, 96, 128))

    assert set(outputs) == {"cls_logits", "bbox_raw"}
    assert model.embedding_proj is None
    assert model.embedding_head is None


def test_old_style_config_dict_constructs_and_loads_checkpoint_state():
    old_config = dict(
        in_channels=8,
        num_classes=7,
        feature_stride=8,
        width=8,
        architecture="simple",
        fusion_mode="gated_two_branch",
        event_frame_channels=2,
        voxel_grid_channels=6,
        component_channels=(2, 6),
    )
    model = SimpleDenseDetector(SimpleDetectorConfig(**old_config))
    state = model.state_dict()

    reloaded = SimpleDenseDetector(SimpleDetectorConfig(**old_config))
    reloaded.load_state_dict(state)
    outputs = reloaded(torch.zeros(1, 8, 96, 128))

    assert set(outputs) == {"cls_logits", "bbox_raw"}


def test_decode_attaches_per_cell_embedding():
    cls_logits = torch.zeros((1, 8, 60, 80))
    bbox_raw = torch.zeros((1, 4, 60, 80))
    cls_logits[0, 3, 5, 6] = 10.0
    embeddings = torch.zeros((1, 16, 60, 80))
    embeddings[0, :, 5, 6] = 0.25

    detections = decode_dense_detections(
        {"cls_logits": cls_logits, "bbox_raw": bbox_raw},
        frame_index=0,
        timestamp=0,
        score_threshold=0.5,
        embeddings=embeddings,
    )

    assert len(detections) == 1
    assert detections[0].embedding == tuple([0.25] * 16)
    assert detections[0].to_dict()["embedding"] == [0.25] * 16


def test_decode_without_embeddings_leaves_record_unchanged():
    cls_logits = torch.zeros((1, 8, 60, 80))
    cls_logits[0, 3, 5, 6] = 10.0

    detections = decode_dense_detections(
        {"cls_logits": cls_logits, "bbox_raw": torch.zeros((1, 4, 60, 80))},
        frame_index=0,
        timestamp=0,
        score_threshold=0.5,
    )

    assert detections[0].embedding is None
    assert "embedding" not in detections[0].to_dict()
