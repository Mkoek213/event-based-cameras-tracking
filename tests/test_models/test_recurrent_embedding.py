"""Tests for object-level R1/R2 association embedding heads."""

import pytest
import torch

from src.models.simple_detector import (
    ConvGRUCell,
    SimpleDenseDetector,
    SimpleDetectorConfig,
    attach_detection_embeddings,
    decode_dense_detections,
    detection_boxes_xyxy,
)

GATED_CONFIG = {
    "in_channels": 8,
    "width": 8,
    "fusion_mode": "gated_two_branch",
    "component_channels": (2, 6),
    "embedding_dim": 256,
    "embedding_hidden_dim": 128,
    "embedding_roi_size": 7,
}


def make_model(recurrent: bool) -> SimpleDenseDetector:
    return SimpleDenseDetector(SimpleDetectorConfig(**GATED_CONFIG, embedding_recurrent=recurrent))


def test_conv_gru_cell_initialises_and_updates_state() -> None:
    cell = ConvGRUCell(4, 6)
    inputs = torch.randn(2, 4, 10, 12)

    first = cell(inputs)
    second = cell(inputs, first)

    assert first.shape == (2, 6, 10, 12)
    assert second.shape == (2, 6, 10, 12)
    assert not torch.equal(first, second)


def test_r1_outputs_feature_map_without_recurrent_state() -> None:
    model = make_model(recurrent=False)
    outputs = model(torch.randn(2, 8, 64, 64))

    assert set(outputs) == {
        "cls_logits",
        "bbox_raw",
        "embedding_feature_map",
        "embedding_state",
    }
    assert outputs["embedding_feature_map"].shape == (2, 128, 8, 8)
    assert outputs["embedding_state"] is None


def test_r2_outputs_feature_map_and_changing_state() -> None:
    torch.manual_seed(0)
    model = make_model(recurrent=True)
    inputs = torch.randn(2, 8, 64, 64)

    first = model(inputs)
    second = model(inputs, first["embedding_state"])
    reset = model(inputs)

    assert first["embedding_feature_map"].shape == (2, 128, 8, 8)
    assert first["embedding_state"].shape == (2, 128, 8, 8)
    assert not torch.equal(first["embedding_state"], second["embedding_state"])
    assert torch.allclose(first["embedding_state"], reset["embedding_state"])


def test_roi_embeddings_are_256d_unit_norm_and_keep_box_order() -> None:
    torch.manual_seed(1)
    model = make_model(recurrent=False).eval()
    outputs = model(torch.randn(2, 8, 64, 64))
    feature_map = outputs["embedding_feature_map"]
    boxes = [
        torch.tensor([[0.0, 0.0, 24.0, 24.0], [32.0, 32.0, 64.0, 64.0]]),
        torch.tensor([[8.0, 8.0, 40.0, 48.0]]),
    ]

    descriptors = model.extract_roi_embeddings(feature_map, boxes)
    first_image = model.extract_roi_embeddings(feature_map[:1], [boxes[0]])
    second_image = model.extract_roi_embeddings(feature_map[1:], [boxes[1]])

    assert descriptors.shape == (3, 256)
    assert torch.allclose(descriptors.norm(dim=1), torch.ones(3), atol=1e-5)
    assert torch.allclose(descriptors, torch.cat([first_image, second_image]), atol=1e-6)


def test_empty_and_single_roi_are_safe_during_training() -> None:
    model = make_model(recurrent=False).train()
    feature_map = model(torch.randn(1, 8, 64, 64))["embedding_feature_map"]

    empty = model.extract_roi_embeddings(feature_map, [torch.empty((0, 4))])
    one = model.extract_roi_embeddings(
        feature_map,
        [torch.tensor([[4.0, 4.0, 40.0, 40.0]])],
    )

    assert empty.shape == (0, 256)
    assert empty.device == feature_map.device
    assert empty.dtype == feature_map.dtype
    assert one.shape == (1, 256)
    assert torch.allclose(one.norm(dim=1), torch.ones(1), atol=1e-5)


def test_roi_embedding_gradient_reaches_head_and_shared_backbone() -> None:
    torch.manual_seed(2)
    model = make_model(recurrent=False).train()
    outputs = model(torch.randn(1, 8, 64, 64))
    descriptors = model.extract_roi_embeddings(
        outputs["embedding_feature_map"],
        [
            torch.tensor(
                [
                    [0.0, 0.0, 24.0, 24.0],
                    [32.0, 32.0, 64.0, 64.0],
                ]
            )
        ],
    )
    loss = (descriptors[0] - descriptors[1]).square().sum()
    loss.backward()

    assert model.embedding_head.weight.grad is not None
    assert model.embedding_proj[0].weight.grad is not None
    assert model.backbone[0][0].weight.grad is not None


def test_detector_only_checkpoint_config_still_loads() -> None:
    old_config = {
        "in_channels": 8,
        "num_classes": 7,
        "feature_stride": 8,
        "width": 8,
        "architecture": "simple",
        "fusion_mode": "gated_two_branch",
        "event_frame_channels": 2,
        "voxel_grid_channels": 6,
        "component_channels": (2, 6),
    }
    model = SimpleDenseDetector(SimpleDetectorConfig(**old_config))
    reloaded = SimpleDenseDetector(SimpleDetectorConfig(**old_config))
    reloaded.load_state_dict(model.state_dict())

    assert set(reloaded(torch.zeros(1, 8, 64, 64))) == {"cls_logits", "bbox_raw"}
    assert reloaded.embedding_head is None


def test_post_nms_embeddings_attach_in_exported_detection_order() -> None:
    cls_logits = torch.zeros((1, 8, 8, 8))
    bbox_raw = torch.zeros((1, 4, 8, 8))
    cls_logits[0, 1, 1, 1] = 9.0
    cls_logits[0, 2, 5, 5] = 8.0
    detections = decode_dense_detections(
        {"cls_logits": cls_logits, "bbox_raw": bbox_raw},
        frame_index=3,
        timestamp=123,
        score_threshold=0.5,
        image_width=64,
        image_height=64,
    )
    reference = torch.zeros(1)
    boxes = detection_boxes_xyxy(detections, reference)
    descriptors = torch.stack([torch.full((256,), 0.1), torch.full((256,), 0.2)])

    attached = attach_detection_embeddings(detections, descriptors)

    assert boxes.shape == (2, 4)
    assert [item.class_id for item in attached] == [0, 1]
    assert attached[0].embedding == pytest.approx(tuple([0.1] * 256))
    assert attached[1].embedding == pytest.approx(tuple([0.2] * 256))
    assert [item.score for item in attached] == [item.score for item in detections]
