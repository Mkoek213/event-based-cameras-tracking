"""Tests for the lightweight dense detector."""

import torch

from src.models.simple_detector import (
    SimpleDenseDetector,
    SimpleDetectorConfig,
    decode_dense_detections,
    normalise_representation_tensor,
    simple_detector_loss,
)


def test_simple_detector_output_shapes():
    model = SimpleDenseDetector(SimpleDetectorConfig(in_channels=10, width=8))
    outputs = model(torch.zeros((2, 10, 480, 640)))

    assert outputs["cls_logits"].shape == (2, 8, 60, 80)
    assert outputs["bbox_raw"].shape == (2, 4, 60, 80)


def test_two_branch_detector_output_shapes():
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=12,
            width=8,
            fusion_mode="two_branch",
            event_frame_channels=2,
            voxel_grid_channels=10,
        )
    )
    outputs = model(torch.zeros((2, 12, 480, 640)))

    assert outputs["cls_logits"].shape == (2, 8, 60, 80)
    assert outputs["bbox_raw"].shape == (2, 4, 60, 80)


def test_two_branch_detector_validates_channel_split():
    try:
        SimpleDenseDetector(
            SimpleDetectorConfig(
                in_channels=12,
                width=8,
                fusion_mode="two_branch",
                event_frame_channels=2,
                voxel_grid_channels=8,
            )
        )
    except ValueError as exc:
        assert "channel split" in str(exc)
    else:
        raise AssertionError("Expected channel split validation error.")


def test_generic_two_branch_detector_output_shapes():
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=3,
            width=8,
            fusion_mode="two_branch",
            component_channels=(2, 1),
        )
    )
    outputs = model(torch.zeros((2, 3, 480, 640)))

    assert outputs["cls_logits"].shape == (2, 8, 60, 80)
    assert outputs["bbox_raw"].shape == (2, 4, 60, 80)


def test_three_branch_detector_output_shapes():
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=13,
            width=8,
            fusion_mode="three_branch",
            component_channels=(2, 10, 1),
        )
    )
    outputs = model(torch.zeros((2, 13, 480, 640)))

    assert outputs["cls_logits"].shape == (2, 8, 60, 80)
    assert outputs["bbox_raw"].shape == (2, 4, 60, 80)


def test_gated_two_branch_detector_output_shapes():
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            in_channels=12,
            width=8,
            fusion_mode="gated_two_branch",
            component_channels=(2, 10),
        )
    )
    outputs = model(torch.zeros((2, 12, 480, 640)))

    assert outputs["cls_logits"].shape == (2, 8, 60, 80)
    assert outputs["bbox_raw"].shape == (2, 4, 60, 80)
    assert model.gate is not None


def test_representation_normalisation_scales_components_independently():
    tensor = torch.zeros((1, 3, 2, 2))
    tensor[:, :2] = 100.0
    tensor[:, 2:] = 1.0

    normalised = normalise_representation_tensor(tensor, (2, 1))

    assert torch.isclose(normalised[:, :2].amax(), torch.tensor(1.0))
    assert torch.isclose(normalised[:, 2:].amax(), torch.tensor(1.0))


def test_simple_detector_loss_is_finite():
    model = SimpleDenseDetector(SimpleDetectorConfig(in_channels=10, width=8))
    outputs = model(torch.zeros((1, 10, 480, 640)))
    cls_targets = torch.zeros((1, 60, 80), dtype=torch.long)
    bbox_targets = torch.zeros((1, 4, 60, 80), dtype=torch.float32)
    pos_mask = torch.zeros((1, 60, 80), dtype=torch.bool)
    cls_targets[0, 10, 12] = 1
    bbox_targets[0, :, 10, 12] = torch.tensor([2.0, 2.0, 3.0, 3.0])
    pos_mask[0, 10, 12] = True

    loss, stats = simple_detector_loss(outputs, cls_targets, bbox_targets, pos_mask)

    assert torch.isfinite(loss)
    assert stats["positive_cells"] == 1


def test_decode_dense_detections_returns_detection_record():
    cls_logits = torch.zeros((1, 8, 60, 80))
    bbox_raw = torch.zeros((1, 4, 60, 80))
    cls_logits[0, 3, 5, 6] = 10.0

    detections = decode_dense_detections(
        {"cls_logits": cls_logits, "bbox_raw": bbox_raw},
        frame_index=4,
        timestamp=123,
        score_threshold=0.5,
    )

    assert len(detections) == 1
    assert detections[0].frame_index == 4
    assert detections[0].timestamp == 123
    assert detections[0].class_id == 2
    assert detections[0].bbox_width > 0
    assert detections[0].bbox_height > 0
