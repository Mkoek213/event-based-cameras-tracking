"""Tests for object-level R1/R2 association embedding heads."""

import pytest
import torch

from src.models.simple_detector import (
    ConvGRUCell,
    RecurrentFeatureAdapter,
    SimpleDenseDetector,
    SimpleDetectorConfig,
    attach_detection_embeddings,
    decode_dense_detections,
    detection_boxes_xyxy,
    simple_detector_config_from_checkpoint,
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


def test_bn_neck_can_be_applied_once_to_the_complete_clip_microbatch() -> None:
    model = make_model(recurrent=False).train()
    outputs = model(torch.randn(2, 8, 64, 64))
    boxes = [
        torch.tensor([[0.0, 0.0, 24.0, 24.0]]),
        torch.tensor([[32.0, 32.0, 64.0, 64.0]]),
    ]
    before = int(model.embedding_bn.num_batches_tracked)

    pre_bn = model.project_roi_embeddings(outputs["embedding_feature_map"], boxes)
    post_bn = model.apply_embedding_bn(pre_bn)

    assert pre_bn.shape == post_bn.shape == (2, 256)
    assert int(model.embedding_bn.num_batches_tracked) == before + 1
    assert not torch.allclose(pre_bn, post_bn)


def test_frozen_detector_stays_in_eval_mode_while_embedding_head_trains() -> None:
    model = make_model(recurrent=True)
    model.set_detector_trainable(False)
    model.train()

    assert all(
        not parameter.requires_grad
        for module in model.detector_modules()
        for parameter in module.parameters()
    )
    assert all(not module.training for module in model.detector_modules())
    assert model.embedding_proj.training
    assert model.embedding_recurrent_cell.training
    assert model.embedding_head.training
    assert model.embedding_bn.training


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


@pytest.mark.parametrize("recurrent", [False, True], ids=["D1", "D2"])
def test_dense_head_outputs_unit_embedding_map_and_optional_memory(recurrent: bool) -> None:
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            **GATED_CONFIG,
            embedding_head_type="dense",
            embedding_recurrent=recurrent,
        )
    )
    inputs = torch.randn(2, 8, 64, 64)

    first = model(inputs)
    second = model(inputs, first["embedding_state"])

    assert first["embeddings"].shape == (2, 256, 8, 8)
    assert torch.allclose(
        first["embeddings"].norm(dim=1),
        torch.ones(2, 8, 8),
        atol=1e-5,
    )
    assert "embedding_feature_map" not in first
    if recurrent:
        assert first["embedding_state"].shape == (2, 128, 8, 8)
        assert not torch.equal(first["embedding_state"], second["embedding_state"])
    else:
        assert first["embedding_state"] is None
        assert second["embedding_state"] is None


def test_dense_embeddings_are_gathered_from_box_centre_cells_in_order() -> None:
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            **GATED_CONFIG,
            embedding_head_type="dense",
            embedding_recurrent=False,
        )
    )
    embedding_map = torch.zeros((2, 256, 8, 8))
    embedding_map[0, 3, 1, 1] = 1.0
    embedding_map[0, 7, 5, 5] = 1.0
    embedding_map[1, 11, 3, 4] = 1.0
    boxes = [
        torch.tensor([[4.0, 4.0, 20.0, 20.0], [36.0, 36.0, 52.0, 52.0]]),
        torch.tensor([[28.0, 20.0, 44.0, 36.0]]),
    ]

    descriptors = model.extract_dense_embeddings_at_boxes(embedding_map, boxes)

    assert descriptors.shape == (3, 256)
    assert descriptors.argmax(dim=1).tolist() == [3, 7, 11]


def test_dense_decode_attaches_normalised_cell_embedding_after_nms() -> None:
    cls_logits = torch.zeros((1, 8, 8, 8))
    bbox_raw = torch.zeros((1, 4, 8, 8))
    embeddings = torch.zeros((1, 16, 8, 8))
    cls_logits[0, 1, 2, 3] = 9.0
    embeddings[0, 5, 2, 3] = 1.0

    detections = decode_dense_detections(
        {"cls_logits": cls_logits, "bbox_raw": bbox_raw},
        frame_index=0,
        timestamp=10,
        score_threshold=0.5,
        image_width=64,
        image_height=64,
        embeddings=embeddings,
    )

    assert len(detections) == 1
    assert detections[0].embedding is not None
    assert detections[0].embedding[5] == pytest.approx(1.0)
    assert sum(value * value for value in detections[0].embedding) == pytest.approx(1.0)


def test_legacy_checkpoint_head_type_is_inferred_from_weight_rank() -> None:
    dense_checkpoint = {
        "model_config": {**GATED_CONFIG, "embedding_recurrent": False},
        "model_state": {"embedding_head.weight": torch.empty(256, 128, 1, 1)},
    }
    roi_checkpoint = {
        "model_config": {**GATED_CONFIG, "embedding_recurrent": False},
        "model_state": {"embedding_head.weight": torch.empty(256, 128)},
    }

    assert simple_detector_config_from_checkpoint(dense_checkpoint).embedding_head_type == "dense"
    assert simple_detector_config_from_checkpoint(roi_checkpoint).embedding_head_type == "roi"


@pytest.mark.parametrize("cell_type", ["convgru", "convlstm", "convrnn"])
def test_temporal_adapter_residual_mode_is_identity_initialised(cell_type: str) -> None:
    torch.manual_seed(11)
    adapter = RecurrentFeatureAdapter(6, cell_type=cell_type, mode="residual")
    inputs = torch.randn(2, 6, 12, 10)

    first, state = adapter(inputs)
    second, next_state = adapter(inputs, state)

    assert torch.equal(first, inputs)
    assert torch.equal(second, inputs)
    if cell_type == "convlstm":
        assert len(state) == len(next_state) == 2
        assert state[0].shape == state[1].shape == inputs.shape
        assert not torch.equal(state[0], next_state[0])
    else:
        assert state.shape == next_state.shape == inputs.shape
        assert not torch.equal(state, next_state)


@pytest.mark.parametrize("cell_type", ["convgru", "convlstm", "convrnn"])
def test_all_temporal_locations_keep_separate_state_and_preserve_r0_at_start(
    cell_type: str,
) -> None:
    temporal_locations = (
        "backbone_s2",
        "backbone_s4",
        "neck",
        "detection_heads",
        "embedding",
    )
    torch.manual_seed(17)
    control = SimpleDenseDetector(
        SimpleDetectorConfig(
            **GATED_CONFIG,
            embedding_head_type="dense",
            embedding_recurrent=False,
        )
    ).eval()
    torch.manual_seed(17)
    recurrent = SimpleDenseDetector(
        SimpleDetectorConfig(
            **GATED_CONFIG,
            embedding_head_type="dense",
            embedding_recurrent=False,
            temporal_recurrence_locations=temporal_locations,
            temporal_recurrence_type=cell_type,
            temporal_recurrence_mode="residual",
        )
    ).eval()
    inputs = torch.randn(2, 8, 64, 64)

    control_outputs = control(inputs)
    first = recurrent(inputs)
    second = recurrent(inputs, temporal_state=first["temporal_state"])

    assert set(first["temporal_state"]) == {
        "backbone_s2",
        "backbone_s4",
        "neck",
        "detection_cls",
        "detection_bbox",
        "embedding",
    }
    assert torch.equal(first["cls_logits"], control_outputs["cls_logits"])
    assert torch.equal(first["bbox_raw"], control_outputs["bbox_raw"])
    assert torch.equal(first["embeddings"], control_outputs["embeddings"])
    assert set(second["temporal_state"]) == set(first["temporal_state"])


def test_direct_temporal_adapter_changes_features_and_can_be_trained_when_frozen() -> None:
    model = SimpleDenseDetector(
        SimpleDetectorConfig(
            **GATED_CONFIG,
            embedding_head_type="dense",
            embedding_recurrent=False,
            temporal_recurrence_locations=("neck",),
            temporal_recurrence_type="convgru",
            temporal_recurrence_mode="direct",
        )
    )
    model.set_detector_trainable(False)
    model.set_temporal_trainable(True)

    assert all(
        parameter.requires_grad
        for module in model.temporal_modules()
        for parameter in module.parameters()
    )
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("embedding_") and not name.startswith("temporal_adapters.")
    )

    outputs = model(torch.randn(1, 8, 64, 64))
    outputs["cls_logits"].sum().backward()
    assert model.temporal_adapters["neck"].cell.update_gate.weight.grad is not None


def test_checkpoint_config_normalises_temporal_locations_to_tuple() -> None:
    checkpoint = {
        "model_config": {
            **GATED_CONFIG,
            "embedding_head_type": "dense",
            "temporal_recurrence_locations": ["neck", "embedding"],
        },
        "model_state": {},
    }

    config = simple_detector_config_from_checkpoint(checkpoint)

    assert config.temporal_recurrence_locations == ("neck", "embedding")
