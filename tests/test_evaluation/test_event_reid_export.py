"""Integration tests for post-NMS embedding export and sequence-state reset."""

import json
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from src.evaluation.simple_detector_trackeval_cli import (
    export_simple_detector_detections_for_sequence,
    reuse_cached_detection_export,
)


class RecordingEmbeddingModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            embedding_dim=2,
            embedding_recurrent=True,
            feature_stride=8,
        )
        self.received_states: list[torch.Tensor | None] = []
        self.received_boxes: list[torch.Tensor] = []

    def __call__(
        self,
        tensor: torch.Tensor,
        embedding_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.received_states.append(embedding_state)
        cls_logits = tensor.new_zeros((1, 8, 60, 80))
        cls_logits[0, 1, 5, 6] = 10.0
        cls_logits[0, 2, 20, 30] = 9.0
        return {
            "cls_logits": cls_logits,
            "bbox_raw": tensor.new_zeros((1, 4, 60, 80)),
            "embedding_feature_map": tensor.new_zeros((1, 4, 60, 80)),
            "embedding_state": tensor.new_ones((1, 4, 60, 80)),
        }

    def extract_roi_embeddings(
        self,
        feature_map: torch.Tensor,
        boxes_per_image: list[torch.Tensor],
    ) -> torch.Tensor:
        self.received_boxes.append(boxes_per_image[0].detach().cpu())
        return feature_map.new_tensor([[1.0, 0.0], [0.0, 1.0]])


def write_sequence(root: Path, sequence: str) -> None:
    seq_dir = root / "test" / sequence
    (seq_dir / "events_left").mkdir(parents=True)
    timestamp = 1_000_000
    (seq_dir / f"{sequence}_image_timestamps.txt").write_text(f"{timestamp}\n", encoding="utf-8")
    with h5py.File(seq_dir / "events_left" / "events.h5", "w") as handle:
        group = handle.create_group("events")
        group.create_dataset("x", data=np.asarray([320], dtype=np.uint16))
        group.create_dataset("y", data=np.asarray([240], dtype=np.uint16))
        group.create_dataset("p", data=np.asarray([1], dtype=np.uint8))
        group.create_dataset("t", data=np.asarray([timestamp], dtype=np.int64))
        handle.create_dataset("ms_to_idx", data=np.zeros(1, dtype=np.int64))


def export(
    model: RecordingEmbeddingModel,
    root: Path,
    sequence: str,
    output: Path,
) -> dict:
    return export_simple_detector_detections_for_sequence(
        model=model,
        checkpoint={"model_config": {}, "benchmark_config": {}},
        root=root,
        split="test",
        sequence=sequence,
        output_path=output,
        score_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections=10,
        representation="event_frame_voxel_grid",
        num_bins=3,
        time_window_us=50_000,
        device=torch.device("cpu"),
        input_normalisation="component",
    )


def test_export_extracts_descriptors_after_nms_in_detection_order(tmp_path: Path) -> None:
    root = tmp_path / "dsec_mot"
    write_sequence(root, "seq")
    model = RecordingEmbeddingModel()

    payload = export(model, root, "seq", tmp_path / "seq.json")

    assert len(payload["detections"]) == 2
    assert [row["class_id"] for row in payload["detections"]] == [0, 1]
    assert [row["embedding"] for row in payload["detections"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert model.received_boxes[0].shape == (2, 4)


def test_recurrent_embedding_state_resets_for_each_exported_sequence(
    tmp_path: Path,
) -> None:
    root = tmp_path / "dsec_mot"
    write_sequence(root, "seq_a")
    write_sequence(root, "seq_b")
    model = RecordingEmbeddingModel()

    export(model, root, "seq_a", tmp_path / "a.json")
    export(model, root, "seq_b", tmp_path / "b.json")

    assert model.received_states == [None, None]


class RecordingDenseEmbeddingModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            embedding_dim=2,
            embedding_head_type="dense",
            embedding_recurrent=True,
            feature_stride=8,
        )
        self.received_states: list[torch.Tensor | None] = []

    def __call__(
        self,
        tensor: torch.Tensor,
        embedding_state: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self.received_states.append(embedding_state)
        cls_logits = tensor.new_zeros((1, 8, 60, 80))
        cls_logits[0, 1, 5, 6] = 10.0
        cls_logits[0, 2, 20, 30] = 9.0
        embeddings = tensor.new_zeros((1, 2, 60, 80))
        embeddings[0, 0, 5, 6] = 1.0
        embeddings[0, 1, 20, 30] = 1.0
        return {
            "cls_logits": cls_logits,
            "bbox_raw": tensor.new_zeros((1, 4, 60, 80)),
            "embeddings": embeddings,
            "embedding_state": tensor.new_ones((1, 4, 60, 80)),
        }


def test_dense_export_uses_embeddings_from_detection_cells(tmp_path: Path) -> None:
    root = tmp_path / "dsec_mot"
    write_sequence(root, "seq")
    model = RecordingDenseEmbeddingModel()

    payload = export(model, root, "seq", tmp_path / "dense.json")

    assert payload["embedding_head_type"] == "dense"
    assert [row["embedding"] for row in payload["detections"]] == [
        [1.0, 0.0],
        [0.0, 1.0],
    ]
    assert model.received_states == [None]


def test_cached_detection_export_preserves_exact_score_export(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    source = tmp_path / "cache.json"
    output = tmp_path / "filtered.json"
    source.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "score_threshold": 0.9,
                "nms_iou_threshold": 0.5,
                "max_detections": 100,
                "frames": [{"frame_index": 0, "timestamp": 1}],
                "detections": [{"score": 0.9, "embedding": [0.0, 1.0]}],
            }
        ),
        encoding="utf-8",
    )

    payload = reuse_cached_detection_export(
        source_path=source,
        output_path=output,
        checkpoint_path=checkpoint,
        score_threshold=0.9,
        nms_iou_threshold=0.5,
        max_detections=100,
    )

    assert payload["score_threshold"] == 0.9
    assert payload["detection_cache_source"] == str(source)
    assert payload["detections"] == [{"score": 0.9, "embedding": [0.0, 1.0]}]
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_cached_detection_export_rejects_a_mismatched_score(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    source = tmp_path / "cache.json"
    source.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "score_threshold": 0.9,
                "nms_iou_threshold": 0.5,
                "max_detections": 100,
                "detections": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        reuse_cached_detection_export(
            source_path=source,
            output_path=tmp_path / "filtered.json",
            checkpoint_path=checkpoint,
            score_threshold=0.5,
            nms_iou_threshold=0.5,
            max_detections=100,
        )
