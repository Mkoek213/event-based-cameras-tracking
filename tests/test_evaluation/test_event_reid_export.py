"""Integration tests for post-NMS embedding export and sequence-state reset."""

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from src.evaluation.simple_detector_trackeval_cli import (
    export_simple_detector_detections_for_sequence,
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
