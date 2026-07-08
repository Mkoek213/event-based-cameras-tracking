"""Tests for the multi-dataset manifest loader."""

import json

import numpy as np

from src.data.unified_manifest import UnifiedDenseRepresentationDataset, write_jsonl


def test_unified_manifest_loads_component_representations(tmp_path):
    ef_path = tmp_path / "ef.npy"
    vg_path = tmp_path / "vg.npy"
    np.save(ef_path, np.ones((2, 24, 32), dtype=np.float32))
    np.save(vg_path, np.ones((10, 24, 32), dtype=np.float32) * 2)
    manifest = tmp_path / "train.jsonl"
    write_jsonl(
        manifest,
        [
            {
                "dataset": "synthetic",
                "sequence": "seq0",
                "timestamp_us": 1000,
                "frame_index": 0,
                "width": 64,
                "height": 48,
                "representation_paths": {
                    "event_frame": str(ef_path),
                    "voxel_grid": str(vg_path),
                },
                "boxes": [[16, 12, 40, 36]],
                "labels": ["vehicle"],
            }
        ],
    )

    dataset = UnifiedDenseRepresentationDataset(
        manifest,
        representation="event_frame_voxel_grid",
        image_width=64,
        image_height=48,
    )
    sample = dataset[0]

    assert sample["events"].shape == (12, 48, 64)
    assert sample["label"][0].shape == (6, 8)
    assert sample["label"][2].any()
    assert sample["meta"]["dataset"] == "synthetic"


def test_unified_manifest_accepts_preconcatenated_representation(tmp_path):
    representation_path = tmp_path / "representation.npy"
    np.save(representation_path, np.ones((12, 48, 64), dtype=np.float32))
    manifest = tmp_path / "train.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "dataset": "synthetic",
                "sequence": "seq0",
                "timestamp_us": 1000,
                "width": 64,
                "height": 48,
                "representation_path": str(representation_path),
                "boxes": [[16, 12, 40, 36]],
                "labels": [0],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    dataset = UnifiedDenseRepresentationDataset(
        manifest,
        representation="event_frame_voxel_grid",
        image_width=64,
        image_height=48,
    )
    sample = dataset[0]

    assert sample["events"].shape == (12, 48, 64)
    assert sample["label"][0].max() == 1
