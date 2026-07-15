"""Tests for ordered clips with object-level association targets."""

from pathlib import Path

import numpy as np
import torch

from src.data.dense_targets import IDENTITY_IGNORE_INDEX
from src.data.sequence_dataset import DSECClipDataset, collate_clip_batch

FRAME_INTERVAL_US = 50_000
FIRST_TIMESTAMP_US = 1_000_000


def _write_sequence(
    root: Path,
    split: str,
    sequence: str,
    annotations: list[tuple[int, float, float, float, float, int]],
    frames: int = 4,
) -> None:
    import h5py

    seq_dir = root / split / sequence
    (seq_dir / "events_left").mkdir(parents=True)
    (root / "annotations" / split).mkdir(parents=True, exist_ok=True)
    timestamps = [FIRST_TIMESTAMP_US + index * FRAME_INTERVAL_US for index in range(frames)]
    (seq_dir / f"{sequence}_image_timestamps.txt").write_text(
        "\n".join(str(timestamp) for timestamp in timestamps) + "\n",
        encoding="utf-8",
    )
    rows = [
        f"{timestamp},{track_id},{left},{top},{width},{height},{class_id}"
        for timestamp in timestamps
        for track_id, left, top, width, height, class_id in annotations
    ]
    (root / "annotations" / split / f"{sequence}.txt").write_text(
        "\n".join(rows) + "\n",
        encoding="utf-8",
    )

    event_times = np.asarray(timestamps, dtype=np.int64)
    with h5py.File(seq_dir / "events_left" / "events.h5", "w") as handle:
        group = handle.create_group("events")
        group.create_dataset("x", data=np.full(frames, 320, dtype=np.uint16))
        group.create_dataset("y", data=np.full(frames, 240, dtype=np.uint16))
        group.create_dataset("p", data=np.ones(frames, dtype=np.uint8))
        group.create_dataset("t", data=event_times)
        handle.create_dataset("ms_to_idx", data=np.zeros(1, dtype=np.int64))


def make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dsec_mot"
    _write_sequence(
        root,
        "train",
        "seq_a",
        [
            (1, 100, 100, 40, 40, 0),
            (2, 300, 200, 60, 40, 1),
        ],
    )
    _write_sequence(root, "train", "seq_b", [(5, 200, 300, 50, 50, 0)])
    return root


def build_dataset(
    root: Path,
    sequences: list[str],
    **kwargs: object,
) -> DSECClipDataset:
    return DSECClipDataset(
        root=root,
        split="train",
        sequences=sequences,
        representation="event_frame",
        num_bins=1,
        clip_length=2,
        clip_stride=2,
        **kwargs,
    )


def test_clip_and_collate_expose_variable_length_roi_targets(tmp_path: Path) -> None:
    dataset = build_dataset(make_dataset_root(tmp_path), ["seq_a"])
    item = dataset[0]

    assert item["events"].shape == (2, 2, 480, 640)
    assert item["cls"].shape == (2, 60, 80)
    assert len(item["roi_boxes"]) == 2
    assert item["roi_boxes"][0].shape == (2, 4)
    assert item["roi_identity_targets"][0].shape == (2,)
    assert item["roi_track_ids"][0].tolist() == [1, 2]
    assert item["roi_class_ids"][0].tolist() == [0, 1]

    batch = collate_clip_batch([item, item])
    assert batch["events"].shape == (2, 2, 2, 480, 640)
    assert len(batch["roi_boxes"]) == 2
    assert len(batch["roi_boxes"][0]) == 2
    assert torch.equal(batch["roi_track_ids"][0][0], torch.tensor([1, 2]))


def test_clips_never_cross_sequence_boundaries(tmp_path: Path) -> None:
    dataset = DSECClipDataset(
        root=make_dataset_root(tmp_path),
        split="train",
        sequences=["seq_a", "seq_b"],
        representation="event_frame",
        num_bins=1,
        clip_length=3,
        clip_stride=1,
    )

    assert len(dataset) == 4
    for index in range(len(dataset)):
        meta = dataset[index]["meta"]
        assert len({frame["sequence"] for frame in meta}) == 1
        frame_indices = [frame["frame_index"] for frame in meta]
        assert frame_indices == list(range(frame_indices[0], frame_indices[0] + 3))


def test_boxes_are_clipped_and_invalid_boxes_are_removed(tmp_path: Path) -> None:
    root = tmp_path / "dsec_mot"
    _write_sequence(
        root,
        "train",
        "seq",
        [
            (1, -10, -5, 30, 25, 0),
            (2, 630, 470, 30, 30, 1),
            (3, 700, 100, 20, 20, 0),
            (4, 10, 10, -4, 8, 0),
        ],
        frames=2,
    )
    item = build_dataset(root, ["seq"])[0]

    assert item["roi_boxes"][0].tolist() == [
        [0.0, 0.0, 20.0, 20.0],
        [630.0, 470.0, 640.0, 480.0],
    ]
    assert item["roi_track_ids"][0].tolist() == [1, 2]
    assert item["roi_class_ids"][0].tolist() == [0, 1]
    assert item["meta"][0]["num_annotations"] == 2


def test_unknown_validation_ids_are_ignored_for_ce_but_retained_for_retrieval(
    tmp_path: Path,
) -> None:
    root = make_dataset_root(tmp_path)
    train_dataset = build_dataset(root, ["seq_a"])
    val_dataset = build_dataset(
        root,
        ["seq_b"],
        identity_vocabulary=train_dataset.identity_vocabulary,
    )
    frame = val_dataset[0]

    assert (frame["roi_identity_targets"][0] == IDENTITY_IGNORE_INDEX).all()
    assert frame["roi_track_ids"][0].tolist() == [5]
    assert frame["roi_class_ids"][0].tolist() == [0]
    assert frame["meta"][0]["sequence"] == "seq_b"


def test_identity_vocabulary_is_sequence_qualified_and_contiguous(tmp_path: Path) -> None:
    dataset = build_dataset(make_dataset_root(tmp_path), ["seq_a", "seq_b"])

    assert dataset.num_identities == 3
    indices = [
        dataset.identity_vocabulary.lookup("seq_a", 1),
        dataset.identity_vocabulary.lookup("seq_a", 2),
        dataset.identity_vocabulary.lookup("seq_b", 5),
    ]
    assert sorted(indices) == [0, 1, 2]


def test_car_only_filter_keeps_only_car_objects_and_identities(tmp_path: Path) -> None:
    dataset = build_dataset(make_dataset_root(tmp_path), ["seq_a"], class_ids=[0])
    item = dataset[0]

    assert dataset.num_identities == 1
    assert item["roi_track_ids"][0].tolist() == [1]
    assert item["roi_class_ids"][0].tolist() == [0]
    assert set(np.unique(item["cls"])) <= {0, 1}
