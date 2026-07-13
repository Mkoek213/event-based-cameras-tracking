"""Tests for the ordered clip dataset used by recurrent embedding training."""

from pathlib import Path

import numpy as np

from src.data.dense_targets import IDENTITY_IGNORE_INDEX
from src.data.sequence_dataset import DSECClipDataset, collate_clip_batch

FRAME_INTERVAL_US = 50_000
FIRST_TIMESTAMP_US = 1_000_000


def _write_sequence(root: Path, split: str, sequence: str, tracks: dict[int, tuple], frames: int):
    import h5py

    seq_dir = root / split / sequence
    (seq_dir / "events_left").mkdir(parents=True)
    (root / "annotations" / split).mkdir(parents=True, exist_ok=True)

    timestamps = [FIRST_TIMESTAMP_US + index * FRAME_INTERVAL_US for index in range(frames)]
    (seq_dir / f"{sequence}_image_timestamps.txt").write_text(
        "\n".join(str(timestamp) for timestamp in timestamps) + "\n", encoding="utf-8"
    )

    rows = []
    for timestamp in timestamps:
        for track_id, (left, top, width, height) in tracks.items():
            rows.append(f"{timestamp},{track_id},{left},{top},{width},{height},0")
    (root / "annotations" / split / f"{sequence}.txt").write_text(
        "\n".join(rows) + "\n", encoding="utf-8"
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
    _write_sequence(root, "train", "seq_a", {1: (100, 100, 40, 40), 2: (300, 200, 60, 40)}, 6)
    _write_sequence(root, "train", "seq_b", {5: (200, 300, 50, 50)}, 6)
    return root


def test_clip_item_shapes(tmp_path: Path):
    root = make_dataset_root(tmp_path)
    dataset = DSECClipDataset(
        root=root,
        split="train",
        sequences=["seq_a"],
        representation="event_frame",
        num_bins=1,
        clip_length=4,
    )

    assert len(dataset) == 1
    item = dataset[0]
    assert item["events"].shape == (4, 2, 480, 640)
    assert item["cls"].shape == (4, 60, 80)
    assert item["bbox"].shape == (4, 4, 60, 80)
    assert item["pos_mask"].shape == (4, 60, 80)
    assert item["identity"].shape == (4, 60, 80)
    assert len(item["meta"]) == 4

    batch = collate_clip_batch([item, item])
    assert batch["events"].shape == (2, 4, 2, 480, 640)
    assert batch["identity_targets"].shape == (2, 4, 60, 80)


def test_clips_never_cross_sequence_boundaries(tmp_path: Path):
    root = make_dataset_root(tmp_path)
    dataset = DSECClipDataset(
        root=root,
        split="train",
        sequences=["seq_a", "seq_b"],
        representation="event_frame",
        num_bins=1,
        clip_length=4,
        clip_stride=1,
    )

    assert len(dataset) == 6
    for index in range(len(dataset)):
        meta = dataset[index]["meta"]
        sequences = {frame["sequence"] for frame in meta}
        frame_indices = [frame["frame_index"] for frame in meta]
        assert len(sequences) == 1
        assert frame_indices == sorted(frame_indices)


def test_identity_map_aligns_with_pos_mask(tmp_path: Path):
    root = make_dataset_root(tmp_path)
    dataset = DSECClipDataset(
        root=root,
        split="train",
        sequences=["seq_a", "seq_b"],
        representation="event_frame",
        num_bins=1,
        clip_length=4,
    )

    item = dataset[0]
    assert item["pos_mask"].any()
    assert np.array_equal(item["identity"] != IDENTITY_IGNORE_INDEX, item["pos_mask"])


def test_identity_vocabulary_is_contiguous_and_stable(tmp_path: Path):
    root = make_dataset_root(tmp_path)

    def build() -> DSECClipDataset:
        return DSECClipDataset(
            root=root,
            split="train",
            sequences=["seq_a", "seq_b"],
            representation="event_frame",
            num_bins=1,
            clip_length=4,
        )

    first = build()
    second = build()

    assert first.num_identities == 3
    pairs = [("seq_a", 1), ("seq_a", 2), ("seq_b", 5)]
    indices = [first.identity_vocabulary.lookup(sequence, track) for sequence, track in pairs]
    assert sorted(indices) == [0, 1, 2]
    second_indices = [
        second.identity_vocabulary.lookup(sequence, track) for sequence, track in pairs
    ]
    assert indices == second_indices


def test_unknown_validation_identities_map_to_ignore_index(tmp_path: Path):
    root = make_dataset_root(tmp_path)
    train_dataset = DSECClipDataset(
        root=root,
        split="train",
        sequences=["seq_a"],
        representation="event_frame",
        num_bins=1,
        clip_length=4,
    )
    val_dataset = DSECClipDataset(
        root=root,
        split="train",
        sequences=["seq_b"],
        representation="event_frame",
        num_bins=1,
        clip_length=4,
        identity_vocabulary=train_dataset.identity_vocabulary,
    )

    item = val_dataset[0]
    assert item["pos_mask"].any()
    assert (item["identity"] == IDENTITY_IGNORE_INDEX).all()
