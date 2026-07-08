import numpy as np

from src.data.converters.event_manifest import (
    BoxRecord,
    RepresentationConfig,
    event_array,
    group_boxes_by_timestamp,
    load_structured_boxes,
    read_h5_event_file,
    read_metavision_dat,
    save_dense_representations,
    select_event_window,
    split_rows,
)


def test_event_window_and_representation_cache(tmp_path):
    events = event_array(
        x=np.array([1, 2, 3]),
        y=np.array([1, 2, 3]),
        t=np.array([100, 200, 300]),
        p=np.array([0, 1, 1]),
    )

    window = select_event_window(events, end_timestamp_us=300, time_window_us=150)

    assert window["t"].tolist() == [200, 300]

    paths = save_dense_representations(
        window,
        tmp_path / "sample",
        RepresentationConfig(width=8, height=8, num_bins=3, time_window_us=150),
    )

    ef = np.load(paths["event_frame"])
    vg = np.load(paths["voxel_grid"])
    assert ef.shape == (2, 8, 8)
    assert vg.shape == (6, 8, 8)
    assert ef.dtype == np.float16


def test_structured_box_loading_and_grouping(tmp_path):
    path = tmp_path / "boxes.npy"
    dtype = np.dtype(
        [
            ("t", "<i8"),
            ("x", "<f4"),
            ("y", "<f4"),
            ("w", "<f4"),
            ("h", "<f4"),
            ("class_id", "u1"),
            ("track_id", "<u4"),
        ]
    )
    boxes = np.array([(100, 1, 2, 3, 4, 2, 9), (100, 5, 6, 7, 8, 1, 10)], dtype=dtype)
    np.save(path, boxes)

    records = load_structured_boxes(path, class_id_map={1: "pedestrian", 2: "vehicle"})
    grouped = group_boxes_by_timestamp(records)

    assert len(records) == 2
    assert records[0] == BoxRecord(100, 1.0, 2.0, 4.0, 6.0, "vehicle", 9)
    assert set(grouped) == {100}
    assert [box.label for box in grouped[100]] == ["vehicle", "pedestrian"]


def test_split_rows_is_deterministic_and_non_empty():
    rows = [{"id": index} for index in range(10)]

    train_a, val_a = split_rows(rows, val_fraction=0.2, seed=7)
    train_b, val_b = split_rows(rows, val_fraction=0.2, seed=7)

    assert train_a == train_b
    assert val_a == val_b
    assert len(train_a) == 8
    assert len(val_a) == 2


def test_h5_event_reader_applies_t_offset(tmp_path):
    import h5py

    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as handle:
        group = handle.create_group("events")
        group.create_dataset("x", data=np.array([1, 2], dtype=np.uint16))
        group.create_dataset("y", data=np.array([3, 4], dtype=np.uint16))
        group.create_dataset("p", data=np.array([0, 1], dtype=np.uint8))
        group.create_dataset("t", data=np.array([10, 20], dtype=np.int64))
        group.create_dataset("t_offset", data=np.array(1000, dtype=np.int64))

    events = read_h5_event_file(path)

    assert events["t"].tolist() == [1010, 1020]
    assert events["x"].tolist() == [1, 2]


def test_metavision_dat_fallback_skips_binary_type_size_header(tmp_path):
    path = tmp_path / "sample_td.dat"
    timestamp = np.array([1234], dtype="<u4")
    data = np.array([(1 << 28) | (20 << 14) | 10], dtype="<u4")
    path.write_bytes(
        b"% Width 1280\n% Height 720\n" + bytes([12, 8]) + timestamp.tobytes() + data.tobytes()
    )

    events = read_metavision_dat(path)

    assert events["t"].tolist() == [1234]
    assert events["x"].tolist() == [10]
    assert events["y"].tolist() == [20]
    assert events["p"].tolist() == [True]
