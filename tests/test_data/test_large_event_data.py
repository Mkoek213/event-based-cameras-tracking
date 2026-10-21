"""Tests for the large event MOT manifests, EF/VG cache, clips, and samplers.

Every source is a small synthetic file written into ``tmp_path``; no dataset, network,
or GPU is used.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

h5py = pytest.importorskip("h5py")
pytest.importorskip("hdf5plugin")

from src.data import large_event_cache as lc  # noqa: E402
from src.data import large_event_dataset as ld  # noqa: E402
from src.data import large_event_schema as schema  # noqa: E402
from src.data.converters import dsec_detection as dd  # noqa: E402
from src.data.converters import large_event_pretraining as lp  # noqa: E402
from src.data.preprocessing import EventPreprocessor  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs/large_event_mot/data.json"
CONFIG = json.loads(CONFIG_PATH.read_text())
LABEL_DTYPE = np.dtype(
    [
        ("t", "<u8"),
        ("x", "<f4"),
        ("y", "<f4"),
        ("w", "<f4"),
        ("h", "<f4"),
        ("class_id", "u1"),
        ("class_confidence", "<f4"),
        ("track_id", "<u4"),
    ]
)
STEM = "moorea_2019-01-30_000_td_500000_60500000"
GEN4_GEOM = lc.dataset_geometry(CONFIG, schema.GEN4_1MPX)
DSEC_GEOM = lc.dataset_geometry(CONFIG, schema.DSEC_MOT)


# --------------------------------------------------------------------------- synthetic sources


def file_version(path):
    """Identify one write of a file.

    Outputs are replaced atomically, so a rewrite gets a new inode even when the file
    system stores modification times in whole seconds (Lustre).
    """

    stat = Path(path).stat()
    return stat.st_ino, stat.st_mtime_ns


def write_gen4_chunk(root, stem, x, y, t, p, labels, split="train"):
    directory = Path(root) / split
    directory.mkdir(parents=True, exist_ok=True)
    with h5py.File(directory / f"{stem}_td.h5", "w") as handle:
        group = handle.create_group("events")
        group["x"] = np.asarray(x, np.int32)
        group["y"] = np.asarray(y, np.int32)
        group["t"] = np.asarray(t, np.uint32)
        group["p"] = np.asarray(p, np.int32)
    np.save(directory / f"{stem}_bbox.npy", np.asarray(labels, LABEL_DTYPE))


def write_dsec_sequence(root, split, name, t_abs, x, y, p, rectify, t_offset):
    directory = Path(root) / split / name / "events_left"
    directory.mkdir(parents=True, exist_ok=True)
    relative = np.asarray(t_abs, np.int64) - t_offset
    with h5py.File(directory / "events.h5", "w") as handle:
        group = handle.create_group("events")
        group["x"] = np.asarray(x, np.uint16)
        group["y"] = np.asarray(y, np.uint16)
        group["t"] = relative.astype(np.uint32)
        group["p"] = np.asarray(p, np.uint8)
        handle["t_offset"] = np.int64(t_offset)
    with h5py.File(directory / "rectify_map.h5", "w") as handle:
        handle["rectify_map"] = rectify.astype(np.float32)


def affine_map(scale=1.0, shift=(0.0, 0.0)):
    v, u = np.mgrid[0:480, 0:640].astype(np.float64)
    return np.stack([u * scale + shift[0], v * scale + shift[1]], -1)


def label_rows(rows):
    return np.array([tuple(row) for row in rows], LABEL_DTYPE)


def gen4_settings(**overrides):
    settings = lp.dataset_settings(CONFIG)
    settings.update(overrides)
    return settings


def dense_events(window_ends, rng, box=None, per_step=1500):
    """Uniform events plus a dense cluster inside ``box`` before every window end."""

    xs, ys, ts, ps = [], [], [], []
    for end in window_ends:
        ts.append(rng.integers(end - 50_000, end, per_step))
        xs.append(rng.integers(0, 1280, per_step))
        ys.append(rng.integers(0, 720, per_step))
        ps.append(rng.choice([-1, 1], per_step))
        if box is not None:
            bx, by, bw, bh = box
            ts.append(rng.integers(end - 50_000, end, 300))
            xs.append(rng.integers(bx, bx + bw, 300))
            ys.append(rng.integers(by, by + bh, 300))
            ps.append(rng.choice([-1, 1], 300))
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    return (
        np.concatenate(xs)[order],
        np.concatenate(ys)[order],
        t[order],
        np.concatenate(ps)[order],
    )


def dsec_mot_files(name, split="train"):
    return {
        "events": f"{split}/{name}/events_left/events.h5",
        "rectify_map": f"{split}/{name}/events_left/rectify_map.h5",
        "labels": f"{split}/{name}.txt",
        "image_timestamps": f"{split}/{name}/{name}_image_timestamps.txt",
    }


def convert_dsec_mot(root, name, images, rows, split="train", events=None, out=None):
    """Write a DSEC-MOT sequence with ``rows`` = (image index, track, x, y, w, h, class)."""

    rng = np.random.default_rng(len(name))
    t0 = int(images[0]) - 50_000
    if events is None:
        t = np.sort(rng.integers(t0, int(images[-1]), 20_000))
        events = (t, rng.integers(0, 640, t.size), rng.integers(0, 480, t.size))
    t, x, y = events
    write_dsec_sequence(
        root / "dsec", split, name, t, x, y, rng.integers(0, 2, t.size), affine_map(), t0
    )
    (root / f"dsec/{split}/{name}/{name}_image_timestamps.txt").write_text(
        "\n".join(str(int(value)) for value in images)
    )
    annotations = root / "ann" / split
    annotations.mkdir(parents=True, exist_ok=True)
    (annotations / f"{name}.txt").write_text(
        "".join(
            f"{int(images[k])},{track},{bx},{by},{bw},{bh},{cls}\n"
            for k, track, bx, by, bw, bh, cls in rows
        )
    )
    settings = dd.dataset_settings(CONFIG, schema.DSEC_MOT)
    roots = {"dsec_root": root / "dsec", "dsec_mot_annotations_root": root / "ann"}
    out = root / "manifest" if out is None else out
    record = dd.convert_sequence(
        schema.DSEC_MOT,
        name,
        split,
        roots,
        dsec_mot_files(name, split),
        settings,
        out / schema.DSEC_MOT,
    )
    return record, out


def entry(dataset, sequence, split):
    return {
        "dataset": dataset,
        "sequence": sequence,
        "split": split,
        "recording": schema.recording_of(dataset, sequence),
    }


# --------------------------------------------------------------------------- configuration


def test_seven_class_ontology_and_source_mappings():
    assert schema.TARGET_CLASSES == (
        "car",
        "pedestrian",
        "bicycle",
        "motorcycle",
        "bus",
        "truck",
        "train",
    )
    target = {name: index for index, name in enumerate(schema.TARGET_CLASSES)}

    def named(dataset):
        mapping = CONFIG[dataset]["source_to_target"]
        names = schema.SOURCE_CLASSES[dataset]
        return {
            names[int(key)]: None if value is None else schema.TARGET_CLASSES[value]
            for key, value in mapping.items()
        }

    # Absent or coarse source classes are never copied into a distinct target class.
    assert named(schema.GEN4_1MPX) == {
        "pedestrian": "pedestrian",
        "two_wheeler": None,
        "car": "car",
        "truck": "truck",
        "bus": "bus",
        "traffic_sign": None,
        "traffic_light": None,
    }
    # Riders are DSEC-MOT persons; DSEC-Detection has no train boxes, so its column is masked.
    assert named(schema.DSEC_DETECTION) == {
        "pedestrian": "pedestrian",
        "rider": "pedestrian",
        "car": "car",
        "bus": "bus",
        "truck": "truck",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        "train": None,
    }
    assert CONFIG[schema.DSEC_DETECTION]["masked_target_classes"] == [target["train"]]
    assert schema.HEAD_ONTOLOGY == {
        schema.GEN4_1MPX: "source",
        schema.DSEC_MOT: "target",
        schema.DSEC_DETECTION: "target",
    }
    assert CONFIG[schema.DSEC_MOT]["identity_supervision"] is True
    assert CONFIG[schema.DSEC_DETECTION]["identity_supervision"] is False


def test_frozen_splits_and_recording_exclusions():
    mot = CONFIG[schema.DSEC_MOT]["splits"]
    assert len(mot["train"]) == 9
    assert mot["val"] == ["zurich_city_01_d"]
    assert mot["test"] == ["interlaken_00_d", "zurich_city_00_b"]
    detection = CONFIG[schema.DSEC_DETECTION]
    excluded = set(detection["excluded_recordings"])
    # Every recording hosting DSEC-MOT validation or test data is excluded from DSEC-Detection.
    held_out = {schema.dsec_recording(name) for name in mot["val"] + mot["test"]}
    assert held_out <= excluded
    assert excluded == {"interlaken_00", "zurich_city_00", "zurich_city_01"}
    listed = [name for names in detection["splits"].values() for name in names]
    assert not {schema.dsec_recording(name) for name in listed} & excluded
    assert not set(listed) & {name for names in mot.values() for name in names}
    assert len(detection["splits"]["train"]) == 33
    assert detection["splits"]["val_monitor"] == [
        "zurich_city_16_a",
        "zurich_city_17_a",
        "zurich_city_18_a",
        "zurich_city_20_a",
        "zurich_city_21_a",
    ]
    finetune = CONFIG["sampling"]["finetune"]["splits"]
    assert finetune == {schema.DSEC_MOT: "train", schema.DSEC_DETECTION: "train"}


# --------------------------------------------------------------------------- temporal grid


def test_grid_snaps_60hz_and_30hz_labels_to_20hz_by_time():
    sixty = np.arange(0, 2_000_000, 16_683)
    steps, snapped = schema.build_label_grid(sixty, 50_000, 2_000)
    assert np.all(snapped) and np.array_equal(steps, sixty[::3])
    thirty = np.arange(0, 2_000_000, 33_367)
    steps, snapped = schema.build_label_grid(thirty, 50_000, 2_000)
    assert snapped.tolist()[:5] == [True, False, True, False, True]
    assert np.all(np.isin(steps[snapped], thirty))
    assert steps[1] == 50_000 and steps[2] == thirty[3]
    assert np.all((np.diff(steps) >= 48_000) & (np.diff(steps) <= 52_000))
    assert schema.build_label_grid(np.array([], np.int64), 50_000, 2_000)[0].size == 0
    with pytest.raises(schema.SchemaError):
        schema.build_label_grid(sixty, 50_000, 25_000)


def test_window_iterator_assigns_boundary_event_to_next_window():
    t = np.array([0, 49_999, 50_000, 99_999, 100_000])
    blocks = [{"t": t[:2]}, {"t": t[2:]}]
    ends = np.array([50_000, 100_000, 150_000])
    windows = list(schema.iter_window_events(blocks, ends - 50_000, ends))
    assert [window["t"].tolist() for window in windows] == [
        [0, 49_999],
        [50_000, 99_999],
        [100_000],
    ]


# --------------------------------------------------------------------------- 1Mpx conversion


def test_gen4_events_cleaned_sorted_stably_and_polarity_mapped(tmp_path):
    t = np.array([10, 30, 20, 20, 40, 35, 50, 60], np.int64)
    x = np.array([1, 2, 1300, 3, 4, 5, 6, 7])
    y = np.array([1, 1, 1, 1, 1, 1, 720, 1])
    p = np.array([1, -1, 1, -1, 1, -1, 1, 1])
    write_gen4_chunk(tmp_path, STEM, x, y, t, p, [])
    stats = {}
    blocks = list(
        lp.iter_gen4_events(
            tmp_path / "train" / f"{STEM}_td.h5", gen4_settings(), stats, block_events=3
        )
    )
    out = {key: np.concatenate([block[key] for block in blocks]) for key in ("x", "y", "t", "p")}
    assert out["t"].tolist() == [10, 20, 30, 35, 40, 60]
    assert out["x"].tolist() == [1, 3, 2, 5, 4, 7]
    assert out["p"].tolist() == [1, 0, 0, 0, 1, 1] and out["p"].dtype == np.uint8
    assert stats["dropped_out_of_sensor"] == 2 and stats["events"] == 6
    assert stats["late_events"] == 3 and stats["max_lateness_us"] == 10


def test_gen4_lateness_beyond_one_block_and_bad_polarity_rejected(tmp_path):
    t = np.array([100, 200, 300, 400, 500, 50])
    write_gen4_chunk(tmp_path, STEM, np.ones(6), np.ones(6), t, np.ones(6), [])
    with pytest.raises(schema.SchemaError, match="lateness"):
        list(
            lp.iter_gen4_events(
                tmp_path / "train" / f"{STEM}_td.h5", gen4_settings(), block_events=2
            )
        )
    with pytest.raises(schema.SchemaError, match="polarity"):
        lp.map_polarity(np.array([-1, 0, 1]), {"-1": 0, "1": 1})
    # A -1 polarity must not become True through a boolean cast.
    assert lp.map_polarity(np.array([-1, 1, -1]), {"-1": 0, "1": 1}).tolist() == [0, 1, 0]


def test_gen4_chunk_label_masks_ignore_regions_and_resume(tmp_path):
    rng = np.random.default_rng(0)
    label_times = np.arange(100_000, 1_100_000, 16_683)
    rows = []
    for k, stamp in enumerate(label_times):
        rows.append((stamp, 100, 100, 80, 60, 2, 0.99, 1))  # car inside the dense box
        rows.append((stamp, 1000, 600, 10, 10, 0, 0.95, 2))  # small pedestrian: ignore
        rows.append((stamp, 1300, 10, 50, 50, 5, 0.95, 3))  # outside the sensor: dropped
        if k == 9:
            rows.append((stamp, 0, 0, 1200, 50, 6, 0.95, 4))  # frame spanning
        if k == 3:
            rows.append((stamp, 500, 300, 60, 60, 1, 0.95, 5))  # two-wheeler without events
    sparse_step = label_times[15]
    ends = [stamp for stamp in label_times[::3] if stamp != sparse_step]
    x, y, t, p = dense_events(ends, rng, box=(100, 100, 80, 60))
    keep = ~((x >= 500) & (x < 560) & (y >= 300) & (y < 360))
    write_gen4_chunk(tmp_path, STEM, x[keep], y[keep], t[keep], p[keep], label_rows(rows))
    out = tmp_path / "out" / schema.GEN4_1MPX
    record = lp.convert_chunk(tmp_path, "train", STEM, gen4_settings(), out, block_events=5000)
    arrays = schema.read_annotations(schema.sequence_paths(out, STEM)[1])
    status, steps = arrays["step_status"], arrays["step_t_us"]
    assert np.array_equal(steps[arrays["step_snapped"]], label_times[::3][: steps.size])
    assert status[np.flatnonzero(steps == label_times[9])[0]] == schema.STEP_FRAME_SPANNING
    assert status[np.flatnonzero(steps == sparse_step)[0]] == schema.STEP_SPARSE_WINDOW
    # Unannotated steps carry no boxes: they can never become background.
    assert np.all(status[arrays["box_step"]] == schema.STEP_ANNOTATED)
    assert record["dropped_boxes"]["zero_area_after_crop"] == len(label_times)
    reasons, source = arrays["box_reason"], arrays["box_source_class"]
    pedestrian, two_wheeler, car = source == 0, source == 1, source == 2
    assert np.all(reasons[pedestrian] == schema.BOX_SMALL)
    assert np.all(arrays["box_ignore"][pedestrian])
    assert two_wheeler.sum() == 1 and reasons[two_wheeler][0] == schema.BOX_ZERO_EVENTS
    assert np.all(~arrays["box_ignore"][car])
    assert np.all(arrays["box_target_class"][car] == 0)
    assert np.all(arrays["box_target_class"][pedestrian] == 1)
    assert np.all(arrays["box_target_class"][two_wheeler] == -1)
    assert record["ontology"]["head"] == "source"
    assert record["identity"] == {
        "supervision": False,
        "namespace": f"{schema.GEN4_1MPX}/{STEM}",
        "reason": CONFIG[schema.GEN4_1MPX]["identity_reason"],
    }
    assert record["alignment"]["aligned"] is True
    json_path = schema.sequence_paths(out, STEM)[0]
    before = file_version(json_path)
    lp.convert_chunk(tmp_path, "train", STEM, gen4_settings(), out)
    assert file_version(json_path) == before
    lp.convert_chunk(tmp_path, "train", STEM, gen4_settings(min_window_events=10), out)
    assert file_version(json_path) != before


def test_malformed_gen4_labels_rejected(tmp_path):
    np.save(tmp_path / "a.npy", np.zeros(2, [("t", "<u8"), ("x", "<f4")]))
    with pytest.raises(schema.SchemaError, match="fields"):
        lp.load_gen4_labels(tmp_path / "a.npy", 60_000_000, 7)
    duplicate = label_rows([(10, 1, 1, 30, 30, 2, 1.0, 7), (10, 50, 50, 30, 30, 2, 1.0, 7)])
    np.save(tmp_path / "b.npy", duplicate)
    with pytest.raises(schema.SchemaError, match="duplicate"):
        lp.load_gen4_labels(tmp_path / "b.npy", 60_000_000, 7)
    np.save(tmp_path / "c.npy", label_rows([(70_000_000, 1, 1, 30, 30, 2, 1.0, 7)]))
    with pytest.raises(schema.SchemaError, match="outside the chunk"):
        lp.load_gen4_labels(tmp_path / "c.npy", 60_000_000, 7)
    with pytest.raises(schema.SchemaError):
        schema.gen4_chunk_identity("moorea_x_td_5_5")


# --------------------------------------------------------------------------- DSEC conversion


def test_raw_to_rectified_box_mapping():
    boxes = np.array([[10.0, 20.0, 30.5, 40.0], [620.0, 400.0, 30.0, 70.0]])
    identity = dd.map_boxes_to_rectified(boxes, affine_map(), 20)
    assert np.allclose(identity[0], boxes[0])
    # The second box leaves the raw sensor: clamped at the last pixel centre.
    assert identity[1][0] == 620 and identity[1][0] + identity[1][2] == 639
    shifted = dd.map_boxes_to_rectified(boxes[:1], affine_map(0.9, (12.0, -5.0)), 20)
    assert np.allclose(shifted[0], [10 * 0.9 + 12, 20 * 0.9 - 5, 30.5 * 0.9, 40 * 0.9])
    # A rotation-like map: every raw pixel inside the box maps inside the mapped box.
    v, u = np.mgrid[0:480, 0:640].astype(np.float64)
    curved = np.stack([u + 0.02 * (v - 240), v - 0.02 * (u - 320)], -1)
    mapped = dd.map_boxes_to_rectified(np.array([[200.0, 150.0, 60.0, 40.0]]), curved, 20)[0]
    pixels = np.mgrid[150:190, 200:260].reshape(2, -1)
    xr, yr, _ = dd.rectify_event_pixels(pixels[1], pixels[0], curved)
    assert np.all((xr >= np.floor(mapped[0]) - 1) & (xr <= np.ceil(mapped[0] + mapped[2])))
    assert np.all((yr >= np.floor(mapped[1]) - 1) & (yr <= np.ceil(mapped[1] + mapped[3])))


def test_dsec_detection_sequence_rules(tmp_path):
    rng = np.random.default_rng(1)
    t0 = 1_000_000_000
    stamps = t0 + 100_000 + np.arange(20) * 50_000
    xs, ys, ts = [], [], []
    for end in stamps:
        ts.append(rng.integers(end - 50_000, end, 2000))
        xs.append(rng.integers(0, 640, 2000))
        ys.append(rng.integers(0, 480, 2000))
        ts.append(rng.integers(end - 50_000, end, 400))
        xs.append(rng.integers(95, 155, 400))
        ys.append(rng.integers(100, 150, 400))
    t = np.concatenate(ts)
    order = np.argsort(t, kind="stable")
    x, y, t = np.concatenate(xs)[order], np.concatenate(ys)[order], t[order]
    # No raw events in [395, 435) x [300, 340): the rider box maps to [400, 440) after +5 px.
    keep = ~((x >= 395) & (x < 435) & (y >= 300) & (y < 340))
    x, y, t = x[keep], y[keep], t[keep]
    name = "zurich_city_02_a"
    write_dsec_sequence(
        tmp_path / "dsec",
        "train",
        name,
        t,
        x,
        y,
        rng.integers(0, 2, t.size),
        affine_map(1.0, (5.0, 0.0)),
        t_offset=t0,
    )
    rows = []
    for k, stamp in enumerate(stamps):
        rows.append((stamp, 95, 100, 60, 50, 2, 0.9, 1))  # car, shifted to x = 100
        rows.append((stamp, 395, 300, 40, 40, 1, 0.9, 2))  # rider without events: ignore
        rows.append((stamp, 10, 10, 10, 10, 0, 0.9, 0))  # 100 px^2 pedestrian: below floor
        if k == 2:
            rows.append((stamp, 300, 200, 40, 40, 7, 0.9, 9))  # train: unmapped class
    rows.append((t0 - 10, 95, 100, 60, 50, 2, 0.9, 1))  # before the first event: dropped
    labels_dir = tmp_path / f"labels/train/{name}/object_detections/left"
    labels_dir.mkdir(parents=True)
    np.save(labels_dir / "tracks.npy", label_rows(rows))
    found = dd.find_detection_labels(tmp_path / "labels")
    assert list(found) == [name]
    roots = {"dsec_root": tmp_path / "dsec", "dsec_detection_labels_root": tmp_path / "labels"}
    files = {
        "events": f"train/{name}/events_left/events.h5",
        "rectify_map": f"train/{name}/events_left/rectify_map.h5",
        "labels": found[name],
    }
    out = tmp_path / "out" / schema.DSEC_DETECTION
    record = dd.convert_sequence(
        schema.DSEC_DETECTION,
        name,
        "train",
        roots,
        files,
        dd.dataset_settings(CONFIG, schema.DSEC_DETECTION),
        out,
        block_events=3000,
    )
    arrays = schema.read_annotations(schema.sequence_paths(out, name)[1])
    source, reason = arrays["box_source_class"], arrays["box_reason"]
    assert record["dropped_boxes"]["outside_event_span"] == 1
    car = source == 2
    assert np.allclose(arrays["box_xywh"][car][0], [100, 100, 60, 50])
    assert np.all(arrays["box_target_class"][car] == 0) and np.all(~arrays["box_ignore"][car])
    rider = source == 1
    assert np.all(arrays["box_target_class"][rider] == 1)
    assert np.all(reason[rider] == schema.BOX_ZERO_EVENTS)
    pedestrian = source == 0
    assert np.all(reason[pedestrian] == schema.BOX_BELOW_AREA_FLOOR)
    # A stored track ID of zero is a valid ID, not "unknown".
    assert np.all(arrays["box_has_track_id"][pedestrian])
    assert np.all(arrays["box_track_id"][pedestrian] == 0)
    train = source == 7
    assert train.sum() == 1 and reason[train][0] == schema.BOX_UNMAPPED_CLASS
    assert record["identity"]["supervision"] is False
    assert record["identity"]["namespace"] == f"{schema.DSEC_DETECTION}/{name}"
    assert record["ontology"]["masked_target_classes"] == [6]
    assert record["labels"]["frame"] == "dsec_raw"
    assert record["events"]["cleaning"]["dropped_outside_rectified"] > 0


def test_dsec_mot_steps_distinguish_unannotated_from_annotated(tmp_path):
    images = 2_000_000_000 + 50_000 * np.arange(1, 7)
    record, out = convert_dsec_mot(
        tmp_path,
        "zurich_city_04_b",
        images,
        [(1, 0, 40, 40, 20, 20, 1), (3, 0, 630, 40, 20, 20, 1)],
    )
    arrays = schema.read_annotations(
        schema.sequence_paths(out / schema.DSEC_MOT, "zurich_city_04_b")[1]
    )
    assert arrays["step_status"].tolist() == [1, 0, 1, 0, 1, 1]
    assert np.array_equal(arrays["step_t_us"], images)
    assert arrays["box_track_id"].tolist() == [0, 0] and arrays["box_has_track_id"].all()
    assert np.allclose(arrays["box_xywh"][1], [630, 40, 10, 20])  # cropped to the frame
    assert record["identity"] == {"supervision": True, "namespace": "dsec_mot/zurich_city_04_b"}
    assert record["labels"]["frame"] == "dsec_rectified"
    annotation = tmp_path / "ann/train/zurich_city_04_b.txt"
    settings = dd.dataset_settings(CONFIG, schema.DSEC_MOT)
    roots = {"dsec_root": tmp_path / "dsec", "dsec_mot_annotations_root": tmp_path / "ann"}
    files = dsec_mot_files("zurich_city_04_b")
    annotation.write_text(f"{images[1] + 7},0,40,40,20,20,1\n")
    with pytest.raises(schema.SchemaError, match="not an image timestamp"):
        dd.convert_sequence(
            schema.DSEC_MOT, "zurich_city_04_b", "train", roots, files, settings, tmp_path / "o2"
        )
    annotation.write_text(f"{images[1]},3,40,40,20,20,1\n{images[2]},3,40,40,20,20,0\n")
    with pytest.raises(schema.SchemaError, match="changes class"):
        dd.convert_sequence(
            schema.DSEC_MOT, "zurich_city_04_b", "train", roots, files, settings, tmp_path / "o3"
        )


@pytest.fixture(scope="module")
def mot_record(tmp_path_factory):
    root = tmp_path_factory.mktemp("mot_record")
    images = 2_000_000_000 + 50_000 * np.arange(1, 7)
    rows = [(1, 0, 40, 40, 60, 50, 0), (1, 1, 200, 100, 60, 50, 1), (3, 0, 50, 40, 60, 50, 0)]
    record, out = convert_dsec_mot(root, "zurich_city_04_b", images, rows)
    arrays = schema.read_annotations(
        schema.sequence_paths(out / schema.DSEC_MOT, "zurich_city_04_b")[1]
    )
    return record, arrays


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda r, a: r["frames"].update(box_frame="dsec_raw"), "frames"),
        (lambda r, a: r.update(split="val_monitor"), "split"),
        (lambda r, a: r.update(recording="zurich_city_05"), "recording"),
        (lambda r, a: r.update(rectify_map=None), "rectification"),
        (lambda r, a: r["events"].update(path="/abs/events.h5"), "source-relative"),
        (lambda r, a: r["identity"].update(namespace="zurich_city_04_b"), "namespace"),
        (lambda r, a: a.__setitem__("box_track_id", np.zeros_like(a["box_track_id"])), "duplicate"),
        (
            lambda r, a: a["step_status"].__setitem__(a["box_step"][0], schema.STEP_NO_LABEL),
            "non-annotated",
        ),
        (lambda r, a: a["box_has_track_id"].__setitem__(0, False), "unknown ID"),
        (lambda r, a: a["box_target_class"].__setitem__(0, 3), "target class"),
        (lambda r, a: a["box_xywh"].__setitem__((0, 0), 639.0), "outside"),
        (lambda r, a: a.__setitem__("box_step", a["box_step"].astype(np.int64)), "dtype"),
        (lambda r, a: r["counts"].update(boxes=1), "counts"),
    ],
)
def test_invalid_records_rejected(mot_record, mutation, message):
    record, arrays = mot_record
    schema.validate_sequence(record, arrays)
    record, arrays = copy.deepcopy(record), {key: value.copy() for key, value in arrays.items()}
    mutation(record, arrays)
    with pytest.raises(schema.SchemaError, match=message):
        schema.validate_sequence(record, arrays)


def test_dsec_mot_resume_reuses_and_reconverts_on_change(tmp_path):
    images = 2_000_000_000 + 50_000 * np.arange(1, 5)
    rows = [(1, 0, 40, 40, 60, 50, 0)]
    convert_dsec_mot(tmp_path, "zurich_city_04_b", images, rows)
    json_path, npz_path = schema.sequence_paths(tmp_path / "manifest/dsec_mot", "zurich_city_04_b")
    before = file_version(json_path)
    convert_dsec_mot(tmp_path, "zurich_city_04_b", images, rows)
    assert file_version(json_path) == before
    # A corrupt annotation file is reconverted instead of reused.
    npz_path.write_bytes(b"corrupt")
    convert_dsec_mot(tmp_path, "zurich_city_04_b", images, rows)
    assert file_version(json_path) != before
    assert schema.read_annotations(npz_path)["box_track_id"].tolist() == [0]
    # Changed labels change the label hash and force a reconversion.
    before = file_version(json_path)
    convert_dsec_mot(tmp_path, "zurich_city_04_b", images, [(2, 4, 40, 40, 60, 50, 0)])
    assert file_version(json_path) != before
    assert schema.read_annotations(npz_path)["box_track_id"].tolist() == [4]


# --------------------------------------------------------------------------- splits and precedence


def test_split_overlap_rejected_at_sequence_and_recording_level():
    allowed = CONFIG["recording_overlap_allowed"]
    base = [
        entry("dsec_mot", "zurich_city_01_d", "val"),
        entry("dsec_mot", "zurich_city_01_e", "train"),
        entry("dsec_detection", "zurich_city_02_a", "train"),
        entry("gen4_1mpx", "moorea_2019-02-19_001_td_500000_60500000", "train"),
        entry("gen4_1mpx", "moorea_2019-02-19_001_td_61500000_121500000", "val"),
    ]
    summary = schema.check_split_overlap(base, allowed)
    assert summary["permitted_recording_overlaps"] == [
        {"dataset": "dsec_mot", "recording": "zurich_city_01"},
        {"dataset": "gen4_1mpx", "recording": "moorea_2019-02-19_001"},
    ]
    # DSEC-Detection training data from the validation recording.
    with pytest.raises(schema.SchemaError, match="held-out"):
        schema.check_split_overlap(
            base + [entry("dsec_detection", "zurich_city_01_a", "train")], allowed
        )
    # The same sequence labelled by both DSEC datasets: precedence must keep one copy.
    with pytest.raises(schema.SchemaError, match="precedence"):
        schema.check_split_overlap(
            base + [entry("dsec_detection", "zurich_city_01_e", "train")], allowed
        )
    # A held-out recording that is not on the allowed list.
    with pytest.raises(schema.SchemaError, match="held-out"):
        schema.check_split_overlap(base + [entry("dsec_mot", "zurich_city_02_b", "test")], allowed)
    # The DSEC-Detection monitor split is held out as well.
    with pytest.raises(schema.SchemaError, match="held-out"):
        schema.check_split_overlap(
            base
            + [
                entry("dsec_detection", "zurich_city_16_a", "val_monitor"),
                entry("dsec_detection", "zurich_city_16_b", "train"),
            ],
            allowed,
        )


def detection_label_sequences():
    """Sixty label sequences shaped like the official DSEC-Detection release."""

    detection = CONFIG[schema.DSEC_DETECTION]
    mot = [name for names in CONFIG[schema.DSEC_MOT]["splits"].values() for name in names]
    recording_excluded = [
        "interlaken_00_f",
        "interlaken_00_g",
        "zurich_city_00_a",
        "zurich_city_01_a",
        "zurich_city_01_b",
        "zurich_city_01_c",
        "zurich_city_01_f",
    ]
    names = (
        mot
        + recording_excluded
        + list(detection["excluded_sequences"])
        + [name for names in detection["splits"].values() for name in names]
    )
    assert len(names) == len(set(names)) == detection["expected_label_sequences"]
    return names


def test_dsec_detection_accounting_exclusions_and_label_precedence():
    settings = dd.dataset_settings(CONFIG, schema.DSEC_DETECTION)
    labels = detection_label_sequences()
    accounting = dd.account_detection_sequences(settings, labels)
    assignment, excluded = accounting["assignment"], accounting["excluded"]
    assert sum(split == "train" for split in assignment.values()) == 33
    assert sorted(name for name, split in assignment.items() if split == "val_monitor") == [
        "zurich_city_16_a",
        "zurich_city_17_a",
        "zurich_city_18_a",
        "zurich_city_20_a",
        "zurich_city_21_a",
    ]
    # Sequences labelled by DSEC-MOT keep only DSEC-MOT labels, whatever their split.
    for name in ("zurich_city_04_b", "zurich_city_01_d", "interlaken_00_d", "zurich_city_00_b"):
        assert excluded[name] == "label_precedence:dsec_mot"
    assert excluded["zurich_city_01_a"] == "excluded_recording:zurich_city_01"
    assert excluded["interlaken_00_f"] == "excluded_recording:interlaken_00"
    assert excluded["zurich_city_19_a"] == "excluded_sequence"
    assert not {schema.dsec_recording(name) for name in assignment} & {
        "interlaken_00",
        "zurich_city_00",
        "zurich_city_01",
    }
    bad = copy.deepcopy(settings)
    bad["splits"]["train"].append("zurich_city_01_a")
    with pytest.raises(schema.SchemaError, match="excluded"):
        dd.account_detection_sequences(bad, labels)
    bad = copy.deepcopy(settings)
    bad["splits"]["train"].append("zurich_city_04_b")
    with pytest.raises(schema.SchemaError, match="excluded"):
        dd.account_detection_sequences(bad, labels)
    with pytest.raises(schema.SchemaError, match="not accounted"):
        dd.account_detection_sequences(settings, labels + ["zurich_city_22_a"])
    with pytest.raises(schema.SchemaError, match="without labels"):
        dd.account_detection_sequences(settings, [n for n in labels if n != "thun_00_a"])
    bad = copy.deepcopy(settings)
    bad["splits"]["val_monitor"].append("zurich_city_15_a")
    with pytest.raises(schema.SchemaError, match="listed in"):
        dd.account_detection_sequences(bad, labels)


def test_output_root_validation_and_overlap_check(tmp_path):
    images = 2_000_000_000 + 50_000 * np.arange(1, 5)
    rows = [(1, 0, 40, 40, 60, 50, 0)]
    records = [
        convert_dsec_mot(tmp_path, "zurich_city_01_e", images, rows)[0],
        convert_dsec_mot(tmp_path, "zurich_city_01_d", images, rows, split="val")[0],
    ]
    settings = dd.dataset_settings(CONFIG, schema.DSEC_MOT)
    schema.write_dataset_manifest(
        tmp_path / "manifest", schema.DSEC_MOT, records, settings, complete=False, plan={}
    )
    summary = schema.validate_output_root(
        tmp_path / "manifest", CONFIG["recording_overlap_allowed"], deep=True
    )
    assert summary["datasets"] == {"dsec_mot": {"sequences": 2, "complete": False}}
    assert summary["permitted_recording_overlaps"] == [
        {"dataset": "dsec_mot", "recording": "zurich_city_01"}
    ]
    with pytest.raises(schema.SchemaError, match="held-out"):
        schema.validate_output_root(tmp_path / "manifest", {})
    npz = schema.sequence_paths(tmp_path / "manifest/dsec_mot", "zurich_city_01_e")[1]
    npz.write_bytes(npz.read_bytes() + b"x")
    with pytest.raises(schema.SchemaError, match="hash"):
        schema.validate_output_root(
            tmp_path / "manifest", CONFIG["recording_overlap_allowed"], deep=True
        )


# --------------------------------------------------------------------------- representation


def brute_force(x, y, t, p, end, geometry):
    """Independent float reference of one window (``np.add.at``, float bins)."""

    start = end - 50_000
    inside = (t >= start) & (t < end)
    xs, ys = x[inside] // geometry.pool, y[inside] // geometry.pool
    ps, tt = p[inside], t[inside].astype(np.float64)
    out = np.zeros((8, geometry.frame_height, geometry.frame_width), np.float64)
    np.add.at(out, (ps, ys, xs), 1)
    bins = np.clip(np.floor(3 * (tt - start) / 50_000), 0, 2).astype(int)
    np.add.at(out, (2 + 3 * ps + bins, ys, xs), 1)
    return np.minimum(out, 255).astype(np.uint8)


def test_causal_window_bounds_and_window_tied_bins():
    end = 1_000_000
    t = np.array([end - 50_000, end - 33_334, end - 33_333, end - 16_667, end - 16_666, end - 1])
    x = np.arange(t.size) * 2
    counts = lc.window_counts(
        x, np.zeros(t.size), t, np.ones(t.size), end, 50_000, 3, DSEC_GEOM, 255
    )
    voxel = counts[lc.VOXEL_GRID]
    assert voxel[3, 0, 0] == 1 and voxel[3, 0, 2] == 1  # bin 0: offsets 0 and 16666
    assert voxel[4, 0, 4] == 1 and voxel[4, 0, 6] == 1  # bin 1: offsets 16667 and 33333
    assert voxel[5, 0, 8] == 1 and voxel[5, 0, 10] == 1  # bin 2: offsets 33334 and 49999
    assert np.array_equal(counts[1], voxel[3:6].sum(0)) and counts[0].sum() == 0
    for outside in (end, end - 50_001):
        with pytest.raises(lc.CacheError, match="causal window"):
            lc.window_counts([0], [0], [outside], [0], end, 50_000, 3, DSEC_GEOM, 255)
    with pytest.raises(lc.CacheError, match="polarity"):
        lc.window_counts([0], [0], [end - 1], [2], end, 50_000, 3, DSEC_GEOM, 255)


def test_pooling_clipping_and_event_frame_match_established_transform():
    rng = np.random.default_rng(5)
    n = 50_000
    x, y = rng.integers(0, 1280, n), rng.integers(0, 720, n)
    t, p = rng.integers(0, 50_000, n), rng.integers(0, 2, n)
    x[:300], y[:300], p[:300] = 1279, 719, 1  # one pooled pixel above the clip
    counts = lc.window_counts(x, y, t, p, 50_000, 50_000, 3, GEN4_GEOM, 255)
    assert counts.shape == (8, 360, 640) and counts.dtype == np.uint8
    assert counts[1, 359, 639] == 255
    events = np.zeros(n, [("x", int), ("y", int), ("t", int), ("p", int)])
    events["x"], events["y"], events["t"], events["p"] = x // 2, y // 2, t, p
    reference = EventPreprocessor(360, 640, "event_frame")(events)
    assert np.array_equal(counts[:2], np.minimum(reference, 255).astype(np.uint8))
    unclipped = lc.window_counts(
        x[300:], y[300:], t[300:], p[300:], 50_000, 50_000, 3, GEN4_GEOM, 255
    ).astype(int)
    assert np.array_equal(unclipped[0], unclipped[2:5].sum(0))
    assert np.array_equal(unclipped[1], unclipped[5:8].sum(0))


def test_normalization_numpy_and_torch_agree():
    counts = np.array([0, 1, 4, 255], np.uint8)
    for scale in (1, 4):
        expected = np.log1p(counts / scale) / np.log1p(255)
        assert np.allclose(lc.normalize_counts(counts, scale), expected)
        assert np.allclose(lc.normalize_counts(torch.from_numpy(counts), scale).numpy(), expected)
    assert lc.normalize_counts(np.array([255], np.uint8), 1)[0] == pytest.approx(1.0)


# --------------------------------------------------------------------------- cache


@pytest.fixture(scope="module")
def gen4_source(tmp_path_factory):
    """Synthetic chunk with out-of-sensor events, late (unsorted) events, and -1/+1."""

    root = tmp_path_factory.mktemp("gen4")
    rng = np.random.default_rng(0)
    label_times = np.arange(100_000, 1_300_000, 16_683)
    rows = []
    for stamp in label_times:
        rows.append((stamp, 100, 100, 80, 60, 2, 0.99, 1))
        rows.append((stamp, 1000, 600, 10, 10, 0, 0.95, 2))
    t = np.concatenate([rng.integers(0, 1_350_000, 400_000), rng.integers(0, 1_350_000, 100_000)])
    x = np.concatenate([rng.integers(0, 1290, 400_000), rng.integers(100, 180, 100_000)])
    y = np.concatenate([rng.integers(0, 725, 400_000), rng.integers(100, 160, 100_000)])
    order = np.argsort(t, kind="stable")
    t, x, y = t[order], x[order], y[order]
    swap = np.arange(10, t.size - 1, 997)  # bounded lateness
    t[swap], t[swap + 1] = t[swap + 1].copy(), t[swap].copy()
    t[0], t[1] = 0, 0
    p = rng.choice([-1, 1], t.size)
    write_gen4_chunk(root, STEM, x, y, t, p, label_rows(rows))
    out = root / "manifest"
    settings = lp.dataset_settings(CONFIG)
    record = lp.convert_chunk(
        root, "train", STEM, settings, out / schema.GEN4_1MPX, block_events=60_000
    )
    schema.write_dataset_manifest(
        out, schema.GEN4_1MPX, [record], settings, complete=False, plan={}
    )
    return {"root": root, "manifest": out, "x": x, "y": y, "t": t, "p": p}


@pytest.fixture(scope="module")
def dsec_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("dsec")
    t0 = 2_000_000_000
    images = t0 + 50_000 * np.arange(1, 13)
    rng = np.random.default_rng(3)
    t = np.sort(np.concatenate([rng.integers(t0, images[-1] + 1, 60_000), np.full(3, images[2])]))
    x = rng.integers(0, 640, t.size)
    y = rng.integers(0, 480, t.size)
    p = rng.integers(0, 2, t.size)
    name = "zurich_city_04_b"
    write_dsec_sequence(root / "dsec", "train", name, t, x, y, p, affine_map(1.0, (3.4, -2.6)), t0)
    (root / f"dsec/train/{name}/{name}_image_timestamps.txt").write_text(
        "\n".join(str(value) for value in images)
    )
    annotations = root / "ann/train"
    annotations.mkdir(parents=True)
    (annotations / f"{name}.txt").write_text(
        "".join(f"{images[k]},{k % 2},40,40,60,50,{k % 2}\n" for k in (1, 2, 3, 5, 8))
    )
    settings = dd.dataset_settings(CONFIG, schema.DSEC_MOT)
    roots = {"dsec_root": root / "dsec", "dsec_mot_annotations_root": root / "ann"}
    out = root / "manifest"
    record = dd.convert_sequence(
        schema.DSEC_MOT,
        name,
        "train",
        roots,
        dsec_mot_files(name),
        settings,
        out / schema.DSEC_MOT,
        block_events=7_000,
    )
    schema.write_dataset_manifest(out, schema.DSEC_MOT, [record], settings, complete=False, plan={})
    return {
        "root": root,
        "manifest": out,
        "name": name,
        "t": t,
        "x": x,
        "y": y,
        "p": p,
        "images": images,
    }


def first_job(source, dataset):
    directory = source["manifest"] / dataset
    return schema.load_sequence(
        directory, schema.read_json(directory / "manifest.json")["sequences"][0]
    )


def test_gen4_cache_equals_independent_source_computation(gen4_source, tmp_path):
    record, arrays = first_job(gen4_source, schema.GEN4_1MPX)
    entry_ = lc.build_sequence_cache(
        record, arrays, {"gen4_root": gen4_source["root"]}, CONFIG, tmp_path, block_events=50_000
    )
    assert entry_["reused"] is False
    s = gen4_source
    inside = (s["x"] < 1280) & (s["y"] < 720)
    order = np.argsort(s["t"][inside], kind="stable")
    x, y, t = s["x"][inside][order], s["y"][inside][order], s["t"][inside][order]
    p = (s["p"][inside][order] == 1).astype(int)
    with h5py.File(lc.cache_path(tmp_path, schema.GEN4_1MPX, STEM), "r") as handle:
        cached = handle["counts"][:]
        assert np.array_equal(handle["step_t_us"][:], arrays["step_t_us"])
    for step, end in enumerate(arrays["step_t_us"]):
        assert np.array_equal(cached[step], brute_force(x, y, t, p, int(end), GEN4_GEOM)), step
    assert int(arrays["step_t_us"][0]) == 100_000 and cached[0].sum() > 0


def test_dsec_cache_equals_independent_source_computation(dsec_source, tmp_path):
    record, arrays = first_job(dsec_source, schema.DSEC_MOT)
    lc.build_sequence_cache(
        record,
        arrays,
        {"dsec_root": dsec_source["root"] / "dsec"},
        CONFIG,
        tmp_path,
        block_events=5_000,
    )
    xr = np.rint(dsec_source["x"] + 3.4).astype(int)
    yr = np.rint(dsec_source["y"] - 2.6).astype(int)
    inside = (xr >= 0) & (xr < 640) & (yr >= 0) & (yr < 480)
    with h5py.File(lc.cache_path(tmp_path, schema.DSEC_MOT, dsec_source["name"]), "r") as handle:
        cached = handle["counts"][:]
    assert np.array_equal(arrays["step_t_us"], dsec_source["images"])
    t, p = dsec_source["t"][inside], dsec_source["p"][inside]
    for step, end in enumerate(arrays["step_t_us"]):
        reference = brute_force(xr[inside], yr[inside], t, p, int(end), DSEC_GEOM)
        assert np.array_equal(cached[step], reference), step
    # Events at exactly images[2] belong to the window ending at images[3].
    on_bound = (t >= dsec_source["images"][2]) & (t < dsec_source["images"][3])
    assert int(cached[3].astype(int)[:2].sum()) == int(on_bound.sum())
    assert int(arrays["step_event_count"][3]) == int(on_bound.sum())


def test_cache_resume_stale_key_and_incomplete_file(gen4_source, tmp_path):
    record, arrays = first_job(gen4_source, schema.GEN4_1MPX)
    roots = {"gen4_root": gen4_source["root"]}
    path = lc.cache_path(tmp_path, schema.GEN4_1MPX, STEM)
    lc.build_sequence_cache(record, arrays, roots, CONFIG, tmp_path)
    before = file_version(path)
    assert lc.build_sequence_cache(record, arrays, roots, CONFIG, tmp_path)["reused"] is True
    assert file_version(path) == before
    (path.parent / f".{path.name}.999.tmp").write_bytes(b"partial")
    with h5py.File(path, "r+") as handle:
        handle.attrs["complete"] = False
    assert lc.build_sequence_cache(record, arrays, roots, CONFIG, tmp_path)["reused"] is False
    assert not list(path.parent.glob("*.tmp"))
    changed = copy.deepcopy(CONFIG)
    changed["representation"]["version"] = 2
    stale = schema.settings_hash(lc.cache_key(record, arrays["step_t_us"], changed))
    with pytest.raises(lc.CacheError, match="stale"):
        lc.CacheReader(path, stale, arrays["step_t_us"].size, GEN4_GEOM).read(0, 1)
    assert lc.build_sequence_cache(record, arrays, roots, changed, tmp_path)["reused"] is False
    reader = lc.CacheReader(path, stale, arrays["step_t_us"].size, GEN4_GEOM)
    assert reader.read(2, 3).shape == (3, 8, 360, 640)
    with pytest.raises(lc.CacheError, match="out of range"):
        reader.read(arrays["step_t_us"].size - 1, 2)


@pytest.mark.parametrize(
    "change",
    [
        lambda c, r: c.update(window_us=40_000),
        lambda c, r: c["representation"].update(voxel_bins=4),
        lambda c, r: c["geometry"]["gen4_1mpx"]["input"].update(height=400),
        lambda c, r: c["cache"].update(format_version=2),
        lambda c, r: r["events"].update(sha256="0" * 64),
        lambda c, r: r.update(converter_version="gen4_1mpx/2"),
    ],
)
def test_cache_key_covers_every_input(gen4_source, change):
    record, arrays = first_job(gen4_source, schema.GEN4_1MPX)
    reference = schema.settings_hash(lc.cache_key(record, arrays["step_t_us"], CONFIG))
    config, record = copy.deepcopy(CONFIG), copy.deepcopy(record)
    change(config, record)
    try:
        changed = schema.settings_hash(lc.cache_key(record, arrays["step_t_us"], config))
    except lc.CacheError:
        return  # inconsistent inputs are rejected outright
    assert changed != reference
    shifted = arrays["step_t_us"].copy()
    shifted[-1] += 1
    assert schema.settings_hash(lc.cache_key(record, shifted, config)) != reference


def test_event_count_disagreement_aborts_without_output(gen4_source, tmp_path):
    record, arrays = first_job(gen4_source, schema.GEN4_1MPX)
    arrays = dict(arrays)
    arrays["step_event_count"] = arrays["step_event_count"].copy()
    arrays["step_event_count"][4] += 1
    with pytest.raises(lc.CacheError, match="step 4"):
        lc.build_sequence_cache(
            record, arrays, {"gen4_root": gen4_source["root"]}, CONFIG, tmp_path
        )
    assert not any((tmp_path / schema.GEN4_1MPX).iterdir())


def test_cache_plan_projects_space_and_dry_run_writes_nothing(gen4_source, tmp_path, capsys):
    job = first_job(gen4_source, schema.GEN4_1MPX)
    cache = tmp_path / "cache"
    plan = lc.plan_cache([job], CONFIG, cache)
    raw = job[1]["step_t_us"].size * 8 * 360 * 640
    assert plan["pending_raw_bytes"] == raw and plan["compressed_fraction_source"] == "assumed"
    assert plan["projected_bytes"] == int(np.ceil(raw * 0.3)) and plan["fits"]
    tight = copy.deepcopy(CONFIG)
    tight["cache"]["free_space_margin_gib"] = 1e9
    with pytest.raises(lc.CacheError, match="exceeds"):
        lc.check_space(lc.plan_cache([job], tight, cache))
    argv = [
        "--manifest-root",
        str(gen4_source["manifest"]),
        "--config",
        str(CONFIG_PATH),
        "--cache-root",
        str(cache),
        "--dataset",
        "gen4_1mpx",
        "--splits",
        "train",
        "--dry-run",
    ]
    assert lc.main(argv) == 0
    assert not cache.exists() and '"pending": 1' in capsys.readouterr().out
    lc.build_sequence_cache(*job, {"gen4_root": gen4_source["root"]}, CONFIG, cache)
    plan = lc.plan_cache([job], CONFIG, cache)
    assert plan["complete"] == 1 and plan["pending"] == 0
    assert plan["compressed_fraction_source"] == "measured" and plan["projected_bytes"] == 0


# --------------------------------------------------------------------------- clips and augmentation


def built_sequence(source, dataset, cache_root, roots):
    lc.build_sequence_cache(*first_job(source, dataset), roots, CONFIG, cache_root)
    return ld.load_sequences(
        source["manifest"], cache_root, CONFIG, dataset, ["train"], require_complete=False
    )[0]


def test_loader_rejects_partial_manifest_and_changed_settings(gen4_source, tmp_path):
    with pytest.raises(schema.SchemaError, match="partial"):
        ld.load_sequences(gen4_source["manifest"], tmp_path, CONFIG, schema.GEN4_1MPX, ["train"])
    changed = copy.deepcopy(CONFIG)
    changed[schema.GEN4_1MPX]["min_window_events"] = 1
    with pytest.raises(schema.SchemaError, match="settings differ"):
        ld.load_sequences(
            gen4_source["manifest"], tmp_path, changed, schema.GEN4_1MPX, ["train"], False
        )
    sequence = ld.load_sequences(
        gen4_source["manifest"], tmp_path, CONFIG, schema.GEN4_1MPX, ["train"], False
    )[0]
    with pytest.raises(lc.CacheError, match="missing"):
        sequence.reader.read(0, 1)


def test_clip_masks_tail_padding_and_source_targets(gen4_source, tmp_path):
    sequence = built_sequence(
        gen4_source, schema.GEN4_1MPX, tmp_path, {"gen4_root": gen4_source["root"]}
    )
    data = ld.LargeEventClipDataset([sequence], CONFIG)
    clip = data[ld.ClipSpec("stream", schema.GEN4_1MPX, STEM, sequence.steps - 2, 2, 5, False)]
    assert clip["event_frame"].shape == (5, 2, 384, 640)
    assert clip["voxel_grid"].shape == (5, 6, 384, 640)
    assert clip["event_frame"].dtype == torch.uint8
    assert clip["valid"].tolist() == [True, True, False, False, False]
    assert clip["supervised"].tolist()[2:] == [False] * 3
    assert clip["step_index"].tolist()[2:] == [-1, -1, -1]
    assert clip["step_t_us"].tolist()[2:] == [-1, -1, -1]
    assert all(target["boxes_xyxy"].shape == (0, 4) for target in clip["targets"][2:])
    assert not clip["event_frame"][2:].any() and not clip["voxel_grid"][:, :, 360:].any()
    raw = sequence.reader.read(sequence.steps - 2, 2)
    assert np.array_equal(clip["event_frame"][:2, :, :360].numpy(), raw[:, :2])
    assert np.array_equal(clip["voxel_grid"][:2, :, :360].numpy(), raw[:, 2:])
    assert clip["reset"] is False
    first = data[ld.ClipSpec("random", schema.GEN4_1MPX, STEM, 0, 5, 5, True)]
    target = first["targets"][0]
    assert torch.allclose(
        target["boxes_xyxy"][~target["ignore"]][0], torch.tensor([50.0, 50.0, 90.0, 80.0])
    )
    assert target["labels"].tolist() == [2, 0]  # native head: car, pedestrian
    assert target["ignore"].tolist() == [False, True]
    assert not target["identity_valid"].any() and target["identity"].tolist() == [-1, -1]
    assert first["class_mask"].all() and first["count_scale"] == 4.0
    with pytest.raises(ld.SamplingError, match="leaves the sequence"):
        data[ld.ClipSpec("random", schema.GEN4_1MPX, STEM, sequence.steps - 2, 5, 5, True)]
    event_frame, voxel_grid = ld.network_input(ld.collate_clips([first, clip]))
    assert event_frame.shape == (2, 5, 2, 384, 640) and voxel_grid.dtype == torch.float32
    assert float(event_frame.max()) <= 1.0
    expected = np.log1p(raw[0, :2].astype(np.float32) / 4) / np.log1p(255)
    assert np.allclose(event_frame[1, 0, :, :360].numpy(), expected, atol=1e-6)


def test_dsec_mot_identity_targets_and_flipped_clip(dsec_source, tmp_path):
    sequence = built_sequence(
        dsec_source, schema.DSEC_MOT, tmp_path, {"dsec_root": dsec_source["root"] / "dsec"}
    )
    namespace = f"dsec_mot/{dsec_source['name']}"
    table = ld.IdentityTable.from_sequences([sequence])
    assert table.keys == ((namespace, 0), (namespace, 1))
    data = ld.LargeEventClipDataset([sequence], CONFIG, table)
    spec = ld.ClipSpec(
        "random",
        schema.DSEC_MOT,
        dsec_source["name"],
        1,
        4,
        4,
        True,
        augmentation=ld.AugmentationParams(hflip=True),
    )
    clip = data[spec]
    assert clip["supervised"].tolist() == [True, True, True, False]
    target = clip["targets"][0]
    assert target["identity"].tolist() == [1] and target["identity_valid"].tolist() == [True]
    assert torch.allclose(
        target["boxes_xyxy"][0], torch.tensor([640 - 100.0, 40.0, 640 - 40.0, 90.0])
    )
    assert clip["targets"][3]["boxes_xyxy"].shape == (0, 4)  # unannotated, not background
    assert np.array_equal(clip["event_frame"].numpy(), sequence.reader.read(1, 4)[:, :2, :, ::-1])
    assert clip["reset"] is True and clip["meta"]["augmentation"]["hflip"] is True
    assert clip["meta"]["namespace"] == namespace and clip["meta"]["identity_supervision"]
    assert clip["class_mask"].all()


def test_nearest_exact_index_matches_torch():
    for out_size, in_size in ((640, 426), (480, 320), (533, 640), (400, 480), (360, 300)):
        source = torch.arange(in_size, dtype=torch.float32).view(1, 1, 1, in_size)
        expected = F.interpolate(source, size=(1, out_size), mode="nearest-exact")[0, 0, 0]
        assert np.array_equal(ld._nearest_exact_index(out_size, in_size), expected.long().numpy())


@pytest.mark.parametrize(
    "params",
    [
        ld.AugmentationParams(hflip=True),
        ld.AugmentationParams(zoom="in", factor=1.37, origin_u=(0.3, 0.8)),
        ld.AugmentationParams(hflip=True, zoom="out", factor=1.17, origin_u=(0.9, 0.1)),
    ],
)
def test_box_and_count_transforms_stay_aligned(params):
    width, height = 640, 480
    boxes = np.array([[200, 150, 260, 200], [400, 300, 455, 371]], np.float64)
    counts = np.zeros((1, 8, height, width), np.uint8)
    for x0, y0, x1, y1 in boxes.astype(int):
        counts[0, :, y0:y1, x0:x1] = 1
    resolved = ld.resolve_augmentation(params, width, height, boxes[:1])
    out = resolved.transform_counts(counts)[0, 0]
    moved = resolved.transform_boxes(boxes)
    covered = np.zeros_like(out, bool)
    for x0, y0, x1, y1 in moved:
        rows = slice(int(np.floor(y0)), int(np.ceil(y1)))
        cols = slice(int(np.floor(x0)), int(np.ceil(x1)))
        assert out[rows, cols].any()
        covered[rows, cols] = True
    assert covered[out > 0].all()
    for x0, y0, x1, y1 in moved:
        interior = out[
            max(int(np.ceil(y0)) + 1, 0) : int(np.floor(y1)) - 1,
            max(int(np.ceil(x0)) + 1, 0) : int(np.floor(x1)) - 1,
        ]
        assert interior.size and interior.all()
    if params.zoom == "in":
        x0, y0, window_w, window_h = resolved.window
        assert boxes[0, 0] >= x0 and boxes[0, 2] <= x0 + window_w
        assert boxes[0, 1] >= y0 and boxes[0, 3] <= y0 + window_h


def test_flip_round_trip_is_exact():
    flip = ld.ResolvedAugmentation(True, "none", (0, 0, 640, 480), 640, 480)
    boxes = np.array([[0.0, 0.0, 640.0, 480.0], [10.5, 3.0, 20.0, 9.0]])
    assert np.allclose(flip.transform_boxes(flip.transform_boxes(boxes)), boxes)
    assert np.allclose(flip.transform_boxes(boxes)[0], boxes[0])
    counts = np.random.default_rng(0).integers(0, 9, (2, 8, 480, 640)).astype(np.uint8)
    assert np.array_equal(flip.transform_counts(flip.transform_counts(counts)), counts)


def test_clipped_boxes_become_ignore_or_are_removed():
    boxes = np.array(
        [
            [-50.0, 10.0, 50.0, 110.0],  # half visible: stays positive
            [-90.0, 10.0, 10.0, 110.0],  # 10% visible: ignore
            [-60.0, 200.0, -10.0, 260.0],  # fully outside: removed
            [630.0, 10.0, 660.0, 30.0],  # clipped to 10x20: below the area floor
            [100.0, 100.0, 110.0, 110.0],  # small but not clipped: unchanged
        ]
    )
    rule = ld.small_box_rule(CONFIG, schema.DSEC_DETECTION)
    xyxy, ignore, keep, counts = ld.clip_transformed_boxes(
        boxes, np.zeros(5, bool), 640, 480, rule, 0.3
    )
    assert keep.tolist() == [True, True, False, True, True]
    assert ignore.tolist() == [False, True, False, True, False]
    assert np.allclose(xyxy[0], [0, 10, 50, 110])
    assert counts == {"boxes_removed": 1, "positives_removed": 1, "positives_to_ignore": 2}
    gen4 = ld.small_box_rule(CONFIG, schema.GEN4_1MPX)
    assert gen4 == {"min_diagonal": 30, "min_side": 10, "min_area": None}
    assert ld.small_box_rule(CONFIG, schema.DSEC_MOT) == {
        "min_diagonal": None,
        "min_side": None,
        "min_area": None,
    }


# --------------------------------------------------------------------------- identities, samplers


def fake_sequence(dataset, name, steps, supervised_every=1, identity=False):
    supervised = np.zeros(steps, bool)
    supervised[::supervised_every] = True
    empty = {
        "xyxy": np.zeros((0, 4)),
        "label": np.zeros(0, int),
        "ignore": np.zeros(0, bool),
        "track_id": np.zeros(0, int),
        "has_track_id": np.zeros(0, bool),
    }
    return ld.SequenceData(
        dataset=dataset,
        sequence=name,
        split="train",
        namespace=f"{dataset}/{name}",
        identity_supervision=identity,
        head="target",
        geometry=DSEC_GEOM,
        step_t_us=np.arange(steps, dtype=np.int64) * 50_000,
        supervised=supervised,
        box_offsets=np.zeros(steps + 1, int),
        boxes=empty,
        class_mask=np.ones(7, bool),
        small_box={},
        reader=None,
    )


def finetune_sequences():
    mot = [
        fake_sequence(schema.DSEC_MOT, f"interlaken_00_{c}", n, identity=True)
        for c, n in zip("abce", (37, 20, 5, 64))
    ]
    detection = [
        fake_sequence(schema.DSEC_DETECTION, f"zurich_city_1{k}_a", 10 + 7 * k, 3) for k in range(5)
    ]
    return {schema.DSEC_MOT: mot, schema.DSEC_DETECTION: detection}


def with_boxes(sequence, track_ids, ignore=None, has_track_id=None):
    count = len(track_ids)
    sequence.boxes = {
        "xyxy": np.zeros((count, 4)),
        "label": np.zeros(count, int),
        "ignore": np.zeros(count, bool) if ignore is None else np.asarray(ignore),
        "track_id": np.asarray(track_ids),
        "has_track_id": np.ones(count, bool) if has_track_id is None else np.asarray(has_track_id),
    }
    return sequence


def test_identity_namespaces_never_collide():
    # Track ID 3 appears in two DSEC-MOT sequences and in a DSEC-Detection sequence.
    first = with_boxes(fake_sequence(schema.DSEC_MOT, "interlaken_00_a", 3, identity=True), [3, 0])
    second = with_boxes(fake_sequence(schema.DSEC_MOT, "interlaken_00_b", 3, identity=True), [3])
    detection = with_boxes(fake_sequence(schema.DSEC_DETECTION, "interlaken_00_a", 3), [3, 7])
    table = ld.IdentityTable.from_sequences([first, second, detection])
    assert table.keys == (
        ("dsec_mot/interlaken_00_a", 0),
        ("dsec_mot/interlaken_00_a", 3),
        ("dsec_mot/interlaken_00_b", 3),
    )
    a = table.lookup("dsec_mot/interlaken_00_a", np.array([3]))[0]
    b = table.lookup("dsec_mot/interlaken_00_b", np.array([3]))[0]
    assert a != b and a >= 0 and b >= 0
    # A detection-only namespace with the same name never resolves to an identity.
    assert table.lookup("dsec_detection/interlaken_00_a", np.array([3, 7])).tolist() == [-1, -1]
    # Ignore boxes and unknown IDs never create identities; ID zero is a valid identity.
    mixed = with_boxes(
        fake_sequence(schema.DSEC_MOT, "zurich_city_04_b", 3, identity=True),
        [5, 0, 9, -1],
        ignore=[False, False, True, False],
        has_track_id=[True, True, True, False],
    )
    table = ld.IdentityTable.from_sequences([mixed])
    assert table.keys == (("dsec_mot/zurich_city_04_b", 0), ("dsec_mot/zurich_city_04_b", 5))
    assert table.lookup("dsec_mot/zurich_city_04_b", np.array([5, 9, 0])).tolist() == [1, -1, 0]
    assert table.sha256() != ld.IdentityTable.from_sequences([first]).sha256()
    with pytest.raises(schema.SchemaError, match="duplicate"):
        ld.IdentityTable([("a", 1), ("a", 1)])


def test_identity_targets_inactive_for_detection_only_clips():
    sequence = with_boxes(fake_sequence(schema.DSEC_DETECTION, "zurich_city_10_a", 2), [4])
    sequence.boxes["xyxy"] = np.array([[10.0, 10.0, 60.0, 60.0]])
    sequence.box_offsets = np.array([0, 1, 1])

    class Reader:
        def read(self, start, count):
            return np.zeros((count, 8, 480, 640), np.uint8)

    sequence.reader = Reader()
    table = ld.IdentityTable([("dsec_detection/zurich_city_10_a", 4)])
    data = ld.LargeEventClipDataset([sequence], CONFIG, table)
    clip = data[ld.ClipSpec("random", schema.DSEC_DETECTION, "zurich_city_10_a", 0, 2, 2, True)]
    target = clip["targets"][0]
    assert target["track_ids"].tolist() == [4]
    assert target["identity"].tolist() == [-1] and target["identity_valid"].tolist() == [False]


def test_mixed_plan_composition_provenance_and_determinism():
    plan = ld.BatchPlan("finetune", CONFIG, finetune_sequences(), seed=7)
    again = ld.BatchPlan("finetune", CONFIG, finetune_sequences(), seed=7)
    other = ld.BatchPlan("finetune", CONFIG, finetune_sequences(), seed=8)
    batches = [plan.batch(step) for step in range(60)]
    assert batches == [again.batch(step) for step in range(60)]
    assert batches != [other.batch(step) for step in range(60)]
    sequences = {s.sequence: s for group in finetune_sequences().values() for s in group}
    for specs in batches:
        assert [(spec.kind, spec.dataset) for spec in specs] == (
            [("random", schema.DSEC_MOT)] * 2
            + [("random", schema.DSEC_DETECTION)] * 2
            + [("stream", schema.DSEC_MOT)] * 2
            + [("stream", schema.DSEC_DETECTION)] * 2
        )
        for spec in specs:
            assert spec.steps == 8 and spec.dataset in sequences[spec.sequence].namespace
            if spec.kind == "random":
                assert spec.reset and spec.length == 8
                assert spec.start + 8 <= sequences[spec.sequence].steps
                assert sequences[spec.sequence].supervised[spec.start : spec.start + 8].any()
            else:
                assert spec.augmentation.zoom in ("none", "out")
    micro = plan.micro_batches(batches[0], 2)
    assert [[(spec.kind, spec.slot) for spec in group] for group in micro] == [
        [("random", 0), ("random", 2), ("stream", 0), ("stream", 2)],
        [("random", 1), ("random", 3), ("stream", 1), ("stream", 3)],
    ]
    with pytest.raises(ld.SamplingError):
        plan.micro_batches(batches[0], 3)


def test_random_starts_are_uniform_over_supervised_clips():
    sequences = [
        fake_sequence(schema.DSEC_MOT, "interlaken_00_a", 12, 20),
        fake_sequence(schema.DSEC_MOT, "interlaken_00_b", 3),
    ]
    assert ld.RandomClipSampler(sequences, 4).total == 1  # b is shorter than T
    sequences[0].supervised[[5, 11]] = True
    sampler = ld.RandomClipSampler(sequences, 4)
    assert sampler.starts[0].tolist() == [0, 2, 3, 4, 5, 8]
    draws = [sampler.draw(ld.counter_rng(1, k))[1] for k in range(3000)]
    frequencies = np.bincount(draws, minlength=9)[[0, 2, 3, 4, 5, 8]] / 3000
    assert np.all(np.abs(frequencies - 1 / 6) < 0.03)
    with pytest.raises(ld.SamplingError):
        ld.RandomClipSampler([fake_sequence(schema.DSEC_MOT, "interlaken_00_c", 3)], 4)


def test_stream_segments_cover_sequences_with_tails_and_resets():
    sequences = finetune_sequences()[schema.DSEC_MOT]
    lengths = {s.sequence: s.steps for s in sequences}
    schedule = ld.StreamSchedule(sequences, 2, 8, seed=3, stage="finetune", dataset="dsec_mot")
    for slot in range(2):
        previous = None
        for k in range(80):
            segment = schedule.segment(slot, k)
            name = segment.sequence.sequence
            if previous is None or segment.reset:
                assert segment.start == 0
                if previous is not None:  # the previous sequence ran to its end first
                    assert previous.start + previous.length == lengths[previous.sequence.sequence]
            else:
                assert name == previous.sequence.sequence
                assert segment.start == previous.start + 8
            assert segment.length == min(8, lengths[name] - segment.start)
            previous = segment
    for epoch in range(3):
        visited = []
        for slot in range(2):
            order, _ = schedule._epochs[slot][epoch]
            visited += [sequences[index].sequence for index in order]
        assert sorted(visited) == sorted(lengths)
    with pytest.raises(ld.SamplingError):
        ld.StreamSchedule(sequences, 5, 8, 3, "finetune", "dsec_mot")


def test_resumed_sampler_reproduces_uninterrupted_order():
    plan = ld.BatchPlan("finetune", CONFIG, finetune_sequences(), seed=11)
    full = list(ld.BatchPlanSampler(plan, 0, 40, micro_batches=2))
    resumed_plan = ld.BatchPlan("finetune", CONFIG, finetune_sequences(), seed=11)
    resumed = list(ld.BatchPlanSampler(resumed_plan, 20, 20, micro_batches=2))
    assert full[40:] == resumed and len(full) == 80


def test_eval_clips_cover_sequence_in_order_with_one_reset():
    sequence = fake_sequence(schema.DSEC_MOT, "zurich_city_01_d", 21)
    clips = list(ld.sequence_clips(sequence, 8))
    assert [(c.start, c.length, c.reset) for c in clips] == [
        (0, 8, True),
        (8, 8, False),
        (16, 5, False),
    ]
    assert all(c.augmentation == ld.AugmentationParams() for c in clips)
