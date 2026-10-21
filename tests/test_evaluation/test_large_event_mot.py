"""Tests for large event MOT exports, BoT-SORT with model embeddings, and seven-class TrackEval.

Tracking needs BoxMOT and the pinned TrackEval checkout (``TRACKEVAL_ROOT``, default
``external/TrackEval``); exports also need the pinned RVT checkout in ``RVT_ROOT``.
Without them the affected tests skip. Sequences are synthetic and evaluated on the CPU.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import src.data.large_event_dataset as dataset_module
from src.data.large_event_dataset import (
    LargeEventClipDataset,
    collate_clips,
    network_input,
    sequence_clips,
)
from src.evaluation import large_event_export as ex
from src.evaluation import large_event_mot as mot
from src.experiments import tracker_tuning
from src.models.large_event_detector import (
    build_large_event_detector,
    config_sha256,
    load_model_config,
)
from src.training.large_event_losses import load_training_config
from src.training.large_event_runtime import smoke_configs, synthetic_sequences

REPO = Path(__file__).resolve().parents[2]
RVT_ROOT = os.environ.get("RVT_ROOT")
TRACKEVAL = Path(os.environ.get("TRACKEVAL_ROOT", REPO / "external/TrackEval"))
needs_rvt = pytest.mark.skipif(
    not RVT_ROOT
    or not Path(RVT_ROOT).is_dir()
    or not all(importlib.util.find_spec(name) for name in ("einops", "omegaconf")),
    reason="set RVT_ROOT to the pinned RVT checkout (needs einops and omegaconf)",
)
needs_tracking = pytest.mark.skipif(
    importlib.util.find_spec("boxmot") is None or not (TRACKEVAL / "trackeval").is_dir(),
    reason="needs boxmot and the pinned TrackEval checkout (TRACKEVAL_ROOT)",
)


def quiet(message):
    return None


@pytest.fixture(scope="module")
def configs():
    _, data, model = smoke_configs(
        load_training_config(),
        json.loads((REPO / "configs/large_event_mot/data.json").read_text()),
        load_model_config(),
    )
    data = copy.deepcopy(data)
    data["dsec_mot"]["splits"] = {
        "train": ["synthetic_train_00"],
        "val": ["synthetic_val_00"],
        "test": ["synthetic_test_00"],
    }
    return data, model, ex.load_evaluation_config()


def sequences_for(data, split, steps=10, objects=3, seed=0):
    return synthetic_sequences(data, "dsec_mot", split, 1, steps, objects, seed)


def write_dsec_root(root: Path, sequences, directory: str):
    """Image timestamps and DSEC-MOT ground truth (rectified pixels) of synthetic sequences."""

    for sequence in sequences:
        pool = sequence.geometry.pool
        folder = root / directory / sequence.sequence
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"{sequence.sequence}_image_timestamps.txt").write_text(
            "\n".join(str(t) for t in sequence.step_t_us) + "\n"
        )
        lines = []
        for step in range(sequence.steps):
            if not sequence.supervised[step]:
                continue
            boxes = sequence.step_boxes(step)
            for xyxy, label, track in zip(boxes["xyxy"], boxes["label"], boxes["track_id"]):
                x0, y0, x1, y1 = xyxy * pool
                lines.append(
                    f"{sequence.step_t_us[step]},{track},{x0:.3f},{y0:.3f},"
                    f"{x1 - x0:.3f},{y1 - y0:.3f},{label}"
                )
        annotations = root / "annotations" / directory
        annotations.mkdir(parents=True, exist_ok=True)
        (annotations / f"{sequence.sequence}.txt").write_text("\n".join(lines) + "\n")


def save_checkpoint(path: Path, model_config, step: int, force: bool = True, seed: int = 0):
    """A fine-tuning inference checkpoint; ``force`` makes the head emit confident boxes."""

    torch.manual_seed(seed)
    model = build_large_event_detector(
        model_config, ontology="target", geometry="dsec_mot", init="random", rvt_root=RVT_ROOT
    )
    if force:
        with torch.no_grad():
            for conv in model.yolox_head.obj_preds:
                conv.bias.add_(8.0)
            for conv in model.yolox_head.cls_preds:
                conv.bias.add_(torch.linspace(-2, 4, conv.bias.numel()))
                conv.weight.normal_(0, 0.05 * (seed + 1))
    payload = {
        "format": "large_event_mot_inference",
        "version": 1,
        "stage": "finetune",
        "step": step,
        "weights": "ema",
        "ontology": "target",
        "classes": list(model.classes),
        "geometry": "dsec_mot",
        "model_config": model_config,
        "model_config_sha256": config_sha256(model_config),
        "model_state": model.state_dict(),
        "metadata": json.loads(json.dumps(model.metadata, default=str)),
        "metrics": None,
        "run_identity_sha256": "r" * 64,
        "data_sha256": "d" * 64,
        "source": {"commit": None},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return model


def context(tmp: Path, data, evaluation, dsec_root: Path) -> mot.Context:
    return mot.Context(
        evaluation_config=evaluation,
        data_config=data,
        dsec_root=dsec_root,
        trackeval_root=TRACKEVAL,
        work_root=tmp / "work",
    )


def perfect_exports(sequence, score=0.9, wrong_class_track=None, drop_frames=()):
    """Exports equal to the ground truth; each embedding is the one-hot track ID."""

    pool = sequence.geometry.pool
    rows, embeddings, offsets = [], [], [0]
    for step in range(sequence.steps):
        frame_rows = []
        if sequence.supervised[step] and step not in drop_frames:
            boxes = sequence.step_boxes(step)
            for xyxy, label, track in zip(boxes["xyxy"], boxes["label"], boxes["track_id"]):
                label = (label + 1) % 7 if track == wrong_class_track else label
                frame_rows.append([*(xyxy * pool), score - 0.01 * track, label])
                vector = np.zeros(128, np.float32)
                vector[track] = 1.0
                embeddings.append(vector)
        rows.extend(frame_rows)
        offsets.append(offsets[-1] + len(frame_rows))
    return {
        "rows": np.asarray(rows, np.float32).reshape(-1, 6),
        "embeddings": np.asarray(embeddings, np.float32).reshape(-1, 128),
        "anchors": np.zeros(len(rows), np.int64),
        "offsets": np.asarray(offsets, np.int64),
        "timestamps": sequence.step_t_us.copy(),
        "cap_reached": np.zeros(sequence.steps, np.bool_),
    }


def params(evaluation, **changes):
    values = dict(evaluation["search"]["base"], score=0.5, buffer=30)
    values.update(changes)
    return mot.canonical_params(values, "reid", evaluation)


# --------------------------------------------------------------------------------- frozen protocol


def test_evaluation_config_freezes_the_protocol(configs):
    _, _, evaluation = configs
    tracker = evaluation["tracker"]
    assert tracker["backend"] == "boxmot_botsort" and tracker["cmc_method"] is None
    assert tracker["isolated_classes"] is True
    assert evaluation["export"]["score_floor"] == 0.05
    grids = {k: v for k, v in evaluation["search"]["grids"].items() if k != "min_area"}
    assert grids == tracker_tuning.GRIDS
    broken = copy.deepcopy(evaluation)
    broken["search"]["grids"]["buffer"] = [2, 3]
    with pytest.raises(mot.EvaluationError, match="small-model"):
        mot.check_search_space(broken)
    mot.check_search_space(evaluation)


def test_canonical_params_validation(configs):
    _, _, evaluation = configs
    base = dict(evaluation["search"]["base"])
    assert mot.canonical_params(base, "reid", evaluation)["min_area"] is False
    with pytest.raises(mot.EvaluationError, match="grid"):
        mot.canonical_params(dict(base, score=0.33), "reid", evaluation)
    with pytest.raises(mot.EvaluationError, match="below"):
        mot.canonical_params(dict(base, low=0.45, high=0.4), "reid", evaluation)
    with pytest.raises(mot.EvaluationError, match="differ"):
        mot.canonical_params(
            {key: value for key, value in base.items() if key != "min_area"}, "reid", evaluation
        )
    motion = mot.canonical_params(dict(base, appearance=0.9, proximity=0.1), "motion", evaluation)
    assert (motion["appearance"], motion["proximity"]) == (0.25, 0.5)
    for score in evaluation["probe"]["score_thresholds"]:
        mot.canonical_params(dict(evaluation["probe"]["params"], score=score), "reid", evaluation)


def test_selected_rows_are_a_score_prefix_with_the_area_filter(configs):
    _, _, evaluation = configs
    rows = np.array(
        [
            [0, 0, 40, 40, 0.9, 0],
            [0, 0, 10, 10, 0.7, 1],
            [0, 0, 30, 30, 0.5, 2],
            [0, 0, 50, 50, 0.1, 0],
        ],
        np.float32,
    )
    base = dict(evaluation["search"]["base"])
    for score in (0.05, 0.1, 0.5, 0.7, 0.9, 0.95):
        selected = mot.select_rows(rows, dict(base, score=score), 614.4)
        assert list(selected) == list(range(len(selected)))
    assert list(mot.select_rows(rows, dict(base, score=0.05, min_area=True), 614.4)) == [0, 2, 3]
    assert list(mot.select_rows(rows, dict(base, score=0.05, min_area=True), 901.0)) == [0, 3]


def test_proposals_start_with_anchors_then_seeded_unique_candidates(configs):
    _, _, evaluation = configs
    grids = evaluation["search"]["grids"]
    probe = dict(evaluation["probe"]["params"], score=0.3)
    anchors = mot.anchors(evaluation, "reid", [probe])
    small = [dict(anchor, min_area=False) for anchor in tracker_tuning.anchors({}, "reid")]
    assert anchors[0] == mot.canonical_params(small[0], "reid", evaluation)
    assert anchors[1]["min_area"] is True
    history, seen = [], set()
    for index in range(150):
        candidate = mot.propose(history, "reid", evaluation, [probe])
        if index < len(anchors):
            assert candidate == anchors[index]
        assert all(candidate[key] in grids[key] for key in grids)
        assert candidate["low"] < candidate["high"]
        key = mot.digest(candidate)
        assert key not in seen
        seen.add(key)
        hota = (index * 7919 % 101) / 101
        history.append({"params": candidate, "metrics": {"HOTA": hota, "AssA": 0.0, "IDF1": 0.0}})
    assert any(item["params"]["min_area"] for item in history[len(anchors) :])
    assert mot.canonical_params(probe, "reid", evaluation) in anchors
    assert mot.propose(history, "reid", evaluation, [probe]) == mot.propose(
        history, "reid", evaluation, [probe]
    )
    first = mot.propose([], "motion", evaluation)
    assert (first["appearance"], first["proximity"]) == (0.25, 0.5)


def test_tie_breaks(configs):
    _, _, evaluation = configs
    metrics = {"HOTA": 0.5, "AssA": 0.4, "IDF1": 0.3}
    a = {"params": params(evaluation, score=0.3), "metrics": dict(metrics)}
    b = {"params": params(evaluation, score=0.4), "metrics": dict(metrics)}
    c = {"params": params(evaluation, score=0.6), "metrics": dict(metrics, AssA=0.41, IDF1=0.0)}
    assert mot.best_trial([a, b, c]) is c
    assert mot.best_trial([a, b]) is min([a, b], key=lambda record: mot.digest(record["params"]))
    points = [{"step": 10, "best": a}, {"step": 20, "best": b}]
    assert [point["step"] for point in mot.rank_points(points)] == [20, 10]


def test_trial_history_survives_a_partial_line(tmp_path):
    path = tmp_path / "reid.jsonl"
    mot.append_history(path, {"a": 1})
    with path.open("a") as handle:
        handle.write('{"a": 2')
    assert mot.read_history(path) == [{"a": 1}]
    mot.append_history(path, {"a": 3})
    assert mot.read_history(path) == [{"a": 1}, {"a": 3}]


# ---------------------------------------------------------------------------- tracking and metrics


@pytest.fixture()
def val_setup(tmp_path, configs):
    data, _, evaluation = configs
    sequences = sequences_for(data, "val", steps=16, objects=4)
    write_dsec_root(tmp_path / "dsec", sequences, "train")
    return context(tmp_path, data, evaluation, tmp_path / "dsec"), sequences[0]


@needs_tracking
def test_perfect_tracking_gives_known_metrics(val_setup, configs):
    ctx, sequence = val_setup
    _, _, evaluation = configs
    exports = {sequence.sequence: perfect_exports(sequence)}
    record = mot.run_trial(exports, params(evaluation), "reid", "val", ctx)
    metrics = record["metrics"]
    assert metrics["HOTA"] > 0.95 and metrics["IDS"] == 0
    assert metrics["IDF1"] == pytest.approx(1.0) and metrics["MOTA"] == pytest.approx(1.0)
    assert metrics["FP"] == 0 and metrics["FN"] == 0
    assert 0.0 <= metrics["DetA"] <= 1.0 and 0.0 <= metrics["AssA"] <= 1.0  # fraction scale
    labels = set(sequence.boxes["label"].tolist())
    for index, name in enumerate(ctx.classes):
        entry = record["by_class"][name]
        assert entry["status"] == ("evaluated" if index in labels else "no_gt")
        if index not in labels:  # a class without GT is n/a, never a zero HOTA
            assert entry["HOTA"] is None and entry["FP"] == 0 and entry["gt_boxes"] == 0
    motion = mot.run_trial(exports, params(evaluation), "motion", "val", ctx)
    assert motion["metrics"]["MOTA"] == pytest.approx(1.0) and motion["metrics"]["HOTA"] > 0.95


@needs_tracking
def test_a_wrong_class_track_gives_the_analytic_error(val_setup, configs):
    ctx, sequence = val_setup
    _, _, evaluation = configs
    gt = len(sequence.boxes["label"])
    exports = {sequence.sequence: perfect_exports(sequence, wrong_class_track=1)}
    record = mot.run_trial(exports, params(evaluation), "reid", "val", ctx)
    track_one = int((sequence.boxes["track_id"] == 1).sum())
    assert record["metrics"]["FN"] == track_one and record["metrics"]["FP"] == track_one
    assert record["metrics"]["MOTA"] == pytest.approx(1 - 2 * track_one / gt)
    wrong = (int(sequence.boxes["label"][sequence.boxes["track_id"] == 1][0]) + 1) % 7
    entry = record["by_class"][ctx.classes[wrong]]
    if entry["gt_boxes"] == 0:
        assert entry["status"] == "no_gt" and entry["FP"] == track_one and entry["HOTA"] is None


@needs_tracking
def test_empty_frames_and_class_isolated_track_ids(val_setup, configs):
    ctx, sequence = val_setup
    _, _, evaluation = configs
    dropped = set(np.flatnonzero(sequence.supervised)[:2].tolist())
    exports = {sequence.sequence: perfect_exports(sequence, drop_frames=dropped)}
    assert np.diff(exports[sequence.sequence]["offsets"])[0] == 0  # frame 0 is empty
    out = ctx.work_root.parent / "out"
    record = mot.run_trial(exports, params(evaluation), "reid", "val", ctx, destination=out)
    lost = sum(int(sequence.box_offsets[s + 1] - sequence.box_offsets[s]) for s in dropped)
    # Tracks born after the first frame stay tentative for one frame, so FN >= lost.
    assert record["metrics"]["FN"] >= lost and record["metrics"]["FP"] == 0
    assert record["metrics"]["IDS"] == 0
    classes_by_track: dict[int, set] = {}
    for line in (out / "tracks" / f"{sequence.sequence}.txt").read_text().splitlines():
        fields = line.split(",")
        classes_by_track.setdefault(int(fields[1]), set()).add(int(fields[7]))
    assert all(len(classes) == 1 for classes in classes_by_track.values())
    for name in (
        "metrics_summary.json",
        "metrics_summary.csv",
        "metrics_by_class.csv",
        "tracker_runs.json",
    ):
        assert (out / name).is_file()
    summary = json.loads((out / "metrics_summary.json").read_text())
    assert summary["scale"] == "fraction" and "raw_trackeval" in summary


@needs_tracking
def test_tracker_receives_embeddings_aligned_with_the_selected_rows(
    val_setup, configs, monkeypatch
):
    ctx, sequence = val_setup
    _, _, evaluation = configs
    exports = perfect_exports(sequence)
    exports["embeddings"][:, 127] = exports["rows"][:, 4]  # tag every row with its score
    seen = []
    original = mot.BoxMotTracker.update

    def update(self, detections, frame, timestamp, embeddings=None):
        if detections:
            assert np.allclose(embeddings[:, 127], [d.score for d in detections])
        seen.append(0 if embeddings is None else embeddings.shape[0])
        return original(self, detections, frame, timestamp, embeddings)

    monkeypatch.setattr(mot.BoxMotTracker, "update", update)
    mot.run_trial({sequence.sequence: exports}, params(evaluation, score=0.85), "reid", "val", ctx)
    assert len(seen) == sequence.steps and min(seen) == 0


def test_test_split_needs_the_frozen_configuration(val_setup, configs):
    ctx, sequence = val_setup
    _, _, evaluation = configs
    exports = {sequence.sequence: perfect_exports(sequence)}
    with pytest.raises(mot.EvaluationError, match="frozen"):
        mot.run_trial(exports, params(evaluation), "reid", "test", ctx)


def test_exports_must_cover_every_image_timestamp(val_setup):
    ctx, sequence = val_setup
    exports = perfect_exports(sequence)
    mot.check_alignment(exports, ctx, "val", sequence.sequence)
    exports["timestamps"] = exports["timestamps"][1:]
    with pytest.raises(mot.EvaluationError, match="image timestamps"):
        mot.check_alignment(exports, ctx, "val", sequence.sequence)


# ----------------------------------------------------------------------------------------- exports


@pytest.fixture(scope="module")
def exported(tmp_path_factory, configs):
    if not RVT_ROOT or not Path(RVT_ROOT).is_dir():
        pytest.skip("set RVT_ROOT to the pinned RVT checkout")
    data, model_config, evaluation = configs
    tmp = tmp_path_factory.mktemp("export")
    checkpoint = tmp / "point.pt"
    save_checkpoint(checkpoint, model_config, step=5)
    loaded = ex.load_detector(checkpoint, "dsec_mot", "cpu", RVT_ROOT)
    sequences = (
        sequences_for(data, "val", steps=10)
        + synthetic_sequences(data, "dsec_mot", "val", 2, 7, 2, 3)[1:]
    )
    meta = ex.export_sequences(loaded, sequences, data, evaluation, tmp / "out", "cpu", log=quiet)
    return data, evaluation, loaded, sequences, tmp, meta


@needs_rvt
def test_export_contents(exported):
    _, _, loaded, sequences, tmp, _ = exported
    for sequence in sequences:
        arrays, metadata = ex.load_export(tmp / "out", sequence.sequence)
        assert arrays["timestamps"].tolist() == sequence.step_t_us.tolist()
        counts = np.diff(arrays["offsets"])
        assert counts.max() <= 300 and counts.sum() > 0
        assert metadata["frames_at_cap"] == int(arrays["cap_reached"].sum())
        for frame in range(sequence.steps):
            rows = arrays["rows"][arrays["offsets"][frame] : arrays["offsets"][frame + 1]]
            assert np.all(np.diff(rows[:, 4]) <= 0) and np.all(rows[:, 4] >= 0.05)
        assert np.allclose(np.linalg.norm(arrays["embeddings"], axis=1), 1.0, atol=1e-4)
        assert arrays["embeddings"].shape == (arrays["rows"].shape[0], 128)
        pool = sequence.geometry.pool
        assert arrays["rows"][:, [0, 2]].max() <= sequence.geometry.frame_width * pool + 1e-3
        assert arrays["rows"][:, [1, 3]].max() <= sequence.geometry.frame_height * pool + 1e-3
        identity = metadata["identity"]
        assert identity["checkpoint_sha256"] == loaded.sha256 and identity["split"] == "val"
        assert identity["decoding"]["score_floor"] == 0.05
        assert identity["rvt_commit"] == loaded.model.metadata["upstream"]["commit"]
        assert metadata["timing"]["frames"] == sequence.steps


@needs_rvt
def test_export_equals_a_manual_stream_and_resets_per_sequence(exported):
    data, evaluation, loaded, sequences, tmp, _ = exported
    model = loaded.model
    sequence = sequences[1]
    dataset = LargeEventClipDataset([sequence], data)
    batch = collate_clips([dataset[next(sequence_clips(sequence, sequence.steps))]])
    event_frame, voxel_grid = network_input(batch)
    arrays, _ = ex.load_export(tmp / "out", sequence.sequence)
    state = None
    with torch.no_grad():
        for step in range(sequence.steps):
            output = model(event_frame[:, step], voxel_grid[:, step], state=state)
            state = output.state
            (found,) = model.decode(output, 0.05, 0.5, 300)
            low, high = arrays["offsets"][step], arrays["offsets"][step + 1]
            rows, embeddings = arrays["rows"][low:high], arrays["embeddings"][low:high]
            boxes = found.boxes.numpy() * sequence.geometry.pool
            assert np.allclose(rows[:, :4], boxes, atol=1e-4)
            assert np.array_equal(rows[:, 5].astype(int), found.labels.numpy())
            expected = output.embeddings[0, found.anchors.numpy()].numpy()
            assert np.allclose(embeddings, expected, atol=1e-5)
    ex.export_sequences(loaded, [sequence], data, evaluation, tmp / "alone", "cpu", log=quiet)
    alone, _ = ex.load_export(tmp / "alone", sequence.sequence)
    for key in ex.ARRAYS:  # no state carried from the preceding sequence
        assert np.array_equal(alone[key], arrays[key]), key


@needs_rvt
def test_export_reuse_tamper_and_test_rejections(exported):
    data, evaluation, loaded, sequences, tmp, meta = exported
    messages = []
    again = ex.export_sequences(
        loaded, sequences[:1], data, evaluation, tmp / "out", "cpu", log=messages.append
    )
    assert "reusing" in messages[0] and again[sequences[0].sequence] == meta[sequences[0].sequence]
    changed = copy.deepcopy(evaluation)
    changed["export"]["score_floor"] = 0.1
    with pytest.raises(ex.ExportError, match="identity differs"):
        ex.export_sequences(loaded, sequences[:1], data, changed, tmp / "out", "cpu", log=quiet)
    archive = tmp / "out" / f"{sequences[0].sequence}.npz"
    backup = archive.read_bytes()
    archive.write_bytes(backup[:-10] + b"0123456789")
    with pytest.raises(ex.ExportError, match="hash mismatch"):
        ex.load_export(tmp / "out", sequences[0].sequence)
    archive.write_bytes(backup)
    with pytest.raises(ex.ExportError, match="frozen final"):
        ex.export_sequences(
            loaded, sequences_for(data, "test"), data, evaluation, tmp / "t", "cpu", log=quiet
        )
    with pytest.raises(SystemExit):
        ex.parse_args(
            [
                "--checkpoint",
                "x",
                "--manifest-root",
                "m",
                "--cache-root",
                "c",
                "--output-dir",
                "o",
                "--split",
                "test",
            ]
        )
    with pytest.raises(ex.ExportError, match="SHA-256"):
        ex.load_detector(loaded.path, "dsec_mot", "cpu", RVT_ROOT, expected_sha256="0" * 64)


@needs_rvt
def test_detector_checkpoint_stage_and_geometry_checks(tmp_path, configs):
    _, model_config, _ = configs
    path = tmp_path / "point.pt"
    save_checkpoint(path, model_config, step=1, force=False)
    payload = torch.load(path, weights_only=True)
    payload["stage"] = "pretrain"
    torch.save(payload, tmp_path / "pretrain.pt")
    with pytest.raises(ex.ExportError, match="fine-tuning"):
        ex.load_detector(tmp_path / "pretrain.pt", "dsec_mot", "cpu", RVT_ROOT)
    payload["stage"], payload["geometry"] = "finetune", "dsec_detection"
    torch.save(payload, tmp_path / "geometry.pt")
    with pytest.raises(ex.ExportError, match="geometry"):
        ex.load_detector(tmp_path / "geometry.pt", "dsec_mot", "cpu", RVT_ROOT)


# ----------------------------------------------------------------------------------- command chain


@needs_rvt
@needs_tracking
def test_probe_search_select_freeze_and_single_frozen_test(tmp_path, configs, monkeypatch):
    data, model_config, evaluation = configs
    evaluation = copy.deepcopy(evaluation)
    evaluation["search"]["trials_per_candidate"] = 4
    evaluation["search"]["diagnostic_motion"]["trials"] = 2
    evaluation["probe"]["retained_candidates"] = 2
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps(evaluation))
    data_path = tmp_path / "data.json"
    data_path.write_text(json.dumps(data))
    val = sequences_for(data, "val", steps=12, objects=3)
    test = sequences_for(data, "test", steps=12, objects=3, seed=5)
    write_dsec_root(tmp_path / "dsec", val, "train")
    write_dsec_root(tmp_path / "dsec", test, "test")
    by_split = {"val": val, "test": test}
    monkeypatch.setattr(
        dataset_module, "load_sequences", lambda m, c, d, ds, splits, **k: by_split[splits[0]]
    )
    run = tmp_path / "run"
    candidates = []
    for point, step in ((1, 4), (2, 8), (3, 12)):
        path = run / "finetune" / f"point_{point}_step_{step}.pt"
        save_checkpoint(path, model_config, step=step, seed=point)
        candidates.append(
            {
                "point": point,
                "step": step,
                "path": str(path.relative_to(run)),
                "sha256": mot.file_sha256(path),
                "dsec_detection_monitor": None,
            }
        )
    requests = {
        "candidates": candidates[:2],
        "finished": False,
        "selection": {
            "dataset": "dsec_mot",
            "split": "val",
            "metric": "seven-class HOTA",
            "sequences": ["synthetic_val_00"],
        },
        "weights": "ema",
        "run_identity_sha256": "r" * 64,
    }
    requests_path = run / "finetune" / "validation_requests.json"
    requests_path.write_text(json.dumps(requests))
    common = [
        "--dsec-root",
        str(tmp_path / "dsec"),
        "--trackeval-root",
        str(TRACKEVAL),
        "--data-config",
        str(data_path),
        "--evaluation-config",
        str(evaluation_path),
    ]
    model_args = ["--manifest-root", "m", "--cache-root", "c", "--rvt-root", RVT_ROOT]
    model_args += ["--device", "cpu"]
    root = tmp_path / "validation"
    probe = ["probe", "--requests", str(requests_path), "--output-root", str(root)]
    probe += common + model_args
    assert mot.main(probe) == 0
    summary = json.loads((root / "probe.json").read_text())
    assert not summary["complete"] and summary["retained"] == []
    assert mot.main(["search", "--output-root", str(root)] + common) == 2  # probe incomplete
    requests["candidates"], requests["finished"] = candidates, True
    requests_path.write_text(json.dumps(requests))
    assert mot.main(probe) == 0
    summary = json.loads((root / "probe.json").read_text())
    assert summary["complete"] and len(summary["retained"]) == 2
    trials = json.loads((root / "probe" / "point_1.json").read_text())["trials"]
    assert [t["params"]["score"] for t in trials] == evaluation["probe"]["score_thresholds"]
    ranking = [point["point"] for point in summary["points"]]
    assert [retained["point"] for retained in summary["retained"]] == ranking[:2]
    changed = copy.deepcopy(evaluation)
    changed["probe"]["retained_candidates"] = 3
    (tmp_path / "changed.json").write_text(json.dumps(changed))
    altered = list(common)
    altered[altered.index(str(evaluation_path))] = str(tmp_path / "changed.json")
    assert mot.main(["search", "--output-root", str(root)] + altered) == 2  # inputs changed
    assert mot.main(["search", "--output-root", str(root), "--max-trials", "2"] + common) == 3
    assert mot.main(["select", "--output-root", str(root)] + common) == 2  # budget not reached
    assert mot.main(["search", "--output-root", str(root), "--workers", "2"] + common) == 0
    for retained in summary["retained"]:
        history = mot.read_history(root / "trials" / f"point_{retained['point']}" / "reid.jsonl")
        assert len(history) == 4 and len({h["candidate_id"] for h in history}) == 4
        assert history[0]["params"] == mot.anchors(evaluation, "reid", [])[0]
    assert mot.main(["select", "--output-root", str(root)] + common) == 0
    selection = json.loads((root / "selection.json").read_text())
    assert selection["winner"]["point"] in [r["point"] for r in summary["retained"]]
    assert (root / "selected" / "metrics_summary.json").is_file()
    assert mot.main(["select", "--output-root", str(root)] + common) == 2  # written once
    assert mot.main(["search", "--output-root", str(root), "--mode", "motion"] + common) == 0
    winner = selection["winner"]["point"]
    motion = mot.read_history(root / "trials" / f"point_{winner}" / "motion.jsonl")
    assert len(motion) == 2
    assert all(not record["tracker_runs"][val[0].sequence]["with_reid"] for record in motion)
    freeze = tmp_path / "freeze.json"
    final = tmp_path / "final"
    freeze_args = ["freeze", "--output-root", str(root), "--freeze", str(freeze)]
    freeze_args += ["--final-output-dir", str(final)] + common
    assert mot.main(freeze_args) == 0
    assert not os.access(freeze, os.W_OK)
    assert mot.main(freeze_args) == 2  # immutable
    frozen = json.loads(freeze.read_text())
    assert frozen["tracker"]["params"] == selection["winner"]["params"]
    assert set(frozen["ground_truth"]["test"]) == {"synthetic_test_00"}
    test_args = ["test", "--freeze", str(freeze)] + common + model_args
    annotations = tmp_path / "dsec" / "annotations" / "test" / "synthetic_test_00.txt"
    original = annotations.read_text()
    annotations.write_text(original + "\n")
    assert mot.main(test_args) == 2 and not final.exists()  # changed test GT rejected
    annotations.write_text(original)
    assert mot.main(test_args) == 0
    for name in (
        "metrics_summary.json",
        "metrics_summary.csv",
        "metrics_by_class.csv",
        "test_run.json",
        "supplementary/L1/metrics_summary.json",
        "supplementary/L2/metrics_summary.json",
    ):
        assert (final / name).is_file(), name
    test_run = json.loads((final / "test_run.json").read_text())
    assert test_run["freeze"]["sha256"] == mot.file_sha256(freeze)
    l1 = json.loads((final / "supplementary/L1/metrics_summary.json").read_text())
    l2 = json.loads((final / "supplementary/L2/metrics_summary.json").read_text())
    assert l1["classes"] == ["car", "pedestrian", "bus", "truck"] and l2["classes"] == ["car"]
    assert l2["collapsed"] is True
    assert mot.main(test_args) == 2  # runs once


@needs_tracking
def test_class_collapsed_literature_counts(tmp_path, configs):
    data, _, evaluation = configs
    sequences = synthetic_sequences(data, "dsec_mot", "test", 1, 12, 6, 2)
    write_dsec_root(tmp_path / "dsec", sequences, "test")
    ctx = context(tmp_path, data, evaluation, tmp_path / "dsec")
    ctx.frozen = True
    sequence = sequences[0]
    record = mot.run_trial(
        {sequence.sequence: perfect_exports(sequence)},
        params(evaluation),
        "reid",
        "test",
        ctx,
        destination=tmp_path / "full",
    )
    assert record["metrics"]["MOTA"] == pytest.approx(1.0)
    literature = ["car", "pedestrian", "bus", "truck"]
    collapsed, _ = mot.trackeval_summary(
        tmp_path / "full" / "tracks",
        "test",
        [sequence.sequence],
        ctx,
        tmp_path / "l2",
        literature,
        collapse=True,
    )
    keep = np.isin(sequence.boxes["label"], [0, 1, 4, 5])
    assert collapsed["by_class"]["car"]["gt_boxes"] == int(keep.sum())
    if keep.any():
        assert collapsed["aggregate"]["MOTA"] == pytest.approx(1.0)
        assert collapsed["aggregate"]["HOTA"] > 0.95


# --------------------------------------------------------------------------------- class isolation


def run_class_switch(isolated: bool):
    """Class 0 seen for 5 frames, gone for 3, then a class-1 box at the same place."""

    from src.evaluation.mot_trackers import BoxMotTracker, DetectionObservation, TrackingConfig

    config = TrackingConfig(
        backend="boxmot_botsort",
        iou_threshold=0.8,
        max_missed_frames=30,
        high_threshold=0.6,
        low_threshold=0.1,
        with_reid=True,
        per_class=True,
        num_classes=7,
        cmc_method=None,
        isolated_classes=isolated,
    )
    tracker = BoxMotTracker("boxmot_botsort", config)
    embedding = np.zeros((1, 128), np.float32)
    embedding[0, 0] = 1.0
    records = []
    for frame in range(12):
        if 5 <= frame < 8:
            detections, embeddings = [], np.empty((0, 128), np.float32)
        else:
            label = 0 if frame < 5 else 1
            detections = [DetectionObservation(frame, frame, label, 0.9, 100, 100, 60, 60)]
            embeddings = embedding
        records.extend(tracker.update(detections, frame, frame, embeddings))
    return records


def classes_by_track(records):
    classes: dict[int, set] = {}
    for record in records:
        classes.setdefault(record.track_id, set()).add(record.class_id)
    return classes


@needs_tracking
def test_isolated_classes_fix_the_shared_lost_track_bug():
    assert any(len(c) > 1 for c in classes_by_track(run_class_switch(isolated=False)).values())
    isolated = classes_by_track(run_class_switch(isolated=True))
    assert all(len(c) == 1 for c in isolated.values()) and len(isolated) == 2


@needs_tracking
def test_isolated_track_ids_are_unique_across_classes():
    from src.evaluation.mot_trackers import BoxMotTracker, DetectionObservation, TrackingConfig

    config = TrackingConfig(
        backend="boxmot_botsort",
        per_class=True,
        with_reid=False,
        cmc_method=None,
        isolated_classes=True,
    )
    tracker = BoxMotTracker("boxmot_botsort", config)
    ids = set()
    for frame in range(4):
        detections = [
            DetectionObservation(frame, frame, label, 0.9, 20 + 80 * label, 50, 40, 40)
            for label in range(7)
        ]
        records = tracker.update(detections, frame, frame)
        assert len({record.track_id for record in records}) == len(records)
        ids |= {(record.track_id, record.class_id) for record in records}
    assert len({track for track, _ in ids}) == len(ids) == 7
    with pytest.raises(ValueError):
        BoxMotTracker(
            "boxmot_botsort", TrackingConfig(backend="boxmot_botsort", isolated_classes=True)
        )
