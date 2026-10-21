"""Protect selection independence, cached decoding and resumable execution."""

import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
import torch

from src.experiments import tracker_tuning as tuning
from src.experiments import tracker_tuning_runtime as runtime
from src.models.simple_detector import decode_dense_detections


def make_checkpoint(repo, relative, embedding_dim=0, num_classes=7):
    path = repo / "runs" / relative / "best.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    runtime.atomic_json(
        path.parent / "config.json",
        {
            "model": {"num_classes": num_classes, "embedding_dim": embedding_dim},
            "representation": "event_frame",
            "num_bins": 3,
            "time_window_us": 50000,
            "val_sequences": [runtime.VAL_SEQUENCE],
            "train_sequences": ["interlaken_00_a"],
        },
    )
    return path


def entry():
    return {
        "id": "fixture",
        "num_classes": 7,
        "embedding_dim": 2,
        "scope": "all_classes",
        "previous_selections": [],
    }


def trial(hota, params=None):
    return {
        "params": params or dict(tuning.BASE_PARAMS),
        "metrics": {key: hota if key not in ("IDS", "FP", "FN") else 0 for key in runtime.METRICS},
        "elapsed_s": 0.01,
        "per_sequence": {},
        "mode": "motion",
        "split": "val",
    }


def make_reference_selection(root, e):
    manifest = {"entries": [e]}
    runtime.atomic_json(root / "manifest.json", manifest)
    params = dict(tuning.BASE_PARAMS, high=0.95, low=0.25, buffer=90, appearance=0.75)
    selection = {
        "manifest_sha256": runtime.digest(manifest),
        "selected": {
            e["id"]: {
                mode: {
                    "params": tuning.canonical(params, mode),
                    "metrics": {"HOTA": 123.0},
                    "split": "val",
                }
                for mode in tuning.modes(e)
            }
        },
    }
    path = root / "selection.json"
    runtime.atomic_json(path, selection)
    return path


def test_runtime_identity_ignores_unrelated_sources_but_tracks_imported_dependencies(
    tmp_path, monkeypatch
):
    sources = {
        "src/__init__.py": "",
        "src/experiments/__init__.py": "",
        "src/experiments/tracker_tuning.py": "from . import tracker_tuning_runtime\n",
        "src/experiments/tracker_tuning_runtime.py": (
            "def export():\n    from src.models import simple_detector\n"
        ),
        "src/models/__init__.py": "from .simple_detector import forward\n",
        "src/models/simple_detector.py": "def forward():\n    return 1\n",
        "src/training/large_event_detector.py": "unrelated = 1\n",
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    trackeval = tmp_path / "external/TrackEval/trackeval"
    trackeval.mkdir(parents=True)
    evaluator = trackeval / "evaluate.py"
    evaluator.write_text("value = 1\n")
    boxmot = tmp_path / "installed_boxmot.py"
    boxmot.write_text("value = 1\n")
    monkeypatch.setattr(tuning.importlib.metadata, "version", lambda name: "fixture")
    monkeypatch.setattr(
        tuning.importlib.metadata,
        "distribution",
        lambda name: SimpleNamespace(files=[boxmot.name], locate_file=lambda relative: boxmot),
    )
    before = tuning.runtime_identity(tmp_path, trackeval.parent)
    assert "src/models/simple_detector.py" in before["code_files"]
    assert "src/training/large_event_detector.py" not in before["code_files"]
    (tmp_path / "src/training/large_event_detector.py").write_text("unrelated = 2\n")
    (tmp_path / "src/training/another_large_model.py").write_text("another = 1\n")
    assert tuning.runtime_identity(tmp_path, trackeval.parent) == before
    (tmp_path / "src/models/simple_detector.py").write_text("def forward():\n    return 2\n")
    after = tuning.runtime_identity(tmp_path, trackeval.parent)
    assert after["code_sha256"] != before["code_sha256"]
    assert (
        after["code_files"]["src/models/simple_detector.py"]
        != before["code_files"]["src/models/simple_detector.py"]
    )
    evaluator.write_text("value = 2\n")
    assert (
        tuning.runtime_identity(tmp_path, trackeval.parent)["code_sha256"] != after["code_sha256"]
    )
    previous = tuning.runtime_identity(tmp_path, trackeval.parent)
    boxmot.write_text("value = 2\n")
    assert (
        tuning.runtime_identity(tmp_path, trackeval.parent)["code_sha256"]
        != previous["code_sha256"]
    )


def test_runtime_source_scope_expands_when_a_new_dependency_is_imported(tmp_path):
    sources = {
        "src/experiments/tracker_tuning.py": "",
        "src/experiments/tracker_tuning_runtime.py": "",
        "src/data/optional.py": "value = 1\n",
    }
    for name, content in sources.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    optional = tmp_path / "src/data/optional.py"
    assert optional not in tuning.runtime_source_files(tmp_path)
    (tmp_path / "src/experiments/tracker_tuning_runtime.py").write_text(
        "def export():\n    from src.data.optional import value\n"
    )
    assert optional in tuning.runtime_source_files(tmp_path)


def test_tuning_cli_defaults_and_separate_outputs():
    assert tuning.parse_args([]).tuning_split == "val"
    for split, output in (
        ("val", tuning.DEFAULT_OUTPUT),
        ("train", tuning.DEFAULT_TRAIN_OUTPUT),
        ("test", tuning.DEFAULT_TEST_OUTPUT),
    ):
        assert tuning.parse_args(["--tuning-split", split]).output_root == output


@pytest.mark.parametrize("split", ["val", "train", "test"])
def test_tuning_cli_rejects_other_splits_default_outputs(split):
    outputs = {
        "val": tuning.DEFAULT_OUTPUT,
        "train": tuning.DEFAULT_TRAIN_OUTPUT,
        "test": tuning.DEFAULT_TEST_OUTPUT,
    }
    for other, output in outputs.items():
        if other == split:
            continue
        with pytest.raises(SystemExit):
            tuning.parse_args(
                ["--tuning-split", split, "--output-root", str(tuning.REPO_ROOT / output)]
            )


def test_training_discovery_excludes_reserved_sequences_and_fingerprints_all_inputs(tmp_path):
    names = ["interlaken_00_a", "interlaken_00_b"]
    for sequence in [*names, runtime.VAL_SEQUENCE, *runtime.TEST_SEQUENCES]:
        write_dataset(tmp_path, "train", sequence)
    for sequence in runtime.TEST_SEQUENCES:
        write_dataset(tmp_path, "test", sequence)
    (tmp_path / "train/not_a_sequence").write_text("file")
    assert runtime.split_sequences("train", {"root": str(tmp_path)}) == names
    before = tuning.dataset_identity(tmp_path, "train")
    validation_identity = tuning.dataset_identity(tmp_path)
    assert not any(runtime.VAL_SEQUENCE in path for path in before)
    assert all(
        f"annotations/train/{sequence}.txt" in before
        and f"train/{sequence}/events_left/events.h5" in before
        for sequence in names
    )
    annotation = tmp_path / "annotations/train/interlaken_00_b.txt"
    annotation.write_text(annotation.read_text() + "2000000,2,40,40,20,20,0\n")
    assert tuning.dataset_identity(tmp_path, "train") != before
    assert tuning.dataset_identity(tmp_path) == validation_identity


def test_training_discovery_requires_nonreserved_sequences(tmp_path):
    write_dataset(tmp_path, "train", runtime.VAL_SEQUENCE)
    for sequence in runtime.TEST_SEQUENCES:
        write_dataset(tmp_path, "train", sequence)
    with pytest.raises(ValueError, match="No training sequences"):
        runtime.split_sequences("train", {"root": str(tmp_path)})
    with pytest.raises(ValueError):
        runtime.split_sequences("unknown", {"root": str(tmp_path)})


def test_training_manifest_fingerprints_eros_for_every_training_sequence(tmp_path):
    sequences = ["interlaken_00_a", "interlaken_00_b"]
    for sequence in sequences:
        write_dataset(tmp_path, "train", sequence)
    eros = tmp_path / "eros"
    for split, names in (("train", sequences), ("test", runtime.TEST_SEQUENCES)):
        for sequence in names:
            directory = eros / split / sequence
            directory.mkdir(parents=True)
            (directory / "metadata.json").write_text("{}")
            (directory / "snapshots.npy").write_bytes(b"snapshot fixture")
    args = {
        "root": str(tmp_path),
        "eros_cache_root": str(eros),
        "trackeval_root": "unused",
        "runtime_identity": {},
        "dataset_identity": {},
        "seed": 42,
        "tuning_split": "train",
    }
    manifest = tuning.build_manifest([dict(entry(), representation="eros")], args)
    assert manifest["tuning_sequences"] == sequences
    assert manifest["selection_policy"][0] == "combined training HOTA"
    assert len(manifest["eros_identity"]) == 8
    assert not any(runtime.VAL_SEQUENCE in path for path in manifest["eros_identity"])


def test_training_contribution_report_records_training_selection(tmp_path):
    records = [
        {
            "entry": dict(
                entry(),
                embedding_dim=0,
                aliases=[{"family": "contribution", "label": label}],
            ),
            "test": {"tuned_motion": {"metrics": trial(hota)["metrics"]}},
        }
        for label, hota in (("baseline", 0.2), ("recurrence_only", 0.3))
    ]
    sequences = ["interlaken_00_a", "interlaken_00_b"]
    tuning.write_contribution_comparisons(
        records, tmp_path, {"tuning_split": "train", "tuning_sequences": sequences}
    )
    comparison = runtime.read_json(tmp_path / "contribution_comparisons.json")
    assert comparison["tuning_sequences"] == sequences
    assert all(row["tuning_split"] == "train" for row in comparison["records"])
    assert comparison["records"][1]["delta_HOTA_vs_baseline"] == pytest.approx(0.1)
    assert "tuning_split" in (tmp_path / "contribution_comparisons.csv").read_text()


def test_tuning_cli_rejects_unknown_split():
    with pytest.raises(SystemExit):
        tuning.parse_args(["--tuning-split", "unknown"])


def test_reference_import_matches_protocol_and_imports_only_parameters(tmp_path):
    make_checkpoint(tmp_path, "simple_detector/tiny", embedding_dim=2)
    e = tuning.discover(tmp_path)[0]
    path = make_reference_selection(tmp_path / "reference", e)
    original = path.read_bytes()
    identity = tuning.import_reference_params(
        [e], {"reference_selection": str(path), "output_root": str(tmp_path / "new")}
    )
    assert identity["sha256"] == runtime.file_hash(path)
    for mode in tuning.modes(e):
        assert e["reference_params"][mode] in tuning.anchors(e, mode)
    assert "123.0" not in json.dumps(e)
    assert path.read_bytes() == original


def test_reference_import_rejects_changed_protocol_and_same_output(tmp_path):
    make_checkpoint(tmp_path, "simple_detector/tiny", embedding_dim=2)
    e = tuning.discover(tmp_path)[0]
    path = make_reference_selection(tmp_path / "reference", e)
    args = {"reference_selection": str(path), "output_root": str(path.parent)}
    with pytest.raises(ValueError, match="different --output-root"):
        tuning.import_reference_params([e], args)
    args["output_root"] = str(tmp_path / "new")
    e["num_bins"] = 5
    with pytest.raises(ValueError, match="protocol mismatch"):
        tuning.import_reference_params([e], args)


@pytest.mark.parametrize("tuning_split", ["train", "test"])
def test_nonvalidation_tuning_requires_initial_candidates_and_freezes_best_trial(
    tmp_path, tuning_split
):
    e = entry()
    e["reference_params"] = {
        mode: tuning.canonical(dict(tuning.BASE_PARAMS, buffer=90), mode)
        for mode in tuning.modes(e)
    }
    args = {"output_root": str(tmp_path), "tuning_split": tuning_split}
    sequences = list(runtime.TEST_SEQUENCES) if tuning_split == "test" else ["interlaken_00_a"]
    manifest = {
        "selection_policy": [f"combined {tuning_split} HOTA"],
        "tuning_sequences": sequences,
    }
    for mode in tuning.modes(e):
        path = tmp_path / "trials/fixture" / f"{mode}.jsonl"
        tuning.append_history(path, dict(trial(0.1), mode=mode, split=tuning_split))
    with pytest.raises(RuntimeError, match="Initial candidates are incomplete"):
        tuning.freeze_selection([e], args, manifest)
    for mode in tuning.modes(e):
        path = tmp_path / "trials/fixture" / f"{mode}.jsonl"
        for params in tuning.anchors(e, mode)[1:]:
            hota = 0.8 if params == e["reference_params"][mode] else 0.1
            tuning.append_history(path, dict(trial(hota, params), mode=mode, split=tuning_split))
    selected = tuning.freeze_selection([e], args, manifest)
    assert selected["tuning_split"] == tuning_split
    if tuning_split == "test":
        assert selected["evaluation_is_independent"] is False
    assert selected["tuning_sequences"] == sequences
    for mode in tuning.modes(e):
        best = selected["selected"][e["id"]][mode]
        assert best["params"] == e["reference_params"][mode]
        assert best["metrics"]["HOTA"] == 0.8
    tuning.append_history(
        tmp_path / "trials/fixture/motion.jsonl", dict(trial(0.99), split=tuning_split)
    )
    assert tuning.freeze_selection([e], args, manifest) == selected


@pytest.mark.parametrize("tuning_split", ["train", "test"])
def test_selection_rejects_mixed_tuning_splits(tmp_path, tuning_split):
    tuning.append_history(tmp_path / "trials/fixture/motion.jsonl", trial(0.5))
    with pytest.raises(RuntimeError, match="Mixed tuning splits"):
        tuning.freeze_selection(
            [entry()],
            {"output_root": str(tmp_path), "tuning_split": tuning_split},
            {"selection_policy": [f"combined {tuning_split} HOTA"]},
        )


def test_discovery_keeps_families_and_deduplicates_only_identical_protocols(tmp_path):
    checkpoint = make_checkpoint(tmp_path, "simple_detector_sweep/fixture")
    for name in ("baseline", "gated_only"):
        source = tmp_path / "results/dsec_mot_novelty_contribution_benchmark/contribution" / name
        runtime.atomic_json(
            source / "variant_summary.json",
            {
                "variant": {"name": name, "suite": "contribution"},
                "selected_checkpoint": str(checkpoint),
                "score_threshold": 0.9,
                "primary_test_metrics": {"HOTA": 999},
            },
        )
    entries = tuning.discover(tmp_path)
    assert len(entries) == 2  # legacy whole normalization vs controlled component normalization
    component = next(e for e in entries if e["normalisation"] == "component")
    assert len(component["aliases"]) == 2
    assert "primary_test_metrics" not in json.dumps(entries)
    assert "999" not in json.dumps(entries)


def test_discovery_rejects_checkpoint_missing_from_selection(tmp_path):
    runtime.atomic_json(
        tmp_path / "results/dsec_mot_recurrent_placement_ablation/x/variant_summary.json",
        {"variant": {"name": "x"}, "selected_checkpoint": "runs/missing.pt"},
    )
    with pytest.raises(FileNotFoundError, match="Selected checkpoint missing"):
        tuning.discover(tmp_path)


def test_discovery_rejects_training_on_validation(tmp_path):
    path = make_checkpoint(tmp_path, "simple_detector/fixture")
    config = runtime.read_json(path.parent / "config.json")
    config["train_sequences"].append(runtime.VAL_SEQUENCE)
    runtime.atomic_json(path.parent / "config.json", config)
    with pytest.raises(ValueError, match="overlap"):
        tuning.discover(tmp_path)


@pytest.mark.parametrize("mode", ["motion", "reid"])
def test_search_is_unique_deterministic_and_includes_supervisor_anchors(mode):
    history = []
    for _ in range(150):
        params = tuning.propose(entry(), mode, history, 42)
        assert params == tuning.propose(entry(), mode, history, 42)
        assert params["low"] < params["high"]
        if mode == "motion":
            assert (params["appearance"], params["proximity"]) == (0.25, 0.5)
        history.append(trial(0.1, params))
    assert len({runtime.digest(r["params"]) for r in history}) == 150
    assert any(r["params"]["high"] == 0.95 and r["params"]["buffer"] == 10 for r in history)


def test_best_trial_uses_validation_and_stable_ties():
    first = trial(0.2)
    second = trial(0.3, dict(tuning.BASE_PARAMS, buffer=10))
    first["test_HOTA"] = 1
    second["test_HOTA"] = 0
    assert tuning.best_trial([first, second]) is second
    equal = trial(0.3, dict(tuning.BASE_PARAMS, buffer=30))
    assert tuning.best_trial([equal, second]) == tuning.best_trial([second, equal])


def test_partial_jsonl_append_is_recovered_without_losing_completed_trials(tmp_path):
    path = tmp_path / "trials.jsonl"
    tuning.append_history(path, trial(0.2))
    with path.open("ab") as stream:
        stream.write(b'{"params":')
    assert len(tuning.read_history(path)) == 1
    tuning.append_history(path, trial(0.3))
    assert [r["metrics"]["HOTA"] for r in tuning.read_history(path)] == [0.2, 0.3]


def test_selection_requires_balanced_round_and_is_immutable(tmp_path):
    e = entry()
    manifest = {"selection_policy": ["validation HOTA"]}
    args = {"output_root": str(tmp_path)}
    motion = tmp_path / "trials/fixture/motion.jsonl"
    reid = tmp_path / "trials/fixture/reid.jsonl"
    tuning.append_history(motion, trial(0.2))
    tuning.append_history(motion, trial(0.3))
    tuning.append_history(reid, trial(0.1))
    with pytest.raises(RuntimeError, match="incomplete round"):
        tuning.freeze_selection([e], args, manifest)
    tuning.append_history(reid, trial(0.4))
    first = tuning.freeze_selection([e], args, manifest)
    tuning.append_history(reid, trial(0.9))
    assert tuning.freeze_selection([e], args, manifest) == first
    assert first["selected"]["fixture"]["reid"]["metrics"]["HOTA"] == 0.4


@pytest.mark.parametrize("score", [0.05, 0.25, 0.5, 0.9, 0.95])
@pytest.mark.parametrize("logit_scale", [3, 30])
def test_cache_matches_direct_decoding_in_crowded_multiclass_frames(score, logit_scale):
    generator = torch.Generator().manual_seed(12)
    outputs = {
        "cls_logits": torch.randn(1, 4, 20, 40, generator=generator) * logit_scale,
        "bbox_raw": torch.full((1, 4, 20, 40), -3.0),
    }
    embeddings = torch.randn(1, 2, 20, 40, generator=generator)
    all_rows = decode_dense_detections(
        outputs,
        0,
        0,
        score_threshold=0.05,
        max_detections=2**31 - 1,
        embeddings=embeddings,
        stable_nms=True,
    )
    rows, vectors = runtime.pack_frame(all_rows, 2)
    indices = runtime.select_indices(rows, score)
    direct = decode_dense_detections(
        outputs,
        0,
        0,
        score_threshold=score,
        max_detections=100,
        embeddings=embeddings,
        stable_nms=True,
    )
    expected, expected_vectors = runtime.pack_frame(direct, 2)
    np.testing.assert_array_equal(rows[indices], expected)
    np.testing.assert_array_equal(vectors[indices], expected_vectors)
    assert len(rows) > 100


@pytest.mark.parametrize("tuning_split", ["val", "train"])
def test_test_evaluation_cannot_run_before_freeze(tmp_path, tuning_split):
    with pytest.raises(RuntimeError, match="frozen selection"):
        runtime.evaluate(
            entry(),
            "reid",
            tuning.BASE_PARAMS,
            "test",
            {"output_root": str(tmp_path), "tuning_split": tuning_split},
        )


def write_dataset(root, split, sequence, frames=5):
    directory = root / split / sequence
    (directory / "events_left").mkdir(parents=True, exist_ok=True)
    timestamps = [1_000_000 + i * 50000 for i in range(frames)]
    (directory / f"{sequence}_image_timestamps.txt").write_text(
        "\n".join(map(str, timestamps)) + "\n"
    )
    annotations = root / "annotations" / split / f"{sequence}.txt"
    annotations.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_text("".join(f"{t},1,40,40,20,20,0\n" for t in timestamps[1:]))
    with h5py.File(directory / "events_left/events.h5", "w") as handle:
        group = handle.create_group("events")
        for key, values in {"x": [42], "y": [42], "p": [1], "t": [timestamps[0]]}.items():
            group.create_dataset(key, data=np.asarray(values))
        handle.create_dataset("ms_to_idx", data=np.zeros(1, dtype=np.int64))
    return timestamps


def test_real_botsort_and_trackeval_cpu_smoke_with_initial_empty_frame(tmp_path):
    pytest.importorskip("boxmot")
    dataset = tmp_path / "dataset"
    timestamps = write_dataset(dataset, "train", runtime.VAL_SEQUENCE)
    cache_path = tmp_path / "output/cache/fixture/val" / f"{runtime.VAL_SEQUENCE}.npz"
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        rows=np.asarray([[40, 40, 60, 60, 0.99, 0]] * 4, dtype=np.float32),
        embeddings=np.asarray([[1, 0]] * 4, dtype=np.float32),
        offsets=np.asarray([0, 0, 1, 2, 3, 4]),
        timestamps=np.asarray(timestamps),
    )
    args = {
        "root": str(dataset),
        "output_root": str(tmp_path / "output"),
        "min_free_gb": 0,
        "trackeval_root": str(tuning.REPO_ROOT / "external/TrackEval"),
    }
    result = runtime.evaluate(entry(), "reid", tuning.BASE_PARAMS, "val", args)
    assert result["metrics"]["HOTA"] > 0.5
    assert result["tracker_runs"][runtime.VAL_SEQUENCE]["frames"] == 5
    assert result["tracker_runs"][runtime.VAL_SEQUENCE]["effective_max_time_lost"] == 1
    assert not list((tmp_path / "output/work").iterdir())


@pytest.mark.parametrize("tuning_split", ["train", "test"])
def test_multi_sequence_evaluation_uses_trackeval_aggregation(tmp_path, tuning_split):
    pytest.importorskip("boxmot")
    dataset = tmp_path / "dataset"
    output = tmp_path / "output"
    sequences = (
        list(runtime.TEST_SEQUENCES)
        if tuning_split == "test"
        else ["interlaken_00_a", "interlaken_00_b"]
    )
    for sequence, frames, detections in zip(sequences, (5, 9), (4, 3)):
        timestamps = write_dataset(dataset, tuning_split, sequence, frames=frames)
        cache_path = output / "cache/fixture" / tuning_split / f"{sequence}.npz"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            rows=np.asarray([[40, 40, 60, 60, 0.99, 0]] * detections, dtype=np.float32),
            embeddings=np.asarray([[1, 0]] * detections, dtype=np.float32),
            offsets=np.asarray([0] + [min(max(0, frame), detections) for frame in range(frames)]),
            timestamps=np.asarray(timestamps),
        )
    args = {
        "root": str(dataset),
        "output_root": str(output),
        "min_free_gb": 0,
        "tuning_split": tuning_split,
        "trackeval_root": str(tuning.REPO_ROOT / "external/TrackEval"),
    }
    result = runtime.evaluate(entry(), "reid", tuning.BASE_PARAMS, tuning_split, args)
    assert not (output / "selection.json").exists()
    assert set(result["per_sequence"]) == set(sequences)
    assert [r["frames"] for r in result["tracker_runs"].values()] == [5, 9]
    arithmetic_mean = np.mean([r["metrics"]["HOTA"] for r in result["per_sequence"].values()])
    assert not np.isclose(result["metrics"]["HOTA"], arithmetic_mean)


class TinyRecurrentModel:
    def __init__(self):
        self.config = SimpleNamespace(
            num_classes=7,
            embedding_dim=2,
            feature_stride=8,
            embedding_head_type="dense",
            temporal_recurrence_locations=("neck",),
        )
        self.states = []

    def __call__(self, tensor, embedding_state=None, temporal_state=None):
        self.states.append((embedding_state, temporal_state))
        logits = tensor.new_full((1, 8, 2, 2), -20)
        logits[:, 0] = 20
        logits[:, 0, 0, 0] = -20
        logits[:, 1, 0, 0] = 20
        return {
            "cls_logits": logits,
            "bbox_raw": tensor.new_zeros((1, 4, 2, 2)),
            "embeddings": tensor.new_ones((1, 2, 2, 2)),
            "embedding_state": 7,
            "temporal_state": 11,
        }


@pytest.mark.parametrize("split", ["train", "test"])
def test_export_resets_both_states_between_sequences_and_reuses_cache(tmp_path, monkeypatch, split):
    from src.evaluation import simple_detector_trackeval_cli as exporter

    root = tmp_path / "data"
    sequences = (
        list(runtime.TEST_SEQUENCES) if split == "test" else ["interlaken_00_a", "interlaken_00_b"]
    )
    for sequence in sequences:
        write_dataset(root, split, sequence, frames=2)
    if split == "train":
        write_dataset(root, "train", runtime.VAL_SEQUENCE, frames=2)
    checkpoint = tmp_path / "fixture.pt"
    checkpoint.write_bytes(b"checkpoint")
    e = dict(
        entry(),
        checkpoint=str(checkpoint),
        checkpoint_identity=runtime.file_identity(checkpoint),
        representation="event_frame",
        num_bins=3,
        time_window_us=50000,
        normalisation="whole",
    )
    model = TinyRecurrentModel()
    monkeypatch.setattr(exporter, "load_model", lambda *args: (model, {}))
    args = {
        "root": str(root),
        "output_root": str(tmp_path / "output"),
        "min_free_gb": 0,
        "device": "cpu",
        "threads": 1,
        "runtime_identity": {},
        "eros_cache_root": "unused",
    }
    metadata = runtime.export_cache(e, split, args)
    assert set(metadata) == set(sequences)
    assert model.states == [(None, None), (7, 11), (None, None), (7, 11)]
    runtime.export_cache(e, split, args)
    assert len(model.states) == 4
    cache = tmp_path / "output/cache/fixture" / split / f"{sequences[0]}.npz"
    cache.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Invalid/stale"):
        runtime.export_cache(e, split, args)


def test_search_resume_does_not_repeat_completed_trials(tmp_path, monkeypatch):
    calls = []

    def evaluate(e, mode, params, split, args):
        calls.append(split)
        return dict(trial(0.3, params), mode=mode)

    monkeypatch.setattr(tuning, "evaluate", evaluate)
    args = {"output_root": str(tmp_path), "seed": 42}
    tuning.search_batch(entry(), "motion", 3, args)
    tuning.search_batch(entry(), "motion", 3, args)
    tuning.search_batch(entry(), "motion", 4, args)
    assert calls == ["val"] * 4


def test_final_cases_toggle_reid_without_changing_geometry():
    selected = {
        "motion": trial(0.2),
        "reid": trial(0.4, dict(tuning.BASE_PARAMS, buffer=30, appearance=0.7, proximity=0.8)),
    }
    cases = tuning.final_cases(entry(), selected)
    off = cases["reid_settings_without_embeddings"]["params"]
    on = cases["tuned_reid"]["params"]
    assert all(off[key] == on[key] for key in ("score", "high", "low", "match", "buffer"))
    assert cases["reid_settings_without_embeddings"]["mode"] == "motion"


@pytest.mark.parametrize("tuning_split", ["val", "train", "test"])
def test_complete_runner_freezes_before_test_and_resumes_without_new_trials(
    tmp_path, monkeypatch, tuning_split
):
    from src.evaluation import simple_detector_trackeval_cli as exporter

    checkpoint = make_checkpoint(tmp_path, "simple_detector/tiny", embedding_dim=2)
    e = tuning.discover(tmp_path)[0]
    dataset = tmp_path / "dataset"
    write_dataset(dataset, "train", runtime.VAL_SEQUENCE, frames=3)
    training_sequences = ["interlaken_00_a", *[f"training_{index}" for index in range(8)]]
    if tuning_split == "train":
        for sequence in training_sequences:
            write_dataset(dataset, "train", sequence, frames=3)
    for sequence in runtime.TEST_SEQUENCES:
        write_dataset(dataset, "test", sequence, frames=3)
    output = tmp_path / "output"
    reference = None
    trial_count = 2
    if tuning_split in ("train", "test"):
        reference = make_reference_selection(tmp_path / "reference", e)
        tuning.import_reference_params(
            [e], {"reference_selection": str(reference), "output_root": str(output)}
        )
        trial_count = max(len(tuning.anchors(e, mode)) for mode in tuning.modes(e))
    model = TinyRecurrentModel()
    monkeypatch.setattr(exporter, "load_model", lambda *args: (model, {}))
    monkeypatch.setattr(tuning, "discover", lambda repo: [e])
    monkeypatch.setattr(tuning, "runtime_identity", lambda *args: {"fixture": True})
    actual_export = tuning.export_cache
    order = []

    def checked_export(item, split, args):
        if split == "test" and tuning_split != "test":
            assert (output / "selection.json").exists()
        if tuning_split == "test":
            assert split == "test"
            assert not (output / "selection.json").exists()
        if split == "train":
            assert tuning_split == "train"
            assert not (output / "selection.json").exists()
        order.append(split)
        return actual_export(item, split, args)

    monkeypatch.setattr(tuning, "export_cache", checked_export)
    args = vars(
        tuning.parse_args(
            [
                "--root",
                str(dataset),
                "--output-root",
                str(output),
                "--device",
                "cpu",
                "--workers",
                "1",
                "--max-trials",
                str(trial_count),
                "--tuning-split",
                tuning_split,
                "--round-size",
                "1",
                "--budget-hours",
                "1",
                "--final-reserve-hours",
                "0",
                "--min-free-gb",
                "0",
                "--trackeval-root",
                str(tuning.REPO_ROOT / "external/TrackEval"),
            ]
            + (["--reference-selection", str(reference)] if reference else [])
        )
    )
    assert tuning.run(args) == 0
    expected_order = [tuning_split, "test"] if tuning_split != "test" else ["test"]
    assert order == expected_order
    summary = runtime.read_json(output / "benchmark_summary.json")
    assert summary["complete"]
    assert len(summary["records"][0]["test"]) == 5
    assert (output / "benchmark_summary.csv").exists()
    assert (output / "report.md").exists()
    if tuning_split == "train":
        assert summary["test_selection"] == "training only"
        assert summary["tuning_sequences"] == sorted(training_sequences)
        assert "validation" not in summary["records"][0]
        assert set(summary["records"][0]["training"]) == {"motion", "reid"}
        assert "Train HOTA" in (output / "report.md").read_text()
        assert "Val HOTA" not in (output / "report.md").read_text()
        csv = (output / "benchmark_summary.csv").read_text()
        assert "training_HOTA" in csv
        assert "validation_HOTA" not in csv
        for mode in tuning.modes(e):
            history = tuning.read_history(output / "trials" / e["id"] / f"{mode}.jsonl")
            assert all(r["split"] == "train" for r in history)
            assert any(r["params"] == e["reference_params"][mode] for r in history)
            assert all(set(r["per_sequence"]) == set(training_sequences) for r in history)
        assert not (output / "cache" / e["id"] / "val").exists()
        assert len(list((output / "cache" / e["id"] / "train").glob("*.npz"))) == 9
        assert runtime.read_json(output / "state.json")["estimated_test_export_s"] > 0
        assert runtime.read_json(output / "selection.json")["tuning_sequences"] == sorted(
            training_sequences
        )
    if tuning_split == "test":
        assert summary["test_selection"] == "test diagnostic (same sequences)"
        assert summary["evaluation_is_independent"] is False
        assert "validation" not in summary["records"][0]
        assert set(summary["records"][0]["tuning"]) == {"motion", "reid"}
        assert "Val HOTA" not in (output / "report.md").read_text()
        assert "Tuning HOTA" in (output / "report.md").read_text()
        assert "validation_HOTA" not in (output / "benchmark_summary.csv").read_text()
        for mode in tuning.modes(e):
            history = tuning.read_history(output / "trials" / e["id"] / f"{mode}.jsonl")
            assert all(r["split"] == "test" for r in history)
            assert any(r["params"] == e["reference_params"][mode] for r in history)
            assert all(set(r["per_sequence"]) == set(runtime.TEST_SEQUENCES) for r in history)
    assert runtime.read_json(output / "state.json")["phase"] == "complete"
    assert not list((output / "cache" / e["id"] / "test").glob("*.npz"))
    before = (output / "selection.json").read_bytes()
    assert tuning.run(args) == 0
    assert order == expected_order
    assert (output / "selection.json").read_bytes() == before
    assert all(
        len(tuning.read_history(output / "trials" / e["id"] / f"{mode}.jsonl")) == trial_count
        for mode in tuning.modes(e)
    )
    assert checkpoint.read_bytes() == b"fixture"
    if tuning_split in ("train", "test"):
        args["tuning_split"] = "val"
        with pytest.raises(RuntimeError, match="Manifest/code/protocol changed"):
            tuning.run(args)
