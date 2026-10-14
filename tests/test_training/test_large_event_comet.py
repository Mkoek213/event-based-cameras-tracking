"""Comet process isolation, resumable identity and trainer logging hooks (no network)."""

from __future__ import annotations

import importlib.util
import json
import os
import random
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training import large_event_comet as comet
from src.training import large_event_detector as trainer

FAKE_SDK = """
import json, os, random, time
from pathlib import Path

def capture(kind, **values):
    with open(os.environ["FAKE_COMET_CAPTURE"], "a") as stream:
        stream.write(json.dumps({"kind": kind, **values}) + "\\n")

def fail(kind):
    if os.environ.get("FAKE_COMET_FAIL") == kind:
        raise RuntimeError("secret-api-key private-hostname")

class ExperimentConfig:
    def __init__(self, **kwargs):
        capture("config", **kwargs)

class Experiment:
    def __init__(self, key):
        self.key = key
    def get_key(self):
        return self.key
    def log_parameters(self, parameters):
        capture("parameters", values=parameters)
    def set_name(self, name):
        capture("name", value=name)
    def log_metrics(self, metrics, step):
        fail("metrics")
        capture("metrics", values=metrics, step=step)
    def end(self):
        time.sleep(float(os.environ.get("FAKE_COMET_END_DELAY", "0")))
        fail("end")
        capture("end")

def start(**kwargs):
    assert os.environ["COMET_DISABLE_AUTO_LOGGING"] == "1"
    random.seed(99)
    if os.environ.get("FAKE_COMET_RNG"):
        import numpy as np, torch
        np.random.seed(99)
        torch.manual_seed(99)
    print("secret-api-key private-hostname")
    time.sleep(float(os.environ.get("FAKE_COMET_START_DELAY", "0")))
    fail("start")
    kwargs.pop("api_key", None)
    kwargs.pop("experiment_config", None)
    capture("start", **kwargs)
    return Experiment(kwargs["experiment_key"])
"""


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("COMET_", "LEMOT_COMET_")):
            monkeypatch.delenv(name)


@pytest.fixture
def fake_sdk(tmp_path, monkeypatch):
    sdk = tmp_path / "sdk"
    sdk.mkdir()
    (sdk / "comet_ml.py").write_text(FAKE_SDK)
    capture = tmp_path / "capture.jsonl"
    monkeypatch.setenv("PYTHONPATH", f"{sdk}{os.pathsep}{Path.cwd()}")
    monkeypatch.setenv("FAKE_COMET_CAPTURE", str(capture))
    monkeypatch.setenv("LEMOT_COMET_MODE", "online")
    monkeypatch.setenv("COMET_API_KEY", "secret-api-key")
    monkeypatch.setattr(comet.importlib.util, "find_spec", lambda name: object())
    return capture


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def logger(tmp_path, messages=None):
    return comet.CometLogger(
        tmp_path / "run",
        {"stage": "pretrain", "variant": "main", "seed": 0},
        log=(messages if messages is not None else []).append,
    )


def test_disabled_mode_never_imports_or_writes(tmp_path, monkeypatch):
    monkeypatch.setattr(comet.importlib.util, "find_spec", lambda name: pytest.fail("SDK lookup"))
    log = logger(tmp_path)
    log.training({})
    log.validation({}, {})
    log.close()
    assert log.process is None and not (tmp_path / "run").exists()


@pytest.mark.parametrize("mode", ["invalid", "online"])
def test_bad_configuration_fails_before_worker(tmp_path, monkeypatch, mode):
    monkeypatch.setenv("LEMOT_COMET_MODE", mode)
    with pytest.raises(comet.CometConfigError):
        logger(tmp_path)
    assert not (tmp_path / "run").exists()


def test_requested_mode_requires_sdk(tmp_path, monkeypatch):
    monkeypatch.setenv("LEMOT_COMET_MODE", "offline")
    monkeypatch.setattr(comet.importlib.util, "find_spec", lambda name: None)
    with pytest.raises(comet.CometConfigError, match="3.57.3"):
        logger(tmp_path)


def test_only_explicit_finite_metrics_and_parameters(fake_sdk, tmp_path, capsys):
    log = comet.CometLogger(
        tmp_path / "run",
        {
            "stage": "pretrain",
            "variant": "main",
            "api_key": "secret-api-key",
            "private_path": "/private/not-uploaded.txt",
        },
    )
    log.training(
        {
            "step": 50,
            "phase": "phase_b",
            "lr": 0.0002,
            "seconds_per_step": 0.8,
            "loss": {"total": 4.2, "iou": float("nan"), "private": 123},
            "counts": {"positive_boxes": 8},
            "state": {"continued": 4},
            "private_path": "/private/not-uploaded.txt",
            "device_peak_mib": float("inf"),
        }
    )
    log.validation(
        {
            "step": 10000,
            "kind": "source_map",
            "metrics": {"map": 0.4, "ap50": 0.6, "per_class_ap": {"car": 0.5, "train": None}},
        },
        {"best": {"step": 10000, "map": 0.4}, "without_improvement": 2},
    )
    log.validation(
        {
            "step": 1000,
            "kind": "validation_point",
            "monitor": {"map": 0.2, "ap50": 0.3},
        },
        {},
    )
    log.close()
    events = records(fake_sdk)
    metrics = [event for event in events if event["kind"] == "metrics"]
    assert [event["step"] for event in metrics] == [50, 10000, 1000]
    assert metrics[0]["values"]["train/loss/total"] == 4.2
    assert metrics[0]["values"]["train/state/continued"] == 4
    assert metrics[1]["values"]["val/source/ap/car"] == 0.5
    assert metrics[1]["values"]["val/source/without_improvement"] == 2
    assert metrics[2]["values"]["val/dsec_detection_monitor/map"] == 0.2
    uploaded = fake_sdk.read_text()
    assert "secret-api-key" not in uploaded and "/private/" not in uploaded
    assert "secret-api-key" not in capsys.readouterr().out
    config = next(event for event in events if event["kind"] == "config")
    assert all(value is False for key, value in config.items() if key.startswith(("log_", "auto_")))
    assert config["parse_args"] is False
    assert not any("loss" in name for name in metrics[1]["values"])


def test_resume_reuses_key_and_repeated_steps(fake_sdk, tmp_path):
    for _ in range(2):
        log = logger(tmp_path)
        log.training({"step": 50, "loss": {"total": 1.0}})
        log.close()
    starts = [event for event in records(fake_sdk) if event["kind"] == "start"]
    assert starts[0]["experiment_key"] == starts[1]["experiment_key"]
    assert all(event["mode"] == "get_or_create" for event in starts)
    saved = json.loads((tmp_path / "run" / "comet.json").read_text())
    assert saved["experiment_key"] == starts[0]["experiment_key"]
    assert set(saved) == {"experiment_key", "project_name", "workspace"}
    assert [event["step"] for event in records(fake_sdk) if event["kind"] == "metrics"] == [50, 50]


def test_different_runs_have_different_keys(fake_sdk, tmp_path):
    for name in ("pretrain/main", "finetune/main", "pretrain/repair"):
        log = comet.CometLogger(tmp_path / name, {}, log=lambda message: None)
        log.close()
    starts = [event for event in records(fake_sdk) if event["kind"] == "start"]
    assert len({event["experiment_key"] for event in starts}) == 3


def test_resume_destination_and_key_guard(fake_sdk, tmp_path, monkeypatch):
    logger(tmp_path).close()
    monkeypatch.setenv("COMET_PROJECT_NAME", "other-project")
    with pytest.raises(comet.CometConfigError, match="differs"):
        logger(tmp_path)
    monkeypatch.delenv("COMET_PROJECT_NAME")
    path = tmp_path / "run" / "comet.json"
    path.write_text(json.dumps({"experiment_key": "bad"}))
    with pytest.raises(comet.CometConfigError, match="invalid experiment"):
        logger(tmp_path)


def test_offline_needs_no_credentials_and_stores_below_run(fake_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("LEMOT_COMET_MODE", "offline")
    monkeypatch.delenv("COMET_API_KEY")
    log = logger(tmp_path)
    log.close()
    events = records(fake_sdk)
    assert next(event for event in events if event["kind"] == "start")["online"] is False
    config = next(event for event in events if event["kind"] == "config")
    assert config["offline_directory"] == str(tmp_path / "run" / "comet-offline")


@pytest.mark.parametrize("kind", ["start", "metrics", "end"])
def test_sdk_failure_is_sanitized_and_nonfatal(fake_sdk, tmp_path, monkeypatch, kind):
    monkeypatch.setenv("FAKE_COMET_FAIL", kind)
    messages = []
    log = logger(tmp_path, messages)
    log.training({"step": 1, "loss": {"total": 2.0}})
    log.close()
    assert log.warned and log.process is None
    assert sum("warning" in message for message in messages) == 1
    assert "secret-api-key" not in " ".join(messages)
    assert "private-hostname" not in " ".join(messages)
    assert (tmp_path / "run" / "comet.json").exists()


@pytest.mark.parametrize("phase", ["START", "END"])
def test_slow_sdk_is_bounded(fake_sdk, tmp_path, monkeypatch, phase):
    monkeypatch.setenv(f"FAKE_COMET_{phase}_DELAY", "5")
    monkeypatch.setattr(comet, f"{phase}_TIMEOUT", 0.15)
    started = time.monotonic()
    log = logger(tmp_path)
    log.close()
    assert time.monotonic() - started < 2.0
    assert log.warned and log.process is None


def test_failed_pipe_does_not_block_or_raise(fake_sdk, tmp_path, monkeypatch):
    log = logger(tmp_path)
    monkeypatch.setattr(comet.os, "write", lambda *args: (_ for _ in ()).throw(BlockingIOError()))
    log.training({"step": 1, "loss": {"total": 1.0}})
    log.training({"step": 2, "loss": {"total": 2.0}})
    log.close()
    assert log.warned and log.process is None


def test_worker_cannot_change_parent_rng_or_signal_handlers(fake_sdk, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COMET_RNG", "1")
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    signal_before = signal.getsignal(signal.SIGUSR1)
    loaded = "comet_ml" in sys.modules
    log = logger(tmp_path)
    log.training({"step": 50, "loss": {"total": 1.0}})
    log.close()
    assert random.getstate() == python_before
    numpy_after = np.random.get_state()
    assert numpy_before[0] == numpy_after[0]
    np.testing.assert_array_equal(numpy_before[1], numpy_after[1])
    assert numpy_before[2:] == numpy_after[2:]
    assert torch.equal(torch_before, torch.get_rng_state())
    assert signal.getsignal(signal.SIGUSR1) == signal_before
    assert ("comet_ml" in sys.modules) == loaded


def test_trainer_validation_logs_existing_local_records(tmp_path, monkeypatch):
    obj = trainer.Trainer.__new__(trainer.Trainer)
    obj.run_dir = obj.stage_dir = tmp_path
    obj.step = 10000
    obj.log = lambda message: None
    obj.settings = {
        "validation": {
            "patience": {"min_delta": 0.002, "evaluations": 6},
            "investigation": {"at_step": 20000, "min_subset_map_at_step": 0.3},
        }
    }
    obj.selection = {
        "evaluations": [],
        "best": None,
        "patience_reference": None,
        "without_improvement": 0,
        "investigation": [],
    }
    metrics = {"map": 0.5, "ap50": 0.7, "subset_map": 0.6, "per_class_ap": {"car": 0.5}}
    obj._evaluate = lambda block: metrics
    obj._inference_payload = lambda values: {}
    obj._write_selection = lambda: None
    monkeypatch.setattr(trainer, "atomic_torch_save", lambda *args: None)
    observed = []

    def observe(record, selection):
        assert records(tmp_path / "validation.jsonl")[-1] == record
        observed.append(record)

    obj.comet = SimpleNamespace(validation=observe)
    obj._source_validation()
    assert observed[0]["metrics"] == metrics
    assert obj.selection["best"]["step"] == 10000
    assert obj.selection["without_improvement"] == 0
    obj.stage = "finetune"
    obj.points = [10000]
    obj.validation_dataset = object()
    obj.settings = {"validation": {"monitor": {}}}
    obj.selection = {"candidates": []}
    monkeypatch.setattr(trainer, "file_sha256", lambda path: "sha256")
    obj._validation_point()
    assert observed[1]["monitor"] == metrics
    assert observed[1]["point"] == 1


def test_trainer_interval_logs_after_local_write(tmp_path):
    obj = trainer.Trainer.__new__(trainer.Trainer)
    obj.run_dir = tmp_path
    obj.step = 50
    obj.device = torch.device("cpu")
    obj.settings = {"phase_a_steps": 20}
    obj.log = lambda message: None
    obj.stop_rules = SimpleNamespace(skips=0)
    obj.bank = SimpleNamespace(counters={"segments": 4})
    obj.bank_counters_logged = {"segments": 0}
    obj._reset_interval = lambda: None
    obj.interval = {
        "steps": 50,
        "updates": 50,
        "validation_seconds": 0,
        "checkpoint_seconds": 0,
        "started": time.perf_counter() - 1,
        "lr": 0.0002,
        "skipped": 0,
        "grad_norm_sum": 50,
        "grad_norm_max": 1,
        "terms": {"total": 100},
        "counts": {},
        "frames": 6000,
        "data_seconds": 0,
    }
    observed = []

    def observe(record):
        assert records(tmp_path / "metrics.jsonl")[-1] == record
        observed.append(record)

    obj.comet = SimpleNamespace(training=observe)
    obj._log_interval()
    assert observed[0]["loss"]["total"] == 2.0
    assert observed[0]["phase"] == "phase_b"


def test_dry_run_does_not_construct_logger(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(trainer, "CometLogger", lambda *args, **kwargs: pytest.fail("logger"))
    monkeypatch.setattr(trainer, "build_inputs", lambda args: object())
    monkeypatch.setattr(trainer, "dry_run_report", lambda inputs, args: {"stage": args.stage})
    args = ["--stage", "pretrain", "--run-dir", str(tmp_path / "run"), "--smoke", "--dry-run"]
    assert trainer.main(args) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "pretrain"
    assert not (tmp_path / "run").exists()


def assert_checkpoint_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_checkpoint_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_checkpoint_equal(a, b)
    else:
        assert left == right


@pytest.mark.skipif(
    not os.environ.get("RVT_ROOT")
    or any(importlib.util.find_spec(name) is None for name in ("einops", "pycocotools")),
    reason="real CPU equivalence needs pinned RVT_ROOT, einops and pycocotools",
)
def test_enabled_logger_preserves_real_training_and_resume(fake_sdk, tmp_path, monkeypatch):
    from src.training import large_event_runtime as runtime

    def run(path, resume=False, limit=None):
        if not resume:
            torch.manual_seed(1234)
        args = trainer.parse_args(
            [
                "--stage",
                "pretrain",
                "--run-dir",
                str(path),
                "--smoke",
                "--rvt-root",
                os.environ["RVT_ROOT"],
            ]
        )
        inputs = trainer.build_inputs(args)
        inputs.model_config["rvt"]["backbone"]["embed_dim"] = 32
        inputs.model_config["fusion"]["dim"] = 32
        inputs.model_config["embedding"]["hidden_dim"] = 32
        inputs.model_config["embedding"]["dim"] = 16
        inputs.model_config["parameter_interval"] = {"min": 0, "max": 90_000_000}
        obj = trainer.Trainer(
            inputs,
            path,
            "cpu",
            resume=resume,
            limits=trainer.JobLimits(max_job_steps=limit),
            log=lambda message: None,
        )
        return obj.run()

    monkeypatch.setenv("LEMOT_COMET_MODE", "disabled")
    assert run(tmp_path / "disabled") == trainer.EXIT_COMPLETE
    assert run(tmp_path / "disabled_resumed", limit=3) == trainer.EXIT_PAUSED
    assert run(tmp_path / "disabled_resumed", resume=True) == trainer.EXIT_COMPLETE
    monkeypatch.setenv("LEMOT_COMET_MODE", "online")
    assert run(tmp_path / "enabled") == trainer.EXIT_COMPLETE
    assert run(tmp_path / "resumed", limit=3) == trainer.EXIT_PAUSED
    assert run(tmp_path / "resumed", resume=True) == trainer.EXIT_COMPLETE
    baseline = runtime.load_checkpoint(tmp_path / "disabled" / "last.pt")
    assert_checkpoint_equal(baseline, runtime.load_checkpoint(tmp_path / "enabled" / "last.pt"))
    assert_checkpoint_equal(
        runtime.load_checkpoint(tmp_path / "disabled_resumed" / "last.pt"),
        runtime.load_checkpoint(tmp_path / "resumed" / "last.pt"),
    )
    resumed = runtime.load_checkpoint(tmp_path / "resumed" / "last.pt")
    for key in ("model", "ema", "optimizer", "bank", "scheduler", "stop_rules", "selection"):
        assert_checkpoint_equal(baseline[key], resumed[key])
    events = records(fake_sdk)
    assert any(event["kind"] == "metrics" for event in events)
    starts = [event["experiment_key"] for event in events if event["kind"] == "start"]
    assert len(starts) == 3 and starts[1] == starts[2]
