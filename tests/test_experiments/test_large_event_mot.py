"""Tests for the bounded large event MOT stage runner, GPU-hour ledger, and Slurm wrapper.

Child commands are mocked except in the pilot test, which drives the real training CLI in
its CPU smoke mode and needs the pinned RVT checkout in ``RVT_ROOT`` plus ``pycocotools``.
``sbatch`` and ``sacct`` are never called.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from src.experiments import large_event_mot as lem

REAL_CHILDREN = lem.Children
CONFIG = json.loads(lem.DEFAULT_CONFIG.read_text())
RVT_ROOT = os.environ.get("RVT_ROOT")
PREP = (
    "convert_gen4",
    "convert_dsec",
    "validate_manifests",
    "cache_gen4",
    "cache_dsec_mot",
    "cache_dsec_detection",
)
PILOT = ("pilot_preflight", "pilot_pretrain", "pilot_finetune")


# ---------------------------------------------------------------------------------------- fixtures


def write_config(tmp_path: Path, config) -> Path:
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps(config, indent=2))
    return path


def init_argv(tmp_path: Path, rvt_root: Path) -> list[str]:
    root = tmp_path / "x"
    return [
        "init",
        "--config",
        str(tmp_path / "experiment.json"),
        "--runs-root",
        str(root / "runs"),
        "--results-root",
        str(root / "results"),
        "--manifest-root",
        str(root / "manifests"),
        "--cache-root",
        str(root / "cache"),
        "--gen4-root",
        str(root / "gen4"),
        "--dsec-root",
        str(root / "dsec"),
        "--dsec-detection-labels-root",
        str(root / "labels"),
        "--dsec-mot-root",
        str(root / "dsec_mot"),
        "--rvt-root",
        str(rvt_root),
        "--rvt-checkpoint",
        str(root / "rvt-b.ckpt"),
        "--venv-root",
        str(root / "venvs"),
        "--gpu-account",
        "grant-gpu-gh200",
        "--cpu-account",
        "grant-cpu",
    ]


def init_experiment(tmp_path: Path, config=None, rvt_root=None) -> lem.Experiment:
    """An initialized experiment whose runtime paths exist as empty placeholders."""

    root = tmp_path / "x"
    for name in ("gen4", "dsec", "labels", "dsec_mot/annotations", "rvt"):
        (root / name).mkdir(parents=True, exist_ok=True)
    (root / "rvt-b.ckpt").write_bytes(b"placeholder")
    for venv in lem.VENVS.values():
        (root / "venvs" / venv / "bin").mkdir(parents=True, exist_ok=True)
        (root / "venvs" / venv / "bin" / "activate").write_text("")
    config_path = write_config(tmp_path, config or CONFIG)
    assert lem.main(init_argv(tmp_path, Path(rvt_root) if rvt_root else root / "rvt")) == 0
    experiment = lem.Experiment.load(root / "runs", config_path)
    experiment.python = sys.executable
    return experiment


@pytest.fixture
def exp(tmp_path):
    return init_experiment(tmp_path)


@pytest.fixture
def sbatch(monkeypatch):
    calls = []

    def fake(argv):
        calls.append(list(argv))
        return str(1000 + len(calls))

    monkeypatch.setattr(lem, "run_sbatch", fake)
    return calls


def complete_stage(exp: lem.Experiment, name: str, attempt: str = "main") -> None:
    """Write plausible outputs for ``name`` and record the attempt as complete."""

    if name == "search":
        trial = exp.validation_root / "trials" / "point_1" / "reid.jsonl"
        trial.parent.mkdir(parents=True, exist_ok=True)
        trial.write_text("{}\n")
    for path in exp.outputs(name, attempt):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            continue
        if path.name == "checks.tsv":
            path.write_text("check\tstatus\tdetail\nx\tPASS\tok\n")
        else:
            payload = {
                "complete": True,
                "finished": True,
                "candidates": [],
                "retained": [{"point": 1}],
                "investigation": [],
            }
            path.write_text(json.dumps(payload))
    assert exp.completion_problems(name, attempt) == [], name
    record = exp.record(name)
    record["active"] = attempt
    record["attempts"][attempt] = {
        "status": "complete",
        "inputs": {},
        "completion": {"outputs": exp.output_hashes(name, attempt)},
    }
    exp.save_record(record)


def complete_all(exp, *names):
    for name in names:
        complete_stage(exp, name)


class FakeChildren:
    def __init__(self, behaviour):
        self.behaviour = behaviour
        self.calls = []

    def __call__(self, log=print):
        outer = self

        class Recorded(REAL_CHILDREN):
            def run(self, argv, env, pausable=False):
                outer.calls.append(list(argv))
                return outer.behaviour(list(argv), env)

        return Recorded(log)


def run_job(monkeypatch, exp, job_id, stage, behaviour, gpu=False):
    monkeypatch.setenv("SLURM_JOB_ID", job_id)
    monkeypatch.delenv("SLURM_JOB_START_TIME", raising=False)
    if gpu:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    else:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    fake = FakeChildren(behaviour)
    monkeypatch.setattr(lem, "Children", fake)
    return exp.job(stage, log=lambda message: None), fake.calls


# ----------------------------------------------------------------------------------- configuration


def test_shipped_config_is_valid_and_sums_to_the_ceiling():
    config = lem.load_experiment_config()
    budget = config["budget"]
    assert sum(budget["phases"].values()) + budget["reserve_gpu_hours"] == 400.0
    assert budget["phases"] == {
        "pilot": 10.0,
        "pretrain": 150.0,
        "finetune": 30.0,
        "evaluation": 10.0,
    }
    assert config["slurm"]["gpu"]["max_wall_hours"] == 47.0
    assert sum(config["stages"][name]["wall_hours"] for name in PILOT) <= budget["phases"]["pilot"]
    assert config["stages"]["pretrain"]["micro_batches"] is None  # frozen after the pilot
    assert config["stages"]["finetune"]["micro_batches"] is None


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda c: c["budget"]["phases"].update(pilot=11.0), "sum to"),
        (lambda c: c["stages"]["pretrain"].update(wall_hours=48.0), "wall time"),
        (lambda c: c["stages"]["pretrain"].update(cpus=73), "per-GPU share"),
        (lambda c: c["stages"]["pilot_pretrain"].update(pause_at_step=1000), "pause_at_step"),
        (lambda c: c["stages"]["test"].update(manual=False), "manual"),
        (lambda c: c["stages"]["probe"].update(phase="evaluation"), "CPU stages"),
        (lambda c: c["stages"]["pretrain"].update(phase=None), "budget phase"),
        (lambda c: c["stages"]["cache_gen4"].update(depends_on=["probe"]), "earlier"),
        (lambda c: c["slurm"]["gpu"].update(max_wall_hours=48.0), "partition limit"),
        (lambda c: c["slurm"]["gpu"].update(gpus=2), "one GPU"),
    ],
)
def test_invalid_configs_are_rejected(mutate, message):
    config = copy.deepcopy(CONFIG)
    mutate(config)
    with pytest.raises(lem.ExperimentError, match=message):
        lem.validate_config(config)


# ------------------------------------------------------------------------------------------ budget


def submit_record(job, phase, planned, gpus=1):
    return {
        "event": "submit",
        "job_id": job,
        "phase": phase,
        "planned_gpu_hours": planned,
        "gpus": gpus,
        "cpus": 32,
    }


def test_budget_arithmetic_for_open_closed_and_external_charges():
    ledger = [
        submit_record("1", "pilot", 2.5),
        {"event": "end", "job_id": "1", "gpu_hours": 1.25, "cpu_hours": 40.0},
        submit_record("2", "pilot", 2.0),
        {"event": "external", "phase": "pilot", "gpu_hours": 0.5},
        {
            "event": "submit",
            "job_id": "3",
            "phase": None,
            "planned_gpu_hours": 0.0,
            "gpus": 0,
            "cpus": 16,
        },
        {"event": "end", "job_id": "3", "gpu_hours": 0.0, "cpu_hours": 16.0},
    ]
    summary = lem.budget_summary(CONFIG, ledger)
    # Closed jobs count elapsed time, open jobs their full requested wall time.
    assert summary["phases"]["pilot"]["used"] == pytest.approx(3.75)
    assert summary["phases"]["pilot"]["open"] == pytest.approx(2.0)
    assert summary["total"]["used"] == pytest.approx(3.75)
    assert summary["cpu_core_hours_closed"] == pytest.approx(56.0)
    assert summary["reserve"]["used"] == 0.0


def test_budget_stops_at_phase_reserve_and_total_ceilings():
    ledger = [submit_record(str(i), "pretrain", 47.0) for i in range(3)]  # 141 of 150
    with pytest.raises(lem.ExperimentError, match="reserve-reason"):
        lem.check_budget(CONFIG, ledger, "pretrain", 47.0, None)
    with pytest.raises(lem.ExperimentError, match="one of"):
        lem.check_budget(CONFIG, ledger, "pretrain", 47.0, "more_search")
    assert lem.check_budget(CONFIG, ledger, "pretrain", 47.0, "job_loss") == pytest.approx(38.0)
    assert lem.check_budget(CONFIG, ledger, "pretrain", 9.0, None) == 0.0
    # 141 + 4 * 47 = 329: 179 of the 200-hour reserve are used.
    full = ledger + [submit_record(str(10 + i), "pretrain", 47.0) for i in range(4)]
    assert lem.budget_summary(CONFIG, full)["reserve"]["used"] == pytest.approx(179.0)
    with pytest.raises(lem.ExperimentError, match="reserve exceeds"):
        lem.check_budget(CONFIG, full, "pretrain", 47.0, "job_loss")
    # Overflow of another phase draws on the same reserve.
    finetune = full + [submit_record("20", "finetune", 30.0)]
    with pytest.raises(lem.ExperimentError, match="reserve exceeds"):
        lem.check_budget(CONFIG, finetune, "finetune", 22.0, "stop_rule_repair")
    assert lem.check_budget(CONFIG, finetune, "finetune", 21.0, "stop_rule_repair") == 21.0
    total = [
        {"event": "external", "phase": phase, "gpu_hours": hours}
        for phase, hours in (("pilot", 10), ("pretrain", 350), ("finetune", 30))
    ]
    with pytest.raises(lem.ExperimentError, match="total ceiling"):
        lem.check_budget(CONFIG, total, "evaluation", 10.5, "job_loss")


def test_slurm_time_format():
    assert lem.slurm_time(47.0) == "1-23:00:00"
    assert lem.slurm_time(0.25) == "0-00:15:00"
    assert lem.requested_hours(2.501) == pytest.approx(2.5)


# ----------------------------------------------------------------------------------- init and plan


def test_init_is_immutable_and_dry_run_writes_nothing(tmp_path, capsys):
    exp = init_experiment(tmp_path)
    manifest = exp.state_dir / lem.MANIFEST
    assert stat.S_IMODE(os.stat(manifest).st_mode) == 0o400
    with pytest.raises(lem.ExperimentError, match="immutable"):
        lem.command_init(lem.parse_args(init_argv(tmp_path, tmp_path / "x" / "rvt")))
    dry = tmp_path / "dry"
    argv = [a.replace(str(tmp_path / "x"), str(dry)) for a in init_argv(tmp_path, tmp_path / "rvt")]
    capsys.readouterr()
    assert lem.main([*argv, "--dry-run"]) == 0
    assert not dry.exists()
    out = capsys.readouterr().out
    plan = json.loads(out[out.index("{") :])
    assert [stage["stage"] for stage in plan["stages"]] == list(lem.STAGES)


def child_parsers():
    from src.data import large_event_cache, large_event_schema
    from src.data.converters import dsec_detection, large_event_pretraining
    from src.evaluation import large_event_export, large_event_mot
    from src.training import large_event_detector

    def via_main_parser(module):
        """Parse with the parser that ``module.main`` builds, stopping before any work."""

        def parse(argv):
            original = argparse.ArgumentParser.parse_args

            class ParserReachedError(Exception):
                pass

            def capture(self, args=None, namespace=None):
                raise ParserReachedError(original(self, args, namespace))

            argparse.ArgumentParser.parse_args = capture
            try:
                module.main(argv)
            except ParserReachedError as parsed:
                return parsed.args[0]
            finally:
                argparse.ArgumentParser.parse_args = original
            raise AssertionError("parser not reached")

        return parse

    return {
        "src.training.large_event_detector": large_event_detector.parse_args,
        "src.evaluation.large_event_export": large_event_export.parse_args,
        "src.evaluation.large_event_mot": large_event_mot.parse_args,
        "src.data.large_event_cache": via_main_parser(large_event_cache),
        "src.data.large_event_schema": via_main_parser(large_event_schema),
        "src.data.converters.large_event_pretraining": via_main_parser(large_event_pretraining),
        "src.data.converters.dsec_detection": via_main_parser(dsec_detection),
    }


def test_complete_plan_and_every_child_command_parses(exp):
    from src.training.large_event_runtime import resolve_stage

    plan = exp.plan()
    assert [stage["stage"] for stage in plan["stages"]] == list(lem.STAGES)
    parsers = child_parsers()
    training = json.loads(lem.CONFIG_FILES["training"].read_text())
    data = json.loads(lem.CONFIG_FILES["data"].read_text())
    for stage in lem.STAGES:
        inputs = {"investigation_note": "x"} if stage == "pilot_finetune" else {}
        for step in exp.steps(stage, "main", inputs, micro=2, point=Path("/p.pt")):
            if step.argv[0] == "bash":
                assert step.argv[1] == str(lem.PREFLIGHT_SCRIPT)
                continue
            assert step.argv[:2] == [sys.executable, "-m"]
            argv = [arg if arg != "<micro_batches>" else "1" for arg in step.argv[3:]]
            args = parsers[step.argv[2]](argv)
            if step.argv[2] == "src.training.large_event_detector":
                resolve_stage(training, data, args.stage, args.set)  # every override is valid
                assert args.device == "cuda"
            if step.argv[2] == "src.evaluation.large_event_mot":
                assert (args.command == "test") == (stage == "test")
                if hasattr(args, "device"):
                    assert args.device == "cpu"
    by_name = {stage["stage"]: stage for stage in plan["stages"]}
    for stage in lem.STAGES:
        command = by_name[stage]["sbatch"]
        assert "--no-requeue" in command and "--signal=B:USR1@600" in command
        gpu = CONFIG["stages"][stage]["kind"] == "gpu"
        assert ("--gres=gpu:1" in command) == gpu and ("grant-gpu-gh200" in command) == gpu
    assert any("micro_batches is not frozen" in b for b in by_name["pretrain"]["blocked"])
    assert by_name["freeze"]["manual"] and by_name["test"]["manual"]
    assert by_name["pilot_pretrain"]["planned_gpu_hours_per_job"] == 2.5
    assert not (exp.state_dir / lem.LEDGER).exists()  # planning writes nothing
    assert not (exp.state_dir / "stages").exists()


def test_frozen_specification_does_not_depend_on_the_submitting_interpreter(exp):
    first = exp.frozen_spec("convert_gen4", "main", {})
    exp.python = "/some/other/python"
    assert exp.frozen_spec("convert_gen4", "main", {}) != first  # tests patch the interpreter
    assert lem.Experiment.load(exp.runs_root, exp.config_path).python == "python"


# ---------------------------------------------------------------------------------- submit and job


def manifest_writer(exp, datasets, complete=True, code=0):
    def behaviour(argv, env):
        for dataset in datasets:
            path = exp.paths["manifest_root"] / dataset / "manifest.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"complete": complete}))
        return lem.ChildResult(code, ["done"])

    return behaviour


def test_submit_job_completion_and_idempotence(exp, sbatch, monkeypatch):
    result = exp.submit("convert_gen4")
    assert result["job_id"] == "1001" and result["planned_gpu_hours"] == 0
    assert sbatch[0][0] == "sbatch" and "--partition=plgrid" in sbatch[0]
    assert sbatch[0][-2:] == [str(exp.runs_root), "convert_gen4"]
    with pytest.raises(lem.ExperimentError, match="open job"):
        exp.submit("convert_gen4")
    with pytest.raises(lem.ExperimentError, match="blocked"):
        exp.submit("validate_manifests")
    code, calls = run_job(
        monkeypatch, exp, "1001", "convert_gen4", manifest_writer(exp, ["gen4_1mpx"])
    )
    assert code == 0 and len(calls) == 1
    state = exp.record("convert_gen4")["attempts"]["main"]
    assert state["status"] == "complete"
    assert list(state["completion"]["outputs"]) == [
        str(exp.paths["manifest_root"] / "gen4_1mpx" / "manifest.json")
    ]
    ends = [record for record in exp.ledger() if record["event"] == "end"]
    assert ends[0]["job_id"] == "1001" and ends[0]["status"] == "complete"
    with pytest.raises(lem.ExperimentError, match="already complete"):
        exp.submit("convert_gen4")
    with pytest.raises(lem.ExperimentError, match="end record"):  # no requeue of a closed job
        run_job(monkeypatch, exp, "1001", "convert_gen4", manifest_writer(exp, ["gen4_1mpx"]))


def test_partial_output_is_not_complete(exp, sbatch, monkeypatch):
    exp.submit("convert_gen4")
    code, _ = run_job(
        monkeypatch, exp, "1001", "convert_gen4", manifest_writer(exp, ["gen4_1mpx"], False)
    )
    assert code == lem.EXIT_FAILED
    state = exp.record("convert_gen4")["attempts"]["main"]
    assert state["status"] == "failed" and "partial" in json.dumps(state["results"])
    assert exp.submit("convert_gen4")["job_id"] == "1002"  # same frozen specification


def test_job_refusals(exp, sbatch, monkeypatch):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(lem.ExperimentError, match="inside a Slurm job"):
        exp.job("convert_gen4")
    with pytest.raises(lem.ExperimentError, match="not submitted"):
        run_job(monkeypatch, exp, "999", "convert_gen4", manifest_writer(exp, ["gen4_1mpx"]))
    exp.submit("convert_gen4")
    with pytest.raises(lem.ExperimentError, match="submitted for convert_gen4"):
        run_job(monkeypatch, exp, "1001", "convert_dsec", manifest_writer(exp, ["gen4_1mpx"]))
    # A GPU stage without a visible GPU is rejected and closed in the ledger.
    exp.submit("pilot_preflight")
    code, calls = run_job(monkeypatch, exp, "1002", "pilot_preflight", lambda a, e: None)
    assert code == lem.EXIT_INPUT and calls == []
    end = [r for r in exp.ledger() if r["event"] == "end" and r["job_id"] == "1002"][0]
    assert end["status"] == "rejected"
    assert exp.record("pilot_preflight")["attempts"]["main"]["status"] == "rejected"


def test_missing_and_stale_dependency_outputs_block_submission(exp, sbatch):
    complete_all(exp, "convert_gen4", "convert_dsec")
    manifest = exp.paths["manifest_root"] / "dsec_mot" / "manifest.json"
    manifest.write_text(json.dumps({"complete": True, "changed": 1}))
    with pytest.raises(lem.ExperimentError, match="changed since completion"):
        exp.submit("validate_manifests")
    manifest.unlink()
    with pytest.raises(lem.ExperimentError, match="is missing"):
        exp.submit("validate_manifests")
    assert sbatch == []


def test_specification_is_frozen_at_first_submission(tmp_path, sbatch, monkeypatch):
    exp = init_experiment(tmp_path)
    exp.submit("convert_gen4")
    run_job(monkeypatch, exp, "1001", "convert_gen4", lambda a, e: lem.ChildResult(1, ["boom"]))
    assert exp.record("convert_gen4")["attempts"]["main"]["status"] == "failed"
    changed = copy.deepcopy(CONFIG)
    changed["stages"]["convert_gen4"]["workers"] = 8
    path = write_config(tmp_path, changed)
    stale = lem.Experiment.load(exp.runs_root, path)
    stale.python = sys.executable
    with pytest.raises(lem.ExperimentError, match=r"stale: \['stage_config', 'steps'\]"):
        stale.submit("convert_gen4")
    blocked = [s for s in stale.plan()["stages"] if s["stage"] == "convert_gen4"][0]["blocked"]
    assert any("stale" in reason for reason in blocked)
    # A configuration change between submission and job start is refused at start.
    write_config(tmp_path, CONFIG)
    exp.submit("convert_gen4")
    write_config(tmp_path, changed)
    late = lem.Experiment.load(exp.runs_root, path)
    late.python = sys.executable
    code, calls = run_job(monkeypatch, late, "1002", "convert_gen4", lambda a, e: None)
    assert code == lem.EXIT_INPUT and calls == []


def frozen_micro(tmp_path, exp, **stages):
    config = copy.deepcopy(CONFIG)
    for name, micro in stages.items():
        config["stages"][name]["micro_batches"] = micro
    loaded = lem.Experiment.load(exp.runs_root, write_config(tmp_path, config))
    loaded.python = sys.executable
    return loaded


def training_child(exp, name, code, finish=False, attempt="main"):
    def behaviour(argv, env):
        run_dir = Path(argv[argv.index("--run-dir") + 1])
        assert run_dir == exp.run_dir(name, attempt)
        hours = float(argv[argv.index("--max-wall-hours") + 1])
        assert 46.5 < hours <= 46.75  # job deadline minus the save margin
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "last.pt").write_text("x")
        if finish:
            stage_dir = run_dir / "pretrain"
            stage_dir.mkdir(parents=True, exist_ok=True)
            selection = {"finished": True, "investigation": []}
            (stage_dir / "selection.json").write_text(json.dumps(selection))
            (stage_dir / "best.pt").write_text("best")
        return lem.ChildResult(code, [])

    return behaviour


def test_training_resume_pause_stop_rule_and_single_repair(tmp_path, sbatch, monkeypatch):
    exp = init_experiment(tmp_path)
    complete_all(exp, *PREP, *PILOT)
    with pytest.raises(lem.ExperimentError, match="micro_batches is not frozen"):
        exp.submit("pretrain")
    exp = frozen_micro(tmp_path, exp, pretrain=2, finetune=1)
    exp.submit("pretrain")
    child = training_child(exp, "pretrain", 3)
    code, calls = run_job(monkeypatch, exp, "1001", "pretrain", child, gpu=True)
    assert code == lem.EXIT_PAUSED and "--resume" not in calls[0]
    assert calls[0][calls[0].index("--micro-batches") + 1] == "2"
    assert exp.record("pretrain")["attempts"]["main"]["status"] == "paused"
    exp.submit("pretrain")  # a paused stage resumes in the next job
    child = training_child(exp, "pretrain", 4)
    code, calls = run_job(monkeypatch, exp, "1002", "pretrain", child, gpu=True)
    assert code == lem.EXIT_STOPPED and calls[0][-1] == "--resume"
    with pytest.raises(lem.ExperimentError, match="stop rule"):
        exp.submit("pretrain")
    with pytest.raises(lem.ExperimentError, match="only allowed together"):
        exp.submit("pretrain", overrides=["precision=fp32"])
    with pytest.raises(lem.ExperimentError, match="only"):
        exp.submit("probe", repair="x")
    result = exp.submit(
        "pretrain", repair="bf16 instability: fp32 fallback", overrides=["precision=fp32"]
    )
    assert result["attempt"] == "repair"
    record = exp.record("pretrain")
    assert record["active"] == "repair"
    assert "--set precision=fp32" in record["attempts"]["repair"]["frozen"]["steps"][0]
    child = training_child(exp, "pretrain", 0, finish=True, attempt="repair")
    code, calls = run_job(monkeypatch, exp, result["job_id"], "pretrain", child, gpu=True)
    assert code == 0 and "--resume" not in calls[0]  # the repair starts its own run
    assert exp.record("pretrain")["attempts"]["repair"]["status"] == "complete"
    with pytest.raises(lem.ExperimentError, match="single repair"):
        exp.submit("pretrain", repair="another")
    finetune = exp.steps("finetune", "main", {})[0]
    checkpoint = finetune.argv[finetune.argv.index("--pretrain-checkpoint") + 1]
    assert checkpoint == str(exp.runs_root / "pretrain" / "repair" / "pretrain" / "best.pt")


def test_finetune_needs_and_freezes_the_investigation_note(tmp_path, sbatch, monkeypatch):
    exp = init_experiment(tmp_path)
    complete_all(exp, *PREP, *PILOT)
    exp = frozen_micro(tmp_path, exp, pretrain=1, finetune=1)
    complete_stage(exp, "pretrain")
    selection = exp.run_dir("pretrain") / "pretrain" / "selection.json"
    os.chmod(selection, 0o600)
    selection.write_text(json.dumps({"finished": True, "investigation": ["low mAP"]}))
    record = exp.record("pretrain")
    record["attempts"]["main"]["completion"]["outputs"] = exp.output_hashes("pretrain", "main")
    exp.save_record(record)
    with pytest.raises(lem.ExperimentError, match="investigation-note"):
        exp.submit("finetune")
    exp.submit("finetune", investigation_note="labels checked; 3-class mAP limited by X")
    exp.reconcile("1001", 0.0, "cancelled before start")
    with pytest.raises(lem.ExperimentError, match="inputs are frozen"):
        exp.submit("finetune", investigation_note="different")
    assert exp.submit("finetune")["attempt"] == "main"
    steps = exp.record("finetune")["attempts"]["main"]["frozen"]["steps"][0]
    assert "labels checked" in steps


def test_reserve_use_needs_a_reason_and_a_note(tmp_path, sbatch):
    exp = init_experiment(tmp_path)
    complete_all(exp, *PREP, *PILOT)
    exp = frozen_micro(tmp_path, exp, pretrain=1, finetune=1)
    exp.charge("pretrain", 140.0, "earlier jobs", None)
    with pytest.raises(lem.ExperimentError, match="reserve-reason"):
        exp.submit("pretrain")
    with pytest.raises(lem.ExperimentError, match="reserve-note"):
        exp.submit("pretrain", reserve_reason="throughput_shortfall")
    assert exp.submit("pretrain", wall_hours=10.0, dry_run=True)["from_reserve"] == 0.0
    assert sbatch == []  # a dry run submits nothing
    result = exp.submit(
        "pretrain", reserve_reason="throughput_shortfall", reserve_note="pilot 30% below plan"
    )
    assert result["from_reserve"] == pytest.approx(37.0)
    assert result["reserve"]["reason"] == "throughput_shortfall"
    status = [s for s in exp.plan()["stages"] if s["stage"] == "pretrain"][0]["status"]
    assert status == "submitted"
    with pytest.raises(lem.ExperimentError, match="positive"):
        exp.charge("pilot", 0.0, "x", None)


def test_freeze_and_test_need_explicit_confirmation(exp, sbatch, monkeypatch):
    complete_all(exp, *PREP, *PILOT, "pretrain", "finetune", "probe", "search", "select")
    with pytest.raises(lem.ExperimentError, match="confirm-selection"):
        exp.submit("freeze")
    with pytest.raises(lem.ExperimentError, match="blocked"):
        exp.submit("test")
    exp.submit("freeze", confirm_selection=lem.file_sha256(exp.validation_root / "selection.json"))

    def freeze_child(argv, env):
        assert argv[3] == "freeze"
        exp.freeze_path.write_text("{}")
        os.chmod(exp.freeze_path, 0o444)
        return lem.ChildResult(0, [])

    assert run_job(monkeypatch, exp, "1001", "freeze", freeze_child)[0] == 0
    with pytest.raises(lem.ExperimentError, match="confirm-freeze"):
        exp.submit("test")
    with pytest.raises(lem.ExperimentError, match="confirm-freeze"):
        exp.submit("test", confirm_freeze="0" * 64)
    result = exp.submit("test", confirm_freeze=lem.file_sha256(exp.freeze_path))
    os.chmod(exp.freeze_path, 0o644)  # a writable freeze manifest is refused at job start
    code, calls = run_job(monkeypatch, exp, result["job_id"], "test", lambda a, e: None)
    assert code == lem.EXIT_INPUT and calls == []
    for stage in lem.STAGES:  # no other stage runs the test command
        for step in exp.steps(stage, "main", {}):
            assert ("test" in step.argv[3:4]) == (stage == "test")


def test_reconcile_with_sacct_and_manually(exp, sbatch, monkeypatch):
    exp.submit("pilot_preflight")
    exp.submit("convert_gen4")
    monkeypatch.setattr(
        lem,
        "query_sacct",
        lambda ids: {
            "1001": {"state": "TIMEOUT", "elapsed_seconds": 900.0},
            "1002": {"state": "RUNNING", "elapsed_seconds": 10.0},
        },
    )
    closed = exp.reconcile()
    assert [record["job_id"] for record in closed] == ["1001"]
    assert closed[0]["gpu_hours"] == pytest.approx(0.25)
    assert exp.record("pilot_preflight")["attempts"]["main"]["status"] == "failed"
    with pytest.raises(lem.ExperimentError, match="elapsed-hours"):
        exp.reconcile("1002")
    exp.reconcile("1002", 0.5, "node lost, sacct unavailable")
    summary = lem.budget_summary(exp.config, exp.ledger())
    assert summary["phases"]["pilot"]["used"] == pytest.approx(0.25)
    assert summary["cpu_core_hours_closed"] == pytest.approx(16 * 0.25 + 32 * 0.5)


def test_sacct_output_parsing(monkeypatch):
    class Result:
        returncode = 0
        stdout = "1001|COMPLETED|3600\n1002|CANCELLED by 123|0\n"
        stderr = ""

    monkeypatch.setattr(lem.subprocess, "run", lambda *args, **kwargs: Result())
    assert lem.query_sacct(["1001", "1002"]) == {
        "1001": {"state": "COMPLETED", "elapsed_seconds": 3600.0},
        "1002": {"state": "CANCELLED", "elapsed_seconds": 0.0},
    }


# ----------------------------------------------------------------------------- signals and wrapper


def test_sigusr1_is_forwarded_to_pausable_children(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        "import signal, sys, time\n"
        "signal.signal(signal.SIGUSR1, lambda *a: sys.exit(3))\n"
        "print('ready', flush=True)\n"
        "time.sleep(30)\n"
    )
    previous = signal.getsignal(signal.SIGUSR1)
    with lem.Children(log=lambda message: None) as children:
        timer = threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGUSR1))
        timer.start()
        started = time.time()
        result = children.run([sys.executable, str(script)], {}, pausable=True)
    assert result.code == 3 and "ready" in result.tail
    assert time.time() - started < 10
    assert children.interrupt == "SIGUSR1"
    assert signal.getsignal(signal.SIGUSR1) == previous
    assert lem.outcome(3, lem.Step("s", [], (0, 3)), children.interrupt)[:2] == (
        "paused",
        lem.EXIT_PAUSED,
    )
    assert lem.outcome(0, lem.Step("s", [], (0,)), "SIGUSR1")[0] == "complete"
    assert lem.outcome(1, lem.Step("s", [], (0, 3)), None)[0] == "failed"
    assert lem.outcome(4, lem.Step("s", [], (0, 3, 4)), None)[0] == "stopped"


def test_sigusr1_is_not_forwarded_to_children_that_cannot_pause(tmp_path):
    script = tmp_path / "child.py"
    script.write_text(
        "import signal, sys, time\n"
        "signal.signal(signal.SIGUSR1, lambda *a: sys.exit(9))\n"
        "print('ready', flush=True)\n"
        "time.sleep(2)\n"
        "print('finished', flush=True)\n"
    )
    messages = []
    with lem.Children(log=messages.append) as children:
        timer = threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGUSR1))
        timer.start()
        result = children.run([sys.executable, str(script)], {})
    assert result.code == 0 and result.tail[-1] == "finished"
    assert children.interrupt == "SIGUSR1"
    assert any("cannot pause" in message for message in messages)
    assert lem.outcome(result.code, lem.Step("s", []), children.interrupt)[0] == "complete"
    with lem.Children(log=messages.append) as children:  # SIGTERM is always forwarded
        timer = threading.Timer(0.8, lambda: os.kill(os.getpid(), signal.SIGTERM))
        timer.start()
        result = children.run([sys.executable, str(script)], {})
    assert result.code == -signal.SIGTERM


def test_preflight_and_training_step_signal_policy(exp):
    (preflight,) = exp.steps("pilot_preflight", "main", {})
    assert not preflight.pausable and preflight.env["PREFLIGHT_TIMEOUT"] == "600"
    assert all(step.pausable for step in exp.steps("pilot_pretrain", "main", {}))
    assert all(not step.pausable for step in exp.steps("search", "main", {}))


def test_sbatch_wrapper(tmp_path):
    script = str(lem.SBATCH_SCRIPT)
    assert subprocess.run(["bash", "-n", script]).returncode == 0
    env = {key: value for key, value in os.environ.items() if not key.startswith("SLURM")}
    result = subprocess.run(["bash", script, "a", "b"], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "outside a Slurm job" in result.stderr
    env["SLURM_JOB_ID"] = "1"
    result = subprocess.run(["bash", script, "a", "b"], env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "LARGE_EVENT_MOT_VENV" in result.stderr
    # With a module stub, the script activates the venv and execs the runner job command.
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    log = tmp_path / "exec.log"
    (venv / "bin" / "activate").write_text(f'python() {{ echo "$@" > {log}; }}\nexport -f python\n')
    stub = tmp_path / "stub"
    stub.mkdir()
    (stub / "python").write_text(f'#!/bin/bash\necho "$@" > {log}\n')
    (stub / "python").chmod(0o755)
    env.update(LARGE_EVENT_MOT_VENV=str(venv), PATH=f"{stub}:{env['PATH']}")
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(
        f'module() {{ echo "module $*" >> {tmp_path / "modules.log"}; }}\n'
        f"export -f module\nexec bash {script} RUNS STAGE\n"
    )
    result = subprocess.run(["bash", str(wrapper)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert log.read_text().split() == [
        "-m",
        "src.experiments.large_event_mot",
        "job",
        "--runs-root",
        "RUNS",
        "--stage",
        "STAGE",
    ]
    assert "module add" in (tmp_path / "modules.log").read_text()


# ---------------------------------------------------------------------- pilot through the real CLI


def smoke_pilot_config():
    config = copy.deepcopy(CONFIG)
    probe = {"micro_batches": [1, 2], "steps": 2, "headroom": 0.1}
    config["stages"]["pilot_pretrain"].update(
        memory_probe={
            **probe,
            "overrides": [
                "total_steps=6",
                "phase_a_steps=1",
                "lr.warmup_steps=1",
                "validation.every=6",
                "checkpoint_every=6",
            ],
        },
        overrides=[
            "total_steps=6",
            "phase_a_steps=2",
            "lr.warmup_steps=2",
            "validation.every=6",
            "checkpoint_every=3",
        ],
        pause_at_step=3,
    )
    config["stages"]["pilot_finetune"].update(
        memory_probe={
            **probe,
            "overrides": [
                "total_steps=6",
                "phase_a_steps=1",
                "lr.warmup_steps=1",
                "validation.points=1",
                "checkpoint_every=6",
            ],
        },
        overrides=["total_steps=6", "validation.points=1", "checkpoint_every=3"],
        pause_at_step=3,
        export=False,
    )
    return config


def narrow_model_config(path: Path) -> Path:
    """A narrower RVT for the CPU pilot; stage, pause, and resume logic is unchanged."""

    config = json.loads(lem.CONFIG_FILES["model"].read_text())
    config["rvt"]["backbone"]["embed_dim"] = 32
    config["fusion"]["dim"] = 32
    config["embedding"]["hidden_dim"] = 32
    config["embedding"]["dim"] = 16
    config["parameter_interval"] = {"min": 0, "max": 90_000_000}
    path.write_text(json.dumps(config))
    return path


@pytest.mark.skipif(
    not RVT_ROOT
    or not Path(RVT_ROOT).is_dir()
    or not all(importlib.util.find_spec(n) for n in ("einops", "omegaconf", "pycocotools")),
    reason="set RVT_ROOT to the pinned RVT checkout (needs einops, omegaconf, pycocotools)",
)
def test_pilot_stages_through_the_real_training_cli(tmp_path, sbatch, monkeypatch):
    """Runner -> training CLI (--smoke, CPU): memory probe, pause, resume, summary."""

    exp = init_experiment(tmp_path, smoke_pilot_config(), rvt_root=RVT_ROOT)
    complete_all(exp, *PREP, "pilot_preflight")
    model_config = narrow_model_config(tmp_path / "model.json")
    original = lem.Experiment.training_argv

    def smoke_argv(self, *args, **kwargs):
        argv = original(self, *args, **kwargs)
        argv[argv.index("--device") + 1] = "cpu"
        return [*argv, "--smoke", "--model-config", str(model_config)]

    monkeypatch.setattr(lem.Experiment, "training_argv", smoke_argv)
    entries = []

    def fake_memory_entry(run_dir, micro, result, headroom):
        assert result.code == lem.EXIT_PAUSED
        metrics = lem.read_jsonl(run_dir / "metrics.jsonl")
        assert metrics and metrics[-1]["step"] == 2  # --max-job-steps of the probe
        entry = {"micro_batches": micro, "fits": micro == 2, "device_peak_mib": 0.0}
        entries.append(entry)
        return entry

    monkeypatch.setattr(lem, "memory_entry", fake_memory_entry)
    paths = [str(lem.REPO_ROOT), *filter(None, [os.environ.get("PYTHONPATH")])]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    monkeypatch.setenv("SLURM_JOB_ID", "1001")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # the GPU-stage guard; children use the CPU
    exp.submit("pilot_pretrain")
    code = exp.job("pilot_pretrain", log=lambda message: None)
    state = exp.record("pilot_pretrain")["attempts"]["main"]
    assert code == 0, state.get("results")
    assert state["measurements"]["micro_batches"] == 2
    assert [entry["micro_batches"] for entry in entries] == [1, 2]
    summary = json.loads(exp.pilot_summary("pilot_pretrain").read_text())
    assert summary["checks"] == {
        "paused_and_resumed": True,
        "finished_at_total": True,
        "metrics_steps_increasing": True,
        "metrics_end_at_total": True,
    }
    assert [job["exit_code"] for job in summary["jobs"]] == [3, 0]
    assert [job["step"] for job in summary["jobs"]] == [3, 6]
    assert summary["micro_batches"] == 2 and summary["nonfinite_free"]
    run = json.loads((exp.run_dir("pilot_pretrain") / "run.json").read_text())
    assert "total_steps=6" in json.dumps(run)
    ledger_end = [record for record in exp.ledger() if record["event"] == "end"][0]
    assert ledger_end["status"] == "complete" and ledger_end["gpu_hours"] > 0

    monkeypatch.setenv("SLURM_JOB_ID", "1002")
    exp.submit("pilot_finetune")
    code = exp.job("pilot_finetune", log=lambda message: None)
    state = exp.record("pilot_finetune")["attempts"]["main"]
    assert code == 0, state.get("results")
    assert state["inputs"]["investigation_note"].startswith("pilot:")
    requests_path = exp.run_dir("pilot_finetune") / "finetune" / "validation_requests.json"
    requests = json.loads(requests_path.read_text())
    assert requests["finished"] and len(requests["candidates"]) == 1
    summary = json.loads(exp.pilot_summary("pilot_finetune").read_text())
    assert summary["checks"]["paused_and_resumed"] and summary["steps"] == 6
    assert "export" not in summary
    with pytest.raises(lem.ExperimentError, match="already complete"):
        exp.submit("pilot_finetune")
    budget = lem.budget_summary(exp.config, exp.ledger())["phases"]["pilot"]
    assert 0 < budget["used"] < 0.1 and budget["open"] == 0.0
