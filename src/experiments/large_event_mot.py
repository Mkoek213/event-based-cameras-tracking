"""Bounded stage runner and Slurm job accounting for the large event MOT experiment.

The experiment is a fixed stage graph (``configs/large_event_mot/experiment.json``):

- preparation on the CPU partition: ``convert_gen4``, ``convert_dsec``,
  ``validate_manifests``, ``cache_gen4``, ``cache_dsec_mot``, ``cache_dsec_detection``;
- the bounded GPU pilot: ``pilot_preflight`` (environment preflight), then
  ``pilot_pretrain`` and ``pilot_finetune``: a micro-batch memory probe (the largest
  micro-batch whose peak device memory leaves the configured headroom), a shortened
  stage run that pauses at ``pause_at_step``, is resumed in the same job, and must
  finish, and for fine-tuning a GPU export of its validation point; each writes a
  measurement summary to ``<results_root>/pilot/<stage>.json``;
- one main training trajectory, ``pretrain`` then ``finetune``, resumable over several
  jobs, with at most one ``--repair`` attempt per stage after a stop rule (or, for
  pretraining, after recorded investigation triggers);
- validation-only selection on the CPU partition: ``probe``, ``search``, ``select``;
- manual stages: ``freeze`` (needs ``--confirm-selection`` with the selection SHA-256)
  and ``test`` (needs ``--confirm-freeze`` with the SHA-256 of the read-only freeze
  manifest). No other stage reads test data and nothing submits a stage implicitly.

State lives in ``<runs_root>/experiment/``: ``manifest.json`` (immutable runtime inputs
written by ``init``), ``ledger.jsonl`` (append-only GPU and CPU accounting),
``stages/<stage>.json`` (attempts, frozen specification, jobs, completion record), and
``logs/`` (Slurm output). A stage's specification (child commands, configuration
hashes, dependency output hashes, attempt inputs) is frozen at its first submission;
later submissions and job starts must reproduce it exactly. A dependency counts only
while it is complete and its recorded outputs still have their recorded SHA-256.

Budget: an open job is charged ``gpus x wall_hours`` and a closed job
``gpus x elapsed hours``; ``charge`` adds GPU time spent outside the runner. A phase may
exceed its ceiling only with ``--reserve-reason`` (one of the configured reasons) and
only while the summed overflow of all phases stays within the reserve; the total
ceiling is never exceeded. CPU core-hours are recorded but not capped.

Commands:

- ``init``: write the manifest (``--dry-run`` prints the plan instead).
- ``plan``: the complete dry-run report; writes nothing.
- ``submit --stage S``: one ``sbatch`` submission after the checks above
  (``--dry-run`` prints the sbatch command without submitting).
- ``job``: the body of ``scripts/helios_large_event_mot.sbatch``; refuses jobs that
  ``submit`` did not record, forwards SIGUSR1 and SIGTERM to the running child, never
  starts another child after a signal, validates completion, and closes the ledger
  record.
- ``status``; ``reconcile`` (close jobs that ended without a record through ``sacct``,
  or with ``--elapsed-hours``); ``charge`` (external GPU time).

Exit codes: 0 done (``job``: stage complete), 1 child failure, 2 input or protocol
error (nothing submitted or run), 3 paused (submit again to continue), 4 stop rule
(``last`` kept; see ``--repair``).

Examples::

    python -m src.experiments.large_event_mot init --runs-root <runs> \\
        --results-root <results> --manifest-root <manifests> --cache-root <cache> \\
        --gen4-root <gen4> --dsec-root <dsec> --dsec-detection-labels-root <labels> \\
        --dsec-mot-root <dsec_mot> --rvt-root <rvt> --rvt-checkpoint <rvt-b.ckpt> \\
        --venv-root <venvs> --gpu-account <grant>-gpu-gh200 --cpu-account <grant>-cpu
    python -m src.experiments.large_event_mot plan --runs-root <runs>
    python -m src.experiments.large_event_mot submit --runs-root <runs> --stage pilot_preflight
    python -m src.experiments.large_event_mot status --runs-root <runs>
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import shlex
import signal
import stat
import subprocess
import sys
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs" / "large_event_mot"
DEFAULT_CONFIG = CONFIG_DIR / "experiment.json"
CONFIG_FILES = {
    name: CONFIG_DIR / f"{name}.json" for name in ("data", "model", "training", "evaluation")
}
SBATCH_SCRIPT = REPO_ROOT / "scripts" / "helios_large_event_mot.sbatch"
PREFLIGHT_SCRIPT = REPO_ROOT / "scripts" / "helios_event_mot_preflight.sh"
VENVS = {"gpu": "large_event_mot_gh200", "cpu": "large_event_mot_cpu"}

STATE_DIR = "experiment"
MANIFEST = "manifest.json"
LEDGER = "ledger.jsonl"
MANIFEST_FORMAT = "large_event_mot_experiment"
EXIT_DONE, EXIT_FAILED, EXIT_INPUT, EXIT_PAUSED, EXIT_STOPPED = 0, 1, 2, 3, 4
STAGES = (
    "convert_gen4",
    "convert_dsec",
    "validate_manifests",
    "cache_gen4",
    "cache_dsec_mot",
    "cache_dsec_detection",
    "pilot_preflight",
    "pilot_pretrain",
    "pilot_finetune",
    "pretrain",
    "finetune",
    "probe",
    "search",
    "select",
    "freeze",
    "test",
)
TRAINING = {
    "pilot_pretrain": "pretrain",
    "pilot_finetune": "finetune",
    "pretrain": "pretrain",
    "finetune": "finetune",
}
REPAIRABLE = ("pretrain", "finetune")
CACHE_DATASETS = {
    "cache_gen4": "gen4_1mpx",
    "cache_dsec_mot": "dsec_mot",
    "cache_dsec_detection": "dsec_detection",
}
MANIFEST_DATASETS = {"convert_gen4": ("gen4_1mpx",), "convert_dsec": ("dsec_mot", "dsec_detection")}
STAGE_CONFIGS = {
    **{name: ("data",) for name in (*MANIFEST_DATASETS, "validate_manifests", *CACHE_DATASETS)},
    "pilot_preflight": (),
    **{name: ("data", "model", "training") for name in TRAINING},
    "pilot_finetune": ("data", "model", "training", "evaluation"),
    **{
        name: ("data", "model", "evaluation")
        for name in ("probe", "search", "select", "freeze", "test")
    },
}
# Runtime paths (``init`` flags); every one is required and stored in the manifest.
PATHS = {
    "runs_root": "runs, pilot runs, and the experiment state",
    "results_root": "pilot summaries, validation selection, freeze manifest, final results",
    "manifest_root": "converted sequence manifests",
    "cache_root": "EF/VG caches",
    "gen4_root": "1Mpx release with <split>/ chunk files",
    "dsec_root": "DSEC events with <split>/<sequence>/ (DSEC-MOT and DSEC-Detection)",
    "dsec_detection_labels_root": "extracted DSEC-Detection labels",
    "dsec_mot_root": "DSEC-MOT root with annotations/ and image timestamps",
    "rvt_root": "RVT checkout at the pinned commit",
    "rvt_checkpoint": "RVT-B 1Mpx checkpoint",
    "venv_root": f"directory holding the {VENVS['gpu']} and {VENVS['cpu']} environments",
}
STAGE_PATHS = {
    "convert_gen4": ("gen4_root",),
    "convert_dsec": ("dsec_root", "dsec_detection_labels_root", "dsec_mot_root"),
    "validate_manifests": (),
    "cache_gen4": ("gen4_root",),
    "cache_dsec_mot": ("dsec_root",),
    "cache_dsec_detection": ("dsec_root",),
    "pilot_preflight": ("rvt_root", "rvt_checkpoint"),
    "pilot_pretrain": ("rvt_root", "rvt_checkpoint"),
    "pilot_finetune": ("rvt_root",),
    "pretrain": ("rvt_root", "rvt_checkpoint"),
    "finetune": ("rvt_root",),
    "probe": ("rvt_root", "dsec_mot_root"),
    "search": ("dsec_mot_root",),
    "select": ("dsec_mot_root",),
    "freeze": ("dsec_mot_root",),
    "test": ("rvt_root", "dsec_mot_root"),
}
TERMINAL_SLURM_STATES = (
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "TIMEOUT",
)
OOM_MARKERS = ("OutOfMemoryError", "CUDA out of memory")
MICRO_PLACEHOLDER = "<memory-probe>"
EPSILON = 1e-9


class ExperimentError(RuntimeError):
    """An input, protocol, dependency, or budget check failed; nothing was submitted."""


# --------------------------------------------------------------------------- utilities


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def source_record() -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "modified": None if status is None else [line[3:] for line in status.splitlines()],
    }


def slurm_time(hours: float) -> str:
    minutes = int(round(hours * 60))
    days, minutes = divmod(minutes, 24 * 60)
    return f"{days}-{minutes // 60:02d}:{minutes % 60:02d}:00"


def requested_hours(hours: float) -> float:
    """Wall hours as Slurm grants them (whole minutes)."""

    return int(round(hours * 60)) / 60.0


def override_value(overrides: Sequence[str], key: str) -> Optional[str]:
    for text in overrides:
        name, _, value = text.partition("=")
        if name == key:
            return value
    return None


# --------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class Stage:
    name: str
    kind: str
    phase: Optional[str]
    depends_on: tuple[str, ...]
    wall_hours: float
    cpus: int
    mem_gb: int
    manual: bool
    settings: Mapping[str, Any]


def load_experiment_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    budget = config["budget"]
    phases = budget["phases"]
    allocated = sum(float(value) for value in phases.values()) + float(budget["reserve_gpu_hours"])
    if abs(allocated - float(budget["total_gpu_hours"])) > EPSILON:
        raise ExperimentError(
            f"phase ceilings and reserve sum to {allocated}, not {budget['total_gpu_hours']}"
        )
    if any(float(value) < 0 for value in phases.values()) or not budget["reserve_reasons"]:
        raise ExperimentError("phase ceilings must be non-negative and reserve reasons given")
    slurm = config["slurm"]
    for kind in ("gpu", "cpu"):
        if float(slurm[kind]["max_wall_hours"]) >= float(slurm[kind]["partition_limit_hours"]):
            raise ExperimentError(f"{kind} max_wall_hours must stay below the partition limit")
    stages = config["stages"]
    if tuple(stages) != STAGES:
        raise ExperimentError(f"stages must be exactly {STAGES} in this order")
    seen: set[str] = set()
    for name, section in stages.items():
        stage = stage_from(config, name)
        for dependency in stage.depends_on:
            if dependency not in seen:
                raise ExperimentError(f"{name}: dependency {dependency} must come earlier")
        seen.add(name)
        if stage.kind == "gpu" and stage.phase not in phases:
            raise ExperimentError(f"{name}: GPU stage needs a budget phase from {sorted(phases)}")
        if stage.kind == "cpu" and stage.phase is not None:
            raise ExperimentError(f"{name}: CPU stages are not charged to a GPU phase")
        check_resources(config, stage, stage.wall_hours)
        if name in TRAINING:
            if "micro_batches" in section and section["micro_batches"] is not None:
                if int(section["micro_batches"]) < 1:
                    raise ExperimentError(f"{name}: micro_batches must be positive")
            if name.startswith("pilot_"):
                pause, total = (
                    section["pause_at_step"],
                    override_value(section["overrides"], "total_steps"),
                )
                if total is None or not 0 < int(pause) < int(total):
                    raise ExperimentError(f"{name}: pause_at_step must lie inside total_steps")
    if not (stages["freeze"].get("manual") and stages["test"].get("manual")):
        raise ExperimentError("freeze and test must be manual stages")


def stage_from(config: Mapping[str, Any], name: str) -> Stage:
    if name not in config["stages"]:
        raise ExperimentError(f"unknown stage {name!r}; stages: {', '.join(STAGES)}")
    section = config["stages"][name]
    if section["kind"] not in ("gpu", "cpu"):
        raise ExperimentError(f"{name}: kind must be gpu or cpu")
    return Stage(
        name=name,
        kind=section["kind"],
        phase=section.get("phase"),
        depends_on=tuple(section["depends_on"]),
        wall_hours=float(section["wall_hours"]),
        cpus=int(section["cpus"]),
        mem_gb=int(section["mem_gb"]),
        manual=bool(section.get("manual", False)),
        settings=section,
    )


def check_resources(config: Mapping[str, Any], stage: Stage, wall_hours: float) -> None:
    slurm = config["slurm"][stage.kind]
    if not 0 < wall_hours <= float(slurm["max_wall_hours"]):
        raise ExperimentError(
            f"{stage.name}: wall time {wall_hours} h outside (0, {slurm['max_wall_hours']}] h"
        )
    if stage.kind == "gpu":
        gpus, share = int(slurm["gpus"]), slurm["per_gpu_share"]
        if gpus != 1:
            raise ExperimentError("the runner plans one GPU per job")
        if stage.cpus > gpus * int(share["cpus"]) or stage.mem_gb > gpus * int(share["mem_gb"]):
            raise ExperimentError(
                f"{stage.name}: {stage.cpus} CPUs / {stage.mem_gb} GB exceed the per-GPU share"
            )


def stage_gpus(config: Mapping[str, Any], stage: Stage) -> int:
    return int(config["slurm"]["gpu"]["gpus"]) if stage.kind == "gpu" else 0


# --------------------------------------------------------------------------- budget


def budget_summary(
    config: Mapping[str, Any], ledger: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """GPU-hours per phase: closed jobs at elapsed time, open jobs at their request."""

    budget = config["budget"]
    ceilings = {name: float(value) for name, value in budget["phases"].items()}
    used = {name: 0.0 for name in ceilings}
    open_hours = {name: 0.0 for name in ceilings}
    ends = {record["job_id"]: record for record in ledger if record["event"] == "end"}
    cpu_hours = 0.0
    for record in ledger:
        if record["event"] == "submit":
            end = ends.get(record["job_id"])
            if end is not None:
                cpu_hours += float(end["cpu_hours"])
            if record["phase"] is None:
                continue
            if end is None:
                used[record["phase"]] += float(record["planned_gpu_hours"])
                open_hours[record["phase"]] += float(record["planned_gpu_hours"])
            else:
                used[record["phase"]] += float(end["gpu_hours"])
        elif record["event"] == "external":
            used[record["phase"]] += float(record["gpu_hours"])
    overflow = {name: max(0.0, used[name] - ceilings[name]) for name in ceilings}
    total = sum(used.values())
    return {
        "unit": budget["unit"],
        "phases": {
            name: {
                "ceiling": ceilings[name],
                "used": used[name],
                "open": open_hours[name],
                "remaining": max(0.0, ceilings[name] - used[name]),
                "from_reserve": overflow[name],
            }
            for name in ceilings
        },
        "reserve": {
            "ceiling": float(budget["reserve_gpu_hours"]),
            "used": sum(overflow.values()),
            "remaining": float(budget["reserve_gpu_hours"]) - sum(overflow.values()),
        },
        "total": {
            "ceiling": float(budget["total_gpu_hours"]),
            "used": total,
            "remaining": float(budget["total_gpu_hours"]) - total,
        },
        "cpu_core_hours_closed": cpu_hours,
    }


def check_budget(
    config: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    phase: str,
    planned: float,
    reserve_reason: Optional[str],
) -> float:
    """Reject a planned charge that does not fit; return the part taken from the reserve."""

    summary = budget_summary(config, ledger)
    current = summary["phases"][phase]
    ceiling, before = current["ceiling"], current["used"]
    after = before + planned
    if summary["total"]["used"] + planned > summary["total"]["ceiling"] + EPSILON:
        raise ExperimentError(
            f"{planned:.2f} GPU-h would exceed the total ceiling "
            f"({summary['total']['used']:.2f} of {summary['total']['ceiling']:.2f} used)"
        )
    from_reserve = max(0.0, after - ceiling) - max(0.0, before - ceiling)
    if from_reserve > EPSILON:
        reasons = config["budget"]["reserve_reasons"]
        if reserve_reason is None:
            raise ExperimentError(
                f"{planned:.2f} GPU-h exceeds the {phase} ceiling ({before:.2f} of {ceiling:.2f} "
                f"used); the reserve needs --reserve-reason ({', '.join(reasons)}) and a note"
            )
        if reserve_reason not in reasons:
            raise ExperimentError(f"reserve reason must be one of {reasons}")
        if summary["reserve"]["used"] + from_reserve > summary["reserve"]["ceiling"] + EPSILON:
            raise ExperimentError(
                f"{from_reserve:.2f} GPU-h from the reserve exceeds its remaining "
                f"{summary['reserve']['remaining']:.2f} GPU-h"
            )
    return max(0.0, from_reserve)


# --------------------------------------------------------------------------- child processes


@dataclass
class Step:
    """One child command of a stage."""

    name: str
    argv: list[str]
    accept: tuple[int, ...] = (EXIT_DONE,)
    env: dict[str, str] = field(default_factory=dict)
    resume_dir: Optional[Path] = None  # training: ``--resume`` once the run has a checkpoint
    wall_flag: Optional[str] = None  # ``--max-wall-hours`` filled from the job deadline
    pausable: bool = False  # saves and exits 3 on SIGUSR1 (the training CLI)

    def render(self) -> str:
        text = shlex.join(
            [*(f"{key}={value}" for key, value in sorted(self.env.items())), *self.argv]
        )
        if self.wall_flag is not None:
            text += f" {self.wall_flag} <hours to the job deadline minus the pause margin>"
        if self.resume_dir is not None:
            text += " [--resume when last.pt or last_prev.pt exists]"
        return text


@dataclass
class ChildResult:
    code: int
    tail: list[str]


class Children:
    """Runs one child at a time and forwards job signals to it.

    SIGTERM is always forwarded. SIGUSR1 (the early warning before the time limit) is
    forwarded only to pausable children; other children would be killed by it, so they
    keep running until they finish or Slurm ends the job. Either signal stops the job
    from starting another child.
    """

    def __init__(self, log=print):
        self.log = log
        self.process: Optional[subprocess.Popen] = None
        self.pausable = False
        self.interrupt: Optional[str] = None
        self.previous: list[tuple[int, Any]] = []

    def __enter__(self) -> "Children":
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            with contextlib.suppress(ValueError):  # not the main thread
                self.previous.append((signum, signal.signal(signum, self._handler)))
        return self

    def __exit__(self, *exc) -> None:
        for signum, handler in self.previous:
            signal.signal(signum, handler)

    def _handler(self, signum, _frame) -> None:
        self.interrupt = signal.Signals(signum).name
        if self.process is None or self.process.poll() is not None:
            self.log(f"received {self.interrupt}; no child is running")
        elif signum == signal.SIGUSR1 and not self.pausable:
            self.log(f"received {self.interrupt}; the running child cannot pause, not forwarded")
        else:
            self.log(f"received {self.interrupt}; forwarding to the running child")
            self.process.send_signal(signum)

    def run(
        self, argv: Sequence[str], env: Mapping[str, str], pausable: bool = False
    ) -> ChildResult:
        self.log(f"$ {shlex.join(argv)}")
        self.pausable = pausable
        tail: deque[str] = deque(maxlen=200)
        environment = {**os.environ, "PYTHONUNBUFFERED": "1", **env}
        self.process = subprocess.Popen(
            list(argv),
            cwd=REPO_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert self.process.stdout is not None
        for line in self.process.stdout:
            sys.stdout.write(line)
            tail.append(line.rstrip("\n"))
        code = self.process.wait()
        self.process = None
        sys.stdout.flush()
        self.log(f"exit {code}")
        return ChildResult(code, list(tail))


def run_sbatch(argv: Sequence[str]) -> str:
    """Submit and return the job ID (``sbatch --parsable``)."""

    result = subprocess.run(list(argv), capture_output=True, text=True)
    if result.returncode != 0:
        raise ExperimentError(f"sbatch failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip().split(";")[0]


def query_sacct(job_ids: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Terminal state and elapsed seconds per job ID from ``sacct``."""

    result = subprocess.run(
        ["sacct", "-X", "-n", "-P", "-j", ",".join(job_ids), "-o", "JobID,State,ElapsedRaw"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExperimentError(f"sacct failed: {result.stderr.strip()}")
    records = {}
    for line in result.stdout.splitlines():
        job_id, state, elapsed = line.split("|")
        records[job_id] = {"state": state.split()[0], "elapsed_seconds": float(elapsed or 0)}
    return records


# --------------------------------------------------------------------------- experiment


class Experiment:
    """The stage graph of one initialized (or planned) experiment."""

    def __init__(self, manifest: Mapping[str, Any], config: Mapping[str, Any], config_path: Path):
        self.manifest = manifest
        self.config = config
        self.config_path = config_path
        self.paths = {name: Path(value) for name, value in manifest["paths"].items()}
        self.runs_root = self.paths["runs_root"]
        self.results_root = self.paths["results_root"]
        self.state_dir = self.runs_root / STATE_DIR
        self.python = "python"  # resolved in the activated environment of the job

    # ---------------------------------------------------------------- state

    @classmethod
    def load(cls, runs_root: str | Path, config_path: str | Path = DEFAULT_CONFIG) -> "Experiment":
        path = Path(runs_root) / STATE_DIR / MANIFEST
        if not path.is_file():
            raise ExperimentError(f"{path} missing; run init first")
        manifest = read_json(path)
        if manifest.get("format") != MANIFEST_FORMAT:
            raise ExperimentError(f"{path} is not an experiment manifest")
        if Path(manifest["repo"]) != REPO_ROOT:
            raise ExperimentError(f"{path} belongs to the checkout {manifest['repo']}")
        return cls(manifest, load_experiment_config(config_path), Path(config_path))

    @contextlib.contextmanager
    def locked(self) -> Iterator[None]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with (self.state_dir / "lock").open("a") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def ledger(self) -> list[dict[str, Any]]:
        return read_jsonl(self.state_dir / LEDGER)

    def record(self, name: str) -> dict[str, Any]:
        path = self.state_dir / "stages" / f"{name}.json"
        if path.exists():
            return read_json(path)
        return {"stage": name, "active": "main", "attempts": {}}

    def save_record(self, record: Mapping[str, Any]) -> None:
        write_json_atomic(self.state_dir / "stages" / f"{record['stage']}.json", record)

    def attempt(self, name: str) -> tuple[str, dict[str, Any]]:
        record = self.record(name)
        active = record["active"]
        return active, record["attempts"].get(active, {"status": "pending"})

    def open_jobs(self, ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        closed = {record["job_id"] for record in ledger if record["event"] == "end"}
        return [r for r in ledger if r["event"] == "submit" and r["job_id"] not in closed]

    # ---------------------------------------------------------------- locations

    def run_dir(self, name: str, attempt: str = "main") -> Path:
        if name == "pilot_preflight":
            return self.runs_root / "pilot" / "preflight"
        if name in ("pilot_pretrain", "pilot_finetune"):
            return self.runs_root / "pilot" / TRAINING[name]
        if name in REPAIRABLE:
            return self.runs_root / name / attempt
        raise ExperimentError(f"{name} has no run directory")

    @property
    def validation_root(self) -> Path:
        return self.results_root / "validation"

    @property
    def freeze_path(self) -> Path:
        return self.results_root / "freeze.json"

    @property
    def final_dir(self) -> Path:
        return self.results_root / "final"

    def pilot_summary(self, name: str) -> Path:
        return self.results_root / "pilot" / f"{name}.json"

    def pretrain_checkpoint(self, name: str) -> Path:
        if name == "pilot_finetune":
            return self.run_dir("pilot_pretrain") / "pretrain" / "best.pt"
        attempt, _ = self.attempt("pretrain")
        return self.run_dir("pretrain", attempt) / "pretrain" / "best.pt"

    def requests_path(self) -> Path:
        attempt, _ = self.attempt("finetune")
        return self.run_dir("finetune", attempt) / "finetune" / "validation_requests.json"

    # ---------------------------------------------------------------- commands

    def module(self, name: str, *args: str) -> list[str]:
        return [self.python, "-m", name, *args]

    def config_args(self) -> list[str]:
        return ["--config", str(CONFIG_FILES["data"])]

    def evaluation_args(self) -> list[str]:
        args = ["--dsec-root", str(self.paths["dsec_mot_root"])]
        if self.manifest["paths"].get("trackeval_root"):
            args += ["--trackeval-root", self.manifest["paths"]["trackeval_root"]]
        return args

    def model_args(self, device: str) -> list[str]:
        return [
            "--manifest-root",
            str(self.paths["manifest_root"]),
            "--cache-root",
            str(self.paths["cache_root"]),
            "--rvt-root",
            str(self.paths["rvt_root"]),
            "--device",
            device,
        ]

    def training_argv(
        self,
        name: str,
        run_dir: Path,
        micro_batches: Any,
        overrides: Sequence[str],
        investigation_note: Optional[str],
    ) -> list[str]:
        stage, settings = TRAINING[name], self.config["stages"][name]
        argv = self.module(
            "src.training.large_event_detector",
            "--stage",
            stage,
            "--run-dir",
            str(run_dir),
            "--manifest-root",
            str(self.paths["manifest_root"]),
            "--cache-root",
            str(self.paths["cache_root"]),
            "--rvt-root",
            str(self.paths["rvt_root"]),
            "--seed",
            str(self.manifest["seed"]),
            "--device",
            "cuda",
            "--micro-batches",
            str(micro_batches),
            "--workers",
            str(settings["workers"]),
        )
        if stage == "pretrain":
            argv += ["--rvt-checkpoint", str(self.paths["rvt_checkpoint"])]
        else:
            argv += ["--pretrain-checkpoint", str(self.pretrain_checkpoint(name))]
            if investigation_note:
                argv += ["--investigation-note", investigation_note]
        for override in overrides:
            argv += ["--set", override]
        return argv

    def memory_probe_steps(self, name: str, inputs: Mapping[str, Any]) -> list[Step]:
        probe = self.config["stages"][name]["memory_probe"]
        steps = []
        for micro in probe["micro_batches"]:
            run_dir = self.runs_root / "pilot" / "memory" / f"{TRAINING[name]}_mb{micro}"
            argv = self.training_argv(
                name, run_dir, micro, probe["overrides"], inputs.get("investigation_note")
            )
            argv += ["--max-job-steps", str(probe["steps"])]
            steps.append(Step(f"memory_probe_mb{micro}", argv, (EXIT_PAUSED,), pausable=True))
        return steps

    def steps(
        self,
        name: str,
        attempt: str,
        inputs: Mapping[str, Any],
        micro: Any = None,
        point: Optional[Path] = None,
    ) -> list[Step]:
        """Child commands of a stage attempt.

        For the pilots, ``micro`` is the memory-probe result and ``point`` the validation
        point to export; both are placeholders until the job knows them.
        """

        settings = self.config["stages"][name]
        manifest_root, cache_root = str(self.paths["manifest_root"]), str(self.paths["cache_root"])
        if name == "convert_gen4":
            argv = self.module(
                "src.data.converters.large_event_pretraining",
                "--gen4-root",
                str(self.paths["gen4_root"]),
                "--output-root",
                manifest_root,
                *self.config_args(),
                "--workers",
                str(settings["workers"]),
            )
            return [Step(name, argv)]
        if name == "convert_dsec":
            argv = self.module(
                "src.data.converters.dsec_detection",
                "--dsec-root",
                str(self.paths["dsec_root"]),
                "--detection-labels-root",
                str(self.paths["dsec_detection_labels_root"]),
                "--mot-annotations-root",
                str(self.paths["dsec_mot_root"] / "annotations"),
                "--output-root",
                manifest_root,
                *self.config_args(),
                "--workers",
                str(settings["workers"]),
            )
            return [Step(name, argv)]
        if name == "validate_manifests":
            argv = self.module(
                "src.data.large_event_schema",
                "--manifest-root",
                manifest_root,
                *self.config_args(),
                "--deep",
            )
            return [Step(name, argv)]
        if name in CACHE_DATASETS:
            dataset = CACHE_DATASETS[name]
            source = ["--gen4-root", str(self.paths["gen4_root"])]
            if dataset != "gen4_1mpx":
                source = ["--dsec-root", str(self.paths["dsec_root"])]
            argv = self.module(
                "src.data.large_event_cache",
                "--manifest-root",
                manifest_root,
                *self.config_args(),
                "--cache-root",
                cache_root,
                "--dataset",
                dataset,
                *source,
                "--workers",
                str(settings["workers"]),
            )
            return [Step(name, argv)]
        if name == "pilot_preflight":
            storage = ":".join(
                str(self.paths[key])
                for key in ("runs_root", "results_root", "manifest_root", "cache_root")
            )
            env = {
                "RVT_ROOT": str(self.paths["rvt_root"]),
                "RVT_CHECKPOINT": str(self.paths["rvt_checkpoint"]),
                "STORAGE_PATHS": storage,
                # Leaves the job's 15 minutes for module loading and the shell checks.
                "PREFLIGHT_TIMEOUT": "600",
            }
            argv = ["bash", str(PREFLIGHT_SCRIPT), "--output-dir", str(self.run_dir(name))]
            return [Step(name, argv, env=env)]
        if name in ("pilot_pretrain", "pilot_finetune"):
            argv = self.training_argv(
                name,
                self.run_dir(name),
                MICRO_PLACEHOLDER if micro is None else micro,
                settings["overrides"],
                inputs.get("investigation_note"),
            )
            steps = [
                *self.memory_probe_steps(name, inputs),
                Step(
                    "segment_1",
                    [*argv, "--max-job-steps", str(settings["pause_at_step"])],
                    (EXIT_PAUSED, EXIT_STOPPED),
                    wall_flag="--max-wall-hours",
                    pausable=True,
                ),
                Step(
                    "segment_2",
                    [*argv, "--resume"],
                    (EXIT_DONE, EXIT_PAUSED, EXIT_STOPPED),
                    wall_flag="--max-wall-hours",
                    pausable=True,
                ),
            ]
            if settings.get("export"):
                argv = self.module(
                    "src.evaluation.large_event_export",
                    "--checkpoint",
                    "<last validation point of the pilot run>" if point is None else str(point),
                    "--output-dir",
                    str(self.results_root / "pilot" / "export"),
                    *self.model_args("cuda"),
                )
                steps.append(Step("export", argv))
            return steps
        if name in REPAIRABLE:
            micro_batches = settings["micro_batches"]
            argv = self.training_argv(
                name,
                self.run_dir(name, attempt),
                "<micro_batches>" if micro_batches is None else micro_batches,
                [*settings["overrides"], *inputs.get("overrides", [])],
                inputs.get("investigation_note"),
            )
            return [
                Step(
                    name,
                    argv,
                    (EXIT_DONE, EXIT_PAUSED, EXIT_STOPPED),
                    resume_dir=self.run_dir(name, attempt),
                    wall_flag="--max-wall-hours",
                    pausable=True,
                )
            ]
        evaluate = "src.evaluation.large_event_mot"
        output = ["--output-root", str(self.validation_root)]
        if name == "probe":
            argv = self.module(
                evaluate,
                "probe",
                "--requests",
                str(self.requests_path()),
                *output,
                *self.evaluation_args(),
                *self.model_args("cpu"),
            )
            return [Step(name, argv)]
        if name == "search":
            argv = self.module(
                evaluate,
                "search",
                *output,
                "--workers",
                str(settings["workers"]),
                "--threads",
                str(settings["threads"]),
                *self.evaluation_args(),
            )
            return [Step(name, argv, (EXIT_DONE, EXIT_PAUSED), wall_flag="--max-wall-hours")]
        if name == "select":
            return [Step(name, self.module(evaluate, "select", *output, *self.evaluation_args()))]
        if name == "freeze":
            argv = self.module(
                evaluate,
                "freeze",
                *output,
                "--freeze",
                str(self.freeze_path),
                "--final-output-dir",
                str(self.final_dir),
                *self.evaluation_args(),
            )
            return [Step(name, argv)]
        if name == "test":
            argv = self.module(
                evaluate,
                "test",
                "--freeze",
                str(self.freeze_path),
                *self.evaluation_args(),
                *self.model_args("cpu"),
            )
            return [Step(name, argv)]
        raise ExperimentError(f"no commands for {name}")

    # ---------------------------------------------------------------- outputs

    def outputs(self, name: str, attempt: str) -> list[Path]:
        """Files whose SHA-256 the completion record keeps (dependency integrity)."""

        if name in MANIFEST_DATASETS:
            return [
                self.paths["manifest_root"] / d / "manifest.json" for d in MANIFEST_DATASETS[name]
            ]
        if name == "validate_manifests":
            return [
                self.paths["manifest_root"] / d / "manifest.json"
                for names in MANIFEST_DATASETS.values()
                for d in names
            ]
        if name in CACHE_DATASETS:
            return [self.paths["cache_root"] / CACHE_DATASETS[name] / "cache_index.json"]
        if name == "pilot_preflight":
            return [self.run_dir(name) / "checks.tsv"]
        if name == "pilot_pretrain":
            run_dir = self.run_dir(name)
            return [
                run_dir / "pretrain" / "selection.json",
                run_dir / "pretrain" / "best.pt",
                self.pilot_summary(name),
            ]
        if name == "pilot_finetune":
            run_dir = self.run_dir(name)
            return [run_dir / "finetune" / "validation_requests.json", self.pilot_summary(name)]
        if name == "pretrain":
            run_dir = self.run_dir(name, attempt) / "pretrain"
            return [run_dir / "selection.json", run_dir / "best.pt"]
        if name == "finetune":
            return [self.run_dir(name, attempt) / "finetune" / "validation_requests.json"]
        if name == "probe":
            return [self.validation_root / "probe.json"]
        if name == "search":
            return sorted((self.validation_root / "trials").glob("point_*/reid.jsonl"))
        if name == "select":
            return [self.validation_root / "selection.json"]
        if name == "freeze":
            return [self.freeze_path]
        if name == "test":
            return [self.final_dir / "metrics_summary.json", self.final_dir / "test_run.json"]
        raise ExperimentError(f"no outputs for {name}")

    def completion_problems(self, name: str, attempt: str) -> list[str]:
        """Why the stage's outputs do not (yet) show a completed stage."""

        problems = [f"missing {path}" for path in self.outputs(name, attempt) if not path.is_file()]
        if name == "search" and not self.outputs(name, attempt):
            problems.append("no reid trial logs")
        if problems:
            return problems
        if name in MANIFEST_DATASETS or name in CACHE_DATASETS:
            for path in self.outputs(name, attempt):
                if not read_json(path).get("complete"):
                    problems.append(f"{path} is partial")
        elif name == "pilot_preflight":
            rows = (self.run_dir(name) / "checks.tsv").read_text(encoding="utf-8").splitlines()
            failed = [row.split("\t")[0] for row in rows[1:] if "\tFAIL\t" in row]
            if failed:
                problems.append(f"preflight checks failed: {failed}")
        elif name in ("pilot_pretrain", "pretrain"):
            directory = self.run_dir(name, attempt) / "pretrain"
            if not read_json(directory / "selection.json").get("finished"):
                problems.append(f"{directory / 'selection.json'} not finished")
        elif name in ("pilot_finetune", "finetune"):
            directory = self.run_dir(name, attempt)
            requests = read_json(directory / "finetune" / "validation_requests.json")
            if not requests.get("finished"):
                problems.append("validation requests not finished")
            for candidate in requests["candidates"]:
                path = directory / candidate["path"]
                if not path.is_file() or file_sha256(path) != candidate["sha256"]:
                    problems.append(f"candidate {path} missing or changed")
        elif name == "probe":
            probe = read_json(self.validation_root / "probe.json")
            if not probe.get("complete") or not probe.get("retained"):
                problems.append("probe.json is not complete with retained candidates")
        elif name == "freeze":
            if os.stat(self.freeze_path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                problems.append(f"{self.freeze_path} is writable")
        return problems

    def output_hashes(self, name: str, attempt: str) -> dict[str, str]:
        return {str(path): file_sha256(path) for path in self.outputs(name, attempt)}

    # ---------------------------------------------------------------- checks

    def dependency_problems(self, name: str) -> tuple[list[str], dict[str, Any]]:
        problems, record = [], {}
        for dependency in stage_from(self.config, name).depends_on:
            attempt, state = self.attempt(dependency)
            if state.get("status") != "complete":
                problems.append(f"{dependency} is {state.get('status', 'pending')}")
                continue
            outputs = state["completion"]["outputs"]
            for path, digest in outputs.items():
                if not Path(path).is_file():
                    problems.append(f"{dependency} output {path} is missing")
                elif file_sha256(path) != digest:
                    problems.append(f"{dependency} output {path} changed since completion")
            record[dependency] = {"attempt": attempt, "outputs": outputs}
        return problems, record

    def input_problems(self, name: str, inputs: Mapping[str, Any]) -> list[str]:
        problems = [
            f"{key} {self.paths[key]} does not exist"
            for key in STAGE_PATHS[name]
            if not self.paths[key].exists()
        ]
        venv = self.paths["venv_root"] / VENVS[stage_from(self.config, name).kind]
        if not (venv / "bin" / "activate").is_file():
            problems.append(f"virtual environment {venv} missing")
        settings = self.config["stages"][name]
        if name in REPAIRABLE and settings["micro_batches"] is None:
            problems.append(f"{name}.micro_batches is not frozen yet (pilot analysis)")
        if name == "finetune":
            selection = self.pretrain_checkpoint(name).parent / "selection.json"
            if selection.is_file() and read_json(selection).get("investigation"):
                if not inputs.get("investigation_note"):
                    problems.append("pretraining investigation triggers need --investigation-note")
        if name == "freeze":
            selection = self.validation_root / "selection.json"
            if not selection.is_file():
                problems.append(f"{selection} missing")
            elif inputs.get("confirm_selection") != file_sha256(selection):
                problems.append("--confirm-selection must equal the SHA-256 of selection.json")
        if name == "test":
            if not self.freeze_path.is_file():
                problems.append(f"{self.freeze_path} missing")
            else:
                if inputs.get("confirm_freeze") != file_sha256(self.freeze_path):
                    problems.append("--confirm-freeze must equal the SHA-256 of freeze.json")
                if os.stat(self.freeze_path).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH):
                    problems.append(f"{self.freeze_path} is writable")
        return problems

    def frozen_spec(self, name: str, attempt: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
        stage = stage_from(self.config, name)
        _, dependencies = self.dependency_problems(name)
        return {
            "stage": name,
            "attempt": attempt,
            "kind": stage.kind,
            "phase": stage.phase,
            "stage_config": self.config["stages"][name],
            "configs": {
                key: canonical_sha256(read_json(CONFIG_FILES[key])) for key in STAGE_CONFIGS[name]
            },
            "steps": [step.render() for step in self.steps(name, attempt, inputs)],
            "dependencies": dependencies,
            "inputs": dict(inputs),
        }

    # ---------------------------------------------------------------- planning

    def sbatch_argv(self, name: str, wall_hours: float) -> list[str]:
        stage = stage_from(self.config, name)
        slurm = self.config["slurm"]
        account = self.manifest["accounts"][stage.kind]
        argv = [
            "sbatch",
            "--parsable",
            f"--job-name={slurm['job_name_prefix']}_{name}",
            f"--account={account}",
            f"--partition={slurm[stage.kind]['partition']}",
            "--nodes=1",
            "--ntasks=1",
            f"--cpus-per-task={stage.cpus}",
            f"--mem={stage.mem_gb}G",
            f"--time={slurm_time(wall_hours)}",
            f"--signal={slurm['signal']}",
            "--no-requeue",
            f"--chdir={REPO_ROOT}",
            f"--output={self.state_dir / 'logs' / '%x_%j.out'}",
            f"--export=ALL,LARGE_EVENT_MOT_VENV={self.paths['venv_root'] / VENVS[stage.kind]}",
        ]
        if stage.kind == "gpu":
            argv.append(f"--gres=gpu:{stage_gpus(self.config, stage)}")
        return [*argv, str(SBATCH_SCRIPT), str(self.runs_root), name]

    def plan(self) -> dict[str, Any]:
        ledger = self.ledger() if (self.state_dir / LEDGER).exists() else []
        stages = []
        for name in STAGES:
            stage = stage_from(self.config, name)
            attempt, state = self.attempt(name)
            inputs = state.get("inputs", {})
            dependency_problems, dependencies = self.dependency_problems(name)
            blocked = [*dependency_problems, *self.input_problems(name, inputs)]
            if state.get("status") not in (None, "pending") and state.get("frozen"):
                current = self.frozen_spec(name, attempt, inputs)
                changed = sorted(k for k in current if current[k] != state["frozen"][k])
                if changed and state.get("status") != "complete":
                    blocked.append(f"stale: {changed} changed since the first submission")
            wall = requested_hours(stage.wall_hours)
            stages.append(
                {
                    "stage": name,
                    "kind": stage.kind,
                    "phase": stage.phase,
                    "manual": stage.manual,
                    "depends_on": list(stage.depends_on),
                    "attempt": attempt,
                    "status": state.get("status", "pending"),
                    "blocked": blocked,
                    "planned_gpu_hours_per_job": stage_gpus(self.config, stage) * wall,
                    "wall_hours": wall,
                    "configs": {
                        key: canonical_sha256(read_json(CONFIG_FILES[key]))
                        for key in STAGE_CONFIGS[name]
                    },
                    "dependencies": dependencies,
                    "commands": [
                        {
                            "name": step.name,
                            "accept_exit": list(step.accept),
                            "command": step.render(),
                        }
                        for step in self.steps(name, attempt, inputs)
                    ],
                    "sbatch": shlex.join(self.sbatch_argv(name, stage.wall_hours)),
                    "outputs": [str(path) for path in self.outputs(name, attempt)]
                    or [str(self.validation_root / "trials" / "point_<k>" / "reid.jsonl")],
                    "completion": completion_rule(name),
                }
            )
        return {
            "manifest": self.manifest,
            "experiment_config_sha256": canonical_sha256(self.config),
            "source": source_record(),
            "stages": stages,
            "budget": budget_summary(self.config, ledger),
            "open_jobs": self.open_jobs(ledger),
        }

    # ---------------------------------------------------------------- submission

    def submit(
        self,
        name: str,
        *,
        wall_hours: Optional[float] = None,
        repair: Optional[str] = None,
        overrides: Sequence[str] = (),
        investigation_note: Optional[str] = None,
        confirm_selection: Optional[str] = None,
        confirm_freeze: Optional[str] = None,
        reserve_reason: Optional[str] = None,
        reserve_note: Optional[str] = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        stage = stage_from(self.config, name)
        wall = requested_hours(stage.wall_hours if wall_hours is None else float(wall_hours))
        check_resources(self.config, stage, wall)
        if reserve_reason is not None and not reserve_note:
            raise ExperimentError("--reserve-reason needs --reserve-note")
        with self.locked():
            ledger = self.ledger()
            if any(job["stage"] == name for job in self.open_jobs(ledger)):
                raise ExperimentError(f"{name} already has an open job; wait, or reconcile it")
            record = self.record(name)
            attempt = record["active"]
            state = record["attempts"].get(attempt)
            if repair is not None:
                attempt, state = self._repair_attempt(name, record, repair, overrides)
            elif overrides:
                raise ExperimentError("--set is only allowed together with --repair")
            if state is not None and state["status"] == "complete":
                raise ExperimentError(f"{name} ({attempt}) is already complete")
            if state is not None and state["status"] == "stopped":
                raise ExperimentError(
                    f"{name} ({attempt}) ended with a stop rule; record a --repair or stop"
                )
            given = {
                key: value
                for key, value in (
                    ("investigation_note", investigation_note),
                    ("confirm_selection", confirm_selection),
                    ("confirm_freeze", confirm_freeze),
                )
                if value is not None
            }
            if name == "pilot_finetune":
                given.setdefault(
                    "investigation_note", self.config["stages"][name]["investigation_note"]
                )
            if state is None or "frozen" not in state:
                inputs = {**(state or {}).get("inputs", {}), **given}
            else:
                inputs = state["inputs"]
                if any(inputs.get(key) != value for key, value in given.items()):
                    raise ExperimentError(f"{name} ({attempt}) inputs are frozen: {inputs}")
            problems, _ = self.dependency_problems(name)
            problems += self.input_problems(name, inputs)
            if problems:
                raise ExperimentError(f"{name} is blocked: " + "; ".join(problems))
            frozen = self.frozen_spec(name, attempt, inputs)
            if state is not None and "frozen" in state and state["frozen"] != frozen:
                changed = sorted(k for k in frozen if frozen[k] != state["frozen"][k])
                raise ExperimentError(
                    f"{name} ({attempt}) is stale: {changed} changed since its first submission"
                )
            planned = stage_gpus(self.config, stage) * wall
            from_reserve = 0.0
            if stage.phase is not None:
                from_reserve = check_budget(
                    self.config, ledger, stage.phase, planned, reserve_reason
                )
            argv = self.sbatch_argv(name, wall)
            if dry_run:
                return {
                    "sbatch": shlex.join(argv),
                    "planned_gpu_hours": planned,
                    "from_reserve": from_reserve,
                    "frozen": frozen,
                }
            (self.state_dir / "logs").mkdir(parents=True, exist_ok=True)
            job_id = run_sbatch(argv)
            submission = {
                "event": "submit",
                "job_id": job_id,
                "stage": name,
                "attempt": attempt,
                "kind": stage.kind,
                "phase": stage.phase,
                "gpus": stage_gpus(self.config, stage),
                "cpus": stage.cpus,
                "wall_hours": wall,
                "planned_gpu_hours": planned,
                "from_reserve": from_reserve,
                "reserve": None
                if from_reserve <= EPSILON
                else {"reason": reserve_reason, "note": reserve_note},
                "sbatch": argv,
                "source": source_record(),
                "time_unix": time.time(),
            }
            append_jsonl(self.state_dir / LEDGER, submission)
            state = record["attempts"].setdefault(attempt, {})
            state.setdefault("frozen", frozen)
            state.setdefault("frozen_sha256", canonical_sha256(frozen))
            state["inputs"] = inputs
            state["status"] = "submitted"
            state.setdefault("jobs", []).append(job_id)
            record["active"] = attempt
            self.save_record(record)
            return {"job_id": job_id, **submission}

    def _repair_attempt(
        self, name: str, record: dict[str, Any], note: str, overrides: Sequence[str]
    ) -> tuple[str, Optional[dict[str, Any]]]:
        if name not in REPAIRABLE:
            raise ExperimentError(f"only {REPAIRABLE} have a repair attempt")
        if "repair" in record["attempts"]:
            stored = record["attempts"]["repair"].get("inputs", {})
            if stored.get("repair_note") != note or stored.get("overrides") != list(overrides):
                raise ExperimentError(
                    f"{name} already has its single repair attempt; submit without --repair "
                    "to continue it"
                )
            return "repair", record["attempts"]["repair"]
        main = record["attempts"].get("main", {})
        allowed = main.get("status") == "stopped"
        if name == "pretrain" and main.get("status") == "complete":
            selection = self.run_dir(name, "main") / "pretrain" / "selection.json"
            allowed = selection.is_file() and bool(read_json(selection).get("investigation"))
        if not allowed:
            raise ExperimentError(
                f"{name}: a repair needs a stop-rule end (or pretraining investigation triggers)"
            )
        if not note.strip():
            raise ExperimentError("--repair needs a non-empty note")
        record["attempts"]["repair"] = {
            "status": "pending",
            "inputs": {"repair_note": note, "overrides": list(overrides)},
        }
        return "repair", record["attempts"]["repair"]

    # ---------------------------------------------------------------- job

    def job(self, stage: str, log=print) -> int:
        job_id = os.environ.get("SLURM_JOB_ID")
        if not job_id:
            raise ExperimentError("job runs only inside a Slurm job submitted by submit")
        started = float(os.environ.get("SLURM_JOB_START_TIME") or time.time())
        with self.locked():
            ledger = self.ledger()
            submissions = [r for r in ledger if r["event"] == "submit" and r["job_id"] == job_id]
            if not submissions:
                raise ExperimentError(f"job {job_id} was not submitted by this runner")
            if any(r["event"] == "end" and r["job_id"] == job_id for r in ledger):
                raise ExperimentError(f"job {job_id} already has an end record (requeued?)")
            submission = submissions[0]
            name, attempt = submission["stage"], submission["attempt"]
            if name != stage:
                raise ExperimentError(f"job {job_id} was submitted for {name}, not {stage}")
            record = self.record(name)
            state = record["attempts"][attempt]
            problems = []
            if submission["kind"] == "gpu" and not (
                os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("SLURM_JOB_GPUS")
            ):
                problems.append("no GPU visible in a GPU stage")
            dependency_problems, _ = self.dependency_problems(name)
            problems += dependency_problems + self.input_problems(name, state["inputs"])
            if not problems and self.frozen_spec(name, attempt, state["inputs"]) != state["frozen"]:
                problems.append("stage specification changed after submission")
            state["status"] = "running" if not problems else "failed"
            self.save_record(record)
        if problems:
            log("refusing to run: " + "; ".join(problems))
            self._close(job_id, submission, started, EXIT_INPUT, "rejected", problems)
            return EXIT_INPUT
        deadline = started + 3600.0 * float(submission["wall_hours"])
        try:
            with Children(log) as children:
                status, code, measurements = self._execute(
                    name, attempt, state, children, deadline, log
                )
        except Exception as error:  # the ledger record is closed for every failure
            log("".join(traceback.format_exception(error)).rstrip())
            status, code, measurements = "failed", EXIT_FAILED, {"error": repr(error)}
        status, code = self._close(job_id, submission, started, code, status, measurements)
        log(f"{name} ({attempt}): {status} (exit {code})")
        return code

    def _close(
        self,
        job_id: str,
        submission: Mapping[str, Any],
        started: float,
        code: int,
        status: str,
        details: Any,
    ) -> tuple[str, int]:
        """Record the job's end; a stage is complete only if its outputs validate."""

        elapsed = max(0.0, time.time() - started)
        name, attempt = submission["stage"], submission["attempt"]
        with self.locked():
            record = self.record(name)
            state = record["attempts"][attempt]
            if status == "complete":
                problems = self.completion_problems(name, attempt)
                if problems:
                    status, code, details = "failed", EXIT_FAILED, {"completion": problems}
                else:
                    state["completion"] = {
                        "outputs": self.output_hashes(name, attempt),
                        "job_id": job_id,
                        "time_unix": time.time(),
                    }
            state["status"] = status
            state.setdefault("results", []).append(
                {"job_id": job_id, "exit": code, "details": details}
            )
            self.save_record(record)
            append_jsonl(
                self.state_dir / LEDGER,
                {
                    "event": "end",
                    "job_id": job_id,
                    "stage": name,
                    "attempt": attempt,
                    "elapsed_seconds": elapsed,
                    "gpu_hours": submission["gpus"] * elapsed / 3600.0,
                    "cpu_hours": submission["cpus"] * elapsed / 3600.0,
                    "exit_code": code,
                    "status": status,
                    "source": "job",
                    "time_unix": time.time(),
                },
            )
        return status, code

    def _wall_args(self, step: Step, deadline: float) -> list[str]:
        if step.wall_flag is None:
            return []
        margin = float(self.config["slurm"]["pause_margin_hours"])
        hours = (deadline - time.time()) / 3600.0 - margin
        if hours <= 0:
            raise ExperimentError("no wall time left before the pause margin")
        return [step.wall_flag, f"{hours:.4f}"]

    def _run_step(
        self, step: Step, children: Children, deadline: float, extra: Sequence[str] = ()
    ) -> ChildResult:
        argv = [*step.argv, *extra, *self._wall_args(step, deadline)]
        if step.resume_dir is not None and has_checkpoint(step.resume_dir):
            argv.append("--resume")
        return children.run(argv, step.env, step.pausable)

    def _execute(
        self,
        name: str,
        attempt: str,
        state: dict[str, Any],
        children: Children,
        deadline: float,
        log,
    ) -> tuple[str, int, dict[str, Any]]:
        if name in ("pilot_pretrain", "pilot_finetune"):
            return self._execute_pilot(name, state, children, deadline, log)
        (step,) = self.steps(name, attempt, state["inputs"])
        result = self._run_step(step, children, deadline)
        return outcome(result.code, step, children.interrupt)

    def _execute_pilot(
        self, name: str, state: dict[str, Any], children: Children, deadline: float, log
    ) -> tuple[str, int, dict[str, Any]]:
        inputs = state["inputs"]
        measurements = state.setdefault("measurements", {"memory_probe": []})
        micro = measurements.get("micro_batches")
        if micro is None:
            micro = self._memory_probe(name, inputs, measurements, children, deadline, log)
            if micro is None:
                status = "paused" if children.interrupt else "failed"
                return status, EXIT_PAUSED if children.interrupt else EXIT_FAILED, measurements
        run_dir, stage = self.run_dir(name), TRAINING[name]
        steps = {step.name: step for step in self.steps(name, "main", inputs, micro=micro)}
        # Segment 1 pauses at ``pause_at_step``; segment 2 resumes from that checkpoint.
        # A later job of the same stage continues with segment 2 from the last checkpoint.
        for segment in ("segment_1", "segment_2"):
            if _finished(run_dir, stage) or (segment == "segment_1" and has_checkpoint(run_dir)):
                continue
            if children.interrupt:
                return "paused", EXIT_PAUSED, measurements
            result = self._run_step(steps[segment], children, deadline)
            status, code, details = outcome(result.code, steps[segment], children.interrupt)
            if segment == "segment_1" and status == "paused" and children.interrupt is None:
                continue
            if status != "complete":
                return status, code, {**measurements, "last": details}
        if children.interrupt:
            return "paused", EXIT_PAUSED, measurements
        if "export" in steps:
            requests = read_json(run_dir / "finetune" / "validation_requests.json")
            point = run_dir / requests["candidates"][-1]["path"]
            (export,) = [
                s for s in self.steps(name, "main", inputs, micro, point) if s.name == "export"
            ]
            result = children.run(export.argv, export.env)
            status, code, details = outcome(result.code, export, children.interrupt)
            if status != "complete":
                return status, code, {**measurements, "last": details}
        summary = pilot_summary(self, name, micro, measurements["memory_probe"])
        write_json_atomic(self.pilot_summary(name), summary)
        return "complete", EXIT_DONE, measurements

    def _memory_probe(
        self,
        name: str,
        inputs: Mapping[str, Any],
        measurements: dict[str, Any],
        children: Children,
        deadline: float,
        log,
    ) -> Optional[int]:
        """First configured micro-batch count (largest micro-batch) that leaves the headroom.

        Runs that ran out of device memory count as not fitting; any other failure
        stops the pilot. Results are stored per candidate, so a later job continues.
        """

        probe = self.config["stages"][name]["memory_probe"]
        done = {entry["micro_batches"]: entry for entry in measurements["memory_probe"]}
        for step, micro in zip(self.memory_probe_steps(name, inputs), probe["micro_batches"]):
            if micro in done:
                if done[micro]["fits"]:
                    measurements["micro_batches"] = micro
                    return micro
                continue
            if children.interrupt:
                return None
            run_dir = Path(step.argv[step.argv.index("--run-dir") + 1])
            if has_checkpoint(run_dir):
                # Measured by an earlier job that ended before recording the result.
                result = ChildResult(EXIT_PAUSED, [])
            else:
                result = self._run_step(step, children, deadline)
            if children.interrupt and result.code != EXIT_PAUSED:
                return None
            entry = memory_entry(run_dir, micro, result, float(probe["headroom"]))
            if entry is None:
                raise ExperimentError(
                    f"memory probe with {micro} micro-batches failed without running out of "
                    f"memory (exit {result.code})"
                )
            measurements["memory_probe"].append(entry)
            if entry["fits"]:
                measurements["micro_batches"] = micro
            log(f"memory probe: {entry}")
            with self.locked():
                record = self.record(name)
                record["attempts"]["main"]["measurements"] = measurements
                self.save_record(record)
            if entry["fits"]:
                return micro
        raise ExperimentError(f"{name}: no micro-batch candidate fits with the configured headroom")

    # ---------------------------------------------------------------- reconcile

    def reconcile(
        self,
        job_id: Optional[str] = None,
        elapsed_hours: Optional[float] = None,
        note: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        with self.locked():
            ledger = self.ledger()
            open_jobs = {job["job_id"]: job for job in self.open_jobs(ledger)}
            if job_id is not None:
                if job_id not in open_jobs:
                    raise ExperimentError(f"job {job_id} is not open")
                if elapsed_hours is None or not note:
                    raise ExperimentError("a manual close needs --elapsed-hours and --note")
                states = {job_id: {"state": "MANUAL", "elapsed_seconds": 3600.0 * elapsed_hours}}
            else:
                states = query_sacct(sorted(open_jobs)) if open_jobs else {}
            closed = []
            for identifier, info in states.items():
                if identifier not in open_jobs or info["state"] not in (
                    *TERMINAL_SLURM_STATES,
                    "MANUAL",
                ):
                    continue
                submission = open_jobs[identifier]
                elapsed = float(info["elapsed_seconds"])
                end = {
                    "event": "end",
                    "job_id": identifier,
                    "stage": submission["stage"],
                    "attempt": submission["attempt"],
                    "elapsed_seconds": elapsed,
                    "gpu_hours": submission["gpus"] * elapsed / 3600.0,
                    "cpu_hours": submission["cpus"] * elapsed / 3600.0,
                    "exit_code": None,
                    "status": "failed",
                    "source": "manual" if info["state"] == "MANUAL" else "sacct",
                    "slurm_state": info["state"],
                    "note": note,
                    "time_unix": time.time(),
                }
                append_jsonl(self.state_dir / LEDGER, end)
                record = self.record(submission["stage"])
                state = record["attempts"][submission["attempt"]]
                if state["status"] in ("submitted", "running"):
                    state["status"] = "failed"
                    state.setdefault("results", []).append(
                        {
                            "job_id": identifier,
                            "exit": None,
                            "details": {"slurm_state": info["state"]},
                        }
                    )
                    self.save_record(record)
                closed.append(end)
            return closed

    def charge(
        self, phase: str, gpu_hours: float, note: str, job_id: Optional[str]
    ) -> dict[str, Any]:
        if phase not in self.config["budget"]["phases"]:
            raise ExperimentError(f"unknown phase {phase}")
        if gpu_hours <= 0 or not note:
            raise ExperimentError("an external charge needs positive GPU-hours and a note")
        record = {
            "event": "external",
            "phase": phase,
            "gpu_hours": float(gpu_hours),
            "note": note,
            "job_id": job_id,
            "time_unix": time.time(),
        }
        with self.locked():
            append_jsonl(self.state_dir / LEDGER, record)
        return record


# --------------------------------------------------------------------------- helpers


def completion_rule(name: str) -> str:
    rules = {
        "convert_gen4": "child exit 0; gen4_1mpx manifest.json complete (not sample-limited)",
        "convert_dsec": "child exit 0; dsec_mot and dsec_detection manifest.json complete",
        "validate_manifests": "large_event_schema --deep exit 0; manifests hashed",
        "pilot_preflight": "preflight exit 0; checks.tsv without FAIL",
        "pilot_pretrain": "memory probe fits; segment 1 exit 3 at pause_at_step; segment 2 "
        "exit 0; selection.json finished; best.pt; summary written",
        "pilot_finetune": "memory probe fits; segment 1 exit 3; segment 2 exit 0; requests "
        "finished with matching candidate SHA-256; GPU export exit 0; summary written",
        "pretrain": "exit 0 with selection.json finished and best.pt (3 pauses: submit "
        "again; 4 stop rule: --repair or stop)",
        "finetune": "exit 0 with validation_requests.json finished and every candidate "
        "SHA-256 matching",
        "probe": "exit 0; probe.json complete with retained candidates",
        "search": "exit 0 (3: paused below the trial budget, submit again)",
        "select": "exit 0; selection.json",
        "freeze": "exit 0; read-only freeze.json",
        "test": "exit 0; final metrics_summary.json and test_run.json",
    }
    return rules.get(
        name, f"child exit 0; {CACHE_DATASETS.get(name, '')} cache_index.json complete"
    )


def has_checkpoint(run_dir: Path) -> bool:
    return (run_dir / "last.pt").is_file() or (run_dir / "last_prev.pt").is_file()


def _selection_file(stage: str) -> str:
    return "selection.json" if stage == "pretrain" else "validation_requests.json"


def _finished(run_dir: Path, stage: str) -> bool:
    path = run_dir / stage / _selection_file(stage)
    return path.is_file() and bool(read_json(path).get("finished"))


def outcome(code: int, step: Step, interrupt: Optional[str]) -> tuple[str, int, dict[str, Any]]:
    details = {"step": step.name, "child_exit": code, "signal": interrupt}
    if code not in step.accept:
        return "failed", EXIT_FAILED, details
    if code == EXIT_PAUSED:
        return "paused", EXIT_PAUSED, details
    if code == EXIT_STOPPED:
        return "stopped", EXIT_STOPPED, details
    return "complete", EXIT_DONE, details  # a child that exits 0 has finished


def memory_entry(run_dir: Path, micro: int, result: ChildResult, headroom: float) -> Optional[dict]:
    """Fit record of one memory-probe run, or ``None`` for a failure other than OOM."""

    if result.code != EXIT_PAUSED:
        if any(marker in line for line in result.tail for marker in OOM_MARKERS):
            return {"micro_batches": micro, "fits": False, "reason": "out of memory"}
        return None
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    resources = read_jsonl(run_dir / "resources.jsonl")
    peak = max((m.get("device_peak_mib", 0.0) for m in metrics), default=None)
    total = resources[-1]["environment"].get("device_memory_bytes") if resources else None
    if peak is None or total is None:
        raise ExperimentError(f"{run_dir}: no device memory measurement")
    limit = (1.0 - headroom) * total / 2**20
    return {
        "micro_batches": micro,
        "fits": peak <= limit,
        "device_peak_mib": peak,
        "device_total_mib": total / 2**20,
        "limit_mib": limit,
        "seconds_per_step": metrics[-1]["seconds_per_step"],
    }


def _median(values: Sequence[float]) -> Optional[float]:
    ordered = sorted(values)
    if not ordered:
        return None
    middle = len(ordered) // 2
    return ordered[middle] if len(ordered) % 2 else 0.5 * (ordered[middle - 1] + ordered[middle])


def pilot_summary(
    experiment: Experiment, name: str, micro: int, memory_probe: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Measurements of a pilot stage and the checks of its pause/resume cycle."""

    settings = experiment.config["stages"][name]
    run_dir = experiment.run_dir(name)
    total = int(override_value(settings["overrides"], "total_steps"))
    pause = int(settings["pause_at_step"])
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    resources = read_jsonl(run_dir / "resources.jsonl")
    steps = [record["step"] for record in metrics]
    checks = {
        "paused_and_resumed": any(
            r["exit_code"] == EXIT_PAUSED and 0 < r["step"] <= pause for r in resources[:-1]
        ),
        "finished_at_total": bool(resources)
        and resources[-1]["exit_code"] == EXIT_DONE
        and resources[-1]["step"] == total,
        "metrics_steps_increasing": all(a < b for a, b in zip(steps, steps[1:])),
        "metrics_end_at_total": bool(steps) and steps[-1] == total,
    }
    if not all(checks.values()):
        raise ExperimentError(f"{name}: pause/resume checks failed: {checks}")
    steady = [record for record in metrics[1:] if not record["validation_seconds"]] or metrics
    summary = {
        "stage": name,
        "micro_batches": micro,
        "memory_probe": list(memory_probe),
        "checks": checks,
        "steps": total,
        "jobs": [
            {
                key: r[key]
                for key in (
                    "start_step",
                    "step",
                    "wall_seconds",
                    "gpu_hours",
                    "exit_code",
                    "reason",
                )
            }
            for r in resources
        ],
        "steady_state": {
            "intervals": len(steady),
            "frames_per_second_median": _median([r["frames_per_second"] for r in steady]),
            "seconds_per_step_median": _median([r["seconds_per_step"] for r in steady]),
            "data_wait_fraction_median": _median([r["data_wait_fraction"] for r in steady]),
        },
        "device_peak_mib_max": max((r.get("device_peak_mib", 0.0) for r in metrics), default=None),
        "host_peak_rss_mib_max": max((r["host_peak_rss_mib"] for r in metrics), default=None),
        "checkpoint_seconds_total": sum(r["checkpoint_seconds"] for r in metrics),
        "validation_seconds_total": sum(r["validation_seconds"] for r in metrics),
        "last_checkpoint_bytes": (run_dir / "last.pt").stat().st_size
        if (run_dir / "last.pt").exists()
        else None,
        "skipped_updates": sum(r["skipped"] for r in metrics),
        "nonfinite_free": all(
            all(math.isfinite(v) for v in r["loss"].values() if isinstance(v, (int, float)))
            for r in metrics
        ),
    }
    if settings.get("export"):
        sequences = read_json(CONFIG_FILES["data"])["dsec_mot"]["splits"]["val"]
        export_dir = experiment.results_root / "pilot" / "export"
        summary["export"] = {
            seq: read_json(export_dir / f"{seq}.json")["timing"] for seq in sequences
        }
    return summary


# --------------------------------------------------------------------------- CLI


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    paths = {name: str(Path(getattr(args, name)).expanduser().resolve()) for name in PATHS}
    if args.trackeval_root:
        paths["trackeval_root"] = str(Path(args.trackeval_root).expanduser().resolve())
    return {
        "format": MANIFEST_FORMAT,
        "version": 1,
        "created_unix": time.time(),
        "repo": str(REPO_ROOT),
        "source": source_record(),
        "paths": paths,
        "accounts": {"gpu": args.gpu_account, "cpu": args.cpu_account},
        "seed": int(args.seed),
    }


def command_init(args: argparse.Namespace) -> int:
    config_path = Path(args.config)
    manifest = build_manifest(args)
    experiment = Experiment(manifest, load_experiment_config(config_path), config_path)
    if args.dry_run:
        print_json(experiment.plan())
        return EXIT_DONE
    path = experiment.state_dir / MANIFEST
    for key in ("runs_root", "results_root", "manifest_root", "cache_root"):
        experiment.paths[key].mkdir(parents=True, exist_ok=True)
    experiment.state_dir.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    except FileExistsError:
        raise ExperimentError(f"{path} exists; the experiment manifest is immutable") from None
    os.chmod(path, stat.S_IRUSR)
    print(f"experiment manifest {path}")
    return EXIT_DONE


def command_plan(args: argparse.Namespace) -> int:
    print_json(Experiment.load(args.runs_root, args.config).plan())
    return EXIT_DONE


def command_submit(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.runs_root, args.config)
    result = experiment.submit(
        args.stage,
        wall_hours=args.wall_hours,
        repair=args.repair,
        overrides=args.set,
        investigation_note=args.investigation_note,
        confirm_selection=args.confirm_selection,
        confirm_freeze=args.confirm_freeze,
        reserve_reason=args.reserve_reason,
        reserve_note=args.reserve_note,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        print_json(result)
    else:
        print(
            f"submitted {args.stage} as job {result['job_id']} "
            f"({result['planned_gpu_hours']:.2f} GPU-h reserved)"
        )
    return EXIT_DONE


def command_job(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.runs_root, args.config)
    return experiment.job(args.stage, log=lambda message: print(message, flush=True))


def command_status(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.runs_root, args.config)
    plan = experiment.plan()
    for stage in plan["stages"]:
        state = "blocked" if stage["blocked"] and stage["status"] != "complete" else stage["status"]
        print(f"{stage['stage']:22s} {stage['kind']:3s} {stage['attempt']:6s} {state}")
        if stage["blocked"] and stage["status"] != "complete":
            for reason in stage["blocked"]:
                print(f"{'':22s}   - {reason}")
    budget = plan["budget"]
    print("\nGPU-hours (open jobs at their requested wall time):")
    for phase, values in budget["phases"].items():
        print(
            f"  {phase:10s} {values['used']:8.2f} of {values['ceiling']:7.2f} "
            f"(open {values['open']:.2f}, from reserve {values['from_reserve']:.2f})"
        )
    print(
        f"  {'reserve':10s} {budget['reserve']['used']:8.2f} of {budget['reserve']['ceiling']:7.2f}"
    )
    print(f"  {'total':10s} {budget['total']['used']:8.2f} of {budget['total']['ceiling']:7.2f}")
    print(f"CPU core-hours (closed jobs): {budget['cpu_core_hours_closed']:.1f}")
    for job in plan["open_jobs"]:
        print(f"open job {job['job_id']}: {job['stage']} ({job['attempt']}), {job['wall_hours']} h")
    return EXIT_DONE


def command_reconcile(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.runs_root, args.config)
    closed = experiment.reconcile(args.job_id, args.elapsed_hours, args.note)
    print_json(closed)
    return EXIT_DONE


def command_charge(args: argparse.Namespace) -> int:
    experiment = Experiment.load(args.runs_root, args.config)
    print_json(experiment.charge(args.phase, args.gpu_hours, args.note, args.job_id))
    return EXIT_DONE


def print_json(value: Any) -> None:
    json.dump(value, sys.stdout, indent=2, default=str)
    sys.stdout.write("\n")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, runs_root: bool = True) -> None:
        if runs_root:
            sub.add_argument("--runs-root", required=True)
        sub.add_argument("--config", default=str(DEFAULT_CONFIG))

    init = commands.add_parser("init", help="write the immutable experiment manifest")
    common(init, runs_root=False)
    for name, text in PATHS.items():
        init.add_argument(f"--{name.replace('_', '-')}", required=True, help=text)
    init.add_argument("--trackeval-root", default=None, help="TrackEval checkout (optional)")
    init.add_argument("--gpu-account", required=True)
    init.add_argument("--cpu-account", required=True)
    init.add_argument("--seed", type=int, default=0)
    init.add_argument("--dry-run", action="store_true", help="print the plan; write nothing")

    plan = commands.add_parser("plan", help="complete stage dry-run report (writes nothing)")
    common(plan)

    submit = commands.add_parser("submit", help="submit one job for one stage")
    common(submit)
    submit.add_argument("--stage", required=True, choices=STAGES)
    submit.add_argument("--wall-hours", type=float, default=None)
    submit.add_argument("--repair", default=None, metavar="NOTE", help="start the repair attempt")
    submit.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    submit.add_argument("--investigation-note", default=None)
    submit.add_argument("--confirm-selection", default=None, metavar="SHA256")
    submit.add_argument("--confirm-freeze", default=None, metavar="SHA256")
    submit.add_argument("--reserve-reason", default=None)
    submit.add_argument("--reserve-note", default=None)
    submit.add_argument("--dry-run", action="store_true", help="print the sbatch command only")

    job = commands.add_parser("job", help="run the submitted stage (inside Slurm only)")
    common(job)
    job.add_argument("--stage", required=True, choices=STAGES)

    status = commands.add_parser("status", help="stage status and budget")
    common(status)

    reconcile = commands.add_parser("reconcile", help="close jobs that ended without a record")
    common(reconcile)
    reconcile.add_argument("--job-id", default=None)
    reconcile.add_argument("--elapsed-hours", type=float, default=None)
    reconcile.add_argument("--note", default=None)

    charge = commands.add_parser("charge", help="record GPU time spent outside the runner")
    common(charge)
    charge.add_argument("--phase", required=True)
    charge.add_argument("--gpu-hours", type=float, required=True)
    charge.add_argument("--note", required=True)
    charge.add_argument("--job-id", default=None)
    return parser.parse_args(argv)


COMMANDS = {
    "init": command_init,
    "plan": command_plan,
    "submit": command_submit,
    "job": command_job,
    "status": command_status,
    "reconcile": command_reconcile,
    "charge": command_charge,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    os.umask(0o077)
    try:
        return COMMANDS[args.command](args)
    except ExperimentError as error:
        print(f"error: {error}", file=sys.stderr)
        return EXIT_INPUT


if __name__ == "__main__":
    raise SystemExit(main())
