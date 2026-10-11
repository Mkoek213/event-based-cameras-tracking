"""Staged training of the large event MOT detector with interruption-safe resume.

Stages (``configs/large_event_mot/training.json``):

- ``pretrain``: 1Mpx train chunks, source head, RVT-B checkpoint initialization. Phase A
  trains the EF/VG stems, gate, fusion LayerNorm, and class predictors; phase B trains
  everything except the embedding head (not run). Source validation (5-class COCO mAP
  on 1Mpx val, EMA weights) selects ``pretrain/best.pt`` (ties: AP50, then the earlier
  step) and drives patience.
- ``finetune``: DSEC-MOT + DSEC-Detection train sequences, target head transferred from a
  pretraining inference checkpoint, new embedding head and identity classifier. Phase A
  trains the embedding head, class predictors, and classifier; phase B trains
  everything. Each validation point saves ``finetune/point_<k>_step_<step>.pt`` (EMA),
  logs the DSEC-Detection detection monitor, and lists the candidate in
  ``finetune/validation_requests.json`` for DSEC-MOT val export and tracking. Selection
  happens there, on DSEC-MOT validation HOTA; this module never reads test data.

Each optimizer step takes the batch plan of that step (random BPTT clips from a zero
state and TBPTT stream segments whose state the :class:`StreamStateBank` carries),
accumulates ``micro_batches`` backward passes, clips the global gradient norm, applies
AdamW at the scheduled learning rate, and updates the EMA. A non-finite loss or
gradient norm skips the update. Sampling is a pure function of the step, so resuming
from ``last.pt`` (model, EMA, optimizer, schedule, state bank, stop rules, selection,
RNG) continues with the identical clip order and stream states.

Run directory: ``run.json`` (resolved configuration and provenance), ``last.pt`` and
``last_prev.pt``, ``metrics.jsonl``, ``validation.jsonl``, ``resources.jsonl`` (one record
per job), and the stage's selection files.

Exit codes: 0 complete (or patience stop), 3 paused (SIGUSR1/SIGTERM, ``--max-job-steps``,
or ``--max-wall-hours``; resume with ``--resume``), 4 stop rule (``last`` kept).

Examples::

    python -m src.training.large_event_detector --stage pretrain --run-dir <run_dir> \\
        --manifest-root <manifests> --cache-root <cache> --rvt-root <rvt> \\
        --rvt-checkpoint <rvt-b.ckpt> --dry-run
    python -m src.training.large_event_detector --stage pretrain --run-dir <scratch> \\
        --rvt-root <rvt> --smoke --device cpu
"""

from __future__ import annotations

import argparse
import json
import resource
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.large_event_cache import CacheError
from src.data.large_event_dataset import (
    BatchPlanSampler,
    LargeEventClipDataset,
    SamplingError,
    SequenceData,
    StageData,
    collate_clips,
    load_sequences,
    network_input,
    small_box_rule,
    stage_data,
)
from src.data.large_event_schema import SchemaError, settings_hash, write_json_atomic
from src.evaluation.large_event_detection_map import evaluate_detection_map, map_settings
from src.models.large_event_detector import (
    DEFAULT_CONFIG as DEFAULT_MODEL_CONFIG,
)
from src.models.large_event_detector import (
    LargeEventDetector,
    ModelConfigError,
    build_large_event_detector,
    config_sha256,
    load_model_config,
    model_geometry,
    transfer_source_to_target,
)
from src.models.large_event_modules import RecurrentState
from src.training.large_event_losses import (
    DEFAULT_CONFIG as DEFAULT_TRAINING_CONFIG,
)
from src.training.large_event_losses import (
    build_identity_classifier,
    load_training_config,
    loss_settings,
    stage_losses,
)
from src.training.large_event_runtime import (
    CHECKPOINT_VERSION,
    INFERENCE_FORMAT,
    LAST,
    LAST_FORMAT,
    LearningRateSchedule,
    ModelEma,
    RuntimeConfigError,
    StateLeakageError,
    StopRuleError,
    StopRules,
    StreamStateBank,
    append_jsonl,
    apply_phase,
    atomic_torch_save,
    batch_positives,
    build_optimizer,
    class_sources,
    environment_record,
    file_sha256,
    load_inference_checkpoint,
    load_last,
    phase_of,
    resolve_stage,
    save_last,
    sequences_record,
    smoke_configs,
    source_commit,
    synthetic_sequences,
    synthetic_stage_data,
    truncate_jsonl,
    validation_points,
)

DEFAULT_DATA_CONFIG = DEFAULT_MODEL_CONFIG.parent / "data.json"
EXIT_COMPLETE, EXIT_PAUSED, EXIT_STOP_RULE = 0, 3, 4
SELECTION_FILE = "selection.json"
REQUESTS_FILE = "validation_requests.json"


@dataclass
class JobLimits:
    """Limits of one job (not part of the run identity)."""

    max_job_steps: Optional[int] = None
    max_wall_hours: Optional[float] = None


@dataclass
class TrainingInputs:
    """Everything a run needs besides its run directory."""

    stage: str
    settings: dict[str, Any]
    training_config: dict[str, Any]
    data_config: dict[str, Any]
    model_config: dict[str, Any]
    data: StageData
    validation_sequences: list[SequenceData]
    seed: int
    smoke: bool
    rvt_root: Optional[str] = None
    rvt_checkpoint: Optional[str] = None
    pretrain_checkpoint: Optional[str] = None
    investigation_note: Optional[str] = None
    overrides: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- initialization


def architecture_sha256(model_config: Mapping[str, Any]) -> str:
    """Model configuration hash without the training-only identity classifier size."""

    config = json.loads(json.dumps(model_config))
    config["embedding"].pop("identity_classifier", None)
    return config_sha256(config)


def stage_geometry(inputs: TrainingInputs) -> str:
    """The stage's model geometry; every stage dataset must share its input size."""

    name = inputs.settings["geometry"]
    reference = model_geometry(inputs.model_config, name)
    for dataset, items in inputs.data.sequences.items():
        geometry = model_geometry(inputs.model_config, dataset)
        data_geometry = items[0].geometry
        if (geometry.input_hw, geometry.valid_hw, geometry.partition_size) != (
            reference.input_hw,
            reference.valid_hw,
            reference.partition_size,
        ) or (data_geometry.input_height, data_geometry.input_width) != reference.input_hw:
            raise RuntimeConfigError(f"{dataset} does not share the {name} model geometry")
    for sequence in inputs.validation_sequences:
        if (sequence.geometry.input_height, sequence.geometry.input_width) != reference.input_hw:
            raise RuntimeConfigError(f"{sequence.sequence}: validation geometry differs")
    return name


def pretraining_source(inputs: TrainingInputs) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Source-head weights for fine-tuning and their provenance.

    A pretraining selection with recorded investigation triggers needs an explicit
    ``--investigation-note`` (design record Section 11.3).
    """

    if inputs.pretrain_checkpoint is None:
        if not inputs.smoke:
            raise RuntimeConfigError("fine-tuning needs --pretrain-checkpoint")
        source = build_large_event_detector(
            inputs.model_config,
            ontology="source",
            geometry="gen4_1mpx",
            init="random",
            rvt_root=inputs.rvt_root,
        )
        return source.state_dict(), {"source": "random source model (smoke mode)"}
    path = Path(inputs.pretrain_checkpoint)
    payload = load_inference_checkpoint(path)
    if payload["stage"] != "pretrain" or payload["ontology"] != "source":
        raise RuntimeConfigError(f"{path}: not a pretraining (source head) checkpoint")
    if architecture_sha256(payload["model_config"]) != architecture_sha256(inputs.model_config):
        raise RuntimeConfigError(f"{path}: built with a different model architecture")
    selection_path = path.parent / SELECTION_FILE
    if not selection_path.is_file():
        raise RuntimeConfigError(f"{path}: no {SELECTION_FILE} next to the checkpoint")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["investigation"] and not inputs.investigation_note:
        raise RuntimeConfigError(
            f"pretraining investigation triggers {selection['investigation']} block fine-tuning; "
            "record the cause with --investigation-note"
        )
    provenance = {
        "source": str(path),
        "sha256": file_sha256(path),
        "step": payload["step"],
        "metrics": payload.get("metrics"),
        "pretraining_finished": selection.get("finished"),
        "investigation": selection["investigation"],
        "investigation_note": inputs.investigation_note,
    }
    return payload["model_state"], provenance


def initial_model(inputs: TrainingInputs, geometry: str) -> tuple[LargeEventDetector, dict]:
    if inputs.stage == "pretrain":
        model = build_large_event_detector(
            inputs.model_config,
            ontology="source",
            geometry=geometry,
            init=inputs.settings["init"],
            rvt_root=inputs.rvt_root,
            checkpoint=inputs.rvt_checkpoint,
        )
        return model, {"init": inputs.settings["init"], "checkpoint": model.metadata["checkpoint"]}
    state, provenance = pretraining_source(inputs)
    model = transfer_source_to_target(
        state,
        inputs.model_config,
        geometry=geometry,
        rvt_root=inputs.rvt_root,
        source_name=f"pretraining checkpoint {provenance.get('sha256', 'random')}",
    )
    return model, {"init": "pretrain_transfer", "pretraining": provenance}


def run_identity(inputs: TrainingInputs) -> dict[str, Any]:
    """Settings that must not change across resumes of one run."""

    return {
        "stage": inputs.stage,
        "settings": inputs.settings,
        "losses": inputs.training_config["losses"],
        "stop_rules": inputs.training_config["stop_rules"],
        "data_config_sha256": settings_hash(inputs.data_config),
        "model_config_sha256": config_sha256(inputs.model_config),
        "data_sha256": sequences_record(inputs.data.sequences)["sha256"],
        "validation_sequences": [item.sequence for item in inputs.validation_sequences],
        "identity_table_sha256": inputs.data.identity_table.sha256(),
        "seed": inputs.seed,
        "smoke": inputs.smoke,
    }


# --------------------------------------------------------------------------- trainer


class Trainer:
    """One training stage in one run directory."""

    def __init__(
        self,
        inputs: TrainingInputs,
        run_dir: str | Path,
        device: str | torch.device,
        micro_batches: int = 1,
        workers: int = 0,
        resume: bool = False,
        limits: Optional[JobLimits] = None,
        log=print,
    ):
        self.inputs = inputs
        self.stage, self.settings = inputs.stage, inputs.settings
        self.run_dir = Path(run_dir)
        self.device = torch.device(device)
        self.micro_batches, self.workers = int(micro_batches), int(workers)
        self.limits = limits or JobLimits()
        self.log = log
        self.plan = inputs.data.plan
        if self.plan.batch_size % self.micro_batches:
            raise RuntimeConfigError(f"{self.micro_batches} micro-batches do not divide the batch")
        self.loss_settings = loss_settings(inputs.training_config, inputs.model_config)
        self.schedule = LearningRateSchedule(self.settings)
        self.identity = run_identity(inputs)
        self.geometry = stage_geometry(inputs)
        self.stage_dir = self.run_dir / self.stage
        self.interrupt: Optional[str] = None
        self.step = 0
        self.saved_step: Optional[int] = None
        self.selection: dict[str, Any] = {}
        self.points = validation_points(self.settings) if self.stage == "finetune" else []

        if resume:
            self._resume()
        else:
            self._fresh()
        self.validation_dataset = (
            LargeEventClipDataset(inputs.validation_sequences, inputs.data_config)
            if inputs.validation_sequences
            else None
        )
        self._reset_interval()

    # ------------------------------------------------------------------- setup

    def _components(self, model: LargeEventDetector) -> None:
        inputs = self.inputs
        self.model = model.to(self.device)
        self.classifier = None
        if self.loss_settings.stage(self.stage).identity_enabled:
            self.classifier = build_identity_classifier(
                self.loss_settings, self.stage, len(inputs.data.identity_table), inputs.seed
            ).to(self.device)
        ema = self.settings["ema"]
        self.ema = ModelEma(self.model, ema["decay"], ema["ramp_steps"])
        self.optimizer, self.optimizer_groups = build_optimizer(
            self.model, self.classifier, self.settings
        )
        self.bank = StreamStateBank([(dataset, local) for dataset, local in self.plan.stream_slots])
        stream_counts: dict[str, int] = {}
        for dataset, _ in self.plan.stream_slots:
            stream_counts[dataset] = stream_counts.get(dataset, 0) + 1
        self.stop_rules = StopRules(
            inputs.training_config["stop_rules"],
            self.settings,
            list(self.model.classes),
            class_sources(inputs.data.sequences),
            stream_counts,
        )
        self.phase: Optional[str] = None

    def _fresh(self) -> None:
        if (self.run_dir / LAST).exists():
            raise RuntimeConfigError(f"{self.run_dir} already holds a run; use --resume")
        model, initialization = initial_model(self.inputs, self.geometry)
        self._components(model)
        self.initialization = initialization
        self.selection = self._initial_selection()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.stage_dir.mkdir(parents=True, exist_ok=True)
        write_json_atomic(self.run_dir / "run.json", self._run_record())

    def _resume(self) -> None:
        payload, path = load_last(self.run_dir)
        stored = payload["identity"]
        if stored != self.identity:
            changed = sorted(key for key in self.identity if stored.get(key) != self.identity[key])
            raise RuntimeConfigError(f"cannot resume {path}: settings changed in {changed}")
        ontology = "source" if self.stage == "pretrain" else "target"
        model = build_large_event_detector(
            self.inputs.model_config,
            ontology=ontology,
            geometry=self.geometry,
            init="random",
            rvt_root=self.inputs.rvt_root,
        )
        model.load_state_dict(payload["model"], strict=True)
        model.metadata = payload["model_metadata"]
        self._components(model)
        if self.classifier is not None:
            self.classifier.load_state_dict(payload["classifier"], strict=True)
        self.ema.load_state_dict(payload["ema"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.bank.load_state_dict(payload["bank"], self.device)
        self.stop_rules.load_state_dict(payload["stop_rules"])
        self.selection = payload["selection"]
        self.initialization = payload["initialization"]
        self.step = int(payload["step"])
        torch.set_rng_state(payload["rng"]["torch"])
        if self.device.type == "cuda" and payload["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state(payload["rng"]["cuda"], self.device)
        for name in ("metrics.jsonl", "validation.jsonl"):
            truncate_jsonl(self.run_dir / name, self.step)
        self.log(f"resumed {path} at step {self.step}")

    def _initial_selection(self) -> dict[str, Any]:
        if self.stage == "pretrain":
            return {
                "best": None,
                "evaluations": [],
                "patience_reference": None,
                "without_improvement": 0,
                "investigation": [],
                "finished": False,
                "stopped_by_patience": False,
            }
        return {"candidates": [], "finished": False}

    def _run_record(self) -> dict[str, Any]:
        return {
            "identity": self.identity,
            "identity_sha256": settings_hash(self.identity),
            "overrides": self.inputs.overrides,
            "initialization": self.initialization,
            "model": {
                key: self.model.metadata.get(key)
                for key in (
                    "upstream",
                    "config_sha256",
                    "ontology",
                    "classes",
                    "parameters",
                    "training_only",
                    "recurrent_state",
                    "transfer",
                )
            },
            "optimizer_groups": self.optimizer_groups,
            "identity_table": {
                "identities": len(self.inputs.data.identity_table),
                "sha256": self.inputs.data.identity_table.sha256(),
            },
            "class_sources": class_sources(self.inputs.data.sequences),
            "sampling": {
                "clip_steps": self.plan.steps,
                "random_slots": self.plan.random_slots,
                "stream_slots": [list(slot) for slot in self.plan.stream_slots],
                "policy": "random clips from a zero state (BPTT); stream segments carry "
                "detached state per slot and reset at every sequence start (TBPTT)",
            },
            "data": sequences_record(self.inputs.data.sequences),
            "validation_sequences": [
                {"dataset": item.dataset, "sequence": item.sequence, "split": item.split}
                for item in self.inputs.validation_sequences
            ],
            "configs": {
                "training": self.inputs.training_config,
                "data": self.inputs.data_config,
                "model": self.inputs.model_config,
            },
            "source": source_commit(),
            "environment": environment_record(self.device),
        }

    # ------------------------------------------------------------------- loop

    def _reset_interval(self) -> None:
        self.interval: dict[str, Any] = {
            "steps": 0,
            "updates": 0,
            "skipped": 0,
            "terms": {},
            "counts": {},
            "grad_norm_sum": 0.0,
            "grad_norm_max": 0.0,
            "frames": 0,
            "data_seconds": 0.0,
            "seconds": 0.0,
            "validation_seconds": 0.0,
            "checkpoint_seconds": 0.0,
            "started": time.perf_counter(),
        }
        self.bank_counters_logged = dict(self.bank.counters)

    def _set_phase(self) -> None:
        phase = phase_of(self.settings, self.step)
        if phase != self.phase:
            counts = apply_phase(self.model, self.classifier, self.settings, phase)
            if self.phase is not None:
                self.log(f"step {self.step}: {phase} ({counts['trainable']:,} trainable)")
            self.phase = phase

    def _install_signals(self) -> list[tuple[int, Any]]:
        def handler(signum, _frame):
            self.interrupt = signal.Signals(signum).name

        previous = []
        for signum in (signal.SIGUSR1, signal.SIGTERM):
            try:
                previous.append((signum, signal.signal(signum, handler)))
            except ValueError:  # Not the main thread (for example inside a test runner).
                pass
        return previous

    def _loader(self) -> DataLoader:
        sampler = BatchPlanSampler(
            self.plan, self.step, int(self.settings["total_steps"]) - self.step, self.micro_batches
        )
        return DataLoader(
            self.inputs.data.dataset,
            batch_sampler=sampler,
            collate_fn=collate_clips,
            num_workers=self.workers,
            pin_memory=self.device.type == "cuda",
            persistent_workers=False,
            prefetch_factor=2 if self.workers else None,
        )

    def run(self) -> int:
        total = int(self.settings["total_steps"])
        started, job_start_step = time.time(), self.step
        previous = self._install_signals()
        reason, code = "complete", EXIT_COMPLETE
        finished = bool(self.selection.get("finished"))  # resuming a completed run is a no-op
        try:
            if self.step < total and not self.selection.get("stopped_by_patience"):
                batches = iter(self._loader())
                while self.step < total:
                    self._optimizer_step(batches)
                    self.step += 1
                    self._after_step()
                    if self.stage == "pretrain" and self.selection["stopped_by_patience"]:
                        reason = "patience"
                        break
                    pause = self._pause_reason(started, job_start_step)
                    if pause is not None:
                        reason, code = pause, EXIT_PAUSED
                        break
            self._log_interval()
            if not finished:
                if code == EXIT_COMPLETE:
                    self._finish()
                if self.saved_step != self.step or code == EXIT_COMPLETE:
                    self._save_last()
        except StopRuleError as error:
            # The previous ``last`` stays the resume point; the log keeps the evidence.
            reason, code = f"stop rule: {error}", EXIT_STOP_RULE
            self._log_interval()
            self.log(f"stopping at step {self.step}: {error}")
        finally:
            for signum, handler in previous:
                signal.signal(signum, handler)
            self._record_job(started, job_start_step, reason, code)
        self.log(f"{self.stage}: {reason} at step {self.step} (exit {code})")
        return code

    def _pause_reason(self, started: float, job_start_step: int) -> Optional[str]:
        if self.interrupt is not None:
            return f"signal {self.interrupt}"
        if self.limits.max_job_steps is not None and (
            self.step - job_start_step >= self.limits.max_job_steps
        ):
            return "max job steps"
        if self.limits.max_wall_hours is not None and (
            time.time() - started >= 3600.0 * self.limits.max_wall_hours
        ):
            return "max wall hours"
        return None

    def _optimizer_step(self, batches) -> None:
        self._set_phase()
        lr = self.schedule(self.step)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.optimizer.zero_grad(set_to_none=True)
        step_started = time.perf_counter()
        terms: dict[str, float] = {}
        counts: dict[str, int] = {}
        flags = {"identity_ce": False, "triplet": False}
        finite = True
        for _ in range(self.micro_batches):
            waited = time.perf_counter()
            batch = next(batches)
            self.interval["data_seconds"] += time.perf_counter() - waited
            loss = self._micro_batch(batch)
            finite &= bool(torch.isfinite(loss.total.detach()))
            for name, value in loss.terms.items():
                terms[name] = terms.get(name, 0.0) + float(value) / self.micro_batches
            for name, value in loss.counts.items():
                counts[name] = counts.get(name, 0) + int(value)
            for name in flags:
                flags[name] |= bool(loss.flags.get(name, False))
            self.interval["frames"] += int(batch["valid"].sum())

        parameters = [
            p
            for group in self.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            parameters, float(self.settings["grad_clip_norm"])
        )
        finite &= bool(torch.isfinite(grad_norm))
        if finite:
            self.optimizer.step()
            self.ema.update(self.model)
        else:
            self.optimizer.zero_grad(set_to_none=True)
        self.stop_rules.record_update(self.step, skipped=not finite)
        self.stop_rules.record_identity(self.step, flags["identity_ce"], flags["triplet"])
        self.stop_rules.check_coverage()

        interval = self.interval
        interval["steps"] += 1
        interval["seconds"] += time.perf_counter() - step_started
        interval["lr"] = lr
        if finite:
            interval["updates"] += 1
            norm = float(grad_norm)
            interval["grad_norm_sum"] += norm
            interval["grad_norm_max"] = max(interval["grad_norm_max"], norm)
            for name, value in terms.items():
                interval["terms"][name] = interval["terms"].get(name, 0.0) + value
        else:
            interval["skipped"] += 1
        for name, value in counts.items():
            interval["counts"][name] = interval["counts"].get(name, 0) + value
        for name, value in flags.items():
            key = f"steps_with_{name}"
            interval["counts"][key] = interval["counts"].get(key, 0) + int(value)

    def _micro_batch(self, batch: Mapping[str, Any]):
        metas = batch["meta"]
        zero = self.model.initial_state(1, self.device)
        rows = [self.bank.take(meta, zero) if meta["kind"] == "stream" else zero for meta in metas]
        reset = batch["reset"].to(self.device)
        state = RecurrentState.cat(rows).reset(reset)
        if self.inputs.training_config["stop_rules"]["state_reset_check"] and not bool(
            state.is_zero()[reset].all()
        ):
            raise StateLeakageError("a reset clip starts from a non-zero state")
        event_frame, voxel_grid = network_input(batch, self.device)
        valid = batch["valid"].to(self.device)
        bf16 = self.settings["precision"] == "bf16"
        outputs = []
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16, enabled=bf16):
            for step in range(valid.shape[1]):
                output = self.model(
                    event_frame[:, step],
                    voxel_grid[:, step],
                    state=state,
                    valid=valid[:, step],
                    with_embeddings=bool(self.settings["with_embeddings"]),
                )
                state = output.state
                outputs.append(output)
        loss = stage_losses(
            outputs, batch, self.stage, self.loss_settings, self.model.yolox_head, self.classifier
        )
        (loss.total / self.micro_batches).backward()
        for row, meta in enumerate(metas):
            if meta["kind"] == "stream":
                self.bank.store(meta, state.rows([row]))
        self.stop_rules.record_batch(metas, batch_positives(batch, len(self.model.classes)))
        return loss

    # ------------------------------------------------------------------- bookkeeping

    def _after_step(self) -> None:
        if self.step % int(self.inputs.training_config["logging"]["every_steps"]) == 0:
            self._log_interval()
        validated = False
        started = time.perf_counter()
        if self.stage == "pretrain":
            every = int(self.settings["validation"]["every"])
            if self.step % every == 0 or self.step == int(self.settings["total_steps"]):
                self._source_validation()
                validated = True
        elif self.step in self.points:
            self._validation_point()
            validated = True
        self.interval["validation_seconds"] += time.perf_counter() - started
        final = self.step == int(self.settings["total_steps"])  # run() saves after _finish
        if not final and (validated or self.step % int(self.settings["checkpoint_every"]) == 0):
            self._save_last()

    def _log_interval(self) -> None:
        interval = self.interval
        if not interval["steps"]:
            return
        updates = max(interval["updates"], 1)
        excluded = interval["validation_seconds"] + interval["checkpoint_seconds"]
        elapsed = time.perf_counter() - interval["started"] - excluded
        record: dict[str, Any] = {
            "step": self.step,
            "phase": phase_of(self.settings, self.step - 1),
            "lr": interval["lr"],
            "steps": interval["steps"],
            "skipped": interval["skipped"],
            "skipped_total": self.stop_rules.skips,
            "grad_norm_mean": interval["grad_norm_sum"] / updates,
            "grad_norm_max": interval["grad_norm_max"],
            "loss": {name: value / updates for name, value in interval["terms"].items()},
            "counts": interval["counts"],
            "state": {
                name: self.bank.counters[name] - self.bank_counters_logged.get(name, 0)
                for name in self.bank.counters
            },
            "frames_per_second": interval["frames"] / max(elapsed, 1e-9),
            "data_wait_fraction": interval["data_seconds"] / max(elapsed, 1e-9),
            "seconds_per_step": elapsed / interval["steps"],
            "validation_seconds": interval["validation_seconds"],
            "checkpoint_seconds": interval["checkpoint_seconds"],
            "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        }
        if self.device.type == "cuda":
            record["device_peak_mib"] = torch.cuda.max_memory_allocated(self.device) / 2**20
        append_jsonl(self.run_dir / "metrics.jsonl", record)
        loss = record["loss"].get("total", float("nan"))
        self.log(
            f"step {self.step} {record['phase']} lr {record['lr']:.3g} loss {loss:.4f} "
            f"skipped {record['skipped']} {record['frames_per_second']:.1f} frames/s"
        )
        self._reset_interval()

    def _inference_payload(self, metrics: Optional[Mapping[str, Any]]) -> dict[str, Any]:
        model = self.ema.module
        return {
            "format": INFERENCE_FORMAT,
            "version": CHECKPOINT_VERSION,
            "stage": self.stage,
            "step": self.step,
            "weights": "ema",
            "ontology": model.ontology,
            "classes": list(model.classes),
            "geometry": self.geometry,
            "model_config": self.inputs.model_config,
            "model_config_sha256": config_sha256(self.inputs.model_config),
            "model_state": model.state_dict(),
            "metadata": self.model.metadata,
            "metrics": None if metrics is None else dict(metrics),
            "run_identity_sha256": settings_hash(self.identity),
            "data_sha256": self.identity["data_sha256"],
            "source": source_commit(),
        }

    def _evaluate(self, block: Mapping[str, Any]) -> dict[str, Any]:
        dataset = self.inputs.validation_sequences[0].dataset
        settings = map_settings(
            block, self.model.classes, small_box_rule(self.inputs.data_config, dataset)
        )
        return evaluate_detection_map(
            self.ema.module,
            self.validation_dataset,
            self.inputs.validation_sequences,
            settings,
            self.device,
        )

    def _source_validation(self) -> None:
        validation = self.settings["validation"]
        metrics = self._evaluate(validation)
        path = self.stage_dir / f"eval_{self.step}.pt"
        payload = self._inference_payload(metrics)
        atomic_torch_save(payload, path)
        selection = self.selection
        entry = {
            "step": self.step,
            "map": metrics["map"],
            "ap50": metrics["ap50"],
            "subset_map": metrics.get("subset_map"),
            "path": str(path.relative_to(self.run_dir)),
        }
        selection["evaluations"].append(entry)
        key = (metrics["map"] or 0.0, metrics["ap50"] or 0.0, -self.step)
        best = selection["best"]
        if best is None or key > (best["map"] or 0.0, best["ap50"] or 0.0, -best["step"]):
            atomic_torch_save({**payload, "selected": True}, self.stage_dir / "best.pt")
            selection["best"] = entry
        patience = validation["patience"]
        reference = selection["patience_reference"]
        current = metrics["map"] or 0.0
        if reference is None or current >= reference + float(patience["min_delta"]):
            selection["patience_reference"] = current
            selection["without_improvement"] = 0
        else:
            selection["patience_reference"] = max(reference, current)
            selection["without_improvement"] += 1
        selection["stopped_by_patience"] = selection["without_improvement"] >= int(
            patience["evaluations"]
        )
        investigation = validation["investigation"]
        if self.step == int(investigation["at_step"]) and (
            metrics.get("subset_map") or 0.0
        ) < float(investigation["min_subset_map_at_step"]):
            selection["investigation"].append(
                f"3-class mAP {metrics.get('subset_map')} below "
                f"{investigation['min_subset_map_at_step']} at step {self.step}"
            )
        self._write_selection()
        append_jsonl(
            self.run_dir / "validation.jsonl",
            {
                "step": self.step,
                "kind": "source_map",
                "metrics": metrics,
                "best": selection["best"],
            },
        )
        self.log(
            f"step {self.step}: source mAP {metrics['map']} AP50 {metrics['ap50']} "
            f"3-class {metrics.get('subset_map')} (best step {selection['best']['step']})"
        )

    def _validation_point(self) -> None:
        validation = self.settings["validation"]
        point = self.points.index(self.step) + 1
        path = self.stage_dir / f"point_{point}_step_{self.step}.pt"
        monitor = self._evaluate(validation["monitor"]) if self.validation_dataset else None
        atomic_torch_save(self._inference_payload({"dsec_detection_monitor": monitor}), path)
        self.selection["candidates"].append(
            {
                "point": point,
                "step": self.step,
                "path": str(path.relative_to(self.run_dir)),
                "sha256": file_sha256(path),
                "dsec_detection_monitor": None
                if monitor is None
                else {key: monitor[key] for key in ("map", "ap50", "per_class_ap")},
            }
        )
        self._write_selection()
        append_jsonl(
            self.run_dir / "validation.jsonl",
            {"step": self.step, "kind": "validation_point", "point": point, "monitor": monitor},
        )
        self.log(
            f"step {self.step}: validation point {point} saved; DSEC-Detection monitor mAP "
            f"{None if monitor is None else monitor['map']}"
        )

    def _write_selection(self) -> None:
        if self.stage == "pretrain":
            record = {
                "metric": "5-class COCO mAP@[0.5:0.95] on 1Mpx val (EMA weights)",
                "tie_breaks": ["higher AP50", "earlier step"],
                **self.selection,
            }
            write_json_atomic(self.stage_dir / SELECTION_FILE, record)
            return
        selection = self.settings["validation"]["selection"]
        sequences = self.inputs.data_config[selection["dataset"]]["splits"][selection["split"]]
        record = {
            "purpose": "DSEC-MOT validation export and BoT-SORT selection of these candidates",
            "selection": {**selection, "sequences": list(sequences)},
            "monitor": "DSEC-Detection val_monitor COCO mAP (logged only, not a selection input)",
            "weights": "ema",
            "run_identity_sha256": settings_hash(self.identity),
            **self.selection,
        }
        write_json_atomic(self.stage_dir / REQUESTS_FILE, record)

    def _finish(self) -> None:
        self.selection["finished"] = True
        if self.stage == "pretrain" and self.selection["best"] is not None:
            minimum = float(self.settings["validation"]["investigation"]["min_best_subset_map"])
            best_subset = max(
                (entry["subset_map"] or 0.0 for entry in self.selection["evaluations"]),
                default=0.0,
            )
            if best_subset < minimum:
                self.selection["investigation"].append(
                    f"best 3-class mAP {best_subset} below {minimum} at the end"
                )
        self._write_selection()

    def _save_last(self) -> None:
        payload = {
            "format": LAST_FORMAT,
            "version": CHECKPOINT_VERSION,
            "stage": self.stage,
            "step": self.step,
            "phase": phase_of(self.settings, self.step),
            "identity": self.identity,
            "initialization": self.initialization,
            "model": self.model.state_dict(),
            "model_metadata": self.model.metadata,
            "classifier": None if self.classifier is None else self.classifier.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {"next_step": self.step, **self.schedule.state_dict()},
            "bank": self.bank.state_dict(),
            "stop_rules": self.stop_rules.state_dict(),
            "selection": self.selection,
            "sampler": {"next_step": self.step, "seed": self.inputs.seed},
            "seed": self.inputs.seed,
            "rng": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state(self.device)
                if self.device.type == "cuda"
                else None,
            },
            "source": source_commit(),
        }
        started = time.perf_counter()
        path = save_last(payload, self.run_dir)
        self.saved_step = self.step
        seconds = time.perf_counter() - started
        self.interval["checkpoint_seconds"] += seconds
        self.log(f"saved {path} at step {self.step} ({seconds:.1f} s)")

    def _record_job(self, started: float, start_step: int, reason: str, code: int) -> None:
        if not self.run_dir.is_dir():
            return
        wall = time.time() - started
        gpus = 1 if self.device.type == "cuda" else 0
        append_jsonl(
            self.run_dir / "resources.jsonl",
            {
                "step": self.step,
                "start_step": start_step,
                "started_unix": started,
                "wall_seconds": wall,
                "gpus": gpus,
                "gpu_hours": gpus * wall / 3600.0,
                "micro_batches": self.micro_batches,
                "workers": self.workers,
                "exit_code": code,
                "reason": reason,
                "environment": environment_record(self.device),
            },
        )


# --------------------------------------------------------------------------- inputs


def evenly_spaced(items: Sequence[SequenceData], limit: Optional[int]) -> list[SequenceData]:
    if limit is None or int(limit) >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, int(limit)).round().astype(int)
    return [items[index] for index in sorted(set(indices.tolist()))]


def validation_block(stage: str, settings: Mapping[str, Any]) -> Mapping[str, Any]:
    return settings["validation"] if stage == "pretrain" else settings["validation"]["monitor"]


def build_inputs(args: argparse.Namespace) -> TrainingInputs:
    training_config = load_training_config(args.training_config)
    data_config = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    model_config = load_model_config(args.model_config)
    if args.smoke:
        training_config, data_config, model_config = smoke_configs(
            training_config, data_config, model_config
        )
    settings = resolve_stage(training_config, data_config, args.stage, args.set)
    block = validation_block(args.stage, settings)
    if args.smoke:
        data = synthetic_stage_data(args.stage, data_config, training_config["smoke"], args.seed)
        smoke = training_config["smoke"]["sequences"]
        validation = synthetic_sequences(
            data_config,
            block["dataset"],
            block["split"],
            int(smoke["val"]),
            int(smoke["steps"]),
            int(smoke["objects"]),
            int(smoke["seed"]) + 1,
        )
        if args.stage == "finetune":
            classifier = model_config["embedding"]["identity_classifier"]
            classifier["identities"] = len(data.identity_table)
    else:
        if args.manifest_root is None or args.cache_root is None:
            raise RuntimeConfigError("--manifest-root and --cache-root are required")
        data = stage_data(args.stage, args.manifest_root, args.cache_root, data_config, args.seed)
        validation = load_sequences(
            args.manifest_root, args.cache_root, data_config, block["dataset"], [block["split"]]
        )
    validation = evenly_spaced(validation, block.get("limit"))
    if any(item.split == "test" for item in validation):
        raise RuntimeConfigError("test sequences are never used during training")
    return TrainingInputs(
        stage=args.stage,
        settings=settings,
        training_config=training_config,
        data_config=data_config,
        model_config=model_config,
        data=data,
        validation_sequences=validation,
        seed=args.seed,
        smoke=args.smoke,
        rvt_root=args.rvt_root,
        rvt_checkpoint=args.rvt_checkpoint,
        pretrain_checkpoint=args.pretrain_checkpoint,
        investigation_note=args.investigation_note,
        overrides=list(args.set),
    )


def dry_run_report(inputs: TrainingInputs, args: argparse.Namespace) -> dict[str, Any]:
    """Resolved plan, data, initialization, and outputs; writes nothing."""

    settings = inputs.settings
    schedule = LearningRateSchedule(settings)
    total = int(settings["total_steps"])
    plan = inputs.data.plan
    run_dir = Path(args.run_dir)
    if args.resume:
        payload, path = load_last(run_dir)
        resume = {"checkpoint": str(path), "step": int(payload["step"])}
    elif (run_dir / LAST).exists():
        raise RuntimeConfigError(f"{run_dir} already holds a run; use --resume")
    else:
        resume = None
    geometry = stage_geometry(inputs)
    model, initialization = initial_model(inputs, geometry)
    first = plan.micro_batches(plan.batch(resume["step"] if resume else 0), args.micro_batches)
    if inputs.stage == "pretrain":
        every = int(settings["validation"]["every"])
        validations = sorted({*range(every, total + 1, every), total})
        selection_files = ["pretrain/eval_<step>.pt", "pretrain/best.pt", "pretrain/selection.json"]
    else:
        validations = validation_points(settings)
        selection_files = [
            "finetune/point_<k>_step_<step>.pt",
            f"finetune/{REQUESTS_FILE}",
        ]
    for sequence in [
        *(s for items in inputs.data.sequences.values() for s in items),
        *inputs.validation_sequences,
    ]:
        if hasattr(sequence.reader, "_open"):
            sequence.reader._open()
            sequence.reader.close()
    return {
        "stage": inputs.stage,
        "smoke": inputs.smoke,
        "settings": settings,
        "overrides": inputs.overrides,
        "resume": resume,
        "geometry": geometry,
        "initialization": initialization,
        "model": {
            "parameters": model.metadata["parameters"],
            "training_only": model.metadata["training_only"],
            "transfer_counts": model.metadata["transfer"]["counts"],
        },
        "data": {
            dataset: {
                "sequences": len(items),
                "steps": sum(item.steps for item in items),
                "supervised_steps": int(sum(item.supervised.sum() for item in items)),
            }
            for dataset, items in inputs.data.sequences.items()
        },
        "identities": len(inputs.data.identity_table),
        "validation_sequences": [item.sequence for item in inputs.validation_sequences],
        "batch": {
            "clips": plan.batch_size,
            "clip_steps": plan.steps,
            "micro_batches": args.micro_batches,
            "first_step": [
                [
                    f"{s.kind}:{s.dataset}/{s.sequence}@{s.start}+{s.length}"
                    f"{' reset' if s.reset else ''}"
                    for s in micro
                ]
                for micro in first
            ],
        },
        "schedule": {
            "total_steps": total,
            "phase_b_from": int(settings["phase_a_steps"]),
            "lr": {
                str(step): schedule(step)
                for step in sorted({0, int(settings["lr"]["warmup_steps"]), total // 2, total - 1})
            },
            "checkpoints": len(
                range(
                    int(settings["checkpoint_every"]), total + 1, int(settings["checkpoint_every"])
                )
            ),
            "validations": validations,
        },
        "outputs": [
            str(run_dir / name)
            for name in (
                "run.json",
                LAST,
                "last_prev.pt",
                "metrics.jsonl",
                "validation.jsonl",
                "resources.jsonl",
                *selection_files,
            )
        ],
    }


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--stage", choices=("pretrain", "finetune"), required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--manifest-root", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--rvt-root", default=None, help="RVT checkout (or RVT_ROOT)")
    parser.add_argument("--rvt-checkpoint", default=None, help="RVT-B 1Mpx checkpoint (pretrain)")
    parser.add_argument("--pretrain-checkpoint", default=None, help="pretrain/best.pt (finetune)")
    parser.add_argument("--investigation-note", default=None)
    parser.add_argument("--training-config", default=str(DEFAULT_TRAINING_CONFIG))
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--model-config", default=str(DEFAULT_MODEL_CONFIG))
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a stage setting, e.g. total_steps=1000 or validation.limit=8",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--micro-batches", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-job-steps", type=int, default=None)
    parser.add_argument("--max-wall-hours", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the resolved plan only")
    parser.add_argument(
        "--smoke", action="store_true", help="synthetic in-memory data and reduced geometry"
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        inputs = build_inputs(args)
        if args.dry_run:
            report = dry_run_report(inputs, args)
            json.dump(report, sys.stdout, indent=2, default=str)
            sys.stdout.write("\n")
            return EXIT_COMPLETE
        trainer = Trainer(
            inputs,
            args.run_dir,
            args.device,
            micro_batches=args.micro_batches,
            workers=args.workers,
            resume=args.resume,
            limits=JobLimits(args.max_job_steps, args.max_wall_hours),
            log=lambda message: print(message, flush=True),
        )
    except (
        RuntimeConfigError,
        SchemaError,
        CacheError,
        SamplingError,
        ModelConfigError,
        FileNotFoundError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return trainer.run()


if __name__ == "__main__":
    raise SystemExit(main())
