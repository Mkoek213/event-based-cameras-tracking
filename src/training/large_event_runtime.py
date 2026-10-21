"""Runtime pieces of large event MOT training: schedules, EMA, phases, state, checkpoints.

Used by :mod:`src.training.large_event_detector`; configuration lives in
``configs/large_event_mot/training.json`` (per-stage fields, ``stop_rules``, ``smoke``).

- :func:`resolve_stage`: the stage settings after ``--set`` overrides, validated so that
  pretraining selects on source validation and fine-tuning only emits DSEC-MOT
  validation candidates (no test split anywhere).
- :class:`LearningRateSchedule`: linear warmup from 0, then linear or cosine decay.
- :class:`ModelEma`: YOLOX ``ModelEMA`` (``decay * (1 - exp(-updates / ramp))`` over
  every floating-point parameter and buffer).
- :func:`apply_phase`: ``requires_grad`` by name prefix; BatchNorm layers without a
  trainable parameter stay in eval mode. :func:`build_optimizer` creates one AdamW over
  every parameter trainable in any phase, so the phase switch keeps optimizer state.
- :class:`StreamStateBank`: the only owner of carried training state. One detached
  ``RecurrentState`` row per stream slot plus its cursor; a segment either starts with a
  reset or continues the slot's cursor exactly, otherwise :class:`StateLeakageError`.
- :class:`StopRules`: non-finite updates, ineffective identity sampling, and classes
  without positives over the first complete stream-epoch of their sources.
- Atomic checkpoint files (temporary file, fsync, rename) with ``last``/``last_prev``.
- Synthetic in-memory sequences for the CPU smoke mode and tests.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from src.data.large_event_cache import CHANNELS, dataset_geometry
from src.data.large_event_dataset import (
    BatchPlan,
    IdentityTable,
    LargeEventClipDataset,
    SequenceData,
    StageData,
    small_box_rule,
)
from src.data.large_event_schema import HEAD_ONTOLOGY, settings_hash
from src.models.large_event_modules import RecurrentState

REPO_ROOT = Path(__file__).resolve().parents[2]
LAST = "last.pt"
LAST_PREV = "last_prev.pt"
LAST_FORMAT = "large_event_mot_last"
INFERENCE_FORMAT = "large_event_mot_inference"
CHECKPOINT_VERSION = 1
PHASES = ("phase_a", "phase_b")
LR_SCHEDULES = ("linear", "cosine")
PRECISIONS = ("bf16", "fp32")
INITIALIZATIONS = {"pretrain": ("rvt_checkpoint", "random"), "finetune": ("pretrain_transfer",)}


class RuntimeConfigError(ValueError):
    """The resolved training settings are inconsistent."""


class StateLeakageError(RuntimeError):
    """A stream segment would continue from a state that does not belong to it."""


class StopRuleError(RuntimeError):
    """A frozen stop rule ended the run (``last`` is kept)."""


# --------------------------------------------------------------------------- configuration


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Recursive merge; mappings merge, every other override value replaces."""

    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_override(text: str) -> tuple[list[str], Any]:
    """``a.b=value`` with a JSON value (bare words are strings)."""

    if "=" not in text:
        raise RuntimeConfigError(f"override {text!r} is not key=value")
    key, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    path = [part for part in key.strip().split(".") if part]
    if not path:
        raise RuntimeConfigError(f"override {text!r} has an empty key")
    return path, value


def apply_overrides(settings: Mapping[str, Any], overrides: Sequence[str]) -> dict[str, Any]:
    """Apply ``--set`` overrides; every key must already exist in the stage settings."""

    resolved = copy.deepcopy(dict(settings))
    for text in overrides:
        path, value = parse_override(text)
        node = resolved
        for part in path[:-1]:
            if not isinstance(node.get(part), dict):
                raise RuntimeConfigError(f"override {text!r}: {part!r} is not a settings block")
            node = node[part]
        if path[-1] not in node:
            raise RuntimeConfigError(f"override {text!r}: unknown key {path[-1]!r}")
        node[path[-1]] = value
    return resolved


def smoke_configs(
    training_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Configurations of the CPU smoke mode: reduced geometry, slots, and schedules."""

    smoke = training_config["smoke"]
    training = deep_merge(
        {key: value for key, value in training_config.items() if key != "smoke"},
        {
            "stages": smoke["stages"],
            "stop_rules": smoke["stop_rules"],
            "logging": smoke["logging"],
        },
    )
    training["smoke"] = copy.deepcopy(smoke)
    return (
        training,
        deep_merge(data_config, smoke["data"]),
        deep_merge(model_config, smoke["model"]),
    )


def resolve_stage(
    training_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    stage: str,
    overrides: Sequence[str] = (),
) -> dict[str, Any]:
    """Validated settings of ``stage`` after ``overrides``."""

    if stage not in INITIALIZATIONS:
        raise RuntimeConfigError(f"unknown stage {stage!r}")
    settings = apply_overrides(training_config["stages"][stage], overrides)
    total, phase_a = int(settings["total_steps"]), int(settings["phase_a_steps"])
    if not 0 <= phase_a < total:
        raise RuntimeConfigError(f"{stage}: phase A must end before step {total}")
    if settings["init"] not in INITIALIZATIONS[stage]:
        raise RuntimeConfigError(f"{stage}: unsupported init {settings['init']!r}")
    if settings["precision"] not in PRECISIONS:
        raise RuntimeConfigError(f"{stage}: unknown precision {settings['precision']!r}")
    lr = settings["lr"]
    if lr["schedule"] not in LR_SCHEDULES or not 0 <= int(lr["warmup_steps"]) <= total:
        raise RuntimeConfigError(f"{stage}: invalid learning-rate schedule {lr}")
    if float(lr["final"]) > float(lr["peak"]) or float(lr["peak"]) <= 0:
        raise RuntimeConfigError(f"{stage}: the learning rate must decay from a positive peak")
    if set(settings["trainable"]) != set(PHASES) or not all(settings["trainable"].values()):
        raise RuntimeConfigError(f"{stage}: trainable prefixes needed for {PHASES}")
    if int(settings["checkpoint_every"]) < 1 or float(settings["grad_clip_norm"]) <= 0:
        raise RuntimeConfigError(f"{stage}: checkpoint_every and grad_clip_norm must be > 0")
    if not 0 < float(settings["ema"]["decay"]) < 1 or float(settings["ema"]["ramp_steps"]) <= 0:
        raise RuntimeConfigError(f"{stage}: invalid EMA settings")

    validation = settings["validation"]
    sampling = data_config["sampling"][stage]["splits"]
    if any(split != "train" for split in sampling.values()):
        raise RuntimeConfigError(f"{stage}: training reads only train splits, got {sampling}")
    if stage == "pretrain":
        if validation["kind"] != "source_map" or validation["split"] != "val":
            raise RuntimeConfigError("pretraining selects on the source val split only")
        if validation["dataset"] not in sampling:
            raise RuntimeConfigError("pretraining validates on its own source dataset")
        if int(validation["every"]) < 1 or validation["selection"]["metric"] != "map":
            raise RuntimeConfigError("pretraining validation needs every >= 1 and metric map")
    else:
        selection, monitor = validation["selection"], validation["monitor"]
        if validation["kind"] != "validation_points" or not 1 <= int(validation["points"]) <= total:
            raise RuntimeConfigError("fine-tuning needs between 1 and total_steps points")
        if selection["dataset"] != "dsec_mot" or selection["split"] != "val":
            raise RuntimeConfigError("fine-tuning candidates are selected on DSEC-MOT val only")
        if monitor["split"] != "val_monitor":
            raise RuntimeConfigError("the fine-tuning monitor reads the val_monitor split only")
    return settings


def phase_of(settings: Mapping[str, Any], step: int) -> str:
    """Phase of optimizer step ``step`` (0-based)."""

    return "phase_a" if step < int(settings["phase_a_steps"]) else "phase_b"


def validation_points(settings: Mapping[str, Any]) -> list[int]:
    """Fine-tuning validation steps ``round(k * total / points)``, ``k = 1 .. points``."""

    total, points = int(settings["total_steps"]), int(settings["validation"]["points"])
    steps = [int(round(k * total / points)) for k in range(1, points + 1)]
    if len(set(steps)) != len(steps) or steps[-1] != total:
        raise RuntimeConfigError(f"{points} validation points do not fit {total} steps")
    return steps


# --------------------------------------------------------------------------- schedule and EMA


class LearningRateSchedule:
    """``lr(k)`` of optimizer step ``k``: ``peak * k / warmup`` for ``k < warmup``, then a
    linear or cosine decay from ``peak`` to ``final`` at ``total_steps``."""

    def __init__(self, settings: Mapping[str, Any]):
        lr = settings["lr"]
        self.peak, self.final = float(lr["peak"]), float(lr["final"])
        self.warmup, self.total = int(lr["warmup_steps"]), int(settings["total_steps"])
        self.schedule = lr["schedule"]

    def __call__(self, step: int) -> float:
        if step < self.warmup:
            return self.peak * step / self.warmup
        progress = min(max((step - self.warmup) / max(self.total - self.warmup, 1), 0.0), 1.0)
        if self.schedule == "linear":
            return self.peak + (self.final - self.peak) * progress
        return self.final + 0.5 * (self.peak - self.final) * (1.0 + math.cos(math.pi * progress))

    def state_dict(self) -> dict[str, Any]:
        return {
            "peak": self.peak,
            "final": self.final,
            "warmup_steps": self.warmup,
            "total_steps": self.total,
            "schedule": self.schedule,
        }


class ModelEma:
    """YOLOX ``ModelEMA``: ``v <- d v + (1 - d) w`` with ``d = decay (1 - exp(-n / ramp))``."""

    def __init__(self, model: nn.Module, decay: float, ramp_steps: float, updates: int = 0):
        self.module = copy.deepcopy(model).eval()
        self.module.requires_grad_(False)
        self.decay, self.ramp, self.updates = float(decay), float(ramp_steps), int(updates)

    def decay_at(self, updates: int) -> float:
        return self.decay * (1.0 - math.exp(-updates / self.ramp))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        decay = self.decay_at(self.updates)
        source = model.state_dict()
        for key, value in self.module.state_dict().items():
            if value.dtype.is_floating_point:
                value.mul_(decay).add_(source[key].detach(), alpha=1.0 - decay)

    def state_dict(self) -> dict[str, Any]:
        return {"updates": self.updates, "module": self.module.state_dict()}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.updates = int(state["updates"])
        self.module.load_state_dict(state["module"], strict=True)


# --------------------------------------------------------------------------- parameters


def named_training_parameters(
    model: nn.Module, classifier: Optional[nn.Module]
) -> list[tuple[str, nn.Parameter]]:
    """Model parameters plus the identity classifier as ``identity_classifier.*``."""

    named = list(model.named_parameters())
    if classifier is not None:
        named += [(f"identity_classifier.{name}", p) for name, p in classifier.named_parameters()]
    return named


def _matches(name: str, prefixes: Iterable[str]) -> bool:
    return any(name.startswith(prefix) for prefix in prefixes)


def apply_phase(
    model: nn.Module,
    classifier: Optional[nn.Module],
    settings: Mapping[str, Any],
    phase: str,
) -> dict[str, int]:
    """Set ``requires_grad`` for ``phase`` and the train/eval mode of every module.

    Frozen BatchNorm layers stay in eval mode so their running statistics do not move.
    """

    prefixes = settings["trainable"][phase]
    trainable = frozen = 0
    for name, parameter in named_training_parameters(model, classifier):
        flag = _matches(name, prefixes)
        parameter.requires_grad_(flag)
        trainable += parameter.numel() if flag else 0
        frozen += 0 if flag else parameter.numel()
    model.train()
    if classifier is not None:
        classifier.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            parameter.requires_grad for parameter in module.parameters(recurse=False)
        ):
            module.eval()
    return {"trainable": trainable, "frozen": frozen}


def no_weight_decay(name: str, parameter: torch.Tensor) -> bool:
    """Norms, biases (``ndim < 2``) and LayerScale ``gamma`` are not decayed."""

    return parameter.ndim < 2 or name.endswith(".gamma")


def build_optimizer(
    model: nn.Module, classifier: Optional[nn.Module], settings: Mapping[str, Any]
) -> tuple[torch.optim.AdamW, dict[str, int]]:
    """AdamW over every parameter trainable in some phase (frozen ones get no gradient
    and are skipped by AdamW), with a weight-decay group and a no-decay group."""

    prefixes = [prefix for phase in PHASES for prefix in settings["trainable"][phase]]
    decay, plain = [], []
    for name, parameter in named_training_parameters(model, classifier):
        if _matches(name, prefixes):
            (plain if no_weight_decay(name, parameter) else decay).append(parameter)
    optimizer_settings = settings["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": float(optimizer_settings["weight_decay"])},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=0.0,
        betas=tuple(optimizer_settings["betas"]),
        eps=float(optimizer_settings["eps"]),
    )
    return optimizer, {
        "decay": sum(p.numel() for p in decay),
        "no_decay": sum(p.numel() for p in plain),
    }


# --------------------------------------------------------------------------- stream state


@dataclass
class StreamCursor:
    """Where a stream slot stands after its last segment."""

    dataset: str
    sequence: str
    next_step: int
    epoch: int
    ordinal: int
    augmentation: dict[str, Any]


class StreamStateBank:
    """Detached recurrent state and cursor of every training stream slot."""

    def __init__(self, slots: Sequence[tuple[str, int]]):
        self.slots = [tuple(slot) for slot in slots]
        self.states: list[Optional[RecurrentState]] = [None] * len(self.slots)
        self.cursors: list[Optional[StreamCursor]] = [None] * len(self.slots)
        self.counters = {"segments": 0, "resets": 0, "continued": 0, "detached": 0}

    def take(self, meta: Mapping[str, Any], zero: RecurrentState) -> RecurrentState:
        """The input state of a stream segment; ``zero`` is a one-row zero state."""

        slot = int(meta["slot"])
        if not 0 <= slot < len(self.slots) or self.slots[slot][0] != meta["dataset"]:
            raise StateLeakageError(f"stream slot {slot} does not serve {meta['dataset']}")
        self.counters["segments"] += 1
        if meta["reset"]:
            if int(meta["start"]) != 0:
                raise StateLeakageError(f"slot {slot}: reset in the middle of {meta['sequence']}")
            self.counters["resets"] += 1
            return zero
        cursor, state = self.cursors[slot], self.states[slot]
        expected = (
            None
            if cursor is None
            else (
                cursor.dataset,
                cursor.sequence,
                cursor.next_step,
                cursor.epoch,
                cursor.ordinal,
                cursor.augmentation,
            )
        )
        found = (
            meta["dataset"],
            meta["sequence"],
            int(meta["start"]),
            int(meta["epoch"]),
            int(meta["ordinal"]),
            dict(meta["augmentation"]),
        )
        if state is None or expected != found:
            raise StateLeakageError(
                f"slot {slot}: segment {found[:5]} does not continue the carried state {expected}"
            )
        self.counters["continued"] += 1
        return state

    def store(self, meta: Mapping[str, Any], state: RecurrentState) -> None:
        """Keep the detached one-row ``state`` after a segment (TBPTT boundary)."""

        slot = int(meta["slot"])
        self.states[slot] = state.detach()
        self.cursors[slot] = StreamCursor(
            dataset=meta["dataset"],
            sequence=meta["sequence"],
            next_step=int(meta["start"]) + int(meta["length"]),
            epoch=int(meta["epoch"]),
            ordinal=int(meta["ordinal"]),
            augmentation=dict(meta["augmentation"]),
        )
        self.counters["detached"] += 1

    def state_dict(self) -> dict[str, Any]:
        occupied = [state is not None for state in self.states]
        states = [state for state in self.states if state is not None]
        return {
            "slots": [list(slot) for slot in self.slots],
            "occupied": occupied,
            "state": RecurrentState.cat(states).state_dict() if states else {},
            "cursors": [None if cursor is None else asdict(cursor) for cursor in self.cursors],
            "counters": dict(self.counters),
        }

    def load_state_dict(self, state: Mapping[str, Any], device: torch.device | str) -> None:
        if [tuple(slot) for slot in state["slots"]] != self.slots:
            raise StateLeakageError("checkpoint stream slots differ from the batch plan")
        occupied = list(state["occupied"])
        carried = (
            RecurrentState.from_state_dict(state["state"]).to(device) if any(occupied) else None
        )
        row = 0
        for slot, present in enumerate(occupied):
            self.states[slot] = carried.rows([row]) if present else None
            row += int(present)
        self.cursors = [
            None if cursor is None else StreamCursor(**cursor) for cursor in state["cursors"]
        ]
        self.counters = dict(state["counters"])


# --------------------------------------------------------------------------- stop rules


class StopRules:
    """Frozen stop rules (design record Section 11.3).

    - Non-finite loss or gradient norm: the update is skipped; stop after
      ``max_consecutive`` consecutive skips or more than ``max_in_window`` skips in any
      ``window_steps``-step window.
    - Identity sampling (identity stages, phase B): per ``window_steps`` window, the
      fractions of optimizer steps with identity-valid anchors and triplet anchors must
      reach the configured minima.
    - Class coverage: a head class with zero positives once every dataset that
      supervises it has completed its first stream-epoch.
    """

    def __init__(
        self,
        rules: Mapping[str, Any],
        settings: Mapping[str, Any],
        class_names: Sequence[str],
        class_sources: Mapping[int, Sequence[str]],
        stream_slots: Mapping[str, int],
    ):
        self.rules = copy.deepcopy(dict(rules))
        self.identity_enabled = bool(settings["identity"]["enabled"])
        self.identity_start = int(settings["phase_a_steps"])
        self.class_names = list(class_names)
        self.class_sources = {int(k): sorted(v) for k, v in class_sources.items()}
        self.stream_slots = dict(stream_slots)
        self.skips = 0
        self.consecutive = 0
        self.skip_steps: deque[int] = deque()
        self.window = {"steps": 0, "identity": 0, "triplet": 0}
        self.windows_checked = 0
        datasets = sorted({d for sources in self.class_sources.values() for d in sources})
        self.positives = {dataset: [0] * len(self.class_names) for dataset in datasets}
        self.slot_epochs: dict[str, dict[int, int]] = {dataset: {} for dataset in datasets}
        self.coverage_checked: list[int] = []

    def record_update(self, step: int, skipped: bool) -> None:
        rule = self.rules["nonfinite"]
        if not skipped:
            self.consecutive = 0
            return
        self.skips += 1
        self.consecutive += 1
        self.skip_steps.append(step)
        while self.skip_steps and self.skip_steps[0] <= step - int(rule["window_steps"]):
            self.skip_steps.popleft()
        if self.consecutive >= int(rule["max_consecutive"]):
            raise StopRuleError(f"{self.consecutive} consecutive non-finite updates at step {step}")
        if len(self.skip_steps) > int(rule["max_in_window"]):
            raise StopRuleError(
                f"{len(self.skip_steps)} non-finite updates within {rule['window_steps']} steps"
            )

    def record_identity(self, step: int, identity: bool, triplet: bool) -> None:
        if not self.identity_enabled or step < self.identity_start:
            return
        rule = self.rules["identity_sampling"]
        self.window["steps"] += 1
        self.window["identity"] += int(identity)
        self.window["triplet"] += int(triplet)
        if self.window["steps"] < int(rule["window_steps"]):
            return
        steps = self.window["steps"]
        identity_fraction = self.window["identity"] / steps
        triplet_fraction = self.window["triplet"] / steps
        self.window = {"steps": 0, "identity": 0, "triplet": 0}
        self.windows_checked += 1
        if identity_fraction < float(rule["min_identity_fraction"]) or triplet_fraction < float(
            rule["min_triplet_fraction"]
        ):
            raise StopRuleError(
                f"ineffective identity sampling before step {step + 1}: identity fraction "
                f"{identity_fraction:.3f}, triplet fraction {triplet_fraction:.3f}"
            )

    def record_batch(self, metas: Sequence[Mapping[str, Any]], positives: Sequence[np.ndarray]):
        """Per-class positive counts and stream epochs of one micro-batch."""

        for meta, counts in zip(metas, positives):
            if meta["dataset"] in self.positives:
                totals = self.positives[meta["dataset"]]
                for label, count in enumerate(counts):
                    totals[label] += int(count)
            if meta["kind"] == "stream" and meta["dataset"] in self.slot_epochs:
                epochs = self.slot_epochs[meta["dataset"]]
                epochs[int(meta["slot"])] = max(
                    epochs.get(int(meta["slot"]), 0), int(meta["epoch"])
                )

    def check_coverage(self) -> None:
        if not self.rules["class_coverage"]:
            return
        finished = {
            dataset
            for dataset, epochs in self.slot_epochs.items()
            if len(epochs) == self.stream_slots.get(dataset, 0)
            and epochs
            and min(epochs.values()) >= 1
        }
        for label, sources in self.class_sources.items():
            if label in self.coverage_checked or not set(sources) <= finished:
                continue
            self.coverage_checked.append(label)
            if sum(self.positives[dataset][label] for dataset in sources) == 0:
                raise StopRuleError(
                    f"class {self.class_names[label]!r} has no positives over the first "
                    f"stream-epoch of {sources}"
                )

    def state_dict(self) -> dict[str, Any]:
        return {
            "skips": self.skips,
            "consecutive": self.consecutive,
            "skip_steps": list(self.skip_steps),
            "window": dict(self.window),
            "windows_checked": self.windows_checked,
            "positives": copy.deepcopy(self.positives),
            "slot_epochs": {
                d: {str(k): v for k, v in e.items()} for d, e in self.slot_epochs.items()
            },
            "coverage_checked": list(self.coverage_checked),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.skips = int(state["skips"])
        self.consecutive = int(state["consecutive"])
        self.skip_steps = deque(int(step) for step in state["skip_steps"])
        self.window = {key: int(value) for key, value in state["window"].items()}
        self.windows_checked = int(state["windows_checked"])
        self.positives = {d: [int(c) for c in counts] for d, counts in state["positives"].items()}
        self.slot_epochs = {
            d: {int(k): int(v) for k, v in e.items()} for d, e in state["slot_epochs"].items()
        }
        self.coverage_checked = [int(label) for label in state["coverage_checked"]]


def class_sources(sequences: Mapping[str, Sequence[SequenceData]]) -> dict[int, list[str]]:
    """Datasets whose class mask supervises each head column."""

    sources: dict[int, list[str]] = {}
    for dataset, items in sequences.items():
        for label in np.flatnonzero(items[0].class_mask):
            sources.setdefault(int(label), []).append(dataset)
    return sources


def batch_positives(batch: Mapping[str, Any], classes: int) -> list[np.ndarray]:
    """Per clip, positive boxes per head class at valid supervised steps."""

    counts = []
    for clip, targets in enumerate(batch["targets"]):
        total = np.zeros(classes, np.int64)
        for step, target in enumerate(targets):
            if batch["valid"][clip, step] and batch["supervised"][clip, step]:
                labels = target["labels"][~target["ignore"]].numpy()
                total += np.bincount(labels, minlength=classes)[:classes]
        counts.append(total)
    return counts


# --------------------------------------------------------------------------- files


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_torch_save(payload: Any, path: str | Path) -> None:
    """Write ``payload`` to a temporary file, fsync it, and rename it over ``path``."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def save_last(payload: Any, run_dir: str | Path) -> Path:
    """Write ``last.pt`` and keep the previous one as ``last_prev.pt``."""

    run_dir = Path(run_dir)
    path, previous = run_dir / LAST, run_dir / LAST_PREV
    temporary = run_dir / f".{LAST}.tmp-{os.getpid()}"
    atomic_torch_save(payload, temporary)
    if path.exists():
        os.replace(path, previous)
    os.replace(temporary, path)
    _fsync_directory(run_dir)
    return path


def load_checkpoint(path: str | Path) -> dict[str, Any]:
    """Load a checkpoint written by this runtime (tensors and plain containers only)."""

    return torch.load(path, map_location="cpu", weights_only=True)


def load_last(run_dir: str | Path) -> tuple[dict[str, Any], Path]:
    """``last.pt``, or ``last_prev.pt`` when ``last.pt`` is missing or unreadable."""

    errors = []
    for name in (LAST, LAST_PREV):
        path = Path(run_dir) / name
        if not path.is_file():
            continue
        try:
            payload = load_checkpoint(path)
        except Exception as error:  # noqa: BLE001  A torn file falls back to last_prev.
            errors.append(f"{path}: {error}")
            continue
        if payload.get("format") != LAST_FORMAT or payload.get("version") != CHECKPOINT_VERSION:
            errors.append(f"{path}: not a {LAST_FORMAT} v{CHECKPOINT_VERSION} checkpoint")
            continue
        return payload, path
    raise FileNotFoundError(f"no resumable checkpoint in {run_dir}: {errors or 'none found'}")


def load_inference_checkpoint(path: str | Path) -> dict[str, Any]:
    payload = load_checkpoint(path)
    if payload.get("format") != INFERENCE_FORMAT or payload.get("version") != CHECKPOINT_VERSION:
        raise RuntimeConfigError(f"{path}: not a {INFERENCE_FORMAT} v{CHECKPOINT_VERSION} file")
    return payload


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def append_jsonl(path: str | Path, record: Mapping[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, allow_nan=True) + "\n")
        handle.flush()


def truncate_jsonl(path: str | Path, max_step: int) -> int:
    """Drop records logged after ``max_step`` (work lost by an interruption); atomic."""

    path = Path(path)
    if not path.is_file():
        return 0
    kept, dropped = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        if int(json.loads(line).get("step", 0)) > max_step:
            dropped += 1
        else:
            kept.append(line)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text("".join(f"{line}\n" for line in kept), encoding="utf-8")
    os.replace(temporary, path)
    return dropped


# --------------------------------------------------------------------------- provenance


def source_commit(root: Path = REPO_ROOT) -> dict[str, Any]:
    def git(*args: str) -> Optional[str]:
        try:
            result = subprocess.run(
                ["git", "-C", str(root), *args], capture_output=True, text=True, check=True
            )
        except (OSError, subprocess.CalledProcessError):
            return None
        return result.stdout.strip()

    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": git("rev-parse", "HEAD"), "dirty": None if status is None else bool(status)}


def environment_record(device: torch.device) -> dict[str, Any]:
    record: dict[str, Any] = {
        "host": socket.gethostname(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device": str(device),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if device.type == "cuda":
        record["device_name"] = torch.cuda.get_device_name(device)
        record["device_memory_bytes"] = torch.cuda.get_device_properties(device).total_memory
    return record


def sequences_record(sequences: Mapping[str, Sequence[SequenceData]]) -> dict[str, Any]:
    """Sequence list with cache keys per dataset and the hash of the whole record."""

    record = {
        dataset: [
            {
                "sequence": item.sequence,
                "split": item.split,
                "steps": item.steps,
                "supervised_steps": int(item.supervised.sum()),
                "cache_key_sha256": getattr(item.reader, "key_sha256", None),
            }
            for item in items
        ]
        for dataset, items in sorted(sequences.items())
    }
    return {"sequences": record, "sha256": settings_hash(record)}


# --------------------------------------------------------------------------- synthetic data


class InMemoryCounts:
    """A cache reader over an in-memory ``(steps, 8, H, W)`` uint8 array."""

    def __init__(self, counts: np.ndarray, key: str):
        self.counts = counts
        self.key_sha256 = key

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.counts.shape[0]:
            raise ValueError(f"steps [{start}, {start + count}) out of range")
        return self.counts[start : start + count]


def synthetic_sequences(
    data_config: Mapping[str, Any],
    dataset: str,
    split: str,
    count: int,
    steps: int,
    objects: int,
    seed: int,
    classes: int = 7,
) -> list[SequenceData]:
    """Deterministic moving-box sequences with event blobs inside the boxes.

    Every fourth step is unannotated, every third annotated step carries one ignore box,
    objects come in same-class pairs (so triplets have negatives) whose labels cycle over
    the dataset's supervised head columns, and track IDs are the object number
    (identity-valid only for identity-supervised datasets).
    """

    geometry = dataset_geometry(data_config, dataset)
    width, height = geometry.frame_width, geometry.frame_height
    settings = data_config[dataset]
    head = HEAD_ONTOLOGY[dataset]
    class_mask = np.ones(classes, np.bool_)
    if head == "target":
        class_mask[list(settings.get("masked_target_classes", []))] = False
    labels_available = np.flatnonzero(class_mask)
    identity = bool(settings.get("identity_supervision", False))
    sequences = []
    for index in range(count):
        name = f"synthetic_{split}_{index:02d}"
        rng = np.random.default_rng([seed, index, len(split), len(dataset)])
        size = np.stack(
            [rng.uniform(0.15, 0.3, objects) * width, rng.uniform(0.2, 0.35, objects) * height], 1
        )
        start = rng.uniform(0, 1, (objects, 2)) * ([width, height] - size)
        velocity = rng.uniform(-1.5, 1.5, (objects, 2))
        pairs = index * objects + np.arange(objects) - np.arange(objects) % 2
        labels = labels_available[pairs % labels_available.size]
        counts = (rng.random((steps, CHANNELS, height, width)) < 0.01).astype(np.uint8)
        step_boxes, offsets = [], [0]
        supervised = np.array([step % 4 != 3 for step in range(steps)])
        for step in range(steps):
            top_left = np.clip(start + velocity * step, 0, [width, height] - size)
            xyxy = np.concatenate([top_left, top_left + size], 1)
            for box in xyxy:
                x0, y0, x1, y1 = np.round(box).astype(int)
                counts[step, :, y0:y1, x0:x1] += rng.integers(
                    1, 4, (CHANNELS, y1 - y0, x1 - x0)
                ).astype(np.uint8)
            if not supervised[step]:
                offsets.append(offsets[-1])
                continue
            ignore = np.zeros(objects, np.bool_)
            if step % 3 == 0:
                ignore[step % objects] = True
            step_boxes.append(
                {
                    "xyxy": xyxy,
                    "label": labels.astype(np.int64),
                    "ignore": ignore,
                    "track_id": np.arange(1, objects + 1, dtype=np.int64),
                    "has_track_id": np.ones(objects, np.bool_),
                }
            )
            offsets.append(offsets[-1] + objects)
        boxes = {key: np.concatenate([item[key] for item in step_boxes]) for key in step_boxes[0]}
        sequences.append(
            SequenceData(
                dataset=dataset,
                sequence=name,
                split=split,
                namespace=f"{dataset}/{name}",
                identity_supervision=identity,
                head=head,
                geometry=geometry,
                step_t_us=50_000 * (np.arange(steps, dtype=np.int64) + 1),
                supervised=supervised,
                box_offsets=np.asarray(offsets, np.int64),
                boxes=boxes,
                class_mask=class_mask,
                small_box=small_box_rule(data_config, dataset),
                reader=InMemoryCounts(counts, settings_hash({"synthetic": name, "seed": seed})),
            )
        )
    return sequences


def synthetic_stage_data(
    stage: str, data_config: Mapping[str, Any], smoke: Mapping[str, Any], seed: int
) -> StageData:
    """The smoke-mode counterpart of :func:`src.data.large_event_dataset.stage_data`."""

    settings = smoke["sequences"]
    sequences = {
        dataset: synthetic_sequences(
            data_config,
            dataset,
            "train",
            int(settings["train"]),
            int(settings["steps"]),
            int(settings["objects"]),
            int(settings["seed"]),
        )
        for dataset in data_config["sampling"][stage]["splits"]
    }
    everything = [item for items in sequences.values() for item in items]
    identity_table = IdentityTable.from_sequences(everything)
    return StageData(
        dataset=LargeEventClipDataset(everything, data_config, identity_table),
        plan=BatchPlan(stage, data_config, sequences, seed),
        identity_table=identity_table,
        sequences=sequences,
    )
