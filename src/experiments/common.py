"""Shared utilities for experiment orchestration.

This module keeps runner scripts small and makes benchmark orchestration reusable
across representation sweeps, EROS experiments and car-only experiments.
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TypeVar

REPO_ROOT = Path(__file__).resolve().parents[2]
T = TypeVar("T")


@dataclass(frozen=True)
class VariantSpec:
    """Input representation and its fusion mode."""

    representation: str
    fusion_mode: str = "single"

    @property
    def label(self) -> str:
        return variant_label(self.representation, self.fusion_mode)

    def checkpoint_name(self, num_bins: int, width: int) -> str:
        name = f"{self.representation}_bins{num_bins}_w{width}"
        return name if self.fusion_mode == "single" else f"{name}_{self.fusion_mode}"


@dataclass(frozen=True)
class EvalTarget:
    """Dataset split and sequence used for validation/test evaluation."""

    split: str
    sequence: str
    label: str


DEFAULT_EVAL_TARGETS = (
    EvalTarget("train", "zurich_city_01_d", "val"),
    EvalTarget("test", "interlaken_00_d", "test"),
)


class CommandRunner:
    """Run shell commands from the repository root and mirror output to a log."""

    def __init__(self, dry_run: bool = False, repo_root: Path = REPO_ROOT) -> None:
        self.dry_run = dry_run
        self.repo_root = repo_root

    def run(self, command: list[str], log_path: Path) -> int:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        printable = " ".join(command)
        print(f"\n$ {printable}")
        print(f"log: {log_path}")
        if self.dry_run:
            return 0

        started = time.time()
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(f"\n$ {printable}\n")
            process = subprocess.Popen(
                command,
                cwd=self.repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_file.write(line)
            code = process.wait()
            elapsed_h = (time.time() - started) / 3600
            log_file.write(f"\nexit_code={code} elapsed_h={elapsed_h:.3f}\n")
            return code


def threshold_label(value: float) -> str:
    """Format a score threshold as a compact label, e.g. 0.95 -> 095."""

    return f"{int(round(value * 100)):03d}"


def window_label(time_window_us: int) -> str:
    """Format an event accumulation window label, e.g. 50000 -> win50ms."""

    return f"win{time_window_us // 1000}ms"


def sweep_label(num_bins: int, time_window_us: int) -> str:
    """Format a representation-parameter label."""

    return f"bins{num_bins}_{window_label(time_window_us)}"


def variant_label(representation: str, fusion_mode: str) -> str:
    """Format representation and fusion mode into a stable run label."""

    return representation if fusion_mode == "single" else f"{representation}_{fusion_mode}"


def unique_specs(specs: list[T]) -> list[T]:
    """Keep specs ordered while removing duplicates."""

    seen: set[object] = set()
    result: list[object] = []
    for spec in specs:
        if spec in seen:
            continue
        seen.add(spec)
        result.append(spec)
    return result


def require_checkpoint(checkpoint: Path, dry_run: bool) -> bool:
    """Return false and print a clear error when a required checkpoint is missing."""

    if checkpoint.exists() or dry_run:
        return True
    print(f"Missing checkpoint, cannot evaluate: {checkpoint}")
    return False


def simple_detector_train_command(
    *,
    python: str,
    root: Path,
    representation: str,
    fusion_mode: str,
    num_bins: int,
    time_window_us: int,
    epochs: int,
    batch_size: int,
    num_workers: int,
    width: int,
    device: str,
    output_dir: Path,
    eros_cache_root: Path | None = None,
    class_ids: list[int] | None = None,
    num_classes: int | None = None,
) -> list[str]:
    """Build the training command for ``SimpleDenseDetector``."""

    command = [
        python,
        "-m",
        "src.training.simple_detector",
        "--root",
        str(root),
        "--representation",
        representation,
        "--fusion-mode",
        fusion_mode,
        "--num-bins",
        str(num_bins),
        "--time-window-us",
        str(time_window_us),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(num_workers),
        "--model-width",
        str(width),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
    ]
    if eros_cache_root is not None:
        command.extend(["--eros-cache-root", str(eros_cache_root)])
    if class_ids is not None:
        command.append("--class-ids")
        command.extend(str(class_id) for class_id in class_ids)
    if num_classes is not None:
        command.extend(["--num-classes", str(num_classes)])
    return command


def simple_detector_eval_command(
    *,
    python: str,
    checkpoint: Path,
    root: Path,
    target: EvalTarget,
    threshold: float,
    device: str,
    max_detections: int,
    output_root: Path,
    run_name: str,
    eros_cache_root: Path | None = None,
    classes_to_eval: list[str] | None = None,
) -> list[str]:
    """Build the TrackEval command for a ``SimpleDenseDetector`` checkpoint."""

    command = [
        python,
        "-m",
        "src.evaluation.simple_detector_trackeval_cli",
        "--checkpoint",
        str(checkpoint),
        "--root",
        str(root),
        "--split",
        target.split,
        "--sequences",
        target.sequence,
        "--device",
        device,
        "--score-threshold",
        str(threshold),
        "--max-detections",
        str(max_detections),
        "--output-root",
        str(output_root),
        "--run-name",
        run_name,
    ]
    if eros_cache_root is not None:
        command.extend(["--eros-cache-root", str(eros_cache_root)])
    if classes_to_eval:
        command.append("--classes-to-eval")
        command.extend(classes_to_eval)
    return command
