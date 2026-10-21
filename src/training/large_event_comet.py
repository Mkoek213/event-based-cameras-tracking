"""Optional Comet monitoring, isolated from the trainer's RNG and framework state.

The SDK runs in a separate process. Nonblocking writes send only selected numeric
records; a slow or failed logger cannot hold up an optimizer step. Local JSONL
files remain authoritative. There is no SDK import in the training process.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import math
import os
import select
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

START_TIMEOUT = 20.0
END_TIMEOUT = 5.0
STATE_FILE = "comet.json"
SCALARS = (
    "lr",
    "steps",
    "skipped",
    "skipped_total",
    "grad_norm_mean",
    "grad_norm_max",
    "frames_per_second",
    "data_wait_fraction",
    "seconds_per_step",
    "validation_seconds",
    "checkpoint_seconds",
    "host_peak_rss_mib",
    "device_peak_mib",
)
LOSSES = (
    "iou",
    "iou_weighted",
    "obj",
    "cls",
    "detection",
    "identity_ce",
    "identity_ce_weighted",
    "triplet",
    "triplet_weighted",
    "total",
)
COUNTS = (
    "supervised_steps",
    "positive_boxes",
    "ignore_boxes",
    "fg_anchors",
    "ignore_masked_anchors",
    "masked_class_columns",
    "identity_valid_boxes",
    "identity_valid_anchors",
    "identity_instances",
    "triplet_anchors",
    "triplet_positive_pairs",
    "triplet_negative_pairs",
    "steps_with_detection",
    "steps_with_identity_ce",
    "steps_with_triplet",
)
CLASSES = (
    "car",
    "pedestrian",
    "two_wheeler",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "train",
    "traffic_sign",
    "traffic_light",
)
PARAMETERS = (
    "stage",
    "variant",
    "seed",
    "total_steps",
    "phase_a_steps",
    "batch_size",
    "clip_steps",
    "micro_batches",
    "precision",
    "peak_lr",
    "weight_decay",
    "ema_decay",
    "class_count",
    "identity_enabled",
)


class CometConfigError(ValueError):
    """An explicitly requested monitoring mode cannot be configured."""


def numeric_fields(record: Mapping[str, Any], names, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}/{name}": value
        for name in names
        if isinstance(value := record.get(name), (int, float)) and math.isfinite(value)
    }


class CometLogger:
    """One experiment per training run directory, reused across allocations."""

    def __init__(self, run_dir: Path, parameters: Mapping[str, Any], log=print):
        self.process: Optional[subprocess.Popen] = None
        self.log = log
        self.warned = False
        self.mode = os.environ.get("LEMOT_COMET_MODE", "disabled")
        if self.mode == "disabled":
            return
        if self.mode not in {"online", "offline"}:
            raise CometConfigError("LEMOT_COMET_MODE must be disabled, online or offline")
        if self.mode == "online" and not os.environ.get("COMET_API_KEY"):
            raise CometConfigError("online monitoring needs COMET_API_KEY in the environment")
        if importlib.util.find_spec("comet_ml") is None:
            raise CometConfigError("monitoring needs comet_ml==3.57.3 in this Python environment")

        try:
            state = self._state(Path(run_dir))
            environment = {**os.environ, "COMET_DISABLE_AUTO_LOGGING": "1"}
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    __name__,
                    "--worker",
                    "--run-dir",
                    str(run_dir),
                    "--mode",
                    self.mode,
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                env=environment,
                start_new_session=True,
            )
            ready, _, _ = select.select([self.process.stdout], [], [], START_TIMEOUT)
            if not ready or self.process.stdout.readline() != b"ready\n":
                self._disable("initialization failed or timed out")
                return
            os.set_blocking(self.process.stdin.fileno(), False)
            public = {name: parameters[name] for name in PARAMETERS if name in parameters}
            self._send({"parameters": public})
            self.log(f"Comet {self.mode}: experiment {state['experiment_key']} ({STATE_FILE})")
        except CometConfigError:
            raise
        except Exception:
            self._disable("initialization failed")

    def _state(self, run_dir: Path) -> dict[str, Any]:
        destination = {
            "project_name": os.environ.get("COMET_PROJECT_NAME") or "large-event-mot",
            "workspace": os.environ.get("COMET_WORKSPACE") or None,
        }
        path = run_dir / STATE_FILE
        if path.exists():
            state = json.loads(path.read_text())
            key = state.get("experiment_key", "")
            if not isinstance(key, str) or not 32 <= len(key) <= 50 or not key.isalnum():
                raise CometConfigError("invalid experiment key in comet.json")
            if any(state.get(name) != value for name, value in destination.items()):
                raise CometConfigError("Comet project/workspace differs from comet.json")
            return state
        state = {"experiment_key": uuid.uuid4().hex, **destination}
        run_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                json.dump(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
        return state

    def _warn(self, reason: str) -> None:
        if not self.warned:
            self.warned = True
            with contextlib.suppress(Exception):
                self.log(f"Comet warning: {reason}; local records and training are unaffected")

    def _disable(self, reason: str) -> None:
        self._warn(reason)
        self._reap(kill=True)

    def _reap(self, kill: bool = False) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        with contextlib.suppress(Exception):
            process.stdin.close()
            if kill:
                process.kill()
            try:
                process.wait(timeout=1.0 if kill else END_TIMEOUT)
            except subprocess.TimeoutExpired:
                self._warn("finalization timed out; remote data may be incomplete")
                process.kill()
                process.wait(timeout=1.0)
            if process.returncode:
                self._warn("logger failed; remote data may be incomplete")
        with contextlib.suppress(Exception):
            process.stdout.close()

    def _send(self, event: Mapping[str, Any]) -> None:
        if self.process is None:
            return
        try:
            if self.process.poll() is not None:
                self._disable("logger exited")
                return
            payload = (json.dumps(event, allow_nan=False, separators=(",", ":")) + "\n").encode()
            pipe = self.process.stdin.fileno()
            # Writes up to PIPE_BUF are atomic; never send a truncated JSON record.
            if len(payload) > os.fpathconf(pipe, "PC_PIPE_BUF"):
                self._disable("record exceeds logger buffer")
            elif os.write(pipe, payload) != len(payload):
                self._disable("incomplete logger write")
        except Exception:
            self._disable("logger unavailable or buffer full")

    def training(self, record: Mapping[str, Any]) -> None:
        if self.process is None:
            return
        metrics = numeric_fields(record, SCALARS, "train")
        metrics.update(numeric_fields(record.get("loss", {}), LOSSES, "train/loss"))
        metrics.update(numeric_fields(record.get("counts", {}), COUNTS, "train/counts"))
        metrics.update(
            numeric_fields(
                record.get("state", {}),
                ("segments", "resets", "continued", "detached"),
                "train/state",
            )
        )
        metrics["train/phase_b"] = int(record.get("phase") == "phase_b")
        self._send({"metrics": metrics, "step": record["step"]})

    def validation(self, record: Mapping[str, Any], selection: Mapping[str, Any]) -> None:
        if self.process is None:
            return
        source = record["kind"] == "source_map"
        prefix = "val/source" if source else "val/dsec_detection_monitor"
        values = record.get("metrics" if source else "monitor") or {}
        metrics = numeric_fields(
            values, ("map", "ap50", "ap75", "subset_map", "subset_ap50"), prefix
        )
        metrics.update(numeric_fields(values.get("per_class_ap", {}), CLASSES, f"{prefix}/ap"))
        if source:
            best = selection.get("best") or {}
            metrics.update(
                numeric_fields(
                    {
                        "best_step": best.get("step"),
                        "best_map": best.get("map"),
                        "without_improvement": selection.get("without_improvement"),
                    },
                    ("best_step", "best_map", "without_improvement"),
                    prefix,
                )
            )
        if metrics:
            self._send({"metrics": metrics, "step": record["step"]})

    def close(self) -> None:
        self._reap()


def worker(run_dir: Path, mode: str) -> int:
    """Run SDK calls outside the training process; stdout carries readiness only."""
    protocol = sys.stdout
    # Bound SDK HTTP/shutdown work too; the parent enforces hard time limits.
    os.environ.update(
        {
            "COMET_DISABLE_AUTO_LOGGING": "1",
            "COMET_TIMEOUT_HTTP": "5",
            "COMET_TIMEOUT_CLEANING": "3",
            "COMET_LOGGING_CONSOLE": "ERROR",
        }
    )
    experiment = None
    try:
        with contextlib.redirect_stdout(sys.stderr):
            import comet_ml

            state = json.loads((run_dir / STATE_FILE).read_text())
            config = comet_ml.ExperimentConfig(
                log_code=False,
                log_graph=False,
                parse_args=False,
                display_summary_level=0,
                log_git_metadata=False,
                log_git_patch=False,
                log_env_details=False,
                log_env_gpu=False,
                log_env_host=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                auto_output_logging=False,
                auto_param_logging=False,
                auto_metric_logging=False,
                auto_log_co2=False,
                auto_histogram_gradient_logging=False,
                auto_histogram_activation_logging=False,
                auto_histogram_tensorboard_logging=False,
                auto_histogram_weight_logging=False,
                offline_directory=str(run_dir / "comet-offline"),
            )
            experiment = comet_ml.start(
                api_key=os.environ.get("COMET_API_KEY") if mode == "online" else None,
                experiment_key=state["experiment_key"],
                project_name=state["project_name"],
                workspace=state["workspace"],
                mode="get_or_create",
                online=mode == "online",
                experiment_config=config,
            )
            if experiment is None or experiment.get_key() != state["experiment_key"]:
                return 1
            protocol.write("ready\n")
            protocol.flush()
            for line in sys.stdin:
                event = json.loads(line)
                if "parameters" in event:
                    parameters = event["parameters"]
                    experiment.log_parameters(parameters)
                    experiment.set_name(
                        f"{parameters.get('stage', 'training')}-{parameters.get('variant', 'main')}"
                    )
                else:
                    # Repeated name/step pairs replace the same point after rollback.
                    experiment.log_metrics(event["metrics"], step=event["step"])
            experiment.end()
        return 0
    except Exception:
        # SDK exceptions can contain credentials or host details: never forward them.
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", action="store_true", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("online", "offline"), required=True)
    args = parser.parse_args()
    return worker(args.run_dir, args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
