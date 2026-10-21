"""Causal detection and embedding export of the large event MOT model for BoT-SORT.

One export per (inference checkpoint, sequence): the EMA weights run in fp32 in eval mode,
the recurrent state starts at zero at the first step of the sequence and is carried
through every step, and every step of the DSEC-MOT image-timestamp grid is exported,
including frame 0 and frames without detections. Decoding follows
``configs/large_event_mot/evaluation.json`` (score floor 0.05, class-aware NMS 0.5, at
most 300 boxes per frame by descending score), so the rows of a frame are sorted by score
and every higher score threshold selects a prefix of them. Each detection carries the
128-d unit-norm descriptor of the anchor that produced it.

Files per sequence in the export directory:

- ``<sequence>.npz``: ``rows`` ``(N, 6)`` float32 ``x1, y1, x2, y2, score, class`` in the
  original (rectified, unpooled) pixel frame; ``embeddings`` ``(N, 128)`` float32;
  ``anchors`` ``(N,)``; ``offsets`` ``(F + 1,)``; ``timestamps`` ``(F,)``;
  ``cap_reached`` ``(F,)``.
- ``<sequence>.json``: the export identity (checkpoint SHA-256, preprocessing cache key,
  split, class mapping, decoding, schema version, RVT commit) and its hash, the archive
  SHA-256, counts, and timing/memory measurements.

An existing export is reused only when its identity and archive hash match; any other
existing file is an error, so a cache is never silently mixed across settings.

The test split is never exported from this command line; the frozen final evaluation in
``src.evaluation.large_event_mot`` calls :func:`export_sequences` with ``allow_test``
after verifying its freeze manifest.

Example::

    python -m src.evaluation.large_event_export --checkpoint <run>/finetune/point_1_step_1000.pt \\
        --manifest-root <manifests> --cache-root <cache> --rvt-root <rvt> \\
        --output-dir <results>/exports/point_1_step_1000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
import torch

from src.data.large_event_cache import step_grid_sha256
from src.data.large_event_dataset import (
    LargeEventClipDataset,
    SequenceData,
    collate_clips,
    load_sequences,
    network_input,
    sequence_clips,
)
from src.data.large_event_schema import read_json, settings_hash, write_json_atomic
from src.models.large_event_detector import LargeEventDetector, build_large_event_detector
from src.training.large_event_runtime import file_sha256, load_inference_checkpoint

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVALUATION_CONFIG = REPO_ROOT / "configs/large_event_mot/evaluation.json"
DEFAULT_DATA_CONFIG = REPO_ROOT / "configs/large_event_mot/data.json"
EXPORT_FORMAT = "large_event_mot_detections"
ARRAYS = ("rows", "embeddings", "anchors", "offsets", "timestamps", "cap_reached")


class ExportError(ValueError):
    """The export request is inconsistent with the checkpoint, data, or an existing file."""


def load_evaluation_config(path: str | Path = DEFAULT_EVALUATION_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    export = config["export"]
    if export["weights"] != "ema" or export["precision"] != "fp32":
        raise ExportError("exports use the EMA weights in fp32")
    if not 0.0 < float(export["score_floor"]) < 1.0 or int(export["max_detections"]) < 100:
        raise ExportError("export floor must lie in (0, 1) and the frame cap must be >= 100")
    return config


# --------------------------------------------------------------------------- model


@dataclass
class LoadedCheckpoint:
    model: LargeEventDetector
    payload: dict[str, Any]
    path: Path
    sha256: str


def load_detector(
    path: str | Path,
    dataset: str,
    device: torch.device | str,
    rvt_root: Optional[str | Path] = None,
    expected_sha256: Optional[str] = None,
) -> LoadedCheckpoint:
    """Seven-class inference model from a fine-tuning checkpoint, in eval mode on ``device``."""

    path = Path(path)
    sha256 = file_sha256(path)
    if expected_sha256 is not None and sha256 != expected_sha256:
        raise ExportError(f"{path}: SHA-256 {sha256} differs from the recorded {expected_sha256}")
    payload = load_inference_checkpoint(path)
    if payload["stage"] != "finetune" or payload["ontology"] != "target":
        raise ExportError(f"{path}: not a fine-tuning (seven-class target head) checkpoint")
    if payload["weights"] != "ema":
        raise ExportError(f"{path}: exports use EMA weights, found {payload['weights']!r}")
    if payload["geometry"] != dataset:
        raise ExportError(f"{path}: trained on geometry {payload['geometry']!r}, not {dataset!r}")
    model = build_large_event_detector(
        payload["model_config"],
        ontology="target",
        geometry=payload["geometry"],
        init="random",
        rvt_root=rvt_root,
    )
    model.load_state_dict(payload["model_state"], strict=True)
    model.to(device).eval()
    return LoadedCheckpoint(model=model, payload=payload, path=path, sha256=sha256)


# --------------------------------------------------------------------------- identity


def export_identity(
    loaded: LoadedCheckpoint,
    sequence: SequenceData,
    data_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Everything that determines the exported rows of ``sequence``."""

    export = evaluation_config["export"]
    payload = loaded.payload
    return {
        "format": EXPORT_FORMAT,
        "schema_version": int(export["schema_version"]),
        "checkpoint_sha256": loaded.sha256,
        "checkpoint_stage": payload["stage"],
        "checkpoint_step": int(payload["step"]),
        "run_identity_sha256": payload["run_identity_sha256"],
        "model_config_sha256": payload["model_config_sha256"],
        "rvt_commit": payload["model_config"]["upstream"]["commit"],
        "weights": export["weights"],
        "precision": export["precision"],
        "dataset": sequence.dataset,
        "split": sequence.split,
        "sequence": sequence.sequence,
        "preprocessing_cache_key": sequence.reader.key_sha256,
        "step_grid_sha256": step_grid_sha256(sequence.step_t_us),
        "representation": dict(data_config["representation"]),
        "geometry": dict(data_config["geometry"][sequence.dataset]),
        "classes": list(payload["classes"]),
        "class_mask": [bool(value) for value in sequence.class_mask],
        "decoding": {
            "score_floor": float(export["score_floor"]),
            "nms_iou": float(export["nms_iou"]),
            "max_detections": int(export["max_detections"]),
            "class_aware_nms": True,
            "order": "descending score per frame",
        },
        "coordinates": "original rectified pixels (model frame x pool), xyxy",
    }


SEQUENCE_FIELDS = ("split", "sequence", "preprocessing_cache_key", "step_grid_sha256")


def shared_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """The identity without its sequence-specific fields (equal across one export setup)."""

    return {key: value for key, value in identity.items() if key not in SEQUENCE_FIELDS}


# --------------------------------------------------------------------------- streaming


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def stream_sequence(
    model: LargeEventDetector,
    dataset: LargeEventClipDataset,
    sequence: SequenceData,
    export: Mapping[str, Any],
    device: torch.device | str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Run ``sequence`` causally from a zero state and decode every step."""

    device = torch.device(device)
    pool = int(sequence.geometry.pool)
    rows, embeddings, anchors, offsets, cap = [], [], [], [0], []
    timestamps: list[int] = []
    latencies: list[float] = []
    state = None
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad(), torch.autocast(device_type=device.type, enabled=False):
        for spec in sequence_clips(sequence, int(export["clip_steps"])):
            batch = collate_clips([dataset[spec]])
            event_frame, voxel_grid = network_input(batch, device)
            if state is None or bool(batch["reset"][0]):
                if spec.start != 0 or not bool(batch["reset"][0]):
                    raise ExportError(f"{sequence.sequence}: reset outside the sequence start")
                state = model.initial_state(1, device)
            for step in range(spec.length):
                _synchronize(device)
                tick = time.perf_counter()
                output = model(event_frame[:, step], voxel_grid[:, step], state=state)
                state = output.state
                (found,) = model.decode(
                    output,
                    score_floor=float(export["score_floor"]),
                    nms_iou=float(export["nms_iou"]),
                    max_detections=int(export["max_detections"]),
                )
                boxes = found.boxes.float().cpu().numpy() * pool
                scores = found.scores.float().cpu().numpy()
                labels = found.labels.cpu().numpy()
                vectors = found.embeddings.float().cpu().numpy()
                _synchronize(device)
                latencies.append(time.perf_counter() - tick)
                if vectors.shape[0] != boxes.shape[0]:
                    raise ExportError(f"{sequence.sequence}: embedding rows do not match boxes")
                if np.any(np.diff(scores) > 0):
                    raise ExportError(f"{sequence.sequence}: decoded scores are not descending")
                rows.append(
                    np.concatenate(
                        [boxes, scores[:, None], labels[:, None].astype(np.float32)], axis=1
                    ).astype(np.float32)
                )
                embeddings.append(vectors.astype(np.float32))
                anchors.append(found.anchors.cpu().numpy().astype(np.int64))
                offsets.append(offsets[-1] + boxes.shape[0])
                cap.append(bool(found.cap_reached))
                timestamps.append(int(batch["step_t_us"][0, step]))
    elapsed = time.perf_counter() - started
    width = model.embedding_head.dim
    arrays = {
        "rows": np.concatenate(rows).reshape(-1, 6) if rows else np.empty((0, 6), np.float32),
        "embeddings": np.concatenate(embeddings).reshape(-1, width)
        if embeddings
        else np.empty((0, width), np.float32),
        "anchors": np.concatenate(anchors) if anchors else np.empty(0, np.int64),
        "offsets": np.asarray(offsets, np.int64),
        "timestamps": np.asarray(timestamps, np.int64),
        "cap_reached": np.asarray(cap, np.bool_),
    }
    if not np.array_equal(arrays["timestamps"], sequence.step_t_us):
        raise ExportError(f"{sequence.sequence}: exported steps differ from the step grid")
    milliseconds = sorted(1000.0 * value for value in latencies)
    timing = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_threads": torch.get_num_threads(),
        "frames": len(timestamps),
        "elapsed_s": elapsed,
        "frames_per_second": len(timestamps) / max(elapsed, 1e-9),
        "latency_ms": {
            "scope": "model step and decoding, excluding cache reading",
            "mean": statistics.fmean(milliseconds) if milliseconds else None,
            "median": statistics.median(milliseconds) if milliseconds else None,
            "p95": milliseconds[int(0.95 * (len(milliseconds) - 1))] if milliseconds else None,
        },
        "host_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    }
    if device.type == "cuda":
        timing["device_peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return arrays, timing


# --------------------------------------------------------------------------- files


def export_paths(directory: str | Path, sequence: str) -> tuple[Path, Path]:
    directory = Path(directory)
    return directory / f"{sequence}.npz", directory / f"{sequence}.json"


def _write_archive(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with open(temporary, "wb") as handle:
        np.savez_compressed(handle, **{name: arrays[name] for name in ARRAYS})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_export(
    directory: str | Path, sequence: str, identity: Optional[Mapping[str, Any]] = None
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Arrays and metadata of an export, verified against its archive hash (and identity)."""

    archive, metadata_path = export_paths(directory, sequence)
    if not archive.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"no export of {sequence} in {directory}")
    metadata = read_json(metadata_path)
    if metadata.get("format") != EXPORT_FORMAT:
        raise ExportError(f"{metadata_path}: not a {EXPORT_FORMAT} file")
    if settings_hash(metadata["identity"]) != metadata["identity_sha256"]:
        raise ExportError(f"{metadata_path}: identity hash mismatch")
    if identity is not None and metadata["identity"] != dict(identity):
        raise ExportError(f"{metadata_path}: export identity differs from the request")
    if file_sha256(archive) != metadata["sha256"]:
        raise ExportError(f"{archive}: archive hash mismatch (torn or modified file)")
    with np.load(archive, allow_pickle=False) as data:
        arrays = {name: data[name] for name in ARRAYS}
    validate_arrays(arrays, sequence)
    return arrays, metadata


def validate_arrays(arrays: Mapping[str, np.ndarray], sequence: str) -> None:
    offsets = arrays["offsets"]
    frames = arrays["timestamps"].size
    if offsets.size != frames + 1 or offsets[0] != 0 or np.any(np.diff(offsets) < 0):
        raise ExportError(f"{sequence}: malformed frame offsets")
    if offsets[-1] != arrays["rows"].shape[0] or arrays["embeddings"].shape[0] != offsets[-1]:
        raise ExportError(f"{sequence}: row, embedding, and offset counts disagree")
    if arrays["cap_reached"].size != frames or arrays["anchors"].size != offsets[-1]:
        raise ExportError(f"{sequence}: per-frame or per-row array lengths disagree")


def export_sequences(
    loaded: LoadedCheckpoint,
    sequences: Sequence[SequenceData],
    data_config: Mapping[str, Any],
    evaluation_config: Mapping[str, Any],
    output_dir: str | Path,
    device: torch.device | str,
    allow_test: bool = False,
    log=None,
) -> dict[str, dict[str, Any]]:
    """Export (or verify and reuse) every sequence; returns the metadata per sequence."""

    if not sequences:
        raise ExportError("no sequences to export")
    log = log or (lambda message: print(message, flush=True))
    results = {}
    dataset = LargeEventClipDataset(sequences, data_config)
    for sequence in sequences:
        if sequence.split == "test" and not allow_test:
            raise ExportError(
                f"{sequence.sequence}: test sequences are exported only by the frozen final "
                "evaluation"
            )
        if sequence.head != "target" or sequence.dataset != loaded.payload["geometry"]:
            raise ExportError(f"{sequence.sequence}: not a target-head sequence of the model")
        identity = export_identity(loaded, sequence, data_config, evaluation_config)
        archive, metadata_path = export_paths(output_dir, sequence.sequence)
        if archive.exists() or metadata_path.exists():
            _, metadata = load_export(output_dir, sequence.sequence, identity)
            log(f"export {sequence.sequence}: reusing verified {archive}")
            results[sequence.sequence] = metadata
            continue
        log(f"export {sequence.sequence}: {sequence.steps} steps on {device}")
        arrays, timing = stream_sequence(
            loaded.model, dataset, sequence, evaluation_config["export"], device
        )
        validate_arrays(arrays, sequence.sequence)
        _write_archive(archive, arrays)
        metadata = {
            "format": EXPORT_FORMAT,
            "identity": identity,
            "identity_sha256": settings_hash(identity),
            "sha256": file_sha256(archive),
            "checkpoint": str(loaded.path),
            "frames": int(arrays["timestamps"].size),
            "detections": int(arrays["rows"].shape[0]),
            "frames_without_detections": int(np.count_nonzero(np.diff(arrays["offsets"]) == 0)),
            "frames_at_cap": int(np.count_nonzero(arrays["cap_reached"])),
            "timing": timing,
            "parameters": loaded.payload["metadata"].get("parameters"),
        }
        write_json_atomic(metadata_path, metadata)
        log(
            f"export {sequence.sequence}: {metadata['detections']} detections, "
            f"{timing['frames_per_second']:.2f} frames/s"
        )
        results[sequence.sequence] = metadata
    return results


# --------------------------------------------------------------------------- CLI


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", required=True, help="fine-tuning inference checkpoint")
    parser.add_argument("--checkpoint-sha256", default=None, help="expected SHA-256")
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="val", choices=["val"], help="test is frozen-only")
    parser.add_argument("--sequences", nargs="*", default=None, help="subset of the split")
    parser.add_argument("--rvt-root", default=None, help="pinned RVT checkout (or RVT_ROOT)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data-config", default=str(DEFAULT_DATA_CONFIG))
    parser.add_argument("--evaluation-config", default=str(DEFAULT_EVALUATION_CONFIG))
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    evaluation_config = load_evaluation_config(args.evaluation_config)
    data_config = json.loads(Path(args.data_config).read_text(encoding="utf-8"))
    dataset = evaluation_config["dataset"]
    sequences = load_sequences(
        args.manifest_root, args.cache_root, data_config, dataset, [args.split]
    )
    if args.sequences:
        missing = set(args.sequences) - {item.sequence for item in sequences}
        if missing:
            raise ExportError(f"not in {dataset} {args.split}: {sorted(missing)}")
        sequences = [item for item in sequences if item.sequence in set(args.sequences)]
    loaded = load_detector(
        args.checkpoint, dataset, args.device, args.rvt_root, args.checkpoint_sha256
    )
    export_sequences(
        loaded, sequences, data_config, evaluation_config, args.output_dir, args.device
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
