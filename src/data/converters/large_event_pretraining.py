"""Convert the Prophesee 1Mpx (Gen4) release into the large event MOT manifest.

Input layout (UZH mirror members): ``<gen4_root>/<split>/<chunk>_td.h5`` with
``events/{x,y,t,p}`` and ``<gen4_root>/<split>/<chunk>_bbox.npy`` labels. Each 60 s
chunk becomes one sequence whose state resets at its start.

Event cleaning (applied by ``iter_gen4_events``, which later stages reuse):
events with ``x >= 1280`` or ``y >= 720`` are dropped, timestamps are stable-sorted,
and polarity -1/+1 is mapped explicitly to 0/1.

Label rules: boxes are cropped to the sensor and zero-area boxes dropped; supervised
steps follow the time-based 20 Hz grid; a step with a frame-spanning box or with
fewer than ``min_window_events`` events in its causal window is unannotated; small
boxes and boxes without events in their window become ignore regions. Source classes
are kept (the pretraining head uses the native ontology). Track IDs are stored but are
never identity supervision.

Example::

    python -m src.data.converters.large_event_pretraining \\
        --gen4-root <gen4_root> --output-root <manifest_root> \\
        --config configs/large_event_mot/data.json --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np

from src.data.large_event_schema import (
    BOX_POSITIVE,
    BOX_SMALL,
    BOX_ZERO_EVENTS,
    GEN4_1MPX,
    SCHEMA_VERSION,
    STEP_ANNOTATED,
    STEP_FRAME_SPANNING,
    STEP_NO_LABEL,
    STEP_SPARSE_WINDOW,
    SchemaError,
    alignment_sample,
    box_density_ratio,
    box_event_counts,
    build_label_grid,
    concatenate_boxes,
    event_count_image,
    gen4_chunk_identity,
    iter_window_events,
    ontology_record,
    read_json,
    reusable_record,
    settings_hash,
    sha256_file,
    shifted_control_ratio,
    store_sequence,
    summarize_alignment,
    validate_output_root,
    write_dataset_manifest,
)

CONVERTER_VERSION = "gen4_1mpx/1"
LABEL_FIELDS = ("t", "x", "y", "w", "h", "class_id", "class_confidence", "track_id")
EVENT_FIELDS = ("x", "y", "t", "p")
EVENT_BLOCK = 8_000_000


def dataset_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    """Conversion settings that define the output; their hash keys resumed records."""

    return {
        "schema_version": SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "window_us": int(config["window_us"]),
        "grid": dict(config["grid"]),
        **config[GEN4_1MPX],
    }


def source_to_target(settings: Mapping[str, Any]) -> dict[int, Optional[int]]:
    return {int(key): value for key, value in settings["source_to_target"].items()}


# --------------------------------------------------------------------------- events


def map_polarity(values: np.ndarray, polarity_map: Mapping[str, int]) -> np.ndarray:
    """Map source polarity values explicitly; any unlisted value is a schema error."""

    values = np.asarray(values)
    mapped = np.full(values.shape, 255, np.uint8)
    for source, target in polarity_map.items():
        mapped[values == int(source)] = int(target)
    if np.any(mapped == 255):
        unexpected = np.unique(values[mapped == 255])[:5].tolist()
        raise SchemaError(f"unexpected polarity values {unexpected}")
    return mapped


def iter_gen4_events(
    path: str | Path,
    settings: Mapping[str, Any],
    stats: Optional[dict[str, Any]] = None,
    block_events: int = EVENT_BLOCK,
) -> Iterator[dict[str, np.ndarray]]:
    """Yield cleaned, globally stable-time-sorted event blocks of one chunk.

    Sorting is streamed: each block is merged with the not-yet-emitted tail of the
    previous ones, and only events earlier than the next raw block's minimum time are
    emitted. If a later block ever reaches behind an emitted event, the bounded-lateness
    assumption fails and the chunk is rejected instead of being silently misordered.
    """

    import h5py
    import hdf5plugin  # noqa: F401  Registers the Blosc filter.

    width, height = settings["sensor"]["width"], settings["sensor"]["height"]
    polarity_map = settings["polarity_map"]
    stats = {} if stats is None else stats
    stats.update(
        input_events=0,
        dropped_out_of_sensor=0,
        late_events=0,
        max_lateness_us=0,
        events=0,
        t_first_us=None,
        t_last_us=None,
    )

    def emit(block: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        if stats["t_first_us"] is None:
            stats["t_first_us"] = int(block["t"][0])
        stats["t_last_us"] = int(block["t"][-1])
        stats["events"] += int(block["t"].size)
        return block

    def raw_blocks(handle) -> Iterator[dict[str, np.ndarray]]:
        events = handle["events"]
        lengths = {events[name].shape[0] for name in EVENT_FIELDS}
        if len(lengths) != 1:
            raise SchemaError(f"{path}: event arrays have different lengths {sorted(lengths)}")
        total = lengths.pop()
        running_max = None
        for start in range(0, total, block_events):
            stop = min(total, start + block_events)
            x = events["x"][start:stop].astype(np.int64)
            y = events["y"][start:stop].astype(np.int64)
            t = events["t"][start:stop].astype(np.int64)
            p = events["p"][start:stop]
            if np.any(t < 0):
                raise SchemaError(f"{path}: negative event timestamps")
            stats["input_events"] += int(t.size)
            previous = np.maximum.accumulate(t)
            if running_max is not None:
                previous = np.maximum(previous, running_max)
            prior = np.concatenate([[t[0] if running_max is None else running_max], previous[:-1]])
            lateness = prior - t
            stats["late_events"] += int(np.count_nonzero(lateness > 0))
            stats["max_lateness_us"] = max(stats["max_lateness_us"], int(lateness.max(initial=0)))
            running_max = int(previous[-1])
            inside = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            stats["dropped_out_of_sensor"] += int(np.count_nonzero(~inside))
            yield {
                "x": x[inside].astype(np.uint16),
                "y": y[inside].astype(np.uint16),
                "t": t[inside],
                "p": map_polarity(p[inside], polarity_map),
            }

    def merged(carry: dict[str, np.ndarray], block: dict[str, np.ndarray]) -> dict:
        joined = {key: np.concatenate([carry[key], block[key]]) for key in block}
        order = np.argsort(joined["t"], kind="stable")
        return {key: value[order] for key, value in joined.items()}

    last_emitted = -1
    with h5py.File(path, "r") as handle:
        carry: Optional[dict[str, np.ndarray]] = None
        for block in raw_blocks(handle):
            if block["t"].size == 0:
                continue
            if carry is None:
                carry = merged({key: value[:0] for key, value in block.items()}, block)
                continue
            if int(block["t"].min()) < last_emitted:
                raise SchemaError(f"{path}: event lateness exceeds one block; cannot stable-sort")
            bound = int(block["t"].min())
            split = int(np.searchsorted(carry["t"], bound, side="left"))
            if split:
                emitted = {key: value[:split] for key, value in carry.items()}
                last_emitted = int(emitted["t"][-1])
                yield emit(emitted)
            carry = merged({key: value[split:] for key, value in carry.items()}, block)
        if carry is not None and carry["t"].size:
            yield emit(carry)
    if stats["events"] + stats["dropped_out_of_sensor"] != stats["input_events"]:
        raise SchemaError(f"{path}: event accounting mismatch after cleaning")


# --------------------------------------------------------------------------- labels


def load_gen4_labels(path: str | Path, chunk_duration_us: int, class_count: int) -> np.ndarray:
    labels = np.load(path, allow_pickle=False)
    if labels.dtype.names is None or tuple(labels.dtype.names) != LABEL_FIELDS:
        raise SchemaError(f"{path}: label fields {labels.dtype.names}, expected {LABEL_FIELDS}")
    if labels.ndim != 1:
        raise SchemaError(f"{path}: labels must be one-dimensional")
    for field in ("x", "y", "w", "h", "class_confidence"):
        if not np.all(np.isfinite(labels[field])):
            raise SchemaError(f"{path}: non-finite {field}")
    times = labels["t"].astype(np.int64)
    if labels.size and (times.min() < 0 or times.max() > chunk_duration_us):
        raise SchemaError(f"{path}: label time outside the chunk")
    if np.any(labels["class_id"] >= class_count):
        raise SchemaError(f"{path}: class_id outside the source ontology")
    keys = np.stack([times, labels["track_id"].astype(np.int64)], 1)
    if np.unique(keys, axis=0).shape[0] != keys.shape[0]:
        raise SchemaError(f"{path}: duplicate (t, track_id) rows")
    order = np.argsort(times, kind="stable")
    return labels[order]


def crop_boxes(labels: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Crop boxes to the sensor; return ``(xywh, keep)`` where keep drops zero area."""

    x0 = np.clip(labels["x"].astype(np.float64), 0, width)
    y0 = np.clip(labels["y"].astype(np.float64), 0, height)
    x1 = np.clip(labels["x"].astype(np.float64) + labels["w"], 0, width)
    y1 = np.clip(labels["y"].astype(np.float64) + labels["h"], 0, height)
    xywh = np.stack([x0, y0, x1 - x0, y1 - y0], 1)
    keep = (xywh[:, 2] > 0) & (xywh[:, 3] > 0)
    return xywh, keep


# --------------------------------------------------------------------------- conversion


def chunk_files(split: str, stem: str) -> tuple[str, str]:
    return f"{split}/{stem}_td.h5", f"{split}/{stem}_bbox.npy"


def convert_chunk(
    gen4_root: str | Path,
    split: str,
    stem: str,
    settings: Mapping[str, Any],
    dataset_dir: str | Path,
    block_events: int = EVENT_BLOCK,
) -> dict[str, Any]:
    """Convert one chunk (or reuse an identical earlier conversion) and return its record."""

    gen4_root = Path(gen4_root)
    events_rel, labels_rel = chunk_files(split, stem)
    events_path, labels_path = gen4_root / events_rel, gen4_root / labels_rel
    digest = settings_hash(settings)
    sources = {
        "labels_sha256": sha256_file(labels_path),
        "events_bytes": events_path.stat().st_size,
    }
    previous = reusable_record(dataset_dir, stem, digest, sources)
    if previous is not None:
        return previous

    session, chunk_start, chunk_end = gen4_chunk_identity(stem)
    width, height = settings["sensor"]["width"], settings["sensor"]["height"]
    mapping = source_to_target(settings)
    labels = load_gen4_labels(labels_path, chunk_end - chunk_start, len(mapping))
    dropped = {
        "zero_area_after_crop": 0,
        "off_grid": 0,
        "frame_spanning_step": 0,
        "sparse_window_step": 0,
    }

    xywh, keep = crop_boxes(labels, width, height)
    dropped["zero_area_after_crop"] = int(np.count_nonzero(~keep))
    labels, xywh = labels[keep], xywh[keep]
    times = labels["t"].astype(np.int64)
    window_us = int(settings["window_us"])
    step_t, snapped = build_label_grid(
        times, settings["grid"]["step_us"], settings["grid"]["tolerance_us"]
    )
    step_of_label = np.full(times.size, -1, np.int64)
    if step_t.size:
        position = np.searchsorted(step_t, times)
        valid = position < step_t.size
        valid[valid] &= (step_t[position[valid]] == times[valid]) & snapped[position[valid]]
        step_of_label[valid] = position[valid]
    dropped["off_grid"] = int(np.count_nonzero(step_of_label < 0))

    status = np.where(snapped, STEP_ANNOTATED, STEP_NO_LABEL).astype(np.uint8)
    spanning_fraction = float(settings["frame_spanning_fraction"])
    spanning = (xywh[:, 2] >= spanning_fraction * width) | (
        xywh[:, 3] >= spanning_fraction * height
    )
    for step in np.unique(step_of_label[spanning & (step_of_label >= 0)]):
        status[step] = STEP_FRAME_SPANNING
    small = settings["small_box"]
    diagonal = np.hypot(xywh[:, 2], xywh[:, 3])
    is_small = (diagonal < small["min_diagonal_px"]) | (
        np.minimum(xywh[:, 2], xywh[:, 3]) < small["min_side_px"]
    )

    rows_by_step: dict[int, np.ndarray] = {}
    for step in np.flatnonzero(status == STEP_ANNOTATED):
        rows_by_step[int(step)] = np.flatnonzero(step_of_label == step)
    sampled = alignment_sample(
        np.asarray(sorted(rows_by_step), np.int64), int(settings["alignment_windows"])
    )

    cleaning: dict[str, Any] = {}
    event_counts = np.zeros(step_t.size, np.int64)
    parts, ratios, controls = [], [], []
    windows = iter_window_events(
        iter_gen4_events(events_path, settings, cleaning, block_events),
        step_t - window_us,
        step_t,
    )
    for step, window in enumerate(windows):
        if not window:
            continue
        count = int(window["t"].size)
        event_counts[step] = count
        if status[step] != STEP_ANNOTATED:
            continue
        if count < int(settings["min_window_events"]):
            status[step] = STEP_SPARSE_WINDOW
            continue
        rows = rows_by_step[step]
        image = event_count_image(window["x"], window["y"], width, height)
        inside = box_event_counts(image, xywh[rows])
        reason = np.full(rows.size, BOX_POSITIVE, np.uint8)
        if settings["zero_event_boxes_ignored"]:
            reason[inside == 0] = BOX_ZERO_EVENTS
        reason[is_small[rows]] = BOX_SMALL
        positive = reason == BOX_POSITIVE
        if step in sampled and np.any(positive):
            ratio = box_density_ratio(image, xywh[rows][positive])
            control = shifted_control_ratio(image, xywh[rows][positive])
            if ratio is not None and control is not None:
                ratios.append(ratio)
                controls.append(control)
        source = labels["class_id"][rows].astype(np.int16)
        parts.append(
            {
                "box_step": np.full(rows.size, step, np.int32),
                "box_xywh": xywh[rows].astype(np.float32),
                "box_source_class": source,
                "box_target_class": np.array(
                    [
                        -1 if mapping[int(value)] is None else mapping[int(value)]
                        for value in source
                    ],
                    np.int16,
                ),
                "box_track_id": labels["track_id"][rows].astype(np.int64),
                "box_has_track_id": np.ones(rows.size, np.bool_),
                "box_ignore": ~positive,
                "box_reason": reason,
                "box_event_count": inside.astype(np.int64),
                "box_confidence": labels["class_confidence"][rows].astype(np.float32),
            }
        )
    for step in np.flatnonzero(status == STEP_FRAME_SPANNING):
        dropped["frame_spanning_step"] += int(np.count_nonzero(step_of_label == step))
    for step in np.flatnonzero(status == STEP_SPARSE_WINDOW):
        dropped["sparse_window_step"] += int(np.count_nonzero(step_of_label == step))
    if cleaning["events"] == 0:
        raise SchemaError(f"{events_path}: no events after cleaning")
    if cleaning["t_last_us"] > chunk_end - chunk_start:
        raise SchemaError(f"{events_path}: event time beyond the chunk length")

    arrays = {
        "step_t_us": step_t,
        "step_status": status,
        "step_snapped": snapped,
        "step_event_count": event_counts,
        **concatenate_boxes(parts),
    }
    record = {
        "schema_version": SCHEMA_VERSION,
        "converter_version": CONVERTER_VERSION,
        "settings_sha256": digest,
        "dataset": GEN4_1MPX,
        "family": "gen4",
        "sequence": stem,
        "recording": session,
        "split": split,
        "sources": sources,
        "events": {
            "root": "gen4_root",
            "path": events_rel,
            "format": "gen4_h5",
            "sha256": sha256_file(events_path),
            "bytes": sources["events_bytes"],
            "t_offset_us": 0,
            "chunk_start_us": chunk_start,
            "chunk_end_us": chunk_end,
            "polarity_map": dict(settings["polarity_map"]),
            "cleaning": cleaning,
        },
        "labels": {
            "root": "gen4_root",
            "path": labels_rel,
            "sha256": sources["labels_sha256"],
            "rows": int(keep.size),
        },
        "rectify_map": None,
        "frames": {
            "event_frame": "gen4_native",
            "box_frame": "gen4_native",
            "width": width,
            "height": height,
        },
        "time": {
            "unit": "us",
            "base": "chunk_relative",
            "window_us": window_us,
            "grid": "label_snap",
            **settings["grid"],
        },
        "ontology": ontology_record(GEN4_1MPX, mapping),
        "identity": {
            "supervision": False,
            "namespace": f"{GEN4_1MPX}/{stem}",
            "reason": settings["identity_reason"],
        },
        "dropped_boxes": dropped,
        "alignment": summarize_alignment(ratios, controls),
    }
    return store_sequence(dataset_dir, record, arrays)


def plan_chunks(
    gen4_root: str | Path, splits: Sequence[str], limit: Optional[int]
) -> list[tuple[str, str]]:
    """List ``(split, stem)`` pairs; every label file needs its event file and vice versa."""

    root = Path(gen4_root)
    jobs = []
    for split in splits:
        split_dir = root / split
        if not split_dir.is_dir():
            raise SchemaError(f"missing split directory {split_dir}")
        label_stems = {path.name[: -len("_bbox.npy")] for path in split_dir.glob("*_bbox.npy")}
        event_stems = {path.name[: -len("_td.h5")] for path in split_dir.glob("*_td.h5")}
        if label_stems != event_stems:
            unmatched = sorted(label_stems ^ event_stems)[:5]
            raise SchemaError(f"{split}: unmatched label/event files, e.g. {unmatched}")
        stems = sorted(label_stems)
        for stem in stems:
            gen4_chunk_identity(stem)
        jobs.extend((split, stem) for stem in (stems if limit is None else stems[:limit]))
    return jobs


def _convert_job(arguments: tuple) -> dict[str, Any]:
    return convert_chunk(*arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Convert Prophesee 1Mpx to the manifest schema.")
    parser.add_argument("--gen4-root", required=True, help="directory with <split>/ chunk files")
    parser.add_argument("--output-root", required=True, help="manifest root directory")
    parser.add_argument("--config", required=True, help="configs/large_event_mot/data.json")
    parser.add_argument("--splits", default=None, help="comma list; default: configured splits")
    parser.add_argument("--limit", type=int, default=None, help="first N chunks per split")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block-events", type=int, default=EVENT_BLOCK)
    parser.add_argument("--dry-run", action="store_true", help="plan and check only; no writes")
    args = parser.parse_args(argv)

    config = read_json(args.config)
    settings = dataset_settings(config)
    splits = args.splits.split(",") if args.splits else list(settings["splits"])
    unknown = sorted(set(splits) - set(settings["splits"]))
    if unknown:
        raise SchemaError(f"splits {unknown} are not configured for {GEN4_1MPX}")
    jobs = plan_chunks(args.gen4_root, splits, args.limit)
    counts = {split: sum(1 for job in jobs if job[0] == split) for split in splits}
    if args.limit is None:
        for split in splits:
            expected = settings["expected_chunks"][split]
            if counts[split] != expected:
                raise SchemaError(f"{split}: found {counts[split]} chunks, expected {expected}")
    plan = {"splits": splits, "chunks": counts, "limit": args.limit}
    print(json.dumps({"dataset": GEN4_1MPX, "plan": plan}, sort_keys=True), flush=True)
    if args.dry_run:
        return 0

    dataset_dir = Path(args.output_root) / GEN4_1MPX
    os.umask(0o077)
    arguments = [
        (args.gen4_root, split, stem, settings, dataset_dir, args.block_events)
        for split, stem in jobs
    ]
    records = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, record in enumerate(pool.map(_convert_job, arguments), 1):
            records.append(record)
            counts_ = record["counts"]
            print(
                f"[{index}/{len(arguments)}] {record['split']}/{record['sequence']} "
                f"steps={counts_['steps']} annotated={counts_['steps_by_status']['annotated']} "
                f"positives={counts_['positive_boxes']}",
                flush=True,
            )
    write_dataset_manifest(
        args.output_root, GEN4_1MPX, records, settings, complete=args.limit is None, plan=plan
    )
    summary = validate_output_root(args.output_root, config["recording_overlap_allowed"])
    print(json.dumps(summary, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
