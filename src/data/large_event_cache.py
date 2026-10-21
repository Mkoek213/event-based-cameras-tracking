"""Causal Event Frame / Voxel Grid cache for the large event MOT pipeline.

Every manifest step ``k`` ends a causal window ``[t_k - window_us, t_k)``; an event at
exactly ``t_k`` belongs to the next window. The window becomes eight uint8 count
channels on the model frame (clipped to ``count_clip``):

- channels 0-1, Event Frame: per-polarity counts;
- channels 2-7, Voxel Grid: per-polarity counts in ``voxel_bins`` equal sub-windows,
  channel ``2 + bins * p + b`` with ``b = floor(bins * (t - t_k + window_us) / window_us)``.
  The bins are tied to the window bounds, not to the first and last event as in
  ``src.data.preprocessing.EventPreprocessor``; the Event Frame channels equal that
  class's ``event_frame`` output on the same (pooled) events.

1Mpx events are area-pooled 2x2 into a 640x360 frame (every event is kept); DSEC events
are already rectified by ``iter_dsec_events``. Events come from the Task 5 readers, so
cleaning, sorting, polarity mapping, and rectification are shared with the manifests,
and every cached step's event count must equal the manifest's ``step_event_count``.

One HDF5 file per sequence, ``<cache_root>/<dataset>/<sequence>.h5``, holds
``counts`` with shape ``(steps, 8, H, W)`` and one Blosc-compressed chunk per step.
Its key covers the event and rectification-map hashes, converter and representation
versions, window, bins, clipping, pooling, frame, and the step grid; a reader rejects a
file whose key differs. Files are written to a temporary name and renamed when
complete, so an interrupted run resumes by skipping complete files.

Example::

    python -m src.data.large_event_cache --manifest-root <manifest_root> \\
        --config configs/large_event_mot/data.json --cache-root <cache_root> \\
        --dataset dsec_mot --dsec-root <dsec_root> --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np

from src.data.large_event_schema import (
    DATASET_FRAMES,
    DATASET_SPLITS,
    DATASETS,
    FRAMES,
    GEN4_1MPX,
    MANIFEST_FILENAME,
    STEP_ANNOTATED,
    canonical_json,
    iter_window_events,
    load_sequence,
    read_json,
    settings_hash,
    sha256_file,
    write_json_atomic,
)

CHANNELS = 8
EVENT_FRAME = slice(0, 2)
VOXEL_GRID = slice(2, 8)
CACHE_SUFFIX = ".h5"
INDEX_FILENAME = "cache_index.json"
EVENT_BLOCK = 8_000_000
SOURCE_ROOTS = {"gen4_root": "--gen4-root", "dsec_root": "--dsec-root"}
_SHUFFLE = {"none": 0, "byte": 1, "bit": 2}


class CacheError(RuntimeError):
    """A cache file is missing, stale, incomplete, or disagrees with its manifest."""


@dataclass(frozen=True)
class Geometry:
    """Model-frame geometry of one dataset: pooling, valid frame, and padded input."""

    pool: int
    frame_width: int
    frame_height: int
    input_width: int
    input_height: int

    @property
    def count_scale(self) -> int:
        """Sensor pixels pooled into one model pixel (``s`` of the input transform)."""

        return self.pool * self.pool


def dataset_geometry(config: Mapping[str, Any], dataset: str) -> Geometry:
    entry = config["geometry"][dataset]
    geometry = Geometry(
        pool=int(entry["pool"]),
        frame_width=int(entry["frame"]["width"]),
        frame_height=int(entry["frame"]["height"]),
        input_width=int(entry["input"]["width"]),
        input_height=int(entry["input"]["height"]),
    )
    sensor = FRAMES[DATASET_FRAMES[dataset][1]]
    if (geometry.frame_width * geometry.pool, geometry.frame_height * geometry.pool) != sensor:
        raise CacheError(f"{dataset}: frame x pool does not cover the {sensor} box frame")
    if geometry.input_width < geometry.frame_width or geometry.input_height < geometry.frame_height:
        raise CacheError(f"{dataset}: model input is smaller than the valid frame")
    return geometry


def representation_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    settings = dict(config["representation"])
    if int(settings["event_frame_channels"]) != 2:
        raise CacheError("the Event Frame has exactly one channel per polarity")
    if 2 + 2 * int(settings["voxel_bins"]) != CHANNELS:
        raise CacheError(f"the cache layout needs {CHANNELS} channels")
    if not 0 < int(settings["count_clip"]) <= 255:
        raise CacheError("count_clip must fit uint8")
    settings["window_us"] = int(config["window_us"])
    return settings


# --------------------------------------------------------------------------- representation


def window_counts(
    x: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    p: np.ndarray,
    step_end_us: int,
    window_us: int,
    voxel_bins: int,
    geometry: Geometry,
    count_clip: int,
) -> np.ndarray:
    """Return the uint8 ``(8, H, W)`` counts of one causal window.

    ``x, y`` are event-frame pixels (native 1Mpx or rectified DSEC), ``p`` is 0/1.
    """

    t = np.asarray(t, np.int64)
    start = int(step_end_us) - int(window_us)
    if t.size and (int(t.min()) < start or int(t.max()) >= int(step_end_us)):
        raise CacheError("event outside its causal window")
    width, height = geometry.frame_width, geometry.frame_height
    xs = np.asarray(x, np.int64) // geometry.pool
    ys = np.asarray(y, np.int64) // geometry.pool
    p = np.asarray(p, np.int64)
    if np.any((xs < 0) | (xs >= width) | (ys < 0) | (ys >= height)):
        raise CacheError("event outside the model frame")
    if np.any((p != 0) & (p != 1)):
        raise CacheError("polarity must be 0 or 1")
    plane = width * height
    pixel = ys * width + xs
    bins = (voxel_bins * (t - start)) // int(window_us)
    index = np.concatenate([p * plane + pixel, (2 + voxel_bins * p + bins) * plane + pixel])
    counts = np.bincount(index, minlength=CHANNELS * plane)
    return np.minimum(counts, count_clip).astype(np.uint8).reshape(CHANNELS, height, width)


def normalize_counts(counts: Any, count_scale: Any, count_clip: int = 255) -> Any:
    """Network input ``log1p(c / s) / log1p(count_clip)`` for numpy or torch counts."""

    if isinstance(counts, np.ndarray):
        values = counts.astype(np.float32) / np.asarray(count_scale, np.float32)
        return np.log1p(values) / np.float32(np.log1p(count_clip))
    import torch

    scale = torch.as_tensor(count_scale, dtype=torch.float32, device=counts.device)
    return torch.log1p(counts.float() / scale) / float(np.log1p(count_clip))


# --------------------------------------------------------------------------- keys and paths


def step_grid_sha256(step_t_us: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(step_t_us, "<i8").tobytes()).hexdigest()


def cache_key(
    record: Mapping[str, Any], step_t_us: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    """Everything that determines the cached counts of one sequence."""

    representation = representation_settings(config)
    if int(record["time"]["window_us"]) != representation["window_us"]:
        raise CacheError(f"{record['sequence']}: manifest window differs from the configuration")
    rectify = record.get("rectify_map")
    return {
        "cache_format_version": int(config["cache"]["format_version"]),
        "representation": representation,
        "geometry": asdict(dataset_geometry(config, record["dataset"])),
        "dataset": record["dataset"],
        "sequence": record["sequence"],
        "converter_version": record["converter_version"],
        "events_sha256": record["events"]["sha256"],
        "events_bytes": int(record["events"]["bytes"]),
        "polarity_map": dict(record["events"]["polarity_map"]),
        "rectify_map_sha256": None if rectify is None else rectify["sha256"],
        "steps": int(np.asarray(step_t_us).size),
        "step_grid_sha256": step_grid_sha256(step_t_us),
    }


def cache_path(cache_root: str | Path, dataset: str, sequence: str) -> Path:
    return Path(cache_root) / dataset / f"{sequence}{CACHE_SUFFIX}"


def _compressor(config: Mapping[str, Any]):
    import hdf5plugin

    settings = config["cache"]["compressor"]
    return hdf5plugin.Blosc(
        cname=settings["cname"],
        clevel=int(settings["clevel"]),
        shuffle=_SHUFFLE[settings["shuffle"]],
    )


def read_cache_key(path: str | Path) -> Optional[str]:
    """Key digest of a complete cache file, or ``None`` when missing or incomplete."""

    import h5py

    try:
        with h5py.File(path, "r") as handle:
            if not bool(handle.attrs.get("complete", False)):
                return None
            return str(handle.attrs["cache_key_sha256"])
    except (OSError, KeyError):
        return None


class CacheReader:
    """Lazily opened, key-checked reader of one sequence cache.

    The HDF5 handle is opened on first use in the calling process (and reopened after a
    fork), so a reader can be created before data-loader workers start.
    """

    def __init__(self, path: str | Path, key_sha256: str, steps: int, geometry: Geometry):
        self.path = Path(path)
        self.key_sha256 = key_sha256
        self.steps = int(steps)
        self.geometry = geometry
        self._handle = None
        self._pid: Optional[int] = None

    def _open(self):
        import h5py
        import hdf5plugin  # noqa: F401  Registers the Blosc filter.

        if self._handle is not None and self._pid == os.getpid():
            return self._handle
        if not self.path.is_file():
            raise CacheError(f"{self.path}: cache file missing; build it with large_event_cache")
        handle = h5py.File(self.path, "r")
        expected = (
            self.steps,
            CHANNELS,
            self.geometry.frame_height,
            self.geometry.frame_width,
        )
        if (
            not bool(handle.attrs.get("complete", False))
            or handle.attrs.get("cache_key_sha256") != self.key_sha256
            or handle["counts"].shape != expected
        ):
            handle.close()
            raise CacheError(f"{self.path}: cache is incomplete or stale for this manifest")
        self._handle, self._pid = handle, os.getpid()
        return handle

    def read(self, start: int, count: int) -> np.ndarray:
        if start < 0 or count < 0 or start + count > self.steps:
            raise CacheError(f"{self.path}: steps [{start}, {start + count}) out of range")
        return self._open()["counts"][start : start + count]

    def close(self) -> None:
        if self._handle is not None and self._pid == os.getpid():
            self._handle.close()
        self._handle, self._pid = None, None

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_handle"], state["_pid"] = None, None
        return state


# --------------------------------------------------------------------------- building


def _event_blocks(
    record: Mapping[str, Any],
    roots: Mapping[str, str | Path],
    config: Mapping[str, Any],
    stats: dict[str, Any],
    block_events: int,
) -> Iterable[Mapping[str, np.ndarray]]:
    """Cleaned event blocks of a manifest record, from the Task 5 readers."""

    events = record["events"]
    root = roots.get(events["root"])
    if not root:
        raise CacheError(f"{record['sequence']}: needs {SOURCE_ROOTS[events['root']]}")
    path = Path(root) / events["path"]
    if path.stat().st_size != int(events["bytes"]):
        raise CacheError(f"{path}: size differs from the manifest record")
    if record["dataset"] == GEN4_1MPX:
        from src.data.converters.large_event_pretraining import dataset_settings, iter_gen4_events

        return iter_gen4_events(path, dataset_settings(config), stats, block_events)
    from src.data.converters.dsec_detection import iter_dsec_events, load_rectify_map

    reference = record["rectify_map"]
    rectify_path = Path(roots[reference["root"]]) / reference["path"]
    if sha256_file(rectify_path) != reference["sha256"]:
        raise CacheError(f"{rectify_path}: hash differs from the manifest record")
    return iter_dsec_events(
        path, load_rectify_map(rectify_path), events["polarity_map"], stats, block_events
    )


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_sequence_cache(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    roots: Mapping[str, str | Path],
    config: Mapping[str, Any],
    cache_root: str | Path,
    block_events: int = EVENT_BLOCK,
) -> dict[str, Any]:
    """Write (or reuse) the cache of one manifest sequence; return its index entry."""

    import h5py

    dataset, sequence = record["dataset"], record["sequence"]
    step_t = np.asarray(arrays["step_t_us"], np.int64)
    key = cache_key(record, step_t, config)
    digest = settings_hash(key)
    path = cache_path(cache_root, dataset, sequence)
    entry = {"dataset": dataset, "sequence": sequence, "steps": int(step_t.size)}
    if read_cache_key(path) == digest:
        return {**entry, "cache_key_sha256": digest, "bytes": path.stat().st_size, "reused": True}

    path.parent.mkdir(parents=True, exist_ok=True)
    for stale in path.parent.glob(f".{path.name}.*.tmp"):
        stale.unlink()
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    representation = key["representation"]
    geometry = dataset_geometry(config, dataset)
    window_us = representation["window_us"]
    expected_counts = np.asarray(arrays["step_event_count"], np.int64)
    stats: dict[str, Any] = {}
    try:
        with h5py.File(temporary, "w") as handle:
            counts = handle.create_dataset(
                "counts",
                shape=(step_t.size, CHANNELS, geometry.frame_height, geometry.frame_width),
                dtype=np.uint8,
                chunks=(1, CHANNELS, geometry.frame_height, geometry.frame_width),
                **_compressor(config),
            )
            handle["step_t_us"] = step_t
            handle["step_event_count"] = expected_counts
            handle["step_annotated"] = np.asarray(arrays["step_status"]) == STEP_ANNOTATED
            handle.attrs["cache_key"] = canonical_json(key)
            handle.attrs["cache_key_sha256"] = digest
            handle.attrs["complete"] = False
            windows = iter_window_events(
                _event_blocks(record, roots, config, stats, block_events),
                step_t - window_us,
                step_t,
            )
            for step, window in enumerate(windows):
                found = int(window["t"].size) if window else 0
                if found != int(expected_counts[step]):
                    raise CacheError(
                        f"{dataset}/{sequence} step {step}: {found} events, "
                        f"manifest records {int(expected_counts[step])}"
                    )
                if not found:
                    continue
                counts[step] = window_counts(
                    window["x"],
                    window["y"],
                    window["t"],
                    window["p"],
                    int(step_t[step]),
                    window_us,
                    int(representation["voxel_bins"]),
                    geometry,
                    int(representation["count_clip"]),
                )
            if stats.get("events") != record["events"]["cleaning"]["events"]:
                raise CacheError(f"{dataset}/{sequence}: cleaned event total differs from manifest")
            handle.attrs["complete"] = True
        _fsync_file(temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {**entry, "cache_key_sha256": digest, "bytes": path.stat().st_size, "reused": False}


# --------------------------------------------------------------------------- planning


def raw_bytes(steps: int, geometry: Geometry) -> int:
    return int(steps) * CHANNELS * geometry.frame_height * geometry.frame_width


def _free_bytes(path: Path) -> int:
    probe = path
    while not probe.exists():
        probe = probe.parent
    return shutil.disk_usage(probe).free


def plan_cache(
    jobs: Sequence[tuple[Mapping[str, Any], Mapping[str, np.ndarray]]],
    config: Mapping[str, Any],
    cache_root: str | Path,
) -> dict[str, Any]:
    """Count complete and pending files and project the space the pending ones need.

    The compressed fraction is measured on complete files when there are any, otherwise
    the configured conservative assumption is used.
    """

    complete_raw = complete_bytes = pending_raw = 0
    complete, pending = [], []
    for record, arrays in jobs:
        geometry = dataset_geometry(config, record["dataset"])
        steps = int(np.asarray(arrays["step_t_us"]).size)
        digest = settings_hash(cache_key(record, arrays["step_t_us"], config))
        path = cache_path(cache_root, record["dataset"], record["sequence"])
        if read_cache_key(path) == digest:
            complete.append(record["sequence"])
            complete_raw += raw_bytes(steps, geometry)
            complete_bytes += path.stat().st_size
        else:
            pending.append(record["sequence"])
            pending_raw += raw_bytes(steps, geometry)
    if complete_raw:
        fraction, source = complete_bytes / complete_raw, "measured"
    else:
        fraction, source = float(config["cache"]["assumed_compressed_fraction"]), "assumed"
    margin = int(float(config["cache"]["free_space_margin_gib"]) * 2**30)
    projected = int(np.ceil(pending_raw * fraction))
    free = _free_bytes(Path(cache_root))
    return {
        "complete": len(complete),
        "pending": len(pending),
        "complete_bytes": complete_bytes,
        "pending_raw_bytes": pending_raw,
        "compressed_fraction": fraction,
        "compressed_fraction_source": source,
        "projected_bytes": projected,
        "free_bytes": free,
        "margin_bytes": margin,
        "fits": projected + margin <= free,
    }


def check_space(plan: Mapping[str, Any]) -> None:
    if not plan["fits"]:
        gib = {key: plan[key] / 2**30 for key in ("projected_bytes", "margin_bytes", "free_bytes")}
        raise CacheError(
            f"projected cache size {gib['projected_bytes']:.1f} GiB plus margin "
            f"{gib['margin_bytes']:.1f} GiB exceeds {gib['free_bytes']:.1f} GiB free"
        )


def write_cache_index(
    cache_root: str | Path, dataset: str, entries: Sequence[Mapping[str, Any]], complete: bool
) -> Path:
    fields = ("sequence", "steps", "cache_key_sha256", "bytes")
    rows = sorted(
        ({key: entry[key] for key in fields} for entry in entries),
        key=lambda row: row["sequence"],
    )
    path = Path(cache_root) / dataset / INDEX_FILENAME
    write_json_atomic(
        path,
        {
            "dataset": dataset,
            "complete": bool(complete),
            "sequences": rows,
            "index_sha256": hashlib.sha256(canonical_json(rows).encode("utf-8")).hexdigest(),
        },
    )
    return path


def load_manifest_jobs(
    manifest_root: str | Path, dataset: str, splits: Sequence[str], limit: Optional[int]
) -> tuple[list[tuple[dict, dict]], bool]:
    """Validated ``(record, arrays)`` pairs of the selected splits and the manifest flag."""

    dataset_dir = Path(manifest_root) / dataset
    manifest = read_json(dataset_dir / MANIFEST_FILENAME)
    jobs = []
    for split in splits:
        entries = [entry for entry in manifest["sequences"] if entry["split"] == split]
        if not entries:
            raise CacheError(f"{dataset}: no manifest sequences in split {split}")
        for entry in entries if limit is None else entries[:limit]:
            jobs.append(load_sequence(dataset_dir, entry))
    return jobs, bool(manifest.get("complete"))


def _build_job(arguments: tuple) -> dict[str, Any]:
    return build_sequence_cache(*arguments)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Build the large event MOT EF/VG cache.")
    parser.add_argument("--manifest-root", required=True)
    parser.add_argument("--config", required=True, help="configs/large_event_mot/data.json")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--splits", default=None, help="comma list; default: every split")
    parser.add_argument("--gen4-root", help="1Mpx release root (gen4_1mpx)")
    parser.add_argument("--dsec-root", help="DSEC root with <split>/<seq>/events_left (DSEC)")
    parser.add_argument("--limit", type=int, default=None, help="first N sequences per split")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block-events", type=int, default=EVENT_BLOCK)
    parser.add_argument("--dry-run", action="store_true", help="plan and check space only")
    args = parser.parse_args(argv)

    config = read_json(args.config)
    splits = args.splits.split(",") if args.splits else list(DATASET_SPLITS[args.dataset])
    jobs, manifest_complete = load_manifest_jobs(
        args.manifest_root, args.dataset, splits, args.limit
    )
    plan = plan_cache(jobs, config, args.cache_root)
    print(json.dumps({"dataset": args.dataset, "splits": splits, "plan": plan}, sort_keys=True))
    check_space(plan)
    if args.dry_run:
        return 0

    roots = {"gen4_root": args.gen4_root, "dsec_root": args.dsec_root}
    os.umask(0o077)
    arguments = [
        (record, arrays, roots, config, args.cache_root, args.block_events)
        for record, arrays in jobs
    ]
    entries = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, entry in enumerate(pool.map(_build_job, arguments), 1):
            entries.append(entry)
            print(
                f"[{index}/{len(arguments)}] {entry['sequence']} steps={entry['steps']} "
                f"bytes={entry['bytes']} reused={entry['reused']}",
                flush=True,
            )
    complete = (
        manifest_complete
        and args.limit is None
        and set(splits) == set(DATASET_SPLITS[args.dataset])
    )
    print(write_cache_index(args.cache_root, args.dataset, entries, complete))
    return 0


if __name__ == "__main__":
    sys.exit(main())
