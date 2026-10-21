"""Ordered EF/VG clips, samplers, and augmentation for the large event MOT pipeline.

Data flow: Task 5 manifests (steps, boxes, ontology, identity policy) plus the
``large_event_cache`` counts give one :class:`SequenceData` per sequence. A sampler
turns an optimizer step into :class:`ClipSpec` objects, and
:class:`LargeEventClipDataset` turns each spec into a clip of ``T`` steps.

Clip contents (see :meth:`LargeEventClipDataset.__getitem__`):

- ``event_frame`` (T, 2, H, W) and ``voxel_grid`` (T, 6, H, W): uint8 counts on the
  padded model input; :func:`network_input` applies the fixed ``log1p`` transform.
- ``valid``: real steps; padded stream-tail steps are False and carry no input.
- ``supervised``: valid steps whose manifest status is ``annotated``; every other step
  advances recurrent state only and is never background.
- ``reset``: the recurrent state must be zero before the first step.
- ``targets``: per step, boxes (xyxy, model frame), head-ontology labels, ignore flags,
  track IDs, and identity indices that are valid only for identity-supervised sources.
- ``class_mask``: head columns that this dataset supervises.

Sampling (design record Section 9): random clips (BPTT) start from a zero state at a
uniformly drawn start whose ``T`` steps contain a supervised step; streams (TBPTT)
walk through whole sequences in ``T``-step segments with tail padding and reset flags.
Every random draw comes from a counter-based generator keyed by
``(seed, stage, slot, step)``, so sampling is a pure function of the optimizer step and
resuming needs no generator state. Clips never cross a sequence boundary, and no
mosaic, mixup, or cross-sequence mixing is applied.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data.large_event_cache import (
    EVENT_FRAME,
    VOXEL_GRID,
    CacheReader,
    Geometry,
    cache_key,
    cache_path,
    dataset_geometry,
    normalize_counts,
)
from src.data.large_event_schema import (
    GEN4_1MPX,
    MANIFEST_FILENAME,
    STEP_ANNOTATED,
    TARGET_CLASSES,
    SchemaError,
    load_sequence,
    read_json,
    settings_hash,
)

HEAD_CLASSES = len(TARGET_CLASSES)
STAGES = ("pretrain", "finetune")
CLIP_KINDS = ("random", "stream", "eval")


class SamplingError(ValueError):
    """The requested sampling cannot be served by the loaded sequences."""


# --------------------------------------------------------------------------- generators


def counter_rng(seed: int, *keys: int | str) -> np.random.Generator:
    """Deterministic generator for a key tuple; strings are hashed with CRC-32."""

    words = [int(seed)]
    for key in keys:
        words.append(zlib.crc32(key.encode("utf-8")) if isinstance(key, str) else int(key))
    if any(word < 0 for word in words):
        raise SamplingError("generator keys must be non-negative")
    return np.random.default_rng(np.random.SeedSequence(words))


@dataclass(frozen=True)
class AugmentationParams:
    """Geometric augmentation shared by every step of one clip or stream sequence.

    ``origin_u`` and ``label_u`` are uniform fractions resolved against the clip's
    frame and labels when the clip is loaded.
    """

    hflip: bool = False
    zoom: str = "none"
    factor: float = 1.0
    origin_u: tuple[float, float] = (0.0, 0.0)
    label_u: float = 0.0


def draw_augmentation(rng: np.random.Generator, settings: Mapping[str, Any]) -> AugmentationParams:
    """Draw RVT-style flip/zoom parameters; the draw order is fixed for every setting."""

    hflip, do_zoom, zoom_in = rng.random(3)
    factor_u, origin_x, origin_y, label_u = rng.random(4)
    zoom = "none"
    factor = 1.0
    if do_zoom < float(settings["zoom_prob"]):
        zoom = "in" if zoom_in < float(settings["zoom_in_prob"]) else "out"
        low, high = settings["zoom_in_factor" if zoom == "in" else "zoom_out_factor"]
        factor = float(low) + float(factor_u) * (float(high) - float(low))
    return AugmentationParams(
        hflip=bool(hflip < float(settings["hflip_prob"])),
        zoom=zoom,
        factor=factor,
        origin_u=(float(origin_x), float(origin_y)),
        label_u=float(label_u),
    )


@dataclass(frozen=True)
class ClipSpec:
    """One clip: ``length`` real steps from ``start`` padded to ``steps``."""

    kind: str
    dataset: str
    sequence: str
    start: int
    length: int
    steps: int
    reset: bool
    slot: int = 0
    index: int = 0
    epoch: int = 0
    ordinal: int = 0
    augmentation: AugmentationParams = field(default_factory=AugmentationParams)


# --------------------------------------------------------------------------- sequences


@dataclass
class SequenceData:
    """Manifest arrays and cache access of one sequence."""

    dataset: str
    sequence: str
    split: str
    namespace: str
    identity_supervision: bool
    head: str
    geometry: Geometry
    step_t_us: np.ndarray
    supervised: np.ndarray
    box_offsets: np.ndarray
    boxes: dict[str, np.ndarray]
    class_mask: np.ndarray
    small_box: dict[str, Optional[float]]
    reader: CacheReader

    @property
    def steps(self) -> int:
        return int(self.step_t_us.size)

    def step_boxes(self, step: int) -> dict[str, np.ndarray]:
        low, high = int(self.box_offsets[step]), int(self.box_offsets[step + 1])
        return {name: value[low:high] for name, value in self.boxes.items()}


def small_box_rule(config: Mapping[str, Any], dataset: str) -> dict[str, Optional[float]]:
    """Model-frame small-box thresholds used when augmentation clips a box."""

    pool = int(config["geometry"][dataset]["pool"])
    settings = config[dataset]
    rule: dict[str, Optional[float]] = {"min_diagonal": None, "min_side": None, "min_area": None}
    if "small_box" in settings:
        rule["min_diagonal"] = float(settings["small_box"]["min_diagonal_px"]) / pool
        rule["min_side"] = float(settings["small_box"]["min_side_px"]) / pool
    if settings.get("min_rectified_area_px2") is not None:
        rule["min_area"] = float(settings["min_rectified_area_px2"]) / pool**2
    return rule


def converter_settings_sha256(config: Mapping[str, Any], dataset: str) -> str:
    if dataset == GEN4_1MPX:
        from src.data.converters.large_event_pretraining import dataset_settings

        return settings_hash(dataset_settings(config))
    from src.data.converters.dsec_detection import dataset_settings as dsec_settings

    return settings_hash(dsec_settings(config, dataset))


def sequence_data(
    record: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
    cache_root: str | Path,
) -> SequenceData:
    dataset = record["dataset"]
    geometry = dataset_geometry(config, dataset)
    if (record["frames"]["width"], record["frames"]["height"]) != (
        geometry.frame_width * geometry.pool,
        geometry.frame_height * geometry.pool,
    ):
        raise SchemaError(f"{record['sequence']}: box frame does not match the geometry")
    head = record["ontology"]["head"]
    class_mask = np.ones(HEAD_CLASSES, np.bool_)
    if head == "target":
        class_mask[list(record["ontology"].get("masked_target_classes", []))] = False
    step_t = np.asarray(arrays["step_t_us"], np.int64)
    box_step = np.asarray(arrays["box_step"], np.int64)
    offsets = np.searchsorted(box_step, np.arange(step_t.size + 1), side="left")
    label_source = "box_source_class" if head == "source" else "box_target_class"
    xywh = np.asarray(arrays["box_xywh"], np.float64) / geometry.pool
    boxes = {
        "xyxy": np.concatenate([xywh[:, :2], xywh[:, :2] + xywh[:, 2:]], 1),
        "label": np.asarray(arrays[label_source], np.int64),
        "ignore": np.asarray(arrays["box_ignore"], np.bool_),
        "track_id": np.asarray(arrays["box_track_id"], np.int64),
        "has_track_id": np.asarray(arrays["box_has_track_id"], np.bool_),
    }
    key = settings_hash(cache_key(record, step_t, config))
    return SequenceData(
        dataset=dataset,
        sequence=record["sequence"],
        split=record["split"],
        namespace=record["identity"]["namespace"],
        identity_supervision=bool(record["identity"]["supervision"]),
        head=head,
        geometry=geometry,
        step_t_us=step_t,
        supervised=np.asarray(arrays["step_status"]) == STEP_ANNOTATED,
        box_offsets=offsets,
        boxes=boxes,
        class_mask=class_mask,
        small_box=small_box_rule(config, dataset),
        reader=CacheReader(
            cache_path(cache_root, dataset, record["sequence"]), key, step_t.size, geometry
        ),
    )


def load_sequences(
    manifest_root: str | Path,
    cache_root: str | Path,
    config: Mapping[str, Any],
    dataset: str,
    splits: Sequence[str],
    require_complete: bool = True,
) -> list[SequenceData]:
    """Validated sequences of ``dataset`` in ``splits``, ordered by split and name.

    The manifest must have been converted with the current configuration, so a cache
    or loader never silently reads labels made under other conversion settings.
    """

    dataset_dir = Path(manifest_root) / dataset
    manifest = read_json(dataset_dir / MANIFEST_FILENAME)
    if require_complete and not manifest.get("complete"):
        raise SchemaError(f"{dataset}: manifest is a partial (sample-limited) conversion")
    if manifest["settings_sha256"] != converter_settings_sha256(config, dataset):
        raise SchemaError(f"{dataset}: manifest settings differ from the data configuration")
    sequences = []
    for split in splits:
        entries = [entry for entry in manifest["sequences"] if entry["split"] == split]
        if not entries:
            raise SchemaError(f"{dataset}: no sequences in split {split}")
        for entry in entries:
            record, arrays = load_sequence(dataset_dir, entry)
            sequences.append(sequence_data(record, arrays, config, cache_root))
    return sequences


class IdentityTable:
    """Global identity indices over ``(namespace, track_id)`` of identity sources.

    Only sequences whose manifest identity policy allows supervision contribute, and
    only through positive boxes with a known track ID.
    """

    def __init__(self, keys: Sequence[tuple[str, int]]):
        self.keys = tuple(sorted((str(namespace), int(track)) for namespace, track in keys))
        if len(set(self.keys)) != len(self.keys):
            raise SchemaError("duplicate identity keys")
        self._index = {key: index for index, key in enumerate(self.keys)}

    @classmethod
    def from_sequences(cls, sequences: Sequence[SequenceData]) -> "IdentityTable":
        keys = set()
        for sequence in sequences:
            if not sequence.identity_supervision:
                continue
            boxes = sequence.boxes
            usable = boxes["has_track_id"] & ~boxes["ignore"]
            keys.update(
                (sequence.namespace, int(track)) for track in np.unique(boxes["track_id"][usable])
            )
        return cls(sorted(keys))

    def __len__(self) -> int:
        return len(self.keys)

    def lookup(self, namespace: str, track_ids: np.ndarray) -> np.ndarray:
        return np.array(
            [self._index.get((namespace, int(track)), -1) for track in track_ids], np.int64
        )

    def sha256(self) -> str:
        return settings_hash([list(key) for key in self.keys])


# --------------------------------------------------------------------------- augmentation


def _nearest_exact_index(output_size: int, input_size: int) -> np.ndarray:
    """Source index of each output pixel for ``interpolate(mode='nearest-exact')``."""

    index = np.floor((np.arange(output_size) + 0.5) * (input_size / output_size)).astype(np.int64)
    return np.minimum(index, input_size - 1)


@dataclass(frozen=True)
class ResolvedAugmentation:
    """Pixel-level augmentation: horizontal flip, then an optional zoom window.

    Box coordinates use the corner convention (pixel ``j`` covers ``[j, j + 1)``), in
    which ``nearest-exact`` resampling of the count maps is exactly the affine map
    ``u' = (u - x0) * W / w`` (zoom-in) or ``u' = u * w / W + x0`` (zoom-out), and the
    flip ``u' = W - u`` maps the frame ``[0, W]`` onto itself.
    """

    hflip: bool
    zoom: str
    window: tuple[int, int, int, int]  # x0, y0, width, height of the zoom window
    width: int
    height: int

    def transform_counts(self, counts: np.ndarray) -> np.ndarray:
        if self.hflip:
            counts = counts[..., ::-1]
        x0, y0, window_w, window_h = self.window
        if self.zoom == "in":
            rows = y0 + _nearest_exact_index(self.height, window_h)
            cols = x0 + _nearest_exact_index(self.width, window_w)
            counts = counts[..., rows, :][..., cols]
        elif self.zoom == "out":
            rows = _nearest_exact_index(window_h, self.height)
            cols = _nearest_exact_index(window_w, self.width)
            canvas = np.zeros_like(counts)
            canvas[..., y0 : y0 + window_h, x0 : x0 + window_w] = counts[..., rows, :][..., cols]
            counts = canvas
        return np.ascontiguousarray(counts)

    def transform_boxes(self, xyxy: np.ndarray) -> np.ndarray:
        boxes = np.asarray(xyxy, np.float64).reshape(-1, 4).copy()
        if self.hflip:
            boxes[:, [0, 2]] = self.width - boxes[:, [2, 0]]
        x0, y0, window_w, window_h = self.window
        if self.zoom == "in":
            boxes[:, [0, 2]] = (boxes[:, [0, 2]] - x0) * (self.width / window_w)
            boxes[:, [1, 3]] = (boxes[:, [1, 3]] - y0) * (self.height / window_h)
        elif self.zoom == "out":
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * (window_w / self.width) + x0
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * (window_h / self.height) + y0
        return boxes


def _window_origin(low: float, high: float, fraction: float) -> int:
    if high < low:
        high = low
    return int(np.floor(low + fraction * (high - low)))


def resolve_augmentation(
    params: AugmentationParams,
    width: int,
    height: int,
    anchor_boxes: np.ndarray,
) -> ResolvedAugmentation:
    """Place the zoom window: zoom-out uniformly; zoom-in so it contains (or lies inside)
    one anchor box chosen by ``label_u``, as RVT samples it from the latest labeled frame;
    zoom-in without an anchor box is placed uniformly."""

    window = (0, 0, width, height)
    if params.zoom in ("in", "out") and params.factor > 1.0:
        window_w, window_h = int(width / params.factor), int(height / params.factor)
        fx, fy = params.origin_u
        x_range = (0.0, float(width - window_w))
        y_range = (0.0, float(height - window_h))
        if params.zoom == "in" and len(anchor_boxes):
            choice = min(int(params.label_u * len(anchor_boxes)), len(anchor_boxes) - 1)
            bx0, by0, bx1, by1 = np.asarray(anchor_boxes, np.float64)[choice]
            x_range = (
                max(min(bx0, bx1 - window_w), 0.0),
                min(max(bx0, bx1 - window_w), float(width - window_w)),
            )
            y_range = (
                max(min(by0, by1 - window_h), 0.0),
                min(max(by0, by1 - window_h), float(height - window_h)),
            )
        window = (_window_origin(*x_range, fx), _window_origin(*y_range, fy), window_w, window_h)
        return ResolvedAugmentation(params.hflip, params.zoom, window, width, height)
    return ResolvedAugmentation(params.hflip, "none", window, width, height)


def clip_transformed_boxes(
    transformed: np.ndarray,
    ignore: np.ndarray,
    width: int,
    height: int,
    small_box: Mapping[str, Optional[float]],
    min_visible_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Clip boxes to the frame; return ``(xyxy, ignore, keep, counts)``.

    Boxes with no area inside the frame are removed (``keep`` False). A clipped box
    whose remaining area is below ``min_visible_fraction`` of its transformed area, or
    below the dataset's small-box threshold, becomes an ignore region.
    """

    boxes = np.asarray(transformed, np.float64).reshape(-1, 4)
    full_area = np.clip(boxes[:, 2] - boxes[:, 0], 0, None) * np.clip(
        boxes[:, 3] - boxes[:, 1], 0, None
    )
    clipped = boxes.copy()
    clipped[:, [0, 2]] = np.clip(clipped[:, [0, 2]], 0, width)
    clipped[:, [1, 3]] = np.clip(clipped[:, [1, 3]], 0, height)
    w = clipped[:, 2] - clipped[:, 0]
    h = clipped[:, 3] - clipped[:, 1]
    area = np.clip(w, 0, None) * np.clip(h, 0, None)
    keep = (w > 0) & (h > 0)
    was_clipped = keep & (area < full_area * (1 - 1e-9))
    small = np.zeros(boxes.shape[0], np.bool_)
    if small_box.get("min_diagonal") is not None:
        small |= np.hypot(w, h) < small_box["min_diagonal"]
    if small_box.get("min_side") is not None:
        small |= np.minimum(w, h) < small_box["min_side"]
    if small_box.get("min_area") is not None:
        small |= area < small_box["min_area"]
    to_ignore = was_clipped & ~ignore & ((area < min_visible_fraction * full_area) | small)
    counts = {
        "boxes_removed": int(np.count_nonzero(~keep)),
        "positives_removed": int(np.count_nonzero(~keep & ~ignore)),
        "positives_to_ignore": int(np.count_nonzero(to_ignore & keep)),
    }
    return clipped, (ignore | to_ignore), keep, counts


# --------------------------------------------------------------------------- dataset


class LargeEventClipDataset(Dataset):
    """Map-style dataset indexed by :class:`ClipSpec` (use with a batch sampler)."""

    def __init__(
        self,
        sequences: Sequence[SequenceData],
        config: Mapping[str, Any],
        identity_table: Optional[IdentityTable] = None,
    ):
        self.sequences = {(item.dataset, item.sequence): item for item in sequences}
        if len(self.sequences) != len(sequences):
            raise SchemaError("duplicate sequence in the clip dataset")
        self.identity_table = identity_table or IdentityTable(())
        self.min_visible_fraction = float(config["augmentation"]["min_visible_fraction"])
        self.count_clip = int(config["representation"]["count_clip"])

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, spec: ClipSpec) -> dict[str, Any]:
        sequence = self.sequences[(spec.dataset, spec.sequence)]
        geometry = sequence.geometry
        if spec.length < 1 or spec.length > spec.steps:
            raise SamplingError(f"clip length {spec.length} outside [1, {spec.steps}]")
        if spec.start < 0 or spec.start + spec.length > sequence.steps:
            raise SamplingError(f"{spec.sequence}: clip leaves the sequence")
        width, height = geometry.frame_width, geometry.frame_height
        step_indices = np.arange(spec.start, spec.start + spec.length)
        step_boxes = [sequence.step_boxes(int(step)) for step in step_indices]

        flip_only = ResolvedAugmentation(
            spec.augmentation.hflip, "none", (0, 0, width, height), width, height
        )
        anchors = np.empty((0, 4))
        for step, boxes in reversed(list(zip(step_indices, step_boxes))):
            positive = ~boxes["ignore"]
            if sequence.supervised[step] and np.any(positive):
                anchors = flip_only.transform_boxes(boxes["xyxy"][positive])
                break
        augmentation = resolve_augmentation(spec.augmentation, width, height, anchors)

        counts = augmentation.transform_counts(sequence.reader.read(spec.start, spec.length))
        padded = np.zeros(
            (spec.steps, counts.shape[1], geometry.input_height, geometry.input_width), np.uint8
        )
        padded[: spec.length, :, :height, :width] = counts

        valid = np.zeros(spec.steps, np.bool_)
        valid[: spec.length] = True
        supervised = np.zeros(spec.steps, np.bool_)
        supervised[: spec.length] = sequence.supervised[step_indices]
        step_t = np.full(spec.steps, -1, np.int64)
        step_t[: spec.length] = sequence.step_t_us[step_indices]
        step_index = np.full(spec.steps, -1, np.int64)
        step_index[: spec.length] = step_indices

        statistics = {"boxes_removed": 0, "positives_removed": 0, "positives_to_ignore": 0}
        targets = []
        for position in range(spec.steps):
            if position >= spec.length:
                targets.append(_empty_targets())
                continue
            boxes = step_boxes[position]
            xyxy, ignore, keep, counts_ = clip_transformed_boxes(
                augmentation.transform_boxes(boxes["xyxy"]),
                boxes["ignore"],
                width,
                height,
                sequence.small_box,
                self.min_visible_fraction,
            )
            for key, value in counts_.items():
                statistics[key] += value
            identity = np.full(keep.size, -1, np.int64)
            if sequence.identity_supervision:
                identity = self.identity_table.lookup(sequence.namespace, boxes["track_id"])
            identity_valid = (
                sequence.identity_supervision & boxes["has_track_id"] & ~ignore & (identity >= 0)
            )
            targets.append(
                {
                    "boxes_xyxy": torch.from_numpy(xyxy[keep].astype(np.float32)),
                    "labels": torch.from_numpy(boxes["label"][keep]),
                    "ignore": torch.from_numpy(ignore[keep]),
                    "track_ids": torch.from_numpy(boxes["track_id"][keep]),
                    "identity": torch.from_numpy(np.where(identity_valid, identity, -1)[keep]),
                    "identity_valid": torch.from_numpy(identity_valid[keep]),
                }
            )
        return {
            "event_frame": torch.from_numpy(np.ascontiguousarray(padded[:, EVENT_FRAME])),
            "voxel_grid": torch.from_numpy(np.ascontiguousarray(padded[:, VOXEL_GRID])),
            "valid": torch.from_numpy(valid),
            "supervised": torch.from_numpy(supervised),
            "reset": bool(spec.reset),
            "step_t_us": torch.from_numpy(step_t),
            "step_index": torch.from_numpy(step_index),
            "count_scale": float(geometry.count_scale),
            "count_clip": self.count_clip,
            "class_mask": torch.from_numpy(sequence.class_mask.copy()),
            "frame_size": (height, width),
            "targets": targets,
            "meta": {
                **asdict(spec),
                "zoom_window": augmentation.window if augmentation.zoom != "none" else None,
                "split": sequence.split,
                "head": sequence.head,
                "namespace": sequence.namespace,
                "identity_supervision": sequence.identity_supervision,
                "augmentation_box_counts": statistics,
            },
        }


def _empty_targets() -> dict[str, torch.Tensor]:
    return {
        "boxes_xyxy": torch.zeros((0, 4), dtype=torch.float32),
        "labels": torch.zeros(0, dtype=torch.int64),
        "ignore": torch.zeros(0, dtype=torch.bool),
        "track_ids": torch.zeros(0, dtype=torch.int64),
        "identity": torch.zeros(0, dtype=torch.int64),
        "identity_valid": torch.zeros(0, dtype=torch.bool),
    }


def collate_clips(samples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Stack clips of one geometry into ``(B, T, ...)`` tensors; targets stay nested."""

    shapes = {tuple(sample["event_frame"].shape) for sample in samples}
    if len(shapes) != 1:
        raise SamplingError(f"clips of one batch need one shape, got {sorted(shapes)}")
    return {
        "event_frame": torch.stack([sample["event_frame"] for sample in samples]),
        "voxel_grid": torch.stack([sample["voxel_grid"] for sample in samples]),
        "valid": torch.stack([sample["valid"] for sample in samples]),
        "supervised": torch.stack([sample["supervised"] for sample in samples]),
        "reset": torch.tensor([sample["reset"] for sample in samples], dtype=torch.bool),
        "step_t_us": torch.stack([sample["step_t_us"] for sample in samples]),
        "step_index": torch.stack([sample["step_index"] for sample in samples]),
        "count_scale": torch.tensor([sample["count_scale"] for sample in samples]),
        "count_clip": samples[0]["count_clip"],
        "class_mask": torch.stack([sample["class_mask"] for sample in samples]),
        "frame_size": [sample["frame_size"] for sample in samples],
        "targets": [sample["targets"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }


def network_input(
    batch: Mapping[str, Any], device: torch.device | str | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalized ``(event_frame, voxel_grid)`` float tensors of a collated batch."""

    scale = batch["count_scale"].view(-1, 1, 1, 1, 1)
    outputs = []
    for name in ("event_frame", "voxel_grid"):
        counts = batch[name].to(device, non_blocking=True) if device is not None else batch[name]
        outputs.append(normalize_counts(counts, scale.to(counts.device), batch["count_clip"]))
    return outputs[0], outputs[1]


# --------------------------------------------------------------------------- samplers


class RandomClipSampler:
    """Uniform draw over every start whose ``T`` steps lie in one sequence and contain a
    supervised step (the same distribution as drawing a sequence proportionally to its
    start positions and rejecting unsupervised clips, without a rejection loop)."""

    def __init__(self, sequences: Sequence[SequenceData], steps: int):
        self.steps = int(steps)
        self.sequences = list(sequences)
        self.starts = []
        for sequence in self.sequences:
            if sequence.steps < self.steps:
                self.starts.append(np.empty(0, np.int64))
                continue
            window = np.convolve(
                sequence.supervised.astype(np.int64), np.ones(self.steps, np.int64), "valid"
            )
            self.starts.append(np.flatnonzero(window > 0))
        self.cumulative = np.cumsum([starts.size for starts in self.starts])
        if not self.cumulative.size or self.cumulative[-1] == 0:
            raise SamplingError("no sequence holds a supervised clip of the requested length")

    @property
    def total(self) -> int:
        return int(self.cumulative[-1])

    def draw(self, rng: np.random.Generator) -> tuple[SequenceData, int]:
        position = int(rng.integers(self.total))
        index = int(np.searchsorted(self.cumulative, position, side="right"))
        offset = position - (int(self.cumulative[index - 1]) if index else 0)
        return self.sequences[index], int(self.starts[index][offset])


@dataclass(frozen=True)
class StreamSegment:
    sequence: SequenceData
    start: int
    length: int
    reset: bool
    epoch: int
    ordinal: int


class StreamSchedule:
    """Deterministic TBPTT segments of one dataset's stream slots.

    Per slot and stream-epoch, the dataset's sequences are permuted with the counter
    generator ``(seed, stage, 'stream_epoch', dataset, epoch)`` and dealt round-robin to
    the dataset's slots; a slot walks its list in ``T``-step segments, the last segment
    of a sequence is padded, and a slot that exhausts its list starts its next epoch.
    ``segment(slot, k)`` is a pure function of ``k`` (the optimizer step), so the
    training state bank only needs to carry recurrent tensors.
    """

    def __init__(
        self,
        sequences: Sequence[SequenceData],
        slots: int,
        steps: int,
        seed: int,
        stage: str,
        dataset: str,
    ):
        self.sequences = [sequence for sequence in sequences if sequence.steps > 0]
        self.slots, self.steps = int(slots), int(steps)
        self.seed, self.stage, self.dataset = int(seed), stage, dataset
        if self.slots < 1 or len(self.sequences) < self.slots:
            raise SamplingError(
                f"{dataset}: {len(self.sequences)} sequences cannot fill {self.slots} stream slots"
            )
        self.segments = np.array(
            [-(-sequence.steps // self.steps) for sequence in self.sequences], np.int64
        )
        self._epochs: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {
            slot: [] for slot in range(self.slots)
        }
        self._epoch_starts: dict[int, list[int]] = {slot: [0] for slot in range(self.slots)}

    def _epoch_list(self, epoch: int) -> np.ndarray:
        rng = counter_rng(self.seed, self.stage, "stream_epoch", self.dataset, epoch)
        return rng.permutation(len(self.sequences))

    def _ensure(self, slot: int, index: int) -> None:
        epochs, starts = self._epochs[slot], self._epoch_starts[slot]
        while starts[-1] <= index:
            order = self._epoch_list(len(epochs))[slot :: self.slots]
            cumulative = np.cumsum(self.segments[order])
            epochs.append((order, cumulative))
            starts.append(starts[-1] + int(cumulative[-1]))

    def segment(self, slot: int, index: int) -> StreamSegment:
        if not 0 <= slot < self.slots or index < 0:
            raise SamplingError(f"stream slot {slot} or segment {index} out of range")
        self._ensure(slot, index)
        starts = self._epoch_starts[slot]
        epoch = int(np.searchsorted(starts, index, side="right")) - 1
        order, cumulative = self._epochs[slot][epoch]
        local = index - starts[epoch]
        ordinal = int(np.searchsorted(cumulative, local, side="right"))
        segment = local - (int(cumulative[ordinal - 1]) if ordinal else 0)
        sequence = self.sequences[int(order[ordinal])]
        start = segment * self.steps
        return StreamSegment(
            sequence=sequence,
            start=start,
            length=min(self.steps, sequence.steps - start),
            reset=segment == 0,
            epoch=epoch,
            ordinal=ordinal,
        )


class BatchPlan:
    """Clip composition of every optimizer step of one training stage.

    Random slots and stream slots are expanded from the stage's per-dataset counts in
    configuration order (random slots first). Every spec carries its dataset, so mixed
    DSEC-MOT/DSEC-Detection batches keep their provenance.
    """

    def __init__(
        self,
        stage: str,
        config: Mapping[str, Any],
        sequences: Mapping[str, Sequence[SequenceData]],
        seed: int,
    ):
        if stage not in STAGES:
            raise SamplingError(f"unknown stage {stage!r}")
        settings = config["sampling"][stage]
        self.stage, self.seed = stage, int(seed)
        self.steps = int(settings["clip_steps"])
        self.random_settings = config["augmentation"]["random"]
        self.stream_settings = config["augmentation"]["stream"]
        self.random_slots = [
            dataset for dataset, count in settings["random_slots"].items() for _ in range(count)
        ]
        self.stream_slots = [
            (dataset, local)
            for dataset, count in settings["stream_slots"].items()
            for local in range(count)
        ]
        self.samplers = {
            dataset: RandomClipSampler(sequences[dataset], self.steps)
            for dataset in dict.fromkeys(self.random_slots)
        }
        self.schedules = {
            dataset: StreamSchedule(
                sequences[dataset], count, self.steps, self.seed, stage, dataset
            )
            for dataset, count in settings["stream_slots"].items()
        }

    @property
    def batch_size(self) -> int:
        return len(self.random_slots) + len(self.stream_slots)

    def batch(self, step: int) -> list[ClipSpec]:
        specs = []
        for slot, dataset in enumerate(self.random_slots):
            rng = counter_rng(self.seed, self.stage, "random", slot, step)
            sequence, start = self.samplers[dataset].draw(rng)
            specs.append(
                ClipSpec(
                    kind="random",
                    dataset=dataset,
                    sequence=sequence.sequence,
                    start=start,
                    length=self.steps,
                    steps=self.steps,
                    reset=True,
                    slot=slot,
                    index=step,
                    augmentation=draw_augmentation(rng, self.random_settings),
                )
            )
        for slot, (dataset, local) in enumerate(self.stream_slots):
            segment = self.schedules[dataset].segment(local, step)
            rng = counter_rng(
                self.seed, self.stage, "stream", dataset, local, segment.epoch, segment.ordinal
            )
            specs.append(
                ClipSpec(
                    kind="stream",
                    dataset=dataset,
                    sequence=segment.sequence.sequence,
                    start=segment.start,
                    length=segment.length,
                    steps=self.steps,
                    reset=segment.reset,
                    slot=slot,
                    index=step,
                    epoch=segment.epoch,
                    ordinal=segment.ordinal,
                    augmentation=draw_augmentation(rng, self.stream_settings),
                )
            )
        return specs

    def micro_batches(self, specs: Sequence[ClipSpec], count: int) -> list[list[ClipSpec]]:
        """Split one step into ``count`` micro-batches with a fixed slot subset each.

        Micro-batch ``m`` holds random slots ``m::count`` and stream slots ``m::count``,
        so every stream slot is in exactly one micro-batch per step and the random:stream
        ratio is kept whenever ``count`` divides both slot counts.
        """

        randoms = [spec for spec in specs if spec.kind == "random"]
        streams = [spec for spec in specs if spec.kind == "stream"]
        if count < 1 or len(randoms) % count or len(streams) % count:
            raise SamplingError(f"{count} micro-batches do not divide the slot counts")
        return [randoms[index::count] + streams[index::count] for index in range(count)]


class BatchPlanSampler(Sampler):
    """Yields the micro-batches of optimizer steps ``start_step, start_step + 1, ...``."""

    def __init__(self, plan: BatchPlan, start_step: int, num_steps: int, micro_batches: int = 1):
        self.plan = plan
        self.start_step, self.num_steps = int(start_step), int(num_steps)
        self.micro_batches = int(micro_batches)

    def __iter__(self) -> Iterator[list[ClipSpec]]:
        for step in range(self.start_step, self.start_step + self.num_steps):
            yield from self.plan.micro_batches(self.plan.batch(step), self.micro_batches)

    def __len__(self) -> int:
        return self.num_steps * self.micro_batches


def sequence_clips(sequence: SequenceData, steps: int) -> Iterator[ClipSpec]:
    """Ordered evaluation segments of a whole sequence: reset first, padded tail, no
    augmentation. Carry state between consecutive segments and reset it per sequence."""

    for ordinal, start in enumerate(range(0, sequence.steps, steps)):
        yield ClipSpec(
            kind="eval",
            dataset=sequence.dataset,
            sequence=sequence.sequence,
            start=start,
            length=min(steps, sequence.steps - start),
            steps=steps,
            reset=start == 0,
            index=ordinal,
            ordinal=ordinal,
        )


# --------------------------------------------------------------------------- stages


@dataclass
class StageData:
    dataset: LargeEventClipDataset
    plan: BatchPlan
    identity_table: IdentityTable
    sequences: dict[str, list[SequenceData]]


def stage_data(
    stage: str,
    manifest_root: str | Path,
    cache_root: str | Path,
    config: Mapping[str, Any],
    seed: int,
    require_complete: bool = True,
) -> StageData:
    """Training data of one stage: sequences, the identity table, dataset, and plan.

    Identities come only from identity-supervised training sequences (DSEC-MOT in
    fine-tuning); pretraining has an empty table.
    """

    splits = config["sampling"][stage]["splits"]
    sequences = {
        dataset: load_sequences(
            manifest_root, cache_root, config, dataset, [split], require_complete
        )
        for dataset, split in splits.items()
    }
    everything = [item for items in sequences.values() for item in items]
    identity_table = IdentityTable.from_sequences(everything)
    return StageData(
        dataset=LargeEventClipDataset(everything, config, identity_table),
        plan=BatchPlan(stage, config, sequences, seed),
        identity_table=identity_table,
        sequences=sequences,
    )
