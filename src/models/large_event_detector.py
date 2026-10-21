"""RVT-B detector with separate EF / VG branches, gated fusion, and object embeddings.

Architecture (``configs/large_event_mot/model.json``)::

    EF (2 ch) --branch--\\
                         gated fusion, stride 4, 64 ch  (replaces RVT stages[0].downsample_cf2cl)
    VG (6 ch) --branch--/  -> RVT-B stages 1-4 (MaxViT + LSTM) -> RVT PAFPN (P3, P4, P5)
                           -> YOLOX head with 7 class rows
                           -> embedding head (128-d unit descriptor per YOLOX anchor)

The pinned RVT checkout supplies the backbone, PAFPN, and YOLOX head modules, so their
parameter names (``backbone.*``, ``fpn.*``, ``yolox_head.*``) match the RVT checkpoint.
The head runs its upstream convolutions, but the per-anchor outputs are assembled here
in the upstream training layout (decoded ``cx, cy, w, h`` in pixels, objectness logit,
class logits) together with the anchor grid, so losses can reuse upstream SimOTA and
decoding never depends on the head's cached grids.

Recurrence: the four RVT stage LSTMs are the model's temporal memory (the stage 2-4
hidden states are the PAFPN inputs). :meth:`LargeEventDetector.forward` takes and returns
an explicit :class:`RecurrentState` (fp32 ``(h, c)`` per stage and stream); ``state=None``
is a zero state, ``reset`` zeroes rows before the step, and rows with ``valid=False``
return their input state unchanged (masked tail steps). The caller owns detaching and
storing states; validation and inference start every sequence from a zero state.

Embeddings: one L2-normalized descriptor per anchor in the YOLOX output order. Decoding
carries the anchor index through score floor, NMS, and cap, so ``Detections.embeddings``
rows are aligned with ``Detections.boxes`` by construction.

Heads: ``source`` keeps the seven 1Mpx rows (pedestrian, two-wheeler, car, truck, bus,
traffic sign, traffic light); ``target`` has the seven DSEC-MOT rows. Initialization:

- ``rvt_checkpoint`` (source only): every RVT tensor is copied, the 20-channel stem conv is
  adapted into the EF and VG stems, the stem LayerNorm becomes the fusion LayerNorm, and
  class rows are copied by name; the rest is new (gate at zero, so ``g = 0.5``).
- ``random``: seeded fresh initialization; nothing is read or downloaded.
- :func:`transfer_source_to_target`: a target model from a source state dict, with class
  rows copied by the configured mapping and ``train`` new; the embedding head starts new
  because pretraining has no identity supervision.

Each initialization records loaded, adapted, new, and dropped tensors in
``model.metadata["transfer"]``.

Example::

    python -m src.models.large_event_detector --rvt-root <rvt_root> \\
        --checkpoint <rvt-b.ckpt> --geometry gen4_1mpx --smoke-forward
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torchvision.ops import batched_nms

from src.models.large_event_modules import (
    EmbeddingHead,
    FusionOutput,
    GatedFusionStem,
    RecurrentState,
    RvtCheckpoint,
    RvtUpstream,
    UpstreamError,
    adapt_rvt_stem,
    import_rvt,
    read_rvt_checkpoint,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "large_event_mot" / "model.json"
ONTOLOGIES = ("source", "target")
INITIALIZATIONS = ("rvt_checkpoint", "random")
FUSION_PREFIX = "backbone.stages.0.downsample_cf2cl."
RVT_STEM_CONV = FUSION_PREFIX + "conv.weight"
CLASS_PREDS_PREFIX = "yolox_head.cls_preds."
EMBEDDING_PREFIX = "embedding_head."
PARAMETER_GROUPS = (
    ("event_frame_branch", FUSION_PREFIX + "event_frame."),
    ("voxel_grid_branch", FUSION_PREFIX + "voxel_grid."),
    ("fusion", FUSION_PREFIX),
    ("backbone", "backbone."),
    ("fpn", "fpn."),
    ("head", "yolox_head."),
    ("embedding_head", EMBEDDING_PREFIX),
)


class ModelConfigError(ValueError):
    """The model configuration or a requested initialization is inconsistent."""


def load_model_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise ModelConfigError(f"{path}: unsupported model schema {config.get('schema_version')}")
    return config


def config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _runtime_path(
    value: Optional[str | Path], env: str, fallback: Optional[str], what: str
) -> Path:
    resolved = value or os.environ.get(env) or fallback
    if not resolved:
        raise ModelConfigError(f"{what} is not set: pass it explicitly or set {env}")
    return Path(resolved).expanduser()


def resolve_rvt_root(config: Mapping[str, Any], rvt_root: Optional[str | Path] = None) -> Path:
    upstream = config["upstream"]
    return _runtime_path(rvt_root, upstream["rvt_root_env"], upstream["rvt_root"], "RVT root")


def resolve_checkpoint(config: Mapping[str, Any], checkpoint: Optional[str | Path] = None) -> Path:
    settings = config["checkpoint"]
    return _runtime_path(checkpoint, settings["path_env"], settings["path"], "RVT checkpoint")


# --------------------------------------------------------------------------- geometry


@dataclass(frozen=True)
class ModelGeometry:
    """Model input size, valid (unpadded) region, and MaxViT partition for a dataset."""

    name: str
    input_hw: tuple[int, int]
    valid_hw: tuple[int, int]
    partition_size: tuple[int, int]


def model_geometry(config: Mapping[str, Any], name: str) -> ModelGeometry:
    try:
        entry = config["geometry"][name]
    except KeyError:
        raise ModelConfigError(f"no model geometry for {name!r}") from None
    geometry = ModelGeometry(
        name=name,
        input_hw=tuple(entry["input_hw"]),
        valid_hw=tuple(entry["valid_hw"]),
        partition_size=tuple(entry["partition_size"]),
    )
    backbone = config["rvt"]["backbone"]
    stride = int(backbone["stem"]["patch_size"])
    for _ in backbone["num_blocks"]:
        for size, part in zip(geometry.input_hw, geometry.partition_size):
            if size % stride or (size // stride) % part:
                raise ModelConfigError(
                    f"{name}: partition {geometry.partition_size} does not divide the "
                    f"stride-{stride} map of input {geometry.input_hw}"
                )
        stride *= 2
    if any(v > i for v, i in zip(geometry.valid_hw, geometry.input_hw)):
        raise ModelConfigError(f"{name}: valid region {geometry.valid_hw} exceeds the input")
    return geometry


# --------------------------------------------------------------------------- outputs


@dataclass
class HeadOutput:
    """Per-anchor head outputs in YOLOX order (levels s8, s16, s32; row-major per level).

    ``boxes`` are decoded ``(cx, cy, w, h)`` in input pixels; logits are raw. All three
    are float32 even under autocast.
    """

    boxes: torch.Tensor
    obj_logits: torch.Tensor
    cls_logits: torch.Tensor
    grid: torch.Tensor
    strides: torch.Tensor
    level_hw: tuple[tuple[int, int], ...]

    def train_layout(self) -> torch.Tensor:
        """``(N, A, 5 + K)`` as upstream ``YOLOXHead.get_losses`` receives it."""

        return torch.cat((self.boxes, self.obj_logits.unsqueeze(-1), self.cls_logits), dim=-1)

    @property
    def x_shifts(self) -> torch.Tensor:
        return self.grid[None, :, 0]

    @property
    def y_shifts(self) -> torch.Tensor:
        return self.grid[None, :, 1]

    @property
    def expanded_strides(self) -> torch.Tensor:
        return self.strides[None, :]

    def scores_and_labels(self) -> tuple[torch.Tensor, torch.Tensor]:
        """YOLOX single-label decoding: ``sigmoid(obj) * max_k sigmoid(cls_k)``."""

        best, labels = torch.sigmoid(self.cls_logits).max(dim=-1)
        return torch.sigmoid(self.obj_logits) * best, labels


@dataclass
class DetectorOutput:
    """One step: stage features, PAFPN levels, head, embeddings, fusion, and new state.

    ``embeddings`` is ``(N, A, D)`` fp32 with unit rows in the ``head`` anchor order, or
    ``None`` when the step ran without the embedding head.
    """

    backbone_features: dict[int, torch.Tensor]
    fpn_features: tuple[torch.Tensor, ...]
    head: HeadOutput
    fusion: FusionOutput
    state: RecurrentState
    embeddings: Optional[torch.Tensor] = None


@dataclass
class Detections:
    """Decoded detections of one image, sorted by descending score."""

    boxes: torch.Tensor  # (D, 4) xyxy in input pixels, clipped to the valid region
    scores: torch.Tensor  # (D,)
    labels: torch.Tensor  # (D,) head rows
    anchors: torch.Tensor  # (D,) anchor index into the HeadOutput order
    cap_reached: bool
    embeddings: Optional[torch.Tensor] = None  # (D, E) rows aligned with boxes


def decode_detections(
    head: HeadOutput,
    valid_hw: Sequence[int],
    score_floor: float,
    nms_iou: float,
    max_detections: int,
    embeddings: Optional[torch.Tensor] = None,
) -> list[Detections]:
    """Score floor, clip to the valid region, class-aware NMS, then a score-ordered cap.

    Boxes with no area left after clipping (for example inside 1Mpx padding rows) are
    dropped. A higher score threshold on the result equals decoding at that threshold.
    With ``embeddings`` (``(N, A, E)``), each detection gets the descriptor of the anchor
    that produced it; an image without detections gets a ``(0, E)`` tensor.
    """

    if embeddings is not None and embeddings.shape[:2] != head.boxes.shape[:2]:
        raise ValueError(
            f"embeddings {tuple(embeddings.shape)} do not match {tuple(head.boxes.shape[:2])} "
            "anchors"
        )

    height, width = valid_hw
    with torch.no_grad():
        scores, labels = head.scores_and_labels()
        center, size = head.boxes[..., :2], head.boxes[..., 2:4]
        xyxy = torch.cat((center - 0.5 * size, center + 0.5 * size), dim=-1)
        xyxy[..., 0::2] = xyxy[..., 0::2].clamp(0.0, float(width))
        xyxy[..., 1::2] = xyxy[..., 1::2].clamp(0.0, float(height))
        positive = (xyxy[..., 2] > xyxy[..., 0]) & (xyxy[..., 3] > xyxy[..., 1])
        results = []
        for image in range(xyxy.shape[0]):
            anchors = torch.nonzero((scores[image] >= score_floor) & positive[image]).squeeze(1)
            keep = batched_nms(
                xyxy[image, anchors], scores[image, anchors], labels[image, anchors], nms_iou
            )
            anchors = anchors[keep]
            cap_reached = anchors.numel() > max_detections
            anchors = anchors[:max_detections]
            results.append(
                Detections(
                    boxes=xyxy[image, anchors],
                    scores=scores[image, anchors],
                    labels=labels[image, anchors],
                    anchors=anchors,
                    cap_reached=bool(cap_reached),
                    embeddings=None if embeddings is None else embeddings[image, anchors],
                )
            )
    return results


# --------------------------------------------------------------------------- model


def _dict_config(value: Mapping[str, Any]) -> Any:
    from omegaconf import OmegaConf

    return OmegaConf.create(copy.deepcopy(dict(value)))


class LargeEventDetector(nn.Module):
    """RVT-B with the gated EF/VG stem, a seven-row YOLOX head, and the embedding head."""

    def __init__(
        self,
        config: Mapping[str, Any],
        upstream: RvtUpstream,
        ontology: str,
        geometry: str,
    ) -> None:
        super().__init__()
        if ontology not in ONTOLOGIES:
            raise ModelConfigError(f"unknown head ontology {ontology!r}")
        fusion = config["fusion"]
        rvt = config["rvt"]
        if fusion["gate_init"] != "zero":
            raise ModelConfigError(f"unsupported gate_init {fusion['gate_init']!r}")
        if fusion["dim"] != rvt["backbone"]["embed_dim"] * rvt["backbone"]["dim_multiplier"][0]:
            raise ModelConfigError("fusion dim must equal the RVT stage-1 dimension")
        classes = tuple(config["heads"][ontology]["classes"])
        if len(classes) != rvt["head"]["num_classes"]:
            raise ModelConfigError(f"{ontology} head has {len(classes)} classes, RVT head differs")
        recurrence = config["recurrence"]
        num_stages = len(rvt["backbone"]["num_blocks"])
        if (
            recurrence["placement"] != "rvt_stage_lstm"
            or list(recurrence["stages"]) != list(range(1, num_stages + 1))
            or recurrence["state_dtype"] != "float32"
        ):
            raise ModelConfigError(f"unsupported recurrence settings {recurrence}")
        if list(config["embedding"]["in_stages"]) != list(rvt["fpn"]["in_stages"]):
            raise ModelConfigError("the embedding head must read the PAFPN levels of the head")

        self.config = copy.deepcopy(dict(config))
        self.ontology = ontology
        self.classes = classes
        self.geometry = model_geometry(config, geometry)

        backbone_cfg = copy.deepcopy(rvt["backbone"])
        backbone_cfg["stage"]["attention"]["partition_size"] = list(self.geometry.partition_size)
        maxvit_rnn = upstream.module("models.detection.recurrent_backbone.maxvit_rnn")
        build = upstream.module("models.detection.yolox_extension.models.build")
        self.backbone = maxvit_rnn.RNNDetector(_dict_config(backbone_cfg))
        self.backbone.stages[0].downsample_cf2cl = GatedFusionStem(
            int(fusion["event_frame_channels"]), int(fusion["voxel_grid_channels"]), fusion
        )
        in_stages = tuple(rvt["fpn"]["in_stages"])
        in_channels = self.backbone.get_stage_dims(in_stages)
        self.fpn = build.build_yolox_fpn(_dict_config(rvt["fpn"]), in_channels=in_channels)
        self.yolox_head = build.build_yolox_head(
            _dict_config(rvt["head"]),
            in_channels=in_channels,
            strides=self.backbone.get_strides(in_stages),
        )
        self.embedding_head = EmbeddingHead(
            in_channels,
            config["embedding"],
            upstream.module("models.detection.yolox.models.network_blocks").BaseConv,
        )
        self._attention_type = upstream.module("models.layers.maxvit.maxvit").PartitionAttentionCl
        self.metadata: dict[str, Any] = {
            "architecture": "RVT-B with gated EF/VG stem, YOLOX head, and embedding head",
            "upstream": {
                "repository": config["upstream"]["repository"],
                "commit": upstream.commit,
                "license": config["upstream"]["license"],
            },
            "config_sha256": config_sha256(config),
            "ontology": ontology,
            "classes": list(classes),
            "recurrence": "RVT stage LSTMs (1x1 gates), carried fp32 (h, c) per stage",
            "embedding_dim": self.embedding_head.dim,
        }

    @property
    def fusion(self) -> GatedFusionStem:
        return self.backbone.stages[0].downsample_cf2cl

    def set_geometry(self, name: str) -> None:
        """Switch input size and MaxViT partition; reset the head's cached grids."""

        self.geometry = model_geometry(self.config, name)
        for module in self.modules():
            if isinstance(module, self._attention_type):
                module.partition_size = self.geometry.partition_size
        self.yolox_head.output_grids = None
        self.yolox_head.output_strides = None
        self.yolox_head.grids = [torch.zeros(1)] * len(self.yolox_head.grids)

    def state_shapes(self, geometry: Optional[str] = None) -> tuple[tuple[int, int, int], ...]:
        """Per-stage LSTM state ``(C, H, W)`` for ``geometry`` (default: the current one)."""

        selected = self.geometry if geometry is None else model_geometry(self.config, geometry)
        height, width = selected.input_hw
        return tuple(
            (int(dim), height // int(stride), width // int(stride))
            for dim, stride in zip(self.backbone.stage_dims, self.backbone.strides)
        )

    def initial_state(
        self, batch_size: int, device: torch.device | str | None = None
    ) -> RecurrentState:
        """Zero fp32 state for ``batch_size`` streams (start of every sequence)."""

        return RecurrentState.zeros(batch_size, self.state_shapes(), device=device)

    def forward(
        self,
        event_frame: torch.Tensor,
        voxel_grid: torch.Tensor,
        state: Optional[RecurrentState] = None,
        reset: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
        with_embeddings: bool = True,
    ) -> DetectorOutput:
        """One step on normalized ``(N, 2, H, W)`` EF and ``(N, 6, H, W)`` VG inputs.

        ``state`` is the state before this step (``None``: zero). ``reset`` (``(N,)``
        bool) zeroes rows before the step. Rows with ``valid`` false return their
        (reset) input state, so masked padding steps do not advance the memory; their
        outputs are computed but carry no meaning. Gradients flow through ``state``;
        detaching at a truncation boundary is the caller's decision.
        """

        for name, tensor in (("event_frame", event_frame), ("voxel_grid", voxel_grid)):
            if tuple(tensor.shape[-2:]) != self.geometry.input_hw:
                raise ValueError(
                    f"{name} is {tuple(tensor.shape[-2:])}, geometry {self.geometry.name} "
                    f"expects {self.geometry.input_hw}"
                )
        batch = event_frame.shape[0]
        if state is None:
            state = self.initial_state(batch, device=event_frame.device)
        elif state.batch_size != batch or state.shapes() != self.state_shapes():
            raise ValueError(
                f"state for {state.batch_size} streams of {state.shapes()} does not match "
                f"{batch} inputs of geometry {self.geometry.name} {self.state_shapes()}"
            )
        if reset is not None:
            state = state.reset(reset)
        features, rvt_states = self.backbone((event_frame, voxel_grid), state.as_rvt())
        new_state = RecurrentState.from_rvt(rvt_states)
        if valid is not None:
            new_state = new_state.hold(state, valid)
        fusion = self.fusion.last_output
        self.fusion.last_output = None
        fpn_features = tuple(self.fpn(features))
        return DetectorOutput(
            backbone_features=features,
            fpn_features=fpn_features,
            head=self._head(fpn_features),
            fusion=fusion,
            state=new_state,
            embeddings=self.embedding_head(fpn_features) if with_embeddings else None,
        )

    def _head(self, features: Sequence[torch.Tensor]) -> HeadOutput:
        head = self.yolox_head
        outputs, grids, strides, level_hw = [], [], [], []
        for level, (x, stride) in enumerate(zip(features, head.strides)):
            x = head.stems[level](x)
            cls_output = head.cls_preds[level](head.cls_convs[level](x))
            reg_feature = head.reg_convs[level](x)
            output = torch.cat(
                (
                    head.reg_preds[level](reg_feature),
                    head.obj_preds[level](reg_feature),
                    cls_output,
                ),
                dim=1,
            )
            height, width = output.shape[-2:]
            outputs.append(output.flatten(start_dim=2).permute(0, 2, 1))
            ys, xs = torch.meshgrid(
                torch.arange(height, device=x.device, dtype=torch.float32),
                torch.arange(width, device=x.device, dtype=torch.float32),
                indexing="ij",
            )
            grids.append(torch.stack((xs, ys), dim=-1).reshape(-1, 2))
            strides.append(torch.full((height * width,), float(stride), device=x.device))
            level_hw.append((int(height), int(width)))
        raw = torch.cat(outputs, dim=1).float()
        grid = torch.cat(grids)
        stride = torch.cat(strides)[:, None]
        boxes = torch.cat(((raw[..., :2] + grid) * stride, torch.exp(raw[..., 2:4]) * stride), -1)
        return HeadOutput(
            boxes=boxes,
            obj_logits=raw[..., 4],
            cls_logits=raw[..., 5:],
            grid=grid,
            strides=stride[:, 0],
            level_hw=tuple(level_hw),
        )

    def decode(
        self,
        output: DetectorOutput,
        score_floor: Optional[float] = None,
        nms_iou: Optional[float] = None,
        max_detections: Optional[int] = None,
    ) -> list[Detections]:
        """Decode with the configured export settings unless overridden."""

        settings = self.config["decoding"]
        return decode_detections(
            output.head,
            self.geometry.valid_hw,
            settings["score_floor"] if score_floor is None else score_floor,
            settings["nms_iou"] if nms_iou is None else nms_iou,
            settings["max_detections"] if max_detections is None else max_detections,
            embeddings=output.embeddings,
        )


# --------------------------------------------------------------------------- accounting


def parameter_counts(model: nn.Module) -> dict[str, Any]:
    """Unique parameters (deduplicated by identity) per group, plus trainable and buffers."""

    groups = {name: 0 for name, _ in PARAMETER_GROUPS}
    groups["other"] = 0
    seen: set[int] = set()
    total = trainable = 0
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        count = parameter.numel()
        total += count
        trainable += count if parameter.requires_grad else 0
        group = next((g for g, prefix in PARAMETER_GROUPS if name.startswith(prefix)), "other")
        groups[group] += count
    buffers = sum(buffer.numel() for buffer in model.buffers())
    return {"total": total, "trainable": trainable, "groups": groups, "buffers": buffers}


def model_accounting(model: LargeEventDetector) -> dict[str, Any]:
    """Inference parameters with the 20-90M check, training-only parameters, state sizes.

    Raises :class:`ModelConfigError` when the complete inference model is outside the
    configured interval.
    """

    counts = parameter_counts(model)
    interval = model.config["parameter_interval"]
    if not interval["min"] <= counts["total"] <= interval["max"]:
        raise ModelConfigError(
            f"{counts['total']} inference parameters are outside "
            f"[{interval['min']}, {interval['max']}]"
        )
    classifier = model.config["embedding"]["identity_classifier"]
    identities = int(classifier["identities"])
    classifier_parameters = identities * model.embedding_head.dim
    classifier_parameters += identities if classifier["bias"] else 0
    state = {}
    for name in model.config["geometry"]:
        shapes = model.state_shapes(name)
        floats = sum(2 * c * h * w for c, h, w in shapes)
        state[name] = {
            "shapes": [list(shape) for shape in shapes],
            "floats_per_stream": floats,
            "bytes_per_stream": 4 * floats,
        }
    return {
        "parameters": {**counts, "interval": [interval["min"], interval["max"]]},
        "training_only": {"identity_classifier": classifier_parameters},
        "recurrent_state": state,
    }


# --------------------------------------------------------------------------- initialization


@dataclass
class TransferReport:
    """Which model tensors were loaded, adapted, newly initialized, or dropped."""

    source: str
    loaded: list[str] = field(default_factory=list)
    adapted: dict[str, str] = field(default_factory=dict)
    new: list[str] = field(default_factory=list)
    dropped: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "loaded": sorted(self.loaded),
            "adapted": dict(sorted(self.adapted.items())),
            "new": sorted(self.new),
            "dropped": dict(sorted(self.dropped.items())),
            "counts": {
                "loaded": len(self.loaded),
                "adapted": len(self.adapted),
                "new": len(self.new),
                "dropped": len(self.dropped),
            },
        }


def _new_model(
    config: Mapping[str, Any], upstream: RvtUpstream, ontology: str, geometry: str
) -> LargeEventDetector:
    """Construct with the configured seed so new tensors are reproducible."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(config["init_seed"]))
        return LargeEventDetector(config, upstream, ontology, geometry)


def _copy_class_rows(
    state: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    rows: Mapping[int, int],
    report: TransferReport,
    rule: str,
) -> set[str]:
    """Copy ``rows`` (model row -> source row) of every ``cls_preds`` weight and bias."""

    used = set()
    for key in [k for k in state if k.startswith(CLASS_PREDS_PREFIX)]:
        if key not in source:
            raise UpstreamError(f"source state has no {key}")
        target = state[key].clone()
        for model_row, source_row in rows.items():
            target[model_row] = source[key][source_row]
        state[key] = target
        report.adapted[key] = rule
        used.add(key)
    return used


def _rows_by_name(
    model_classes: Sequence[str], source_classes: Sequence[str], mapping: Mapping[str, Any]
) -> dict[int, int]:
    rows = {}
    for name, source_name in mapping.items():
        if source_name is None:
            continue
        if name not in model_classes or source_name not in source_classes:
            raise ModelConfigError(f"class mapping {name!r} <- {source_name!r} is not in the heads")
        rows[model_classes.index(name)] = list(source_classes).index(source_name)
    return rows


def _row_rule(
    classes: Sequence[str], rows: Mapping[int, int], source_classes: Sequence[str]
) -> str:
    parts = [
        f"{name} <- {source_classes[rows[i]]}" if i in rows else f"{name} new"
        for i, name in enumerate(classes)
    ]
    return "class rows: " + ", ".join(parts)


def load_rvt_checkpoint(
    model: LargeEventDetector, checkpoint: RvtCheckpoint, config: Mapping[str, Any]
) -> TransferReport:
    """Initialize a source-head model from the verified RVT checkpoint."""

    if model.ontology != "source":
        raise ModelConfigError("the RVT checkpoint initializes the source head only")
    settings = config["checkpoint"]
    if len(checkpoint.state) != settings["tensors"]:
        raise UpstreamError(f"checkpoint has {len(checkpoint.state)} tensors")
    report = TransferReport(source=f"RVT checkpoint sha256 {checkpoint.sha256}")
    state = model.state_dict()
    used: set[str] = set()

    fusion = config["fusion"]
    stem = checkpoint.state[RVT_STEM_CONV]
    event_frame, voxel_grid = adapt_rvt_stem(
        stem,
        polarities=int(settings["input"]["polarities"]),
        source_bins=int(settings["input"]["time_bins"]),
        voxel_bins=int(fusion["voxel_bins"]),
    )
    for key, value, rule in (
        (FUSION_PREFIX + "event_frame.stem.weight", event_frame, "mean of RVT stem over 10 bins"),
        (FUSION_PREFIX + "voxel_grid.stem.weight", voxel_grid, "overlap-weighted RVT stem bins"),
    ):
        if state[key].shape != value.shape:
            raise UpstreamError(f"{key}: adapted shape {tuple(value.shape)}")
        state[key] = value
        report.adapted[key] = f"{rule} ({RVT_STEM_CONV})"
    used.add(RVT_STEM_CONV)
    for key in (FUSION_PREFIX + "norm.weight", FUSION_PREFIX + "norm.bias"):
        state[key] = checkpoint.state[key].clone()
        report.adapted[key] = "RVT stage-1 stem LayerNorm used as the fusion LayerNorm"
        used.add(key)

    rows = _rows_by_name(
        model.classes, settings["classes"], config["heads"]["source"]["from_checkpoint"]
    )
    used |= _copy_class_rows(
        state,
        checkpoint.state,
        rows,
        report,
        _row_rule(model.classes, rows, settings["classes"]),
    )

    for key, tensor in checkpoint.state.items():
        if key in used:
            continue
        if key not in state or state[key].shape != tensor.shape:
            raise UpstreamError(f"checkpoint tensor {key} {tuple(tensor.shape)} has no model match")
        state[key] = tensor.clone()
        report.loaded.append(key)
        used.add(key)
    report.new = [key for key in state if key not in used and key not in report.adapted]
    model.load_state_dict(state, strict=True)
    return report


def build_large_event_detector(
    config: Mapping[str, Any],
    *,
    ontology: str = "source",
    geometry: str = "gen4_1mpx",
    init: str = "rvt_checkpoint",
    rvt_root: Optional[str | Path] = None,
    checkpoint: Optional[str | Path] = None,
) -> LargeEventDetector:
    """Build a detector; ``init`` is ``rvt_checkpoint`` (source head) or ``random``."""

    if init not in INITIALIZATIONS:
        raise ModelConfigError(f"unknown initialization {init!r}")
    upstream = import_rvt(resolve_rvt_root(config, rvt_root), config["upstream"]["commit"])
    model = _new_model(config, upstream, ontology, geometry)
    if init == "random":
        report = TransferReport(source=f"random (seed {config['init_seed']})")
        report.new = list(model.state_dict())
        model.metadata["checkpoint"] = None
    else:
        loaded = read_rvt_checkpoint(resolve_checkpoint(config, checkpoint), config["checkpoint"])
        report = load_rvt_checkpoint(model, loaded, config)
        model.metadata["checkpoint"] = {
            "name": config["checkpoint"]["name"],
            "size_bytes": loaded.size_bytes,
            "sha256": loaded.sha256,
            "tensor_sha256": loaded.tensor_sha256,
            "global_step": loaded.global_step,
        }
    model.metadata["init"] = init
    model.metadata["transfer"] = report.as_dict()
    model.metadata.update(model_accounting(model))
    return model


def transfer_source_to_target(
    source_state: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
    *,
    geometry: str = "dsec_mot",
    rvt_root: Optional[str | Path] = None,
    source_name: str = "source-head state dict",
) -> LargeEventDetector:
    """Seven-class target model from a source-head state dict (fine-tuning start).

    Every tensor except the class rows and the embedding head is copied; target rows
    follow ``heads.target.from_source`` (``train`` and other unmapped rows stay new).
    Source rows without a target row (traffic sign and light) are dropped. The embedding
    head starts new (seeded): pretraining has no identity supervision, and running it
    in training mode would only have moved its BatchNorm statistics.
    """

    upstream = import_rvt(resolve_rvt_root(config, rvt_root), config["upstream"]["commit"])
    model = _new_model(config, upstream, "target", geometry)
    heads = config["heads"]
    source_classes = heads["source"]["classes"]
    rows = _rows_by_name(model.classes, source_classes, heads["target"]["from_source"])
    report = TransferReport(source=source_name)
    state = model.state_dict()
    used = _copy_class_rows(
        state, source_state, rows, report, _row_rule(model.classes, rows, source_classes)
    )
    for row, name in enumerate(source_classes):
        if row not in rows.values():
            report.dropped[f"{CLASS_PREDS_PREFIX}*[{row}]"] = f"source row {name} has no target"
    for key, tensor in source_state.items():
        if key in used:
            continue
        if key.startswith(EMBEDDING_PREFIX):
            report.dropped[key] = "embedding head starts new at fine-tuning"
            continue
        if key not in state:
            report.dropped[key] = "not in the target model"
            continue
        if state[key].shape != tensor.shape:
            raise UpstreamError(f"source tensor {key} {tuple(tensor.shape)} has no target match")
        state[key] = tensor.detach().clone()
        report.loaded.append(key)
        used.add(key)
    report.new = [key for key in state if key not in used]
    model.load_state_dict(state, strict=True)
    model.metadata["init"] = "source_transfer"
    model.metadata["checkpoint"] = None
    model.metadata["transfer"] = report.as_dict()
    model.metadata.update(model_accounting(model))
    return model


# --------------------------------------------------------------------------- CLI


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--rvt-root", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--init", choices=INITIALIZATIONS, default="rvt_checkpoint")
    parser.add_argument("--ontology", choices=ONTOLOGIES, default="source")
    parser.add_argument("--geometry", default="gen4_1mpx")
    parser.add_argument(
        "--smoke-forward", action="store_true", help="two zero-input CPU steps with carried state"
    )
    parser.add_argument("--report", default=None, help="write metadata JSON to this path")
    args = parser.parse_args(argv)

    config = load_model_config(args.config)
    model = build_large_event_detector(
        config,
        ontology=args.ontology,
        geometry=args.geometry,
        init=args.init,
        rvt_root=args.rvt_root,
        checkpoint=args.checkpoint,
    )
    summary = {
        key: model.metadata[key]
        for key in (
            "upstream",
            "ontology",
            "init",
            "checkpoint",
            "parameters",
            "training_only",
            "recurrent_state",
        )
    }
    summary["transfer_counts"] = model.metadata["transfer"]["counts"]
    if args.smoke_forward:
        model.eval()
        fusion = config["fusion"]
        height, width = model.geometry.input_hw
        inputs = (
            torch.zeros(1, fusion["event_frame_channels"], height, width),
            torch.zeros(1, fusion["voxel_grid_channels"], height, width),
        )
        with torch.no_grad():
            output = model(*inputs)
            output = model(*inputs, state=output.state)
        summary["smoke_forward"] = {
            "anchors": int(output.head.boxes.shape[1]),
            "fpn_shapes": [list(feature.shape) for feature in output.fpn_features],
            "embedding_shape": list(output.embeddings.shape),
            "state_floats": output.state.floats_per_stream(),
            "gate_mean": float(output.fusion.gate.mean()),
        }
    if args.report:
        Path(args.report).write_text(json.dumps(model.metadata, indent=2) + "\n", encoding="utf-8")
    json.dump(summary, sys.stdout, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
