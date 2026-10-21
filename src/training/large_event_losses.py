"""Stage-aware detection and identity losses for the large event MOT detector.

Configuration: ``configs/large_event_mot/training.json`` (``losses`` and the per-stage
``head`` / ``identity`` fields); classifier size and descriptor width come from the
``embedding`` section of ``configs/large_event_mot/model.json``.

Detection (both stages), per micro-batch of ``B`` clips and ``T`` steps:

- Only supervised steps (``valid`` and ``supervised``) contribute; unannotated steps,
  padded stream tails, and record-less frames are never background.
- Positive boxes enter upstream YOLOX SimOTA (``YOLOXHead.get_assignments``); class
  columns masked for the clip's dataset are replaced by ``masked_class_logit`` for the
  assignment only, so they add no cost. Ignore boxes never enter assignment.
- ``L_det = (iou_weight * sum_FG L_iou + sum_a w_a L_obj + sum_FG sum_k m_k L_cls_k)
  / max(num_FG, 1)``, with ``w_a = 0`` for non-foreground anchors whose centre
  ``((x_shift + 0.5) * stride, (y_shift + 0.5) * stride)`` lies inside an ignore box
  (inclusive) and ``m`` the dataset's class-column mask. ``num_FG`` is summed over the
  micro-batch. A supervised step without positives supervises background outside its
  ignore boxes.

Identity (stages whose ``identity.enabled`` is true, datasets in ``identity.datasets``):

- Identity-valid anchors are foreground anchors whose matched box has
  ``identity_valid``; their descriptors are the embedding-head rows at those anchors.
- ``L_ce``: cosine-softmax cross-entropy, ``logits = scale * normalize(W) e``, mean over
  identity-valid anchors (:class:`CosineIdentityClassifier`).
- ``L_tri``: one representative per ``(step, box)`` (its foreground anchor with the
  highest matching IoU), class-aware batch-hard cosine triplet with margin over the
  micro-batch, averaged over representatives with a positive and a negative.
- ``L = L_det + ce_weight * L_ce + triplet_weight * L_tri``.

Inactive terms (no supervised step, no identity-valid anchor, identity disabled) are
finite zeros connected to the graph and logged with zero counts. Identity labels from a
dataset outside the stage's identity list, positives in masked class columns, or a
classifier in a stage without identity supervision raise :class:`SupervisionError`, so
detection-only data never create pseudo-identity supervision.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch
from torch import nn
from torch.nn import functional as F

from src.models.large_event_detector import DetectorOutput, HeadOutput

DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "large_event_mot" / "training.json"
)
STAGES = ("pretrain", "finetune")
HEADS = ("source", "target")


class TrainingConfigError(ValueError):
    """The training configuration is inconsistent."""


class SupervisionError(ValueError):
    """A batch carries supervision that its stage or dataset does not allow."""


# --------------------------------------------------------------------------- settings


@dataclass(frozen=True)
class StageSupervision:
    stage: str
    head: str
    identity_enabled: bool
    identity_datasets: frozenset[str]


@dataclass(frozen=True)
class LossSettings:
    iou_loss_type: str
    iou_weight: float
    masked_class_logit: float
    identities: int
    embedding_dim: int
    classifier_scale: float
    classifier_init_std: float
    triplet_margin: float
    ce_weight: float
    triplet_weight: float
    stages: Mapping[str, StageSupervision]

    def stage(self, name: str) -> StageSupervision:
        if name not in self.stages:
            raise TrainingConfigError(f"unknown stage {name!r}, expected one of {STAGES}")
        return self.stages[name]


def load_training_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("schema_version") != 1:
        raise TrainingConfigError(
            f"{path}: unsupported training schema {config.get('schema_version')}"
        )
    return config


def _positive(value: Any, name: str, allow_zero: bool = False) -> float:
    number = float(value)
    if not (number >= 0.0 if allow_zero else number > 0.0):
        raise TrainingConfigError(f"{name} must be {'>= 0' if allow_zero else '> 0'}")
    return number


def loss_settings(
    training_config: Mapping[str, Any], model_config: Mapping[str, Any]
) -> LossSettings:
    """Validated loss settings from ``training.json`` and the model's embedding section."""

    detection = training_config["losses"]["detection"]
    identity = training_config["losses"]["identity"]
    embedding = model_config["embedding"]
    if detection["iou_loss_type"] not in ("iou", "giou"):
        raise TrainingConfigError(f"unknown IoU loss {detection['iou_loss_type']!r}")
    if float(detection["masked_class_logit"]) > -1e3:
        raise TrainingConfigError("masked_class_logit must be <= -1000 to add zero cost")
    if embedding["identity_classifier"].get("bias", False):
        raise TrainingConfigError("the cosine identity classifier has no bias")
    stages = {}
    for name in STAGES:
        stage = training_config["stages"][name]
        if stage["head"] not in HEADS:
            raise TrainingConfigError(f"{name}: unknown head {stage['head']!r}")
        enabled = bool(stage["identity"]["enabled"])
        datasets = frozenset(str(dataset) for dataset in stage["identity"]["datasets"])
        if enabled != bool(datasets):
            raise TrainingConfigError(
                f"{name}: identity supervision needs a dataset list exactly when enabled"
            )
        stages[name] = StageSupervision(name, stage["head"], enabled, datasets)
    return LossSettings(
        iou_loss_type=detection["iou_loss_type"],
        iou_weight=_positive(detection["iou_weight"], "iou_weight"),
        masked_class_logit=float(detection["masked_class_logit"]),
        identities=int(embedding["identity_classifier"]["identities"]),
        embedding_dim=int(embedding["dim"]),
        classifier_scale=_positive(identity["classifier"]["scale"], "classifier scale"),
        classifier_init_std=_positive(identity["classifier"]["init_std"], "classifier init_std"),
        triplet_margin=_positive(identity["triplet"]["margin"], "triplet margin", True),
        ce_weight=_positive(identity["ce_weight"], "ce_weight", True),
        triplet_weight=_positive(identity["triplet_weight"], "triplet_weight", True),
        stages=stages,
    )


# --------------------------------------------------------------------------- identity


class CosineIdentityClassifier(nn.Module):
    """Training-only identity classifier: ``scale * normalize(W) e`` without bias."""

    def __init__(
        self, identities: int, dim: int, scale: float, init_std: float, seed: int = 0
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.scale = float(scale)
        self.weight = nn.Parameter(torch.randn(identities, dim, generator=generator) * init_std)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.scale * F.linear(embeddings, F.normalize(self.weight, dim=1))


def build_identity_classifier(
    settings: LossSettings, stage: str, identities: int, seed: int = 0
) -> CosineIdentityClassifier:
    """A new classifier for an identity-supervised stage over ``identities`` labels.

    ``identities`` is the size of the stage's identity table and must equal the
    configured classifier size, so a changed training set cannot silently resize it.
    """

    if not settings.stage(stage).identity_enabled:
        raise SupervisionError(f"stage {stage!r} has no identity supervision")
    if identities != settings.identities:
        raise TrainingConfigError(
            f"identity table has {identities} identities, classifier is configured for "
            f"{settings.identities}"
        )
    return CosineIdentityClassifier(
        identities,
        settings.embedding_dim,
        settings.classifier_scale,
        settings.classifier_init_std,
        seed,
    )


def batch_hard_cosine_triplet(
    embeddings: torch.Tensor, identities: torch.Tensor, classes: torch.Tensor, margin: float
) -> tuple[torch.Tensor, dict[str, int]]:
    """Class-aware batch-hard cosine triplet over representatives with known identities.

    Positives share the identity (another row); negatives have the same class and a
    different identity. Rows without a positive or a negative are skipped.
    """

    count = embeddings.shape[0]
    normalized = F.normalize(embeddings, dim=1)
    distances = 1.0 - normalized @ normalized.t()
    same = identities[:, None] == identities[None, :]
    eye = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    positives = same & ~eye
    negatives = ~same & (classes[:, None] == classes[None, :])
    active = positives.any(dim=1) & negatives.any(dim=1)
    counts = {
        "triplet_anchors": int(active.sum()),
        "triplet_positive_pairs": int(positives.sum()),
        "triplet_negative_pairs": int(negatives.sum()),
    }
    if not bool(active.any()):
        return embeddings.sum() * 0.0, counts
    hardest_positive = distances.masked_fill(~positives, float("-inf")).amax(dim=1)
    hardest_negative = distances.masked_fill(~negatives, float("inf")).amin(dim=1)
    losses = F.relu(hardest_positive[active] - hardest_negative[active] + margin)
    return losses.mean(), counts


# --------------------------------------------------------------------------- losses


@dataclass
class LossOutput:
    """Total loss with detached per-term values and supervision counts."""

    total: torch.Tensor
    terms: dict[str, torch.Tensor]
    counts: dict[str, int]
    flags: dict[str, bool] = field(default_factory=dict)

    def log(self) -> dict[str, float | int | bool]:
        values: dict[str, float | int | bool] = {
            f"loss/{name}": float(value) for name, value in self.terms.items()
        }
        values.update({f"count/{name}": value for name, value in self.counts.items()})
        values.update({f"active/{name}": value for name, value in self.flags.items()})
        return values


@dataclass
class _Assigned:
    """Foreground assignment of one supervised step."""

    clip: int
    step: int
    fg_anchors: torch.Tensor  # (F,) anchor indices
    matched_gt: torch.Tensor  # (F,) index into the step's positive boxes
    matching_iou: torch.Tensor  # (F,)
    positives: Mapping[str, torch.Tensor]  # the step's positive boxes and labels


def _xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    return torch.cat(((boxes[:, :2] + boxes[:, 2:]) / 2, boxes[:, 2:] - boxes[:, :2]), dim=1)


def _graph_zero(*tensors: Optional[torch.Tensor]) -> torch.Tensor:
    """An exact zero connected to every given tensor, finite even if a tensor is not.

    Summing an empty slice keeps the graph edge without multiplying values by zero.
    """

    present = [tensor for tensor in tensors if tensor is not None]
    return sum(tensor.reshape(-1)[:0].sum() for tensor in present) + 0.0


def _check_batch(
    outputs: Sequence[DetectorOutput],
    batch: Mapping[str, Any],
    supervision: StageSupervision,
    classes: int,
) -> None:
    clips, steps = batch["valid"].shape
    if len(outputs) != steps:
        raise ValueError(f"{len(outputs)} outputs for clips of {steps} steps")
    for output in outputs:
        if output.head.cls_logits.shape[:1] != (clips,):
            raise ValueError("every output needs one row per clip")
        if output.head.cls_logits.shape[-1] != classes:
            raise ValueError("class mask and head disagree on the number of classes")
    for clip, meta in enumerate(batch["meta"]):
        if meta["head"] != supervision.head:
            raise SupervisionError(
                f"{meta['dataset']}/{meta['sequence']} uses the {meta['head']} head, stage "
                f"{supervision.stage} trains the {supervision.head} head"
            )
        for step, target in enumerate(batch["targets"][clip]):
            if not (batch["valid"][clip, step] and batch["supervised"][clip, step]):
                continue
            positive = ~target["ignore"]
            labels = target["labels"][positive]
            if labels.numel() and (labels.min() < 0 or labels.max() >= classes):
                raise SupervisionError(f"{meta['sequence']}: label outside the head")
            if labels.numel() and not bool(batch["class_mask"][clip][labels].all()):
                raise SupervisionError(f"{meta['sequence']}: positive box in a masked class column")
            identity_valid = target["identity_valid"]
            if not bool(identity_valid.any()):
                continue
            if not supervision.identity_enabled:
                raise SupervisionError(
                    f"{meta['sequence']}: identity labels in stage {supervision.stage}, "
                    "which has no identity supervision"
                )
            if meta["dataset"] not in supervision.identity_datasets:
                raise SupervisionError(
                    f"{meta['dataset']}/{meta['sequence']}: identity labels from a dataset "
                    "outside the stage's identity sources"
                )
            if bool((identity_valid & target["ignore"]).any()) or bool(
                (target["identity"][identity_valid] < 0).any()
            ):
                raise SupervisionError(f"{meta['sequence']}: malformed identity labels")


def _ignore_weights(head: HeadOutput, ignore_xyxy: torch.Tensor) -> torch.Tensor:
    """True for anchors whose centre lies inside any ignore box (inclusive)."""

    if ignore_xyxy.numel() == 0:
        return torch.zeros(head.grid.shape[0], dtype=torch.bool, device=head.grid.device)
    centres = (head.grid + 0.5) * head.strides[:, None]
    x, y = centres[:, :1], centres[:, 1:]
    inside = (
        (x >= ignore_xyxy[None, :, 0])
        & (x <= ignore_xyxy[None, :, 2])
        & (y >= ignore_xyxy[None, :, 1])
        & (y <= ignore_xyxy[None, :, 3])
    )
    return inside.any(dim=1)


def stage_losses(
    outputs: Sequence[DetectorOutput],
    batch: Mapping[str, Any],
    stage: str,
    settings: LossSettings,
    assigner: nn.Module,
    classifier: Optional[CosineIdentityClassifier] = None,
) -> LossOutput:
    """Losses of one micro-batch: ``outputs[t]`` is the model step ``t`` for all clips.

    ``batch`` is a :func:`src.data.large_event_dataset.collate_clips` batch on any
    device; ``assigner`` is the model's upstream ``YOLOXHead`` (SimOTA and IoU loss).
    """

    supervision = settings.stage(stage)
    if (classifier is not None) != supervision.identity_enabled:
        raise SupervisionError(
            f"stage {stage!r} {'needs' if supervision.identity_enabled else 'has no'} "
            "identity classifier"
        )
    classes = batch["class_mask"].shape[-1]
    _check_batch(outputs, batch, supervision, classes)
    if supervision.identity_enabled and any(output.embeddings is None for output in outputs):
        raise SupervisionError(f"stage {stage!r} needs embeddings at every step")
    if getattr(assigner.iou_loss, "loss_type", None) != settings.iou_loss_type:
        raise TrainingConfigError("the head's IoU loss differs from the configured type")

    device = outputs[0].head.obj_logits.device
    with torch.autocast(device_type=device.type, enabled=False):
        return _stage_losses(outputs, batch, supervision, settings, assigner, classifier)


def _stage_losses(
    outputs: Sequence[DetectorOutput],
    batch: Mapping[str, Any],
    supervision: StageSupervision,
    settings: LossSettings,
    assigner: nn.Module,
    classifier: Optional[CosineIdentityClassifier],
) -> LossOutput:
    device = outputs[0].head.obj_logits.device
    clips = batch["valid"].shape[0]
    classes = batch["class_mask"].shape[-1]
    counts = {
        "supervised_steps": 0,
        "positive_boxes": 0,
        "ignore_boxes": 0,
        "fg_anchors": 0,
        "ignore_masked_anchors": 0,
        "masked_class_columns": 0,
        "identity_valid_boxes": 0,
        "identity_valid_anchors": 0,
        "identity_instances": 0,
        "triplet_anchors": 0,
        "triplet_positive_pairs": 0,
        "triplet_negative_pairs": 0,
    }
    obj_losses, iou_losses, cls_losses = [], [], []
    assigned: list[_Assigned] = []

    for step, output in enumerate(outputs):
        head = output.head
        x_shifts, y_shifts = head.x_shifts, head.y_shifts
        expanded_strides = head.expanded_strides
        for clip in range(clips):
            if not (batch["valid"][clip, step] and batch["supervised"][clip, step]):
                continue
            target = batch["targets"][clip][step]
            class_mask = batch["class_mask"][clip].to(device)
            ignore = target["ignore"].to(device)
            positive_xyxy = target["boxes_xyxy"].to(device)[~ignore].float()
            positive_labels = target["labels"].to(device)[~ignore]
            counts["supervised_steps"] += 1
            counts["positive_boxes"] += int(positive_labels.numel())
            counts["ignore_boxes"] += int(ignore.sum())
            counts["masked_class_columns"] += int((~class_mask).sum())

            anchors = head.obj_logits.shape[1]
            fg_mask = torch.zeros(anchors, dtype=torch.bool, device=device)
            obj_target = torch.zeros(anchors, device=device)
            if positive_labels.numel():
                cls_logits = head.cls_logits[clip : clip + 1].detach()
                cls_logits = cls_logits.masked_fill(~class_mask, settings.masked_class_logit)
                gt_cxcywh = _xyxy_to_cxcywh(positive_xyxy)
                (_, fg_mask, matching_iou, matched_gt, num_fg) = assigner.get_assignments(
                    0,
                    int(positive_labels.numel()),
                    gt_cxcywh,
                    positive_labels.float(),
                    head.boxes[clip].detach(),
                    expanded_strides,
                    x_shifts,
                    y_shifts,
                    cls_logits,
                    head.obj_logits[clip : clip + 1, :, None].detach(),
                )
                fg_anchors = torch.nonzero(fg_mask).squeeze(1)
                counts["fg_anchors"] += int(num_fg)
                obj_target[fg_mask] = 1.0
                iou_losses.append(
                    assigner.iou_loss(head.boxes[clip][fg_anchors], gt_cxcywh[matched_gt])
                )
                matched_labels = positive_labels[matched_gt]
                cls_target = F.one_hot(matched_labels, classes).float() * matching_iou[:, None]
                cls_loss = F.binary_cross_entropy_with_logits(
                    head.cls_logits[clip][fg_anchors], cls_target, reduction="none"
                )
                cls_losses.append(cls_loss * class_mask.float())
                assigned.append(
                    _Assigned(
                        clip,
                        step,
                        fg_anchors,
                        matched_gt,
                        matching_iou,
                        {
                            "labels": positive_labels,
                            "identity": target["identity"].to(device)[~ignore],
                            "identity_valid": target["identity_valid"].to(device)[~ignore],
                        },
                    )
                )
            ignored = _ignore_weights(head, target["boxes_xyxy"].to(device)[ignore].float())
            ignored &= ~fg_mask
            counts["ignore_masked_anchors"] += int(ignored.sum())
            obj_loss = F.binary_cross_entropy_with_logits(
                head.obj_logits[clip], obj_target, reduction="none"
            )
            obj_losses.append(obj_loss * (~ignored).float())

    head_tensors = [
        tensor
        for output in outputs
        for tensor in (output.head.boxes, output.head.obj_logits, output.head.cls_logits)
    ]
    terms: dict[str, torch.Tensor] = {}
    if counts["supervised_steps"]:
        normalizer = float(max(counts["fg_anchors"], 1))
        loss_obj = torch.cat(obj_losses).sum() / normalizer
        loss_iou = (
            torch.cat(iou_losses).sum() / normalizer if iou_losses else _graph_zero(*head_tensors)
        )
        loss_cls = (
            torch.cat(cls_losses).sum() / normalizer if cls_losses else _graph_zero(*head_tensors)
        )
        detection = settings.iou_weight * loss_iou + loss_obj + loss_cls
    else:
        loss_obj = loss_iou = loss_cls = detection = _graph_zero(*head_tensors)
    terms.update(
        {
            "iou": loss_iou,
            "iou_weighted": settings.iou_weight * loss_iou,
            "obj": loss_obj,
            "cls": loss_cls,
            "detection": detection,
        }
    )

    identity_active = triplet_active = False
    if supervision.identity_enabled:
        embedding_tensors = [output.embeddings for output in outputs]
        idle = _graph_zero(*embedding_tensors, classifier.weight)
        anchor_rows, anchor_identities = [], []
        instance_rows, instance_identities, instance_classes = [], [], []
        for item in assigned:
            valid_gt = item.positives["identity_valid"]
            if not bool(valid_gt.any()):
                continue
            counts["identity_valid_boxes"] += int(valid_gt.sum())
            embeddings = outputs[item.step].embeddings[item.clip]
            anchor_valid = valid_gt[item.matched_gt]
            anchor_rows.append(embeddings[item.fg_anchors[anchor_valid]])
            anchor_identities.append(item.positives["identity"][item.matched_gt[anchor_valid]])
            for gt in torch.nonzero(valid_gt).squeeze(1).tolist():
                matched = item.matched_gt == gt
                if not bool(matched.any()):
                    continue
                ious = item.matching_iou.masked_fill(~matched, float("-inf"))
                instance_rows.append(embeddings[item.fg_anchors[torch.argmax(ious)]])
                instance_identities.append(item.positives["identity"][gt])
                instance_classes.append(item.positives["labels"][gt])
        if anchor_rows:
            rows = torch.cat(anchor_rows)
            labels = torch.cat(anchor_identities)
            if int(labels.max()) >= classifier.weight.shape[0]:
                raise SupervisionError("identity index outside the identity classifier")
            counts["identity_valid_anchors"] = int(rows.shape[0])
            identity_active = rows.shape[0] > 0
            loss_ce = F.cross_entropy(classifier(rows.float()), labels) if identity_active else idle
        else:
            loss_ce = idle
        counts["identity_instances"] = len(instance_rows)
        if instance_rows:
            loss_tri, triplet_counts = batch_hard_cosine_triplet(
                torch.stack(instance_rows).float(),
                torch.stack(instance_identities),
                torch.stack(instance_classes),
                settings.triplet_margin,
            )
            counts.update(triplet_counts)
            triplet_active = triplet_counts["triplet_anchors"] > 0
            if not triplet_active:
                loss_tri = idle
        else:
            loss_tri = idle
    else:
        loss_ce = loss_tri = _graph_zero(*head_tensors)

    total = detection + settings.ce_weight * loss_ce + settings.triplet_weight * loss_tri
    terms.update(
        {
            "identity_ce": loss_ce,
            "identity_ce_weighted": settings.ce_weight * loss_ce,
            "triplet": loss_tri,
            "triplet_weighted": settings.triplet_weight * loss_tri,
            "total": total,
        }
    )
    return LossOutput(
        total=total,
        terms={name: value.detach() for name, value in terms.items()},
        counts=counts,
        flags={
            "detection": counts["supervised_steps"] > 0,
            "identity_ce": identity_active,
            "triplet": triplet_active,
        },
    )
