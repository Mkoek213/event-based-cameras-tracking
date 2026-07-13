"""Small dense detector for controlled event-representation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.evaluation.detection_export import DetectionRecord


@dataclass(frozen=True)
class SimpleDetectorConfig:
    in_channels: int
    num_classes: int = 7
    feature_stride: int = 8
    width: int = 32
    architecture: str = "simple"
    fusion_mode: str = "single"
    event_frame_channels: int = 2
    voxel_grid_channels: int = 0
    component_channels: tuple[int, ...] = ()
    embedding_dim: int = 0
    embedding_recurrent: bool = False
    embedding_hidden_dim: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell keeping a spatial hidden state across frames."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.update_gate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.reset_gate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )
        self.candidate = nn.Conv2d(
            input_channels + hidden_channels, hidden_channels, kernel_size=3, padding=1
        )

    def forward(self, x: torch.Tensor, hidden: torch.Tensor | None = None) -> torch.Tensor:
        if hidden is None:
            hidden = torch.zeros(
                x.shape[0],
                self.hidden_channels,
                x.shape[2],
                x.shape[3],
                device=x.device,
                dtype=x.dtype,
            )
        combined = torch.cat([hidden, x], dim=1)
        update = torch.sigmoid(self.update_gate(combined))
        reset = torch.sigmoid(self.reset_gate(combined))
        candidate = torch.tanh(self.candidate(torch.cat([reset * hidden, x], dim=1)))
        return (1.0 - update) * hidden + update * candidate


class ResidualBlock(nn.Module):
    """Small residual block used by the stronger detector backbone."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.layers(x)


class CSPBlock(nn.Module):
    """Compact CSP-style block inspired by YOLO/CSPDarknet backbones."""

    def __init__(self, channels: int, num_blocks: int) -> None:
        super().__init__()
        hidden = max(channels // 2, 8)
        self.left = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
            *[ResidualBlock(hidden) for _ in range(num_blocks)],
        )
        self.right = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.SiLU(inplace=True),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(2 * hidden, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([self.left(x), self.right(x)], dim=1))


class CSPStage(nn.Sequential):
    """Downsample then process features with a CSP block."""

    def __init__(self, in_channels: int, out_channels: int, num_blocks: int) -> None:
        super().__init__(
            ConvBlock(in_channels, out_channels, stride=2),
            CSPBlock(out_channels, num_blocks=num_blocks),
        )


class CSPPANBackbone(nn.Module):
    """CSPDarknet/PAN-style backbone that returns stride-8 features."""

    def __init__(self, in_channels: int, width: int, depth: int = 2) -> None:
        super().__init__()
        depth = max(depth, 1)
        self.stem = ConvBlock(in_channels, width)
        self.stage2 = CSPStage(width, width, num_blocks=depth)
        self.stage4 = CSPStage(width, 2 * width, num_blocks=depth)
        self.stage8 = CSPStage(2 * width, 4 * width, num_blocks=depth + 1)
        self.stage16 = CSPStage(4 * width, 8 * width, num_blocks=depth + 1)

        self.lateral16 = nn.Sequential(
            nn.Conv2d(8 * width, 4 * width, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * width),
            nn.SiLU(inplace=True),
        )
        self.fuse8 = CSPBlock(8 * width, num_blocks=depth)
        self.output = nn.Sequential(
            nn.Conv2d(8 * width, 4 * width, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * width),
            nn.SiLU(inplace=True),
            ConvBlock(4 * width, 4 * width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        c2 = self.stage2(x)
        c4 = self.stage4(c2)
        c8 = self.stage8(c4)
        c16 = self.stage16(c8)
        up16 = F.interpolate(self.lateral16(c16), size=c8.shape[-2:], mode="nearest")
        return self.output(self.fuse8(torch.cat([c8, up16], dim=1)))


class SimpleDenseDetector(nn.Module):
    """A YOLO-like dense detector with a fixed stride-8 output grid.

    The model is intentionally small. It is meant to compare event input
    representations under a controlled architecture, not to be a SOTA detector.
    """

    def __init__(self, config: SimpleDetectorConfig) -> None:
        super().__init__()
        if config.feature_stride != 8:
            raise ValueError("SimpleDenseDetector currently supports feature_stride=8 only.")
        if config.architecture not in ("simple", "csp_pan"):
            raise ValueError(f"Unknown architecture '{config.architecture}'.")
        if config.fusion_mode not in (
            "single",
            "two_branch",
            "three_branch",
            "gated_two_branch",
        ):
            raise ValueError(f"Unknown fusion_mode '{config.fusion_mode}'.")
        self.config = config
        w = config.width
        backbone_in_channels = config.in_channels
        self.input_stems = None
        self.gate = None
        self.component_channels: tuple[int, ...] = ()
        if config.component_channels:
            component_channels = tuple(config.component_channels)
            expected_branches = {
                "two_branch": 2,
                "three_branch": 3,
                "gated_two_branch": 2,
            }.get(config.fusion_mode)
            if expected_branches is None:
                raise ValueError("component_channels requires a multi-branch fusion mode.")
            if len(component_channels) != expected_branches:
                raise ValueError(
                    f"{config.fusion_mode} requires {expected_branches} component channel splits, "
                    f"got {component_channels}."
                )
            if any(channels <= 0 for channels in component_channels):
                raise ValueError("All component channel counts must be positive.")
            if sum(component_channels) != config.in_channels:
                raise ValueError(
                    "Component channel split does not match in_channels: "
                    f"{sum(component_channels)} != {config.in_channels}."
                )
            branch_width = max(w // expected_branches, 4)
            self.component_channels = component_channels
            self.input_stems = nn.ModuleList(
                ConvBlock(channels, branch_width) for channels in component_channels
            )
            self.fusion = nn.Sequential(
                nn.Conv2d(expected_branches * branch_width, w, kernel_size=1, bias=False),
                nn.BatchNorm2d(w),
                nn.SiLU(inplace=True),
            )
            if config.fusion_mode == "gated_two_branch":
                hidden = max(branch_width, 4)
                self.gate = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(expected_branches * branch_width, hidden),
                    nn.SiLU(inplace=True),
                    nn.Linear(hidden, expected_branches),
                    nn.Sigmoid(),
                )
            self.event_frame_channels = 0
            self.voxel_grid_channels = 0
            self.event_frame_stem = None
            self.voxel_grid_stem = None
            backbone_in_channels = w
        elif config.fusion_mode == "two_branch":
            # Legacy EF+VG path retained so existing checkpoints remain loadable.
            ef_channels = config.event_frame_channels
            vg_channels = config.voxel_grid_channels or (config.in_channels - ef_channels)
            if ef_channels <= 0 or vg_channels <= 0:
                raise ValueError(
                    "two_branch fusion requires positive event-frame and voxel-grid channel counts."
                )
            if ef_channels + vg_channels != config.in_channels:
                raise ValueError(
                    "two_branch fusion channel split does not match in_channels: "
                    f"{ef_channels} + {vg_channels} != {config.in_channels}."
                )
            branch_width = max(w // 2, 4)
            self.event_frame_channels = ef_channels
            self.voxel_grid_channels = vg_channels
            self.event_frame_stem = ConvBlock(ef_channels, branch_width)
            self.voxel_grid_stem = ConvBlock(vg_channels, branch_width)
            self.fusion = nn.Sequential(
                nn.Conv2d(2 * branch_width, w, kernel_size=1, bias=False),
                nn.BatchNorm2d(w),
                nn.SiLU(inplace=True),
            )
            backbone_in_channels = w
        else:
            self.event_frame_channels = 0
            self.voxel_grid_channels = 0
            self.event_frame_stem = None
            self.voxel_grid_stem = None
            self.fusion = None
            self.gate = None

        if config.architecture == "simple":
            self.backbone = nn.Sequential(
                ConvBlock(backbone_in_channels, w, stride=2),
                ConvBlock(w, w),
                ConvBlock(w, 2 * w, stride=2),
                ConvBlock(2 * w, 2 * w),
                ConvBlock(2 * w, 4 * w, stride=2),
                ConvBlock(4 * w, 4 * w),
                ConvBlock(4 * w, 4 * w),
            )
        else:
            self.backbone = CSPPANBackbone(backbone_in_channels, w)
        self.cls_head = nn.Sequential(
            ConvBlock(4 * w, 4 * w),
            nn.Conv2d(4 * w, config.num_classes + 1, kernel_size=1),
        )
        self.bbox_head = nn.Sequential(
            ConvBlock(4 * w, 4 * w),
            nn.Conv2d(4 * w, 4, kernel_size=1),
        )

        if config.embedding_dim > 0:
            hidden = config.embedding_hidden_dim or config.embedding_dim
            self.embedding_proj = ConvBlock(4 * w, hidden)
            self.embedding_recurrent_cell = (
                ConvGRUCell(hidden, hidden) if config.embedding_recurrent else None
            )
            self.embedding_head = nn.Conv2d(hidden, config.embedding_dim, kernel_size=1)
        else:
            self.embedding_proj = None
            self.embedding_recurrent_cell = None
            self.embedding_head = None

    def forward(
        self, x: torch.Tensor, embedding_state: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        if self.input_stems is not None:
            components = torch.split(x, self.component_channels, dim=1)
            features = [stem(component) for stem, component in zip(self.input_stems, components)]
            if self.gate is not None:
                weights = self.gate(torch.cat(features, dim=1))
                features = [
                    feature * weights[:, index].view(-1, 1, 1, 1)
                    for index, feature in enumerate(features)
                ]
            x = self.fusion(torch.cat(features, dim=1))
        elif self.config.fusion_mode == "two_branch":
            event_frame = x[:, : self.event_frame_channels]
            voxel_grid = x[:, self.event_frame_channels :]
            event_frame_features = self.event_frame_stem(event_frame)
            voxel_grid_features = self.voxel_grid_stem(voxel_grid)
            x = self.fusion(torch.cat([event_frame_features, voxel_grid_features], dim=1))
        features = self.backbone(x)
        outputs = {
            "cls_logits": self.cls_head(features),
            "bbox_raw": self.bbox_head(features),
        }
        if self.embedding_head is not None:
            embedding_features = self.embedding_proj(features)
            if self.embedding_recurrent_cell is not None:
                embedding_state = self.embedding_recurrent_cell(embedding_features, embedding_state)
                embeddings = self.embedding_head(embedding_state)
            else:
                embeddings = self.embedding_head(embedding_features)
            outputs["embeddings"] = F.normalize(embeddings, dim=1)
            outputs["embedding_state"] = embedding_state
        return outputs

    @staticmethod
    def bbox_distances(bbox_raw: torch.Tensor) -> torch.Tensor:
        return F.softplus(bbox_raw) + 1e-3


def normalise_event_tensor(events: torch.Tensor) -> torch.Tensor:
    """Log-compress and scale each event tensor independently."""
    events = torch.log1p(events.clamp_min(0.0))
    flat = events.flatten(start_dim=1)
    scale = flat.amax(dim=1).clamp_min(1.0).view(-1, 1, 1, 1)
    return events / scale


def normalise_representation_tensor(
    events: torch.Tensor, component_channels: tuple[int, ...] | list[int] = ()
) -> torch.Tensor:
    """Normalise each representation component independently when fused."""
    splits = tuple(component_channels)
    if not splits:
        return normalise_event_tensor(events)
    if sum(splits) != events.shape[1]:
        raise ValueError(
            f"Component channel split {splits} does not match tensor "
            f"with {events.shape[1]} channels."
        )
    normalised: list[torch.Tensor] = []
    for component in torch.split(events, splits, dim=1):
        component = torch.log1p(component.clamp_min(0.0))
        scale = component.flatten(start_dim=1).amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        normalised.append(component / scale)
    return torch.cat(normalised, dim=1)


def simple_detector_loss(
    outputs: dict[str, torch.Tensor],
    cls_targets: torch.Tensor,
    bbox_targets: torch.Tensor,
    pos_mask: torch.Tensor,
    background_weight: float = 0.05,
    bbox_weight: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    cls_logits = outputs["cls_logits"]
    bbox_pred = SimpleDenseDetector.bbox_distances(outputs["bbox_raw"])

    weight = torch.ones(cls_logits.shape[1], device=cls_logits.device)
    weight[0] = background_weight
    cls_loss = F.cross_entropy(cls_logits, cls_targets.long(), weight=weight)

    if pos_mask.any():
        pred_pos = bbox_pred.permute(0, 2, 3, 1)[pos_mask]
        target_pos = bbox_targets.permute(0, 2, 3, 1)[pos_mask]
        bbox_loss = F.smooth_l1_loss(pred_pos, target_pos)
    else:
        bbox_loss = bbox_pred.sum() * 0.0

    loss = cls_loss + bbox_weight * bbox_loss
    stats = {
        "loss": float(loss.detach().cpu()),
        "cls_loss": float(cls_loss.detach().cpu()),
        "bbox_loss": float(bbox_loss.detach().cpu()),
        "positive_cells": int(pos_mask.sum().detach().cpu()),
    }
    return loss, stats


def _box_iou_xyxy(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    x0 = torch.maximum(box[0], boxes[:, 0])
    y0 = torch.maximum(box[1], boxes[:, 1])
    x1 = torch.minimum(box[2], boxes[:, 2])
    y1 = torch.minimum(box[3], boxes[:, 3])
    inter = (x1 - x0).clamp_min(0) * (y1 - y0).clamp_min(0)
    area_box = (box[2] - box[0]).clamp_min(0) * (box[3] - box[1]).clamp_min(0)
    area_boxes = (boxes[:, 2] - boxes[:, 0]).clamp_min(0) * (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
    return inter / (area_box + area_boxes - inter).clamp_min(1e-6)


def _class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
    max_detections: int,
) -> list[int]:
    keep: list[int] = []
    for class_id in labels.unique(sorted=True):
        idxs = torch.nonzero(labels == class_id, as_tuple=False).flatten()
        idxs = idxs[scores[idxs].argsort(descending=True)]
        while idxs.numel() > 0 and len(keep) < max_detections:
            current = int(idxs[0])
            keep.append(current)
            if idxs.numel() == 1:
                break
            ious = _box_iou_xyxy(boxes[current], boxes[idxs[1:]])
            idxs = idxs[1:][ious <= iou_threshold]
    keep.sort(key=lambda idx: float(scores[idx]), reverse=True)
    return keep[:max_detections]


@torch.inference_mode()
def decode_dense_detections(
    outputs: dict[str, torch.Tensor],
    frame_index: int,
    timestamp: int,
    score_threshold: float = 0.25,
    nms_iou_threshold: float = 0.5,
    max_detections: int = 100,
    image_width: int = EVENT_WIDTH,
    image_height: int = EVENT_HEIGHT,
    feature_stride: int = 8,
    embeddings: torch.Tensor | None = None,
) -> list[DetectionRecord]:
    """Decode one model output into DetectionRecord rows.

    When ``embeddings`` is given (``(D, H, W)`` cell embeddings), each detection
    carries the embedding vector of the feature cell that produced it.
    """
    cls_logits = outputs["cls_logits"]
    bbox_raw = outputs["bbox_raw"]
    if cls_logits.ndim == 4:
        cls_logits = cls_logits[0]
    if bbox_raw.ndim == 4:
        bbox_raw = bbox_raw[0]
    if embeddings is not None and embeddings.ndim == 4:
        embeddings = embeddings[0]

    probabilities = cls_logits.softmax(dim=0)
    foreground = probabilities[1:]
    scores, labels = foreground.max(dim=0)
    mask = scores >= score_threshold
    if not mask.any():
        return []

    ys, xs = torch.nonzero(mask, as_tuple=True)
    labels = labels[ys, xs]
    scores = scores[ys, xs]
    distances = SimpleDenseDetector.bbox_distances(bbox_raw)[:, ys, xs] * feature_stride

    centers_x = (xs.float() + 0.5) * feature_stride
    centers_y = (ys.float() + 0.5) * feature_stride
    left = (centers_x - distances[0]).clamp(0, image_width - 1)
    top = (centers_y - distances[1]).clamp(0, image_height - 1)
    right = (centers_x + distances[2]).clamp(0, image_width - 1)
    bottom = (centers_y + distances[3]).clamp(0, image_height - 1)
    boxes = torch.stack([left, top, right, bottom], dim=1)

    valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    if not valid.any():
        return []
    boxes = boxes[valid]
    scores = scores[valid]
    labels = labels[valid]
    ys = ys[valid]
    xs = xs[valid]

    keep = _class_aware_nms(boxes, scores, labels, nms_iou_threshold, max_detections)
    detections: list[DetectionRecord] = []
    for idx in keep:
        box = boxes[idx].detach().cpu()
        score = float(scores[idx].detach().cpu())
        class_id = int(labels[idx].detach().cpu())
        embedding = None
        if embeddings is not None:
            vector = embeddings[:, ys[idx], xs[idx]].detach().cpu()
            embedding = tuple(float(value) for value in vector)
        detections.append(
            DetectionRecord(
                frame_index=frame_index,
                timestamp=timestamp,
                class_id=class_id,
                score=score,
                bbox_left=float(box[0]),
                bbox_top=float(box[1]),
                bbox_width=float(box[2] - box[0]),
                bbox_height=float(box[3] - box[1]),
                embedding=embedding,
            )
        )
    return detections
