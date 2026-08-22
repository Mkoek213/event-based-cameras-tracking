"""Small dense detector for controlled event-representation benchmarks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import roi_align

from src.data.dataset import EVENT_HEIGHT, EVENT_WIDTH
from src.evaluation.detection_export import DetectionRecord

TEMPORAL_RECURRENCE_LOCATIONS = (
    "backbone_s2",
    "backbone_s4",
    "neck",
    "detection_heads",
    "embedding",
)
TEMPORAL_RECURRENCE_TYPES = ("convgru", "convlstm", "convrnn")
TEMPORAL_RECURRENCE_MODES = ("residual", "direct")


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
    embedding_head_type: str = "roi"
    embedding_roi_size: int = 7
    temporal_recurrence_locations: tuple[str, ...] = ()
    temporal_recurrence_type: str = "convgru"
    temporal_recurrence_mode: str = "residual"

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


class ConvLSTMCell(nn.Module):
    """Convolutional LSTM cell with spatial hidden and cell states."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            input_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size=3,
            padding=1,
        )

    def forward(
        self,
        x: torch.Tensor,
        state: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if state is None:
            shape = (x.shape[0], self.hidden_channels, x.shape[2], x.shape[3])
            hidden = torch.zeros(shape, device=x.device, dtype=x.dtype)
            cell = torch.zeros_like(hidden)
        else:
            hidden, cell = state
        input_gate, forget_gate, output_gate, candidate = self.gates(
            torch.cat([hidden, x], dim=1)
        ).chunk(4, dim=1)
        input_gate = torch.sigmoid(input_gate)
        forget_gate = torch.sigmoid(forget_gate)
        output_gate = torch.sigmoid(output_gate)
        candidate = torch.tanh(candidate)
        cell = forget_gate * cell + input_gate * candidate
        hidden = output_gate * torch.tanh(cell)
        return hidden, cell


class ConvRNNCell(nn.Module):
    """Simple tanh convolutional Elman cell."""

    def __init__(self, input_channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.transition = nn.Conv2d(
            input_channels + hidden_channels,
            hidden_channels,
            kernel_size=3,
            padding=1,
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
        return torch.tanh(self.transition(torch.cat([hidden, x], dim=1)))


TemporalCellState = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


class RecurrentFeatureAdapter(nn.Module):
    """Apply a recurrent cell directly or as an identity-initialised residual."""

    def __init__(self, channels: int, cell_type: str, mode: str) -> None:
        super().__init__()
        if cell_type not in TEMPORAL_RECURRENCE_TYPES:
            raise ValueError(f"Unknown temporal recurrence type {cell_type!r}.")
        if mode not in TEMPORAL_RECURRENCE_MODES:
            raise ValueError(f"Unknown temporal recurrence mode {mode!r}.")
        cell_class = {
            "convgru": ConvGRUCell,
            "convlstm": ConvLSTMCell,
            "convrnn": ConvRNNCell,
        }[cell_type]
        self.cell_type = cell_type
        self.mode = mode
        self.cell = cell_class(channels, channels)
        self.residual_gain = nn.Parameter(torch.zeros(())) if mode == "residual" else None

    def forward(
        self,
        x: torch.Tensor,
        state: TemporalCellState | None = None,
    ) -> tuple[torch.Tensor, TemporalCellState]:
        if self.cell_type == "convlstm":
            next_state = self.cell(x, state)
            recurrent_features = next_state[0]
        else:
            next_state = self.cell(x, state)
            recurrent_features = next_state
        if self.residual_gain is not None:
            recurrent_features = x + torch.tanh(self.residual_gain) * recurrent_features
        return recurrent_features, next_state


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
        temporal_locations = tuple(config.temporal_recurrence_locations)
        unknown_locations = sorted(set(temporal_locations) - set(TEMPORAL_RECURRENCE_LOCATIONS))
        if unknown_locations:
            raise ValueError(f"Unknown temporal recurrence locations: {unknown_locations}.")
        if len(set(temporal_locations)) != len(temporal_locations):
            raise ValueError("Temporal recurrence locations must be unique.")
        if config.temporal_recurrence_type not in TEMPORAL_RECURRENCE_TYPES:
            raise ValueError(
                f"Unknown temporal recurrence type {config.temporal_recurrence_type!r}."
            )
        if config.temporal_recurrence_mode not in TEMPORAL_RECURRENCE_MODES:
            raise ValueError(
                f"Unknown temporal recurrence mode {config.temporal_recurrence_mode!r}."
            )
        if config.architecture != "simple" and {
            "backbone_s2",
            "backbone_s4",
        }.intersection(temporal_locations):
            raise ValueError("Stride-2/4 temporal adapters require architecture='simple'.")
        if config.embedding_recurrent and "embedding" in temporal_locations:
            raise ValueError(
                "Legacy embedding recurrence and the temporal embedding adapter "
                "cannot be enabled together."
            )
        if "embedding" in temporal_locations and config.embedding_dim <= 0:
            raise ValueError("Embedding recurrence requires embedding_dim > 0.")
        self.temporal_recurrence_locations = temporal_locations
        self.config = config
        self._detector_frozen = False
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

        if config.embedding_head_type not in ("roi", "dense"):
            raise ValueError(f"Unknown embedding_head_type {config.embedding_head_type!r}.")
        if config.embedding_dim > 0:
            hidden = config.embedding_hidden_dim or config.embedding_dim
            if config.embedding_head_type == "roi" and config.embedding_roi_size <= 0:
                raise ValueError("embedding_roi_size must be positive.")
            self.embedding_proj = ConvBlock(4 * w, hidden)
            self.embedding_recurrent_cell = (
                ConvGRUCell(hidden, hidden) if config.embedding_recurrent else None
            )
            if config.embedding_head_type == "dense":
                self.embedding_head = nn.Conv2d(hidden, config.embedding_dim, kernel_size=1)
                self.embedding_bn = None
            else:
                self.embedding_head = nn.Linear(hidden, config.embedding_dim)
                self.embedding_bn = nn.BatchNorm1d(config.embedding_dim)
        else:
            self.embedding_proj = None
            self.embedding_recurrent_cell = None
            self.embedding_head = None
            self.embedding_bn = None

        adapter_type = config.temporal_recurrence_type
        adapter_mode = config.temporal_recurrence_mode
        adapter_channels = {
            "backbone_s2": w,
            "backbone_s4": 2 * w,
            "neck": 4 * w,
            "detection_cls": 4 * w,
            "detection_bbox": 4 * w,
            "embedding": config.embedding_hidden_dim or config.embedding_dim,
        }
        adapter_keys: list[str] = []
        for location in temporal_locations:
            if location == "detection_heads":
                adapter_keys.extend(("detection_cls", "detection_bbox"))
            else:
                adapter_keys.append(location)
        self.temporal_adapters = nn.ModuleDict(
            {
                key: RecurrentFeatureAdapter(
                    adapter_channels[key],
                    cell_type=adapter_type,
                    mode=adapter_mode,
                )
                for key in adapter_keys
            }
        )

    def detector_temporal_modules(self) -> tuple[nn.Module, ...]:
        """Return temporal adapters that modify detector features or predictions."""

        return tuple(
            adapter for key, adapter in self.temporal_adapters.items() if key != "embedding"
        )

    def temporal_modules(self) -> tuple[nn.Module, ...]:
        """Return every configurable temporal adapter."""

        return tuple(self.temporal_adapters.values())

    def detector_modules(self) -> tuple[nn.Module, ...]:
        """Return modules belonging to the detector rather than the ReID head."""

        modules = (
            self.input_stems,
            self.event_frame_stem,
            self.voxel_grid_stem,
            self.fusion,
            self.gate,
            self.backbone,
            self.cls_head,
            self.bbox_head,
            *self.detector_temporal_modules(),
        )
        return tuple(module for module in modules if isinstance(module, nn.Module))

    def set_detector_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze detector weights while leaving the ReID head untouched."""

        for module in self.detector_modules():
            module.requires_grad_(trainable)
        self._detector_frozen = not trainable
        if self._detector_frozen:
            for module in self.detector_modules():
                module.eval()

    def set_temporal_trainable(self, trainable: bool) -> None:
        """Independently enable adapters after freezing the pre-trained detector."""

        for module in self.temporal_modules():
            module.requires_grad_(trainable)

    def train(self, mode: bool = True) -> SimpleDenseDetector:
        """Keep frozen detector BatchNorm statistics fixed during head training."""

        super().train(mode)
        if mode and self._detector_frozen:
            for module in self.detector_modules():
                module.eval()
        return self

    def forward(
        self,
        x: torch.Tensor,
        embedding_state: torch.Tensor | None = None,
        temporal_state: dict[str, TemporalCellState] | None = None,
    ) -> dict[str, object]:
        current_temporal_state = temporal_state or {}
        next_temporal_state: dict[str, TemporalCellState] = {}

        def apply_temporal(key: str, features: torch.Tensor) -> torch.Tensor:
            adapter = self.temporal_adapters[key] if key in self.temporal_adapters else None
            if adapter is None:
                return features
            features, next_state = adapter(features, current_temporal_state.get(key))
            next_temporal_state[key] = next_state
            return features

        if self.input_stems is not None:
            components = torch.split(x, self.component_channels, dim=1)
            fused_features = [
                stem(component) for stem, component in zip(self.input_stems, components)
            ]
            if self.gate is not None:
                weights = self.gate(torch.cat(fused_features, dim=1))
                fused_features = [
                    feature * weights[:, index].view(-1, 1, 1, 1)
                    for index, feature in enumerate(fused_features)
                ]
            x = self.fusion(torch.cat(fused_features, dim=1))
        elif self.config.fusion_mode == "two_branch":
            event_frame = x[:, : self.event_frame_channels]
            voxel_grid = x[:, self.event_frame_channels :]
            event_frame_features = self.event_frame_stem(event_frame)
            voxel_grid_features = self.voxel_grid_stem(voxel_grid)
            x = self.fusion(torch.cat([event_frame_features, voxel_grid_features], dim=1))

        if self.config.architecture == "simple":
            for index, block in enumerate(self.backbone):
                x = block(x)
                if index == 1:
                    x = apply_temporal("backbone_s2", x)
                elif index == 3:
                    x = apply_temporal("backbone_s4", x)
            features = x
        else:
            features = self.backbone(x)
        features = apply_temporal("neck", features)

        cls_features = self.cls_head[0](features)
        cls_features = apply_temporal("detection_cls", cls_features)
        bbox_features = self.bbox_head[0](features)
        bbox_features = apply_temporal("detection_bbox", bbox_features)
        outputs: dict[str, object] = {
            "cls_logits": self.cls_head[1](cls_features),
            "bbox_raw": self.bbox_head[1](bbox_features),
        }

        if self.embedding_proj is not None:
            embedding_features = self.embedding_proj(features)
            if self.embedding_recurrent_cell is not None:
                embedding_state = self.embedding_recurrent_cell(embedding_features, embedding_state)
                embedding_features = embedding_state
            else:
                embedding_state = None
            embedding_features = apply_temporal("embedding", embedding_features)
            if self.config.embedding_head_type == "dense":
                outputs["embeddings"] = F.normalize(self.embedding_head(embedding_features), dim=1)
            else:
                outputs["embedding_feature_map"] = embedding_features
            outputs["embedding_state"] = embedding_state
        if self.temporal_recurrence_locations:
            outputs["temporal_state"] = next_temporal_state
        return outputs

    def project_roi_embeddings(
        self,
        feature_map: torch.Tensor,
        boxes_per_image: list[torch.Tensor],
    ) -> torch.Tensor:
        """Project RoIs into pre-BN embedding vectors, preserving supplied box order."""

        if self.embedding_head is None or self.config.embedding_head_type != "roi":
            raise RuntimeError("RoI embeddings require an enabled RoI embedding head.")
        if feature_map.ndim != 4:
            raise ValueError("feature_map must have shape (B, C, H, W).")
        if len(boxes_per_image) != feature_map.shape[0]:
            raise ValueError(
                "boxes_per_image length must match the feature-map batch dimension: "
                f"{len(boxes_per_image)} != {feature_map.shape[0]}."
            )
        boxes = [
            box.to(device=feature_map.device, dtype=feature_map.dtype).reshape(-1, 4)
            for box in boxes_per_image
        ]
        roi_count = sum(int(box.shape[0]) for box in boxes)
        if roi_count == 0:
            return feature_map.new_empty((0, self.config.embedding_dim))

        pooled = roi_align(
            feature_map,
            boxes,
            output_size=(self.config.embedding_roi_size, self.config.embedding_roi_size),
            spatial_scale=1.0 / self.config.feature_stride,
            aligned=True,
        )
        return self.embedding_head(pooled.mean(dim=(-2, -1)))

    def apply_embedding_bn(self, vectors: torch.Tensor) -> torch.Tensor:
        """Apply the BN neck, safely handling empty and single-RoI training batches."""

        if self.embedding_bn is None or self.config.embedding_head_type != "roi":
            raise RuntimeError("RoI embeddings require an enabled embedding BN neck.")
        if vectors.ndim != 2 or vectors.shape[1] != self.config.embedding_dim:
            raise ValueError(
                "vectors must have shape "
                f"(N, {self.config.embedding_dim}), got {tuple(vectors.shape)}."
            )
        if vectors.shape[0] == 0:
            return vectors
        if self.training and vectors.shape[0] < 2:
            return F.batch_norm(
                vectors,
                self.embedding_bn.running_mean,
                self.embedding_bn.running_var,
                self.embedding_bn.weight,
                self.embedding_bn.bias,
                training=False,
                momentum=0.0,
                eps=self.embedding_bn.eps,
            )
        return self.embedding_bn(vectors)

    def extract_roi_embeddings(
        self,
        feature_map: torch.Tensor,
        boxes_per_image: list[torch.Tensor],
    ) -> torch.Tensor:
        """Extract one L2-normalised post-BN descriptor per supplied box."""

        pre_bn = self.project_roi_embeddings(feature_map, boxes_per_image)
        return F.normalize(self.apply_embedding_bn(pre_bn), dim=1)

    def extract_dense_embeddings_at_boxes(
        self,
        embedding_map: torch.Tensor,
        boxes_per_image: list[torch.Tensor],
    ) -> torch.Tensor:
        """Gather dense descriptors from the grid cells containing box centres."""

        if self.config.embedding_head_type != "dense" or self.embedding_head is None:
            raise RuntimeError("Dense embedding extraction requires a dense embedding head.")
        if embedding_map.ndim != 4:
            raise ValueError("embedding_map must have shape (B, D, H, W).")
        if embedding_map.shape[1] != self.config.embedding_dim:
            raise ValueError(
                f"Expected {self.config.embedding_dim} embedding channels, "
                f"got {embedding_map.shape[1]}."
            )
        if len(boxes_per_image) != embedding_map.shape[0]:
            raise ValueError("boxes_per_image length must match the embedding-map batch dimension.")

        rows: list[torch.Tensor] = []
        height, width = embedding_map.shape[-2:]
        for batch_index, boxes in enumerate(boxes_per_image):
            boxes = boxes.to(device=embedding_map.device, dtype=embedding_map.dtype).reshape(-1, 4)
            if boxes.shape[0] == 0:
                continue
            centres_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
            centres_y = 0.5 * (boxes[:, 1] + boxes[:, 3])
            grid_x = torch.round(centres_x / self.config.feature_stride - 0.5).long()
            grid_y = torch.round(centres_y / self.config.feature_stride - 0.5).long()
            grid_x = grid_x.clamp(0, width - 1)
            grid_y = grid_y.clamp(0, height - 1)
            rows.append(embedding_map[batch_index, :, grid_y, grid_x].transpose(0, 1))
        if not rows:
            return embedding_map.new_empty((0, self.config.embedding_dim))
        return torch.cat(rows, dim=0)

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
    """Decode one model output and optionally attach its per-cell embeddings."""
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


def simple_detector_config_from_checkpoint(checkpoint: dict) -> SimpleDetectorConfig:
    """Load config while recognising legacy dense and current RoI checkpoints."""

    payload = dict(checkpoint["model_config"])
    if "embedding_head_type" not in payload and int(payload.get("embedding_dim", 0)) > 0:
        weight = checkpoint.get("model_state", {}).get("embedding_head.weight")
        payload["embedding_head_type"] = (
            "dense" if isinstance(weight, torch.Tensor) and weight.ndim == 4 else "roi"
        )
    payload["temporal_recurrence_locations"] = tuple(
        payload.get("temporal_recurrence_locations", ())
    )
    return SimpleDetectorConfig(**payload)


def detection_boxes_xyxy(
    detections: list[DetectionRecord],
    reference: torch.Tensor,
) -> torch.Tensor:
    """Convert exported-order detections to an RoI tensor on ``reference``."""

    if not detections:
        return reference.new_empty((0, 4))
    return reference.new_tensor(
        [
            [
                detection.bbox_left,
                detection.bbox_top,
                detection.bbox_left + detection.bbox_width,
                detection.bbox_top + detection.bbox_height,
            ]
            for detection in detections
        ]
    )


def attach_detection_embeddings(
    detections: list[DetectionRecord], embeddings: torch.Tensor
) -> list[DetectionRecord]:
    """Attach descriptor rows to detections without changing their order."""

    if embeddings.ndim != 2 or embeddings.shape[0] != len(detections):
        raise ValueError(
            f"Embedding shape {tuple(embeddings.shape)} does not align with "
            f"{len(detections)} detections."
        )
    rows = embeddings.detach().cpu()
    return [
        replace(detection, embedding=tuple(float(value) for value in row))
        for detection, row in zip(detections, rows)
    ]
