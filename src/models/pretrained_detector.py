"""RGB-pretrained detector with learnable adapters for event representations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from src.data.representations import representation_channel_splits


@dataclass(frozen=True)
class PretrainedDetectorConfig:
    representation: str
    num_bins: int = 5
    num_classes: int = 7
    adapter_mode: str = "single"
    adapter_width: int = 32
    min_size: int = 480
    max_size: int = 640
    pretrained_weights: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


class AdapterBranch(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        groups = min(8, out_channels)
        while out_channels % groups:
            groups -= 1
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class RepresentationAdapter(nn.Module):
    """Map an event representation to a learned three-channel pseudo-RGB image."""

    def __init__(
        self, channel_splits: tuple[int, ...], mode: str = "single", width: int = 32
    ) -> None:
        super().__init__()
        if mode not in ("single", "multi_branch"):
            raise ValueError(f"Unknown adapter mode '{mode}'.")
        if mode == "multi_branch" and len(channel_splits) < 2:
            raise ValueError("multi_branch requires a representation with at least two components.")
        self.channel_splits = channel_splits
        self.mode = mode
        if mode == "single":
            self.branches = None
            self.fusion = AdapterBranch(sum(channel_splits), width)
        else:
            branch_width = max(width // len(channel_splits), 8)
            self.branches = nn.ModuleList(
                AdapterBranch(channels, branch_width) for channels in channel_splits
            )
            self.fusion = AdapterBranch(branch_width * len(channel_splits), width)
        self.output = nn.Sequential(nn.Conv2d(width, 3, kernel_size=1), nn.Sigmoid())

    @staticmethod
    def normalise(images: torch.Tensor) -> torch.Tensor:
        images = torch.log1p(images.clamp_min(0.0))
        scale = images.flatten(start_dim=1).amax(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return images / scale

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        chunks = torch.split(images, self.channel_splits, dim=1)
        chunks = [self.normalise(chunk) for chunk in chunks]
        if self.branches is not None:
            images = torch.cat(
                [branch(chunk) for branch, chunk in zip(self.branches, chunks)], dim=1
            )
        else:
            images = torch.cat(chunks, dim=1)
        return self.output(self.fusion(images))


class PretrainedEventDetector(nn.Module):
    """Wrap torchvision Faster R-CNN while preserving its RGB-pretrained backbone."""

    def __init__(self, config: PretrainedDetectorConfig) -> None:
        super().__init__()
        self.config = config
        splits = representation_channel_splits(config.representation, config.num_bins)
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT if config.pretrained_weights else None
        self.detector = fasterrcnn_resnet50_fpn(
            weights=weights,
            weights_backbone=None,
            min_size=config.min_size,
            max_size=config.max_size,
        )
        input_features = self.detector.roi_heads.box_predictor.cls_score.in_features
        self.detector.roi_heads.box_predictor = FastRCNNPredictor(
            input_features, config.num_classes + 1
        )
        # Build the representation-specific adapter last so all variants created
        # with the same seed receive an identical DSEC-MOT predictor initialization.
        self.adapter = RepresentationAdapter(splits, config.adapter_mode, config.adapter_width)

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ):
        batch = torch.stack(images)
        pseudo_rgb = self.adapter(batch)
        return self.detector(list(pseudo_rgb), targets)

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.detector.backbone.parameters():
            parameter.requires_grad = trainable

    def freeze_batch_norm_stats(self) -> None:
        """Preserve pretrained BatchNorm statistics during small-batch fine-tuning."""
        for module in self.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()

    @property
    def parameter_counts(self) -> dict[str, int]:
        return {
            "total": sum(parameter.numel() for parameter in self.parameters()),
            "adapter": sum(parameter.numel() for parameter in self.adapter.parameters()),
            "trainable": sum(
                parameter.numel() for parameter in self.parameters() if parameter.requires_grad
            ),
        }
