"""Single-stage object detector head on top of EventBackbone."""

from __future__ import annotations

import torch
import torch.nn as nn

from .backbone import EventBackbone


class EventDetector(nn.Module):
    """Simple anchor-free detection head.

    Predicts per-pixel (class scores, bounding-box offsets) from the
    backbone feature map.

    Args:
        backbone: Feature extractor module.
        num_classes: Number of foreground object classes.
    """

    def __init__(self, backbone: EventBackbone, num_classes: int = 10) -> None:
        super().__init__()
        self.backbone = backbone
        c = backbone.out_channels
        self.cls_head = nn.Conv2d(c, num_classes, kernel_size=1)
        self.reg_head = nn.Conv2d(c, 4, kernel_size=1)  # (dx, dy, dw, dh)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        cls_logits = self.cls_head(features)
        bbox_offsets = self.reg_head(features)
        return cls_logits, bbox_offsets
