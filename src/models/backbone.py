"""Backbone feature extractor for event-frame inputs."""

from __future__ import annotations

import torch
import torch.nn as nn


class EventBackbone(nn.Module):
    """Lightweight CNN backbone designed for event-based representations.

    Accepts a 4-D tensor of shape ``(B, C, H, W)`` and returns a feature
    map with reduced spatial resolution.

    Args:
        in_channels: Number of input channels (2 for polarity event frames,
            ``num_bins`` for voxel grids, etc.).
        base_channels: Width multiplier for the network.
    """

    def __init__(self, in_channels: int = 2, base_channels: int = 64) -> None:
        super().__init__()
        c = base_channels
        self.encoder = nn.Sequential(
            # Stage 1 – 1/2
            nn.Conv2d(in_channels, c, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
            # Stage 2 – 1/4
            nn.Conv2d(c, c * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(inplace=True),
            # Stage 3 – 1/8
            nn.Conv2d(c * 2, c * 4, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(inplace=True),
        )
        self.out_channels = c * 4

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
