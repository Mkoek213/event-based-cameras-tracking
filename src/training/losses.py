"""Loss functions for detection training."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DetectionLoss(nn.Module):
    """Combined classification + regression loss for anchor-free detection.

    Args:
        cls_weight: Weighting factor for the classification term.
        reg_weight: Weighting factor for the regression term.
    """

    def __init__(self, cls_weight: float = 1.0, reg_weight: float = 1.0) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight

    def forward(
        self,
        cls_logits: torch.Tensor,
        bbox_pred: torch.Tensor,
        cls_targets: torch.Tensor,
        bbox_targets: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute the combined loss.

        Args:
            cls_logits: ``(B, C, H, W)`` raw class logits.
            bbox_pred: ``(B, 4, H, W)`` bounding-box offset predictions.
            cls_targets: ``(B, H, W)`` integer class labels.
            bbox_targets: ``(B, 4, H, W)`` target bounding-box offsets.
            pos_mask: ``(B, H, W)`` boolean mask of positive (foreground) cells.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where *loss_dict* contains
            individual loss components for logging.
        """
        cls_loss = F.cross_entropy(cls_logits, cls_targets)

        if pos_mask.any():
            reg_loss = F.smooth_l1_loss(
                bbox_pred.permute(0, 2, 3, 1)[pos_mask],
                bbox_targets.permute(0, 2, 3, 1)[pos_mask],
            )
        else:
            reg_loss = bbox_pred.sum() * 0.0

        total = self.cls_weight * cls_loss + self.reg_weight * reg_loss
        return total, {"cls_loss": cls_loss.item(), "reg_loss": reg_loss.item()}
