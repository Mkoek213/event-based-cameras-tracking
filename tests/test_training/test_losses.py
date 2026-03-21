"""Unit tests for loss functions."""

import pytest
import torch

from src.training.losses import DetectionLoss


class TestDetectionLoss:
    def _make_batch(self, B: int = 2, C: int = 4, H: int = 8, W: int = 8):
        cls_logits = torch.randn(B, C, H, W)
        bbox_pred = torch.randn(B, 4, H, W)
        cls_targets = torch.randint(0, C, (B, H, W))
        bbox_targets = torch.randn(B, 4, H, W)
        pos_mask = torch.rand(B, H, W) > 0.5
        return cls_logits, bbox_pred, cls_targets, bbox_targets, pos_mask

    def test_loss_is_finite(self):
        criterion = DetectionLoss()
        args = self._make_batch()
        total, loss_dict = criterion(*args)
        assert torch.isfinite(total)

    def test_loss_dict_keys(self):
        criterion = DetectionLoss()
        args = self._make_batch()
        _, loss_dict = criterion(*args)
        assert "cls_loss" in loss_dict
        assert "reg_loss" in loss_dict

    def test_no_positives(self):
        criterion = DetectionLoss()
        cls_logits, bbox_pred, cls_targets, bbox_targets, _ = self._make_batch()
        pos_mask = torch.zeros_like(cls_targets, dtype=torch.bool)
        total, _ = criterion(cls_logits, bbox_pred, cls_targets, bbox_targets, pos_mask)
        assert torch.isfinite(total)
