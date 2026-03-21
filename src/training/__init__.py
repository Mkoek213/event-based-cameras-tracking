"""Training sub-package."""

from .losses import DetectionLoss
from .trainer import Trainer

__all__ = ["DetectionLoss", "Trainer"]
