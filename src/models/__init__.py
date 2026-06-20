"""Trainable models used by the local DSEC-MOT benchmark."""

from .simple_detector import SimpleDenseDetector, SimpleDetectorConfig, decode_dense_detections

__all__ = ["SimpleDenseDetector", "SimpleDetectorConfig", "decode_dense_detections"]
