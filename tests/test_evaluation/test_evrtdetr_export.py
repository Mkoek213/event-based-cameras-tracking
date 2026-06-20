"""Tests for EvRT-DETR detection export helpers."""

import pytest

from src.evaluation.evrtdetr_export import build_dsec_class_id_by_model_label


def test_public_gen4_classes_map_to_dsec_ids():
    assert build_dsec_class_id_by_model_label(["pedestrian", "two-wheeler", "car"]) == [1, 2, 0]


def test_unknown_model_class_fails_explicitly():
    with pytest.raises(ValueError, match="Cannot map model class"):
        build_dsec_class_id_by_model_label(["unknown"])
