"""Documented converter targets for the OVHCloud multi-dataset pretraining run.

The concrete parsers are intentionally added per downloaded dataset, because the
public packages differ in format and some links require manual acceptance. This
module keeps the implementation contract explicit and testable before the data
is available.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetConversionTarget:
    name: str
    task: str
    expected_modalities: tuple[str, ...]
    preferred_use: str
    notes: str


CONVERSION_TARGETS = (
    DatasetConversionTarget(
        name="DSEC-MOT",
        task="MOT",
        expected_modalities=("events",),
        preferred_use="fine_tune_and_final_eval",
        notes="Already supported directly; also usable as unified manifest rows.",
    ),
    DatasetConversionTarget(
        name="FEMOT",
        task="RGB-event MOT",
        expected_modalities=("events", "rgb"),
        preferred_use="mot_pretraining",
        notes="Best external MOT candidate because it includes track IDs and traffic scenes.",
    ),
    DatasetConversionTarget(
        name="MEVDT",
        task="event/grayscale vehicle MOT",
        expected_modalities=("events", "grayscale"),
        preferred_use="traffic_mot_pretraining",
        notes="Good match for vehicle tracking; verify package format after download.",
    ),
    DatasetConversionTarget(
        name="TUMTraf Event/EMOT",
        task="traffic detection/MOT",
        expected_modalities=("events", "rgb"),
        preferred_use="traffic_pretraining",
        notes="Use event labels where available; EMOT availability must be verified manually.",
    ),
    DatasetConversionTarget(
        name="Prophesee Gen1 Automotive",
        task="event detection",
        expected_modalities=("events",),
        preferred_use="detector_pretraining",
        notes="Large traffic detection set for cars and pedestrians; no track IDs required.",
    ),
    DatasetConversionTarget(
        name="Prophesee 1MP Automotive",
        task="event detection",
        expected_modalities=("events",),
        preferred_use="large_detector_pretraining",
        notes="Very strong source for cars, pedestrians and two-wheelers.",
    ),
    DatasetConversionTarget(
        name="EventVOT",
        task="event SOT",
        expected_modalities=("events",),
        preferred_use="low_weight_representation_pretraining",
        notes="Use low sampling weight because it is SOT, not MOT.",
    ),
    DatasetConversionTarget(
        name="COESOT",
        task="RGB-event SOT",
        expected_modalities=("events", "rgb"),
        preferred_use="low_weight_fusion_pretraining",
        notes="Useful for robustness, but keep separate from final MOT benchmark.",
    ),
    DatasetConversionTarget(
        name="VisEvent",
        task="RGB-event SOT",
        expected_modalities=("events", "rgb"),
        preferred_use="low_weight_fusion_pretraining",
        notes="Auxiliary tracking pretraining source.",
    ),
    DatasetConversionTarget(
        name="CRSOT",
        task="RGB-event SOT",
        expected_modalities=("events", "rgb"),
        preferred_use="low_weight_fusion_pretraining",
        notes="Useful for hard alignment conditions; SOT-only supervision.",
    ),
)
