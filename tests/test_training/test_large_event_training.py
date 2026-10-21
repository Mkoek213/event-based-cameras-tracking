"""Tests for the large event MOT losses, training runtime, staged trainer, and source mAP.

Model-dependent tests need the pinned RVT checkout in ``RVT_ROOT`` (and ``einops`` and
``omegaconf``); validation needs ``pycocotools``. Without them the tests skip. Trainer
runs use the CPU smoke mode (synthetic in-memory sequences, reduced geometry) with a
narrowed RVT configuration, because resume and state semantics do not depend on the
model width; the full-width model is covered by ``tests/test_models``.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
import os
import shutil
import signal
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.evaluation import large_event_detection_map as dmap
from src.models.large_event_detector import (
    DetectorOutput,
    HeadOutput,
    build_large_event_detector,
    load_model_config,
)
from src.models.large_event_modules import RecurrentState
from src.training import large_event_detector as trainer_module
from src.training import large_event_runtime as rt
from src.training.large_event_detector import (
    EXIT_COMPLETE,
    EXIT_PAUSED,
    EXIT_STOP_RULE,
    JobLimits,
    Trainer,
    build_inputs,
    parse_args,
)
from src.training.large_event_losses import (
    SupervisionError,
    TrainingConfigError,
    batch_hard_cosine_triplet,
    build_identity_classifier,
    load_training_config,
    loss_settings,
    stage_losses,
)
from src.training.recurrent_embedding_detector import batch_hard_cosine_triplet_loss

RVT_ROOT = os.environ.get("RVT_ROOT")
HAS_RVT = (
    bool(RVT_ROOT)
    and Path(RVT_ROOT).is_dir()
    and all(importlib.util.find_spec(name) for name in ("einops", "omegaconf"))
)
needs_rvt = pytest.mark.skipif(
    not HAS_RVT, reason="set RVT_ROOT to the pinned RVT checkout (needs einops and omegaconf)"
)
needs_coco = pytest.mark.skipif(
    importlib.util.find_spec("pycocotools") is None,
    reason="validation mAP needs pycocotools (requirements/large_event_mot.txt)",
)
REPO = Path(__file__).resolve().parents[2]
TRAINING_CONFIG = json.loads((REPO / "configs/large_event_mot/training.json").read_text())
DATA_CONFIG = json.loads((REPO / "configs/large_event_mot/data.json").read_text())
K = 7
LEVELS = ((4, 4, 8.0), (2, 2, 16.0))  # (h, w, stride): 20 anchors on a 32x32 image
TINY = {"input_hw": [128, 192], "valid_hw": [120, 192], "partition_size": [2, 3]}


@pytest.fixture(scope="module", autouse=True)
def no_fsync():
    """Checkpoint durability is not under test; fsync of every save dominates the runtime."""

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(rt.os, "fsync", lambda descriptor: None)
        yield


@pytest.fixture(scope="module")
def model_config():
    config = load_model_config()
    config["geometry"]["tiny"] = copy.deepcopy(TINY)
    return config


@pytest.fixture(scope="module")
def settings(model_config):
    return loss_settings(load_training_config(), model_config)


@pytest.fixture(scope="module")
def target_model(model_config):
    if not HAS_RVT:
        pytest.skip("set RVT_ROOT to the pinned RVT checkout")
    return build_large_event_detector(
        model_config, ontology="target", geometry="tiny", init="random", rvt_root=RVT_ROOT
    )


@pytest.fixture(scope="module")
def assigner(target_model):
    return target_model.yolox_head


# -------------------------------------------------------------------------- synthetic head outputs


def grid_and_strides():
    grids, strides = [], []
    for h, w, s in LEVELS:
        ys, xs = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
        grids.append(torch.stack((xs, ys), -1).reshape(-1, 2))
        strides.append(torch.full((h * w,), s))
    return torch.cat(grids), torch.cat(strides)


def synthetic_step(clips, seed, embeddings=True):
    """A ``DetectorOutput`` with leaf tensors for gradient checks (``output.leaves``)."""

    generator = torch.Generator().manual_seed(seed)
    grid, strides = grid_and_strides()
    anchors = grid.shape[0]
    raw = (torch.randn(clips, anchors, 4, generator=generator) * 0.3).requires_grad_(True)
    boxes = torch.cat(
        ((raw[..., :2] + grid) * strides[:, None], torch.exp(raw[..., 2:]) * strides[:, None]), -1
    )
    obj = torch.randn(clips, anchors, generator=generator).requires_grad_(True)
    cls = torch.randn(clips, anchors, K, generator=generator).requires_grad_(True)
    embedding_leaf = embedding = None
    if embeddings:
        embedding_leaf = torch.randn(clips, anchors, 128, generator=generator).requires_grad_(True)
        embedding = F.normalize(embedding_leaf, dim=-1)
    head = HeadOutput(boxes, obj, cls, grid, strides, tuple((h, w) for h, w, _ in LEVELS))
    output = DetectorOutput({}, (), head, None, None, embedding)
    output.leaves = (raw, obj, cls) + ((embedding_leaf,) if embeddings else ())
    return output


def target(boxes=(), labels=(), ignore=(), identity=None, identity_valid=None):
    count = len(labels)
    return {
        "boxes_xyxy": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "ignore": torch.tensor(ignore if ignore else [False] * count, dtype=torch.bool),
        "track_ids": torch.zeros(count, dtype=torch.int64),
        "identity": torch.tensor(identity if identity is not None else [-1] * count),
        "identity_valid": torch.tensor(
            identity_valid if identity_valid is not None else [False] * count, dtype=torch.bool
        ),
    }


def id_target(boxes, labels, identity):
    return target(boxes, labels, identity=identity, identity_valid=[i >= 0 for i in identity])


def make_batch(targets, metas, supervised=None, valid=None, class_mask=None):
    clips, steps = len(targets), len(targets[0])
    ones = torch.ones(clips, steps, dtype=torch.bool)
    return {
        "valid": ones if valid is None else torch.tensor(valid),
        "supervised": ones.clone() if supervised is None else torch.tensor(supervised),
        "class_mask": torch.ones(clips, K, dtype=torch.bool)
        if class_mask is None
        else torch.tensor(class_mask),
        "targets": targets,
        "meta": metas,
    }


def meta(dataset="dsec_mot", sequence="seq_a", head="target"):
    return {"dataset": dataset, "sequence": sequence, "head": head}


def gen4(sequence="chunk"):
    return meta("gen4_1mpx", sequence, "source")


def classifier_for(settings):
    return build_identity_classifier(settings, "finetune", 320)


# ----------------------------------------------------------------------------------- loss settings


def test_loss_settings_and_identity_classifier(settings, model_config):
    assert settings.stage("pretrain").identity_enabled is False
    assert settings.stage("finetune").identity_datasets == frozenset({"dsec_mot"})
    classifier = build_identity_classifier(settings, "finetune", 320, seed=0)
    assert sum(p.numel() for p in classifier.parameters()) == 40_960
    assert torch.equal(
        classifier.weight, build_identity_classifier(settings, "finetune", 320).weight
    )
    assert abs(float(classifier.weight.std()) - 0.01) < 1e-3
    embeddings = F.normalize(torch.randn(5, 128), dim=1)
    logits = classifier(embeddings)
    assert logits.abs().max() <= 16.0 + 1e-4
    torch.testing.assert_close(logits, 16 * embeddings @ F.normalize(classifier.weight, dim=1).t())
    with pytest.raises(SupervisionError):
        build_identity_classifier(settings, "pretrain", 320)
    with pytest.raises(TrainingConfigError, match="320"):
        build_identity_classifier(settings, "finetune", 321)
    bad = load_training_config()
    bad["stages"]["finetune"]["identity"]["datasets"] = []
    with pytest.raises(TrainingConfigError):
        loss_settings(bad, model_config)
    bad = load_training_config()
    bad["losses"]["detection"]["masked_class_logit"] = -10.0
    with pytest.raises(TrainingConfigError):
        loss_settings(bad, model_config)
    with pytest.raises(TrainingConfigError):
        settings.stage("other")


# -------------------------------------------------------------------------------- detection losses


@needs_rvt
def test_no_supervised_step_gives_a_connected_zero(settings, assigner):
    output = synthetic_step(2, 0, embeddings=False)
    batch = make_batch(
        [[target([[1, 1, 9, 9]], [0])], [target()]],
        [gen4("a"), gen4("b")],
        supervised=[[False], [False]],
    )
    loss = stage_losses([output], batch, "pretrain", settings, assigner)
    assert float(loss.total) == 0.0 and loss.total.requires_grad
    assert loss.counts["supervised_steps"] == 0
    assert loss.flags == {"detection": False, "identity_ce": False, "triplet": False}
    loss.total.backward()
    for leaf in output.leaves:
        assert leaf.grad is not None and torch.count_nonzero(leaf.grad) == 0


@needs_rvt
def test_padded_and_unannotated_steps_change_nothing(settings, assigner):
    first, second = synthetic_step(1, 1, False), synthetic_step(1, 2, False)
    labeled = target([[2, 2, 14, 12]], [2])
    extra = target([[0, 0, 30, 30]], [1])
    single = stage_losses(
        [first], make_batch([[labeled]], [gen4()]), "pretrain", settings, assigner
    )
    padded = stage_losses(
        [first, second],
        make_batch([[labeled, extra]], [gen4()], valid=[[True, False]]),
        "pretrain",
        settings,
        assigner,
    )
    unannotated = stage_losses(
        [first, second],
        make_batch([[labeled, extra]], [gen4()], supervised=[[True, False]]),
        "pretrain",
        settings,
        assigner,
    )
    assert float(padded.total) == float(single.total) == float(unannotated.total)
    assert padded.counts["supervised_steps"] == unannotated.counts["supervised_steps"] == 1


@needs_rvt
def test_labeled_empty_step_supervises_background_only(settings, assigner):
    output = synthetic_step(1, 3, False)
    loss = stage_losses(
        [output], make_batch([[target()]], [gen4()]), "pretrain", settings, assigner
    )
    expected = F.binary_cross_entropy_with_logits(
        output.head.obj_logits[0], torch.zeros(20), reduction="sum"
    )
    torch.testing.assert_close(loss.total, expected)
    assert float(loss.terms["iou"]) == 0.0 and float(loss.terms["cls"]) == 0.0
    assert loss.counts["fg_anchors"] == 0 and loss.flags["detection"]
    loss.total.backward()
    raw, obj, cls = output.leaves
    assert torch.count_nonzero(obj.grad) == 20
    assert torch.count_nonzero(cls.grad) == 0 and torch.count_nonzero(raw.grad) == 0


@needs_rvt
def test_ignore_regions_mask_background_objectness(settings, assigner):
    output = synthetic_step(1, 4, False)
    # Inclusive: s8 centres (4,4), (12,4), (4,12), (12,12) and the s16 centre (8,8).
    batch = make_batch([[target([[0.0, 0.0, 12.0, 12.0]], [0], ignore=[True])]], [gen4()])
    loss = stage_losses([output], batch, "pretrain", settings, assigner)
    grid, strides = grid_and_strides()
    inside = (((grid + 0.5) * strides[:, None]) <= 12).all(dim=1)
    assert int(inside.sum()) == 5 and loss.counts["ignore_masked_anchors"] == 5
    expected = F.binary_cross_entropy_with_logits(
        output.head.obj_logits[0][~inside], torch.zeros(15), reduction="sum"
    )
    torch.testing.assert_close(loss.total, expected)
    assert loss.counts["positive_boxes"] == 0 and loss.counts["ignore_boxes"] == 1


@needs_rvt
def test_detection_losses_match_upstream_yolox(settings, assigner):
    outputs = [synthetic_step(3, 10 + step, False) for step in range(2)]
    targets = [
        [target([[2, 2, 14, 12], [16, 8, 30, 28]], [2, 0]), target([[4, 4, 20, 20]], [1])],
        [target(), target([[1, 10, 9, 30], [10, 1, 31, 15], [0, 0, 32, 32]], [3, 4, 6])],
        [target([[5, 5, 27, 27]], [5]), target()],
    ]
    batch = make_batch(targets, [gen4("a"), gen4("b"), gen4("c")])
    loss = stage_losses(outputs, batch, "pretrain", settings, assigner)
    rows, labels = [], []
    for clip in range(3):
        for step in range(2):
            rows.append(outputs[step].head.train_layout()[clip])
            xyxy = targets[clip][step]["boxes_xyxy"]
            cxcywh = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)
            labels.append(torch.cat((targets[clip][step]["labels"][:, None].float(), cxcywh), 1))
    padded = torch.zeros(len(labels), 3, 5)
    for index, label in enumerate(labels):
        padded[index, : label.shape[0]] = label
    head = outputs[0].head
    upstream = assigner.get_losses(
        [head.x_shifts],
        [head.y_shifts],
        [head.expanded_strides],
        padded,
        torch.stack(rows),
        None,
        torch.float32,
    )
    torch.testing.assert_close(loss.total, upstream[0])
    torch.testing.assert_close(loss.terms["iou_weighted"], upstream[1])
    torch.testing.assert_close(loss.terms["obj"], upstream[2])
    torch.testing.assert_close(loss.terms["cls"], upstream[3])
    assert loss.counts["supervised_steps"] == 6 and loss.counts["positive_boxes"] == 7


@needs_rvt
def test_masked_class_column_adds_no_cost_or_gradient(settings, assigner):
    mask = [[True] * 6 + [False]]
    targets = [[target([[2, 2, 14, 12], [16, 8, 30, 28]], [2, 0])]]
    batch = make_batch(targets, [meta("dsec_detection")], class_mask=mask)
    base = synthetic_step(1, 20)
    loss = stage_losses([base], batch, "finetune", settings, assigner, classifier_for(settings))
    altered = synthetic_step(1, 20)
    with torch.no_grad():
        altered.head.cls_logits[..., 6] = 50.0
    other = stage_losses([altered], batch, "finetune", settings, assigner, classifier_for(settings))
    torch.testing.assert_close(loss.total, other.total)
    assert loss.counts["fg_anchors"] == other.counts["fg_anchors"]
    assert loss.counts["masked_class_columns"] == 1
    loss.total.backward()
    assert torch.count_nonzero(base.leaves[2].grad[..., 6]) == 0
    assert torch.count_nonzero(base.leaves[2].grad[..., :6]) > 0
    with pytest.raises(SupervisionError, match="masked class"):
        stage_losses(
            [synthetic_step(1, 21)],
            make_batch(
                [[target([[2, 2, 14, 12]], [6])]], [meta("dsec_detection")], class_mask=mask
            ),
            "finetune",
            settings,
            assigner,
            classifier_for(settings),
        )


@needs_rvt
def test_wrong_head_or_label_is_rejected(settings, assigner):
    with pytest.raises(SupervisionError, match="head"):
        stage_losses(
            [synthetic_step(1, 0, False)],
            make_batch([[target()]], [meta()]),
            "pretrain",
            settings,
            assigner,
        )
    with pytest.raises(SupervisionError, match="outside the head"):
        stage_losses(
            [synthetic_step(1, 0, False)],
            make_batch([[target([[0, 0, 8, 8]], [7])]], [gen4()]),
            "pretrain",
            settings,
            assigner,
        )


# --------------------------------------------------------------------------------- identity losses


@needs_rvt
def test_identity_labels_rejected_outside_identity_sources(settings, assigner):
    labeled = id_target([[2, 2, 14, 12]], [0], [5])
    with pytest.raises(SupervisionError, match="no identity supervision"):
        stage_losses(
            [synthetic_step(1, 0, False)],
            make_batch([[labeled]], [gen4()]),
            "pretrain",
            settings,
            assigner,
        )
    with pytest.raises(SupervisionError, match="identity sources"):
        stage_losses(
            [synthetic_step(1, 0)],
            make_batch([[labeled]], [meta("dsec_detection")]),
            "finetune",
            settings,
            assigner,
            classifier_for(settings),
        )
    with pytest.raises(SupervisionError, match="identity classifier"):
        stage_losses(
            [synthetic_step(1, 0)],
            make_batch([[labeled]], [meta()]),
            "finetune",
            settings,
            assigner,
        )
    with pytest.raises(SupervisionError, match="identity classifier"):
        stage_losses(
            [synthetic_step(1, 0, False)],
            make_batch([[target()]], [gen4()]),
            "pretrain",
            settings,
            assigner,
            classifier_for(settings),
        )
    with pytest.raises(SupervisionError, match="embeddings"):
        stage_losses(
            [synthetic_step(1, 0, False)],
            make_batch([[labeled]], [meta()]),
            "finetune",
            settings,
            assigner,
            classifier_for(settings),
        )
    malformed = id_target([[2, 2, 14, 12]], [0], [5])
    malformed["ignore"][:] = True
    with pytest.raises(SupervisionError, match="malformed"):
        stage_losses(
            [synthetic_step(1, 0)],
            make_batch([[malformed]], [meta()]),
            "finetune",
            settings,
            assigner,
            classifier_for(settings),
        )
    with pytest.raises(SupervisionError, match="outside the identity classifier"):
        stage_losses(
            [synthetic_step(1, 0)],
            make_batch([[id_target([[2, 2, 14, 12]], [0], [320])]], [meta()]),
            "finetune",
            settings,
            assigner,
            classifier_for(settings),
        )


@needs_rvt
def test_clips_without_ids_give_finite_inactive_identity_terms(settings, assigner):
    output = synthetic_step(2, 30)
    classifier = classifier_for(settings)
    batch = make_batch(
        [[target([[2, 2, 14, 12]], [0])], [target([[4, 4, 20, 20]], [1])]],
        [meta(), meta("dsec_detection", "det_seq")],
    )
    loss = stage_losses([output], batch, "finetune", settings, assigner, classifier)
    assert float(loss.terms["identity_ce"]) == 0.0 and float(loss.terms["triplet"]) == 0.0
    assert loss.counts["identity_valid_anchors"] == 0 and loss.counts["triplet_anchors"] == 0
    assert loss.flags == {"detection": True, "identity_ce": False, "triplet": False}
    torch.testing.assert_close(loss.total, loss.terms["detection"])
    loss.total.backward()
    assert torch.count_nonzero(output.leaves[3].grad) == 0
    assert classifier.weight.grad is not None and torch.count_nonzero(classifier.weight.grad) == 0
    assert all(torch.isfinite(leaf.grad).all() for leaf in output.leaves)


@needs_rvt
def test_cross_entropy_uses_identity_valid_anchors_only(settings, assigner):
    output = synthetic_step(1, 40)
    classifier = classifier_for(settings)
    labeled = id_target([[2, 2, 14, 12], [16, 8, 30, 28]], [0, 0], [7, -1])
    batch = make_batch([[labeled]], [meta()])
    loss = stage_losses([output], batch, "finetune", settings, assigner, classifier)
    head = output.head
    gt = torch.tensor([[8.0, 7.0, 12.0, 10.0], [23.0, 18.0, 14.0, 20.0]])
    _, fg, _, matched, _ = assigner.get_assignments(
        0,
        2,
        gt,
        torch.zeros(2),
        head.boxes[0].detach(),
        head.expanded_strides,
        head.x_shifts,
        head.y_shifts,
        head.cls_logits[:1].detach(),
        head.obj_logits[:1, :, None].detach(),
    )
    anchors = torch.nonzero(fg).squeeze(1)[matched == 0]
    expected = F.cross_entropy(
        classifier(output.embeddings[0][anchors]), torch.full((anchors.numel(),), 7)
    )
    torch.testing.assert_close(loss.terms["identity_ce"], expected.detach())
    assert loss.counts["identity_valid_anchors"] == anchors.numel() > 0
    assert loss.counts["identity_valid_boxes"] == 1 and loss.counts["identity_instances"] == 1
    assert loss.flags["identity_ce"] and not loss.flags["triplet"]
    loss.total.backward()
    assert torch.count_nonzero(classifier.weight.grad[7]) > 0
    touched = torch.nonzero(output.leaves[3].grad[0].abs().sum(1)).squeeze(1)
    assert set(touched.tolist()) == set(anchors.tolist())


@needs_rvt
def test_triplet_needs_negatives_and_reused_ids_stay_distinct(settings, assigner):
    classifier = classifier_for(settings)
    box = [[2, 2, 14, 12]]
    # One identity in two steps and no other identity of its class: no negative.
    outputs = [synthetic_step(1, 50), synthetic_step(1, 51)]
    batch = make_batch([[id_target(box, [0], [3]), id_target(box, [0], [3])]], [meta()])
    loss = stage_losses(outputs, batch, "finetune", settings, assigner, classifier)
    assert loss.counts["identity_instances"] == 2
    assert loss.counts["triplet_positive_pairs"] == 2 and loss.counts["triplet_negative_pairs"] == 0
    assert not loss.flags["triplet"] and float(loss.terms["triplet"]) == 0.0
    # Track ID 1 in two sequences has two global identities: negatives, not positives.
    outputs = [synthetic_step(2, 52), synthetic_step(2, 53)]
    first, second = id_target(box, [0], [10]), id_target(box, [0], [11])
    for item in (first, second):
        item["track_ids"][:] = 1
    batch = make_batch(
        [[first, copy.deepcopy(first)], [second, copy.deepcopy(second)]],
        [meta("dsec_mot", "seq_a"), meta("dsec_mot", "seq_b")],
    )
    loss = stage_losses(outputs, batch, "finetune", settings, assigner, classifier)
    assert loss.counts["identity_instances"] == 4
    assert loss.counts["triplet_positive_pairs"] == 4 and loss.counts["triplet_negative_pairs"] == 8
    assert loss.counts["triplet_anchors"] == 4 and loss.flags["triplet"]
    # A different class is not a negative.
    for step in range(2):
        batch["targets"][1][step]["labels"][:] = 1
    loss = stage_losses(outputs, batch, "finetune", settings, assigner, classifier)
    assert loss.counts["triplet_negative_pairs"] == 0 and not loss.flags["triplet"]


@needs_rvt
def test_triplet_representatives_match_the_reference_loss(settings, assigner):
    outputs = [synthetic_step(2, 60), synthetic_step(2, 61)]
    boxes = [[2, 2, 14, 12], [16, 8, 30, 28]]
    batch = make_batch(
        [
            [id_target(boxes, [0, 0], [1, 2]), id_target(boxes, [0, 0], [1, 2])],
            [id_target(boxes, [0, 1], [3, 4]), id_target(boxes, [0, 1], [3, 4])],
        ],
        [meta("dsec_mot", "a"), meta("dsec_mot", "b")],
    )
    loss = stage_losses(outputs, batch, "finetune", settings, assigner, classifier_for(settings))
    rows, identities, classes = [], [], []
    for clip in range(2):
        for step in range(2):
            head = outputs[step].head
            item = batch["targets"][clip][step]
            xyxy = item["boxes_xyxy"]
            gt = torch.cat(((xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]), 1)
            _, fg, iou, matched, _ = assigner.get_assignments(
                0,
                2,
                gt,
                item["labels"].float(),
                head.boxes[clip].detach(),
                head.expanded_strides,
                head.x_shifts,
                head.y_shifts,
                head.cls_logits[clip : clip + 1].detach(),
                head.obj_logits[clip : clip + 1, :, None].detach(),
            )
            anchors = torch.nonzero(fg).squeeze(1)
            for gt_index in range(2):
                chosen = matched == gt_index
                best = anchors[chosen][torch.argmax(iou[chosen])]
                rows.append(outputs[step].embeddings[clip][best])
                identities.append(item["identity"][gt_index])
                classes.append(item["labels"][gt_index])
    rows, identities, classes = torch.stack(rows), torch.stack(identities), torch.stack(classes)
    reference, count = batch_hard_cosine_triplet_loss(rows, identities, classes, margin=0.3)
    torch.testing.assert_close(loss.terms["triplet"], reference.detach())
    assert loss.counts["triplet_anchors"] == count == 6  # identity 4 (class 1) has no negative
    torch.testing.assert_close(
        loss.total, loss.terms["detection"] + loss.terms["identity_ce"] + loss.terms["triplet"]
    )


def test_vectorized_triplet_equals_the_reference_on_random_sets():
    generator = torch.Generator().manual_seed(0)
    for _ in range(20):
        count = int(torch.randint(1, 30, (1,), generator=generator))
        embeddings = torch.randn(count, 16, generator=generator, requires_grad=True)
        identities = torch.randint(0, 5, (count,), generator=generator)
        classes = torch.randint(0, 2, (count,), generator=generator)
        reference, anchors = batch_hard_cosine_triplet_loss(embeddings, identities, classes, 0.3)
        mine, counts = batch_hard_cosine_triplet(embeddings, identities, classes, 0.3)
        torch.testing.assert_close(mine, reference)
        assert counts["triplet_anchors"] == anchors
        expected = torch.autograd.grad(reference, embeddings, allow_unused=True)[0]
        if expected is not None:
            torch.testing.assert_close(
                torch.autograd.grad(mine, embeddings, allow_unused=True)[0], expected
            )


# ------------------------------------------------------------------------------- full model losses


def run_model(model, inputs, with_embeddings):
    state, outputs = None, []
    for ef, vg in inputs:
        output = model(ef, vg, state=state, with_embeddings=with_embeddings)
        state = output.state
        outputs.append(output)
    return outputs


def tiny_inputs(seed, clips):
    generator = torch.Generator().manual_seed(seed)
    return [
        (
            torch.rand(clips, 2, 128, 192, generator=generator) * 0.3,
            torch.rand(clips, 6, 128, 192, generator=generator) * 0.3,
        )
        for _ in range(2)
    ]


def class_rows_have_gradient(model):
    return any(
        layer.weight.grad is not None and torch.count_nonzero(layer.weight.grad) > 0
        for layer in model.yolox_head.cls_preds
    )


@needs_rvt
def test_full_model_finetune_gradients_by_dataset(target_model, settings):
    model = target_model
    model.train()
    classifier = classifier_for(settings)
    boxes = [[20, 20, 60, 50], [100, 60, 160, 110]]
    inputs = tiny_inputs(0, 2)
    # A detection-only DSEC-Detection clip: no gradient into the embedding head or classifier.
    model.zero_grad()
    outputs = run_model(model, [(ef[:1], vg[:1]) for ef, vg in inputs], True)
    batch = make_batch(
        [[target(boxes, [0, 1]), target(boxes, [0, 1])]],
        [meta("dsec_detection")],
        class_mask=[[True] * 6 + [False]],
    )
    loss = stage_losses(outputs, batch, "finetune", settings, model.yolox_head, classifier)
    assert torch.isfinite(loss.total)
    loss.total.backward()
    for parameter in model.embedding_head.parameters():
        assert parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
    assert torch.count_nonzero(classifier.weight.grad) == 0
    assert class_rows_have_gradient(model)
    assert any(torch.count_nonzero(p.grad) for p in model.fusion.parameters() if p.grad is not None)
    # DSEC-MOT clips with identities over two carried steps train every identity term.
    model.zero_grad()
    classifier.zero_grad()
    outputs = run_model(model, inputs, True)
    batch = make_batch(
        [
            [id_target(boxes, [0, 1], [1, 2]), id_target(boxes, [0, 1], [1, 2])],
            [id_target(boxes, [0, 0], [3, 4]), id_target(boxes, [0, 0], [3, 4])],
        ],
        [meta("dsec_mot", "a"), meta("dsec_mot", "b")],
    )
    loss = stage_losses(outputs, batch, "finetune", settings, model.yolox_head, classifier)
    assert loss.flags == {"detection": True, "identity_ce": True, "triplet": True}
    loss.total.backward()
    for parameter in list(model.parameters()) + [classifier.weight]:
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()
    assert any(torch.count_nonzero(p.grad) for p in model.embedding_head.parameters() if p.ndim > 1)
    assert torch.count_nonzero(classifier.weight.grad[1:5]) > 0
    model.zero_grad()
    model.eval()


@needs_rvt
def test_full_model_pretraining_leaves_the_embedding_head_alone(model_config, settings):
    model = build_large_event_detector(
        model_config, ontology="source", geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    model.train()
    inputs = tiny_inputs(1, 1)
    batch = make_batch(
        [
            [
                target([[20, 20, 70, 60]], [2]),
                target([[20, 20, 70, 60], [100, 0, 192, 120]], [5, 6], ignore=[False, True]),
            ]
        ],
        [gen4()],
    )
    loss = stage_losses(
        run_model(model, inputs, False), batch, "pretrain", settings, model.yolox_head
    )
    assert float(loss.terms["identity_ce"]) == 0.0 and float(loss.terms["triplet"]) == 0.0
    assert loss.counts["ignore_masked_anchors"] > 0
    loss.total.backward()
    assert all(p.grad is None for p in model.embedding_head.parameters())
    assert class_rows_have_gradient(model)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        mixed = stage_losses(
            run_model(model, inputs, False), batch, "pretrain", settings, model.yolox_head
        )
    assert mixed.total.dtype == torch.float32 and torch.isfinite(mixed.total)


# ---------------------------------------------------------------------------------- stage settings


def test_learning_rate_schedules_and_validation_points():
    pretrain = rt.LearningRateSchedule(rt.resolve_stage(TRAINING_CONFIG, DATA_CONFIG, "pretrain"))
    assert pretrain(0) == 0 and pretrain(1000) == pytest.approx(1e-4)
    assert pretrain(2000) == pytest.approx(2e-4)
    assert pretrain(101000) == pytest.approx(2e-4 + (2e-8 - 2e-4) * 0.5)
    assert pretrain(200000) == pytest.approx(2e-8)
    finetune_settings = rt.resolve_stage(TRAINING_CONFIG, DATA_CONFIG, "finetune")
    finetune = rt.LearningRateSchedule(finetune_settings)
    assert finetune(250) == pytest.approx(5e-5) and finetune(500) == pytest.approx(1e-4)
    assert finetune(6250) == pytest.approx(1e-6 + 0.5 * (1e-4 - 1e-6))
    assert finetune(12000) == pytest.approx(1e-6)
    assert rt.validation_points(finetune_settings) == list(range(1000, 12001, 1000))
    short = rt.resolve_stage(
        TRAINING_CONFIG, DATA_CONFIG, "finetune", ["total_steps=600", "phase_a_steps=500"]
    )
    assert rt.validation_points(short) == list(range(50, 601, 50))
    assert rt.phase_of(finetune_settings, 499) == "phase_a"
    assert rt.phase_of(finetune_settings, 500) == "phase_b"


def test_stage_resolution_never_reaches_test_data():
    resolve = rt.resolve_stage
    with pytest.raises(rt.RuntimeConfigError, match="unknown key"):
        resolve(TRAINING_CONFIG, DATA_CONFIG, "pretrain", ["validation.evry=3"])
    with pytest.raises(rt.RuntimeConfigError, match="source val split"):
        resolve(TRAINING_CONFIG, DATA_CONFIG, "pretrain", ["validation.split=test"])
    with pytest.raises(rt.RuntimeConfigError, match="DSEC-MOT val only"):
        resolve(TRAINING_CONFIG, DATA_CONFIG, "finetune", ["validation.selection.split=test"])
    with pytest.raises(rt.RuntimeConfigError, match="DSEC-MOT val only"):
        resolve(
            TRAINING_CONFIG,
            DATA_CONFIG,
            "finetune",
            ["validation.selection.dataset=dsec_detection"],
        )
    with pytest.raises(rt.RuntimeConfigError, match="val_monitor"):
        resolve(TRAINING_CONFIG, DATA_CONFIG, "finetune", ["validation.monitor.split=train"])
    bad = copy.deepcopy(DATA_CONFIG)
    bad["sampling"]["finetune"]["splits"]["dsec_mot"] = "test"
    with pytest.raises(rt.RuntimeConfigError, match="train splits"):
        resolve(TRAINING_CONFIG, bad, "finetune")
    with pytest.raises(rt.RuntimeConfigError, match="phase A"):
        resolve(TRAINING_CONFIG, DATA_CONFIG, "pretrain", ["phase_a_steps=200000"])
    with pytest.raises(rt.RuntimeConfigError, match="init"):
        resolve(TRAINING_CONFIG, DATA_CONFIG, "finetune", ["init=random"])
    resolved = resolve(
        TRAINING_CONFIG,
        DATA_CONFIG,
        "pretrain",
        ["total_steps=1000", "phase_a_steps=100", "lr.warmup_steps=100", "validation.limit=8"],
    )
    assert resolved["total_steps"] == 1000 and resolved["validation"]["limit"] == 8


def test_ema_follows_the_yolox_formula():
    net = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.BatchNorm1d(2))
    ema = rt.ModelEma(net, 0.9999, 2000)
    start = {key: value.clone() for key, value in ema.module.state_dict().items()}
    with torch.no_grad():
        for parameter in net.parameters():
            parameter.add_(1.0)
    ema.update(net)
    decay = 0.9999 * (1 - math.exp(-1 / 2000))
    for key, value in ema.module.state_dict().items():
        if value.dtype.is_floating_point:
            assert torch.allclose(value, decay * start[key] + (1 - decay) * net.state_dict()[key])
        else:
            assert torch.equal(value, start[key])
    assert not any(p.requires_grad for p in ema.module.parameters()) and not ema.module.training


# ----------------------------------------------------------------------- state bank and stop rules


def stream_meta(
    slot, sequence, start, length=3, reset=False, epoch=0, ordinal=0, dataset="dsec_mot"
):
    return {
        "slot": slot,
        "dataset": dataset,
        "sequence": sequence,
        "start": start,
        "length": length,
        "reset": reset,
        "epoch": epoch,
        "ordinal": ordinal,
        "kind": "stream",
        "augmentation": {
            "hflip": False,
            "zoom": "none",
            "factor": 1.0,
            "origin_u": (0.0, 0.0),
            "label_u": 0.0,
        },
    }


def test_stream_state_bank_isolates_streams(tmp_path):
    zero = RecurrentState.zeros(1, [(2, 2, 2)])
    one = RecurrentState(((torch.ones(1, 2, 2, 2), torch.ones(1, 2, 2, 2)),))
    bank = rt.StreamStateBank([("dsec_mot", 0), ("dsec_mot", 1)])
    with pytest.raises(rt.StateLeakageError):
        bank.take(stream_meta(0, "a", 3), zero)  # continuation without a carried state
    with pytest.raises(rt.StateLeakageError):
        bank.take(stream_meta(0, "a", 3, reset=True), zero)  # reset in the middle
    with pytest.raises(rt.StateLeakageError):
        bank.take(stream_meta(0, "a", 0, reset=True, dataset="dsec_detection"), zero)
    assert bank.take(stream_meta(0, "a", 0, reset=True), zero) is zero
    bank.store(stream_meta(0, "a", 0), one)
    for wrong in (
        stream_meta(0, "a", 4),  # gap
        stream_meta(0, "b", 3),  # other sequence
        stream_meta(0, "a", 3, epoch=1),
        stream_meta(0, "a", 3, ordinal=1),
        stream_meta(1, "a", 3),  # other slot
    ):
        with pytest.raises(rt.StateLeakageError):
            bank.take(wrong, zero)
    flipped = stream_meta(0, "a", 3)
    flipped["augmentation"] = dict(flipped["augmentation"], hflip=True)
    with pytest.raises(rt.StateLeakageError):
        bank.take(flipped, zero)
    assert torch.equal(bank.take(stream_meta(0, "a", 3), zero).stages[0][0], one.stages[0][0])
    assert bank.states[0].stages[0][0].requires_grad is False
    path = tmp_path / "bank.pt"
    torch.save(bank.state_dict(), path)
    restored = rt.StreamStateBank([("dsec_mot", 0), ("dsec_mot", 1)])
    restored.load_state_dict(torch.load(path, weights_only=True), "cpu")
    assert restored.states[1] is None
    assert torch.equal(restored.states[0].stages[0][1], one.stages[0][1])
    assert restored.cursors == bank.cursors and restored.counters == bank.counters
    with pytest.raises(rt.StateLeakageError):
        rt.StreamStateBank([("dsec_mot", 0)]).load_state_dict(bank.state_dict(), "cpu")


def stop_rules():
    rules = {
        "nonfinite": {"max_consecutive": 3, "max_in_window": 5, "window_steps": 1000},
        "identity_sampling": {
            "window_steps": 4,
            "min_identity_fraction": 0.75,
            "min_triplet_fraction": 0.5,
        },
        "class_coverage": True,
    }
    settings = {"identity": {"enabled": True}, "phase_a_steps": 2}
    sources = {0: ["x"], 1: ["x", "y"], 2: ["y"]}
    return rt.StopRules(rules, settings, ["a", "b", "c"], sources, {"x": 1, "y": 2})


def test_stop_rule_for_non_finite_updates():
    rules = stop_rules()
    rules.record_update(0, skipped=True)
    rules.record_update(1, skipped=True)
    rules.record_update(2, skipped=False)
    rules.record_update(3, skipped=True)
    rules.record_update(4, skipped=True)
    with pytest.raises(rt.StopRuleError, match="consecutive"):
        rules.record_update(5, skipped=True)
    rules = stop_rules()
    for step in (0, 200, 400, 600, 800):
        rules.record_update(step, skipped=True)
        rules.record_update(step + 1, skipped=False)
    rules.record_update(1000, skipped=True)  # step 0 left the window
    with pytest.raises(rt.StopRuleError, match="within 1000"):
        rules.record_update(1100, skipped=True)
    state = rules.state_dict()
    restored = stop_rules()
    restored.load_state_dict(json.loads(json.dumps(state)))
    assert restored.state_dict() == state


def test_stop_rules_for_identity_sampling_and_class_coverage():
    rules = stop_rules()
    for step in range(2):
        rules.record_identity(step, False, False)  # phase A is not checked
    for step, flags in zip(range(2, 6), [(1, 1), (1, 1), (1, 0), (0, 0)]):
        rules.record_identity(step, *flags)  # 0.75 and 0.5 pass
    with pytest.raises(rt.StopRuleError, match="identity fraction 0.500"):
        for step, flags in zip(range(6, 10), [(1, 1), (1, 1), (0, 0), (0, 0)]):
            rules.record_identity(step, *flags)
    rules = stop_rules()
    rules.record_batch([dict(stream_meta(0, "s", 0, dataset="x"), epoch=1)], [np.array([1, 0, 0])])
    rules.check_coverage()  # y has not finished an epoch: only class 0 is checked
    assert rules.coverage_checked == [0]
    rules.record_batch([dict(stream_meta(0, "s", 0, dataset="y"), epoch=1)], [np.array([0, 0, 1])])
    rules.check_coverage()  # one of the two y slots finished
    assert rules.coverage_checked == [0]
    rules.record_batch([dict(stream_meta(1, "s", 0, dataset="y"), epoch=1)], [np.array([0, 0, 0])])
    with pytest.raises(rt.StopRuleError, match="'b'"):
        rules.check_coverage()


# ---------------------------------------------------------------------------------- staged trainer


def narrow(model_config):
    """A narrower RVT for fast CPU runs; the stage, state, and resume logic is unchanged."""

    model_config["rvt"]["backbone"]["embed_dim"] = 32
    model_config["fusion"]["dim"] = 32
    model_config["embedding"]["hidden_dim"] = 32
    model_config["embedding"]["dim"] = 16
    model_config["parameter_interval"] = {"min": 0, "max": 90_000_000}
    return model_config


def inputs_for(stage, *extra, pretrain=None, note=True):
    argv = ["--stage", stage, "--run-dir", "unused", "--smoke", "--rvt-root", RVT_ROOT, *extra]
    if pretrain is not None:
        argv += ["--pretrain-checkpoint", str(pretrain)]
        if note:
            argv += ["--investigation-note", "smoke"]
    inputs = build_inputs(parse_args(argv))
    narrow(inputs.model_config)
    return inputs


class Recording(Trainer):
    """Records the clips and reset flags of every micro-batch."""

    def __init__(self, *args, **kwargs):
        self.seen = []
        super().__init__(*args, **kwargs)

    def _micro_batch(self, batch):
        self.seen.append(
            (
                self.step,
                tuple(
                    (m["kind"], m["dataset"], m["sequence"], m["start"], m["length"], m["reset"])
                    for m in batch["meta"]
                ),
            )
        )
        return super()._micro_batch(batch)


def run(stage, run_dir, *extra, resume=False, limits=None, micro=1, cls=Recording, pretrain=None):
    trainer = cls(
        inputs_for(stage, *extra, pretrain=pretrain),
        run_dir,
        "cpu",
        micro_batches=micro,
        resume=resume,
        limits=limits,
        log=lambda message: None,
    )
    return trainer, trainer.run()


def assert_states_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        if left[key].dtype.is_floating_point:
            assert torch.allclose(left[key], right[key], rtol=1e-6, atol=0), key
        else:
            assert torch.equal(left[key], right[key]), key


def metric_lines(path):
    timing = {
        "frames_per_second",
        "data_wait_fraction",
        "seconds_per_step",
        "host_peak_rss_mib",
        "validation_seconds",
        "checkpoint_seconds",
    }
    return [
        {key: value for key, value in json.loads(line).items() if key not in timing}
        for line in Path(path).read_text().splitlines()
    ]


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


@pytest.fixture(scope="module")
def pretrain_run(tmp_path_factory):
    if not HAS_RVT or importlib.util.find_spec("pycocotools") is None:
        pytest.skip("trainer runs need RVT_ROOT and pycocotools")
    run_dir = tmp_path_factory.mktemp("pretrain_full")
    trainer, code = run("pretrain", run_dir)
    assert code == EXIT_COMPLETE
    return trainer, run_dir


@needs_rvt
def test_phases_freeze_parameters_and_their_batchnorm():
    inputs = inputs_for("pretrain")
    model = build_large_event_detector(
        inputs.model_config,
        ontology="source",
        geometry="gen4_1mpx",
        init="random",
        rvt_root=RVT_ROOT,
    )
    counts = rt.apply_phase(model, None, inputs.settings, "phase_a")
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    assert trainable and all(
        name.startswith(("backbone.stages.0.downsample_cf2cl.", "yolox_head.cls_preds."))
        for name in trainable
    )
    assert counts["trainable"] == sum(
        p.numel() for name, p in model.named_parameters() if name in trainable
    )
    norms = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert norms and all(not m.training for m in norms) and model.training
    rt.apply_phase(model, None, inputs.settings, "phase_b")
    frozen = {name for name, p in model.named_parameters() if not p.requires_grad}
    assert frozen and all(name.startswith("embedding_head.") for name in frozen)
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert module.training == (not name.startswith("embedding_head."))


@needs_rvt
def test_optimizer_groups_and_weight_decay():
    finetune = inputs_for("finetune")
    model = build_large_event_detector(
        finetune.model_config,
        ontology="target",
        geometry="dsec_mot",
        init="random",
        rvt_root=RVT_ROOT,
    )
    classifier = build_identity_classifier(
        loss_settings(finetune.training_config, finetune.model_config),
        "finetune",
        len(finetune.data.identity_table),
    )
    optimizer, groups = rt.build_optimizer(model, classifier, finetune.settings)
    named = dict(rt.named_training_parameters(model, classifier))
    names = {id(p): name for name, p in named.items()}
    decay = {id(p) for p in optimizer.param_groups[0]["params"]}
    assert optimizer.param_groups[0]["weight_decay"] == 0.05
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    assert all(names[i].endswith("weight") and not names[i].endswith(".gamma") for i in decay)
    assert id(classifier.weight) in decay
    gammas = [name for name in named if name.endswith(".gamma")]
    assert gammas and all(id(named[name]) not in decay for name in gammas)
    assert groups["decay"] + groups["no_decay"] == sum(p.numel() for p in named.values())
    pretrain = inputs_for("pretrain")
    source = build_large_event_detector(
        pretrain.model_config,
        ontology="source",
        geometry="gen4_1mpx",
        init="random",
        rvt_root=RVT_ROOT,
    )
    optimizer, groups = rt.build_optimizer(source, None, pretrain.settings)
    assert all(group["weight_decay"] == 0.0 for group in optimizer.param_groups)
    embedding = sum(
        p.numel() for n, p in source.named_parameters() if n.startswith("embedding_head.")
    )
    assert (
        groups["decay"] + groups["no_decay"]
        == sum(p.numel() for p in source.parameters()) - embedding
    )


@needs_rvt
@needs_coco
def test_interrupted_pretraining_equals_the_uninterrupted_run(pretrain_run, tmp_path):
    full, full_dir = pretrain_run
    first, code = run("pretrain", tmp_path, limits=JobLimits(max_job_steps=3))
    assert code == EXIT_PAUSED and first.step == 3
    resumed, code = run("pretrain", tmp_path, resume=True)
    assert code == EXIT_COMPLETE and resumed.step == 6
    assert first.seen + resumed.seen == full.seen
    resets = [m for _, metas in full.seen for m in metas if m[0] == "stream" and m[5]]
    assert resets and all(m[3] == 0 for m in resets)  # streams reset at sequence starts only
    assert_states_equal(full.model.state_dict(), resumed.model.state_dict())
    assert_states_equal(full.ema.module.state_dict(), resumed.ema.module.state_dict())
    assert full.ema.updates == resumed.ema.updates == 6
    full_optimizer = full.optimizer.state_dict()["state"]
    resumed_optimizer = resumed.optimizer.state_dict()["state"]
    assert full_optimizer.keys() == resumed_optimizer.keys()
    for key in full_optimizer:
        assert_states_equal(full_optimizer[key], resumed_optimizer[key])
    assert full.bank.counters == resumed.bank.counters
    assert full.bank.cursors == resumed.bank.cursors
    assert metric_lines(full_dir / "metrics.jsonl") == metric_lines(tmp_path / "metrics.jsonl")
    assert json.loads((full_dir / "pretrain/selection.json").read_text()) == json.loads(
        (tmp_path / "pretrain/selection.json").read_text()
    )
    assert [job["reason"] for job in jsonl(tmp_path / "resources.jsonl")] == [
        "max job steps",
        "complete",
    ]
    # Pretraining never optimizes the embedding head.
    initial = build_large_event_detector(
        full.inputs.model_config,
        ontology="source",
        geometry="gen4_1mpx",
        init="random",
        rvt_root=RVT_ROOT,
    ).state_dict()
    embedding = [key for key in initial if key.startswith("embedding_head.")]
    assert embedding and all(torch.equal(full.model.state_dict()[k], initial[k]) for k in embedding)


@needs_rvt
@needs_coco
def test_torn_last_checkpoint_falls_back_to_last_prev(pretrain_run, tmp_path):
    full, full_dir = pretrain_run
    _, code = run("pretrain", tmp_path, limits=JobLimits(max_job_steps=3))
    assert code == EXIT_PAUSED
    (tmp_path / "last.pt").write_bytes(b"torn")  # a crash during the step-3 save
    resumed, code = run("pretrain", tmp_path, resume=True)
    assert code == EXIT_COMPLETE and resumed.seen[0][0] == 2
    assert_states_equal(full.model.state_dict(), resumed.model.state_dict())
    assert metric_lines(full_dir / "metrics.jsonl") == metric_lines(tmp_path / "metrics.jsonl")
    assert [record["step"] for record in jsonl(tmp_path / "validation.jsonl")] == [3, 6]


@needs_rvt
@needs_coco
def test_resume_guards(pretrain_run, tmp_path):
    _, full_dir = pretrain_run
    with pytest.raises(rt.RuntimeConfigError, match="already holds a run"):
        run("pretrain", full_dir)
    with pytest.raises(rt.RuntimeConfigError, match="settings changed"):
        run("pretrain", full_dir, "--set", "total_steps=8", resume=True)
    with pytest.raises(FileNotFoundError):
        run("pretrain", tmp_path, resume=True)
    before = (full_dir / "last.pt").stat().st_mtime_ns
    trainer, code = run("pretrain", full_dir, resume=True)  # a completed run is a no-op
    assert code == EXIT_COMPLETE and trainer.seen == []
    assert (full_dir / "last.pt").stat().st_mtime_ns == before


@needs_rvt
@needs_coco
def test_interrupted_finetuning_equals_the_uninterrupted_run(pretrain_run, tmp_path):
    _, pretrain_dir = pretrain_run
    best = pretrain_dir / "pretrain" / "best.pt"
    full, code = run("finetune", tmp_path / "full", pretrain=best)
    assert code == EXIT_COMPLETE
    part, code = run(
        "finetune", tmp_path / "part", pretrain=best, limits=JobLimits(max_job_steps=4)
    )
    assert code == EXIT_PAUSED
    rest, code = run("finetune", tmp_path / "part", pretrain=best, resume=True)
    assert code == EXIT_COMPLETE
    assert part.seen + rest.seen == full.seen
    assert_states_equal(full.model.state_dict(), rest.model.state_dict())
    assert_states_equal(full.classifier.state_dict(), rest.classifier.state_dict())
    assert_states_equal(full.ema.module.state_dict(), rest.ema.module.state_dict())
    # Mixed batches: both datasets in every step, each stream resets only at sequence starts.
    for _, metas in full.seen:
        assert {m[1] for m in metas} == {"dsec_mot", "dsec_detection"}
    requests = json.loads((tmp_path / "full/finetune/validation_requests.json").read_text())
    assert [candidate["step"] for candidate in requests["candidates"]] == [3, 6]
    assert requests["finished"] and requests["selection"]["split"] == "val"
    assert requests["selection"]["sequences"] == ["zurich_city_01_d"]
    for candidate in requests["candidates"]:
        path = tmp_path / "full" / candidate["path"]
        payload = rt.load_inference_checkpoint(path)
        assert payload["weights"] == "ema" and payload["ontology"] == "target"
        assert rt.file_sha256(path) == candidate["sha256"]
    point = rt.load_inference_checkpoint(tmp_path / "full/finetune/point_2_step_6.pt")
    assert_states_equal(point["model_state"], full.ema.module.state_dict())
    record = json.loads((tmp_path / "full/run.json").read_text())
    assert record["initialization"]["pretraining"]["sha256"] == rt.file_sha256(best)
    assert record["identity_table"]["identities"] == len(full.inputs.data.identity_table) > 0
    dropped = record["model"]["transfer"]["dropped"]
    assert any(key.startswith("embedding_head.") for key in dropped)
    metrics = jsonl(tmp_path / "full/metrics.jsonl")
    assert all(
        line["counts"]["steps_with_triplet"] == line["steps"]
        for line in metrics
        if line["phase"] == "phase_b"
    )


@needs_rvt
@needs_coco
def test_finetuning_initialization_guards(pretrain_run, tmp_path):
    _, pretrain_dir = pretrain_run
    best = pretrain_dir / "pretrain/best.pt"
    with pytest.raises(rt.RuntimeConfigError, match="investigation"):
        Trainer(
            inputs_for("finetune", pretrain=best, note=False),
            tmp_path / "a",
            "cpu",
            log=lambda m: None,
        )
    lonely = tmp_path / "no_selection"
    lonely.mkdir()
    shutil.copy(best, lonely / "best.pt")
    with pytest.raises(rt.RuntimeConfigError, match="selection.json"):
        run("finetune", tmp_path / "b", pretrain=lonely / "best.pt")
    run("finetune", tmp_path / "c", pretrain=best, limits=JobLimits(max_job_steps=3))
    with pytest.raises(rt.RuntimeConfigError, match="not a pretraining"):
        run("finetune", tmp_path / "d", pretrain=tmp_path / "c/finetune/point_1_step_3.pt")


@needs_rvt
@needs_coco
def test_non_finite_updates_are_skipped_then_stop(tmp_path, monkeypatch):
    original = trainer_module.stage_losses
    calls = {"count": 0}

    def poisoned(*args, **kwargs):
        loss = original(*args, **kwargs)
        calls["count"] += 1
        if calls["count"] >= 3:  # steps 2, 3, 4 are non-finite
            loss.total = loss.total * float("nan")
        return loss

    monkeypatch.setattr(trainer_module, "stage_losses", poisoned)
    trainer, code = run("pretrain", tmp_path)
    assert code == EXIT_STOP_RULE and trainer.step == 4 and trainer.stop_rules.skips == 3
    assert trainer.ema.updates == 2
    # The stop rule keeps the last regular save (step 4) as the resume point.
    assert rt.load_checkpoint(tmp_path / "last.pt")["step"] == 4
    assert jsonl(tmp_path / "resources.jsonl")[-1]["exit_code"] == EXIT_STOP_RULE


@needs_rvt
def test_skipped_update_leaves_weights_and_ema(tmp_path, monkeypatch):
    original = trainer_module.stage_losses
    captured = {}

    class Probe(Recording):
        def _micro_batch(self, batch):
            poisoned.step = self.step
            return super()._micro_batch(batch)

        def _optimizer_step(self, batches):
            if self.step == 1:
                captured["before"] = copy.deepcopy(self.model.state_dict())
                captured["ema"] = copy.deepcopy(self.ema.module.state_dict())
            super()._optimizer_step(batches)
            if self.step == 1:
                captured["after"] = self.model.state_dict()
                captured["ema_after"] = self.ema.module.state_dict()

    def poisoned(*args, **kwargs):
        loss = original(*args, **kwargs)
        if poisoned.step == 1:
            loss.total = loss.total + float("inf")
        return loss

    monkeypatch.setattr(trainer_module, "stage_losses", poisoned)
    trainer, code = run("pretrain", tmp_path, limits=JobLimits(max_job_steps=2), cls=Probe)
    assert code == EXIT_PAUSED and trainer.stop_rules.skips == 1
    for key, value in captured["before"].items():
        if value.dtype.is_floating_point and "running" not in key:
            assert torch.equal(value, captured["after"][key]), key
    for key, value in captured["ema"].items():
        assert torch.equal(value, captured["ema_after"][key]), key


@needs_rvt
def test_sigusr1_pauses_after_the_step_with_a_checkpoint(tmp_path):
    class Signalled(Recording):
        def _micro_batch(self, batch):
            if self.step == 1:
                os.kill(os.getpid(), signal.SIGUSR1)
            return super()._micro_batch(batch)

    previous = signal.getsignal(signal.SIGUSR1)
    trainer, code = run("pretrain", tmp_path, cls=Signalled)
    assert code == EXIT_PAUSED and trainer.step == 2
    assert rt.load_checkpoint(tmp_path / "last.pt")["step"] == 2
    assert signal.getsignal(signal.SIGUSR1) == previous
    assert jsonl(tmp_path / "resources.jsonl")[-1]["reason"] == "signal SIGUSR1"


@needs_rvt
@needs_coco
def test_micro_batches_keep_the_state_semantics(pretrain_run, tmp_path):
    full, _ = pretrain_run
    split, code = run("pretrain", tmp_path, micro=2)
    assert code == EXIT_COMPLETE

    def by_step(seen):
        return {
            step: sorted(m for s, metas in seen if s == step for m in metas) for step, _ in seen
        }

    assert by_step(split.seen) == by_step(full.seen)
    assert all(len(metas) == 2 for _, metas in split.seen)
    assert all(sum(m[0] == "stream" for m in metas) == 1 for _, metas in split.seen)
    assert split.bank.cursors == full.bank.cursors and split.bank.counters == full.bank.counters


@needs_rvt
def test_dry_run_reports_the_plan_and_writes_nothing(tmp_path, capsys):
    run_dir = tmp_path / "run"
    argv = ["--stage", "finetune", "--run-dir", str(run_dir), "--smoke", "--rvt-root", RVT_ROOT]
    assert trainer_module.main([*argv, "--dry-run"]) == EXIT_COMPLETE
    report = json.loads(capsys.readouterr().out)
    assert not run_dir.exists()
    assert report["schedule"]["validations"] == [3, 6]
    assert report["model"]["parameters"]["total"] == 22_453_860
    assert report["batch"]["clips"] == 4 and report["resume"] is None
    assert trainer_module.main([*argv, "--dry-run", "--resume"]) == 2  # nothing to resume
    assert trainer_module.main([*argv, "--set", "total_steps=1"]) == 2


# -------------------------------------------------------------------------------------- source mAP


def collected(gt, predictions, images):
    return dmap._Collected(
        images=[{"id": index} for index in range(1, images + 1)],
        ground_truth=[(i, np.array(box, float), label) for i, box, label in gt],
        predictions=[
            (i, np.array(box, float), score, label) for i, box, score, label in predictions
        ],
    )


@needs_coco
def test_coco_metrics_on_known_cases():
    gt = [(1, [0, 0, 10, 10], 0), (2, [5, 5, 25, 25], 1)]
    perfect = collected(gt, [(1, [0, 0, 10, 10], 0.9, 0), (2, [5, 5, 25, 25], 0.8, 1)], 3)
    result = dmap.coco_metrics(perfect, (0, 1, 2), 100)
    assert result["map"] == pytest.approx(1.0) and result["ap50"] == pytest.approx(1.0)
    assert result["per_class_ap"] == {0: pytest.approx(1.0), 1: pytest.approx(1.0), 2: None}
    # Class confusion: class 0 is missed; class 1 has a confident FP before its TP.
    confused = collected(gt, [(1, [0, 0, 10, 10], 0.9, 1), (2, [5, 5, 25, 25], 0.8, 1)], 3)
    result = dmap.coco_metrics(confused, (0, 1), 100)
    assert result["per_class_ap"][0] == 0.0
    assert result["per_class_ap"][1] == pytest.approx(0.5, abs=0.01)
    assert dmap.coco_metrics(collected(gt, [], 2), (0, 1), 100)["map"] == 0.0
    assert dmap.coco_metrics(collected([], [], 2), (0, 1), 100)["map"] is None
    # IoU 1/3 matches at no COCO threshold.
    shifted = collected(gt[:1], [(1, [5, 0, 15, 10], 0.9, 0)], 1)
    assert dmap.coco_metrics(shifted, (0,), 100)["map"] == 0.0


def test_size_rule():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 40, 5], [0, 0, 30, 30], [0, 0, 0, 5]])
    gen4 = dmap.passes_size_rule(boxes, {"min_diagonal": 30, "min_side": 10})
    assert gen4.tolist() == [False, False, True, False]
    floor = dmap.passes_size_rule(boxes, {"min_area": 614.4})
    assert floor.tolist() == [False, False, True, False]


@needs_rvt
def test_validation_lanes_are_isolated_and_test_split_rejected(monkeypatch):
    from src.data.large_event_dataset import LargeEventClipDataset
    from src.models.large_event_detector import Detections

    inputs = inputs_for("pretrain")
    detector = build_large_event_detector(
        inputs.model_config,
        ontology="source",
        geometry="gen4_1mpx",
        init="random",
        rvt_root=RVT_ROOT,
    )
    validation = inputs.validation_sequences
    assert len(validation) == 2
    dataset = LargeEventClipDataset(validation, inputs.data_config)
    no_size_rule = {"min_diagonal": None, "min_side": None, "min_area": None}
    # Record every decoded row's raw head output; the score carries the record index. A
    # random-init head has near-tied scores, so comparing decoded boxes would be unstable.
    records = []

    def recording(head, *args, **kwargs):
        detections = []
        for row in range(head.obj_logits.shape[0]):
            records.append(head.train_layout()[row].clone())
            detections.append(
                Detections(
                    boxes=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
                    scores=torch.tensor([float(len(records) - 1)]),
                    labels=torch.tensor([0]),
                    anchors=torch.tensor([0]),
                    cap_reached=False,
                )
            )
        return detections

    monkeypatch.setattr(dmap, "decode_detections", recording)

    def outputs(lanes, clip_steps):
        block = dict(inputs.settings["validation"], lanes=lanes, clip_steps=clip_steps)
        settings = dmap.map_settings(block, detector.classes, no_size_rule)
        found = dmap.collect_detections(detector, dataset, validation, settings, "cpu")
        key = {image["id"]: (image["sequence"], image["step"]) for image in found.images}
        return {key[image]: records[int(score)] for image, _, score, _ in found.predictions}

    reference = outputs(1, 3)
    supervised = sum(int(sequence.supervised.sum()) for sequence in validation)
    assert len(reference) == supervised
    for lanes, clip_steps in ((2, 3), (3, 4)):
        other = outputs(lanes, clip_steps)
        assert other.keys() == reference.keys()
        for key, value in reference.items():
            torch.testing.assert_close(other[key], value, rtol=0.0, atol=2e-4)
    # Negative control: without the per-sequence reset, the second sequence of a shared lane
    # starts from the first one's state, which moves its outputs well beyond that tolerance.
    monkeypatch.setattr(RecurrentState, "reset", lambda self, mask: self)
    leaked = outputs(1, 3)
    second = [key for key in reference if key[0] == validation[1].sequence]
    assert max(float((leaked[key] - reference[key]).abs().max()) for key in second) > 5e-4
    test_split = [type(validation[0])(**{**validation[0].__dict__, "split": "test"})]
    settings = dmap.map_settings(inputs.settings["validation"], detector.classes, no_size_rule)
    with pytest.raises(dmap.DetectionMapError, match="test"):
        dmap.collect_detections(detector, dataset, test_split, settings, "cpu")
