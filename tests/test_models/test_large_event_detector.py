"""Tests for the RVT-B detector with gated EF/VG fusion, recurrence, and embeddings.

The model needs the pinned RVT checkout: set ``RVT_ROOT`` to it. Checkpoint tests also
need the verified RVT-B 1Mpx checkpoint in ``RVT_CHECKPOINT``. Without them the tests
skip; nothing is downloaded. Most tests use a reduced ``tiny`` geometry on the CPU.
"""

from __future__ import annotations

import copy
import importlib.util
import io
import math
import os
import pickle
import sys
import zipfile
from pathlib import Path

import pytest
import torch

from src.models import large_event_detector as det
from src.models import large_event_modules as mod
from src.models.large_event_modules import RecurrentState

RVT_ROOT = os.environ.get("RVT_ROOT")
RVT_CHECKPOINT = os.environ.get("RVT_CHECKPOINT")
pytestmark = [
    pytest.mark.skipif(
        not RVT_ROOT or not Path(RVT_ROOT).is_dir(),
        reason="set RVT_ROOT to the pinned RVT checkout to run the large detector tests",
    ),
    pytest.mark.skipif(
        any(importlib.util.find_spec(name) is None for name in ("einops", "omegaconf")),
        reason="RVT needs einops and omegaconf (requirements/large_event_mot.txt)",
    ),
]
needs_checkpoint = pytest.mark.skipif(
    not RVT_CHECKPOINT or not Path(RVT_CHECKPOINT).is_file(),
    reason="set RVT_CHECKPOINT to the verified RVT-B 1Mpx checkpoint",
)
PRIOR = -math.log((1 - 0.01) / 0.01)
TINY = (128, 192)


@pytest.fixture(scope="module")
def config():
    config = det.load_model_config()
    # Reduced CPU geometry: stage maps 32x48 down to 4x6, partition (2, 3).
    config["geometry"]["tiny"] = {
        "input_hw": list(TINY),
        "valid_hw": [120, 192],
        "partition_size": [2, 3],
    }
    return config


@pytest.fixture(scope="module")
def model(config):
    model = det.build_large_event_detector(
        config, geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    model.eval()
    return model


@pytest.fixture(scope="module")
def source_model(config):
    if not RVT_CHECKPOINT or not Path(RVT_CHECKPOINT).is_file():
        pytest.skip("set RVT_CHECKPOINT to the verified RVT-B 1Mpx checkpoint")
    return det.build_large_event_detector(
        config, geometry="tiny", rvt_root=RVT_ROOT, checkpoint=RVT_CHECKPOINT
    )


@pytest.fixture(scope="module")
def checkpoint(config):
    if not RVT_CHECKPOINT or not Path(RVT_CHECKPOINT).is_file():
        pytest.skip("set RVT_CHECKPOINT to the verified RVT-B 1Mpx checkpoint")
    return mod.read_rvt_checkpoint(RVT_CHECKPOINT, config["checkpoint"])


def inputs(batch=2, hw=TINY, seed=0, scale=1.0):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.rand(batch, 2, *hw, generator=generator) * scale,
        torch.rand(batch, 6, *hw, generator=generator) * scale,
    )


# --------------------------------------------------------------------------------- upstream access


def test_import_shim_isolates_project_packages(config):
    import src.models as project_models

    before_path = list(sys.path)
    before = {n: m for n, m in sys.modules.items() if n.split(".")[0] in mod.RVT_TOP_LEVEL}
    upstream = mod.import_rvt(RVT_ROOT, config["upstream"]["commit"])
    assert sys.path == before_path
    after = {n: m for n, m in sys.modules.items() if n.split(".")[0] in mod.RVT_TOP_LEVEL}
    assert after == before
    assert upstream.commit == config["upstream"]["commit"]
    rnn = sys.modules["_rvt_upstream.models.layers.rnn"]
    assert Path(rnn.__file__).resolve().is_relative_to(Path(RVT_ROOT).resolve())
    assert sys.modules["src.models"] is project_models


def test_import_shim_survives_a_colliding_package(config, tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "__init__.py").write_text("MARK = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    import models  # noqa: F401  A regular package named like RVT's namespace package.

    saved = dict(mod._RVT_CACHE)
    mod._RVT_CACHE.clear()
    try:
        upstream = mod.import_rvt(RVT_ROOT, config["upstream"]["commit"])
        assert sys.modules["models"].MARK == 1
        location = Path(upstream.module("models.layers.rnn").__file__).resolve()
        assert location.is_relative_to(Path(RVT_ROOT).resolve())
    finally:
        del sys.modules["models"]
        mod._RVT_CACHE.update(saved)


def test_import_shim_rejects_wrong_commit_and_missing_code(tmp_path):
    with pytest.raises(mod.UpstreamError, match="pinned"):
        mod.import_rvt(RVT_ROOT, "0" * 40)
    with pytest.raises(mod.UpstreamError, match="RVT model code"):
        mod.import_rvt(tmp_path, "0" * 40)


def test_runtime_paths_are_required(config, monkeypatch):
    monkeypatch.delenv("RVT_ROOT", raising=False)
    monkeypatch.delenv("RVT_CHECKPOINT", raising=False)
    with pytest.raises(det.ModelConfigError, match="RVT_ROOT"):
        det.resolve_rvt_root(config)
    with pytest.raises(det.ModelConfigError, match="RVT_CHECKPOINT"):
        det.resolve_checkpoint(config)
    monkeypatch.setenv("RVT_ROOT", "/x/rvt")
    assert det.resolve_rvt_root(config) == Path("/x/rvt")
    assert det.resolve_rvt_root(config, "/y") == Path("/y")


# ------------------------------------------------------------------------------- checkpoint reader


def settings_for(path, base):
    settings = copy.deepcopy(base)
    settings["size_bytes"] = path.stat().st_size
    settings["sha256"] = mod.file_sha256(path)
    return settings


def test_checkpoint_reader_round_trip_and_hash_checks(config, tmp_path):
    state = {
        "mdl.a": torch.randn(3, 4),
        "mdl.b": torch.arange(5),
        "mdl.c": torch.randn(2).double(),
    }
    path = tmp_path / "t.ckpt"
    torch.save({"state_dict": state, "global_step": 7}, path)
    settings = settings_for(path, config["checkpoint"])
    with pytest.raises(mod.UpstreamError, match="tensor SHA-256"):
        mod.read_rvt_checkpoint(path, settings)
    settings["tensor_sha256"] = mod.tensor_sha256({k[4:]: v for k, v in state.items()})
    loaded = mod.read_rvt_checkpoint(path, settings)
    assert loaded.global_step == 7
    for key, value in state.items():
        assert torch.equal(loaded.state[key[4:]], value)
        assert loaded.state[key[4:]].dtype == value.dtype
    wrong = dict(settings, size_bytes=settings["size_bytes"] + 1)
    with pytest.raises(mod.UpstreamError, match="expected"):
        mod.read_rvt_checkpoint(path, wrong)
    with pytest.raises(mod.UpstreamError, match="SHA-256"):
        mod.read_rvt_checkpoint(path, dict(settings, sha256="0" * 64))
    with pytest.raises(mod.UpstreamError, match="does not exist"):
        mod.read_rvt_checkpoint(tmp_path / "none.ckpt", settings)


class _Malicious:
    def __reduce__(self):
        return (os.system, ("echo unsafe",))


def test_checkpoint_reader_rejects_disallowed_globals(config, tmp_path):
    path = tmp_path / "unsafe.ckpt"
    with zipfile.ZipFile(path, "w") as archive:
        payload = pickle.dumps({"state_dict": {}, "x": _Malicious()}, protocol=2)
        archive.writestr("archive/data.pkl", payload)
    with pytest.raises(pickle.UnpicklingError, match="not allowed"):
        mod.read_rvt_checkpoint(path, settings_for(path, config["checkpoint"]))


@needs_checkpoint
def test_pinned_checkpoint_provenance(checkpoint, config):
    assert len(checkpoint.state) == config["checkpoint"]["tensors"] == 424
    assert checkpoint.global_step == 390000
    assert checkpoint.state[det.RVT_STEM_CONV].shape == (64, 20, 7, 7)
    assert checkpoint.tensor_sha256 == config["checkpoint"]["tensor_sha256"]


# ------------------------------------------------------------------------------------- fusion stem


def test_bin_overlap_and_stem_adaptation_keep_the_rvt_response():
    overlap = mod.bin_overlap(10, 3)
    assert torch.allclose(overlap.sum(1), torch.full((3,), 10 / 3, dtype=torch.float64))
    assert torch.allclose(overlap.sum(0), torch.ones(10, dtype=torch.float64))
    assert overlap[0, 3].item() == pytest.approx(1 / 3)
    generator = torch.Generator().manual_seed(1)
    weight = torch.randn(64, 20, 7, 7, generator=generator, dtype=torch.float64)
    w_ef, w_vg = mod.adapt_rvt_stem(weight, 2, 10, 3)
    assert w_ef.shape == (64, 2, 7, 7) and w_vg.shape == (64, 6, 7, 7)

    def conv(x, kernel):
        return torch.nn.functional.conv2d(x, kernel, stride=4, padding=3)

    # Counts spread evenly over RVT's 10 bins give the same stem response.
    ef = torch.rand(1, 2, 16, 16, generator=generator, dtype=torch.float64)
    rvt_ef = ef[:, :, None].expand(1, 2, 10, 16, 16).reshape(1, 20, 16, 16) / 10
    assert torch.allclose(conv(rvt_ef, weight), conv(ef, w_ef))
    vg = torch.rand(1, 2, 3, 16, 16, generator=generator, dtype=torch.float64)
    rvt_vg = torch.einsum("bt,npbhw->npthw", overlap, vg) * (3 / 10)
    assert torch.allclose(
        conv(rvt_vg.reshape(1, 20, 16, 16), weight), conv(vg.reshape(1, 6, 16, 16), w_vg)
    )
    with pytest.raises(mod.UpstreamError, match="channels"):
        mod.adapt_rvt_stem(weight[:, :18], 2, 10, 3)


def test_gate_equation_and_branch_separation(config):
    stem = mod.GatedFusionStem(2, 6, config["fusion"])
    assert torch.equal(stem.gate.weight, torch.zeros_like(stem.gate.weight))
    torch.nn.init.normal_(stem.gate.weight, std=0.05)
    torch.nn.init.normal_(stem.gate.bias, std=0.5)
    ef, vg = inputs(1, (32, 32))
    fused_map = stem((ef, vg))
    output = stem.last_output
    expected = torch.sigmoid(stem.gate(torch.cat((output.event_frame, output.voxel_grid), 1)))
    assert torch.allclose(output.gate, expected)
    assert 0 < output.gate.min() and output.gate.max() < 1 and output.gate.std() > 0.01
    fused = output.gate * output.event_frame + (1 - output.gate) * output.voxel_grid
    assert torch.allclose(fused_map, stem.norm(fused.permute(0, 2, 3, 1)))
    assert fused_map.shape == (1, 8, 8, 64)
    # Changing VG leaves F_EF unchanged and vice versa.
    stem((ef, torch.rand_like(vg)))
    assert torch.equal(stem.last_output.event_frame, output.event_frame)
    assert not torch.equal(stem.last_output.voxel_grid, output.voxel_grid)
    stem((torch.rand_like(ef), vg))
    assert torch.equal(stem.last_output.voxel_grid, output.voxel_grid)
    with pytest.raises(ValueError, match="expected"):
        stem((vg, ef))


# ------------------------------------------------------------------------ structure and accounting


def test_full_model_parameter_accounting(model):
    accounting = det.model_accounting(model)
    parameters = accounting["parameters"]
    assert parameters["groups"] == {
        "event_frame_branch": 42_880,
        "voxel_grid_branch": 55_424,
        "fusion": 8_384,
        "backbone": 12_721_920,
        "fpn": 3_860_992,
        "head": 1_892_644,
        "embedding_head": 3_871_616,
        "other": 0,
    }
    assert parameters["total"] == 22_453_860 == model.metadata["parameters"]["total"]
    assert parameters["trainable"] == parameters["total"]
    assert 20_000_000 <= parameters["total"] <= 90_000_000
    assert parameters["interval"] == [20_000_000, 90_000_000]
    assert accounting["training_only"] == {"identity_classifier": 40_960}
    state = accounting["recurrent_state"]
    assert state["gen4_1mpx"]["floats_per_stream"] == 3_686_400
    assert state["dsec_mot"]["floats_per_stream"] == 4_608_000
    # Frozen parameters still count; shared parameters count once.
    frozen = copy.deepcopy(model)
    frozen.backbone.requires_grad_(False)
    counts = det.parameter_counts(frozen)
    assert (
        counts["total"] == 22_453_860
        and counts["trainable"] == 22_453_860 - 12_721_920 - 98_304 - 8_384
    )
    shared = torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))
    shared.append(shared[0])
    assert det.parameter_counts(shared)["total"] == 4
    narrow = copy.deepcopy(model)
    narrow.config["parameter_interval"] = {"min": 22_453_861, "max": 90_000_000}
    with pytest.raises(det.ModelConfigError, match="outside"):
        det.model_accounting(narrow)


def test_shapes_gate_initialization_and_head_layout(model):
    with torch.no_grad():
        output = model(*inputs())
    fusion = output.fusion
    assert (
        fusion.event_frame.shape == fusion.voxel_grid.shape == fusion.gate.shape == (2, 64, 32, 48)
    )
    assert torch.allclose(fusion.gate, torch.full_like(fusion.gate, 0.5))
    assert model.fusion.last_output is None
    assert [tuple(t.shape[1:]) for t in output.backbone_features.values()] == [
        (64, 32, 48),
        (128, 16, 24),
        (256, 8, 12),
        (512, 4, 6),
    ]
    assert [tuple(t.shape[1:]) for t in output.fpn_features] == [
        (128, 16, 24),
        (256, 8, 12),
        (512, 4, 6),
    ]
    anchors = 16 * 24 + 8 * 12 + 4 * 6
    assert output.head.level_hw == ((16, 24), (8, 12), (4, 6))
    assert output.head.boxes.shape == (2, anchors, 4) and output.head.cls_logits.shape[-1] == 7
    assert output.embeddings.shape == (2, anchors, 128)
    assert output.head.train_layout().shape == (2, anchors, 12)
    assert output.head.strides.unique().tolist() == [8.0, 16.0, 32.0]


def test_head_matches_upstream_decoding_and_training_layout(model):
    head = model.yolox_head
    with torch.no_grad():
        output = model(*inputs())
        upstream, _ = head(list(output.fpn_features))
        parts = []
        for level, (x, stride) in enumerate(zip(output.fpn_features, head.strides)):
            x = head.stems[level](x)
            reg = head.reg_convs[level](x)
            raw = torch.cat(
                [
                    head.reg_preds[level](reg),
                    head.obj_preds[level](reg),
                    head.cls_preds[level](head.cls_convs[level](x)),
                ],
                1,
            )
            parts.append(head.get_output_and_grid(raw, level, stride, x.type())[0])
    assert torch.allclose(upstream[..., :4], output.head.boxes, rtol=1e-5, atol=1e-4)
    assert torch.allclose(upstream[..., 4], torch.sigmoid(output.head.obj_logits), atol=1e-6)
    assert torch.allclose(upstream[..., 5:], torch.sigmoid(output.head.cls_logits), atol=1e-6)
    assert torch.allclose(torch.cat(parts, 1), output.head.train_layout(), rtol=1e-5, atol=1e-4)


def test_branch_gate_recurrence_and_embedding_gradients(config):
    model = det.build_large_event_detector(
        config, geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    model.train()
    ef, vg = inputs()
    ef.requires_grad_(True)
    vg.requires_grad_(True)
    first = model(ef, vg)
    second = model(*inputs(seed=1), state=first.state)
    loss = (
        second.head.train_layout().square().mean()
        + second.embeddings[..., 0].mean()
        + first.head.train_layout().square().mean()
    )
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
    fusion = model.fusion
    for branch in (fusion.event_frame, fusion.voxel_grid):
        for name, parameter in branch.named_parameters():
            assert parameter.grad.abs().sum() > 0, name
    assert fusion.event_frame.stem.weight.grad.shape == (64, 2, 7, 7)
    assert fusion.voxel_grid.stem.weight.grad.shape == (64, 6, 7, 7)
    assert fusion.gate.weight.grad.abs().sum() > 0
    for stage in range(4):
        assert (
            model.get_parameter(f"backbone.stages.{stage}.lstm.conv1x1.weight").grad.abs().sum() > 0
        )
    assert model.get_parameter("embedding_head.levels.0.3.weight").grad.abs().sum() > 0
    assert ef.grad.abs().sum() > 0 and vg.grad.abs().sum() > 0


def test_gradients_stay_in_their_branch(config):
    model = det.build_large_event_detector(
        config, geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    model(*inputs(1)).fusion.event_frame.sum().backward()
    assert all(
        p.grad is None or p.grad.abs().sum() == 0 for p in model.fusion.voxel_grid.parameters()
    )
    assert model.fusion.event_frame.stem.weight.grad.abs().sum() > 0
    model.zero_grad()
    head_only = model(*inputs(1), with_embeddings=False)
    assert head_only.embeddings is None
    head_only.head.train_layout().square().mean().backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in model.embedding_head.parameters())


def test_bf16_autocast_keeps_fp32_outputs_and_state(model):
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        first = model(*inputs(1))
        second = model(*inputs(1, seed=1), state=first.state)
    assert second.head.boxes.dtype == second.head.cls_logits.dtype == torch.float32
    assert second.embeddings.dtype == torch.float32
    assert all(t.dtype == torch.float32 for pair in second.state.stages for t in pair)
    assert torch.isfinite(second.head.train_layout()).all()


def test_geometry_switch_resets_grids_and_rejects_stale_state(config):
    model = det.build_large_event_detector(
        config, geometry="gen4_1mpx", init="random", rvt_root=RVT_ROOT
    )
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 2, 384, 640), torch.zeros(1, 6, 384, 640))
        model.yolox_head(list(output.fpn_features))
        assert output.head.boxes.shape[1] == 5040
        model.set_geometry("dsec_mot")
        partitions = {
            m.partition_size for m in model.modules() if isinstance(m, model._attention_type)
        }
        assert partitions == {(5, 10)}
        ef, vg = torch.zeros(1, 2, 480, 640), torch.zeros(1, 6, 480, 640)
        with pytest.raises(ValueError, match="does not match"):
            model(ef, vg, state=output.state)
        output = model(ef, vg, state=model.initial_state(1))
        upstream, _ = model.yolox_head(list(output.fpn_features))
    assert output.head.boxes.shape[1] == upstream.shape[1] == 6300
    assert output.embeddings.shape == (1, 6300, 128)
    assert output.state.shapes()[0] == (64, 120, 160)
    with pytest.raises(ValueError, match="expects"):
        model(torch.zeros(1, 2, 384, 640), torch.zeros(1, 6, 384, 640))


def test_bad_geometry_rejected(config):
    bad = copy.deepcopy(config)
    bad["geometry"]["bad"] = {
        "input_hw": [480, 640],
        "valid_hw": [480, 640],
        "partition_size": [6, 10],
    }
    with pytest.raises(det.ModelConfigError, match="does not divide"):
        det.model_geometry(bad, "bad")
    with pytest.raises(det.ModelConfigError, match="no model geometry"):
        det.model_geometry(bad, "nope")


# --------------------------------------------------------------------------------- recurrent state


def assert_outputs_close(left, right):
    torch.testing.assert_close(
        left.head.train_layout(), right.head.train_layout(), rtol=1e-5, atol=1e-5
    )
    torch.testing.assert_close(left.embeddings, right.embeddings, rtol=1e-5, atol=1e-5)


def test_zero_state_equivalences(model):
    ef, vg = inputs()
    with torch.no_grad():
        none = model(ef, vg)
        zero = model(ef, vg, state=model.initial_state(2))
        noisy = RecurrentState.zeros(2, model.state_shapes())._map(torch.randn_like)
        reset = model(ef, vg, state=noisy, reset=torch.tensor([True, True]))
    assert_outputs_close(none, zero)
    assert_outputs_close(none, reset)
    assert none.state.stages[1][0].shape == (2, 128, 16, 24)
    assert all(t.dtype == torch.float32 for pair in none.state.stages for t in pair)


def test_carried_state_and_per_row_reset(model):
    steps = [inputs(seed=seed) for seed in (1, 2, 3)]
    with torch.no_grad():
        state = None
        for ef, vg in steps[:2]:
            state = model(ef, vg, state=state).state
        ef, vg = steps[2]
        carried = model(ef, vg, state=state)
        fresh = model(ef, vg)
        mixed = model(ef, vg, state=state, reset=torch.tensor([True, False]))
    assert not torch.allclose(carried.head.train_layout(), fresh.head.train_layout())
    torch.testing.assert_close(mixed.head.train_layout()[0], fresh.head.train_layout()[0])
    torch.testing.assert_close(mixed.head.train_layout()[1], carried.head.train_layout()[1])
    torch.testing.assert_close(mixed.embeddings[0], fresh.embeddings[0])


def test_no_state_leaks_across_sequences(model):
    sequence_a = [inputs(1, seed=seed) for seed in (10, 11, 12)]
    sequence_b = [inputs(1, seed=seed) for seed in (20, 21)]
    with torch.no_grad():
        state = None
        for ef, vg in sequence_a:
            state = model(ef, vg, state=state).state
        after = []
        for index, (ef, vg) in enumerate(sequence_b):
            reset = torch.tensor([True]) if index == 0 else None
            output = model(ef, vg, state=state, reset=reset)
            state = output.state
            after.append(output)
        state = None
        alone = []
        for ef, vg in sequence_b:
            output = model(ef, vg, state=state)
            state = output.state
            alone.append(output)
    for left, right in zip(after, alone):
        assert torch.equal(left.head.train_layout(), right.head.train_layout())
        assert torch.equal(left.state.stages[3][1], right.state.stages[3][1])


def test_invalid_rows_hold_their_state(model):
    (ef1, vg1), (ef2, vg2), (ef3, vg3) = (inputs(seed=seed) for seed in (30, 31, 32))
    with torch.no_grad():
        state = model(ef1, vg1).state
        held = model(ef2, vg2, state=state, valid=torch.tensor([True, False])).state
        stepped = model(ef2, vg2, state=state).state
        after_hold = model(ef3, vg3, state=held)
        skipped = model(ef3, vg3, state=state)
    for (h_held, c_held), (h_old, c_old), (h_new, c_new) in zip(
        held.stages, state.stages, stepped.stages
    ):
        assert torch.equal(h_held[1], h_old[1]) and torch.equal(c_held[1], c_old[1])
        assert torch.equal(h_held[0], h_new[0]) and torch.equal(c_held[0], c_new[0])
    torch.testing.assert_close(after_hold.head.train_layout()[1], skipped.head.train_layout()[1])


def test_bptt_reaches_earlier_steps_and_detach_cuts_it(config):
    model = det.build_large_event_detector(
        config, geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    model.train()
    steps = [inputs(seed=seed) for seed in (40, 41, 42)]
    first_ef = steps[0][0].clone().requires_grad_(True)
    state = None
    for index, (ef, vg) in enumerate(steps):
        output = model(first_ef if index == 0 else ef, vg, state=state)
        state = output.state
    (output.head.train_layout().square().mean() + output.embeddings[..., 0].mean()).backward()
    assert first_ef.grad is not None and first_ef.grad.abs().sum() > 0
    first_ef = steps[0][0].clone().requires_grad_(True)
    state = model(first_ef, steps[0][1]).state.detach()
    for ef, vg in steps[1:]:
        output = model(ef, vg, state=state)
        state = output.state
    output.head.train_layout().square().mean().backward()
    assert first_ef.grad is None
    assert all(not t.requires_grad for pair in state.detach().stages for t in pair)


def test_state_serialization_rows_and_checks(model):
    with torch.no_grad():
        state = model(*inputs(3, seed=60)).state
    buffer = io.BytesIO()
    torch.save(state.state_dict(), buffer)
    buffer.seek(0)
    restored = RecurrentState.from_state_dict(torch.load(buffer, weights_only=True))
    for (h, c), (h2, c2) in zip(state.stages, restored.stages):
        assert torch.equal(h, h2) and torch.equal(c, c2)
    rebuilt = RecurrentState.cat([state.rows([2]), state.rows([0, 1])]).rows([1, 2, 0])
    for left, right in zip(rebuilt.state_dict().values(), state.state_dict().values()):
        assert torch.equal(left, right)
    assert state.is_zero().tolist() == [False, False, False]
    assert state.reset(torch.tensor([False, True, False])).is_zero().tolist() == [
        False,
        True,
        False,
    ]
    assert state.floats_per_stream() == sum(2 * c * h * w for c, h, w in model.state_shapes())
    with pytest.raises(ValueError, match="keys"):
        RecurrentState.from_state_dict({"stage1.h": state.stages[0][0]})
    with pytest.raises(ValueError, match="float32"):
        RecurrentState(((state.stages[0][0].double(), state.stages[0][1].double()),))
    with pytest.raises(ValueError, match="bool mask"):
        state.reset(torch.tensor([1, 0, 0]))
    with pytest.raises(ValueError, match="different layouts"):
        RecurrentState.cat([state, RecurrentState.zeros(1, [(1, 2, 2)])])
    with pytest.raises(ValueError, match="does not match"):
        model(*inputs(2, seed=61), state=state)


# ------------------------------------------------------------------------- embeddings and decoding


def fake_head(boxes_xyxy, obj, cls):
    xyxy = torch.tensor(boxes_xyxy, dtype=torch.float32)[None]
    count = xyxy.shape[1]
    return det.HeadOutput(
        boxes=torch.cat(((xyxy[..., :2] + xyxy[..., 2:]) / 2, xyxy[..., 2:] - xyxy[..., :2]), -1),
        obj_logits=torch.logit(torch.tensor(obj))[None],
        cls_logits=torch.logit(torch.tensor(cls))[None],
        grid=torch.zeros(count, 2),
        strides=torch.ones(count),
        level_hw=((1, count),),
    )


def test_decode_floor_class_aware_nms_clip_and_cap():
    boxes = [
        [0, 0, 10, 10],
        [1, 1, 10, 10],
        [0, 0, 10, 10],
        [50, 50, 60, 60],
        [-5, 200, 10, 230],
        [5, 5, 15, 15],
    ]
    obj = [0.9, 0.8, 0.9, 0.04, 0.9, 0.5]
    cls = [[0.9, 0.1], [0.9, 0.1], [0.1, 0.8], [0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]
    head = fake_head(boxes, obj, cls)
    embeddings = torch.nn.functional.normalize(torch.randn(1, 6, 4), dim=-1)
    (detections,) = det.decode_detections(head, (100, 100), 0.05, 0.5, 300, embeddings)
    # 1 is suppressed by 0 (same class), 2 survives (other class), 3 is under the floor,
    # 4 has no height inside the valid region, 5 overlaps 0 below the NMS threshold.
    assert detections.anchors.tolist() == [0, 2, 5]
    assert detections.labels.tolist() == [0, 1, 0]
    assert torch.allclose(detections.scores, torch.tensor([0.81, 0.72, 0.45]))
    assert torch.equal(detections.embeddings, embeddings[0, [0, 2, 5]])
    assert not detections.cap_reached
    (capped,) = det.decode_detections(head, (100, 100), 0.05, 0.5, 2)
    assert capped.anchors.tolist() == [0, 2] and capped.cap_reached
    (clipped,) = det.decode_detections(
        fake_head([[-5, -5, 20, 120]], [0.9], [[0.9, 0.1]]), (100, 50), 0.05, 0.5, 300
    )
    assert clipped.boxes.tolist() == [[0.0, 0.0, 20.0, 100.0]]
    (empty,) = det.decode_detections(
        fake_head([[0, 0, 1, 1]], [0.01], [[0.5, 0.5]]),
        (100, 100),
        0.05,
        0.5,
        300,
        embeddings[:, :1],
    )
    assert empty.boxes.shape == (0, 4) and empty.embeddings.shape == (0, 4)
    with pytest.raises(ValueError, match="anchors"):
        det.decode_detections(head, (100, 100), 0.05, 0.5, 300, embeddings[:, :3])


def test_embeddings_unit_norm_grid_alignment_and_decoded_rows(model):
    ef, vg = inputs(seed=80)
    with torch.no_grad():
        output = model(ef, vg)
        norms = output.embeddings.norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms))
        offset = 0
        for level, (feature, (h, w)) in enumerate(zip(output.fpn_features, output.head.level_hw)):
            raw = model.embedding_head.levels[level](feature)
            y, x = h - 2, w // 3
            anchor = offset + y * w + x
            expected = torch.nn.functional.normalize(raw[1, :, y, x], dim=0)
            torch.testing.assert_close(output.embeddings[1, anchor], expected)
            assert output.head.grid[anchor].tolist() == [float(x), float(y)]
            assert output.head.strides[anchor].item() == 8 * 2**level
            offset += h * w
        # Force confident detections so the alignment check is not vacuous.
        head = copy.copy(output.head)
        head.obj_logits = torch.full_like(output.head.obj_logits, 10.0)
        head.cls_logits = output.head.cls_logits + 10.0
        detections = det.decode_detections(head, (120, 192), 0.05, 0.5, 300, output.embeddings)
        empty = model.decode(output, score_floor=1.5)
        without = model(ef, vg, with_embeddings=False)
    for image, result in enumerate(detections):
        assert result.boxes.shape[0] > 0
        assert torch.equal(result.embeddings, output.embeddings[image, result.anchors])
        assert torch.all(result.boxes[:, 3] <= 120)
    for result in empty:
        assert result.boxes.shape == (0, 4) and result.embeddings.shape == (0, 128)
    assert without.embeddings is None and model.decode(without)[0].embeddings is None


def test_decoding_threshold_is_a_prefix(model):
    with torch.no_grad():
        output = model(*inputs(seed=3))
    head = copy.copy(output.head)
    head.obj_logits = output.head.obj_logits + 6.0
    head.cls_logits = output.head.cls_logits + 6.0
    output = copy.copy(output)
    output.head = head
    low = model.decode(output, score_floor=0.05)
    high = model.decode(output, score_floor=0.3)
    assert any(len(result.anchors) for result in high)
    for a, b in zip(low, high):
        assert torch.equal(a.anchors[a.scores >= 0.3], b.anchors)


# --------------------------------------------------------------------- initialization and transfer


def test_random_initialization_is_seeded_and_reads_nothing(config, monkeypatch):
    monkeypatch.delenv("RVT_CHECKPOINT", raising=False)
    calls = []
    monkeypatch.setattr(det, "read_rvt_checkpoint", lambda *args: calls.append(args))
    first = det.build_large_event_detector(
        config, geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    second = det.build_large_event_detector(
        config, geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    assert calls == [] and first.metadata["checkpoint"] is None
    assert first.metadata["transfer"]["counts"]["new"] == len(first.state_dict())
    a, b = first.state_dict(), second.state_dict()
    assert all(torch.equal(a[key], b[key]) for key in a)
    target = det.build_large_event_detector(
        config, ontology="target", geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    assert target.classes == ("car", "pedestrian", "bicycle", "motorcycle", "bus", "truck", "train")
    with pytest.raises(det.ModelConfigError, match="source head only"):
        det.build_large_event_detector(
            config, ontology="target", rvt_root=RVT_ROOT, checkpoint="unused"
        )
    with pytest.raises(det.ModelConfigError, match="unknown initialization"):
        det.build_large_event_detector(config, init="pretrained", rvt_root=RVT_ROOT)


@needs_checkpoint
def test_checkpoint_load_report(source_model, checkpoint):
    report = source_model.metadata["transfer"]
    assert report["counts"] == {"loaded": 415, "adapted": 10, "new": 84, "dropped": 0}
    adapted_from_checkpoint = {key for key in report["adapted"] if key in checkpoint.state}
    assert set(report["loaded"]) | {det.RVT_STEM_CONV} | adapted_from_checkpoint == set(
        checkpoint.state
    )
    assert sum(key.startswith(det.EMBEDDING_PREFIX) for key in report["new"]) == 60
    for key in report["new"]:
        assert key.startswith(det.EMBEDDING_PREFIX) or (
            key.startswith(det.FUSION_PREFIX)
            and any(part in key for part in (".event_frame.", ".voxel_grid.", ".gate."))
        )
    state = source_model.state_dict()
    for key in report["loaded"]:
        assert torch.equal(state[key], checkpoint.state[key]), key
    for level in range(3):
        key = f"yolox_head.cls_preds.{level}"
        assert torch.equal(state[key + ".weight"][:3], checkpoint.state[key + ".weight"])
        assert torch.equal(state[key + ".bias"][:3], checkpoint.state[key + ".bias"])
        assert torch.allclose(state[key + ".bias"][3:], torch.full((4,), PRIOR))
    norm = det.FUSION_PREFIX + "norm.weight"
    assert torch.equal(state[norm], checkpoint.state[norm])
    assert torch.equal(state[det.FUSION_PREFIX + "gate.weight"], torch.zeros(64, 128, 1, 1))
    metadata = source_model.metadata
    assert metadata["checkpoint"]["tensor_sha256"] == checkpoint.tensor_sha256
    assert metadata["upstream"]["commit"] == source_model.config["upstream"]["commit"]
    assert metadata["parameters"]["total"] == 22_453_860


@needs_checkpoint
def test_checkpoint_load_is_deterministic(config, source_model):
    again = det.build_large_event_detector(
        config, geometry="tiny", rvt_root=RVT_ROOT, checkpoint=RVT_CHECKPOINT
    )
    assert again.metadata["transfer"] == source_model.metadata["transfer"]
    a, b = source_model.state_dict(), again.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


def test_class_head_transfer_to_seven_targets(config, model):
    source_state = {key: value.clone() for key, value in model.state_dict().items()}
    for key in source_state:
        if key.startswith(det.EMBEDDING_PREFIX) and source_state[key].is_floating_point():
            source_state[key] += 1.0
    target = det.transfer_source_to_target(source_state, config, geometry="tiny", rvt_root=RVT_ROOT)
    fresh = det.build_large_event_detector(
        config, ontology="target", geometry="tiny", init="random", rvt_root=RVT_ROOT
    )
    assert target.ontology == "target" and target.metadata["init"] == "source_transfer"
    report = target.metadata["transfer"]
    assert report["counts"]["adapted"] == 6
    embedding_keys = {key for key in target.state_dict() if key.startswith(det.EMBEDDING_PREFIX)}
    assert set(report["new"]) == embedding_keys and len(embedding_keys) == 60
    assert {"yolox_head.cls_preds.*[5]", "yolox_head.cls_preds.*[6]"} <= set(report["dropped"])
    assert embedding_keys <= set(report["dropped"])
    state = target.state_dict()
    # Two-wheeler initializes both bicycle and motorcycle; train is new at the prior.
    source_rows = {"car": 2, "pedestrian": 0, "bicycle": 1, "motorcycle": 1, "bus": 4, "truck": 3}
    for level in range(3):
        for suffix in (".weight", ".bias"):
            key = f"yolox_head.cls_preds.{level}{suffix}"
            for row, name in enumerate(target.classes):
                if name in source_rows:
                    assert torch.equal(state[key][row], source_state[key][source_rows[name]])
        assert state[f"yolox_head.cls_preds.{level}.bias"][6].item() == pytest.approx(PRIOR)
    for key in report["loaded"]:
        assert torch.equal(state[key], source_state[key])
    assert len(report["loaded"]) == len(state) - 6 - 60
    for key in embedding_keys:
        assert torch.equal(state[key], fresh.state_dict()[key])
    assert target.metadata["parameters"]["total"] == 22_453_860
