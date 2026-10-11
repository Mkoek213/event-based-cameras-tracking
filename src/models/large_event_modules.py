"""Upstream access and EF/VG fusion modules for the large event MOT detector.

The pieces live here so that :mod:`src.models.large_event_detector` only assembles:

- :func:`import_rvt` loads the pinned RVT checkout (``RVT_ROOT``) without colliding with
  this project's top-level ``data``, ``models``, and ``utils`` packages. RVT's packages
  are namespace packages, so a regular package of the same name anywhere on
  ``sys.path`` wins even when the checkout comes first; the shim therefore imports with
  a ``sys.path`` that excludes every entry providing one of those names, re-registers
  the imported RVT modules under the private ``_rvt_upstream.`` prefix, and restores the
  project modules and ``sys.path`` afterwards. Every RVT import happens at module import
  time (the only function-level imports are guarded optional ones), so nothing resolves
  the bare names later.
- :func:`read_rvt_checkpoint` reads the public RVT Lightning checkpoint with a
  restricted unpickler: an explicit allowlist of globals, optimizer and scheduler
  objects replaced by inert stubs, and tensors rebuilt from the zip archive. It never
  calls ``torch.load``. Size, file SHA-256, and the canonical tensor SHA-256 are checked.
- :class:`GatedFusionStem` replaces RVT's stage-1 patch embedding
  (``stages[0].downsample_cf2cl``): separate Event Frame and Voxel Grid stems, each a
  7x7 stride-4 conv, LayerNorm, and one ConvNeXt block, joined per pixel and channel by
  ``Z = LN(g * F_EF + (1 - g) * F_VG)`` with ``g = sigmoid(Conv1x1([F_EF; F_VG]))``.
  :func:`adapt_rvt_stem` derives both stem convs from the pretrained 20-channel conv.
- :class:`RecurrentState` is the explicit carried state of the four RVT stage LSTMs:
  one fp32 ``(h, c)`` pair per stage with per-row reset, hold, detach, and serialization.
- :class:`EmbeddingHead` gives one L2-normalized descriptor per YOLOX anchor:
  ``E_l = L2(Conv1x1(CBS3x3(CBS3x3(CBS1x1(P_l)))))`` per PAFPN level.
"""

from __future__ import annotations

import collections
import hashlib
import importlib
import io
import os
import pickle
import subprocess
import sys
import types
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

RVT_MODULE_PREFIX = "_rvt_upstream"
# Top-level RVT packages that the used modules import; the project has the same names.
RVT_TOP_LEVEL = ("data", "models", "utils")
RVT_MODULES = (
    "models.detection.recurrent_backbone.maxvit_rnn",
    "models.layers.maxvit.maxvit",
    "models.layers.rnn",
    "models.detection.yolox_extension.models.build",
    "models.detection.yolox_extension.models.yolo_pafpn",
    "models.detection.yolox.models.yolo_head",
    "models.detection.yolox.models.network_blocks",
    "models.detection.yolox.models.losses",
)
# Third-party packages RVT imports; loaded before the import path is narrowed.
RVT_DEPENDENCIES = ("numpy", "torchvision", "omegaconf", "einops")


class UpstreamError(RuntimeError):
    """The RVT checkout or checkpoint does not match the pinned provenance."""


# --------------------------------------------------------------------------- RVT import


@dataclass(frozen=True)
class RvtUpstream:
    """Imported RVT modules, keyed by their RVT-relative dotted name."""

    root: Path
    commit: str
    modules: Mapping[str, types.ModuleType]

    def module(self, name: str) -> types.ModuleType:
        return self.modules[name]


_RVT_CACHE: dict[Path, RvtUpstream] = {}


def rvt_commit(root: Path) -> str:
    """``git rev-parse HEAD`` of an RVT checkout."""

    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise UpstreamError(f"{root} is not a readable git checkout of RVT: {error}") from error
    return result.stdout.strip()


def _provides_top_level(entry: str, names: tuple[str, ...]) -> bool:
    base = Path(entry or os.getcwd())
    return any((base / name).is_dir() or (base / f"{name}.py").is_file() for name in names)


def _is_top_level(name: str, tops: tuple[str, ...]) -> bool:
    return name.split(".", 1)[0] in tops


def import_rvt(root: str | Path, expected_commit: Optional[str]) -> RvtUpstream:
    """Import the used RVT modules from ``root`` under the ``_rvt_upstream.`` prefix.

    ``expected_commit`` is compared with ``git rev-parse HEAD``; ``None`` skips the check
    (tests only). The result is cached per resolved root.
    """

    root = Path(root).expanduser().resolve()
    if root in _RVT_CACHE:
        upstream = _RVT_CACHE[root]
        if expected_commit is not None and upstream.commit != expected_commit:
            raise UpstreamError(f"RVT commit {upstream.commit} != pinned {expected_commit}")
        return upstream
    if not (root / "models" / "detection" / "recurrent_backbone" / "maxvit_rnn.py").is_file():
        raise UpstreamError(f"{root} does not contain the RVT model code")
    commit = rvt_commit(root) if expected_commit is not None else "unverified"
    if expected_commit is not None and commit != expected_commit:
        raise UpstreamError(f"RVT checkout {root} is at {commit}, pinned {expected_commit}")

    for dependency in RVT_DEPENDENCIES:
        importlib.import_module(dependency)
    saved_path = list(sys.path)
    saved_modules = {
        name: module for name, module in sys.modules.items() if _is_top_level(name, RVT_TOP_LEVEL)
    }
    for name in saved_modules:
        del sys.modules[name]
    before = set(sys.modules)
    imported: dict[str, types.ModuleType] = {}
    try:
        sys.path[:] = [str(root)] + [
            entry
            for entry in saved_path
            if Path(entry or os.getcwd()).resolve() != root
            and not _provides_top_level(entry, RVT_TOP_LEVEL)
        ]
        importlib.invalidate_caches()
        for name in RVT_MODULES:
            importlib.import_module(name)
        for name in set(sys.modules) - before:
            if not _is_top_level(name, RVT_TOP_LEVEL):
                continue
            module = sys.modules[name]
            origin = getattr(module, "__file__", None)
            locations = [origin] if origin else list(getattr(module, "__path__", []))
            if not all(Path(location).resolve().is_relative_to(root) for location in locations):
                raise UpstreamError(f"RVT import resolved {name} outside {root}: {locations}")
            imported[name] = module
    finally:
        for name in [name for name in sys.modules if _is_top_level(name, RVT_TOP_LEVEL)]:
            del sys.modules[name]
        sys.modules.update(saved_modules)
        sys.path[:] = saved_path
        importlib.invalidate_caches()

    if RVT_MODULE_PREFIX not in sys.modules:
        sys.modules[RVT_MODULE_PREFIX] = types.ModuleType(RVT_MODULE_PREFIX)
    for name, module in imported.items():
        sys.modules[f"{RVT_MODULE_PREFIX}.{name}"] = module
    upstream = RvtUpstream(root=root, commit=commit, modules=dict(imported))
    _RVT_CACHE[root] = upstream
    return upstream


# --------------------------------------------------------------------------- checkpoint


class _InertObject:
    """Stands in for pickled optimizer, scheduler, and bound-method objects."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        pass

    def __setstate__(self, state: Any) -> None:
        pass


def _inert_getattr(*args: Any) -> _InertObject:
    return _InertObject()


_STORAGE_DTYPES = {
    "FloatStorage": torch.float32,
    "DoubleStorage": torch.float64,
    "LongStorage": torch.int64,
}
_ALLOWED_GLOBALS = {
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("collections", "defaultdict"): collections.defaultdict,
    ("__builtin__", "dict"): dict,
    ("builtins", "dict"): dict,
    ("__builtin__", "getattr"): _inert_getattr,
    ("builtins", "getattr"): _inert_getattr,
    ("torch._utils", "_rebuild_tensor_v2"): torch._utils._rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): torch._utils._rebuild_parameter,
    ("torch.optim.adamw", "AdamW"): _InertObject,
    ("torch.optim.lr_scheduler", "OneCycleLR"): _InertObject,
}


class _RestrictedUnpickler(pickle.Unpickler):
    def __init__(self, data: bytes, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(io.BytesIO(data))
        self._archive = archive
        self._prefix = prefix

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch" and name in _STORAGE_DTYPES:
            return name
        try:
            return _ALLOWED_GLOBALS[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError(f"global {module}.{name} is not allowed") from None

    def persistent_load(self, pid: Any) -> Any:
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, storage_type, key, _, numel = pid
        dtype = _STORAGE_DTYPES[storage_type]
        raw = self._archive.read(f"{self._prefix}/data/{key}")
        flat = torch.frombuffer(bytearray(raw), dtype=dtype)
        if flat.numel() != numel:
            raise pickle.UnpicklingError(f"storage {key}: {flat.numel()} != {numel} elements")
        return torch.storage.TypedStorage(
            wrap_storage=flat.untyped_storage(), dtype=dtype, _internal=True
        )


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """SHA-256 over sorted keys, then per tensor key, dtype, shape tuple, and raw bytes."""

    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RvtCheckpoint:
    """The ``mdl.``-stripped RVT state dict and its verified provenance."""

    state: dict[str, torch.Tensor]
    size_bytes: int
    sha256: str
    tensor_sha256: str
    global_step: Optional[int]


def read_rvt_checkpoint(path: str | Path, settings: Mapping[str, Any]) -> RvtCheckpoint:
    """Verify and read the RVT checkpoint described by ``settings`` (model config)."""

    path = Path(path).expanduser()
    if not path.is_file():
        raise UpstreamError(f"RVT checkpoint {path} does not exist")
    size = path.stat().st_size
    if size != settings["size_bytes"]:
        raise UpstreamError(f"{path}: {size} B, expected {settings['size_bytes']} B")
    sha256 = file_sha256(path)
    if sha256 != settings["sha256"]:
        raise UpstreamError(f"{path}: SHA-256 {sha256}, expected {settings['sha256']}")
    with zipfile.ZipFile(path) as archive:
        pickles = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(pickles) != 1:
            raise UpstreamError(f"{path}: expected one data.pkl, found {pickles}")
        prefix = pickles[0][: -len("/data.pkl")]
        content = _RestrictedUnpickler(archive.read(pickles[0]), archive, prefix).load()
    prefix = settings["state_dict_prefix"]
    raw_state = content["state_dict"]
    if not all(key.startswith(prefix) for key in raw_state):
        raise UpstreamError(f"{path}: state_dict keys without the {prefix!r} prefix")
    state = {key[len(prefix) :]: tensor for key, tensor in raw_state.items()}
    digest = tensor_sha256(state)
    if digest != settings["tensor_sha256"]:
        raise UpstreamError(
            f"{path}: tensor SHA-256 {digest}, expected {settings['tensor_sha256']}"
        )
    return RvtCheckpoint(
        state=state,
        size_bytes=size,
        sha256=sha256,
        tensor_sha256=digest,
        global_step=content.get("global_step"),
    )


# --------------------------------------------------------------------------- fusion stem


class ChannelsFirstLayerNorm(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW tensor (per pixel)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    """``y + gamma * Linear_2(GELU(Linear_1(LN(DWConv7x7(y)))))`` on NCHW tensors."""

    def __init__(
        self, dim: int, kernel_size: int, mlp_ratio: int, layer_scale_init: float, norm_eps: float
    ) -> None:
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=norm_eps)
        self.pwconv1 = nn.Linear(dim, mlp_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(mlp_ratio * dim, dim)
        self.gamma = nn.Parameter(torch.full((dim,), float(layer_scale_init)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pwconv2(self.act(self.pwconv1(self.norm(y))))
        return x + (self.gamma * y).permute(0, 3, 1, 2)


class EventBranch(nn.Module):
    """``F = Block(LN(Conv7x7_stride4(x)))`` for one input channel group (NCHW out)."""

    def __init__(self, in_channels: int, dim: int, settings: Mapping[str, Any]) -> None:
        super().__init__()
        kernel = int(settings["stem_kernel"])
        self.in_channels = in_channels
        self.stem = nn.Conv2d(
            in_channels,
            dim,
            kernel_size=kernel,
            stride=int(settings["stem_stride"]),
            padding=kernel // 2,
            bias=False,
        )
        self.norm = ChannelsFirstLayerNorm(dim, eps=float(settings["stem_norm_eps"]))
        block = settings["convnext"]
        self.block = ConvNeXtBlock(
            dim,
            kernel_size=int(block["kernel"]),
            mlp_ratio=int(block["mlp_ratio"]),
            layer_scale_init=float(block["layer_scale_init"]),
            norm_eps=float(block["norm_eps"]),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected (N, {self.in_channels}, H, W), got {tuple(x.shape)}")
        return self.block(self.norm(self.stem(x)))


@dataclass
class FusionOutput:
    """Branch features and gate of the last :class:`GatedFusionStem` call (NCHW)."""

    event_frame: torch.Tensor
    voxel_grid: torch.Tensor
    gate: torch.Tensor


class GatedFusionStem(nn.Module):
    """Separate EF/VG branches and the per-pixel, per-channel gate at stride 4.

    Drop-in for RVT's ``stages[0].downsample_cf2cl``: it takes the ``(event_frame,
    voxel_grid)`` pair in NCHW and returns the normalized fused map in NHWC, so the
    unchanged RVT stage feeds it to its first attention block (``skip_first_norm``).
    The gate conv starts at zero, so ``g = 0.5`` everywhere. The LayerNorm is named
    ``norm`` like the RVT stem norm it inherits from the checkpoint.
    """

    def __init__(
        self, event_frame_channels: int, voxel_grid_channels: int, settings: Mapping[str, Any]
    ) -> None:
        super().__init__()
        dim = int(settings["dim"])
        self.event_frame = EventBranch(event_frame_channels, dim, settings)
        self.voxel_grid = EventBranch(voxel_grid_channels, dim, settings)
        self.gate = nn.Conv2d(2 * dim, dim, kernel_size=1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.norm = nn.LayerNorm(dim, eps=float(settings["fusion_norm_eps"]))
        self.last_output: Optional[FusionOutput] = None

    @staticmethod
    def output_is_normed() -> bool:
        return True

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        event_frame, voxel_grid = inputs
        f_ef = self.event_frame(event_frame)
        f_vg = self.voxel_grid(voxel_grid)
        if f_ef.shape != f_vg.shape:
            raise ValueError(f"branch shapes differ: {tuple(f_ef.shape)} vs {tuple(f_vg.shape)}")
        gate = torch.sigmoid(self.gate(torch.cat((f_ef, f_vg), dim=1)))
        fused = gate * f_ef + (1.0 - gate) * f_vg
        self.last_output = FusionOutput(event_frame=f_ef, voxel_grid=f_vg, gate=gate)
        return self.norm(fused.permute(0, 2, 3, 1))


def bin_overlap(source_bins: int, target_bins: int) -> torch.Tensor:
    """``(target_bins, source_bins)`` lengths of ``[t, t+1)`` within target bin ``b``.

    Target bin ``b`` spans ``[S * b / B, S * (b + 1) / B)`` in source-bin units.
    """

    overlap = torch.zeros(target_bins, source_bins, dtype=torch.float64)
    for b in range(target_bins):
        low, high = source_bins * b / target_bins, source_bins * (b + 1) / target_bins
        for t in range(source_bins):
            overlap[b, t] = max(0.0, min(high, t + 1) - max(low, t))
    return overlap


def adapt_rvt_stem(
    weight: torch.Tensor, polarities: int, source_bins: int, voxel_bins: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """EF and VG stem weights from RVT's polarity-major ``(C, P * S, k, k)`` stem conv.

    ``W_EF[:, p] = mean_t W[:, p, t]`` and
    ``W_VG[:, B_count * p + b] = (B / S) * sum_t overlap(t, b) * W[:, p, t]``.
    """

    out_channels, in_channels, *kernel = weight.shape
    if in_channels != polarities * source_bins:
        raise UpstreamError(
            f"RVT stem has {in_channels} channels, expected {polarities * source_bins}"
        )
    per_bin = weight.detach().to(torch.float64).view(out_channels, polarities, source_bins, *kernel)
    event_frame = per_bin.mean(dim=2)
    mixing = bin_overlap(source_bins, voxel_bins) * (voxel_bins / source_bins)
    voxel_grid = torch.einsum("bt,cpt...->cpb...", mixing, per_bin)
    voxel_grid = voxel_grid.reshape(out_channels, polarities * voxel_bins, *kernel)
    return event_frame.to(weight.dtype), voxel_grid.to(weight.dtype)


# --------------------------------------------------------------------------- recurrent state


def _row_mask(mask: torch.Tensor, batch: int, like: torch.Tensor) -> torch.Tensor:
    if mask.dtype != torch.bool or tuple(mask.shape) != (batch,):
        raise ValueError(f"expected a ({batch},) bool mask, got {mask.dtype} {tuple(mask.shape)}")
    return mask.to(like.device).view(batch, *([1] * (like.ndim - 1)))


@dataclass(frozen=True)
class RecurrentState:
    """Carried ``(h, c)`` of the RVT stage LSTMs, one NCHW pair per stage, in fp32.

    Rows are streams (batch entries). Every operation returns a new state, so a state
    that is still part of an autograd graph is never modified in place.
    """

    stages: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a recurrent state needs at least one stage")
        batch = self.stages[0][0].shape[0]
        for index, (hidden, cell) in enumerate(self.stages, start=1):
            if hidden.shape != cell.shape or hidden.ndim != 4 or hidden.shape[0] != batch:
                raise ValueError(
                    f"stage {index}: h {tuple(hidden.shape)} and c {tuple(cell.shape)} "
                    f"are not matching (N={batch}, C, H, W) maps"
                )
            if hidden.dtype != torch.float32 or cell.dtype != torch.float32:
                raise ValueError(f"stage {index}: state must be float32")

    @classmethod
    def zeros(
        cls,
        batch_size: int,
        shapes: Sequence[tuple[int, int, int]],
        device: torch.device | str | None = None,
    ) -> RecurrentState:
        """Zero state for ``batch_size`` streams; ``shapes`` are per-stage ``(C, H, W)``."""

        return cls(
            tuple(
                (
                    torch.zeros(batch_size, *shape, device=device),
                    torch.zeros(batch_size, *shape, device=device),
                )
                for shape in shapes
            )
        )

    @classmethod
    def from_rvt(cls, states: Sequence[tuple[torch.Tensor, torch.Tensor]]) -> RecurrentState:
        """From the backbone's per-stage ``(h_t, c_t)`` list, cast to fp32."""

        return cls(tuple((hidden.float(), cell.float()) for hidden, cell in states))

    def as_rvt(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """The ``prev_states`` list that ``RNNDetector.forward`` takes."""

        return [(hidden, cell) for hidden, cell in self.stages]

    @property
    def batch_size(self) -> int:
        return int(self.stages[0][0].shape[0])

    @property
    def device(self) -> torch.device:
        return self.stages[0][0].device

    def shapes(self) -> tuple[tuple[int, int, int], ...]:
        """Per-stage ``(C, H, W)``."""

        return tuple(tuple(hidden.shape[1:]) for hidden, _ in self.stages)

    def floats_per_stream(self) -> int:
        return sum(2 * hidden[0].numel() for hidden, _ in self.stages)

    def _map(self, function: Callable[[torch.Tensor], torch.Tensor]) -> RecurrentState:
        return RecurrentState(tuple((function(h), function(c)) for h, c in self.stages))

    def detach(self) -> RecurrentState:
        """Cut the graph (TBPTT segment boundary); values are unchanged."""

        return self._map(torch.Tensor.detach)

    def to(self, device: torch.device | str) -> RecurrentState:
        return self._map(lambda tensor: tensor.to(device))

    def reset(self, mask: torch.Tensor) -> RecurrentState:
        """Zero the rows where ``mask`` (``(N,)`` bool) is true."""

        return self._map(
            lambda t: torch.where(_row_mask(mask, self.batch_size, t), torch.zeros_like(t), t)
        )

    def hold(self, previous: RecurrentState, mask: torch.Tensor) -> RecurrentState:
        """Rows where ``mask`` is false keep ``previous`` (masked tail steps)."""

        if previous.shapes() != self.shapes() or previous.batch_size != self.batch_size:
            raise ValueError("held state has a different layout")
        return RecurrentState(
            tuple(
                tuple(
                    torch.where(_row_mask(mask, self.batch_size, new), new, old)
                    for new, old in zip(new_pair, old_pair)
                )
                for new_pair, old_pair in zip(self.stages, previous.stages)
            )
        )

    def rows(self, index: torch.Tensor | Sequence[int]) -> RecurrentState:
        """Select streams by index (for a state bank)."""

        index = torch.as_tensor(index, dtype=torch.long, device=self.device)
        return self._map(lambda tensor: tensor.index_select(0, index))

    @classmethod
    def cat(cls, states: Sequence[RecurrentState]) -> RecurrentState:
        """Concatenate the streams of states with equal per-stage shapes."""

        if not states or any(state.shapes() != states[0].shapes() for state in states):
            raise ValueError("cannot concatenate states with different layouts")
        return cls(
            tuple(
                (
                    torch.cat([state.stages[i][0] for state in states]),
                    torch.cat([state.stages[i][1] for state in states]),
                )
                for i in range(len(states[0].stages))
            )
        )

    def is_zero(self) -> torch.Tensor:
        """``(N,)`` bool: rows whose every ``h`` and ``c`` entry is zero."""

        zero = torch.ones(self.batch_size, dtype=torch.bool, device=self.device)
        for hidden, cell in self.stages:
            zero &= (hidden.flatten(1) == 0).all(dim=1) & (cell.flatten(1) == 0).all(dim=1)
        return zero

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Detached tensors keyed ``stage<k>.h`` and ``stage<k>.c``."""

        state = {}
        for index, (hidden, cell) in enumerate(self.stages, start=1):
            state[f"stage{index}.h"] = hidden.detach()
            state[f"stage{index}.c"] = cell.detach()
        return state

    @classmethod
    def from_state_dict(cls, state: Mapping[str, torch.Tensor]) -> RecurrentState:
        count = len(state) // 2
        expected = {f"stage{i}.{part}" for i in range(1, count + 1) for part in ("h", "c")}
        if set(state) != expected:
            raise ValueError(f"recurrent state keys {sorted(state)} are not {sorted(expected)}")
        return cls(tuple((state[f"stage{i}.h"], state[f"stage{i}.c"]) for i in range(1, count + 1)))


# --------------------------------------------------------------------------- embedding head


class EmbeddingHead(nn.Module):
    """Dense per-anchor descriptors on the YOLOX grid, one branch per PAFPN level.

    ``base_conv`` is RVT's ``BaseConv`` (conv without bias, BatchNorm, activation). The
    output is ``(N, A, dim)`` in fp32 with unit L2 norm, flattened level by level in
    row-major order like the YOLOX head outputs, so anchor ``a`` of both is the same cell.
    """

    def __init__(
        self,
        in_channels: Sequence[int],
        settings: Mapping[str, Any],
        base_conv: Callable[..., nn.Module],
    ) -> None:
        super().__init__()
        hidden = int(settings["hidden_dim"])
        act = settings["act"]
        self.dim = int(settings["dim"])
        self.eps = float(settings["normalize_eps"])
        self.levels = nn.ModuleList(
            nn.Sequential(
                base_conv(int(channels), hidden, 1, 1, act=act),
                base_conv(hidden, hidden, 3, 1, act=act),
                base_conv(hidden, hidden, 3, 1, act=act),
                nn.Conv2d(hidden, self.dim, kernel_size=1),
            )
            for channels in in_channels
        )

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.levels):
            raise ValueError(f"expected {len(self.levels)} feature levels, got {len(features)}")
        outputs = [
            branch(feature).float().flatten(start_dim=2)
            for branch, feature in zip(self.levels, features)
        ]
        return F.normalize(torch.cat(outputs, dim=2).permute(0, 2, 1), dim=-1, eps=self.eps)
