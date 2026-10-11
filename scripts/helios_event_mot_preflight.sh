#!/usr/bin/env bash
# Bounded environment preflight for the large event-based MOT pipeline.
#
# Records host, software, Slurm allocation, device memory, and storage, then
# runs short checks of the pinned dependencies, the RVT checkout, the RVT-B
# checkpoint hash, and native GPU operations (bf16 forward/backward, batched
# NMS). Nothing is downloaded or trained.
#
# Usage (inside an allocated job; the virtual environment must be active):
#   scripts/helios_event_mot_preflight.sh [--cpu-only] [--output-dir DIR]
#
# Runtime inputs (environment variables):
#   RVT_ROOT          RVT git checkout (required for the RVT check)
#   RVT_CHECKPOINT    path to rvt-b.ckpt (required for the checkpoint check)
#   STORAGE_PATHS     colon-separated directories whose free space is recorded
#   PREFLIGHT_TIMEOUT seconds allowed for the Python checks (default 900)
#
# Without --cpu-only the script refuses to run outside a Slurm job that has a
# GPU allocated, so GPU work never runs on a login node. --cpu-only skips the
# GPU checks and is meant for local syntax/import checks and the x86 CPU
# environment. Exit status is 0 only when every required check passed.

set -uo pipefail

RVT_COMMIT="b80f5683a6e2d5de65d4bde8105d796ccb50dbb1"
RVT_CKPT_BYTES="297169732"
RVT_CKPT_SHA256="65d83e68236e99290757c1305e06da6c0cc650c435eed03cea70e9ffed8a1572"

cpu_only=0
output_dir=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --cpu-only) cpu_only=1; shift ;;
        --output-dir) output_dir="$2"; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [[ "$cpu_only" -eq 0 ]]; then
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "refusing GPU checks outside a Slurm job; use --cpu-only for local checks" >&2
        exit 2
    fi
    if [[ -z "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo "refusing GPU checks: job ${SLURM_JOB_ID} has no GPU allocated" >&2
        exit 2
    fi
fi

host="$(hostname -s)"
stamp="$(date +%Y%m%d_%H%M%S)"
output_dir="${output_dir:-runs/large_event_mot/preflight/${host}_${stamp}}"
umask 077
mkdir -p "$output_dir" || exit 2
log="$output_dir/preflight.log"
summary="$output_dir/checks.tsv"
printf 'check\tstatus\tdetail\n' > "$summary"
failures=0

record() {
    printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$summary"
    printf '[%s] %s: %s\n' "$2" "$1" "$3" | tee -a "$log"
    if [[ "$2" == "FAIL" ]]; then failures=$((failures + 1)); fi
}

section() {
    printf '\n## %s\n' "$1" >> "$log"
    shift
    "$@" >> "$log" 2>&1
}

echo "preflight output: $output_dir" | tee "$log"
section "date" date --iso-8601=seconds
section "host" uname -a
section "os" cat /etc/os-release
section "cpu" lscpu
section "memory" free -g
section "slurm" bash -c 'env | grep -E "^SLURM_(JOB_ID|JOB_ACCOUNT|JOB_PARTITION|JOB_NODELIST|CPUS_ON_NODE|CPUS_PER_TASK|MEM_PER_NODE|MEM_PER_CPU|JOB_GPUS|GPUS_ON_NODE|TIMELIMIT)=" | sort'
section "cuda visible devices" bash -c 'echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"'
if command -v module >/dev/null 2>&1; then
    section "modules" module -t list
fi
if command -v nvidia-smi >/dev/null 2>&1; then
    section "nvidia-smi" nvidia-smi \
        --query-gpu=name,memory.total,driver_version,compute_cap --format=csv
fi
section "python" bash -c 'command -v python; python --version'
section "pip freeze" python -m pip freeze
if command -v hpc-fs >/dev/null 2>&1; then
    section "hpc-fs" hpc-fs
fi

record "architecture" "INFO" "$(uname -m)"
if [[ -n "${STORAGE_PATHS:-}" ]]; then
    IFS=':' read -r -a storage_paths <<< "$STORAGE_PATHS"
    for path in "${storage_paths[@]}"; do
        if [[ -d "$path" ]]; then
            free="$(df -Ph "$path" | awk 'NR == 2 {print $4 " free of " $2}')"
            inodes="$(df -Pi "$path" | awk 'NR == 2 {print $4 " inodes free"}')"
            record "storage $path" "INFO" "$free, $inodes"
        else
            record "storage $path" "FAIL" "not a directory"
        fi
    done
else
    record "storage" "SKIP" "STORAGE_PATHS not set"
fi

if [[ -z "${RVT_ROOT:-}" ]]; then
    record "rvt commit" "SKIP" "RVT_ROOT not set"
elif ! head="$(git -C "$RVT_ROOT" rev-parse HEAD 2>/dev/null)"; then
    record "rvt commit" "FAIL" "$RVT_ROOT is not a git checkout"
elif [[ "$head" != "$RVT_COMMIT" ]]; then
    record "rvt commit" "FAIL" "HEAD $head, expected $RVT_COMMIT"
elif [[ -n "$(git -C "$RVT_ROOT" status --porcelain --untracked-files=no)" ]]; then
    record "rvt commit" "FAIL" "$head has modified tracked files"
else
    record "rvt commit" "PASS" "$head"
fi

if [[ -z "${RVT_CHECKPOINT:-}" ]]; then
    record "rvt checkpoint" "SKIP" "RVT_CHECKPOINT not set"
elif [[ ! -f "$RVT_CHECKPOINT" ]]; then
    record "rvt checkpoint" "FAIL" "$RVT_CHECKPOINT not found"
else
    bytes="$(stat -c %s "$RVT_CHECKPOINT")"
    sha="$(sha256sum "$RVT_CHECKPOINT" | awk '{print $1}')"
    if [[ "$bytes" == "$RVT_CKPT_BYTES" && "$sha" == "$RVT_CKPT_SHA256" ]]; then
        record "rvt checkpoint" "PASS" "$bytes B, sha256 $sha"
    else
        record "rvt checkpoint" "FAIL" "$bytes B, sha256 $sha"
    fi
fi

export PREFLIGHT_CPU_ONLY="$cpu_only"
export PREFLIGHT_SCRATCH="$output_dir"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python_out="$output_dir/python_checks.tsv"
timeout "${PREFLIGHT_TIMEOUT:-900}" python - > "$python_out" 2>> "$log" <<'PY'
import contextlib
import importlib
import importlib.metadata as metadata
import os
import platform
import sys
import time

cpu_only = os.environ["PREFLIGHT_CPU_ONLY"] == "1"
arch = platform.machine()


report = sys.stdout


def emit(name, status, detail):
    print(f"{name}\t{status}\t{detail}", file=report, flush=True)


def check(name, fn, required=True):
    try:
        with contextlib.redirect_stdout(sys.stderr):  # keep library prints out of the TSV
            detail = fn()
        emit(name, "PASS", detail)
    except Exception as exc:  # every failure is recorded, never raised
        emit(name, "FAIL" if required else "WARN", f"{type(exc).__name__}: {exc}")


emit("python", "INFO", platform.python_version())

# Distribution metadata is recorded; for example boxmot reports an older
# __version__ than its installed distribution.
gh200 = arch == "aarch64"
packages = ["torch", "torchvision", "numpy", "h5py", "hdf5plugin"]
packages += ["omegaconf", "einops", "pycocotools"] if gh200 else ["boxmot", "trackeval"]
for package in packages:
    check(f"import {package}",
          lambda p=package: importlib.import_module(p) and metadata.version(p))


def hdf5_blosc():
    import h5py
    import hdf5plugin
    import numpy as np

    path = os.path.join(os.environ["PREFLIGHT_SCRATCH"], "blosc_check.h5")
    data = np.arange(100_000, dtype=np.int64)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("t", data=data, **hdf5plugin.Blosc(cname="zstd"))
    with h5py.File(path, "r") as handle:
        ok = bool((handle["t"][:] == data).all())
    os.remove(path)
    if not ok:
        raise RuntimeError("blosc round trip changed data")
    return "blosc zstd round trip"


def coco_eval():
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt = COCO()
    gt.dataset = {
        "info": {},
        "images": [{"id": 1, "width": 640, "height": 480}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40],
                         "area": 2000, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "car"}],
    }
    gt.createIndex()
    dt = gt.loadRes([{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40],
                      "score": 0.9}])
    evaluator = COCOeval(gt, dt, "bbox")
    evaluator.evaluate()
    evaluator.accumulate()
    ap = float(evaluator.eval["precision"][0, :, 0, 0, -1].mean())
    if abs(ap - 1.0) > 1e-6:
        raise RuntimeError(f"AP@0.5 {ap}, expected 1.0")
    return "AP@0.5 1.0 on a perfect detection"


check("hdf5plugin blosc", hdf5_blosc)
if gh200:
    check("pycocotools COCOeval", coco_eval)


def gpu_checks():
    import torch
    import torch.nn.functional as F
    from torchvision.ops import batched_nms

    if not torch.cuda.is_available():
        emit("cuda available", "FAIL", "torch.cuda.is_available() is False")
        return
    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    emit("cuda device", "PASS",
         f"{props.name}, {props.total_memory / 2**30:.1f} GiB, capability "
         f"{props.major}.{props.minor}, torch CUDA {torch.version.cuda}, "
         f"cuDNN {torch.backends.cudnn.version()}, {torch.cuda.device_count()} visible")
    check("bf16 supported", lambda: str(torch.cuda.is_bf16_supported()))

    def forward_backward():
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()
        stem = torch.nn.Conv2d(8, 64, 7, stride=4, padding=3).to(device)
        norm = torch.nn.LayerNorm(64).to(device)
        gate = torch.nn.Conv2d(64, 64, 1).to(device)
        x = torch.randn(2, 8, 384, 640, device=device)
        h = torch.zeros(2, 64, 96, 160, device=device)
        start = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(2):  # two steps with carried recurrent state
                feat = norm(stem(x).permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
                h = torch.sigmoid(gate(feat)) * h + feat
            tokens = h.flatten(2).transpose(1, 2).reshape(2 * 240, 64, 64)
            q = tokens.view(2 * 240, 64, 2, 32).transpose(1, 2)
            attn = F.scaled_dot_product_attention(q, q, q)
            loss = attn.float().pow(2).mean() + h.float().mean()
        loss.backward()
        torch.cuda.synchronize()
        grads = [p.grad for p in stem.parameters()]
        if not torch.isfinite(loss) or any(not torch.isfinite(g).all() for g in grads):
            raise RuntimeError("non-finite loss or gradient")
        peak = torch.cuda.max_memory_allocated() / 2**20
        return (f"loss {loss.item():.4f}, {time.perf_counter() - start:.2f} s, "
                f"peak {peak:.0f} MiB")

    def nms():
        torch.manual_seed(0)
        xy = torch.rand(5000, 2, device=device) * 600
        wh = torch.rand(5000, 2, device=device) * 60 + 4
        boxes = torch.cat([xy, xy + wh], dim=1)
        scores = torch.rand(5000, device=device)
        labels = torch.randint(0, 7, (5000,), device=device)
        keep_gpu = batched_nms(boxes, scores, labels, 0.5)
        keep_cpu = batched_nms(boxes.cpu(), scores.cpu(), labels.cpu(), 0.5)
        if set(keep_gpu.tolist()) != set(keep_cpu.tolist()):
            raise RuntimeError("GPU and CPU batched_nms disagree")
        return f"{keep_gpu.numel()} of 5000 kept, matches CPU"

    check("bf16 forward/backward", forward_backward)
    check("torchvision batched_nms", nms)


if cpu_only:
    emit("gpu checks", "SKIP", "--cpu-only")
else:
    check("gpu checks", lambda: gpu_checks() or "completed")
PY
python_status=$?
cat "$python_out" >> "$summary"
while IFS=$'\t' read -r name status detail; do
    printf '[%s] %s: %s\n' "$status" "$name" "$detail" | tee -a "$log"
    if [[ "$status" == "FAIL" ]]; then failures=$((failures + 1)); fi
done < "$python_out"
if [[ "$python_status" -ne 0 ]]; then
    record "python checks" "FAIL" "exit status $python_status (124 means timeout)"
fi

if [[ "$failures" -gt 0 ]]; then
    echo "PREFLIGHT FAILED: $failures failed checks, see $summary" | tee -a "$log"
    exit 1
fi
echo "PREFLIGHT PASSED: see $summary" | tee -a "$log"
