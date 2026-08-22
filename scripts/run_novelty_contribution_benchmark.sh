#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
MAX_PARALLEL="${MAX_PARALLEL:-4}"
DEVICE="${DEVICE:-cuda}"

exec "${PYTHON_BIN}" -m src.experiments.novelty_contribution_benchmark \
  --root data/datasets/dsec_mot \
  --suite all \
  --epochs 30 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --memory-effective-frames 64 \
  --num-workers 0 \
  --device "${DEVICE}" \
  --max-parallel "${MAX_PARALLEL}" \
  --recurrence-variant gru_direct_neck \
  "$@"
