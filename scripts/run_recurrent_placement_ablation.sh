#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

exec "$REPO_ROOT/.venv/bin/python" \
  -m src.experiments.recurrent_placement_ablation \
  "$@"
