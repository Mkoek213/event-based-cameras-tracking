#!/usr/bin/env bash
set -euo pipefail

# Train the four embedding variants concurrently, then reuse the canonical runner
# for HOTA checkpoint/threshold selection and test evaluation.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
TRAINER=src.training.recurrent_embedding_detector
MAIN_RUNNER="$REPO_ROOT/scripts/run_frozen_r0_dense_embedding_benchmark.sh"
DATA_ROOT="$REPO_ROOT/data/datasets/dsec_mot"
INITIALIZATION="${INITIALIZATION:-scratch}"
if [[ "$INITIALIZATION" != "scratch" && "$INITIALIZATION" != "warmstart" ]]; then
  echo "INITIALIZATION must be scratch or warmstart." >&2
  exit 1
fi
RUNS_ROOT="$REPO_ROOT/runs/dense_event_reid_embedding_frozen/$INITIALIZATION"
LOG_DIR="$RUNS_ROOT/parallel_training_logs"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
PARALLEL_TRAININGS="${PARALLEL_TRAININGS:-2}"
DRY_RUN="${DRY_RUN:-0}"
ALLOW_CONCURRENT_BENCHMARKS="${ALLOW_CONCURRENT_BENCHMARKS:-0}"

ALL_R0="$REPO_ROOT/runs/simple_detector_sweep/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
CAR_R0="$REPO_ROOT/runs/simple_detector_car_only/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
D1_INITIAL_HEAD="$REPO_ROOT/runs/recurrent_embedding/event_frame_voxel_grid_bins3_w32_gated_two_branch_embed/best.pt"
D2_INITIAL_HEAD="$REPO_ROOT/runs/recurrent_embedding/event_frame_voxel_grid_bins3_w32_gated_two_branch_recurrent_embed/best.pt"

if [[ "$PARALLEL_TRAININGS" != "2" && "$PARALLEL_TRAININGS" != "4" ]]; then
  echo "PARALLEL_TRAININGS must be 2 or 4." >&2
  exit 1
fi
if [[ "$DRY_RUN" != "1" && "$ALLOW_CONCURRENT_BENCHMARKS" != "1" ]] \
  && pgrep -f 'src[.]training[.]recurrent_embedding_detector' >/dev/null; then
  echo "Another embedding trainer is active. Wait for it to finish or set ALLOW_CONCURRENT_BENCHMARKS=1." >&2
  exit 1
fi
required_paths=("$PYTHON" "$MAIN_RUNNER" "$ALL_R0" "$CAR_R0")
if [[ "$INITIALIZATION" == "warmstart" ]]; then
  required_paths+=("$D1_INITIAL_HEAD" "$D2_INITIAL_HEAD")
fi
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done
mkdir -p "$LOG_DIR"

pids=()
names=()

launch_variant() {
  local scope="$1"
  local variant="$2"
  local detector_checkpoint="$3"
  local recurrent_flag=--no-recurrent-embedding
  local run_name="event_frame_voxel_grid_bins3_w32_gated_two_branch_dense_d1_frozen_r0_${INITIALIZATION}"
  local initial_embedding_checkpoint="$D1_INITIAL_HEAD"
  if [[ "$variant" == "D2" ]]; then
    recurrent_flag=--recurrent-embedding
    run_name="event_frame_voxel_grid_bins3_w32_gated_two_branch_dense_d2_frozen_r0_recurrent_${INITIALIZATION}"
    initial_embedding_checkpoint="$D2_INITIAL_HEAD"
  fi

  local command=(
    "$PYTHON" -m "$TRAINER"
    --root "$DATA_ROOT"
    --representation event_frame_voxel_grid
    --fusion-mode gated_two_branch
    --num-bins 3
    --time-window-us 50000
    --model-width 32
    --embedding-hidden-dim 128
    --embedding-dim 128
    --embedding-head-type dense
    --identity-ce-weight 1.0
    --triplet-weight 1.0
    --triplet-margin 0.3
    --clip-length 8
    --clip-stride 8
    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --grad-accum-steps "$GRAD_ACCUM_STEPS"
    --num-workers "$NUM_WORKERS"
    --lr 0.001
    --weight-decay 0.0001
    --grad-clip-norm 5.0
    --seed 0
    --device "$DEVICE"
    --initial-detector-checkpoint "$detector_checkpoint"
    --freeze-detector
    --save-every-epoch
    --resume
    --output-dir "$RUNS_ROOT/$scope"
    --run-name "$run_name"
    "$recurrent_flag"
  )
  if [[ "$scope" == "car_only" ]]; then
    command+=(--class-ids 0 --num-classes 1)
  fi
  if [[ "$INITIALIZATION" == "warmstart" ]]; then
    command+=(--initial-embedding-checkpoint "$initial_embedding_checkpoint")
  fi

  local name="${scope}_${variant}"
  local log_path="$LOG_DIR/${name}.log"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '$'
    printf ' %q' "${command[@]}"
    printf ' > %q 2>&1\n' "$log_path"
    return
  fi

  echo "Starting $name; log: $log_path"
  "${command[@]}" >"$log_path" 2>&1 &
  pids+=("$!")
  names+=("$name")
}

wait_for_variants() {
  local failed=0
  local index
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "Finished ${names[$index]}"
    else
      echo "Failed ${names[$index]}; inspect $LOG_DIR/${names[$index]}.log" >&2
      failed=1
    fi
  done
  pids=()
  names=()
  if [[ "$failed" != "0" ]]; then
    exit 1
  fi
}

launch_variant all_classes D1 "$ALL_R0"
launch_variant all_classes D2 "$ALL_R0"
if [[ "$PARALLEL_TRAININGS" == "2" ]]; then
  wait_for_variants
fi
launch_variant car_only D1 "$CAR_R0"
launch_variant car_only D2 "$CAR_R0"

if [[ "$DRY_RUN" == "1" ]]; then
  echo "Dry-run only. After parallel training, the canonical $INITIALIZATION runner performs validation and test evaluation."
  exit 0
fi

wait_for_variants

echo "All $INITIALIZATION training variants are complete. Starting sequential HOTA selection and evaluation."
INITIALIZATION="$INITIALIZATION" \
DEVICE="$DEVICE" \
EPOCHS="$EPOCHS" \
BATCH_SIZE="$BATCH_SIZE" \
GRAD_ACCUM_STEPS="$GRAD_ACCUM_STEPS" \
NUM_WORKERS="$NUM_WORKERS" \
  "$MAIN_RUNNER"
