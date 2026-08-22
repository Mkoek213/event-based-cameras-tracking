#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
DATA_ROOT="$REPO_ROOT/data/datasets/dsec_mot"
RESULTS_ROOT="$REPO_ROOT/results/dsec_mot_event_reid_embedding"
LOG_DIR="$REPO_ROOT/runs/event_reid_embedding/logs"
LOG_PATH="$LOG_DIR/fixed_score090_test.log"
DEVICE="${DEVICE:-cuda}"

mkdir -p "$RESULTS_ROOT" "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Nie znaleziono interpretera: $PYTHON" >&2
  exit 1
fi

if ! command -v jq >/dev/null 2>&1; then
  echo "Do wypisania tabeli wyników wymagany jest program jq." >&2
  exit 1
fi

exec > >(tee "$LOG_PATH") 2>&1

COMMON_ARGS=(
  --root "$DATA_ROOT"
  --split test
  --sequences interlaken_00_d zurich_city_00_b
  --device "$DEVICE"
  --score-threshold 0.90
  --max-detections 100
  --tracker-backend boxmot_botsort
  --input-normalisation component
  --output-root "$RESULTS_ROOT"
)

run_evaluation() {
  local checkpoint="$1"
  local run_name="$2"
  local tracker_name="$3"
  shift 3

  if [[ ! -f "$checkpoint" ]]; then
    echo "Nie znaleziono checkpointu: $checkpoint" >&2
    exit 1
  fi

  echo
  echo "Uruchamiam: $run_name"
  "$PYTHON" -m src.evaluation.simple_detector_trackeval_cli \
    --checkpoint "$checkpoint" \
    "${COMMON_ARGS[@]}" \
    --run-name "$run_name" \
    --tracker-name "$tracker_name" \
    "$@"
}

ALL_R1="$REPO_ROOT/runs/event_reid_embedding/all_classes/event_frame_voxel_grid_bins3_w32_gated_two_branch_r1_non_recurrent/best.pt"
ALL_R2="$REPO_ROOT/runs/event_reid_embedding/all_classes/event_frame_voxel_grid_bins3_w32_gated_two_branch_r2_recurrent/best.pt"
CAR_R1="$REPO_ROOT/runs/event_reid_embedding/car_only/event_frame_voxel_grid_bins3_w32_gated_two_branch_r1_non_recurrent/best.pt"
CAR_R2="$REPO_ROOT/runs/event_reid_embedding/car_only/event_frame_voxel_grid_bins3_w32_gated_two_branch_r2_recurrent/best.pt"

run_evaluation \
  "$ALL_R1" \
  all_classes_R1_motion_test_fixed090 \
  event_reid_motion_fixed090

run_evaluation \
  "$ALL_R1" \
  all_classes_R1_reid_test_fixed090 \
  event_reid_reid_fixed090 \
  --track-with-reid \
  --track-appearance-thresh 0.50 \
  --track-proximity-thresh 0.70

run_evaluation \
  "$ALL_R2" \
  all_classes_R2_motion_test_fixed090 \
  event_reid_motion_fixed090

run_evaluation \
  "$ALL_R2" \
  all_classes_R2_reid_test_fixed090 \
  event_reid_reid_fixed090 \
  --track-with-reid \
  --track-appearance-thresh 0.25 \
  --track-proximity-thresh 0.30

run_evaluation \
  "$CAR_R1" \
  car_only_R1_motion_test_fixed090 \
  event_reid_motion_fixed090 \
  --classes-to-eval car

run_evaluation \
  "$CAR_R1" \
  car_only_R1_reid_test_fixed090 \
  event_reid_reid_fixed090 \
  --track-with-reid \
  --track-appearance-thresh 0.10 \
  --track-proximity-thresh 0.70 \
  --classes-to-eval car

run_evaluation \
  "$CAR_R2" \
  car_only_R2_motion_test_fixed090 \
  event_reid_motion_fixed090 \
  --classes-to-eval car

run_evaluation \
  "$CAR_R2" \
  car_only_R2_reid_test_fixed090 \
  event_reid_reid_fixed090 \
  --track-with-reid \
  --track-appearance-thresh 0.20 \
  --track-proximity-thresh 0.70 \
  --classes-to-eval car

echo
printf 'RUN\tHOTA\tAssA\tDetA\tMOTA\tIDF1\tIDS\tFP\tFN\n'

shopt -s nullglob
summary_files=("$RESULTS_ROOT"/*fixed090/metrics_summary.json)
for summary_file in "${summary_files[@]}"; do
  run_name="$(basename "$(dirname "$summary_file")")"
  jq -r --arg run "$run_name" \
    '[
      $run,
      .aggregate.HOTA,
      .aggregate.AssA,
      .aggregate.DetA,
      .aggregate.MOTA,
      .aggregate.IDF1,
      .aggregate.IDS,
      .aggregate.FP,
      .aggregate.FN
    ] | @tsv' \
    "$summary_file"
done

echo
echo "Gotowe. Log: $LOG_PATH"
