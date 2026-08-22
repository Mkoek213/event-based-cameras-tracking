#!/usr/bin/env bash
set -euo pipefail

# Keep decimal parsing stable on systems whose locale uses a comma separator.
export LC_ALL=C.UTF-8

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="$REPO_ROOT/.venv/bin/python"
DATA_ROOT="$REPO_ROOT/data/datasets/dsec_mot"
RUNS_ROOT="$REPO_ROOT/runs/event_reid_embedding_frozen"
RESULTS_ROOT="$REPO_ROOT/results/dsec_mot_event_reid_embedding_frozen"
LOG_DIR="$RUNS_ROOT/logs"
LOG_PATH="$LOG_DIR/full_frozen_r0_benchmark.log"

DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-2}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"
MAX_TRAIN_CLIPS="${MAX_TRAIN_CLIPS:-0}"
MAX_VAL_CLIPS="${MAX_VAL_CLIPS:-0}"
MAX_EVAL_FRAMES="${MAX_EVAL_FRAMES:-0}"
DRY_RUN="${DRY_RUN:-0}"

ALL_R0="$REPO_ROOT/runs/simple_detector_sweep/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"
CAR_R0="$REPO_ROOT/runs/simple_detector_car_only/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt"

ALL_SCORE_THRESHOLDS=(0.10 0.25 0.50 0.70 0.90 0.95)
CAR_SCORE_THRESHOLDS=(0.90 0.95 0.97 0.99)
APPEARANCE_THRESHOLDS=(0.10 0.20 0.25 0.30 0.40 0.50)
PROXIMITY_THRESHOLDS=(0.30 0.50 0.70)

mkdir -p "$RUNS_ROOT" "$RESULTS_ROOT" "$LOG_DIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "Nie znaleziono interpretera: $PYTHON" >&2
  exit 1
fi
if ! command -v jq >/dev/null 2>&1; then
  echo "Wymagany jest program jq." >&2
  exit 1
fi
for checkpoint in "$ALL_R0" "$CAR_R0"; do
  if [[ ! -f "$checkpoint" ]]; then
    echo "Nie znaleziono checkpointu R0: $checkpoint" >&2
    exit 1
  fi
done

exec > >(tee -a "$LOG_PATH") 2>&1

run_command() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '$'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

threshold_label() {
  printf '%.2f' "$1" | tr -d '.'
}

label_to_threshold() {
  local value=$((10#$1))
  printf '0.%02d' "$value"
}

run_name_for() {
  local variant="$1"
  if [[ "$variant" == "F1" ]]; then
    printf 'event_frame_voxel_grid_bins3_w32_gated_two_branch_f1_frozen_r0'
  else
    printf 'event_frame_voxel_grid_bins3_w32_gated_two_branch_f2_frozen_r0_recurrent'
  fi
}

run_dir_for() {
  local scope="$1"
  local variant="$2"
  printf '%s/%s/%s' "$RUNS_ROOT" "$scope" "$(run_name_for "$variant")"
}

train_variant() {
  local scope="$1"
  local variant="$2"
  local initial_checkpoint="$3"
  local output_dir="$RUNS_ROOT/$scope"
  local run_name
  run_name="$(run_name_for "$variant")"

  local recurrent_flag=--no-recurrent-embedding
  if [[ "$variant" == "F2" ]]; then
    recurrent_flag=--recurrent-embedding
  fi

  local command=(
    "$PYTHON" -m src.training.recurrent_embedding_detector
    --root "$DATA_ROOT"
    --representation event_frame_voxel_grid
    --fusion-mode gated_two_branch
    --num-bins 3
    --time-window-us 50000
    --model-width 32
    --embedding-hidden-dim 128
    --embedding-dim 256
    --roi-size 7
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
    --initial-detector-checkpoint "$initial_checkpoint"
    --freeze-detector
    --save-every-epoch
    --resume
    --output-dir "$output_dir"
    --run-name "$run_name"
    "$recurrent_flag"
  )

  if [[ "$MAX_TRAIN_CLIPS" != "0" ]]; then
    command+=(--max-train-clips "$MAX_TRAIN_CLIPS")
  fi
  if [[ "$MAX_VAL_CLIPS" != "0" ]]; then
    command+=(--max-val-clips "$MAX_VAL_CLIPS")
  fi
  if [[ "$scope" == "car_only" ]]; then
    command+=(--class-ids 0 --num-classes 1)
  fi

  echo
  echo "Trening $scope $variant z zamrożonego R0"
  run_command "${command[@]}"
}

run_evaluation() {
  local scope="$1"
  local variant="$2"
  local tag="$3"
  local mode="$4"
  local split="$5"
  local checkpoint="$6"
  local score="$7"
  local appearance="$8"
  local proximity="$9"

  local score_label
  local appearance_label
  local proximity_label
  score_label="$(threshold_label "$score")"
  appearance_label="$(threshold_label "$appearance")"
  proximity_label="$(threshold_label "$proximity")"
  local run_name="${scope}_${variant}_${tag}_${mode}_${split}_score${score_label}_appearance${appearance_label}_proximity${proximity_label}"
  local summary_path="$RESULTS_ROOT/$run_name/metrics_summary.json"

  if [[ "$DRY_RUN" != "1" && -f "$summary_path" ]]; then
    echo "Pomijam gotową ewaluację: $run_name"
    return
  fi

  local dataset_split=test
  local sequences=(interlaken_00_d zurich_city_00_b)
  if [[ "$split" == "val" ]]; then
    dataset_split=train
    sequences=(zurich_city_01_d)
  fi

  local command=(
    "$PYTHON" -m src.evaluation.simple_detector_trackeval_cli
    --checkpoint "$checkpoint"
    --root "$DATA_ROOT"
    --split "$dataset_split"
    --sequences "${sequences[@]}"
    --device "$DEVICE"
    --score-threshold "$score"
    --max-detections 100
    --tracker-backend boxmot_botsort
    --track-per-class
    --track-disable-cmc
    --input-normalisation component
    --track-appearance-thresh "$appearance"
    --track-proximity-thresh "$proximity"
    --tracker-name "frozen_r0_${mode}"
    --output-root "$RESULTS_ROOT"
    --run-name "$run_name"
  )
  if [[ "$mode" == "reid" ]]; then
    command+=(--track-with-reid)
  fi
  if [[ "$scope" == "car_only" ]]; then
    command+=(--classes-to-eval car)
  fi
  if [[ "$MAX_EVAL_FRAMES" != "0" ]]; then
    command+=(--max-frames "$MAX_EVAL_FRAMES")
  fi

  echo
  echo "Ewaluacja: $run_name"
  run_command "${command[@]}"
}

best_summary_from_pattern() {
  local pattern="$1"
  local summaries=()
  local directory
  shopt -s nullglob
  for directory in $pattern; do
    [[ -f "$directory/metrics_summary.json" ]] || continue
    summaries+=("$(jq -r --arg directory "$directory" '[.aggregate.HOTA, .aggregate.AssA, .aggregate.IDF1, $directory] | @tsv' "$directory/metrics_summary.json")")
  done
  if [[ "${#summaries[@]}" -eq 0 ]]; then
    echo "Brak kompletnych wyników walidacyjnych dla wzorca: $pattern" >&2
    exit 1
  fi
  printf '%s\n' "${summaries[@]}" \
    | sort -t $'\t' -k1,1nr -k2,2nr -k3,3nr \
    | head -1
}

select_detector_score() {
  local scope="$1"
  local checkpoint="$2"
  local scores=("${ALL_SCORE_THRESHOLDS[@]}")
  if [[ "$scope" == "car_only" ]]; then
    scores=("${CAR_SCORE_THRESHOLDS[@]}")
  fi

  local score
  for score in "${scores[@]}"; do
    run_evaluation "$scope" F1 detector_score motion val "$checkpoint" "$score" 0.25 0.50
  done

  local best_line
  best_line="$(best_summary_from_pattern "$RESULTS_ROOT/${scope}_F1_detector_score_motion_val_score*_appearance025_proximity050")"
  local best_hota best_assa best_idf1 best_directory
  IFS=$'\t' read -r best_hota best_assa best_idf1 best_directory <<< "$best_line"
  local base
  base="$(basename "$best_directory")"
  if [[ ! "$base" =~ _score([0-9]{3})_ ]]; then
    echo "Nie można odczytać progu z $base" >&2
    exit 1
  fi
  SELECTED_SCORE="$(label_to_threshold "${BASH_REMATCH[1]}")"
}

select_hota_checkpoint() {
  local scope="$1"
  local variant="$2"
  local score="$3"
  local run_dir
  run_dir="$(run_dir_for "$scope" "$variant")"

  local epoch_checkpoint
  shopt -s nullglob
  local epoch_checkpoints=("$run_dir"/epoch_*.pt)
  if [[ "${#epoch_checkpoints[@]}" -eq 0 ]]; then
    echo "Brak checkpointów epokowych w $run_dir" >&2
    exit 1
  fi

  for epoch_checkpoint in "${epoch_checkpoints[@]}"; do
    local epoch_name
    epoch_name="$(basename "$epoch_checkpoint" .pt)"
    run_evaluation "$scope" "$variant" "$epoch_name" reid val "$epoch_checkpoint" "$score" 0.25 0.50
  done

  local score_label
  score_label="$(threshold_label "$score")"
  local best_line
  best_line="$(best_summary_from_pattern "$RESULTS_ROOT/${scope}_${variant}_epoch_???_reid_val_score${score_label}_appearance025_proximity050")"
  local best_hota best_assa best_idf1 best_directory
  IFS=$'\t' read -r best_hota best_assa best_idf1 best_directory <<< "$best_line"
  local base
  base="$(basename "$best_directory")"
  if [[ ! "$base" =~ _epoch_([0-9]{3})_ ]]; then
    echo "Nie można odczytać epoki z $base" >&2
    exit 1
  fi
  local epoch="${BASH_REMATCH[1]}"
  local source_checkpoint="$run_dir/epoch_${epoch}.pt"
  local selected_checkpoint="$run_dir/best_hota.pt"
  "$PYTHON" -c 'import sys, torch; source, target, epoch, hota, assa, idf1 = sys.argv[1:]; checkpoint = torch.load(source, map_location="cpu"); checkpoint["selected_epoch"] = int(epoch); checkpoint["selection_metric"] = "validation_hota"; checkpoint["best_hota_selection"] = {"HOTA": float(hota), "AssA": float(assa), "IDF1": float(idf1)}; torch.save(checkpoint, target)' \
    "$source_checkpoint" "$selected_checkpoint" "$((10#$epoch))" "$best_hota" "$best_assa" "$best_idf1"
  jq -n \
    --argjson epoch "$((10#$epoch))" \
    --argjson HOTA "$best_hota" \
    --argjson AssA "$best_assa" \
    --argjson IDF1 "$best_idf1" \
    --arg score_threshold "$score" \
    --arg source_checkpoint "$source_checkpoint" \
    '{selection_policy: ["HOTA", "AssA", "IDF1"], epoch: $epoch, validation: {HOTA: $HOTA, AssA: $AssA, IDF1: $IDF1}, score_threshold: ($score_threshold | tonumber), source_checkpoint: $source_checkpoint}' \
    > "$run_dir/best_hota_selection.json"

  SELECTED_CHECKPOINT="$selected_checkpoint"
  SELECTED_EPOCH="$((10#$epoch))"
}

select_association_thresholds() {
  local scope="$1"
  local variant="$2"
  local checkpoint="$3"
  local score="$4"
  local tag="${5:-best_hota}"

  local appearance proximity
  for appearance in "${APPEARANCE_THRESHOLDS[@]}"; do
    for proximity in "${PROXIMITY_THRESHOLDS[@]}"; do
      run_evaluation "$scope" "$variant" "$tag" reid val "$checkpoint" "$score" "$appearance" "$proximity"
    done
  done

  local score_label
  score_label="$(threshold_label "$score")"
  local best_line
  best_line="$(best_summary_from_pattern "$RESULTS_ROOT/${scope}_${variant}_${tag}_reid_val_score${score_label}_appearance*_proximity*")"
  local best_hota best_assa best_idf1 best_directory
  IFS=$'\t' read -r best_hota best_assa best_idf1 best_directory <<< "$best_line"
  local base
  base="$(basename "$best_directory")"
  if [[ ! "$base" =~ appearance([0-9]{3})_proximity([0-9]{3})$ ]]; then
    echo "Nie można odczytać progów z $base" >&2
    exit 1
  fi
  SELECTED_APPEARANCE="$(label_to_threshold "${BASH_REMATCH[1]}")"
  SELECTED_PROXIMITY="$(label_to_threshold "${BASH_REMATCH[2]}")"
  SELECTED_VAL_HOTA="$best_hota"
  SELECTED_VAL_ASSA="$best_assa"
  SELECTED_VAL_IDF1="$best_idf1"
}

for scope in all_classes car_only; do
  if [[ "$scope" == "all_classes" ]]; then
    r0_checkpoint="$ALL_R0"
  else
    r0_checkpoint="$CAR_R0"
  fi
  train_variant "$scope" F1 "$r0_checkpoint"
  train_variant "$scope" F2 "$r0_checkpoint"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo
  echo "Dry-run: po treningu skrypt wybiera score threshold przez HOTA motion na walidacji, następnie best_hota.pt przez HOTA ReID każdej epoki, a na końcu appearance/proximity również przez HOTA walidacyjne. Ostatnia epoka jest dodatkowo oceniana diagnostycznie."
  exit 0
fi

for scope in all_classes car_only; do
  reference_checkpoint="$(run_dir_for "$scope" F1)/epoch_001.pt"
  select_detector_score "$scope" "$reference_checkpoint"
  selected_score="$SELECTED_SCORE"
  echo
  echo "Wybrany score threshold dla $scope: $selected_score"

  for variant in F1 F2; do
    select_hota_checkpoint "$scope" "$variant" "$selected_score"
    selected_checkpoint="$SELECTED_CHECKPOINT"
    selected_epoch="$SELECTED_EPOCH"
    echo "Wybrany checkpoint $scope $variant: $selected_checkpoint"

    select_association_thresholds "$scope" "$variant" "$selected_checkpoint" "$selected_score" best_hota
    appearance="$SELECTED_APPEARANCE"
    proximity="$SELECTED_PROXIMITY"
    echo "Wybrane progi $scope $variant: appearance=$appearance proximity=$proximity HOTA=$SELECTED_VAL_HOTA AssA=$SELECTED_VAL_ASSA IDF1=$SELECTED_VAL_IDF1"

    run_evaluation "$scope" "$variant" best_hota motion test "$selected_checkpoint" "$selected_score" 0.25 0.50
    run_evaluation "$scope" "$variant" best_hota reid test "$selected_checkpoint" "$selected_score" "$appearance" "$proximity"

    last_epoch_number="$((10#$EPOCHS))"
    if [[ "$selected_epoch" -eq "$last_epoch_number" ]]; then
      echo "Ostatnia epoka jest już checkpointem best_hota dla $scope $variant."
    else
      printf -v last_epoch_label '%03d' "$last_epoch_number"
      last_checkpoint="$(run_dir_for "$scope" "$variant")/epoch_${last_epoch_label}.pt"
      if [[ ! -f "$last_checkpoint" ]]; then
        echo "Brak checkpointu ostatniej epoki: $last_checkpoint" >&2
        exit 1
      fi
      select_association_thresholds "$scope" "$variant" "$last_checkpoint" "$selected_score" last_epoch
      last_appearance="$SELECTED_APPEARANCE"
      last_proximity="$SELECTED_PROXIMITY"
      echo "Diagnostyka ostatniej epoki $scope $variant: appearance=$last_appearance proximity=$last_proximity HOTA=$SELECTED_VAL_HOTA AssA=$SELECTED_VAL_ASSA IDF1=$SELECTED_VAL_IDF1"
      run_evaluation "$scope" "$variant" last_epoch reid test "$last_checkpoint" "$selected_score" "$last_appearance" "$last_proximity"
    fi
  done
done

echo
printf 'RUN\tHOTA\tAssA\tDetA\tMOTA\tIDF1\tIDS\tFP\tFN\n'
shopt -s nullglob
for summary_file in \
  "$RESULTS_ROOT"/*_best_hota_*_test_score*/metrics_summary.json \
  "$RESULTS_ROOT"/*_last_epoch_reid_test_score*/metrics_summary.json; do
  run_name="$(basename "$(dirname "$summary_file")")"
  jq -r --arg run "$run_name" '[
    $run,
    .aggregate.HOTA,
    .aggregate.AssA,
    .aggregate.DetA,
    .aggregate.MOTA,
    .aggregate.IDF1,
    .aggregate.IDS,
    .aggregate.FP,
    .aggregate.FN
  ] | @tsv' "$summary_file"
done

echo
echo "Gotowe. Log: $LOG_PATH"
echo "Wyniki: $RESULTS_ROOT"
