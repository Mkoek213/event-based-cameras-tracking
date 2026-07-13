#!/usr/bin/env bash
# Kompletna macierz ewaluacyjna eksperymentu recurrent-embedding (warianty A/B/C/D).
# Idempotentny: pomija ukonczone treningi/ewaluacje, mozna przerywac i wznawiac.
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
ROOT=data/datasets/dsec_mot
OUT=results/dsec_mot_trackeval_simple_detector
LOGDIR=runs/rec_embed_matrix_logs
mkdir -p "$LOGDIR"

B_CKPT=runs/recurrent_embedding/event_frame_voxel_grid_bins3_w32_gated_two_branch_recurrent_embed/best.pt
C_CKPT=runs/recurrent_embedding/event_frame_voxel_grid_bins3_w32_gated_two_branch_embed/best.pt
PLAIN_CKPT=runs/simple_detector_sweep/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt

THRS="0.10 0.25 0.50 0.70 0.90 0.95"
VAL_ARGS="--split train --sequences zurich_city_01_d"
TEST_ARGS="--split test --sequences interlaken_00_d zurich_city_00_b"

log() { echo "[$(date +%H:%M:%S)] $*"; }

thr_label() { echo "$1" | tr -d '.'; }

run_eval() { # run_name output_root checkpoint threshold split_args extra_args...
  local name=$1 out=$2 ckpt=$3 thr=$4; shift 4
  if [ -f "$out/$name/metrics_summary.json" ]; then
    log "SKIP  $name (gotowe)"
    return 0
  fi
  log "EVAL  $name"
  $PY -m src.evaluation.simple_detector_trackeval_cli \
    --checkpoint "$ckpt" --root "$ROOT" \
    --score-threshold "$thr" \
    --device cuda --output-root "$out" --run-name "$name" "$@" \
    > "$LOGDIR/$name.log" 2>&1
}

best_thr() { # results_root prefix -> najlepszy prog wg val HOTA (etykieta "090" -> "0.90")
  $PY - "$1" "$2" <<'EOF'
import json, sys
from pathlib import Path
root, prefix = sys.argv[1], sys.argv[2]
best, label = -1.0, None
for d in Path(root).glob(prefix + "*"):
    p = d / "metrics_summary.json"
    if not p.exists():
        continue
    hota = json.loads(p.read_text())["aggregate"]["HOTA"]
    if hota > best:
        best, label = hota, d.name.split("thr")[-1]
if label is None:
    sys.exit(f"Brak wynikow walidacji dla prefiksu {prefix} w {root}")
print(f"{label[0]}.{label[1:]}")
EOF
}

# ---------------------------------------------------------------- 1. trening C
if [ -f "$C_CKPT" ]; then
  log "SKIP  trening C (checkpoint istnieje)"
else
  log "TRAIN wariant C (bez rekurencji), ~2 h"
  $PY -m src.training.recurrent_embedding_detector \
    --root "$ROOT" --no-recurrent-embedding --resume \
    --num-workers 8 --device cuda \
    --output-dir runs/recurrent_embedding \
    > "$LOGDIR/train_c.log" 2>&1
fi

# ------------------------------------------- 2. sweepy walidacyjne (6 progow)
MM30="--track-max-missed-frames 30"
for thr in $THRS; do
  t=$(thr_label "$thr")
  # B i D-na-B: sweepy juz istnieja (rec_embed_reid/motion_val_thr*), zostana pominiete
  run_eval "rec_embed_reid_val_thr$t"    "$OUT" "$B_CKPT" "$thr" $VAL_ARGS $MM30 --tracker-backend boxmot_botsort --track-with-reid
  run_eval "rec_embed_motion_val_thr$t"  "$OUT" "$B_CKPT" "$thr" $VAL_ARGS $MM30 --tracker-backend boxmot_botsort
  run_eval "c_embed_reid_val_thr$t"      "$OUT" "$C_CKPT" "$thr" $VAL_ARGS $MM30 --tracker-backend boxmot_botsort --track-with-reid
  run_eval "a_gated_iou_val_thr$t"       "$OUT" "$PLAIN_CKPT" "$thr" $VAL_ARGS $MM30 --tracker-backend iou --input-normalisation whole
  run_eval "a_gated_iou_compnorm_val_thr$t" "$OUT" "$PLAIN_CKPT" "$thr" $VAL_ARGS $MM30 --tracker-backend iou --input-normalisation component
  run_eval "d_plain_botsort_val_thr$t"   "$OUT" "$PLAIN_CKPT" "$thr" $VAL_ARGS $MM30 --tracker-backend boxmot_botsort --input-normalisation whole
done

# ------------------- 3. testy (obie sekwencje) przy progu wybranym na walidacji
B_THR=$(best_thr "$OUT" "rec_embed_reid_val_thr")
DB_THR=$(best_thr "$OUT" "rec_embed_motion_val_thr")
C_THR=$(best_thr "$OUT" "c_embed_reid_val_thr")
A_THR=$(best_thr "$OUT" "a_gated_iou_val_thr")
AC_THR=$(best_thr "$OUT" "a_gated_iou_compnorm_val_thr")
DP_THR=$(best_thr "$OUT" "d_plain_botsort_val_thr")
log "progi z walidacji: B=$B_THR D(na B)=$DB_THR C=$C_THR A=$A_THR A(comp)=$AC_THR D(plain)=$DP_THR"

run_eval "rec_embed_reid_testboth_thr$(thr_label "$B_THR")"   "$OUT" "$B_CKPT" "$B_THR" $TEST_ARGS $MM30 --tracker-backend boxmot_botsort --track-with-reid
run_eval "rec_embed_motion_testboth_thr$(thr_label "$DB_THR")" "$OUT" "$B_CKPT" "$DB_THR" $TEST_ARGS $MM30 --tracker-backend boxmot_botsort
# dodatkowo D-na-B przy progu B, do porownania przy dopasowanym progu
run_eval "rec_embed_motion_testboth_thr$(thr_label "$B_THR")" "$OUT" "$B_CKPT" "$B_THR" $TEST_ARGS $MM30 --tracker-backend boxmot_botsort
run_eval "c_embed_reid_testboth_thr$(thr_label "$C_THR")"     "$OUT" "$C_CKPT" "$C_THR" $TEST_ARGS $MM30 --tracker-backend boxmot_botsort --track-with-reid
run_eval "a_gated_iou_testboth_thr$(thr_label "$A_THR")"      "$OUT" "$PLAIN_CKPT" "$A_THR" $TEST_ARGS $MM30 --tracker-backend iou --input-normalisation whole
run_eval "a_gated_iou_compnorm_testboth_thr$(thr_label "$AC_THR")" "$OUT" "$PLAIN_CKPT" "$AC_THR" $TEST_ARGS $MM30 --tracker-backend iou --input-normalisation component
run_eval "d_plain_botsort_testboth_thr$(thr_label "$DP_THR")" "$OUT" "$PLAIN_CKPT" "$DP_THR" $TEST_ARGS $MM30 --tracker-backend boxmot_botsort --input-normalisation whole

# ------- 3b. powtorka ewaluacji z rozdzialu benchmarkowego na OBU sekwencjach
# Kazdy wpis: results_root|baza_run_name|checkpoint|dodatkowe_flagi
# Tracker i normalizacja jak w oryginalnych przebiegach rozdzialu (IoU, domyslne
# ustawienia, normalizacja whole); prog wybierany po istniejacej walidacji.
CHAPTER_ENTRIES=(
  "results/dsec_mot_trackeval_simple_detector|event_frame|runs/simple_detector/event_frame_bins5_w32/best.pt|"
  "results/dsec_mot_trackeval_simple_detector|voxel_grid|runs/simple_detector/voxel_grid_bins5_w32/best.pt|"
  "results/dsec_mot_trackeval_simple_detector|event_frame_voxel_grid|runs/simple_detector/event_frame_voxel_grid_bins5_w32/best.pt|"
  "results/dsec_mot_trackeval_simple_detector|event_frame_voxel_grid_two_branch|runs/simple_detector/event_frame_voxel_grid_bins5_w32_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_eros|eros|runs/simple_detector/eros_bins5_w32/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_eros|event_frame_eros_two_branch|runs/simple_detector/event_frame_eros_bins5_w32_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_eros|voxel_grid_eros_two_branch|runs/simple_detector/voxel_grid_eros_bins5_w32_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_eros|event_frame_voxel_grid_eros_three_branch|runs/simple_detector/event_frame_voxel_grid_eros_bins5_w32_three_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_sweep|bins5_win50ms_event_frame_voxel_grid_two_branch|runs/simple_detector_sweep/bins5_win50ms/event_frame_voxel_grid_bins5_w32_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_sweep|bins5_win50ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_sweep/bins5_win50ms/event_frame_voxel_grid_bins5_w32_gated_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_sweep|bins3_win50ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_sweep/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_sweep|bins5_win100ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_sweep/bins5_win100ms/event_frame_voxel_grid_bins5_w32_gated_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_sweep|bins7_win50ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_sweep/bins7_win50ms/event_frame_voxel_grid_bins7_w32_gated_two_branch/best.pt|"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_event_frame|runs/simple_detector_car_only/bins5_win50ms/event_frame_bins5_w32/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_voxel_grid|runs/simple_detector_car_only/bins5_win50ms/voxel_grid_bins5_w32/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_event_frame_voxel_grid|runs/simple_detector_car_only/bins5_win50ms/event_frame_voxel_grid_bins5_w32/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_event_frame_voxel_grid_two_branch|runs/simple_detector_car_only/bins5_win50ms/event_frame_voxel_grid_bins5_w32_two_branch/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_car_only/bins5_win50ms/event_frame_voxel_grid_bins5_w32_gated_two_branch/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins3_win50ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_car_only/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win100ms_event_frame_voxel_grid_gated_two_branch|runs/simple_detector_car_only/bins5_win100ms/event_frame_voxel_grid_bins5_w32_gated_two_branch/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_eros|runs/simple_detector_car_only/bins5_win50ms/eros_bins5_w32/best.pt|car"
  "results/dsec_mot_trackeval_simple_detector_car_only|bins5_win50ms_event_frame_voxel_grid_eros_three_branch|runs/simple_detector_car_only/bins5_win50ms/event_frame_voxel_grid_eros_bins5_w32_three_branch/best.pt|car"
)

for entry in "${CHAPTER_ENTRIES[@]}"; do
  IFS='|' read -r res_root base ckpt car <<< "$entry"
  thr=$(best_thr "$res_root" "${base}_val_thr")
  extra=()
  if [ "$car" = "car" ]; then
    extra+=(--classes-to-eval car)
  fi
  run_eval "${base}_testboth_thr$(thr_label "$thr")" "$res_root" "$ckpt" "$thr" \
    $TEST_ARGS "${extra[@]}"
done

# ---------------------------------------------------------- 4. raport zbiorczy
log "RAPORT"
$PY - <<'EOF'
import json
from pathlib import Path

root = Path("results/dsec_mot_trackeval_simple_detector")

FAMILIES = [
    ("A  gated bins3 + IoU (norm. jak w raportach)", "a_gated_iou_val_thr", "a_gated_iou_testboth_thr"),
    ("A' gated bins3 + IoU (norm. jak w treningu)", "a_gated_iou_compnorm_val_thr", "a_gated_iou_compnorm_testboth_thr"),
    ("D  gated bins3 + BoT-SORT motion", "d_plain_botsort_val_thr", "d_plain_botsort_testboth_thr"),
    ("D' detektor B + BoT-SORT motion", "rec_embed_motion_val_thr", "rec_embed_motion_testboth_thr"),
    ("C  embed bez rekurencji + BoT-SORT reid", "c_embed_reid_val_thr", "c_embed_reid_testboth_thr"),
    ("B  recurrent embed + BoT-SORT reid", "rec_embed_reid_val_thr", "rec_embed_reid_testboth_thr"),
]

def agg(path):
    p = path / "metrics_summary.json"
    return json.loads(p.read_text()) if p.exists() else None

print(f"\n{'wariant':46s} {'thr':>5s} {'valHOTA':>8s} {'tstHOTA':>8s} {'tstIDF1':>8s} {'IDS':>4s} {'interlaken':>10s} {'zurich00b':>10s}")
for label, valp, testp in FAMILIES:
    best, thr = -1.0, None
    for d in root.glob(valp + "*"):
        s = agg(d)
        if s and s["aggregate"]["HOTA"] > best:
            best, thr = s["aggregate"]["HOTA"], d.name.split("thr")[-1]
    if thr is None:
        print(f"{label:46s}  brak wynikow walidacji"); continue
    row = f"{label:46s} {thr[0]}.{thr[1:]:<3s} {best:8.4f}"
    t = agg(root / f"{testp}{thr}")
    if t:
        a = t["aggregate"]; per = t["per_sequence"]
        def seq(name):
            m = per.get(name, {}).get("metrics", {})
            return f"{m.get('HOTA', float('nan')):10.4f}"
        row += f" {a['HOTA']:8.4f} {a['IDF1']:8.4f} {a['IDS']:4d} {seq('interlaken_00_d')} {seq('zurich_city_00_b')}"
    else:
        row += "   (test w toku)"
    print(row)
print("\nPary do wnioskow: B vs D' = wklad embeddingow (te same detekcje); B vs C = wklad rekurencji;")
print("A vs A' = efekt naprawy normalizacji; D vs D' = efekt wspoltrenowania detektora; A vs B = calosc.")

CHAPTER = [
    ("tab:wynikiTest", "EF", "dsec_mot_trackeval_simple_detector", "event_frame"),
    ("tab:wynikiTest", "VG", "dsec_mot_trackeval_simple_detector", "voxel_grid"),
    ("tab:wynikiTest", "EF+VG single", "dsec_mot_trackeval_simple_detector", "event_frame_voxel_grid"),
    ("tab:wynikiTest", "EF+VG two-branch", "dsec_mot_trackeval_simple_detector", "event_frame_voxel_grid_two_branch"),
    ("tab:wynikiTest", "EROS", "dsec_mot_trackeval_simple_detector_eros", "eros"),
    ("tab:wynikiTest", "EROS+EF two-branch", "dsec_mot_trackeval_simple_detector_eros", "event_frame_eros_two_branch"),
    ("tab:wynikiTest", "EROS+VG two-branch", "dsec_mot_trackeval_simple_detector_eros", "voxel_grid_eros_two_branch"),
    ("tab:wynikiTest", "EROS+EF+VG three-br", "dsec_mot_trackeval_simple_detector_eros", "event_frame_voxel_grid_eros_three_branch"),
    ("tab:gatedSweep", "two-branch 5b/50ms", "dsec_mot_trackeval_simple_detector_sweep", "bins5_win50ms_event_frame_voxel_grid_two_branch"),
    ("tab:gatedSweep", "gated 5b/50ms", "dsec_mot_trackeval_simple_detector_sweep", "bins5_win50ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:gatedSweep", "gated 3b/50ms", "dsec_mot_trackeval_simple_detector_sweep", "bins3_win50ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:gatedSweep", "gated 5b/100ms", "dsec_mot_trackeval_simple_detector_sweep", "bins5_win100ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:gatedSweep", "gated 7b/50ms", "dsec_mot_trackeval_simple_detector_sweep", "bins7_win50ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:wynikiCarOnly", "EF", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_event_frame"),
    ("tab:wynikiCarOnly", "VG", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_voxel_grid"),
    ("tab:wynikiCarOnly", "EF+VG single", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_event_frame_voxel_grid"),
    ("tab:wynikiCarOnly", "EF+VG two-branch", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_event_frame_voxel_grid_two_branch"),
    ("tab:wynikiCarOnly", "gated 5b/50ms", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:wynikiCarOnly", "gated 3b/50ms", "dsec_mot_trackeval_simple_detector_car_only", "bins3_win50ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:wynikiCarOnly", "gated 5b/100ms", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win100ms_event_frame_voxel_grid_gated_two_branch"),
    ("tab:wynikiCarOnly", "EROS", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_eros"),
    ("tab:wynikiCarOnly", "EROS+EF+VG three-br", "dsec_mot_trackeval_simple_detector_car_only", "bins5_win50ms_event_frame_voxel_grid_eros_three_branch"),
]

print("\n--- Powtorka rozdzialu benchmarkowego na 2 sekwencjach testowych ---")
print(f"{'tabela':18s} {'wariant':20s} {'thr':>5s} {'stare(inter)':>12s} {'nowe(2 sekw.)':>13s} {'interlaken':>10s} {'zurich00b':>10s} {'MOTA':>8s} {'IDF1':>8s} {'IDS':>4s}")
for table, name, res_root, base in CHAPTER:
    rroot = Path("results") / res_root
    best, thr = -1.0, None
    for d in rroot.glob(base + "_val_thr*"):
        s = agg(d)
        if s and s["aggregate"]["HOTA"] > best:
            best, thr = s["aggregate"]["HOTA"], d.name.split("thr")[-1]
    if thr is None:
        print(f"{table:18s} {name:20s}  brak walidacji"); continue
    old = agg(rroot / f"{base}_test_thr{thr}")
    new = agg(rroot / f"{base}_testboth_thr{thr}")
    old_h = f"{old['aggregate']['HOTA']:12.4f}" if old else " " * 12
    if new:
        a = new["aggregate"]; per = new["per_sequence"]
        def seq(nm):
            m = per.get(nm, {}).get("metrics", {})
            return f"{m.get('HOTA', float('nan')):10.4f}"
        print(f"{table:18s} {name:20s} {thr[0]}.{thr[1:]:<3s} {old_h} {a['HOTA']:13.4f} {seq('interlaken_00_d')} {seq('zurich_city_00_b')} {a['MOTA']:8.4f} {a['IDF1']:8.4f} {a['IDS']:4d}")
    else:
        print(f"{table:18s} {name:20s} {thr[0]}.{thr[1:]:<3s} {old_h}   (testboth w toku)")
EOF

log "GOTOWE"
