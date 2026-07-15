# Event-ReID Association Embedding Benchmark

The R0/R1/R2 benchmark uses the selected event-only detector configuration:

```text
DSEC-MOT events
  -> Event Frame (2 channels) + 3-bin polarity Voxel Grid (6 channels), 50 ms
  -> gated two-branch SimpleDenseDetector, width 32, stride 8
  -> class-aware NMS
  -> optional post-NMS RoI association descriptors
  -> BoT-SORT
  -> TrackEval
```

R0 keeps the historical detector-only checkpoints and motion-only BoT-SORT. R1
adds non-recurrent object descriptors. R2 adds a spatial ConvGRU only inside the
association head.

## Association head

The shared stride-8 backbone map is passed through:

```text
ConvBlock(4 * width, 128)
  -> optional ConvGRU(128, 128)
  -> RoIAlign(7 x 7, spatial_scale=1/8, aligned=True)
  -> global average pooling
  -> Linear(128, 256)
  -> BatchNorm neck
  -> L2 normalisation
```

RoIAlign runs after detector decoding and class-aware NMS at inference. Descriptor
rows are attached to detections in their existing exported order and are passed to
BoxMOT BoT-SORT as external embeddings. R1 always returns no recurrent state. R2
resets state at clip boundaries during training and at sequence boundaries during
evaluation.

Training uses:

```text
detection loss
+ 1.0 * identity cross-entropy
+ 1.0 * class-aware batch-hard cosine triplet loss
```

The triplet margin is 0.3. Positives have the same sequence-qualified training
identity; negatives have another identity and the same object class. Mining uses all
objects from the current batch of clips and all eight time steps. Gradient
accumulation does not enlarge that mining pool.

Validation identities are not classifier classes. Checkpoint selection uses
same-class retrieval on `train/zurich_city_01_d`: highest mAP, then Rank-1, then
lower detection loss. Validation fails if there are no valid retrieval queries.

## Checkpoint paths

R0 paths are unchanged:

```text
runs/simple_detector_sweep/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt
runs/simple_detector_car_only/bins3_win50ms/event_frame_voxel_grid_bins3_w32_gated_two_branch/best.pt
```

R1/R2 paths are:

```text
runs/event_reid_embedding/all_classes/event_frame_voxel_grid_bins3_w32_gated_two_branch_r1_non_recurrent/best.pt
runs/event_reid_embedding/all_classes/event_frame_voxel_grid_bins3_w32_gated_two_branch_r2_recurrent/best.pt
runs/event_reid_embedding/car_only/event_frame_voxel_grid_bins3_w32_gated_two_branch_r1_non_recurrent/best.pt
runs/event_reid_embedding/car_only/event_frame_voxel_grid_bins3_w32_gated_two_branch_r2_recurrent/best.pt
```

Every R1/R2 directory contains `best.pt`, `last.pt`, `config.json`, and
`history.json`.

## Full training commands

Run the four trainings through the benchmark orchestrator:

```bash
.venv/bin/python -m src.experiments.event_reid_embedding_benchmark \
  --device cuda \
  --epochs 30 \
  --batch-size 2 \
  --grad-accum-steps 4 \
  --num-workers 4 \
  --skip-val \
  --skip-test
```

This expands to R1 and R2 for both `all_classes` and `car_only`, always with
seed 0, clip length/stride 8/8, AdamW at `1e-3`, weight decay `1e-4`, cosine
annealing, gradient clipping at 5, and CUDA AMP. Add `--resume` to resume
incomplete `last.pt` checkpoints. Use `--scope all_classes` or
`--scope car_only` to run only one protocol.

## Validation and selected test

Inspect the full four-training and validation command matrix without executing it:

```bash
.venv/bin/python -m src.experiments.event_reid_embedding_benchmark \
  --dry-run \
  --device cuda
```

After all four checkpoints exist, run only validation sweeps:

```bash
.venv/bin/python -m src.experiments.event_reid_embedding_benchmark \
  --skip-train \
  --skip-test \
  --device cuda
```

The validation grids are:

- all classes: detector score `0.10 0.25 0.50 0.70 0.90 0.95`;
- car-only: detector score `0.90 0.95 0.97 0.99`;
- ReID appearance threshold `0.10 0.20 0.25 0.30 0.40 0.50`;
- ReID proximity threshold `0.30 0.50 0.70`.

R0 and the R1/R2 motion-only diagnostics tune only detector score. R1/R2 primary
runs tune all three thresholds. Selection is by HOTA, AssA, then IDF1.

Run the two final test sequences once with validation-selected settings:

```bash
.venv/bin/python -m src.experiments.event_reid_embedding_benchmark \
  --skip-train \
  --skip-val \
  --device cuda
```

The test stage evaluates `test/interlaken_00_d` and
`test/zurich_city_00_b` together. It never derives thresholds from test metrics.

Results are stored under `results/dsec_mot_event_reid_embedding/`. Stable run
names include scope, variant, mode, split, detector score, appearance threshold,
and proximity threshold. The final files are:

```text
validation_selection.json
benchmark_summary.json
benchmark_summary.csv
```

They report checkpoint, selected epoch and thresholds, retrieval mAP/Rank-1,
ReID status, HOTA, AssA, DetA, MOTA, IDF1, IDS, FP, and FN for each test sequence
and the combined test set.

## Verification commands

Complete repository checks:

```bash
.venv/bin/python -m pytest
.venv/bin/python -m ruff check src tests
```

Small standalone CPU smokes can reduce dimensions while retaining the R1/R2 data
flow. The implementation verification used:

```bash
.venv/bin/python -m src.training.recurrent_embedding_detector \
  --root data/datasets/dsec_mot \
  --train-sequences interlaken_00_a \
  --val-sequences zurich_city_01_d \
  --representation event_frame_voxel_grid \
  --fusion-mode gated_two_branch \
  --num-bins 3 \
  --time-window-us 50000 \
  --model-width 4 \
  --embedding-hidden-dim 8 \
  --embedding-dim 16 \
  --roi-size 3 \
  --no-recurrent-embedding \
  --clip-length 2 \
  --clip-stride 2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --num-workers 0 \
  --max-train-clips 1 \
  --max-val-clips 1 \
  --device cpu \
  --no-amp \
  --output-dir /tmp/event_reid_r1_smoke \
  --run-name r1_cpu_smoke

.venv/bin/python -m src.training.recurrent_embedding_detector \
  --root data/datasets/dsec_mot \
  --train-sequences interlaken_00_a \
  --val-sequences zurich_city_01_d \
  --representation event_frame_voxel_grid \
  --fusion-mode gated_two_branch \
  --num-bins 3 \
  --time-window-us 50000 \
  --model-width 4 \
  --embedding-hidden-dim 8 \
  --embedding-dim 16 \
  --roi-size 3 \
  --recurrent-embedding \
  --clip-length 2 \
  --clip-stride 2 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum-steps 1 \
  --num-workers 0 \
  --max-train-clips 1 \
  --max-val-clips 1 \
  --device cpu \
  --no-amp \
  --output-dir /tmp/event_reid_r2_smoke \
  --run-name r2_cpu_smoke
```

The reduced smoke dimensions are verification-only. Full benchmark commands retain
the decision-complete 128-hidden/256-output/7x7 architecture. No complete 30-epoch
benchmark is launched by these instructions.
