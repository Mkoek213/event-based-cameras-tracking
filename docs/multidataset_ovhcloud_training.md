# Multi-Dataset OVHCloud Training Plan

This document describes the planned large pretraining run for the `EF+VG gated_two_branch` detector on a temporary OVHCloud machine with 2 GPUs and about 64 GB VRAM.

## Goal

Train one detector architecture on substantially more event data than DSEC-MOT alone, then fine-tune and evaluate on DSEC-MOT with the same TrackEval protocol used in the thesis benchmark.

The final reported benchmark should still be DSEC-MOT. External datasets are used for pretraining, not for one mixed leaderboard.

## Dataset Priority

### Directly relevant MOT / traffic datasets

1. `DSEC-MOT` - final fine-tuning and final benchmark.
2. `FEMOT` - RGB-event MOT, traffic scenes, track IDs.
3. `MEVDT` - event/grayscale vehicle detection and tracking.
4. `TUMTraf Event` / `TUMTraf EMOT` - roadside traffic event/RGB data.
5. `Prophesee Gen1 Automotive` - large event detection data for cars/pedestrians.
6. `Prophesee 1MP Automotive` - large event detection data for cars/pedestrians/two-wheelers.

### Auxiliary SOT datasets for representation pretraining

Use lower sampling weights because these datasets provide single-object tracking supervision, not MOT supervision.

1. `EventVOT`
2. `COESOT`
3. `VisEvent`
4. `CRSOT`
5. `N-MuPeTS` only if pedestrian robustness becomes important.

## Shared Class Space

Pretraining should use a coarse class space:

- `vehicle`: car, truck, bus, train, generic vehicle
- `pedestrian`: pedestrian, person
- `two_wheeler`: bicycle, motorcycle
- `object`: unknown SOT target

For final DSEC-MOT fine-tuning, either keep the 3 traffic classes or replace the classifier with the 7 DSEC-MOT classes and load compatible backbone/fusion weights with `--resume-from-pretrained`.

## Manifest Format

External dataset converters should write JSONL manifests consumed by `UnifiedDenseRepresentationDataset`.

Required fields:

```json
{
  "dataset": "femot",
  "sequence": "dvSave-...",
  "timestamp_us": 123456789,
  "frame_index": 0,
  "width": 640,
  "height": 480,
  "representation_paths": {
    "event_frame": "cache/femot/seq/frame_000001_ef.npy",
    "voxel_grid": "cache/femot/seq/frame_000001_vg.npy"
  },
  "boxes": [[120.0, 80.0, 180.0, 160.0]],
  "labels": ["vehicle"],
  "track_ids": [42]
}
```

A row may also use `representation_path` when all channels are already concatenated in one `.npy` or `.npz` tensor.


## Manifest Conversion Pipeline

External datasets are normalized with:

```bash
python -m src.data.converters.build_manifest <command> ...
```

The converter writes cached dense representations and two JSONL manifests:

- `cache/<dataset>/<sequence>/*_ef.npy`
- `cache/<dataset>/<sequence>/*_vg.npy`
- `manifests/pretrain_train.jsonl`
- `manifests/pretrain_val.jsonl`

Use `inspect` before converting a new dataset package:

```bash
python -m src.data.converters.build_manifest inspect PATH_TO_EVENTS PATH_TO_BOXES
```

### DSEC Detection

```bash
python -m src.data.converters.build_manifest dsec-detection \
  --manifest ~/work/data/datasets/dsec_detection/usable_manifest/dsec_detection_usable_abs.csv \
  --output-root data/unified_event_mot/dsec_detection \
  --width 640 \
  --height 480 \
  --num-bins 5 \
  --time-window-us 50000 \
  --sample-stride 5
```

Use `--max-samples 20` for a smoke test before the full conversion.

### eTraM Static

Run `inspect` on one `*_td.h5` file first and set `--width/--height` to the reported sensor resolution. Then run:

```bash
python -m src.data.converters.build_manifest etram \
  --hdf5-root ~/work/data/datasets/etram/static/usable/hdf5 \
  --bbox-root ~/work/data/datasets/etram/static/usable/annotations \
  --output-root data/unified_event_mot/etram \
  --width WIDTH_FROM_INSPECT \
  --height HEIGHT_FROM_INSPECT \
  --num-bins 5 \
  --time-window-us 50000 \
  --sample-stride 5
```

### Prophesee 1MP Automotive

Run after all `.7z` archives have been extracted. Inspect one `*_td.dat` and matching bbox file first, then convert:

```bash
python -m src.data.converters.build_manifest prophesee-1mp \
  --root ~/work/data/datasets/prophesee_1mp/extracted \
  --output-root data/unified_event_mot/prophesee_1mp \
  --width 1280 \
  --height 720 \
  --num-bins 5 \
  --time-window-us 50000 \
  --sample-stride 5
```

For the first run use `--max-samples 100` to verify that `.dat` decoding, bbox loading and cache writing are correct.

### Merge Converted Datasets

```bash
python -m src.data.converters.build_manifest merge \
  --output-root data/unified_event_mot/merged \
  data/unified_event_mot/dsec_detection \
  data/unified_event_mot/etram \
  data/unified_event_mot/prophesee_1mp
```

The training command should point to `data/unified_event_mot/merged/manifests/pretrain_train.jsonl` and `data/unified_event_mot/merged/manifests/pretrain_val.jsonl`.

## OVHCloud Pretraining Command

Run inside `tmux` and keep logs in `runs/`:

```bash
mkdir -p runs/ovh_gated_multidataset/logs

CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
torchrun --standalone --nproc_per_node=2 -m src.training.simple_detector \
  --train-manifest data/unified_event_mot/merged/manifests/pretrain_train.jsonl \
  --val-manifest data/unified_event_mot/merged/manifests/pretrain_val.jsonl \
  --representation event_frame_voxel_grid \
  --fusion-mode gated_two_branch \
  --architecture csp_pan \
  --model-width 128 \
  --num-classes 4 \
  --num-bins 5 \
  --time-window-us 50000 \
  --max-steps 1200000 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --num-workers 12 \
  --lr 3e-4 \
  --weight-decay 1e-4 \
  --device cuda \
  --output-dir runs/ovh_gated_multidataset/pretrain \
  --resume \
  2>&1 | tee -a runs/ovh_gated_multidataset/logs/pretrain.log
```

## DSEC-MOT Fine-Tuning Command

```bash
mkdir -p runs/ovh_gated_multidataset/logs

CUDA_VISIBLE_DEVICES=0,1 \
NCCL_DEBUG=WARN \
OMP_NUM_THREADS=8 \
torchrun --standalone --nproc_per_node=2 -m src.training.simple_detector \
  --root data/datasets/dsec_mot \
  --representation event_frame_voxel_grid \
  --fusion-mode gated_two_branch \
  --architecture csp_pan \
  --model-width 128 \
  --num-bins 5 \
  --time-window-us 50000 \
  --epochs 80 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --num-workers 12 \
  --lr 1e-4 \
  --weight-decay 1e-4 \
  --device cuda \
  --resume-from-pretrained runs/ovh_gated_multidataset/pretrain/event_frame_voxel_grid_bins5_w128_csp_pan_gated_two_branch/best.pt \
  --output-dir runs/ovh_gated_multidataset/finetune_dsec \
  --resume \
  2>&1 | tee -a runs/ovh_gated_multidataset/logs/finetune_dsec.log
```

## Implementation Status

Implemented now:

- `src.data.unified_manifest.UnifiedDenseRepresentationDataset`
- manifest JSONL reader/writer
- shared dense target encoding in `src.data.dense_targets`
- manifest mode in `src.training.simple_detector`
- single-node DDP through `torchrun`
- compatible checkpoint loading through `--resume-from-pretrained`
- external dataset converters in `src.data.converters.build_manifest`
- EF/VG cache generation for DSEC Detection, eTraM Static and Prophesee 1MP

Still dataset-dependent:

- final Prophesee 1MP extraction verification
- class-id mapping verification for every external dataset
- final full conversion runtime and storage estimate
