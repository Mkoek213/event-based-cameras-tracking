# Architecture Overview

## Directory Structure

```
event-based-cameras-tracking/
├── codex/                     # Working status notes for Codex sessions
├── data/                      # Local data storage (large files are NOT tracked)
│   ├── raw/                   # Optional raw downloads
│   ├── processed/             # Optional preprocessed files
│   ├── datasets/              # Local datasets, including DSEC-MOT
│   └── cache/                 # Local representation cache, e.g. EROS snapshots
├── docs/                      # Project documentation
│   ├── architecture.md        # This file
│   └── setup.md               # Environment setup instructions
├── scripts/                   # Legacy standalone tools and external-baseline utilities
├── src/                       # Importable Python package
│   ├── data/                  # Dataset classes, preprocessing and EROS cache generation
│   ├── experiments/           # Benchmark orchestration and summary generation
│   ├── models/                # SimpleDenseDetector and pretrained adapters
│   ├── training/              # Training entry-point implementations
│   ├── utils/                 # Metrics, visualisation, I/O helpers
│   └── evaluation/            # DSEC-MOT export / tracking / TrackEval glue
├── tests/                     # Pytest unit tests
│   ├── test_data/
│   ├── test_evaluation/
│   └── test_models/
├── runs/                      # Local training checkpoints and logs (ignored)
├── results/                   # Local benchmark outputs and summaries (ignored)
└── external/                  # Local external repos, e.g. TrackEval (ignored)
```

The empty directories `configs/`, `experiments/`, `models/` and `notebooks/`
are old scaffolding from an earlier project layout. They are not used by the
current benchmark pipeline and should not be treated as active locations for new
experiments.

## Key Design Decisions

* **`src/` layout** – placing source code under `src/` prevents accidental
  imports of uninstalled packages and is the modern Python packaging
  standard.
* **Controlled detector-first benchmark** – the active thesis path uses a small
  in-repo `SimpleDenseDetector` trained from scratch, because the main research
  variable is the event representation rather than the absolute detector SOTA.
  EvRT-DETR remains useful as infrastructure and external-baseline code.
* **Separation of concerns** – dataset access, export logic, tracking, and
  evaluation are kept separate to make detector swapping easier.
* **Module-first execution** – active benchmark code is run with
  `python -m src...`. Experiment orchestration and result summarisation live in
  `src/experiments/`. Dataset/cache logic lives in `src/data/`, training logic
  lives in `src/training/`, and metric/evaluation logic lives in
  `src/evaluation/`. The `scripts/` directory is kept only for legacy
  standalone tools and external-baseline utilities.
* **Event-based processing** – `src/data/preprocessing.py` and
  `src/data/representations.py` build dense event representations used by the
  controlled benchmark.
* **Same tracking/evaluation backend** – all simple-detector variants use NMS,
  the same IoU tracker and the same TrackEval adapter.

## Current Benchmark Architecture

The controlled benchmark uses:

1. `BenchmarkRepresentation` to create one of:
   - `event_frame`
   - `voxel_grid`
   - `event_frame_voxel_grid`
   - `eros`
   - `event_frame_eros`
   - `voxel_grid_eros`
   - `event_frame_voxel_grid_eros`
2. `SimpleDenseDetector` to predict dense class logits and bounding-box
   distances on a stride-8 grid.
3. Dense decoding + NMS.
4. `SimpleIoUTracker` for class-aware tracking-by-detection.
5. `TrackEval` for `HOTA`, `MOTA`, `IDF1`, `IDS`, `FP`, `FN`.
6. `src.evaluation.detection_metrics_cli` for detection-side `AP50`,
   precision, recall and F1.

## Active Entry Points

Active benchmark commands are Python modules run with `python -m`. The table
below lists the modules that should be used for new experiments.

| Module | Purpose | Main outputs |
| --- | --- | --- |
| `src.training.simple_detector` | Train `SimpleDenseDetector` for one representation/fusion configuration. | `runs/.../best.pt`, `runs/.../last.pt`, `config.json`, `history.json`. |
| `src.evaluation.simple_detector_trackeval_cli` | Export detections from a simple-detector checkpoint, run NMS, IoU tracking and TrackEval. | `results/.../detections/*.json`, `tracks/*.txt`, TrackEval reports, `metrics_summary.json`, `metrics_summary.csv`. |
| `src.data.eros_precompute` | Precompute persistent EROS snapshots at DSEC-MOT image timestamps. | `data/cache/dsec_mot_eros/.../snapshots.npy`, `metadata.json`. |
| `src.evaluation.detection_metrics_cli` | Compute detection-only metrics from exported detection JSON files. | `detection_metrics.json` per run and optional aggregate CSV. |
| `src.experiments.simple_detector_representation_sweep` | Run EF/VG/EF+VG representation sweeps over bins, windows and thresholds. | Trained checkpoints, TrackEval summaries for validation/test, logs. |
| `src.experiments.simple_detector_eros_benchmark` | Run EROS and EROS-fusion benchmark variants. | EROS cache, checkpoints, TrackEval summaries. |
| `src.experiments.simple_detector_car_only_benchmark` | Train and evaluate the same benchmark with car-only labels/evaluation. | Car-only checkpoints, `car_summary.txt`, detection metrics CSV, selected-threshold CSV. |
| `src.experiments.simple_detector_large_benchmark` | Run the same representation benchmark with a wider detector and optional seeds. | Large-model checkpoints and TrackEval summaries. |
| `src.experiments.pretrained_representation_benchmark` | Fine-tune a COCO-pretrained Faster R-CNN through representation adapters. | Pretrained-detector checkpoints and TrackEval summaries. |
| `src.experiments.summarise_*` | Select validation thresholds and produce compact result tables. | Summary CSVs used for thesis tables and analysis. |

Typical command shape:

```bash
.venv/bin/python -m src.experiments.simple_detector_representation_sweep \
  --num-bins 5 \
  --time-window-us 50000 \
  --variants event_frame_voxel_grid_two_branch \
  --epochs 30 \
  --batch-size 16 \
  --device cuda
```

## Fusion Modes

`SimpleDenseDetector` supports four input/fusion modes:

- `single` – the whole representation tensor goes directly into the shared
  backbone.
- `two_branch` – two representation components are first processed by separate
  convolutional input blocks, then concatenated and fused with a `1x1`
  convolution.
- `three_branch` – the same idea for three components, used for `EF+VG+EROS`.
- `gated_two_branch` – two branches are weighted by a small gating network
  before feature fusion.

## Main Experiment Outputs

Local experiment outputs are intentionally not tracked by Git:

- `runs/` – checkpoints and logs.
- `results/` – detection exports, tracks, TrackEval reports and summary CSVs.
- `data/cache/` – EROS snapshots.
- `external/` – external repositories such as TrackEval and EvRT-DETR.
- Empty legacy scaffold directories such as `configs/`, `experiments/`,
  `models/` and `notebooks/` are not part of the active workflow.

The repository should track code, tests and small documentation only.
