# Environment Setup

## Prerequisites

* Python ≥ 3.9
* CUDA ≥ 11.8 (optional, for GPU acceleration)

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/Mkoek213/event-based-cameras-tracking.git
cd event-based-cameras-tracking

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Install the project as an editable package
pip install -e .
```

For the current DSEC-MOT workflow, `requirements.txt` is enough. Optional
packages for other event formats such as `.dat` / `.raw` can be installed
later if they become necessary.

## Validating DSEC-MOT

```bash
.venv/bin/python scripts/validate_dsec_mot.py
```

## Running Public EvRT-DETR Inference

```bash
.venv/bin/python scripts/run_evrtdetr_dsec_mot.py \
  --split test \
  --sequence interlaken_00_d \
  --max-frames 100
```

## Converting DSEC-MOT For EvRT-DETR Fine-Tuning

```bash
.venv/bin/python scripts/convert_dsec_mot_to_evrtdetr_npz.py
```

## Running EvRT-DETR Export + MOT Evaluation

```bash
.venv/bin/python scripts/export_evrtdetr_dsec_mot.py \
  --split test \
  --sequence interlaken_00_d

.venv/bin/python scripts/evaluate_evrtdetr_dsec_mot_trackeval.py \
  --split test
```

## Running The Active Representation Benchmark

The active thesis benchmark is module-first and should be run with
`python -m src...`, not through wrappers in `scripts/`.

Main representation sweep:

```bash
.venv/bin/python -m src.experiments.simple_detector_representation_sweep \
  --num-bins 5 \
  --time-window-us 50000 \
  --variants event_frame voxel_grid event_frame_voxel_grid_two_branch \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda
```

EROS cache and EROS benchmark:

```bash
.venv/bin/python -m src.data.eros_precompute

.venv/bin/python -m src.experiments.simple_detector_eros_benchmark \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda
```

Car-only benchmark:

```bash
.venv/bin/python -m src.experiments.simple_detector_car_only_benchmark \
  --epochs 30 \
  --batch-size 16 \
  --num-workers 4 \
  --device cuda
```

## Running Tests

```bash
.venv/bin/python -m pytest
```

## Pre-commit Hooks

The repository uses `pre-commit` for lightweight checks before commits:

- whitespace and end-of-file cleanup,
- YAML/TOML/JSON sanity checks,
- merge-conflict and private-key detection,
- Python syntax checks,
- `ruff` linting with safe fixes,
- `ruff-format` formatting.

Install the hooks once after setting up the virtual environment:

```bash
.venv/bin/python -m pre_commit install
```

Run the same checks manually on all tracked files:

```bash
.venv/bin/python -m pre_commit run --all-files
```

For a faster Python-only check during development, run Ruff on the active
package code and tests:

```bash
.venv/bin/python -m ruff check src tests
.venv/bin/python -m ruff format src tests
```
