# Architecture Overview

## Directory Structure

```
event-based-cameras-tracking/
├── configs/                   # Hydra / OmegaConf configuration files
│   ├── base_config.yaml       # Shared default settings
│   └── experiments/           # Per-experiment config overrides
├── data/                      # Data storage (NOT tracked by Git, except .gitkeep)
│   ├── raw/                   # Raw event recordings (.dat, .raw, .h5)
│   ├── processed/             # Preprocessed event tensors / frames
│   └── datasets/              # Dataset split manifests (CSV / JSON)
├── docs/                      # Project documentation
│   ├── architecture.md        # This file
│   └── setup.md               # Environment setup instructions
├── experiments/               # Experiment artefacts (NOT tracked by Git)
│   └── logs/                  # TensorBoard / MLflow logs
├── models/                    # Model weights (NOT tracked by Git)
│   ├── checkpoints/           # Training checkpoints
│   └── pretrained/            # Pre-trained backbone weights
├── notebooks/                 # Jupyter notebooks for exploration
│   └── exploration/
├── scripts/                   # Entry-point scripts (train / eval / infer)
│   ├── train.py
│   ├── evaluate.py
│   └── inference.py
├── src/                       # Importable Python package
│   ├── data/                  # Dataset classes & preprocessing transforms
│   ├── models/                # Backbone, detector, and tracker definitions
│   ├── utils/                 # Metrics, visualisation, I/O helpers
│   ├── training/              # Trainer loop & loss functions
│   └── evaluation/            # Evaluation pipeline
└── tests/                     # Pytest unit tests (mirrors src/)
    ├── test_data/
    ├── test_models/
    ├── test_utils/
    ├── test_training/
    └── test_evaluation/
```

## Key Design Decisions

* **`src/` layout** – placing source code under `src/` prevents accidental
  imports of uninstalled packages and is the modern Python packaging
  standard.
* **Hydra + OmegaConf** – configuration is managed declaratively; experiment
  overrides live in `configs/experiments/` and are composed at run-time.
* **Separation of concerns** – data loading, model definition, training
  logic, and evaluation are kept in distinct sub-packages to maximise
  testability and reuse.
* **Event-based processing** – the `src/data/` module wraps the `tonic`
  library for event-stream transforms and integrates directly with standard
  PyTorch `DataLoader`.
