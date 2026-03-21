# event-based-cameras-tracking

> **Object tracking using event-based cameras** — an engineering research project
> applying modern deep-learning techniques to high-speed, low-latency event streams.

---

## Project Structure

```
event-based-cameras-tracking/
├── configs/                   # Hydra / OmegaConf configuration files
│   ├── base_config.yaml       # Shared default settings
│   └── experiments/           # Per-experiment config overrides
├── data/                      # Data storage (NOT committed to Git)
│   ├── raw/                   # Raw event recordings (.dat, .raw, .h5)
│   ├── processed/             # Preprocessed event tensors / frames
│   └── datasets/              # Dataset split manifests (CSV / JSON)
├── docs/                      # Project documentation
│   ├── architecture.md        # Directory structure & design decisions
│   └── setup.md               # Environment setup instructions
├── experiments/               # Experiment artefacts (NOT committed to Git)
│   └── logs/                  # TensorBoard / MLflow logs
├── models/                    # Model weights (NOT committed to Git)
│   ├── checkpoints/           # Training checkpoints
│   └── pretrained/            # Pre-trained backbone weights
├── notebooks/                 # Jupyter notebooks for data exploration
│   └── exploration/
├── scripts/                   # Entry-point scripts
│   ├── train.py               # Start a training run
│   ├── evaluate.py            # Evaluate a checkpoint
│   └── inference.py           # Run inference on new data
├── src/                       # Importable Python package
│   ├── data/                  # Dataset classes & preprocessing transforms
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── models/                # Backbone, detector, and tracker
│   │   ├── backbone.py
│   │   ├── detector.py
│   │   └── tracker.py
│   ├── utils/                 # Metrics, visualisation, I/O helpers
│   │   ├── metrics.py
│   │   ├── visualization.py
│   │   └── io.py
│   ├── training/              # Trainer loop & loss functions
│   │   ├── trainer.py
│   │   └── losses.py
│   └── evaluation/            # Evaluation pipeline
│       └── evaluator.py
└── tests/                     # Pytest unit tests (mirrors src/)
    ├── test_data/
    ├── test_models/
    ├── test_utils/
    ├── test_training/
    └── test_evaluation/
```

---

## Quick Start

```bash
# 1. Clone & set up environment
git clone https://github.com/Mkoek213/event-based-cameras-tracking.git
cd event-based-cameras-tracking
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Train
python scripts/train.py --config configs/base_config.yaml

# 3. Evaluate
python scripts/evaluate.py --config configs/base_config.yaml \
    --checkpoint models/checkpoints/checkpoint_epoch_0100.pt

# 4. Run tests
pytest
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions and
[docs/architecture.md](docs/architecture.md) for design decisions.
