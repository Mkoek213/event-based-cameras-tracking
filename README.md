# event-based-cameras-tracking

> **Object tracking using event-based cameras** — an engineering research project
> applying modern deep-learning techniques to high-speed, low-latency event streams.

---

## Project Structure

```
event-based-cameras-tracking/
├── configs/                   # Configuration files for active experiments
│   └── experiments/
├── data/                      # Data storage (NOT committed to Git)
│   ├── raw/                   # Raw event recordings (.dat, .raw, .h5)
│   ├── processed/             # Preprocessed event tensors / frames
│   └── datasets/              # Dataset split manifests (CSV / JSON)
├── docs/                      # Project documentation
│   ├── architecture.md        # Directory structure & design decisions
│   └── setup.md               # Environment setup instructions
├── experiments/               # Experiment artefacts (NOT committed to Git)
│   └── logs/
├── models/                    # Model weights (NOT committed to Git)
│   ├── checkpoints/
│   └── pretrained/
├── notebooks/                 # Jupyter notebooks for data exploration
│   └── exploration/
├── scripts/                   # Entry-point scripts
│   ├── convert_dsec_mot_to_evrtdetr_npz.py
│   ├── run_evrtdetr_dsec_mot.py
│   ├── export_evrtdetr_dsec_mot.py
│   ├── evaluate_evrtdetr_dsec_mot_trackeval.py
│   ├── run_simple_tracker_dsec_mot.py
│   ├── render_dsec_mot_video.py
│   └── validate_dsec_mot.py
├── src/                       # Importable Python package
│   ├── data/                  # Dataset classes & preprocessing transforms
│   │   ├── dataset.py
│   │   └── preprocessing.py
│   ├── evaluation/            # DSEC-MOT export / tracking / TrackEval glue
│   ├── utils/                 # Metrics, visualisation, I/O helpers
│   │   ├── metrics.py
│   │   └── io.py
└── tests/                     # Pytest unit tests (mirrors src/)
    ├── test_data/
    └── test_utils/
```

---

## Quick Start

```bash
# 1. Clone & set up environment
git clone https://github.com/Mkoek213/event-based-cameras-tracking.git
cd event-based-cameras-tracking
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Validate local DSEC-MOT layout
.venv/bin/python scripts/validate_dsec_mot.py

# 3. Run the public pretrained EvRT-DETR checkpoint on a DSEC-MOT sequence
.venv/bin/python scripts/run_evrtdetr_dsec_mot.py \
    --split test \
    --sequence interlaken_00_d \
    --max-frames 100

# 4. Run tests
.venv/bin/python -m pytest
```

See [docs/setup.md](docs/setup.md) for detailed setup instructions and
[docs/architecture.md](docs/architecture.md) for design decisions.
