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

## Running Training

```bash
python scripts/train.py --config-name base_config
```

Override any config value on the command line:

```bash
python scripts/train.py training.epochs=200 data.batch_size=32
```

## Running Evaluation

```bash
python scripts/evaluate.py --config-name base_config
```

## Running Inference

```bash
python scripts/inference.py --input data/raw/sample.dat --output results/
```

## Running Tests

```bash
pytest
```
