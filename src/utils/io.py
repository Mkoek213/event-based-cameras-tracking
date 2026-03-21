"""I/O helpers: configuration loading and result saving."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict:
    """Load a YAML configuration file.

    Args:
        path: Path to the ``.yaml`` file.

    Returns:
        Configuration dictionary.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML is required. pip install pyyaml")

    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_results(results: Any, path: str | Path) -> None:
    """Serialise *results* to a JSON file.

    Args:
        results: JSON-serialisable object (dict, list, …).
        path: Destination file path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
