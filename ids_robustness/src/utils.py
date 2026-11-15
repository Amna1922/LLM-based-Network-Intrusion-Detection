"""Utility helpers for the IDS robustness project."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]


def resolve_path(path: str | Path) -> Path:
    """Resolve a repo-relative path to an absolute path."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return (ROOT / candidate).resolve()


def load_config(path: str | Path | None = None) -> dict:
    """Load the project configuration from YAML."""
    config_path = Path(path) if path is not None else ROOT / "config.yaml"
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def set_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> str:
    """Return the target compute device for training and inference."""
    return "cuda" if torch.cuda.is_available() else "cpu"
