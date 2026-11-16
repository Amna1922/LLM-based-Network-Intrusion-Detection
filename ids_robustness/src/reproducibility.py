"""Save run environment metadata and split membership for reproducibility."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch

from src.utils import ROOT

PACKAGE_NAMES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "scapy",
    "PyYAML",
    "torch",
    "transformers",
    "peft",
    "accelerate",
    "matplotlib",
)


def collect_environment(peak_gpu_memory_bytes: int | None = None) -> dict[str, object]:
    """Collect Python, package, CUDA, hardware, and Git version details."""
    packages = {}
    for package in PACKAGE_NAMES:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = "not installed"
    gpu_names = (
        [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if torch.cuda.is_available()
        else []
    )
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_names": gpu_names,
        "peak_gpu_memory_bytes": peak_gpu_memory_bytes,
        "packages": packages,
        "git_commit": commit,
    }


def save_environment(
    path: str | Path,
    peak_gpu_memory_bytes: int | None = None,
) -> dict[str, object]:
    """Write environment metadata as JSON and return the saved record."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = collect_environment(peak_gpu_memory_bytes)
    output_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
