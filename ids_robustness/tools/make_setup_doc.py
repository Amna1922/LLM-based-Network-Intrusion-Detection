"""Generate reproducibility instructions from project configuration and results."""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils import load_config


def _measured_training_summary(results_dir: Path) -> str:
    """Summarize measured LoRA durations without estimating missing runs."""
    durations = []
    for path in (results_dir / "baseline_runs").glob("qwen_lora_seed_*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        seconds = result.get("training_time_seconds")
        if seconds is not None:
            durations.append(float(seconds))
    if not durations:
        return "No completed LoRA training durations are recorded yet."
    return (
        f"LoRA runs with recorded training durations: {len(durations)}; measured training duration "
        f"median={statistics.median(durations):.1f} seconds, "
        f"minimum={min(durations):.1f} seconds, maximum={max(durations):.1f} seconds. "
        "These measurements do not include the pending ablation runs."
    )


def build_setup_doc() -> str:
    """Build Windows setup, execution, and provenance instructions."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    environment_path = results_dir / "environment.json"
    if environment_path.is_file():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment_lines = [
            f"- Python: `{environment.get('python', 'not recorded')}`",
            f"- PyTorch: `{environment.get('torch', 'not recorded')}`",
            f"- CUDA runtime: `{environment.get('cuda_runtime', 'not recorded')}`",
            f"- CUDA available during preparation: `{environment.get('cuda_available', 'not recorded')}`",
            f"- GPUs: `{', '.join(environment.get('gpu_names', [])) or 'none recorded'}`",
            f"- Git revision: `{environment.get('git_commit', 'not recorded')}`",
        ]
    else:
        environment_lines = ["Environment metadata has not been recorded; run data preparation."]

    return "\n".join(
        [
            "# Reproduction setup",
            "",
            "Run these commands from the `ids_robustness` project directory in Windows PowerShell.",
            "",
            "## Create and activate the environment",
            "",
            "```powershell",
            "py -3.11 -m venv .venv",
            ".\\.venv\\Scripts\\Activate.ps1",
            "python -m pip install --upgrade pip",
            "python -m pip install -r requirements.txt",
            "```",
            "",
            "In Command Prompt, activate the same environment with:",
            "",
            "```cmd",
            ".venv\\Scripts\\activate.bat",
            "```",
            "",
            "## Confirm CUDA",
            "",
            "```powershell",
            "nvidia-smi",
            "python -c \"import torch; print('CUDA available:', torch.cuda.is_available()); print('PyTorch CUDA:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')\"",
            "```",
            "",
            "The CUDA-enabled PyTorch wheel is required for GPU training. The training code selects CUDA when PyTorch reports it available; otherwise it runs on CPU. No separate CUDA toolkit is used by the Python code.",
            "",
            "## Run and resume the experiment pipeline",
            "",
            "```powershell",
            "python -m pytest -q",
            "python -m src.run_experiments",
            "```",
            "",
            "The runner executes preparation, delay audit, baselines and multi-seed statistics, Task 4 ablations, Tasks 5-6 analyses, plots/tables, the base-paper comparison, and this setup document in order. Existing complete stage outputs are reused. Task 4 trains three separate LoRA ablations if their results are missing and can take several hours on a laptop GPU. It was not run while preparing these docs.",
            "",
            "Resume from a particular stage:",
            "",
            "```powershell",
            "python -m src.run_experiments --start-at task4-ablation",
            "python -m src.run_experiments --start-at task7-plots-tables",
            "```",
            "",
            "To run the main evaluation separately, use `python -m src.run_eval --quick` for the smoke test or `python -m src.run_eval` for the configured full evaluation.",
            "",
            "## Inputs and outputs",
            "",
            f"- CSV baseline: `{config['csv_path']}`",
            f"- Attack capture: `{config['pcap_path']}`",
            f"- Benign captures: `{config['benign_pcap_dir']}`",
            f"- Result directory: `{config['results_dir']}`",
            "- Model files may be downloaded from Hugging Face when absent. After model preparation, evaluation reads local PCAPs and writes local outputs; it does not scan or send network traffic.",
            "",
            "## Recorded environment",
            "",
            *environment_lines,
            "",
            _measured_training_summary(results_dir),
            "",
            "The project seed and configured model/training settings are in `config.yaml`. `results/splits/` records split membership when data preparation has completed. `results/environment.json` records package and hardware metadata for that run.",
            "Each new training run writes `training_loss.csv` inside its adapter run folder. Fig. 10 reads those per-seed histories; older runs without per-step logs remain unavailable.",
            "",
        ]
    )


def main() -> None:
    """Write the generated reproduction setup document."""
    output = ROOT / "docs" / "SETUP.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_setup_doc(), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
