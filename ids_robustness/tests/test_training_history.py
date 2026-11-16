"""Verify per-run loss CSV persistence and Figure 10 input paths."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.plots import plot_training_losses, training_loss_paths
from src.train_llm import save_training_artifacts


class _SavedArtifact:
    """Provide the minimal save interface used by training artifact output."""

    def save_pretrained(self, path) -> None:
        """Create the requested output directory."""
        Path(path).mkdir(parents=True, exist_ok=True)


class _Trainer:
    """Provide the trainer step count used by the training summary."""

    class _State:
        """Store a representative optimizer-step count."""

        global_step = 2

    state = _State()


def test_training_history_is_saved_in_adapter_run_folder(tmp_path) -> None:
    """Persist measured steps beside the adapter rather than beside the log."""
    run_dir = tmp_path / "llm_adapter"
    result = save_training_artifacts(
        model=_SavedArtifact(),
        tokenizer=_SavedArtifact(),
        trainer=_Trainer(),
        training_loss=0.25,
        device="cpu",
        quick=False,
        row_count=4,
        epoch_count=1,
        adapter_dir=str(run_dir),
        log_path=str(tmp_path / "training_log.txt"),
        training_history=[
            {"step": 1, "loss": 0.5},
            {"step": 2, "loss": 0.25},
        ],
    )

    history_path = run_dir / "training_loss.csv"
    history = pd.read_csv(history_path)
    assert result["training_history_path"] == str(history_path)
    assert history.to_dict(orient="records") == [
        {"step": 1, "loss": 0.5},
        {"step": 2, "loss": 0.25},
    ]


def test_figure_10_reads_each_seed_run_history(tmp_path) -> None:
    """Plot loss histories from the adapter folder for each saved seed."""
    results_dir = tmp_path / "results"
    baseline_dir = results_dir / "baseline_runs"
    baseline_dir.mkdir(parents=True)
    seeds = (42, 123, 2025)
    for seed in seeds:
        (baseline_dir / f"qwen_lora_seed_{seed}.json").write_text(
            json.dumps({"seed": seed}),
            encoding="utf-8",
        )
        run_dir = (
            results_dir / "llm_adapter"
            if seed == 42
            else results_dir / "seed_runs" / f"llm_seed_{seed}"
        )
        run_dir.mkdir(parents=True)
        pd.DataFrame(
            [{"step": 1, "loss": 0.5}, {"step": 2, "loss": 0.25}]
        ).to_csv(run_dir / "training_loss.csv", index=False)

    paths = training_loss_paths(results_dir)
    assert set(paths) == set(seeds)
    assert all(path.parent.is_dir() for path in paths.values())
    caption = plot_training_losses(results_dir, tmp_path / "figures")

    assert caption == "Training loss histories for 3 saved LoRA seeds."
    assert (tmp_path / "figures" / "fig10_training_loss.png").is_file()
    assert (tmp_path / "figures" / "fig10_training_loss.pdf").is_file()
