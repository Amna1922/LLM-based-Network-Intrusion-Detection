"""Tests for complete and incomplete ablation summary tables."""

import pandas as pd

from src.ablation import write_notes


def _ablation_rows(names: list[str]) -> pd.DataFrame:
    """Create one synthetic measurement row for each ablation name."""
    return pd.DataFrame(
        [
            {
                "ablation": name,
                "macro_f1": 0.8,
                "variant": "original",
                "detection_rate": 0.75,
            }
            for name in names
        ]
    )


def test_write_notes_accepts_saved_a4_zero_shot_name(tmp_path) -> None:
    """Summarize every saved ablation, including A4_zero_shot."""
    names = [
        "A0_main_reference",
        "A1",
        "A2",
        "A3",
        "A4_zero_shot",
        "A5_padding_only",
        "A5_delay_only",
        "A5_padding_and_delay",
    ]

    report = write_notes(_ablation_rows(names), tmp_path / "ablation_notes.md")

    assert "A4 (zero-shot Qwen reused from Task 2)" in report
    assert "test macro-F1=0.800000" in report
    assert "A5 (reused seed-42 results" in report


def test_write_notes_skips_missing_ablation_rows(tmp_path, capsys) -> None:
    """Report absent ablations clearly rather than indexing an empty frame."""
    names = [
        "A0_main_reference",
        "A1",
        "A3",
        "A5_padding_only",
    ]

    report = write_notes(_ablation_rows(names), tmp_path / "ablation_notes.md")
    output = capsys.readouterr().out

    assert "Skipping A2: no ablation result rows are available." in output
    assert "Skipping A4_zero_shot: no ablation result rows are available." in output
    assert "Skipping A5_delay_only: no ablation result rows are available." in output
    assert "A4 (zero-shot Qwen reused from Task 2) was not available" in report
    assert "delay only: not available" in report
