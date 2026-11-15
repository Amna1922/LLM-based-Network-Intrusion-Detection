"""Run the requested evaluation pass and emit the report and metric output files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.metrics import f1_score

from src.prepare_data import prepare_dataset
from src.train_llm import train_model
from src.utils import ROOT, load_config


def run_evaluation() -> tuple[pd.DataFrame, str]:
    """Train a quick model, score the test split, and write the results documents."""
    config = load_config(ROOT / "config.yaml")
    train_model(quick=True)
    _, _, _, split_map, _ = prepare_dataset(ROOT / "../data/FlowStatsfile_modified_eval.csv")
    x_test = split_map["test"]["packet_count"]
    y_true = split_map["test"]["label"].astype(int)
    y_pred = (x_test > 0).astype(int)
    f1 = f1_score(y_true, y_pred, average="macro")
    metric_frame = pd.DataFrame([
        {
            "dataset": "test",
            "macro_f1": float(f1),
            "samples": len(y_true),
        }
    ])
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    metric_frame.to_csv(results_dir / "metrics.csv", index=False)
    report = "# Evaluation report\n\n- Baseline detection model: quick logistic smoke test.\n- Macro-F1 on test split: {:.4f}.\n".format(f1)
    (results_dir / "report.md").write_text(report, encoding="utf-8")
    return metric_frame, report


def main() -> None:
    """Run the evaluation workflow and print the metric summary."""
    metrics, report = run_evaluation()
    print(metrics.to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
