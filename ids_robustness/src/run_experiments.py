"""Run or resume the ordered project analyses and reporting stages."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.utils import ROOT, load_config

STAGES = (
    "prepare-data",
    "task1-delay-audit",
    "tasks2-3-baselines",
    "task4-ablation",
    "task5-feature-space",
    "task6-failure-analysis",
    "task7-plots-tables",
    "task8-base-comparison",
    "task9-setup-doc",
)


def _stage_outputs(stage: str, results_dir: Path) -> list[Path]:
    """Return the completion markers for one pipeline stage."""
    outputs = {
        "prepare-data": [
            results_dir / "data_summary.csv",
            results_dir / "environment.json",
            results_dir / "splits" / "train.csv",
            results_dir / "splits" / "validation.csv",
            results_dir / "splits" / "test.csv",
        ],
        "task1-delay-audit": [results_dir / "delay_audit.md"],
        "tasks2-3-baselines": [
            results_dir / "baselines.csv",
            results_dir / "baseline_detection.csv",
            results_dir / "stats_summary.csv",
        ],
        "task4-ablation": [
            results_dir / "ablation.csv",
            results_dir / "ablation_notes.md",
        ],
        "task5-feature-space": [results_dir / "feature_vs_problem_space.csv"],
        "task6-failure-analysis": [
            results_dir / "failure_breakdown.csv",
            results_dir / "qualitative_examples.md",
        ],
        "task7-plots-tables": [
            results_dir / "figures" / "captions.md",
            results_dir / "figures" / "fig01_pipeline_architecture.png",
            results_dir / "figures" / "fig02_baseline_comparison.png",
            results_dir / "figures" / "fig03_classical_roc.png",
            results_dir / "figures" / "fig04_detection_confidence.png",
            results_dir / "figures" / "fig05_delay_mode_heatmaps.png",
            results_dir / "figures" / "fig07_ablation.png",
            results_dir / "figures" / "fig08_feature_vs_problem_space.png",
            results_dir / "figures" / "fig09_llm_confusion_matrix.png",
            results_dir / "tables" / "table_02_baselines.csv",
            results_dir / "tables" / "table_04_ablation.csv",
        ],
        "task8-base-comparison": [
            results_dir / "base_paper_comparison.csv",
            results_dir / "tables" / "table_07_base_comparison.csv",
        ],
        "task9-setup-doc": [ROOT / "docs" / "SETUP.md"],
    }
    return outputs[stage]


def _is_complete(stage: str, results_dir: Path) -> bool:
    """Check that every output marker for a stage exists."""
    return all(path.is_file() for path in _stage_outputs(stage, results_dir))


def _run_stage(stage: str, results_dir: Path) -> None:
    """Execute one stage by calling its existing project implementation."""
    if stage == "prepare-data":
        from src.prepare_data import prepare_dataset

        prepare_dataset()
    elif stage == "task1-delay-audit":
        from src.delay_audit import audit_delay_modes

        audit_delay_modes()
    elif stage == "tasks2-3-baselines":
        from src.baselines import run_baselines

        run_baselines()
    elif stage == "task4-ablation":
        from src.ablation import run_ablation

        run_ablation()
    elif stage == "task5-feature-space":
        from src.feature_space import run_feature_space

        run_feature_space()
    elif stage == "task6-failure-analysis":
        from src.analysis import run_analysis

        run_analysis()
    elif stage == "task7-plots-tables":
        from src.make_tables import build_tables
        from src.plots import run_plots

        run_plots()
        build_tables(results_dir, results_dir / "tables")
    elif stage == "task8-base-comparison":
        from src.base_comparison import run_comparison
        from src.make_tables import build_tables

        run_comparison()
        build_tables(results_dir, results_dir / "tables")
    elif stage == "task9-setup-doc":
        from tools.make_setup_doc import build_setup_doc

        output = ROOT / "docs" / "SETUP.md"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(build_setup_doc(), encoding="utf-8")
        print(f"Wrote {output}")
    else:
        raise ValueError(f"Unknown experiment stage: {stage}")


def run_experiments(
    start_at: str = STAGES[0],
    stop_after: str = STAGES[-1],
) -> None:
    """Run selected ordered stages, resuming from their saved completion files."""
    if start_at not in STAGES or stop_after not in STAGES:
        raise ValueError("Start and stop stages must be members of STAGES.")
    first = STAGES.index(start_at)
    last = STAGES.index(stop_after)
    if first > last:
        raise ValueError("The start stage must not follow the stop stage.")
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    for stage in STAGES[first : last + 1]:
        if _is_complete(stage, results_dir):
            print(f"Resume: skip {stage}; completion outputs are present.")
            continue
        print(f"Running {stage}...")
        _run_stage(stage, results_dir)
        if not _is_complete(stage, results_dir):
            missing = [
                str(path)
                for path in _stage_outputs(stage, results_dir)
                if not path.is_file()
            ]
            raise RuntimeError(
                f"Stage {stage} did not create its required outputs: {missing}"
            )
        print(f"Completed {stage}.")


def parse_args() -> argparse.Namespace:
    """Parse start and stop stages for a full or partial pipeline run."""
    parser = argparse.ArgumentParser(
        description="Run or resume the offline IDS robustness experiment stages."
    )
    parser.add_argument("--start-at", choices=STAGES, default=STAGES[0])
    parser.add_argument("--stop-after", choices=STAGES, default=STAGES[-1])
    return parser.parse_args()


def main() -> None:
    """Run the selected experiment stages in the documented order."""
    args = parse_args()
    run_experiments(args.start_at, args.stop_after)


if __name__ == "__main__":
    main()
