"""Generate CSV and LaTeX result tables from saved project outputs."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from src.utils import ROOT, load_config

TABLE_CAPTIONS = {
    "table_01_base_paper": "Reported settings and results from the supplied base paper.",
    "table_02_baselines": "Mean baseline test metrics over the saved runs.",
    "table_03_detection": "Per-variant detection rates and confidence intervals.",
    "table_04_ablation": "Ablation macro-F1 and perturbation detection measurements.",
    "table_05_seed_runs": "Per-seed Qwen LoRA test metrics and original-flow detection.",
    "table_06_reproduction": "Configured reproduction settings and recorded environment.",
    "table_07_base_comparison": "Descriptive comparison with the supplied base-paper values.",
}


def write_table(
    name: str,
    frame: pd.DataFrame,
    output_dir: Path,
    caption: str | None = None,
) -> tuple[Path, Path]:
    """Write a table as CSV and a standalone LaTeX tabular document."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / f"{name}.csv"
    tex_path = output_dir / f"{name}.tex"
    frame.to_csv(csv_path, index=False)
    latex = frame.to_latex(
        index=False,
        escape=True,
        caption=caption or TABLE_CAPTIONS.get(name, name.replace("_", " ")),
        label=f"tab:{name}",
        na_rep="N/A",
        float_format=lambda value: f"{value:.4f}",
    )
    tex_path.write_text(latex, encoding="utf-8")
    return csv_path, tex_path


def _paper_table(path: Path) -> pd.DataFrame:
    """Flatten the supplied paper's reported values and mark them with asterisks."""
    paper = yaml.safe_load(path.read_text(encoding="utf-8"))
    rows = []

    def add(section: str, item: str, value: object) -> None:
        """Append one asterisk-marked reported value."""
        rows.append(
            {"section": section, "item": item, "reported value": f"{value}*"}
        )

    add("paper", "title", paper["paper"]["title"])
    add("paper", "author", paper["paper"]["author"])
    add("paper", "identifier and date", paper["paper"]["id"])
    add("problem", "scope", paper["problem"])
    add("method", "approach", paper["method"])
    setup = paper["setup"]
    for field in (
        "models",
        "precision",
        "lora",
        "epochs",
        "effective_batch",
        "learning_rate",
        "seed",
        "inference",
        "certification",
    ):
        add("setup", field, setup[field])
    for dataset, values in paper["datasets"].items():
        for field, value in values.items():
            add(f"dataset {dataset}", field, value)
    for model, values in paper["results"][
        "CA_0p05_sigma_0p25_clean_vs_noise_aug_percent"
    ].items():
        add("certified accuracy percent", model, values)
    for attack, values in paper["results"]["transfer_attack_ASR_percent"].items():
        add("transfer ASR percent", attack, values)
    add(
        "clean accuracy",
        "CIC clean-trained",
        paper["results"]["clean_accuracy_CIC_clean_trained"],
    )
    add("duplicate audit", "reported duplicate result", paper["results"]["duplicate_audit"])
    return pd.DataFrame(rows)


def _baseline_table(results_dir: Path) -> pd.DataFrame:
    """Aggregate saved baseline metrics by model without mixing run-level data."""
    frame = pd.read_csv(results_dir / "baselines.csv")
    numeric = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "macro_f1",
        "false_positive_rate",
        "roc_auc",
        "training_time_seconds",
        "test_inference_seconds_per_1000_flows",
        "invalid_outputs",
    ]
    aggregated = frame.groupby("model", dropna=False).agg(
        n_runs=("seed", "nunique"),
        **{
            f"mean_{column}": (column, "mean")
            for column in numeric
        },
    )
    detection = pd.read_csv(results_dir / "baseline_detection.csv")
    asr = (
        detection[detection["variant"] != "original"]
        .groupby("model")["asr"]
        .mean()
        .rename("mean_conditional_asr")
    )
    return aggregated.join(asr).reset_index()


def _detection_table(results_dir: Path) -> pd.DataFrame:
    """Select seed-aggregated detection rates and their saved confidence limits."""
    stats = pd.read_csv(results_dir / "stats_summary.csv")
    return stats[stats["metric"] == "detection_rate"][
        [
            "model",
            "variant",
            "n_seeds",
            "mean",
            "std",
            "wilson_lower",
            "wilson_upper",
        ]
    ].rename(
        columns={
            "mean": "mean_detection_rate",
            "std": "seed_standard_deviation",
            "wilson_lower": "ci_95_lower",
            "wilson_upper": "ci_95_upper",
        }
    )


def _ablation_table(results_dir: Path) -> pd.DataFrame | None:
    """Load completed ablation measurements when the experiment has run."""
    path = results_dir / "ablation.csv"
    return pd.read_csv(path) if path.is_file() else None


def _seed_table(results_dir: Path) -> pd.DataFrame:
    """Collect per-seed Qwen LoRA metrics and original-flow detection."""
    baseline_path = results_dir / "baseline_runs"
    detection = pd.read_csv(results_dir / "baseline_detection.csv")
    original = detection[
        (detection["model"] == "Qwen2.5-0.5B-LoRA")
        & (detection["variant"] == "original")
    ].set_index("seed")
    rows = []
    for path in sorted(baseline_path.glob("qwen_lora_seed_*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        seed = int(result["seed"])
        metrics = result["test_metrics"]
        rows.append(
            {
                "seed": seed,
                "test_macro_f1": metrics["macro_f1"],
                "test_accuracy": metrics["accuracy"],
                "invalid_model_outputs": result.get("invalid_outputs", 0),
                "original_flow_detection_rate": (
                    float(original.loc[seed, "detection_rate"])
                    if seed in original.index
                    else None
                ),
                "training_time_seconds": result.get("training_time_seconds"),
            }
        )
    return pd.DataFrame(rows)


def _reproduction_table(results_dir: Path) -> pd.DataFrame:
    """Summarize configuration and recorded machine metadata without inference."""
    config = load_config()
    environment_path = results_dir / "environment.json"
    environment = (
        json.loads(environment_path.read_text(encoding="utf-8"))
        if environment_path.is_file()
        else {}
    )
    training_log = results_dir / "training_log.txt"
    log_values = {}
    if training_log.is_file():
        for line in training_log.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                log_values[key] = value
    fields = {
        "python": environment.get("python", "not recorded"),
        "torch": environment.get("torch", "not recorded"),
        "cuda runtime": environment.get("cuda_runtime", "not recorded"),
        "GPU names": ", ".join(environment.get("gpu_names", [])) or "not recorded",
        "compute device": log_values.get("device", "not recorded"),
        "PCAP-primary source": (
            pd.read_csv(results_dir / "data_summary.csv")["source"].iloc[0]
            if (results_dir / "data_summary.csv").is_file()
            else "not recorded"
        ),
        "seed": config["seed"],
        "max rows": config["max_rows"],
        "epochs": config["epochs"],
        "micro-batch": config["cuda_micro_batch_size"],
        "gradient accumulation": config["cuda_gradient_accumulation_steps"],
        "LoRA rank": config["lora_rank"],
        "LoRA alpha": config["lora_alpha"],
        "learning rate": config["learning_rate"],
        "training rows": log_values.get("train_rows", "not recorded"),
        "final training loss": log_values.get("train_loss", "not recorded"),
    }
    return pd.DataFrame(
        [{"setting": key, "value": value} for key, value in fields.items()]
    )


def build_tables(results_dir: Path, output_dir: Path) -> list[Path]:
    """Generate all tables supported by the result files currently available."""
    paper_path = ROOT / "base_paper_reported.yaml"
    tables: list[tuple[str, pd.DataFrame]] = [
        ("table_01_base_paper", _paper_table(paper_path)),
        ("table_02_baselines", _baseline_table(results_dir)),
        ("table_03_detection", _detection_table(results_dir)),
        ("table_05_seed_runs", _seed_table(results_dir)),
        ("table_06_reproduction", _reproduction_table(results_dir)),
    ]
    ablation = _ablation_table(results_dir)
    if ablation is not None:
        tables.insert(3, ("table_04_ablation", ablation))
    comparison_path = results_dir / "base_paper_comparison.csv"
    if comparison_path.is_file():
        tables.append(("table_07_base_comparison", pd.read_csv(comparison_path)))
    output_paths = []
    for name, frame in tables:
        csv_path, tex_path = write_table(
            name,
            frame,
            output_dir,
            TABLE_CAPTIONS[name],
        )
        output_paths.extend((csv_path, tex_path))
        print(f"Wrote {csv_path} and {tex_path} ({len(frame)} rows)")
    if ablation is None:
        print("Skipped table 4: results/ablation.csv does not exist.")
    if not comparison_path.is_file():
        print("Skipped table 7: results/base_paper_comparison.csv does not exist.")
    return output_paths


def main() -> None:
    """Generate result tables in CSV and LaTeX formats."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    build_tables(results_dir, results_dir / "tables")


if __name__ == "__main__":
    main()
