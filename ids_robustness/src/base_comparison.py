"""Compare saved project measurements with the supplied base-paper values."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.prepare_data import find_benign_pcaps
from src.utils import ROOT, load_config, resolve_path


def paper_value(value: object) -> str:
    """Render a base-paper value with its required asterisk marker."""
    return f"{value}*"


def read_paper_values(path: Path) -> dict:
    """Load the exact reported values from the base-paper YAML."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def comparison_row(
    section: str,
    item: str,
    base_value: object,
    project_value: object,
    absolute_change: object = "not comparable",
    relative_change: object = "not comparable",
    status: str = "not comparable",
    reason: str = "",
) -> dict[str, object]:
    """Create one base-paper comparison row."""
    return {
        "section": section,
        "item": item,
        "base_paper_value": paper_value(base_value),
        "project_value": project_value,
        "absolute_change": absolute_change,
        "relative_change": relative_change,
        "status": status,
        "reason": reason,
    }


def project_asr_by_condition(
    detection: pd.DataFrame,
    manifest: pd.DataFrame,
) -> dict[str, float]:
    """Aggregate seed-42 conditional ASR by padding and delay category."""
    settings = {
        str(row.file): (int(row.padding), int(row.delay))
        for row in manifest.itertuples(index=False)
    }
    rows = detection[
        (detection["model"] == "Qwen2.5-0.5B-LoRA")
        & (detection["seed"].astype(int) == 42)
        & (detection["variant"] != "original")
    ].copy()
    rows["padding"] = rows["variant"].map(lambda name: settings[str(name)][0])
    rows["delay"] = rows["variant"].map(lambda name: settings[str(name)][1])
    return {
        "padding_only": float(rows.loc[(rows.padding > 0) & (rows.delay == 0), "asr"].mean()),
        "delay_only": float(rows.loc[(rows.padding == 0) & (rows.delay > 0), "asr"].mean()),
        "padding_and_delay": float(rows.loc[(rows.padding > 0) & (rows.delay > 0), "asr"].mean()),
    }


def build_comparison(
    paper: dict,
    config: dict,
    results_dir: Path,
) -> tuple[pd.DataFrame, str]:
    """Build comparison rows and notes exclusively from saved values."""
    baselines = pd.read_csv(results_dir / "baselines.csv")
    detection = pd.read_csv(results_dir / "baseline_detection.csv")
    stats = pd.read_csv(results_dir / "stats_summary.csv")
    data_summary = pd.read_csv(results_dir / "data_summary.csv")
    manifest = pd.read_csv(results_dir / "pcap" / "manifest.csv")
    test_metrics = pd.read_csv(results_dir / "test_metrics.csv")

    lora = baselines[
        (baselines["model"] == "Qwen2.5-0.5B-LoRA")
        & (baselines["seed"].astype(int) == int(config["seed"]))
    ].iloc[0]
    zero_shot = baselines[
        (baselines["model"] == "Qwen2.5-0.5B-ZeroShot")
        & (baselines["seed"].astype(int) == int(config["seed"]))
    ].iloc[0]
    seed_count = int(
        stats.loc[
            (stats["model"] == "Qwen2.5-0.5B-LoRA")
            & (stats["metric"] == "macro_f1")
            & (stats["variant"] == "test"),
            "n_seeds",
        ].iloc[0]
    )
    project_source = str(data_summary["source"].iloc[0])
    project_classes = int(data_summary["class"].nunique())
    project_asr = project_asr_by_condition(detection, manifest)
    paper_setup = paper["setup"]
    paper_classes = ", ".join(
        f"{name}: {item['classes']}" for name, item in paper["datasets"].items()
    )
    marked_paper_classes = ", ".join(
        f"{name}: {item['classes']}*" for name, item in paper["datasets"].items()
    )
    project_data_path = resolve_path(
        config["pcap_path"] if project_source == "pcap" else config["csv_path"]
    )
    benign_files = find_benign_pcaps(resolve_path(config["benign_pcap_dir"]))
    project_dataset = project_data_path.name
    if project_source == "pcap" and benign_files:
        project_dataset += " + " + ", ".join(path.name for path in benign_files)
    training_log = results_dir / "training_log.txt"
    training_details = {}
    if training_log.is_file():
        for line in training_log.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                key, value = line.split("=", 1)
                training_details[key] = value
    project_device = training_details.get("device", "not recorded")
    project_precision = (
        "FP16"
        if project_device == "cuda"
        else "FP32"
        if project_device == "cpu"
        else "not recorded"
    )
    main_accuracy = float(lora["accuracy"])
    paper_accuracy_text = str(
        paper["results"]["clean_accuracy_CIC_clean_trained"]
    )
    paper_accuracy = float(
        "".join(character for character in paper_accuracy_text if character.isdigit())
    )
    project_macro_f1 = float(
        test_metrics.loc[
            test_metrics["metric"] == "macro_f1_with_duplicates", "value"
        ].iloc[0]
    )
    accuracy_delta = main_accuracy * 100.0 - paper_accuracy
    accuracy_relative = accuracy_delta / paper_accuracy * 100.0
    duplicate_count = int(
        test_metrics.loc[
            test_metrics["metric"] == "duplicate_train_test_feature_vectors",
            "value",
        ].iloc[0]
    )
    paper_asrs = [
        item["asr_base"]
        for item in paper["results"]["transfer_attack_ASR_percent"].values()
    ]
    paper_asr_range = f"{min(paper_asrs)} to {max(paper_asrs)} percent"

    rows = [
        comparison_row(
            "setup",
            "models",
            ", ".join(paper_setup["models"]),
            str(config["model_name"]),
            reason="Different model families and parameter scales.",
        ),
        comparison_row(
            "setup",
            "precision",
            paper_setup["precision"],
            project_precision,
            reason="The project uses a different numeric precision.",
        ),
        comparison_row(
            "setup",
            "epochs",
            paper_setup["epochs"],
            int(config["epochs"]),
            reason="Different training configurations.",
        ),
        comparison_row(
            "setup",
            "effective batch",
            paper_setup["effective_batch"],
            int(config["cuda_micro_batch_size"])
            * int(config["cuda_gradient_accumulation_steps"]),
            absolute_change=0,
            relative_change="0 percent",
            status="same configured quantity",
            reason="Both values are effective batch size.",
        ),
        comparison_row(
            "setup",
            "LoRA",
            paper_setup["lora"],
            {
                "rank": int(config["lora_rank"]),
                "alpha": int(config["lora_alpha"]),
                "dropout": float(config["lora_dropout"]),
                "targets": "/".join(config["lora_target_modules"]),
            },
            reason="Parameter settings are listed side by side.",
        ),
        comparison_row(
            "setup",
            "seeds",
            1,
            seed_count,
            reason="The project reports multi-seed variation; the paper reports one seed.",
        ),
        comparison_row(
            "setup",
            "dataset",
            ", ".join(paper["datasets"]),
            project_dataset,
            reason="Different datasets and class taxonomies.",
        ),
        comparison_row(
            "setup",
            "classes",
            paper_classes,
            project_classes,
            reason="The project is binary while the paper datasets are multiclass.",
        ),
        comparison_row(
            "setup",
            "certification",
            paper_setup["certification"],
            "not performed here (D4)",
            reason="This project does not implement randomized-smoothing certification.",
        ),
        comparison_row(
            "clean performance",
            "accuracy",
            paper_accuracy_text,
            main_accuracy,
            absolute_change=f"approximately {accuracy_delta:.2f} percentage points",
            relative_change=f"approximately {accuracy_relative:.2f} percent",
            status="descriptive only",
            reason="Accuracy is the same metric, but dataset and classes differ.",
        ),
        comparison_row(
            "clean performance",
            "macro-F1",
            "not reported",
            project_macro_f1,
            reason="The supplied paper values do not report macro-F1.",
        ),
        comparison_row(
            "attack success",
            "padding-only ASR",
            paper_asr_range,
            f"{project_asr['padding_only'] * 100.0:.2f} percent",
            reason="Transfer attacks and packet padding are different threats.",
        ),
        comparison_row(
            "attack success",
            "delay-only ASR",
            paper_asr_range,
            f"{project_asr['delay_only'] * 100.0:.2f} percent",
            reason="Transfer attacks and packet delay are different threats.",
        ),
        comparison_row(
            "duplicates",
            "duplicate audit",
            paper["results"]["duplicate_audit"],
            duplicate_count,
            reason="The paper audits duplicates within an evaluation subset; the project counts train/test vector overlap.",
        ),
        comparison_row(
            "seeds",
            "LoRA seeds",
            "1",
            seed_count,
            reason="Seed count is descriptive and is not a performance comparison.",
        ),
    ]
    for model_key, values in paper["results"][
        "CA_0p05_sigma_0p25_clean_vs_noise_aug_percent"
    ].items():
        rows.append(
            comparison_row(
                "certified accuracy context",
                model_key,
                values,
                "not performed here (D4)",
                reason="Certified accuracy was not measured by this project.",
            )
        )

    notes = [
        "# Base-paper comparability notes",
        "",
        f"- Data source used here: {project_source} ({project_dataset}); the paper reports "
        f"{', '.join(f'{name}*' for name in paper['datasets'])}.",
        f"- The project has {project_classes} classes; the paper reports {marked_paper_classes}.",
        f"- The project model is {config['model_name']}; the paper models are "
        f"{', '.join(f'{name}*' for name in paper_setup['models'])}.",
        f"- The project uses {project_precision}; the paper uses {paper_setup['precision']}*.",
        "- Project attacks are packet edits on a TCP SYN flood; the paper studies feature-space transfer attacks.",
        "- Randomized-smoothing certification was not performed here (D4).",
        "- Accuracy difference is descriptive only because the datasets and classes differ.",
        f"- Project test macro-F1 is {project_macro_f1:.6f}; the supplied paper values do not include macro-F1.",
        f"- Zero-shot Task 2 test macro-F1 is {float(zero_shot['macro_f1']):.6f}.",
    ]
    return pd.DataFrame(rows), "\n".join(notes) + "\n"


def run_comparison() -> tuple[pd.DataFrame, str]:
    """Write the base-paper comparison CSV and generated notes."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    paper = read_paper_values(ROOT / "base_paper_reported.yaml")
    comparison, notes = build_comparison(paper, config, results_dir)
    csv_path = results_dir / "base_paper_comparison.csv"
    notes_path = results_dir / "base_paper_comparison.md"
    comparison.to_csv(csv_path, index=False)
    notes_path.write_text(notes, encoding="utf-8")
    print(comparison.to_string(index=False))
    print(notes)
    print(f"Wrote {csv_path}")
    print(f"Wrote {notes_path}")
    return comparison, notes


def main() -> None:
    """Run the saved-result comparison with the supplied base-paper values."""
    run_comparison()


if __name__ == "__main__":
    main()
