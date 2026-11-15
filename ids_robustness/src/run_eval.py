"""Run preparation, training, PCAP perturbation, validation, prediction, and reporting."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.pcap_features import extract_pcap_flows
from src.perturb_pcap import generate_variants, write_session_subset
from src.predict_llm import load_finetuned_model, predict_unique_prompts
from src.prepare_data import (
    FEATURES,
    build_text_rows,
    prepare_dataset,
    transform_features,
)
from src.train_llm import train_model
from src.utils import ROOT, load_config, pick_device, resolve_path
from src.validate import validate_variants


def scale_flow_frame(frame: pd.DataFrame, scaler: object) -> pd.DataFrame:
    """Apply the saved log transform and training scaler to raw flow features."""
    transformed = transform_features(frame)
    scaled = pd.DataFrame(
        scaler.transform(transformed),
        columns=FEATURES,
        index=frame.index,
    )
    result = pd.concat([frame.drop(columns=FEATURES), scaled], axis=1)
    return result[[*frame.drop(columns=FEATURES).columns, *FEATURES]]


def sample_rows(frame: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    """Select a seeded random sample no larger than the requested row limit."""
    if len(frame) <= limit:
        return frame.copy()
    return frame.sample(n=limit, random_state=seed).copy()


def select_attack_flows(
    attacks: pd.DataFrame,
    prepared: tuple,
    source: str,
    attack_path: Path,
    limit: int,
    seed: int,
) -> pd.DataFrame:
    """Restrict PCAP-primary runs to test attack sessions and sample test flows."""
    if source == "pcap":
        splits = prepared[3]
        sessions = set(
            splits["test"].loc[splits["test"]["label"] == 1, "session_id"]
        )
        prefix = attack_path.name + "|"
        raw_sessions = {session[len(prefix) :] for session in sessions if session.startswith(prefix)}
        attacks = attacks[attacks["session_id"].isin(raw_sessions)].copy()
    if attacks.empty:
        raise ValueError("No test-split attack flows are available for evaluation.")
    return sample_rows(attacks, limit, seed)


def detection_summary(
    predictions: list[str | None],
    seed: int,
    resamples: int,
    confidence: float,
) -> dict[str, float | int]:
    """Calculate detection rate and a percentile bootstrap confidence interval."""
    detected = np.array([prediction == "DDoS" for prediction in predictions], dtype=float)
    rate = float(detected.mean()) if len(detected) else 0.0
    rng = np.random.default_rng(seed)
    bootstrap_rates = [
        float(rng.choice(detected, size=len(detected), replace=True).mean())
        for _ in range(resamples)
    ] if len(detected) else [0.0]
    alpha = (1.0 - confidence) / 2.0
    return {
        "samples": len(detected),
        "detection_rate": rate,
        "ci_lower": float(np.quantile(bootstrap_rates, alpha)),
        "ci_upper": float(np.quantile(bootstrap_rates, 1.0 - alpha)),
    }


def metric_row(
    metric: str,
    variant: str,
    values: dict[str, float | int],
    baseline_rate: float | None = None,
) -> dict[str, object]:
    """Create one standardized result row for the metrics CSV."""
    rate = values.get("detection_rate")
    return {
        "metric": metric,
        "variant": variant,
        "samples": values.get("samples", ""),
        "value": values.get("detection_rate", values.get("macro_f1", "")),
        "ci_lower": values.get("ci_lower", ""),
        "ci_upper": values.get("ci_upper", ""),
        "change_from_original": (
            float(rate) - baseline_rate
            if baseline_rate is not None and rate is not None
            else ""
        ),
    }


def load_training_summary(path: Path) -> dict[str, str]:
    """Read key-value training metadata from the training log."""
    if not path.is_file():
        return {}
    summary = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            summary[key] = value
    return summary


def package_versions() -> dict[str, str]:
    """Read installed versions for the report."""
    packages = ("torch", "transformers", "peft", "scapy", "pandas", "scikit-learn")
    return {package: importlib.metadata.version(package) for package in packages}


def format_table(frame: pd.DataFrame) -> str:
    """Format a result frame as a Markdown table."""
    if frame.empty:
        return "_No rows._"
    columns = list(frame.columns)
    rows = ["| " + " | ".join(columns) + " |"]
    rows.append("| " + " | ".join("---" for _ in columns) + " |")
    for values in frame.itertuples(index=False, name=None):
        rendered = [str(value) for value in values]
        rows.append("| " + " | ".join(rendered) + " |")
    return "\n".join(rows)


def write_report(
    config: dict,
    device: str,
    source: str,
    training: dict[str, str],
    test_metrics: pd.DataFrame,
    detection_metrics: pd.DataFrame,
    functionality: pd.DataFrame,
    domain_gap: pd.DataFrame,
    shortcut_audit: pd.DataFrame,
    duplicate_vectors: int,
    invalid_outputs: int,
    warning: str | None,
    report_path: Path,
) -> str:
    """Write all measured evaluation results and limitations to Markdown."""
    lines = ["# Evaluation report", ""]
    passing_share = (
        float(functionality["pass_all"].astype(bool).mean())
        if not functionality.empty
        else 0.0
    )
    if warning:
        lines.extend([f"> **{warning}**", ""])
    lines.extend(
        [
            "## Run",
            "",
            f"- Device: {device}",
            f"- Data source: {source}",
            f"- Training rows: {training.get('train_rows', 'unknown')}",
            f"- Epochs: {training.get('epochs', 'unknown')}",
            f"- Final training loss: {training.get('train_loss', 'unknown')}",
            f"- Invalid model outputs: {invalid_outputs}",
            f"- Versions: {package_versions()}",
            "",
            "## Test-split classification",
            "",
            format_table(test_metrics),
            "",
            f"Train/test duplicate feature vectors: {duplicate_vectors}",
            "",
            "## PCAP attack-flow detection",
            "",
            format_table(detection_metrics),
            "",
            "## Functionality validation",
            "",
            f"Variants passing all checks: {passing_share:.3f}",
            "",
            format_table(functionality),
            "",
            "## CSV and PCAP domain gap",
            "",
            format_table(domain_gap),
            "",
            "## Shortcut audit",
            "",
            format_table(shortcut_audit),
            "",
            "## Limitations",
            "",
            "- The evaluated attack type is TCP SYN flood only.",
            "- The detector is Qwen2.5-0.5B-Instruct with LoRA.",
            f"- The configured seed is {config['seed']}; no multi-seed confidence is claimed.",
            f"- Training data source: {source}.",
        ]
    )
    if warning:
        lines.append(f"- Warning fired: {warning}.")
    report = "\n".join(lines) + "\n"
    report_path.write_text(report, encoding="utf-8")
    return report


def run_evaluation(quick: bool = False, retrain: bool = False) -> tuple[pd.DataFrame, str]:
    """Execute the full evaluation pipeline and write its result files."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    print("Step 1/7: prepare data")
    prepared = prepare_dataset()
    _, _, domain_gap, splits, audit = prepared
    source = str(audit.attrs["source"])

    adapter_dir = results_dir / "llm_adapter"
    training_log = results_dir / "training_log.txt"
    print("Step 2/7: train or reuse LoRA adapter")
    if retrain or not (adapter_dir / "adapter_config.json").is_file():
        train_model(quick=quick, prepared_data=prepared)
    training = load_training_summary(training_log)

    attack_path = resolve_path(config["pcap_path"])
    variants_dir = results_dir / "pcap"
    evaluation_attack_path = attack_path
    if source == "pcap":
        test_sessions = set(
            splits["test"].loc[splits["test"]["label"] == 1, "session_id"]
        )
        prefix = attack_path.name + "|"
        selected_sessions = {
            session[len(prefix) :]
            for session in test_sessions
            if session.startswith(prefix)
        }
        evaluation_attack_path = variants_dir / "attack_test_sessions.pcap"
        write_session_subset(attack_path, evaluation_attack_path, selected_sessions)
    pairs = config["quick_variants"] if quick else None
    print("Step 3/7: perturb attack PCAP")
    manifest = generate_variants(
        evaluation_attack_path,
        variants_dir,
        config["padding_values"],
        config["delay_values_ms"],
        variant_pairs=pairs,
        delay_mode=str(config.get("delay_mode", "client_and_reply")),
    )
    manifest_path = variants_dir / "manifest.csv"

    print("Step 4/7: validate variants")
    functionality = validate_variants(
        evaluation_attack_path,
        manifest_path,
        variants_dir,
        results_dir / "functionality.csv",
        float(config["min_peak_rate_ratio"]),
        float(config["peak_window_ms"]) / 1000.0,
    )
    functionality_columns = [
        "file",
        "padding",
        "delay",
        "invariant_pass",
        "invariant_mismatches",
        "same_syn_count",
        "same_syn_5_tuples",
        "same_rst_ack_count",
        "peak_rate_pass",
        "checksums_valid",
        "pass_all",
    ]
    functionality.to_csv(results_dir / "functionality.csv", index=False)

    print("Step 5/7: extract and sample flows")
    original_flows = extract_pcap_flows(
        evaluation_attack_path,
        float(config["duration_floor"]),
    )
    original_flows["label"] = 1
    selected = select_attack_flows(
        original_flows,
        prepared,
        source,
        attack_path,
        int(config["quick_eval_flows"] if quick else config["max_eval_flows"]),
        int(config["seed"]),
    )
    selected_ids = set(selected["flow_id"])
    flow_variants = {}
    for row in manifest:
        feature_table = extract_pcap_flows(
            variants_dir / str(row["file"]),
            float(config["duration_floor"]),
        )
        feature_table["label"] = 1
        flow_variants[str(row["file"])] = feature_table[
            feature_table["flow_id"].isin(selected_ids)
        ].copy()

    print("Step 6/7: predict unique prompts")
    test_rows = splits["test"]
    if quick:
        test_rows = sample_rows(
            test_rows,
            int(config["quick_eval_flows"]),
            int(config["seed"]),
        )
    model, tokenizer, device = load_finetuned_model()
    scaler = joblib.load(results_dir / "feature_scaler.joblib")
    test_texts = build_text_rows(test_rows)["text"].tolist()
    pcap_tables = {"original": selected}
    pcap_tables.update(flow_variants)
    pcap_prompts = {}
    for name, flows in pcap_tables.items():
        scaled = scale_flow_frame(flows, scaler)
        pcap_prompts[name] = build_text_rows(scaled)["text"].tolist()
    all_prompts = test_texts + [
        prompt for prompts in pcap_prompts.values() for prompt in prompts
    ]
    names = predict_unique_prompts(
        all_prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(config["prediction_batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        max_input_length=int(config["max_seq_length"]),
    )
    test_predictions = names[: len(test_texts)]
    invalid_outputs = sum(prediction is None for prediction in names)
    test_labels = test_rows["label"].astype(int).to_numpy()
    class_ids = {"Benign": 0, "DDoS": 1}
    predicted_ids = np.array([class_ids.get(value, -1) for value in test_predictions])
    all_f1 = float(
        f1_score(test_labels, predicted_ids, labels=[0, 1], average="macro")
    )
    unique_indices = test_rows.assign(prompt=test_texts).drop_duplicates("prompt").index
    unique_labels = test_rows.loc[unique_indices, "label"].astype(int).to_numpy()
    prompt_to_prediction = dict(zip(test_texts, test_predictions))
    unique_prompts = test_rows.loc[unique_indices].pipe(build_text_rows)["text"].tolist()
    unique_predictions = np.array(
        [class_ids.get(prompt_to_prediction[prompt], -1) for prompt in unique_prompts]
    )
    unique_f1 = float(
        f1_score(unique_labels, unique_predictions, labels=[0, 1], average="macro")
    )
    test_metrics = pd.DataFrame(
        [
            {"metric": "macro_f1_with_duplicates", "value": all_f1, "samples": len(test_rows)},
            {"metric": "macro_f1_without_duplicates", "value": unique_f1, "samples": len(unique_labels)},
            {
                "metric": "duplicate_train_test_feature_vectors",
                "value": int(audit.attrs["duplicate_count"]),
                "samples": "",
            },
        ]
    )
    test_metrics.to_csv(results_dir / "test_metrics.csv", index=False)

    attack_predictions = {}
    offset = len(test_texts)
    for name, prompts in pcap_prompts.items():
        attack_predictions[name] = names[offset : offset + len(prompts)]
        offset += len(prompts)
    baseline = detection_summary(
        attack_predictions["original"],
        int(config["seed"]),
        int(config["bootstrap_resamples"]),
        float(config["bootstrap_confidence"]),
    )
    detection_rows = [metric_row("detection_rate", "original", baseline)]
    for row in manifest:
        name = str(row["file"])
        summary = detection_summary(
            attack_predictions[name],
            int(config["seed"]),
            int(config["bootstrap_resamples"]),
            float(config["bootstrap_confidence"]),
        )
        summary["padding"] = int(row["padding"])
        summary["delay_ms"] = int(row["delay"])
        detection_rows.append(
            metric_row(
                "detection_rate",
                name,
                summary,
                float(baseline["detection_rate"]),
            )
        )
    detection_metrics = pd.DataFrame(detection_rows)
    test_export = pd.DataFrame(
        [
            {
                "metric": row.metric,
                "variant": "test",
                "samples": row.samples,
                "value": row.value,
                "ci_lower": "",
                "ci_upper": "",
                "change_from_original": "",
            }
            for row in test_metrics.itertuples(index=False)
        ]
    )
    metrics = pd.concat([test_export, detection_metrics], ignore_index=True)
    metrics.to_csv(results_dir / "metrics.csv", index=False)

    warning = None
    if (
        source == "csv"
        and float(baseline["detection_rate"])
        < float(config["min_baseline_detection"])
    ):
        warning = (
            "RESULTS NOT MEANINGFUL: no benign PCAP, CSV and PCAP are different "
            "sources; baseline detection too low"
        )
    print("Step 7/7: report")
    report = write_report(
        config=config,
        device=device,
        source=source,
        training=training,
        test_metrics=test_metrics,
        detection_metrics=detection_metrics,
        functionality=functionality[functionality_columns],
        domain_gap=domain_gap,
        shortcut_audit=audit,
        duplicate_vectors=int(audit.attrs["duplicate_count"]),
        invalid_outputs=invalid_outputs,
        warning=warning,
        report_path=results_dir / "report.md",
    )
    return metrics, report


def main() -> None:
    """Parse quick and retrain flags, then run the evaluation pipeline."""
    parser = argparse.ArgumentParser(description="Run the IDS robustness evaluation")
    parser.add_argument("--quick", action="store_true", help="Evaluate a small sample and two variants")
    parser.add_argument("--retrain", action="store_true", help="Train a fresh LoRA adapter")
    args = parser.parse_args()
    metrics, report = run_evaluation(quick=args.quick, retrain=args.retrain)
    print(metrics.to_string(index=False))
    print(report)


if __name__ == "__main__":
    main()
