"""Evaluate classical baselines, zero-shot Qwen, and seeded LoRA runs."""

from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.stats import norm
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.pcap_features import extract_pcap_flows
from src.perturb_pcap import generate_variants, write_session_subset
from src.predict_llm import load_finetuned_model, predict_unique_prompts
from src.prepare_data import FEATURES, build_text_rows, prepare_dataset
from src.run_eval import sample_rows, scale_flow_frame, select_attack_flows
from src.train_llm import load_base_model, train_model
from src.utils import ROOT, load_config, resolve_path, set_seed

BASELINE_SEEDS = list(range(42, 52))
LLM_SEEDS = [42, 123, 2025]
CLASSICAL_MODELS = ("LogisticRegression", "RandomForest", "HistGradientBoosting")


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Return the Wilson score interval for a binomial proportion."""
    if total <= 0:
        return float("nan"), float("nan")
    z_value = float(norm.ppf(0.5 + confidence / 2.0))
    proportion = successes / total
    denominator = 1.0 + z_value**2 / total
    center = (proportion + z_value**2 / (2.0 * total)) / denominator
    margin = (
        z_value
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z_value**2 / (4.0 * total**2)
        )
        / denominator
    )
    return float(center - margin), float(center + margin)


def json_number_list(values: np.ndarray) -> str:
    """Serialize numeric ROC values as JSON, replacing non-finite numbers."""
    return json.dumps(
        [float(value) if np.isfinite(value) else None for value in values]
    )


def classification_metrics(
    labels: np.ndarray,
    predictions: np.ndarray,
    positive_scores: np.ndarray | None = None,
) -> dict[str, object]:
    """Compute binary classification metrics and optional ROC data."""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_negative, false_positive, false_negative, true_positive = (
        int(value) for value in matrix.ravel()
    )
    result: dict[str, object] = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "macro_f1": float(
            f1_score(labels, predictions, labels=[0, 1], average="macro")
        ),
        "false_positive_rate": (
            false_positive / (false_positive + true_negative)
            if false_positive + true_negative
            else 0.0
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "confusion_matrix": matrix.tolist(),
        "roc_auc": None,
        "roc_fpr": None,
        "roc_tpr": None,
    }
    if positive_scores is not None:
        result["roc_auc"] = float(roc_auc_score(labels, positive_scores))
        fpr, tpr, _ = roc_curve(labels, positive_scores)
        result["roc_fpr"] = [float(value) for value in fpr]
        result["roc_tpr"] = [float(value) for value in tpr]
    return result


def make_classical_model(name: str, seed: int):
    """Create one configured classical baseline with a deterministic seed."""
    if name == "LogisticRegression":
        return LogisticRegression(max_iter=1000, random_state=seed)
    if name == "RandomForest":
        return RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "HistGradientBoosting":
        return HistGradientBoostingClassifier(random_state=seed)
    raise ValueError(f"Unknown classical baseline: {name}")


def build_evaluation_frames(
    config: dict,
    prepared: tuple,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Load the held-out attack sample and matching original and variant flows."""
    results_dir = ROOT / config["results_dir"]
    attack_path = resolve_path(config["pcap_path"])
    _, attack_flows, _, _, audit = prepared
    source = str(audit.attrs["source"])
    attack_eval_path = attack_path
    if source == "pcap":
        splits = prepared[3]
        test_sessions = set(
            splits["test"].loc[splits["test"]["label"] == 1, "session_id"]
        )
        prefix = attack_path.name + "|"
        selected_sessions = {
            session[len(prefix) :]
            for session in test_sessions
            if session.startswith(prefix)
        }
        attack_eval_path = results_dir / "pcap" / "attack_test_sessions.pcap"
        if not attack_eval_path.is_file():
            write_session_subset(attack_path, attack_eval_path, selected_sessions)

    pcap_dir = results_dir / "pcap"
    manifest_path = pcap_dir / "manifest.csv"
    if not manifest_path.is_file():
        generate_variants(
            attack_eval_path,
            pcap_dir,
            config["padding_values"],
            config["delay_values_ms"],
            delay_mode=str(config.get("delay_mode", "client_and_reply")),
        )
    manifest_frame = pd.read_csv(manifest_path)
    manifest = manifest_frame.to_dict(orient="records")
    selected = select_attack_flows(
        attack_flows,
        prepared,
        source,
        attack_path,
        int(config["max_eval_flows"]),
        int(config["seed"]),
    )
    selected["label"] = 1
    flow_ids = selected["flow_id"].astype(str).tolist()
    tables = {"original": selected}
    for item in manifest:
        filename = str(item["file"])
        variant_path = pcap_dir / filename
        if not variant_path.is_file():
            raise FileNotFoundError(
                f"Expected perturbation variant is missing: {variant_path}"
            )
        flows = extract_pcap_flows(
            variant_path,
            float(config["duration_floor"]),
        )
        indexed = flows.set_index("flow_id").reindex(flow_ids).reset_index()
        if indexed["flow_id"].isna().any():
            raise ValueError(f"Flows are missing from perturbation {filename}.")
        indexed["label"] = 1
        tables[filename] = indexed
    return selected, tables, manifest


def prepare_inputs(
    config: dict,
    prepared: tuple,
    tables: dict[str, pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], list[str], dict[str, list[str]]]:
    """Build scaled feature matrices and normalized prompts for all evaluations."""
    splits = prepared[3]
    scaler = joblib.load(ROOT / config["results_dir"] / "feature_scaler.joblib")
    train_features = splits["train"][FEATURES].to_numpy(dtype=float)
    test_features = splits["test"][FEATURES].to_numpy(dtype=float)
    labels = splits["test"]["label"].astype(int).to_numpy()
    test_prompts = build_text_rows(splits["test"])["text"].tolist()
    attack_matrices = {}
    attack_prompts = {}
    for name, flows in tables.items():
        scaled = scale_flow_frame(flows, scaler)
        attack_matrices[name] = scaled[FEATURES].to_numpy(dtype=float)
        attack_prompts[name] = build_text_rows(scaled)["text"].tolist()
    return (
        train_features,
        test_features,
        attack_matrices,
        test_prompts,
        attack_prompts,
    )


def evaluate_detection(
    predictions: dict[str, list[int]],
    flow_count: int,
    confidence: float,
) -> dict[str, dict[str, object]]:
    """Compute per-variant detection, conditional ASR, and Wilson intervals."""
    original = predictions["original"]
    original_detected = int(sum(value == 1 for value in original))
    rows = {}
    for variant, values in predictions.items():
        detected = int(sum(value == 1 for value in values))
        lower, upper = wilson_interval(detected, flow_count, confidence)
        evaded = sum(
            before == 1 and after != 1
            for before, after in zip(original, values)
        )
        rows[variant] = {
            "samples": flow_count,
            "detected": detected,
            "detection_rate": detected / flow_count if flow_count else 0.0,
            "wilson_lower": lower,
            "wilson_upper": upper,
            "originally_detected": original_detected,
            "evaded": int(evaded),
            "asr": (
                evaded / original_detected
                if original_detected
                else None
            ),
        }
    return rows


def save_run(path: Path, result: dict[str, object]) -> None:
    """Write a completed model run so later invocations can resume."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")


def load_run(path: Path, model_name: str, seed: int) -> dict[str, object]:
    """Load a cached run and validate its model and seed identity."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result["model"] != model_name or int(result["seed"]) != seed:
        raise ValueError(f"Cached result identity does not match {path}.")
    return result


def evaluate_classical_run(
    name: str,
    seed: int,
    run_path: Path,
    config: dict,
    prepared: tuple,
    data: tuple,
) -> dict[str, object]:
    """Fit one classical baseline and evaluate its test and PCAP predictions."""
    if run_path.is_file():
        print(f"Reuse cached run: {name}, seed={seed}")
        return load_run(run_path, name, seed)
    train_x, test_x, attack_matrices, _, _ = data
    splits = prepared[3]
    train_y = splits["train"]["label"].astype(int).to_numpy()
    test_y = splits["test"]["label"].astype(int).to_numpy()
    model = make_classical_model(name, seed)
    started = time.perf_counter()
    model.fit(train_x, train_y)
    training_time = time.perf_counter() - started

    started = time.perf_counter()
    test_probabilities = model.predict_proba(test_x)[:, 1]
    test_time = time.perf_counter() - started
    test_predictions = (test_probabilities >= 0.5).astype(int)
    test_metrics = classification_metrics(
        test_y,
        test_predictions,
        test_probabilities,
    )
    attack_predictions = {}
    attack_time = 0.0
    for variant, features in attack_matrices.items():
        started = time.perf_counter()
        probabilities = model.predict_proba(features)[:, 1]
        attack_time += time.perf_counter() - started
        attack_predictions[variant] = (probabilities >= 0.5).astype(int).tolist()
    detection = evaluate_detection(
        attack_predictions,
        len(attack_predictions["original"]),
        float(config["bootstrap_confidence"]),
    )
    result = {
        "model": name,
        "seed": seed,
        "training_time_seconds": training_time,
        "test_inference_seconds_per_1000_flows": (
            test_time / len(test_y) * 1000 if len(test_y) else 0.0
        ),
        "attack_inference_seconds_per_1000_flows": (
            attack_time
            / (len(attack_predictions) * len(attack_predictions["original"]))
            * 1000
            if attack_predictions["original"]
            else 0.0
        ),
        "test_metrics": test_metrics,
        "detection": detection,
    }
    save_run(run_path, result)
    return result


def prepare_prompts(
    test_prompts: list[str],
    attack_prompts: dict[str, list[str]],
) -> tuple[list[str], list[str]]:
    """Flatten attack prompts in stable variant order for one batched LLM call."""
    names = list(attack_prompts)
    return names, test_prompts + [
        prompt for name in names for prompt in attack_prompts[name]
    ]


def llm_predictions(
    adapter_dir: Path | None,
    test_prompts: list[str],
    attack_prompts: dict[str, list[str]],
    config: dict,
) -> tuple[list[int], dict[str, list[int]], int, float, float]:
    """Run zero-shot or LoRA Qwen inference on test and attack flow prompts."""
    if adapter_dir is None:
        model, tokenizer = load_base_model(str(config["model_name"]), "cuda" if torch.cuda.is_available() else "cpu")
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
        model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        model, tokenizer, device = load_finetuned_model(adapter_dir)
    batch_size = int(config["prediction_batch_size"])
    max_tokens = int(config["max_new_tokens"])
    max_length = int(config["max_seq_length"])
    started = time.perf_counter()
    test_answers = predict_unique_prompts(
        test_prompts,
        model,
        tokenizer,
        device,
        batch_size,
        max_tokens,
        max_length,
    )
    test_time = time.perf_counter() - started
    names, combined_prompts = prepare_prompts([], attack_prompts)
    started = time.perf_counter()
    attack_answers = predict_unique_prompts(
        combined_prompts,
        model,
        tokenizer,
        device,
        batch_size,
        max_tokens,
        max_length,
    )
    attack_time = time.perf_counter() - started
    test_predictions = [1 if answer == "DDoS" else 0 for answer in test_answers]
    attack_predictions = {}
    offset = 0
    for name in names:
        count = len(attack_prompts[name])
        attack_predictions[name] = [
            1 if answer == "DDoS" else 0
            for answer in attack_answers[offset : offset + count]
        ]
        offset += count
    invalid = sum(answer is None for answer in test_answers + attack_answers)
    if device == "cuda":
        del model
        torch.cuda.empty_cache()
    return (
        test_predictions,
        attack_predictions,
        invalid,
        test_time,
        attack_time,
    )


def evaluate_llm_run(
    model_name: str,
    seed: int,
    adapter_dir: Path | None,
    run_path: Path,
    config: dict,
    prepared: tuple,
    data: tuple,
    training_time: float | None = None,
) -> dict[str, object]:
    """Evaluate one zero-shot or LoRA Qwen model and cache its measurements."""
    if run_path.is_file():
        print(f"Reuse cached run: {model_name}, seed={seed}")
        return load_run(run_path, model_name, seed)
    _, test_x, _, test_prompts, attack_prompts = data
    test_y = prepared[3]["test"]["label"].astype(int).to_numpy()
    (
        test_predictions,
        attack_predictions,
        invalid,
        test_time,
        attack_time,
    ) = llm_predictions(adapter_dir, test_prompts, attack_prompts, config)
    test_metrics = classification_metrics(
        test_y,
        np.asarray(test_predictions, dtype=int),
    )
    test_metrics["roc_auc"] = None
    detection = evaluate_detection(
        attack_predictions,
        len(attack_predictions["original"]),
        float(config["bootstrap_confidence"]),
    )
    row_count = len(test_y)
    attack_count = len(attack_predictions["original"])
    result = {
        "model": model_name,
        "seed": seed,
        "training_time_seconds": training_time,
        "test_inference_seconds_per_1000_flows": (
            test_time / row_count * 1000 if row_count else 0.0
        ),
        "attack_inference_seconds_per_1000_flows": (
            attack_time
            / (len(attack_predictions) * attack_count)
            * 1000
            if attack_count
            else 0.0
        ),
        "invalid_outputs": invalid,
        "test_metrics": test_metrics,
        "detection": detection,
    }
    save_run(run_path, result)
    return result


def result_rows(result: dict[str, object]) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Convert a cached run into baseline and per-variant CSV rows."""
    metrics = result["test_metrics"]
    row = {
        "model": result["model"],
        "seed": result["seed"],
        **{
            key: metrics.get(key)
            for key in (
                "accuracy",
                "precision",
                "recall",
                "f1",
                "macro_f1",
                "false_positive_rate",
                "true_negative",
                "false_positive",
                "false_negative",
                "true_positive",
                "roc_auc",
            )
        },
        "confusion_matrix": json.dumps(metrics["confusion_matrix"]),
        "roc_fpr": (
            json.dumps(metrics["roc_fpr"])
            if metrics["roc_fpr"] is not None
            else "N/A"
        ),
        "roc_tpr": (
            json.dumps(metrics["roc_tpr"])
            if metrics["roc_tpr"] is not None
            else "N/A"
        ),
        "training_time_seconds": result.get("training_time_seconds"),
        "test_inference_seconds_per_1000_flows": result[
            "test_inference_seconds_per_1000_flows"
        ],
        "attack_inference_seconds_per_1000_flows": result[
            "attack_inference_seconds_per_1000_flows"
        ],
        "invalid_outputs": result.get("invalid_outputs", 0),
    }
    detection_rows = [
        {
            "model": result["model"],
            "seed": result["seed"],
            "variant": variant,
            **values,
        }
        for variant, values in result["detection"].items()
    ]
    return row, detection_rows


def summarize_seed_results(
    runs: list[dict[str, object]],
) -> pd.DataFrame:
    """Summarize macro-F1, detection, and ASR across repeated model seeds."""
    rows = []
    by_model: dict[str, list[dict[str, object]]] = {}
    for result in runs:
        by_model.setdefault(str(result["model"]), []).append(result)
    for model, model_runs in by_model.items():
        f1_values = [
            float(result["test_metrics"]["macro_f1"])
            for result in model_runs
        ]
        rows.append(
            {
                "model": model,
                "metric": "macro_f1",
                "variant": "test",
                "n_seeds": len(f1_values),
                "mean": float(np.mean(f1_values)),
                "std": float(np.std(f1_values, ddof=1)) if len(f1_values) > 1 else None,
                "wilson_lower": None,
                "wilson_upper": None,
            }
        )
        variants = list(model_runs[0]["detection"])
        for variant in variants:
            detections = [
                result["detection"][variant]
                for result in model_runs
            ]
            detection_values = [
                float(item["detection_rate"]) for item in detections
            ]
            rows.append(
                {
                    "model": model,
                    "metric": "detection_rate",
                    "variant": variant,
                    "n_seeds": len(detection_values),
                    "mean": float(np.mean(detection_values)),
                    "std": (
                        float(np.std(detection_values, ddof=1))
                        if len(detection_values) > 1
                        else None
                    ),
                    "wilson_lower": wilson_interval(
                        sum(int(item["detected"]) for item in detections),
                        sum(int(item["samples"]) for item in detections),
                    )[0],
                    "wilson_upper": wilson_interval(
                        sum(int(item["detected"]) for item in detections),
                        sum(int(item["samples"]) for item in detections),
                    )[1],
                }
            )
            if variant == "original":
                continue
            asr_values = [
                float(item["asr"])
                for item in detections
                if item["asr"] is not None
            ]
            rows.append(
                {
                    "model": model,
                    "metric": "asr",
                    "variant": variant,
                    "n_seeds": len(asr_values),
                    "mean": float(np.mean(asr_values)) if asr_values else None,
                    "std": (
                        float(np.std(asr_values, ddof=1))
                        if len(asr_values) > 1
                        else None
                    ),
                    "wilson_lower": None,
                    "wilson_upper": None,
                }
            )
    return pd.DataFrame(rows)


def run_baselines() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run or resume all requested baselines and repeated seed evaluations."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    run_dir = results_dir / "baseline_runs"
    run_dir.mkdir(parents=True, exist_ok=True)
    prepared = prepare_dataset()
    tables_selected, tables, manifest = build_evaluation_frames(config, prepared)
    data = prepare_inputs(config, prepared, tables)
    train_rows = len(prepared[3]["train"])
    print(
        f"Evaluation data ready: train_rows={train_rows}, "
        f"test_rows={len(prepared[3]['test'])}, "
        f"attack_flows={len(tables_selected)}, variants={len(manifest)}"
    )

    runs: list[dict[str, object]] = []
    for name in CLASSICAL_MODELS:
        for seed in BASELINE_SEEDS:
            print(f"Baseline {name}, seed={seed}/{BASELINE_SEEDS[-1]}")
            path = run_dir / f"{name.lower()}_seed_{seed}.json"
            runs.append(
                evaluate_classical_run(
                    name,
                    seed,
                    path,
                    config,
                    prepared,
                    data,
                )
            )

    zero_path = run_dir / "qwen_zero_shot.json"
    runs.append(
        evaluate_llm_run(
            "Qwen2.5-0.5B-ZeroShot",
            int(config["seed"]),
            None,
            zero_path,
            config,
            prepared,
            data,
        )
    )

    for seed in LLM_SEEDS:
        adapter_dir = (
            results_dir / "llm_adapter"
            if seed == int(config["seed"])
            else results_dir / "seed_runs" / f"llm_seed_{seed}"
        )
        run_path = run_dir / f"qwen_lora_seed_{seed}.json"
        if run_path.is_file():
            runs.append(
                evaluate_llm_run(
                    "Qwen2.5-0.5B-LoRA",
                    seed,
                    adapter_dir,
                    run_path,
                    config,
                    prepared,
                    data,
                )
            )
            continue
        if seed != int(config["seed"]) and not (
            adapter_dir / "adapter_config.json"
        ).is_file():
            print(f"Training Qwen LoRA, seed={seed}")
            train_model(
                prepared_data=prepared,
                seed=seed,
                adapter_dir=str(adapter_dir),
            )
        elif seed != int(config["seed"]):
            print(f"Reuse LoRA adapter: seed={seed}")
        summary_path = (
            results_dir / "training_log.txt"
            if seed == int(config["seed"])
            else adapter_dir / "training_log.txt"
        )
        training_time = None
        if summary_path.is_file():
            for line in summary_path.read_text(encoding="utf-8").splitlines():
                if line.startswith("training_time_seconds="):
                    training_time = float(line.split("=", 1)[1])
        runs.append(
            evaluate_llm_run(
                "Qwen2.5-0.5B-LoRA",
                seed,
                adapter_dir,
                run_path,
                config,
                prepared,
                data,
                training_time=training_time,
            )
        )

    baseline_rows = []
    detection_rows = []
    for result in runs:
        baseline_row, detection_part = result_rows(result)
        baseline_rows.append(baseline_row)
        detection_rows.extend(detection_part)
    baselines = pd.DataFrame(baseline_rows)
    detection = pd.DataFrame(detection_rows)
    stats = summarize_seed_results(
        [
            result
            for result in runs
            if result["model"] != "Qwen2.5-0.5B-ZeroShot"
        ]
    )
    baselines.to_csv(results_dir / "baselines.csv", index=False)
    detection.to_csv(results_dir / "baseline_detection.csv", index=False)
    stats.to_csv(results_dir / "stats_summary.csv", index=False)
    print("Baseline metrics:")
    print(baselines.to_string(index=False))
    print("Baseline detection:")
    print(detection.to_string(index=False))
    print("Seed statistics:")
    print(stats.to_string(index=False))
    return baselines, detection, stats


def main() -> None:
    """Run baseline and multi-seed experiments."""
    run_baselines()


if __name__ == "__main__":
    main()
