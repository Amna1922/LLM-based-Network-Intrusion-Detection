"""Run resumable feature and serialization ablations against the saved baseline."""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import joblib
import pandas as pd
import torch
from sklearn.metrics import f1_score

from src.baselines import build_evaluation_frames
from src.pcap_features import FEATURE_COLUMNS
from src.predict_llm import load_finetuned_model, predict_unique_prompts
from src.prepare_data import build_text_rows, prepare_dataset
from src.run_eval import scale_flow_frame
from src.train_llm import train_model
from src.utils import ROOT, load_config

ABLATIONS = {
    "A1": {
        "description": "Timing features removed",
        "feature_columns": ["ip_proto", "tp_dst", "packet_count", "byte_count"],
        "include_feature_names": True,
    },
    "A2": {
        "description": "Size features removed",
        "feature_columns": ["ip_proto", "tp_dst", "flow_duration"],
        "include_feature_names": True,
    },
    "A3": {
        "description": "Feature names removed from serialization",
        "feature_columns": FEATURE_COLUMNS,
        "include_feature_names": False,
    },
}


def adapter_is_ready(adapter_dir: Path) -> bool:
    """Return whether a completed adapter and its training log are present."""
    return (
        (adapter_dir / "adapter_config.json").is_file()
        and (adapter_dir / "training_log.txt").is_file()
    )


def completed_training_times(results_dir: Path) -> list[float]:
    """Read measured LoRA training durations from completed seed runs."""
    durations = []
    for path in (results_dir / "baseline_runs").glob("*.json"):
        result = json.loads(path.read_text(encoding="utf-8"))
        duration = result.get("training_time_seconds")
        if (
            result.get("model") == "Qwen2.5-0.5B-LoRA"
            and int(result.get("seed", -1)) in {123, 2025}
            and duration is not None
        ):
            durations.append(float(duration))
    return durations


def format_duration(seconds: float) -> str:
    """Format a measured duration estimate as hours, minutes, and seconds."""
    total_seconds = round(seconds)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, remaining_seconds = divmod(remainder, 60)
    return f"{hours}h {minutes}m {remaining_seconds}s"


def print_time_estimate(results_dir: Path) -> None:
    """Print the retraining estimate from measured prior LoRA runs."""
    run_dir = results_dir / "ablation_runs"
    adapter_root = results_dir / "ablation_adapters"
    pending = sum(
        1
        for name in ABLATIONS
        if not (run_dir / f"{name}.json").is_file()
        and not adapter_is_ready(adapter_root / name)
    )
    if pending == 0:
        print("No ablation retraining is pending; inference time remains.")
        return
    durations = completed_training_times(results_dir)
    if not durations:
        print(
            f"Estimated GPU retrains: {pending}. No measured training durations "
            "are available; estimate cannot be computed."
        )
        return
    average_seconds = statistics.mean(durations)
    total_seconds = average_seconds * pending
    print(
        f"Estimated training time for {pending} pending ablations: "
        f"{format_duration(total_seconds)} "
        f"(mean of {len(durations)} measured LoRA runs). Inference time is additional."
    )


def read_cached_run(path: Path, name: str, seed: int) -> dict:
    """Load a completed ablation result and verify its identity."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("ablation") != name or int(result.get("seed", -1)) != seed:
        raise ValueError(f"Cached ablation identity does not match {path}.")
    return result


def evaluate_trained_ablation(
    name: str,
    config: dict,
    prepared: tuple,
    tables: dict[str, pd.DataFrame],
    scaler: object,
    results_dir: Path,
) -> dict:
    """Train or reuse one ablation adapter and measure its predictions."""
    run_dir = results_dir / "ablation_runs"
    cache_path = run_dir / f"{name}.json"
    seed = int(config["seed"])
    if cache_path.is_file():
        print(f"Reuse cached ablation: {name}")
        return read_cached_run(cache_path, name, seed)

    settings = ABLATIONS[name]
    adapter_dir = results_dir / "ablation_adapters" / name
    if adapter_is_ready(adapter_dir):
        print(f"Reuse completed adapter: {name}")
    else:
        print(f"Training {name}: {settings['description']}")
        train_model(
            quick=False,
            prepared_data=prepared,
            seed=seed,
            adapter_dir=str(adapter_dir),
            feature_columns=settings["feature_columns"],
            include_feature_names=settings["include_feature_names"],
        )

    splits = prepared[3]
    test_rows = splits["test"]
    test_prompts = build_text_rows(
        test_rows,
        features=settings["feature_columns"],
        include_feature_names=settings["include_feature_names"],
    )["text"].tolist()
    attack_prompts = {}
    for variant, flows in tables.items():
        scaled = scale_flow_frame(flows, scaler)
        attack_prompts[variant] = build_text_rows(
            scaled,
            features=settings["feature_columns"],
            include_feature_names=settings["include_feature_names"],
        )["text"].tolist()

    all_prompts = test_prompts + [
        prompt for prompts in attack_prompts.values() for prompt in prompts
    ]
    model, tokenizer, device = load_finetuned_model(adapter_dir)
    answers = predict_unique_prompts(
        all_prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(config["prediction_batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        max_input_length=int(config["max_seq_length"]),
    )
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    test_count = len(test_prompts)
    test_answers = answers[:test_count]
    class_ids = {"Benign": 0, "DDoS": 1}
    test_predictions = [class_ids.get(answer, -1) for answer in test_answers]
    test_labels = test_rows["label"].astype(int).tolist()
    macro_f1 = float(
        f1_score(
            test_labels,
            test_predictions,
            labels=[0, 1],
            average="macro",
            zero_division=0,
        )
    )
    detection = {}
    offset = test_count
    for variant, prompts in attack_prompts.items():
        variant_answers = answers[offset : offset + len(prompts)]
        offset += len(prompts)
        detected = sum(answer == "DDoS" for answer in variant_answers)
        detection[variant] = {
            "flows": len(variant_answers),
            "detected": detected,
            "detection_rate": detected / len(variant_answers) if variant_answers else 0.0,
        }
    result = {
        "ablation": name,
        "seed": seed,
        "macro_f1": macro_f1,
        "invalid_outputs": sum(answer is None for answer in answers),
        "detection": detection,
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def result_rows(
    name: str,
    description: str,
    macro_f1: float,
    detection: dict[str, dict],
    manifest: pd.DataFrame,
) -> list[dict[str, object]]:
    """Create one output row per original or perturbed evaluation variant."""
    variant_settings = {
        str(row.file): (int(row.padding), int(row.delay))
        for row in manifest.itertuples(index=False)
    }
    variant_settings["original"] = (0, 0)
    rows = []
    for variant, metrics in detection.items():
        padding, delay = variant_settings[variant]
        rows.append(
            {
                "ablation": name,
                "description": description,
                "variant": variant,
                "padding": padding,
                "delay_ms": delay,
                "macro_f1": macro_f1,
                "flows": int(metrics["flows"]),
                "detected": int(metrics["detected"]),
                "detection_rate": float(metrics["detection_rate"]),
            }
        )
    return rows


def main_reference_rows(
    results_dir: Path,
    manifest: pd.DataFrame,
) -> tuple[list[dict[str, object]], float]:
    """Read the seed-42 main-run metrics for reference and reused comparisons."""
    test_metrics = pd.read_csv(results_dir / "test_metrics.csv")
    macro_f1 = float(
        test_metrics.loc[
            test_metrics["metric"] == "macro_f1_with_duplicates", "value"
        ].iloc[0]
    )
    metrics = pd.read_csv(results_dir / "metrics.csv")
    detection = metrics[metrics["metric"] == "detection_rate"].set_index("variant")
    detection_map = {
        variant: {
            "flows": int(row["samples"]),
            "detected": round(float(row["value"]) * int(row["samples"])),
            "detection_rate": float(row["value"]),
        }
        for variant, row in detection.iterrows()
    }
    rows = result_rows(
        "A0_main_reference",
        "Seed-42 main run",
        macro_f1,
        detection_map,
        manifest,
    )
    return rows, macro_f1


def zero_shot_rows(
    results_dir: Path,
    manifest: pd.DataFrame,
) -> list[dict[str, object]]:
    """Reuse Task 2 zero-shot metrics without rerunning the base model."""
    baseline_path = results_dir / "baselines.csv"
    detection_path = results_dir / "baseline_detection.csv"
    if not baseline_path.is_file() or not detection_path.is_file():
        raise FileNotFoundError("Task 2 baseline result files are required for A4.")
    baselines = pd.read_csv(baseline_path)
    selected = baselines[
        (baselines["model"] == "Qwen2.5-0.5B-ZeroShot")
        & (baselines["seed"].astype(int) == 42)
    ]
    if selected.empty:
        raise ValueError("Task 2 has no seed-42 zero-shot result.")
    macro_f1 = float(selected.iloc[0]["macro_f1"])
    detections = pd.read_csv(detection_path)
    detections = detections[
        (detections["model"] == "Qwen2.5-0.5B-ZeroShot")
        & (detections["seed"].astype(int) == 42)
    ].set_index("variant")
    detection_map = {
        variant: {
            "flows": int(row["samples"]),
            "detected": int(row["detected"]),
            "detection_rate": float(row["detection_rate"]),
        }
        for variant, row in detections.iterrows()
    }
    return result_rows(
        "A4_zero_shot",
        "Zero-shot Qwen reused from Task 2",
        macro_f1,
        detection_map,
        manifest,
    )


def write_notes(rows: pd.DataFrame, output_path: Path) -> str:
    """Write one measured-results sentence for each requested ablation."""
    lines = ["# Ablation notes", ""]
    descriptions = {
        "A1": ("A1", "timing features removed"),
        "A2": ("A2", "size features removed"),
        "A3": ("A3", "feature names removed from serialization"),
        "A4_zero_shot": ("A4", "zero-shot Qwen reused from Task 2"),
    }
    for key, (name, description) in descriptions.items():
        subset = rows[rows["ablation"] == key]
        if subset.empty:
            print(f"Skipping {key}: no ablation result rows are available.")
            lines.append(f"{name} ({description}) was not available; no result rows were saved.")
            continue
        macro_f1 = float(subset["macro_f1"].iloc[0])
        rates = ", ".join(
            f"{row.variant}={float(row.detection_rate):.6f}"
            for row in subset.itertuples(index=False)
        )
        lines.append(
            f"{name} ({description}) had test macro-F1={macro_f1:.6f} "
            f"and detection rates {rates}."
        )

    main_rows = rows[rows["ablation"] == "A0_main_reference"]
    main_macro_f1 = (
        float(main_rows["macro_f1"].iloc[0])
        if not main_rows.empty
        else None
    )
    grouped = []
    for category, label in (
        ("padding_only", "padding only"),
        ("delay_only", "delay only"),
        ("padding_and_delay", "padding and delay"),
    ):
        name = f"A5_{category}"
        subset = rows[rows["ablation"] == name]
        if subset.empty:
            print(f"Skipping {name}: no ablation result rows are available.")
            grouped.append(f"{label}: not available")
            continue
        rates = ", ".join(
            f"{row.variant}={float(row.detection_rate):.6f}"
            for row in subset.itertuples(index=False)
        )
        grouped.append(f"{label}: {rates}")
    if main_macro_f1 is None:
        print("Skipping A0_main_reference summary: no ablation result rows are available.")
        lines.append("A0_main_reference was not available; no result rows were saved.")
    if any(value != "not available" for value in grouped):
        macro_text = (
            f"{main_macro_f1:.6f}" if main_macro_f1 is not None else "not available"
        )
        lines.append(
            f"A5 (reused seed-42 results, main test macro-F1={macro_text}) "
            f"had detection rates {'; '.join(grouped)}."
        )
    else:
        lines.append("A5 reused perturbation comparisons were not available.")
    report = "\n".join(lines) + "\n"
    output_path.write_text(report, encoding="utf-8")
    return report


def run_ablation() -> tuple[pd.DataFrame, str]:
    """Run or resume all ablations and write their CSV and notes."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    output_path = results_dir / "ablation.csv"
    notes_path = results_dir / "ablation_notes.md"
    if output_path.is_file() and notes_path.is_file():
        results = pd.read_csv(output_path)
        notes = notes_path.read_text(encoding="utf-8")
        print(f"Reusing saved results: {output_path}")
        print(notes)
        return results, notes

    print_time_estimate(results_dir)
    prepared = prepare_dataset()
    _, tables, manifest = build_evaluation_frames(config, prepared)
    manifest_frame = pd.DataFrame(manifest)
    scaler = joblib.load(results_dir / "feature_scaler.joblib")

    rows, _ = main_reference_rows(results_dir, manifest_frame)
    for name, settings in ABLATIONS.items():
        result = evaluate_trained_ablation(
            name,
            config,
            prepared,
            tables,
            scaler,
            results_dir,
        )
        rows.extend(
            result_rows(
                name,
                str(settings["description"]),
                float(result["macro_f1"]),
                result["detection"],
                manifest_frame,
            )
        )

    rows.extend(zero_shot_rows(results_dir, manifest_frame))
    main_detection = {
        row["variant"]: {
            "flows": row["flows"],
            "detected": row["detected"],
            "detection_rate": row["detection_rate"],
        }
        for row in rows
        if row["ablation"] == "A0_main_reference"
    }
    main_macro_f1 = next(
        float(row["macro_f1"])
        for row in rows
        if row["ablation"] == "A0_main_reference"
    )
    for item in manifest:
        padding = int(item["padding"])
        delay = int(item["delay"])
        if padding > 0 and delay == 0:
            category = "padding_only"
        elif padding == 0 and delay > 0:
            category = "delay_only"
        else:
            category = "padding_and_delay"
        variant = str(item["file"])
        rows.extend(
            result_rows(
                f"A5_{category}",
                f"Reused main-run result: padding={padding}, delay={delay} ms",
                main_macro_f1,
                {variant: main_detection[variant]},
                manifest_frame,
            )
        )

    results = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    notes = write_notes(results, notes_path)
    print(results.to_string(index=False))
    print(f"Wrote {output_path}")
    print(f"Wrote {notes_path}")
    return results, notes


def main() -> None:
    """Run resumable ablation experiments and print the saved results."""
    run_ablation()


if __name__ == "__main__":
    main()
