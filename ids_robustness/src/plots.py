"""Generate result-backed project figures with Matplotlib."""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch

from src.pcap_features import FEATURE_COLUMNS, extract_pcap_flows
from src.prepare_data import (
    build_pcap_dataset,
    cap_rows_by_session,
    find_benign_pcaps,
)
from src.utils import ROOT, load_config, resolve_path

FEATURE_UNITS = {
    "ip_proto": "protocol",
    "tp_dst": "port",
    "packet_count": "packets/flow",
    "byte_count": "bytes/flow",
    "flow_duration": "seconds/flow",
    "packet_count_per_second": "packets/s",
    "byte_count_per_second": "bytes/s",
}

MODEL_NAMES = (
    "LogisticRegression",
    "RandomForest",
    "HistGradientBoosting",
    "Qwen2.5-0.5B-ZeroShot",
    "Qwen2.5-0.5B-LoRA",
)


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> None:
    """Save a figure as a 300 dpi PNG and a PDF."""
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(directory / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def configure_plotting() -> None:
    """Set readable default fonts and figure styling."""
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.titlesize": 13,
        }
    )


def plot_pipeline(directory: Path) -> str:
    """Draw the local PCAP-to-detection processing architecture."""
    figure, axis = plt.subplots(figsize=(14, 4))
    axis.set_xlim(0, 14)
    axis.set_ylim(0, 4)
    axis.axis("off")
    labels = [
        "PCAP",
        "Perturbation",
        "Flow extraction",
        "Invariant gate",
        "Text serialization",
        "LLM",
        "Detection",
        "Functionality check",
    ]
    positions = np.linspace(0.9, 13.1, len(labels))
    for x_value, label in zip(positions, labels):
        box = FancyBboxPatch(
            (x_value - 0.72, 1.65),
            1.44,
            0.75,
            boxstyle="round,pad=0.08",
            facecolor="#e8f0fa",
            edgecolor="#284b63",
        )
        axis.add_patch(box)
        axis.text(x_value, 2.025, label, ha="center", va="center", wrap=True)
    for left, right in zip(positions[:-1], positions[1:]):
        axis.add_patch(
            FancyArrowPatch(
                (left + 0.75, 2.025),
                (right - 0.75, 2.025),
                arrowstyle="-|>",
                mutation_scale=12,
                color="#333333",
            )
        )
    axis.set_title("Offline intrusion-detection evaluation pipeline")
    axis.legend(
        handles=[Line2D([0], [0], color="#333333", marker=">", label="Local data flow")],
        loc="lower center",
    )
    save_figure(figure, directory, "fig01_pipeline_architecture")
    return "Architecture diagram of the local PCAP evaluation pipeline."


def plot_baseline_comparison(
    results_dir: Path,
    directory: Path,
) -> str:
    """Plot macro-F1 and mean conditional ASR for each measured model."""
    metrics = pd.read_csv(results_dir / "baselines.csv")
    detection = pd.read_csv(results_dir / "baseline_detection.csv")
    macro = metrics.groupby("model")["macro_f1"].mean()
    attack = detection[detection["variant"] != "original"]
    asr = attack.groupby("model")["asr"].mean()
    names = [name for name in MODEL_NAMES if name in macro.index]
    x_values = np.arange(len(names))
    width = 0.36
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(
        x_values - width / 2,
        [macro[name] for name in names],
        width,
        label="Macro-F1",
        color="#3b82a0",
    )
    asr_values = [asr.get(name, np.nan) for name in names]
    axis.bar(
        x_values + width / 2,
        asr_values,
        width,
        label="Mean conditional ASR",
        color="#e08e45",
    )
    for position, value in zip(x_values, asr_values):
        if pd.isna(value):
            axis.text(position + width / 2, 0.02, "N/A", ha="center", va="bottom")
    axis.set_xticks(x_values, names, rotation=20, ha="right")
    axis.set_ylabel("Score (0 to 1)")
    axis.set_xlabel("Model")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.set_title("Baseline macro-F1 and attack success rate")
    save_figure(figure, directory, "fig02_baseline_comparison")
    return "Mean test macro-F1 and mean conditional ASR across saved model runs."


def _roc_values(value: object) -> np.ndarray:
    """Decode a stored JSON ROC coordinate column."""
    if isinstance(value, str):
        return np.asarray(json.loads(value), dtype=float)
    return np.asarray(value, dtype=float)


def plot_roc_curves(results_dir: Path, directory: Path) -> str:
    """Plot mean ROC curves for the saved classical baseline seeds."""
    metrics = pd.read_csv(results_dir / "baselines.csv")
    classical = metrics[metrics["model"].isin(MODEL_NAMES[:3])]
    grid = np.linspace(0.0, 1.0, 201)
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, group in classical.groupby("model", sort=False):
        curves = []
        for row in group.itertuples(index=False):
            fpr = _roc_values(row.roc_fpr)
            tpr = _roc_values(row.roc_tpr)
            curves.append(np.interp(grid, fpr, tpr))
        mean_curve = np.mean(curves, axis=0)
        auc = float(group["roc_auc"].mean())
        axis.plot(grid, mean_curve, label=f"{model_name} (mean AUC={auc:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set_xlabel("False positive rate")
    axis.set_ylabel("True positive rate")
    axis.set_title("Mean ROC curves on the saved test split")
    axis.legend()
    save_figure(figure, directory, "fig03_classical_roc")
    return "Mean classical-baseline ROC curves interpolated from saved seed runs."


def plot_variant_detection(results_dir: Path, directory: Path) -> str:
    """Plot per-model detection rates with their saved 95 percent intervals."""
    stats = pd.read_csv(results_dir / "stats_summary.csv")
    stats = stats[stats["metric"] == "detection_rate"]
    variants = ["original"] + [
        value
        for value in stats["variant"].drop_duplicates().tolist()
        if value != "original"
    ]
    models = [name for name in MODEL_NAMES if name in set(stats["model"])]
    figure, axis = plt.subplots(figsize=(14, 6))
    width = 0.8 / max(len(models), 1)
    positions = np.arange(len(variants))
    for index, model_name in enumerate(models):
        group = stats[stats["model"] == model_name].set_index("variant")
        means = group.reindex(variants)["mean"].to_numpy(dtype=float)
        lower = group.reindex(variants)["wilson_lower"].to_numpy(dtype=float)
        upper = group.reindex(variants)["wilson_upper"].to_numpy(dtype=float)
        errors = np.vstack(
            [np.maximum(means - lower, 0), np.maximum(upper - means, 0)]
        )
        axis.bar(
            positions + (index - (len(models) - 1) / 2) * width,
            means,
            width,
            yerr=errors,
            capsize=2,
            label=model_name,
        )
    axis.set_xticks(positions, variants, rotation=35, ha="right")
    axis.set_ylabel("Detection rate (flows detected / evaluated flows)")
    axis.set_xlabel("Original or perturbation variant")
    axis.set_ylim(0, 1.08)
    axis.set_title("Attack detection per variant with Wilson 95% intervals")
    axis.legend()
    save_figure(figure, directory, "fig04_detection_confidence")
    return "Per-variant detection rates and Wilson intervals from the statistics CSV."


def _client_only_rate(report_path: Path) -> float | None:
    """Read the measured client-only delay detection rate from the audit report."""
    if not report_path.is_file():
        return None
    text = report_path.read_text(encoding="utf-8")
    match = re.search(
        r"\|\s*client_only\s*\|\s*\d+/\d+\s*\|\s*([0-9.]+)\s*\|",
        text,
    )
    return float(match.group(1)) if match else None


def plot_delay_heatmaps(
    results_dir: Path,
    config: dict,
    directory: Path,
) -> str:
    """Plot full-grid configured-mode and measured client-only detection."""
    detection = pd.read_csv(results_dir / "baseline_detection.csv")
    detection = detection[
        (detection["model"] == "Qwen2.5-0.5B-LoRA")
        & (detection["seed"].astype(int) == int(config["seed"]))
    ]
    manifest = pd.read_csv(results_dir / "pcap" / "manifest.csv")
    configured = np.full(
        (len(config["padding_values"]), len(config["delay_values_ms"])),
        np.nan,
    )
    padding_index = {
        int(value): index for index, value in enumerate(config["padding_values"])
    }
    delay_index = {
        int(value): index for index, value in enumerate(config["delay_values_ms"])
    }
    rates = detection.set_index("variant")["detection_rate"].to_dict()
    configured[padding_index[0], delay_index[0]] = float(
        detection.loc[detection["variant"] == "original", "detection_rate"].iloc[0]
    )
    for row in manifest.itertuples(index=False):
        configured[padding_index[int(row.padding)], delay_index[int(row.delay)]] = float(
            rates[str(row.file)]
        )
    client_only = np.full_like(configured, np.nan)
    client_only_rate = _client_only_rate(results_dir / "delay_audit.md")
    if client_only_rate is not None and 0 in padding_index and 50 in delay_index:
        client_only[padding_index[0], delay_index[50]] = client_only_rate

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, matrix, title in (
        (axes[0], configured, str(config.get("delay_mode", "configured"))),
        (axes[1], client_only, "client_only"),
    ):
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        axis.set_xticks(
            np.arange(len(config["delay_values_ms"])),
            config["delay_values_ms"],
        )
        axis.set_yticks(
            np.arange(len(config["padding_values"])),
            config["padding_values"],
        )
        axis.set_xlabel("Added delay (ms)")
        axis.set_ylabel("TCP payload padding (bytes)")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix[row_index, column_index]
                label = f"{value:.2f}" if np.isfinite(value) else "N/M"
                axis.text(column_index, row_index, label, ha="center", va="center")
    figure.colorbar(image, ax=axes, label="Detection rate (fraction of flows)")
    figure.suptitle("Detection heatmaps by padding, delay, and delay mode")
    save_figure(figure, directory, "fig05_delay_mode_heatmaps")
    return (
        "The configured delay mode has the full saved grid; client-only has only "
        "the measured zero-padding, 50 ms cell, with other cells marked not measured."
    )


def _distribution_data(results_dir: Path, config: dict) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load capped class distributions and all saved test attack variants."""
    attack_path = resolve_path(config["pcap_path"])
    benign_files = find_benign_pcaps(resolve_path(config["benign_pcap_dir"]))
    if not benign_files:
        raise FileNotFoundError("Feature distribution plot requires benign PCAP files.")
    labeled = build_pcap_dataset(
        attack_path,
        benign_files,
        float(config["duration_floor"]),
    )
    class_flows = cap_rows_by_session(
        labeled,
        int(config["max_rows"]),
        int(config["seed"]),
    )
    pcap_dir = results_dir / "pcap"
    attack_variants = {
        "original": extract_pcap_flows(
            pcap_dir / "attack_test_sessions.pcap",
            float(config["duration_floor"]),
        )
    }
    for path in sorted(pcap_dir.glob("pad_*.pcap")):
        attack_variants[path.stem] = extract_pcap_flows(
            path,
            float(config["duration_floor"]),
        )
    return class_flows, attack_variants


def _boxplot(
    axis: plt.Axes,
    frames: list[pd.DataFrame],
    feature: str,
    labels: list[str],
    colors: list[str],
) -> None:
    """Draw log-like box plots for one feature across data groups."""
    values = [
        frame[feature].astype(float).replace([np.inf, -np.inf], np.nan).dropna()
        for frame in frames
    ]
    artists = axis.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showfliers=False,
    )
    for box, color in zip(artists["boxes"], colors):
        box.set_facecolor(color)
    axis.set_yscale("symlog", linthresh=1)
    axis.set_ylabel(f"Value ({FEATURE_UNITS[feature]})")
    axis.tick_params(axis="x", labelrotation=35)
    axis.set_title(feature)


def plot_feature_distributions(
    results_dir: Path,
    config: dict,
    directory: Path,
) -> str | None:
    """Plot benign and attack features plus real PCAP perturbation distributions."""
    if not find_benign_pcaps(resolve_path(config["benign_pcap_dir"])):
        print("Skip Fig. 6: no benign PCAP files are available.")
        return None
    class_flows, attack_variants = _distribution_data(results_dir, config)
    benign = class_flows[class_flows["label"] == 0]
    ddos = class_flows[class_flows["label"] == 1]
    attack_names = list(attack_variants)
    figure, axes = plt.subplots(4, 4, figsize=(18, 12))
    for index, feature in enumerate(FEATURE_COLUMNS):
        row_index, column_index = divmod(index, 4)
        top = axes[row_index, column_index]
        _boxplot(
            top,
            [benign, ddos],
            feature,
            ["Benign", "DDoS"],
            ["#4c78a8", "#f58518"],
        )
        bottom = axes[row_index + 2, column_index]
        _boxplot(
            bottom,
            [attack_variants[name] for name in attack_names],
            feature,
            [
                name.replace(".pcap", "").replace("pad_", "p").replace("_delay_", "d")
                for name in attack_names
            ],
            ["#777777"] + ["#54a24b"] * (len(attack_names) - 1),
        )
    axes[1, -1].axis("off")
    axes[3, -1].axis("off")
    figure.legend(
        handles=[
            Patch(facecolor="#4c78a8", label="Benign"),
            Patch(facecolor="#f58518", label="DDoS"),
            Patch(facecolor="#777777", label="Original attack"),
            Patch(facecolor="#54a24b", label="Perturbed attack"),
        ],
        loc="upper center",
        ncol=4,
    )
    figure.suptitle("Feature distributions on log-like axes")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(figure, directory, "fig06_feature_distributions")
    return (
        f"Raw feature distributions for {len(benign)} benign and {len(ddos)} DDoS "
        f"training-source flows, plus {len(attack_names) - 1} real perturbed variants."
    )


def plot_ablation(results_dir: Path, directory: Path) -> str | None:
    """Plot test macro-F1 and original-flow detection for each available ablation."""
    path = results_dir / "ablation.csv"
    if not path.is_file():
        return None
    rows = pd.read_csv(path)
    labels = ["A0 main", "A1 timing", "A2 size", "A3 names", "A4 zero-shot"]
    names = [
        "A0_main_reference",
        "A1",
        "A2",
        "A3",
        "A4_zero_shot",
    ]
    macros = []
    detection_rates = []
    for name in names:
        group = rows[rows["ablation"] == name]
        macros.append(float(group["macro_f1"].iloc[0]))
        detection_rates.append(float(group["detection_rate"].mean()))
    for name, label in (
        ("A5_padding_only", "A5 pad"),
        ("A5_delay_only", "A5 delay"),
        ("A5_padding_and_delay", "A5 both"),
    ):
        group = rows[rows["ablation"] == name]
        if not group.empty:
            labels.append(label)
            macros.append(float(group["macro_f1"].mean()))
            detection_rates.append(float(group["detection_rate"].mean()))
    positions = np.arange(len(labels))
    width = 0.38
    figure, axis = plt.subplots(figsize=(11, 5))
    axis.bar(positions - width / 2, macros, width, label="Test macro-F1")
    axis.bar(
        positions + width / 2,
        detection_rates,
        width,
        label="Mean detection rate across evaluated variants",
    )
    axis.set_xticks(positions, labels, rotation=20, ha="right")
    axis.set_ylabel("Score (0 to 1)")
    axis.set_xlabel("Ablation")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.set_title("Ablation test macro-F1 and original-flow detection")
    save_figure(figure, directory, "fig07_ablation")
    return "Ablation macro-F1 and original-flow detection from results/ablation.csv."


def plot_feature_space(results_dir: Path, directory: Path) -> str:
    """Plot real-PCAP and simulated feature-space detection rates."""
    data = pd.read_csv(results_dir / "feature_vs_problem_space.csv")
    positions = np.arange(len(data))
    width = 0.38
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(
        positions - width / 2,
        data["real_pcap_detection_rate"],
        width,
        label="Real perturbed PCAP",
    )
    axis.bar(
        positions + width / 2,
        data["feature_space_detection_rate"],
        width,
        label="Feature-space simulation",
    )
    axis.set_xticks(positions, data["variant"], rotation=35, ha="right")
    axis.set_ylabel("Detection rate (flows)")
    axis.set_xlabel("Perturbation variant")
    axis.set_ylim(0, 1.05)
    axis.legend()
    axis.set_title("Feature-space versus problem-space detection")
    save_figure(figure, directory, "fig08_feature_vs_problem_space")
    return (
        f"Comparison of feature-space and real-PCAP detection for {len(data)} "
        "saved perturbation variants."
    )


def plot_confusion_matrix(results_dir: Path, directory: Path) -> str:
    """Plot the seed-42 LoRA confusion matrix from its cached run."""
    run_path = results_dir / "baseline_runs" / "qwen_lora_seed_42.json"
    result = json.loads(run_path.read_text(encoding="utf-8"))
    matrix = np.asarray(result["test_metrics"]["confusion_matrix"], dtype=int)
    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], ["Benign", "DDoS"])
    axis.set_yticks([0, 1], ["Benign", "DDoS"])
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Qwen LoRA test confusion matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
            )
    figure.colorbar(image, ax=axis, label="Flow count")
    axis.legend(
        handles=[Patch(facecolor="#4c78a8", label="Confusion-matrix cells")],
        loc="upper right",
    )
    save_figure(figure, directory, "fig09_llm_confusion_matrix")
    return f"Seed-42 Qwen LoRA confusion matrix over {int(matrix.sum())} test flows."


def training_loss_paths(results_dir: Path) -> dict[int, Path]:
    """Locate per-step loss traces for saved Qwen LoRA seed runs."""
    paths = {}
    for result_path in (results_dir / "baseline_runs").glob("qwen_lora_seed_*.json"):
        seed = int(result_path.stem.rsplit("_", 1)[-1])
        run_dir = (
            results_dir / "llm_adapter"
            if seed == 42
            else results_dir / "seed_runs" / f"llm_seed_{seed}"
        )
        paths[seed] = run_dir / "training_loss.csv"
    return paths


def plot_training_losses(results_dir: Path, directory: Path) -> str | None:
    """Plot loss curves only when per-step traces exist for every saved seed."""
    paths = training_loss_paths(results_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if not paths or missing:
        print(
            "Skip Fig. 10: per-step training loss traces are missing for "
            + ", ".join(missing)
        )
        return None
    figure, axis = plt.subplots(figsize=(8, 5))
    for seed, path in sorted(paths.items()):
        history = pd.read_csv(path)
        if history.empty or not {"step", "loss"}.issubset(history.columns):
            raise ValueError(f"Training history is empty or malformed: {path}")
        history[["step", "loss"]] = history[["step", "loss"]].apply(
            pd.to_numeric,
            errors="raise",
        )
        axis.plot(history["step"], history["loss"], label=f"Seed {seed}")
    axis.set_xlabel("Optimizer step")
    axis.set_ylabel("Training loss")
    axis.set_title("Qwen LoRA training loss by seed")
    axis.legend()
    save_figure(figure, directory, "fig10_training_loss")
    return f"Training loss histories for {len(paths)} saved LoRA seeds."


def write_captions(captions: list[str | None], path: Path) -> None:
    """Write captions and explicit reasons when required measurements are absent."""
    lines = ["# Figure captions", ""]
    for index, caption in enumerate(captions, start=1):
        if caption is not None:
            text = caption
        elif index == 7:
            text = (
                "Not generated: Task 4 has not produced results/ablation.csv. "
                "Complete the ablation stage to generate this figure."
            )
        elif index == 10:
            text = (
                "Not generated: completed historical runs have no per-step loss "
                "traces. New training runs save these histories."
            )
        else:
            text = "Not generated because the required measurement file is not available."
        lines.append(f"**Fig. {index}.** {text}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_plots() -> list[str | None]:
    """Generate all figures whose required result files are available."""
    configure_plotting()
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    directory = results_dir / "figures"
    captions: list[str | None] = [
        plot_pipeline(directory),
        plot_baseline_comparison(results_dir, directory),
        plot_roc_curves(results_dir, directory),
        plot_variant_detection(results_dir, directory),
        plot_delay_heatmaps(results_dir, config, directory),
        plot_feature_distributions(results_dir, config, directory),
        plot_ablation(results_dir, directory),
        plot_feature_space(results_dir, directory),
        plot_confusion_matrix(results_dir, directory),
        plot_training_losses(results_dir, directory),
    ]
    write_captions(captions, directory / "captions.md")
    print(f"Wrote figure captions to {directory / 'captions.md'}")
    return captions


def main() -> None:
    """Generate the result-backed project figures."""
    captions = run_plots()
    generated = sum(caption is not None for caption in captions)
    print(f"Generated {generated} of {len(captions)} figures with available data.")


if __name__ == "__main__":
    main()
