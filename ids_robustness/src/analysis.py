"""Summarize failure subgroups and produce real-model qualitative examples."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.feature_space import load_full_manifest
from src.pcap_features import FEATURE_COLUMNS, extract_pcap_flows
from src.predict_llm import (
    generate_unique_raw_answers,
    load_finetuned_model,
    parse_class_name,
)
from src.prepare_data import build_text_rows
from src.run_eval import scale_flow_frame
from src.utils import ROOT, load_config


def load_flow_variants(
    results_dir: Path,
    manifest: pd.DataFrame,
    duration_floor: float,
) -> dict[str, pd.DataFrame]:
    """Extract original and perturbed test flows, checking flow identity."""
    original_path = results_dir / "pcap" / "attack_test_sessions.pcap"
    original = extract_pcap_flows(original_path, duration_floor)
    if original.empty or original["flow_id"].duplicated().any():
        raise ValueError("The test-session PCAP must contain unique directional flows.")
    frames = {"original": original}
    expected_ids = set(original["flow_id"])
    for row in manifest.itertuples(index=False):
        path = results_dir / "pcap" / str(row.file)
        frame = extract_pcap_flows(path, duration_floor)
        if set(frame["flow_id"]) != expected_ids:
            raise ValueError(f"Flow identities differ in {path.name}.")
        frames[str(row.file)] = (
            frame.set_index("flow_id").loc[original["flow_id"]].reset_index()
        )
    return frames


def build_prompt_groups(
    frames: dict[str, pd.DataFrame],
    scaler: object,
) -> dict[str, pd.DataFrame]:
    """Build scaled model prompts while preserving each flow identifier."""
    groups = {}
    for name, frame in frames.items():
        scaled = scale_flow_frame(frame, scaler)
        scaled["label"] = 1
        text_rows = build_text_rows(scaled)
        groups[name] = pd.DataFrame(
            {
                "flow_id": frame["flow_id"].tolist(),
                "prompt": text_rows["text"].tolist(),
            }
        )
    return groups


def build_failure_breakdown(
    frames: dict[str, pd.DataFrame],
    answers: dict[str, dict[str, str]],
    duration_floor: float,
) -> pd.DataFrame:
    """Calculate detection rates for packet-count and duration-floor groups."""
    rows = []
    for variant, frame in frames.items():
        detections = frame["flow_id"].map(
            lambda flow_id: parse_class_name(answers[variant][flow_id]) == "DDoS"
        )
        for count in (2, 4):
            mask = frame["packet_count"].eq(count)
            rows.append(
                {
                    "variant": variant,
                    "dimension": "packet_count",
                    "group": str(count),
                    "flows": int(mask.sum()),
                    "detected": int(detections[mask].sum()),
                    "detection_rate": (
                        float(detections[mask].mean()) if mask.any() else ""
                    ),
                }
            )
        at_floor = frame["flow_duration"].le(duration_floor)
        for label, mask in (("yes", at_floor), ("no", ~at_floor)):
            rows.append(
                {
                    "variant": variant,
                    "dimension": "duration_at_floor",
                    "group": label,
                    "flows": int(mask.sum()),
                    "detected": int(detections[mask].sum()),
                    "detection_rate": (
                        float(detections[mask].mean()) if mask.any() else ""
                    ),
                }
            )
    return pd.DataFrame(rows)


def format_features(frame: pd.DataFrame, flow_id: str) -> str:
    """Format the seven raw extracted features for one flow."""
    row = frame.set_index("flow_id").loc[flow_id]
    return ", ".join(f"{feature}={float(row[feature]):.6g}" for feature in FEATURE_COLUMNS)


def write_qualitative_examples(
    output_path: Path,
    frames: dict[str, pd.DataFrame],
    prompts: dict[str, pd.DataFrame],
    answers: dict[str, dict[str, str]],
) -> str:
    """Write three evaded and three retained detections with real model text."""
    variant = "pad_0_delay_50ms.pcap"
    original = frames["original"]
    changed = frames[variant]
    original_answers = answers["original"]
    changed_answers = answers[variant]
    original_ids = set(original["flow_id"])
    evaded = sorted(
        flow_id
        for flow_id in original_ids
        if parse_class_name(original_answers[flow_id]) == "DDoS"
        and parse_class_name(changed_answers[flow_id]) != "DDoS"
    )
    retained = sorted(
        flow_id
        for flow_id in original_ids
        if parse_class_name(original_answers[flow_id]) == "DDoS"
        and parse_class_name(changed_answers[flow_id]) == "DDoS"
    )
    if len(evaded) < 3 or len(retained) < 3:
        raise ValueError(
            "The selected delay variant needs at least three evaded and three "
            "retained original detections for qualitative examples."
        )

    prompt_lookup = {
        name: group.set_index("flow_id")["prompt"].to_dict()
        for name, group in prompts.items()
    }
    lines = [
        "# Qualitative examples",
        "",
        f"Original flows are compared with `{variant}`. Examples are selected by "
        "sorted flow ID from the measured predictions.",
        "",
    ]
    for heading, flow_ids in (("Evaded", evaded[:3]), ("Still detected", retained[:3])):
        lines.extend([f"## {heading}", ""])
        for index, flow_id in enumerate(flow_ids, start=1):
            lines.extend(
                [
                    f"### {heading} example {index}",
                    "",
                    f"- Flow ID: `{flow_id}`",
                    f"- Original raw features: {format_features(original, flow_id)}",
                    f"- Perturbed raw features: {format_features(changed, flow_id)}",
                    f"- Original parsed prediction: {parse_class_name(original_answers[flow_id])}",
                    f"- Perturbed parsed prediction: {parse_class_name(changed_answers[flow_id])}",
                    "- Original model output:",
                    "",
                    "```text",
                    original_answers[flow_id],
                    "```",
                    "- Perturbed model output:",
                    "",
                    "```text",
                    changed_answers[flow_id],
                    "```",
                    "- Original text prompt:",
                    "",
                    "```text",
                    prompt_lookup["original"][flow_id],
                    "```",
                    "- Perturbed text prompt:",
                    "",
                    "```text",
                    prompt_lookup[variant][flow_id],
                    "```",
                    "",
                ]
            )
    report = "\n".join(lines)
    output_path.write_text(report, encoding="utf-8")
    return report


def run_analysis() -> tuple[pd.DataFrame, str]:
    """Run real-model subgroup predictions and write both analysis artifacts."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    breakdown_path = results_dir / "failure_breakdown.csv"
    examples_path = results_dir / "qualitative_examples.md"
    if breakdown_path.is_file() and examples_path.is_file():
        breakdown = pd.read_csv(breakdown_path)
        report = examples_path.read_text(encoding="utf-8")
        print(f"Reusing saved results: {breakdown_path}")
        print(breakdown.to_string(index=False))
        print(report)
        return breakdown, report

    manifest = load_full_manifest(results_dir, config)
    frames = load_flow_variants(
        results_dir,
        manifest,
        float(config["duration_floor"]),
    )
    scaler = joblib.load(results_dir / "feature_scaler.joblib")
    prompt_groups = build_prompt_groups(frames, scaler)
    ordered_names = list(prompt_groups)
    flat_prompts = [
        prompt
        for group in prompt_groups.values()
        for prompt in group["prompt"].tolist()
    ]
    model, tokenizer, device = load_finetuned_model()
    raw_answers = generate_unique_raw_answers(
        flat_prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(config["prediction_batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        max_input_length=int(config["max_seq_length"]),
    )
    answers = {}
    offset = 0
    for name in ordered_names:
        flow_ids = prompt_groups[name]["flow_id"].tolist()
        group_answers = raw_answers[offset : offset + len(flow_ids)]
        offset += len(flow_ids)
        answers[name] = dict(zip(flow_ids, group_answers))

    breakdown = build_failure_breakdown(
        frames,
        answers,
        float(config["duration_floor"]),
    )
    breakdown.to_csv(breakdown_path, index=False)
    report = write_qualitative_examples(
        examples_path,
        frames,
        prompt_groups,
        answers,
    )
    print(breakdown.to_string(index=False))
    print(f"Wrote {breakdown_path}")
    print(f"Wrote {examples_path}")
    return breakdown, report


def main() -> None:
    """Run failure-case breakdown and qualitative example generation."""
    run_analysis()


if __name__ == "__main__":
    main()
