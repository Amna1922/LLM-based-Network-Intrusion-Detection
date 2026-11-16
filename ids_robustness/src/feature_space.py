"""Compare direct feature perturbations with detections from perturbed captures."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import joblib
import pandas as pd
from scapy.all import IP, TCP, PcapReader

from src.pcap_features import extract_pcap_flows
from src.perturb_pcap import session_identifier, tcp_payload_bytes
from src.predict_llm import (
    generate_unique_raw_answers,
    load_finetuned_model,
    parse_class_name,
)
from src.prepare_data import build_text_rows
from src.run_eval import scale_flow_frame
from src.utils import ROOT, load_config


def load_full_manifest(results_dir: Path, config: dict) -> pd.DataFrame:
    """Read the saved full perturbation grid and reject incomplete runs."""
    manifest_path = results_dir / "pcap" / "manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"PCAP manifest not found: {manifest_path}")
    manifest = pd.read_csv(manifest_path)
    actual = {
        (int(row.padding), int(row.delay))
        for row in manifest.itertuples(index=False)
    }
    expected = {
        (int(padding), int(delay))
        for padding in config["padding_values"]
        for delay in config["delay_values_ms"]
        if int(padding) != 0 or int(delay) != 0
    }
    if actual != expected:
        raise ValueError(
            "The saved PCAP manifest is not the complete perturbation grid; "
            "run the full evaluation before this analysis."
        )
    return manifest


def count_attacker_payload_packets(pcap_path: str | Path) -> dict[str, int]:
    """Map client-direction flows to their declared TCP payload packet counts."""
    clients: dict[str, tuple[str, int]] = {}
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if IP not in packet or TCP not in packet:
                continue
            session = session_identifier(packet)
            if session is None:
                continue
            tcp = packet[TCP]
            flags = int(tcp.flags)
            if not flags & 0x02:
                continue
            client = (
                (packet[IP].dst, int(tcp.dport))
                if flags & 0x10
                else (packet[IP].src, int(tcp.sport))
            )
            previous = clients.setdefault(session, client)
            if previous != client:
                raise ValueError(f"Conflicting TCP client direction in session {session}.")

    counts: dict[str, int] = defaultdict(int)
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if IP not in packet or TCP not in packet:
                continue
            session = session_identifier(packet)
            if session not in clients:
                continue
            source = (packet[IP].src, int(packet[TCP].sport))
            if source != clients[session]:
                continue
            flow_id = "|".join(
                (
                    packet[IP].src,
                    str(int(packet[TCP].sport)),
                    packet[IP].dst,
                    str(int(packet[TCP].dport)),
                    str(int(packet[IP].proto)),
                )
            )
            counts.setdefault(flow_id, 0)
            if tcp_payload_bytes(packet):
                counts[flow_id] += 1
    return dict(counts)


def simulate_feature_variant(
    flows: pd.DataFrame,
    payload_counts: dict[str, int],
    padding: int,
    delay_ms: int,
    duration_floor: float,
) -> pd.DataFrame:
    """Apply padding and delay directly to raw flow features and recalculate rates."""
    simulated = flows.copy()
    attacker_flow = simulated["flow_id"].isin(payload_counts)
    packet_counts = (
        simulated["flow_id"].map(payload_counts).fillna(0).astype(float)
    )
    simulated["byte_count"] = (
        simulated["byte_count"].astype(float) + padding * packet_counts
    )
    simulated["flow_duration"] = (
        simulated["flow_duration"].astype(float)
        + attacker_flow.astype(float) * delay_ms / 1000.0
    ).clip(lower=duration_floor)
    simulated["packet_count_per_second"] = (
        simulated["packet_count"].astype(float) / simulated["flow_duration"]
    )
    simulated["byte_count_per_second"] = (
        simulated["byte_count"].astype(float) / simulated["flow_duration"]
    )
    return simulated


def run_feature_space() -> pd.DataFrame:
    """Run the saved LoRA model on feature-space variants and write comparisons."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    output_path = results_dir / "feature_vs_problem_space.csv"
    if output_path.is_file():
        results = pd.read_csv(output_path)
        print(f"Reusing saved results: {output_path}")
        print(results.to_string(index=False))
        return results

    manifest = load_full_manifest(results_dir, config)
    original_path = results_dir / "pcap" / "attack_test_sessions.pcap"
    original = extract_pcap_flows(original_path, float(config["duration_floor"]))
    if original.empty or original["flow_id"].duplicated().any():
        raise ValueError("The test-session PCAP must contain unique directional flows.")
    metrics_path = results_dir / "metrics.csv"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Evaluation metrics not found: {metrics_path}")
    metric_rows = pd.read_csv(metrics_path)
    real_rates = metric_rows[
        (metric_rows["metric"] == "detection_rate")
        & (metric_rows["variant"] != "original")
    ].set_index("variant")
    payload_counts = count_attacker_payload_packets(original_path)
    print(
        "Attacker-direction payload packets counted: "
        f"{sum(payload_counts.values())}"
    )

    scaler_path = results_dir / "feature_scaler.joblib"
    scaler = joblib.load(scaler_path)
    prompt_groups = {}
    for row in manifest.itertuples(index=False):
        variant_name = str(row.file)
        if variant_name not in real_rates.index:
            raise ValueError(f"Real-PCAP metrics are missing {variant_name}.")
        if int(real_rates.loc[variant_name, "samples"]) != len(original):
            raise ValueError(
                f"Real-PCAP sample count for {variant_name} does not match test flows."
            )
        simulated = simulate_feature_variant(
            original,
            payload_counts,
            int(row.padding),
            int(row.delay),
            float(config["duration_floor"]),
        )
        scaled = scale_flow_frame(simulated, scaler)
        scaled["label"] = 1
        prompt_groups[variant_name] = build_text_rows(scaled)["text"].tolist()

    all_prompts = [
        prompt for prompts in prompt_groups.values() for prompt in prompts
    ]
    model, tokenizer, device = load_finetuned_model()
    answers = generate_unique_raw_answers(
        all_prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(config["prediction_batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        max_input_length=int(config["max_seq_length"]),
    )
    output_rows = []
    offset = 0
    for row in manifest.itertuples(index=False):
        variant_name = str(row.file)
        count = len(prompt_groups[variant_name])
        parsed = [parse_class_name(answer) for answer in answers[offset : offset + count]]
        offset += count
        simulated_rate = sum(answer == "DDoS" for answer in parsed) / count
        real_rate = float(real_rates.loc[variant_name, "value"])
        output_rows.append(
            {
                "variant": variant_name,
                "padding": int(row.padding),
                "delay_ms": int(row.delay),
                "flows": count,
                "feature_detected": sum(answer == "DDoS" for answer in parsed),
                "feature_invalid_outputs": sum(answer is None for answer in parsed),
                "feature_space_detection_rate": simulated_rate,
                "real_pcap_detection_rate": real_rate,
                "difference_pp": (simulated_rate - real_rate) * 100.0,
            }
        )
    results = pd.DataFrame(output_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    print(results.to_string(index=False))
    print(f"Wrote {output_path}")
    return results


def main() -> None:
    """Run the feature-space versus problem-space comparison."""
    run_feature_space()


if __name__ == "__main__":
    main()
