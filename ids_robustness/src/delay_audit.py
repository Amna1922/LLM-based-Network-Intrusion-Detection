"""Compare client-only and client-plus-reply delays on held-out attack flows."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scapy.all import IP, TCP, PcapReader

from src.pcap_features import FEATURE_COLUMNS, extract_pcap_flows
from src.perturb_pcap import session_identifier, write_variant, write_session_subset
from src.predict_llm import load_finetuned_model, predict_unique_prompts
from src.prepare_data import build_text_rows, prepare_dataset
from src.run_eval import sample_rows, scale_flow_frame
from src.utils import ROOT, load_config, resolve_path


def flow_floor_summary(frame: pd.DataFrame, floor: float) -> list[str]:
    """Return per-class flow-duration floor counts and fractions."""
    lines = ["Flow-duration floor fractions in capped primary PCAP data:"]
    for label, class_name in ((0, "Benign"), (1, "DDoS")):
        class_frame = frame[frame["label"] == label]
        at_floor = int(
            np.isclose(
                class_frame["flow_duration"].astype(float),
                floor,
                rtol=0,
                atol=1e-12,
            ).sum()
        )
        total = len(class_frame)
        fraction = at_floor / total if total else 0.0
        lines.append(
            f"- {class_name}: {at_floor}/{total} ({fraction:.6f})"
        )
    return lines


def read_session_packet_map(
    path: Path,
    session_ids: set[str],
) -> dict[str, list[dict[str, object]]]:
    """Read packet details once for a set of sessions."""
    sessions: dict[str, list[dict[str, object]]] = {
        session_id: [] for session_id in session_ids
    }
    client_endpoints: dict[str, tuple[str, int]] = {}
    with PcapReader(str(path)) as reader:
        for packet in reader:
            if IP not in packet or TCP not in packet:
                continue
            session_id = session_identifier(packet)
            if session_id not in sessions:
                continue
            tcp = packet[TCP]
            source = (packet[IP].src, int(tcp.sport))
            destination = (packet[IP].dst, int(tcp.dport))
            if session_id not in client_endpoints:
                client_endpoints[session_id] = (
                    destination
                    if tcp.flags & 0x02 and tcp.flags & 0x10
                    else source
                )
            direction = (
                "attacker -> server"
                if source == client_endpoints[session_id]
                else "server -> attacker"
            )
            sessions[session_id].append(
                {
                    "timestamp": float(packet.time),
                    "direction": direction,
                    "flags": str(tcp.flags),
                    "source": source,
                    "destination": destination,
                }
            )
    return sessions


def format_packet_lines(
    original: list[dict[str, object]],
    variant: list[dict[str, object]],
) -> list[str]:
    """Format packet records with timestamp shifts relative to the original."""
    lines = []
    for index, record in enumerate(variant):
        baseline_time = (
            float(original[index]["timestamp"])
            if index < len(original)
            else float(record["timestamp"])
        )
        shift_ms = (float(record["timestamp"]) - baseline_time) * 1000
        lines.append(
            f"  packet {index + 1}: {float(record['timestamp']):.6f}, "
            f"{record['direction']}, flags={record['flags']}, "
            f"shift={shift_ms:+.3f} ms"
        )
    return lines


def describe_features(frame: pd.DataFrame, flow_id: str) -> str:
    """Format the seven raw features for one directional flow."""
    rows = frame[frame["flow_id"] == flow_id]
    if rows.empty:
        return "flow not present"
    row = rows.iloc[0]
    return ", ".join(
        f"{feature}={float(row[feature]):.6g}"
        for feature in FEATURE_COLUMNS
    )


def choose_examples(
    selected: pd.DataFrame,
    original_predictions: list[str | None],
    delayed_predictions: list[str | None],
) -> list[tuple[int, str]]:
    """Choose one retained detection and two evaded detections if available."""
    retained: list[tuple[int, str]] = []
    evaded: list[tuple[int, str]] = []
    for index, (_, row) in enumerate(selected.iterrows()):
        if original_predictions[index] != "DDoS":
            continue
        if delayed_predictions[index] == "DDoS":
            retained.append((index, str(row["session_id"])))
        else:
            evaded.append((index, str(row["session_id"])))
    examples = []
    used_sessions: set[str] = set()
    for index, session_id in retained:
        if session_id not in used_sessions:
            examples.append((index, "still detected"))
            used_sessions.add(session_id)
            break
    for index, session_id in evaded:
        if session_id not in used_sessions:
            examples.append((index, "evaded"))
            used_sessions.add(session_id)
        if sum(outcome == "evaded" for _, outcome in examples) == 2:
            break
    return examples


def prediction_summary(
    predictions: list[str | None],
    original_predictions: list[str | None],
) -> tuple[float, float, int]:
    """Return detection rate, conditional attack success rate, and invalid count."""
    rate = (
        sum(prediction == "DDoS" for prediction in predictions) / len(predictions)
        if predictions
        else 0.0
    )
    originally_detected = sum(
        prediction == "DDoS" for prediction in original_predictions
    )
    evaded = sum(
        original == "DDoS" and current != "DDoS"
        for original, current in zip(original_predictions, predictions)
    )
    asr = evaded / originally_detected if originally_detected else 0.0
    invalid = sum(prediction is None for prediction in predictions)
    return rate, asr, invalid


def shifted_packet_counts(
    original: dict[str, list[dict[str, object]]],
    variant: dict[str, list[dict[str, object]]],
) -> dict[str, int]:
    """Count shifted attacker packets and server replies across sampled sessions."""
    counts = {"attacker -> server": 0, "server -> attacker": 0}
    for session_id, original_packets in original.items():
        changed_packets = variant.get(session_id, [])
        for index, packet in enumerate(changed_packets):
            if index >= len(original_packets):
                continue
            shift = float(packet["timestamp"]) - float(
                original_packets[index]["timestamp"]
            )
            if abs(shift) > 1e-6:
                counts[str(packet["direction"])] += 1
    return counts


def audit_delay_modes() -> str:
    """Run the held-out delay comparison and write its detailed Markdown report."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    report_path = results_dir / "delay_audit.md"
    if report_path.is_file():
        report = report_path.read_text(encoding="utf-8")
        print(report, end="")
        return report

    prepared = prepare_dataset()
    primary, attack_flows, _, splits, audit = prepared
    source = str(audit.attrs["source"])
    if source != "pcap":
        raise RuntimeError(
            "The delay audit requires a PCAP-primary dataset with benign captures."
        )
    attack_path = resolve_path(config["pcap_path"])
    sample_limit = int(config["max_eval_flows"])
    selected = attack_flows
    if source == "pcap":
        test_sessions = set(
            splits["test"].loc[splits["test"]["label"] == 1, "session_id"]
        )
        prefix = attack_path.name + "|"
        raw_sessions = {
            session[len(prefix) :]
            for session in test_sessions
            if session.startswith(prefix)
        }
        selected = selected[selected["session_id"].isin(raw_sessions)].copy()
    selected = sample_rows(selected, sample_limit, int(config["seed"]))
    if selected.empty:
        raise ValueError("No held-out attack flows are available for the delay audit.")

    audit_dir = results_dir / "delay_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    selected_sessions = set(selected["session_id"].astype(str))
    subset_path = audit_dir / "attack_test_sessions.pcap"
    if not subset_path.is_file():
        write_session_subset(attack_path, subset_path, selected_sessions)

    variant_paths = {
        "client_only": audit_dir / "pad_0_delay_50ms_client_only.pcap",
        "client_and_reply": audit_dir / "pad_0_delay_50ms.pcap",
    }
    for mode, path in variant_paths.items():
        if not path.is_file():
            write_variant(
                subset_path,
                path,
                padding=0,
                delay_ms=50,
                delay_mode=mode,
            )
    packet_maps = {
        "original": read_session_packet_map(subset_path, selected_sessions)
    }
    for mode, path in variant_paths.items():
        packet_maps[mode] = read_session_packet_map(path, selected_sessions)

    scaler = joblib.load(results_dir / "feature_scaler.joblib")
    feature_tables = {
        "original": extract_pcap_flows(
            subset_path,
            float(config["duration_floor"]),
        )
    }
    for mode, path in variant_paths.items():
        feature_tables[mode] = extract_pcap_flows(
            path,
            float(config["duration_floor"]),
        )
    selected_ids = selected["flow_id"].astype(str).tolist()
    prompts = {}
    for mode, frame in feature_tables.items():
        matched = frame.set_index("flow_id").reindex(selected_ids).reset_index()
        if matched["flow_id"].isna().any():
            raise ValueError(f"Some selected flow IDs were missing for {mode}.")
        matched["label"] = 1
        scaled = scale_flow_frame(matched, scaler)
        prompts[mode] = build_text_rows(scaled)["text"].tolist()

    model, tokenizer, device = load_finetuned_model()
    all_prompts = [
        prompt for mode in ("original", "client_only", "client_and_reply")
        for prompt in prompts[mode]
    ]
    all_predictions = predict_unique_prompts(
        all_prompts,
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=int(config["prediction_batch_size"]),
        max_new_tokens=int(config["max_new_tokens"]),
        max_input_length=int(config["max_seq_length"]),
    )
    count = len(selected)
    predictions = {
        "original": all_predictions[:count],
        "client_only": all_predictions[count : count * 2],
        "client_and_reply": all_predictions[count * 2 : count * 3],
    }
    lines = [
        "# Delay audit",
        "",
        f"- Primary dataset source: {source.upper()}",
        f"- Held-out directional attack flows sampled: {count}",
        "- Delay applied: 50 ms per attacker-sent packet after the first packet of its flow.",
        "",
        "## Detection by delay mode",
        "",
        "| Mode | Detected / flows | Detection rate | ASR | Invalid outputs |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    original_rate, _, original_invalid = prediction_summary(
        predictions["original"],
        predictions["original"],
    )
    lines.append(
        f"| original | {sum(item == 'DDoS' for item in predictions['original'])}/{count} "
        f"| {original_rate:.6f} | N/A | {original_invalid} |"
    )
    for mode in ("client_only", "client_and_reply"):
        rate, asr, invalid = prediction_summary(
            predictions[mode],
            predictions["original"],
        )
        detected = sum(item == "DDoS" for item in predictions[mode])
        lines.append(
            f"| {mode} | {detected}/{count} | {rate:.6f} | {asr:.6f} | {invalid} |"
        )
    lines.extend(["", "## Packets with changed timestamps", ""])
    lines.extend(
        [
            "| Mode | Attacker packets shifted | Server replies shifted |",
            "| --- | ---: | ---: |",
        ]
    )
    for mode in ("client_only", "client_and_reply"):
        changed = shifted_packet_counts(packet_maps["original"], packet_maps[mode])
        lines.append(
            f"| {mode} | {changed['attacker -> server']} "
            f"| {changed['server -> attacker']} |"
        )
    lines.extend(["", "## Duration-at-floor fractions", ""])
    lines.extend(flow_floor_summary(primary, float(config["duration_floor"])))
    examples = choose_examples(
        selected,
        predictions["original"],
        predictions["client_and_reply"],
    )
    lines.extend(
        [
            "",
            "## Packet and feature examples",
            "",
            "The primary packet comparison is original versus `pad_0_delay_50ms.pcap` "
            "(client-and-reply mode). Client-only packets are also shown for comparison.",
            "",
        ]
    )
    if len(examples) < 3:
        lines.append(
            "Could not select one retained detection and two evaded detections from "
            "the measured predictions."
        )
    for example_number, (index, outcome) in enumerate(examples, start=1):
        row = selected.iloc[index]
        session_id = str(row["session_id"])
        flow_id = str(row["flow_id"])
        lines.extend(
            [
                f"### Example {example_number}: {outcome}",
                "",
                f"- Flow ID: `{flow_id}`",
                f"- Session ID: `{session_id}`",
                f"- Flow direction: "
                f"{'attacker-to-server SYN' if int(row['syn_count']) > 0 else 'server-to-attacker reply'}",
                f"- Original prediction: {predictions['original'][index]}",
                f"- Client-only prediction: {predictions['client_only'][index]}",
                f"- Client-and-reply prediction: {predictions['client_and_reply'][index]}",
                f"- Original 7 features: {describe_features(feature_tables['original'], flow_id)}",
                f"- Client-only 7 features: {describe_features(feature_tables['client_only'], flow_id)}",
                f"- Client-and-reply 7 features: {describe_features(feature_tables['client_and_reply'], flow_id)}",
                "",
            ]
        )
        original_packets = packet_maps["original"].get(session_id, [])
        for mode, path in variant_paths.items():
            variant_packets = packet_maps[mode].get(session_id, [])
            lines.append(f"Packets, {mode}:")
            lines.extend(format_packet_lines(original_packets, variant_packets))
            if not variant_packets:
                lines.append("  no packets found")
            lines.append("")

    lines.extend(
        [
            "## Delay interpretation",
            "",
            "- `client_only` shifts attacker-to-server packet timestamps after the first "
            "packet and leaves server replies at their original timestamps.",
            "- `client_and_reply` shifts attacker packets and shifts replies by the same "
            "cumulative offset, with replies clamped not to precede the latest request.",
            "",
        ]
    )
    report = "\n".join(lines)
    report_path.write_text(report, encoding="utf-8")
    print(report, end="")
    return report


def main() -> None:
    """Run the resumable delay audit."""
    audit_delay_modes()


if __name__ == "__main__":
    main()
