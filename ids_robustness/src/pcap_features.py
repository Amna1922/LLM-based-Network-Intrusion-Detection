"""Extract unidirectional TCP flow features from PCAP and PCAPNG files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scapy.all import IP, TCP, PcapReader

from src.utils import ROOT, load_config, resolve_path

FEATURE_COLUMNS = [
    "ip_proto",
    "tp_dst",
    "packet_count",
    "byte_count",
    "flow_duration",
    "packet_count_per_second",
    "byte_count_per_second",
]
FLAG_COLUMNS = ["syn_count", "ack_count", "rst_count", "fin_count", "psh_count"]
TUPLE_COLUMNS = ["src_ip", "src_port", "dst_ip", "dst_port"]


def extract_pcap_flows(
    pcap_path: str | Path,
    duration_floor: float | None = None,
) -> pd.DataFrame:
    """Stream packets and aggregate features by unidirectional TCP 5-tuple."""
    if duration_floor is None:
        duration_floor = float(load_config()["duration_floor"])
    stats: dict[tuple[str, int, str, int, int], dict[str, float]] = {}
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if IP not in packet or TCP not in packet:
                continue
            ip_packet = packet[IP]
            tcp_packet = packet[TCP]
            flow_key = (
                ip_packet.src,
                int(tcp_packet.sport),
                ip_packet.dst,
                int(tcp_packet.dport),
                int(ip_packet.proto),
            )
            if flow_key not in stats:
                stats[flow_key] = {
                    "packet_count": 0,
                    "byte_count": 0,
                    "start_time": float(packet.time),
                    "end_time": float(packet.time),
                    **{flag: 0 for flag in FLAG_COLUMNS},
                }
            record = stats[flow_key]
            record["packet_count"] += 1
            record["byte_count"] += len(packet)
            record["end_time"] = max(record["end_time"], float(packet.time))
            record["start_time"] = min(record["start_time"], float(packet.time))
            flags = int(tcp_packet.flags)
            for name, mask in (
                ("syn_count", 0x02),
                ("ack_count", 0x10),
                ("rst_count", 0x04),
                ("fin_count", 0x01),
                ("psh_count", 0x08),
            ):
                if flags & mask:
                    record[name] += 1

    return make_flow_table(stats, duration_floor)


def make_flow_table(
    stats: dict[tuple, dict[str, float]],
    duration_floor: float,
) -> pd.DataFrame:
    """Build the feature table and derive stable bidirectional session IDs."""
    rows = []
    for (src_ip, src_port, dst_ip, dst_port, protocol), record in stats.items():
        duration = max(record["end_time"] - record["start_time"], duration_floor)
        endpoints = sorted([(src_ip, src_port), (dst_ip, dst_port)])
        session_id = f"{endpoints[0][0]}:{endpoints[0][1]}|{endpoints[1][0]}:{endpoints[1][1]}|{protocol}"
        rows.append(
            {
                "src_ip": src_ip,
                "src_port": src_port,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "ip_proto": protocol,
                "tp_dst": dst_port,
                "packet_count": record["packet_count"],
                "byte_count": record["byte_count"],
                "flow_duration": duration,
                "packet_count_per_second": record["packet_count"] / duration,
                "byte_count_per_second": record["byte_count"] / duration,
                "syn_count": record["syn_count"],
                "ack_count": record["ack_count"],
                "rst_count": record["rst_count"],
                "fin_count": record["fin_count"],
                "psh_count": record["psh_count"],
                "session_id": session_id,
                "flow_id": "|".join(map(str, (src_ip, src_port, dst_ip, dst_port, protocol))),
            }
        )
    columns = TUPLE_COLUMNS + FEATURE_COLUMNS + FLAG_COLUMNS + ["session_id", "flow_id"]
    return pd.DataFrame(rows, columns=columns)


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    table = extract_pcap_flows(
        resolve_path(config["pcap_path"]),
        float(config["duration_floor"]),
    )
    print(table.head().to_string(index=False))
