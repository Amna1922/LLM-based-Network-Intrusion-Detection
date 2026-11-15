"""Convert a PCAP into one-row-per-flow feature summaries without loading the whole file into memory."""

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


def extract_pcap_flows(pcap_path: str | Path) -> pd.DataFrame:
    """Read a PCAP in a streaming fashion and aggregate one row per unidirectional 5-tuple."""
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
                    "syn_count": 0,
                    "ack_count": 0,
                    "rst_count": 0,
                    "fin_count": 0,
                    "psh_count": 0,
                }
            record = stats[flow_key]
            record["packet_count"] += 1
            record["byte_count"] += len(packet)
            record["end_time"] = max(record["end_time"], float(packet.time))
            record["start_time"] = min(record["start_time"], float(packet.time))
            flags = int(tcp_packet.flags)
            if flags & 0x02:
                record["syn_count"] += 1
            if flags & 0x10:
                record["ack_count"] += 1
            if flags & 0x04:
                record["rst_count"] += 1
            if flags & 0x01:
                record["fin_count"] += 1
            if flags & 0x08:
                record["psh_count"] += 1

    rows = []
    for (src_ip, src_port, dst_ip, dst_port, ip_proto), record in stats.items():
        duration = max(record["end_time"] - record["start_time"], 1e-9)
        packet_count = record["packet_count"]
        byte_count = record["byte_count"]
        rows.append(
            {
                "ip_proto": ip_proto,
                "tp_dst": dst_port,
                "packet_count": packet_count,
                "byte_count": byte_count,
                "flow_duration": duration,
                "packet_count_per_second": packet_count / duration,
                "byte_count_per_second": byte_count / duration,
                "syn_count": record["syn_count"],
                "ack_count": record["ack_count"],
                "rst_count": record["rst_count"],
                "fin_count": record["fin_count"],
                "psh_count": record["psh_count"],
            }
        )

    return pd.DataFrame(rows)[FEATURE_COLUMNS + ["syn_count", "ack_count", "rst_count", "fin_count", "psh_count"]]


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    table = extract_pcap_flows(resolve_path(config["pcap_path"]))
    print(table.head().to_string(index=False))
