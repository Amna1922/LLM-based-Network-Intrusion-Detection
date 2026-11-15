"""Check PCAP flow invariants, SYN-flood functionality, and packet checksums."""

from __future__ import annotations

import struct
from pathlib import Path

import pandas as pd
from scapy.all import IP, TCP, PcapReader

from src.perturb_pcap import packet_flow
from src.utils import ROOT, load_config, resolve_path

FLAG_COLUMNS = (
    "packet_count",
    "syn_count",
    "ack_count",
    "rst_count",
    "fin_count",
    "psh_count",
)


def invariant_mismatch_columns(original: pd.DataFrame, altered: pd.DataFrame) -> list[str]:
    """Return invariant columns that differ for aligned flow rows."""
    identifiers = ["src_ip", "src_port", "dst_ip", "dst_port", "ip_proto"]
    keys = identifiers if all(column in original and column in altered for column in identifiers) else ["flow_id"]
    invariant_cols = ["ip_proto", "tp_dst", *FLAG_COLUMNS, *identifiers]
    common = [column for column in invariant_cols if column in original and column in altered]
    if not all(column in original and column in altered for column in keys):
        return [column for column in common if not original[column].equals(altered[column])]
    left = original.set_index(keys).sort_index()
    right = altered.set_index(keys).sort_index()
    if not left.index.equals(right.index):
        return list(dict.fromkeys([*common, *identifiers]))
    compared = [column for column in common if column not in keys]
    return [column for column in compared if not left[column].equals(right[column])]


def collect_packet_facts(pcap_path: str | Path) -> dict[str, object]:
    """Stream a PCAP and count flags, SYN tuples/times, and invalid checksums."""
    flows: dict[tuple, dict[str, int]] = {}
    syn_tuples = set()
    syn_times = []
    rst_ack_count = 0
    checksum_errors = 0
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if IP not in packet:
                continue
            ip = packet[IP]
            if TCP not in packet:
                continue
            tcp = ip[TCP]
            flow = (
                ip.src,
                int(tcp.sport),
                ip.dst,
                int(tcp.dport),
                int(ip.proto),
            )
            counts = flows.setdefault(flow, {name: 0 for name in FLAG_COLUMNS})
            counts["packet_count"] += 1
            flags = int(tcp.flags)
            for name, mask in (
                ("syn_count", 0x02),
                ("ack_count", 0x10),
                ("rst_count", 0x04),
                ("fin_count", 0x01),
                ("psh_count", 0x08),
            ):
                if flags & mask:
                    counts[name] += 1
            if flags & 0x02 and not flags & 0x10:
                syn_tuples.add(flow)
                syn_times.append(float(packet.time))
            if flags & 0x04 and flags & 0x10:
                rst_ack_count += 1
            if not packet_checksums_valid(packet):
                checksum_errors += 1
    return {
        "flows": flows,
        "syn_tuples": syn_tuples,
        "syn_count": len(syn_times),
        "syn_times": sorted(syn_times),
        "rst_ack_count": rst_ack_count,
        "checksum_errors": checksum_errors,
    }


def packet_checksums_valid(packet: object) -> bool:
    """Check stored IPv4 and TCP checksums using the packet bytes."""
    if IP not in packet or TCP not in packet:
        return True
    ip_bytes = bytes(packet[IP])
    header_length = (ip_bytes[0] & 0x0F) * 4
    total_length = struct.unpack("!H", ip_bytes[2:4])[0]
    if header_length < 20 or total_length < header_length:
        return False
    tcp_segment = ip_bytes[header_length:total_length]
    pseudo_header = (
        ip_bytes[12:20]
        + bytes((0, ip_bytes[9]))
        + struct.pack("!H", len(tcp_segment))
    )
    return internet_checksum(ip_bytes[:header_length]) == 0 and internet_checksum(
        pseudo_header + tcp_segment
    ) == 0


def internet_checksum(data: bytes) -> int:
    """Return the one's-complement checksum of a byte sequence."""
    if len(data) % 2:
        data += b"\x00"
    total = sum(struct.unpack(f"!{len(data) // 2}H", data)) if data else 0
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def peak_syn_rate(syn_times: list[float], window_seconds: float = 0.1) -> int:
    """Return the largest number of SYNs in any sliding time window."""
    peak = 0
    start = 0
    for end, timestamp in enumerate(syn_times):
        while timestamp - syn_times[start] > window_seconds:
            start += 1
        peak = max(peak, end - start + 1)
    return peak


def check_variant_functionality(
    original_facts: dict[str, object],
    variant_facts: dict[str, object],
    min_peak_rate_ratio: float,
    peak_window_seconds: float = 0.1,
) -> dict[str, object]:
    """Check SYN counts, tuples, reset replies, peak rate, and checksums."""
    original_syn_times = original_facts["syn_times"]
    variant_syn_times = variant_facts["syn_times"]
    original_peak = peak_syn_rate(original_syn_times, peak_window_seconds)
    variant_peak = peak_syn_rate(variant_syn_times, peak_window_seconds)
    checks = {
        "same_syn_count": original_facts["syn_count"] == variant_facts["syn_count"],
        "same_syn_5_tuples": original_facts["syn_tuples"] == variant_facts["syn_tuples"],
        "same_rst_ack_count": original_facts["rst_ack_count"] == variant_facts["rst_ack_count"],
        "peak_rate_pass": variant_peak >= original_peak * min_peak_rate_ratio,
        "checksums_valid": variant_facts["checksum_errors"] == 0,
    }
    return {
        "same_syn_count": checks["same_syn_count"],
        "same_syn_5_tuples": checks["same_syn_5_tuples"],
        "same_rst_ack_count": checks["same_rst_ack_count"],
        "peak_rate_pass": checks["peak_rate_pass"],
        "checksums_valid": checks["checksums_valid"],
        "original_syn_count": original_facts["syn_count"],
        "variant_syn_count": variant_facts["syn_count"],
        "original_syn_tuples": len(original_facts["syn_tuples"]),
        "variant_syn_tuples": len(variant_facts["syn_tuples"]),
        "original_rst_ack_count": original_facts["rst_ack_count"],
        "variant_rst_ack_count": variant_facts["rst_ack_count"],
        "original_peak_syn_per_100ms": original_peak,
        "variant_peak_syn_per_100ms": variant_peak,
        "checksum_errors": variant_facts["checksum_errors"],
        "pass_all": all(checks.values()),
    }


def validate_variants(
    original_path: str | Path,
    manifest_path: str | Path,
    variants_dir: str | Path,
    output_path: str | Path,
    min_peak_rate_ratio: float,
    peak_window_seconds: float = 0.1,
) -> pd.DataFrame:
    """Run functionality and invariant gates for every manifest variant."""
    original_facts = collect_packet_facts(original_path)
    rows = []
    invariant_failures = []
    manifest = pd.read_csv(manifest_path)
    for variant in manifest.itertuples(index=False):
        path = Path(variants_dir) / str(variant.file)
        facts = collect_packet_facts(path)
        result = check_variant_functionality(
            original_facts,
            facts,
            min_peak_rate_ratio,
            peak_window_seconds,
        )
        original_table = facts_to_frame(original_facts["flows"])
        variant_table = facts_to_frame(facts["flows"])
        mismatches = invariant_mismatch_columns(original_table, variant_table)
        result["file"] = str(variant.file)
        result["padding"] = int(variant.padding)
        result["delay"] = int(variant.delay)
        result["invariant_pass"] = not mismatches
        result["invariant_mismatches"] = ",".join(mismatches)
        result["pass_all"] = bool(result["pass_all"] and not mismatches)
        rows.append(result)
        if mismatches:
            print(f"Invariant gate failed for {variant.file}: {', '.join(mismatches)}")
            invariant_failures.append((str(variant.file), mismatches))
        print(
            f"{variant.file}: invariant={not mismatches}, "
            f"functionality={result['pass_all']}"
        )
    results = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_path, index=False)
    if invariant_failures:
        raise RuntimeError(f"Invariant gate failed: {invariant_failures}")
    passing_share = float(results["pass_all"].mean()) if not results.empty else 0.0
    print(f"Variants passing all checks: {passing_share:.3f}")
    return results


def facts_to_frame(flows: dict[tuple, dict[str, int]]) -> pd.DataFrame:
    """Convert per-direction packet facts to a table with full tuple columns."""
    rows = []
    for flow, counts in flows.items():
        src_ip, src_port, dst_ip, dst_port, protocol = flow
        rows.append(
            {
                "src_ip": src_ip,
                "src_port": src_port,
                "dst_ip": dst_ip,
                "dst_port": dst_port,
                "ip_proto": protocol,
                "tp_dst": dst_port,
                **counts,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Validate all generated PCAP variants using configured thresholds."""
    config = load_config()
    results_dir = ROOT / config["results_dir"]
    validate_variants(
        resolve_path(config["pcap_path"]),
        results_dir / "pcap" / "manifest.csv",
        results_dir / "pcap",
        results_dir / "functionality.csv",
        float(config["min_peak_rate_ratio"]),
        float(config["peak_window_ms"]) / 1000.0,
    )


if __name__ == "__main__":
    main()
