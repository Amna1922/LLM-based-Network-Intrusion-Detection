"""Write padded and delayed PCAP variants using streaming Scapy readers and writers."""

from __future__ import annotations

import csv
import socket
import struct
from pathlib import Path

from scapy.all import IP, TCP, Padding, PcapReader, PcapWriter, Raw

from src.utils import ROOT, load_config, resolve_path


def pad_payload(payload: bytes, padding: int) -> bytes:
    """Append the requested number of zero bytes to a TCP payload."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return payload + (b"\x00" * padding)


def packet_flow(packet: object) -> tuple[tuple[str, int], tuple[str, int], int] | None:
    """Return a direction-independent TCP flow key and protocol number."""
    if IP not in packet or TCP not in packet:
        return None
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    endpoints = sorted(
        [(ip_layer.src, int(tcp_layer.sport)), (ip_layer.dst, int(tcp_layer.dport))]
    )
    return endpoints[0], endpoints[1], int(ip_layer.proto)


def tcp_payload_bytes(packet: object) -> bytes:
    """Return TCP payload bytes excluding any captured link-layer padding."""
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    ip_header_length = int(ip_layer.ihl or 5) * 4
    tcp_header_length = int(tcp_layer.dataofs or 5) * 4
    declared_length = int(ip_layer.len or len(bytes(ip_layer)))
    payload_length = max(0, declared_length - ip_header_length - tcp_header_length)
    return bytes(tcp_layer.payload)[:payload_length]


def internet_checksum(data: bytes) -> int:
    """Calculate an IPv4 or TCP one's-complement checksum."""
    if len(data) % 2:
        data += b"\x00"
    words = struct.unpack(f"!{len(data) // 2}H", data) if data else ()
    total = sum(words)
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return (~total) & 0xFFFF


def recompute_checksums(packet: object, ip_length: int) -> None:
    """Set the declared IP length and calculate IPv4 and TCP checksums."""
    ip_layer = packet[IP]
    tcp_layer = packet[TCP]
    ip_layer.len = ip_length
    ip_layer.chksum = 0
    tcp_layer.chksum = 0
    ip_bytes = bytes(ip_layer)
    header_length = int(ip_layer.ihl or 5) * 4
    header = bytearray(ip_bytes[:header_length])
    header[10:12] = b"\x00\x00"
    ip_layer.chksum = internet_checksum(bytes(header))
    ip_bytes = bytes(ip_layer)
    tcp_segment = bytearray(ip_bytes[header_length:ip_length])
    tcp_checksum_offset = 16
    tcp_segment[tcp_checksum_offset : tcp_checksum_offset + 2] = b"\x00\x00"
    pseudo_header = (
        socket.inet_aton(ip_layer.src)
        + socket.inet_aton(ip_layer.dst)
        + bytes((0, int(ip_layer.proto)))
        + struct.pack("!H", len(tcp_segment))
    )
    tcp_layer.chksum = internet_checksum(pseudo_header + bytes(tcp_segment))


def has_client_payload(pcap_path: str | Path) -> bool:
    """Check that at least one initial client SYN carries TCP payload."""
    with PcapReader(str(pcap_path)) as reader:
        for packet in reader:
            if IP in packet and TCP in packet:
                tcp = packet[TCP]
                if (
                    tcp.flags & 0x02
                    and not tcp.flags & 0x10
                    and tcp_payload_bytes(packet)
                ):
                    return True
    return False


def session_identifier(packet: object) -> str | None:
    """Return the stable bidirectional TCP session ID for a packet."""
    flow = packet_flow(packet)
    if flow is None:
        return None
    endpoint_a, endpoint_b, protocol = flow
    return (
        f"{endpoint_a[0]}:{endpoint_a[1]}|"
        f"{endpoint_b[0]}:{endpoint_b[1]}|{protocol}"
    )


def write_session_subset(
    source_path: str | Path,
    output_path: str | Path,
    session_ids: set[str],
) -> int:
    """Stream packets for the requested bidirectional sessions into a new PCAP."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with PcapReader(str(source_path)) as reader, PcapWriter(
        str(target), append=False, sync=True
    ) as writer:
        for packet in reader:
            if session_identifier(packet) in session_ids:
                writer.write(packet)
                count += 1
    if count == 0:
        raise ValueError("No packets matched the selected test attack sessions.")
    return count


def write_variant(
    source_path: str | Path,
    output_path: str | Path,
    padding: int,
    delay_ms: int,
) -> int:
    """Stream one PCAP variant to disk and return its number of packets."""
    client_by_flow: dict[tuple, tuple[str, int]] = {}
    packet_counts: dict[tuple, int] = {}
    offsets: dict[tuple, float] = {}
    latest_client_times: dict[tuple, float] = {}
    count = 0
    with PcapReader(str(source_path)) as reader, PcapWriter(
        str(output_path), append=False, sync=True
    ) as writer:
        for packet in reader:
            flow = packet_flow(packet)
            if flow is None:
                writer.write(packet)
                count += 1
                continue
            endpoint_a, endpoint_b, protocol = flow
            flow_key = (endpoint_a, endpoint_b, protocol)
            source_endpoint = (packet[IP].src, int(packet[TCP].sport))
            tcp = packet[TCP]
            old_ip_length = int(packet[IP].len or len(bytes(packet[IP])))
            frame_padding = bytes(packet[Padding]) if Padding in packet else b""
            original_payload = tcp_payload_bytes(packet)
            tcp.remove_payload()
            if original_payload:
                tcp.add_payload(Raw(original_payload))
            if flow_key not in client_by_flow:
                if tcp.flags & 0x02 and tcp.flags & 0x10:
                    client_by_flow[flow_key] = (
                        packet[IP].dst,
                        int(tcp.dport),
                    )
                else:
                    client_by_flow[flow_key] = source_endpoint
                packet_counts[flow_key] = 0
                offsets[flow_key] = 0.0
            client = client_by_flow[flow_key]
            original_time = float(packet.time)
            if source_endpoint == client:
                packet_counts[flow_key] += 1
                if packet_counts[flow_key] > 1:
                    offsets[flow_key] += delay_ms / 1000.0
                shift = offsets[flow_key]
                effective_time = original_time + shift
                latest_client_times[flow_key] = max(
                    latest_client_times.get(flow_key, effective_time),
                    effective_time,
                )
                if padding and original_payload:
                    tcp.remove_payload()
                    tcp.add_payload(Raw(pad_payload(original_payload, padding)))
            else:
                shift = offsets[flow_key]
                effective_time = original_time + shift
                if flow_key in latest_client_times:
                    effective_time = max(effective_time, latest_client_times[flow_key])
            if frame_padding:
                packet.add_payload(Padding(frame_padding))
            payload_padding = (
                padding
                if source_endpoint == client and original_payload
                else 0
            )
            recompute_checksums(packet, old_ip_length + payload_padding)
            if delay_ms:
                packet.time = effective_time
            writer.write(packet)
            count += 1
    return count


def generate_variants(
    source_path: str | Path,
    output_dir: str | Path,
    padding_values: list[int],
    delay_values_ms: list[int],
    variant_pairs: list[dict[str, int]] | None = None,
) -> list[dict[str, object]]:
    """Generate the configured PCAP grid and write its manifest."""
    if not has_client_payload(source_path):
        raise ValueError("The PCAP has no client-to-server TCP packet carrying payload.")
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    if variant_pairs is None:
        variant_pairs = [
            {"padding": padding, "delay": delay}
            for padding in padding_values
            for delay in delay_values_ms
            if padding != 0 or delay != 0
        ]
    for pair in variant_pairs:
        padding = int(pair["padding"])
        delay_ms = int(pair["delay"])
        filename = f"pad_{padding}_delay_{delay_ms}ms.pcap"
        write_variant(
            source_path,
            target_dir / filename,
            padding,
            delay_ms,
        )
        manifest.append(
            {"file": filename, "padding": padding, "delay": delay_ms}
        )
    with (target_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["file", "padding", "delay"])
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def main() -> None:
    """Generate configured PCAP variants and print the manifest entries."""
    config = load_config()
    source = resolve_path(config["pcap_path"])
    output = ROOT / config["results_dir"] / "pcap"
    manifest = generate_variants(
        source,
        output,
        config["padding_values"],
        config["delay_values_ms"],
    )
    for item in manifest:
        print(item)


if __name__ == "__main__":
    main()
