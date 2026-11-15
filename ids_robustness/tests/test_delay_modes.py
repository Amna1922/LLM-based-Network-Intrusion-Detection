"""Verify packet timestamp behavior for the two delay policies."""

from __future__ import annotations

from scapy.all import Ether, IP, TCP, PcapReader, PcapWriter

from src.perturb_pcap import write_variant


def _write_exchange(path) -> None:
    """Write a small client-server TCP exchange with fixed timestamps."""
    packets = [
        (1.0, "10.0.0.1", "10.0.0.2", 1234, 80, "S"),
        (1.01, "10.0.0.2", "10.0.0.1", 80, 1234, "SA"),
        (1.02, "10.0.0.1", "10.0.0.2", 1234, 80, "A"),
        (1.03, "10.0.0.2", "10.0.0.1", 80, 1234, "A"),
    ]
    with PcapWriter(str(path), sync=True) as writer:
        for timestamp, source, destination, sport, dport, flags in packets:
            packet = Ether() / IP(src=source, dst=destination) / TCP(
                sport=sport,
                dport=dport,
                flags=flags,
            )
            packet.time = timestamp
            writer.write(packet)


def _timestamps(path) -> list[float]:
    """Read packet timestamps from a PCAP."""
    with PcapReader(str(path)) as reader:
        return [float(packet.time) for packet in reader]


def test_client_only_delays_only_following_client_packets(tmp_path) -> None:
    """Keep replies unchanged in client-only mode and shift them otherwise."""
    source = tmp_path / "source.pcap"
    client_only = tmp_path / "client_only.pcap"
    both = tmp_path / "both.pcap"
    _write_exchange(source)

    write_variant(source, client_only, padding=0, delay_ms=50, delay_mode="client_only")
    write_variant(source, both, padding=0, delay_ms=50, delay_mode="client_and_reply")

    client_only_times = _timestamps(client_only)
    both_times = _timestamps(both)
    assert client_only_times[0] == 1.0
    assert client_only_times[1] == 1.01
    assert client_only_times[2] == 1.07
    assert client_only_times[3] == 1.03
    assert both_times[2] == 1.07
    assert both_times[1] == 1.01
    assert both_times[3] >= both_times[2]
