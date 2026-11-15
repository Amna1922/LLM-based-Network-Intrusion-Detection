"""Verify that modified PCAP packets have valid IPv4 and TCP checksums."""

from __future__ import annotations

from scapy.all import Ether, IP, TCP, Raw, PcapReader, PcapWriter

from src.perturb_pcap import write_variant
from src.validate import packet_checksums_valid


def test_perturbed_packets_have_valid_checksums(tmp_path) -> None:
    """Recompute and verify checksums after appending TCP payload bytes."""
    source = tmp_path / "source.pcap"
    output = tmp_path / "padded.pcap"
    packet = Ether() / IP(src="10.0.0.1", dst="10.0.0.2") / TCP(
        sport=1234,
        dport=80,
        flags="S",
    ) / Raw(b"payload")
    packet.time = 1.0
    original_length = len(bytes(packet))
    with PcapWriter(str(source), sync=True) as writer:
        writer.write(Ether(bytes(packet)))
    write_variant(source, output, padding=16, delay_ms=0)

    with PcapReader(str(output)) as reader:
        packets = list(reader)
    assert len(packets) == 1
    assert len(bytes(packets[0][TCP].payload)) == len(b"payload") + 16
    assert len(bytes(packets[0])) == original_length + 16
    assert packet_checksums_valid(packets[0])
