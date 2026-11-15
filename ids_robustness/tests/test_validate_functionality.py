"""Verify that SYN removal fails the PCAP functionality gate."""

from __future__ import annotations

from src.validate import check_variant_functionality


def test_removed_syn_fails_syn_count_check() -> None:
    """Reject a variant that has one fewer initial SYN packet."""
    flow = ("10.0.0.1", 1234, "10.0.0.2", 80, 6)
    original = {
        "flows": {flow: {"syn_count": 1}},
        "syn_tuples": {flow},
        "syn_count": 1,
        "syn_times": [1.0],
        "rst_ack_count": 0,
        "checksum_errors": 0,
    }
    altered = {
        "flows": {},
        "syn_tuples": set(),
        "syn_count": 0,
        "syn_times": [],
        "rst_ack_count": 0,
        "checksum_errors": 0,
    }
    result = check_variant_functionality(original, altered, 0.5)
    assert not result["same_syn_count"]
    assert not result["pass_all"]
