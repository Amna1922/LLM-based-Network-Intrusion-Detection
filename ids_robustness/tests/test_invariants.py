"""Tests for the invariant gate in the PCAP validation step."""

from __future__ import annotations

import pandas as pd

from src.validate import invariant_mismatch_columns


def test_invariant_gate_detects_flag_count_change() -> None:
    original = pd.DataFrame([
        {"ip_proto": 6, "tp_dst": 80, "packet_count": 3, "syn_count": 2, "ack_count": 1, "rst_count": 0, "fin_count": 0, "psh_count": 0, "flow_id": "a"},
    ])
    altered = pd.DataFrame([
        {"ip_proto": 6, "tp_dst": 80, "packet_count": 3, "syn_count": 1, "ack_count": 1, "rst_count": 0, "fin_count": 0, "psh_count": 0, "flow_id": "a"},
    ])
    mismatch = invariant_mismatch_columns(original, altered)
    assert "syn_count" in mismatch
