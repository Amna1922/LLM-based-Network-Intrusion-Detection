"""Tests for the flow-aware stratified split."""

from __future__ import annotations

import pandas as pd

from src.prepare_data import stratified_group_split


def test_flow_ids_do_not_overlap_between_splits() -> None:
    rows = []
    for flow_id, label in [("f1", 0), ("f2", 0), ("f3", 1), ("f4", 1)]:
        rows.append({"flow_id": flow_id, "label": label, "ip_proto": 6, "tp_dst": 80, "packet_count": 10, "byte_count": 100, "flow_duration": 1.0, "packet_count_per_second": 10.0, "byte_count_per_second": 100.0})
    frame = pd.DataFrame(rows)
    split_map = stratified_group_split(frame, seed=42)
    overlap = set(split_map["train"]["flow_id"]) & set(split_map["test"]["flow_id"])
    assert not overlap
