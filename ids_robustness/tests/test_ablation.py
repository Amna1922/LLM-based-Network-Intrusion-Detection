"""Test ablation prompt serialization."""

from __future__ import annotations

import pandas as pd

from src.prepare_data import build_text_rows


def test_text_rows_support_feature_selection_and_bare_values() -> None:
    """Format selected features with names or as bare comma-separated values."""
    frame = pd.DataFrame(
        [
            {
                "tp_dst": 80.0,
                "flow_duration": 0.5,
                "label": 1,
            }
        ]
    )

    named = build_text_rows(
        frame,
        features=["tp_dst", "flow_duration"],
    )
    bare = build_text_rows(
        frame,
        features=["tp_dst", "flow_duration"],
        include_feature_names=False,
    )

    assert named.loc[0, "text"].endswith(
        "Flow: tp_dst: 80.0000, flow_duration: 0.5000"
    )
    assert bare.loc[0, "text"].endswith("Flow: 80.0000, 0.5000")
