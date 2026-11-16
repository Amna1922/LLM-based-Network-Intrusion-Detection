"""Test direct feature-space simulation."""

from __future__ import annotations

from math import isclose

import pandas as pd

from src.feature_space import simulate_feature_variant


def test_feature_space_padding_raises_byte_count_by_payload_packets() -> None:
    """Add the configured padding once for every attacker payload packet."""
    flows = pd.DataFrame(
        [
            {
                "flow_id": "attacker-flow",
                "packet_count": 4,
                "byte_count": 400,
                "flow_duration": 0.1,
                "packet_count_per_second": 40.0,
                "byte_count_per_second": 4000.0,
            },
            {
                "flow_id": "reply-flow",
                "packet_count": 2,
                "byte_count": 200,
                "flow_duration": 0.1,
                "packet_count_per_second": 20.0,
                "byte_count_per_second": 2000.0,
            },
        ]
    )
    simulated = simulate_feature_variant(
        flows,
        {"attacker-flow": 3},
        padding=16,
        delay_ms=50,
        duration_floor=0.000001,
    )

    assert simulated.loc[0, "byte_count"] == 448
    assert simulated.loc[1, "byte_count"] == 200
    assert isclose(simulated.loc[0, "flow_duration"], 0.15)
    assert isclose(simulated.loc[1, "flow_duration"], 0.1)
    assert isclose(simulated.loc[0, "packet_count_per_second"], 4 / 0.15)
