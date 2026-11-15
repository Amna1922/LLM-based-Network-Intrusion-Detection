"""Verify delay-audit example selection and packet timing summaries."""

from __future__ import annotations

import pandas as pd

from src.delay_audit import choose_examples, shifted_packet_counts


def test_choose_examples_uses_distinct_retained_and_evaded_sessions() -> None:
    """Choose one retained detection and two evaded flows from distinct sessions."""
    selected = pd.DataFrame(
        {
            "session_id": ["retained", "retained", "evaded-1", "evaded-2"],
        }
    )

    examples = choose_examples(
        selected,
        ["DDoS", "DDoS", "DDoS", "DDoS"],
        ["DDoS", "Benign", "Benign", None],
    )

    assert examples == [(0, "still detected"), (2, "evaded"), (3, "evaded")]


def test_shifted_packet_counts_separate_client_and_reply_directions() -> None:
    """Count packet timestamp changes by packet direction."""
    original = {
        "session": [
            {"timestamp": 1.0, "direction": "attacker -> server"},
            {"timestamp": 1.1, "direction": "server -> attacker"},
        ]
    }
    variant = {
        "session": [
            {"timestamp": 1.05, "direction": "attacker -> server"},
            {"timestamp": 1.15, "direction": "server -> attacker"},
        ]
    }

    assert shifted_packet_counts(original, variant) == {
        "attacker -> server": 1,
        "server -> attacker": 1,
    }
