"""Validate invariant fields and SYN-flood properties for perturbed PCAP variants."""

from __future__ import annotations

import pandas as pd


def invariant_mismatch_columns(original: pd.DataFrame, altered: pd.DataFrame) -> list[str]:
    """Return the names of invariant columns whose values changed between two flow tables."""
    invariant_cols = [
        "ip_proto",
        "tp_dst",
        "packet_count",
        "syn_count",
        "ack_count",
        "rst_count",
        "fin_count",
        "psh_count",
        "flow_id",
    ]
    mismatches: list[str] = []
    for column in invariant_cols:
        if column in original.columns and column in altered.columns:
            if not original[column].equals(altered[column]):
                mismatches.append(column)
    return mismatches


def main() -> None:
    """Placeholder validation entry point."""
    print("Invariant validation workflow ready.")


if __name__ == "__main__":
    main()
