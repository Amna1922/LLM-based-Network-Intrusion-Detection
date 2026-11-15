"""Apply padding and timing changes to a PCAP while preserving the flow metadata."""

from __future__ import annotations

from pathlib import Path


def pad_payload(payload: bytes, padding: int) -> bytes:
    """Return the payload with the requested number of trailing padding bytes added."""
    if padding < 0:
        raise ValueError("padding must be non-negative")
    return payload + (b"\x00" * padding)


def main() -> None:
    """Placeholder entry point for the PCAP perturbation workflow."""
    print("PCAP perturbation workflow is not yet implemented for the full pipeline.")


if __name__ == "__main__":
    main()
