"""Tests for the padding helper used in the PCAP perturbation step."""

from __future__ import annotations

from src.perturb_pcap import pad_payload


def test_padding_increases_payload_length() -> None:
    payload = b"abcdef"
    padded = pad_payload(payload, 16)
    assert len(padded) == len(payload) + 16
    assert padded[: len(payload)] == payload
