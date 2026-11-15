"""Tests for selecting benign-PCAP-backed training data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import prepare_data


def _pcap_flows(prefix: str, count: int) -> pd.DataFrame:
    """Create distinct synthetic PCAP flows for a dataset-preparation test."""
    rows = []
    for index in range(count):
        packet_count = index + 1
        duration = float(index + 1)
        rows.append(
            {
                "flow_id": f"{prefix}-flow-{index}",
                "session_id": f"{prefix}-session-{index}",
                "ip_proto": 6,
                "tp_dst": 80 + index,
                "packet_count": packet_count,
                "byte_count": packet_count * 100,
                "flow_duration": duration,
                "packet_count_per_second": packet_count / duration,
                "byte_count_per_second": packet_count * 100 / duration,
            }
        )
    return pd.DataFrame(rows)


def _csv_rows() -> pd.DataFrame:
    """Create a small labeled CSV baseline with enough rows for grouped splits."""
    rows = []
    for label in (0, 1):
        for index in range(30):
            packet_count = index + 1
            rows.append(
                {
                    "flow_id": f"csv-{label}-{index}",
                    "ip_proto": 6,
                    "tp_dst": 80 + index,
                    "packet_count": packet_count,
                    "byte_count": packet_count * 100,
                    "flow_duration_sec": 1,
                    "flow_duration_nsec": 0,
                    "packet_count_per_second": packet_count,
                    "byte_count_per_second": packet_count * 100,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_prepare_dataset_selects_and_reports_benign_pcap(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    """Use PCAP flows as the primary dataset and report source/counts/splits."""
    benign_dir = tmp_path / "benign_pcaps"
    benign_dir.mkdir()
    benign_file = benign_dir / "benign_browsing.pcap"
    benign_file.write_bytes(b"synthetic test capture")
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    config = {
        "csv_path": str(tmp_path / "flows.csv"),
        "pcap_path": str(tmp_path / "attack.pcap"),
        "benign_pcap_dir": str(benign_dir),
        "duration_floor": 0.000001,
        "max_rows": 1000,
        "seed": 42,
    }

    def fake_extract(path: str | Path, duration_floor: float) -> pd.DataFrame:
        """Return synthetic captures in place of reading files."""
        prefix = "attack" if Path(path).name == "attack.pcap" else "benign"
        return _pcap_flows(prefix, 30)

    monkeypatch.setattr(prepare_data, "load_config", lambda: config)
    monkeypatch.setattr(prepare_data, "ROOT", tmp_path)
    monkeypatch.setattr(prepare_data, "extract_pcap_flows", fake_extract)
    monkeypatch.setattr(prepare_data.pd, "read_csv", lambda _: _csv_rows())

    primary, _, _, splits, audit = prepare_data.prepare_dataset()
    output = capsys.readouterr().out

    assert audit.attrs["source"] == "pcap"
    assert audit.attrs["train_rows"] == len(splits["train"])
    assert primary["label"].value_counts().to_dict() == {1: 30, 0: 30}
    assert "Benign PCAP folder checked: " + str(benign_dir) in output
    assert "  - benign_browsing.pcap" in output
    assert "Primary dataset source: PCAP" in output
    assert "PCAP flows extracted before row cap: Benign=30, DDoS=30" in output
    assert "Primary dataset flows after cap: Benign=30, DDoS=30, total=60" in output
    assert "PCAP feature summary by class (raw units, after row cap):" in output
    assert "Benign ip_proto: median=" in output
    assert "DDoS byte_count_per_second: median=" in output
    assert (
        f"Shortcut audit training split: source=PCAP, rows={len(splits['train'])}"
        in output
    )
    assert all(len(splits[name]) > 0 for name in ("train", "val", "test"))
