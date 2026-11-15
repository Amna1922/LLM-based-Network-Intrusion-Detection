"""Verify that the quick evaluation writes result files without loading a model."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import run_eval
from src.prepare_data import FEATURES, transform_features


def _flows() -> pd.DataFrame:
    """Create a small labeled flow set for the orchestration test."""
    rows = []
    for index in range(4):
        row = {
            "flow_id": f"flow-{index}",
            "session_id": f"session-{index}",
            "src_ip": "10.0.0.1",
            "src_port": 1000 + index,
            "dst_ip": "10.0.0.2",
            "dst_port": 80,
            "ip_proto": 6,
            "tp_dst": 80,
            "packet_count": 2 + index,
            "byte_count": 100 + index * 10,
            "flow_duration": 0.1 + index * 0.01,
            "packet_count_per_second": (2 + index) / (0.1 + index * 0.01),
            "byte_count_per_second": (100 + index * 10) / (0.1 + index * 0.01),
            "label": index % 2,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_run_eval_quick_writes_report(tmp_path, monkeypatch) -> None:
    """Run orchestration with tiny data and deterministic fake model predictions."""
    flows = _flows()
    root = tmp_path
    results = root / "results"
    adapter = results / "llm_adapter"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (results / "training_log.txt").write_text(
        "train_rows=2\nepochs=1\ntrain_loss=0.1\n",
        encoding="utf-8",
    )
    joblib.dump(
        StandardScaler().fit(transform_features(flows)),
        results / "feature_scaler.joblib",
    )
    audit = pd.DataFrame([{"feature": "packet_count", "single_feature_accuracy": 0.5}])
    audit.attrs.update(
        {
            "source": "csv",
            "duplicate_count": 0,
            "train_rows": 2,
        }
    )
    prepared = (
        flows,
        flows,
        pd.DataFrame([{"feature": "packet_count", "csv_median": 1.0}]),
        {"train": flows.iloc[:2], "test": flows.copy()},
        audit,
    )
    config = {
        "results_dir": "results",
        "seed": 42,
        "quick_eval_flows": 3,
        "max_eval_flows": 4,
        "quick_variants": [{"padding": 16, "delay": 0}, {"padding": 0, "delay": 50}],
        "padding_values": [0, 16, 64],
        "delay_values_ms": [0, 50, 100],
        "pcap_path": "attack.pcap",
        "duration_floor": 0.000001,
        "min_peak_rate_ratio": 0.5,
        "peak_window_ms": 100,
        "bootstrap_resamples": 10,
        "bootstrap_confidence": 0.95,
        "min_baseline_detection": 0.5,
        "prediction_batch_size": 16,
        "max_new_tokens": 16,
        "max_seq_length": 256,
    }

    def fake_variants(source, output, padding_values, delay_values_ms, variant_pairs=None):
        """Create a small quick manifest without writing packet data."""
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [
                {"file": "one.pcap", "padding": 16, "delay": 0},
                {"file": "two.pcap", "padding": 0, "delay": 50},
            ]
        ).to_csv(output / "manifest.csv", index=False)
        return [
            {"file": "one.pcap", "padding": 16, "delay": 0},
            {"file": "two.pcap", "padding": 0, "delay": 50},
        ]

    def fake_validation(*args, **kwargs):
        """Return passing variant checks for the report."""
        return pd.DataFrame(
            [
                {
                    "file": "one.pcap",
                    "padding": 16,
                    "delay": 0,
                    "invariant_pass": True,
                    "invariant_mismatches": "",
                    "same_syn_count": True,
                    "same_syn_5_tuples": True,
                    "same_rst_ack_count": True,
                    "peak_rate_pass": True,
                    "checksums_valid": True,
                    "pass_all": True,
                },
                {
                    "file": "two.pcap",
                    "padding": 0,
                    "delay": 50,
                    "invariant_pass": True,
                    "invariant_mismatches": "",
                    "same_syn_count": True,
                    "same_syn_5_tuples": True,
                    "same_rst_ack_count": True,
                    "peak_rate_pass": True,
                    "checksums_valid": True,
                    "pass_all": True,
                },
            ]
        )

    monkeypatch.setattr(run_eval, "ROOT", root)
    monkeypatch.setattr(run_eval, "load_config", lambda: config)
    monkeypatch.setattr(run_eval, "resolve_path", lambda value: root / "attack.pcap")
    monkeypatch.setattr(run_eval, "prepare_dataset", lambda: prepared)
    monkeypatch.setattr(run_eval, "generate_variants", fake_variants)
    monkeypatch.setattr(run_eval, "validate_variants", fake_validation)
    monkeypatch.setattr(run_eval, "extract_pcap_flows", lambda *args: flows.copy())
    monkeypatch.setattr(
        run_eval,
        "load_finetuned_model",
        lambda: (object(), object(), "cpu"),
    )
    monkeypatch.setattr(
        run_eval,
        "predict_unique_prompts",
        lambda prompts, **kwargs: ["DDoS"] * len(prompts),
    )
    monkeypatch.setattr(run_eval, "package_versions", lambda: {"torch": "test"})

    run_eval.run_evaluation(quick=True)
    assert (results / "report.md").is_file()
    assert (results / "metrics.csv").is_file()
