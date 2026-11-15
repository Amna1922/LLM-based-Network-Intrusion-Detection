"""Load the CSV, clean the features, split the flows, and write the domain-gap report."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

from src.pcap_features import extract_pcap_flows
from src.utils import ROOT, load_config, resolve_path

FEATURES = [
    "ip_proto",
    "tp_dst",
    "packet_count",
    "byte_count",
    "flow_duration",
    "packet_count_per_second",
    "byte_count_per_second",
]


def _clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop unneeded columns, fix the duration field, and remove invalid rows."""
    cleaned = frame.copy()
    cleaned = cleaned.drop(columns=["timestamp", "datapath_id", "ip_src", "tp_src", "ip_dst", "icmp_code", "icmp_type", "idle_timeout", "hard_timeout", "flags"], errors="ignore")
    cleaned["flow_duration"] = cleaned["flow_duration_sec"].astype(float) + cleaned["flow_duration_nsec"].astype(float) / 1_000_000_000.0
    cleaned = cleaned.drop(columns=["flow_duration_sec", "flow_duration_nsec"], errors="ignore")
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned


def print_label_evidence(frame: pd.DataFrame) -> None:
    """Print the label counts and the evidence that label 1 is the DDoS class."""
    counts = frame["label"].value_counts().sort_index()
    print("Label evidence:")
    for label, count in counts.items():
        label_name = "Benign" if label == 0 else "DDoS"
        print(f"  label {label}: {count} rows -> {label_name}")
    label_1 = frame[frame["label"] == 1]
    print(f"  all label 1 rows are TCP: {label_1['ip_proto'].nunique()} unique proto values -> {sorted(label_1['ip_proto'].unique().tolist())[:5]}")
    print(f"  label 1 TCP packets median: {label_1['packet_count'].median():.2f}, byte_count median: {label_1['byte_count'].median():.2f}")


def build_domain_gap(csv_frame: pd.DataFrame, pcap_frame: pd.DataFrame) -> pd.DataFrame:
    """Compare the feature ranges for CSV attack rows and pcap-derived flows."""
    attack_rows = csv_frame[csv_frame["label"] == 1].copy()
    gap_rows = []
    for feature in FEATURES:
        csv_values = attack_rows[feature].astype(float)
        pcap_values = pcap_frame[feature].astype(float)
        gap_rows.append(
            {
                "feature": feature,
                "csv_min": float(csv_values.min()),
                "csv_median": float(csv_values.median()),
                "csv_max": float(csv_values.max()),
                "pcap_min": float(pcap_values.min()),
                "pcap_median": float(pcap_values.median()),
                "pcap_max": float(pcap_values.max()),
            }
        )
    return pd.DataFrame(gap_rows)


def stratified_group_split(frame: pd.DataFrame, seed: int = 42) -> dict[str, pd.DataFrame]:
    """Split flow IDs into train, validation, and test sets while preserving label balance."""
    flow_labels = frame.groupby("flow_id")["label"].first().reset_index()
    flow_ids = flow_labels["flow_id"].tolist()
    labels = flow_labels["label"].tolist()

    test_size = max(2, int(round(len(flow_ids) * 0.2)))
    if test_size >= len(flow_ids):
        test_size = len(flow_ids) - 1
    splitter_test = StratifiedShuffleSplit(n_splits=1, test_size=test_size / len(flow_ids), random_state=seed)
    train_val_ids, test_ids = next(splitter_test.split(flow_ids, labels))
    train_val_flow_ids = [flow_ids[i] for i in train_val_ids]
    test_flow_ids = [flow_ids[i] for i in test_ids]

    train_val_labels = [labels[i] for i in train_val_ids]
    if len(train_val_flow_ids) < 4:
        train_flow_ids = train_val_flow_ids
        val_flow_ids = []
    else:
        splitter_val = StratifiedShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed)
        train_ids, val_ids = next(splitter_val.split(train_val_flow_ids, train_val_labels))
        train_flow_ids = [train_val_flow_ids[i] for i in train_ids]
        val_flow_ids = [train_val_flow_ids[i] for i in val_ids]

    split_map = {
        "train": frame[frame["flow_id"].isin(train_flow_ids)].copy(),
        "val": frame[frame["flow_id"].isin(val_flow_ids)].copy(),
        "test": frame[frame["flow_id"].isin(test_flow_ids)].copy(),
    }
    return split_map


def build_text_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert each row into a single prompt string with normalised feature values."""
    text_rows = []
    for _, row in frame.iterrows():
        items = []
        for feature in FEATURES:
            value = float(row[feature])
            items.append(f"{feature}: {value:.4f}")
        text_rows.append({
            "text": "Classify this network flow as Benign or DDoS. Flow: " + ", ".join(items),
            "label": int(row["label"]),
        })
    return pd.DataFrame(text_rows)


def prepare_dataset(csv_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Load the CSV, clean it, build the train/val/test splits, and calculate the domain gap."""
    config = load_config(ROOT / "config.yaml")
    raw = pd.read_csv(csv_path)
    clean = _clean_frame(raw)
    clean = clean[clean["label"].isin([0, 1])].copy()
    clean["label"] = clean["label"].astype(int)

    print_label_evidence(clean)
    pcap_features = extract_pcap_flows(resolve_path(config["pcap_path"]))
    gap = build_domain_gap(clean, pcap_features)
    gap_path = ROOT / "results" / "domain_gap.csv"
    gap.to_csv(gap_path, index=False)
    print(f"Domain gap written to {gap_path}")
    print(gap.to_string(index=False))

    split_map = stratified_group_split(clean, seed=int(config["seed"]))
    scaler = StandardScaler()
    for name, subset in split_map.items():
        subset[FEATURES] = scaler.fit_transform(subset[FEATURES]) if name == "train" else scaler.transform(subset[FEATURES])
    duplicate_count = set(tuple(row) for _, row in split_map["train"][FEATURES].iterrows()) & set(tuple(row) for _, row in split_map["test"][FEATURES].iterrows())
    print(f"Duplicate feature vectors between train and test: {len(duplicate_count)}")

    return clean, pcap_features, gap, split_map, pd.DataFrame({})


if __name__ == "__main__":
    config = load_config(ROOT / "config.yaml")
    csv_path = resolve_path(config["csv_path"])
    prepare_dataset(csv_path)
