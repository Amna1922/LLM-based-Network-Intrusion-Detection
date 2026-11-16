"""Prepare CSV or labeled PCAP flows, grouped splits, scaling, and audits."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.pcap_features import FEATURE_COLUMNS, extract_pcap_flows
from src.reproducibility import save_environment
from src.utils import ROOT, load_config, resolve_path

FEATURES = FEATURE_COLUMNS
LOG_FEATURES = [
    "packet_count",
    "byte_count",
    "flow_duration",
    "packet_count_per_second",
    "byte_count_per_second",
]


def _clean_frame(frame: pd.DataFrame, duration_floor: float) -> pd.DataFrame:
    """Keep model features and flow IDs, derive duration and recompute rates."""
    cleaned = frame.copy()
    cleaned["flow_duration"] = (
        cleaned["flow_duration_sec"].astype(float)
        + cleaned["flow_duration_nsec"].astype(float) / 1_000_000_000.0
    ).clip(lower=duration_floor)
    cleaned["packet_count_per_second"] = (
        cleaned["packet_count"].astype(float) / cleaned["flow_duration"]
    )
    cleaned["byte_count_per_second"] = (
        cleaned["byte_count"].astype(float) / cleaned["flow_duration"]
    )
    keep = [column for column in ["flow_id", *FEATURES, "label"] if column in cleaned]
    cleaned = cleaned[keep].replace([np.inf, -np.inf], np.nan).dropna()
    return cleaned


def print_label_evidence(frame: pd.DataFrame) -> None:
    """Print the CSV label counts and evidence that label 1 is TCP DDoS."""
    counts = frame["label"].value_counts().sort_index()
    print("CSV label evidence:")
    for label, count in counts.items():
        class_name = "Benign" if label == 0 else "DDoS"
        print(f"  label {label}: {count} rows -> {class_name}")
    attacks = frame[frame["label"] == 1]
    print(
        "  label 1 protocol values: "
        f"{sorted(attacks['ip_proto'].unique().tolist())}"
    )
    print(
        f"  label 1 packet/byte medians: "
        f"{attacks['packet_count'].median():.2f}/"
        f"{attacks['byte_count'].median():.2f}"
    )


def transform_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the shared log1p transform to count, duration, and rate features."""
    transformed = frame[FEATURES].astype(float).copy()
    for column in LOG_FEATURES:
        transformed[column] = np.log1p(transformed[column].clip(lower=0))
    return transformed


def build_domain_gap(
    csv_frame: pd.DataFrame,
    pcap_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Compare log-transformed feature ranges for CSV attacks and PCAP flows."""
    csv_attack = transform_features(csv_frame[csv_frame["label"] == 1])
    pcap_values = transform_features(pcap_frame)
    rows = []
    for feature in FEATURES:
        csv_values = csv_attack[feature]
        pcap_column = pcap_values[feature]
        rows.append(
            {
                "feature": feature,
                "csv_min": float(csv_values.min()),
                "csv_median": float(csv_values.median()),
                "csv_max": float(csv_values.max()),
                "pcap_min": float(pcap_column.min()),
                "pcap_median": float(pcap_column.median()),
                "pcap_max": float(pcap_column.max()),
            }
        )
    return pd.DataFrame(rows)


def stratified_group_split(
    frame: pd.DataFrame,
    seed: int = 42,
    group_column: str = "flow_id",
) -> dict[str, pd.DataFrame]:
    """Create a stratified 70/10/20 split without splitting any group."""
    group_labels = frame.groupby(group_column)["label"].first().reset_index()
    groups = group_labels[group_column].tolist()
    labels = group_labels["label"].tolist()
    if len(groups) < 3 or min(pd.Series(labels).value_counts()) < 2:
        raise ValueError("Each class needs at least two groups for a grouped split.")
    test_size = max(2, round(len(groups) * 0.2))
    test_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_size / len(groups),
        random_state=seed,
    )
    train_val_idx, test_idx = next(test_splitter.split(groups, labels))
    train_val_groups = [groups[index] for index in train_val_idx]
    train_val_labels = [labels[index] for index in train_val_idx]
    test_groups = [groups[index] for index in test_idx]
    if len(train_val_groups) < 4:
        return {
            "train": frame[frame[group_column].isin(train_val_groups)].copy(),
            "val": frame.iloc[0:0].copy(),
            "test": frame[frame[group_column].isin(test_groups)].copy(),
        }
    val_size = max(2, round(len(groups) * 0.1))
    val_fraction = val_size / len(train_val_groups)
    val_splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_fraction,
        random_state=seed,
    )
    train_idx, val_idx = next(val_splitter.split(train_val_groups, train_val_labels))
    train_groups = [train_val_groups[index] for index in train_idx]
    val_groups = [train_val_groups[index] for index in val_idx]
    return {
        "train": frame[frame[group_column].isin(train_groups)].copy(),
        "val": frame[frame[group_column].isin(val_groups)].copy(),
        "test": frame[frame[group_column].isin(test_groups)].copy(),
    }


def find_benign_pcaps(folder: str | Path) -> list[Path]:
    """Return all PCAP and PCAPNG paths in the benign capture folder."""
    path = Path(folder)
    if not path.is_dir():
        return []
    return sorted(
        item
        for item in path.iterdir()
        if item.is_file() and item.suffix.lower() in {".pcap", ".pcapng"}
    )


def cap_rows_by_session(
    frame: pd.DataFrame,
    max_rows: int,
    seed: int,
) -> pd.DataFrame:
    """Select whole sessions up to a row cap, alternating labels for balance."""
    if len(frame) <= max_rows:
        return frame.copy()
    by_label = {
        int(label): group["session_id"].drop_duplicates().sample(
            frac=1,
            random_state=seed,
        ).tolist()
        for label, group in frame.groupby("label")
    }
    selected: list[str] = []
    selected_count = 0
    labels = sorted(by_label)
    cursors = {label: 0 for label in labels}
    while selected_count < max_rows and any(
        cursors[label] < len(by_label[label]) for label in labels
    ):
        for label in labels:
            if cursors[label] >= len(by_label[label]):
                continue
            session = by_label[label][cursors[label]]
            cursors[label] += 1
            session_count = int((frame["session_id"] == session).sum())
            if selected_count + session_count <= max_rows:
                selected.append(session)
                selected_count += session_count
    return frame[frame["session_id"].isin(selected)].copy()


def build_pcap_dataset(
    attack_pcap: str | Path,
    benign_pcaps: list[Path],
    duration_floor: float,
) -> pd.DataFrame:
    """Combine attack and benign PCAP flows without including CSV rows."""
    attacks = extract_pcap_flows(attack_pcap, duration_floor)
    attacks["label"] = 1
    benign_tables = []
    for path in benign_pcaps:
        flows = extract_pcap_flows(path, duration_floor)
        flows["label"] = 0
        flows["source_file"] = path.name
        flows["session_id"] = path.name + "|" + flows["session_id"]
        benign_tables.append(flows)
    attacks["source_file"] = Path(attack_pcap).name
    attacks["session_id"] = Path(attack_pcap).name + "|" + attacks["session_id"]
    return pd.concat([attacks, *benign_tables], ignore_index=True)


def print_class_feature_summary(frame: pd.DataFrame) -> None:
    """Print raw feature median and range for each PCAP dataset class."""
    print("PCAP feature summary by class (raw units, after row cap):")
    for label, class_name in ((0, "Benign"), (1, "DDoS")):
        class_frame = frame[frame["label"] == label]
        for feature in FEATURES:
            values = class_frame[feature]
            print(
                f"  {class_name} {feature}: median={values.median():.6g}, "
                f"min={values.min():.6g}, max={values.max():.6g}"
            )


def shortcut_audit(
    training_frame: pd.DataFrame,
    seed: int,
    source: str,
) -> pd.DataFrame:
    """Measure best one-feature decision-stump accuracy on training data."""
    rows = []
    labels = training_frame["label"].astype(int)
    print(
        "Shortcut audit training split: "
        f"source={source.upper()}, rows={len(training_frame)}"
    )
    for feature in FEATURES:
        model = DecisionTreeClassifier(max_depth=1, random_state=seed)
        model.fit(training_frame[[feature]], labels)
        accuracy = float(model.score(training_frame[[feature]], labels))
        rows.append({"feature": feature, "single_feature_accuracy": accuracy})
        print(f"Shortcut audit {feature}: accuracy={accuracy:.6f}")
        if accuracy > 0.99:
            print(f"WARNING: {feature} alone has accuracy above 0.99.")
    return pd.DataFrame(rows)


def save_data_summary(
    source: str,
    before_cap: pd.Series,
    after_cap: pd.Series,
) -> None:
    """Save class counts before and after the configured row cap."""
    rows = []
    for label, class_name in ((0, "Benign"), (1, "DDoS")):
        rows.append(
            {
                "source": source,
                "class": class_name,
                "flows_before_cap": int(before_cap.get(label, 0)),
                "flows_after_cap": int(after_cap.get(label, 0)),
            }
        )
    output_path = ROOT / "results" / "data_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)


def save_split_membership(
    splits: dict[str, pd.DataFrame],
    output_dir: str | Path,
) -> None:
    """Save per-flow split IDs and labels for exact dataset reproduction."""
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    columns = [
        column
        for column in ("flow_id", "session_id", "label")
        if any(column in frame.columns for frame in splits.values())
    ]
    for name in ("train", "val", "test"):
        frame = splits[name]
        present_columns = [column for column in columns if column in frame.columns]
        split_name = "validation" if name == "val" else name
        frame[present_columns].to_csv(
            target_dir / f"{split_name}.csv",
            index=False,
        )


def build_text_rows(
    frame: pd.DataFrame,
    features: list[str] | None = None,
    include_feature_names: bool = True,
) -> pd.DataFrame:
    """Format selected normalized features as named or bare-value prompts."""
    selected_features = FEATURES if features is None else features
    unknown_features = set(selected_features) - set(FEATURES)
    if not selected_features or unknown_features:
        raise ValueError(f"Invalid prompt feature selection: {sorted(unknown_features)}")
    text_rows = []
    for row in frame.itertuples(index=False):
        if include_feature_names:
            values = [
                f"{name}: {getattr(row, name):.4f}"
                for name in selected_features
            ]
        else:
            values = [
                f"{getattr(row, name):.4f}"
                for name in selected_features
            ]
        text_rows.append(
            {
                "text": "Classify this network flow as Benign or DDoS. Flow: "
                + ", ".join(values),
                "label": int(row.label),
            }
        )
    return pd.DataFrame(text_rows, index=frame.index)


def prepare_dataset(
    csv_path: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame], pd.DataFrame]:
    """Build the primary dataset, grouped splits, scaler, domain gap, and audit."""
    config = load_config()
    csv_path = resolve_path(csv_path or config["csv_path"])
    duration_floor = float(config["duration_floor"])
    raw_csv = pd.read_csv(csv_path)
    csv_frame = _clean_frame(raw_csv, duration_floor)
    csv_frame = csv_frame[csv_frame["label"].isin([0, 1])].copy()
    csv_frame["label"] = csv_frame["label"].astype(int)
    print_label_evidence(csv_frame)

    attack_path = resolve_path(config["pcap_path"])
    attack_flows = extract_pcap_flows(attack_path, duration_floor)
    domain_gap = build_domain_gap(csv_frame, attack_flows)
    domain_gap.to_csv(ROOT / "results" / "domain_gap.csv", index=False)
    print("Domain gap:")
    print(domain_gap.to_string(index=False))

    benign_dir = resolve_path(config["benign_pcap_dir"])
    benign_paths = find_benign_pcaps(benign_dir)
    print(f"Benign PCAP folder checked: {benign_dir}")
    if benign_paths:
        print("Benign PCAP files found:")
        for path in benign_paths:
            print(f"  - {path.name} ({path.stat().st_size:,} bytes)")
    else:
        print("Benign PCAP files found: none")

    if benign_paths:
        source = "pcap"
        print("Primary dataset source: PCAP")
        primary = build_pcap_dataset(attack_path, benign_paths, duration_floor)
        before_cap = primary["label"].value_counts()
        print(
            "PCAP flows extracted before row cap: "
            f"Benign={int(before_cap.get(0, 0))}, "
            f"DDoS={int(before_cap.get(1, 0))}"
        )
        primary = cap_rows_by_session(
            primary,
            int(config["max_rows"]),
            int(config["seed"]),
        )
        group_column = "session_id"
    else:
        source = "csv"
        print("Primary dataset source: CSV")
        primary = csv_frame
        before_cap = primary["label"].value_counts()
        group_column = "flow_id"

    primary_counts = primary["label"].value_counts()
    save_data_summary(source, before_cap, primary_counts)
    unit = "flows" if source == "pcap" else "rows"
    print(
        f"Primary dataset {unit} after cap: "
        f"Benign={int(primary_counts.get(0, 0))}, "
        f"DDoS={int(primary_counts.get(1, 0))}, total={len(primary)}"
    )
    if source == "pcap":
        print_class_feature_summary(primary)

    splits = stratified_group_split(
        primary,
        seed=int(config["seed"]),
        group_column=group_column,
    )
    save_split_membership(splits, ROOT / "results" / "splits")
    save_environment(ROOT / "results" / "environment.json")
    print("Split sizes:")
    for name in ("train", "val", "test"):
        subset = splits[name]
        counts = subset["label"].value_counts()
        split_name = "validation" if name == "val" else name
        print(
            f"  {split_name}: total={len(subset)}, "
            f"Benign={int(counts.get(0, 0))}, DDoS={int(counts.get(1, 0))}"
        )
    audit = shortcut_audit(
        splits["train"],
        int(config["seed"]),
        source,
    )

    train_values = transform_features(splits["train"])
    scaler = StandardScaler().fit(train_values)
    for name, subset in splits.items():
        values = transform_features(subset)
        scaled = pd.DataFrame(
            scaler.transform(values),
            columns=FEATURES,
            index=subset.index,
        )
        splits[name] = pd.concat([subset.drop(columns=FEATURES), scaled], axis=1)
        ordered = [column for column in subset.columns if column not in FEATURES]
        splits[name] = splits[name][ordered + FEATURES]
    joblib.dump(scaler, ROOT / "results" / "feature_scaler.joblib")

    train_vectors = set(map(tuple, splits["train"][FEATURES].to_numpy()))
    test_vectors = set(map(tuple, splits["test"][FEATURES].to_numpy()))
    duplicates = len(train_vectors & test_vectors)
    print(f"Duplicate feature vectors between train and test: {duplicates}")
    audit.attrs["source"] = source
    audit.attrs["group_column"] = group_column
    audit.attrs["duplicate_count"] = duplicates
    audit.attrs["train_rows"] = len(splits["train"])
    audit.to_csv(ROOT / "results" / "shortcut_audit.csv", index=False)
    audit.attrs["csv_rows"] = len(csv_frame)
    return primary, attack_flows, domain_gap, splits, audit


if __name__ == "__main__":
    prepare_dataset()
