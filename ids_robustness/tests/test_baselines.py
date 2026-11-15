"""Test classical baseline fitting and binary metric calculations."""

from __future__ import annotations

import numpy as np

from src.baselines import (
    classification_metrics,
    make_classical_model,
    wilson_interval,
)


def test_classical_baselines_fit_tiny_synthetic_data() -> None:
    """Fit each requested estimator and calculate its metrics on a tiny split."""
    features = np.concatenate(
        [np.linspace(-1.0, -0.1, 20), np.linspace(0.1, 1.0, 20)]
    ).reshape(-1, 1)
    labels = np.array([0] * 20 + [1] * 20)
    for name in (
        "LogisticRegression",
        "RandomForest",
        "HistGradientBoosting",
    ):
        model = make_classical_model(name, seed=42)
        model.fit(features, labels)
        probabilities = model.predict_proba(features)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(labels, predictions, probabilities)

        assert metrics["accuracy"] == 1.0
        assert metrics["macro_f1"] == 1.0
        assert metrics["roc_auc"] == 1.0
        assert metrics["confusion_matrix"] == [[20, 0], [0, 20]]
        assert len(metrics["roc_fpr"]) > 0


def test_wilson_interval_handles_all_detected_and_no_samples() -> None:
    """Return bounded confidence intervals and NaNs when a sample is empty."""
    lower, upper = wilson_interval(10, 10)
    empty_lower, empty_upper = wilson_interval(0, 0)

    assert 0 < lower < upper <= 1
    assert np.isnan(empty_lower)
    assert np.isnan(empty_upper)
