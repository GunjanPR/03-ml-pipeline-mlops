import numpy as np
import pytest

from ml_pipeline.evaluate import best_f1_threshold, bootstrap_auc_ci, classification_metrics


def test_perfect_scores():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    m = classification_metrics(y, p, threshold=0.5)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0
    assert m["f1"] == 1.0
    assert m["brier"] == pytest.approx(0.025)


def test_best_f1_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7])
    threshold = best_f1_threshold(y, p)
    assert 0.3 < threshold <= 0.6


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p = np.clip(y * 0.3 + rng.random(500) * 0.7, 0, 1)
    low, high = bootstrap_auc_ci(y, p, n_boot=200)
    point = classification_metrics(y, p, 0.5)["roc_auc"]
    assert low < point < high
