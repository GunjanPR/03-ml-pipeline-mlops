"""Classification metrics used for model comparison and the promotion gate."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Decision threshold that maximises F1. Choose it on validation data, never on test."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # precision/recall have one more entry than thresholds; the last point has no threshold.
    return float(thresholds[int(np.argmax(f1[:-1]))])


def classification_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_prob >= threshold)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y_true)),
        "n": int(len(y_true)),
    }


def bootstrap_auc_ci(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile bootstrap interval for ROC-AUC."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y_true[idx].min() == y_true[idx].max():
            continue
        scores.append(roc_auc_score(y_true[idx], y_prob[idx]))
    low, high = np.percentile(scores, [2.5, 97.5])
    return float(low), float(high)
