import numpy as np

from ml_pipeline.promote import decide, paired_bootstrap_auc_diff


def _scores(n: int = 3000, seed: int = 0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    weak = np.clip(0.5 + 0.1 * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    strong = np.clip(0.5 + 0.4 * (y - 0.5) + rng.normal(0, 0.2, n), 0.01, 0.99)
    return y, weak, strong


def test_clearly_better_challenger_passes():
    y, weak, strong = _scores()
    passed, stats = decide(y, strong, weak, min_auc_lift=0.01)
    assert passed, stats["reasons"]
    assert stats["auc_lift_ci_low"] > 0


def test_worse_challenger_is_blocked():
    y, weak, strong = _scores()
    passed, stats = decide(y, weak, strong, min_auc_lift=0.01)
    assert not passed
    assert any("AUC lift" in r for r in stats["reasons"])


def test_identical_model_is_blocked():
    y, _, strong = _scores()
    passed, stats = decide(y, strong, strong.copy(), min_auc_lift=0.0)
    assert not passed  # zero lift: the CI cannot exclude 0
    assert stats["auc_lift"] == 0


def test_miscalibrated_challenger_is_blocked_on_brier():
    y, _, strong = _scores()
    shifted = np.clip(strong * 0.3 + 0.6, 0, 1)  # same ranking, worse calibration
    passed, stats = decide(y, shifted, strong, min_auc_lift=-1.0)
    assert not passed
    assert any("Brier" in r for r in stats["reasons"])


def test_paired_bootstrap_interval_brackets_point_difference():
    y, weak, strong = _scores()
    low, high = paired_bootstrap_auc_diff(y, strong, weak, n_boot=300)
    assert 0 < low < high
