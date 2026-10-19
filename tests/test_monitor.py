import json

import numpy as np
import pytest

from conftest import fake_raw
from ml_pipeline.data import FEATURES, clean
from ml_pipeline.monitor import PREDICTION, detect_drift
from ml_pipeline.serve import DRIFT_DETECTED, DRIFT_SHARE, update_drift_gauges


@pytest.fixture(scope="module")
def frames():
    reference = clean(fake_raw(3000, seed=1))[FEATURES]
    same = clean(fake_raw(800, seed=2))[FEATURES]
    return reference, same


def _shift(frame):
    shifted = frame.copy()
    for col in ["number_inpatient", "num_medications", "time_in_hospital", "num_procedures"]:
        shifted[col] = shifted[col] * 3 + 5
    for col in ["age", "gender", "insulin", "metformin", "A1Cresult", "diag_1_group"]:
        shifted[col] = shifted[col].iloc[0]
    return shifted


def test_same_distribution_is_not_flagged(frames):
    reference, same = frames
    result = detect_drift(reference, same, threshold=0.3)
    assert not result.drifted
    assert result.drift_share < 0.3


def test_shifted_batch_is_flagged_with_the_right_columns(frames, tmp_path):
    reference, same = frames
    result = detect_drift(reference, _shift(same), threshold=0.2, html_path=tmp_path / "r.html")
    assert result.drifted
    assert {"number_inpatient", "num_medications", "age", "insulin"} <= set(result.drifted_columns)
    assert (tmp_path / "r.html").exists()


def test_prediction_column_is_checked_but_not_counted_as_a_feature(frames):
    reference, same = frames
    rng = np.random.default_rng(0)
    ref = reference.assign(**{PREDICTION: rng.uniform(0, 0.2, len(reference))})
    cur = same.assign(**{PREDICTION: rng.uniform(0.6, 0.9, len(same))})
    result = detect_drift(ref, cur, threshold=0.3)
    assert result.prediction_drifted
    assert PREDICTION not in result.drifted_columns


def test_drift_gauges_read_latest_json(tmp_path):
    path = tmp_path / "latest.json"
    update_drift_gauges(path)  # missing file is ignored
    path.write_text(
        json.dumps({"drift_share": 0.4, "drifted": True, "checked_at": "2026-10-19T00:00:00+00:00"})
    )
    update_drift_gauges(path)
    assert DRIFT_SHARE._value.get() == 0.4
    assert DRIFT_DETECTED._value.get() == 1
