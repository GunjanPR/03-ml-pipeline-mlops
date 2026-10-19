import numpy as np
import pytest
from fastapi.testclient import TestClient

from conftest import fake_raw
from ml_pipeline.data import FEATURES, clean, split
from ml_pipeline.fastpath import FastScorer, build_fast_scorer
from ml_pipeline.serve import FAST_PATH_MAX_ROWS, PATH, ModelState, create_app
from ml_pipeline.train import fit_baseline, fit_xgboost


@pytest.fixture(scope="module")
def frames():
    return split(clean(fake_raw(900)), seed=0)


def _with_unseen_values(frame):
    odd = frame.copy()
    odd.loc[odd.index[:5], "medical_specialty"] = "Never-Seen"  # unknown category
    odd.loc[odd.index[5:10], "gender"] = "Unknown/Invalid"  # unseen value of a binary column
    return odd


@pytest.mark.parametrize("trainer", [fit_baseline, fit_xgboost])
def test_fast_path_matches_sklearn_exactly(frames, settings, trainer):
    pipeline = trainer(frames, settings).pipeline
    test = _with_unseen_values(frames["test"][FEATURES])
    scorer = FastScorer(pipeline)
    fast = scorer.predict_proba(test.to_dict("records"))
    np.testing.assert_allclose(fast, pipeline.predict_proba(test)[:, 1], atol=1e-9)
    assert build_fast_scorer(pipeline, frames["val"][FEATURES]) is not None


def test_fast_path_is_refused_without_a_reference_or_on_mismatch(frames, settings, monkeypatch):
    import ml_pipeline.fastpath as fastpath

    pipeline = fit_baseline(frames, settings).pipeline
    assert build_fast_scorer(pipeline, None) is None
    broken = FastScorer(pipeline)
    broken.mean = broken.mean + 1.0  # simulate a layout bug
    monkeypatch.setattr(fastpath, "FastScorer", lambda _: broken)
    assert fastpath.build_fast_scorer(pipeline, frames["val"][FEATURES]) is None


def test_api_uses_fast_path_for_small_requests_and_sklearn_for_large(frames, settings):
    state = ModelState()
    state.model = fit_baseline(frames, settings).pipeline
    state.scorer = FastScorer(state.model)
    state.version = "1"
    records = frames["test"][FEATURES].head(FAST_PATH_MAX_ROWS + 1).to_dict("records")
    fast_before = PATH.labels("fast")._value.get()
    slow_before = PATH.labels("sklearn")._value.get()
    with TestClient(create_app(settings, state)) as client:
        small = client.post("/predict", json={"records": records[:2]}).json()
        large = client.post("/predict", json={"records": records}).json()
    assert PATH.labels("fast")._value.get() == fast_before + 1
    assert PATH.labels("sklearn")._value.get() == slow_before + 1
    # both paths agree on the shared rows
    for a, b in zip(small["predictions"], large["predictions"][:2], strict=True):
        assert a["probability"] == pytest.approx(b["probability"], abs=1e-9)
