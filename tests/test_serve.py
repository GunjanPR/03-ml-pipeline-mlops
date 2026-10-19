import json

import pytest
from fastapi.testclient import TestClient

from conftest import fake_raw
from ml_pipeline.data import FEATURES, clean, split
from ml_pipeline.serve import ModelState, create_app
from ml_pipeline.train import fit_baseline


@pytest.fixture(scope="module")
def frames():
    return split(clean(fake_raw(900)), seed=0)


@pytest.fixture
def client(frames, settings):
    state = ModelState()
    state.model = fit_baseline(frames, settings).pipeline
    state.version, state.threshold = "7", 0.2
    with TestClient(create_app(settings, state)) as c:
        yield c


def _records(frames, n=3):
    return frames["test"][FEATURES].head(n).to_dict("records")


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json()["model_version"] == "7"


def test_predict_returns_one_probability_per_record_and_logs_inputs(client, frames, settings):
    response = client.post("/predict", json={"records": _records(frames)})
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "7"
    assert len(body["predictions"]) == 3
    for p in body["predictions"]:
        assert 0 <= p["probability"] <= 1
        assert p["readmit_30d"] == (p["probability"] >= 0.2)
    client.app.state.inference_log.flush()
    logged = settings.inference_log_path.read_text().splitlines()
    assert len(logged) == 3
    assert json.loads(logged[0])["model_version"] == "7"


def test_unseen_category_is_accepted(client, frames):
    record = _records(frames, 1)[0] | {"medical_specialty": "Never-Seen-Before"}
    assert client.post("/predict", json={"records": [record]}).status_code == 200


@pytest.mark.parametrize(
    "mutate",
    [
        lambda r: r.pop("insulin"),  # missing feature (name contains no dash)
        lambda r: r.pop("glyburide-metformin"),  # missing feature with a dash in its name
        lambda r: r.update(num_medications=-1),  # negative count
        lambda r: r.update(extra_field="x"),  # unknown field
    ],
)
def test_invalid_records_are_rejected(client, frames, mutate):
    record = _records(frames, 1)[0]
    mutate(record)
    assert client.post("/predict", json={"records": [record]}).status_code == 422


def test_empty_and_oversized_batches(client, frames, settings):
    assert client.post("/predict", json={"records": []}).status_code == 422
    too_many = _records(frames, 1) * (settings.max_batch_size + 1)
    assert client.post("/predict", json={"records": too_many}).status_code == 413


def test_metrics_exposed(client, frames):
    client.post("/predict", json={"records": _records(frames)})
    text = client.get("/metrics").text
    assert "mlp_predictions_total" in text
    assert "mlp_predict_latency_seconds" in text


def test_not_ready_without_model(settings):
    settings = settings.model_copy(update={"model_uri": "models:/does-not-exist@champion"})
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.post("/predict", json={"records": [{}]}).status_code in (422, 503)
