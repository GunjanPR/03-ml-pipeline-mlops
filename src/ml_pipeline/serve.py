"""FastAPI inference service for the registered champion model.

    uvicorn ml_pipeline.serve:app --port 8020        # or: make serve

Endpoints: `POST /predict`, `GET /health` (process up), `GET /ready` (model loaded),
`GET /model` (served version and threshold), `GET /metrics` (Prometheus).
Every prediction is appended to `INFERENCE_LOG_PATH` (JSON lines) for drift monitoring.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest
from pydantic import BaseModel, ConfigDict, Field, create_model

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES
from ml_pipeline.fastpath import build_fast_scorer

log = logging.getLogger(__name__)

PREDICTIONS = Counter("mlp_predictions_total", "Rows scored", ["model_version"])
POSITIVES = Counter("mlp_positive_predictions_total", "Rows above threshold", ["model_version"])
LATENCY = Histogram(
    "mlp_predict_latency_seconds",
    "Latency of /predict, including logging",
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
BATCH = Histogram("mlp_predict_batch_size", "Rows per /predict call", buckets=(1, 10, 100, 1000))
SCORE = Histogram(
    "mlp_predicted_probability",
    "Distribution of predicted readmission probability",
    buckets=tuple(i / 20 for i in range(1, 21)),
)
ERRORS = Counter("mlp_predict_errors_total", "Failed /predict calls", ["kind"])
PATH = Counter("mlp_predict_path_total", "/predict calls by scoring path", ["path"])

# Requests up to this many rows use the verified fast encoder (see ml_pipeline.fastpath);
# larger batches go through the sklearn pipeline, which is faster per row at volume.
FAST_PATH_MAX_ROWS = 32
MODEL_VERSION = Gauge("mlp_model_version", "Registry version currently served")
DRIFT_SHARE = Gauge("mlp_drift_share", "Share of drifted features in the last check")
DRIFT_DETECTED = Gauge("mlp_drift_detected", "1 if the last drift check crossed the threshold")
DRIFT_CHECKED = Gauge("mlp_drift_last_check_timestamp", "Unix time of the last drift check")


def _field_name(feature: str) -> str:
    return feature.replace("-", "_")


Encounter: type[BaseModel] = create_model(
    "Encounter",
    __config__=ConfigDict(populate_by_name=True, extra="forbid"),
    **{_field_name(f): (float, Field(..., alias=f, ge=0)) for f in NUMERIC_FEATURES},
    **{_field_name(f): (str, Field(..., alias=f, min_length=1)) for f in CATEGORICAL_FEATURES},
)


class PredictRequest(BaseModel):
    records: list[Encounter] = Field(..., min_length=1)  # type: ignore[valid-type]


class Prediction(BaseModel):
    probability: float
    readmit_30d: bool


class PredictResponse(BaseModel):
    model_version: str
    threshold: float
    predictions: list[Prediction]


def update_drift_gauges(path: Path) -> None:
    """Expose the latest `ml_pipeline.monitor` result (if any) as gauges."""
    try:
        latest = json.loads(path.read_text())
    except (OSError, ValueError):
        return
    DRIFT_SHARE.set(latest["drift_share"])
    DRIFT_DETECTED.set(int(latest["drifted"]))
    DRIFT_CHECKED.set(datetime.fromisoformat(latest["checked_at"]).timestamp())


class ModelState:
    def __init__(self) -> None:
        self.model: Any = None
        self.version = "unloaded"
        self.threshold = 0.5
        self.error: str | None = None
        self.scorer = None
        self.lock = threading.Lock()

    def load(self, settings: Settings) -> None:
        import mlflow
        import mlflow.sklearn
        from mlflow import MlflowClient

        bundle = settings.model_bundle
        uri = str(bundle / "model") if bundle else settings.resolved_model_uri
        try:
            mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
            self.model = mlflow.sklearn.load_model(uri)
            self.version, self.threshold = "unknown", 0.5
            if bundle:
                metadata = json.loads((bundle / "metadata.json").read_text())
                self.version, self.threshold = metadata["version"], metadata["threshold"]
            elif uri.startswith("models:/") and "@" in uri:
                name, alias = uri.removeprefix("models:/").split("@")
                mv = MlflowClient().get_model_version_by_alias(name, alias)
                self.version = str(mv.version)
                self.threshold = float(mv.tags.get("threshold", 0.5))
            MODEL_VERSION.set(float(self.version) if self.version.isdigit() else -1)
            self.scorer = build_fast_scorer(self.model, _reference_sample(settings))
            self.error = None
            log.info("serving %s (version %s, threshold %.4f)", uri, self.version, self.threshold)
        except Exception as exc:  # noqa: BLE001 - keep the process up; /ready reports it
            self.model, self.error = None, f"{type(exc).__name__}: {exc}"
            log.error("could not load %s: %s", uri, self.error)


def _reference_sample(settings: Settings) -> pd.DataFrame | None:
    """Rows to verify the fast path against: the bundle's reference sample, else training
    data when it is available locally."""
    try:
        if settings.model_bundle:
            return pd.read_parquet(settings.model_bundle / "reference.parquet")
        from ml_pipeline.monitor import reference_frame

        return reference_frame(settings)
    except Exception:  # noqa: BLE001 - no reference means no fast path, not a failure
        return None


class InferenceLog:
    """Appends each scored record as a JSON line. Writes happen on a background thread, so
    disk I/O never adds to request latency; `close()` flushes what is queued."""

    def __init__(self, path) -> None:
        self.path = path
        self.queue: queue.Queue = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True, name="inference-log")
        self.thread.start()

    def write(self, records: list[dict], proba, version: str) -> None:
        self.queue.put((records, [float(p) for p in proba], version, datetime.now(UTC)))

    def _run(self) -> None:
        while True:
            item = self.queue.get()
            try:
                if item is None:
                    return
                records, proba, version, ts = item
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as fh:
                    for record, p in zip(records, proba, strict=True):
                        row = {**record, "prediction": p, "model_version": version,
                               "ts": ts.isoformat()}  # fmt: skip
                        fh.write(json.dumps(row) + "\n")
            except Exception:  # noqa: BLE001 - logging must never take the API down
                log.exception("inference log write failed")
            finally:
                self.queue.task_done()

    def flush(self) -> None:
        self.queue.join()

    def close(self) -> None:
        self.queue.put(None)
        self.thread.join(timeout=5)


def create_app(settings: Settings | None = None, state: ModelState | None = None) -> FastAPI:
    settings = settings or get_settings()
    state = state or ModelState()
    inference_log = InferenceLog(settings.inference_log_path)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state.model is None:
            state.load(settings)
        yield
        inference_log.close()

    app = FastAPI(title="Readmission risk API", version="0.1.0", lifespan=lifespan)
    app.state.model_state = state
    app.state.inference_log = inference_log

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        if state.model is None:
            raise HTTPException(503, detail=state.error or "model not loaded")
        return {"status": "ready", "model_version": state.version}

    @app.get("/model")
    def model_info() -> dict:
        return {
            "uri": str(settings.model_bundle or settings.resolved_model_uri),
            "version": state.version,
            "threshold": state.threshold,
            "features": FEATURES,
        }

    @app.post("/model/reload")
    def reload() -> dict:
        with state.lock:
            state.load(settings)
        if state.model is None:
            raise HTTPException(503, detail=state.error)
        return {"status": "reloaded", "model_version": state.version}

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        if state.model is None:
            ERRORS.labels("not_ready").inc()
            raise HTTPException(503, detail="model not loaded")
        if len(request.records) > settings.max_batch_size:
            ERRORS.labels("batch_too_large").inc()
            raise HTTPException(413, detail=f"at most {settings.max_batch_size} records per call")
        start = time.perf_counter()
        records = [r.model_dump(by_alias=True) for r in request.records]
        model, scorer = state.model, state.scorer
        version, threshold = state.version, state.threshold
        if scorer is not None and len(records) <= FAST_PATH_MAX_ROWS:
            proba = scorer.predict_proba(records)
            PATH.labels("fast").inc()
        else:
            proba = model.predict_proba(pd.DataFrame(records)[FEATURES])[:, 1]
            PATH.labels("sklearn").inc()
        inference_log.write(records, proba, version)
        flags = proba >= threshold
        PREDICTIONS.labels(version).inc(len(proba))
        POSITIVES.labels(version).inc(int(flags.sum()))
        BATCH.observe(len(proba))
        for p in proba:
            SCORE.observe(float(p))
        LATENCY.observe(time.perf_counter() - start)
        return PredictResponse(
            model_version=version,
            threshold=threshold,
            predictions=[
                Prediction(probability=float(p), readmit_30d=bool(f))
                for p, f in zip(proba, flags, strict=True)
            ],
        )

    @app.get("/metrics")
    def metrics() -> Response:
        update_drift_gauges(settings.reports_dir / "drift" / "latest.json")
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
