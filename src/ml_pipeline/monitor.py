"""Data-drift monitoring with Evidently.

    python -m ml_pipeline.monitor                          # inference log vs training data
    python -m ml_pipeline.monitor --current batch.parquet  # any batch vs training data
    python -m ml_pipeline.monitor --retrain                # also retrain + gate on drift

Each feature (and the model's predicted probability) is compared with the training split
using Evidently's per-column tests (K-S / chi-square for small samples, Wasserstein /
Jensen-Shannon distance for large ones). The batch counts as drifted when the share of
drifted columns reaches `DRIFT_THRESHOLD`. Results go to `reports/drift/`: an HTML report
and `latest.json`, which the API exposes as Prometheus gauges.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, load_splits

log = logging.getLogger(__name__)

PREDICTION = "prediction"
REFERENCE_ROWS = 20_000


@dataclass
class DriftResult:
    rows: int
    drift_share: float
    threshold: float
    drifted: bool
    drifted_columns: list[str] = field(default_factory=list)
    prediction_drifted: bool = False
    column_scores: dict[str, dict] = field(default_factory=dict)
    checked_at: str = ""
    report_html: str = ""


def reference_frame(settings: Settings, model=None) -> pd.DataFrame:
    """Training rows (sampled), with the model's predictions when a model is given."""
    train = load_splits(settings)["train"]
    sample = train.sample(min(REFERENCE_ROWS, len(train)), random_state=settings.random_seed)
    ref = sample[FEATURES].reset_index(drop=True)
    if model is not None:
        ref[PREDICTION] = model.predict_proba(ref)[:, 1]
    return ref


def read_inference_log(path: Path, last: int | None = None) -> pd.DataFrame:
    frame = pd.read_json(path, lines=True)
    if last:
        frame = frame.tail(last)
    return frame


def _column_drifted(metric: dict) -> tuple[str, dict] | None:
    config = metric.get("config", {})
    if not config.get("type", "").endswith("ValueDrift"):
        return None
    method, threshold, value = config["method"], config["threshold"], metric["value"]
    # p-value tests drift when small; distance metrics drift when large.
    drifted = value < threshold if "p_value" in method else value >= threshold
    return config["column"], {
        "method": method,
        "score": float(value),
        "threshold": float(threshold),
        "drifted": bool(drifted),
    }


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    threshold: float,
    html_path: Path | None = None,
) -> DriftResult:
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset

    numeric = [*NUMERIC_FEATURES]
    if PREDICTION in reference.columns and PREDICTION in current.columns:
        numeric.append(PREDICTION)
    columns = numeric + CATEGORICAL_FEATURES
    definition = DataDefinition(numerical_columns=numeric, categorical_columns=CATEGORICAL_FEATURES)
    snapshot = Report([DataDriftPreset(drift_share=threshold)]).run(
        Dataset.from_pandas(current[columns].reset_index(drop=True), data_definition=definition),
        Dataset.from_pandas(reference[columns], data_definition=definition),
    )
    scores = dict(filter(None, (_column_drifted(m) for m in snapshot.dict()["metrics"])))
    features = {k: v for k, v in scores.items() if k != PREDICTION}
    drifted_columns = sorted(k for k, v in features.items() if v["drifted"])
    share = len(drifted_columns) / len(features)
    if html_path is not None:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot.save_html(str(html_path))
    return DriftResult(
        rows=len(current),
        drift_share=share,
        threshold=threshold,
        drifted=share >= threshold,
        drifted_columns=drifted_columns,
        prediction_drifted=bool(scores.get(PREDICTION, {}).get("drifted", False)),
        column_scores=scores,
        checked_at=datetime.now(UTC).isoformat(),
        report_html=str(html_path or ""),
    )


def monitor(
    settings: Settings | None = None,
    current: pd.DataFrame | None = None,
    retrain: bool = False,
) -> DriftResult:
    import mlflow
    import mlflow.sklearn

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(settings.resolved_model_uri)
    if current is None:
        current = read_inference_log(settings.inference_log_path)
    if PREDICTION not in current.columns:
        current = current.assign(**{PREDICTION: model.predict_proba(current[FEATURES])[:, 1]})

    out_dir = settings.reports_dir / "drift"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    result = detect_drift(
        reference_frame(settings, model),
        current,
        settings.drift_threshold,
        html_path=out_dir / f"drift_{stamp}.html",
    )
    (out_dir / "latest.json").write_text(json.dumps(asdict(result), indent=2))
    log.info(
        "drift share %.2f (threshold %.2f) on %d rows -> %s%s",
        result.drift_share,
        result.threshold,
        result.rows,
        "DRIFT" if result.drifted else "ok",
        f": {', '.join(result.drifted_columns)}" if result.drifted_columns else "",
    )
    if result.drifted and retrain:
        retrain_and_gate(settings)
    return result


def retrain_and_gate(settings: Settings) -> None:
    """Retrain the challenger and run the promotion gate (the drift-triggered job)."""
    from ml_pipeline.promote import promote
    from ml_pipeline.train import train

    log.info("drift detected: retraining challenger")
    train("xgboost", settings, shap=False)
    promote(settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--current", type=Path, help="parquet/CSV batch (default: inference log)")
    parser.add_argument("--retrain", action="store_true", help="retrain + gate when drifted")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    current = None
    if args.current:
        reader = pd.read_parquet if args.current.suffix == ".parquet" else pd.read_csv
        current = reader(args.current)
    result = monitor(current=current, retrain=args.retrain)
    sys.exit(3 if result.drifted else 0)


if __name__ == "__main__":
    main()
