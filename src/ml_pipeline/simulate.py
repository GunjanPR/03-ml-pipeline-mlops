"""Drift simulation: does monitoring catch realistic shifts, and how much do they hurt?

    python -m ml_pipeline.simulate                   # all scenarios, 10 batches each
    python -m ml_pipeline.simulate --batches 20 --batch-size 1000

Batches are drawn from the held-out **test** split (patients never seen in training), so
each batch has labels and the champion's ROC-AUC can be measured on it. Scenarios:

* `no_shift` - random batches. Any alarm here is a false alarm.
* `older_sicker` - case mix shifts to patients aged 70+ with prior inpatient stays.
* `emergency_intake` - a new site where almost every admission comes through the ER.
* `icd10_migration` - half of the diagnosis codes arrive in a coding system the ICD-9
  grouping does not recognise (the 2015 ICD-9 -> ICD-10 switch).
* `lab_panel_change` - a new admission lab panel adds about 15 lab procedures per stay.

Writes `reports/drift_simulation.json` and `reports/drift_simulation.md`. With `--e2e`, one
shifted batch is also sent through the real API (`/predict`) and the monitor reads it back
from the inference log, to check the whole loop.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import FEATURES, TARGET, load_splits
from ml_pipeline.monitor import PREDICTION, detect_drift, reference_frame

log = logging.getLogger(__name__)

Scenario = Callable[[pd.DataFrame, np.random.Generator, int], pd.DataFrame]


def _weighted(pool: pd.DataFrame, rng, n: int, weights: np.ndarray) -> pd.DataFrame:
    idx = rng.choice(len(pool), n, replace=False, p=weights / weights.sum())
    return pool.iloc[idx].reset_index(drop=True)


def no_shift(pool, rng, n):
    return pool.iloc[rng.choice(len(pool), n, replace=False)].reset_index(drop=True)


def older_sicker(pool, rng, n):
    old = pool["age"].isin(["[70-80)", "[80-90)", "[90-100)"])
    sick = pool["number_inpatient"] >= 1
    return _weighted(pool, rng, n, np.where(old & sick, 20.0, np.where(old | sick, 3.0, 1.0)))


def emergency_intake(pool, rng, n):
    er = pool["admission_source_id"] == "7"
    return _weighted(pool, rng, n, np.where(er, 20.0, 1.0))


def icd10_migration(pool, rng, n):
    batch = no_shift(pool, rng, n)
    for col in ("diag_1_group", "diag_2_group", "diag_3_group"):
        recoded = rng.random(n) < 0.5
        batch.loc[recoded, col] = "other"  # unknown ICD-10 codes fall into "other"
    return batch


def lab_panel_change(pool, rng, n):
    batch = no_shift(pool, rng, n)
    batch["num_lab_procedures"] = batch["num_lab_procedures"] + rng.poisson(15, n)
    return batch


SCENARIOS: dict[str, Scenario] = {
    "no_shift": no_shift,
    "older_sicker": older_sicker,
    "emergency_intake": emergency_intake,
    "icd10_migration": icd10_migration,
    "lab_panel_change": lab_panel_change,
}


def simulate(
    settings: Settings | None = None, batches: int = 10, batch_size: int = 1000
) -> pd.DataFrame:
    import mlflow
    import mlflow.sklearn

    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model = mlflow.sklearn.load_model(settings.resolved_model_uri)
    reference = reference_frame(settings, model)
    pool = load_splits(settings)["test"]
    rows = []
    for name, scenario in SCENARIOS.items():
        for b in range(batches):
            rng = np.random.default_rng([settings.random_seed, b, len(rows)])
            batch = scenario(pool, rng, batch_size)
            proba = model.predict_proba(batch[FEATURES])[:, 1]
            current = batch[FEATURES].assign(**{PREDICTION: proba})
            result = detect_drift(reference, current, settings.drift_threshold)
            rows.append(
                {
                    "scenario": name,
                    "batch": b,
                    "drift_share": result.drift_share,
                    "drifted": result.drifted,
                    "prediction_drifted": result.prediction_drifted,
                    "drifted_columns": ", ".join(result.drifted_columns),
                    "roc_auc": roc_auc_score(batch[TARGET], proba),
                    "positive_rate": batch[TARGET].mean(),
                    "mean_prediction": proba.mean(),
                }
            )
        log.info("%s: %d batches", name, batches)
    return pd.DataFrame(rows)


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    summary = results.groupby("scenario", sort=False).agg(
        batches=("batch", "size"),
        alarm_rate=("drifted", "mean"),
        mean_drift_share=("drift_share", "mean"),
        prediction_drift_rate=("prediction_drifted", "mean"),
        roc_auc=("roc_auc", "mean"),
        positive_rate=("positive_rate", "mean"),
        mean_prediction=("mean_prediction", "mean"),
    )
    top = results.assign(col=results["drifted_columns"].str.split(", ")).explode("col")
    top = top[top["col"] != ""]
    summary["most_flagged"] = (
        top.groupby("scenario")["col"]
        .agg(lambda s: ", ".join(s.value_counts().head(3).index))
        .reindex(summary.index)
        .fillna("")
    )
    return summary


def end_to_end(settings: Settings, batch_size: int) -> dict:
    """Send one shifted batch through /predict, then run the monitor on the inference log."""
    import tempfile
    from pathlib import Path

    from fastapi.testclient import TestClient

    from ml_pipeline.monitor import monitor
    from ml_pipeline.serve import create_app

    pool = load_splits(settings)["test"]
    batch = older_sicker(pool, np.random.default_rng(settings.random_seed), batch_size)
    with tempfile.TemporaryDirectory() as tmp:
        local = settings.model_copy(update={"inference_log_path": Path(tmp) / "inference.jsonl"})
        with TestClient(create_app(local)) as client:
            for start in range(0, len(batch), 250):
                records = batch[FEATURES].iloc[start : start + 250].to_dict("records")
                client.post("/predict", json={"records": records}).raise_for_status()
        result = monitor(local)
    return {"scenario": "older_sicker", "rows": result.rows, "drifted": result.drifted,
            "drift_share": result.drift_share}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batches", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--e2e", action="store_true", help="also run one batch through the API")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    results = simulate(settings, args.batches, args.batch_size)
    summary = summarise(results)
    payload = {
        "batch_size": args.batch_size,
        "drift_threshold": settings.drift_threshold,
        "summary": summary.reset_index().to_dict("records"),
    }
    if args.e2e:
        payload["end_to_end"] = end_to_end(settings, args.batch_size)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "drift_simulation.json").write_text(json.dumps(payload, indent=2))
    (settings.reports_dir / "drift_simulation.md").write_text(
        summary.to_markdown(floatfmt=".3f"), encoding="utf-8"
    )
    log.info("\n%s", summary.to_string(float_format="%.3f"))


if __name__ == "__main__":
    main()
