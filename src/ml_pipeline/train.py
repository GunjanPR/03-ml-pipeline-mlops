"""Train a model, score it on val/test, and log the run to MLflow.

    python -m ml_pipeline.train                    # baseline logistic regression
    python -m ml_pipeline.train --model baseline

The F1 threshold is chosen on validation data; test is scored once with it. Metrics are
also written to `reports/<model>_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable

import mlflow
import mlflow.sklearn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import load_splits, xy
from ml_pipeline.evaluate import best_f1_threshold, bootstrap_auc_ci, classification_metrics
from ml_pipeline.features import build_preprocessor

log = logging.getLogger(__name__)


def logistic_baseline(seed: int) -> BaseEstimator:
    return LogisticRegression(max_iter=2000, random_state=seed)


MODELS: dict[str, Callable[[int], BaseEstimator]] = {"baseline": logistic_baseline}


def build_pipeline(model: str, seed: int) -> Pipeline:
    return Pipeline([("prep", build_preprocessor()), ("model", MODELS[model](seed))])


def train(model: str = "baseline", settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    splits = load_splits(settings)
    x_train, y_train = xy(splits["train"])
    x_val, y_val = xy(splits["val"])
    x_test, y_test = xy(splits["test"])

    pipeline = build_pipeline(model, settings.random_seed)
    start = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start

    p_val = pipeline.predict_proba(x_val)[:, 1]
    threshold = best_f1_threshold(y_val, p_val)
    start = time.perf_counter()
    p_test = pipeline.predict_proba(x_test)[:, 1]
    predict_seconds = time.perf_counter() - start

    val = classification_metrics(y_val, p_val, threshold)
    test = classification_metrics(y_test, p_test, threshold)
    test["roc_auc_ci_low"], test["roc_auc_ci_high"] = bootstrap_auc_ci(
        y_test, p_test, seed=settings.random_seed
    )
    result = {
        "model": model,
        "n_train": len(y_train),
        "val": val,
        "test": test,
        "fit_seconds": fit_seconds,
        "predict_ms_per_1k_rows": 1000 * predict_seconds / len(y_test) * 1000,
    }

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=model) as run:
        mlflow.log_params({"model": model, "seed": settings.random_seed, "n_train": len(y_train)})
        mlflow.log_params({f"model__{k}": v for k, v in pipeline["model"].get_params().items()})
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics(
            {
                "fit_seconds": fit_seconds,
                "predict_ms_per_1k_rows": result["predict_ms_per_1k_rows"],
            }
        )
        mlflow.sklearn.log_model(pipeline, name="model", input_example=x_val.head(5))
        result["mlflow_run_id"] = run.info.run_id

    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    out = settings.reports_dir / f"{model}_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    log.info(
        "%s test ROC-AUC %.4f [%.4f, %.4f] PR-AUC %.4f F1 %.4f Brier %.4f (fit %.1fs) -> %s",
        model,
        test["roc_auc"],
        test["roc_auc_ci_low"],
        test["roc_auc_ci_high"],
        test["pr_auc"],
        test["f1"],
        test["brier"],
        fit_seconds,
        out,
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=sorted(MODELS), default="baseline")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    train(args.model)


if __name__ == "__main__":
    main()
