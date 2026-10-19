"""Train a model, score it on val/test, log it to MLflow and register it.

    python -m ml_pipeline.train --model baseline     # logistic regression (the named baseline)
    python -m ml_pipeline.train --model xgboost      # XGBoost tuned with Optuna on val

Protocol: hyperparameters, early stopping and the F1 decision threshold are all chosen on
the validation split. The test split is scored once, after every choice is fixed. Each run
registers a new version of `REGISTERED_MODEL_NAME` in the MLflow registry with the
`challenger` alias; `python -m ml_pipeline.promote` decides whether it becomes `champion`.
Metrics are also written to `reports/<model>_metrics.json`.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import FEATURES, load_splits, xy
from ml_pipeline.evaluate import best_f1_threshold, bootstrap_auc_ci, classification_metrics
from ml_pipeline.features import build_preprocessor

log = logging.getLogger(__name__)

# MLflow 3 serialises sklearn models with skops, which only loads allow-listed types.
TRUSTED_TYPES = ["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"]

CHALLENGER = "challenger"
CHAMPION = "champion"


@dataclass
class Fitted:
    pipeline: Pipeline
    params: dict = field(default_factory=dict)


def fit_baseline(
    splits: dict[str, pd.DataFrame], settings: Settings, exclude: frozenset[str] = frozenset()
) -> Fitted:
    x_train, y_train = xy(splits["train"])
    model = LogisticRegression(max_iter=2000, random_state=settings.random_seed)
    pipeline = Pipeline([("prep", build_preprocessor(exclude)), ("model", model)])
    pipeline.fit(x_train, y_train)
    return Fitted(pipeline, {"C": model.C})


def fit_xgboost(
    splits: dict[str, pd.DataFrame], settings: Settings, exclude: frozenset[str] = frozenset()
) -> Fitted:
    """Optuna (TPE) search maximising val ROC-AUC, with early stopping on val."""
    import optuna
    from sklearn.metrics import roc_auc_score
    from xgboost import XGBClassifier

    x_train, y_train = xy(splits["train"])
    x_val, y_val = xy(splits["val"])
    prep = build_preprocessor(exclude).fit(x_train)
    xt, xv = prep.transform(x_train), prep.transform(x_val)

    def make(params: dict, n_estimators: int, early_stopping: int | None) -> XGBClassifier:
        return XGBClassifier(
            n_estimators=n_estimators,
            tree_method="hist",
            eval_metric="auc",
            early_stopping_rounds=early_stopping,
            random_state=settings.random_seed,
            n_jobs=-1,
            **params,
        )

    def objective(trial: optuna.Trial) -> float:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1, 50, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True),
            "gamma": trial.suggest_float("gamma", 0, 5),
        }
        model = make(params, 2000, early_stopping=50)
        model.fit(xt, y_train, eval_set=[(xv, y_val)], verbose=False)
        trial.set_user_attr("n_estimators", int(model.best_iteration) + 1)
        return float(roc_auc_score(y_val, model.predict_proba(xv)[:, 1]))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=settings.random_seed)
    )

    def progress(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        if (trial.number + 1) % 10 == 0:
            log.info("optuna trial %d: best val ROC-AUC %.4f", trial.number + 1, study.best_value)

    study.optimize(objective, n_trials=settings.optuna_n_trials, callbacks=[progress])
    best = study.best_trial
    n_estimators = best.user_attrs["n_estimators"]
    log.info("optuna best val ROC-AUC %.4f after %d trials", best.value, len(study.trials))

    pipeline = Pipeline(
        [("prep", build_preprocessor(exclude)), ("model", make(best.params, n_estimators, None))]
    )
    pipeline.fit(x_train, y_train)
    return Fitted(
        pipeline,
        {**best.params, "n_estimators": n_estimators, "optuna_trials": len(study.trials)},
    )


TRAINERS = {"baseline": fit_baseline, "xgboost": fit_xgboost}


def feature_names(pipeline: Pipeline) -> list[str]:
    return [n.split("__", 1)[1] for n in pipeline["prep"].get_feature_names_out()]


def shap_artifacts(pipeline: Pipeline, x: pd.DataFrame, out_dir, seed: int, n: int = 2000):
    """Mean |SHAP| per feature (CSV) and a beeswarm summary plot (PNG) on `n` rows of `x`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    sample = x.sample(min(n, len(x)), random_state=seed)
    transformed = pipeline["prep"].transform(sample)
    dense = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    names = feature_names(pipeline)
    model = pipeline["model"]
    if hasattr(model, "get_booster"):
        values = shap.TreeExplainer(model).shap_values(dense)
    else:
        background = shap.maskers.Independent(dense, max_samples=200)
        values = shap.LinearExplainer(model, background).shap_values(dense)
    importance = (
        pd.DataFrame({"feature": names, "mean_abs_shap": np.abs(values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    csv_path = out_dir / "shap_importance.csv"
    importance.to_csv(csv_path, index=False)
    plt.figure()
    shap.summary_plot(values, dense, feature_names=names, max_display=20, show=False)
    png_path = out_dir / "shap_summary.png"
    plt.savefig(png_path, dpi=120, bbox_inches="tight")
    plt.close("all")
    return importance, [csv_path, png_path]


KEY_FEATURES = 5


def key_features(importance: pd.DataFrame, n: int = KEY_FEATURES) -> list[str]:
    """Top raw features by summed mean |SHAP| over their one-hot columns."""
    by_raw: dict[str, float] = {}
    for name, value in zip(importance["feature"], importance["mean_abs_shap"], strict=True):
        raw = next(
            (f for f in sorted(FEATURES, key=len, reverse=True)
             if name == f or name.startswith(f + "_")),
            name,
        )  # fmt: skip
        by_raw[raw] = by_raw.get(raw, 0.0) + float(value)
    return [f for f, _ in sorted(by_raw.items(), key=lambda kv: -kv[1])[:n]]


def train(
    model: str = "baseline",
    settings: Settings | None = None,
    shap: bool = True,
    exclude: frozenset[str] = frozenset(),
    register: bool = True,
) -> dict:
    """Train, evaluate and log one model. `exclude` drops features (ablations); ablation runs
    pass `register=False` so they never reach the registry or the default report files."""
    settings = settings or get_settings()
    splits = load_splits(settings)
    x_val, y_val = xy(splits["val"])
    x_test, y_test = xy(splits["test"])

    start = time.perf_counter()
    fitted = TRAINERS[model](splits, settings, exclude)
    fit_seconds = time.perf_counter() - start
    pipeline = fitted.pipeline

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
        "n_train": len(splits["train"]),
        "params": fitted.params,
        "excluded_features": sorted(exclude),
        "val": val,
        "test": test,
        "fit_seconds": fit_seconds,
        "predict_ms_per_1k_rows": 1000 * predict_seconds / len(y_test) * 1000,
    }

    run_name = model if not exclude else f"{model}-without-{'-'.join(sorted(exclude))}"
    out_dir = settings.reports_dir / (model if register else f"ablation/{run_name}")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "test_proba.npy", p_test)  # used for paired comparisons in promote.py

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tag("model_kind", model)
        mlflow.set_tag("excluded_features", ",".join(sorted(exclude)))
        mlflow.log_params({"model": model, "seed": settings.random_seed, **fitted.params})
        mlflow.log_metrics({f"val_{k}": v for k, v in val.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test.items()})
        mlflow.log_metrics(
            {"fit_seconds": fit_seconds, "predict_ms_per_1k_rows": result["predict_ms_per_1k_rows"]}
        )
        if shap:
            importance, paths = shap_artifacts(pipeline, x_test, out_dir, settings.random_seed)
            result["top_features"] = importance.head(10).to_dict("records")
            result["key_features"] = key_features(importance)
            for path in paths:
                mlflow.log_artifact(str(path), artifact_path="explainability")
        info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            input_example=x_val.head(5),
            registered_model_name=settings.registered_model_name if register else None,
            skops_trusted_types=TRUSTED_TYPES,
        )
        result["mlflow_run_id"] = run.info.run_id

    if not register:
        (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
        log.info("%s test ROC-AUC %.4f (not registered)", run_name, test["roc_auc"])
        return result

    client = MlflowClient()
    version = info.registered_model_version
    for key, value in {
        "model_kind": model,
        "threshold": threshold,
        "val_roc_auc": val["roc_auc"],
        "test_roc_auc": test["roc_auc"],
        "key_features": ",".join(result.get("key_features", [])),
    }.items():
        client.set_model_version_tag(settings.registered_model_name, version, key, str(value))
    client.set_registered_model_alias(settings.registered_model_name, CHALLENGER, version)
    result["registered_version"] = int(version)

    out = settings.reports_dir / f"{model}_metrics.json"
    out.write_text(json.dumps(result, indent=2))
    log.info(
        "%s v%s test ROC-AUC %.4f [%.4f, %.4f] PR-AUC %.4f F1 %.4f Brier %.4f (fit %.1fs) -> %s",
        model,
        version,
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
    parser.add_argument("--model", choices=sorted(TRAINERS), default="baseline")
    parser.add_argument("--no-shap", action="store_true", help="skip SHAP artifacts")
    parser.add_argument("--trials", type=int, help="override OPTUNA_N_TRIALS")
    parser.add_argument(
        "--exclude", default="", help="comma-separated features to drop (ablation; not registered)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = get_settings()
    if args.trials:
        settings = settings.model_copy(update={"optuna_n_trials": args.trials})
    exclude = frozenset(filter(None, args.exclude.split(",")))
    unknown = exclude - set(FEATURES)
    if unknown:
        parser.error(f"unknown features: {sorted(unknown)}")
    train(args.model, settings, shap=not args.no_shap, exclude=exclude, register=not exclude)


if __name__ == "__main__":
    main()
