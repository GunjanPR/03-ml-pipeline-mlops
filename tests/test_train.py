import json

import pytest
from mlflow import MlflowClient

from conftest import fake_raw
from ml_pipeline.data import clean, split
from ml_pipeline.promote import promote
from ml_pipeline.train import CHALLENGER, CHAMPION, train


@pytest.fixture
def prepared(settings):
    settings.processed_dir.mkdir(parents=True)
    for name, frame in split(clean(fake_raw(900)), seed=0).items():
        frame.to_parquet(settings.processed_dir / f"{name}.parquet", index=False)
    return settings


def test_baseline_trains_logs_registers_and_reports(prepared):
    result = train("baseline", prepared)

    assert result["test"]["roc_auc"] > 0.6  # the fake data has a real signal
    assert result["registered_version"] == 1
    saved = json.loads((prepared.reports_dir / "baseline_metrics.json").read_text())
    assert saved["test"]["n"] == result["test"]["n"]
    assert (prepared.reports_dir / "baseline" / "shap_summary.png").exists()
    assert result["top_features"][0]["feature"]

    version = MlflowClient().get_model_version_by_alias(prepared.registered_model_name, CHALLENGER)
    assert version.tags["model_kind"] == "baseline"
    assert 0 < float(version.tags["threshold"]) < 1


def test_first_baseline_becomes_champion_then_xgboost_goes_through_the_gate(prepared):
    train("baseline", prepared, shap=False)
    assert promote(prepared) is None
    client = MlflowClient()
    name = prepared.registered_model_name
    assert str(client.get_model_version_by_alias(name, CHAMPION).version) == "1"

    result = train("xgboost", prepared)
    assert result["params"]["optuna_trials"] == 3
    assert (prepared.reports_dir / "xgboost" / "shap_importance.csv").exists()

    gate = promote(prepared)
    assert gate.challenger_version == 2 and gate.reference_version == 1
    champion = str(client.get_model_version_by_alias(name, CHAMPION).version)
    assert champion == ("2" if gate.passed else "1")
    assert json.loads((prepared.reports_dir / "gate.json").read_text())["passed"] == gate.passed


def test_gate_always_leaves_a_champion_on_first_comparison(prepared):
    # The DVC order: both models trained before the first gate run.
    train("baseline", prepared, shap=False)
    train("xgboost", prepared, shap=False)
    gate = promote(prepared)
    champion = MlflowClient().get_model_version_by_alias(prepared.registered_model_name, CHAMPION)
    assert str(champion.version) == ("2" if gate.passed else "1")


def test_ablation_run_is_logged_but_not_registered(prepared):
    train("baseline", prepared, shap=False)
    result = train(
        "baseline", prepared, shap=False, exclude=frozenset({"prior_encounters"}), register=False
    )
    assert result["excluded_features"] == ["prior_encounters"]
    versions = MlflowClient().search_model_versions(f"name='{prepared.registered_model_name}'")
    assert len(versions) == 1
    assert (prepared.reports_dir / "ablation" / "baseline-without-prior_encounters").exists()
