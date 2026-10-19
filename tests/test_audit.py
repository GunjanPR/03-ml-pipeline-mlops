import json

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.audit import (
    audit_model,
    expected_calibration_error,
    load_model_outputs,
    markdown,
    subgroups,
    top_share_capture,
)


def test_ece_is_zero_when_predictions_match_observed_rates():
    rng = np.random.default_rng(0)
    p = np.repeat([0.1, 0.5], 5000)
    y = (rng.random(len(p)) < p).astype(int)
    assert expected_calibration_error(y, p) < 0.02
    assert expected_calibration_error(y, np.clip(p * 2, 0, 1)) > 0.1  # over-predicts


def test_top_share_capture():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    p = np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    top = top_share_capture(y, p, 0.2)  # flags rows 0 and 2
    assert top["flagged"] == 2
    assert top["recall"] == pytest.approx(0.5)
    assert top["precision"] == pytest.approx(0.5)
    assert top["lift"] == pytest.approx(2.5)


def test_subgroup_labels():
    frame = pd.DataFrame(
        {"age": ["[10-20)", "[50-60)", "[70-80)", "[90-100)"], "gender": ["Male"] * 4,
         "race": ["Asian", "missing", "Caucasian", "Hispanic"]}
    )  # fmt: skip
    groups = subgroups(frame)
    assert groups["age"].tolist() == ["<40", "40-59", "60-79", "80+"]
    assert groups["race"].tolist() == ["Asian", "missing", "Caucasian", "Hispanic"]


def test_audit_reports_a_gap_the_model_has():
    """Scores carry signal for one group only: the audit must show that group's lower AUC."""
    rng = np.random.default_rng(1)
    n = 4000
    frame = pd.DataFrame(
        {
            "age": rng.choice(["[50-60)", "[70-80)"], n),
            "gender": rng.choice(["Male", "Female"], n),
            "race": rng.choice(["A", "B"], n),
        }
    )
    y = (rng.random(n) < 0.2).astype(int)
    informative = frame["race"].eq("A").to_numpy()
    p = np.where(informative, 0.1 + 0.3 * y, 0.2) + rng.normal(0, 0.05, n)
    p = np.clip(p, 0.01, 0.99)
    report = audit_model(frame, y, p, threshold=0.3, seed=0)
    race = report["subgroups"]["race"]["groups"]
    assert race["A"]["roc_auc"] > 0.9 and race["B"]["roc_auc"] < 0.6
    assert race["A"]["roc_auc_ci"][0] < race["A"]["roc_auc"] < race["A"]["roc_auc_ci"][1]
    assert report["subgroups"]["race"]["auc_gap"] > 0.3
    assert race["B"]["tpr"] < race["A"]["tpr"]
    md = markdown({"n_test": n, "models": {"m": report}})
    assert "| race | B |" in md and "AUC gap" in md


def test_small_groups_get_no_auc():
    rng = np.random.default_rng(2)
    n = 500
    frame = pd.DataFrame(
        {"age": ["[50-60)"] * n, "gender": ["Male"] * n,
         "race": ["A"] * (n - 40) + ["B"] * 40}
    )  # fmt: skip
    y = (rng.random(n) < 0.1).astype(int)
    report = audit_model(frame, y, rng.random(n), threshold=0.5, seed=0)
    assert report["subgroups"]["race"]["groups"]["B"]["roc_auc"] is None


def test_loads_registered_and_ablation_outputs(tmp_path):
    from ml_pipeline.config import Settings

    settings = Settings(reports_dir=tmp_path)
    for name, report in [
        ("baseline", "baseline_metrics.json"),
        ("ablation/baseline-without-race", "ablation/baseline-without-race/metrics.json"),
    ]:  # noqa: E501
        (tmp_path / name).mkdir(parents=True)
        np.save(tmp_path / name / "test_proba.npy", np.array([0.1, 0.9]))
        (tmp_path / report).write_text(json.dumps({"test": {"threshold": 0.25}}))
    outputs = load_model_outputs(settings, ["baseline", "ablation/baseline-without-race"])
    assert {k: v[1] for k, v in outputs.items()} == {
        "baseline": 0.25,
        "ablation/baseline-without-race": 0.25,
    }
