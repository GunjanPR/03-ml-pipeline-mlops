import numpy as np
import pandas as pd
import pytest

from ml_pipeline.age_gap import (
    BANDS,
    band_thresholds,
    comorbidity_flags,
    gap_reduction_ci,
    markdown,
    per_band_auc,
    policy_metrics,
)


def test_comorbidity_flags_match_any_of_the_three_codes():
    codes = pd.DataFrame(
        {
            "code_1": ["428", "250.13", "486", "missing"],
            "code_2": ["missing", "V58", "585", "250.02"],
            "code_3": ["427.31", "missing", "missing", "E888"],
        }
    )
    flags = comorbidity_flags(codes)
    assert flags["cc_heart_failure"].tolist() == [1, 0, 0, 0]
    assert flags["cc_arrhythmia"].tolist() == [1, 0, 0, 0]
    assert flags["cc_dka_or_coma"].tolist() == [0, 1, 0, 0]
    assert flags["cc_pneumonia"].tolist() == [0, 0, 1, 0]
    assert flags["cc_renal"].tolist() == [0, 0, 1, 0]
    # fifth digit 2 or 3 = uncontrolled: 250.13 (DKA, type 1) and 250.02
    assert flags["cc_uncontrolled_diabetes"].tolist() == [0, 1, 0, 1]


def _bands(n_per_band: int) -> np.ndarray:
    return np.repeat(np.array(BANDS), n_per_band)


def test_band_thresholds_reach_the_target_sensitivity_in_every_band():
    rng = np.random.default_rng(0)
    bands = _bands(2000)
    y = (rng.random(len(bands)) < 0.2).astype(int)
    # each band has its own score scale, so one global threshold treats them differently
    shift = pd.Series(bands).map(dict(zip(BANDS, [0.0, 0.1, 0.2, 0.3], strict=True))).to_numpy()
    p = 0.3 * y + rng.random(len(y)) * 0.5 + shift
    thresholds = band_thresholds(y, p, bands, target_tpr=0.6)
    metrics = policy_metrics(y, p, bands, thresholds, seed=0)
    for b in BANDS:
        assert metrics["tpr"][b] == pytest.approx(0.6, abs=0.01)
    single = policy_metrics(y, p, bands, {b: 0.6 for b in BANDS}, seed=0)
    assert single["tpr_gap"] > 0.3 > metrics["tpr_gap"]
    low, high = gap_reduction_ci(y, p, bands, {b: 0.6 for b in BANDS}, thresholds, seed=0)
    assert 0 < low <= high


def test_per_band_auc_and_gap():
    rng = np.random.default_rng(1)
    bands = _bands(1000)
    y = (rng.random(len(bands)) < 0.3).astype(int)
    noise = np.where(bands == "<40", 0.1, 1.0)  # scores are informative for the young only
    p = y + rng.normal(0, 1, len(y)) * noise
    result = per_band_auc(y, p, bands)
    assert result["<40"] > 0.95 > 0.8 > result["80+"]
    assert result["gap"] == pytest.approx(
        max(result[b] for b in BANDS) - min(result[b] for b in BANDS)
    )


def test_markdown_renders_every_section():
    policy = {
        "thresholds": dict.fromkeys(BANDS, 0.1),
        "tpr": dict.fromkeys(BANDS, 0.5),
        "tpr_ci": {b: [0.4, 0.6] for b in BANDS},
        "tpr_gap": 0.1,
        "tpr_gap_ci": [0.0, 0.2],
        "overall_tpr": 0.5,
        "flag_rate": 0.25,
        "precision": 0.2,
    }
    signal = {
        "n": 10,
        "readmission_rate": 0.1,
        "feature_auc": {},
        "readmission_rate_prior_inpatient": 0.2,
        "readmission_rate_no_prior_inpatient": 0.05,
        "share_discharged_home": 0.5,
        "share_acute_or_uncontrolled_diabetes": 0.1,
        "share_heart_failure": 0.1,
    }
    report = {
        "champion_refit_test_auc": 0.668,
        "committed_test_auc": 0.668,
        "signal": {b: {**signal, "feature_auc": {f: 0.6 for f in (
            "number_inpatient", "number_emergency", "prior_encounters",
            "prior_readmissions_30d", "total_prior_visits", "number_diagnoses")}}
            for b in BANDS},
        "young_phenotype_test": {
            "acute_or_uncontrolled": {"n": 3, "readmission_rate": 0.1, "roc_auc": 0.8},
            "other": {"n": 5, "readmission_rate": 0.1, "roc_auc": 0.7},
        },
        "model_attempts": {"lr": {"all": 0.65, **dict.fromkeys(BANDS, 0.6), "gap": 0.1}},
        "threshold_policies": {
            "target_tpr_val": 0.41, "global": policy, "per_band": policy,
            "gap_reduction_ci": [-0.1, 0.1],
        },
    }  # fmt: skip
    text = markdown(report)
    for heading in ("## 1.", "## 2.", "## 3.", "paired bootstrap"):
        assert heading in text
