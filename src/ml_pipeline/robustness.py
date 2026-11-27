"""Robustness of the headline comparison: a clinical baseline, a tuned baseline, tuning
variance and a time-ordered split.

    python -m ml_pipeline.robustness            # writes reports/robustness.{json,md}
    python -m ml_pipeline.robustness --seeds 3  # fewer Optuna re-runs (each is ~50 trials)

Four questions a reviewer asks about the gate's decision:

1. **Clinical baseline.** Hospitals use the LACE index (Length of stay, Acuity, Charlson
   comorbidity, ED visits), not logistic regression. How does a LACE score computed from this
   extract rank patients on the same test rows?
2. **Tuned baseline.** The logistic-regression baseline ran with sklearn defaults (C = 1).
   Does tuning C on val change the comparison?
3. **Tuning variance.** One Optuna search moved test AUC by about 0.005 across machines. What
   is the spread of the XGBoost lift over several search seeds?
4. **The other split.** The gate uses a time-ordered split (since 2026-11-27: train on the
   earliest encounters, test on the latest; `encounter_id` follows time). Re-running the same
   recipes on the earlier random patient-grouped split shows how much the choice of split
   moves the decision. (Before the switch this section ran the time-ordered split and found
   the lift that led to it.)

The test split is the same gating set the promotion gate uses; nothing here changes the
registry or the champion.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import (
    EXCLUDED_DISCHARGE_IDS,
    GROUP,
    TARGET,
    clean,
    load_raw,
    split,
    xy,
)
from ml_pipeline.evaluate import bootstrap_auc_ci
from ml_pipeline.features import build_preprocessor
from ml_pipeline.promote import paired_bootstrap_auc_diff
from ml_pipeline.train import fit_baseline, fit_xgboost

log = logging.getLogger(__name__)

REPORT_JSON = Path("reports/robustness.json")
REPORT_MD = Path("reports/robustness.md")
SEED_CACHE = Path("reports/robustness_xgb_seeds.json")
C_GRID = [float(c) for c in np.logspace(-3, 2, 11)]

# ---------------------------------------------------------------------------------- LACE
# Charlson comorbidity weights from ICD-9 codes (Deyo adaptation, Quan 2005 code lists,
# simplified to code prefixes). Each entry: (weight, condition, test on the code string).


def _num(code: str) -> float | None:
    try:
        return float(code)
    except ValueError:
        return None


def _in(lo: float, hi: float):
    return lambda c: (v := _num(c)) is not None and lo <= v < hi


def _prefix(*prefixes: str):
    return lambda c: any(c.startswith(p) for p in prefixes)


CHARLSON: list[tuple[int, str, object]] = [
    (1, "myocardial_infarction", _prefix("410", "412")),
    (1, "heart_failure", lambda c: c.startswith("428") or _in(425.4, 426)(c)),
    (1, "peripheral_vascular", lambda c: _prefix("440", "441")(c) or _in(443.1, 444)(c)),
    (1, "cerebrovascular", _in(430, 439)),
    (1, "dementia", _prefix("290", "294.1", "331.2")),
    (1, "chronic_pulmonary", _in(490, 506)),
    (1, "rheumatic", _prefix("710.0", "710.1", "710.4", "714", "725")),
    (1, "peptic_ulcer", _in(531, 535)),
    (1, "mild_liver", _prefix("570", "571", "573.3", "573.4", "573.8", "573.9")),
    (2, "hemiplegia", _prefix("342", "343", "344")),
    (2, "renal", lambda c: _prefix("582", "585", "586", "588.0", "V42.0", "V45.1", "V56")(c)
     or _in(583.0, 583.8)(c)),
    (2, "malignancy", lambda c: _in(140, 173)(c) or _in(174, 196)(c) or _in(200, 209)(c)),
    (3, "severe_liver", lambda c: _in(456.0, 456.3)(c) or _in(572.2, 572.9)(c)),
    (6, "metastatic", _in(196, 200)),
    (6, "aids", _prefix("042", "043", "044")),
]  # fmt: skip
DIABETES_COMPLICATED = _in(250.4, 250.8)


def charlson(codes: list[str]) -> int:
    """Charlson index from the (at most three) coded diagnoses of one encounter.

    Every patient in this cohort has diabetes, so diabetes always scores 1, or 2 when a
    coded diagnosis shows end-organ complications (250.4-250.7)."""
    codes = [c for c in codes if c and c != "missing"]
    score = 2 if any(DIABETES_COMPLICATED(c) for c in codes) else 1
    for weight, _name, test in CHARLSON:
        if any(test(c) for c in codes):  # type: ignore[operator]
            score += weight
    return score


def lace_points(los: float, emergent: bool, cci: int, ed_visits: float) -> int:
    """LACE index (van Walraven et al., 2010). L: length of stay in days; A: 3 points for an
    emergent admission; C: Charlson index (4 or more scores 5); E: ED visits, capped at 4."""
    los = int(los)
    if los < 1:
        lp = 0
    elif los <= 3:
        lp = los
    elif los <= 6:
        lp = 4
    elif los <= 13:
        lp = 5
    else:
        lp = 7
    cp = 5 if cci >= 4 else cci
    return lp + (3 if emergent else 0) + cp + int(min(ed_visits, 4))


def lace_scores(frame: pd.DataFrame) -> np.ndarray:
    """LACE from this extract. Approximations: ED visits are `number_emergency` (previous
    year, not 6 months); acuity is `admission_type_id` 1 (emergency) or 7 (trauma centre);
    Charlson uses the three coded diagnoses the extract keeps."""
    out = []
    for row in frame[["time_in_hospital", "admission_type_id", "number_emergency",
                      "diag_1", "diag_2", "diag_3"]].itertuples(index=False):  # fmt: skip
        cci = charlson([str(row.diag_1), str(row.diag_2), str(row.diag_3)])
        emergent = str(row.admission_type_id) in {"1", "7"}
        out.append(lace_points(row.time_in_hospital, emergent, cci, row.number_emergency))
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------------- data
def frames_with_extras(settings: Settings) -> dict[str, pd.DataFrame]:
    """The same splits as `data.prepare`, plus the raw ICD-9 codes, which the processed files
    drop."""
    raw = load_raw(settings.raw_dir / "diabetic_data.csv")
    cleaned = clean(raw)
    kept = raw[~raw["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)].reset_index(drop=True)
    assert len(kept) == len(cleaned) and (kept[GROUP].values == cleaned[GROUP].values).all()
    df = cleaned.copy()
    df["encounter_id"] = kept["encounter_id"].values
    for i in (1, 2, 3):
        df[f"diag_{i}"] = kept[f"diag_{i}"].fillna("missing").astype(str).values
    gate = split(df, settings.random_seed, strategy=settings.split_strategy)
    return {"gate": gate, "df": df}  # type: ignore[dict-item]


def other_strategy(strategy: str) -> str:
    return "grouped" if strategy == "time" else "time"


SPLIT_LABELS = {
    "time": "Time-ordered split (train on the earliest 70% of encounters, test on the latest 15%)",
    "grouped": "Random patient-grouped split (the gate's split before 2026-11-27)",
}


def auc_with_ci(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    low, high = bootstrap_auc_ci(y, p)
    return {"roc_auc": float(roc_auc_score(y, p)), "ci_low": low, "ci_high": high}


def lift(y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray) -> dict[str, float]:
    low, high = paired_bootstrap_auc_diff(y, p_new, p_ref)
    diff = float(roc_auc_score(y, p_new) - roc_auc_score(y, p_ref))
    return {"lift": diff, "ci_low": low, "ci_high": high}


def tuned_logreg(splits: dict[str, pd.DataFrame], seed: int) -> tuple[Pipeline, dict]:
    """Pick C on val ROC-AUC over an 11-point log grid, then refit on train."""
    x_train, y_train = xy(splits["train"])
    x_val, y_val = xy(splits["val"])
    scores = {}
    for c in C_GRID:
        model = LogisticRegression(max_iter=3000, C=c, random_state=seed)
        pipe = Pipeline([("prep", build_preprocessor()), ("model", model)]).fit(x_train, y_train)
        scores[c] = float(roc_auc_score(y_val, pipe.predict_proba(x_val)[:, 1]))
    best = max(scores, key=lambda c: scores[c])
    model = LogisticRegression(max_iter=3000, C=best, random_state=seed)
    pipe = Pipeline([("prep", build_preprocessor()), ("model", model)]).fit(x_train, y_train)
    return pipe, {"best_C": best, "val_auc_by_C": scores}


def xgb_fixed(splits: dict[str, pd.DataFrame], params: dict, seed: int) -> Pipeline:
    from xgboost import XGBClassifier

    params = {k: v for k, v in params.items() if k != "optuna_trials"}
    model = XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1, **params)
    x_train, y_train = xy(splits["train"])
    return Pipeline([("prep", build_preprocessor()), ("model", model)]).fit(x_train, y_train)


def prob(pipe: Pipeline, frame: pd.DataFrame) -> np.ndarray:
    x, _ = xy(frame)
    return pipe.predict_proba(x)[:, 1]


# ---------------------------------------------------------------------------------- run
def run(settings: Settings, n_seeds: int) -> dict:
    built = frames_with_extras(settings)
    splits, df = built["gate"], built["df"]
    processed_test = pd.read_parquet(settings.processed_dir / "test.parquet")
    assert (splits["test"][GROUP].values == processed_test[GROUP].values).all(), (
        "splits differ from data/processed; run `dvc repro prepare` first"
    )
    test = splits["test"]
    y = test[TARGET].to_numpy()
    seed = settings.random_seed
    result: dict = {
        "split_strategy": settings.split_strategy,
        "n_test": int(len(test)),
        "positive_rate": float(y.mean()),
    }

    base = fit_baseline(splits, settings).pipeline
    p_base = prob(base, test)
    lace = lace_scores(test)
    tuned, tuned_info = tuned_logreg(splits, seed)
    p_tuned = prob(tuned, test)
    xgb_params = json.loads(Path("reports/xgboost_metrics.json").read_text())["params"]
    p_xgb = prob(xgb_fixed(splits, xgb_params, seed), test)

    result["lace"] = {
        **auc_with_ci(y, lace),
        "mean_score": float(lace.mean()),
        "high_risk_share": float((lace >= 10).mean()),
        "readmit_rate_high": float(y[lace >= 10].mean()),
        "readmit_rate_low": float(y[lace < 10].mean()),
        "logreg_minus_lace": lift(y, p_base, lace),
        "xgboost_minus_lace": lift(y, p_xgb, lace),
    }
    result["logreg_default"] = auc_with_ci(y, p_base)
    result["logreg_tuned"] = {
        **auc_with_ci(y, p_tuned),
        **tuned_info,
        "tuned_minus_default": lift(y, p_tuned, p_base),
        "xgboost_minus_tuned": lift(y, p_xgb, p_tuned),
    }
    result["xgboost_committed"] = {**auc_with_ci(y, p_xgb), "minus_default": lift(y, p_xgb, p_base)}

    # 3. tuning variance: full Optuna search per seed, cached so reruns are cheap
    cache = json.loads(SEED_CACHE.read_text()) if SEED_CACHE.exists() else {}
    seeds = [seed + i for i in range(n_seeds)]
    for s in seeds:
        key = f"{settings.split_strategy}:{s}:{settings.optuna_n_trials}"
        if key in cache:
            continue
        start = time.perf_counter()
        fitted = fit_xgboost(splits, settings.model_copy(update={"random_seed": s}))
        p = prob(fitted.pipeline, test)
        cache[key] = {
            "seed": s,
            "test_auc": float(roc_auc_score(y, p)),
            "minus_default": lift(y, p, p_base),
            "params": fitted.params,
            "seconds": round(time.perf_counter() - start, 1),
        }
        SEED_CACHE.write_text(json.dumps(cache, indent=2))
        log.info("seed %d: test AUC %.4f", s, cache[key]["test_auc"])
    runs = [cache[f"{settings.split_strategy}:{s}:{settings.optuna_n_trials}"] for s in seeds]
    lifts = np.array([r["minus_default"]["lift"] for r in runs])
    gate_pass = [
        r["minus_default"]["lift"] >= settings.min_auc_lift and r["minus_default"]["ci_low"] > 0
        for r in runs
    ]
    result["xgboost_seeds"] = {
        "n_trials": settings.optuna_n_trials,
        "runs": runs,
        "test_auc_min": float(min(r["test_auc"] for r in runs)),
        "test_auc_max": float(max(r["test_auc"] for r in runs)),
        "lift_mean": float(lifts.mean()),
        "lift_sd": float(lifts.std(ddof=1)) if len(lifts) > 1 else 0.0,
        "lift_min": float(lifts.min()),
        "lift_max": float(lifts.max()),
        "gate_would_pass": int(sum(gate_pass)),
    }

    # 4. the other split, no re-tuning (same recipes and XGBoost parameters)
    alt = other_strategy(settings.split_strategy)
    ts = split(df, seed, strategy=alt)
    yt = ts["test"][TARGET].to_numpy()
    t_base = prob(fit_baseline(ts, settings).pipeline, ts["test"])
    t_xgb = prob(xgb_fixed(ts, xgb_params, seed), ts["test"])
    t_lace = lace_scores(ts["test"])
    seen = ts["test"][GROUP].isin(set(ts["train"][GROUP])).to_numpy()
    result["other_split"] = {
        "strategy": alt,
        "n_train": int(len(ts["train"])),
        "n_test": int(len(ts["test"])),
        "test_positive_rate": float(yt.mean()),
        "test_patients_seen_in_train": float(seen.mean()),
        "logreg": auc_with_ci(yt, t_base),
        "xgboost": auc_with_ci(yt, t_xgb),
        "lace": auc_with_ci(yt, t_lace),
        "xgboost_minus_logreg": lift(yt, t_xgb, t_base),
        "logreg_minus_lace": lift(yt, t_base, t_lace),
    }
    return result


def _f(x: float) -> str:
    return f"{x:.3f}"


def _ci(d: dict) -> str:
    return f"{_f(d['roc_auc'])} ({_f(d['ci_low'])}-{_f(d['ci_high'])})"


def _lift(d: dict) -> str:
    return f"{d['lift']:+.4f} ({d['ci_low']:+.4f}, {d['ci_high']:+.4f})"


def to_markdown(r: dict) -> str:
    la, lt, sd, ts = r["lace"], r["logreg_tuned"], r["xgboost_seeds"], r["other_split"]
    rows = "\n".join(
        f"| {x['seed']} | {_f(x['test_auc'])} | {_lift(x['minus_default'])} | "
        f"depth {x['params'].get('max_depth')}, {x['params'].get('n_estimators')} trees |"
        for x in sd["runs"]
    )
    return f"""# Robustness of the model comparison

Generated by `python -m ml_pipeline.robustness` (a DVC stage). Gate split:
`{r["split_strategy"]}`. Test = the same {r["n_test"]:,} gating rows as the promotion gate
({100 * r["positive_rate"]:.1f}% readmitted).
ROC-AUC CIs are 1,000-sample bootstraps; lifts use the paired bootstrap from the gate.

## 1. Clinical baseline: LACE index

| Score | Test ROC-AUC (95% CI) |
|---|---|
| LACE (computed from this extract) | {_ci(la)} |
| Logistic regression (baseline) | {_ci(r["logreg_default"])} |
| XGBoost (committed run) | {_ci(r["xgboost_committed"])} |

Logistic regression minus LACE: {_lift(la["logreg_minus_lace"])}. XGBoost minus LACE:
{_lift(la["xgboost_minus_lace"])}. LACE >= 10 flags {100 * la["high_risk_share"]:.1f}% of
encounters; {100 * la["readmit_rate_high"]:.1f}% of them are readmitted vs
{100 * la["readmit_rate_low"]:.1f}% below 10.

Approximations: ED visits are the previous year's `number_emergency` (LACE uses 6 months),
acuity is emergency or trauma-centre admission type, and the Charlson index uses the three
diagnosis codes the extract keeps (every patient scores at least 1 for diabetes).

## 2. Tuned logistic regression

C chosen on val over an 11-point grid from 0.001 to 100: best C = {lt["best_C"]:g}.
Test ROC-AUC {_ci(lt)} vs {_ci(r["logreg_default"])} with C = 1. Tuned minus default:
{_lift(lt["tuned_minus_default"])}. XGBoost minus tuned: {_lift(lt["xgboost_minus_tuned"])}.

## 3. XGBoost tuning variance ({len(sd["runs"])} Optuna searches of {sd["n_trials"]} trials)

| Seed | Test ROC-AUC | Lift over logistic regression (95% CI) | Best parameters |
|---|---|---|---|
{rows}

Lift mean {sd["lift_mean"]:+.4f}, SD {sd["lift_sd"]:.4f}, range {sd["lift_min"]:+.4f} to
{sd["lift_max"]:+.4f}. The gate (lift >= +0.01 and CI above 0) would promote XGBoost in
{sd["gate_would_pass"]} of {len(sd["runs"])} searches.

## 4. For comparison: {SPLIT_LABELS[ts["strategy"]]}

{ts["n_train"]:,} train / {ts["n_test"]:,} test encounters,
{100 * ts["test_positive_rate"]:.1f}% readmitted in test;
{100 * ts["test_patients_seen_in_train"]:.1f}% of test encounters belong to a patient seen in
train. Same recipes and XGBoost parameters as the gate's run, no re-tuning.

| Model | Test ROC-AUC (95% CI) |
|---|---|
| Logistic regression | {_ci(ts["logreg"])} |
| XGBoost (committed parameters) | {_ci(ts["xgboost"])} |
| LACE | {_ci(ts["lace"])} |

XGBoost minus logistic regression: {_lift(ts["xgboost_minus_logreg"])}. Logistic regression
minus LACE: {_lift(ts["logreg_minus_lace"])}.
"""


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seeds", type=int, default=5, help="Optuna searches to run")
    args = parser.parse_args(argv)
    result = run(get_settings(), args.seeds)
    REPORT_JSON.write_text(json.dumps(result, indent=2))
    REPORT_MD.write_text(to_markdown(result))
    log.info("wrote %s and %s", REPORT_JSON, REPORT_MD)


if __name__ == "__main__":
    main()
