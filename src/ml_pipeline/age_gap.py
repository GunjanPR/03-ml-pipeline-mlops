"""Why does ROC-AUC fall with age, and what can be done about it?

    python -m ml_pipeline.age_gap     # writes reports/age_gap.json and reports/age_gap.md

The audit (`ml_pipeline.audit`) shows the champion ranks patients under 40 far better than
patients over 80 (test AUC 0.745 vs 0.619). This module answers three questions, with the
same protocol as training (choices on train/val, test touched once at the end):

1. **Where is the signal?** Univariate AUC of the prior-utilisation features and the
   readmission rate with and without a prior inpatient stay, per age band (train split),
   plus the share of young patients with acute or uncontrolled diabetes (DKA, coma,
   "uncontrolled" ICD-9 250.x2/250.x3), a phenotype that keeps coming back.
2. **Can a better model close the gap?** Train on train, score val by age band: the logistic
   baseline, the baseline plus comorbidity flags from the full ICD-9 codes (heart failure,
   renal disease, COPD, ...), one logistic model per age band, and XGBoost with the tuned
   parameters from `reports/xgboost_metrics.json`, with and without the flags.
3. **Can the deployed decision be made fairer?** One global threshold gives each age band a
   different sensitivity. Per-band thresholds are fitted on val so every band reaches the
   val sensitivity of the global threshold, then the test split measures the sensitivity
   gap, the alert volume and precision under both policies (bootstrap 95% CIs).
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline import data as D
from ml_pipeline.audit import AGE_BANDS, N_BOOT
from ml_pipeline.config import Settings, get_settings
from ml_pipeline.features import MIN_CATEGORY_COUNT

log = logging.getLogger(__name__)

BANDS = tuple(AGE_BANDS)
UTILISATION = [
    "number_inpatient",
    "number_emergency",
    "prior_encounters",
    "prior_readmissions_30d",
    "total_prior_visits",
    "number_diagnoses",
]
CODE_COLUMNS = ["code_1", "code_2", "code_3"]


def _num(code: str) -> float | None:
    try:
        return float(code)
    except ValueError:
        return None


def _in(lo: float, hi: float):
    return lambda c, v: v is not None and lo <= v < hi


# Comorbidity flags from ICD-9 codes (Elixhauser/Charlson-style prefixes). Any of the three
# recorded diagnoses counts.
CONDITIONS = {
    "heart_failure": lambda c, v: c.startswith(("428", "404")),
    "renal": lambda c, v: c.startswith(("403", "V56")) or _in(584, 587)(c, v),
    "copd": _in(490, 497),
    "arrhythmia": lambda c, v: c.startswith("427"),
    "pneumonia": _in(480, 487),
    "uti_sepsis": lambda c, v: c.startswith(("599", "038", "995.9")),
    "dementia": lambda c, v: c.startswith(("290", "294", "331")),
    "injury": _in(800, 830),
    "anemia": _in(280, 286),
    "cancer": _in(140, 209),
    # 250.1x ketoacidosis, 250.2x hyperosmolarity, 250.3x coma
    "dka_or_coma": lambda c, v: c.startswith(("250.1", "250.2", "250.3")),
    # fifth digit 2/3 = "uncontrolled"
    "uncontrolled_diabetes": lambda c, v: c.startswith("250") and len(c) >= 6 and c[5] in "23",
}
FLAGS = [f"cc_{name}" for name in CONDITIONS]


def comorbidity_flags(codes: pd.DataFrame) -> pd.DataFrame:
    """0/1 column per condition, set when any of `code_1..3` matches."""
    out = {}
    for name, rule in CONDITIONS.items():
        hit = np.zeros(len(codes), dtype=bool)
        for col in CODE_COLUMNS:
            hit |= codes[col].map(lambda c, rule=rule: bool(rule(c, _num(c)))).to_numpy()
        out[f"cc_{name}"] = hit.astype(float)
    return pd.DataFrame(out, index=codes.index)


def splits_with_codes(settings: Settings) -> dict[str, pd.DataFrame]:
    """The processed splits plus the raw ICD-9 codes and comorbidity flags.

    Rebuilds `data.clean` + `data.split` with three extra columns. The split depends only on
    row order and patient ids, so the rows match `data/processed/*.parquet`; that is checked.
    """
    raw = D.load_raw(D.fetch(settings))
    codes = D.add_history_features(raw)
    codes = codes[~codes["discharge_disposition_id"].isin(D.EXCLUDED_DISCHARGE_IDS)]
    codes = pd.DataFrame(
        {f"code_{i}": codes[f"diag_{i}"].fillna("missing").astype(str) for i in (1, 2, 3)}
    ).reset_index(drop=True)
    frame = pd.concat([D.clean(raw), codes], axis=1)
    frames = D.split(frame, seed=settings.random_seed)
    processed = D.load_splits(settings)
    for name, f in frames.items():
        ref = processed[name]
        same = len(f) == len(ref) and (f[D.GROUP].to_numpy() == ref[D.GROUP].to_numpy()).all()
        if not same or not (f[D.TARGET].to_numpy() == ref[D.TARGET].to_numpy()).all():
            raise SystemExit(f"{name}: rebuilt split does not match data/processed; re-run prepare")
        frames[name] = pd.concat([f, comorbidity_flags(f[CODE_COLUMNS])], axis=1)
    return frames


def age_band(frame: pd.DataFrame) -> np.ndarray:
    band = {raw: name for name, raws in AGE_BANDS.items() for raw in raws}
    return frame["age"].map(band).to_numpy()


def per_band_auc(y: np.ndarray, p: np.ndarray, bands: np.ndarray) -> dict[str, float]:
    out = {"all": float(roc_auc_score(y, p))}
    for b in BANDS:
        mask = bands == b
        out[b] = float(roc_auc_score(y[mask], p[mask]))
    out["gap"] = max(out[b] for b in BANDS) - min(out[b] for b in BANDS)
    return out


# ---- 1. where is the signal ---------------------------------------------------------------


def signal_by_band(train: pd.DataFrame) -> dict:
    bands, y = age_band(train), train[D.TARGET].to_numpy()
    rows = {}
    for b in BANDS:
        g, yb = train[bands == b], y[bands == b]
        prior = g["number_inpatient"].to_numpy() > 0
        rows[b] = {
            "n": int(len(g)),
            "readmission_rate": float(yb.mean()),
            "feature_auc": {f: float(roc_auc_score(yb, g[f])) for f in UTILISATION},
            "share_prior_inpatient": float(prior.mean()),
            "readmission_rate_prior_inpatient": float(yb[prior].mean()),
            "readmission_rate_no_prior_inpatient": float(yb[~prior].mean()),
            "share_discharged_home": float((g["discharge_disposition_id"] == "1").mean()),
            "share_acute_or_uncontrolled_diabetes": float(
                ((g["cc_dka_or_coma"] + g["cc_uncontrolled_diabetes"]) > 0).mean()
            ),
            "share_heart_failure": float(g["cc_heart_failure"].mean()),
        }
    return rows


def phenotype_auc(test: pd.DataFrame, p: np.ndarray) -> dict:
    """Champion test AUC inside the youngest band, split by the diabetes-crisis phenotype."""
    young = age_band(test) == "<40"
    crisis = ((test["cc_dka_or_coma"] + test["cc_uncontrolled_diabetes"]) > 0).to_numpy()
    y = test[D.TARGET].to_numpy()
    out = {}
    for label, mask in (("acute_or_uncontrolled", young & crisis), ("other", young & ~crisis)):
        out[label] = {
            "n": int(mask.sum()),
            "readmission_rate": float(y[mask].mean()),
            "roc_auc": float(roc_auc_score(y[mask], p[mask])),
        }
    return out


# ---- 2. can a better model close the gap --------------------------------------------------


def _prep(numeric: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), numeric),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT
                ),
                D.CATEGORICAL_FEATURES,
            ),
        ]
    )


def _logistic(numeric: list[str], seed: int, c: float = 1.0) -> Pipeline:
    model = LogisticRegression(max_iter=3000, C=c, random_state=seed)
    return Pipeline([("prep", _prep(numeric)), ("model", model)])


def _xgboost(numeric: list[str], params: dict, seed: int) -> Pipeline:
    from xgboost import XGBClassifier

    model = XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1, **params)
    return Pipeline([("prep", _prep(numeric)), ("model", model)])


def model_attempts(train: pd.DataFrame, val: pd.DataFrame, settings: Settings) -> dict:
    seed = settings.random_seed
    y_tr, y_va = train[D.TARGET].to_numpy(), val[D.TARGET].to_numpy()
    bands_tr, bands_va = age_band(train), age_band(val)
    base, flags = list(D.NUMERIC_FEATURES), list(D.NUMERIC_FEATURES) + FLAGS
    tuned = json.loads((settings.reports_dir / "xgboost_metrics.json").read_text())["params"]
    tuned = {k: v for k, v in tuned.items() if k != "optuna_trials"}
    results = {}

    def score(name: str, pipeline: Pipeline, numeric: list[str]) -> np.ndarray:
        cols = numeric + D.CATEGORICAL_FEATURES
        pipeline.fit(train[cols], y_tr)
        p = pipeline.predict_proba(val[cols])[:, 1]
        results[name] = per_band_auc(y_va, p, bands_va)
        log.info("%s: %s", name, {k: round(v, 4) for k, v in results[name].items()})
        return p

    score("logistic regression (champion recipe)", _logistic(base, seed), base)
    score("+ comorbidity flags", _logistic(flags, seed), flags)
    per_band = np.zeros(len(val))
    cols = flags + D.CATEGORICAL_FEATURES
    for b in BANDS:
        model = _logistic(flags, seed, c=0.3).fit(train[cols][bands_tr == b], y_tr[bands_tr == b])
        per_band[bands_va == b] = model.predict_proba(val[cols][bands_va == b])[:, 1]
    results["one logistic model per age band (+ flags)"] = per_band_auc(y_va, per_band, bands_va)
    score("XGBoost (tuned params)", _xgboost(base, tuned, seed), base)
    score("XGBoost + comorbidity flags", _xgboost(flags, tuned, seed), flags)
    return results


# ---- 3. per-band thresholds ---------------------------------------------------------------


def band_thresholds(
    y: np.ndarray, p: np.ndarray, bands: np.ndarray, target_tpr: float
) -> dict[str, float]:
    """Per band, the threshold whose sensitivity on (y, p) is `target_tpr`."""
    out = {}
    for b in BANDS:
        positives = np.sort(p[(bands == b) & (y == 1)])
        out[b] = float(np.quantile(positives, 1 - target_tpr))
    return out


def policy_metrics(
    y: np.ndarray, p: np.ndarray, bands: np.ndarray, thresholds: dict[str, float], seed: int
) -> dict:
    cut = np.array([thresholds[b] for b in bands])
    flagged = p >= cut
    rng = np.random.default_rng(seed)

    def tprs(idx: np.ndarray) -> dict[str, float]:
        yi, fi, bi = y[idx], flagged[idx], bands[idx]
        return {b: float(fi[(bi == b) & (yi == 1)].mean()) for b in BANDS}

    point = tprs(np.arange(len(y)))
    gaps, per_band = [], {b: [] for b in BANDS}
    for _ in range(N_BOOT):
        draw = tprs(rng.integers(0, len(y), len(y)))
        gaps.append(max(draw.values()) - min(draw.values()))
        for b in BANDS:
            per_band[b].append(draw[b])
    return {
        "thresholds": thresholds,
        "tpr": point,
        "tpr_ci": {b: [float(x) for x in np.percentile(per_band[b], [2.5, 97.5])] for b in BANDS},
        "tpr_gap": max(point.values()) - min(point.values()),
        "tpr_gap_ci": [float(x) for x in np.percentile(gaps, [2.5, 97.5])],
        "overall_tpr": float(flagged[y == 1].mean()),
        "flag_rate": float(flagged.mean()),
        "precision": float(y[flagged].mean()),
    }


def threshold_policies(
    val: pd.DataFrame, test: pd.DataFrame, p_val: np.ndarray, p_test: np.ndarray,
    threshold: float, seed: int,
) -> dict:  # fmt: skip
    y_va, y_te = val[D.TARGET].to_numpy(), test[D.TARGET].to_numpy()
    b_va, b_te = age_band(val), age_band(test)
    target = float((p_val[y_va == 1] >= threshold).mean())
    per_band = band_thresholds(y_va, p_val, b_va, target)
    single = {b: threshold for b in BANDS}
    return {
        "target_tpr_val": target,
        "global": policy_metrics(y_te, p_test, b_te, single, seed),
        "per_band": policy_metrics(y_te, p_test, b_te, per_band, seed),
        "gap_reduction_ci": gap_reduction_ci(y_te, p_test, b_te, single, per_band, seed),
    }


def gap_reduction_ci(
    y: np.ndarray, p: np.ndarray, bands: np.ndarray, before: dict, after: dict, seed: int
) -> list[float]:
    """Paired bootstrap 95% CI of (sensitivity gap before) - (gap after) on the same rows."""
    flags = [p >= np.array([t[b] for b in bands]) for t in (before, after)]
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yi, bi = y[idx], bands[idx]
        gaps = []
        for f in flags:
            tpr = [f[idx][(bi == b) & (yi == 1)].mean() for b in BANDS]
            gaps.append(max(tpr) - min(tpr))
        diffs.append(gaps[0] - gaps[1])
    return [float(x) for x in np.percentile(diffs, [2.5, 97.5])]


# ---- report -------------------------------------------------------------------------------


def markdown(report: dict) -> str:
    sig = report["signal"]
    lines = [
        "# Why AUC falls with age",
        "",
        "Generated by `python -m ml_pipeline.age_gap`. Sections 1-2 use train and val only; "
        "section 3 fits thresholds on val and measures them once on test "
        f"({N_BOOT} bootstrap resamples).",
        "",
        "## 1. Where the signal is (train split)",
        "",
        "| | " + " | ".join(BANDS) + " |",
        "|---|" + "---|" * len(BANDS),
        "| Encounters | " + " | ".join(f"{sig[b]['n']:,}" for b in BANDS) + " |",
        "| Readmitted <30 d | "
        + " | ".join(f"{sig[b]['readmission_rate']:.1%}" for b in BANDS)
        + " |",
    ]
    for f in UTILISATION:
        lines.append(
            f"| AUC of `{f}` alone | "
            + " | ".join(f"{sig[b]['feature_auc'][f]:.3f}" for b in BANDS)
            + " |"
        )
    rows = [
        ("Readmitted, prior inpatient stay", "readmission_rate_prior_inpatient"),
        ("Readmitted, no prior inpatient stay", "readmission_rate_no_prior_inpatient"),
        ("Discharged home", "share_discharged_home"),
        ("Acute or uncontrolled diabetes code", "share_acute_or_uncontrolled_diabetes"),
        ("Heart failure code", "share_heart_failure"),
    ]
    for label, key in rows:
        lines.append(f"| {label} | " + " | ".join(f"{sig[b][key]:.1%}" for b in BANDS) + " |")
    ph = report["young_phenotype_test"]
    lines += [
        "",
        "Champion test AUC inside the under-40 band: "
        f"{ph['acute_or_uncontrolled']['roc_auc']:.3f} for the "
        f"{ph['acute_or_uncontrolled']['n']} encounters with an acute or uncontrolled "
        f"diabetes code, {ph['other']['roc_auc']:.3f} for the other {ph['other']['n']}.",
        "",
        "## 2. Models tried (train -> val ROC-AUC)",
        "",
        "| Model | All | " + " | ".join(BANDS) + " | Gap |",
        "|---|---|" + "---|" * len(BANDS) + "---|",
    ]
    for name, r in report["model_attempts"].items():
        lines.append(
            f"| {name} | {r['all']:.4f} | "
            + " | ".join(f"{r[b]:.4f}" for b in BANDS)
            + f" | {r['gap']:.3f} |"
        )
    pol = report["threshold_policies"]
    lines += [
        "",
        "## 3. Decision thresholds (champion, test split)",
        "",
        f"Per-band thresholds are fitted on val so each band reaches the val sensitivity of "
        f"the global threshold ({pol['target_tpr_val']:.3f}).",
        "",
        "| Policy | Thresholds | "
        + " | ".join(f"Sensitivity {b}" for b in BANDS)
        + " | Sensitivity gap | Overall sensitivity | Flagged | Precision |",
        "|---|---|" + "---|" * len(BANDS) + "---|---|---|---|",
    ]
    for name in ("global", "per_band"):
        m = pol[name]
        thr = ", ".join(f"{m['thresholds'][b]:.3f}" for b in BANDS)
        cells = [
            f"{m['tpr'][b]:.3f} ({m['tpr_ci'][b][0]:.3f}-{m['tpr_ci'][b][1]:.3f})" for b in BANDS
        ]
        lines.append(
            f"| {name.replace('_', '-')} | {thr} | "
            + " | ".join(cells)
            + f" | {m['tpr_gap']:.3f} ({m['tpr_gap_ci'][0]:.3f}-{m['tpr_gap_ci'][1]:.3f}) | "
            f"{m['overall_tpr']:.3f} | {m['flag_rate']:.1%} | {m['precision']:.1%} |"
        )
    ci = pol["gap_reduction_ci"]
    lines += [
        "",
        "Sensitivity gap reduction (paired bootstrap): "
        f"{pol['global']['tpr_gap'] - pol['per_band']['tpr_gap']:.3f} "
        f"(95% CI {ci[0]:.3f} to {ci[1]:.3f}).",
    ]
    return "\n".join(lines) + "\n"


def run(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    frames = splits_with_codes(settings)
    train, val, test = frames["train"], frames["val"], frames["test"]
    from ml_pipeline.train import fit_baseline

    champion = fit_baseline(frames, settings).pipeline
    p_val = champion.predict_proba(D.xy(val)[0])[:, 1]
    p_test = champion.predict_proba(D.xy(test)[0])[:, 1]
    committed = json.loads((settings.reports_dir / "baseline_metrics.json").read_text())["test"]
    refit_auc = roc_auc_score(test[D.TARGET], p_test)
    if abs(refit_auc - committed["roc_auc"]) > 1e-6:
        # lbfgs results differ slightly across platforms (Windows vs Linux: 0.6682 vs 0.6679)
        log.warning(
            "refitted champion test AUC %.6f != reports/baseline_metrics.json %.6f; "
            "this report will not match reports/audit.md exactly",
            refit_auc,
            committed["roc_auc"],
        )
    threshold = committed["threshold"]
    report = {
        "champion_refit_test_auc": float(refit_auc),
        "committed_test_auc": float(committed["roc_auc"]),
        "signal": signal_by_band(train),
        "young_phenotype_test": phenotype_auc(test, p_test),
        "model_attempts": model_attempts(train, val, settings),
        "threshold_policies": threshold_policies(
            val, test, p_val, p_test, threshold, settings.random_seed
        ),
    }
    # LF on every platform, so the DVC hashes in dvc.lock match a Linux checkout
    out = settings.reports_dir
    (out / "age_gap.json").write_text(json.dumps(report, indent=2) + "\n", newline="\n")
    (out / "age_gap.md").write_text(markdown(report), newline="\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run()


if __name__ == "__main__":
    main()
