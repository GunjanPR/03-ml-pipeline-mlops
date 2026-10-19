"""Model audit on the held-out test split: calibration, decision value and subgroups.

    python -m ml_pipeline.audit       # writes reports/audit.json and reports/audit.md

ROC-AUC says how well a model ranks patients overall. Before a risk score drives who gets
a follow-up call, three more questions matter:

* **Calibration.** Does a predicted 20% risk mean about 20% of those patients come back?
  Expected calibration error (ECE) over 10 equal-count bins, and the ratio of mean predicted
  to observed risk.
* **Decision value.** A care team can only follow up a fixed share of discharges. Of the
  patients readmitted, what share is in the top 10% / 20% of scores (recall), how many of
  those flagged come back (precision), and the lift over picking at random.
* **Subgroups.** The same metrics by age band, sex and race, with bootstrap 95% CIs, at the
  deployed decision threshold. Gaps are reported, not hidden: the dataset's race field is
  self-reported and incomplete, and small groups have wide intervals.

Every model is scored on the same test rows (`reports/<model>/test_proba.npy`, written by
`ml_pipeline.train`), so the champion and the challenger can be compared like for like.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import TARGET, load_splits

log = logging.getLogger(__name__)

N_BINS = 10
TOP_SHARES = (0.1, 0.2)
# Fewer positives than this and a subgroup AUC is not worth reporting (its CI spans
# most of [0.5, 1]); counts and rates are still shown.
MIN_POSITIVES_FOR_AUC = 30
N_BOOT = 1000
AGE_BANDS = {
    "<40": ("[0-10)", "[10-20)", "[20-30)", "[30-40)"),
    "40-59": ("[40-50)", "[50-60)"),
    "60-79": ("[60-70)", "[70-80)"),
    "80+": ("[80-90)", "[90-100)"),
}


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> float:
    """Mean |observed - predicted| over equal-count bins of the predicted probability,
    weighted by bin size."""
    order = np.argsort(p)
    error = 0.0
    for idx in np.array_split(order, n_bins):
        if len(idx):
            error += len(idx) * abs(y[idx].mean() - p[idx].mean())
    return float(error / len(y))


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> list[dict]:
    order = np.argsort(p)
    return [
        {
            "bin": i + 1,
            "n": len(idx),
            "mean_predicted": float(p[idx].mean()),
            "observed_rate": float(y[idx].mean()),
        }
        for i, idx in enumerate(np.array_split(order, n_bins))
        if len(idx)
    ]


def top_share_capture(y: np.ndarray, p: np.ndarray, share: float) -> dict:
    """Flag the highest-scoring `share` of patients: recall, precision and lift."""
    k = max(1, int(round(share * len(p))))
    flagged = np.argsort(-p, kind="stable")[:k]
    hits = y[flagged].sum()
    precision = hits / k
    return {
        "share": share,
        "flagged": k,
        "recall": float(hits / max(y.sum(), 1)),
        "precision": float(precision),
        "lift": float(precision / y.mean()),
    }


@dataclass
class Interval:
    value: float | None
    low: float | None
    high: float | None


def _bootstrap(stat, y: np.ndarray, p: np.ndarray, rng: np.random.Generator) -> Interval:
    try:
        value = stat(y, p)
    except ValueError:
        return Interval(None, None, None)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        try:
            draws.append(stat(y[idx], p[idx]))
        except ValueError:  # a resample with one class
            continue
    low, high = np.percentile(draws, [2.5, 97.5])
    return Interval(float(value), float(low), float(high))


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    if y.min() == y.max():
        raise ValueError("one class")
    return roc_auc_score(y, p)


def _tpr(threshold: float):
    def stat(y: np.ndarray, p: np.ndarray) -> float:
        if not y.any():
            raise ValueError("no positives")
        return float((p[y == 1] >= threshold).mean())

    return stat


def group_metrics(y: np.ndarray, p: np.ndarray, threshold: float, rng: np.random.Generator) -> dict:
    positives = int(y.sum())
    flagged = p >= threshold
    row = {
        "n": len(y),
        "positives": positives,
        "observed_rate": float(y.mean()),
        "mean_predicted": float(p.mean()),
        "calibration_ratio": float(p.mean() / y.mean()) if positives else None,
        "flag_rate": float(flagged.mean()),
        "precision": float(y[flagged].mean()) if flagged.any() else None,
        "brier": float(np.mean((p - y) ** 2)),
    }
    tpr = _bootstrap(_tpr(threshold), y, p, rng)
    row.update(tpr=tpr.value, tpr_ci=[tpr.low, tpr.high])
    if positives >= MIN_POSITIVES_FOR_AUC:
        auc = _bootstrap(_auc, y, p, rng)
        row.update(roc_auc=auc.value, roc_auc_ci=[auc.low, auc.high])
    else:
        row.update(roc_auc=None, roc_auc_ci=[None, None])
    return row


def subgroups(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Attribute -> group label per row."""
    band = {raw: name for name, raws in AGE_BANDS.items() for raw in raws}
    return {
        "age": frame["age"].map(band).fillna("unknown"),
        "sex": frame["gender"].astype(str),
        "race": frame["race"].astype(str),
    }


def audit_model(
    frame: pd.DataFrame, y: np.ndarray, p: np.ndarray, threshold: float, seed: int
) -> dict:
    rng = np.random.default_rng(seed)
    result = {
        "threshold": threshold,
        "overall": group_metrics(y, p, threshold, rng),
        "ece": expected_calibration_error(y, p),
        "reliability": reliability_table(y, p),
        "top_share": [top_share_capture(y, p, s) for s in TOP_SHARES],
        "subgroups": {},
    }
    for attribute, labels in subgroups(frame).items():
        rows = {}
        present = set(labels.unique())
        order = [b for b in AGE_BANDS if b in present] if attribute == "age" else []
        for label in order + sorted(present - set(order)):
            mask = (labels == label).to_numpy()
            if mask.sum() >= 20:
                rows[label] = group_metrics(y[mask], p[mask], threshold, rng)
        aucs = [r["roc_auc"] for r in rows.values() if r["roc_auc"] is not None]
        tprs = [r["tpr"] for r in rows.values() if r["tpr"] is not None and r["positives"] >= 30]
        result["subgroups"][attribute] = {
            "groups": rows,
            "auc_gap": float(max(aucs) - min(aucs)) if len(aucs) > 1 else None,
            "tpr_gap": float(max(tprs) - min(tprs)) if len(tprs) > 1 else None,
        }
    return result


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _ci(ci: list) -> str:
    return "" if ci[0] is None else f" ({ci[0]:.3f}-{ci[1]:.3f})"


def markdown(report: dict) -> str:
    lines = [
        "# Model audit (test split)",
        "",
        f"Generated by `python -m ml_pipeline.audit`. {report['n_test']:,} test encounters; "
        "95% bootstrap CIs. Subgroup AUC is omitted below "
        f"{MIN_POSITIVES_FOR_AUC} readmissions.",
        "",
        "## Calibration and decision value",
        "",
        "| Model | ROC-AUC | ECE | Predicted / observed risk | Top 10%: recall, precision, lift "
        "| Top 20%: recall, precision, lift |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in report["models"].items():
        top = {t["share"]: t for t in m["top_share"]}
        cells = [
            f"{top[s]['recall']:.1%}, {top[s]['precision']:.1%}, {top[s]['lift']:.2f}x"
            for s in TOP_SHARES
        ]
        lines.append(
            f"| {name} | {_fmt(m['overall']['roc_auc'])} | {m['ece']:.4f} | "
            f"{m['overall']['calibration_ratio']:.3f} | {cells[0]} | {cells[1]} |"
        )
    for name, m in report["models"].items():
        lines += [
            "",
            f"## Subgroups: {name} (threshold {m['threshold']:.3f})",
            "",
            "| Attribute | Group | n | Readmitted | Predicted / observed | Flagged | "
            "Sensitivity (TPR) | ROC-AUC |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for attribute, block in m["subgroups"].items():
            for label, r in block["groups"].items():
                lines.append(
                    f"| {attribute} | {label} | {r['n']:,} | {r['observed_rate']:.1%} | "
                    f"{_fmt(r['calibration_ratio'], 2)} | {r['flag_rate']:.1%} | "
                    f"{_fmt(r['tpr'])}{_ci(r['tpr_ci'])} | "
                    f"{_fmt(r['roc_auc'])}{_ci(r['roc_auc_ci'])} |"
                )
        gaps = [
            f"{a}: AUC gap {_fmt(b['auc_gap'])}, sensitivity gap {_fmt(b['tpr_gap'])}"
            for a, b in m["subgroups"].items()
        ]
        lines += ["", "Largest gaps between groups with enough readmissions: " + "; ".join(gaps)]
    return "\n".join(lines) + "\n"


def load_model_outputs(settings: Settings, models: list[str]) -> dict[str, tuple]:
    """Test probabilities and decision threshold per model, from the training reports.

    A name such as `ablation/baseline-without-race` reads an ablation run (`train --exclude`).
    """
    outputs = {}
    for name in models:
        folder = settings.reports_dir / name
        report = folder / "metrics.json" if name.startswith("ablation/") else None
        report = report or settings.reports_dir / f"{name}_metrics.json"
        metrics = json.loads(report.read_text())
        p = np.load(folder / "test_proba.npy")
        outputs[name] = (p, metrics["test"]["threshold"])
    return outputs


def audit(
    settings: Settings | None = None, models: list[str] | None = None, out: str = "audit"
) -> dict:
    settings = settings or get_settings()
    test = load_splits(settings)["test"]
    y = test[TARGET].to_numpy()
    report = {"n_test": len(test), "models": {}}
    outputs = load_model_outputs(settings, models or ["baseline", "xgboost"])
    for name, (p, threshold) in outputs.items():
        if len(p) != len(y):
            raise SystemExit(f"{name}: {len(p)} predictions for {len(y)} test rows; re-train")
        report["models"][name] = audit_model(test, y, p, threshold, settings.random_seed)
        log.info(
            "%s: ECE %.4f, top-10%% recall %.1f%%",
            name,
            report["models"][name]["ece"],
            100 * report["models"][name]["top_share"][0]["recall"],
        )
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / f"{out}.json").write_text(json.dumps(report, indent=2))
    (settings.reports_dir / f"{out}.md").write_text(markdown(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", default="baseline,xgboost")
    parser.add_argument("--out", default="audit", help="report name: reports/<out>.json/.md")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    audit(models=args.models.split(","), out=args.out)


if __name__ == "__main__":
    main()
