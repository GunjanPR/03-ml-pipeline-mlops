"""Quality gate: promote the `challenger` model version to `champion` only if it is better.

    python -m ml_pipeline.promote            # decide, move aliases, write reports/gate.json
    python -m ml_pipeline.promote --strict   # same, but exit 1 when blocked (CI)

The challenger is compared with the current champion, or with the latest baseline version
when there is no champion yet (a lone baseline becomes the first champion). Both are scored
on the same held-out test rows, and the gate requires all of:

* ROC-AUC lift of at least `MIN_AUC_LIFT`;
* the lower bound of a paired-bootstrap 95% CI for that lift above zero;
* Brier score no worse than the reference by more than `MAX_BRIER_INCREASE`.

The decision is written to `reports/gate.json` and the challenger version is tagged with it.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass

import mlflow
import mlflow.sklearn
import numpy as np
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.metrics import brier_score_loss, roc_auc_score

from ml_pipeline.config import Settings, get_settings
from ml_pipeline.data import load_splits, xy
from ml_pipeline.train import CHALLENGER, CHAMPION

log = logging.getLogger(__name__)

MAX_BRIER_INCREASE = 0.002


@dataclass
class GateResult:
    challenger_version: int
    reference_version: int
    reference_kind: str
    challenger_auc: float
    reference_auc: float
    auc_lift: float
    auc_lift_ci_low: float
    auc_lift_ci_high: float
    challenger_brier: float
    reference_brier: float
    min_auc_lift: float
    passed: bool
    reasons: list[str]


def paired_bootstrap_auc_diff(
    y: np.ndarray, p_new: np.ndarray, p_ref: np.ndarray, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile interval for AUC(p_new) - AUC(p_ref), resampling rows jointly."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if y[idx].min() == y[idx].max():
            continue
        diffs.append(roc_auc_score(y[idx], p_new[idx]) - roc_auc_score(y[idx], p_ref[idx]))
    low, high = np.percentile(diffs, [2.5, 97.5])
    return float(low), float(high)


def decide(
    y: np.ndarray,
    p_new: np.ndarray,
    p_ref: np.ndarray,
    min_auc_lift: float,
    seed: int = 0,
) -> tuple[bool, dict]:
    auc_new, auc_ref = roc_auc_score(y, p_new), roc_auc_score(y, p_ref)
    brier_new, brier_ref = brier_score_loss(y, p_new), brier_score_loss(y, p_ref)
    low, high = paired_bootstrap_auc_diff(y, p_new, p_ref, seed=seed)
    reasons = []
    if auc_new - auc_ref < min_auc_lift:
        reasons.append(f"AUC lift {auc_new - auc_ref:+.4f} < required {min_auc_lift:+.4f}")
    if low <= 0:
        reasons.append(f"lift CI lower bound {low:+.4f} is not above 0")
    if brier_new > brier_ref + MAX_BRIER_INCREASE:
        reasons.append(f"Brier {brier_new:.4f} worse than {brier_ref:.4f} + {MAX_BRIER_INCREASE}")
    stats = {
        "challenger_auc": float(auc_new),
        "reference_auc": float(auc_ref),
        "auc_lift": float(auc_new - auc_ref),
        "auc_lift_ci_low": low,
        "auc_lift_ci_high": high,
        "challenger_brier": float(brier_new),
        "reference_brier": float(brier_ref),
        "reasons": reasons,
    }
    return not reasons, stats


def _reference(client: MlflowClient, name: str):
    try:
        return client.get_model_version_by_alias(name, CHAMPION)
    except MlflowException:
        versions = client.search_model_versions(
            f"name='{name}' and tag.model_kind='baseline'", order_by=["version_number DESC"]
        )
        if not versions:
            raise SystemExit("no champion and no baseline version registered") from None
        return versions[0]


def promote(settings: Settings | None = None, dry_run: bool = False) -> GateResult | None:
    settings = settings or get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.registered_model_name
    challenger = client.get_model_version_by_alias(name, CHALLENGER)
    reference = _reference(client, name)
    if str(challenger.version) == str(reference.version):
        if CHAMPION not in reference.aliases and not dry_run:
            # First run: the baseline itself becomes the initial champion.
            client.set_registered_model_alias(name, CHAMPION, challenger.version)
            log.info("no champion yet: baseline v%s set as champion", challenger.version)
        else:
            log.info("challenger v%s is already the reference; nothing to do", challenger.version)
        return None

    x_test, y_test = xy(load_splits(settings)["test"])
    p_new = mlflow.sklearn.load_model(f"models:/{name}/{challenger.version}").predict_proba(x_test)[
        :, 1
    ]
    p_ref = mlflow.sklearn.load_model(f"models:/{name}/{reference.version}").predict_proba(x_test)[
        :, 1
    ]
    passed, stats = decide(y_test, p_new, p_ref, settings.min_auc_lift, settings.random_seed)
    result = GateResult(
        challenger_version=int(challenger.version),
        reference_version=int(reference.version),
        reference_kind=reference.tags.get("model_kind", "unknown"),
        min_auc_lift=settings.min_auc_lift,
        passed=passed,
        **stats,
    )
    client.set_model_version_tag(
        name, challenger.version, "gate", "passed" if passed else "blocked"
    )
    if not dry_run:
        if passed:
            client.set_registered_model_alias(name, CHAMPION, challenger.version)
        elif CHAMPION not in reference.aliases:
            # Blocked with no champion yet: serve the baseline rather than nothing.
            client.set_registered_model_alias(name, CHAMPION, reference.version)
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "gate.json").write_text(json.dumps(asdict(result), indent=2))
    log.info(
        "gate %s: v%s vs v%s (%s) AUC %.4f vs %.4f, lift %+.4f [%+.4f, %+.4f]%s",
        "PASSED" if passed else "BLOCKED",
        result.challenger_version,
        result.reference_version,
        result.reference_kind,
        result.challenger_auc,
        result.reference_auc,
        result.auc_lift,
        result.auc_lift_ci_low,
        result.auc_lift_ci_high,
        "" if passed else " - " + "; ".join(result.reasons),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="evaluate without moving aliases")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 when the challenger is blocked (CI gates)"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = promote(dry_run=args.dry_run)
    blocked = result is not None and not result.passed
    sys.exit(1 if blocked and args.strict else 0)


if __name__ == "__main__":
    main()
