"""Fetch, clean and split the UCI Diabetes 130-US Hospitals readmission dataset.

Task: predict whether an inpatient encounter is followed by a readmission within 30 days
(`readmitted == "<30"`). Splits are grouped by `patient_nbr`, so no patient appears in more
than one of train / val / test.

    python -m ml_pipeline.data fetch     # download and unzip the raw CSV (if missing)
    python -m ml_pipeline.data prepare   # validate, clean, split -> data/processed/
"""

from __future__ import annotations

import argparse
import json
import logging
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ml_pipeline.config import Settings, get_settings

log = logging.getLogger(__name__)

RAW_FILE = "diabetic_data.csv"
TARGET = "readmitted_30d"
GROUP = "patient_nbr"
SPLITS = ("train", "val", "test")

# Discharged to hospice or expired: these patients cannot be readmitted, so they are dropped
# (as in Strack et al., 2014).
EXCLUDED_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

RAW_NUMERIC_FEATURES = [
    "time_in_hospital",
    "num_lab_procedures",
    "num_procedures",
    "num_medications",
    "number_outpatient",
    "number_emergency",
    "number_inpatient",
    "number_diagnoses",
]
# Patient-history features derived from earlier encounters of the same patient (v2).
HISTORY_FEATURES = ["prior_encounters", "prior_readmissions_30d", "total_prior_visits"]
NUMERIC_FEATURES = RAW_NUMERIC_FEATURES + HISTORY_FEATURES
MEDICATIONS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide", "examide",
    "citoglipton", "insulin", "glyburide-metformin", "glipizide-metformin",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]  # fmt: skip
CATEGORICAL_FEATURES = [
    "race",
    "gender",
    "age",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
    "payer_code",
    "medical_specialty",
    "max_glu_serum",
    "A1Cresult",
    "change",
    "diabetesMed",
    "diag_1_group",
    "diag_2_group",
    "diag_3_group",
    *MEDICATIONS,
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def icd9_group(code: str) -> str:
    """Map an ICD-9 code to one of the disease groups used by Strack et al. (2014)."""
    if not code or code == "missing":
        return "missing"
    if code[0] in "VE":
        return "other"
    try:
        value = float(code)
    except ValueError:
        return "other"
    if int(value) == 250:
        return "diabetes"
    if 390 <= value <= 459 or int(value) == 785:
        return "circulatory"
    if 460 <= value <= 519 or int(value) == 786:
        return "respiratory"
    if 520 <= value <= 579 or int(value) == 787:
        return "digestive"
    if 800 <= value <= 999:
        return "injury"
    if 710 <= value <= 739:
        return "musculoskeletal"
    if 580 <= value <= 629 or int(value) == 788:
        return "genitourinary"
    if 140 <= value <= 239:
        return "neoplasms"
    return "other"


def fetch(settings: Settings | None = None) -> Path:
    """Download and unzip the raw CSV if it is not already present. Returns its path."""
    settings = settings or get_settings()
    raw_dir = settings.raw_dir
    csv_path = raw_dir / RAW_FILE
    if csv_path.exists():
        return csv_path
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "diabetes.zip"
    if not zip_path.exists():
        log.info("downloading %s", settings.dataset_url)
        urllib.request.urlretrieve(settings.dataset_url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(raw_dir)
    return csv_path


def load_raw(path: Path) -> pd.DataFrame:
    # "None" is a real category in max_glu_serum / A1Cresult, so only "?" means missing.
    return pd.read_csv(path, na_values=["?"], keep_default_na=False, low_memory=False)


def add_history_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Per-patient history from *earlier* encounters only (ordered by `encounter_id`, which the
    dataset assigns in time order). All of it is known when the patient is admitted:

    * `prior_encounters` - earlier encounters of this patient in the data;
    * `prior_readmissions_30d` - how many of them were followed by a readmission within 30
      days (that outcome is settled before, or by, the current admission);
    * `total_prior_visits` - outpatient + emergency + inpatient visits in the preceding year.
    """
    df = raw.sort_values("encounter_id").copy()
    by_patient = df.groupby(GROUP, sort=False)
    df["prior_encounters"] = by_patient.cumcount()
    readmitted = (df["readmitted"] == "<30").astype(int)
    df["prior_readmissions_30d"] = readmitted.groupby(df[GROUP]).cumsum() - readmitted
    df["total_prior_visits"] = (
        df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    )
    return df.sort_index()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Derive history features (on all encounters), filter, derive the target and the
    remaining features. Output columns: GROUP, FEATURES, TARGET."""
    df = add_history_features(raw)
    df = df[~df["discharge_disposition_id"].isin(EXCLUDED_DISCHARGE_IDS)].copy()
    for i in (1, 2, 3):
        df[f"diag_{i}_group"] = df[f"diag_{i}"].fillna("missing").astype(str).map(icd9_group)
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("missing").astype(str)
    df[NUMERIC_FEATURES] = df[NUMERIC_FEATURES].astype(float)
    df[TARGET] = (df["readmitted"] == "<30").astype(int)
    return df[[GROUP, *FEATURES, TARGET]].reset_index(drop=True)


def split(
    df: pd.DataFrame, seed: int, val_size: float = 0.15, test_size: float = 0.15
) -> dict[str, pd.DataFrame]:
    """Patient-grouped train / val / test split."""
    outer = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    rest_idx, test_idx = next(outer.split(df, groups=df[GROUP]))
    rest = df.iloc[rest_idx]
    inner = GroupShuffleSplit(
        n_splits=1, test_size=val_size / (1 - test_size), random_state=seed + 1
    )
    train_idx, val_idx = next(inner.split(rest, groups=rest[GROUP]))
    return {
        "train": rest.iloc[train_idx].reset_index(drop=True),
        "val": rest.iloc[val_idx].reset_index(drop=True),
        "test": df.iloc[test_idx].reset_index(drop=True),
    }


def profile(raw: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> dict:
    """Row counts, patients, positive rates and the raw missing-value rates."""
    missing = raw.isna().mean()
    return {
        "raw_rows": len(raw),
        "raw_patients": int(raw[GROUP].nunique()),
        "excluded_rows": len(raw) - sum(len(f) for f in frames.values()),
        "raw_missing_rate": {k: round(float(v), 4) for k, v in missing[missing > 0].items()},
        "splits": {
            name: {
                "rows": len(f),
                "patients": int(f[GROUP].nunique()),
                "positive_rate": round(float(f[TARGET].mean()), 4),
            }
            for name, f in frames.items()
        },
    }


def prepare(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Fetch, validate, clean, split and validate again; writes `data/processed/{split}.parquet`
    and `reports/data_profile.json`. Deterministic for a given seed, so re-running it rewrites
    the same files instead of adding rows."""
    from ml_pipeline.validation import PROCESSED_SCHEMA, RAW_SCHEMA, validate

    settings = settings or get_settings()
    raw = validate(load_raw(fetch(settings)), RAW_SCHEMA, "raw")
    frames = {
        name: validate(frame, PROCESSED_SCHEMA, f"processed/{name}")
        for name, frame in split(clean(raw), seed=settings.random_seed).items()
    }
    settings.processed_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(settings.processed_dir / f"{name}.parquet", index=False)
        log.info("%s: %d rows, positive rate %.3f", name, len(frame), frame[TARGET].mean())
    settings.reports_dir.mkdir(parents=True, exist_ok=True)
    (settings.reports_dir / "data_profile.json").write_text(
        json.dumps(profile(raw, frames), indent=2)
    )
    return frames


def load_splits(settings: Settings | None = None) -> dict[str, pd.DataFrame]:
    """Load processed splits, preparing them first if needed."""
    settings = settings or get_settings()
    paths = {name: settings.processed_dir / f"{name}.parquet" for name in SPLITS}
    if not all(p.exists() for p in paths.values()):
        return prepare(settings)
    return {name: pd.read_parquet(p) for name, p in paths.items()}


def xy(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    return frame[FEATURES], frame[TARGET].to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch or prepare the readmission dataset")
    parser.add_argument("step", choices=["fetch", "prepare"], nargs="?", default="prepare")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.step == "fetch":
        log.info("raw data at %s", fetch())
    else:
        prepare()


if __name__ == "__main__":
    main()
