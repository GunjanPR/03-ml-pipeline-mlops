import numpy as np
import pandas as pd
import pytest

from ml_pipeline.config import Settings
from ml_pipeline.data import CATEGORICAL_FEATURES, MEDICATIONS, NUMERIC_FEATURES


def fake_raw(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """A small frame with the raw dataset's columns and a learnable signal."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"encounter_id": np.arange(n), "patient_nbr": rng.integers(0, n // 3, n)})
    for col in NUMERIC_FEATURES:
        df[col] = rng.integers(0, 10, n)
    raw_categoricals = [c for c in CATEGORICAL_FEATURES if not c.startswith("diag_")]
    for col in raw_categoricals:
        df[col] = rng.choice(["a", "b", "c"], n)
    for col in MEDICATIONS:
        df[col] = rng.choice(["No", "Steady", "Up"], n)
    df["discharge_disposition_id"] = rng.choice([1, 3, 6, 11], n)
    df["admission_type_id"] = rng.choice([1, 2, 3], n)
    for i in (1, 2, 3):
        df[f"diag_{i}"] = rng.choice(["250.01", "428", "486", "V58", None], n)
    risky = df["number_inpatient"] > 6
    df["readmitted"] = np.where(risky | (rng.random(n) < 0.08), "<30", "NO")
    return df


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        mlflow_tracking_uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        _env_file=None,
    )
