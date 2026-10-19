import pytest

from conftest import fake_raw
from ml_pipeline.data import FEATURES, GROUP, TARGET, clean, icd9_group, split


@pytest.mark.parametrize(
    ("code", "group"),
    [
        ("250.83", "diabetes"),
        ("428", "circulatory"),
        ("785", "circulatory"),
        ("486", "respiratory"),
        ("V58", "other"),
        ("E888", "other"),
        ("820", "injury"),
        ("missing", "missing"),
        ("not-a-code", "other"),
    ],
)
def test_icd9_group(code, group):
    assert icd9_group(code) == group


def test_clean_drops_hospice_and_expired_and_builds_target():
    raw = fake_raw()
    df = clean(raw)
    assert len(df) == (raw["discharge_disposition_id"] != 11).sum()
    assert list(df.columns) == [GROUP, *FEATURES, TARGET]
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df[FEATURES].isna().sum().sum() == 0


def test_split_is_grouped_by_patient_and_covers_all_rows():
    df = clean(fake_raw())
    parts = split(df, seed=1)
    patients = [set(parts[name][GROUP]) for name in ("train", "val", "test")]
    assert not patients[0] & patients[1]
    assert not patients[0] & patients[2]
    assert not patients[1] & patients[2]
    assert sum(len(p) for p in parts.values()) == len(df)


def test_split_is_deterministic():
    df = clean(fake_raw())
    first, second = split(df, seed=7), split(df, seed=7)
    assert first["test"].equals(second["test"])
