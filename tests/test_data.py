import pytest

from conftest import fake_raw
from ml_pipeline.data import FEATURES, GROUP, ORDER, TARGET, clean, icd9_group, split


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
    assert list(df.columns) == [ORDER, GROUP, *FEATURES, TARGET]
    assert set(df[TARGET].unique()) <= {0, 1}
    assert df[FEATURES].isna().sum().sum() == 0


def test_split_is_grouped_by_patient_and_covers_all_rows():
    df = clean(fake_raw())
    parts = split(df, seed=1, strategy="grouped")
    patients = [set(parts[name][GROUP]) for name in ("train", "val", "test")]
    assert not patients[0] & patients[1]
    assert not patients[0] & patients[2]
    assert not patients[1] & patients[2]
    assert sum(len(p) for p in parts.values()) == len(df)


def test_time_split_trains_on_the_past_and_tests_on_the_latest():
    df = clean(fake_raw()).sample(frac=1.0, random_state=3)  # order must not matter
    parts = split(df, seed=1)  # default strategy: time
    assert sum(len(p) for p in parts.values()) == len(df)
    assert parts["train"][ORDER].max() < parts["val"][ORDER].min()
    assert parts["val"][ORDER].max() < parts["test"][ORDER].min()
    assert abs(len(parts["test"]) - 0.15 * len(df)) <= 1


def test_unknown_split_strategy_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        split(clean(fake_raw()), seed=1, strategy="random")


def test_split_is_deterministic():
    df = clean(fake_raw())
    first, second = split(df, seed=7), split(df, seed=7)
    assert first["test"].equals(second["test"])


def test_history_features_use_only_earlier_encounters():
    import pandas as pd

    from ml_pipeline.data import add_history_features

    raw = pd.DataFrame(
        {
            "encounter_id": [30, 10, 20, 40],
            "patient_nbr": [1, 1, 1, 2],
            "readmitted": ["NO", "<30", ">30", "<30"],
            "number_outpatient": [0, 1, 0, 2],
            "number_emergency": [1, 0, 0, 0],
            "number_inpatient": [2, 0, 1, 1],
        }
    )
    out = add_history_features(raw).set_index("encounter_id")
    assert out.loc[[10, 20, 30], "prior_encounters"].tolist() == [0, 1, 2]
    # encounter 10 was readmitted <30; that is history for 20 and 30, but not for 10 itself
    assert out.loc[[10, 20, 30], "prior_readmissions_30d"].tolist() == [0, 1, 1]
    assert out.loc[40, "prior_encounters"] == 0 and out.loc[40, "prior_readmissions_30d"] == 0
    assert out.loc[30, "total_prior_visits"] == 3
    assert list(add_history_features(raw).index) == list(raw.index)  # original order kept
