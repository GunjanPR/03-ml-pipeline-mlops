import json

import pytest

from conftest import fake_raw
from ml_pipeline.data import TARGET, clean, prepare
from ml_pipeline.validation import (
    PROCESSED_SCHEMA,
    RAW_SCHEMA,
    DataValidationError,
    validate,
)


def test_fake_raw_and_its_cleaned_form_are_valid():
    raw = validate(fake_raw(), RAW_SCHEMA, "raw")
    validate(clean(raw), PROCESSED_SCHEMA, "processed")


@pytest.mark.parametrize(
    ("column", "value", "check"),
    [
        ("readmitted", "maybe", "isin"),
        ("age", "[100-110)", "isin"),
        ("num_medications", -3, "greater_than_or_equal_to"),
        ("insulin", "Sideways", "isin"),
    ],
)
def test_raw_schema_rejects_bad_values(column, value, check):
    raw = fake_raw()
    raw.loc[5, column] = value
    with pytest.raises(DataValidationError) as err:
        validate(raw, RAW_SCHEMA, "raw")
    cases = err.value.failures
    assert column in set(cases["column"])
    assert any(check in str(c) for c in cases["check"])


def test_raw_schema_rejects_duplicate_encounters_and_missing_columns():
    raw = fake_raw()
    raw.loc[1, "encounter_id"] = raw.loc[0, "encounter_id"]
    with pytest.raises(DataValidationError):
        validate(raw, RAW_SCHEMA, "raw")
    with pytest.raises(DataValidationError):
        validate(fake_raw().drop(columns="readmitted"), RAW_SCHEMA, "raw")


def test_processed_schema_rejects_missing_values_and_extra_columns():
    df = clean(fake_raw())
    with pytest.raises(DataValidationError):
        validate(df.assign(extra=1), PROCESSED_SCHEMA, "processed")
    df.loc[0, TARGET] = 2
    with pytest.raises(DataValidationError):
        validate(df, PROCESSED_SCHEMA, "processed")


def test_prepare_is_deterministic_and_idempotent(settings):
    settings.raw_dir.mkdir(parents=True)
    fake_raw(900).to_csv(settings.raw_dir / "diabetic_data.csv", index=False)
    first = prepare(settings)
    second = prepare(settings)
    for name in first:
        assert first[name].equals(second[name])
    profile = json.loads((settings.reports_dir / "data_profile.json").read_text())
    assert profile["raw_rows"] == 900
    assert sum(s["rows"] for s in profile["splits"].values()) + profile["excluded_rows"] == 900
