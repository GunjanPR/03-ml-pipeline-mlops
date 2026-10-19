"""Pandera schemas: bad data stops the pipeline before it reaches training or serving.

* `RAW_SCHEMA` checks the downloaded CSV: required columns, types, value ranges, the known
  category sets, and that `encounter_id` is unique.
* `PROCESSED_SCHEMA` checks what training consumes: no missing values, non-negative counts,
  a binary target, and categoricals that are non-empty strings.

Validation is lazy, so one report lists every failure rather than the first one.
"""

from __future__ import annotations

import pandera.pandas as pa
from pandera.errors import SchemaErrors

from ml_pipeline.data import (
    CATEGORICAL_FEATURES,
    GROUP,
    MEDICATIONS,
    NUMERIC_FEATURES,
    RAW_NUMERIC_FEATURES,
    TARGET,
)

AGE_BRACKETS = [f"[{i}-{i + 10})" for i in range(0, 100, 10)]
MEDICATION_VALUES = ["No", "Steady", "Up", "Down"]

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "encounter_id": pa.Column(int, unique=True),
        "patient_nbr": pa.Column(int),
        "gender": pa.Column(str, pa.Check.isin(["Male", "Female", "Unknown/Invalid"])),
        "age": pa.Column(str, pa.Check.isin(AGE_BRACKETS)),
        "admission_type_id": pa.Column(int, pa.Check.in_range(1, 8)),
        "discharge_disposition_id": pa.Column(int, pa.Check.in_range(1, 30)),
        "admission_source_id": pa.Column(int, pa.Check.in_range(1, 26)),
        "time_in_hospital": pa.Column(int, pa.Check.in_range(1, 14)),
        **{
            col: pa.Column(int, pa.Check.ge(0))
            for col in RAW_NUMERIC_FEATURES
            if col != "time_in_hospital"
        },
        **{col: pa.Column(str, pa.Check.isin(MEDICATION_VALUES)) for col in MEDICATIONS},
        "change": pa.Column(str, pa.Check.isin(["No", "Ch"])),
        "diabetesMed": pa.Column(str, pa.Check.isin(["No", "Yes"])),
        "readmitted": pa.Column(str, pa.Check.isin(["NO", ">30", "<30"])),
        **{f"diag_{i}": pa.Column(str, nullable=True) for i in (1, 2, 3)},
    },
    strict=False,  # other raw columns (weight, payer_code, ...) are allowed through
    coerce=False,
)

PROCESSED_SCHEMA = pa.DataFrameSchema(
    {
        GROUP: pa.Column(int),
        **{col: pa.Column(float, pa.Check.ge(0)) for col in NUMERIC_FEATURES},
        **{col: pa.Column(str, pa.Check.str_length(min_value=1)) for col in CATEGORICAL_FEATURES},
        TARGET: pa.Column(int, pa.Check.isin([0, 1])),
    },
    strict=True,
    checks=[pa.Check(lambda df: df[TARGET].mean() > 0, error="target has no positives")],
)


class DataValidationError(ValueError):
    def __init__(self, stage: str, failures):
        self.stage = stage
        self.failures = failures
        super().__init__(f"{stage} data failed validation:\n{failures.to_string(max_rows=20)}")


def validate(df, schema: pa.DataFrameSchema, stage: str):
    """Validate lazily; raise `DataValidationError` with every failure case."""
    try:
        return schema.validate(df, lazy=True)
    except SchemaErrors as err:
        cases = err.failure_cases[["column", "check", "failure_case"]]
        raise DataValidationError(stage, cases.drop_duplicates().head(100)) from None
