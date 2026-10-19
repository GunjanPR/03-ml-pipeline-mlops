"""Feature preprocessing shared by every model."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Categories seen fewer times than this in train are pooled into one "infrequent" level.
MIN_CATEGORY_COUNT = 50


def build_preprocessor(exclude: frozenset[str] = frozenset()) -> ColumnTransformer:
    """Scale numeric counts and one-hot encode categoricals (unknown levels are ignored).
    Columns in `exclude` are dropped (used for feature ablations)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), [c for c in NUMERIC_FEATURES if c not in exclude]),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT
                ),
                [c for c in CATEGORICAL_FEATURES if c not in exclude],
            ),
        ]
    )
