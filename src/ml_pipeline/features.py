"""Feature preprocessing shared by every model."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml_pipeline.data import CATEGORICAL_FEATURES, NUMERIC_FEATURES

# Categories seen fewer times than this in train are pooled into one "infrequent" level.
MIN_CATEGORY_COUNT = 50


def build_preprocessor() -> ColumnTransformer:
    """Scale numeric counts and one-hot encode categoricals (unknown levels are ignored)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist", min_frequency=MIN_CATEGORY_COUNT
                ),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
