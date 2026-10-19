"""Low-latency scoring path for small requests.

scikit-learn's `ColumnTransformer` + `OneHotEncoder` cost about 14 ms per call for a single
row, almost all of it per-column bookkeeping that is the same for 1 row as for 100. For
small batches the API instead encodes each record with plain dict lookups built from the
*fitted* preprocessor (scaler means/scales, one-hot layout including the "infrequent"
bucket), then calls the fitted model on the resulting matrix.

The fast path is only switched on after it reproduces the sklearn pipeline's probabilities
on a reference sample (max abs difference < 1e-9); otherwise the API keeps using sklearn.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger(__name__)

INFREQUENT = "infrequent_sklearn"
TOLERANCE = 1e-9


class FastScorer:
    def __init__(self, pipeline: Pipeline):
        prep: ColumnTransformer = pipeline["prep"]
        self.model = pipeline["model"]
        transformers = {name: (est, cols) for name, est, cols in prep.transformers_}
        scaler, numeric = transformers["num"]
        encoder, categorical = transformers["cat"]
        if not isinstance(scaler, StandardScaler) or not isinstance(encoder, OneHotEncoder):
            raise TypeError("unsupported preprocessor layout")
        if encoder.drop_idx_ is not None:
            raise TypeError("one-hot drop is not supported")
        self.numeric = list(numeric)
        self.mean = scaler.mean_
        self.scale = scaler.scale_
        num_slice, cat_slice = prep.output_indices_["num"], prep.output_indices_["cat"]
        self.n_features = cat_slice.stop
        self.num_offset = num_slice.start
        names = encoder.get_feature_names_out(categorical)
        index = {name: cat_slice.start + i for i, name in enumerate(names)}
        # per column: value -> output index; plus the column's "infrequent" index (or None)
        self.categorical = [
            (col, {name[len(col) + 1 :]: idx for name, idx in index.items()
                   if name.startswith(col + "_")}, index.get(f"{col}_{INFREQUENT}"))
            for col in categorical
        ]  # fmt: skip

    def transform(self, records: list[dict]) -> np.ndarray:
        x = np.zeros((len(records), self.n_features))
        for row, record in enumerate(records):
            values = np.fromiter((record[c] for c in self.numeric), float, len(self.numeric))
            x[row, self.num_offset : self.num_offset + len(values)] = (
                values - self.mean
            ) / self.scale
            for col, lookup, infrequent in self.categorical:
                idx = lookup.get(str(record[col]), infrequent)
                if idx is not None:
                    x[row, idx] = 1.0
        return x

    def predict_proba(self, records: list[dict]) -> np.ndarray:
        return self.model.predict_proba(self.transform(records))[:, 1]


def build_fast_scorer(pipeline, reference: pd.DataFrame | None) -> FastScorer | None:
    """A verified fast scorer, or None when the pipeline is not supported or disagrees."""
    try:
        scorer = FastScorer(pipeline)
    except (KeyError, TypeError, AttributeError) as exc:
        log.info("fast path off: %s", exc)
        return None
    if reference is None or reference.empty:
        log.info("fast path off: no reference sample to verify it against")
        return None
    sample = reference[pipeline["prep"].feature_names_in_].head(2000)
    expected = pipeline.predict_proba(sample)[:, 1]
    got = scorer.predict_proba(sample.to_dict("records"))
    diff = float(np.max(np.abs(expected - got)))
    if diff > TOLERANCE:
        log.warning("fast path off: differs from sklearn by %.3g", diff)
        return None
    log.info("fast path on (verified on %d rows, max diff %.1e)", len(sample), diff)
    return scorer
