"""Feature preparation for the KOI disposition classifier.

No leakage risk here worth flagging like in other repos in this portfolio:
every feature is a transit/stellar parameter measured independently of the
human/pipeline vetting decision that produced `koi_disposition` -- unlike
the `koi_fpflag_*` columns (deliberately excluded), which directly encode
sub-decisions of the label itself.
"""
from __future__ import annotations

import numpy as np
import polars as pl

FEATURE_COLUMNS = [
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]

# Right-skewed physical quantities spanning orders of magnitude -- log1p
# keeps the tree/NN models from being dominated by a handful of extreme
# outlier planets (e.g. koi_depth in ppm can range from single digits to
# hundreds of thousands).
LOG_COLUMNS = ["koi_period", "koi_depth", "koi_prad", "koi_insol", "koi_model_snr"]

LABEL_COLUMN = "koi_disposition"


def prepare_features(df: pl.DataFrame) -> pl.DataFrame:
    clean = df.drop_nulls(subset=FEATURE_COLUMNS + [LABEL_COLUMN])

    exprs = []
    for col in FEATURE_COLUMNS:
        if col in LOG_COLUMNS:
            exprs.append((pl.col(col).clip(lower_bound=0) + 1.0).log().alias(col))
        else:
            exprs.append(pl.col(col))

    return clean.with_columns(exprs)


def to_xy(df: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    X = df.select(FEATURE_COLUMNS).to_numpy()
    y = df[LABEL_COLUMN].to_numpy()
    return X, y, FEATURE_COLUMNS
