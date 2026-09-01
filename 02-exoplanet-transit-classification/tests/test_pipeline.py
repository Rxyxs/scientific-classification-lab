"""Tests run entirely offline against the already-downloaded real parquet
(data/raw/koi_cumulative.parquet) -- no network calls, no mocking of the
model logic itself."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.engineering import FEATURE_COLUMNS, LABEL_COLUMN, prepare_features, to_xy

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "koi_cumulative.parquet"

pytestmark = pytest.mark.skipif(not RAW_PATH.exists(), reason="run src/data/fetch_koi.py first")


@pytest.fixture(scope="module")
def raw_df() -> pl.DataFrame:
    return pl.read_parquet(RAW_PATH)


@pytest.fixture(scope="module")
def clean_df(raw_df: pl.DataFrame) -> pl.DataFrame:
    return prepare_features(raw_df)


def test_raw_data_has_real_disposition_labels(raw_df: pl.DataFrame):
    labels = set(raw_df[LABEL_COLUMN].unique().to_list())
    assert labels == {"CONFIRMED", "CANDIDATE", "FALSE POSITIVE"}


def test_raw_data_is_nontrivially_sized(raw_df: pl.DataFrame):
    assert raw_df.height > 5000


def test_prepare_features_drops_nulls(clean_df: pl.DataFrame):
    assert clean_df.select(FEATURE_COLUMNS + [LABEL_COLUMN]).null_count().sum_horizontal()[0] == 0


def test_log_transform_removes_negative_or_zero_skew_columns(clean_df: pl.DataFrame):
    # koi_period, koi_depth, etc. are log1p-transformed; after transform they
    # should not retain their original multi-order-of-magnitude spread as raw values.
    assert clean_df["koi_period"].max() < 20  # log-space, real periods go up to ~1000s of days


def test_to_xy_shapes_match(clean_df: pl.DataFrame):
    X, y, feature_names = to_xy(clean_df)
    assert X.shape[0] == y.shape[0] == clean_df.height
    assert X.shape[1] == len(feature_names) == len(FEATURE_COLUMNS)


def test_xgboost_beats_majority_class_baseline(clean_df: pl.DataFrame):
    """The actual claim this repo makes -- verified as a real, reproducible test,
    not just asserted in the README."""
    X, y_str, _ = to_xy(clean_df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_acc = accuracy_score(y_test, baseline.predict(X_test))

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        objective="multi:softprob", num_class=3, eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)
    xgb_acc = accuracy_score(y_test, model.predict(X_test))

    assert xgb_acc > baseline_acc + 0.15  # real, meaningful margin, not a coin flip
