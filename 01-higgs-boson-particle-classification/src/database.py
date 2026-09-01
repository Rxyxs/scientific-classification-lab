"""Persistencia de features + predicciones en DuckDB."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "outputs" / "higgs.duckdb"


def export_results(features: pd.DataFrame, predictions: pd.DataFrame, metrics: pd.DataFrame) -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM features")
    con.execute("CREATE OR REPLACE TABLE predictions AS SELECT * FROM predictions")
    con.execute("CREATE OR REPLACE TABLE model_metrics AS SELECT * FROM metrics")
    con.close()


def export_activation_comparison(results: pd.DataFrame) -> None:
    """Persiste la tabla comparativa de activaciones (ReLU/GELU/Swish +
    FocalLoss) en el mismo DuckDB usado por el pipeline principal."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB_PATH))
    con.execute("CREATE OR REPLACE TABLE activation_comparison AS SELECT * FROM results")
    con.close()
