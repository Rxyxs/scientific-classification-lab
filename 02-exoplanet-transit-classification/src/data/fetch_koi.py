"""Download the real Kepler Objects of Interest (KOI) cumulative catalog from
NASA's Exoplanet Archive TAP API (public, no API key required).

Each row is a Kepler Threshold Crossing Event vetted by the Kepler pipeline
and/or a human reviewer, labeled with the real, published disposition:
CONFIRMED, CANDIDATE, or FALSE POSITIVE.
"""
from __future__ import annotations

import urllib.parse
from pathlib import Path

import polars as pl
import requests

RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
TAP_URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"

COLUMNS = [
    "kepoi_name", "kepler_name", "koi_disposition",
    "koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_teq",
    "koi_insol", "koi_model_snr", "koi_impact",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag",
]


def fetch_koi_catalog() -> pl.DataFrame:
    query = f"select {','.join(COLUMNS)} from cumulative"
    resp = requests.get(TAP_URL, params={"query": query, "format": "csv"}, timeout=60)
    resp.raise_for_status()
    if resp.text.lstrip().startswith("<?xml"):
        raise RuntimeError(f"NASA Exoplanet Archive returned an error, not CSV:\n{resp.text[:500]}")

    from io import StringIO
    df = pl.read_csv(StringIO(resp.text))
    return df


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = fetch_koi_catalog()
    out_path = RAW_DIR / "koi_cumulative.parquet"
    df.write_parquet(out_path)

    print(f"{df.height} filas descargadas del catalogo KOI real de NASA")
    print(df["koi_disposition"].value_counts())
    print(f"-> {out_path.relative_to(RAW_DIR.parents[1])}")


if __name__ == "__main__":
    main()
