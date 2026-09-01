"""Exporta las probabilidades del LightGBM afinado sobre el public-test
(mas la etiqueta real y el peso KaggleWeight) para que julia/ams_sweep.jl
pueda reproducir el barrido de AMS de forma independiente.

    python -m src.export_for_julia
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.data import load_raw, split_by_kaggle_set
from src.features import engineer_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"


def main() -> None:
    print("[1/2] Cargando datos, modelo afinado y prediciendo sobre public-test...")
    df = load_raw()
    train_df, public_df, _ = split_by_kaggle_set(df)
    _, medians = engineer_features(train_df)
    X_public, _ = engineer_features(public_df, medians_by_jet=medians)

    model = joblib.load(MODELS_DIR / "lightgbm_tuned.joblib")
    proba = model.predict_proba(X_public)[:, 1]

    print("[2/2] Guardando CSV de referencia para Julia...")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = pd.DataFrame(
        {
            "proba": proba,
            "is_signal": public_df["is_signal"].to_numpy(),
            "kaggle_weight": public_df["KaggleWeight"].to_numpy(),
        }
    )
    out.to_csv(REPORTS_DIR / "julia_ams_reference.csv", index=False)
    print(f"Guardado en: {REPORTS_DIR / 'julia_ams_reference.csv'} ({len(out)} filas)")


if __name__ == "__main__":
    main()
