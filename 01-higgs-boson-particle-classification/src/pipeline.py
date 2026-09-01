"""Pipeline end-to-end: ingesta -> feature engineering -> iteracion de
modelos (Decision Tree -> LightGBM -> MLP PyTorch) -> DuckDB -> reportes.

    python -m src.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data import load_raw, split_by_kaggle_set
from src.features import engineer_features
from src.modeling import (
    evaluate_model, mlp_predict_proba, train_baseline_tree, train_lightgbm, train_mlp,
)
from src.database import export_results

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"
REPORTS_DIR = ROOT / "outputs" / "reports"


def main() -> None:
    print("[1/6] Cargando dataset ATLAS Higgs (CERN Open Data)...")
    df = load_raw()
    train_df, public_df, private_df = split_by_kaggle_set(df)
    print(f"  train={len(train_df)}  public_test={len(public_df)}  private_test={len(private_df)}")

    print("[2/6] Feature engineering (imputacion por PRI_jet_num + flags + ratios)...")
    X_train, medians = engineer_features(train_df)
    X_public, _ = engineer_features(public_df, medians_by_jet=medians)
    X_private, _ = engineer_features(private_df, medians_by_jet=medians)

    y_train = train_df["is_signal"].to_numpy()
    y_public = public_df["is_signal"].to_numpy()
    y_private = private_df["is_signal"].to_numpy()
    w_public = public_df["KaggleWeight"].to_numpy()
    w_private = private_df["KaggleWeight"].to_numpy()

    results = []

    print("[3/6] Baseline: Decision Tree...")
    tree = train_baseline_tree(X_train, y_train)
    tree_proba = tree.predict_proba(X_public)[:, 1]
    results.append(evaluate_model("decision_tree", tree_proba, y_public, w_public))

    print("[4/6] Gradient Boosting: LightGBM...")
    lgbm = train_lightgbm(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_public)[:, 1]
    results.append(evaluate_model("lightgbm", lgbm_proba, y_public, w_public))

    print("[5/6] Red Neuronal: PyTorch MLP (Dropout + BatchNorm)...")
    mlp, scaler = train_mlp(X_train.to_numpy(), y_train, X_public.to_numpy(), y_public)
    mlp_proba = mlp_predict_proba(mlp, scaler, X_public.to_numpy())
    results.append(evaluate_model("pytorch_mlp", mlp_proba, y_public, w_public))

    public_metrics = pd.DataFrame(results)
    print("\n=== Metricas en public-test (leaderboard historico) ===")
    print(public_metrics.to_string(index=False))

    best_name = public_metrics.loc[public_metrics["ams"].idxmax(), "model"]
    print(f"\n[6/6] Mejor modelo por AMS: {best_name} -> evaluando en private-test (held-out final)...")

    if best_name == "decision_tree":
        best_proba_private = tree.predict_proba(X_private)[:, 1]
    elif best_name == "lightgbm":
        best_proba_private = lgbm.predict_proba(X_private)[:, 1]
    else:
        best_proba_private = mlp_predict_proba(mlp, scaler, X_private.to_numpy())

    private_result = evaluate_model(f"{best_name}_private_test", best_proba_private, y_private, w_private)
    print(private_result)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"tree": tree, "lgbm": lgbm}, MODELS_DIR / "gbm_models.joblib")
    joblib.dump(medians, MODELS_DIR / "jet_medians.joblib")
    import torch

    torch.save(mlp.state_dict(), MODELS_DIR / "mlp_state.pt")
    joblib.dump(scaler, MODELS_DIR / "mlp_scaler.joblib")

    all_metrics = pd.concat([public_metrics, pd.DataFrame([private_result])], ignore_index=True)
    all_metrics.to_csv(REPORTS_DIR / "model_metrics.csv", index=False)
    with open(REPORTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics.to_dict(orient="records"), f, indent=2, ensure_ascii=False)

    predictions_df = public_df[["EventId", "is_signal"]].copy()
    predictions_df["proba_lightgbm"] = lgbm_proba
    predictions_df["proba_mlp"] = mlp_proba
    predictions_df["proba_tree"] = tree_proba

    export_results(X_train.assign(EventId=train_df["EventId"], is_signal=y_train), predictions_df, all_metrics)
    print(f"\nGuardado en: {MODELS_DIR}, {REPORTS_DIR}, outputs/higgs.duckdb")


if __name__ == "__main__":
    main()
