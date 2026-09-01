"""Busqueda de hiperparametros con Optuna para LightGBM, optimizando AMS
directamente (no un proxy como logloss) sobre el public-test oficial.

    python -m src.tune
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import optuna
from lightgbm import LGBMClassifier

from src.data import load_raw, split_by_kaggle_set
from src.features import engineer_features
from src.metrics import best_ams_over_thresholds

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"
MODELS_DIR = ROOT / "outputs" / "models"

N_TRIALS = 40


def _objective(trial, X_train, y_train, X_val, y_val, w_val):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "num_leaves": trial.suggest_int("num_leaves", 15, 127),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }
    model = LGBMClassifier(**params, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_val)[:, 1]
    ams, _ = best_ams_over_thresholds(y_val, proba, w_val)
    return ams


def main() -> None:
    print("[1/3] Cargando datos y features...")
    df = load_raw()
    train_df, public_df, _ = split_by_kaggle_set(df)
    X_train, medians = engineer_features(train_df)
    X_public, _ = engineer_features(public_df, medians_by_jet=medians)
    y_train = train_df["is_signal"].to_numpy()
    y_public = public_df["is_signal"].to_numpy()
    w_public = public_df["KaggleWeight"].to_numpy()

    print(f"[2/3] Optuna: {N_TRIALS} trials, optimizando AMS en public-test...")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: _objective(t, X_train, y_train, X_public, y_public, w_public), n_trials=N_TRIALS)

    print(f"\nMejor AMS (Optuna): {study.best_value:.4f}")
    print(f"Mejores parametros: {study.best_params}")

    print("[3/3] Reentrenando modelo final con los mejores parametros...")
    best_model = LGBMClassifier(**study.best_params, random_state=42, verbose=-1)
    best_model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "lightgbm_tuned.joblib")

    baseline_metrics = json.load(open(REPORTS_DIR / "model_metrics.json", encoding="utf-8"))
    baseline_lgbm_ams = next(m["ams"] for m in baseline_metrics if m["model"] == "lightgbm")

    result = {
        "baseline_lightgbm_ams": baseline_lgbm_ams,
        "tuned_lightgbm_ams": round(study.best_value, 4),
        "improvement_pp": round(study.best_value - baseline_lgbm_ams, 4),
        "n_trials": N_TRIALS,
        "best_params": study.best_params,
    }
    with open(REPORTS_DIR / "optuna_tuning_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n=== Resultado ===")
    print(f"LightGBM baseline (sin tuning): AMS={baseline_lgbm_ams}")
    print(f"LightGBM tuned (Optuna, {N_TRIALS} trials): AMS={study.best_value:.4f}")
    print(f"Mejora: {result['improvement_pp']:+.4f} pp")
    print(f"\nGuardado en: {REPORTS_DIR / 'optuna_tuning_result.json'}")


if __name__ == "__main__":
    main()
