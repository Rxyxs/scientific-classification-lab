"""Train and evaluate the KOI disposition classifier (CONFIRMED / CANDIDATE /
FALSE POSITIVE): a majority-class baseline vs. XGBoost, with SHAP
explainability -- mandatory here, not optional, since this is a scientific
classification task where "why did the model flag this KOI" matters as much
as the accuracy number.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import shap
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.engineering import FEATURE_COLUMNS, LABEL_COLUMN, prepare_features, to_xy

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "koi_cumulative.parquet"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "data" / "processed"


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    raw = pl.read_parquet(RAW_PATH)
    df = prepare_features(raw)
    print(f"n={df.height} filas tras dropear nulos en features/label (de {raw.height} originales)")

    X, y_str, feature_names = to_xy(df)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    baseline = DummyClassifier(strategy="most_frequent", random_state=42)
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_acc = accuracy_score(y_test, baseline_pred)
    baseline_f1 = f1_score(y_test, baseline_pred, average="macro")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        eval_metric="mlogloss",
        random_state=42,
    )
    model.fit(X_train, y_train)
    xgb_pred = model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    xgb_f1 = f1_score(y_test, xgb_pred, average="macro")

    report = classification_report(
        y_test, xgb_pred, target_names=label_encoder.classes_, output_dict=True
    )

    print(f"\nBaseline (clase mayoritaria): accuracy={baseline_acc:.4f}  F1-macro={baseline_f1:.4f}")
    print(f"XGBoost: accuracy={xgb_acc:.4f}  F1-macro={xgb_f1:.4f}")
    print("\n" + classification_report(y_test, xgb_pred, target_names=label_encoder.classes_))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=(0, 2))
    shap_importance = (
        pl.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
        .sort("mean_abs_shap", descending=True)
    )
    print("\nSHAP feature importance (promedio sobre las 3 clases):")
    print(shap_importance)

    model.save_model(MODELS_DIR / "koi_xgboost.json")
    shap_importance.write_csv(REPORTS_DIR / "shap_feature_importance.csv")

    metrics = {
        "n_rows_used": df.height,
        "n_rows_raw": raw.height,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "classes": label_encoder.classes_.tolist(),
        "baseline_accuracy": round(float(baseline_acc), 4),
        "baseline_f1_macro": round(float(baseline_f1), 4),
        "xgboost_accuracy": round(float(xgb_acc), 4),
        "xgboost_f1_macro": round(float(xgb_f1), 4),
        "xgboost_classification_report": report,
    }
    with open(REPORTS_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nMetricas guardadas en: {REPORTS_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
