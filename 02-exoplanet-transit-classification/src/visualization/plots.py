"""Result plots for the KOI disposition classifier, generated from the real
metrics/artifacts already written by train.py and torch_classifier.py --
nothing here is recomputed or hardcoded separately."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import shap
import xgboost as xgb
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.engineering import prepare_features, to_xy

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "koi_cumulative.parquet"
REPORTS_DIR = ROOT / "reports"
FIG_DIR = REPORTS_DIR / "figures"

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10})


def plot_class_distribution(raw: pl.DataFrame) -> None:
    counts = raw["koi_disposition"].value_counts().sort("count", descending=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(counts["koi_disposition"], counts["count"], color=["#2A6F97", "#C1440E", "#6C757D"])
    ax.set_ylabel("count")
    ax.set_title(f"KOI disposition — real Kepler catalog (n={raw.height})")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(model: xgb.XGBClassifier, X_test, y_test, class_names: list[str]) -> None:
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("XGBoost — confusion matrix (held-out test)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_shap_importance() -> None:
    shap_df = pl.read_csv(REPORTS_DIR / "shap_feature_importance.csv").sort("mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#2A6F97")
    ax.set_xlabel("mean |SHAP value| (averaged over 3 classes)")
    ax.set_title("KOI classifier — SHAP feature importance")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "shap_importance.png", dpi=150)
    plt.close(fig)


def plot_activation_comparison() -> None:
    with open(REPORTS_DIR / "torch_activation_comparison.json", encoding="utf-8") as f:
        results = json.load(f)
    with open(REPORTS_DIR / "metrics.json", encoding="utf-8") as f:
        metrics = json.load(f)

    names = list(results.keys()) + ["xgboost"]
    accs = [results[n]["val_accuracy"] for n in results] + [metrics["xgboost_accuracy"]]
    colors = ["#6C757D"] * len(results) + ["#2A6F97"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(names, accs, color=colors)
    ax.axhline(metrics["baseline_accuracy"], color="#C1440E", linestyle="--", label=f"baseline (majority class) = {metrics['baseline_accuracy']:.3f}")
    ax.set_ylabel("accuracy (held-out)")
    ax.set_title("Model comparison: PyTorch activations vs. XGBoost vs. baseline")
    ax.legend()
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "model_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    raw = pl.read_parquet(RAW_PATH)
    df = prepare_features(raw)
    X, y_str, feature_names = to_xy(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        objective="multi:softprob", num_class=len(label_encoder.classes_), eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)

    plot_class_distribution(raw)
    plot_confusion_matrix(model, X_test, y_test, label_encoder.classes_.tolist())
    plot_shap_importance()
    plot_activation_comparison()
    print(f"4 charts -> {FIG_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
