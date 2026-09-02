"""Interactive Plotly chart: real KOI test-set predictions in feature space.

Standalone, self-contained HTML (inline plotly.js) built from the actual
held-out predictions of the XGBoost classifier trained in src/models/train.py
-- same train/test split (random_state=42, stratified), same features. No
numbers here are invented; every point is a real KOI from the NASA Exoplanet
Archive cumulative table, plotted at its real (log) transit period and real
model signal-to-noise ratio, colored by its real published disposition, with
hover text showing the model's prediction so correct/incorrect calls are
visible interactively.

    python -m src.visualization.interactive
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.features.engineering import prepare_features, to_xy

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "koi_cumulative.parquet"
OUT_DIR = ROOT / "outputs" / "interactive"

COLORS = {
    "CONFIRMED": "#2A6F97",
    "CANDIDATE": "#C1440E",
    "FALSE POSITIVE": "#6C757D",
}


def main() -> None:
    raw = pl.read_parquet(RAW_PATH)
    df = prepare_features(raw)
    X, y_str, feature_names = to_xy(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(X)), test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05, subsample=0.8,
        colsample_bytree=0.8, objective="multi:softprob",
        num_class=len(label_encoder.classes_), eval_metric="mlogloss", random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_label = label_encoder.inverse_transform(y_pred)
    y_true_label = label_encoder.inverse_transform(y_test)

    # koi_period and koi_model_snr are log1p-transformed by prepare_features;
    # undo that here so axes read in physical units (days, SNR).
    period_days = np.expm1(X_test[:, feature_names.index("koi_period")])
    snr = np.expm1(X_test[:, feature_names.index("koi_model_snr")])
    depth_ppm = np.expm1(X_test[:, feature_names.index("koi_depth")])
    prad = np.expm1(X_test[:, feature_names.index("koi_prad")])

    test_df = df[idx_test.tolist()]
    kepoi = test_df["kepoi_name"].to_list() if "kepoi_name" in test_df.columns else [""] * len(idx_test)

    fig = go.Figure()
    for cls in label_encoder.classes_:
        mask = y_true_label == cls
        correct = y_pred_label[mask] == cls
        hover = [
            f"real disposition: {cls}<br>predicted: {pred}<br>"
            f"period: {p:.2f} d<br>SNR: {s:.1f}<br>depth: {d:.0f} ppm<br>radius: {r:.2f} R⊕"
            + ("<br><b>correctly classified</b>" if ok else "<br><b>misclassified</b>")
            for pred, p, s, d, r, ok in zip(
                y_pred_label[mask], period_days[mask], snr[mask], depth_ppm[mask], prad[mask], correct
            )
        ]
        # correctly classified: filled marker; misclassified: outlined X, same color family
        fig.add_trace(go.Scattergl(
            x=period_days[mask], y=snr[mask], mode="markers",
            name=f"{cls} (correct)",
            marker=dict(
                color=COLORS[cls], size=7,
                symbol=np.where(correct, "circle", "x"),
                line=dict(width=np.where(correct, 0, 1.5), color="black"),
                opacity=np.where(correct, 0.65, 0.95),
            ),
            text=hover, hoverinfo="text",
        ))

    fig.update_layout(
        title="Kepler Objects of Interest — held-out test set (2,300 KOIs), real disposition vs. XGBoost prediction",
        xaxis_title="orbital period (days, log scale)",
        yaxis_title="transit model SNR (log scale)",
        xaxis_type="log", yaxis_type="log",
        template="plotly_white",
        legend_title="real disposition",
        width=1000, height=650,
        margin=dict(t=80),
    )
    fig.update_layout(
        annotations=[dict(
            text="X markers = misclassified by the model. Circles = correctly classified. "
                 "Hover for real vs. predicted disposition, period, SNR, depth, radius.",
            xref="paper", yref="paper", x=0, y=-0.15, showarrow=False, font=dict(size=11, color="#555"),
        )]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "koi_feature_space.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    n_correct = int((y_pred_label == y_true_label).sum())
    print(f"{len(y_test)} held-out KOIs plotted, {n_correct} correctly classified ({n_correct/len(y_test):.1%})")
    print(f"-> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
