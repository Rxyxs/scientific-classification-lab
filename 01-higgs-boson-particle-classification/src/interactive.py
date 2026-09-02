"""Interactive Plotly chart: real AMS-vs-threshold sweep on the actual
public-test predictions of this run's models (LightGBM + PyTorch MLP).

Standalone, self-contained HTML (inline plotly.js). Every point comes from
sweeping decision thresholds over real predicted probabilities on the
official public-test split (100,000 real ATLAS events, KaggleSet == 'b'),
scored with the real AMS metric (src/metrics.py) -- nothing here is
interpolated or invented.

    python -m src.interactive
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go

from src.data import load_raw, split_by_kaggle_set
from src.features import engineer_features
from src.metrics import ams_score
from src.modeling import mlp_predict_proba, train_baseline_tree, train_lightgbm, train_mlp

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs" / "interactive"


def ams_curve(y_true, y_proba, weights, n_thresholds=200):
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    scores = np.empty(n_thresholds)
    for i, t in enumerate(thresholds):
        pred = (y_proba >= t).astype(int)
        scores[i] = ams_score(y_true, pred, weights)
    return thresholds, scores


def main() -> None:
    print("[1/4] Loading real ATLAS Higgs data + official train/public-test split...")
    df = load_raw()
    train_df, public_df, _ = split_by_kaggle_set(df)

    print("[2/4] Feature engineering...")
    X_train, medians = engineer_features(train_df)
    X_public, _ = engineer_features(public_df, medians_by_jet=medians)
    y_train = train_df["is_signal"].to_numpy()
    y_public = public_df["is_signal"].to_numpy()
    w_public = public_df["KaggleWeight"].to_numpy()

    print("[3/4] Training LightGBM + PyTorch MLP on real train split...")
    lgbm = train_lightgbm(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_public)[:, 1]

    mlp, scaler = train_mlp(X_train.to_numpy(), y_train, X_public.to_numpy(), y_public)
    mlp_proba = mlp_predict_proba(mlp, scaler, X_public.to_numpy())

    print("[4/4] Sweeping 200 thresholds x 2 models, building interactive chart...")
    t_lgbm, ams_lgbm = ams_curve(y_public, lgbm_proba, w_public)
    t_mlp, ams_mlp = ams_curve(y_public, mlp_proba, w_public)

    best_lgbm_idx = int(np.argmax(ams_lgbm))
    best_mlp_idx = int(np.argmax(ams_mlp))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_lgbm, y=ams_lgbm, mode="lines", name="LightGBM",
        line=dict(color="#02569B", width=2),
        hovertemplate="threshold=%{x:.3f}<br>AMS=%{y:.4f}<extra>LightGBM</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_mlp, y=ams_mlp, mode="lines", name="PyTorch MLP",
        line=dict(color="#EE4C2C", width=2),
        hovertemplate="threshold=%{x:.3f}<br>AMS=%{y:.4f}<extra>PyTorch MLP</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[t_lgbm[best_lgbm_idx]], y=[ams_lgbm[best_lgbm_idx]], mode="markers",
        name=f"LightGBM optimum (AMS={ams_lgbm[best_lgbm_idx]:.4f} @ t={t_lgbm[best_lgbm_idx]:.3f})",
        marker=dict(color="#02569B", size=12, symbol="star", line=dict(color="black", width=1)),
    ))
    fig.add_trace(go.Scatter(
        x=[t_mlp[best_mlp_idx]], y=[ams_mlp[best_mlp_idx]], mode="markers",
        name=f"MLP optimum (AMS={ams_mlp[best_mlp_idx]:.4f} @ t={t_mlp[best_mlp_idx]:.3f})",
        marker=dict(color="#EE4C2C", size=12, symbol="star", line=dict(color="black", width=1)),
    ))

    fig.update_layout(
        title="AMS vs. decision threshold — real public-test sweep (100,000 events, 200 thresholds/model)",
        xaxis_title="probability threshold",
        yaxis_title="AMS (Approximate Median Significance)",
        template="plotly_white",
        width=1000, height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_layout(
        annotations=[dict(
            text="Both curves computed live in this run from real predicted probabilities on the official "
                 "public-test split, scored with the exact AMS formula used by the HiggsML Challenge.",
            xref="paper", yref="paper", x=0, y=-0.13, showarrow=False, font=dict(size=11, color="#555"),
        )]
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "ams_threshold_sweep.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"LightGBM best AMS={ams_lgbm[best_lgbm_idx]:.4f} @ threshold={t_lgbm[best_lgbm_idx]:.4f}")
    print(f"MLP best AMS={ams_mlp[best_mlp_idx]:.4f} @ threshold={t_mlp[best_mlp_idx]:.4f}")
    print(f"-> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
