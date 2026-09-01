"""Comparacion de funciones de activacion (ReLU / GELU / Swish=SiLU) para el
PyTorch MLP, entrenadas con una **loss custom** (Focal Loss) en vez de
BCEWithLogitsLoss plano.

Motivacion: el dataset tiene desbalance moderado (~34% senal / 66% fondo,
ver class_balance.png) y AMS castiga fuerte los falsos positivos de fondo
cuando hay pocos eventos senal bien clasificados. Focal Loss (Lin et al.,
2017) reduce el peso de ejemplos "faciles" ya bien clasificados y concentra
el gradiente en los dificiles -- una alternativa razonable a BCE plano para
este tipo de desbalance.

    python -m src.activation_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from src.data import load_raw, split_by_kaggle_set
from src.features import engineer_features
from src.metrics import best_ams_over_thresholds
from src.modeling import HiggsMLP, RANDOM_STATE

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "outputs" / "reports"

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "ReLU": nn.ReLU,
    "GELU": nn.GELU,
    "Swish": nn.SiLU,  # SiLU(x) = x * sigmoid(x) == Swish
}


class FocalLoss(nn.Module):
    """Focal Loss binaria sobre logits (Lin et al., 2017):

        FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Con gamma=0 y alpha=0.5 se reduce a BCE ponderada; gamma>0 des-enfatiza
    ejemplos faciles (p_t alto) para que el gradiente se concentre en los
    eventos dificiles de clasificar, algo relevante aqui porque la mayoria
    de eventos de fondo son "faciles" y dominarian el gradiente en BCE plano.
    """

    def __init__(self, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_term = (1.0 - p_t).clamp(min=1e-6) ** self.gamma
        return (alpha_t * focal_term * bce).mean()


def train_mlp_variant(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    activation: type[nn.Module],
    loss_fn: nn.Module,
    epochs: int = 25,
    batch_size: int = 2048,
    lr: float = 1e-3,
    patience: int = 5,
) -> tuple[HiggsMLP, StandardScaler, list[float], list[float]]:
    """Entrena una variante de HiggsMLP con la activacion y loss dadas.
    Devuelve el modelo, el scaler y las curvas de loss (train/val) por epoca.
    """
    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    torch.manual_seed(RANDOM_STATE)
    model = HiggsMLP(n_features=X_train.shape[1], activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    n = Xt.shape[0]
    train_losses, val_losses = [], []
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    rng = np.random.default_rng(RANDOM_STATE)

    for _epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        train_losses.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler, train_losses, val_losses


def mlp_predict_proba(model: HiggsMLP, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(scaler.transform(X), dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def run_comparison(epochs: int = 25) -> pd.DataFrame:
    """Entrena un HiggsMLP con FocalLoss para cada activacion en ACTIVATIONS
    y devuelve una tabla comparativa (AMS/AUC en public-test). Tambien
    guarda las curvas de loss/epoca y el grafico comparativo en
    outputs/reports/, y persiste el resultado en DuckDB (tabla
    `activation_comparison`)."""
    print("Cargando dataset ATLAS Higgs...")
    df = load_raw()
    train_df, public_df, _private_df = split_by_kaggle_set(df)

    X_train, medians = engineer_features(train_df)
    X_public, _ = engineer_features(public_df, medians_by_jet=medians)

    y_train = train_df["is_signal"].to_numpy()
    y_public = public_df["is_signal"].to_numpy()
    w_public = public_df["KaggleWeight"].to_numpy()

    loss_fn = FocalLoss(alpha=0.5, gamma=2.0)

    results = []
    curves: dict[str, dict[str, list[float]]] = {}

    for name, activation in ACTIVATIONS.items():
        print(f"Entrenando MLP con activacion={name}, loss=FocalLoss(alpha=0.5, gamma=2.0)...")
        model, scaler, train_losses, val_losses = train_mlp_variant(
            X_train.to_numpy(), y_train, X_public.to_numpy(), y_public,
            activation=activation, loss_fn=loss_fn, epochs=epochs,
        )
        proba = mlp_predict_proba(model, scaler, X_public.to_numpy())
        ams, threshold = best_ams_over_thresholds(y_public, proba, w_public)
        from sklearn.metrics import roc_auc_score

        results.append({
            "activation": name,
            "loss": "FocalLoss(alpha=0.5, gamma=2.0)",
            "ams": round(ams, 4),
            "best_threshold": round(float(threshold), 4),
            "auc": round(float(roc_auc_score(y_public, proba)), 4),
            "epochs_trained": len(train_losses),
        })
        curves[name] = {"train_loss": train_losses, "val_loss": val_losses}

    results_df = pd.DataFrame(results)
    print("\n=== Comparacion de activaciones (FocalLoss, public test) ===")
    print(results_df.to_string(index=False))

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(REPORTS_DIR / "activation_comparison.csv", index=False)
    with open(REPORTS_DIR / "activation_loss_curves.json", "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    _plot_loss_curves(curves)
    _plot_loss_curves_animated(curves)
    _plot_ams_comparison(results_df)

    try:
        from src.database import export_activation_comparison

        export_activation_comparison(results_df)
    except Exception as exc:  # pragma: no cover - DuckDB persistence is best-effort here
        print(f"[activation_comparison] no se pudo persistir en DuckDB: {exc}")

    return results_df


def _plot_loss_curves(curves: dict[str, dict[str, list[float]]]) -> None:
    sns.set_theme(style="whitegrid")
    colors = {"ReLU": "#4C72B0", "GELU": "#DD8452", "Swish": "#55A868"}
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, series in curves.items():
        ax.plot(range(1, len(series["val_loss"]) + 1), series["val_loss"],
                label=name, color=colors.get(name), linewidth=2)
    ax.set_title("FocalLoss (val) por epoca -- ReLU vs GELU vs Swish")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Focal loss (validacion)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "activation_loss_curves.png", dpi=150)
    plt.close(fig)


def _plot_loss_curves_animated(curves: dict[str, dict[str, list[float]]]) -> None:
    """Version animada (GIF) de _plot_loss_curves: dibuja las curvas de
    FocalLoss (val) por epoca de forma progresiva ("racing line chart"),
    con una etiqueta flotante en la punta de cada linea mostrando el valor
    actual. Usa exactamente los mismos datos reales que la version estatica
    (curves[name]["val_loss"]), sin inventar epocas adicionales."""
    plt.style.use("dark_background")
    colors = {"ReLU": "#4C72B0", "GELU": "#DD8452", "Swish": "#55A868"}

    n_epochs = max(len(series["val_loss"]) for series in curves.values())
    n_frames = min(n_epochs, 60)
    # Indices de epoca (1-based) que se revelan en cada frame, subsampleando
    # sobre los datos reales ya calculados (sin fabricar valores nuevos).
    frame_epochs = sorted(set(
        int(round(x)) for x in np.linspace(1, n_epochs, n_frames)
    ))

    all_val_losses = [v for series in curves.values() for v in series["val_loss"]]
    y_min, y_max = min(all_val_losses), max(all_val_losses)
    y_pad = (y_max - y_min) * 0.15 or 0.05

    fig, ax = plt.subplots(figsize=(12, 6))
    lines = {}
    annotations = {}
    for name, series in curves.items():
        (line,) = ax.plot([], [], label=name, color=colors.get(name), linewidth=2.5)
        lines[name] = line
        annotations[name] = ax.annotate(
            "", xy=(0, 0), xytext=(10, 0), textcoords="offset points",
            color="white", fontsize=10, fontweight="bold", va="center",
            bbox=dict(boxstyle="round,pad=0.35", fc=colors.get(name), ec="white", alpha=0.9),
        )

    ax.set_xlim(1, n_epochs)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)
    ax.set_title("FocalLoss (val) por epoca -- ReLU vs GELU vs Swish", color="white")
    ax.set_xlabel("Epoca")
    ax.set_ylabel("Focal loss (validacion)")
    ax.legend(loc="upper right")

    def update(frame_idx: int):
        cur_epoch = frame_epochs[frame_idx]
        for name, series in curves.items():
            val_loss = series["val_loss"]
            e = min(cur_epoch, len(val_loss))
            xs = range(1, e + 1)
            ys = val_loss[:e]
            lines[name].set_data(xs, ys)
            x_tip, y_tip = e, val_loss[e - 1]
            annotations[name].set_position((10, 0))
            annotations[name].xy = (x_tip, y_tip)
            annotations[name].set_text(f"{name}: {y_tip:.4f}")
        return list(lines.values()) + list(annotations.values())

    ani = FuncAnimation(fig, update, frames=len(frame_epochs), interval=150, blit=False)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ani.save(REPORTS_DIR / "activation_loss_curves_animated.gif", writer="pillow")
    plt.close(fig)
    plt.style.use("default")


def _plot_ams_comparison(results_df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid")
    colors = {"ReLU": "#4C72B0", "GELU": "#DD8452", "Swish": "#55A868"}
    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars = ax.bar(results_df["activation"], results_df["ams"],
                   color=[colors.get(a, "#8C8C8C") for a in results_df["activation"]])
    ax.set_title("AMS por activacion (MLP + FocalLoss, public test)")
    ax.set_ylabel("AMS")
    for bar, val in zip(bars, results_df["ams"]):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03, f"{val:.3f}", ha="center")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "activation_ams_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    run_comparison()
