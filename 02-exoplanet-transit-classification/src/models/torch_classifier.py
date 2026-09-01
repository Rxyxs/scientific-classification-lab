"""PyTorch MLP comparison for KOI disposition classification, with a
ReLU/GELU/Swish activation ablation -- same protocol used across this
portfolio's other PyTorch comparisons (real train/val split, same seed,
same architecture except activation).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn

from src.features.engineering import LABEL_COLUMN, prepare_features, to_xy

ROOT = Path(__file__).resolve().parents[2]
RAW_PATH = ROOT / "data" / "raw" / "koi_cumulative.parquet"
REPORTS_DIR = ROOT / "reports"

ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "swish": nn.SiLU}


class KOIClassifierNet(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden: int = 64, activation: str = "relu"):
        super().__init__()
        act = ACTIVATIONS[activation]
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            act(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, hidden),
            act(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.3),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_one(X_train, y_train, X_val, y_val, n_classes: int, activation: str, epochs: int = 60, patience: int = 10):
    torch.manual_seed(42)
    model = KOIClassifierNet(X_train.shape[1], n_classes, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt, yt = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    Xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

    best_val_loss, best_epoch, best_state, bad_epochs = float("inf"), 0, None, 0
    val_losses = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch + 1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        val_pred = model(Xv).argmax(dim=1).numpy()

    return {
        "activation": activation,
        "best_epoch": best_epoch,
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_f1_macro": float(f1_score(y_val, val_pred, average="macro")),
    }


def main() -> None:
    raw = pl.read_parquet(RAW_PATH)
    df = prepare_features(raw)
    X, y_str, _ = to_xy(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    results = {}
    for name in ACTIVATIONS:
        print(f"Entrenando MLP con activacion '{name}'...")
        res = train_one(X_train, y_train, X_test, y_test, n_classes=len(label_encoder.classes_), activation=name)
        results[name] = res
        print(f"  [{name}] mejor epoch {res['best_epoch']} | accuracy val {res['val_accuracy']:.4f} | F1-macro {res['val_f1_macro']:.4f}")

    with open(REPORTS_DIR / "torch_activation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nComparacion guardada en: {REPORTS_DIR / 'torch_activation_comparison.json'}")


if __name__ == "__main__":
    main()
