"""Iteracion de modelos: Decision Tree (baseline) -> Gradient Boosting
(LightGBM) -> Red Neuronal (PyTorch, Dropout+BatchNorm). Los 3 se evaluan
con AMS (no accuracy/AUC) sobre el public-test y el private-test del split
oficial de la competencia."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMClassifier

from src.metrics import ams_score, best_ams_over_thresholds

RANDOM_STATE = 42


def train_baseline_tree(X_train, y_train) -> DecisionTreeClassifier:
    model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train) -> LGBMClassifier:
    model = LGBMClassifier(
        n_estimators=600,
        num_leaves=63,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=30,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


class HiggsMLP(nn.Module):
    """MLP tabular con Dropout + BatchNorm -- estabiliza el entrenamiento con
    ~30 features de escalas muy dispares (GeV vs. angulos en radianes)."""

    def __init__(
        self,
        n_features: int,
        hidden: tuple[int, ...] = (128, 64, 32),
        dropout: float = 0.3,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            act = activation() if activation is not None else nn.ReLU()
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), act, nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = 40, batch_size: int = 2048, lr: float = 1e-3, patience: int = 5,
) -> tuple[HiggsMLP, StandardScaler]:
    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    model = HiggsMLP(n_features=X_train.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = nn.BCEWithLogitsLoss()

    n = Xt.shape[0]
    best_val_loss, best_state, bad_epochs = float("inf"), None, 0
    rng = np.random.default_rng(RANDOM_STATE)

    for epoch in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            xb, yb = Xt[idx], yt[idx]
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_val_loss:
            best_val_loss, best_state, bad_epochs = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return model, scaler


def mlp_predict_proba(model: HiggsMLP, scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(scaler.transform(X), dtype=torch.float32))
        return torch.sigmoid(logits).numpy()


def evaluate_model(name: str, y_proba: np.ndarray, y_true: np.ndarray, weights: np.ndarray) -> dict:
    ams, threshold = best_ams_over_thresholds(y_true, y_proba, weights)
    pred_at_threshold = (y_proba >= threshold).astype(int)
    from sklearn.metrics import roc_auc_score, accuracy_score

    return {
        "model": name,
        "ams": round(ams, 4),
        "best_threshold": round(float(threshold), 4),
        "auc": round(float(roc_auc_score(y_true, y_proba)), 4),
        "accuracy_at_threshold": round(float(accuracy_score(y_true, pred_at_threshold)), 4),
    }
