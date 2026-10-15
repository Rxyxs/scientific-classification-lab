"""Iteracion de modelos: Decision Tree (baseline) -> Gradient Boosting
(LightGBM) -> Red Neuronal (PyTorch, Dropout+BatchNorm). Los 3 se evaluan
con AMS (no accuracy/AUC) sobre el public-test y el private-test del split
oficial de la competencia."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

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


MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 300
PATIENCE = 15
VAL_FRACTION = 0.15  # validacion estratificada tomada del split de train, nunca del public/private test


def split_validation(X_train: np.ndarray, y_train: np.ndarray, seed: int = RANDOM_STATE):
    """Separa una validacion estratificada del train oficial. El public-test y el
    private-test quedan solo para medir: antes el early stopping miraba el
    public-test, el mismo conjunto en que se reportaba el AMS."""
    from sklearn.model_selection import train_test_split

    return train_test_split(X_train, y_train, test_size=VAL_FRACTION, stratify=y_train, random_state=seed)


def _fingerprint(*arrays: np.ndarray, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    h.update(json.dumps(config, sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def fit_mlp(
    model: nn.Module, loss_fn: nn.Module, Xt: torch.Tensor, yt: torch.Tensor, Xv: torch.Tensor, yv: torch.Tensor,
    epochs: int = MAX_EPOCHS, min_epochs: int = MIN_EPOCHS, patience: int = PATIENCE,
    batch_size: int = 2048, lr: float = 1e-3, checkpoint_path: Path | None = None,
) -> tuple[list[float], list[float], int]:
    """Bucle comun de entrenamiento: al menos ``min_epochs`` epocas (o ``epochs``
    si es menor); desde ahi early stopping tras ``patience`` epocas sin mejora
    en la loss de validacion, restaurando la mejor epoca. Con
    ``checkpoint_path`` el estado completo se guarda cada epoca y una corrida
    interrumpida retoma donde quedo. Devuelve (loss train, loss val, mejor epoca)."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    fp = _fingerprint(Xt.numpy(), yt.numpy(), Xv.numpy(), yv.numpy(), model=repr(model), loss=repr(loss_fn),
                      epochs=epochs, min_epochs=min_epochs, patience=patience, batch=batch_size, lr=lr)
    state = {"epoch": 0, "train": [], "val": [], "best_val": float("inf"), "best_epoch": 0,
             "best_state": None, "bad_epochs": 0, "done": False}
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if ckpt.get("fingerprint") == fp:
            model.load_state_dict(ckpt["model"])
            optimizer.load_state_dict(ckpt["optimizer"])
            state = ckpt["state"]

    def save_checkpoint() -> None:
        if checkpoint_path is None:
            return
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        torch.save({"fingerprint": fp, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "state": state}, tmp)
        tmp.replace(path)

    n = Xt.shape[0]
    while not state["done"] and state["epoch"] < epochs:
        rng = np.random.default_rng(RANDOM_STATE + state["epoch"])  # mismo orden de batches al retomar
        torch.manual_seed(RANDOM_STATE + 10_000 + state["epoch"])
        model.train()
        perm = rng.permutation(n)
        epoch_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        state["epoch"] += 1
        state["train"].append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        state["val"].append(val_loss)
        if val_loss < state["best_val"] - 1e-7:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        save_checkpoint()

    if state["best_state"] is not None:
        model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    return state["train"], state["val"], state["best_epoch"]


def train_mlp(
    X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray,
    epochs: int = MAX_EPOCHS, batch_size: int = 2048, lr: float = 1e-3, patience: int = PATIENCE,
    min_epochs: int = MIN_EPOCHS, checkpoint_path: Path | None = None,
) -> tuple[HiggsMLP, StandardScaler]:
    """``X_val``/``y_val`` deben salir del train (``split_validation``), no del test."""
    scaler = StandardScaler().fit(X_train)
    Xt = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    Xv = torch.tensor(scaler.transform(X_val), dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)

    torch.manual_seed(RANDOM_STATE)
    model = HiggsMLP(n_features=X_train.shape[1])
    _, _, best_epoch = fit_mlp(model, nn.BCEWithLogitsLoss(), Xt, yt, Xv, yv, epochs=epochs, min_epochs=min_epochs,
                               patience=patience, batch_size=batch_size, lr=lr, checkpoint_path=checkpoint_path)
    print(f"  MLP: mejor epoca {best_epoch}")
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
