"""PyTorch MLP comparison for KOI disposition classification, with a
ReLU/GELU/Swish activation ablation -- same protocol used across this
portfolio's other PyTorch comparisons (real train/val split, same seed,
same architecture except activation).
"""
from __future__ import annotations

import copy
import hashlib
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


MIN_EPOCHS = 100   # ninguna red se detiene antes de 100 epocas
MAX_EPOCHS = 3000  # epocas full-batch: cada una es un solo paso de gradiente
PATIENCE = 50
VAL_FRACTION = 0.15
CHECKPOINT_EVERY = 10
CHECKPOINT_DIR = ROOT / "outputs" / "models" / "checkpoints"


def _fingerprint(*arrays, **config) -> str:
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(np.asarray(a, dtype=np.float32)).tobytes())
    h.update(json.dumps(config, sort_keys=True).encode())
    return h.hexdigest()[:16]


def train_one(X_train, y_train, X_val, y_val, n_classes: int, activation: str,
              epochs: int = MAX_EPOCHS, patience: int = PATIENCE, min_epochs: int = MIN_EPOCHS,
              X_test=None, y_test=None, checkpoint_path: Path | None = None):
    """Entrena con early stopping sobre (X_val, y_val), que debe salir del train;
    al menos ``min_epochs`` epocas, desde ahi corta tras ``patience`` sin mejora y
    restaura la mejor epoca. El estado se guarda cada ``CHECKPOINT_EVERY`` epocas.
    Las metricas finales se calculan sobre (X_test, y_test), que el
    entrenamiento nunca mira."""
    torch.manual_seed(42)
    model = KOIClassifierNet(X_train.shape[1], n_classes, activation=activation)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    Xt, yt = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)
    Xv, yv = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)

    fp = _fingerprint(X_train, y_train, X_val, y_val, activation=activation, epochs=epochs,
                      patience=patience, min_epochs=min_epochs)
    state = {"epoch": 0, "val_losses": [], "best_val": float("inf"), "best_epoch": 0, "best_state": None,
             "bad_epochs": 0, "done": False}
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

    while not state["done"] and state["epoch"] < epochs:
        torch.manual_seed(42 + state["epoch"])  # mascaras de dropout reproducibles al retomar
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(Xt), yt)
        loss.backward()
        optimizer.step()
        state["epoch"] += 1

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        state["val_losses"].append(val_loss)
        if val_loss < state["best_val"] - 1e-7:
            state.update(best_val=val_loss, best_epoch=state["epoch"], bad_epochs=0,
                         best_state=copy.deepcopy(model.state_dict()))
        else:
            state["bad_epochs"] += 1
        if state["epoch"] >= min(min_epochs, epochs) and state["bad_epochs"] >= patience:
            state["done"] = True
        if state["epoch"] % CHECKPOINT_EVERY == 0:
            save_checkpoint()

    model.load_state_dict(state["best_state"])
    state["done"] = True
    save_checkpoint()
    model.eval()
    X_eval, y_eval = (X_val, y_val) if X_test is None else (X_test, y_test)
    with torch.no_grad():
        pred = model(torch.tensor(X_eval, dtype=torch.float32)).argmax(dim=1).numpy()

    return {
        "activation": activation,
        "best_epoch": state["best_epoch"],
        "epochs_run": state["epoch"],
        "test_accuracy": float(accuracy_score(y_eval, pred)),
        "test_f1_macro": float(f1_score(y_eval, pred, average="macro")),
    }


def main() -> None:
    raw = pl.read_parquet(RAW_PATH)
    df = prepare_features(raw)
    X, y_str, _ = to_xy(df)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    # Validacion para early stopping: 15% estratificado del train. El test (25%, mismo split que XGBoost)
    # solo se usa para las metricas finales; antes el early stopping miraba el test.
    X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=VAL_FRACTION, random_state=42,
                                                  stratify=y_train)
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(X_fit)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    results = {}
    for name in ACTIVATIONS:
        print(f"Entrenando MLP con activacion '{name}'...")
        res = train_one(X_fit, y_fit, X_val, y_val, n_classes=len(label_encoder.classes_), activation=name,
                        X_test=X_test, y_test=y_test, checkpoint_path=CHECKPOINT_DIR / f"mlp_{name}.ckpt")
        results[name] = res
        print(f"  [{name}] mejor epoch {res['best_epoch']} de {res['epochs_run']} | accuracy test "
              f"{res['test_accuracy']:.4f} | F1-macro {res['test_f1_macro']:.4f}")

    with open(REPORTS_DIR / "torch_activation_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nComparacion guardada en: {REPORTS_DIR / 'torch_activation_comparison.json'}")


if __name__ == "__main__":
    main()
