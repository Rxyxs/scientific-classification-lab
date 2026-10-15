import numpy as np
import pytest
import torch

from src.models.torch_classifier import train_one


def _data(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6)).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(int) + (X[:, 2] > 1).astype(int)
    return X[:200], y[:200], X[200:250], y[200:250], X[250:], y[250:]


def test_minimum_epochs_and_metrics_on_test_split():
    Xt, yt, Xv, yv, Xs, ys = _data()
    res = train_one(Xt, yt, Xv, yv, n_classes=3, activation="relu", epochs=300, min_epochs=120, patience=3,
                    X_test=Xs, y_test=ys)
    assert res["epochs_run"] >= 120
    assert 0 <= res["test_accuracy"] <= 1


def test_resume_from_checkpoint_gives_same_result(tmp_path, monkeypatch):
    Xt, yt, Xv, yv, Xs, ys = _data(seed=1)
    kw = dict(n_classes=3, activation="gelu", epochs=40, min_epochs=40, patience=1000, X_test=Xs, y_test=ys)
    ref = train_one(Xt, yt, Xv, yv, **kw)

    ckpt = tmp_path / "koi.ckpt"
    real_step = torch.optim.Adam.step
    calls = {"n": 0}

    def flaky_step(self, *args, **k):
        calls["n"] += 1
        if calls["n"] > 25:  # corta en la epoca 26; el ultimo checkpoint es la 20
            raise KeyboardInterrupt
        return real_step(self, *args, **k)

    monkeypatch.setattr(torch.optim.Adam, "step", flaky_step)
    with pytest.raises(KeyboardInterrupt):
        train_one(Xt, yt, Xv, yv, checkpoint_path=ckpt, **kw)
    monkeypatch.setattr(torch.optim.Adam, "step", real_step)

    res = train_one(Xt, yt, Xv, yv, checkpoint_path=ckpt, **kw)
    assert res["test_accuracy"] == pytest.approx(ref["test_accuracy"])
    assert res["test_f1_macro"] == pytest.approx(ref["test_f1_macro"])
