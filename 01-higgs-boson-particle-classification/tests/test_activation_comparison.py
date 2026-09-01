import numpy as np
import torch

from src.activation_comparison import ACTIVATIONS, FocalLoss, train_mlp_variant, mlp_predict_proba
from src.modeling import HiggsMLP


def test_focal_loss_matches_bce_when_gamma_zero_and_alpha_half():
    torch.manual_seed(0)
    logits = torch.randn(32)
    targets = torch.randint(0, 2, (32,)).float()

    focal = FocalLoss(alpha=0.5, gamma=0.0)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    # Con gamma=0, (1-p_t)^gamma == 1, y alpha_t == 0.5 constante (alpha=0.5
    # aplicado tanto a targets=1 como targets=0), asi que FocalLoss == 0.5*BCE.
    assert abs(focal(logits, targets).item() - 0.5 * bce.item()) < 1e-5


def test_focal_loss_downweights_easy_examples():
    # Ejemplo "facil": logit muy negativo con target=0 (ya bien clasificado).
    easy_logits = torch.tensor([-8.0])
    easy_targets = torch.tensor([0.0])
    # Ejemplo "dificil": logit cercano a 0 con target=1 (incierto).
    hard_logits = torch.tensor([0.0])
    hard_targets = torch.tensor([1.0])

    focal = FocalLoss(alpha=0.5, gamma=2.0)
    easy_loss = focal(easy_logits, easy_targets).item()
    hard_loss = focal(hard_logits, hard_targets).item()

    assert easy_loss < hard_loss


def test_all_three_activations_are_registered():
    assert set(ACTIVATIONS) == {"ReLU", "GELU", "Swish"}
    assert ACTIVATIONS["Swish"] is torch.nn.SiLU


def test_higgs_mlp_accepts_custom_activation():
    model = HiggsMLP(n_features=5, hidden=(8, 4), activation=torch.nn.GELU)
    assert isinstance(model.net[2], torch.nn.GELU)


def test_train_mlp_variant_reduces_loss_on_toy_data():
    rng = np.random.default_rng(0)
    n, d = 400, 6
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
    X_train, y_train = X[:300], y[:300]
    X_val, y_val = X[300:], y[300:]

    loss_fn = FocalLoss(alpha=0.5, gamma=2.0)
    model, scaler, train_losses, val_losses = train_mlp_variant(
        X_train, y_train, X_val, y_val,
        activation=torch.nn.ReLU, loss_fn=loss_fn, epochs=10, batch_size=64,
    )

    assert len(train_losses) >= 1
    assert train_losses[-1] < train_losses[0]

    proba = mlp_predict_proba(model, scaler, X_val)
    assert proba.shape == (100,)
    assert (proba >= 0).all() and (proba <= 1).all()
