"""AMS (Approximate Median Significance) -- la metrica oficial del HiggsML
Challenge, no una metrica de sklearn. Formula (Cowan et al., usada por
ATLAS/Kaggle):

    AMS = sqrt( 2 * ((s+b+b_reg) * ln(1 + s/(b+b_reg)) - s) )

donde `s` es la suma de pesos de eventos senal (Label='s') correctamente
predichos como senal, `b` la suma de pesos de eventos background
predichos (incorrectamente) como senal, y b_reg=10 es un termino de
regularizacion que evita AMS mal definida cuando b es muy chico.
"""

from __future__ import annotations

import numpy as np

B_REG = 10.0


def ams_score(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray) -> float:
    """y_true/y_pred binarios (1=senal). `weights` = KaggleWeight del subset evaluado."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    weights = np.asarray(weights)

    predicted_signal = y_pred == 1
    s = weights[predicted_signal & (y_true == 1)].sum()
    b = weights[predicted_signal & (y_true == 0)].sum()

    radicand = 2.0 * ((s + b + B_REG) * np.log(1.0 + s / (b + B_REG)) - s)
    return float(np.sqrt(radicand)) if radicand > 0 else 0.0


def best_ams_over_thresholds(
    y_true: np.ndarray, y_proba: np.ndarray, weights: np.ndarray, n_thresholds: int = 200
) -> tuple[float, float]:
    """Barre umbrales de probabilidad y devuelve (mejor_AMS, umbral_optimo)."""
    thresholds = np.linspace(0.0, 1.0, n_thresholds)
    best_ams, best_threshold = 0.0, 0.5
    for t in thresholds:
        pred = (y_proba >= t).astype(int)
        ams = ams_score(y_true, pred, weights)
        if ams > best_ams:
            best_ams, best_threshold = ams, t
    return best_ams, best_threshold
