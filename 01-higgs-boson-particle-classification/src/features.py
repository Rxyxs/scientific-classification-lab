"""Feature engineering para el dataset ATLAS Higgs.

El -999.0 no es un dato faltante aleatorio: es la convencion de ATLAS para
"variable fisicamente indefinida" -- p. ej. una masa invariante dijet
(`DER_mass_jet_jet`) no existe si el evento tiene menos de 2 jets
(`PRI_jet_num < 2`). Imputar con la mediana global sin mas mezclaria
eventos de 0 jets con eventos de 2+ jets, perdiendo exactamente la
informacion fisica que la variable buscaba capturar. Aqui la imputacion es
por grupo de `PRI_jet_num` (misma topologia de evento), mas un flag binario
explicito por columna -- para que el modelo pueda distinguir "el evento no
tiene esta variable" de "la variable vale la mediana".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import FEATURE_COLUMNS, MISSING_SENTINEL

# Columnas cuya indefinicion depende de PRI_jet_num (verificado empiricamente:
# DER_mass_jet_jet/DER_deltaeta_jet_jet/DER_prodeta_jet_jet son -999 en el
# 100% de eventos con jet_num<2; las *_subleading en jet_num<2; DER_mass_MMC
# es indefinida por fallo de reconstruccion, no por jet_num).
JET_DEPENDENT_COLUMNS = [
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi",
]
RECONSTRUCTION_DEPENDENT_COLUMNS = ["DER_mass_MMC", "DER_pt_h"]


def engineer_features(df: pd.DataFrame, medians_by_jet: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Imputa por grupo de PRI_jet_num, agrega flags de faltante y 3 ratios
    cinematicos derivados. Devuelve (features, medianas_por_jet_usadas) --
    las medianas se calculan SOLO sobre el set pasado (tipicamente train) y
    se reutilizan para transformar validation/test, evitando fuga."""
    out = df.copy()

    missing_cols = JET_DEPENDENT_COLUMNS + RECONSTRUCTION_DEPENDENT_COLUMNS
    for col in missing_cols:
        out[f"{col}_missing"] = (out[col] == MISSING_SENTINEL).astype(int)

    if medians_by_jet is None:
        medians_by_jet = (
            out[out[JET_DEPENDENT_COLUMNS].ne(MISSING_SENTINEL).all(axis=1)]
            .groupby("PRI_jet_num")[JET_DEPENDENT_COLUMNS]
            .median()
        )

    for col in JET_DEPENDENT_COLUMNS:
        fallback = out[col].replace(MISSING_SENTINEL, np.nan).median()
        jet_medians = out["PRI_jet_num"].map(medians_by_jet[col]).fillna(fallback)
        mask = out[col] == MISSING_SENTINEL
        out.loc[mask, col] = jet_medians[mask]

    for col in RECONSTRUCTION_DEPENDENT_COLUMNS:
        median = out.loc[out[col] != MISSING_SENTINEL, col].median()
        out.loc[out[col] == MISSING_SENTINEL, col] = median

    # Ratios cinematicos derivados (sobre variables ya imputadas):
    out["ratio_met_sumpt"] = out["PRI_met"] / (out["DER_sum_pt"] + 1.0)
    out["ratio_tau_lep_pt"] = out["PRI_tau_pt"] / (out["PRI_lep_pt"] + 1.0)
    out["delta_phi_tau_met"] = np.abs(np.mod(out["PRI_tau_phi"] - out["PRI_met_phi"] + np.pi, 2 * np.pi) - np.pi)

    feature_cols = (
        FEATURE_COLUMNS
        + [f"{c}_missing" for c in missing_cols]
        + ["ratio_met_sumpt", "ratio_tau_lep_pt", "delta_phi_tau_met"]
    )
    return out[feature_cols], medians_by_jet
