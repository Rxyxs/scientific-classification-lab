"""Ingesta del dataset ATLAS Higgs (CERN Open Data, registro 328): union del
conjunto original de la competencia Kaggle 2014 + su extension oficial.

El split train/test no es aleatorio: se usa la columna KaggleSet original,
que reproduce exactamente los conjuntos de la competencia (t=train,
b=public leaderboard, v=private leaderboard) -- evita fuga y permite
comparar contra el leaderboard historico real.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "atlas-higgs.csv"

MISSING_SENTINEL = -999.0

FEATURE_COLUMNS = [
    "DER_mass_MMC", "DER_mass_transverse_met_lep", "DER_mass_vis", "DER_pt_h",
    "DER_deltaeta_jet_jet", "DER_mass_jet_jet", "DER_prodeta_jet_jet",
    "DER_deltar_tau_lep", "DER_pt_tot", "DER_sum_pt", "DER_pt_ratio_lep_tau",
    "DER_met_phi_centrality", "DER_lep_eta_centrality",
    "PRI_tau_pt", "PRI_tau_eta", "PRI_tau_phi",
    "PRI_lep_pt", "PRI_lep_eta", "PRI_lep_phi",
    "PRI_met", "PRI_met_phi", "PRI_met_sumet", "PRI_jet_num",
    "PRI_jet_leading_pt", "PRI_jet_leading_eta", "PRI_jet_leading_phi",
    "PRI_jet_subleading_pt", "PRI_jet_subleading_eta", "PRI_jet_subleading_phi",
    "PRI_jet_all_pt",
]


def load_raw() -> pd.DataFrame:
    df = pd.read_csv(RAW_PATH)
    df["is_signal"] = (df["Label"] == "s").astype(int)
    return df


def split_by_kaggle_set(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Reproduce el split original de la competencia: train / public LB / private LB.

    `u` (unused, 18.238 filas que ATLAS nunca asigno a ningun subconjunto de la
    competencia) se descarta explicitamente -- no es un split valido para
    comparar contra el leaderboard historico.
    """
    train = df[df["KaggleSet"] == "t"].reset_index(drop=True)
    public_test = df[df["KaggleSet"] == "b"].reset_index(drop=True)
    private_test = df[df["KaggleSet"] == "v"].reset_index(drop=True)
    return train, public_test, private_test
