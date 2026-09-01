"""API FastAPI de scoring en tiempo real (modelo LightGBM, el de mejor AMS).

    uvicorn src.api:app --reload
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data import FEATURE_COLUMNS
from src.features import JET_DEPENDENT_COLUMNS, RECONSTRUCTION_DEPENDENT_COLUMNS, engineer_features

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "outputs" / "models"

app = FastAPI(title="Higgs Boson Event Classifier", version="1.0.0")

_models = None
_medians = None


def _lazy_load():
    global _models, _medians
    if _models is None:
        _models = joblib.load(MODELS_DIR / "gbm_models.joblib")
        _medians = joblib.load(MODELS_DIR / "jet_medians.joblib")


class HiggsEvent(BaseModel):
    DER_mass_MMC: float
    DER_mass_transverse_met_lep: float
    DER_mass_vis: float
    DER_pt_h: float
    DER_deltaeta_jet_jet: float
    DER_mass_jet_jet: float
    DER_prodeta_jet_jet: float
    DER_deltar_tau_lep: float
    DER_pt_tot: float
    DER_sum_pt: float
    DER_pt_ratio_lep_tau: float
    DER_met_phi_centrality: float
    DER_lep_eta_centrality: float
    PRI_tau_pt: float
    PRI_tau_eta: float
    PRI_tau_phi: float
    PRI_lep_pt: float
    PRI_lep_eta: float
    PRI_lep_phi: float
    PRI_met: float
    PRI_met_phi: float
    PRI_met_sumet: float
    PRI_jet_num: int
    PRI_jet_leading_pt: float
    PRI_jet_leading_eta: float
    PRI_jet_leading_phi: float
    PRI_jet_subleading_pt: float
    PRI_jet_subleading_eta: float
    PRI_jet_subleading_phi: float
    PRI_jet_all_pt: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/score")
def score(event: HiggsEvent):
    _lazy_load()
    row = pd.DataFrame([event.model_dump()])
    X, _ = engineer_features(row, medians_by_jet=_medians)
    proba = float(_models["lgbm"].predict_proba(X)[0, 1])
    return {"signal_probability": round(proba, 6), "predicted_label": "s" if proba >= 0.5 else "b"}
