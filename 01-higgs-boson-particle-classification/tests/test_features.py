import pandas as pd

from src.data import MISSING_SENTINEL, load_raw, split_by_kaggle_set
from src.features import JET_DEPENDENT_COLUMNS, engineer_features


def test_missing_flag_matches_sentinel():
    df = pd.DataFrame(
        {
            "DER_mass_MMC": [100.0, MISSING_SENTINEL],
            "DER_pt_h": [10.0, 20.0],
            "DER_deltaeta_jet_jet": [1.0, MISSING_SENTINEL],
            "DER_mass_jet_jet": [50.0, MISSING_SENTINEL],
            "DER_prodeta_jet_jet": [0.5, MISSING_SENTINEL],
            "PRI_jet_leading_pt": [30.0, MISSING_SENTINEL],
            "PRI_jet_leading_eta": [0.1, MISSING_SENTINEL],
            "PRI_jet_leading_phi": [0.2, MISSING_SENTINEL],
            "PRI_jet_subleading_pt": [MISSING_SENTINEL, MISSING_SENTINEL],
            "PRI_jet_subleading_eta": [MISSING_SENTINEL, MISSING_SENTINEL],
            "PRI_jet_subleading_phi": [MISSING_SENTINEL, MISSING_SENTINEL],
            "PRI_jet_num": [1, 0],
            "DER_mass_transverse_met_lep": [1.0, 1.0], "DER_mass_vis": [1.0, 1.0],
            "DER_pt_tot": [1.0, 1.0], "DER_sum_pt": [10.0, 10.0],
            "DER_pt_ratio_lep_tau": [1.0, 1.0], "DER_met_phi_centrality": [1.0, 1.0],
            "DER_lep_eta_centrality": [1.0, 1.0], "DER_deltar_tau_lep": [1.0, 1.0],
            "PRI_tau_pt": [1.0, 1.0],
            "PRI_tau_eta": [1.0, 1.0], "PRI_tau_phi": [1.0, 1.0],
            "PRI_lep_pt": [1.0, 1.0], "PRI_lep_eta": [1.0, 1.0], "PRI_lep_phi": [1.0, 1.0],
            "PRI_met": [1.0, 1.0], "PRI_met_phi": [1.0, 1.0], "PRI_met_sumet": [1.0, 1.0],
            "PRI_jet_all_pt": [1.0, 1.0],
        }
    )
    features, _ = engineer_features(df)
    assert features["DER_mass_jet_jet_missing"].tolist() == [0, 1]
    assert features["DER_mass_MMC_missing"].tolist() == [0, 1]
    # imputado: no debe quedar ningun -999 tras engineer_features
    assert not (features[JET_DEPENDENT_COLUMNS] == MISSING_SENTINEL).any().any()


def test_kaggle_set_split_is_disjoint_and_matches_official_sizes():
    df = load_raw()
    train, public, private = split_by_kaggle_set(df)
    assert len(train) == 250_000
    assert len(public) == 100_000
    assert len(private) == 450_000
    ids = set(train["EventId"]) | set(public["EventId"]) | set(private["EventId"])
    assert len(ids) == len(train) + len(public) + len(private)  # sin solapamiento
