[ 🇺🇸 English ] | [ 🇨🇱 [Leer en Español](README.es.md) ]

# hunting-exoplanets-kepler

Classifying **Kepler Objects of Interest (KOI)** — CONFIRMED / CANDIDATE / FALSE POSITIVE — on the real catalog published by **NASA's Exoplanet Archive** (public API, no key required). Same spirit as `hunting-the-higgs-boson`: real scientific data, no synthetic data anywhere, honest results even when they're not perfect.

## The data

`src/data/fetch_koi.py` downloads the `cumulative` table live via NASA's TAP API (`https://exoplanetarchive.ipac.caltech.edu/TAP/sync`) — **9,564 real KOIs**, each with its real published disposition after Kepler pipeline vetting and/or human review. No authentication, no hand-downloaded dataset.

```bash
python -m src.data.fetch_koi
```

**Honest note on features**: only transit and stellar parameters are used (`koi_period`, `koi_depth`, `koi_prad`, `koi_model_snr`, `koi_steff`, etc.) — the `koi_fpflag_*` columns are deliberately excluded, since they're sub-decisions of the vetting pipeline itself and would leak the label almost perfectly. The goal is predicting disposition from observed physics, not from the verdict already reached.

## Task and models

```bash
python -m src.models.train              # baseline + XGBoost + SHAP
python -m src.models.torch_classifier    # ReLU/GELU/Swish ablation (PyTorch)
python -m src.visualization.plots        # generates all 4 charts in reports/figures/
```

| Model | Accuracy (holdout) | F1-macro |
|---|---|---|
| Baseline (majority class) | 0.498 | 0.222 |
| PyTorch MLP (Swish) | 0.748 | 0.711 |
| PyTorch MLP (GELU) | 0.760 | 0.722 |
| PyTorch MLP (ReLU) | 0.763 | 0.722 |
| **XGBoost** | **0.793** | **0.756** |

![Model comparison](reports/figures/model_comparison.png)

**Honest finding**: XGBoost beats all three MLP variants on this tabular dataset — consistent with the well-known pattern that gradient-boosted trees tend to outperform dense networks on moderate-sized tabular features (9,200 rows, 12 features). No PyTorch run is reported as the winner because none was.

![Confusion matrix](reports/figures/confusion_matrix.png)

The `CANDIDATE` class is hardest (F1=0.58) — and that makes physical sense: it's the class of genuinely unresolved KOIs, neither confirmed nor ruled out, so it's expected that the model (and astronomers themselves) carry more uncertainty there than on already-settled cases.

![SHAP importance](reports/figures/shap_importance.png)

`koi_model_snr` (transit-model signal-to-noise ratio) dominates — matches physical intuition: a weak, noisy transit signal is the most common reason a KOI ends up as a false positive or stays unresolved.

![Class distribution](reports/figures/class_distribution.png)

## Interactive chart

`python -m src.visualization.interactive` produces an interactive scatter (Plotly, self-contained HTML) of the **2,300 real holdout predictions**: orbital period vs. transit-model SNR (both log-scaled), colored by real disposition, with hover text showing real vs. predicted disposition, transit depth, and planet radius. `X` markers are the model's actual misclassifications — you can see live where it fails, not just the aggregate accuracy number.

**[View the interactive chart](https://htmlpreview.github.io/?https://github.com/Rxyxs/scientific-classification-lab/blob/main/02-exoplanet-transit-classification/outputs/interactive/koi_feature_space.html)**

## Techniques used

- **Real data ingestion**: live download via NASA's TAP API (`requests` + SQL-like query), no hand-downloaded dataset or static cache.
- **Feature engineering**: `log1p` transform on 5 heavily right-skewed physical quantities (period, depth, radius, insolation, SNR) so tree/NN models aren't dominated by extreme outliers; deliberate exclusion of `koi_fpflag_*` columns to avoid label leakage.
- **Modeling**: majority-class baseline (`DummyClassifier`) → multiclass XGBoost (`multi:softprob`, 300 trees, depth 5) → PyTorch MLP activation ablation (ReLU/GELU/Swish, BatchNorm+Dropout, early stopping on validation loss).
- **Explainability**: SHAP (`TreeExplainer`) on the XGBoost model — mandatory, not optional, for a scientific use case where "why" matters as much as accuracy.
- **Evaluation**: stratified 75/25 split (`random_state=42`), accuracy + F1-macro (more informative than accuracy alone given the 3-class imbalance), confusion matrix and per-class classification report.
- **Tests**: 6 pytest tests, offline against the already-downloaded parquet, that verify the repo's central claim (XGBoost beats the baseline) as a reproducible assertion, not just a README number.

## Tests

```bash
pytest
```

6 tests, all offline against the already-downloaded parquet: real labels verified, no nulls after cleaning, and the repo's central claim verified as a reproducible test (XGBoost beats the baseline by a real margin, not just asserted in the README).

## Installation

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt   # Windows
```

## Stack

Python · Polars · XGBoost · PyTorch · SHAP · scikit-learn · NASA Exoplanet Archive TAP API

## Author

Pablo Reyes — Data Scientist, Santiago, Chile.
