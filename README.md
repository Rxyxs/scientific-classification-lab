[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Scientific Classification Lab

Two physical-sciences classification problems, same core approach: extract domain-specific features from raw scientific data, then compare a gradient-boosted model against a neural network. Each folder is self-contained with its own README, dependencies, and tests. This repo replaces two separate single-domain repos that used to live on this profile.

## Techniques

| # | Domain | Folder | What it does |
|---|---|---|---|
| 01 | Particle physics (ATLAS/CERN) | [`01-higgs-boson-particle-classification`](01-higgs-boson-particle-classification) | Classifies Higgs-to-tau-tau decay events vs. background from kinematic variables: gradient-boosted model vs. PyTorch neural network, evaluated with the AMS metric, served via FastAPI. |
| 02 | Exoplanet detection (Kepler) | [`02-exoplanet-transit-classification`](02-exoplanet-transit-classification) | Classifies Kepler Objects of Interest as confirmed exoplanets vs. false positives from transit-signal features. |

## Why one repo instead of two

Both projects are real, runnable, and independently tested — this isn't about hiding scope, it's about representing it accurately. Two repos in two unrelated-sounding domains (particle physics, astronomy) hide the fact that they share the same core technique — feature engineering from raw scientific measurements into a supervised classifier, GBM vs. neural net compared head-to-head; one lab makes that shared method the actual point.

## Running a technique

Each folder is self-contained — see its own README for the exact setup and entry point, real results from an actual run, and any honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
