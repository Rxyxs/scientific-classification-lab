[ 🇺🇸 English ] | [ 🇨🇱 Leer en Español ](README.es.md)

# Scientific Classification Lab

Two physical-sciences classification problems, same core approach: extract domain-specific features from raw scientific data, then compare a gradient-boosted model against a neural network. Each folder is self-contained with its own README, dependencies, and tests. This repo replaces two separate single-domain repos that used to live on this profile.

## Techniques

| # | Domain | Folder | What it does |
|---|---|---|---|
| 01 | Particle physics (ATLAS/CERN) | [`01-higgs-boson-particle-classification`](01-higgs-boson-particle-classification) | Classifies Higgs-to-tau-tau decay events vs. background from kinematic variables: gradient-boosted model vs. PyTorch neural network, evaluated with the AMS metric, served via FastAPI. |
| 02 | Exoplanet detection (Kepler) | [`02-exoplanet-transit-classification`](02-exoplanet-transit-classification) | Classifies Kepler Objects of Interest as confirmed exoplanets vs. false positives from transit-signal features. |

## What the two projects found

Both use real scientific data with published ground truth — no synthetic generation, no simulated labels. Every number comes from an actual run.

| # | Project | Headline number | What it actually says |
|---|---|---|---|
| **01** | Higgs boson (CERN/ATLAS, 818k events) | AMS **2.931 → 3.641** across the iteration | Decision tree → LightGBM → PyTorch MLP → Optuna-tuned LightGBM. The tuning gain is verified on a **450,000-event private set never touched during model selection** (3.629, within 0.33% of the public-test value) |
| **01** | — same folder | 2014 Kaggle winners reached AMS ≈ **3.8–3.9** | Stated explicitly. This project reaches 3.63–3.64 with no ensembling, and reports that as the result of its iteration rather than as a leaderboard-matching claim |
| **01** | — same folder | Julia cross-check: **0.0000 difference**, and a **0.0235-wide threshold plateau** | A fresh from-scratch AMS implementation in Julia matches Python exactly. Its speed then buys a 2,000-threshold sweep that answers a question the 200-point Python sweep can't: the optimum is a **wide plateau (0.7979–0.8214), not a knife edge** |
| **02** | Kepler exoplanets (NASA, 9,564 real KOIs) | XGBoost **0.793** accuracy / **0.756** F1-macro vs. a 0.498 majority-class baseline | Downloaded live from NASA's public TAP API, no manual dataset, no key |
| **02** | — same folder | `koi_fpflag_*` columns **deliberately excluded** | Those flags are the Kepler vetting pipeline's own sub-decisions. Including them would predict the label from the verdict instead of from the observed physics — the single most consequential choice in the project, and it costs accuracy |
| **02** | — same folder | `CANDIDATE` is the hardest class (F1 = **0.58**) | And that is physically correct: it is the class of genuinely unresolved objects, where the astronomers hadn't decided either |

---

## Evidence

### Iteration on a real physics problem, and where it stops

![AMS by model, Higgs](01-higgs-boson-particle-classification/outputs/reports/ams_comparison.png)

**How to read it.** AMS (Approximate Median Significance) on the public test set at each model's own optimal threshold — the official metric of the 2014 ATLAS challenge, not accuracy. Higher is better. The three bars are the untuned comparison; the Optuna-tuned LightGBM that reaches 3.641 is not in this figure.

The jump that matters is the first one: a decision-tree baseline at 2.931 to LightGBM at 3.553, a 21% gain. After that the returns flatten — **the PyTorch MLP at 3.578 edges LightGBM by 0.025**, a difference small enough that it is reported as a tie in substance rather than a win for the neural network. Tuning then adds 0.088 on top of LightGBM, and that gain is confirmed on a private set that was never used for selection.

The honest ceiling is stated rather than hidden: the challenge's 2014 winners reached roughly 3.8–3.9 with heavily tuned ensembles. This project reaches 3.63–3.64 without ensembling at all.

### On tabular data, the neural network is not automatically the answer

![Model comparison, exoplanets](02-exoplanet-transit-classification/reports/figures/model_comparison.png)

**How to read it.** Held-out accuracy on 9,564 real Kepler Objects of Interest, three classes (CONFIRMED / CANDIDATE / FALSE POSITIVE). The dashed line at 0.498 is the majority-class baseline — anything below it is worse than guessing the most common answer. Three PyTorch activations in grey, XGBoost in blue. *(The legend box partially covers the ReLU and GELU labels; their values are 0.763 and 0.760, from the folder's results table.)*

XGBoost wins at 0.793 against the best MLP at 0.763. Taken together with the Higgs figure above, the pair makes a point neither makes alone: on 818k physics events the network ties the gradient-boosted trees, and on 9,564 tabular KOIs it loses to them. Neither result is dressed up as a verdict on deep learning — the folders report what each run produced and name the dataset size and shape as the likely reason.

---

## The pattern across both

Two problems from two sciences, and one shared discipline:

> **The result is defined by what was left out.**

- **02 excludes the `koi_fpflag_*` columns.** They would have predicted the disposition almost perfectly, because they *are* the vetting pipeline's intermediate verdicts. Keeping them would have produced a far better number and a worthless model. The project predicts from observed transit and stellar physics instead, and pays for it in accuracy.
- **01 excludes the private test set from every selection decision**, then uses it once to check that a 40-trial Optuna gain was real rather than fitted to the search.
- **01 also refuses the easy framing.** Reaching 3.63 is reported next to the 3.8–3.9 the 2014 winners achieved, rather than alone.
- **Neither project reports a neural network as the winner where it wasn't.** In 01 the MLP's 0.025 edge over untuned LightGBM is treated as a tie; in 02 XGBoost simply wins and that is what the folder says.

The `CANDIDATE` class result is the clearest illustration of why this matters. Its F1 of 0.58 looks like the model's weakest point until you notice what the class is: KOIs that the Kepler vetting process itself left unresolved. A model that scored 0.95 there would be worth distrusting.

---

## Why one repo instead of two

Both projects are real, runnable, and independently tested — this isn't about hiding scope, it's about representing it accurately. Two repos in two unrelated-sounding domains (particle physics, astronomy) hide the fact that they share the same core technique — feature engineering from raw scientific measurements into a supervised classifier, GBM vs. neural net compared head-to-head; one lab makes that shared method the actual point.

## Running a technique

Each folder is self-contained — see its own README for the exact setup and entry point, real results from an actual run, and any honest negative findings.

## Author

Pablo Reyes — [github.com/Rxyxs](https://github.com/Rxyxs)
Code: MIT — see [LICENSE](LICENSE)
