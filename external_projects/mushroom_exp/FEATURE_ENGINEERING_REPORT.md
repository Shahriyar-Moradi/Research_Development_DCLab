# Mushroom — Feature Engineering & Selection Report

**Project:** `external_projects/mushroom_exp/`  
**Source:** [https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)  
**Rows / features (raw):** 8124 / 22  
**Positive rate:** 0.482  
**Protocol:** stratified 80/20, seed 42; all stateful FE fit on **train only**

---

## 1. Problem & decision-time policy

Classify mushrooms as edible vs poisonous from categorical traits.

### Leakage / safe vs unsafe
- **Has classic leakage columns:** False
- **Unsafe-only features:** `[]`
- **Rationale:** Physical traits are observed before edibility label; no leakage identified.

When leakage exists, **safe** drops post-outcome columns (HyperAck analog of final fares).  
When it does not, safe and unsafe matrices are identical — the ladder still documents FE and optimization lifts.

---

## 2. Feature engineering ladder (what we tried)

| Stage | Idea | Why (HyperAck playbook) |
|---|---|---|
| raw | Original numeric matrix | Floor score |
| logs | `log1p` on skewed non-negative cols | Tame heavy tails |
| ratios | Pairwise intensity ratios | Scale-normalized signals |
| interactions | Products of top-MI features | Non-linear combinations |
| full_fe | + KMeans clusters + quantile bins | Spatial/soft thresholds (train-fit) |
| selected | Top mutual-information subset | Test dilution vs compact set |

---

## 3. Safe-mode experiment results

| ID | Experiment | FE stage | #Feats | ROC-AUC | Recall | F1 | Optimization |
|---|---|---|---:|---:|---:|---:|---|
| 01 | baseline_logistic | raw | 22 | 0.9996 | 0.9808 | 0.9871 | none |
| 02 | baseline_lightgbm | raw | 22 | 1.0000 | 1.0000 | 1.0000 | none |
| 03 | fe_log_transforms | logs | 31 | 1.0000 | 1.0000 | 1.0000 | none |
| 04 | fe_ratios | ratios | 46 | 1.0000 | 1.0000 | 1.0000 | none |
| 05 | fe_interactions | interactions | 54 | 1.0000 | 1.0000 | 1.0000 | none |
| 06 | fe_full_clusters_bins | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | none |
| 07 | feature_selection_mi | selected | 25 | 1.0000 | 1.0000 | 1.0000 | mutual_info_selection |
| 08 | tuned_xgboost | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | RandomizedSearchCV_roc_auc |
| 09 | tuned_lightgbm | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | RandomizedSearchCV_roc_auc |
| 10 | hist_gradient_boosting | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | none |
| 11 | extra_trees_strong | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | hand_tuned |
| 12 | calibrated_lgbm_threshold | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | isotonic_calibration |
| 13 | stacking_ensemble | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | oof_stacking |
| 14 | soft_vote_blend | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | soft_probability_blend |
| 15 | leakage_honesty_check | full_fe | 59 | 1.0000 | 1.0000 | 1.0000 | none |

**Best safe model:** `hist_gradient_boosting` — ROC-AUC **1.0000**, F1 **1.0000**  
**Best optimization method (safe):** `mutual_info_selection` via `feature_selection_mi` (AUC 1.0000)

### FE stage chart
![FE ladder](benchmark_outputs/fe_ladder_safe.png)

---

## 4. Feature selection findings

Experiment `07 feature_selection_mi` keeps a top-MI subset of the full FE matrix.  
Compare its ROC-AUC against `06 fe_full_clusters_bins`:

- Full FE AUC: **1.0000** (59 feats)
- Selected AUC: **1.0000** (25 feats)
- Δ: **+0.0000** — selection helped or matched.


---

## 5. Figures
- `benchmark_outputs/fe_ladder_safe.png`
- `benchmark_outputs/safe_vs_unsafe_by_experiment.png`
- `benchmark_outputs/ladder_results.csv`
