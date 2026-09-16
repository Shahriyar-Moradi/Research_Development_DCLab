# Bank Marketing — Feature Engineering & Selection Report

**Project:** `external_projects/bank_marketing_exp/`  
**Source:** [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**Rows / features (raw):** 45211 / 16  
**Positive rate:** 0.117  
**Protocol:** stratified 80/20, seed 42; all stateful FE fit on **train only**

---

## 1. Problem & decision-time policy

Predict term-deposit subscription from phone marketing campaign data.

### Leakage / safe vs unsafe
- **Has classic leakage columns:** True
- **Unsafe-only features:** `['duration']`
- **Rationale:** `duration` is call length known only after the call ends — classic target leakage for predicting subscription (y). Safe models must drop it (HyperAck analog of final fares).

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
| 01 | baseline_logistic | raw | 15 | 0.7260 | 0.1346 | 0.2234 | none |
| 02 | baseline_lightgbm | raw | 15 | 0.8029 | 0.2329 | 0.3333 | none |
| 03 | fe_log_transforms | logs | 23 | 0.8029 | 0.2329 | 0.3333 | none |
| 04 | fe_ratios | ratios | 38 | 0.7872 | 0.2543 | 0.3536 | none |
| 05 | fe_interactions | interactions | 46 | 0.7892 | 0.2479 | 0.3558 | none |
| 06 | fe_full_clusters_bins | full_fe | 51 | 0.7877 | 0.2436 | 0.3502 | none |
| 07 | feature_selection_mi | selected | 25 | 0.7837 | 0.2244 | 0.3216 | mutual_info_selection |
| 08 | tuned_xgboost | full_fe | 51 | 0.7973 | 0.2030 | 0.3105 | RandomizedSearchCV_roc_auc |
| 09 | tuned_lightgbm | full_fe | 51 | 0.7976 | 0.2158 | 0.3248 | RandomizedSearchCV_roc_auc |
| 10 | hist_gradient_boosting | full_fe | 51 | 0.7990 | 0.2137 | 0.3180 | none |
| 11 | extra_trees_strong | full_fe | 51 | 0.7924 | 0.1859 | 0.2876 | hand_tuned |
| 12 | calibrated_lgbm_threshold | full_fe | 51 | 0.7930 | 0.1987 | 0.3054 | isotonic_calibration |
| 13 | stacking_ensemble | full_fe | 51 | 0.7998 | 0.2308 | 0.3380 | oof_stacking |
| 14 | soft_vote_blend | full_fe | 51 | 0.8002 | 0.2073 | 0.3139 | soft_probability_blend |
| 15 | leakage_honesty_check | full_fe | 51 | 0.7919 | 0.2222 | 0.3312 | none |

**Best safe model:** `baseline_lightgbm` — ROC-AUC **0.8029**, F1 **0.3333**  
**Best optimization method (safe):** `soft_probability_blend` via `soft_vote_blend` (AUC 0.8002)

### FE stage chart
![FE ladder](benchmark_outputs/fe_ladder_safe.png)

---

## 4. Feature selection findings

Experiment `07 feature_selection_mi` keeps a top-MI subset of the full FE matrix.  
Compare its ROC-AUC against `06 fe_full_clusters_bins`:

- Full FE AUC: **0.7877** (51 feats)
- Selected AUC: **0.7837** (25 feats)
- Δ: **-0.0041** — full FE retained more signal (dilution not an issue / selection too aggressive).


---

## 5. Figures
- `benchmark_outputs/fe_ladder_safe.png`
- `benchmark_outputs/safe_vs_unsafe_by_experiment.png`
- `benchmark_outputs/ladder_results.csv`
