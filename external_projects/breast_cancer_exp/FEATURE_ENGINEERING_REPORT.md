# Breast Cancer Wisconsin (Diagnostic) — Feature Engineering & Selection Report

**Project:** `external_projects/breast_cancer_exp/`  
**Source:** [https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)  
**Rows / features (raw):** 569 / 30  
**Positive rate:** 0.373  
**Protocol:** stratified 80/20, seed 42; all stateful FE fit on **train only**

---

## 1. Problem & decision-time policy

Binary diagnosis (malignant/benign) from cell nuclei measurements.

### Leakage / safe vs unsafe
- **Has classic leakage columns:** False
- **Unsafe-only features:** `[]`
- **Rationale:** All morphometry features are available at diagnosis time; no temporal leakage.

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
| 01 | baseline_logistic | raw | 30 | 0.9960 | 0.9286 | 0.9512 | none |
| 02 | baseline_lightgbm | raw | 30 | 0.9967 | 0.9048 | 0.9500 | none |
| 03 | fe_log_transforms | logs | 54 | 0.9967 | 0.9048 | 0.9500 | none |
| 04 | fe_ratios | ratios | 69 | 1.0000 | 0.9286 | 0.9630 | none |
| 05 | fe_interactions | interactions | 77 | 0.9987 | 0.9286 | 0.9512 | none |
| 06 | fe_full | full_fe | 82 | 0.9987 | 0.9286 | 0.9512 | none |
| 06 | fe_full_clusters_bins | full_fe | 82 | 0.9987 | 0.9286 | 0.9512 | none |
| 07 | feature_selection_mi | selected | 25 | 0.9937 | 0.9286 | 0.9512 | mutual_info_selection |
| 08 | tuned_xgboost | full_fe | 82 | 0.9997 | 0.9286 | 0.9630 | RandomizedSearchCV_roc_auc |
| 09 | tuned_lightgbm | full_fe | 82 | 0.9990 | 0.9286 | 0.9630 | RandomizedSearchCV_roc_auc |
| 10 | hist_gradient_boosting | full_fe | 82 | 0.9970 | 0.9286 | 0.9512 | none |
| 11 | extra_trees_strong | full_fe | 82 | 0.9997 | 0.9524 | 0.9756 | hand_tuned |
| 12 | calibrated_lgbm_threshold | full_fe | 82 | 0.9983 | 0.9286 | 0.9512 | isotonic_calibration |
| 13 | stacking_ensemble | full_fe | 82 | 0.9993 | 0.9524 | 0.9756 | oof_stacking |
| 14 | soft_vote_blend | full_fe | 82 | 0.9993 | 0.9524 | 0.9756 | soft_probability_blend |
| 15 | leakage_honesty_check | full_fe | 82 | 0.9980 | 0.9286 | 0.9512 | none |

**Best safe model:** `fe_ratios` — ROC-AUC **1.0000**, F1 **0.9630**  
**Best optimization method (safe):** `RandomizedSearchCV_roc_auc` via `tuned_xgboost` (AUC 0.9997)

### FE stage chart
![FE ladder](benchmark_outputs/fe_ladder_safe.png)

---

## 4. Feature selection findings

Experiment `07 feature_selection_mi` keeps a top-MI subset of the full FE matrix.  
Compare its ROC-AUC against `06 fe_full_clusters_bins`:

- Full FE AUC: **0.9987** (82 feats)
- Selected AUC: **0.9937** (25 feats)
- Δ: **-0.0050** — full FE retained more signal (dilution not an issue / selection too aggressive).


---

## 5. Figures
- `benchmark_outputs/fe_ladder_safe.png`
- `benchmark_outputs/safe_vs_unsafe_by_experiment.png`
- `benchmark_outputs/ladder_results.csv`
