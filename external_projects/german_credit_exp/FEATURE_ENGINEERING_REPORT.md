# Statlog German Credit — Feature Engineering & Selection Report

**Project:** `external_projects/german_credit_exp/`  
**Source:** [https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)  
**Rows / features (raw):** 1000 / 20  
**Positive rate:** 0.300  
**Protocol:** stratified 80/20, seed 42; all stateful FE fit on **train only**

---

## 1. Problem & decision-time policy

Classify credit applicants as good/bad risk.

### Leakage / safe vs unsafe
- **Has classic leakage columns:** False
- **Unsafe-only features:** `[]`
- **Rationale:** Applicant attributes are known at underwriting time; no leakage identified.

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
| 01 | baseline_logistic | raw | 20 | 0.7748 | 0.4000 | 0.4848 | none |
| 02 | baseline_lightgbm | raw | 20 | 0.7827 | 0.5333 | 0.5714 | none |
| 03 | fe_log_transforms | logs | 33 | 0.7827 | 0.5333 | 0.5714 | none |
| 04 | fe_ratios | ratios | 48 | 0.8051 | 0.5667 | 0.6296 | none |
| 05 | fe_interactions | interactions | 56 | 0.7958 | 0.5833 | 0.6542 | none |
| 06 | fe_full_clusters_bins | full_fe | 61 | 0.7989 | 0.6000 | 0.6372 | none |
| 07 | feature_selection_mi | selected | 25 | 0.7662 | 0.4667 | 0.5138 | mutual_info_selection |
| 08 | tuned_xgboost | full_fe | 61 | 0.8082 | 0.5000 | 0.5882 | RandomizedSearchCV_roc_auc |
| 09 | tuned_lightgbm | full_fe | 61 | 0.7905 | 0.4500 | 0.5400 | RandomizedSearchCV_roc_auc |
| 10 | hist_gradient_boosting | full_fe | 61 | 0.7926 | 0.5167 | 0.5688 | none |
| 11 | extra_trees_strong | full_fe | 61 | 0.7985 | 0.4167 | 0.5000 | hand_tuned |
| 12 | calibrated_lgbm_threshold | full_fe | 61 | 0.7840 | 0.4667 | 0.5714 | isotonic_calibration |
| 13 | stacking_ensemble | full_fe | 61 | 0.8062 | 0.5333 | 0.6038 | oof_stacking |
| 14 | soft_vote_blend | full_fe | 61 | 0.8050 | 0.5333 | 0.6214 | soft_probability_blend |
| 15 | leakage_honesty_check | full_fe | 61 | 0.7973 | 0.5833 | 0.6306 | none |

**Best safe model:** `tuned_xgboost` — ROC-AUC **0.8082**, F1 **0.5882**  
**Best optimization method (safe):** `RandomizedSearchCV_roc_auc` via `tuned_xgboost` (AUC 0.8082)

### FE stage chart
![FE ladder](benchmark_outputs/fe_ladder_safe.png)

---

## 4. Feature selection findings

Experiment `07 feature_selection_mi` keeps a top-MI subset of the full FE matrix.  
Compare its ROC-AUC against `06 fe_full_clusters_bins`:

- Full FE AUC: **0.7989** (61 feats)
- Selected AUC: **0.7662** (25 feats)
- Δ: **-0.0327** — full FE retained more signal (dilution not an issue / selection too aggressive).


---

## 5. Figures
- `benchmark_outputs/fe_ladder_safe.png`
- `benchmark_outputs/safe_vs_unsafe_by_experiment.png`
- `benchmark_outputs/ladder_results.csv`
