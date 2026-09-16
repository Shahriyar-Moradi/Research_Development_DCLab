# Default of Credit Card Clients — Optimization Methods Report

**Question:** Which optimization methods lift ROC-AUC the most on this dataset?

---

## 1. Methods tested

| Method | Experiment | Description |
|---|---|---|
| none (baseline) | 02 baseline_lightgbm | Default LightGBM on raw features |
| FE only | 03–06 | Feature engineering without hyperparameter search |
| mutual_info_selection | 07 | Compact MI feature subset |
| RandomizedSearchCV | 08 tuned_xgboost, 09 tuned_lightgbm | 3-fold CV maximizing ROC-AUC |
| hand_tuned | 11 extra_trees_strong | Playbook ExtraTrees region |
| isotonic_calibration | 12 calibrated_lgbm | Probability calibration |
| oof_stacking | 13 stacking_ensemble | ET+LGBM+XGB → logistic meta |
| soft_probability_blend | 14 soft_vote_blend | Weighted probability average |

---

## 2. Safe-mode ranking (best → worst)

| Rank | Experiment | Method | ROC-AUC | F1 | Notes |
|---:|---|---|---:|---:|---|
| 1 | stacking_ensemble | oof_stacking | 0.8005 | 0.4848 | ET+LGBM+XGB stacked with logistic meta-learner (HyperAck exp 13). |
| 2 | soft_vote_blend | soft_probability_blend | 0.7996 | 0.4863 | Soft-vote blend ET+LGBM+XGB (HyperAck exp 15 / optimized safe champion style). |
| 3 | leakage_honesty_check | none | 0.7996 | 0.4866 | Honesty marker: LightGBM on full FE under this mode. Compare safe vs unsafe of t |
| 4 | tuned_xgboost | RandomizedSearchCV_roc_auc | 0.7988 | 0.4825 | RandomizedSearchCV on XGBoost over full FE (HyperAck exp 08). |
| 5 | tuned_lightgbm | RandomizedSearchCV_roc_auc | 0.7985 | 0.4802 | RandomizedSearchCV on LightGBM over full FE (HyperAck exp 09). |
| 6 | hist_gradient_boosting | none | 0.7973 | 0.4874 | Sklearn HistGB on full FE (HyperAck exp 10). |
| 7 | extra_trees_strong | hand_tuned | 0.7957 | 0.4789 | Strong ExtraTrees (n=600, depth=14) on full FE. |
| 8 | fe_ratios | none | 0.7948 | 0.4900 | Add pairwise intensity ratios among high-variance features. |
| 9 | fe_full_clusters_bins | none | 0.7930 | 0.4775 | Full FE: logs+ratios+interactions+KMeans clusters+quantile bins (train-fit). |
| 10 | calibrated_lgbm_threshold | isotonic_calibration | 0.7929 | 0.4762 | Isotonic-calibrated LightGBM; threshold kept at 0.5 for ranking metrics. |
| 11 | fe_interactions | none | 0.7899 | 0.4870 | Multiplicative interactions among top-MI features. |
| 12 | feature_selection_mi | mutual_info_selection | 0.7880 | 0.4779 | Keep top-MI subset of full FE matrix (HyperAck exp 07). |
| 13 | baseline_lightgbm | none | 0.7867 | 0.4880 | Strong GBDT baseline on raw features (HyperAck exp 02). |
| 14 | fe_log_transforms | none | 0.7867 | 0.4880 | Add log1p transforms for skewed non-negative columns. |
| 15 | baseline_logistic | none | 0.7209 | 0.3755 | Floor: logistic regression on raw features. |

**Winner:** `stacking_ensemble` with method `oof_stacking`  
**Best pure search method:** `stacking_ensemble` (`oof_stacking`)

![Optimization methods](benchmark_outputs/optimization_methods_safe.png)

---

## 3. Safe vs Optimized Safe

| Arm | Experiment | ROC-AUC | Recall | F1 |
|---|---|---:|---:|---:|
| Safe baseline | baseline_lightgbm | 0.7867 | — | — |
| Safe optimized (best) | stacking_ensemble | 0.8005 | 0.3706 | 0.4848 |

Lift (best − baseline LGBM): see ranking table and `quadrant_safe_unsafe_opt.png`.

---

## 4. Unsafe vs Optimized Unsafe / Safe vs Unsafe

No post-outcome leakage columns for this dataset — safe and unsafe feature matrices are **identical**. Differences across modes (if any) are numerical noise only. Focus optimization comparisons on the safe ranking above.


---

## 5. Recommendation for production

1. Prefer **safe** features only when leakage columns exist.
2. Start from **full FE** then test MI selection.
3. Apply **RandomizedSearchCV** on LightGBM/XGBoost before stacking.
4. Use **soft-vote / stacking** only if they beat the best single tuned model on the locked test set.
