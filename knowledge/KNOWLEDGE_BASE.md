# DCLab R&D Knowledge Base

This file is generated from the repository's experiment JSON. Run `python -m dclab_rnd sync` after every experiment cycle.

## Evidence inventory

- **569** normalized experiment records across **11** datasets and **6** suites.
- **473** completed results are eligible for deployment comparisons.
- **0** records contain full code/data/runtime provenance.

## Deployment-eligible champions

| Dataset | Model family | Experiment | ROC-AUC | F1 | Recall | Evidence |
|---|---|---|---:|---:|---:|---|
| adult | catboost | catboost | 0.9274 | 0.7046 | 0.6405 | `general_pipeline/results_external/adult__optimized__catboost.json` |
| bank_marketing | lightgbm | baseline_lightgbm | 0.8029 | 0.3333 | 0.2329 | `external_projects/bank_marketing_exp/results/ladder/02_safe_baseline_lightgbm.json` |
| breast_cancer | lightgbm | fe_ratios | 1.0000 | 0.9630 | 0.9286 | `external_projects/breast_cancer_exp/results/ladder/04_safe_fe_ratios.json` |
| credit_default | ensemble | stacking_ensemble | 0.8005 | 0.4848 | 0.3706 | `external_projects/credit_default_exp/results/ladder/13_safe_stacking_ensemble.json` |
| german_credit | catboost | catboost | 0.8112 | 0.5962 | 0.5167 | `general_pipeline/results_external/german_credit__optimized__catboost.json` |
| heart_disease | hist_gradient_boosting | hist_gradient_boosting | 0.9740 | 0.9153 | 0.9643 | `external_projects/heart_disease_exp/results/ladder/10_safe_hist_gradient_boosting.json` |
| hyperack | ensemble | softvote_etbag_lgbmwinner_xgb | 0.9455 | 0.8505 | 0.7784 | `optimized_safe_model/results/53_softvote_etbag_lgbmwinner_xgb.json` |
| mushroom | catboost | catboost | 1.0000 | 1.0000 | 1.0000 | `general_pipeline/results_external/mushroom__baseline__catboost.json` |
| online_shoppers | xgboost | tuned_xgboost | 0.7737 | 0.1212 | 0.0681 | `external_projects/online_shoppers_exp/results/ladder/08_safe_tuned_xgboost.json` |
| spambase | lightgbm | lightgbm | 0.9878 | 0.9350 | 0.9311 | `general_pipeline/results_external/spambase__baseline__lightgbm.json` |
| wine_quality | extra_trees | extra_trees | 0.9160 | 0.8833 | 0.9149 | `general_pipeline/results_external/wine_quality__baseline__extra_trees.json` |

## Cross-dataset model evidence

Each model family contributes at most one best deployment-eligible result per dataset, preventing large experiment suites from dominating the conclusion.

| Model family | Datasets | Wins | Mean rank | Mean ROC-AUC | Range |
|---|---:|---:|---:|---:|---:|
| lightgbm | 11 | 3 | 2.55 | 0.8998 | 0.7719–1.0000 |
| xgboost | 11 | 1 | 3.55 | 0.8982 | 0.7737–1.0000 |
| ensemble | 11 | 2 | 3.64 | 0.8982 | 0.7735–1.0000 |
| hist_gradient_boosting | 11 | 1 | 4.36 | 0.8975 | 0.7696–1.0000 |
| extra_trees | 11 | 1 | 4.55 | 0.8948 | 0.7532–1.0000 |
| logistic_regression | 11 | 0 | 7.64 | 0.8528 | 0.7048–0.9997 |
| catboost | 9 | 3 | 3.56 | 0.9217 | 0.7907–1.0000 |
| random_forest | 9 | 0 | 5.22 | 0.9226 | 0.7961–1.0000 |
| tabular_transformer | 1 | 0 | 8.00 | 0.9186 | 0.9186–0.9186 |
| linear_svc | 1 | 0 | 9.00 | 0.8829 | 0.8829–0.8829 |

## Leakage evidence

| Dataset | Best safe ROC-AUC | Best unsafe ROC-AUC | Apparent lift |
|---|---:|---:|---:|
| bank_marketing | 0.8029 | 0.9357 | +0.1328 |
| hyperack | 0.9455 | 0.9802 | +0.0347 |
| online_shoppers | 0.7737 | 0.9311 | +0.1575 |

## Interpretation rules

1. Deployment comparisons exclude known post-outcome feature leakage.
2. A point-estimate win is a hypothesis until repeated-CV or bootstrap uncertainty supports it.
3. Cross-dataset mean rank is stronger evidence for a default algorithm than one dataset's highest ROC-AUC.
4. Runtime, calibration, recall, and business cost remain selection constraints even when ROC-AUC is primary.

See [NEXT_EXPERIMENTS.md](NEXT_EXPERIMENTS.md) for the evidence-driven backlog and [DATA_QUALITY.md](DATA_QUALITY.md) for registry warnings.
