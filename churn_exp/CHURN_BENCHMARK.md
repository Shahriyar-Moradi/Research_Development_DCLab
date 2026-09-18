# Telco Churn: 15-experiment benchmark

> Development evidence only. The same CV is reused across candidates; no independent test or deployment approval.

| Rank | ID | Experiment | Model | ROC-AUC | AP | Log loss | Δ AUC vs logistic |
|---:|---|---|---|---:|---:|---:|---:|
| 1 | CHURN-014 | Logistic with charge features | logistic_regression | 0.8499 | 0.6708 | 0.4108 | +0.0050 |
| 2 | CHURN-010 | Regularized histogram boosting | hist_gradient_boosting | 0.8478 | 0.6670 | 0.4133 | +0.0030 |
| 3 | CHURN-008 | Regularized Random Forest | random_forest | 0.8472 | 0.6623 | 0.4153 | +0.0024 |
| 4 | CHURN-012 | Regularized LightGBM | lightgbm | 0.8468 | 0.6666 | 0.4143 | +0.0020 |
| 5 | CHURN-015 | LightGBM with charge features | lightgbm | 0.8466 | 0.6633 | 0.4147 | +0.0018 |
| 6 | CHURN-013 | XGBoost baseline | xgboost | 0.8460 | 0.6666 | 0.4152 | +0.0011 |
| 7 | CHURN-004 | Flexible logistic C=3 | logistic_regression | 0.8449 | 0.6534 | 0.4181 | +0.0001 |
| 8 | CHURN-002 | Logistic baseline C=1 | logistic_regression | 0.8448 | 0.6535 | 0.4181 | +0.0000 |
| 9 | CHURN-003 | Regularized logistic C=0.2 | logistic_regression | 0.8445 | 0.6529 | 0.4182 | -0.0004 |
| 10 | CHURN-011 | LightGBM baseline | lightgbm | 0.8402 | 0.6578 | 0.4236 | -0.0046 |
| 11 | CHURN-006 | Regularized Extra Trees | extra_trees | 0.8397 | 0.6418 | 0.4239 | -0.0052 |
| 12 | CHURN-009 | Histogram gradient boosting | hist_gradient_boosting | 0.8382 | 0.6540 | 0.4274 | -0.0066 |
| 13 | CHURN-007 | Random Forest baseline | random_forest | 0.8364 | 0.6443 | 0.4360 | -0.0085 |
| 14 | CHURN-005 | Extra Trees baseline | extra_trees | 0.8246 | 0.6134 | 0.4421 | -0.0202 |
| 15 | CHURN-001 | Prior-only floor | dummy | 0.5000 | 0.2654 | 0.5787 | -0.3448 |

## Protocol

- `customerID` is excluded before modeling.
- Blank `TotalCharges` at zero tenure becomes zero; all other missing numeric values are imputed inside each fold.
- Categorical encoding, numeric imputation/scaling and derived features are fitted/executed inside each fold.
- Exact duplicate allowed-input rows share a fold. The source has no verified household/account-history grouping or event-time split.
- Every result includes OOF predictions, calibration, missing-input stress tests, sensitivity, data/code hashes and a replay recipe.

## Interpretation

The highest score is a candidate for independent temporal/external confirmation, not a universal best model. Prefer probability quality, robustness, operating constraints and stability—not ROC-AUC alone.
