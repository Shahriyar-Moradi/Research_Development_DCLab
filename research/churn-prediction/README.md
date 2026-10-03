# Churn prediction (Telco customer churn)

**Status:** active

## Research question

For a subscription business, which modeling choices give a reliable churn-risk ranking, and do deep tabular models beat simple ones on this kind of data?

## Prediction contract

| Item | Value |
|---|---|
| Unit | one customer at the snapshot date |
| Target | `Churn` (Yes/No) |
| Forbidden | `customerID` (identifier; target-encoding it inflated ROC-AUC from 0.82 to 1.00 in PIT-004) |
| Open issue | the real prediction timestamp and outcome window are not recorded in the public data |
| Primary metric | ROC-AUC and average precision |

## What is here

| Folder | What it is | Key result |
|---|---|---|
| `churn_exp/` | Fixed 15-experiment development campaign (2×3-fold, duplicate-grouped CV) | Logistic regression with charge features led: ROC-AUC 0.850, AP 0.671 ([benchmark](churn_exp/CHURN_BENCHMARK.md)) |
| `tabular_transformers/` | TabTransformer, FT-Transformer and SAINT vs trees and logistic regression | Logistic regression 0.845 ≥ FT-Transformer 0.842 ≥ XGBoost 0.842 ≥ SAINT 0.842 ≥ TabTransformer 0.833 |
| `logistic_regression/` | First baselines and a beginner pandas/NumPy notebook | Teaching material |

## How to run

```bash
# Fixed 15-experiment campaign (writes churn_exp/results and CHURN_BENCHMARK.md)
make churn-run
make churn-status

# Deep tabular models (extra dependencies)
pip install -r research/churn-prediction/tabular_transformers/requirements-tabular-transformers.txt
.venv/bin/python research/churn-prediction/tabular_transformers/tabular_transformer_churn.py

# Baseline script and its tests
.venv/bin/python research/churn-prediction/logistic_regression/main.py
.venv/bin/python -m unittest discover -s research/churn-prediction/logistic_regression -p 'test_model*.py'
```

## Conclusions so far

- A well-specified linear model matched or beat every boosted and transformer model on this dataset. Deep tabular models did not earn their cost here.
- These are development-CV numbers. Independent confirmation is still pending.
