# External Projects Index

HyperAck-style folders for each public dataset — full experiment ladder, notebooks, and detailed reports.

## Per-project layout

```
external_projects/<dataset>_exp/
  README.md
  run_all.py
  data/                          → external_data/<dataset>
  notebooks/
    01_<key>_full_ladder.ipynb
    02_<key>_feature_engineering.ipynb
    03_<key>_optimization_methods.ipynb
  results/
    ladder/                      → exp 01–15 × safe/unsafe JSON
  benchmark_outputs/             → FE, safe vs unsafe, optimization plots
  FEATURE_ENGINEERING_REPORT.md
  OPTIMIZATION_REPORT.md
  COMPLETE_MASTER_REPORT.md
  MASTER_BENCHMARK_REPORT.md
```

## Experiment ladder (mirrors HyperAck 01–15)

| ID | Focus |
|---|---|
| 01–02 | Baselines (logistic, LightGBM) |
| 03–06 | FE: logs → ratios → interactions → clusters/bins |
| 07 | Mutual-information feature selection |
| 08–09 | RandomizedSearchCV (XGB, LGBM) |
| 10–11 | HistGB / ExtraTrees |
| 12 | Isotonic calibration |
| 13–14 | Stacking / soft-vote |
| 15 | Leakage honesty marker |

**Comparisons:** Safe vs Unsafe · Safe baseline vs Safe optimized · Unsafe baseline vs Unsafe optimized · FE ablation · Optimization method ranking

## Leakage policies

| Dataset | Unsafe-only columns |
|---|---|
| bank_marketing | `duration` |
| online_shoppers | `PageValues` |
| others | none (safe ≡ unsafe matrix; FE/opt ladder still runs) |

## Projects

- [`adult_exp`](adult_exp/)
- [`bank_marketing_exp`](bank_marketing_exp/)
- [`breast_cancer_exp`](breast_cancer_exp/)
- [`heart_disease_exp`](heart_disease_exp/)
- [`credit_default_exp`](credit_default_exp/)
- [`german_credit_exp`](german_credit_exp/)
- [`mushroom_exp`](mushroom_exp/)
- [`spambase_exp`](spambase_exp/)
- [`online_shoppers_exp`](online_shoppers_exp/)
- [`wine_quality_exp`](wine_quality_exp/)

## Run

```bash
# Full HyperAck playbook for all / one dataset
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset all
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset bank_marketing

# Future new dataset
# 1) register in external_catalog.py + download
# 2) define leakage in playbook/policy.py
# 3) run playbook --dataset <key>
```

See [`general_pipeline/playbook/FUTURE_DATASET_PIPELINE.md`](../general_pipeline/playbook/FUTURE_DATASET_PIPELINE.md).
