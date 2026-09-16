# Statlog German Credit — Complete Master Report

HyperAck-style end-to-end study for **Statlog German Credit**.

## Executive summary

| Item | Value |
|---|---|
| Best safe ROC-AUC | **0.8082** (`tuned_xgboost`) |
| Best unsafe ROC-AUC | **0.8082** (`tuned_xgboost`) |
| Best optimization method | `RandomizedSearchCV_roc_auc` |
| Leakage present | False |
| Leakage columns | `[]` |

```
Unsafe ceiling (if leakage)  ──▶  0.8082
        │ drop leakage
        ▼
Safe baseline / FE ladder    ──▶  see FEATURE_ENGINEERING_REPORT.md
        │ RandomizedSearch / ensembles
        ▼
Safe optimized champion      ──▶  0.8082  (tuned_xgboost)
```

## Detailed documents

1. [FEATURE_ENGINEERING_REPORT.md](FEATURE_ENGINEERING_REPORT.md) — FE stages, selection, safe results  
2. [OPTIMIZATION_REPORT.md](OPTIMIZATION_REPORT.md) — tuning/ensembling methods & which won  
3. [MASTER_BENCHMARK_REPORT.md](MASTER_BENCHMARK_REPORT.md) — model-zoo benchmark  
4. Notebooks in `notebooks/` — step-by-step reproducible narrative  

## Key figures

![Quadrants](benchmark_outputs/quadrant_safe_unsafe_opt.png)

![FE ladder](benchmark_outputs/fe_ladder_safe.png)

![Optimization](benchmark_outputs/optimization_methods_safe.png)

![Safe vs Unsafe](benchmark_outputs/safe_vs_unsafe_by_experiment.png)

## Reproduce

```bash
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset german_credit
.venv/bin/python external_projects/german_credit_exp/run_all.py
```

## Future datasets

Use the same playbook pipeline:

```bash
# 1) add dataset to external_catalog + download
# 2) define leakage in playbook/policy.py
# 3) run:
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset <new_key>
```
