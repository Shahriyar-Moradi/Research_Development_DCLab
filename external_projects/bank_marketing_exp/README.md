# Bank Marketing experiments

HyperAck-style tabular classification project for **Bank Marketing**.

- Source: [https://archive.ics.uci.edu/dataset/222/bank+marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Cached data: `data/` → `external_data/bank_marketing/`
- Locked split: stratified 80/20, seed `42`
- Supports **safe** (drop leakage cols) vs **unsafe** (keep all) plus baseline/optimized.

## Run

```bash
.venv/bin/python external_projects/bank_marketing_exp/run_all.py
# or from toolkit:
.venv/bin/python general_pipeline/build_external_projects.py --dataset bank_marketing --run
```

## Layout

| Path | Purpose |
|---|---|
| `data/` | Features + target parquet + meta |
| `results/` | Per-experiment JSON metrics |
| `benchmark_outputs/` | CSVs + diagnostic PNGs |
| `MASTER_BENCHMARK_REPORT.md` | Project master report |

## Models

logistic_regression, linear_svc, random_forest, extra_trees, hist_gradient_boosting, lightgbm, xgboost, catboost, stacking_ensemble, soft_vote_blend


## HyperAck-style playbook artifacts

| Document | Content |
|---|---|
| `COMPLETE_MASTER_REPORT.md` | Executive summary + links |
| `FEATURE_ENGINEERING_REPORT.md` | FE stages & selection |
| `OPTIMIZATION_REPORT.md` | Tuning / ensembles ranking |
| `notebooks/` | Step-by-step Jupyter notebooks |
| `results/ladder/` | Exp 01–15 JSON (safe & unsafe) |

```bash
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset bank_marketing
```
