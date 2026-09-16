# Mushroom — Master Benchmark Report

HyperAck-style project folder: `external_projects/mushroom_exp/`

- **Source:** [https://archive.ics.uci.edu/dataset/73/mushroom](https://archive.ics.uci.edu/dataset/73/mushroom)
- **Description:** Classify mushrooms as edible vs poisonous from categorical traits.
- **Split:** stratified 80/20, seed 42
- **Champion (safe optimized):** `catboost` — ROC-AUC **1.0000**, F1 **1.0000**
- **Overall best:** `catboost` (safe/baseline) — ROC-AUC **1.0000**

## Full results

| Mode | Opt | Model | ROC-AUC | Acc | Prec | Rec | F1 |
|---|---|---|---:|---:|---:|---:|---:|
| safe | baseline | catboost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | extra_trees | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | hist_gradient_boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | lightgbm | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | linear_svc | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | soft_vote_blend | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | stacking_ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | xgboost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | baseline | logistic_regression | 0.9996 | 0.9877 | 0.9935 | 0.9808 | 0.9871 |
| safe | optimized | catboost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | extra_trees | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | hist_gradient_boosting | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | random_forest | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | soft_vote_blend | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | stacking_ensemble | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | xgboost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | lightgbm | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| safe | optimized | linear_svc | 1.0000 | 0.9994 | 1.0000 | 0.9987 | 0.9994 |
| safe | optimized | logistic_regression | 0.9997 | 0.9883 | 0.9948 | 0.9808 | 0.9878 |

## Figures

![ROC baseline vs optimized](benchmark_outputs/roc_baseline_vs_optimized.png)

![Safe optimized 4 metrics](benchmark_outputs/safe_optimized_4metrics_profile.png)

![Metrics heatmap](benchmark_outputs/metrics_heatmap.png)

![Pareto ROC vs Recall](benchmark_outputs/pareto_roc_vs_recall.png)

![Optimization lift](benchmark_outputs/optimization_lift.png)

## Reproduce

```bash
.venv/bin/python external_projects/mushroom_exp/run_all.py
```
