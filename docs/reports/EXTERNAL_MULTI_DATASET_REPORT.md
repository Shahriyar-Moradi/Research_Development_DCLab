# External Multi-Dataset Classification Benchmark

Replicates the HyperAck tabular pipeline across **10 public UCI classification datasets** with the same model zoo (baseline + optimized).

## Datasets (cached under `external_data/`)

| Key | Dataset | Rows | Feats | Pos rate | Source |
|---|---|---:|---:|---:|---|
| adult | Adult (Census Income) | 48,842 | 14 | 0.239 | [UCI #2](https://archive.ics.uci.edu/dataset/2/adult) |
| bank_marketing | Bank Marketing | 45,211 | 16 | 0.117 | [UCI #222](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| breast_cancer | Breast Cancer Wisconsin | 569 | 30 | 0.373 | [UCI #17](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) |
| heart_disease | Heart Disease | 303 | 13 | 0.459 | [UCI #45](https://archive.ics.uci.edu/dataset/45/heart+disease) |
| credit_default | Credit Card Default | 30,000 | 23 | 0.221 | [UCI #350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients) |
| german_credit | German Credit | 1,000 | 20 | 0.300 | [UCI #144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) |
| mushroom | Mushroom | 8,124 | 22 | 0.482 | [UCI #73](https://archive.ics.uci.edu/dataset/73/mushroom) |
| spambase | Spambase | 4,601 | 57 | 0.394 | [UCI #94](https://archive.ics.uci.edu/dataset/94/spambase) |
| online_shoppers | Online Shoppers Intent | 12,330 | 17 | 0.155 | [UCI #468](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset) |
| wine_quality | Wine Quality (binary ≥6) | 6,497 | 11 | 0.633 | [UCI #186](https://archive.ics.uci.edu/dataset/186/wine+quality) |

Protocol: stratified 80/20 split (seed 42); datasets >20k rows subsampled to 20k before split; median imputation via sklearn pipelines.

## Experiment grid

- **Models:** Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, LightGBM, XGBoost, CatBoost
- **Optimization:** baseline + optimized (HyperAck safe-tuned configs)
- **Runs:** 10 × 2 × 7 = **140** completed (~140s wall time)

## Best ROC-AUC by dataset

| Dataset | Best model | Opt | ROC-AUC | F1 |
|---|---|---|---:|---:|
| adult | catboost | optimized | 0.927 | 0.705 |
| bank_marketing | catboost | optimized | 0.939 | 0.535 |
| breast_cancer | extra_trees | optimized | 0.999 | 0.950 |
| credit_default | lightgbm | optimized | 0.797 | 0.483 |
| german_credit | catboost | optimized | 0.811 | 0.596 |
| heart_disease | random_forest | optimized | 0.962 | 0.897 |
| mushroom | random_forest | baseline | 1.000 | 1.000 |
| online_shoppers | lightgbm | optimized | 0.932 | 0.667 |
| spambase | lightgbm | baseline | 0.988 | 0.935 |
| wine_quality | extra_trees | baseline | 0.916 | 0.883 |

## Per-dataset HyperAck-style projects + full playbook

Each dataset has `external_projects/<key>_exp/` with:

- **15-step ladder** (baseline → FE → selection → RandomizedSearch → calibration → stacking/blend)
- **Safe vs unsafe** (leakage drop where defined: Bank `duration`, Shoppers `PageValues`)
- **Notebooks** (`notebooks/01–03`)
- **Separate reports:** `FEATURE_ENGINEERING_REPORT.md`, `OPTIMIZATION_REPORT.md`, `COMPLETE_MASTER_REPORT.md`

Reusable future-dataset pipeline: `general_pipeline/playbook/FUTURE_DATASET_PIPELINE.md`

```bash
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset all
.venv/bin/python general_pipeline/playbook/run_playbook.py --dataset bank_marketing
```

## Artifacts

- Per-project: `external_projects/*_exp/`
- Cross-dataset tables: `benchmark_outputs/external/`
- Playbook code: `general_pipeline/playbook/`


## Notes

- These public tasks do not have HyperAck-style safe/unsafe fare leakage quadrants; experiments use `mode=external` with safe-tuned hyperparameters for the optimized arm.
- Bank Marketing includes call `duration`, which is often treated as post-outcome leakage in production settings (similar spirit to HyperAck final fares).
