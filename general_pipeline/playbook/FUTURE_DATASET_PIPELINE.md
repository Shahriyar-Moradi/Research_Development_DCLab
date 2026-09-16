# Future Dataset Pipeline (HyperAck Playbook)

Reusable path to produce the same depth of analysis as HyperAck for any new tabular binary classification problem.

## Steps

1. **Register** the dataset in `general_pipeline/external_catalog.py`.
2. **Download / cache** via `download_external_datasets.py` (or drop parquet into `external_data/<key>/`).
3. **Define leakage policy** in `general_pipeline/playbook/policy.py`:
   - List post-outcome columns in `unsafe_only_features`.
   - Write a clear decision-time rationale.
4. **Run the playbook**:
   ```bash
   .venv/bin/python general_pipeline/playbook/run_playbook.py --dataset <key>
   ```
5. **Read** in `external_projects/<key>_exp/`:
   - `FEATURE_ENGINEERING_REPORT.md`
   - `OPTIMIZATION_REPORT.md`
   - `COMPLETE_MASTER_REPORT.md`
   - `notebooks/*.ipynb`
6. **Iterate** optimization: extend `ladder_specs()` or add Optuna trials following the tabular playbook rule.

## Ladder (mirrors HyperAck 01–15)

| ID | Focus |
|---|---|
| 01–02 | Baselines |
| 03–06 | FE families (logs, ratios, interactions, clusters/bins) |
| 07 | Mutual-information feature selection |
| 08–09 | RandomizedSearchCV (XGB, LGBM) |
| 10–11 | HistGB / ExtraTrees |
| 12 | Isotonic calibration |
| 13–14 | Stacking / soft-vote |
| 15 | Leakage honesty marker |

## Quadrants compared

- Safe baseline vs Safe optimized
- Unsafe baseline vs Unsafe optimized (when leakage exists)
- Safe vs Unsafe (leakage cost)
- FE stage ablation
- Optimization method ranking
