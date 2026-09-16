# Optimized safe models

Same HyperAck problem, **still leakage-safe** (no final fares), with stronger improvements:

1. Richer safe feature engineering (optional; recovery uses proven 25-feature matrix)
2. Class-weight / scale_pos_weight for recall (01–15)
3. Heavier hyperparameter search + local search around prior safe winner
4. Seed-bagging, soft-voting, OOF stacks, CatBoost, Optuna
5. Threshold tuning for F1+recall

**Result:** best ROC-AUC **0.9455** (exp 53 soft-vote) — beats prior safe `09` (0.9450).
Stacking of tuned ET+LGBM+XGB also reaches **0.9450**. Tuned ExtraTrees alone is **0.9444**.

## Run

```bash
.venv/bin/python optimized_safe_model/run_all_optimized.py
.venv/bin/python optimized_safe_model/run_recovery.py
.venv/bin/python optimized_safe_model/run_recovery2.py
.venv/bin/python optimized_safe_model/run_recovery3.py
.venv/bin/python optimized_safe_model/run_recovery4.py

# RandomizedSearchCV on every model family (LR/RF/ET/HGB/SVC/LGBM/XGB/CatBoost + ensembles)
.venv/bin/python optimized_safe_model/run_tune_all_models.py

.venv/bin/python optimized_safe_model/compare_optimized.py

# Visual benchmark vs unsafe (same exp_id + champions)
.venv/bin/python optimized_safe_model/benchmark_vs_unsafe.py
```

Uses the same locked split as `hyperack_exp` / `safe_leakage_exp`.

## Visual outputs
- `results/optimized_vs_unsafe_side_by_side.png` — matched models ROC + recall
- `results/optimized_vs_unsafe_roc_gap.png` — leakage cost per model
- `results/optimized_vs_unsafe_champions.png` — best unsafe / prev safe / optimized
- `results/optimized_top_vs_unsafe_ceiling.png` — top optimized vs ceilings
- Canvas: `optimized-vs-unsafe-benchmark`
