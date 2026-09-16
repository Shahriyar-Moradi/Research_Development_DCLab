# Safe leakage experiments

Same model strategies as `hyperack_exp/`, but **all training drops unsafe features**:

- dropped: `final_customer_fare`, `final_biker_fare`
- kept: first fare, distance, time, geo, category, products, and safe engineered features

Uses the **same train/test split** as `hyperack_exp` so scores are comparable.

## Run

```bash
.venv/bin/python safe_leakage_exp/run_all_safe.py
.venv/bin/python safe_leakage_exp/compare_safe_vs_unsafe.py
```

## Outputs

- `results/*.json` — safe-run metrics
- `results/safe_vs_unsafe_comparison.csv`
- `SAFE_VS_UNSAFE_REPORT.md`
