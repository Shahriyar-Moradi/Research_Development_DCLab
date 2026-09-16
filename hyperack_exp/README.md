# HyperAck experiments

This directory contains 15 focused experiments for maximizing held-out `hyper_ack`
classification performance plus a final leaderboard.

For a plain-language explanation of features, feature engineering, and why each
experiment chose its inputs, read [FEATURE_ENGINEERING_REPORT.md](FEATURE_ENGINEERING_REPORT.md).

For how the top models were trained and a **general playbook** for other
classification/prediction problems, read
[TOP_MODELS_TRAINING_REPORT.md](TOP_MODELS_TRAINING_REPORT.md).

## Run order

1. Run `01_baseline_current.ipynb` through `15_full_fe_best_blend.ipynb`.
2. Run `99_final_benchmark.ipynb`.

Every experiment reads `../hyper_ackt-dataset.csv`, uses the same persisted
stratified 80/20 split (`results/split_indices.npz`), and writes one record to
`results/`. Do not change the split file while comparing experiments.

## Metrics

The leaderboard ranks held-out ROC-AUC first, then F1 and accuracy. Average
precision, confusion matrix, and timings are saved for context. A high score
from final fares can be operational leakage: compare it with experiment 14,
which excludes `final_customer_fare` and `final_biker_fare`.

## Reproducibility

All random seeds use `42`. The supplied tuning notebooks use randomized search
on training folds; Optuna is included for a more extensive follow-up search.
Install dependencies from the repository root before running:

```bash
.venv/bin/pip install -r requirements.txt
```
