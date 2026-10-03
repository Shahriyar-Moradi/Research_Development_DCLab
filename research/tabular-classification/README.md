# Tabular classification (HyperAck and public benchmarks)

**Status:** active · the most mature track

## Research question

How do we build the most accurate *honest* classifier for tabular business data, and which steps (leakage control, feature engineering, algorithm choice, tuning, calibration) actually move the score?

## Why it matters for DCLab

This track produced most of DCLab's rules (`knowledge/model_building_rules.jsonl`), the leakage precedents the copilot cites, and the five-stage workflow the agent follows.

## Prediction contract (HyperAck)

| Item | Value |
|---|---|
| Unit | one delivery order |
| Prediction moment | when the order is created, before dispatch |
| Target | `hyper_ack` (order accepted, 0/1) |
| Forbidden | `final_customer_fare`, `final_biker_fare` (only known after the outcome) |
| Primary metric | ROC-AUC, with recall at the operating threshold |

## What is here

| Folder | What it is | Key result |
|---|---|---|
| `hyperack_exp/` | Original 15-experiment ladder, **including** the leaky final fares | ROC-AUC 0.9793, inflated, research only |
| `safe_leakage_exp/` | The same ladder without final fares | ROC-AUC 0.9450, the honest baseline |
| `optimized_safe_model/` | 53 leakage-safe optimizations: tuning, recovery runs, ensembles | ROC-AUC 0.9455 (exp 53 soft vote ET + LGBM + XGB) |
| `external_projects/` | The same ladder on 10 public UCI datasets, with notebooks and reports | Per-dataset reports in each `<dataset>_exp/` |
| `benchmark_outputs/` | 4-way benchmark plots and CSVs (safe/unsafe × baseline/optimized) | Used by `docs/reports/MASTER_4WAY_BENCHMARK_REPORT.md` |
| `notebooks/` | First exploratory HyperAck notebook and its saved models | Superseded by the suites above |

The shared pipeline code is in [`general_pipeline/`](../../general_pipeline/) at the repository root, because other tracks and `dclab_rnd` import it.

## How to run

From the repository root, with the ML environment installed (`pip install -r requirements.txt`):

```bash
# Leakage-safe ladder and the safe-vs-unsafe comparison
.venv/bin/python research/tabular-classification/safe_leakage_exp/run_all_safe.py
.venv/bin/python research/tabular-classification/safe_leakage_exp/compare_safe_vs_unsafe.py

# Tune all model families, then the visual benchmark against the unsafe ceiling
.venv/bin/python research/tabular-classification/optimized_safe_model/run_tune_all_models.py
.venv/bin/python research/tabular-classification/optimized_safe_model/benchmark_vs_unsafe.py

# Unified pipeline: one experiment, or the full 44-model 4-way benchmark
.venv/bin/python general_pipeline/run_experiments.py --mode safe --optimization optimized --model lightgbm
.venv/bin/python general_pipeline/run_experiments.py --mode all --optimization all --model all

# One public dataset end to end (ladder, notebooks, reports)
.venv/bin/python research/tabular-classification/external_projects/adult_exp/run_all.py

# After any new result: rebuild the registry and knowledge
make rd-sync
```

The HyperAck notebooks in `hyperack_exp/` run from that folder (`cd research/tabular-classification/hyperack_exp && jupyter lab`).

## Conclusions so far

- Leakage was the largest effect: +0.035 ROC-AUC on HyperAck from the final fares, and +0.13 to +0.20 on public datasets.
- 25 lean features beat 46 engineered ones (about 0.945 vs 0.937).
- Ensembles of ExtraTrees, LightGBM and XGBoost gave the best safe ranking. Isotonic calibration plus a threshold sweep gave the best recall (0.937).
- Details: [`docs/reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md`](../../docs/reports/MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md).
