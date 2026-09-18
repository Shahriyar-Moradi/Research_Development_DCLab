# Research & Development — DCLab

HyperAck order-acceptance tabular classification R&D: feature engineering, leakage-safe modeling, multi-algorithm optimization, and empirical playbooks for production-honest classifiers.

**Dataset:** 11,107 orders · locked stratified 80/20 split (seed 42) · primary metric **ROC-AUC**

---

## Automated R&D control plane

All historical result JSON is normalized into one evidence registry. The automation excludes known leakage from deployment leaderboards, ranks algorithm families across datasets, detects malformed/duplicate results and benchmark regressions, and generates the next experiment backlog from evidence gaps.

```bash
# Rebuild registry + knowledge base after any experiment
make rd-sync

# Tests, result validation, regression gates, and stale-artifact check
make rd-check

# Fast view of deployment-eligible champions
make rd-status

# Run a fast, safe HyperAck experiment and refresh all evidence
make rd-smoke

# Or target an existing runner through one controlled lifecycle command
.venv/bin/python -m dclab_rnd cycle external --dataset adult --model lightgbm --optimization optimized
```

Generated research memory:

| Artifact | Purpose |
|---|---|
| [`knowledge/KNOWLEDGE_BASE.md`](knowledge/KNOWLEDGE_BASE.md) | Champions, cross-dataset evidence, leakage gaps |
| [`knowledge/NEXT_EXPERIMENTS.md`](knowledge/NEXT_EXPERIMENTS.md) | Prioritized, evidence-driven experiment backlog |
| [`knowledge/experiment_registry.csv`](knowledge/experiment_registry.csv) | Canonical machine-readable ledger of every run |
| [`knowledge/DATA_QUALITY.md`](knowledge/DATA_QUALITY.md) | Duplicate, schema, metric, and regression findings |

New runs created by the shared runners now include `schema_version` and reproducibility provenance: git commit/dirty state, Python and ML package versions, random seed, and SHA-256 hashes of source data. CI verifies the automation layer and refuses stale knowledge artifacts.

The GitHub workflow also runs a daily evidence-health check at 01:15 UTC. Full training stays explicit through `dclab_rnd cycle`, preventing an accidental all-model/all-dataset compute run.

---

## Headline results (same test split)

| Suite | Best ROC-AUC | Best recall (same or related) | Notes |
|---|---:|---:|---|
| Unsafe (`hyperack_exp/`) | **0.9793** | 0.9114 | Inflated by final-fare **leakage** — not deployable |
| Safe baseline (`safe_leakage_exp/`) | **0.9450** | 0.7707 | Honest ceiling after dropping final fares |
| Optimized safe (`optimized_safe_model/`) | **0.9455** | 0.9374 (calibrated) | Soft-vote champion; stacking @ 0.9450 / recall 0.8006 |

```
Unsafe (leaked final fares)  ──▶  ROC 0.9793   (research only)
Drop leakage
Safe baseline                ──▶  ROC 0.9450
Systematic optimization
Optimized safe champion      ──▶  ROC 0.9455   (exp 53 soft-vote ET+LGBM+XGB)
```

---

## Master report & playbook (start here)

### Full written reports
Complete empirical evaluations across all experiments:

| Report | Markdown | PDF | Focus |
|---|---|---|---|
| **Master 4-Way Quadrant Benchmark** | [`MASTER_4WAY_BENCHMARK_REPORT.md`](MASTER_4WAY_BENCHMARK_REPORT.md) | [`MASTER_4WAY_BENCHMARK_REPORT.pdf`](MASTER_4WAY_BENCHMARK_REPORT.pdf) | 4-quadrant benchmark, Tabular Transformer (FT-Transformer), General Pipeline |
| **Comprehensive R&D Lifecycle** | [`MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md`](MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md) | [`MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf`](MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf) | Leakage forensics, 25 vs 46 FE dilution, 83-experiment optimization ladder |

### General Tabular Pipeline (`general_pipeline/`)
A unified, modular pipeline supporting 11 model architectures across Safe vs Unsafe regimes, with baseline and optimized hyperparameter factories:

```bash
# Run any single experiment
.venv/bin/python general_pipeline/run_experiments.py --mode safe --optimization optimized --model tabular_transformer

# Run full 44-model benchmark & regenerate all 4-way plots
.venv/bin/python general_pipeline/run_experiments.py --mode all --optimization all --model all
```

### Cursor / agent playbook rule
Persistent SOP for tabular classification in this repo (decision-time gate, FE ladder, algorithm search regions, optimization hierarchy, threshold calibration):

- [`.cursor/rules/tabular-classification-playbook.mdc`](.cursor/rules/tabular-classification-playbook.mdc)

**Always apply** for tabular classification / prediction work. Summary of the ladder:

1. Baseline GBDT + first-order FE  
2. Domain FE (time / pricing / geo) — keep lean; avoid feature dilution  
3. `RandomizedSearchCV` (40–60 trials, train CV only)  
4. Seed bagging (5–10 seeds)  
5. Soft-vote / stack **ExtraTrees + LightGBM + XGBoost**  
6. Post-hoc isotonic calibration + threshold sweep for recall  

**Non-negotiable:** never deploy models that use post-outcome features (`final_customer_fare`, `final_biker_fare`, etc.).

---

## Experiment suites

| Folder | What it is | Key outputs |
|---|---|---|
| [`hyperack_exp/`](hyperack_exp/) | Original 15-exp ladder (includes unsafe final fares) | `results/`, [`FEATURE_ENGINEERING_REPORT.md`](hyperack_exp/FEATURE_ENGINEERING_REPORT.md), [`TOP_MODELS_TRAINING_REPORT.md`](hyperack_exp/TOP_MODELS_TRAINING_REPORT.md) |
| [`safe_leakage_exp/`](safe_leakage_exp/) | Same strategies **without** final fares | [`SAFE_VS_UNSAFE_REPORT.md`](safe_leakage_exp/SAFE_VS_UNSAFE_REPORT.md), `results/safe_vs_unsafe_*.png` |
| [`optimized_safe_model/`](optimized_safe_model/) | 53 safe optimizations: tune-all, recovery, ensembles | Champion **0.9455**, [`OPTIMIZED_VS_UNSAFE_BENCHMARK.md`](optimized_safe_model/OPTIMIZED_VS_UNSAFE_BENCHMARK.md), visual PNGs under `results/` |

### Quick run (optimized safe + visual vs unsafe)

```bash
.venv/bin/pip install -r requirements.txt

# Full tune-all (RandomizedSearchCV across LR/RF/ET/HGB/SVC/LGBM/XGB/CatBoost + ensembles)
.venv/bin/python optimized_safe_model/run_tune_all_models.py

# Visual benchmark: optimized safe vs unsafe (same exp_id + champions)
.venv/bin/python optimized_safe_model/benchmark_vs_unsafe.py
```

Safe vs unsafe baseline ladder:

```bash
.venv/bin/python safe_leakage_exp/run_all_safe.py
.venv/bin/python safe_leakage_exp/compare_safe_vs_unsafe.py
```

---

## Visual benchmarks

| Plot | Location |
|---|---|
| Safe vs unsafe (baseline suite) | `safe_leakage_exp/results/safe_vs_unsafe_side_by_side.png` |
| Optimized safe vs unsafe (matched models) | `optimized_safe_model/results/optimized_vs_unsafe_side_by_side.png` |
| Leakage ROC gap | `optimized_safe_model/results/optimized_vs_unsafe_roc_gap.png` |
| Champions (unsafe / prev safe / optimized) | `optimized_safe_model/results/optimized_vs_unsafe_champions.png` |
| Top optimized vs ceilings | `optimized_safe_model/results/optimized_top_vs_unsafe_ceiling.png` |

Interactive canvases (Cursor): `optimized-vs-unsafe-benchmark`, `safe-vs-unsafe-benchmark`, `optimized-safe-benchmark`.

---

## Production pointers (from the report)

| Goal | Model | ROC-AUC | Recall |
|---|---|---:|---:|
| Peak ranking (safe) | Exp **53** soft-vote ET bag + LGBM + XGB | **0.9455** | 0.7784 |
| Balanced stack | Exp **51** stacking ET+LGBM+XGB | 0.9450 | **0.8006** |
| Low-latency single model | Exp **21** tuned LightGBM (25 safe features) | 0.9450 | 0.7707 |
| Max recall | Exp **12** isotonic-calibrated LGBM + threshold | 0.9375 | **0.9374** |

Winning safe matrix: **25 features** (raw + cyclical time + fare/distance ratios + haversine/bearing). Expanding to 46 interactive features diluted GBDTs (~0.945 → ~0.937).

---

## Repo layout

```
R&D/
├── README.md                                          ← this file
├── MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.md   ← full report
├── MASTER_CLASSIFICATION_AND_OPTIMIZATION_REPORT.pdf
├── generate_pdf.py
├── .cursor/rules/tabular-classification-playbook.mdc  ← FE / train / tune SOP
├── hyperack_exp/                                      ← unsafe + FE ladder
├── safe_leakage_exp/                                  ← no final fares
└── optimized_safe_model/                              ← tune-all + champion ensembles
```

---

## License / remote

Remote: `git@github.com:Shahriyar-Moradi/Research_Development_DCLab.git`
