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

### Evidence-first 50-experiment campaign

The second-generation campaign teaches people and LLM agents the **workflow of model building**, rather than merely collecting more leaderboard rows. It defines exactly 50 experiments: 10 real public UCI datasets × 5 scientific stages.

1. data understanding and production-risk profiling;
2. declared plus heuristic leakage review;
3. train-fitted feature-engineering ablation;
4. stability-aware algorithm-family screening; and
5. conservative optimization, calibration/reliability evidence, and one final holdout evaluation.

```bash
# Materialize the exact 50-task plan and agent-memory files
make rd-campaign-plan

# Run the complete resumable campaign (10k-row cap; CV can take time)
.venv/bin/python -m dclab_rnd campaign run

# Faster first pass; completed tasks are skipped on resume
.venv/bin/python -m dclab_rnd campaign run --quick

# Target one dataset or one scientific stage
.venv/bin/python -m dclab_rnd campaign run --dataset adult --experiment leakage_audit

# Inspect progress or rebuild the evidence/LLM context
make rd-campaign-status
make rd-campaign-report
make rd-campaign-verify
```

The campaign writes to `campaigns/model_building_50_v1/`:

| Artifact | Purpose |
|---|---|
| `manifest.json` | Exact questions, hypotheses, sources, and experiment IDs |
| `results/*.json` | Observed evidence, claims, limitations, and provenance |
| `CAMPAIGN_REPORT.md` | Human-readable cross-dataset synthesis |
| `agent_memory.jsonl` | Claim-level retrieval memory with evidence paths |
| `llm_review_queue.jsonl` | Bounded critic/hypothesis tasks for a capable LLM agent |
| `llm_review_memory.jsonl` | Durable cited conclusions from completed OpenAI reviews |
| `AGENT_CONTEXT.md` | Agent operating contract and required answer structure |
| `MODEL_BUILDING_WORKFLOW.md` | Reusable code/process blocks |

```bash
# Run pending LLM critic reviews (requires OPENAI_API_KEY in .env)
make rd-campaign-review
# or:
.venv/bin/python -m dclab_rnd campaign review --model gpt-5.6-terra
.venv/bin/python -m dclab_rnd campaign review --model gpt-5.6-terra --limit 5
.venv/bin/python -m dclab_rnd campaign review --experiment EXP-001,EXP-002
```

Default model is **gpt-5.6-terra**. Override with `--model` or `OPENAI_MODEL` in `.env`.

The LLM is deliberately a critic and hypothesis generator. Deterministic code owns metrics, split discipline, provenance, and evidence; a human owns ambiguous decision-time semantics and promotion.

### Agentic Research Studio

The interactive research system turns the fixed campaign into an adaptive learning loop. LangGraph owns the durable loop; NVIDIA NOOA `PredictStrategy` specialists interpret goals, audit data, choose a bounded experiment, critique measured results, and select the next test. The OpenAI Responses API supplies the configured reasoning model. Generated Python is never executed: agents return a validated experiment language, and an isolated deterministic worker owns data access, preprocessing, training, and metrics.

```bash
# Python 3.12 or 3.13 is required by NOOA; keep it separate from the ML environment.
uv venv --python 3.13 .venv-agent
uv pip install --python .venv-agent/bin/python -r requirements-agent.lock.txt

# OPENAI_API_KEY is read from the process or local .env; never enter it in the UI.
make agent-test
make agent-serve
# Open http://127.0.0.1:8765
```

The UI supports 1–50-experiment budgets, selected public datasets, pause/resume, run history, input/output diagnostics, evidence-linked knowledge, and trajectory/recipe export. Current results are adaptive development CV. Exact duplicate input rows stay in one fold, preprocessing is fit inside each training fold, decision-time-blocked columns and their descendants are rejected, and source-column masking propagates through derived features. The system records data/code fingerprints, repeat-level metrics, OOF predictions, calibration, input sensitivity, missing-input stress, failures, counterevidence, theory, and reusable workflow blocks.

Agent artifacts live under the ignored `agent_runs/` directory. A trajectory is explicitly marked `training_ready: false`: a human must verify claims, licenses, privacy and source-grouped train/evaluation separation before using it for model training. Promotion still requires fresh external confirmation and the DCLab Product Core lifecycle.

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

## First-class agentic research projects

The local Research Studio now keeps three evidence contexts separate: the open
tabular lab, HyperAck, and Telco Churn. It uses typed NVIDIA Object-Oriented
Agents (NOOA), a durable LangGraph loop, and the OpenAI Responses API. The
default LLM is `gpt-5.6-terra`; raw dataset rows and API keys are not sent to the
LLM, and provider storage is disabled with `store=false`.

### Run locally

```bash
# One-time environments must already be installed; load OPENAI_API_KEY from .env.
.venv-agent/bin/python -m dclab_rnd.agentic serve
# Open http://127.0.0.1:8765

# Re-run/read the fixed 15-experiment churn campaign (no LLM required).
.venv/bin/python -m dclab_rnd.churn_suite run
.venv/bin/python -m dclab_rnd.churn_suite status

# Launch live agentic follow-up research (uses OpenAI API calls).
.venv-agent/bin/python -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4 --rows 11118
.venv-agent/bin/python -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4 --rows 7043
```

HyperAck contains 83 historical experiments: 15 original, 15 leakage-safe and
53 optimized-safe. Its reported optimized-safe leader has ROC-AUC 0.9455. The
0.9793 historical ceiling is explicitly unsafe because it uses post-outcome
final-fare features. New agentic HyperAck experiments block those fields.

Telco Churn now has a fixed, fully executed 15-experiment 2×3-fold development
campaign under `churn_exp/results/`. Its leaderboard and protocol are in
`churn_exp/CHURN_BENCHMARK.md`. These are adaptive development comparisons—not
independent confirmation, production approval, or a universal model ranking.

### Model-building field guide and future LLM memory

```bash
# Deterministically rebuild the human and machine-readable synthesis.
make master-guide

# Run four typed GPT-6 Astra advisory reviewers, then include their cited
# critiques in the guide. Requires OPENAI_API_KEY with available API credit.
make master-review
```

The main entry point is [`knowledge/MODEL_BUILDING_FIELD_GUIDE.md`](knowledge/MODEL_BUILDING_FIELD_GUIDE.md), also available inside the local UI under **Knowledge**. Machine consumers should use `knowledge/master_evidence_pack.json`, `knowledge/model_building_rules.jsonl`, and `knowledge/workflow_blocks.json`. Agent reviews remain advisory; deterministic measurements and repository evidence are authoritative.

The current reviewer lifecycle is recorded in
`knowledge/gpt6_astra_master_review_status.json`. The last provider attempt was
blocked by exhausted API credit, so no invented LLM findings were merged into
the guide; rerun `make master-review` after credits are available.
