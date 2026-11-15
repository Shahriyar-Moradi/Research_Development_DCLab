# DCLab | Agentic ML Dataset, Fine-Tuning & Evaluation Plan (V1)

**Original design date:** November 5, 2026  
**Owner:** DCLab Research & Engineering  
**Status:** Proposed implementation plan, not an execution report or benchmark result.  
**Scope:** This workstream is independent of Hierarchical Planning V1 and V2. Integration occurs only through DCLab's LLM Gateway, Tool Registry and validated contracts.

## 1. Objective and success definition

Build a DCLab-specific, open-weight, fine-tuned **Agentic Machine Learning model** that could become DCLab's default main model **only after satisfying pre-registered evaluation gates**. This model will propose interpretations and decisions; it will never replace deterministic execution, scientific validators, authorization, safety boundaries or artifact verification.

**Proposed 30-day outputs:** audited and versioned datasets, one or two genuinely trained adapters, a comparison with the unfine-tuned base and strong OpenAI/Claude baselines, and an evidence-backed `PROMOTE`, `LIMITED-PROMOTION` or `REJECT` verdict. Becoming the main model is not guaranteed; a candidate that fails a gate remains in shadow mode.

| Program | Responsibility and deliverables | Dependency |
|---|---|---|
| Hierarchical Planning V1 | TaskGraph, TaskExecutionPlan, ordered execution, safeguards, verification | Must be completed and accepted |
| Hierarchical Planning V2 | Dynamic collaboration, debate, parallel experiments, experience memory | Only after V1 acceptance |
| **Agentic ML Model Program** | Dataset creation, model training, evaluation, controlled rollout | Can begin independently; integrates operationally only through validated V1/V2 gates |

**Critical update (November 15):** Before treating fine-tuning as an intelligence fix, perform the separate Model-Only vs Minimal Agent vs Current Harness vs Late-Binding study. Fine-tuning cannot resolve a hard-coded solution space that prevents the model from making better proposals.

## 2. Data, training and evaluation architecture

```text
DCLab events and traces + human-written examples + permitted open datasets
       |  inventory, provenance, consent, source licensing
       v
Canonical episode records
       |  privacy, quality, deduplication, independent scientific verification
       v
    TRAIN         DEV         SEALED TEST
       |                        |
SFT / LoRA / QLoRA      Immutable evaluation manifest
       |                        |
Candidate adapters -- identical DCLab harness, tools and budgets
                          |
           Deterministic validators and scientific review
                          |
               Promotion decision and rollback
```

**Separation of responsibilities:** The model understands user intent, chooses tools, plans, writes/repairs code, interprets evidence, and abstains when necessary. Deterministic infrastructure owns permissions, execution, budgets, artifacts, leakage detection, metrics, final holdout access and event records. Training data must have proven rights and defensible quality. Benchmarks and sealed answers must never be used to tune prompts, select examples or train candidates.

## 3. Data sources and permitted use

| Tier | Source | Intended use | Conditions |
|---|---|---|---|
| A | Owned DCLab execution traces and actual results | Primary SFT and repair corpus | Ownership/consent, no exposed PII, independently verified tool outcomes |
| B | Human-authored or reviewed examples | Gold supervision, hard failures, safe refusal | Reviewer identity and acceptance rubric recorded |
| C | Permissively licensed public datasets and code, including suitable OpenML/UCI/repositories | Training tasks and executed examples | Verify each dataset's and repository's actual license, URL and commit |
| D | Permitted open-weight model generations paired with executed results | Diversity and recovery examples | Independent verification; `synthetic=true` |
| E | OpenAI / Claude API runs | Independent comparison and evaluation | Do not reuse outputs or judgments to train or rank training examples without explicit applicable authorization |
| F | MLE-bench Lite, verified ScienceAgentBench, DSBench and internal sealed suites | Final evaluation | Never move tasks, answers, traces or feedback to train/dev |

**Provider rights:** Recheck current OpenAI and Anthropic account-specific terms and their effective dates before export or training. The November 2025/2026 policy discussion in the source plan is an audit flag, not legal clearance. Keep `provider_output_train_allowed=false` until the specific use is documented as permitted. Even model-derived labels or preference rankings can raise contractual concerns. Use independent human and executable validators instead.

**Privacy:** No secrets, API keys, unauthorized customer data, sensitive unredacted values, private chain-of-thought, or protected benchmark content in training examples. Store safe references and content hashes where possible.

## 4. Dataset taxonomy and initial targets

An initial **1,000-3,000 verified high-quality episodes** is a planning target, not a quality guarantee. If fewer valid episodes exist, do not manufacture volume.

| Capability | Illustrative initial proportion | Cases |
|---|---:|---|
| Problem Understanding & Routing | 15% | Ambiguous goals, missing targets, invalid intents, correct interpretation |
| Hierarchical Task Planning | 20% | Valid TaskGraph, dependencies, blocked tasks |
| Code & ML Engineering | 20% | Python, features, fit/evaluation, debugging |
| Tool Selection & Arguments | 15% | Registry choice, JSON schema, scope and missing arguments |
| Evidence & Scientific Validation | 15% | Metrics, leakage, split, holdout and grounded findings |
| Failure Recovery & Safe Abstention | 10% | Timeout, no data, permissions, stale graph |
| User-Facing Explanation | 5% | Evidence-based decisions, honest uncertainty |

Coverage should include classification, regression, clustering/segmentation, time series, anomaly detection, NLP, CV, multi-table work and data-quality problems. Add a family to the **supported** set only if execution tools and actual scientific evaluators exist; otherwise teach honest abstention.

**Required negative examples:** leakage after splitting, wrong metric, train/test contamination, fabricated results, permission bypass, wrong tool or arguments, duplicate write after timeout, holdout reuse, claimed success without artifacts, and data-work requests incorrectly routed to advice.

## 5. Canonical Episode Record

A single episode may produce multiple SFT and evaluation examples, provided all derived examples preserve common group IDs, provenance and split identity. The following is conceptual; implement against actual repository schemas and model chat templates.

```json
{
  "schema_version": "agentic_ml_episode.v1",
  "episode_id": "ep_000001",
  "source": {"type": "dclab_run", "origin_id": "run_42", "license": "owned", "train_allowed": true},
  "case": {"case_id": "case_17", "family": "classification", "difficulty": "medium", "group_id": "dataset_family_17"},
  "inputs": {"user_objective": "Train a churn classifier", "context_refs": ["sha256:..."]},
  "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}],
  "tools": [{"name": "tool_name", "json_schema_version": "v1"}],
  "trajectory": [{"step": 1, "type": "tool_call", "tool": "tool_name", "args_hash": "sha256:...", "outcome": "verified"}],
  "artifacts": [{"kind": "code", "uri": "artifact://...", "sha256": "..."}],
  "validation": {"verdict": "verified", "verifier_version": "v1", "evidence_ref": "evidence://...", "reviewer": "human_or_rule"},
  "labels": {"tool_valid": true, "scientific_correct": true, "safe_stop": false},
  "metadata": {"created_at": "2026-11-05", "generation_source": "owned", "contains_pii": false},
  "split": "train",
  "quality": {"status": "approved", "score": 0.95}
}
```

**Exports:** SFT uses template-compatible `messages`/`tools`; preference data uses `prompt/chosen/rejected` only when labeling rights and validation are established; evaluation uses a case manifest and a private independent verifier. Never expose gold answers to the model. Preserve executable Python/notebook/diff, environment versions, input references, exit code, bounded logs, reproducible metrics, commit and dataset version. Code becomes gold only after passing execution and scientific tests.

## 6. Data-quality pipeline

1. **Inventory:** Count each record, type, source, license, owner and approval status.
2. **Normalize:** Convert into a versioned schema with distinct episode, step and artifact entities.
3. **Privacy and license gate:** Reject secrets, PII, rights-unknown records and prohibited commercial-model outputs.
4. **Validity:** Check schema, tool registry version, arguments and actual sequence of steps.
5. **Scientific review:** Independently rerun code/tests and check metrics, leakage and holdout usage.
6. **Deduplicate:** Use exact and near-duplicate fingerprints across all candidate splits.
7. **Group split:** Initial planning ratio `70/15/15` for train/dev/sealed at **dataset/source/task-family level**, not at message or row level. All derivatives of one task remain together.
8. **Freeze:** Record manifest, snapshot hash, split counts, timestamp, group ratios and owner.
9. **Quarantine:** Send uncertain or incomplete examples to review, never directly to SFT.

Maintain a shared **Sealed Benchmark Registry** containing competition IDs, dataset names, source hashes, URLs, task fingerprints and similarity checks applied during both collection and export.

## 7. Base-model choice and training

Compare candidates in **the same harness before training**, rather than presuming a winner:

| Candidate mentioned in original plan | Role | Verification before adoption |
|---|---|---|
| Qwen3.5-4B | Lower-cost routing/tool-calling option | Model availability, license, tool-template reliability, memory and latency |
| Qwen3-8B | Comparable open-weight coding/planning baseline | License, quality and tool calling |
| Qwen3.5-9B | Potential higher-capability ML-engineering candidate | License, inference/training memory and performance |

The original proposal prioritizes 4B and 9B for the first comparison and fine-tunes only the strongest with an alternative as a control. Treat specific license/version statements as claims requiring model-card verification, not automatically valid forever.

**Stages:** A) Evaluate base models without fine-tuning on dev, including JSON validity, tool choice, science, latency and memory. B) Train SFT with LoRA/QLoRA using approved examples and objectives for tool use, planning, safe stopping and repair. C) Train a small targeted repair iteration on legally permitted dev-discovered failure categories, followed by full regressions. D) Run optional DPO only if independently reliable and train-permitted chosen/rejected pairs exist. E) Freeze base/adapter hashes, dataset, tokenizer, chat template, seed, precision, recipe and environment.

**Compute:** Local Apple Silicon with ~32GB unified memory is appropriate for development and quantized smoke tests, but reproducible 8B/9B training with larger contexts may require rented GPUs. Run a short GPU/environment smoke test before committing a large budget. Check current PEFT/TRL and MLX-LM support for the selected model and tool format.

**Prevent catastrophic forgetting:** Try a legally permitted 15-30% replay buffer of general reasoning, coding and tool-use tasks, with ablation rather than assuming the ratio is ideal. Use modest epochs, early stopping and versioned checkpoints. Measure general capabilities and scientific/code performance before and after each stage. Keep adapters individually reversible without replacing their base model.

## 8. Evaluation suite and promotion criteria

| Arm | Configuration | Question answered |
|---|---|---|
| A | Base open-weight model | Net effect of fine-tuning |
| B | DCLab fine-tuned candidate | Main-model candidate quality |
| C | Strong OpenAI API model | Capability/cost reference |
| D | Strong Claude API model | Capability/cost reference |
| E | DCLab deterministic/fixed V1 fallback | Existing safety/reliability baseline |
| F, later | DCLab V2 multi-agent + candidate | Multi-agent effect, only after V2 exists |

All comparisons use matched input snapshots, prompts where applicable, tool access, allowed budget and hardware reporting. Log model/adapter IDs, prompt and registry hashes, dataset hash, seed, token use, spend, latency, GPU time, steps, tool calls, errors and evidence. Run stochastic cases at least three times when feasible, report uncertainty and failure classes. **Model score and system score are distinct.**

**Test layers:** L0 format/schema/refusals; L1 routing, planning, tool choice, arguments, repair and evidence; L2 scientific correctness (leakage, split, metrics, reproducibility); L3 isolated end-to-end execution with verified artifacts; L4 adversarial and recovery (injection, permissions, no data, exhausted budget, holdout); L5 external sealed benchmarks (MLE-bench Lite, verified ScienceAgentBench and DSBench if applicable).

**Sample planning:** Around 200 varied dev cases and at least 100 independent internal sealed cases, adjusted to budget and confidence intervals. For MLE-bench Lite (22 competitions in the source proposal), do infrastructure rehearsals with a distinct dev route, then run a fixed official-style protocol where feasible. Never present cherry-picked contests as a full benchmark result.

| Gate | Proposed acceptance criterion, to freeze before sealed testing |
|---|---|
| G0 Data | 100% permitted and provenance-verified training records; zero train/sealed overlap |
| G1 Safety | Zero **observed critical** unauthorized writes, holdout re-reads, policy bypasses or fabricated completion on the defined set |
| G2 Format/Tools | >=98% schema-valid outputs and >=95% valid tool arguments in dev |
| G3 Science | >=95% pass on defined scientific guard cases; zero critical science failure observed |
| G4 End-to-End | At least equal to V1 on priority families; no predefined meaningful regression |
| G5 General capability | No degradation larger than 2 percentage points on general regression suite, adjusted for uncertainty |
| G6 Economics | Meets pre-registered all-in GPU/API cost and latency SLO, including fallback rate |
| G7 External | Reproducible protocol and results; threshold defined before seeing outcomes |

These are **illustrative gates**, not measured results or statistical proof of zero risk. Validate power, sample size and business risk first. Deterministic high-risk validators and human stop controls remain in production.

**Promotion decision:** `REJECT` on severe rights, safety, science, end-to-end or capability failures. `LIMITED-PROMOTION` for narrow verified strengths such as routing/tool selection, with escalation elsewhere. `PROMOTE` only after all frozen gates, shadow run, 5% then 25% canary, regression observation, kill-switch test and successful rollback. No OpenAI/Claude outputs enter training or preferences without explicit approval.

## 9. Independent 30-day roadmap

| Week | Work | Artifacts | Exit condition |
|---|---|---|---|
| 1, days 1-7 | Inventory, provenance/rights, schema, privacy, dedup, grouped split, contamination guards | `DATASET_CARD.md`, `data_manifest.json`, `schema.json`, `quality_report.md`, train/dev/sealed manifests | Every trainable record has rights/source; sealed isolated |
| 2, days 8-14 | Untrained base comparisons, small LoRA/QLoRA first SFT, replay and checkpoints | `baseline_report.md`, `sft_recipe.yaml`, `adapter-v0`, `model_card.md` | No unacceptable general/safety regression |
| 3, days 15-21 | Failure taxonomy, legally permitted repair samples, v1 adapter and ablations | `error_analysis.md`, `adapter-v1`, `ablation_report.md` | Practical gains on target tasks, not merely lower loss |
| 4, days 22-30 | Freeze, blind internal/sealed testing, comparable strong-API baselines, economic curves and shadow | `model_comparison.md`, `benchmark_report.md`, `promotion_verdict.md`, `rollback_runbook.md` | Evidence-backed promotion/rejection verdict |

## 10. Proposed repository structure

Create only after inspecting actual repository HEAD and ownership conventions:

```text
planning/
  agentic_ml_model/
    README.md
    DATASET_PLAN.md
    DATASET_SCHEMA.json
    DATA_POLICY.md
    MODEL_SELECTION.md
    TRAINING_RECIPE.md
    EVAL_PLAN.md
    PROMOTION_GATES.md
    MONTH_01_ROADMAP.md
    codex_handoff.md

dclab_rnd/agentic_training/  # only if no equivalent exists
  ingest.py
  normalize.py
  validate.py
  split.py
  export_sft.py
  eval_manifest.py
  quality_report.py
```

Reuse `dclab_rnd/agents/harness.py`, traces, registry, model gateway and existing benchmarks. Keep training datasets and evidence in a versioned private artifact store, not Git. Test ETL on tiny CPU-only fixtures in CI.

## 11. Codex handoff: audit only

```text
You are implementing the independent DCLab Agentic ML Model Program.
This is NOT Hierarchical Planning V2; do not edit V1 or implement V2 features.

First inspect HEAD, git status, README, CLAUDE.md, dataset locations,
current benchmark harness/traces, model gateway, storage conventions,
privacy policies and repo tests. Do not overwrite user changes.

For this first package perform an AUDIT AND DESIGN ONLY:
1. Inventory existing training/evaluation data by source, schema,
   licensing, quality, lineage and privacy risk.
2. Identify the minimum canonical Episode schema and versioning method.
3. Map existing tool traces, code artifacts and independent evidence
   to that schema; no invented evidence or results.
4. Identify all benchmark contamination risks and freeze a proposed
   blocked-source registry; do not access sealed answers for training.
5. Document OpenAI/Anthropic output restrictions; never auto-import
   their outputs or judgments into training datasets.
6. Specify Train/Dev/Sealed group splits and deterministic rules.
7. Produce a gap report, acceptance criteria and small implementation
   packages for dataset ETL, model training, and evaluation.
8. Avoid live API calls, GPU training, repository writes outside the
   requested documentation and all cloud spending without approval.

Deliver: inventory, source policy, canonical schema proposal,
Data Quality pipeline spec, contamination controls, repository paths,
tests to write, a one-month roadmap, known blockers and exact
commands/tests actually executed. No claims of model capability.
```

## 12. Official and reference resources

- [OpenAI MLE-bench](https://github.com/openai/mle-bench)
- [ScienceAgentBench](https://github.com/OSU-NLP-Group/ScienceAgentBench)
- [OpenML benchmark suites](https://docs.openml.org/benchmark/)
- [Hugging Face TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats)
- [Hugging Face PEFT integration](https://huggingface.co/docs/trl/peft_integration)
- [MLX-LM](https://github.com/ml-explore/mlx-lm)
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)
- [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B)
- [Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
- [OpenAI Business Terms](https://openai.com/policies/may-2025-business-terms/)
- [OpenAI Usage Policies](https://openai.com/policies/usage-policies/)
- [Anthropic Legal Center](https://www.anthropic.com/legal)
