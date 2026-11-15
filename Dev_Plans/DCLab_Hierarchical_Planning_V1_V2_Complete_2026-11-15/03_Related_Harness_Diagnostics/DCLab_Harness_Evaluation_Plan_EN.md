# DCLab Harness Evaluation and Model-Only Comparison

**Project:** DCLab Harness Evaluation & Model-Only Baseline  
**Document version:** 1.0 (English edition)  
**Prepared:** November 15, 2026  
**Proposed start:** Monday, November 16, 2026  
**Status:** Ready for implementation and controlled evaluation; no new live-model results have been collected  
**Priority:** P0 / Architectural Diagnosis  
**Reference repository:** `Shahriyar-Moradi/Research_Development_DCLab`, branch `main`

## 1. Executive Objective and Decision

This evaluation is **not** intended to prove that a particular language model is good or bad. Its purpose is to determine whether the current DCLab harness **improves, preserves, or constrains the model's ability** to understand a problem, design a suitable solution, plan the work, and complete the task safely.

We will test three competing hypotheses:

- **H1 - Architectural constraint:** DCLab forces early routing to a problem type, workflow, or solution family before the model can examine the full problem, eliminating viable approaches.
- **H2 - Harness implementation issue:** Context assembly, prompts, tool descriptions, schemas, validation gates, fallback paths, or budgets reduce performance independently of early problem classification.
- **H3 - Model or evidence limitation:** The same underlying model cannot solve the task well even with comparable context and tools outside the current harness.

**Decision rule:** Do not rewrite the architecture, add compensating AI features, or fine-tune a specialist model until the controlled comparisons identify the actual failure mode. A plausible architectural explanation is not experimental evidence.

### Main questions

1. What does the model see before DCLab frames and routes the problem?
2. What does the model propose before code, gates, or tool contracts intervene?
3. Where is the first material divergence between a minimal agent and the current harness?
4. Does each intervention protect correctness and safety, or does it unnecessarily eliminate useful reasoning?
5. Does the current harness justify its additional complexity, latency, and cost?

## 2. What the Existing Code Indicates (Not Yet Live Results)

The following observations are based on a review of the project's source code. They are **not** claims about the live configuration or actual runtime behavior of any particular session.

| Component | Observed design | What the evaluation must verify |
|:--|:--|:--|
| `dclab_rnd/agents/case_agent.py` | Model-led Case Agent with `DCLAB_CASE_AGENT`, route checks, readiness gates, and a policy allowing up to 14 tool steps and 8 model replies per run. | Did this agent actually run? What was the first wrong route or rejected action? |
| `dclab_rnd/agents/runtime.py` | Model/tool execution loop with tool validation, errors returned to the model, stop conditions, and tool-result truncation; default `result_chars=4000`. | Were results truncated, budgets exhausted, or tool calls malformed? |
| `dclab_rnd/agents/context.py` | Context builder with an approximately 48,000-character budget; the initial table view shows metadata, not raw rows. | Was the information available through tools? Was relevant evidence actually observed? |
| `dclab_core/problem_map.py` | Finite problem-family taxonomy; one single-problem path may default to `classification` when no family is identified. | Did a genuinely novel task get bound to the wrong family? Was this code path used in production? |
| `dclab_rnd/draft/chat.py` | Both model-led Home and deterministic script/fallback modes; a ceiling of five primary questions. | Did fallback behavior or compulsory questions obstruct an otherwise valid route? |
| `dclab_rnd/draft/coordinator.py` | Coordinator uses situations, blockers, and validated state transitions. | Was a valid model proposal rejected because of a product/workflow constraint rather than a correctness or safety issue? |
| `dclab_rnd/agent_eval/task_planning_v4` | Existing benchmark runners, dev/sealed splits, and planning gates. | Reuse for preflight checks, but do not mistake the existing benchmark for this new model-only-versus-harness comparison. |

**Evidence boundary:** We have not yet inspected the effective live flags and model/provider configuration for the target runs, captured their complete sent prompts, analyzed production traces, or run fresh model evaluations. Code inspection alone cannot show whether the harness actually lowers model capability.

### Execution path to audit

```text
User request + uploaded files
  -> Entry point / feature flags / selected model
  -> Context builder + evidence + memory
  -> Prompt + offered tools + model response
  -> Problem framing / route / family selection
  -> Solution proposal + tool calls
  -> Validation gates + coordinator state
  -> Execution + observations + replanning
  -> Final response / project artifact
```

At every boundary, capture four things: **what the model observed, what it proposed, what DCLab changed or rejected, and what the user ultimately received.**

## 3. Experimental Design: Four Arms (A/B/C/D)

| Arm | Configuration | What it can establish |
|:--|:--|:--|
| **A. Model Only** | The same base model, problem brief, and prepared evidence; no mandatory workflow and no execution tools. It produces a problem interpretation, options, and a plan. | Baseline for **reasoning and solution-design quality only**, not project execution success. |
| **B. Minimal Agent** | Same model, with essential read/analysis tools and comparable tool permissions and budgets, but no forced early problem type. | Primary control to estimate the additional benefit or harm of DCLab's current harness. |
| **C. Current DCLab** | Existing routing, policies, context builder, gates, coordinator, and fallbacks, with no behavioral changes. | Actual product baseline and evidence for the earliest failure point. |
| **D. Late-Binding Prototype** | Model first constructs a general `ProblemFrame` and multiple solution candidates; only then are they mapped to DCLab's existing type/workflow contracts. Safety checks remain intact. | Targeted intervention to test whether early structural binding is the root cause. |

### Three separate experiments

**P: Planning-quality comparison.** Run A, B, and C against the same frozen evidence snapshot. Do not allow project execution. Standardize the output into a `ProblemFrame` and `SolutionPlan`; blind expert judges score the results. This is the valid setting for comparing a model alone with a harness on thinking and planning.

**E: Execution-quality comparison.** Run B and C with equivalent tools, data, permissions, and execution limits. Evaluate successful completion, correctness, recovery, safety, cost, and latency. **Do not compare A with B/C on end-to-end execution**, because A has no tools.

**D: Causal ablation.** Once baselines are stable, change one component at a time: early routing, context construction, tool schemas, planner constraints, gate feedback, or model tier. Do not change multiple factors simultaneously; that would undermine causal interpretation.

### Fairness and reproducibility controls

- Pin the precise model identifier, provider, model/version availability, sampling parameters, prompt version, tool schemas, and output-token budget. Repeat cases when deterministic seeding is not available.
- Use identical briefs, evidence snapshots, dataset versions, temporal cutoffs, and permission rules across comparable runs.
- Ensure the Model-Only arm sees only information available to the harness, not hidden labels, expected answers, or ground truth.
- Log model requests, input/output tokens, actual wall-clock time, model spending, tool-call count, and limits. Record unavoidable differences explicitly.
- Use a sandbox with synthetic or de-identified data. Do not perform production writes or external actions without authorization.
- Blind judges to the arm identity, and rotate run order to reduce judge and cache/order effects.
- Treat the initial eight-case pilot as a debugging exercise, not statistical proof of superiority.

## 4. Evaluation Dataset and Case Design

### Stage 1 - Initial pilot

Use the **eight development cases** already included in the previous diagnostic package (`intelligence_pilot_cases.json`). First review and freeze their expected behaviors and scoring rubrics. Do not limit the benchmark to straightforward churn/classification examples.

### Stage 2 - Broader study

Create **at least 30 distinct, independently reviewed cases**, spanning:

1. **Ambiguous business problems:** declining sales or increased complaints where the best response may be diagnosis, process change, or instrumentation rather than training a model.
2. **Multi-stage decision problems:** fraud across drivers and customers with limited human review capacity and asymmetric false-positive costs.
3. **Temporal and multi-table problems:** ETA, demand forecasting, or risk prediction where join timing, prediction horizon, and leakage matter.
4. **Causal/decision problems:** churn plus Next Best Action with no historical treatment-outcome dataset. The system must not invent uplift estimates or the causal effect of discounts.
5. **Out-of-taxonomy problems:** experimental design, root-cause analysis, optimization, graph-based problems, or a situation where the correct solution family is initially unknown.
6. **Failure and adaptation:** missing data, failed tools, contradictory evidence, low budgets, or an initially incorrect route. The agent should diagnose and recover.

At least **one-third of cases** should have a correct solution that does **not** involve directly building an ML model.

Each case must have a unique ID and frozen: brief, evidence, allowed tools, expected behaviors, forbidden actions, evaluation rubric, dataset version, and evidence cutoff. Keep answer keys separate from model-visible inputs.

## 5. Evaluation Framework and Metrics

| Dimension | Evaluation question | Measurement |
|:--|:--|:--|
| **Understanding** | Does it identify the actual decision, decision unit, horizon, constraints, unknowns, and cost of mistakes? | 0-4 rubric |
| **Solution Design** | Does it propose defensible alternatives and explain trade-offs? | 0-4 rubric |
| **Planning** | Is the plan executable, with dependencies, evaluation criteria, stop conditions, and fallback paths? | 0-4 rubric |
| **Evidence Discipline** | Does it distinguish facts from assumptions and avoid leakage and unsupported numeric claims? | 0-4 rubric |
| **Adaptation** | Does it meaningfully replan after tool failures or new evidence? | 0-4 rubric |
| **Execution (B/C)** | Does it finish safely with correct tools, arguments, and reproducible results? | Percentages/counts |
| **Safety** | Does it leak a holdout, invent facts, perform unauthorized side effects, or disclose restricted data? | Incident counts; **hard gate** |
| **Efficiency** | What tokens, cost, latency, tool calls, and human interventions are required? | Raw values and cost per accepted task |

**Rubric interpretation:** 0 = missing/incorrect; 1 = poor; 2 = incomplete but useful; 3 = good; 4 = strong and well-supported.

Report the five qualitative dimensions separately. A single average score can hide serious weaknesses or unsafe behavior.

- **Primary planning KPI:** Expert-Accepted Plan Rate.
- **Primary execution KPI:** Safe Task Completion Rate.
- **Primary harness-effect comparison:** Paired B-versus-C delta with first-divergence cause; A-versus-C is valid only within Planning experiment P.
- **Secondary metrics:** Tool-selection accuracy, argument correctness, unnecessary steps, repeated questions, recovery success, failure-class distribution, cost per passed case, and P50/P95 wall-clock latency.

Report paired case-level differences and uncertainty intervals in the expanded study. For categorical pass/fail outcomes, also record the cases where B succeeds and C fails, and vice versa.

## 6. Trace and Observability Contract

Produce a `run_manifest.json` and timestamped `trace.jsonl` per run. Minimum structure:

```json
{
  "run_id": "pilot-001-C",
  "case_id": "INT-01",
  "arm": "C_current_dclab",
  "git_sha": "<pinned commit>",
  "model_id": "<provider/model/version>",
  "flags": {"DCLAB_CASE_AGENT": "<effective value>"},
  "prompt_sha256": "<hash>",
  "tool_schema_sha256": "<hash>",
  "stage": "route_or_gate",
  "event": "gate_reject",
  "chosen_route": "<route or null>",
  "proposed_action": "<action or null>",
  "gate_name": "<name>",
  "gate_reason": "<machine-readable reason>",
  "stop_reason": null,
  "tokens_in": 0,
  "tokens_out": 0,
  "latency_ms": 0,
  "cost_usd": null
}
```

Also capture: evidence hashes, sanitized tool arguments, tool result or error, omitted-context count, the moment a problem family was selected, first candidate plan, plan after validation, user acceptance/rejection, and final artifact version.

**Trace privacy:** Never persist secrets, private customer records, raw personal information, or proprietary hidden model reasoning without explicit permission and the appropriate usage rights. Observable proposals, tool actions, validation decisions, and outcomes are generally enough to diagnose the harness. Separate debug records from training-eligible datasets and track provenance/licensing for any later reuse.

## 7. First-Divergence Root-Cause Analysis

For every paired run, compare more than the final answer:

1. **Observation parity:** Did the compared arms actually receive the same relevant facts and usable evidence?
2. **Initial framing:** How did each arm define the problem? Did C select a type before sufficient investigation?
3. **First viable plan:** Did the model inside C produce a good alternative that the harness later discarded or rewrote?
4. **Gate/tool boundary:** Was rejection driven by safety/data validity or simply by the current product's expected shape?
5. **Recovery:** Could the model revise the plan, choose an alternate tool, or recover from a rejection or failure?

Use one or more explicit reason codes:

```text
early_binding | context_loss | model_error | bad_prompt |
tool_missing | tool_schema_error | gate_rejection | fallback |
budget_exhausted | execution_error | dataset_ambiguity | unknown
```

**Interpretation safeguard:** Better prose from a Model-Only run does not prove the harness suppresses intelligence. Demonstrate a specific lost observation, forced decision, invalid rejection, or failed recovery in the trace.

## 8. Implementation Roadmap and Checkpoints

### P0 - Source/Runtime Audit (no production behavior changes)

- [ ] Pin the Git commit, effective feature flags, provider/model, prompt versions, and tool schemas.
- [ ] Trace at least **two previously failed real requests** through `entrypoint -> case_agent / Home / fallback -> coordinator`.
- [ ] For each, reconstruct the sequence of model proposal, route, gate decision, tool output, and user-visible artifact.
- [ ] Check the `classification` fallback in `problem_map`, route checks, question ceilings, tool-result truncation, and exact stop reasons.
- [ ] Classify each failure as understanding, planning, routing, validation, or execution.

**Checkpoint:** A factual architecture trace exists and the run configuration is known. No architectural change is authorized yet.

### P1 - Existing Evaluation Infrastructure Preflight

From the repository root, use the currently documented task-planning benchmark entry point (confirm command availability and arguments in the checked-out commit):

```bash
python -m dclab_rnd.agent_eval.task_planning_v4 verify-sealed
python -m dclab_rnd.agent_eval.task_planning_v4 run-offline --split dev --limit 8
```

These commands test existing infrastructure; they **do not** implement the new A/B/C comparison. If the installed CLI differs, inspect `--help` or the module entry point rather than assuming the commands are valid.

**Checkpoint:** Evaluation scaffolding, evidence capture, and benchmark integrity work without live-model spending. Record any preflight failure independently.

### P2 - Build a Non-Production Comparison Runner

- [ ] Add a runner parameterized by `case_id x arm x repetition`.
- [ ] Implement three separate adapters: `ModelOnlyAdapter`, `MinimalAgentAdapter`, and `CurrentDCLabAdapter`.
- [ ] Normalize planning responses to shared `ProblemFrame` and `PlanArtifact` output contracts **without requiring a family to be chosen upfront**.
- [ ] Make all arms consume the same frozen evidence reference, and audit their available tools/permissions.
- [ ] Run the eight dev cases through A/B/C for experiment P. Run only B/C for experiment E.
- [ ] Write one manifest, one structured trace, one scorecard row, and one first-divergence record per run.
- [ ] Freeze prompts and conditions during baseline collection; version every subsequent change.

**Checkpoint:** Reproducible pilot results, not merely anecdotal chat transcripts.

### P3 - Targeted Ablations

- [ ] If `early_binding` dominates, implement D behind a separate feature flag, allowing flexible framing before mapping to existing workflows.
- [ ] If `context_loss` dominates, repair evidence retrieval/observation parity before changing planning.
- [ ] If `gate_rejection` dominates, distinguish safety from format constraints and improve recoverable gate feedback; **do not remove safety gates**.
- [ ] If `model_error` dominates, compare a stronger model or revised prompt under identical conditions before considering fine-tuning.
- [ ] Change one factor at a time and retain the original C baseline.

**Checkpoint:** A specific mechanism explains the observed improvement or failure, with supporting traces.

### P4 - Expanded Study and Architecture Decision

- [ ] Freeze evaluation cases, scoring rubrics, and thresholds **before** looking at final-set outcomes.
- [ ] Evaluate at least 30 reviewed cases with repetitions where supported, paired scoring, and uncertainty intervals.
- [ ] Include safety incidents and latency/cost alongside quality gains.
- [ ] Recommend one of: keep the existing harness, repair a focused subsystem, introduce late binding gradually, upgrade the model, or collect more evidence.

**Checkpoint:** Documented Go/No-Go decision backed by accepted plans, safe completions, paired deltas, and trace-level explanations.

## 9. Decision Matrix

| Observed pattern | Likely explanation | Recommended next action |
|:--|:--|:--|
| A and B perform well; C is poor | Routing, context, gates, or DCLab-specific constraints | Fix the identified boundary; do not immediately fine-tune. |
| A plans well; B and C fail during execution | Tool contracts, permissions, or execution integration | Repair tool interfaces, results, and recovery behavior. |
| A, B, and C all frame or plan poorly | Weak model, inadequate evidence, prompt, or task definition | Test a stronger model and better evidence first; evaluate training later. |
| C performs well; B is weaker | The existing harness adds value | Retain architecture and optimize its overhead. |
| D beats C without new safety failures | Late binding may be beneficial | Gradually roll out behind a feature flag. |
| C proposes a valid approach but a gate suppresses it | Unnecessarily rigid schema or non-safety gate | Improve contract semantics and feedback, preserving real safety checks. |

**Provisional change trigger, not a result:** An improvement on the order of **15 percentage points in Expert-Accepted Plan Rate** could warrant an architectural intervention in the expanded study, but only with sufficient confidence, no new critical safety failures, and acceptable cost/latency. An eight-case pilot cannot establish this statistically.

## 10. Definition of Done and Deliverables

For the first evaluation cycle, consider the work ready for review only if:

- [ ] We can explain the real execution path of at least two known failures.
- [ ] Arms A, B, and C have been run with pinned models and matched evidence, or a precise technical blocker is recorded.
- [ ] Every case has a scorecard, including unsuccessful and interrupted runs.
- [ ] We have identified the **earliest** meaningful difference between C and B (or A for planning only), linked to a trace.
- [ ] Exactly one priority root cause and a narrow next experiment have been selected.

**Proposed repository deliverables (not yet created in the repository):**

```text
docs/evaluations/harness-vs-model/PLAN.md
experiments/harness_eval/cases_dev.json
experiments/harness_eval/run_manifest.json
experiments/harness_eval/traces/*.jsonl
experiments/harness_eval/scorecard.csv
experiments/harness_eval/summary.md
```

Recommended additional artifacts: `rubric_v1.json`, `first_divergence.csv`, `ablation_registry.json`, and `go_no_go_decision.md`.

## 11. GitHub Issue: Ready-to-Use Specification

**Title:** `P0: Harness Audit & Controlled Model-Only vs DCLab Evaluation`

**Description:**

Audit the real DCLab runtime path, including the Case Agent, Home/fallback path, Context Builder, prompts, routing, ProblemMap, coordinator, and tool/validation gates. In parallel, perform a controlled comparison of Model Only (understanding and planning), Minimal Agent (matched tools and permissions), and Current DCLab on nontrivial scenarios. The primary objective is to identify the **first divergence** and establish whether early deterministic binding, context/tool contract limitations, or underlying model capability explains the difference in quality. All evaluation should be sandboxed and observe the same safety boundaries. Produce an evidence-backed decision to make a focused repair, test late binding, or pursue a model improvement.

**Acceptance criteria:**

- At least two complete traces of known failures.
- Paired pilot of eight cases with all applicable arms.
- Blind scorecards and explicit first-divergence classifications.
- Written Go/No-Go decision and next experiment.
- No unauthorized production changes or side effects.

**Labels:** `P0`, `architecture`, `harness`, `evaluation`, `agentic-ai`, `research`.

## 12. Source References and Status

- [Case Agent](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/agents/case_agent.py)
- [Agent Runtime](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/agents/runtime.py)
- [Context Builder](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/agents/context.py)
- [Problem Map](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_core/problem_map.py)
- [Home Agent](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/draft/chat.py)
- [Coordinator](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/draft/coordinator.py)
- [Existing Planning Benchmark](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/tree/main/dclab_rnd/agent_eval/task_planning_v4)

**Current execution status:** The original review consisted of static source inspection. No new live comparisons, GitHub commits, or project issues were created as part of preparing this English plan. The metrics and acceptance thresholds above are evaluation specifications, not measured outcomes.
