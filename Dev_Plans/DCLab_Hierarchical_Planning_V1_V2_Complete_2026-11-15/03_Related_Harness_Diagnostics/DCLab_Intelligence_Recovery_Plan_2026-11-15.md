# DCLab Intelligence Recovery: Architecture Diagnosis and Ablation Plan

**Status:** Proposed / diagnosis not yet established  
**Recorded:** 2026-11-15 (Asia/Dubai)  
**Next working session:** 2026-11-16  
**Priority:** P0 architecture-and-evaluation investigation  
**Scope:** Problem Understanding → Solution Design → Task Planning → Tool Selection → Safe Execution  
**Repository:** `Shahriyar-Moradi/Research_Development_DCLab` (main branch inspected on 2026-11-15)

> **Decision for tomorrow:** Freeze feature expansion and model fine-tuning until a controlled, trace-backed comparison isolates the main failure mode. This is a proposed decision, not a claim that any experiment has passed.

## 1. The problem we are trying to solve

The owner's current experience is that DCLab performs narrow/simple modeling tasks but does not reliably:

1. Understand an open-ended, underspecified, or novel business/data-science problem.
2. Frame a meaningful decision problem: unit, horizon, available evidence, constraints, objective, and costs.
3. Generate and compare plausible solution strategies instead of quickly mapping a request onto a catalog of known problem types.
4. Create a viable multi-stage research/ML/decision plan, handle obstacles, and revise it using new evidence.
5. Deliver nontrivial value beyond standard churn/classification workflow execution.

**Working hypothesis:** parts of the routing, problem ontology, context handling, planning, and deterministic gates may cause *effective intelligence suppression*: the capable underlying model's useful decisions do not reliably survive the harness. This does **not** establish that the model itself is strong enough, or that all deterministic constraints are bad.

**Important distinction:** deterministic **validation, permissions, reproducibility, measurement and side-effect boundaries** are desirable; deterministic **premature choice of problem framing and solution approach** can be harmful. The goal is not an unrestricted agent.

## 2. What repository inspection establishes, and what remains unknown

The following are source-code observations from a limited GitHub read, **not runtime findings**:

| Path | Relevant observation | Hypothesis to test |
|---|---|---|
| `dclab_rnd/agents/case_agent.py` | Has an AI-first lead `case_agent`, tools, readiness checks, a second route check and a `DCLAB_CASE_AGENT` switch; code comments say enabled by default when a model serves. | Is this path used in the failing user sessions, or is a fallback/route gate taking over? |
| `dclab_rnd/draft/chat.py` | A narrow Home role and an optional deterministic script exist; explicit question and model/step budgets are present. | Does Home's framing prematurely select the wrong task or constrain discovery? |
| `dclab_core/problem_map.py` | `Family` is a closed taxonomy; `decompose` uses rules/templates plus validation. | Are novel/compound problems flattened, misclassified, or rejected before a useful free-form hypothesis is explored? |
| `dclab_rnd/draft/coordinator.py` | Walks from problem to built project using state-machine gates and allowed next actions. | Does a gate cause a premature handoff, refuse a good proposal, or route toward the simplest permitted build? |
| `dclab_rnd/agents/context.py` | Constructs bounded context containing case facts, table column summaries and turns; raw table values are not included in this view and older turns may be dropped. | Is the agent reasoning from insufficient evidence, or missing important context after compaction? |
| `dclab_rnd/agents/runtime.py` | Tool loop, step/reply/time limits, schema checks, tracer and deterministic tool-side validation. | Is model intent lost through tool schemas, truncation, budget exhaustion, or rejected calls? |
| `dclab_rnd/agent_eval/task_planning_v4/` | Already contains dev/sealed planning cases, gates, offline/live runners and SHA verification. | Reuse existing eval infrastructure rather than invent another scorer; add a separate harness-ablation study. |

Source links:
- [case_agent.py](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/agents/case_agent.py)
- [problem_map.py](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_core/problem_map.py)
- [coordinator.py](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/draft/coordinator.py)
- [context.py](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/blob/main/dclab_rnd/agents/context.py)
- [TP-08 planning benchmark](https://github.com/Shahriyar-Moradi/Research_Development_DCLab/tree/main/dclab_rnd/agent_eval/task_planning_v4)

**What we have NOT verified:** Which code path actually served the owner's failing sessions, prompts/tool schemas as delivered to the provider, final artifacts/traces, current production configuration, model tiers, tool failures, or benchmark execution results. Do not call a suspected source-code bottleneck a proven root cause.

### Prior signal to reproduce, not generalize

An earlier owner-reported evaluation showed a Claude-alone arm completing 19/19 tasks while the DCLab lead-agent arm completed 15/19, with higher cost and extra turns. This is a strong reason to investigate, **not** a fair current claim about the latest version without reproducing the same conditions.

## 3. Hypothesis register

| ID | Hypothesis | Observable signature | Falsifier / alternative explanation |
|---|---|---|---|
| H1 | Premature classification into fixed `ProblemType`/family distorts understanding. | Early route or family chosen before unit/horizon/decision is understood; viable alternatives never considered. | Correct open-ended framing already exists upstream and survives to plan. |
| H2 | Model-proposed options/plans are suppressed by validation/coordination gates. | A sensible proposal is visible in a trace, then rejected/rewritten for a non-safety reason. | Rejected proposal was infeasible, fabricated, or unsafe. |
| H3 | Context/observation bottleneck hides crucial evidence. | Minimal agent with equivalent evidence succeeds; current agent omits key facts or misses discarded turns. | Both agents see identical useful evidence and make same error. |
| H4 | Planner is optimizing a predetermined workflow rather than exploring alternatives. | Only one template path is produced where independent alternative strategies are called for. | Alternatives exist and are assessed against explicit constraints. |
| H5 | Routing/provider/model tier or tool budget is the dominant source. | Error tracks weaker model, fallback, cap hit, or malformed tool calls. | Error persists with same strong model, enough budget, and stable tool calls. |
| H6 | Tool APIs/schemas make the right action impossible or excessively indirect. | Agent recognizes right action but has no permitted tool or usable arguments. | Tools expose the action and work when directly invoked. |
| H7 | Benchmark or expectation is misaligned with the product contract. | "Failures" are ambiguous tasks or capabilities not yet within product scope. | Expert rubric confirms in-scope, actionable failures. |
| H8 | The base model's reasoning is genuinely insufficient for this class of problems. | Strong base, minimal agent and DCLab all fail despite adequate context. | Base or minimal agent reliably solves with the same model and evidence. |

## 4. Controlled comparison: separate intelligence from execution

Run the **same cases** under the **same underlying model version**, with fixed seed/temperature where supported, matching token/time/cost budget and the same authorized dataset/tools wherever applicable. Record any non-comparability.

| Arm | Description | What it tests | Comparison scope |
|---|---|---|---|
| **A: Direct LLM** | Strong-model answer to a full *read-only* problem brief, no tools. | Can the model understand and propose good strategies? | **Understanding and plan quality only**; not execution success. |
| **B: Minimal Agent** | One model-driven loop, neutral mission prompt, access to equivalent read/query/validate/execution tools. No mandatory early problem-family selection. | Does a simple harness recover the missing capability? | Primary fair execution comparison against C. |
| **C: DCLab Current** | The actual active DCLab Home/Case/Coordinator path, unchanged. Log route/fallback. | Real baseline and failure trace. | Primary baseline. |
| **D: Late-binding Hybrid** | Free-form problem framing and candidate plans **before** family/workflow binding; deterministic checks at execution boundaries. | Does moving classification later fix quality while preserving safety? | Proposed intervention; implement after baseline. |
| **E: Rules fallback (optional)** | Force documented fallback in a sandbox. | Identify how much error comes from rule-based selection. | Diagnostic only. |

**Critical fairness rule:** A (no tools) cannot be used to claim superior end-to-end task completion against C. For B/C/D, tools, data and side-effect permissions must be equivalent. Preserve the same hard policy/safety gates in all executing arms; ablate *premature choice constraints*, not safety checks.

### Exact trace fields to collect

`case_id`, `arm`, `run_id`, `git_sha`, `model_provider`, `model_name`, `model_params`, `prompt_hash`, `tool_schema_hash`, `context_hash`, `context_chars`, `turns_dropped`, `route`, `family_selected`, `when_family_selected`, `fallback_reason`, `gate_name`, `gate_decision`, `gate_reason`, `tool_name`, `tool_args`, `tool_error`, `model_step`, `tokens_in`, `tokens_out`, `cost`, `latency`, `final_artifact`, `human_review`.

Do not record secret keys, private customer rows, hidden evaluations, or raw provider reasoning that the provider does not authorize retaining. Capture observable proposals, decisions, tool calls and errors.

## 5. Test set and case quality

**Tomorrow's pilot:** 8 diagnostic development cases in `intelligence_pilot_cases.json`. These are proposed fixtures, not certified ground truth. Have an expert review acceptable decisions before scoring. Include at least one genuinely open-ended case and cases with competing plausible solutions.

**After pilot:** 30+ *independent* cases across 3 strata: (a) underspecified problem framing, (b) multi-stage/novel solution design, (c) data/safety traps and corrective planning. Use fresh real user briefs when possible. Maintain separate development and sealed holdout sets. **Never tune prompts against the sealed set.** Use repeated seeds if the model and budget allow.

The repository already has `task_planning_v4` dev and sealed cases. Preserve their sealed SHA; use them as a preflight/reference suite, but do not pretend they automatically measure complete product intelligence or run the new arms.

### Examples of required reasoning

- Define prediction/decision unit and time window before selecting a model family.
- Detect post-event leakage and time-based validation needs.
- Distinguish churn *risk prediction* from next-best-action *causal effect*, especially without treatment outcomes.
- Treat small numbers of fraud labels and review capacity as a decision/evidence problem, not simply a classifier request.
- Ask targeted clarifications where uncertainty changes the solution; avoid repeated obvious questions.
- Propose multiple viable approaches and justify tradeoffs, including *no ML yet* where appropriate.
- Recover when an assumption fails, rather than sticking to the first task type.

## 6. Independent evaluation rubric

Use an expert-blind score of **0–4** on each dimension (0 = absent/incorrect, 1 = mostly wrong, 2 = partial, 3 = competent, 4 = strong and evidence-backed):

1. **Problem framing:** identifies business objective, decision/action, unit, moment/horizon, target/evidence, constraints and unknowns without inventing facts.
2. **Solution intelligence:** generates nontrivial alternatives, reasons about tradeoffs and selects a defensible approach.
3. **Plan quality:** ordered, executable, appropriately scoped tasks with dependencies, data needs, validation, stop conditions and contingency handling.
4. **Evidence discipline:** distinguishes supplied facts/observations from assumptions, quantifies only with evidence, detects material ambiguity.
5. **Adaptive behavior:** changes course when observations/constraints conflict with first assumptions; asks the right question at the right time.

Additional binary outcomes: valid build/analysis, correct tool selection, safe stop, exact policy violations, evidence fabrication, leakage, unauthorized actions, unresolved blocker, and expert acceptance. Report call count, tokens, provider cost, latency and correction time separately. Avoid a single opaque "intelligence score" as the only outcome.

**Hard safety gate:** zero fabricated evaluation numbers, zero unauthorized side effects, zero known holdout/leakage violations for any candidate considered for adoption. A planning gain cannot compensate for those violations.

**Decision thresholds (provisional, adjust before looking at sealed cases):**
- For 8-case pilot, judge per-case disagreements and trace explanations; **no statistical victory claims**.
- For the 30+ case study, look for ≥15 percentage points absolute improvement in expert-accepted solution/planning rate for B or D versus C, with paired uncertainty interval and no safety regressions. Treat a smaller reproducible gain as worth investigation but not automatically worth migration.
- Performance/cost: report latency and cost per expert-accepted task and any tool-failure overhead.

## 7. Diagnostic decision matrix

| Observation | Most likely next investigation | What **not** to conclude |
|---|---|---|
| A frames well; B plans well; C fails | Current path routing, prompts, context, gates, tool schemas, fallback. | "All deterministic validation is bad." |
| A frames well; B and C fail at execution | Tools, permissions, contract translation, observation quality, budgets. | "Need a fine-tuned model." |
| B/C plan well but results fail | Execution, validation integration, dataset/label quality, monitoring. | "Reasoning is necessarily weak." |
| A/B/C all weak on same cases | Insufficient evidence, model capability, scope, task ambiguity and tool affordances. | "Fine-tuning is proven necessary." |
| D outperforms B and C safely | Late-binding framing and staged validation add value. | "Complete rewrite is required." |
| C already matches/exceeds B | Existing agent may be appropriate; improve identified subcomponent, not architecture wholesale. | "The harness is definitely suppressing intelligence." |

## 8. Proposed architecture if evidence supports H1/H2/H4

**Do not delete the current ProblemMap/SolutionPlan/SolutionWorkflow contracts.** Add a pre-binding "ProblemFrame" artifact as an *internal proposal*; only map its relevant parts into current contracts after review.

**Stage 1: Problem exploration (open-ended, read-only).**
- Capture goal, stakeholder, decision, units, temporal contract, available data, constraints, unknowns and confidence.
- Generate candidate framings and hypotheses, including no-model alternatives.
- Preserve raw user intent and source references, not merely a normalized problem type.
- Ask a question only if its answer would materially alter solution choice or safe execution.

**Stage 2: Solution design (adaptive, evaluated).**
- Draft 2–3 diverse, defensible strategies, expected data/evidence needs, evaluation plans and failure modes.
- Choose a strategy provisionally, expose tradeoffs and allow deliberate backtracking after observations.
- Convert proposal into a task graph with preconditions, outputs, assertions and checkpoints.

**Stage 3: Late binding.**
- Reuse `ProblemMap`, `SolutionPlan`, `SolutionWorkflow` only where their contracts express the chosen design.
- If no family matches, preserve an explicit `unmapped/research` state; **do not** force a false classification or silently fall back to an unrelated template.
- Keep taxonomy for analytics, capability matching and execution adapters, not as the prerequisite for thinking.

**Stage 4: Controlled execution.**
- Deterministically enforce data access, user approval, budget, correct schema, provenance, leakage/holdout rules, idempotency and audit logging.
- A rejected proposal returns a precise, machine-readable reason and available alternatives; the agent may revise or stop.

Suggested pipeline:

```text
User brief + evidence
  → Open-ended ProblemFrame + uncertainties + alternatives
  → Grounding/critic checks (without mandatory early family choice)
  → Chosen Solution Strategy + contingent TaskGraph
  → Late-bind into compatible DCLab contracts (or explicit research mode)
  → Deterministic validated tools and execution
  → Observe outcomes → revise plan or stop → evidence-backed deliverable
```

## 9. Tomorrow's implementation sequence (Monday 2026-11-16)

**P0: Reproduce and observe before making changes**

- [ ] Pin repository commit SHA, model/provider, tool schemas, prompts, flags, temperature and budgets.
- [ ] Take one owner-reported failed prompt and inspect actual `case_agent` vs Home/fallback dispatch.
- [ ] Capture turn-by-turn path: framed facts → chosen route/family → planner → tool proposal → gate accept/reject → outcome.
- [ ] Run current offline benchmark sanity checks (below).
- [ ] Obtain blind expert expected behaviors for pilot fixtures.

**P1: Matched pilot**

- [ ] Build minimal-agent arm B using existing read/query/validation interfaces, with unchanged safety controls.
- [ ] Capture C current baseline on all 8 fixtures; no prompt edits during this baseline.
- [ ] Capture A read-only model plans on the same briefs as a reasoning upper-bound diagnostic.
- [ ] Run B with identical executing permissions and comparable budgets.
- [ ] Blind-score outputs and compare both scorecard and trace-level rejection/failure reason.
- [ ] Mark each failure root cause `wrong_route`, `missing_context`, `bad_plan`, `no_tool`, `gate_reject`, `tool_failure`, `model_failure`, `under_specified`, or `unknown`.

**P2: Intervention only after evidence**

- [ ] If H1/H4 supported, add *late-binding* ProblemFrame→contracts behind a feature flag.
- [ ] If H2 supported, make rejection reasons actionable without relaxing safety.
- [ ] If H3 supported, improve retrieval/observations and grounded context.
- [ ] If H5/H6 supported, fix model tier, tool interface or budgets before architectural changes.
- [ ] Re-run the same **development** pilot and then an untouched sealed study.

**Stop conditions:** no silent side effects; stop a test if budget is exceeded or patient/customer data are exposed; do not relax permission, leakage or holdout protections to improve benchmark scores.

### Repository commands to run on the developer machine

```bash
# Existing project preflight. These are NOT the new A/B/C/D comparison.
python -m dclab_rnd.agent_eval.task_planning_v4 verify-sealed
python -m dclab_rnd.agent_eval.task_planning_v4 run-offline --split dev --limit 8

# Optional live benchmark: consumes provider credits; confirm current model configuration and cap first.
python -m dclab_rnd.agent_eval.task_planning_v4 run-live \
  --split dev --limit 8 --workers 1 --cap-usd 1.20 --tag 20261116_intelligence_preflight

# Preserve sealed benchmark for one final run, after design freezes.
```

**Do not claim these commands implement the new harness comparison.** A new adapter/runner is still needed to dispatch the same pilot to A/B/C/(D), standardize artifacts and capture identical trace fields. Existing runners' B0/B1/C labels may describe a *different* experiment.

## 10. Deliverables, ownership, acceptance

**Deliverables**
- `DCLab_Intelligence_Recovery_Plan_2026-11-15.md` (this document)
- `intelligence_pilot_cases.json` (8 case briefs with diagnostic expected behaviors)
- `intelligence_scorecard.csv` (empty blind-review template)
- `score_ablation.py` (local scorecard sanity checker and paired summarizer; **not** a model runner)

**Tomorrow is complete when** (a) real failing path is known for at least one case, (b) 8 cases are reviewed and frozen for a development pilot, (c) B and C complete matched runs or a concrete blocking incompatibility is documented, (d) every failed case has a specific trace-backed failure category, (e) a go/no-go for late-binding intervention is written down.

### Registration note for issue tracker

**Title:** `P0: Diagnose DCLab effective intelligence loss via harness ablation (do not fine-tune yet)`  
**Description:** Existing AI-first CaseAgent coexists with a closed problem-family taxonomy, rules fallback, coordinator gates and bounded context. We need a matched, reproducible comparison of direct-model, minimal-agent, current-DCLab and late-binding-hybrid paths to discover whether model intelligence is suppressed by premature routing, context/tool constraints or planning gates. Preserve execution safety, log traces, grade nontrivial problems blind, and decide about architecture changes only from evidence.  
**Labels:** `architecture`, `agent-evaluation`, `priority:P0`, `research`, `harness`  
**Target working date:** 2026-11-16.  
**No production code or GitHub issue was changed as part of preparing this document.**

## 11. Fine-tuning decision gate

Do **not** start fine-tuning simply because the application feels unintelligent. Consider it only if all are documented:

- Comparable prompt/context/tool paths have been repaired or ruled out as bottlenecks.
- The base model itself repeatedly fails a *well-defined, in-scope* behavior on a held-out set.
- You can supply legally usable, high-quality demonstrations of desired decisions and counterexamples.
- A smaller scoped SFT/LoRA experiment has a measurable success criterion over the strongest prompted baseline.
- Safety, tool-call validity and generalization are checked separately from style/format compliance.

Otherwise the fastest path is likely harness/evaluation repair, not another model.

---

**Evidence boundary:** This plan is based on the owner's reports and an inspection of selected GitHub files. It is not a completed code audit, reproducible experiment, live model run or validated architectural redesign. Keep interpretations provisional until traces and paired results exist.
