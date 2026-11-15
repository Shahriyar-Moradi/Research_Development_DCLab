# DCLab Agent Framework Bake-off: Experiment and Implementation Plan

**Date:** 15 November 2026  
**Status:** Proposed; no framework trials have been run  
**Scope:** Controlled testing of whether advanced agent harnesses add measurable value **after and alongside** repair of current DCLab routing/context/planning issues.

## 1. Hypotheses

- **H0:** A lean agent with the same model and authorized tools matches or outperforms complicated harnesses on DCLab's target cases.
- **H1:** Early problem-type binding, model-visible context loss or hard workflow routing suppresses correct model proposals in the current harness.
- **H2:** A late-bound `ProblemFrame` and dynamically compiled, validated plan improve novel/ambiguous problem solving.
- **H3:** Deep Agents' context offload and delegated expertise improve long-horizon cases *after* the fundamental problem-frame issue is fixed.
- **H4:** Pydantic AI Harness yields comparable or better typed scientific decisions at lower integration/runtime overhead.
- **H5:** An isolated coding worker (OpenHands SDK) improves independently checked artifact quality, not just generated code volume.

These are hypotheses, not findings.

## 2. Comparison arms and permitted inferences

| Arm | Name | Configuration | Primary comparison |
|---|---|---|---|
| A | Model-only | Same underlying model, fixed brief/evidence, no tools | Understanding/planning **only** |
| B | Minimal agent | Same model, essential read/analysis tools, simple loop, no forced early type | Primary agent baseline |
| C | Existing DCLab | Pinned current harness, all current gates and prompts | Product baseline |
| D | DCLab late-binding | C + only the controlled problem-framing/route timing change | Causal architectural ablation |
| E0 | Deep Agents minimal | Same model/tools, skills and subagents off where feasible | Harness overhead and baseline |
| E1 | Deep Agents scoped | E0 + isolated evidence profiling/critic, context offload and controlled skills | Incremental Deep Agents gain |
| F0 | Pydantic AI minimal | Typed model/tool loop using identical inputs and permissions | Lean framework comparator |
| F1 | Pydantic Harness scoped | F0 + selected memory/planning/context capabilities, one at a time | Incremental Pydantic capability gain |
| G* | Coding specialist | Current DCLab coding runner vs OpenHands SDK; optional Codex/Claude | **Execution tasks only** |

Do not interpret A versus G as a fair end-to-end comparison: A has no tools. Prefer matched B/C/D/E/F for planning and B/C/D/E/F for execution once permissions/tool parity is established.

## 3. Phase 0: preflight, version pinning and evidence

- [ ] Freeze repository SHA and model/provider/version for each benchmark cohort.
- [ ] Capture actual effective feature flags; prove which `case_agent`, fallback and coordinator paths execute in failed sessions.
- [ ] Export redacted model-visible messages, offered tool schemas, context sections, first proposals, gate feedback and subsequent revisions.
- [ ] Audit `ProblemMap` route default and verify whether it is actually triggered on the target cases.
- [ ] Record current permissions, training/holdout boundaries, tool budgets, wall-clock caps, temp, token limits and max model steps.
- [ ] Run existing offline baseline and hard safety/regression tests. Do not change production behavior or disclose sealed references.

**Exit:** reproducible, privacy-safe historical traces for at least two prior failures plus frozen benchmark runner configuration.

## 4. Phase 1: dataset and grading rubric

Create eight first-pass development cases and expand to at least 30 independently reviewed cases. Suggested category distribution for the 30-case stage:

| Category | Cases | Critical expected skill |
|---|---:|---|
| Ambiguous business diagnosis / no-ML solution | 7 | Don't force classification; request specific missing evidence |
| Temporal, multitable, graph/forecasting | 6 | Entity/horizon/split/causal time validity |
| Causal and retention intervention decisions | 5 | No invented uplift; viable experiment or policy fallback |
| Scientific model selection / error analysis | 5 | Independent metrics, uncertainty and falsifiable strategy |
| Unfamiliar optimization and non-model tasks | 3 | Out-of-taxonomy reasoning |
| Tool failure, contradictory evidence, recovery | 4 | Diagnose failure, adapt plan, no false success |

Mark at least ten cases as `ML_NOT_REQUIRED` or `ML_PREMATURE` where appropriate; these tags may overlap categories but should be pre-registered. For every case store:

```text
case_id, brief, evidence_manifest, dataset_snapshot_hash,
allowed_tool_set, forbidden_access, expected_behavior,
independent_reference_review, rubric_version, split=dev|sealed
```

Scoring dimensions (0–4): problem framing, scientific validity, alternative options, plan structure, evidence discipline, failure recovery. Execution: actual independently verified completion, invalid/unauthorized tool calls, reproducing results, holdout/leakage incidents, latency and spend. Use blind expert review and independently executable assertions where possible.

## 5. Phase 2: framework adapters with no authority migration

Create a uniform adapter returning:

```text
AgentRunResult {
  arm_id, case_id, run_id, provider_model, version_hashes,
  problem_frame?, solution_candidates[], plan?,
  steps[], tool_events[], artifacts[], verifier_results[],
  tokens, cost, latency, stop_reason, observed_errors[]
}
```

Implementation rules:

1. All agents invoke **DCLab-owned** capability endpoints; never receive raw database credentials.
2. Refuse requests for sealed data, unauthorized network/file access, unsupported user-scoped operations.
3. Reuse exactly the same dataset snapshot, typed tool schemas, redaction rule and permitted input evidence.
4. Convert provider/framework event streams to a common `trace.jsonl` schema; retain native traces as linked diagnostics.
5. Allow no model-generated plan to execute before `PlanValidator` and explicit approval policy have accepted it.
6. Run all code in a separate sandbox with network restrictions and an immutable training dataset mount.
7. Apply total cost/step/time ceilings equally; log exceptions from library-specific mechanics.
8. Preserve DCLab's existing checkpoint and source-of-truth IDs, even if experimental frameworks have their own memory/state.

**Exit:** every adapter passes schema, authorization, cost-limit, repeated-call, cancellation, prompt-injection and identical-input preflight tests.

## 6. Phase 3: staged comparisons

**Comparison P: plan-only.** Run A/B/C/D/E0/F0 against frozen briefs and available evidence, with execution suppressed. Grade accepted `ProblemFrame`, solution options and plan blind to arm. Use multiple repeats where the model is non-deterministic.

**Comparison X: execution.** B/C/D/E0/F0 execute the same safe tasks using the same DCLab tool contracts. Verify final artifacts independently. Do not let differences in shell access stand in for reasoning quality.

**Comparison M: incremental agent capabilities.** When E0 and F0 baselines are stable, add exactly one of context offloading, specialist critique, subagent delegation, skills retrieval, or memory. Compare E1/F1 against their own E0/F0 controls and the strongest baseline.

**Comparison G: code execution.** On a subset of code-authoring cases, compare current native coding worker vs OpenHands SDK under identical sandbox images, tool permissions and tests. Report working artifacts and verified scientific outputs, not number of files changed.

**Comparison S: model tier.** If most arms fail identically on framing while evidence parity and tools are sound, hold harness constant and test model capability before concluding the whole architecture is bad.

### Trace diff checklist

- Did both runs actually see the same crucial evidence and unknowns?
- Was a problem type chosen before the actual business problem was framed?
- Did the model propose a good alternative that the harness discarded or failed to expose?
- Was a rejection caused by a safety/scientific invariant or by an unnecessary output-shape rule?
- Did context compaction omit a crucial source pointer?
- Did the agent recover from a failed tool call or repeatedly retry bad arguments?
- Did it claim success despite failing independent verifier checks?

## 7. Release gates and decision points

**Hard gates:** no critical authorized-data/holdout leakage found in the declared safety test suite; no unauthorized writes; no unreproducible claimed success; model prompts and outputs redacted as required; reproducible artifact fingerprints; protected current baseline workflows not regressed.

**Quality gates:** pre-register a practical +10 percentage-point acceptance/completion target against current DCLab where plausible; show paired-case differences and uncertainty rather than declaring a statistically established winner from 30 cases. Compare also with the minimal agent, not only with the current weak harness. Test long-horizon and out-of-taxonomy cases separately.

**Cost gates:** report accepted outcomes per dollar, median/p95 wall-clock latency, model/tool steps and human interventions. If a richer harness only improves fluent commentary, reject it. If it produces better validated outcomes at a moderate cost increase, decide explicitly whether the marginal benefit justifies it.

**Decision tree:**

```text
B > C and D ~= B       -> fix early binding; keep core LangGraph
B > C and D < B        -> investigate context/gates/tool contracts further
C >= B                 -> keep and simplify; do not presume a harness problem
E1 > E0 and E1 > B/D   -> add only winning Deep Agents capability(s)
F1 > F0 and F1 > B/D   -> consider typed Pydantic agent boundary
G > native worker      -> delegate approved coding tasks to OpenHands
All weak              -> investigate model, data evidence, rubric and tools
Safety gate fails     -> do not deploy regardless of average quality
```

## 8. Integration sequence following proven gains

- [ ] **P0** Diagnose and repair current DCLab before adopting a framework as a new core dependency.
- [ ] **P1** Implement `ProblemFrame -> SolutionCandidates -> late-bound PlanCompiler` with open `not_ml` option; preserve controlled execution.
- [ ] **P2** Use Deep Agents for a single read-only investigation/critic boundary if E1 wins; restrict tools and subagents.
- [ ] **P3** Introduce an isolated OpenHands coding worker only if G wins and the sandbox is verified safe.
- [ ] **P4** Store verified experiments in evidence-linked memory; use DSPy to optimize narrowed framing/planning modules only after reliable metrics exist.
- [ ] **P5** Reassess external Model/Problem/Project Agent product separately; reuse the same typed tools, identity and governance without sharing a global mutable agent state.

## 9. Suggested repository structure (proposal, adapt to actual HEAD)

```text
docs/architecture/agent-framework-evaluation/
  FRAMEWORK_RESEARCH.md
  EXPERIMENT_PROTOCOL.md
experiments/harness_framework_bakeoff/
  cases_dev.json
  schemas.py
  adapters/
    model_only.py
    minimal_agent.py
    dclab_current.py
    dclab_late_binding.py
    deepagents_adapter.py
    pydantic_adapter.py
    coding_worker_adapter.py
  runners/
    planning.py
    execution.py
  grading/
    scientific_rubric.py
    independent_verifiers.py
  traces/
  results/
  DECISION_RECORD.md
```

These are **suggested paths**, not verified existing modules. Follow current DCLab evaluator conventions when implementing rather than creating a second testing platform.

## 10. Deliverables / definition of done

1. Runtime route and first-divergence audit with concrete trace evidence.
2. Frozen 8-case dev pilot and 30-case wider benchmark with independently reviewed scoring.
3. Reproducible A/B/C/D/E0/E1/F0/F1 test protocol; optional coding G sub-study.
4. Paired scorecards with confidence/uncertainty, cost, latency and hard safety findings.
5. An architecture decision record choosing **Keep / Repair / Adopt Specific Capability / Insufficient Evidence**.
6. One feature-flagged and reversible integration only if the declared gates pass.

**No implementations, calls to provider models, production writes, or benchmark measurements have been performed in preparing this plan.**

## Primary source links

- https://docs.langchain.com/oss/python/deepagents/overview
- https://github.com/langchain-ai/docs/blob/main/src/oss/langgraph/overview.mdx
- https://github.com/pydantic/pydantic-ai/blob/main/docs/harness/index.md
- https://github.com/OpenHands/software-agent-sdk
- https://developers.openai.com/api/docs/guides/agents/sdk
- https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/optimization/optimizers.md
- https://langfuse.com/docs/evaluation/overview
