# DCLab | Hierarchical Planning V2: Additive Capability Pack

**Date:** 2026-11-05  
**Status:** Design specification; not an implementation report  
**Dependency:** V1 must pass its predeclared acceptance gates before V2 implementation begins.  
**Design rule:** V2 adds capability to V1; do not rewrite V1 TaskGraph, TaskExecutionPlan, execution, verification, permission guards, six intent choices, or persistent event infrastructure.

## 0. Non-negotiable separation

- **V1:** deterministic and optionally model-assisted decomposition and task plans, sequential guarded execution, durable evidence/verification/recovery, V1 baselines and UI.
- **V2:** dynamic specialist-role collaboration, structured critique of solutions, constrained concurrent experiments, evidence-backed experience memory, optional adapter-backed agent roles, and multi-agent evaluation.
- **Agentic ML Data/Model Program:** independently collect/curate data and train a new base+adapter candidate. Only connect to V2 through existing model gateway/registry and *after* the relevant gates. Data collection can run while V1 is being finished.
- Avoid hidden chain-of-thought capture. Save concise, user-visible decision summaries and evidence from actual tools instead.

## 1. Entry checklist (must be satisfied before V2)

- The current V1 branch has a release report, commit hash, schema version and baseline fixture hashes.
- TaskGraph and TaskExecutionPlan contracts remain validated and stable.
- V1 sequential resume/recovery is demonstrated without duplicate writes or premature holdout reads.
- V1 scientific checks include leakage, wrong metric, unsupported-task and incomplete-evidence cases.
- Existing route choices `Model / Plan / Report / Analyze / Prepare / Learn` remain intact.
- No critical V1 gate failures, and all V1 regression tests are frozen for V2 comparisons.
- If any check fails, repair V1 in V1; V2 is not a workaround.

## 2. Capability packages (implement and accept independently)

### HP2-01 | Dynamic specialist collaboration

**Objective:** Pick a minimal role set for a problem and coordinate their contributions on a validated V1 task, not create a new orchestrator.

**Design:** Add an `AgentRoleContract` mapping task type/complexity to bounded role skills. Suggested logical roles: Data Examiner, Experiment Designer, ML Implementer, Scientific Reviewer. The same model may perform multiple roles; separate roles need not mean separate paid models.

**Guardrails:** V1 scheduler/dispatcher remains authority. All agents see only scoped context and registry tools. The coordinator cannot grant permissions, alter policies, or claim completion. Cap agents, rounds, tokens and tool calls.

**Acceptance:** Correct role selection on simple/ambiguous/multi-table cases, appropriate single-agent fallback, no routing regression, no unauthorized tool access, lower/equal overhead for simple cases.

### HP2-02 | Structured proposals and evidence-based critique

**Objective:** Enable collaboration where disagreement is settled by a reproducible experiment or deterministic validator rather than persuasion.

**Protocol:**
1. `proposal`: hypothesis, dataset facts, planned metric, expected artifact, expected cost.
2. `critique`: precise fault claim, severity, evidence request, proposed discriminating test.
3. `experiment_request`: reference to a V1-valid leaf task and scoped permitted tools.
4. `verification`: observed metrics, provenance, decision, unresolved limitations.
5. `decision`: accepted/rejected/blocked with evidence references.

**Guardrails:** No unlimited chat loops. At most 2 critique rounds by default. Human intervention when intent/safety/science ambiguity persists. Reviewers cannot override deterministic fails. No hidden reasoning logs.

**Acceptance:** Reasoned disagreement on leakage, class imbalance, split methodology and poor metrics; zero false-completion acceptance; no extra holdout access.

### HP2-03 | Constrained parallel experimental branches

**Objective:** Run independent candidate experiments when the predicted information gain justifies the compute cost.

**Design:**
- Read-only preparation may be shared; branch-specific writes are isolated by worktree/container/artifact namespace.
- Only dependency-ready V1 leaf tasks can fan out. Shared-state mutations and final/holdout work remain serialized and governed by V1.
- Race/conflict detection, per-branch budget, idempotency/reconciliation before retries, and deterministic merge policy.
- Compare all candidates using the same untouched evaluation contract.
- Select a winner only with an independently verified artifact and outcome.

**Acceptance:** Run at least two independent model candidates with isolated artifacts; crash and resume without double-write; preserve siblings on one branch failure; zero cross-branch contamination.

### HP2-04 | Evidence-backed experiment memory

**Objective:** Retain usable lessons without treating unverified agent text as truth.

**Memory entry:** case signature; proposal; hypothesis; tool/code/dataset fingerprints; test conditions; metric and statistical uncertainty; verifier outcome; applicability scope; failure modes; lesson status `candidate/verified/refuted/stale`.

**Policies:** Save signed-off lessons only after verifier approval; source-data changes invalidate applicability; recency/confidence scoring affects retrieval but does not bypass registry. Never retrieve benchmark answer keys.

**Acceptance:** A previously proven failure mode can be avoided on a new but compatible case; a conflicting lesson is surfaced as uncertain; revoked lesson stops being used; retrieval never exposes held-out labels.

### HP2-05 | Open-weight candidate model integration

**Objective:** Run the user’s fine-tuned Agentic ML model as a gateway-selectable engine for *bounded roles* first.

**Design:** Add no training code to V2 itself. Import a pinned adapter manifest from the independent Data/Model Program. Log base+adapter+tokenizer+template hashes and cost/latency. Use model-level fallback only for unresolved or high-complexity roles. Keep all tool/safety/verification contracts outside the LLM.

**Acceptance:** One-click rollback and shadow execution; identical function schemas to paid-model baseline; no novel authority granted to the model; bounded fallback cost. Full default promotion requires model-program gates.

### HP2-06 | Ablations, benchmark and release gate

**Arms:** `A: fixed workflow`, `B: accepted V1`, `C: V2 collaboration without parallel`, `D: V2 + parallel`, `E: V2 + experience memory`, `F: V2 + own fine-tuned adapter`.

**Evaluation:** matched cases, frozen tool registry, matched budgets/compute/harness, repeated stochastic seeds, separate safety/science/e2e scores, provenance and cost. Compare per capability, not only final total. No train/dev cases in sealed external benchmarks.

**Metrics:** validated completion, scientific correctness, tool choice/args validity, false completions, leakage/holdout incidents, recoverability, work duplication, wall-clock latency, API+GPU cost, quality per dollar.

**Release rule:** 0 observed critical safety violations in declared test suite, no material V1 regression, proven quality/cost tradeoff at preset uncertainty and sample size. Otherwise `NOT PROMOTED` or `INSUFFICIENT EVIDENCE`.

## 3. Project artifacts (adapt paths after HEAD inspection)

```text
planning/v2/
  README.md
  hp2_scope_and_entry_gate.md
  hp2_role_contracts.md
  hp2_collaboration_protocol.md
  hp2_parallel_experiment_policy.md
  hp2_memory_evidence_policy.md
  hp2_model_gateway_integration.md
  hp2_eval_protocol.md
  hp2_release_checklist.md
  tests_and_results/
```

Do not create duplicated runtime or graph layers just to match these file names. Keep docs/versioned contracts close to the repo’s existing architecture.

## 4. Data requirements for collaboration (collect later, not via unverified chat)

A successful collaborative episode should expose: objective; confirmed context; V1 task/plan IDs; each role’s declared claim; proposed code/artifact; critique and evidence; actually executed tool calls; test outputs; final independent verification; human feedback; rejection reason; budget/timing. Record provenance, license and the exact model/template versions. Use post-hoc verified evidence summaries, not proprietary hidden chain of thought.

These data become useful as V2-specific training examples **only after** the parallel Agentic ML Dataset Program’s license/privacy/quality gates. This avoids circularly treating a new, untested V2 behavior as gold data.

## 5. Suggested V2-only implementation sequence after V1 signoff

1. Freeze V1 baseline; HP2-01 role routing, with single-agent fallback.
2. HP2-02 proposal/critique protocols, no tool execution added in debate.
3. HP2-03 scoped parallel experiments and conflict/retry validation.
4. HP2-04 experience memory with evaluator-backed provenance.
5. HP2-05 integrate an approved adapter from independent model program.
6. HP2-06 compare A–F; keep disabled by default until pass.

Do **not** assume all six packages fit into the same month as data/model training. V2 begins on the date V1 is accepted, not on a calendar promise.

## 6. Codex handoff prompt

> Inspect the current DCLab repository at HEAD and the full Hierarchical Planning V1 implementation/report before editing. Treat this V2 spec as an additive, sequential follow-up to an already accepted V1. First produce an architecture/contract gap audit and a proposed HP2-01 implementation diff. Do not create a second scheduler, change the six Home routes, modify V1 safety rules, read sealed benchmarks, spend API/GPU budget, or enable production behavior. List all current contracts, exact files to change, test cases, regression risks, and V1 acceptance evidence. If V1 acceptance is missing or any relevant gates fail, report BLOCKED rather than executing V2. Implement only the first reviewed package after user approval, with honest test output and rollback instructions.
