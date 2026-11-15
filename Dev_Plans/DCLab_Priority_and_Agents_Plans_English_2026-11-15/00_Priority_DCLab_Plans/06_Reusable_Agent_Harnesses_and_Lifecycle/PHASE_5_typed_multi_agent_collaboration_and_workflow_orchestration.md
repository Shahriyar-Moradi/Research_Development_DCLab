# Phase 5: Typed Multi-Agent Collaboration and Workflow Orchestration
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Coordinate three specialized harnesses with structured tasks, explicit dependencies, bounded execution and evidence-bearing results.
**Entry criteria:** G4 Project Agent, G3 Problem Agent and G2 Model Agent contracts are available and stable.
**Exit gate G5:** Project -> Problem -> Model delegation completes a real churn investigation with bounded calls, policy enforcement, durable jobs and cited evidence.
**Mandatory live demonstration:** Ask Project Agent to investigate declining churn-model recall and recommend a solution; inspect decomposed tasks, model diagnostics, business KPI evidence and decision proposal.
**Important boundary:** Typed, deterministic task routing is the default. Do not implement unrestricted peer-to-peer agent conversations, recursive tool loops or multiple competing workflow engines.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 5A | Delegation protocols and capability routing | P5-01, P5-02 | Implement in listed order unless explicitly parallel-safe |
| 5B | Planning, execution and failure recovery | P5-03, P5-04, P5-05, P5-06 | Implement in listed order unless explicitly parallel-safe |
| 5C | Evidence synthesis and real workflow demonstration | P5-07, P5-08, P5-09 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 5A: Delegation protocols and capability routing
**Goal:** Complete P5-01, P5-02 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P5-01 - Define typed Agent-to-Agent protocol
**Outcome and scope.** Replace fragile conversational delegation with structured contracts.
**Preconditions/dependencies.** P4-10
**Existing integration to inspect.** External task contracts aligned with DCLab TaskGraph concepts.

**Implementation steps (ordered):
1. Define TaskRequest containing source/target agent and entity IDs, capability, input evidence refs, constraints, budget, principal, correlation ID and expected output schema.
2. Define TaskResult containing status, typed findings, artifact refs, execution receipt, cost, warning codes and provenance.
3. Specify schema and version negotiation, idempotency, cancellation and authorization context propagation.
4. Separate a mere analysis proposal from an execution request with side effects.
5. Validate forged or stale task requests and reject unknown schema versions.

**Artifacts and interfaces:** AgentTaskRequest/Result schemas, versioned protocol and fixtures.

**Acceptance tests and evidence:** Unknown target, incompatible version, missing approval or changed arguments fail deterministically before execution.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P5-02 - Implement deterministic capability discovery and routing
**Outcome and scope.** Select agents from declared capabilities and relations, not invented capabilities.
**Preconditions/dependencies.** P5-01
**Existing integration to inspect.** Template registry, relationship graph and guarded tool registry.

**Implementation steps (ordered):
1. Index available Model/Problem/Project instances with entity bindings and allowed capability IDs.
2. Define a task router that resolves exact target using relation graph, task type and user scope.
3. Permit LLM semantic interpretation only when deterministic resolution leaves genuine ambiguity.
4. Return conflict or multiple-choice state rather than silently picking a model for ambiguous requests.
5. Log selected route and rejected candidate reasons for evaluation.

**Artifacts and interfaces:** AgentRouter, capability index and routing decisions.

**Acceptance tests and evidence:** Wrong-agent routing cases are caught; authorization is checked again by the target harness.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 5B: Planning, execution and failure recovery
**Goal:** Complete P5-03, P5-04, P5-05, P5-06 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P5-03 - Implement task-level decomposition and typed plans
**Outcome and scope.** Turn broad project requests into checkable, authorized tasks.
**Preconditions/dependencies.** P5-02
**Existing integration to inspect.** `dclab_core/task_graph.py`, `task_execution_plan.py`, `task_validation.py` patterns.

**Implementation steps (ordered):
1. Translate user goal into objective, required evidence, nodes, dependencies, owner agent and estimated risk/cost.
2. Represent immutable action proposal, argument references, expected inputs and stop conditions.
3. Validate graph acyclicity, dependency readiness, capability membership and evidence availability.
4. Use business planning artifacts for rationale while leaving execution to task engine.
5. Generate explicit approval checkpoints and alternatives when resources are missing.

**Artifacts and interfaces:** TaskPlanner adapter, TaskGraph, TaskExecutionPlan and validation report.

**Acceptance tests and evidence:** Plans cannot reference nonexistent artifacts or execute unapproved literal SQL/URLs; absent evidence stops the plan.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P5-04 - Build durable orchestration and dependency scheduler
**Outcome and scope.** Execute validated plans without duplicate side effects.
**Preconditions/dependencies.** P5-03
**Existing integration to inspect.** Shared jobs, checkpoints and approval guard.

**Implementation steps (ordered):
1. Implement orchestration state: proposed, validated, ready, queued, running, waiting_approval, failed, completed.
2. Schedule DAG nodes with dependency validation and per-agent concurrent-run limits.
3. Use idempotent step receipts and durable checkpoints before/after effectful calls.
4. Support safe parallel read-only tasks but serialize conflicting writes and shared-model changes.
5. Expose process tree and status for dashboard with cursor-based progress updates.

**Artifacts and interfaces:** WorkflowScheduler, orchestration state machine and run APIs.

**Acceptance tests and evidence:** Crash/restart does not rerun a completed retrain; denied step stops dependent tasks without false completion.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P5-05 - Implement recovery, timeout and circuit breakers
**Outcome and scope.** Prevent hard failures and runaway loops from turning into invisible bad states.
**Preconditions/dependencies.** P5-04
**Existing integration to inspect.** Shared worker and tool adapter failure interfaces.

**Implementation steps (ordered):
1. Classify transient provider error, permanent validation error, permission denial and user-cancelled task.
2. Configure bounded retries, exponential backoff, timeout, job heartbeat and provider circuit breaker.
3. Record explicit partial-success results and cleanup/compensation requirements for effectful operations.
4. Define human intervention queues for unrecoverable or ambiguous side effects.
5. Run fault-injection tests for missing tool response, duplicate event, job crash and revoked permission.

**Artifacts and interfaces:** RecoveryPolicy, circuit breaker and fault-injection report.

**Acceptance tests and evidence:** No automatic retry of a potentially non-idempotent write without verified receipt and safe retry key.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P5-06 - Implement plan budget, delegation depth and approval limits
**Outcome and scope.** Keep multi-agent architecture measurable and bounded.
**Preconditions/dependencies.** P5-04
**Existing integration to inspect.** Shared execution kernel and observability.

**Implementation steps (ordered):
1. Set caps on LLM turns, token costs, workflow wall-clock, tool calls, retry count and delegation depth.
2. Scope budgets by run, problem, project, workspace and provider.
3. Make planned cost a pre-flight check with refusal or user escalation when budgets are exceeded.
4. Track time/cost per completed task and unnecessary tool calls.
5. Implement tests to ensure delegated child work counts against parent budget.

**Artifacts and interfaces:** BudgetLedger, run limits and efficiency telemetry.

**Acceptance tests and evidence:** Recursive ping-pong loops terminate; child agents cannot bypass parent budget with a fresh run ID.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 5C: Evidence synthesis and real workflow demonstration
**Goal:** Complete P5-07, P5-08, P5-09 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P5-07 - Implement evidence merging, contradiction detection and review
**Outcome and scope.** Produce reliable final answers from multiple specialized agents.
**Preconditions/dependencies.** P5-03
**Existing integration to inspect.** All harness answer contracts and evidence resolver.

**Implementation steps (ordered):
1. Merge TaskResults by source, entity, version, measurement unit, window, cohort and freshness.
2. Detect contradictory metrics from incompatible thresholds, datasets or label timing.
3. Distinguish observed facts from hypotheses, actions and policy decisions in final answer.
4. Apply deterministic completeness checks; add targeted critic review only when needed and measurable.
5. Publish evidence bundle mapping every quantitative assertion to a canonical result.

**Artifacts and interfaces:** EvidenceMerger, consistency report and final response structure.

**Acceptance tests and evidence:** Conflicting metric evidence is flagged; the agent does not average incompatible recall scores into a fake answer.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P5-08 - Build complete churn investigation workflow
**Outcome and scope.** Demonstrate the intended agentic ML product end to end.
**Preconditions/dependencies.** P5-07
**Existing integration to inspect.** All three harnesses, data binding, evaluation, workflow scheduler.

**Implementation steps (ordered):
1. Trigger investigation from Project Agent for declining recall in a named period.
2. Delegate problem-scoped question to Churn Problem Agent and precise model diagnostics to Churn Model Agent.
3. Query approved labeled performance and business churn KPI windows independently.
4. Produce hypotheses, data gaps, candidate experiments, chosen next steps and any approval request.
5. Persist investigation as a resumable project/prob/ model-linked record with clear final verdict.

**Artifacts and interfaces:** Vertical-slice demo, walkthrough fixture and trace bundle.

**Acceptance tests and evidence:** No fabricated scores, no unauthorized retraining, and a complete evidence-linked task tree across three agents.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P5-09 - Run sealed collaboration benchmark and performance ablations
**Outcome and scope.** Prove that three agents actually help versus simpler alternatives.
**Preconditions/dependencies.** P5-08
**Existing integration to inspect.** Existing DCLab sealed benchmark practices; independent external evaluations.

**Implementation steps (ordered):
1. Compare direct deterministic tool workflow, single general agent and three-template orchestration on same tasks.
2. Measure success, tool/route accuracy, policy violations, recovery, human effort, latency and cost per task.
3. Study alternatives including no project agent, no memory retrieval, deterministic-only routing and fewer delegation steps.
4. Identify failure clusters and adjust routing or tool boundaries rather than just increasing prompt size.
5. Gate only if collaboration improves a named product capability at acceptable cost.

**Artifacts and interfaces:** G5 collaboration gate and comparative benchmark report.

**Acceptance tests and evidence:** No unjustified agent proliferation: more complex architecture must have documented net benefit.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 9 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Ask Project Agent to investigate declining churn-model recall and recommend a solution; inspect decomposed tasks, model diagnostics, business KPI evidence and decision proposal.
- [ ] G5 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G5 provides the stable interfaces, immutable evidence and regression tests required by Phase 6.

---

## Download or open each implementation subplan separately

- [Implementation Subplan 5A: Delegation protocols and capability routing](../plans/PLAN_5A_delegation_protocols_and_capability_routing.md)
- [Implementation Subplan 5B: Planning, execution and failure recovery](../plans/PLAN_5B_planning_execution_and_failure_recovery.md)
- [Implementation Subplan 5C: Evidence synthesis and real workflow demonstration](../plans/PLAN_5C_evidence_synthesis_and_real_workflow_demonstration.md)

These subplans are independent implementation documents containing work packages, detailed engineering tasks, outputs, tests, and acceptance criteria.
