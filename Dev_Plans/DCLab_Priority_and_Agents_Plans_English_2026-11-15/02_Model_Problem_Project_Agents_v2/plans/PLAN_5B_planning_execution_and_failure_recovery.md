# Implementation Subplan 5B: Planning, execution and failure recovery
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 5: Typed Multi-Agent Collaboration and Workflow Orchestration
**Phase exit gate:** G5
**Mission:** Coordinate three specialized harnesses with structured tasks, explicit dependencies, bounded execution and evidence-bearing results.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P5-03` | Implement task-level decomposition and typed plans | TaskPlanner adapter, TaskGraph, TaskExecutionPlan and validation report. | P5-02 |
| `P5-04` | Build durable orchestration and dependency scheduler | WorkflowScheduler, orchestration state machine and run APIs. | P5-03 |
| `P5-05` | Implement recovery, timeout and circuit breakers | RecoveryPolicy, circuit breaker and fault-injection report. | P5-04 |
| `P5-06` | Implement plan budget, delegation depth and approval limits | BudgetLedger, run limits and efficiency telemetry. | P5-04 |

## P5-03: Implement task-level decomposition and typed plans
**Desired capability:** Turn broad project requests into checkable, authorized tasks.
**Prerequisites:** P5-02
**Reuse/integration points:** `dclab_core/task_graph.py`, `task_execution_plan.py`, `task_validation.py` patterns.

### Engineering tasks
- [ ] **01** Translate user goal into objective, required evidence, nodes, dependencies, owner agent and estimated risk/cost.
- [ ] **02** Represent immutable action proposal, argument references, expected inputs and stop conditions.
- [ ] **03** Validate graph acyclicity, dependency readiness, capability membership and evidence availability.
- [ ] **04** Use business planning artifacts for rationale while leaving execution to task engine.
- [ ] **05** Generate explicit approval checkpoints and alternatives when resources are missing.

### Outputs and checks
- Artifacts: TaskPlanner adapter, TaskGraph, TaskExecutionPlan and validation report.
- Acceptance: Plans cannot reference nonexistent artifacts or execute unapproved literal SQL/URLs; absent evidence stops the plan.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P5-04: Build durable orchestration and dependency scheduler
**Desired capability:** Execute validated plans without duplicate side effects.
**Prerequisites:** P5-03
**Reuse/integration points:** Shared jobs, checkpoints and approval guard.

### Engineering tasks
- [ ] **01** Implement orchestration state: proposed, validated, ready, queued, running, waiting_approval, failed, completed.
- [ ] **02** Schedule DAG nodes with dependency validation and per-agent concurrent-run limits.
- [ ] **03** Use idempotent step receipts and durable checkpoints before/after effectful calls.
- [ ] **04** Support safe parallel read-only tasks but serialize conflicting writes and shared-model changes.
- [ ] **05** Expose process tree and status for dashboard with cursor-based progress updates.

### Outputs and checks
- Artifacts: WorkflowScheduler, orchestration state machine and run APIs.
- Acceptance: Crash/restart does not rerun a completed retrain; denied step stops dependent tasks without false completion.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P5-05: Implement recovery, timeout and circuit breakers
**Desired capability:** Prevent hard failures and runaway loops from turning into invisible bad states.
**Prerequisites:** P5-04
**Reuse/integration points:** Shared worker and tool adapter failure interfaces.

### Engineering tasks
- [ ] **01** Classify transient provider error, permanent validation error, permission denial and user-cancelled task.
- [ ] **02** Configure bounded retries, exponential backoff, timeout, job heartbeat and provider circuit breaker.
- [ ] **03** Record explicit partial-success results and cleanup/compensation requirements for effectful operations.
- [ ] **04** Define human intervention queues for unrecoverable or ambiguous side effects.
- [ ] **05** Run fault-injection tests for missing tool response, duplicate event, job crash and revoked permission.

### Outputs and checks
- Artifacts: RecoveryPolicy, circuit breaker and fault-injection report.
- Acceptance: No automatic retry of a potentially non-idempotent write without verified receipt and safe retry key.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P5-06: Implement plan budget, delegation depth and approval limits
**Desired capability:** Keep multi-agent architecture measurable and bounded.
**Prerequisites:** P5-04
**Reuse/integration points:** Shared execution kernel and observability.

### Engineering tasks
- [ ] **01** Set caps on LLM turns, token costs, workflow wall-clock, tool calls, retry count and delegation depth.
- [ ] **02** Scope budgets by run, problem, project, workspace and provider.
- [ ] **03** Make planned cost a pre-flight check with refusal or user escalation when budgets are exceeded.
- [ ] **04** Track time/cost per completed task and unnecessary tool calls.
- [ ] **05** Implement tests to ensure delegated child work counts against parent budget.

### Outputs and checks
- Artifacts: BudgetLedger, run limits and efficiency telemetry.
- Acceptance: Recursive ping-pong loops terminate; child agents cannot bypass parent budget with a fresh run ID.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
