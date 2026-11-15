# Implementation Subplan 3C: Execution readiness and problem-level experience
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 3: Problem Agent Template and Specialized Problem Harness
**Phase exit gate:** G3
**Mission:** Give each independently versioned business/technical problem an agent that understands its objectives, complete solution history, evidence, context and operational life.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P3-08` | Connect problem decomposition, solution planning and task readiness | ProblemExecutionPlan and task readiness reports. | P3-07 |
| `P3-09` | Enable Model Agent discovery and problem-level delegation proposals | Problem-to-model capability bindings and typed request prototype. | P3-08, P2-10 |
| `P3-10` | Deliver Problem Agent screen and sealed G3 tests | Problem Agent UI, release candidate and G3 report. | P3-09 |

## P3-08: Connect problem decomposition, solution planning and task readiness
**Desired capability:** Integrate existing DCLab thought artifacts without copying the execution engine.
**Prerequisites:** P3-07
**Reuse/integration points:** `dclab_core/task_graph.py`, `task_execution_plan.py`, `task_validation.py`.

### Engineering tasks
- [ ] **01** Link ProblemMap and SolutionPlan IDs to Problem Agent; version the active decomposition and solution plan.
- [ ] **02** Translate adopted solution into typed TaskGraph proposals with capability bindings and evidence requirements.
- [ ] **03** Validate graph structure, dependencies, approvals and available tools before execution planning.
- [ ] **04** Mark steps proposed/ready/blocked/executed based on independent host receipts.
- [ ] **05** Stop if data readiness, label timing, rights or required approvals are missing.

### Outputs and checks
- Artifacts: ProblemExecutionPlan and task readiness reports.
- Acceptance: The agent cannot mark task complete from chat claims; blocked prerequisites show a specific remedy.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-09: Enable Model Agent discovery and problem-level delegation proposals
**Desired capability:** Make specialization visible while keeping authority with each harness.
**Prerequisites:** P3-08, P2-10
**Reuse/integration points:** Shared registry, Model Agent tool APIs, task contracts.

### Engineering tasks
- [ ] **01** Resolve linked Model Agent IDs and allowed capabilities by model/solution association.
- [ ] **02** Build typed task request for model metrics, diagnostics, feature analysis or challenger comparison.
- [ ] **03** Check permission, version, expected output and budget before submission to central collaboration layer.
- [ ] **04** Store correlation IDs and evidence handles; avoid free-text agent messages as executable commands.
- [ ] **05** In Phase 3, support a read-only dispatch pilot; Phase 5 adds robust full orchestration.

### Outputs and checks
- Artifacts: Problem-to-model capability bindings and typed request prototype.
- Acceptance: An unrelated model or wrong version cannot be delegated to without explicit grant; responses keep provenance.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-10: Deliver Problem Agent screen and sealed G3 tests
**Desired capability:** Prove persistent problem and solution intelligence.
**Prerequisites:** P3-09
**Reuse/integration points:** External UI, sealed benchmark suite and current Model Agent.

### Engineering tasks
- [ ] **01** Create Overview, Goals, Hypotheses, Solutions, Models, Data, Decisions, Conversations and Work tabs.
- [ ] **02** Support versioned solution comparison, decision timeline and evidence-linked chat answers.
- [ ] **03** Run multi-turn evaluation of correction, historical rationale, contradictory KPI and missing evidence.
- [ ] **04** Validate read-only/delegation permissions and isolation across two problems in same project.
- [ ] **05** Publish G3 defect review and freeze stable Problem Harness contract.

### Outputs and checks
- Artifacts: Problem Agent UI, release candidate and G3 report.
- Acceptance: One churn problem successfully preserves solution and conversation revisions; independent second problem uses same template and harness.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
