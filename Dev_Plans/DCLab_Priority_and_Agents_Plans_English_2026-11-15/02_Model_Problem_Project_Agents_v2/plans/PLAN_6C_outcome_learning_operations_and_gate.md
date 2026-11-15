# Implementation Subplan 6C: Outcome learning, operations and gate
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 6: Event-Driven Monitoring, Investigation and Continuous Lifecycle
**Phase exit gate:** G6
**Mission:** Run proactive model/problem/project operations through governed events and evidence-based improvement loops.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P6-07` | Implement outcome feedback and decision learning | OutcomeLedger, KPI-linked effectiveness report. | P6-06 |
| `P6-08` | Implement operations dashboards, alert controls and incident runbooks | Monitoring UI, event timeline, incident runbook. | P6-07 |
| `P6-09` | Pass proactive lifecycle gate with fault injection | G6 event lifecycle acceptance and remediation report. | P6-08 |

## P6-07: Implement outcome feedback and decision learning
**Desired capability:** Close the loop between proposed actions and real-world effect.
**Prerequisites:** P6-06
**Reuse/integration points:** Problem decisions, Project milestones, KPI registry.

### Engineering tasks
- [ ] **01** Define OutcomeObservation with intervention, owner, reference population, before/after window and measurement source.
- [ ] **02** Link outcomes to decision, solution revision, model version, investigation and originating hypothesis.
- [ ] **03** Separate randomized or quasi-experimental outcome evidence from unadjusted before/after correlation.
- [ ] **04** Use outcome feedback to update recommendations and case records without silently modifying training data.
- [ ] **05** Track unresolved actions and measured time to detection/response/recovery.

### Outputs and checks
- Artifacts: OutcomeLedger, KPI-linked effectiveness report.
- Acceptance: Agent never asserts a retention lift from model recall alone; decisions remain auditable after supersession.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P6-08: Implement operations dashboards, alert controls and incident runbooks
**Desired capability:** Give humans visibility and intervention when automation is uncertain.
**Prerequisites:** P6-07
**Reuse/integration points:** External UI and event/investigation APIs.

### Engineering tasks
- [ ] **01** Create model health, problem outcome and project portfolio alert screens.
- [ ] **02** Show source health, event severity, suppression rules, investigation tasks and related approval requests.
- [ ] **03** Implement acknowledge, assign, mute and close with actor-bound audit records.
- [ ] **04** Write incident runbooks for connector outage, retrain failure, harmful release and cross-tenant access incident.
- [ ] **05** Test accessibility, empty states, missing data and cancelled investigations.

### Outputs and checks
- Artifacts: Monitoring UI, event timeline, incident runbook.
- Acceptance: Operators can identify whether a model alert is data, performance, or service-related and see who owns the next action.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P6-09: Pass proactive lifecycle gate with fault injection
**Desired capability:** Validate safety and usefulness under realistic failures.
**Prerequisites:** P6-08
**Reuse/integration points:** All previous phase suites, event replay fixture runner.

### Engineering tasks
- [ ] **01** Replay label lag, drift, missing batch, duplicate event, provider timeout and recovery cases.
- [ ] **02** Check that investigation records persist after restart and unresolved status remains visible.
- [ ] **03** Measure alert precision, unnecessary investigations, response latency and agent cost.
- [ ] **04** Attempt permission escalation through event payload and verify server rejects it.
- [ ] **05** Sign G6 report with policy refusals and outcome accuracy.

### Outputs and checks
- Artifacts: G6 event lifecycle acceptance and remediation report.
- Acceptance: No unapproved effectful actions, no alert storm, every conclusion grounded in retained evidence.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
