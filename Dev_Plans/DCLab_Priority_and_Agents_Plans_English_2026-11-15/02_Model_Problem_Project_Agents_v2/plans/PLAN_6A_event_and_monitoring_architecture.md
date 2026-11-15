# Implementation Subplan 6A: Event and monitoring architecture
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 6: Event-Driven Monitoring, Investigation and Continuous Lifecycle
**Phase exit gate:** G6
**Mission:** Run proactive model/problem/project operations through governed events and evidence-based improvement loops.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P6-01` | Define canonical lifecycle EventEnvelope | EventEnvelope schema, source adapter API, event catalog. | P5-09 |
| `P6-02` | Implement monitoring collectors, windows and source health | MonitoringCollector, health status and audit trail. | P6-01 |
| `P6-03` | Implement severity, hysteresis and event routing policies | EventPolicyEngine, route table and notifications. | P6-02 |

## P6-01: Define canonical lifecycle EventEnvelope
**Desired capability:** Create one event format for model, problem and project signals.
**Prerequisites:** P5-09
**Reuse/integration points:** Model monitoring and shared event infrastructure.

### Engineering tasks
- [ ] **01** Define event_id, source, kind, severity, entity refs, event_time, observed_window, policy version, evidence refs and dedupe key.
- [ ] **02** Create source adapters for model telemetry, DCLab jobs, dataset updates, solution decisions and business KPI anomalies.
- [ ] **03** Validate signature/trust, schema evolution and unknown or out-of-order event handling.
- [ ] **04** Publish event catalog with semantics for drift, labels available, recall decline, failed job and business outcome change.
- [ ] **05** Create replay fixtures and synthetic generators without production credentials.

### Outputs and checks
- Artifacts: EventEnvelope schema, source adapter API, event catalog.
- Acceptance: Duplicate and out-of-order messages preserve one coherent audit timeline without retriggering identical effects.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P6-02: Implement monitoring collectors, windows and source health
**Desired capability:** Ingest signals with quality guarantees instead of merely polling LLMs.
**Prerequisites:** P6-01
**Reuse/integration points:** DCLab monitoring signals and external adapter streams.

### Engineering tasks
- [ ] **01** Define scheduled/streaming collectors with source authorization, freshness limits and backpressure.
- [ ] **02** Store window metadata, sample sizes, missing counts, label delays and baseline references.
- [ ] **03** Verify metrics against their own typed data source and model version before publication.
- [ ] **04** Detect missing telemetry, unhealthy data connectors and unreachable evaluation providers.
- [ ] **05** Keep raw sensitive rows in approved source stores and emit derived aggregates with refs.

### Outputs and checks
- Artifacts: MonitoringCollector, health status and audit trail.
- Acceptance: No alert is presented as model deterioration when underlying sample size or collection quality is insufficient.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P6-03: Implement severity, hysteresis and event routing policies
**Desired capability:** Prevent alert storms and unsupported automatic action.
**Prerequisites:** P6-02
**Reuse/integration points:** Three harness lifecycle policies and shared registry.

### Engineering tasks
- [ ] **01** Define thresholds by model, problem objective and project risk tolerance.
- [ ] **02** Implement cooldown, consecutive-window criteria, minimum sample and deduplication.
- [ ] **03** Route events to Model Agent for model health, Problem Agent for solution impact and Project Agent for portfolio escalation.
- [ ] **04** Record suppression reasons, event-policy version and escalation deadlines.
- [ ] **05** Support manual mute, acknowledgement and on-call override with audit.

### Outputs and checks
- Artifacts: EventPolicyEngine, route table and notifications.
- Acceptance: One sustained drift produces bounded investigations; urgent high-impact event reaches correct agent and owner.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
