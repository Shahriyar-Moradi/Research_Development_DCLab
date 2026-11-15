# Phase 6: Event-Driven Monitoring, Investigation and Continuous Lifecycle
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Run proactive model/problem/project operations through governed events and evidence-based improvement loops.
**Entry criteria:** G5 typed delegation works; Model Agent has evaluation/monitoring and approval-aware job adapter.
**Exit gate G6:** Monitoring event can open one deduplicated investigation, coordinate diagnostics, request approved action, and measure the effect later.
**Mandatory live demonstration:** Inject synthetic feature-drift event and labeled-performance decline; follow event through Model, Problem and Project Agents with no unnecessary retraining.
**Important boundary:** Event processing should be deterministic and cost controlled. Agent reasoning is invoked for ambiguous investigations rather than for every telemetry record.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 6A | Event and monitoring architecture | P6-01, P6-02, P6-03 | Implement in listed order unless explicitly parallel-safe |
| 6B | Investigations and controlled response | P6-04, P6-05, P6-06 | Implement in listed order unless explicitly parallel-safe |
| 6C | Outcome learning, operations and gate | P6-07, P6-08, P6-09 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 6A: Event and monitoring architecture
**Goal:** Complete P6-01, P6-02, P6-03 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P6-01 - Define canonical lifecycle EventEnvelope
**Outcome and scope.** Create one event format for model, problem and project signals.
**Preconditions/dependencies.** P5-09
**Existing integration to inspect.** Model monitoring and shared event infrastructure.

**Implementation steps (ordered):
1. Define event_id, source, kind, severity, entity refs, event_time, observed_window, policy version, evidence refs and dedupe key.
2. Create source adapters for model telemetry, DCLab jobs, dataset updates, solution decisions and business KPI anomalies.
3. Validate signature/trust, schema evolution and unknown or out-of-order event handling.
4. Publish event catalog with semantics for drift, labels available, recall decline, failed job and business outcome change.
5. Create replay fixtures and synthetic generators without production credentials.

**Artifacts and interfaces:** EventEnvelope schema, source adapter API, event catalog.

**Acceptance tests and evidence:** Duplicate and out-of-order messages preserve one coherent audit timeline without retriggering identical effects.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P6-02 - Implement monitoring collectors, windows and source health
**Outcome and scope.** Ingest signals with quality guarantees instead of merely polling LLMs.
**Preconditions/dependencies.** P6-01
**Existing integration to inspect.** DCLab monitoring signals and external adapter streams.

**Implementation steps (ordered):
1. Define scheduled/streaming collectors with source authorization, freshness limits and backpressure.
2. Store window metadata, sample sizes, missing counts, label delays and baseline references.
3. Verify metrics against their own typed data source and model version before publication.
4. Detect missing telemetry, unhealthy data connectors and unreachable evaluation providers.
5. Keep raw sensitive rows in approved source stores and emit derived aggregates with refs.

**Artifacts and interfaces:** MonitoringCollector, health status and audit trail.

**Acceptance tests and evidence:** No alert is presented as model deterioration when underlying sample size or collection quality is insufficient.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P6-03 - Implement severity, hysteresis and event routing policies
**Outcome and scope.** Prevent alert storms and unsupported automatic action.
**Preconditions/dependencies.** P6-02
**Existing integration to inspect.** Three harness lifecycle policies and shared registry.

**Implementation steps (ordered):
1. Define thresholds by model, problem objective and project risk tolerance.
2. Implement cooldown, consecutive-window criteria, minimum sample and deduplication.
3. Route events to Model Agent for model health, Problem Agent for solution impact and Project Agent for portfolio escalation.
4. Record suppression reasons, event-policy version and escalation deadlines.
5. Support manual mute, acknowledgement and on-call override with audit.

**Artifacts and interfaces:** EventPolicyEngine, route table and notifications.

**Acceptance tests and evidence:** One sustained drift produces bounded investigations; urgent high-impact event reaches correct agent and owner.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 6B: Investigations and controlled response
**Goal:** Complete P6-04, P6-05, P6-06 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P6-04 - Implement persistent Investigation entity and ownership
**Outcome and scope.** Make investigations resumable, reviewable and measurable.
**Preconditions/dependencies.** P6-03
**Existing integration to inspect.** Task scheduler, knowledge graph and evidence receipts.

**Implementation steps (ordered):
1. Create Investigation with trigger evidence, scope, owner agent, suspected causes, plan, linked tasks, status and findings.
2. Correlate repeated alerts into existing open investigation when rules indicate same event family.
3. Start with read-only diagnostics and explicit evidence gathering.
4. Support human assignment, conversation updates and closure reasons with immutable timeline.
5. Represent insufficient evidence and unresolved investigation as first-class outcomes.

**Artifacts and interfaces:** InvestigationService, correlation and status APIs.

**Acceptance tests and evidence:** A duplicate alert links to an existing investigation, not an unbounded second workflow.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P6-05 - Implement safe diagnostic playbooks and triage
**Outcome and scope.** Reuse tested scientific analysis rather than inventing ad hoc remediation.
**Preconditions/dependencies.** P6-04
**Existing integration to inspect.** Model diagnostics plus problem KPI and lineage tools.

**Implementation steps (ordered):
1. Create typed playbooks for schema mismatch, missing feature, distribution shift, label delay and prediction/performance drift.
2. Gather feature profiles, model evaluations, data lineage and business KPIs with independent provenance.
3. Apply deterministic prerequisites and choose investigations based on available labels and windows.
4. Mark suspected causes as hypotheses until confirmed by targeted checks.
5. Persist next actions, data gaps and failure reasons for each playbook.

**Artifacts and interfaces:** DiagnosticPlaybookRegistry, triage findings, evidence bundles.

**Acceptance tests and evidence:** Synthetic concept shift is distinguished from data-quality failure where fixtures support that distinction; missing labels remain uncertainty.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P6-06 - Implement policy-gated improvement and retrain triggers
**Outcome and scope.** Allow proactive improvement without unsafe model churn.
**Preconditions/dependencies.** P6-05
**Existing integration to inspect.** Model retraining, solution assessment, approval state machine.

**Implementation steps (ordered):
1. Define eligibility policy for retrain: sufficient labeled new data, allowed source, matching feature contract and validated degradation.
2. Create candidate specs and approval requests when the event warrants a controlled experiment.
3. Compare candidate to champion on appropriate held-out windows and business-relevant thresholds.
4. Require separate authorization for adopted solution changes, model deployment and external business interventions.
5. Record experiments not worth running to avoid recurring expensive recommendations.

**Artifacts and interfaces:** ImprovementProposal, retrain policy and execution receipt.

**Acceptance tests and evidence:** Feature drift alone cannot automatically deploy or claim concept drift; policy refusal is visible and audit-safe.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 6C: Outcome learning, operations and gate
**Goal:** Complete P6-07, P6-08, P6-09 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P6-07 - Implement outcome feedback and decision learning
**Outcome and scope.** Close the loop between proposed actions and real-world effect.
**Preconditions/dependencies.** P6-06
**Existing integration to inspect.** Problem decisions, Project milestones, KPI registry.

**Implementation steps (ordered):
1. Define OutcomeObservation with intervention, owner, reference population, before/after window and measurement source.
2. Link outcomes to decision, solution revision, model version, investigation and originating hypothesis.
3. Separate randomized or quasi-experimental outcome evidence from unadjusted before/after correlation.
4. Use outcome feedback to update recommendations and case records without silently modifying training data.
5. Track unresolved actions and measured time to detection/response/recovery.

**Artifacts and interfaces:** OutcomeLedger, KPI-linked effectiveness report.

**Acceptance tests and evidence:** Agent never asserts a retention lift from model recall alone; decisions remain auditable after supersession.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P6-08 - Implement operations dashboards, alert controls and incident runbooks
**Outcome and scope.** Give humans visibility and intervention when automation is uncertain.
**Preconditions/dependencies.** P6-07
**Existing integration to inspect.** External UI and event/investigation APIs.

**Implementation steps (ordered):
1. Create model health, problem outcome and project portfolio alert screens.
2. Show source health, event severity, suppression rules, investigation tasks and related approval requests.
3. Implement acknowledge, assign, mute and close with actor-bound audit records.
4. Write incident runbooks for connector outage, retrain failure, harmful release and cross-tenant access incident.
5. Test accessibility, empty states, missing data and cancelled investigations.

**Artifacts and interfaces:** Monitoring UI, event timeline, incident runbook.

**Acceptance tests and evidence:** Operators can identify whether a model alert is data, performance, or service-related and see who owns the next action.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P6-09 - Pass proactive lifecycle gate with fault injection
**Outcome and scope.** Validate safety and usefulness under realistic failures.
**Preconditions/dependencies.** P6-08
**Existing integration to inspect.** All previous phase suites, event replay fixture runner.

**Implementation steps (ordered):
1. Replay label lag, drift, missing batch, duplicate event, provider timeout and recovery cases.
2. Check that investigation records persist after restart and unresolved status remains visible.
3. Measure alert precision, unnecessary investigations, response latency and agent cost.
4. Attempt permission escalation through event payload and verify server rejects it.
5. Sign G6 report with policy refusals and outcome accuracy.

**Artifacts and interfaces:** G6 event lifecycle acceptance and remediation report.

**Acceptance tests and evidence:** No unapproved effectful actions, no alert storm, every conclusion grounded in retained evidence.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 9 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Inject synthetic feature-drift event and labeled-performance decline; follow event through Model, Problem and Project Agents with no unnecessary retraining.
- [ ] G6 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G6 provides the stable interfaces, immutable evidence and regression tests required by Phase 7.
