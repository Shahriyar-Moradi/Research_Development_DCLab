# Implementation Subplan 6B: Investigations and controlled response
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 6: Event-Driven Monitoring, Investigation and Continuous Lifecycle
**Phase exit gate:** G6
**Mission:** Run proactive model/problem/project operations through governed events and evidence-based improvement loops.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P6-04` | Implement persistent Investigation entity and ownership | InvestigationService, correlation and status APIs. | P6-03 |
| `P6-05` | Implement safe diagnostic playbooks and triage | DiagnosticPlaybookRegistry, triage findings, evidence bundles. | P6-04 |
| `P6-06` | Implement policy-gated improvement and retrain triggers | ImprovementProposal, retrain policy and execution receipt. | P6-05 |

## P6-04: Implement persistent Investigation entity and ownership
**Desired capability:** Make investigations resumable, reviewable and measurable.
**Prerequisites:** P6-03
**Reuse/integration points:** Task scheduler, knowledge graph and evidence receipts.

### Engineering tasks
- [ ] **01** Create Investigation with trigger evidence, scope, owner agent, suspected causes, plan, linked tasks, status and findings.
- [ ] **02** Correlate repeated alerts into existing open investigation when rules indicate same event family.
- [ ] **03** Start with read-only diagnostics and explicit evidence gathering.
- [ ] **04** Support human assignment, conversation updates and closure reasons with immutable timeline.
- [ ] **05** Represent insufficient evidence and unresolved investigation as first-class outcomes.

### Outputs and checks
- Artifacts: InvestigationService, correlation and status APIs.
- Acceptance: A duplicate alert links to an existing investigation, not an unbounded second workflow.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P6-05: Implement safe diagnostic playbooks and triage
**Desired capability:** Reuse tested scientific analysis rather than inventing ad hoc remediation.
**Prerequisites:** P6-04
**Reuse/integration points:** Model diagnostics plus problem KPI and lineage tools.

### Engineering tasks
- [ ] **01** Create typed playbooks for schema mismatch, missing feature, distribution shift, label delay and prediction/performance drift.
- [ ] **02** Gather feature profiles, model evaluations, data lineage and business KPIs with independent provenance.
- [ ] **03** Apply deterministic prerequisites and choose investigations based on available labels and windows.
- [ ] **04** Mark suspected causes as hypotheses until confirmed by targeted checks.
- [ ] **05** Persist next actions, data gaps and failure reasons for each playbook.

### Outputs and checks
- Artifacts: DiagnosticPlaybookRegistry, triage findings, evidence bundles.
- Acceptance: Synthetic concept shift is distinguished from data-quality failure where fixtures support that distinction; missing labels remain uncertainty.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P6-06: Implement policy-gated improvement and retrain triggers
**Desired capability:** Allow proactive improvement without unsafe model churn.
**Prerequisites:** P6-05
**Reuse/integration points:** Model retraining, solution assessment, approval state machine.

### Engineering tasks
- [ ] **01** Define eligibility policy for retrain: sufficient labeled new data, allowed source, matching feature contract and validated degradation.
- [ ] **02** Create candidate specs and approval requests when the event warrants a controlled experiment.
- [ ] **03** Compare candidate to champion on appropriate held-out windows and business-relevant thresholds.
- [ ] **04** Require separate authorization for adopted solution changes, model deployment and external business interventions.
- [ ] **05** Record experiments not worth running to avoid recurring expensive recommendations.

### Outputs and checks
- Artifacts: ImprovementProposal, retrain policy and execution receipt.
- Acceptance: Feature drift alone cannot automatically deploy or claim concept drift; policy refusal is visible and audit-safe.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
