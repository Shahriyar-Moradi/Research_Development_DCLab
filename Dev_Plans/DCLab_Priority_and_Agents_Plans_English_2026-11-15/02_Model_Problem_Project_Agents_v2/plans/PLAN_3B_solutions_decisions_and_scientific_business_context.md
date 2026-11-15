# Implementation Subplan 3B: Solutions, decisions and scientific/business context
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 3: Problem Agent Template and Specialized Problem Harness
**Phase exit gate:** G3
**Mission:** Give each independently versioned business/technical problem an agent that understands its objectives, complete solution history, evidence, context and operational life.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P3-04` | Implement first-class Solution Portfolio under each problem | SolutionRepository, solution comparison and adoption workflow. | P3-02 |
| `P3-05` | Implement Decision and Assumption Ledger | Decision ledger, assumption register and comparison explanations. | P3-04 |
| `P3-06` | Implement problem data sources, feature maps and business KPI contracts | ProblemDataContext, KPI registry, scoped query adapter. | P3-02 |
| `P3-07` | Implement multi-criteria solution assessment | SolutionAssessment DTO, comparison table and decision proposal. | P3-04, P3-06 |

## P3-04: Implement first-class Solution Portfolio under each problem
**Desired capability:** Treat solution alternatives as independent, versioned entities, not paragraphs in chat.
**Prerequisites:** P3-02
**Reuse/integration points:** DCLab `dclab_core/plan.py`, `solution_workflow.py`, Studio solution APIs.

### Engineering tasks
- [ ] **01** Define solution status proposed/under_review/validated/rejected/adopted/superseded/retired and revision history.
- [ ] **02** Store approach, constraints, assumptions, associated model versions, evaluation methods, planned workflow and business hypothesis.
- [ ] **03** Create compare/clone/revise/adopt operations with immutable source evidence and acceptance gates.
- [ ] **04** Integrate `SolutionPlan`, `SolutionWorkflow` and Studio Solution as referenced artifacts, not duplicate planners.
- [ ] **05** Display rejected candidates and reasons, and require approval for replacing the adopted solution.

### Outputs and checks
- Artifacts: SolutionRepository, solution comparison and adoption workflow.
- Acceptance: Changing current solution preserves old alternatives, explicit decision rationale, and linked evaluation records.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-05: Implement Decision and Assumption Ledger
**Desired capability:** Answer why choices were made using recorded evidence.
**Prerequisites:** P3-04
**Reuse/integration points:** DCLab ledgers and transitions through adapter where available.

### Engineering tasks
- [ ] **01** Define DecisionRecord: actor, alternatives, chosen option, rationale, evidence refs, uncertainties, time, scope and approval.
- [ ] **02** Distinguish observation, unverified hypothesis, assumption, proposal, accepted decision and later outcome.
- [ ] **03** Implement append-only superseding history and source-level access control.
- [ ] **04** Build read tools for "Why XGBoost?", "Which alternatives failed?" and "What changed?".
- [ ] **05** Enforce no-invented-rationale policy; infer possible causes only when labeled inference.

### Outputs and checks
- Artifacts: Decision ledger, assumption register and comparison explanations.
- Acceptance: Missing historical rationale produces "not recorded" rather than a reconstructed fiction.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-06: Implement problem data sources, feature maps and business KPI contracts
**Desired capability:** Bridge business context with ML metrics correctly.
**Prerequisites:** P3-02
**Reuse/integration points:** `dclab_rnd/connectors/db.py`, Studio feature/data tools, data audit.

### Engineering tasks
- [ ] **01** Define dataset binding and approved query views with data owner, schema, freshness, row cap and workspace permission.
- [ ] **02** Create KPI definitions including unit, numerator/denominator, cohort, period, timezone, source and allowed comparison rule.
- [ ] **03** Link problem features and model inputs with prediction moment and leakage review.
- [ ] **04** Implement read-only business cohort queries with parameter validation and PII minimization.
- [ ] **05** Test business churn rate vs model recall confusion, stale source and missing labels.

### Outputs and checks
- Artifacts: ProblemDataContext, KPI registry, scoped query adapter.
- Acceptance: Agent never represents recall as churn rate; every KPI states population, period, source and missing-data caveats.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P3-07: Implement multi-criteria solution assessment
**Desired capability:** Compare candidate approaches using evidence, cost and business goals.
**Prerequisites:** P3-04, P3-06
**Reuse/integration points:** DCLab evaluation/report tools via Model Agent, business KPI registry.

### Engineering tasks
- [ ] **01** Define assessment criteria: model quality, expected or observed business outcome, resource cost, latency, operational risk and data readiness.
- [ ] **02** Distinguish offline metrics, forecasts and controlled intervention outcomes with separate evidence classes.
- [ ] **03** Create comparable cohort/window and threshold check before ranking solution candidates.
- [ ] **04** Persist a scored trade-off matrix with qualitative limitations and stakeholder-approved weights.
- [ ] **05** Provide an explainable recommendation with experiments needed to close uncertainty.

### Outputs and checks
- Artifacts: SolutionAssessment DTO, comparison table and decision proposal.
- Acceptance: Better model AUC alone cannot be reported as proven customer-retention improvement.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
