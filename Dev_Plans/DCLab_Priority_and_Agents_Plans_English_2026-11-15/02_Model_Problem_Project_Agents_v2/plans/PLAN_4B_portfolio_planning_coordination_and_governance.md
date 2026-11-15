# Implementation Subplan 4B: Portfolio planning, coordination and governance
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 4: Project Agent Template and Specialized Project Harness
**Phase exit gate:** G4
**Mission:** Create the long-lived portfolio-level agent coordinating multiple problems, solutions, people, decisions and outcomes without duplicating Problem Agent expertise.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P4-04` | Implement project objectives, budgets and resource constraints | ProjectGoals, ResourceBudget and constraint evaluator. | P4-01 |
| `P4-05` | Implement project portfolio and solution strategy | ProjectPortfolioService, strategy assessment and review API. | P4-02, P4-04 |
| `P4-06` | Implement project milestones, workflow and task coordination | ProjectRoadmap, milestone/ticket APIs and provenance-backed status view. | P4-05 |
| `P4-07` | Implement cross-problem governance and approvals | Portfolio policy evaluator, cross-problem approval contract. | P4-06 |

## P4-04: Implement project objectives, budgets and resource constraints
**Desired capability:** Let the Project Agent plan within real business boundaries.
**Prerequisites:** P4-01
**Reuse/integration points:** Existing external authorization and job scheduler, DCLab project metadata where available.

### Engineering tasks
- [ ] **01** Define business objectives and success metrics with measurement windows and accountable owners.
- [ ] **02** Represent operational budgets for compute, engineer time, data spend and agent/tool usage as distinct constraints.
- [ ] **03** Define resource allocations, priorities, deadlines and allowed escalation paths for child problems.
- [ ] **04** Build budget/ownership check tools that run before cross-problem execution.
- [ ] **05** Represent infeasible demands and trade-offs as explicit decisions rather than silent partial execution.

### Outputs and checks
- Artifacts: ProjectGoals, ResourceBudget and constraint evaluator.
- Acceptance: A plan exceeding a budget cannot start unauthorized jobs; the failure explains which limit blocks the work.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-05: Implement project portfolio and solution strategy
**Desired capability:** Manage competing solution paths and dependencies across problems.
**Prerequisites:** P4-02, P4-04
**Reuse/integration points:** Problem Agent solution assessments and project resources.

### Engineering tasks
- [ ] **01** Aggregate solution alternatives, model linkages and status from Problem Agents.
- [ ] **02** Build cross-problem dependency and conflict graph: shared features, scarce datasets, competing interventions, rollout risk.
- [ ] **03** Support portfolio review comparing impact, effort, uncertainty, risk and expected sequencing.
- [ ] **04** Require signed decisions for priority changes affecting budget or multiple problem owners.
- [ ] **05** Retain historical portfolio snapshots and explain rationales from decision ledger.

### Outputs and checks
- Artifacts: ProjectPortfolioService, strategy assessment and review API.
- Acceptance: Agent can prioritize churn and fraud work with explicit criteria without claiming false causal business impact.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-06: Implement project milestones, workflow and task coordination
**Desired capability:** Make high-level plans executable through child-owned capabilities.
**Prerequisites:** P4-05
**Reuse/integration points:** TaskGraph and job infrastructure patterns.

### Engineering tasks
- [ ] **01** Translate approved project objectives to milestones with owner, scope, dependencies and acceptance evidence.
- [ ] **02** Attach each work item to a Problem or Model Agent capability rather than creating a new agent per task.
- [ ] **03** Track proposed, approved, executing, completed, failed and cancelled statuses with execution receipt links.
- [ ] **04** Define handoffs, replan triggers and cancelation propagation with approval-aware checks.
- [ ] **05** Expose project-level status and blockers without mutating child scientific records.

### Outputs and checks
- Artifacts: ProjectRoadmap, milestone/ticket APIs and provenance-backed status view.
- Acceptance: Completed milestone requires verified child task evidence; project view survives service restart and revisions.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-07: Implement cross-problem governance and approvals
**Desired capability:** Ensure portfolio delegation respects local ownership and shared resources.
**Prerequisites:** P4-06
**Reuse/integration points:** Shared policy engine plus Problem and Model Harness policies.

### Engineering tasks
- [ ] **01** Define project-to-problem permissions, child-agent delegation grants and policy inheritance.
- [ ] **02** Implement approval for solution adoption, budget increases, new data access and production-impacting actions.
- [ ] **03** Require explicit scope, target entities, arguments and actor for a high-impact approval.
- [ ] **04** Audit shared model changes that affect more than one problem and require stakeholder review.
- [ ] **05** Test revocation, stale approvals and attempted escalation through chat or tool output.

### Outputs and checks
- Artifacts: Portfolio policy evaluator, cross-problem approval contract.
- Acceptance: Cross-problem writes fail unless both project and affected resource policies allow them.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
