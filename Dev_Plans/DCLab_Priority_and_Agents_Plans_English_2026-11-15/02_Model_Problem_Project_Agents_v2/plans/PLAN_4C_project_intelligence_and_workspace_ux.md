# Implementation Subplan 4C: Project intelligence and workspace UX
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 4: Project Agent Template and Specialized Project Harness
**Phase exit gate:** G4
**Mission:** Create the long-lived portfolio-level agent coordinating multiple problems, solutions, people, decisions and outcomes without duplicating Problem Agent expertise.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P4-08` | Implement project intelligence Q&A and executive synthesis | ProjectAnswer DTO, executive status tool and Q&A benchmark. | P4-06 |
| `P4-09` | Implement Project Agent dashboard and portfolio navigation | Project Agent UI, portfolio board and navigation test suite. | P4-08 |
| `P4-10` | Pass G4 Project Agent reuse and continuity benchmark | G4 gate report and stable Project Agent template. | P4-09 |

## P4-08: Implement project intelligence Q&A and executive synthesis
**Desired capability:** Give leaders current project reality with evidence, not persuasive fiction.
**Prerequisites:** P4-06
**Reuse/integration points:** Project Harness, child-agent contracts, EvidenceBundle.

### Engineering tasks
- [ ] **01** Create intents for status, next steps, milestones, what changed, choices, risks and comparative outcomes.
- [ ] **02** Collect structured child-agent answers with evidence refs and resource freshness rather than asking an LLM to reconstruct all data.
- [ ] **03** Distinguish proven results, pending experiments, forecasts, assumptions and recommendations.
- [ ] **04** Summarize decisions, dependencies and unresolved uncertainties with links to exact records.
- [ ] **05** Set limits on evidence volume and provide drill-down paths for domain-specific details.

### Outputs and checks
- Artifacts: ProjectAnswer DTO, executive status tool and Q&A benchmark.
- Acceptance: Status report never calls a failed job completed; all numeric values resolve to source with window and unit.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-09: Implement Project Agent dashboard and portfolio navigation
**Desired capability:** Give users a coherent top-level workspace with specialist drill-down.
**Prerequisites:** P4-08
**Reuse/integration points:** External dashboard components from earlier phases.

### Engineering tasks
- [ ] **01** Create overview, goals, workstreams, problems, solutions, models, milestones, budget, timeline and approvals panels.
- [ ] **02** Build project chat with evidence chips, agent delegation trace, current active work and corrective actions.
- [ ] **03** Add navigation from project to problem solution assessment to exact model evaluation.
- [ ] **04** Design empty states, conflict warnings, access-denied explanations and status labels.
- [ ] **05** Run usability checks against representative ML engineer and business stakeholder tasks.

### Outputs and checks
- Artifacts: Project Agent UI, portfolio board and navigation test suite.
- Acceptance: A user can traverse Project -> Problem -> Solution -> ModelVersion -> Evaluation without losing provenance.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P4-10: Pass G4 Project Agent reuse and continuity benchmark
**Desired capability:** Prove the three-layer distinction in end-to-end scenarios.
**Prerequisites:** P4-09
**Reuse/integration points:** All three harnesses and current benchmark framework.

### Engineering tasks
- [ ] **01** Create one project with two problems, at least two solutions for one problem, and shared or separate models.
- [ ] **02** Test project-level priority requests, linked histories, solution changes and active-job visibility.
- [ ] **03** Simulate removed model permission, stale decision and lost provider connection.
- [ ] **04** Check child identity and policy isolation; run sealed answer/authorization suite.
- [ ] **05** Publish project acceptance report and update compatibility matrix.

### Outputs and checks
- Artifacts: G4 gate report and stable Project Agent template.
- Acceptance: Two projects can be created from the same Project template; no duplicate harness classes or cross-project leakage.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
