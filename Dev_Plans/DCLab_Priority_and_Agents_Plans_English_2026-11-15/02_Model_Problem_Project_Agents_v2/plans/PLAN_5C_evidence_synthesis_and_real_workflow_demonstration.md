# Implementation Subplan 5C: Evidence synthesis and real workflow demonstration
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 5: Typed Multi-Agent Collaboration and Workflow Orchestration
**Phase exit gate:** G5
**Mission:** Coordinate three specialized harnesses with structured tasks, explicit dependencies, bounded execution and evidence-bearing results.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P5-07` | Implement evidence merging, contradiction detection and review | EvidenceMerger, consistency report and final response structure. | P5-03 |
| `P5-08` | Build complete churn investigation workflow | Vertical-slice demo, walkthrough fixture and trace bundle. | P5-07 |
| `P5-09` | Run sealed collaboration benchmark and performance ablations | G5 collaboration gate and comparative benchmark report. | P5-08 |

## P5-07: Implement evidence merging, contradiction detection and review
**Desired capability:** Produce reliable final answers from multiple specialized agents.
**Prerequisites:** P5-03
**Reuse/integration points:** All harness answer contracts and evidence resolver.

### Engineering tasks
- [ ] **01** Merge TaskResults by source, entity, version, measurement unit, window, cohort and freshness.
- [ ] **02** Detect contradictory metrics from incompatible thresholds, datasets or label timing.
- [ ] **03** Distinguish observed facts from hypotheses, actions and policy decisions in final answer.
- [ ] **04** Apply deterministic completeness checks; add targeted critic review only when needed and measurable.
- [ ] **05** Publish evidence bundle mapping every quantitative assertion to a canonical result.

### Outputs and checks
- Artifacts: EvidenceMerger, consistency report and final response structure.
- Acceptance: Conflicting metric evidence is flagged; the agent does not average incompatible recall scores into a fake answer.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P5-08: Build complete churn investigation workflow
**Desired capability:** Demonstrate the intended agentic ML product end to end.
**Prerequisites:** P5-07
**Reuse/integration points:** All three harnesses, data binding, evaluation, workflow scheduler.

### Engineering tasks
- [ ] **01** Trigger investigation from Project Agent for declining recall in a named period.
- [ ] **02** Delegate problem-scoped question to Churn Problem Agent and precise model diagnostics to Churn Model Agent.
- [ ] **03** Query approved labeled performance and business churn KPI windows independently.
- [ ] **04** Produce hypotheses, data gaps, candidate experiments, chosen next steps and any approval request.
- [ ] **05** Persist investigation as a resumable project/prob/ model-linked record with clear final verdict.

### Outputs and checks
- Artifacts: Vertical-slice demo, walkthrough fixture and trace bundle.
- Acceptance: No fabricated scores, no unauthorized retraining, and a complete evidence-linked task tree across three agents.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P5-09: Run sealed collaboration benchmark and performance ablations
**Desired capability:** Prove that three agents actually help versus simpler alternatives.
**Prerequisites:** P5-08
**Reuse/integration points:** Existing DCLab sealed benchmark practices; independent external evaluations.

### Engineering tasks
- [ ] **01** Compare direct deterministic tool workflow, single general agent and three-template orchestration on same tasks.
- [ ] **02** Measure success, tool/route accuracy, policy violations, recovery, human effort, latency and cost per task.
- [ ] **03** Study alternatives including no project agent, no memory retrieval, deterministic-only routing and fewer delegation steps.
- [ ] **04** Identify failure clusters and adjust routing or tool boundaries rather than just increasing prompt size.
- [ ] **05** Gate only if collaboration improves a named product capability at acceptable cost.

### Outputs and checks
- Artifacts: G5 collaboration gate and comparative benchmark report.
- Acceptance: No unjustified agent proliferation: more complex architecture must have documented net benefit.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
