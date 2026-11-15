# Implementation Subplan 7C: Reuse validation, docs and release
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 7: External Dashboard, API Productization, Generalization and Production
**Phase exit gate:** G7
**Mission:** Ship a standalone customer-facing product and verify that scale and entity variation do not require new agent or harness implementations.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P7-07` | Prove template reuse on 5 models, 3 problems and 2 projects | Reuse-test fixtures, onboarding scripts and measured compatibility report. | P7-06 |
| `P7-08` | Benchmark accuracy, authorization, task completion and cost | Comparative benchmark and production readiness scorecard. | P7-07 |
| `P7-09` | Complete customer onboarding, operator docs and governance | Installation guides, API docs, user manuals and operations playbook. | P7-08 |
| `P7-10` | Run production cutover and post-launch controls | G7 release sign-off, tagged binaries and live operations checklist. | P7-09 |

## P7-07: Prove template reuse on 5 models, 3 problems and 2 projects
**Desired capability:** Validate the primary economic and engineering requirement.
**Prerequisites:** P7-06
**Reuse/integration points:** Three shipped templates and harnesses, factory, adapter SDK.

### Engineering tasks
- [ ] **01** Onboard five distinct model instances including two algorithm families and at least one different task-kind adapter where supported.
- [ ] **02** Create three problem instances with independent objectives, alternative solutions and histories.
- [ ] **03** Create two project instances coordinating multiple problems and shared model links.
- [ ] **04** Require all instantiation via manifests, bindings, seed data and supported adapters; no new `Agent` or `Harness` class.
- [ ] **05** Measure onboarding effort, changed files, number of reusable modules and cost per new entity.

### Outputs and checks
- Artifacts: Reuse-test fixtures, onboarding scripts and measured compatibility report.
- Acceptance: 5/3/2 gate passes without per-entity harness/agent class; any new adapter is versioned and generic to a capability or model kind.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-08: Benchmark accuracy, authorization, task completion and cost
**Desired capability:** Ensure the product beats simpler alternatives where it should.
**Prerequisites:** P7-07
**Reuse/integration points:** Sealed tests and benchmark runner created in P0-06 and expanded each phase.

### Engineering tasks
- [ ] **01** Run sealed QA and tool scenarios for each agent, plus full end-to-end collaboration and event cases.
- [ ] **02** Report completion, factual support, argument validation, tool routing, policy denials, unneeded steps, latency and total cost.
- [ ] **03** Compare against direct deterministic tools and single-agent baseline; document meaningful trade-offs.
- [ ] **04** Run negative tests for unsupported providers, missing data, misleading conversation context and leakage traps.
- [ ] **05** Publish readiness threshold table and unsolved defects rather than burying failures.

### Outputs and checks
- Artifacts: Comparative benchmark and production readiness scorecard.
- Acceptance: All critical security/safety cases pass; performance and value thresholds are explicitly approved before launch.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-09: Complete customer onboarding, operator docs and governance
**Desired capability:** Make installation, support and responsible use repeatable.
**Prerequisites:** P7-08
**Reuse/integration points:** All phase artifacts and current external UI.

### Engineering tasks
- [ ] **01** Write documentation for tenant setup, data rights, DCLab connection, manifest creation and approval policies.
- [ ] **02** Create standard runbooks for agent onboarding, model monitoring, retraining, decision revisions and incident response.
- [ ] **03** Publish user guides for Model, Problem and Project Agents and dashboard terminology.
- [ ] **04** Document supported/unsupported algorithms, connectors and business actions.
- [ ] **05** Create migration guide from older two-harness roadmap to three independent specialized templates.

### Outputs and checks
- Artifacts: Installation guides, API docs, user manuals and operations playbook.
- Acceptance: New engineer can onboard a supported model/problem/project from docs alone without changing platform source.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-10: Run production cutover and post-launch controls
**Desired capability:** Release only what is tested and controllable.
**Prerequisites:** P7-09
**Reuse/integration points:** All prior phases.

### Engineering tasks
- [ ] **01** Define rollout cohort, owner/operator roster, alerting and rollback decision thresholds.
- [ ] **02** Run final smoke: create entities, query evidence, delegate analysis, request approval, replay event and inspect outcome.
- [ ] **03** Verify data/privacy agreements, backups, security review, API compatibility and observability dashboards.
- [ ] **04** Sign release gates with evidence refs and explicit scope exclusions, including no automatic production deployment.
- [ ] **05** Establish measured improvement backlog and quarterly review of template specialization versus configuration sprawl.

### Outputs and checks
- Artifacts: G7 release sign-off, tagged binaries and live operations checklist.
- Acceptance: 5/3/2 acceptance, safety, docs, rollback and incident operations are verified; unimplemented capabilities are disclosed.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
