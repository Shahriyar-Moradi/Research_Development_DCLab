# Implementation Subplan 7B: Enterprise and runtime hardening
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 7: External Dashboard, API Productization, Generalization and Production
**Phase exit gate:** G7
**Mission:** Ship a standalone customer-facing product and verify that scale and entity variation do not require new agent or harness implementations.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P7-04` | Harden tenant security, privacy and secret management | Security review, penetration-test backlog, privacy controls. | P7-03 |
| `P7-05` | Harden reliability, performance and failure isolation | Performance benchmark, capacity plan, runbooks and recovery drills. | P7-04 |
| `P7-06` | Implement reproducible CI/CD and versioned releases | CI/CD pipelines, release manifest and rollback procedure. | P7-05 |

## P7-04: Harden tenant security, privacy and secret management
**Desired capability:** Prove isolation at API, agent, tool, data and event layers.
**Prerequisites:** P7-03
**Reuse/integration points:** External policy and identity system; DCLab ACL is not substituted.

### Engineering tasks
- [ ] **01** Run threat model focused on prompt injection, permission confusion, cross-tenant retrieval, malicious tool output and webhook spoofing.
- [ ] **02** Enforce least-privileged provider credentials, secret rotation, encryption and audit retention policies.
- [ ] **03** Verify policy at ingress and connector call, including job resumptions and background events.
- [ ] **04** Add PII redaction/data minimization controls and deletion/export workflow for customer-owned information.
- [ ] **05** Perform adversarial tests and formal review of high-impact approval path.

### Outputs and checks
- Artifacts: Security review, penetration-test backlog, privacy controls.
- Acceptance: No cross-tenant data access in test matrix; compromised tool output cannot authorize action or expose credentials.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-05: Harden reliability, performance and failure isolation
**Desired capability:** Keep agent processes stable with many registered instances.
**Prerequisites:** P7-04
**Reuse/integration points:** External jobs, state, tracing, caching and adapter layers.

### Engineering tasks
- [ ] **01** Measure concurrent runs, queue delays, slow providers, embedding retrieval and dashboard response under load.
- [ ] **02** Set SLOs for API availability, evidence lookup, queued job start and supported read-only request completion.
- [ ] **03** Introduce rate limits, resource quotas, backpressure, circuit breakers and cache invalidation.
- [ ] **04** Test backup/restore, migration rollback, service restart and disaster recovery procedures.
- [ ] **05** Define cost limits for model calls, storage, data transfers and per-tenant work.

### Outputs and checks
- Artifacts: Performance benchmark, capacity plan, runbooks and recovery drills.
- Acceptance: API stays responsive under worker saturation; data restored with consistent entity/decision histories.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-06: Implement reproducible CI/CD and versioned releases
**Desired capability:** Make templates and harness changes safe to upgrade.
**Prerequisites:** P7-05
**Reuse/integration points:** Template Registry and external deployment stack.

### Engineering tasks
- [ ] **01** Set CI quality gates: lint, typing, schema migration, contract, unit, integration, security and sealed scenario tests.
- [ ] **02** Build staging/production promotion with signed artifacts and pinned template versions.
- [ ] **03** Add canary rollout and safe rollback for template/harness releases independent of agent instance state.
- [ ] **04** Preserve compatibility of manifests, saved execution checkpoints and public API versions.
- [ ] **05** Define release notes and audit trail of which template version produced each recommendation.

### Outputs and checks
- Artifacts: CI/CD pipelines, release manifest and rollback procedure.
- Acceptance: An older pinned agent remains runnable or has an explicit migration path; failed template rollout can be reversed.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
