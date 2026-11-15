# Implementation Subplan 0C: Quality, security and integration readiness
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 0: Foundation, DCLab Gap Audit and Reference Scenario
**Phase exit gate:** G0
**Mission:** Create a validated, provenance-aware specification and reproducible churn reference before coding the external platform.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P0-06` | Specify evaluation and safety benchmarks | Sealed scenario pack, scoring rubric, evaluation runner skeleton. | Previous approved package and phase prerequisites |
| `P0-07` | Define security, tenancy and risk policies | Threat model, RBAC/ABAC matrix, privacy and retention specifications. | Previous approved package and phase prerequisites |
| `P0-08` | Validate DCLab adapter feasibility | Adapter spike, example payloads, integration risk register. | Previous approved package and phase prerequisites |
| `P0-09` | Create project delivery and PR conventions | Backlog tracker, repository contributing guide, PR checklist. | Previous approved package and phase prerequisites |

## P0-06: Specify evaluation and safety benchmarks
**Desired capability:** Establish evidence-based gates before agent prompts are tuned.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** Existing `dclab_rnd/agent_eval`, test fixtures and agent trace format.

### Engineering tasks
- [ ] **01** Create answer cases for exact version retrieval, feature lineage, missing metrics, outdated context and contradictory conversations.
- [ ] **02** Create execution cases for invalid tool arguments, withheld approval, failed job, duplicate event and cross-tenant attempt.
- [ ] **03** Create problem cases that distinguish a rise in business churn from declining model recall.
- [ ] **04** Specify human-scored rubric and deterministic oracle checks for citations, task completion, unauthorized side effects and recovery.
- [ ] **05** Record baselines for direct tools, single agent and three-agent architecture where feasible.

### Outputs and checks
- Artifacts: Sealed scenario pack, scoring rubric, evaluation runner skeleton.
- Acceptance: Every critical pathway has a negative test and traceable expected outcome; sealed cases are not leaked into prompts.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P0-07: Define security, tenancy and risk policies
**Desired capability:** Create a control matrix that can be enforced outside LLM prompts.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** Reuse DCLab guard concepts, not implicit trust in DCLab sessions.

### Engineering tasks
- [ ] **01** Identify actor types, workspace ownership, project membership, model sharing and service-to-service identities.
- [ ] **02** Define permissions for read, evaluate, query, create candidate, retrain, adopt solution, deploy and act on production.
- [ ] **03** Mark each capability read-only, experimental write, controlled write or high-impact operation.
- [ ] **04** Specify retention, deletion, PII/secret redaction, audit evidence and cross-tenant isolation rules.
- [ ] **05** Design approval expirations, immutable action receipts and revocation behavior.

### Outputs and checks
- Artifacts: Threat model, RBAC/ABAC matrix, privacy and retention specifications.
- Acceptance: Unsafe operations are denied in a policy simulation independent of agent outputs; privacy owner approves data paths.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P0-08: Validate DCLab adapter feasibility
**Desired capability:** Prove the external platform can integrate without a fork.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** DCLab APIs, `agentic/routers/projects.py`, `jobs.py`, evaluation and lineage modules.

### Engineering tasks
- [ ] **01** Identify callable DCLab interfaces for project metadata, model lineage, dataset references, evaluation, retraining and job status.
- [ ] **02** Prototype a versioned HTTP or command adapter with least privilege and an explicit timeout.
- [ ] **03** Specify failure semantics, event mapping, data redaction and idempotent job request receipts.
- [ ] **04** Test with a fake DCLab provider first and a local integration instance second.
- [ ] **05** Document incompatibilities and isolate any necessary bridge behind an adapter rather than coupling the new service to internals.

### Outputs and checks
- Artifacts: Adapter spike, example payloads, integration risk register.
- Acceptance: External service continues to operate in read-only degraded mode when DCLab is unavailable; no core modifications required for the spike.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P0-09: Create project delivery and PR conventions
**Desired capability:** Turn the architecture into independently testable implementation slices.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** Existing DCLab Makefile/test practices inform new repo conventions.

### Engineering tasks
- [ ] **01** Assign package IDs, owners, estimate bands, dependencies, branch/PR template, and acceptance criteria.
- [ ] **02** Establish CI jobs for schema validation, formatting, type checks, fast unit tests and integration smoke tests.
- [ ] **03** Define compatibility/version policy for templates, manifests, tools, events and REST APIs.
- [ ] **04** Set quality gates for provenance, policy refusals, rollback and performance limits.
- [ ] **05** Publish a prioritized MVP cut that can demonstrate model questions before all remaining phases finish.

### Outputs and checks
- Artifacts: Backlog tracker, repository contributing guide, PR checklist.
- Acceptance: Every package has one accountable owner, a failing-test-first definition and a reviewable evidence link.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
