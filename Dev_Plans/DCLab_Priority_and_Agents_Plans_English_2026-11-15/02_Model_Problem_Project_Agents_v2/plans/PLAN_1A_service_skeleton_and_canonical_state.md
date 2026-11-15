# Implementation Subplan 1A: Service skeleton and canonical state
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 1: Shared Infrastructure, Template Registry and Agent Factory
**Phase exit gate:** G1
**Mission:** Deliver an independent service runtime that can instantiate all three agent types from configuration, with no new agent/harness classes per entity.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P1-01` | Bootstrap standalone external service | Bootable API, worker and local dev stack. | P0-02 |
| `P1-02` | Implement authoritative persistence and migrations | SQL migrations, repositories, model factories and seed fixtures. | P0-03 |

## P1-01: Bootstrap standalone external service
**Desired capability:** Establish a separately deployable API/worker/UI workspace.
**Prerequisites:** P0-02
**Reuse/integration points:** New external service; existing DCLab does not change.

### Engineering tasks
- [ ] **01** Create Python package, FastAPI app factory, dependency injection container, environment schema and health endpoints.
- [ ] **02** Create directories for `core`, `templates`, `harnesses`, `agents`, `adapters`, `domain`, `api`, `workers`, and `tests`.
- [ ] **03** Implement local Docker Compose or equivalent for API, PostgreSQL and a worker; keep separate DB credentials from DCLab.
- [ ] **04** Add log correlation, feature flags, readiness checks and graceful shutdown.
- [ ] **05** Document local startup with fake external integrations so no DCLab credentials are required.

### Outputs and checks
- Artifacts: Bootable API, worker and local dev stack.
- Acceptance: Cold start/health smoke tests pass with mocked DCLab provider and no LLM credentials.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-02: Implement authoritative persistence and migrations
**Desired capability:** Persist identity, versions and relations without using chat messages as database records.
**Prerequisites:** P0-03
**Reuse/integration points:** External PostgreSQL; DCLab storage mapped only through adapters.

### Engineering tasks
- [ ] **01** Create database tables for tenants, projects, problems, solutions, model families/versions, agent identities and template versions.
- [ ] **02** Add relations, artifact/evidence refs, conversations, decisions, executions, approvals and audit records.
- [ ] **03** Apply foreign keys, immutable revision constraints, unique tenant-scoped IDs and optimistic concurrency.
- [ ] **04** Implement transaction boundaries for active revision pointers and create/update operations.
- [ ] **05** Write fixtures, reversible migrations and a backfill strategy for externally linked DCLab entities.

### Outputs and checks
- Artifacts: SQL migrations, repositories, model factories and seed fixtures.
- Acceptance: Wrong-tenant reads fail; invalid state edges roll back; migration up/down and restart persistence tests pass.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
