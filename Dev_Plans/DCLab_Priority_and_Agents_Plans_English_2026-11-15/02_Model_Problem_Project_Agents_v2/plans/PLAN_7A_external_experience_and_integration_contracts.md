# Implementation Subplan 7A: External experience and integration contracts
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 7: External Dashboard, API Productization, Generalization and Production
**Phase exit gate:** G7
**Mission:** Ship a standalone customer-facing product and verify that scale and entity variation do not require new agent or harness implementations.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P7-01` | Deliver complete external dashboard information architecture | Production UI design system, routes and accessibility tests. | P6-09 |
| `P7-02` | Version public APIs and publish integration SDK | OpenAPI v1, SDK reference and interoperability suite. | P7-01 |
| `P7-03` | Implement outbound events, webhooks and connector packaging | Webhook delivery service, connector manifest, integration examples. | P7-02 |

## P7-01: Deliver complete external dashboard information architecture
**Desired capability:** Present the three agent layers and their lifecycle as a coherent product.
**Prerequisites:** P6-09
**Reuse/integration points:** External React or comparable UI stack; no dependency on DCLab studio frontend.

### Engineering tasks
- [ ] **01** Create Project, Problem and Model home pages with persistent navigation and breadcrumbs.
- [ ] **02** Build chat/workspace, solution portfolio, model lineage/evaluation, tasks, approvals, monitoring and evidence explorer.
- [ ] **03** Show agent kind, template version, current binding, resource scope and executable capabilities.
- [ ] **04** Add task-run timeline, blocking conditions, cost and permission states in detail views.
- [ ] **05** Validate experience with ML engineer and stakeholder walkthroughs.

### Outputs and checks
- Artifacts: Production UI design system, routes and accessibility tests.
- Acceptance: A user can locate any current entity, track changes, and drill from a decision to underlying data without relying solely on chat.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-02: Version public APIs and publish integration SDK
**Desired capability:** Make the platform usable by apps and external automation.
**Prerequisites:** P7-01
**Reuse/integration points:** Existing external v1 shell from Phase 1.

### Engineering tasks
- [ ] **01** Publish stable OpenAPI for templates, instances, projects, problems, models, solutions, tasks, approvals, events and evidence.
- [ ] **02** Define error codes, pagination, filters, optimistic versioning and event/idempotency request headers.
- [ ] **03** Provide Python SDK and basic TypeScript usage examples for common read and delegated operations.
- [ ] **04** Add contract tests against fake DCLab integration and controlled real-adapter environment.
- [ ] **05** Document backwards compatibility, deprecation and API authentication.

### Outputs and checks
- Artifacts: OpenAPI v1, SDK reference and interoperability suite.
- Acceptance: A separate sample client creates project/problem/model instances, lists evidence and watches a task through supported APIs.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P7-03: Implement outbound events, webhooks and connector packaging
**Desired capability:** Allow optional data-provider and customer integrations.
**Prerequisites:** P7-02
**Reuse/integration points:** DCLab APIs and external adapters.

### Engineering tasks
- [ ] **01** Create signed webhooks for task status, evaluation completed, monitoring alert, approval and solution decision.
- [ ] **02** Implement delivery retries, dedupe, ordering guidance and tenant-scoped webhook secrets.
- [ ] **03** Package DCLab provider as opt-in adapter with feature discovery and compatibility checks.
- [ ] **04** Define connector test suite for timeouts, revoked credentials and partially supported capabilities.
- [ ] **05** Document no-direct-DB-access rule between external platform and DCLab.

### Outputs and checks
- Artifacts: Webhook delivery service, connector manifest, integration examples.
- Acceptance: DCLab downtime reduces only provider-dependent operations; external identity, UI and audits remain available.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
