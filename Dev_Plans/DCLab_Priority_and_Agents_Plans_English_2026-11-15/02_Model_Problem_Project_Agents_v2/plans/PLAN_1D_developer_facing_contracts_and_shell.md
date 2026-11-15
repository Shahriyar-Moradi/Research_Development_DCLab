# Implementation Subplan 1D: Developer-facing contracts and shell
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 1: Shared Infrastructure, Template Registry and Agent Factory
**Phase exit gate:** G1
**Mission:** Deliver an independent service runtime that can instantiate all three agent types from configuration, with no new agent/harness classes per entity.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P1-12` | Expose minimal API, SDK and proof-of-reuse scenario | Versioned v1 API, CLI/SDK examples, G1 demo. | P1-04, P1-06, P1-11 |

## P1-12: Expose minimal API, SDK and proof-of-reuse scenario
**Desired capability:** Give downstream phase developers stable operations and examples.
**Prerequisites:** P1-04, P1-06, P1-11
**Reuse/integration points:** New external frontend shell, built before full product dashboard.

### Engineering tasks
- [ ] **01** Expose template listing, instance creation, run submission, job lookup, approved tool list and evidence fetch endpoints.
- [ ] **02** Publish typed OpenAPI schemas, Python client examples and three starter manifests.
- [ ] **03** Create a minimal non-production dashboard for agent identity/run status and a trace viewer.
- [ ] **04** Write contract tests using fake DCLab adapters to instantiate one of each agent kind.
- [ ] **05** Measure cold-start and run overhead without LLM and record baseline budgets.

### Outputs and checks
- Artifacts: Versioned v1 API, CLI/SDK examples, G1 demo.
- Acceptance: Three instances with isolated records and shared kernel can be created, inspected, and resumed via API.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
