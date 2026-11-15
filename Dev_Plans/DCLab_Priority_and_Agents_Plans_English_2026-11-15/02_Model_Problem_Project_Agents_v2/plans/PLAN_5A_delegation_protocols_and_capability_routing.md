# Implementation Subplan 5A: Delegation protocols and capability routing
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 5: Typed Multi-Agent Collaboration and Workflow Orchestration
**Phase exit gate:** G5
**Mission:** Coordinate three specialized harnesses with structured tasks, explicit dependencies, bounded execution and evidence-bearing results.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P5-01` | Define typed Agent-to-Agent protocol | AgentTaskRequest/Result schemas, versioned protocol and fixtures. | P4-10 |
| `P5-02` | Implement deterministic capability discovery and routing | AgentRouter, capability index and routing decisions. | P5-01 |

## P5-01: Define typed Agent-to-Agent protocol
**Desired capability:** Replace fragile conversational delegation with structured contracts.
**Prerequisites:** P4-10
**Reuse/integration points:** External task contracts aligned with DCLab TaskGraph concepts.

### Engineering tasks
- [ ] **01** Define TaskRequest containing source/target agent and entity IDs, capability, input evidence refs, constraints, budget, principal, correlation ID and expected output schema.
- [ ] **02** Define TaskResult containing status, typed findings, artifact refs, execution receipt, cost, warning codes and provenance.
- [ ] **03** Specify schema and version negotiation, idempotency, cancellation and authorization context propagation.
- [ ] **04** Separate a mere analysis proposal from an execution request with side effects.
- [ ] **05** Validate forged or stale task requests and reject unknown schema versions.

### Outputs and checks
- Artifacts: AgentTaskRequest/Result schemas, versioned protocol and fixtures.
- Acceptance: Unknown target, incompatible version, missing approval or changed arguments fail deterministically before execution.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P5-02: Implement deterministic capability discovery and routing
**Desired capability:** Select agents from declared capabilities and relations, not invented capabilities.
**Prerequisites:** P5-01
**Reuse/integration points:** Template registry, relationship graph and guarded tool registry.

### Engineering tasks
- [ ] **01** Index available Model/Problem/Project instances with entity bindings and allowed capability IDs.
- [ ] **02** Define a task router that resolves exact target using relation graph, task type and user scope.
- [ ] **03** Permit LLM semantic interpretation only when deterministic resolution leaves genuine ambiguity.
- [ ] **04** Return conflict or multiple-choice state rather than silently picking a model for ambiguous requests.
- [ ] **05** Log selected route and rejected candidate reasons for evaluation.

### Outputs and checks
- Artifacts: AgentRouter, capability index and routing decisions.
- Acceptance: Wrong-agent routing cases are caught; authorization is checked again by the target harness.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
