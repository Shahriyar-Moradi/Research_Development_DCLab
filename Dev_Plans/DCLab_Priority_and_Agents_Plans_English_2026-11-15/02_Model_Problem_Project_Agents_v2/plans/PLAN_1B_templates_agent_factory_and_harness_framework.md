# Implementation Subplan 1B: Templates, agent factory and harness framework
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 1: Shared Infrastructure, Template Registry and Agent Factory
**Phase exit gate:** G1
**Mission:** Deliver an independent service runtime that can instantiate all three agent types from configuration, with no new agent/harness classes per entity.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P1-03` | Create versioned Agent Template Registry | Template registry, version resolution, manifest validation CLI/API. | P1-01 |
| `P1-04` | Implement Agent Factory and identity lifecycle | Agent Factory, instance CRUD, lifecycle events and seed onboarding scripts. | P1-02, P1-03 |
| `P1-05` | Implement shared execution kernel | Core Executor, RunState and typed tool dispatcher. | P1-02, P1-07 |
| `P1-06` | Implement three specialized Harness contracts | Three harness implementations and state-machine matrices. | P1-03, P1-05 |

## P1-03: Create versioned Agent Template Registry
**Desired capability:** Maintain exactly three first-class template families with extensions, not copies.
**Prerequisites:** P1-01
**Reuse/integration points:** Informed by DCLab Agent registry; separate external canonical implementation.

### Engineering tasks
- [ ] **01** Define TemplateSpec including type, template version, schema version, instruction renderer, tool allowlist and lifecycle contract.
- [ ] **02** Implement `model`, `problem` and `project` templates with separate default manifests and policies.
- [ ] **03** Validate template/manifest compatibility and semantic version constraints at startup and instantiation.
- [ ] **04** Implement template evolution and migration strategy preserving previously created instances.
- [ ] **05** Expose registry list, inspect and validate operations, and prohibit unsigned arbitrary plugin loading.

### Outputs and checks
- Artifacts: Template registry, version resolution, manifest validation CLI/API.
- Acceptance: Older agent instances resolve pinned template version; new entity creation changes only manifest and bindings.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-04: Implement Agent Factory and identity lifecycle
**Desired capability:** Make agents persistent logical resources rather than always-running LLM processes.
**Prerequisites:** P1-02, P1-03
**Reuse/integration points:** All three template families; no per-model generated Python code.

### Engineering tasks
- [ ] **01** Implement `create_instance(template_id, entity_ref, bindings, policy_ref, actor)` with idempotent response.
- [ ] **02** Validate entity existence, permissions, manifest version and capability compatibility.
- [ ] **03** Persist agent ID, owner, linked entity, template version, current status and activation policy.
- [ ] **04** Add activate, suspend, resume, archive and inspect operations; keep run state separate.
- [ ] **05** Implement factory hooks for typed context loading and specialized harness resolution.

### Outputs and checks
- Artifacts: Agent Factory, instance CRUD, lifecycle events and seed onboarding scripts.
- Acceptance: Two instances of the same type share the same class/harness binaries while retaining isolated scope and state.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-05: Implement shared execution kernel
**Desired capability:** Build common turn/tool/task orchestration for every harness.
**Prerequisites:** P1-02, P1-07
**Reuse/integration points:** Concepts from `dclab_rnd/agents/runtime.py` and existing task validator.

### Engineering tasks
- [ ] **01** Define bounded execution loop, capability resolution, tool-call validation and result envelopes.
- [ ] **02** Add budget limits for calls, tokens, time, side effects and recursive delegation.
- [ ] **03** Persist run IDs, correlation IDs, checkpoints and resumable execution pointers.
- [ ] **04** Make read-only versus effectful steps explicit; an LLM proposal never implies action authorization.
- [ ] **05** Offer a deterministic state-machine path; allow LangGraph adapter only if it improves measured performance.

### Outputs and checks
- Artifacts: Core Executor, RunState and typed tool dispatcher.
- Acceptance: Every visible response and tool use is traced; step/time limits terminate safely; no scientific metric computed by LLM.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-06: Implement three specialized Harness contracts
**Desired capability:** Keep process ownership distinct without duplicating basic execution infrastructure.
**Prerequisites:** P1-03, P1-05
**Reuse/integration points:** External implementation inspired by existing DCLab harness patterns.

### Engineering tasks
- [ ] **01** Define `Harness` protocol: bind_context, available_capabilities, plan, validate, execute, summarize, checkpoint.
- [ ] **02** Implement `ModelHarness` state transitions around versions, evaluation, monitoring, challenger and deployment proposal.
- [ ] **03** Implement `ProblemHarness` transitions around problem scope, hypotheses, solutions, analysis and validation.
- [ ] **04** Implement `ProjectHarness` transitions around portfolio, resources, decisions, cross-problem coordination and outcomes.
- [ ] **05** Add independent policy packages and tests for each; common kernel remains unaware of business-specific transitions.

### Outputs and checks
- Artifacts: Three harness implementations and state-machine matrices.
- Acceptance: Attempting a Problem transition with Model Harness is refused; adding a new churn model requires no harness subclass.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
