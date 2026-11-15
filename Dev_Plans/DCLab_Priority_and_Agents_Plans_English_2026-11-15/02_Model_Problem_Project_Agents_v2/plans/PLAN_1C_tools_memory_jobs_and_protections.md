# Implementation Subplan 1C: Tools, memory, jobs and protections
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 1: Shared Infrastructure, Template Registry and Agent Factory
**Phase exit gate:** G1
**Mission:** Deliver an independent service runtime that can instantiate all three agent types from configuration, with no new agent/harness classes per entity.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P1-07` | Build typed Capability and Tool Registry | Capability registry and integration adapter SDK. | P0-08 |
| `P1-08` | Implement context assembly and memory layers | ContextResolver, RetrievalStore, memory freshness and provenance tests. | P1-02, P1-10 |
| `P1-09` | Implement durable job and approval-aware workflow execution | Job API, worker adapter, approval state machine and test harness. | P1-02, P1-10 |
| `P1-10` | Implement authorization and human-approval boundaries | Policy evaluator, approval APIs, signed or integrity-protected receipts. | P0-07 |
| `P1-11` | Implement observability, evidence and reproducibility | Trace schema, evidence resolver, telemetry pipeline and replay fixtures. | P1-05, P1-07 |

## P1-07: Build typed Capability and Tool Registry
**Desired capability:** Give all three harnesses a single safe dispatch surface.
**Prerequisites:** P0-08
**Reuse/integration points:** DCLab registry schema and guards are a design reference.

### Engineering tasks
- [ ] **01** Define tools with id, JSON/Pydantic schema, effect class, principal scope, timeout, connector, idempotency behavior and result schema.
- [ ] **02** Implement capability-discovery filtered by entity permissions and template allowlist.
- [ ] **03** Normalize provider errors, not-found, untrusted output and partial results; store evidence refs rather than raw sensitive payloads.
- [ ] **04** Create adapters for external DCLab facts, evaluation, job and dataset metadata; add fake in-memory adapters.
- [ ] **05** Test input validation, unsupported tool kinds and denial of unknown effects.

### Outputs and checks
- Artifacts: Capability registry and integration adapter SDK.
- Acceptance: No unregistered tool executes; unsupported capability surfaces a typed unsupported result, never a hallucinated success.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-08: Implement context assembly and memory layers
**Desired capability:** Make all relevant knowledge retrievable without overloading prompts.
**Prerequisites:** P1-02, P1-10
**Reuse/integration points:** External store; DCLab memory readers available via scoped adapter.

### Engineering tasks
- [ ] **01** Define context bundle by workspace/entity/version, allowed relationship graph, pinned authoritative facts, recent events and requested task.
- [ ] **02** Persist conversation turns and semantic summaries separately from canonical data, metrics and decisions.
- [ ] **03** Implement bounded retrieval ranking by relevance, freshness, provenance and permission scope.
- [ ] **04** Add optional semantic search via pgvector only after evaluating recall and cost against keyword/structured retrieval.
- [ ] **05** Create contradiction and supersession rules; summaries never overwrite signed facts.

### Outputs and checks
- Artifacts: ContextResolver, RetrievalStore, memory freshness and provenance tests.
- Acceptance: The agent correctly reports missing facts and ignores inaccessible or superseded memories; no raw dataset bulk injected into prompts.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-09: Implement durable job and approval-aware workflow execution
**Desired capability:** Handle long-running evaluation and retraining without busy LLM loops.
**Prerequisites:** P1-02, P1-10
**Reuse/integration points:** Patterned after `dclab_rnd/jobs/store.py` and `handlers.py`.

### Engineering tasks
- [ ] **01** Create jobs with states queued/running/waiting_approval/completed/failed/cancelled and durable checkpoint payload.
- [ ] **02** Add idempotency keys, retries with caps, attempt IDs, result artifact refs and resume semantics.
- [ ] **03** Integrate approvals into the job state machine so actions never proceed on an expired or wrong-scope approval.
- [ ] **04** Support cancellation, timeouts, heartbeat, stuck-job recovery and duplicate request suppression.
- [ ] **05** Map DCLab long jobs into external statuses through a connector, keeping provider IDs.

### Outputs and checks
- Artifacts: Job API, worker adapter, approval state machine and test harness.
- Acceptance: Kill/restart does not duplicate effectful work; approval refusal prevents invocation of provider write endpoint.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-10: Implement authorization and human-approval boundaries
**Desired capability:** Enforce least privilege at every connector and tool invocation.
**Prerequisites:** P0-07
**Reuse/integration points:** Policy concepts from existing Studio graph validator; external source of truth is independent.

### Engineering tasks
- [ ] **01** Resolve authenticated principal, tenant, grants, entity memberships and resource scopes for each request.
- [ ] **02** Enforce risk-based policies before dispatch with default deny for effects and sensitive reads.
- [ ] **03** Create approval requests bound to exact arguments, task fingerprint, actor, deadline and target entity.
- [ ] **04** Re-check privileges on resumption; prohibit prompt injection from changing policy.
- [ ] **05** Add audit receipts and tests covering cross-tenant access, tool argument escalation and approval replay.

### Outputs and checks
- Artifacts: Policy evaluator, approval APIs, signed or integrity-protected receipts.
- Acceptance: Negative authorization tests run independently of LLM; no actuation can bypass guard with a direct tool call.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P1-11: Implement observability, evidence and reproducibility
**Desired capability:** Make every agent result traceable to verified inputs.
**Prerequisites:** P1-05, P1-07
**Reuse/integration points:** DCLab tracing architecture as reference; no cross-database direct joins.

### Engineering tasks
- [ ] **01** Emit spans for runs, tool calls, connector requests and long-running jobs with correlation IDs.
- [ ] **02** Build EvidenceRef and EvidenceBundle linking versioned data, evaluations, conversations and policy decisions.
- [ ] **03** Capture cost, latency, selected capabilities, authorization results and retry counts.
- [ ] **04** Create a deterministic replay stub for mocked tools and a redacted diagnostic export.
- [ ] **05** Define monitoring and dashboards for tool failures, queue age, denial counts and agent accuracy.

### Outputs and checks
- Artifacts: Trace schema, evidence resolver, telemetry pipeline and replay fixtures.
- Acceptance: Every numerical claim in a test answer resolves to a canonical artifact or is marked unsupported.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
