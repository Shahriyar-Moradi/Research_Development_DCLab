# Phase 1: Shared Infrastructure, Template Registry and Agent Factory
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Deliver an independent service runtime that can instantiate all three agent types from configuration, with no new agent/harness classes per entity.
**Entry criteria:** G0 approved schemas, policy matrix, pilot dataset and integration spike.
**Exit gate G1:** Model, Problem and Project instances can be created from versioned manifests; all route through one core executor and distinct scoped specialized harnesses.
**Mandatory live demonstration:** Instantiate three fake agents from YAML and run read-only typed tools; restart the service and resume a job without losing state.
**Important boundary:** Three templates are first-class versioned artifacts; harnesses own process-specific policy/state logic. Shared runtime modules must not contain entity-specific if/else chains.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 1A | Service skeleton and canonical state | P1-01, P1-02 | Implement in listed order unless explicitly parallel-safe |
| 1B | Templates, agent factory and harness framework | P1-03, P1-04, P1-05, P1-06 | Implement in listed order unless explicitly parallel-safe |
| 1C | Tools, memory, jobs and protections | P1-07, P1-08, P1-09, P1-10, P1-11 | Implement in listed order unless explicitly parallel-safe |
| 1D | Developer-facing contracts and shell | P1-12 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 1A: Service skeleton and canonical state
**Goal:** Complete P1-01, P1-02 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P1-01 - Bootstrap standalone external service
**Outcome and scope.** Establish a separately deployable API/worker/UI workspace.
**Preconditions/dependencies.** P0-02
**Existing integration to inspect.** New external service; existing DCLab does not change.

**Implementation steps (ordered):
1. Create Python package, FastAPI app factory, dependency injection container, environment schema and health endpoints.
2. Create directories for `core`, `templates`, `harnesses`, `agents`, `adapters`, `domain`, `api`, `workers`, and `tests`.
3. Implement local Docker Compose or equivalent for API, PostgreSQL and a worker; keep separate DB credentials from DCLab.
4. Add log correlation, feature flags, readiness checks and graceful shutdown.
5. Document local startup with fake external integrations so no DCLab credentials are required.

**Artifacts and interfaces:** Bootable API, worker and local dev stack.

**Acceptance tests and evidence:** Cold start/health smoke tests pass with mocked DCLab provider and no LLM credentials.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-02 - Implement authoritative persistence and migrations
**Outcome and scope.** Persist identity, versions and relations without using chat messages as database records.
**Preconditions/dependencies.** P0-03
**Existing integration to inspect.** External PostgreSQL; DCLab storage mapped only through adapters.

**Implementation steps (ordered):
1. Create database tables for tenants, projects, problems, solutions, model families/versions, agent identities and template versions.
2. Add relations, artifact/evidence refs, conversations, decisions, executions, approvals and audit records.
3. Apply foreign keys, immutable revision constraints, unique tenant-scoped IDs and optimistic concurrency.
4. Implement transaction boundaries for active revision pointers and create/update operations.
5. Write fixtures, reversible migrations and a backfill strategy for externally linked DCLab entities.

**Artifacts and interfaces:** SQL migrations, repositories, model factories and seed fixtures.

**Acceptance tests and evidence:** Wrong-tenant reads fail; invalid state edges roll back; migration up/down and restart persistence tests pass.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 1B: Templates, agent factory and harness framework
**Goal:** Complete P1-03, P1-04, P1-05, P1-06 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P1-03 - Create versioned Agent Template Registry
**Outcome and scope.** Maintain exactly three first-class template families with extensions, not copies.
**Preconditions/dependencies.** P1-01
**Existing integration to inspect.** Informed by DCLab Agent registry; separate external canonical implementation.

**Implementation steps (ordered):
1. Define TemplateSpec including type, template version, schema version, instruction renderer, tool allowlist and lifecycle contract.
2. Implement `model`, `problem` and `project` templates with separate default manifests and policies.
3. Validate template/manifest compatibility and semantic version constraints at startup and instantiation.
4. Implement template evolution and migration strategy preserving previously created instances.
5. Expose registry list, inspect and validate operations, and prohibit unsigned arbitrary plugin loading.

**Artifacts and interfaces:** Template registry, version resolution, manifest validation CLI/API.

**Acceptance tests and evidence:** Older agent instances resolve pinned template version; new entity creation changes only manifest and bindings.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-04 - Implement Agent Factory and identity lifecycle
**Outcome and scope.** Make agents persistent logical resources rather than always-running LLM processes.
**Preconditions/dependencies.** P1-02, P1-03
**Existing integration to inspect.** All three template families; no per-model generated Python code.

**Implementation steps (ordered):
1. Implement `create_instance(template_id, entity_ref, bindings, policy_ref, actor)` with idempotent response.
2. Validate entity existence, permissions, manifest version and capability compatibility.
3. Persist agent ID, owner, linked entity, template version, current status and activation policy.
4. Add activate, suspend, resume, archive and inspect operations; keep run state separate.
5. Implement factory hooks for typed context loading and specialized harness resolution.

**Artifacts and interfaces:** Agent Factory, instance CRUD, lifecycle events and seed onboarding scripts.

**Acceptance tests and evidence:** Two instances of the same type share the same class/harness binaries while retaining isolated scope and state.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-05 - Implement shared execution kernel
**Outcome and scope.** Build common turn/tool/task orchestration for every harness.
**Preconditions/dependencies.** P1-02, P1-07
**Existing integration to inspect.** Concepts from `dclab_rnd/agents/runtime.py` and existing task validator.

**Implementation steps (ordered):
1. Define bounded execution loop, capability resolution, tool-call validation and result envelopes.
2. Add budget limits for calls, tokens, time, side effects and recursive delegation.
3. Persist run IDs, correlation IDs, checkpoints and resumable execution pointers.
4. Make read-only versus effectful steps explicit; an LLM proposal never implies action authorization.
5. Offer a deterministic state-machine path; allow LangGraph adapter only if it improves measured performance.

**Artifacts and interfaces:** Core Executor, RunState and typed tool dispatcher.

**Acceptance tests and evidence:** Every visible response and tool use is traced; step/time limits terminate safely; no scientific metric computed by LLM.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-06 - Implement three specialized Harness contracts
**Outcome and scope.** Keep process ownership distinct without duplicating basic execution infrastructure.
**Preconditions/dependencies.** P1-03, P1-05
**Existing integration to inspect.** External implementation inspired by existing DCLab harness patterns.

**Implementation steps (ordered):
1. Define `Harness` protocol: bind_context, available_capabilities, plan, validate, execute, summarize, checkpoint.
2. Implement `ModelHarness` state transitions around versions, evaluation, monitoring, challenger and deployment proposal.
3. Implement `ProblemHarness` transitions around problem scope, hypotheses, solutions, analysis and validation.
4. Implement `ProjectHarness` transitions around portfolio, resources, decisions, cross-problem coordination and outcomes.
5. Add independent policy packages and tests for each; common kernel remains unaware of business-specific transitions.

**Artifacts and interfaces:** Three harness implementations and state-machine matrices.

**Acceptance tests and evidence:** Attempting a Problem transition with Model Harness is refused; adding a new churn model requires no harness subclass.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 1C: Tools, memory, jobs and protections
**Goal:** Complete P1-07, P1-08, P1-09, P1-10, P1-11 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P1-07 - Build typed Capability and Tool Registry
**Outcome and scope.** Give all three harnesses a single safe dispatch surface.
**Preconditions/dependencies.** P0-08
**Existing integration to inspect.** DCLab registry schema and guards are a design reference.

**Implementation steps (ordered):
1. Define tools with id, JSON/Pydantic schema, effect class, principal scope, timeout, connector, idempotency behavior and result schema.
2. Implement capability-discovery filtered by entity permissions and template allowlist.
3. Normalize provider errors, not-found, untrusted output and partial results; store evidence refs rather than raw sensitive payloads.
4. Create adapters for external DCLab facts, evaluation, job and dataset metadata; add fake in-memory adapters.
5. Test input validation, unsupported tool kinds and denial of unknown effects.

**Artifacts and interfaces:** Capability registry and integration adapter SDK.

**Acceptance tests and evidence:** No unregistered tool executes; unsupported capability surfaces a typed unsupported result, never a hallucinated success.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-08 - Implement context assembly and memory layers
**Outcome and scope.** Make all relevant knowledge retrievable without overloading prompts.
**Preconditions/dependencies.** P1-02, P1-10
**Existing integration to inspect.** External store; DCLab memory readers available via scoped adapter.

**Implementation steps (ordered):
1. Define context bundle by workspace/entity/version, allowed relationship graph, pinned authoritative facts, recent events and requested task.
2. Persist conversation turns and semantic summaries separately from canonical data, metrics and decisions.
3. Implement bounded retrieval ranking by relevance, freshness, provenance and permission scope.
4. Add optional semantic search via pgvector only after evaluating recall and cost against keyword/structured retrieval.
5. Create contradiction and supersession rules; summaries never overwrite signed facts.

**Artifacts and interfaces:** ContextResolver, RetrievalStore, memory freshness and provenance tests.

**Acceptance tests and evidence:** The agent correctly reports missing facts and ignores inaccessible or superseded memories; no raw dataset bulk injected into prompts.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-09 - Implement durable job and approval-aware workflow execution
**Outcome and scope.** Handle long-running evaluation and retraining without busy LLM loops.
**Preconditions/dependencies.** P1-02, P1-10
**Existing integration to inspect.** Patterned after `dclab_rnd/jobs/store.py` and `handlers.py`.

**Implementation steps (ordered):
1. Create jobs with states queued/running/waiting_approval/completed/failed/cancelled and durable checkpoint payload.
2. Add idempotency keys, retries with caps, attempt IDs, result artifact refs and resume semantics.
3. Integrate approvals into the job state machine so actions never proceed on an expired or wrong-scope approval.
4. Support cancellation, timeouts, heartbeat, stuck-job recovery and duplicate request suppression.
5. Map DCLab long jobs into external statuses through a connector, keeping provider IDs.

**Artifacts and interfaces:** Job API, worker adapter, approval state machine and test harness.

**Acceptance tests and evidence:** Kill/restart does not duplicate effectful work; approval refusal prevents invocation of provider write endpoint.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-10 - Implement authorization and human-approval boundaries
**Outcome and scope.** Enforce least privilege at every connector and tool invocation.
**Preconditions/dependencies.** P0-07
**Existing integration to inspect.** Policy concepts from existing Studio graph validator; external source of truth is independent.

**Implementation steps (ordered):
1. Resolve authenticated principal, tenant, grants, entity memberships and resource scopes for each request.
2. Enforce risk-based policies before dispatch with default deny for effects and sensitive reads.
3. Create approval requests bound to exact arguments, task fingerprint, actor, deadline and target entity.
4. Re-check privileges on resumption; prohibit prompt injection from changing policy.
5. Add audit receipts and tests covering cross-tenant access, tool argument escalation and approval replay.

**Artifacts and interfaces:** Policy evaluator, approval APIs, signed or integrity-protected receipts.

**Acceptance tests and evidence:** Negative authorization tests run independently of LLM; no actuation can bypass guard with a direct tool call.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P1-11 - Implement observability, evidence and reproducibility
**Outcome and scope.** Make every agent result traceable to verified inputs.
**Preconditions/dependencies.** P1-05, P1-07
**Existing integration to inspect.** DCLab tracing architecture as reference; no cross-database direct joins.

**Implementation steps (ordered):
1. Emit spans for runs, tool calls, connector requests and long-running jobs with correlation IDs.
2. Build EvidenceRef and EvidenceBundle linking versioned data, evaluations, conversations and policy decisions.
3. Capture cost, latency, selected capabilities, authorization results and retry counts.
4. Create a deterministic replay stub for mocked tools and a redacted diagnostic export.
5. Define monitoring and dashboards for tool failures, queue age, denial counts and agent accuracy.

**Artifacts and interfaces:** Trace schema, evidence resolver, telemetry pipeline and replay fixtures.

**Acceptance tests and evidence:** Every numerical claim in a test answer resolves to a canonical artifact or is marked unsupported.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 1D: Developer-facing contracts and shell
**Goal:** Complete P1-12 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P1-12 - Expose minimal API, SDK and proof-of-reuse scenario
**Outcome and scope.** Give downstream phase developers stable operations and examples.
**Preconditions/dependencies.** P1-04, P1-06, P1-11
**Existing integration to inspect.** New external frontend shell, built before full product dashboard.

**Implementation steps (ordered):
1. Expose template listing, instance creation, run submission, job lookup, approved tool list and evidence fetch endpoints.
2. Publish typed OpenAPI schemas, Python client examples and three starter manifests.
3. Create a minimal non-production dashboard for agent identity/run status and a trace viewer.
4. Write contract tests using fake DCLab adapters to instantiate one of each agent kind.
5. Measure cold-start and run overhead without LLM and record baseline budgets.

**Artifacts and interfaces:** Versioned v1 API, CLI/SDK examples, G1 demo.

**Acceptance tests and evidence:** Three instances with isolated records and shared kernel can be created, inspected, and resumed via API.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 12 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Instantiate three fake agents from YAML and run read-only typed tools; restart the service and resume a job without losing state.
- [ ] G1 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G1 provides the stable interfaces, immutable evidence and regression tests required by Phase 2.

---

## Download or open each implementation subplan separately

- [Implementation Subplan 1A: Service skeleton and canonical state](../plans/PLAN_1A_service_skeleton_and_canonical_state.md)
- [Implementation Subplan 1B: Templates, agent factory and harness framework](../plans/PLAN_1B_templates_agent_factory_and_harness_framework.md)
- [Implementation Subplan 1C: Tools, memory, jobs and protections](../plans/PLAN_1C_tools_memory_jobs_and_protections.md)
- [Implementation Subplan 1D: Developer-facing contracts and shell](../plans/PLAN_1D_developer_facing_contracts_and_shell.md)

These subplans are independent implementation documents containing work packages, detailed engineering tasks, outputs, tests, and acceptance criteria.
