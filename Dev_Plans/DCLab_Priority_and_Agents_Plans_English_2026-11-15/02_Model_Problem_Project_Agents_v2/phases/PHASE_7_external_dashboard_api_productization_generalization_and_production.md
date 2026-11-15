# Phase 7: External Dashboard, API Productization, Generalization and Production
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Ship a standalone customer-facing product and verify that scale and entity variation do not require new agent or harness implementations.
**Entry criteria:** G6 proven events; three reusable templates and data-bound agent instances are operational.
**Exit gate G7:** Standalone external platform passes tenant isolation, reusable onboarding of five models/three problems/two projects, end-to-end scenario and release readiness.
**Mandatory live demonstration:** Onboard 5 model agents, 3 problem agents and 2 project agents using manifests and adapters only; inspect dashboard and run governed model/problem/project workflows.
**Important boundary:** Charging, billing tiers and broad enterprise integrations are deferred unless chosen later; build observability and security first. External platform must not require DCLab core modifications to serve basic views.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 7A | External experience and integration contracts | P7-01, P7-02, P7-03 | Implement in listed order unless explicitly parallel-safe |
| 7B | Enterprise and runtime hardening | P7-04, P7-05, P7-06 | Implement in listed order unless explicitly parallel-safe |
| 7C | Reuse validation, docs and release | P7-07, P7-08, P7-09, P7-10 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 7A: External experience and integration contracts
**Goal:** Complete P7-01, P7-02, P7-03 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P7-01 - Deliver complete external dashboard information architecture
**Outcome and scope.** Present the three agent layers and their lifecycle as a coherent product.
**Preconditions/dependencies.** P6-09
**Existing integration to inspect.** External React or comparable UI stack; no dependency on DCLab studio frontend.

**Implementation steps (ordered):
1. Create Project, Problem and Model home pages with persistent navigation and breadcrumbs.
2. Build chat/workspace, solution portfolio, model lineage/evaluation, tasks, approvals, monitoring and evidence explorer.
3. Show agent kind, template version, current binding, resource scope and executable capabilities.
4. Add task-run timeline, blocking conditions, cost and permission states in detail views.
5. Validate experience with ML engineer and stakeholder walkthroughs.

**Artifacts and interfaces:** Production UI design system, routes and accessibility tests.

**Acceptance tests and evidence:** A user can locate any current entity, track changes, and drill from a decision to underlying data without relying solely on chat.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-02 - Version public APIs and publish integration SDK
**Outcome and scope.** Make the platform usable by apps and external automation.
**Preconditions/dependencies.** P7-01
**Existing integration to inspect.** Existing external v1 shell from Phase 1.

**Implementation steps (ordered):
1. Publish stable OpenAPI for templates, instances, projects, problems, models, solutions, tasks, approvals, events and evidence.
2. Define error codes, pagination, filters, optimistic versioning and event/idempotency request headers.
3. Provide Python SDK and basic TypeScript usage examples for common read and delegated operations.
4. Add contract tests against fake DCLab integration and controlled real-adapter environment.
5. Document backwards compatibility, deprecation and API authentication.

**Artifacts and interfaces:** OpenAPI v1, SDK reference and interoperability suite.

**Acceptance tests and evidence:** A separate sample client creates project/problem/model instances, lists evidence and watches a task through supported APIs.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-03 - Implement outbound events, webhooks and connector packaging
**Outcome and scope.** Allow optional data-provider and customer integrations.
**Preconditions/dependencies.** P7-02
**Existing integration to inspect.** DCLab APIs and external adapters.

**Implementation steps (ordered):
1. Create signed webhooks for task status, evaluation completed, monitoring alert, approval and solution decision.
2. Implement delivery retries, dedupe, ordering guidance and tenant-scoped webhook secrets.
3. Package DCLab provider as opt-in adapter with feature discovery and compatibility checks.
4. Define connector test suite for timeouts, revoked credentials and partially supported capabilities.
5. Document no-direct-DB-access rule between external platform and DCLab.

**Artifacts and interfaces:** Webhook delivery service, connector manifest, integration examples.

**Acceptance tests and evidence:** DCLab downtime reduces only provider-dependent operations; external identity, UI and audits remain available.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 7B: Enterprise and runtime hardening
**Goal:** Complete P7-04, P7-05, P7-06 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P7-04 - Harden tenant security, privacy and secret management
**Outcome and scope.** Prove isolation at API, agent, tool, data and event layers.
**Preconditions/dependencies.** P7-03
**Existing integration to inspect.** External policy and identity system; DCLab ACL is not substituted.

**Implementation steps (ordered):
1. Run threat model focused on prompt injection, permission confusion, cross-tenant retrieval, malicious tool output and webhook spoofing.
2. Enforce least-privileged provider credentials, secret rotation, encryption and audit retention policies.
3. Verify policy at ingress and connector call, including job resumptions and background events.
4. Add PII redaction/data minimization controls and deletion/export workflow for customer-owned information.
5. Perform adversarial tests and formal review of high-impact approval path.

**Artifacts and interfaces:** Security review, penetration-test backlog, privacy controls.

**Acceptance tests and evidence:** No cross-tenant data access in test matrix; compromised tool output cannot authorize action or expose credentials.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-05 - Harden reliability, performance and failure isolation
**Outcome and scope.** Keep agent processes stable with many registered instances.
**Preconditions/dependencies.** P7-04
**Existing integration to inspect.** External jobs, state, tracing, caching and adapter layers.

**Implementation steps (ordered):
1. Measure concurrent runs, queue delays, slow providers, embedding retrieval and dashboard response under load.
2. Set SLOs for API availability, evidence lookup, queued job start and supported read-only request completion.
3. Introduce rate limits, resource quotas, backpressure, circuit breakers and cache invalidation.
4. Test backup/restore, migration rollback, service restart and disaster recovery procedures.
5. Define cost limits for model calls, storage, data transfers and per-tenant work.

**Artifacts and interfaces:** Performance benchmark, capacity plan, runbooks and recovery drills.

**Acceptance tests and evidence:** API stays responsive under worker saturation; data restored with consistent entity/decision histories.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-06 - Implement reproducible CI/CD and versioned releases
**Outcome and scope.** Make templates and harness changes safe to upgrade.
**Preconditions/dependencies.** P7-05
**Existing integration to inspect.** Template Registry and external deployment stack.

**Implementation steps (ordered):
1. Set CI quality gates: lint, typing, schema migration, contract, unit, integration, security and sealed scenario tests.
2. Build staging/production promotion with signed artifacts and pinned template versions.
3. Add canary rollout and safe rollback for template/harness releases independent of agent instance state.
4. Preserve compatibility of manifests, saved execution checkpoints and public API versions.
5. Define release notes and audit trail of which template version produced each recommendation.

**Artifacts and interfaces:** CI/CD pipelines, release manifest and rollback procedure.

**Acceptance tests and evidence:** An older pinned agent remains runnable or has an explicit migration path; failed template rollout can be reversed.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 7C: Reuse validation, docs and release
**Goal:** Complete P7-07, P7-08, P7-09, P7-10 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P7-07 - Prove template reuse on 5 models, 3 problems and 2 projects
**Outcome and scope.** Validate the primary economic and engineering requirement.
**Preconditions/dependencies.** P7-06
**Existing integration to inspect.** Three shipped templates and harnesses, factory, adapter SDK.

**Implementation steps (ordered):
1. Onboard five distinct model instances including two algorithm families and at least one different task-kind adapter where supported.
2. Create three problem instances with independent objectives, alternative solutions and histories.
3. Create two project instances coordinating multiple problems and shared model links.
4. Require all instantiation via manifests, bindings, seed data and supported adapters; no new `Agent` or `Harness` class.
5. Measure onboarding effort, changed files, number of reusable modules and cost per new entity.

**Artifacts and interfaces:** Reuse-test fixtures, onboarding scripts and measured compatibility report.

**Acceptance tests and evidence:** 5/3/2 gate passes without per-entity harness/agent class; any new adapter is versioned and generic to a capability or model kind.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-08 - Benchmark accuracy, authorization, task completion and cost
**Outcome and scope.** Ensure the product beats simpler alternatives where it should.
**Preconditions/dependencies.** P7-07
**Existing integration to inspect.** Sealed tests and benchmark runner created in P0-06 and expanded each phase.

**Implementation steps (ordered):
1. Run sealed QA and tool scenarios for each agent, plus full end-to-end collaboration and event cases.
2. Report completion, factual support, argument validation, tool routing, policy denials, unneeded steps, latency and total cost.
3. Compare against direct deterministic tools and single-agent baseline; document meaningful trade-offs.
4. Run negative tests for unsupported providers, missing data, misleading conversation context and leakage traps.
5. Publish readiness threshold table and unsolved defects rather than burying failures.

**Artifacts and interfaces:** Comparative benchmark and production readiness scorecard.

**Acceptance tests and evidence:** All critical security/safety cases pass; performance and value thresholds are explicitly approved before launch.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-09 - Complete customer onboarding, operator docs and governance
**Outcome and scope.** Make installation, support and responsible use repeatable.
**Preconditions/dependencies.** P7-08
**Existing integration to inspect.** All phase artifacts and current external UI.

**Implementation steps (ordered):
1. Write documentation for tenant setup, data rights, DCLab connection, manifest creation and approval policies.
2. Create standard runbooks for agent onboarding, model monitoring, retraining, decision revisions and incident response.
3. Publish user guides for Model, Problem and Project Agents and dashboard terminology.
4. Document supported/unsupported algorithms, connectors and business actions.
5. Create migration guide from older two-harness roadmap to three independent specialized templates.

**Artifacts and interfaces:** Installation guides, API docs, user manuals and operations playbook.

**Acceptance tests and evidence:** New engineer can onboard a supported model/problem/project from docs alone without changing platform source.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P7-10 - Run production cutover and post-launch controls
**Outcome and scope.** Release only what is tested and controllable.
**Preconditions/dependencies.** P7-09
**Existing integration to inspect.** All prior phases.

**Implementation steps (ordered):
1. Define rollout cohort, owner/operator roster, alerting and rollback decision thresholds.
2. Run final smoke: create entities, query evidence, delegate analysis, request approval, replay event and inspect outcome.
3. Verify data/privacy agreements, backups, security review, API compatibility and observability dashboards.
4. Sign release gates with evidence refs and explicit scope exclusions, including no automatic production deployment.
5. Establish measured improvement backlog and quarterly review of template specialization versus configuration sprawl.

**Artifacts and interfaces:** G7 release sign-off, tagged binaries and live operations checklist.

**Acceptance tests and evidence:** 5/3/2 acceptance, safety, docs, rollback and incident operations are verified; unimplemented capabilities are disclosed.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 10 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Onboard 5 model agents, 3 problem agents and 2 project agents using manifests and adapters only; inspect dashboard and run governed model/problem/project workflows.
- [ ] G7 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
G7 provides the independently deployable external product, onboarding suite, and operational runbooks.
