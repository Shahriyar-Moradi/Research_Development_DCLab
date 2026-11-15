# Phase 0: Foundation, DCLab Gap Audit and Reference Scenario
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Create a validated, provenance-aware specification and reproducible churn reference before coding the external platform.
**Entry criteria:** The existing DCLab repository, previous planning documents, team requirements, and a permitted churn dataset are available.
**Exit gate G0:** Approved architecture decision records, tested baseline model, fixture pack, integration map, and backlog estimate with named owners.
**Mandatory live demonstration:** Demonstrate reproducible churn training and an API mock that resolves the same project, problem, solution, and model IDs.
**Important boundary:** Inventory status is not implementation status. A symbol existing in DCLab proves neither that it is stable nor that it meets the new external product contract.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 0A | Repository and architecture baseline | P0-01, P0-02, P0-03 | Implement in listed order unless explicitly parallel-safe |
| 0B | Data, baseline model and evidence | P0-04, P0-05 | Implement in listed order unless explicitly parallel-safe |
| 0C | Quality, security and integration readiness | P0-06, P0-07, P0-08, P0-09 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 0A: Repository and architecture baseline
**Goal:** Complete P0-01, P0-02, P0-03 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P0-01 - Inventory existing DCLab capabilities
**Outcome and scope.** Establish exact boundaries for reuse, adaptation and new external code.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** Existing DCLab `agents`, `studio`, `dclab_core`, `jobs`, `memory`, `storage`, `agent_eval`.

**Implementation steps (ordered):
1. Inspect `dclab_rnd/agents/{harness,runtime,registry}.py`, `agentic/`, `jobs/`, `memory/`, `storage/` and relevant API routes; record public contracts and tests.
2. Inspect `dclab_core/{model_evaluation,monitoring,task_graph,task_execution_plan,task_validation}.py` and Studio evaluation, lineage and retraining code.
3. Classify each capability as reusable by API, reusable through an adapter, unsuitable, or missing; state assumptions and actual evidence.
4. Run existing unit and integration checks in a local checkout, capture versions and failures without modifying the system.
5. Publish compatibility matrix with source path, owner, failure mode, and planned external adapter.

**Artifacts and interfaces:** `capability_inventory.md`, baseline test report, reuse/extend/build ledger.

**Acceptance tests and evidence:** Each reused capability has a confirmed contract and test evidence; no proposed external behavior silently depends on an untested internal function.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P0-02 - Ratify independent-service architecture
**Outcome and scope.** Make external-service and three-template boundaries explicit.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** Earlier v1.1 plan is superseded; preserve existing DCLab semantics.

**Implementation steps (ordered):
1. Write ADRs: separate deployable/API, independent database and credentials; DCLab is an optional integration provider.
2. Define a shared infrastructure kernel plus three specialized harness implementations: Model, Problem and Project.
3. Define identity semantics: one model-family agent, one problem agent per problem, one project agent per project; distinct entity versions.
4. Specify the compatibility rule: reuse concepts and adapters, not private DCLab runtime internals as an unconditional dependency.
5. Approve an incremental migration path that does not change existing Case Agent, studio gates, or deployed behavior.

**Artifacts and interfaces:** ADRs 001–006, dependency boundary diagram, service inventory.

**Acceptance tests and evidence:** Architecture review confirms three agent/harness templates, no cloned harness per instance, and DCLab running unchanged when external service is down.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P0-03 - Domain model and entity cardinalities
**Outcome and scope.** Prevent Project, Problem, Solution, and Model scopes from becoming ambiguous.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** DCLab problem cards, `ProblemMap`, `SolutionPlan`, Studio project records.

**Implementation steps (ordered):
1. Specify Workspace, Project, Problem, Solution, ModelFamily, ModelVersion, DataAsset, FeatureContract, Experiment and AgentInstance IDs.
2. Write cardinalities: project 1..N problems; problem 0..N candidate solutions; one solution may reference multiple models; a model may support multiple problems with scoped grants.
3. Define provenance references, immutable revisions, current-alias pointers, and status transitions for each lifecycle.
4. Define transfer of confirmed Case Agent intake into project/problem creation without fabricating historical decisions.
5. Model authorization and deletion/tombstoning behavior for shared models and cross-project links.

**Artifacts and interfaces:** ER diagram, Pydantic contracts, entity state-transition specification.

**Acceptance tests and evidence:** Contract validators reject ambiguous IDs, cross-workspace links, missing versions and invalid transitions.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 0B: Data, baseline model and evidence
**Goal:** Complete P0-04, P0-05 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P0-04 - Curate churn dataset with training provenance
**Outcome and scope.** Make data readiness the first hard gate.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** DCLab Telco data, dataset catalog, leakage audit and lineage records.

**Implementation steps (ordered):
1. Select permitted Telco churn source; document license, origin URL or ownership, collection time and allowed training use.
2. Pin raw file hash, schema, target, prediction moment, entity key, sensitive fields, label availability and temporal assumptions.
3. Generate validation report for nulls, duplicates, leakage suspects, inconsistent types and invalid targets.
4. Create train/validation/test protocol with held-out evidence, leakage guard and stable random seed or justified time split.
5. Publish dataset card and approved training manifest; exclude unlicensed commercial model outputs or undocumented derived labels.

**Artifacts and interfaces:** Dataset card, SHA-256 manifest, data-quality report, split protocol.

**Acceptance tests and evidence:** Independent rerun loads identical snapshot and split; leaks and licensing violations block use.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P0-05 - Produce versioned baseline model and metric evidence
**Outcome and scope.** Create deterministic reference artifacts for Model Agent read/evaluate workflows.
**Preconditions/dependencies.** P0-04
**Existing integration to inspect.** `general_pipeline`, `studio/lineage.py`, `studio/retrain.py`, `dclab_core/model_evaluation.py`.

**Implementation steps (ordered):
1. Train an initial tabular classifier using existing science pipeline rather than coding a second training implementation.
2. Record preprocessing, feature contract, dataset version, hyperparameters, code commit, seed and training run ID.
3. Evaluate with precision, recall, F1, PR-AUC, ROC-AUC, calibration and chosen threshold on correctly separated windows.
4. Register at least two model versions for subsequent comparison; mark baseline versus candidate status clearly.
5. Export fabricated or approved non-sensitive inference fixtures for API and UI tests.

**Artifacts and interfaces:** Two model versions, model cards, evaluation artifacts, serving contract.

**Acceptance tests and evidence:** A rerun reproduces results within recorded numerical tolerance; metrics include window, threshold, cohort and source ref.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 0C: Quality, security and integration readiness
**Goal:** Complete P0-06, P0-07, P0-08, P0-09 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P0-06 - Specify evaluation and safety benchmarks
**Outcome and scope.** Establish evidence-based gates before agent prompts are tuned.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** Existing `dclab_rnd/agent_eval`, test fixtures and agent trace format.

**Implementation steps (ordered):
1. Create answer cases for exact version retrieval, feature lineage, missing metrics, outdated context and contradictory conversations.
2. Create execution cases for invalid tool arguments, withheld approval, failed job, duplicate event and cross-tenant attempt.
3. Create problem cases that distinguish a rise in business churn from declining model recall.
4. Specify human-scored rubric and deterministic oracle checks for citations, task completion, unauthorized side effects and recovery.
5. Record baselines for direct tools, single agent and three-agent architecture where feasible.

**Artifacts and interfaces:** Sealed scenario pack, scoring rubric, evaluation runner skeleton.

**Acceptance tests and evidence:** Every critical pathway has a negative test and traceable expected outcome; sealed cases are not leaked into prompts.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P0-07 - Define security, tenancy and risk policies
**Outcome and scope.** Create a control matrix that can be enforced outside LLM prompts.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** Reuse DCLab guard concepts, not implicit trust in DCLab sessions.

**Implementation steps (ordered):
1. Identify actor types, workspace ownership, project membership, model sharing and service-to-service identities.
2. Define permissions for read, evaluate, query, create candidate, retrain, adopt solution, deploy and act on production.
3. Mark each capability read-only, experimental write, controlled write or high-impact operation.
4. Specify retention, deletion, PII/secret redaction, audit evidence and cross-tenant isolation rules.
5. Design approval expirations, immutable action receipts and revocation behavior.

**Artifacts and interfaces:** Threat model, RBAC/ABAC matrix, privacy and retention specifications.

**Acceptance tests and evidence:** Unsafe operations are denied in a policy simulation independent of agent outputs; privacy owner approves data paths.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P0-08 - Validate DCLab adapter feasibility
**Outcome and scope.** Prove the external platform can integrate without a fork.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** DCLab APIs, `agentic/routers/projects.py`, `jobs.py`, evaluation and lineage modules.

**Implementation steps (ordered):
1. Identify callable DCLab interfaces for project metadata, model lineage, dataset references, evaluation, retraining and job status.
2. Prototype a versioned HTTP or command adapter with least privilege and an explicit timeout.
3. Specify failure semantics, event mapping, data redaction and idempotent job request receipts.
4. Test with a fake DCLab provider first and a local integration instance second.
5. Document incompatibilities and isolate any necessary bridge behind an adapter rather than coupling the new service to internals.

**Artifacts and interfaces:** Adapter spike, example payloads, integration risk register.

**Acceptance tests and evidence:** External service continues to operate in read-only degraded mode when DCLab is unavailable; no core modifications required for the spike.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P0-09 - Create project delivery and PR conventions
**Outcome and scope.** Turn the architecture into independently testable implementation slices.
**Preconditions/dependencies.** Previous approved package and phase prerequisites
**Existing integration to inspect.** Existing DCLab Makefile/test practices inform new repo conventions.

**Implementation steps (ordered):
1. Assign package IDs, owners, estimate bands, dependencies, branch/PR template, and acceptance criteria.
2. Establish CI jobs for schema validation, formatting, type checks, fast unit tests and integration smoke tests.
3. Define compatibility/version policy for templates, manifests, tools, events and REST APIs.
4. Set quality gates for provenance, policy refusals, rollback and performance limits.
5. Publish a prioritized MVP cut that can demonstrate model questions before all remaining phases finish.

**Artifacts and interfaces:** Backlog tracker, repository contributing guide, PR checklist.

**Acceptance tests and evidence:** Every package has one accountable owner, a failing-test-first definition and a reviewable evidence link.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 9 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Demonstrate reproducible churn training and an API mock that resolves the same project, problem, solution, and model IDs.
- [ ] G0 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G0 provides the stable interfaces, immutable evidence and regression tests required by Phase 1.
