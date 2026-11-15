# Implementation Subplan 0A: Repository and architecture baseline
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 0: Foundation, DCLab Gap Audit and Reference Scenario
**Phase exit gate:** G0
**Mission:** Create a validated, provenance-aware specification and reproducible churn reference before coding the external platform.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P0-01` | Inventory existing DCLab capabilities | `capability_inventory.md`, baseline test report, reuse/extend/build ledger. | Previous approved package and phase prerequisites |
| `P0-02` | Ratify independent-service architecture | ADRs 001–006, dependency boundary diagram, service inventory. | Previous approved package and phase prerequisites |
| `P0-03` | Domain model and entity cardinalities | ER diagram, Pydantic contracts, entity state-transition specification. | Previous approved package and phase prerequisites |

## P0-01: Inventory existing DCLab capabilities
**Desired capability:** Establish exact boundaries for reuse, adaptation and new external code.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** Existing DCLab `agents`, `studio`, `dclab_core`, `jobs`, `memory`, `storage`, `agent_eval`.

### Engineering tasks
- [ ] **01** Inspect `dclab_rnd/agents/{harness,runtime,registry}.py`, `agentic/`, `jobs/`, `memory/`, `storage/` and relevant API routes; record public contracts and tests.
- [ ] **02** Inspect `dclab_core/{model_evaluation,monitoring,task_graph,task_execution_plan,task_validation}.py` and Studio evaluation, lineage and retraining code.
- [ ] **03** Classify each capability as reusable by API, reusable through an adapter, unsuitable, or missing; state assumptions and actual evidence.
- [ ] **04** Run existing unit and integration checks in a local checkout, capture versions and failures without modifying the system.
- [ ] **05** Publish compatibility matrix with source path, owner, failure mode, and planned external adapter.

### Outputs and checks
- Artifacts: `capability_inventory.md`, baseline test report, reuse/extend/build ledger.
- Acceptance: Each reused capability has a confirmed contract and test evidence; no proposed external behavior silently depends on an untested internal function.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P0-02: Ratify independent-service architecture
**Desired capability:** Make external-service and three-template boundaries explicit.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** Earlier v1.1 plan is superseded; preserve existing DCLab semantics.

### Engineering tasks
- [ ] **01** Write ADRs: separate deployable/API, independent database and credentials; DCLab is an optional integration provider.
- [ ] **02** Define a shared infrastructure kernel plus three specialized harness implementations: Model, Problem and Project.
- [ ] **03** Define identity semantics: one model-family agent, one problem agent per problem, one project agent per project; distinct entity versions.
- [ ] **04** Specify the compatibility rule: reuse concepts and adapters, not private DCLab runtime internals as an unconditional dependency.
- [ ] **05** Approve an incremental migration path that does not change existing Case Agent, studio gates, or deployed behavior.

### Outputs and checks
- Artifacts: ADRs 001–006, dependency boundary diagram, service inventory.
- Acceptance: Architecture review confirms three agent/harness templates, no cloned harness per instance, and DCLab running unchanged when external service is down.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P0-03: Domain model and entity cardinalities
**Desired capability:** Prevent Project, Problem, Solution, and Model scopes from becoming ambiguous.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** DCLab problem cards, `ProblemMap`, `SolutionPlan`, Studio project records.

### Engineering tasks
- [ ] **01** Specify Workspace, Project, Problem, Solution, ModelFamily, ModelVersion, DataAsset, FeatureContract, Experiment and AgentInstance IDs.
- [ ] **02** Write cardinalities: project 1..N problems; problem 0..N candidate solutions; one solution may reference multiple models; a model may support multiple problems with scoped grants.
- [ ] **03** Define provenance references, immutable revisions, current-alias pointers, and status transitions for each lifecycle.
- [ ] **04** Define transfer of confirmed Case Agent intake into project/problem creation without fabricating historical decisions.
- [ ] **05** Model authorization and deletion/tombstoning behavior for shared models and cross-project links.

### Outputs and checks
- Artifacts: ER diagram, Pydantic contracts, entity state-transition specification.
- Acceptance: Contract validators reject ambiguous IDs, cross-workspace links, missing versions and invalid transitions.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
