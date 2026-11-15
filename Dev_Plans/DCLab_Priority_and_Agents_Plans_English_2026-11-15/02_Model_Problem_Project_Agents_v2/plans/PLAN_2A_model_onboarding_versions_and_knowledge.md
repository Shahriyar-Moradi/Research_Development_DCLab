# Implementation Subplan 2A: Model onboarding, versions and knowledge
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 2: Model Agent Template and Specialized Model Harness
**Phase exit gate:** G2
**Mission:** Deliver repeatable model-knowledge, evaluation, diagnostics and controlled improvement capabilities for any supported model.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P2-01` | Build ModelFamily/ModelVersion onboarding adapter | Model registry adapter, import API, model agent auto-instantiation. | P1-04, P0-05 |
| `P2-02` | Implement immutable model knowledge and lineage graph | ModelKnowledgeService and citation-bearing DTOs. | P2-01 |
| `P2-03` | Implement Model Agent conversational context and actions | Model conversation API and benchmark prompts. | P2-02 |

## P2-01: Build ModelFamily/ModelVersion onboarding adapter
**Desired capability:** Register models and create agents automatically without per-model classes.
**Prerequisites:** P1-04, P0-05
**Reuse/integration points:** `dclab_rnd/studio/lineage.py`, model cards, MLflow integration `watch_mlflow.py`.

### Engineering tasks
- [ ] **01** Define model-family record with task kind, owning tenant, algorithms, artifact registry references, version aliases, feature schema and lifecycle status.
- [ ] **02** Implement import from DCLab model/evidence records and optional MLflow registry adapter, validating identifiers and artifact hashes.
- [ ] **03** Create or resolve Model Agent instance from the `model` template; never generate custom source code.
- [ ] **04** Record safe backend adapter selection for scikit-learn, LightGBM/XGBoost, PyTorch or unsupported task families.
- [ ] **05** Test duplicate registration, version supersession, missing artifact and cross-project model-sharing grants.

### Outputs and checks
- Artifacts: Model registry adapter, import API, model agent auto-instantiation.
- Acceptance: Onboard two classifiers using one template/harness; unsupported format reports capability gap rather than failing silently.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-02: Implement immutable model knowledge and lineage graph
**Desired capability:** Answer factual model questions from authoritative sources.
**Prerequisites:** P2-01
**Reuse/integration points:** Existing lineage, evidence registry and DCLab feature metadata.

### Engineering tasks
- [ ] **01** Link model version to code revision, environment, dataset snapshot/hash, schema, feature pipeline, hyperparameters and training run.
- [ ] **02** Build `model.get_overview`, `model.list_versions`, `model.get_training_sources`, `model.get_features`, `model.get_experiments` tools.
- [ ] **03** Add evidence resolver for figures, comparison tables and source records with access checks.
- [ ] **04** Detect incomplete provenance and express specific missing fields and source freshness.
- [ ] **05** Create lineage tests for one version trained from different snapshot versus another.

### Outputs and checks
- Artifacts: ModelKnowledgeService and citation-bearing DTOs.
- Acceptance: Agent answers "Which data and features trained v2?" with correct pinned references and no guesses.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-03: Implement Model Agent conversational context and actions
**Desired capability:** Provide useful model-aware dialog without confusing analysis and execution.
**Prerequisites:** P2-02
**Reuse/integration points:** Model Harness plus shared context, tool and trace services.

### Engineering tasks
- [ ] **01** Create Model Harness system instructions and typed intents: explain, inspect, evaluate, compare, diagnose, retrain_request, deploy_proposal.
- [ ] **02** Resolve requested version and time window explicitly and carry constraints between turns.
- [ ] **03** Filter tool availability based on adapter support, user permissions and model status.
- [ ] **04** Answer with sections: verified fact, evidence, limitation, next action; refuse to invent missing metrics.
- [ ] **05** Support resume by context cursor and audit each tool response.

### Outputs and checks
- Artifacts: Model conversation API and benchmark prompts.
- Acceptance: Multi-turn follow-ups preserve version/cohort/threshold; unsupported requests have grounded explanations.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
