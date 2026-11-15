# Implementation Subplan 0B: Data, baseline model and evidence
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 0: Foundation, DCLab Gap Audit and Reference Scenario
**Phase exit gate:** G0
**Mission:** Create a validated, provenance-aware specification and reproducible churn reference before coding the external platform.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P0-04` | Curate churn dataset with training provenance | Dataset card, SHA-256 manifest, data-quality report, split protocol. | Previous approved package and phase prerequisites |
| `P0-05` | Produce versioned baseline model and metric evidence | Two model versions, model cards, evaluation artifacts, serving contract. | P0-04 |

## P0-04: Curate churn dataset with training provenance
**Desired capability:** Make data readiness the first hard gate.
**Prerequisites:** Previous approved package and phase prerequisites
**Reuse/integration points:** DCLab Telco data, dataset catalog, leakage audit and lineage records.

### Engineering tasks
- [ ] **01** Select permitted Telco churn source; document license, origin URL or ownership, collection time and allowed training use.
- [ ] **02** Pin raw file hash, schema, target, prediction moment, entity key, sensitive fields, label availability and temporal assumptions.
- [ ] **03** Generate validation report for nulls, duplicates, leakage suspects, inconsistent types and invalid targets.
- [ ] **04** Create train/validation/test protocol with held-out evidence, leakage guard and stable random seed or justified time split.
- [ ] **05** Publish dataset card and approved training manifest; exclude unlicensed commercial model outputs or undocumented derived labels.

### Outputs and checks
- Artifacts: Dataset card, SHA-256 manifest, data-quality report, split protocol.
- Acceptance: Independent rerun loads identical snapshot and split; leaks and licensing violations block use.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P0-05: Produce versioned baseline model and metric evidence
**Desired capability:** Create deterministic reference artifacts for Model Agent read/evaluate workflows.
**Prerequisites:** P0-04
**Reuse/integration points:** `general_pipeline`, `studio/lineage.py`, `studio/retrain.py`, `dclab_core/model_evaluation.py`.

### Engineering tasks
- [ ] **01** Train an initial tabular classifier using existing science pipeline rather than coding a second training implementation.
- [ ] **02** Record preprocessing, feature contract, dataset version, hyperparameters, code commit, seed and training run ID.
- [ ] **03** Evaluate with precision, recall, F1, PR-AUC, ROC-AUC, calibration and chosen threshold on correctly separated windows.
- [ ] **04** Register at least two model versions for subsequent comparison; mark baseline versus candidate status clearly.
- [ ] **05** Export fabricated or approved non-sensitive inference fixtures for API and UI tests.

### Outputs and checks
- Artifacts: Two model versions, model cards, evaluation artifacts, serving contract.
- Acceptance: A rerun reproduces results within recorded numerical tolerance; metrics include window, threshold, cohort and source ref.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
