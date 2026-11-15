# Implementation Subplan 2B: Evaluation, diagnostics and monitoring
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 2: Model Agent Template and Specialized Model Harness
**Phase exit gate:** G2
**Mission:** Deliver repeatable model-knowledge, evaluation, diagnostics and controlled improvement capabilities for any supported model.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P2-04` | Implement versioned deterministic evaluation | EvaluationService, job handler, versioned metric records. | P2-02, P1-09 |
| `P2-05` | Implement feature, prediction and quality diagnostics | DiagnosticReport DTO, feature-quality adapters and plot/evidence refs. | P2-04 |
| `P2-06` | Implement monitoring baselines and per-model policy | MonitoringPolicy, MonitoringReport and model telemetry view. | P2-05 |

## P2-04: Implement versioned deterministic evaluation
**Desired capability:** Make accuracy/recall and other metrics reliable rather than verbal estimates.
**Prerequisites:** P2-02, P1-09
**Reuse/integration points:** `dclab_core/model_evaluation.py` and existing Studio evaluation path.

### Engineering tasks
- [ ] **01** Resolve exact model version, approved evaluation dataset, label state, prediction horizon, segment and threshold.
- [ ] **02** Check evaluation dataset compatibility, leakage, duplication and model-serving preprocessing schema.
- [ ] **03** Run task-specific evaluation through the authorized compute adapter with class-appropriate measures and uncertainty.
- [ ] **04** Write immutable EvaluationRun referencing code, model, window, metrics, confusion matrix, calibration and costs.
- [ ] **05** Expose `model.evaluate`, `model.compare_evaluations` and `model.get_evaluation` tools.

### Outputs and checks
- Artifacts: EvaluationService, job handler, versioned metric records.
- Acceptance: Metric calculations match independent oracle; stale, unlabeled or incompatible cohorts produce explicit errors.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-05: Implement feature, prediction and quality diagnostics
**Desired capability:** Expose reproducible root-cause evidence without claiming causality.
**Prerequisites:** P2-04
**Reuse/integration points:** `dclab_core/monitoring.py`, feature checks, Studio audit.

### Engineering tasks
- [ ] **01** Validate feature distribution, null rates, category changes, freshness and feature-contract mismatches by version.
- [ ] **02** Compare training and serving samples with documented distance statistics and sample sizes.
- [ ] **03** Calculate segment-level model quality only when ground truth exists; distinguish prediction drift from performance drift.
- [ ] **04** Integrate compatible explainability adapters with explicit model version, data and limitations.
- [ ] **05** Produce ranked diagnostic findings as observed signals, hypotheses and recommended further checks.

### Outputs and checks
- Artifacts: DiagnosticReport DTO, feature-quality adapters and plot/evidence refs.
- Acceptance: A synthetic null spike and a distribution shift are detected; missing labels never result in invented recall.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-06: Implement monitoring baselines and per-model policy
**Desired capability:** Maintain model health against explicit reference windows.
**Prerequisites:** P2-05
**Reuse/integration points:** Existing DCLab monitoring plan and `watch_mlflow.py` where usable.

### Engineering tasks
- [ ] **01** Define monitoring policy: required features, production alias, baseline window, thresholds, minimum count and notification routing.
- [ ] **02** Create baseline from approved training/reference data and store its version and hash.
- [ ] **03** Read monitoring batches/events through source adapters, compute schema/quality/drift checks and record monitor windows.
- [ ] **04** Apply label-availability logic for delayed ground truth and calibration checks.
- [ ] **05** Provide `model.monitor_status`, `model.list_alerts`, `model.explain_alert`.

### Outputs and checks
- Artifacts: MonitoringPolicy, MonitoringReport and model telemetry view.
- Acceptance: The same window/policy yields reproducible reports; changed thresholds create a new policy version, not silent retroactive changes.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
