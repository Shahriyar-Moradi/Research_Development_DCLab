# Phase 2: Model Agent Template and Specialized Model Harness
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Deliver repeatable model-knowledge, evaluation, diagnostics and controlled improvement capabilities for any supported model.
**Entry criteria:** G1 shared kernel, model template, contract registry and pilot model versions are operational.
**Exit gate G2:** Churn Model Agent can answer model questions with evidence, launch approved evaluation and challenger jobs, and explain monitoring signals.
**Mandatory live demonstration:** Ask a churn Model Agent: "Which data trained v2?", "What is recall for cohort X?", "Did input distributions drift?", "Train a challenger". Each yields traceable evidence or a correct refusal.
**Important boundary:** The Model Agent owns a stable model-family identity; every answer specifies the exact relevant model version, dataset, time window and threshold. Production actions remain guarded.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 2A | Model onboarding, versions and knowledge | P2-01, P2-02, P2-03 | Implement in listed order unless explicitly parallel-safe |
| 2B | Evaluation, diagnostics and monitoring | P2-04, P2-05, P2-06 | Implement in listed order unless explicitly parallel-safe |
| 2C | Model operations and user experience | P2-07, P2-08, P2-09, P2-10 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 2A: Model onboarding, versions and knowledge
**Goal:** Complete P2-01, P2-02, P2-03 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P2-01 - Build ModelFamily/ModelVersion onboarding adapter
**Outcome and scope.** Register models and create agents automatically without per-model classes.
**Preconditions/dependencies.** P1-04, P0-05
**Existing integration to inspect.** `dclab_rnd/studio/lineage.py`, model cards, MLflow integration `watch_mlflow.py`.

**Implementation steps (ordered):
1. Define model-family record with task kind, owning tenant, algorithms, artifact registry references, version aliases, feature schema and lifecycle status.
2. Implement import from DCLab model/evidence records and optional MLflow registry adapter, validating identifiers and artifact hashes.
3. Create or resolve Model Agent instance from the `model` template; never generate custom source code.
4. Record safe backend adapter selection for scikit-learn, LightGBM/XGBoost, PyTorch or unsupported task families.
5. Test duplicate registration, version supersession, missing artifact and cross-project model-sharing grants.

**Artifacts and interfaces:** Model registry adapter, import API, model agent auto-instantiation.

**Acceptance tests and evidence:** Onboard two classifiers using one template/harness; unsupported format reports capability gap rather than failing silently.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-02 - Implement immutable model knowledge and lineage graph
**Outcome and scope.** Answer factual model questions from authoritative sources.
**Preconditions/dependencies.** P2-01
**Existing integration to inspect.** Existing lineage, evidence registry and DCLab feature metadata.

**Implementation steps (ordered):
1. Link model version to code revision, environment, dataset snapshot/hash, schema, feature pipeline, hyperparameters and training run.
2. Build `model.get_overview`, `model.list_versions`, `model.get_training_sources`, `model.get_features`, `model.get_experiments` tools.
3. Add evidence resolver for figures, comparison tables and source records with access checks.
4. Detect incomplete provenance and express specific missing fields and source freshness.
5. Create lineage tests for one version trained from different snapshot versus another.

**Artifacts and interfaces:** ModelKnowledgeService and citation-bearing DTOs.

**Acceptance tests and evidence:** Agent answers "Which data and features trained v2?" with correct pinned references and no guesses.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-03 - Implement Model Agent conversational context and actions
**Outcome and scope.** Provide useful model-aware dialog without confusing analysis and execution.
**Preconditions/dependencies.** P2-02
**Existing integration to inspect.** Model Harness plus shared context, tool and trace services.

**Implementation steps (ordered):
1. Create Model Harness system instructions and typed intents: explain, inspect, evaluate, compare, diagnose, retrain_request, deploy_proposal.
2. Resolve requested version and time window explicitly and carry constraints between turns.
3. Filter tool availability based on adapter support, user permissions and model status.
4. Answer with sections: verified fact, evidence, limitation, next action; refuse to invent missing metrics.
5. Support resume by context cursor and audit each tool response.

**Artifacts and interfaces:** Model conversation API and benchmark prompts.

**Acceptance tests and evidence:** Multi-turn follow-ups preserve version/cohort/threshold; unsupported requests have grounded explanations.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 2B: Evaluation, diagnostics and monitoring
**Goal:** Complete P2-04, P2-05, P2-06 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P2-04 - Implement versioned deterministic evaluation
**Outcome and scope.** Make accuracy/recall and other metrics reliable rather than verbal estimates.
**Preconditions/dependencies.** P2-02, P1-09
**Existing integration to inspect.** `dclab_core/model_evaluation.py` and existing Studio evaluation path.

**Implementation steps (ordered):
1. Resolve exact model version, approved evaluation dataset, label state, prediction horizon, segment and threshold.
2. Check evaluation dataset compatibility, leakage, duplication and model-serving preprocessing schema.
3. Run task-specific evaluation through the authorized compute adapter with class-appropriate measures and uncertainty.
4. Write immutable EvaluationRun referencing code, model, window, metrics, confusion matrix, calibration and costs.
5. Expose `model.evaluate`, `model.compare_evaluations` and `model.get_evaluation` tools.

**Artifacts and interfaces:** EvaluationService, job handler, versioned metric records.

**Acceptance tests and evidence:** Metric calculations match independent oracle; stale, unlabeled or incompatible cohorts produce explicit errors.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-05 - Implement feature, prediction and quality diagnostics
**Outcome and scope.** Expose reproducible root-cause evidence without claiming causality.
**Preconditions/dependencies.** P2-04
**Existing integration to inspect.** `dclab_core/monitoring.py`, feature checks, Studio audit.

**Implementation steps (ordered):
1. Validate feature distribution, null rates, category changes, freshness and feature-contract mismatches by version.
2. Compare training and serving samples with documented distance statistics and sample sizes.
3. Calculate segment-level model quality only when ground truth exists; distinguish prediction drift from performance drift.
4. Integrate compatible explainability adapters with explicit model version, data and limitations.
5. Produce ranked diagnostic findings as observed signals, hypotheses and recommended further checks.

**Artifacts and interfaces:** DiagnosticReport DTO, feature-quality adapters and plot/evidence refs.

**Acceptance tests and evidence:** A synthetic null spike and a distribution shift are detected; missing labels never result in invented recall.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-06 - Implement monitoring baselines and per-model policy
**Outcome and scope.** Maintain model health against explicit reference windows.
**Preconditions/dependencies.** P2-05
**Existing integration to inspect.** Existing DCLab monitoring plan and `watch_mlflow.py` where usable.

**Implementation steps (ordered):
1. Define monitoring policy: required features, production alias, baseline window, thresholds, minimum count and notification routing.
2. Create baseline from approved training/reference data and store its version and hash.
3. Read monitoring batches/events through source adapters, compute schema/quality/drift checks and record monitor windows.
4. Apply label-availability logic for delayed ground truth and calibration checks.
5. Provide `model.monitor_status`, `model.list_alerts`, `model.explain_alert`.

**Artifacts and interfaces:** MonitoringPolicy, MonitoringReport and model telemetry view.

**Acceptance tests and evidence:** The same window/policy yields reproducible reports; changed thresholds create a new policy version, not silent retroactive changes.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 2C: Model operations and user experience
**Goal:** Complete P2-07, P2-08, P2-09, P2-10 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P2-07 - Implement controlled retraining and champion-challenger
**Outcome and scope.** Turn retraining into an auditable, reversible workflow.
**Preconditions/dependencies.** P2-04, P1-09, P1-10
**Existing integration to inspect.** Existing `studio/retrain.py` semantics and worker infrastructure.

**Implementation steps (ordered):
1. Validate training data license, source freshness, feature compatibility, label quality, sufficient sample and permitted operation.
2. Generate immutable candidate training spec; request approval when policy requires it.
3. Submit provider training job and persist idempotency key, provider job ID, logs and output artifacts.
4. Evaluate challenger against champion on a shared or explicitly comparable untouched window and include decision threshold.
5. Produce recommendation showing benefits, regression risks, costs, uncertainties and whether candidate is eligible for promotion.

**Artifacts and interfaces:** RetrainingJob, ModelCandidate and champion-challenger comparison.

**Acceptance tests and evidence:** Duplicate submissions start only one effectful job; holdout reuse and missing consent are refused; a candidate is not auto-deployed.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-08 - Implement deployment-state adapter and safe actions
**Outcome and scope.** Represent production reality without creating an unsafe deployment bot.
**Preconditions/dependencies.** P2-07
**Existing integration to inspect.** Studio service export; external deployment connector separate from model science.

**Implementation steps (ordered):
1. Read deployed alias, serving environment, release ID, inference errors, canary status and rollback options.
2. Define proposed versus approved versus executed state and require target-specific approval and provider receipt.
3. Add dry-run release checks for schema changes, latency, resource cost and rollback compatibility.
4. Expose deploy proposal/approval flow behind provider adapter; defer actual deployment automation if not supported.
5. Persist explicit incident and rollback references for later audit.

**Artifacts and interfaces:** DeploymentSnapshot, ReleaseProposal and gate tests.

**Acceptance tests and evidence:** An unapproved proposal never changes production; existing deployment state is read from provider, not inferred from a chat message.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-09 - Implement Model Agent screens and evidence views
**Outcome and scope.** Make model lifecycle intelligible to ML engineers.
**Preconditions/dependencies.** P2-03, P2-06
**Existing integration to inspect.** External UI shell from P1-12.

**Implementation steps (ordered):
1. Create model overview with alias, versions, source data, features and experiment history.
2. Show immutable evaluation comparison and monitoring timeline, threshold/cohort labels, warnings and sample sizes.
3. Build chat with tool-call timeline, evidence links, approvals, job status and comparison reports.
4. Offer safe forms for evaluation and challenger requests, with clear action previews.
5. Test keyboard navigation, loading, missing evidence, and authorization-specific visibility.

**Artifacts and interfaces:** Model Agent UI views and end-to-end tests.

**Acceptance tests and evidence:** User can open the exact metric source, compare versions and see whether a job is proposed/running/completed.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P2-10 - Gate model template compatibility and release slice
**Outcome and scope.** Prove template reuse instead of demo-specific engineering.
**Preconditions/dependencies.** P2-09
**Existing integration to inspect.** Existing `agent_eval` provides baseline patterns; external sealed tests are independent.

**Implementation steps (ordered):
1. Run sealed model-agent question suite against original churn model and a second classifier instance.
2. Measure exact metric retrieval, version resolution, tool selection, unauthorized effects, cost and latency.
3. Run injected failures: DCLab timeout, unsupported model kind, missing feature artifact and revoked model access.
4. Verify two agent instances were instantiated from identical template binary with distinct manifests.
5. Record observed defects, patch adapters/contract, rerun entire model gate and publish evidence.

**Artifacts and interfaces:** G2 gate report, defect register, Model Agent MVP release tag.

**Acceptance tests and evidence:** No metric hallucinations in deterministic tests, no unauthorized writes, successful onboarding without agent/harness code copy.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 10 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Ask a churn Model Agent: "Which data trained v2?", "What is recall for cohort X?", "Did input distributions drift?", "Train a challenger". Each yields traceable evidence or a correct refusal.
- [ ] G2 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G2 provides the stable interfaces, immutable evidence and regression tests required by Phase 3.

---

## Download or open each implementation subplan separately

- [Implementation Subplan 2A: Model onboarding, versions and knowledge](../plans/PLAN_2A_model_onboarding_versions_and_knowledge.md)
- [Implementation Subplan 2B: Evaluation, diagnostics and monitoring](../plans/PLAN_2B_evaluation_diagnostics_and_monitoring.md)
- [Implementation Subplan 2C: Model operations and user experience](../plans/PLAN_2C_model_operations_and_user_experience.md)

These subplans are independent implementation documents containing work packages, detailed engineering tasks, outputs, tests, and acceptance criteria.
