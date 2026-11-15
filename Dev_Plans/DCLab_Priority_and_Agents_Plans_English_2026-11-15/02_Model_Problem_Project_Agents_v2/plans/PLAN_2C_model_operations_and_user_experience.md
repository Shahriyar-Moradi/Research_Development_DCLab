# Implementation Subplan 2C: Model operations and user experience
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

**Parent:** Phase 2: Model Agent Template and Specialized Model Harness
**Phase exit gate:** G2
**Mission:** Deliver repeatable model-knowledge, evaluation, diagnostics and controlled improvement capabilities for any supported model.

## Delivery contract
This subplan is a standalone implementation increment: preserve its interfaces, ship tests with the change, provide evidence and do not claim a capability before validating it.

## Work packages and sequencing
| Package | Work | Concrete deliverable | Dependency |
|---|---|---|---|
| `P2-07` | Implement controlled retraining and champion-challenger | RetrainingJob, ModelCandidate and champion-challenger comparison. | P2-04, P1-09, P1-10 |
| `P2-08` | Implement deployment-state adapter and safe actions | DeploymentSnapshot, ReleaseProposal and gate tests. | P2-07 |
| `P2-09` | Implement Model Agent screens and evidence views | Model Agent UI views and end-to-end tests. | P2-03, P2-06 |
| `P2-10` | Gate model template compatibility and release slice | G2 gate report, defect register, Model Agent MVP release tag. | P2-09 |

## P2-07: Implement controlled retraining and champion-challenger
**Desired capability:** Turn retraining into an auditable, reversible workflow.
**Prerequisites:** P2-04, P1-09, P1-10
**Reuse/integration points:** Existing `studio/retrain.py` semantics and worker infrastructure.

### Engineering tasks
- [ ] **01** Validate training data license, source freshness, feature compatibility, label quality, sufficient sample and permitted operation.
- [ ] **02** Generate immutable candidate training spec; request approval when policy requires it.
- [ ] **03** Submit provider training job and persist idempotency key, provider job ID, logs and output artifacts.
- [ ] **04** Evaluate challenger against champion on a shared or explicitly comparable untouched window and include decision threshold.
- [ ] **05** Produce recommendation showing benefits, regression risks, costs, uncertainties and whether candidate is eligible for promotion.

### Outputs and checks
- Artifacts: RetrainingJob, ModelCandidate and champion-challenger comparison.
- Acceptance: Duplicate submissions start only one effectful job; holdout reuse and missing consent are refused; a candidate is not auto-deployed.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-08: Implement deployment-state adapter and safe actions
**Desired capability:** Represent production reality without creating an unsafe deployment bot.
**Prerequisites:** P2-07
**Reuse/integration points:** Studio service export; external deployment connector separate from model science.

### Engineering tasks
- [ ] **01** Read deployed alias, serving environment, release ID, inference errors, canary status and rollback options.
- [ ] **02** Define proposed versus approved versus executed state and require target-specific approval and provider receipt.
- [ ] **03** Add dry-run release checks for schema changes, latency, resource cost and rollback compatibility.
- [ ] **04** Expose deploy proposal/approval flow behind provider adapter; defer actual deployment automation if not supported.
- [ ] **05** Persist explicit incident and rollback references for later audit.

### Outputs and checks
- Artifacts: DeploymentSnapshot, ReleaseProposal and gate tests.
- Acceptance: An unapproved proposal never changes production; existing deployment state is read from provider, not inferred from a chat message.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-09: Implement Model Agent screens and evidence views
**Desired capability:** Make model lifecycle intelligible to ML engineers.
**Prerequisites:** P2-03, P2-06
**Reuse/integration points:** External UI shell from P1-12.

### Engineering tasks
- [ ] **01** Create model overview with alias, versions, source data, features and experiment history.
- [ ] **02** Show immutable evaluation comparison and monitoring timeline, threshold/cohort labels, warnings and sample sizes.
- [ ] **03** Build chat with tool-call timeline, evidence links, approvals, job status and comparison reports.
- [ ] **04** Offer safe forms for evaluation and challenger requests, with clear action previews.
- [ ] **05** Test keyboard navigation, loading, missing evidence, and authorization-specific visibility.

### Outputs and checks
- Artifacts: Model Agent UI views and end-to-end tests.
- Acceptance: User can open the exact metric source, compare versions and see whether a job is proposed/running/completed.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## P2-10: Gate model template compatibility and release slice
**Desired capability:** Prove template reuse instead of demo-specific engineering.
**Prerequisites:** P2-09
**Reuse/integration points:** Existing `agent_eval` provides baseline patterns; external sealed tests are independent.

### Engineering tasks
- [ ] **01** Run sealed model-agent question suite against original churn model and a second classifier instance.
- [ ] **02** Measure exact metric retrieval, version resolution, tool selection, unauthorized effects, cost and latency.
- [ ] **03** Run injected failures: DCLab timeout, unsupported model kind, missing feature artifact and revoked model access.
- [ ] **04** Verify two agent instances were instantiated from identical template binary with distinct manifests.
- [ ] **05** Record observed defects, patch adapters/contract, rerun entire model gate and publish evidence.

### Outputs and checks
- Artifacts: G2 gate report, defect register, Model Agent MVP release tag.
- Acceptance: No metric hallucinations in deterministic tests, no unauthorized writes, successful onboarding without agent/harness code copy.
- Risk to control: Scope creep, stale evidence, or unauthorized side effects
- Quality gate: typed schemas, positive/negative tests, source and permission checks, durable receipts, updated contract docs.
- Implementation status: **Not started / requires evidence** (this document is a plan, not a completion claim).

## Subplan acceptance
- [ ] Every listed work package has a reviewed PR and reproducible test evidence.
- [ ] The implementation reuses the approved shared services and preserves the three-template boundary.
- [ ] No unauthorized state changes occur under negative and retry tests.
- [ ] Migration, observability and recovery are documented where applicable.
- [ ] Parent phase gate evidence is updated with residual risks.
