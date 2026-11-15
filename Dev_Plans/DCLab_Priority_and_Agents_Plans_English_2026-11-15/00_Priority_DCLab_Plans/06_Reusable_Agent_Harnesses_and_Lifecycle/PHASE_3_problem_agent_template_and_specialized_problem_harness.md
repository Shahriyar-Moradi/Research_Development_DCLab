# Phase 3: Problem Agent Template and Specialized Problem Harness
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Give each independently versioned business/technical problem an agent that understands its objectives, complete solution history, evidence, context and operational life.
**Entry criteria:** G2 Model Agent and G1 template/factory capabilities are available; churn problem and at least one linked solution are defined.
**Exit gate G3:** A churn Problem Agent can explain its objective, all considered solutions and rationale, analyze KPI definitions, preserve conversation continuity, and safely request model-level work.
**Mandatory live demonstration:** Ask: "What churn problem are we solving, what alternatives were rejected, and why?" Then revise the solution and verify prior revisions remain accessible.
**Important boundary:** A Problem Agent is not a narrow problem statement bot. It owns the problem, its solution portfolio, context, data, conversation and decisions; Project Agent coordinates multiple such problems.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 3A | Problem identity and knowledge | P3-01, P3-02, P3-03 | Implement in listed order unless explicitly parallel-safe |
| 3B | Solutions, decisions and scientific/business context | P3-04, P3-05, P3-06, P3-07 | Implement in listed order unless explicitly parallel-safe |
| 3C | Execution readiness and problem-level experience | P3-08, P3-09, P3-10 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 3A: Problem identity and knowledge
**Goal:** Complete P3-01, P3-02, P3-03 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P3-01 - Implement Problem entity and Problem Agent identity
**Outcome and scope.** Separate one problem lifecycle from its containing project.
**Preconditions/dependencies.** P1-04, P0-03
**Existing integration to inspect.** DCLab problem card and `dclab_core/problem_map.py` via adapter.

**Implementation steps (ordered):
1. Implement Problem record with business owner, goals, prediction/decision moment, constraints, stakeholders, outcome KPI and hypotheses.
2. Define lifecycle proposed/clarifying/active/paused/resolved/retired with guarded transitions.
3. Create Problem Agent by factory using `problem` template, stable problem ID and project-scoped permissions.
4. Link confirmed intake from existing DCLab Case Agent through one import adapter, preserving source and uncertainty.
5. Support multiple problems under a project without requiring multiple project agents.

**Artifacts and interfaces:** ProblemAgent manifest, problem lifecycle service and API.

**Acceptance tests and evidence:** Two problems share one Project yet keep independent identity, conversations, policies and solution histories.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-02 - Implement Problem Context Graph and evidence resolution
**Outcome and scope.** Keep goals, datasets, models, experiments and conversations correctly connected.
**Preconditions/dependencies.** P3-01, P1-08
**Existing integration to inspect.** External structured store and DCLab data catalog / lineage adapters.

**Implementation steps (ordered):
1. Define typed edges problem->dataset/feature/model/experiment/solution/decision/KPI/conversation and enforce tenant ACLs.
2. Build context resolver with pinned authoritative facts plus relevant historical evidence and explicit freshness dates.
3. Track competing hypotheses, questions, assumptions and contradictions with source provenance.
4. Support exact relationship traversal and query-by-question without loading all material into LLM context.
5. Preserve tombstones and superseding revisions on deleted or updated source items.

**Artifacts and interfaces:** ProblemKnowledgeGraph service and evidence-aware context bundle.

**Acceptance tests and evidence:** A newly updated problem goal supersedes old chat assumptions; inaccessible linked model evidence is never leaked.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-03 - Implement conversation memory and decision continuity
**Outcome and scope.** Make problem-level conversation a persistent, verifiable history.
**Preconditions/dependencies.** P3-02
**Existing integration to inspect.** DCLab Case Agent session records via explicit import grants.

**Implementation steps (ordered):
1. Store immutable conversation events, actor, timestamps, linked problem, relevant entity refs and summarization cursor.
2. Record confirmations, corrections and open questions separately from informal discussion.
3. Create bounded history summarization with provenance and human-reviewed high-impact edits.
4. Implement retrieval "what changed since last time" using event offsets and current canonical facts.
5. Allow authorized migration of confirmed Case Agent context without exposing all intake sessions by default.

**Artifacts and interfaces:** ProblemConversationMemory, summary versioning and continuity APIs.

**Acceptance tests and evidence:** Correction in a later turn takes precedence; agent can cite which event changed its understanding.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 3B: Solutions, decisions and scientific/business context
**Goal:** Complete P3-04, P3-05, P3-06, P3-07 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P3-04 - Implement first-class Solution Portfolio under each problem
**Outcome and scope.** Treat solution alternatives as independent, versioned entities, not paragraphs in chat.
**Preconditions/dependencies.** P3-02
**Existing integration to inspect.** DCLab `dclab_core/plan.py`, `solution_workflow.py`, Studio solution APIs.

**Implementation steps (ordered):
1. Define solution status proposed/under_review/validated/rejected/adopted/superseded/retired and revision history.
2. Store approach, constraints, assumptions, associated model versions, evaluation methods, planned workflow and business hypothesis.
3. Create compare/clone/revise/adopt operations with immutable source evidence and acceptance gates.
4. Integrate `SolutionPlan`, `SolutionWorkflow` and Studio Solution as referenced artifacts, not duplicate planners.
5. Display rejected candidates and reasons, and require approval for replacing the adopted solution.

**Artifacts and interfaces:** SolutionRepository, solution comparison and adoption workflow.

**Acceptance tests and evidence:** Changing current solution preserves old alternatives, explicit decision rationale, and linked evaluation records.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-05 - Implement Decision and Assumption Ledger
**Outcome and scope.** Answer why choices were made using recorded evidence.
**Preconditions/dependencies.** P3-04
**Existing integration to inspect.** DCLab ledgers and transitions through adapter where available.

**Implementation steps (ordered):
1. Define DecisionRecord: actor, alternatives, chosen option, rationale, evidence refs, uncertainties, time, scope and approval.
2. Distinguish observation, unverified hypothesis, assumption, proposal, accepted decision and later outcome.
3. Implement append-only superseding history and source-level access control.
4. Build read tools for "Why XGBoost?", "Which alternatives failed?" and "What changed?".
5. Enforce no-invented-rationale policy; infer possible causes only when labeled inference.

**Artifacts and interfaces:** Decision ledger, assumption register and comparison explanations.

**Acceptance tests and evidence:** Missing historical rationale produces "not recorded" rather than a reconstructed fiction.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-06 - Implement problem data sources, feature maps and business KPI contracts
**Outcome and scope.** Bridge business context with ML metrics correctly.
**Preconditions/dependencies.** P3-02
**Existing integration to inspect.** `dclab_rnd/connectors/db.py`, Studio feature/data tools, data audit.

**Implementation steps (ordered):
1. Define dataset binding and approved query views with data owner, schema, freshness, row cap and workspace permission.
2. Create KPI definitions including unit, numerator/denominator, cohort, period, timezone, source and allowed comparison rule.
3. Link problem features and model inputs with prediction moment and leakage review.
4. Implement read-only business cohort queries with parameter validation and PII minimization.
5. Test business churn rate vs model recall confusion, stale source and missing labels.

**Artifacts and interfaces:** ProblemDataContext, KPI registry, scoped query adapter.

**Acceptance tests and evidence:** Agent never represents recall as churn rate; every KPI states population, period, source and missing-data caveats.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-07 - Implement multi-criteria solution assessment
**Outcome and scope.** Compare candidate approaches using evidence, cost and business goals.
**Preconditions/dependencies.** P3-04, P3-06
**Existing integration to inspect.** DCLab evaluation/report tools via Model Agent, business KPI registry.

**Implementation steps (ordered):
1. Define assessment criteria: model quality, expected or observed business outcome, resource cost, latency, operational risk and data readiness.
2. Distinguish offline metrics, forecasts and controlled intervention outcomes with separate evidence classes.
3. Create comparable cohort/window and threshold check before ranking solution candidates.
4. Persist a scored trade-off matrix with qualitative limitations and stakeholder-approved weights.
5. Provide an explainable recommendation with experiments needed to close uncertainty.

**Artifacts and interfaces:** SolutionAssessment DTO, comparison table and decision proposal.

**Acceptance tests and evidence:** Better model AUC alone cannot be reported as proven customer-retention improvement.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 3C: Execution readiness and problem-level experience
**Goal:** Complete P3-08, P3-09, P3-10 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P3-08 - Connect problem decomposition, solution planning and task readiness
**Outcome and scope.** Integrate existing DCLab thought artifacts without copying the execution engine.
**Preconditions/dependencies.** P3-07
**Existing integration to inspect.** `dclab_core/task_graph.py`, `task_execution_plan.py`, `task_validation.py`.

**Implementation steps (ordered):
1. Link ProblemMap and SolutionPlan IDs to Problem Agent; version the active decomposition and solution plan.
2. Translate adopted solution into typed TaskGraph proposals with capability bindings and evidence requirements.
3. Validate graph structure, dependencies, approvals and available tools before execution planning.
4. Mark steps proposed/ready/blocked/executed based on independent host receipts.
5. Stop if data readiness, label timing, rights or required approvals are missing.

**Artifacts and interfaces:** ProblemExecutionPlan and task readiness reports.

**Acceptance tests and evidence:** The agent cannot mark task complete from chat claims; blocked prerequisites show a specific remedy.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-09 - Enable Model Agent discovery and problem-level delegation proposals
**Outcome and scope.** Make specialization visible while keeping authority with each harness.
**Preconditions/dependencies.** P3-08, P2-10
**Existing integration to inspect.** Shared registry, Model Agent tool APIs, task contracts.

**Implementation steps (ordered):
1. Resolve linked Model Agent IDs and allowed capabilities by model/solution association.
2. Build typed task request for model metrics, diagnostics, feature analysis or challenger comparison.
3. Check permission, version, expected output and budget before submission to central collaboration layer.
4. Store correlation IDs and evidence handles; avoid free-text agent messages as executable commands.
5. In Phase 3, support a read-only dispatch pilot; Phase 5 adds robust full orchestration.

**Artifacts and interfaces:** Problem-to-model capability bindings and typed request prototype.

**Acceptance tests and evidence:** An unrelated model or wrong version cannot be delegated to without explicit grant; responses keep provenance.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P3-10 - Deliver Problem Agent screen and sealed G3 tests
**Outcome and scope.** Prove persistent problem and solution intelligence.
**Preconditions/dependencies.** P3-09
**Existing integration to inspect.** External UI, sealed benchmark suite and current Model Agent.

**Implementation steps (ordered):
1. Create Overview, Goals, Hypotheses, Solutions, Models, Data, Decisions, Conversations and Work tabs.
2. Support versioned solution comparison, decision timeline and evidence-linked chat answers.
3. Run multi-turn evaluation of correction, historical rationale, contradictory KPI and missing evidence.
4. Validate read-only/delegation permissions and isolation across two problems in same project.
5. Publish G3 defect review and freeze stable Problem Harness contract.

**Artifacts and interfaces:** Problem Agent UI, release candidate and G3 report.

**Acceptance tests and evidence:** One churn problem successfully preserves solution and conversation revisions; independent second problem uses same template and harness.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 10 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Ask: "What churn problem are we solving, what alternatives were rejected, and why?" Then revise the solution and verify prior revisions remain accessible.
- [ ] G3 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G3 provides the stable interfaces, immutable evidence and regression tests required by Phase 4.

---

## Download or open each implementation subplan separately

- [Implementation Subplan 3A: Problem identity and knowledge](../plans/PLAN_3A_problem_identity_and_knowledge.md)
- [Implementation Subplan 3B: Solutions, decisions and scientific/business context](../plans/PLAN_3B_solutions_decisions_and_scientific_business_context.md)
- [Implementation Subplan 3C: Execution readiness and problem-level experience](../plans/PLAN_3C_execution_readiness_and_problem_level_experience.md)

These subplans are independent implementation documents containing work packages, detailed engineering tasks, outputs, tests, and acceptance criteria.
