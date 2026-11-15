# Phase 4: Project Agent Template and Specialized Project Harness
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

## Executive objective

Create the long-lived portfolio-level agent coordinating multiple problems, solutions, people, decisions and outcomes without duplicating Problem Agent expertise.
**Entry criteria:** G3 Problem Agents are proven, one Project root and two Problem objects are available.
**Exit gate G4:** A Project Agent can answer overall status, coordinate problem portfolios, explain cross-problem decisions and continue project-level work.
**Mandatory live demonstration:** Ask: "What have we tried across churn and retention, what is blocked, and which solution should we prioritize?" Verify source-based result and workload implications.
**Important boundary:** A Project is a durable coordination boundary, not a synonym for Problem; projects may contain multiple business objectives, unrelated models and multiple teams.

## Subplans and delivery sequence
| Subplan | Objective | Work packages | Execution rule |
|---|---|---|---|
| 4A | Project identity, relationships and organizational memory | P4-01, P4-02, P4-03 | Implement in listed order unless explicitly parallel-safe |
| 4B | Portfolio planning, coordination and governance | P4-04, P4-05, P4-06, P4-07 | Implement in listed order unless explicitly parallel-safe |
| 4C | Project intelligence and workspace UX | P4-08, P4-09, P4-10 | Implement in listed order unless explicitly parallel-safe |

## Work-package dependency map
The packages are implementation units, not one large undifferentiated phase. Implement and review each unit through a small pull request. Explicit dependencies override numeric order; work not listed as safe to parallelize should stay sequential.

## Detailed subplans

### Subplan 4A: Project identity, relationships and organizational memory
**Goal:** Complete P4-01, P4-02, P4-03 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P4-01 - Implement Project Agent identity and lifecycle
**Outcome and scope.** Create a durable program-level representative with independent rules.
**Preconditions/dependencies.** P3-10
**Existing integration to inspect.** DCLab Studio project records and external Agent Factory.

**Implementation steps (ordered):
1. Define Project with scope, owner, team, objectives, budget, linked problem IDs, launch date and lifecycle status.
2. Create one stable Project Agent instance per project from `project` template.
3. Implement create/adopt/pause/archive and project version events with optimistic concurrency.
4. Import DCLab project metadata via adapter while keeping external identity separate and mapping explicit.
5. Define project-wide policy limits inherited and tightened by individual Problem and Model agents.

**Artifacts and interfaces:** ProjectAgent entity, manifest and guarded lifecycle APIs.

**Acceptance tests and evidence:** Replacing the active solution or one problem model never recreates Project Agent identity.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-02 - Implement project-to-problem/model relationship graph
**Outcome and scope.** Support many-to-many model and solution use without cross-tenant leakage.
**Preconditions/dependencies.** P4-01
**Existing integration to inspect.** Problem/Model registries and context resolver.

**Implementation steps (ordered):
1. Add project membership, problem ownership, shared model access, solution relationships and resource references.
2. Enforce scoped link/unlink, versioned relation events and clearly defined orphan or retirement handling.
3. Aggregate canonical statuses from children without overwriting their authoritative records.
4. Represent separate owner/grant for shared model serving multiple problems.
5. Build tests for hierarchy changes, duplicate links, stale references and cross-tenant attempts.

**Artifacts and interfaces:** ProjectRelationshipService and portfolio traversal API.

**Acceptance tests and evidence:** Every project view identifies correct child agents and relation permissions; linked model cannot leak to unauthorized project.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-03 - Implement project-level history, conversations and decisions
**Outcome and scope.** Preserve work continuity across multiple problems and teams.
**Preconditions/dependencies.** P4-02
**Existing integration to inspect.** Problem Agent conversation contract and shared memory infrastructure.

**Implementation steps (ordered):
1. Store project-specific conversations, meeting decisions, accepted constraints, milestones, approvals and references to child-agent records.
2. Create cross-problem retrieval with permission-aware summarization and source attribution.
3. Track decision changes and contradictory assumptions with resolution ownership.
4. Support "what changed since last meeting" by canonical event cursor and temporal ordering.
5. Prevent importing all private child conversations into general project memory without grants.

**Artifacts and interfaces:** ProjectMemoryService, timeline views, decision history.

**Acceptance tests and evidence:** Agent can distinguish a project-level decision from an unresolved problem-specific hypothesis; restricted conversation remains private.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 4B: Portfolio planning, coordination and governance
**Goal:** Complete P4-04, P4-05, P4-06, P4-07 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P4-04 - Implement project objectives, budgets and resource constraints
**Outcome and scope.** Let the Project Agent plan within real business boundaries.
**Preconditions/dependencies.** P4-01
**Existing integration to inspect.** Existing external authorization and job scheduler, DCLab project metadata where available.

**Implementation steps (ordered):
1. Define business objectives and success metrics with measurement windows and accountable owners.
2. Represent operational budgets for compute, engineer time, data spend and agent/tool usage as distinct constraints.
3. Define resource allocations, priorities, deadlines and allowed escalation paths for child problems.
4. Build budget/ownership check tools that run before cross-problem execution.
5. Represent infeasible demands and trade-offs as explicit decisions rather than silent partial execution.

**Artifacts and interfaces:** ProjectGoals, ResourceBudget and constraint evaluator.

**Acceptance tests and evidence:** A plan exceeding a budget cannot start unauthorized jobs; the failure explains which limit blocks the work.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-05 - Implement project portfolio and solution strategy
**Outcome and scope.** Manage competing solution paths and dependencies across problems.
**Preconditions/dependencies.** P4-02, P4-04
**Existing integration to inspect.** Problem Agent solution assessments and project resources.

**Implementation steps (ordered):
1. Aggregate solution alternatives, model linkages and status from Problem Agents.
2. Build cross-problem dependency and conflict graph: shared features, scarce datasets, competing interventions, rollout risk.
3. Support portfolio review comparing impact, effort, uncertainty, risk and expected sequencing.
4. Require signed decisions for priority changes affecting budget or multiple problem owners.
5. Retain historical portfolio snapshots and explain rationales from decision ledger.

**Artifacts and interfaces:** ProjectPortfolioService, strategy assessment and review API.

**Acceptance tests and evidence:** Agent can prioritize churn and fraud work with explicit criteria without claiming false causal business impact.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-06 - Implement project milestones, workflow and task coordination
**Outcome and scope.** Make high-level plans executable through child-owned capabilities.
**Preconditions/dependencies.** P4-05
**Existing integration to inspect.** TaskGraph and job infrastructure patterns.

**Implementation steps (ordered):
1. Translate approved project objectives to milestones with owner, scope, dependencies and acceptance evidence.
2. Attach each work item to a Problem or Model Agent capability rather than creating a new agent per task.
3. Track proposed, approved, executing, completed, failed and cancelled statuses with execution receipt links.
4. Define handoffs, replan triggers and cancelation propagation with approval-aware checks.
5. Expose project-level status and blockers without mutating child scientific records.

**Artifacts and interfaces:** ProjectRoadmap, milestone/ticket APIs and provenance-backed status view.

**Acceptance tests and evidence:** Completed milestone requires verified child task evidence; project view survives service restart and revisions.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-07 - Implement cross-problem governance and approvals
**Outcome and scope.** Ensure portfolio delegation respects local ownership and shared resources.
**Preconditions/dependencies.** P4-06
**Existing integration to inspect.** Shared policy engine plus Problem and Model Harness policies.

**Implementation steps (ordered):
1. Define project-to-problem permissions, child-agent delegation grants and policy inheritance.
2. Implement approval for solution adoption, budget increases, new data access and production-impacting actions.
3. Require explicit scope, target entities, arguments and actor for a high-impact approval.
4. Audit shared model changes that affect more than one problem and require stakeholder review.
5. Test revocation, stale approvals and attempted escalation through chat or tool output.

**Artifacts and interfaces:** Portfolio policy evaluator, cross-problem approval contract.

**Acceptance tests and evidence:** Cross-problem writes fail unless both project and affected resource policies allow them.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

### Subplan 4C: Project intelligence and workspace UX
**Goal:** Complete P4-08, P4-09, P4-10 with independent verification and preserve their interfaces for dependent packages.
**Execution method:** Create/update contracts first, write failure tests, implement adapters and handlers, add end-to-end evidence, run gate review.

#### P4-08 - Implement project intelligence Q&A and executive synthesis
**Outcome and scope.** Give leaders current project reality with evidence, not persuasive fiction.
**Preconditions/dependencies.** P4-06
**Existing integration to inspect.** Project Harness, child-agent contracts, EvidenceBundle.

**Implementation steps (ordered):
1. Create intents for status, next steps, milestones, what changed, choices, risks and comparative outcomes.
2. Collect structured child-agent answers with evidence refs and resource freshness rather than asking an LLM to reconstruct all data.
3. Distinguish proven results, pending experiments, forecasts, assumptions and recommendations.
4. Summarize decisions, dependencies and unresolved uncertainties with links to exact records.
5. Set limits on evidence volume and provide drill-down paths for domain-specific details.

**Artifacts and interfaces:** ProjectAnswer DTO, executive status tool and Q&A benchmark.

**Acceptance tests and evidence:** Status report never calls a failed job completed; all numeric values resolve to source with window and unit.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-09 - Implement Project Agent dashboard and portfolio navigation
**Outcome and scope.** Give users a coherent top-level workspace with specialist drill-down.
**Preconditions/dependencies.** P4-08
**Existing integration to inspect.** External dashboard components from earlier phases.

**Implementation steps (ordered):
1. Create overview, goals, workstreams, problems, solutions, models, milestones, budget, timeline and approvals panels.
2. Build project chat with evidence chips, agent delegation trace, current active work and corrective actions.
3. Add navigation from project to problem solution assessment to exact model evaluation.
4. Design empty states, conflict warnings, access-denied explanations and status labels.
5. Run usability checks against representative ML engineer and business stakeholder tasks.

**Artifacts and interfaces:** Project Agent UI, portfolio board and navigation test suite.

**Acceptance tests and evidence:** A user can traverse Project -> Problem -> Solution -> ModelVersion -> Evaluation without losing provenance.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

#### P4-10 - Pass G4 Project Agent reuse and continuity benchmark
**Outcome and scope.** Prove the three-layer distinction in end-to-end scenarios.
**Preconditions/dependencies.** P4-09
**Existing integration to inspect.** All three harnesses and current benchmark framework.

**Implementation steps (ordered):
1. Create one project with two problems, at least two solutions for one problem, and shared or separate models.
2. Test project-level priority requests, linked histories, solution changes and active-job visibility.
3. Simulate removed model permission, stale decision and lost provider connection.
4. Check child identity and policy isolation; run sealed answer/authorization suite.
5. Publish project acceptance report and update compatibility matrix.

**Artifacts and interfaces:** G4 gate report and stable Project Agent template.

**Acceptance tests and evidence:** Two projects can be created from the same Project template; no duplicate harness classes or cross-project leakage.

**Primary implementation risk:** Scope creep, stale evidence, or unauthorized side effects

**PR definition of done:** typed contracts committed; negative and happy-path tests green; trace/evidence included; security scope confirmed; docs updated; no unreviewed modifications to DCLab core.

## Phase gate checklist
- [ ] All 10 work packages have linked passing PRs and owners.
- [ ] Contract, integration, authorization and negative/error tests pass.
- [ ] Data provenance and permission requirements verified for every new source.
- [ ] Every side-effecting action has an approval/receipt and recovery path.
- [ ] UI/API shows accurate proposed/running/completed status without inventing results.
- [ ] Demonstration performed: Ask: "What have we tried across churn and retention, what is blocked, and which solution should we prioritize?" Verify source-based result and workload implications.
- [ ] G4 sign-off includes evidence links, exceptions and residual risks.

## Handoff to next phase
Gate G4 provides the stable interfaces, immutable evidence and regression tests required by Phase 5.
