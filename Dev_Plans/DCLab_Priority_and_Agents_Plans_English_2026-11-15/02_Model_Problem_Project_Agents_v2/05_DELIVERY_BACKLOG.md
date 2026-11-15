# Work Package Backlog and Delivery Tracker
**Version:** 2.0  |  **Date:** 10 November 2026  |  **Status:** proposed engineering plan; not a claim of implemented features.

All work packages begin in **Not started** state; existence of related DCLab files is not completion evidence.
Change status only when a linked PR, tests and release-gate evidence exist.

## Status workflow
`Not started -> In progress -> In review -> Verified -> Done`; `Blocked` is a temporary state with owner and unblock action. Reopened packages move back to In progress.

## Recommended pull-request boundaries
Each work package can be one PR or a reviewable series if required. Avoid a monolithic framework rewrite. Begin with a read-only golden path before exposing any write or retraining capability.

## Phase 0: Foundation, DCLab Gap Audit and Reference Scenario
**Phase gate:** G0. **Packages:** 9.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P0-01 | Inventory existing DCLab capabilities | Previous approved package and phase prerequisites | Not started | Each reused capability has a confirmed contract and test evidence; no proposed external behavior silently depends on an untested internal function. |
| P0-02 | Ratify independent-service architecture | Previous approved package and phase prerequisites | Not started | Architecture review confirms three agent/harness templates, no cloned harness per instance, and DCLab running unchanged when external service is down. |
| P0-03 | Domain model and entity cardinalities | Previous approved package and phase prerequisites | Not started | Contract validators reject ambiguous IDs, cross-workspace links, missing versions and invalid transitions. |
| P0-04 | Curate churn dataset with training provenance | Previous approved package and phase prerequisites | Not started | Independent rerun loads identical snapshot and split; leaks and licensing violations block use. |
| P0-05 | Produce versioned baseline model and metric evidence | P0-04 | Not started | A rerun reproduces results within recorded numerical tolerance; metrics include window, threshold, cohort and source ref. |
| P0-06 | Specify evaluation and safety benchmarks | Previous approved package and phase prerequisites | Not started | Every critical pathway has a negative test and traceable expected outcome; sealed cases are not leaked into prompts. |
| P0-07 | Define security, tenancy and risk policies | Previous approved package and phase prerequisites | Not started | Unsafe operations are denied in a policy simulation independent of agent outputs; privacy owner approves data paths. |
| P0-08 | Validate DCLab adapter feasibility | Previous approved package and phase prerequisites | Not started | External service continues to operate in read-only degraded mode when DCLab is unavailable; no core modifications required for the spike. |
| P0-09 | Create project delivery and PR conventions | Previous approved package and phase prerequisites | Not started | Every package has one accountable owner, a failing-test-first definition and a reviewable evidence link. |

## Phase 1: Shared Infrastructure, Template Registry and Agent Factory
**Phase gate:** G1. **Packages:** 12.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P1-01 | Bootstrap standalone external service | P0-02 | Not started | Cold start/health smoke tests pass with mocked DCLab provider and no LLM credentials. |
| P1-02 | Implement authoritative persistence and migrations | P0-03 | Not started | Wrong-tenant reads fail; invalid state edges roll back; migration up/down and restart persistence tests pass. |
| P1-03 | Create versioned Agent Template Registry | P1-01 | Not started | Older agent instances resolve pinned template version; new entity creation changes only manifest and bindings. |
| P1-04 | Implement Agent Factory and identity lifecycle | P1-02, P1-03 | Not started | Two instances of the same type share the same class/harness binaries while retaining isolated scope and state. |
| P1-05 | Implement shared execution kernel | P1-02, P1-07 | Not started | Every visible response and tool use is traced; step/time limits terminate safely; no scientific metric computed by LLM. |
| P1-06 | Implement three specialized Harness contracts | P1-03, P1-05 | Not started | Attempting a Problem transition with Model Harness is refused; adding a new churn model requires no harness subclass. |
| P1-07 | Build typed Capability and Tool Registry | P0-08 | Not started | No unregistered tool executes; unsupported capability surfaces a typed unsupported result, never a hallucinated success. |
| P1-08 | Implement context assembly and memory layers | P1-02, P1-10 | Not started | The agent correctly reports missing facts and ignores inaccessible or superseded memories; no raw dataset bulk injected into prompts. |
| P1-09 | Implement durable job and approval-aware workflow execution | P1-02, P1-10 | Not started | Kill/restart does not duplicate effectful work; approval refusal prevents invocation of provider write endpoint. |
| P1-10 | Implement authorization and human-approval boundaries | P0-07 | Not started | Negative authorization tests run independently of LLM; no actuation can bypass guard with a direct tool call. |
| P1-11 | Implement observability, evidence and reproducibility | P1-05, P1-07 | Not started | Every numerical claim in a test answer resolves to a canonical artifact or is marked unsupported. |
| P1-12 | Expose minimal API, SDK and proof-of-reuse scenario | P1-04, P1-06, P1-11 | Not started | Three instances with isolated records and shared kernel can be created, inspected, and resumed via API. |

## Phase 2: Model Agent Template and Specialized Model Harness
**Phase gate:** G2. **Packages:** 10.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P2-01 | Build ModelFamily/ModelVersion onboarding adapter | P1-04, P0-05 | Not started | Onboard two classifiers using one template/harness; unsupported format reports capability gap rather than failing silently. |
| P2-02 | Implement immutable model knowledge and lineage graph | P2-01 | Not started | Agent answers "Which data and features trained v2?" with correct pinned references and no guesses. |
| P2-03 | Implement Model Agent conversational context and actions | P2-02 | Not started | Multi-turn follow-ups preserve version/cohort/threshold; unsupported requests have grounded explanations. |
| P2-04 | Implement versioned deterministic evaluation | P2-02, P1-09 | Not started | Metric calculations match independent oracle; stale, unlabeled or incompatible cohorts produce explicit errors. |
| P2-05 | Implement feature, prediction and quality diagnostics | P2-04 | Not started | A synthetic null spike and a distribution shift are detected; missing labels never result in invented recall. |
| P2-06 | Implement monitoring baselines and per-model policy | P2-05 | Not started | The same window/policy yields reproducible reports; changed thresholds create a new policy version, not silent retroactive changes. |
| P2-07 | Implement controlled retraining and champion-challenger | P2-04, P1-09, P1-10 | Not started | Duplicate submissions start only one effectful job; holdout reuse and missing consent are refused; a candidate is not auto-deployed. |
| P2-08 | Implement deployment-state adapter and safe actions | P2-07 | Not started | An unapproved proposal never changes production; existing deployment state is read from provider, not inferred from a chat message. |
| P2-09 | Implement Model Agent screens and evidence views | P2-03, P2-06 | Not started | User can open the exact metric source, compare versions and see whether a job is proposed/running/completed. |
| P2-10 | Gate model template compatibility and release slice | P2-09 | Not started | No metric hallucinations in deterministic tests, no unauthorized writes, successful onboarding without agent/harness code copy. |

## Phase 3: Problem Agent Template and Specialized Problem Harness
**Phase gate:** G3. **Packages:** 10.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P3-01 | Implement Problem entity and Problem Agent identity | P1-04, P0-03 | Not started | Two problems share one Project yet keep independent identity, conversations, policies and solution histories. |
| P3-02 | Implement Problem Context Graph and evidence resolution | P3-01, P1-08 | Not started | A newly updated problem goal supersedes old chat assumptions; inaccessible linked model evidence is never leaked. |
| P3-03 | Implement conversation memory and decision continuity | P3-02 | Not started | Correction in a later turn takes precedence; agent can cite which event changed its understanding. |
| P3-04 | Implement first-class Solution Portfolio under each problem | P3-02 | Not started | Changing current solution preserves old alternatives, explicit decision rationale, and linked evaluation records. |
| P3-05 | Implement Decision and Assumption Ledger | P3-04 | Not started | Missing historical rationale produces "not recorded" rather than a reconstructed fiction. |
| P3-06 | Implement problem data sources, feature maps and business KPI contracts | P3-02 | Not started | Agent never represents recall as churn rate; every KPI states population, period, source and missing-data caveats. |
| P3-07 | Implement multi-criteria solution assessment | P3-04, P3-06 | Not started | Better model AUC alone cannot be reported as proven customer-retention improvement. |
| P3-08 | Connect problem decomposition, solution planning and task readiness | P3-07 | Not started | The agent cannot mark task complete from chat claims; blocked prerequisites show a specific remedy. |
| P3-09 | Enable Model Agent discovery and problem-level delegation proposals | P3-08, P2-10 | Not started | An unrelated model or wrong version cannot be delegated to without explicit grant; responses keep provenance. |
| P3-10 | Deliver Problem Agent screen and sealed G3 tests | P3-09 | Not started | One churn problem successfully preserves solution and conversation revisions; independent second problem uses same template and harness. |

## Phase 4: Project Agent Template and Specialized Project Harness
**Phase gate:** G4. **Packages:** 10.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P4-01 | Implement Project Agent identity and lifecycle | P3-10 | Not started | Replacing the active solution or one problem model never recreates Project Agent identity. |
| P4-02 | Implement project-to-problem/model relationship graph | P4-01 | Not started | Every project view identifies correct child agents and relation permissions; linked model cannot leak to unauthorized project. |
| P4-03 | Implement project-level history, conversations and decisions | P4-02 | Not started | Agent can distinguish a project-level decision from an unresolved problem-specific hypothesis; restricted conversation remains private. |
| P4-04 | Implement project objectives, budgets and resource constraints | P4-01 | Not started | A plan exceeding a budget cannot start unauthorized jobs; the failure explains which limit blocks the work. |
| P4-05 | Implement project portfolio and solution strategy | P4-02, P4-04 | Not started | Agent can prioritize churn and fraud work with explicit criteria without claiming false causal business impact. |
| P4-06 | Implement project milestones, workflow and task coordination | P4-05 | Not started | Completed milestone requires verified child task evidence; project view survives service restart and revisions. |
| P4-07 | Implement cross-problem governance and approvals | P4-06 | Not started | Cross-problem writes fail unless both project and affected resource policies allow them. |
| P4-08 | Implement project intelligence Q&A and executive synthesis | P4-06 | Not started | Status report never calls a failed job completed; all numeric values resolve to source with window and unit. |
| P4-09 | Implement Project Agent dashboard and portfolio navigation | P4-08 | Not started | A user can traverse Project -> Problem -> Solution -> ModelVersion -> Evaluation without losing provenance. |
| P4-10 | Pass G4 Project Agent reuse and continuity benchmark | P4-09 | Not started | Two projects can be created from the same Project template; no duplicate harness classes or cross-project leakage. |

## Phase 5: Typed Multi-Agent Collaboration and Workflow Orchestration
**Phase gate:** G5. **Packages:** 9.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P5-01 | Define typed Agent-to-Agent protocol | P4-10 | Not started | Unknown target, incompatible version, missing approval or changed arguments fail deterministically before execution. |
| P5-02 | Implement deterministic capability discovery and routing | P5-01 | Not started | Wrong-agent routing cases are caught; authorization is checked again by the target harness. |
| P5-03 | Implement task-level decomposition and typed plans | P5-02 | Not started | Plans cannot reference nonexistent artifacts or execute unapproved literal SQL/URLs; absent evidence stops the plan. |
| P5-04 | Build durable orchestration and dependency scheduler | P5-03 | Not started | Crash/restart does not rerun a completed retrain; denied step stops dependent tasks without false completion. |
| P5-05 | Implement recovery, timeout and circuit breakers | P5-04 | Not started | No automatic retry of a potentially non-idempotent write without verified receipt and safe retry key. |
| P5-06 | Implement plan budget, delegation depth and approval limits | P5-04 | Not started | Recursive ping-pong loops terminate; child agents cannot bypass parent budget with a fresh run ID. |
| P5-07 | Implement evidence merging, contradiction detection and review | P5-03 | Not started | Conflicting metric evidence is flagged; the agent does not average incompatible recall scores into a fake answer. |
| P5-08 | Build complete churn investigation workflow | P5-07 | Not started | No fabricated scores, no unauthorized retraining, and a complete evidence-linked task tree across three agents. |
| P5-09 | Run sealed collaboration benchmark and performance ablations | P5-08 | Not started | No unjustified agent proliferation: more complex architecture must have documented net benefit. |

## Phase 6: Event-Driven Monitoring, Investigation and Continuous Lifecycle
**Phase gate:** G6. **Packages:** 9.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P6-01 | Define canonical lifecycle EventEnvelope | P5-09 | Not started | Duplicate and out-of-order messages preserve one coherent audit timeline without retriggering identical effects. |
| P6-02 | Implement monitoring collectors, windows and source health | P6-01 | Not started | No alert is presented as model deterioration when underlying sample size or collection quality is insufficient. |
| P6-03 | Implement severity, hysteresis and event routing policies | P6-02 | Not started | One sustained drift produces bounded investigations; urgent high-impact event reaches correct agent and owner. |
| P6-04 | Implement persistent Investigation entity and ownership | P6-03 | Not started | A duplicate alert links to an existing investigation, not an unbounded second workflow. |
| P6-05 | Implement safe diagnostic playbooks and triage | P6-04 | Not started | Synthetic concept shift is distinguished from data-quality failure where fixtures support that distinction; missing labels remain uncertainty. |
| P6-06 | Implement policy-gated improvement and retrain triggers | P6-05 | Not started | Feature drift alone cannot automatically deploy or claim concept drift; policy refusal is visible and audit-safe. |
| P6-07 | Implement outcome feedback and decision learning | P6-06 | Not started | Agent never asserts a retention lift from model recall alone; decisions remain auditable after supersession. |
| P6-08 | Implement operations dashboards, alert controls and incident runbooks | P6-07 | Not started | Operators can identify whether a model alert is data, performance, or service-related and see who owns the next action. |
| P6-09 | Pass proactive lifecycle gate with fault injection | P6-08 | Not started | No unapproved effectful actions, no alert storm, every conclusion grounded in retained evidence. |

## Phase 7: External Dashboard, API Productization, Generalization and Production
**Phase gate:** G7. **Packages:** 10.
| Package | Objective | Depends on | Status | Acceptance evidence |
|---|---|---|---|---|
| P7-01 | Deliver complete external dashboard information architecture | P6-09 | Not started | A user can locate any current entity, track changes, and drill from a decision to underlying data without relying solely on chat. |
| P7-02 | Version public APIs and publish integration SDK | P7-01 | Not started | A separate sample client creates project/problem/model instances, lists evidence and watches a task through supported APIs. |
| P7-03 | Implement outbound events, webhooks and connector packaging | P7-02 | Not started | DCLab downtime reduces only provider-dependent operations; external identity, UI and audits remain available. |
| P7-04 | Harden tenant security, privacy and secret management | P7-03 | Not started | No cross-tenant data access in test matrix; compromised tool output cannot authorize action or expose credentials. |
| P7-05 | Harden reliability, performance and failure isolation | P7-04 | Not started | API stays responsive under worker saturation; data restored with consistent entity/decision histories. |
| P7-06 | Implement reproducible CI/CD and versioned releases | P7-05 | Not started | An older pinned agent remains runnable or has an explicit migration path; failed template rollout can be reversed. |
| P7-07 | Prove template reuse on 5 models, 3 problems and 2 projects | P7-06 | Not started | 5/3/2 gate passes without per-entity harness/agent class; any new adapter is versioned and generic to a capability or model kind. |
| P7-08 | Benchmark accuracy, authorization, task completion and cost | P7-07 | Not started | All critical security/safety cases pass; performance and value thresholds are explicitly approved before launch. |
| P7-09 | Complete customer onboarding, operator docs and governance | P7-08 | Not started | New engineer can onboard a supported model/problem/project from docs alone without changing platform source. |
| P7-10 | Run production cutover and post-launch controls | P7-09 | Not started | 5/3/2 acceptance, safety, docs, rollback and incident operations are verified; unimplemented capabilities are disclosed. |

## Release demonstration sequence
1. G0: audited integration map and reproducible churn baseline.
2. G1: one factory creates all three template types with independent context and shared policy.
3. G2: Model Agent answers model-version questions and executes a safe evaluation.
4. G3: Problem Agent explains why a solution was selected, using actual evidence.
5. G4: Project Agent aggregates two problems and delegates authorized work.
6. G5: typed project-to-problem-to-model delegated investigation survives restart.
7. G6: a drift event opens one deduplicated investigation, not an unsafe auto-deployment.
8. G7: independent external UI/API and five-model/three-problem/two-project template reuse demonstration.

## Quality standard for every PR
- Purpose, before/after behavior and scope boundary stated.
- Positive and negative unit tests plus integration tests where relevant.
- Input/output schemas, errors, idempotency and version handling documented.
- No model score without evaluation version, dataset window and source.
- No training/data use without provenance and rights.
- Security and tenant access reviewed at actual tool execution.
- Trace, job receipt, audit and rollback included for state changes.
- Evidence and link to the work-package row recorded.
