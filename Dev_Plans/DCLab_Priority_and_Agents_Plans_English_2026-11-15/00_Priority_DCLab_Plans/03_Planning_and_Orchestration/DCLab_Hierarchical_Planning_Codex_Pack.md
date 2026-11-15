# DCLab Hierarchical Planning Implementation Pack

Repository [Research_Development_DCLab](https://github.com/Shahriyar-Moradi/Research_Development_DCLab)

Prepared 2026-11-04. Initial review on commit `81e8a9009174031c64148ebd22ef035500fccd08`. The branch can advance, so **every Codex session must first inspect the current HEAD and diff**.

## Scope and architectural decision

The user's six Home choices are **Model, Plan, Report, Analyze, Prepare, Learn**. These are **intent routing hints**, not a replacement for reasoning. On 2026-11-04 they are already present in `dclab_rnd/agentic/web/src/views/home.html` and mapped to `prediction`, `business`, `reporting`, `analysis`, `preparation`, `advice`. The user-selected intent is represented by `dclab_rnd/draft/intent.py::chosen` and the Lead Agent tracks chip selection in `dclab_rnd/agents/case_agent.py`. The newest commit changed the Home acknowledgement for a user-picked kind of help. **Do not implement this feature from scratch.** First audit behavior and close verified gaps only.

The visible **Plan** intent currently means a **business question or decision** (`business`). The new **task planning engine** is an internal execution capability, and must not silently alter that product intent or rename its contract.

Existing sources of truth

| Concern | Existing implementation | Rule for this project |
| --- | --- | --- |
| Business problem decomposition | `dclab_core/problem_map.py`, `dclab_core/mapping.py` | Reuse, do not conflate subproblems with executable tasks |
| Solution decisions | `dclab_core/plan.py`, `dclab_core/contracts.py::SolutionPlan` | Preserve findings, sources, alternatives |
| Validated solution steps | `dclab_core/solution_workflow.py` | Do not rewrite safety templates or validation |
| Complex ML solution topology | `dclab_core/solution_map.py` | Preserve split and composition semantics |
| Complex ML execution and immutable revisions | `dclab_core/programme.py`, `dclab_rnd/studio/programme.py` | Keep the programme executor authoritative |
| Workflow evidence | `dclab_core/workflow_run.py` | Do not schedule these 12 conceptual steps again if the engine already does them |
| Project moves and approvals | `dclab_rnd/studio/graph.py` | All effects must pass the original checks |
| Deterministic ML engine | `dclab_rnd/studio/engine.py` | Do not rebuild splitting, fitting, selection, or final scoring |
| Tools and guarded calls | `dclab_rnd/agents/registry.py`, `dclab_rnd/intern/tools.py` | Use existing schemas, scopes and guards |
| Team and checkpoints | `dclab_rnd/orchestra/solution.py`, `dclab_rnd/orchestra/checkpoints.py` | No second LangGraph graph or database by default |
| Agent harness and traces | `dclab_rnd/agents/harness.py`, `dclab_rnd/agents/traces.py` | Reuse, redact sensitive values |
| Task-ish intern note | `dclab_rnd/intern/loop.py::write_plan` | A human-readable 2,000-character note, not an executable graph |
| Core schemas and wheel | `dclab_core/contracts.py`, `dclab_core/schemas`, `scripts/core_wheel_check.py` | Versioned, frozen schemas, and both environments |
| Benchmark | `dclab_rnd/bench/`, `docs/guides/BENCHMARK.md`, `docs/SYSTEM_EVALUATION.md` | No claims without measured outcomes |

Core planning principle

`understood objective → existing solution decision/workflow → executable task graph → validated per-task plan → existing guarded execution → independent verification → selective replanning`.

Crucial distinction. The **task graph is a compilation/coordination representation**, not a second implementation of DCLab’s ML protocol. Some solution-workflow steps correspond to a guard, an implicit operation inside an engine stage, or a person’s approval, not to another executable tool call. **No double execution.** In particular, the held-out evaluation may be invoked only using the existing protected final-stage or programme route.

## Shared operating rules for all prompts

Paste the following invariant block at the beginning of a new Codex session, or retain it as the project's controlling instructions. Each following prompt also assumes these rules.

```text
You are working in the current local checkout of
Shahriyar-Moradi/Research_Development_DCLab.

Before editing, inspect git status, current HEAD, README, CLAUDE.md,
relevant modules, their call sites, existing tests, and current API/schema
conventions. The remote review observed 81e8a900 on 2026-11-04 but the
repository may have advanced. Prefer the actual current checkout.
Never overwrite unrelated edits. Do not silently reset or stash user work.

Scope only the package in the current prompt. Use the smallest cohesive
changes, keep compatibility with existing file and PostgreSQL stores,
maintain the no-live-model test guarantees, and follow existing conventions.
Write tests that fail before the implementation and pass after. If a proposed
file/function is obsolete or not present, discover the correct current owner,
explain the adjustment, and do not create a duplicate.

Hard invariants
1. An LLM may propose a plan, never execute during planning.
2. No new unguarded write path or arbitrary code/network/file execution.
3. Existing registry tool schema, scope and authorization are authoritative.
4. Existing studio.graph / programme / engine own holdout, approvals, splits,
   leakage, model selection and final scoring.
5. No holdout rereads or mutable 'finished' artifact reuse after input changes.
6. Preserve human decisions and their provenance; a model cannot erase gates.
7. Task completion requires measured evidence, not agent self-report.
8. Do not store raw table values, secrets, personal values, sealed benchmark
   rows or hidden reasoning in task logs/traces.
9. Keep offline/no-model mode functional and backward compatible.
10. No default rollout or new dependency without demonstrated necessity.
11. No repeated side effects after checkpoint/resume or ambiguous failures.
12. No production-readiness claim from unit tests or synthetic results.

Run focused tests after each package, the repository's check-quick gate where
possible, and report exactly what ran and what was skipped. Never execute
live-model or sealed benchmark runs without the owner's explicit approval.
At the end of every package report changed files, public contract changes,
new tests, commands/outputs, blockers, open risks, and a proposed single
reviewable commit. Do not push a branch without authorization.
```

## Execution sequence

| Order | Package | What it produces | Dependency |
| --- | --- | --- | --- |
| R-01 | Audit and harden explicit Home routing | Six-choice behavior validated | None, independent of task planning |
| TP-01A | Map existing planning/execution architecture | Executable capability inventory and invariants matrix | None |
| TP-01B | Task graph and plan contracts | Minimal frozen v1 schema models | TP-01A |
| TP-01C | Graph validation and fingerprints | Safe deterministic graph check | TP-01B |
| TP-01D | Binary-workflow adapter | Real workflow converted without double execution | TP-01C |
| TP-01E | Packaging, regression and handoff | TP-01 accept/reject verdict | TP-01D |
| TP-02 | Model-assisted decomposition | Candidate task graphs with deterministic fallback | TP-01E |
| TP-03 | Per-task execution planning | Registered-tool plans, costs and preconditions | TP-02 |
| TP-04 | Guarded orchestration and persistence | Sequential scheduling, durable task states, resume | TP-03 |
| TP-05 | Independent verification and safe replanning | Evidence-based completion and selective invalidation | TP-04 |
| TP-06 | Comparative benchmark and quality gates | A/B/C measurements and promotion verdict | TP-05 |
| TP-07 | Product task visibility and controlled rollout | Read-only task board, flag, gradual rollout | TP-06 |

Do each prompt in its own Codex session or incremental working branch, and review the previous output before the next one. Do not execute the later prompts early.

---

# R-01 — Audit and finish the existing six-choice Home routing

## Codex prompt

```text
Goal
Audit and harden the already-implemented six-choice Home routing system.
This package is an independent user-intent improvement, NOT TP-01.

Current repository evidence (verify at HEAD)
- Home buttons are in dclab_rnd/agentic/web/src/views/home.html.
- Labels and tracks are Model/prediction, Plan/business,
  Report/reporting, Analyze/analysis, Prepare/preparation,
  Learn/advice.
- dclab_rnd/draft/intent.py::chosen creates confirmed user intent.
- dclab_rnd/draft/chat.py::HomeAgent and its chat flow preserve choice.
- dclab_rnd/agents/case_agent.py contains chip, chip_checked,
  chip_intent, changed_mind, chip_note and related route checks.
- tests/test_home_intent.py, tests/test_case_agent.py,
  tests/test_home_routing_points.py cover parts of this behavior.
- The 2026-11-04 commit 81e8a90 changed 'I read this as' to
  'You picked' for explicit selections.

Steps
1. Trace the selection end to end. Button click → client request →
   draft creation/update → stored intent → Home or case-agent route →
   response → follow-up messages → new project/session.
   Locate the actual JS owner, API endpoint and storage handler.
2. Produce a short written route precedence table before edits.
   Explicit selection outranks inference for this action unless the
   user explicitly changes their mind. A contradiction must be surfaced
   once rather than silently ignored or endlessly questioned. An absent
   selection uses the current model/rules routing path. Changing a choice
   must not erase unrelated conversation or user-provided information.
3. Check behavior for each of the six choices, including blank input,
   clicks before typing, typing then clicking, selecting and deselecting,
   selection after a draft exists, refresh, navigation, resume, new
   project, and multi-turn follow-ups. Follow existing product semantics
   when a category is changed after actions already started.
4. Check mixed and conflicting requests. Examples 'Model + show sales
   summary first', 'Learn + build a churn model', 'Report + clean my
   tables', 'Plan + give me an execution plan for the ML workflow', and
   'Find customer segments' which can mean analysis or clustering.
   Do not hard-code the word 'plan' as hierarchical execution routing.
5. Keep routing as a product intent hint, never authorization for builds,
   changes, holdout reads or arbitrary tool calls. Check permission and
   approval guards still apply after a selected chip.
6. Add regression cases using scripted/no-model clients. If a behavior
   already passes, keep it and do not rewrite it.
7. Add an offline route decision comparison for selected vs unselected
   requests using held-out DEV cases only. Report sample sizes, success,
   disagreements, invalid writes and added LLM calls. Do not rerun a
   sealed set or claim a new model evaluation from scripted tests.
8. If needed, make the smallest corrective edits in the existing
   frontend, draft or case-agent code. Avoid a new router framework.

Tests and verification
- Run tests/test_home_intent.py, tests/test_case_agent.py,
  tests/test_home_routing_points.py, tests/test_home_conversation.py,
  and relevant browser / API checks using the repo's commands.
- Test for an explicit choice surviving refresh and a conflicting
  Lead Agent route being refused or clarified, never silently switched.
- Test that selecting a chip alone cannot execute a model build.
- Run make check-quick where supported; use make browser-test if needed.
- Do not use live models by default.

Deliverables
- docs/guides/HOME_ROUTING_AUDIT.md with traced flow, precedence,
  scenarios, any confirmed defect and its fix, and unanswered risks.
- Only code and tests necessary for defects that are actually reproduced.
- Summary with the routing behavior before/after, measured results,
  commands and exact output, and a small reviewable commit.
```

# TP-01A — Architecture and capability mapping

## Codex prompt

```text
Goal
Produce a factually verified architecture/capability mapping for a new
hierarchical task planning layer. This is investigation and design ONLY.
Do not modify the execution runtime or any public contract.

Read in depth
- dclab_core/problem_map.py, mapping.py, plan.py, solution_workflow.py,
  solution_map.py, programme.py, workflow_run.py, orchestra.py,
  contracts.py and relevant schema registrations.
- dclab_rnd/studio/graph.py, engine.py, programme.py,
  solution.py, store.py, and stage/run APIs.
- dclab_rnd/intern/loop.py and tools.py, agents/registry.py,
  agents/runtime.py, agents/harness.py, agents/traces.py,
  orchestra/solution.py, orchestra/checkpoints.py.
- Relevant test files, Makefile, core-wheel check, BENCHMARK docs.

Deliver an auditable matrix for EACH mandatory SolutionWorkflow step.
Fields
  step_id, family, WF block, prerequisites, artifacts consumed,
  expected outputs, parent business goal, suggested task group,
  existing authority, actual callable action, guard, approval,
  verification record, dependency, failure meaning, replay safety,
  runnable classification.
Classifications
  'direct_tool', 'engine_implicit', 'guard_only', 'human_gate',
  'evidence_only', 'programme_atomic', 'unsupported'.
Document when one engine run covers several conceptual steps. Do not
make duplicate side-effect nodes for the same engine operation.

Add an independent capability inventory using the actual registry.
List tool name, scope, effect, schema requirements, permissions,
handler, studio.graph move, and whether safe to repeat. Determine
which calls can be checked without invoking the tool and which
already have side effects. Inventory status of project, artifact,
tracing, event stores and checkpoints in both file and PostgreSQL
modes. Document the differences between a flat project and a
complex programme.

Architectural decisions to settle explicitly
1. What belongs in a task graph vs the SolutionWorkflow, SolutionMap,
   WorkflowRun and Programme? Define one representative case of each.
2. What is the graph's structural source of truth and what stores
   mutable execution status? Explain the data model for revision and
   crash recovery without a duplicate database.
3. Which object owns user intent, tool authorization, run state,
   human approvals, the holdout read count, and all model results?
4. How is task execution prevented before an approved solution exists?
5. What fields are essential to v1 and what remains deferred?
6. How does one task bind to a real capability or become blocked?
7. What are the exact compatibility and upgrade requirements for the
   existing dclab_core wheel and main-product adapters?

Produce three artifacts
- docs/guides/TASK_PLANNING_ARCHITECTURE.md, target architecture,
  boundaries, integration diagram, threat model, design decisions.
- docs/guides/TASK_CAPABILITY_MATRIX.md, existing step and tool maps.
- docs/guides/TASK_PLANNING_ACCEPTANCE.md, cross-phase hard gates,
  benchmark plan, scope and exclusions.

Include a first implementation slice, one binary classification / Telco
churn scenario using a currently supported sample. Identify exactly
which existing stage produces each artifact and which operations are
only checks. Cite source file and function for each mapping.

No implementation, no new dependency, no new datastore, no new agent.
Report unresolved ambiguity instead of inventing a contract.
```

# TP-01B — Minimal typed task contracts and frozen schemas

## Codex prompt

```text
Goal
Add the smallest public, versioned, deterministic contracts needed
for task decomposition and per-task execution planning. Implement
only contracts and schema registration in this package.

Prerequisite
Read TASK_PLANNING_ARCHITECTURE.md and TASK_CAPABILITY_MATRIX.md.
Refuse to proceed if the matrix has unresolved fundamental ownership
or holdout-authorization conflicts. Inspect current dclab_core
contract/versioning/schema generation patterns first.

Suggested files (adapt to actual current conventions)
- dclab_core/task_graph.py
- dclab_core/task_execution_plan.py
- dclab_core/contracts.py (registration, minimal additions)
- dclab_core/schemas/task_graph.json and
  dclab_core/schemas/task_execution_plan.json
- frozen versioned baseline schemas using the current schema command
- dclab_core/README.md, CHANGELOG.md
- tests/test_task_graph.py, tests/test_task_execution_plan.py

TaskGraph v1 should express an immutable, structural plan
- schema_version, graph_id, revision, objective/source references,
  workflow/solution fingerprints, nodes, edges, creation provenance.
- Task node contract with id, parent id (optional), title/objective,
  source_step_ids, kind (group, action, gate, verification, blocked),
  needs, produces, dependencies, acceptance checks, capability binding,
  scope, and blocker reason where unsupported.
- Use stable bounded IDs and known enums when appropriate. Do not
  invent arbitrary executable Python actions or arbitrary URLs.
- Distinguish logical grouping edges from execution-dependency edges.
- Group nodes are not executable. Any kind other than 'action' must
  have clearly specified interpretation and never run arbitrary code.
- A task may be human-gated or check-only without inventing a tool.
- A declared action must point to a capability in existing registry,
  not copy that capability's schema.

TaskExecutionPlan v1
- graph/task/revision identity, immutable plan revision and fingerprint,
  expected input fingerprints and artifact IDs, selected authorized
  capability names, ordered bounded actions, controlled argument refs,
  acceptance/evidence requirements, compute/tool budgets, stop and
  compensation/escalation policy, plan provenance and status.
- Allow deferred arguments only as typed references to already
  declared outputs; do not permit raw model-supplied expressions/eval.
- Plans must support no-op and explicit blocker as distinct from
  successful task completion.
- Choose a minimal model based on measured adapter needs rather than
  fields added just for theoretical completeness.

Schema/version rules
- Reuse dclab_core.contracts.Contract/Versioned/_version and
  dclab_core.contracts.CONTRACTS registration conventions.
- Freeze JSON schemas at v1. Never modify a frozen version after
  publication. Verify unknown fields and missing version are refused.
- Keep core import-safe and without network, agent runtime, database
  access or circular imports. Inspect the installed wheel behavior.
- No task execution, graph traversal or model request here.

Tests
- JSON round-trip, version rejection, unknown field rejection,
  wrong enum/ID rejection, duplicate IDs, empty objective,
  untrusted argument expressions, bounded lists and schema freshness.
- Check that unchanged schemas remain byte-compatible.
- Include a minimal valid example and an unsupported-capability node.
- No network/model calls.

Deliverables
New typed contracts, frozen schemas, source documentation, failing-
before/passing-after tests. Run focused core tests and the applicable
schema/wheel checks. Report the schema diff and compatibility risks.
```

# TP-01C — Deterministic graph validator, planning invariants and fingerprints

## Codex prompt

```text
Goal
Implement pure graph and plan validation with reproducible structural
fingerprints. No scheduler, tool execution or model calls in this step.

Prerequisite
Contracts and schema baselines from TP-01B must be green.

Suggested owner
- dclab_core/task_graph.py and task_execution_plan.py or small
  pure companion modules if necessary.
- tests/test_task_graph.py, tests/test_task_execution_plan.py.

Functions to define after matching repository conventions
- validate_graph(graph, source_workflow=None, source_solution_map=None)
  -> list of actionable problems.
- ready_tasks(graph, observed_state) or equivalent PURE dependency
  traversal. Observed_state is supplied by the host, never fetched.
- graph_fingerprint(graph) and plan_fingerprint(plan), stable hashes
  over canonical semantic content, excluding volatile timestamps,
  transient statuses, raw user data, and the hash field itself.
- validate_execution_plan(plan, graph, known_capabilities) -> issues;
  the core receives a pure capability snapshot, not a runtime registry.

Requirements
1. Reject duplicate IDs, absent parents, parent cycles, dependency
   cycles, self-dependencies, dependencies on unknown/invalid nodes.
2. Reject executable group nodes and dependencies that cannot be
   satisfied with the requested outputs/artifacts.
3. Preserve source-workflow provenance. Mandatory steps must be
   accounted for by action, guard, explicit gate, evidence check or
   blocked node. The adapter may use a justified many-to-one mapping
   but must never silently omit a mandatory safety step.
4. Enforce temporal constraints from existing SolutionWorkflow
   validation, including leakage-before-model-selection, human gate,
   single final holdout read, and reporting after final evaluation.
5. Prohibit model execution intents before approval and tools outside
   the provided capability snapshot/scope. A core validator should
   never invoke a tool or import studio/agents.
6. Prevent multiple nodes from scheduling the same non-idempotent
   engine effect, especially final score and programme run.
7. Deterministic ordering when multiple nodes are ready; no parallel
   execution in v1. No node is marked complete by validation.
8. Clearly distinguish a changed revision from a changed input artifact.
9. Fingerprints must be stable across dictionary insertion order,
   restarts, file vs PostgreSQL storage, and round-trip serialization.
10. Fingerprint mismatch must be explicit and refused; never silently
    rewrite the old graph or plan. Preserve old revision for audit.

Red-team tests
- cycle of length 2 and 3, missing parent, dangling dependency,
  duplicate executable action, final-before-approval, holdout twice,
  missing leakage, cross-scope tool, missing capability, inconsistent
  input artifact reference, stale fingerprint, benign reordering.
- A valid binary case passes and has deterministic topological order.

Document exactly which conditions are enforced by the pure planning
validator and which must be rechecked by the authoritative runtime
at execution time to avoid time-of-check/time-of-use vulnerabilities.

Run focused tests and schema checks, report precise failures.
```

# TP-01D — Deterministic solution-workflow-to-task adapter

## Codex prompt

```text
Goal
Convert one validated binary-classification SolutionWorkflow into an
accurate, non-executing TaskGraph. Prove real integration without any
new ML computation or new project writes.

Prerequisites
TP-01A capability map, TP-01B contracts and TP-01C validator.

Suggested file
- dclab_rnd/tasks/adapter.py (and __init__.py)
- tests/test_task_adapter.py
- fixture based on the existing Telco churn or another verified
  binary-classification sample in the repository.

Inputs
- Validated existing SolutionWorkflow, source SolutionPlan,
  current project/data/solution metadata and, when applicable,
  verified capability snapshot from the live registry.
- Return TaskGraph + structured mapping report, no side effects.

Approach
1. Build a deterministic mapping from existing Step.id, Step.block,
   Step.kind, Step.inputs/outputs, decision/check/gate and required
   family steps to group/action/gate/check nodes.
2. Tie each actually runnable unit to an EXISTING tool or stage. For
   example the stage owner may be run_stage(data, leakage, features,
   models, final) but resolve the exact schema from the registry.
   Treat split locking as existing guarded engine behavior, not a
   second 'lock_split' operation. Use the precise actual call mapping.
3. Represent many-to-one coverage when one studio engine run
   implements multiple abstract workflow steps. One physical
   side effect = one action task with several source step IDs.
4. If a required capability is missing, represent a blocked task
   naming why; do not invent tools, mark complete, or guess data.
5. Carry artifact lineage and stable hashes, not full table rows.
6. Do not open holdout, fit, query raw values, or set approvals.
7. Use deterministic topological order for v1, with no concurrency.
8. Keep Programme execution as an atomic existing capability when
   implementing complex cases later. Do not decompose its leaves into
   new independently scheduled writes here.

Positive scenario
- Generate a valid current classification workflow from the
  existing core builder.
- Run the adapter twice and compare semantic fingerprints.
- Validate that mandatory steps, guards and gates are represented.
- Show the resulting task tree and mapping without any execution.

Negative scenarios
- Missing dataset/solution or unknown family blocks appropriately.
- An unconfirmed label/outcome never becomes a runnable modeling task.
- A planted invalid workflow is refused before adaptation.
- No phantom registered tools; no duplicate final-stage action.
- Changing the data or solution changes the relevant source hash.
- Do not mutate the original workflow or project.

Add a simple read-only CLI/developer inspection path ONLY if there is
already an appropriate pattern. Otherwise keep the adapter callable in
Python and exercised by tests. Do not add product API or UI here.

Deliver example JSON in tests/fixtures or docs, source mapping summary,
tests, exact commands and output. No runtime defaults changed.
```

# TP-01E — Cross-environment tests, regression gate and TP-01 handoff

## Codex prompt

```text
Goal
Close TP-01 as a verifiable architecture-and-contracts milestone.
Make no architectural expansion beyond fixes required by its gates.

Review files
- New TP-01 contracts, schemas, validator, adapter and tests.
- tests/test_core_contracts.py, tests/test_core_wheel.py,
  scripts/core_wheel_check.py, Makefile, requirements/base.txt,
  requirements/ci.txt, requirements/agent.lock.txt.
- tests/test_solution_workflow.py, tests/test_programme.py,
  tests/test_orchestra_solution.py, tests/test_graph.py,
  tests/test_agent_harness.py, tests/test_intern.py.

Check and repair
1. The core package imports independently and the installed wheel
   exposes the new contract. No accidental dependency from the core
   onto ML runtime, database, Web UI or agent modules.
2. Importing the agent harness in the lean agent virtual environment
   does not accidentally require NumPy/scikit-learn. The 2026-11-04
   repo had an import regression from a top-level dclab_core import
   into agent_metrics; explicitly regression-test the separation.
3. Existing generated schemas and changelog are current.
4. Backwards-compatible reading of existing project, programme,
   workflow and experience artifacts is preserved.
5. File-based and PostgreSQL tests use actual isolated test storage.
6. The adapter is read-only. A test spies on registry handlers,
   project writes, holdout reads and model gateway calls, expecting 0.
7. Test an actual binary workflow fixture, an invalid workflow,
   a missing capability and a changed input fingerprint.
8. Existing workflow, programme, home and intern tests remain green.
9. Explicitly record tests skipped because the local environment is
   not provisioned. Do not claim them passed.

Commands
- Use the repo's provided Python environments and the exact
  focused test invocation accepted by the current test suite.
- Run make core-schemas / core-wheel-check if targets exist at HEAD.
- Run make check-quick and, if feasible, make rd-check.
- Run make test-pg and make check-all only when PostgreSQL and the
  documented environments are present, no destructive setup.
- No live model, no sealed benchmark, no external API.

Write docs/guides/TASK_PLANNING_TP01_REPORT.md with
- Architecture mapping links and ownership matrix.
- Full coverage of the original seven-item TP-01 checklist.
- Which paths are physically executable vs implicit, guard or blocked.
- New contract versions and schema names.
- Test commands and their exact pass/fail/skip statuses.
- Known limits and acceptance gate pass/fail.
- Decision whether TP-02 may begin.

Do not proceed into TP-02 in this package. A failure or unresolved
ownership conflict must be reported as a blocker, not hidden.
```

# TP-02 — Bounded model-assisted task decomposition

## Codex prompt

```text
Goal
Add one model-assisted task decomposer that turns an already
validated solution workflow (and its deterministic TP-01 adapter
result) into an improved executable task hierarchy, under deterministic
validation. Preserve the rule-generated graph as fallback.

Prerequisite
TP-01 acceptance report explicitly passes. No default rollout.

Implementation
- Suggested dclab_rnd/tasks/decomposer.py and a narrowly scoped
  prompt in the existing dclab_rnd/prompts/ registration convention.
- Discover whether the existing solution_planner/decomposer roles and
  the gateway can own this role. Prefer extending a measured existing
  role only if its semantics fit; otherwise register one new role with
  harness purpose, prompt hash and model routing. Do not build an
  independent LLM client.
- Input context contains confirmed goal, sanitized data-card metadata,
  validated workflow, task-capability map, budgets, unresolved
  questions, deterministic graph and authorized tool names. Never raw
  rows or untrusted instructions promoted to system policy.
- Model proposes TASK BOUNDARIES/dependencies only; cannot alter
  mandatory solution steps, signed decisions, safety guards or run
  arbitrary tools. Do not let it execute while planning.
- Validate the proposed TaskGraph against TP-01 checks. If invalid,
  retain the deterministic graph and log refusal reasons. A bounded
  one-repair attempt is optional only if measured.
- Limit node count, depth, model calls, tokens, cost, and planning
  time; reuse existing budget conventions. Do not split each tool
  argument or micro-operation into a separate task.
- Require grounding of all proposed subtask objectives in source
  workflow steps, confirmed case facts or registered capabilities.
- Preserve source workflow fingerprints and plan provenance.
- Explicitly report unsupported tasks as blockers with reasons.

Evaluate in scripted/no-model mode using honest cases
- One step/task, multi-step churn, branch dependencies, missing
  capability, contradictory model response, hallucinated tool,
  attempts to remove safety steps, missing required data, and change
  of objective after confirmation.
- Check fallback output equals the deterministic TP-01 graph on
  invalid or unavailable LLM responses.
- Benchmark decomposition coverage and invalid proposal rate against
  deterministic adapter on dev examples.

Deliver code, tests, tracing, a prompt version, docs and a recorded
DEV result. The runtime must not execute decomposed tasks yet.
```

# TP-03 — Per-task execution planner and plan gate

## Codex prompt

```text
Goal
Plan HOW to execute one dependency-ready leaf task. Produce a typed
TaskExecutionPlan with authorized calls, concrete expected results,
verification and recovery guidance. Planning never calls a write tool.

Read dclab_rnd/agents/registry.py, intern/tools.py, agents/runtime.py,
models/gateway and the TaskExecutionPlan contract at HEAD.

Design
1. Input is one task plus current project state, declared artifact
   hashes, allowed registry tool schemas for that scope, remaining
   budget and past failures. It cannot see unrelated workspace data.
2. Registry metadata is the single authority for tool names, JSON
   argument schemas, read/write effects, scopes and project moves.
3. Create bounded ordered action intents and explicit artifact
   bindings to earlier outputs. Unknown capabilities are blockers.
4. Validate all static arguments through the existing registry.check
   or a pure schema snapshot. Revalidate resolved arguments immediately
   before runtime execution. Do not execute here.
5. Prohibit side effects from trying to 'preview' a write. Read-only
   tools may be used only when already permitted by agent policy.
6. The output requires preconditions, acceptance checks, budget,
   selected steps, failure classifications and stop/escalation rules.
7. An existing task may have no tool action because it is a gate or
   verification. Never falsely mark it as completed.
8. Scripted fallback produces deterministic plans for existing five
   studio stages. Model planning is behind an off-by-default flag.
9. The existing workflow and programme guards remain authoritative.
10. Track model/tool schema hashes in the plan so stale plans cannot
    run after changed permissions or tool definitions.

Tests
- Good plan for each supported stage, missing required tool args,
  arbitrary tool, wrong scope, write without move, forbidden data,
  missing artifact, invalid binding, stale capability hash, timeout,
  overbudget, model refusal, changed user intent.
- Confirm 0 write-tool calls during planning and no extra holdout
  accesses. Confirm fallback if model is down.

Integrate with the shared harness and gateway, not another agent loop.
Deliver code, tests and a DEV plan validity comparison. No scheduler.
```

# TP-04 — Sequential guarded execution and durable task progression

## Codex prompt

```text
Goal
Execute one validated TaskExecutionPlan at a time, using existing
registered tools and the original studio guard. Add durable task
progress and safe resume without duplicating side effects.

Architecture
- Inspect SolutionTeam, Intern, Registry, studio.graph,
  studio.programme, jobs/checkpoints, harness/traces and the existing
  project stores. Choose ONE existing orchestration host for the first
  binary-use-case integration. Explain why before edits.
- Do not build a second general scheduler or LangGraph graph. Introduce
  a small dependency-ready task dispatcher only if the existing host
  has no appropriate equivalent.
- Keep structural TaskGraph immutable. Store task attempts and
  transitions through current event/project persistence conventions.
- A task attempt records graph/plan versions, source and input hashes,
  declared tool actions, observed results, timestamps, decision IDs,
  evidence references, budget consumption, and checkpoint pointer.
- Before every action recheck current project state, input fingerprints,
  registry schema version, write guard and human approvals.
- All calls go through Registry.call/acall with the intended scope.
  No direct studio writes by the new task runtime.
- Prevent a retry from repeating an unknown-completion action. After
  timeouts/crashes reconcile project state and evidence before any
  repeat. Document each action's safe repeat/idempotency policy.
- Respect stop/pause/resume and existing job budgets.
- Sequential only, no fan-out/parallelism.
- Programmes are atomic existing operations unless a future package
  proves their leaves can be safely checkpointed individually.
- Do not call final/holdout early. Never run another holdout read to
  recover from a failed task.

Acceptance demonstration
- The Telco binary workflow is represented as a task graph.
- Dispatcher runs permitted stages in dependency order through real
  existing tools using a temp isolated project fixture.
- Each stage leaves a task attempt and existing stage evidence.
- Process interruption at several boundaries resumes from saved
  state. No valid completed stage is rerun and no double writes occur.
- A denied guard prevents the call, records a blocker and preserves
  original state.
- A stale graph or changed dataset is refused before dispatch.

Add integration tests on file store and PostgreSQL where provisioned.
Keep off by default. No new user-facing API without a strong need.
```

# TP-05 — Independent task verification, recovery and selective invalidation

## Codex prompt

```text
Goal
Make task completion evidence-based and implement bounded, safe
replanning when a task fails or source artifacts change.

Verification
- Read task's acceptance criteria, evidence records produced by
  actual existing tools, graph/plan fingerprints, source data/solution
  hashes, project approvals and holdout usage.
- Decide verified / failed / blocked / stale, with machine-readable
  reason, evidence links, and violated criteria. Do not accept an
  agent's 'done' statement as verification.
- Separate the verifier from the planner/executor. Start with
  deterministic checks; add an advisory model reviewer only if it
  helps on measured cases and cannot override deterministic failures.
- Record each result as an existing experience/ledger event. Create a
  standalone versioned TaskVerification contract only if required for
  interoperability and after updating schema/wheel tests.

Replanning
- Classify transient failure, invalid tool arguments, changed input,
  missing capability, policy refusal, missing human approval,
  irreversible action ambiguity and domain-science failure.
- For transient failures, reconcile side effects before retry.
- For changed inputs, invalidate only downstream nodes whose
  dependency fingerprints are stale; preserve independent verified
  tasks, human decisions and immutable prior results.
- For a policy refusal, missing capability or consumed holdout, stop
  with an actionable blocker. Do not 'reason around' a guard.
- New plan revision must pass the same deterministic validator.
- Limit retries and replan attempts; report exhaustion transparently.
- Updating the objective or mandatory safety steps needs an explicit
  new confirmed solution revision, not an automatic task rewrite.

Fault-injection tests
- Tool returns error before any write, timeout after successful write,
  restart mid-task, replan cycle, changed source table, changed schema,
  changed solution, approval revoked, blocked related table leakage,
  already-consumed holdout, duplicate final task, and independent
  sibling task surviving another sibling's failure.
- In every case verify no unauthorized write, no additional holdout
  read, no repeated valid work and honest status/evidence.

Deliver implementation, persistence tests, recovery timeline and
failing-before/passing-after test evidence. Maintain the old fixed
workflow as unchanged fallback.
```

# TP-06 — Evaluation suite, baselines and release gates

## Codex prompt

```text
Goal
Prove, using the existing benchmark infrastructure, whether task
planning improves DCLab rather than simply consuming more tokens.

Arms
A = current fixed workflow and original execution.
B = deterministic task decomposition + deterministic task plans.
C = model-assisted task decomposition/planning + guarded execution,
    independent verification and replanning.

Before running
- Inventory existing dclab_rnd/bench and agent_eval/system_tasks.
- Freeze DEV evaluation cases and gate rules in code before results;
  record versions, datasets, seeds, prompt hashes, capability schema
  hash, commit, cost cap, and project/data fingerprints.
- No sealed reuse, no live-model costs without owner approval.
- Keep cases from evaluation out of training and prompt tuning.

Case strata
- Clear model task, ambiguous intent, explicit Home chip route,
  churn binary, segmentation vs reporting ambiguity, multi-table
  fraud dependency, missing label, incomplete data, contradictory
  request, unsupported approach, human approval, holdout already
  used, guard denied, transient failure, stale input, resume/retry.
- Use fixed deterministic fixtures first. Distinguish synthetic
  outcomes from real-user tasks and production performance.

Metrics
- End-to-end correctness and verified completion rate.
- Task coverage, valid graph/plan rate, proper blocked/abstention rate.
- Unauthorized tool calls, false completion, holdout violations,
  leakage-policy bypasses, repeated non-idempotent actions.
- Failure recovery by category, re-execution avoided, provenance.
- Extra prompts, latency, compute, tokens, EUR, user confirmations.
- Paired comparison per case, with intervals/uncertainty and
  separate failure counts. Do not hide any red layer in a total score.

Promotion rules
- Hard 0 critical safety violations for all runs.
- No regression in the existing required workflow and route tests.
- A predeclared positive gain in end-to-end task completion or
  recovery versus A on matched cases with acceptable cost and delay.
- Missing evidence or insufficient sample size means NOT PROMOTED,
  not a favorable inference. Define numerical non-safety thresholds
  and sample sizes before the test, document justification.
- Do not promote C solely because it beats B on dev cases.

Deliver a new benchmark layer/track mapping using existing conventions,
result files, comparison report, CI-compatible offline suite, and an
explicit PASS / FAIL / INSUFFICIENT-EVIDENCE recommendation. Never
silently switch the product default.
```

# TP-07 — Product task visibility, opt-in rollout and hardening

## Codex prompt

```text
Goal
Expose meaningful task progress in DCLab's existing white/light
Home and project experience, then enable the planner only after
its TP-06 gates pass. Preserve the six intent buttons exactly.

Rules
- Model/Plan/Report/Analyze/Prepare/Learn remain intent selections.
  'Plan' continues mapping to business intent. The new execution
  planner appears as a project-specific TASK PLAN and progress panel,
  not a new seventh intent or a renamed existing one.
- Use the existing web/src build process and design tokens. Never
  hand-edit generated static/app outputs without rebuilding sources.
- Use existing session/task board conventions where possible.
- No visual overhaul, no dark-mode-only UI and no second product.

User experience
- Read-only tree of task groups and leaf tasks, ordered dependencies,
  completed/running/blocked/stale status, concise reason, evidence
  link, source workflow step, selected capability, plan revision,
  approved/awaiting approval indicator and current budget usage.
- Make it clear when a task is a guard/check instead of tool work.
- Show why a blocked task requires data, permission or a person.
- Show a plan revision diff and which valid tasks were preserved.
- Do not display raw private rows, secrets or model hidden reasoning.
- Correct accessibility labels, keyboard navigation, screen reader
  status, responsive narrow-width layout, loading/error/empty states.

Backend and rollout
- Reuse existing project API/service and task event store. Add only
  minimal read-only endpoints under appropriate role/owner guards.
- Use current feature flag/settings conventions. Start disabled.
- If required gates did not pass, provide only developer-preview
  views and no default execution switch.
- Ensure old/legacy sessions render unchanged, and a user who never
  selects a chip still follows current Home routing behavior.
- Add audit trail for a switch, explicit user confirmation before
  any newly introduced write behavior, and straightforward rollback.

Tests
- Unit, API, file/PostgreSQL, browser 1280 and 375px, resume,
  permission, role, no-data and blocked-state tests.
- Browser evidence via existing screenshots and test runner, no
  fabricated screenshots or timings.
- Run make browser-test / product-e2e / check-all when provisioned.

Deliver code, tests, docs, screenshots if actually generated, feature
flag, rollback plan, release note, measured results, and a final
unresolved-risk list. Do not claim all of DCLab is production-ready.
```

---

## TP-01 acceptance checklist

- [ ] Source mapping between `SolutionWorkflow`, `SolutionMap`, `WorkflowRun`, Studio stages and `Programme` leaves is complete, and no logical step becomes a duplicate side effect.
- [ ] Ownership is explicit for input truth, execution state, authorization, approvals, holdout count, budget, provenance, and mutable run status.
- [ ] `TaskGraph` v1 and `TaskExecutionPlan` v1 exist as typed, versioned, registered, schema-frozen contracts.
- [ ] Deterministic dependency, cycle, mandatory-step coverage, capability-scope and safety validation exists.
- [ ] Canonical stable fingerprinting is tested under serialization/reordering, and stale-input refusal is explicit.
- [ ] One supported binary classification workflow maps to a task hierarchy without executing a model or reading holdout.
- [ ] Negative graph tests and compatibility with the lean agent environment pass.
- [ ] Existing tests, schemas, core-wheel and regression checks are reported with honest pass/fail/skip status.
- [ ] A reviewable TP-01 acceptance report decides whether TP-02 can begin.

## Why this order is important

Starting with a new executor would duplicate `studio.engine` and `programme.run`. Starting with a model decomposer before a graph validator would permit invented tasks. Starting with dynamic replanning before idempotency/reconciliation would risk repeating irreversible actions and the final holdout read. Starting with UI would confuse the user's **Plan** intent with execution planning. The architecture, contracts, adapter, runtime and evaluation order is therefore intentional.

## Recommended handoff from each Codex run

Use a factual handoff, not a confidence claim. Ask Codex to print the base commit and new commit, changed files, decisions and tradeoffs, tests that were actually run, failures and skips, schema diffs, compatibility risks, safety behavior, regression evidence, and the exact work remaining for the next package. Keep each package small enough to inspect in a normal pull request.
