<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

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
