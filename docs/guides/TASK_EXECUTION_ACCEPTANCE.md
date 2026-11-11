# TP-04 — guarded sequential task execution

Status: complete under the owner's focused-test instruction; independent review passed (2026-11-12).
Scope: TP-04, using the existing basic binary-workflow adapter. No new product route or default rollout.

## Host and execution boundary

The existing `SolutionTeam` is the only orchestration host. Its stage order, leakage audit, human pauses, holdout
approval and canonical report remain authoritative. `SolutionTeam(..., task_execution=True)` explicitly enables
the developer path. The default remains false. The programme path remains with its existing atomic owner; the
task mode refuses to flatten a programme into independent stage writes.

`TaskDispatcher.dispatch(project_id, task_id, plan=None)` executes at most one validated atomic plan. With no plan
supplied, it asks the existing TP-03 deterministic planner. It refreshes current project metadata, immutable source
graph, registry capability descriptors, actual dataset bytes and observed owner evidence. `resolve_action` then
rechecks typed references, the registered argument schema, current permissions, input hashes and remaining budget.
The effect runs through `Registry.call(..., scope="project")`, the existing `ProjectGuard`, and `engine.execute`.
The new dispatcher does not write a solution, approve a gate, fit a model, select a split or score a holdout itself.

The physical chain is data → leakage → features → models → owner approval → final → export. Final remains one
existing atomic operation. The task export uses the registered `export_notebook` capability; the team's existing
canonical solution report is then read normally. These are distinct artifacts, and export success is not proof of
all report acceptance criteria.

For example, if data processing finishes and the process stops before its task checkpoint is saved, the next
dispatcher compares the stored stage run with that attempt. A matching completed owner record is reconciled and
the next stage can proceed. The data processing is not repeated.

## Durable attempts and exclusive turns

Task attempts use the solution team's existing SQLite/PostgreSQL checkpoint saver under the `task_execution`
namespace of the same project thread. There is no second graph, datastore, task table or project-document field.
Each snapshot has a new channel version, including the PostgreSQL blob version. Old snapshots remain history.
The attempt intent is durably saved before the registry call. An ordinary later save of a project cannot erase it.
Complete source decisions are not duplicated into checkpoints because even a confirmed positive-class label can
be a private table value. The source plan and workflow are rebuilt from the authenticated current project and
compared with the immutable source hashes already in the task graph.

Each attempt stores its ID and task, graph/plan revisions and fingerprints, full typed plan, input hashes, declared
action, start/finish times, owner run/evidence references, complete output-record hash, decision IDs where available,
budget reservation/usage and checkpoint pointer. Results are code-owned statuses and references, never raw tool
replies, table values or provider error text. The existing decision ledger remains best effort: unavailable links
are explicitly recorded as unavailable rather than fabricated or used to erase valid stage-completion evidence.

`Projects.turn(project_id, purpose="project", require_process=False)` is the small added storage interface. The
file backend reuses its thread and advisory file locks, scoped by workspace directory, project and lock purpose.
PostgreSQL uses a workspace/project-scoped advisory session lock with shared thread-local reentrancy across store
instances. Lock connections are detached from the data pool before waiting, so a one-connection pool and waiting
dispatchers cannot starve the owner of the connection it needs to save its result. The detached session is unlocked
and closed on exit.

The solution invocation lock is separate from the project effect lock. A graph worker can therefore acquire the
effect lock without waiting for its own parent thread. Stage effects, stage-record writes/clears and project writes
reuse the effect lock. Existing intentional owner reruns remain existing behavior; they make changed task evidence
stale rather than silently authorizing a task retry.

Task mode requires a cross-process lock. A file platform without `fcntl` refuses it before execution. Legacy file
operations keep their old thread-only fallback. Checkpoint reads authenticate project ownership through the project
store first; shared PostgreSQL checkpoint tables do not grant another workspace access to a plan or paused question.

## Resume, unknown effects and holdout

An unfinished opted-in graph checkpoint resumes with `graph.invoke(None, config)`. A human answer still uses the
existing interrupt/resume path. Task checkpoints are never deleted to restart a walk. A legacy-mode team refuses
to bypass an existing task namespace; callers must resume the opted-in host.

Engine stage lineage associates the owner run with its task attempt through `RunRecord.parent`. This is
nonsemantic provenance: the attempt ID is not inserted into scientific inputs or their fingerprint. Existing
lineage replay semantics are retained.

Reconciliation checks the typed completed stage run, project/stage/attempt identity, current dataset/solution/
settings/configuration and upstream run IDs, owner project metadata, and the complete canonical stage-record hash.
The lineage input fingerprint alone is insufficient because it excludes outputs, code and upstream run IDs.
Sampling is the existing engine's responsibility; the dispatcher does not recompute its folds or infer extra
scientific proof from a recorded stage.

Final reconciliation additionally requires the matching consumed holdout receipt, split fingerprint, final evidence
and exactly one recorded project holdout use. An intent-only receipt, missing final record, mismatch or incomplete
owner metadata blocks. No recovery path supplies a reuse reason or invokes final to discover whether it finished.
The actual owner permission is observed and bound to current prerequisite evidence. A permission already present
before the models attempt, a consumed permission, or ambiguous approval history is not adopted as fresh permission.
This host binding does not pretend that legacy approval entries were cryptographic source-bound signatures.

A crash before the effect may leave only an intent. That outcome remains unknown and blocks automatic retry.
If a completed stage record exists but the engine did not finish saving its own project metadata, the task also
stops for owner reconciliation. TP-04 does not repair scientific owner state directly. After a completed export,
recovery requires its owner completion event, current source-record hash and both registered artifact hashes.
Partial export files or a bare `captured` event are insufficient.

Operation completion and scientific acceptance are separate. Attempts explicitly keep
`acceptance_verification="pending_tp05"`. The metadata receipts used for dependency readiness verify only the
existing owner operation and its lineage. They do not invent missing per-fold class counts, optional threshold
proof or canonical report limits. TP-05 owns the independent acceptance verdict and selective recovery policy.

## Stops, budgets and privacy

The existing job checkpoint is checked before each task and after its durable result. A started deterministic
stage finishes before Stop takes effect; stopping after completion does not lose its attempt or repeat the work.
No automatic retry or parallel dispatch is introduced. An unresolved attempt blocks later effects.

Tool-call allowance is charged durably before execution and cannot reset on resume. Time is reserved before the
effect, reconciled conservatively after a crash, and checked between tasks. A known owner duration beyond the
plan limit retains its overrun and prevents another action. This is a boundary budget, not hard cancellation of
an in-flight scientific computation. Unknown time is not reclaimed. The host's existing step/euro limits remain.

The first task-execution slice uses the existing gateway's no-model scope. No model request is added, and this
mode does not spend on model planning, explanations or critiques. The off-by-default TP-03 proposal API remains
separate. Existing default solution-team behavior and its model settings are unchanged.

Only names, hashes, counts, statuses and evidence/decision identifiers enter new task records and traces. Actual
dataset bytes are checked through `studio.data.data_path`; no raw cells are read by the dispatcher or copied into
its checkpoints. The existing guarded engine reads the data it is authorized to process.

## Verification ledger

The new integration fixture is an invented, seeded, 240-row Telco-shaped binary table. Its approvals are made through
the existing owner approval function inside isolated temporary projects. It is not a live customer table or a
sealed benchmark. Tests use the existing fast engine profile and no live model.

Test first: the initial positive real-stage/no-repeat test and unready-task test failed because the dispatcher
module did not exist. Implementation then passed these tests and the complete five-stage/export chain.

Early affected checks passed 14 PostgreSQL dispatch tests and then 38 file/PostgreSQL cases as crash, source,
schema, guard, concurrency, budget and host-resume coverage expanded. Focused review reproduced three defects:

- Advisory lock connections could exhaust a one-slot PostgreSQL data pool. Detaching those sessions fixes it.
- A bare checkpoint read could expose another workspace's task state. Project ownership is now checked first.
- A known duration overrun could disappear after crash recovery. Reconciliation now preserves the conservative
  charge and overrun before another action is allowed.

Review also required the file task path to refuse unsupported process locking. Targeted tests cover that boundary,
workspace-scoped file locks, separate invocation/effect locks and reentrant store instances. Final independent
review passed with no remaining correctness findings. Final affected test commands/results and completion-corpus
checks follow below. A final self-check found the private-class-value risk in complete source-plan payloads;
the hash-only source binding fixed it, and its failing-before/passing-after regression runs on both backends.

- **46 dispatcher tests passed**, with no skips: 22 on files and 24 on PostgreSQL. This includes the complete
  real-tool chain, host pause/crash resume, unknown effects, receipts, isolation, one-slot pool, privacy and budgets.
  Command: `DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 2 test_task_dispatch`.
- **Six storage-interface/lock tests and six holdout crash-safety tests passed**, with no skips. These were run
  with the repository runner beside the earlier dispatcher group. Their code was unchanged afterward, so those
  passing groups were not repeated for the final source-payload privacy fix.
- **16 selected compatibility tests passed**, with no skips: file/PostgreSQL project documents, memory merge,
  stage records, deletion, numeric JSON and draft-lock compatibility; two existing solution-team cases; three
  stage lineage/replay/failure cases; and two harness ownership checks. No full storage or PostgreSQL suite ran.
- Total application/compatibility coverage: **74 distinct relevant tests passed**. The final dispatcher runner
  archive and the prior lock/holdout archive are kept in the repository's ignored training-record test-run store.
- Four development-record/corpus checks passed after correcting a numeric-grounding wording issue in the record.
  `make sft-dev` rebuilt 6,746 examples from 644 packages (5,328 train and 1,418 validation), with no refusals.
  `git diff --check` passed. No application tests were repeated solely for these documentation records.

Only affected test groups run under the owner's standing instruction. No full repository, full PostgreSQL, broad
product flow, live-model or sealed evaluation is run. The inherited TP-03 global prompt snapshot staleness is not
changed or misrepresented by this package. Portable core contracts and schemas remain unchanged.

## Limits and next package

This is an explicit developer API for a fresh basic binary project. It does not silently adopt prior legacy
execution, flatten programmes, add a product setting or promote the planner. Code/source/capability changes refuse
the old checkpoint rather than rewriting its graph. Missing or ambiguous evidence needs the existing owner.
TP-05 will add independent acceptance checks, failure classification and safe selective invalidation.
