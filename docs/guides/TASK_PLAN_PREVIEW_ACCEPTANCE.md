# Task Plan developer preview — TP-07

The project Workflow page now includes a read-only Task Plan panel. Home's built-project context also links to it.
This completes TP-07's permitted developer preview. **Planner execution rollout remains blocked** by the failed
[TP-06 comparison](TASK_PLANNING_EVALUATION_ACCEPTANCE.md). Existing stage actions keep their existing guards.

## Enable and roll back

The workspace owner opens **Admin → Teams**, selects **Read-only Task Plan developer preview · no execution**,
and presses **Save**. The existing owner-only, CSRF-protected settings endpoint records the change in the workspace
audit. The setting is `task_plan_preview` in the existing team settings; its default is false.

Open a real project's **Workflow → Overview → Task Plan**, or use the **Task Plan** link in Home's built-project
context. Refresh reads current records. It cannot start, retry, reconcile, verify or approve a task. Rollback is
the same checkbox, cleared and saved. No migration, project rewrite or execution switch is involved.
The six Home intent buttons and their business-intent meanings remain unchanged.

## What the panel means

The tree separates logical groups, guarded tool work and approval/check nodes. Details show dependencies, source
workflow steps, allowlisted capabilities, plan revisions, approval state, current verification and evidence links.
It also shows recorded execution budget usage and revision summaries with preserved work. Budget figures describe
execution, not the separate cost of model planning requests.

Completed means a matching saved independent verification and current owner evidence exist. Missing proofs block
dependent work; changed source, code or evidence makes affected progress stale. Ambiguous attempts remain blocked
until the existing owner path reconciles them. Viewing an active attempt does not resume it. Old projects without
a task journal receive a metadata-only deterministic preview, without invented completion claims. Unsupported,
unconfirmed and no-data inputs explain that a supported plan needs data or confirmation.

The read-only `GET /api/projects/{project_id}/task-plan` reuses project access checks and the existing task saver.
SQLite opens only an existing database in read-only/query-only mode; PostgreSQL uses a read-only connection.
Neither path runs saver setup or creates storage. The presenter does not call proof-writing verification or
effect reconciliation. Response fields are allowlisted; raw plans, private rows, hidden reasoning and exception
details are excluded. Public core contracts, schemas and main-product field mappings are unchanged.

## Actual validation

Only relevant groups ran under the owner's standing test scope. No broad repository gate or sealed evaluation is claimed.

| Check | Result |
|---|---|
| Task Plan presenter/API, real file and PostgreSQL stores | 33 passed, no skips |
| Existing team settings and web build/client groups | 24 passed, no skips |
| Real browser: Developer and Data science, each 1280 and 375 px | All four final checks passed |
| Frontend generation | `make web` passed; generated API client and static assets rebuilt |
| Development corpus | `make sft-dev` and all four validation/freshness checks passed; zero refused examples |
| Independent review | PASS after proof-propagation and adjacent-revision fixes; final focused regression passed |

Browser checks cover default-off, no-data, recorded progress, visible completion/blocked labels, keyboard details,
evidence, read-only refresh, a visible error/retry, rollback and narrow layout. The isolated fixture dispatches only
the existing guarded data stage. API tests cover viewer read access, owner-only switching, unauthenticated refusal,
workspace isolation, audit scope and absence of a POST execution endpoint. Recovery tests cover resumed records,
missing upstream proof, current owner approval, running/unknown effects and revision preservation.

Failing-before records are retained: the absent initial presenter, lost-proof dependency propagation, replacement
plan preservation, adjacent graph revision/preserved-attempt history and a UI helper that rendered status as evidence controls. Each implementation defect has a
passing regression. The initial browser run lacked pinned Chromium; installation and the successful rerun are
recorded. The browser keeper declined copying the whole temporary workspace because free disk was below its
existing 2 GB threshold. Screenshots and explicit invented project/checkpoint/owner JSON fixtures were saved
separately before teardown. These are real browser captures, not generated mockups.

## Bounded live planning diagnostic

The separately authorized test budget was **$2**. Four fresh invented binary scenarios used 27 actual model
requests: four accepted decompositions and 23 accepted task-plan proposals. Three guarded workflows completed,
each reading its holdout once. The changed-contract scenario correctly stopped after data, with no holdout read.
No unauthorized, repeated or leaking handler was observed; planning offered no tool schemas. All 27 complete
provider exchanges and returned reasoning records were retained in the opted-out evaluation folder.

At provider catalog prices verified before the run, token-priced cost was **$0.025211** and permanently reserved
worst-case exposure was **$0.08684772**. These are USD token accounting, not an invoice. A run-local verified price
file supplied numeric USD prices to the gateway's EUR-named fields; an independent USD reservation counter enforced
the cap and never refunded failures. Application-wide prices were not changed. No additional live calls are needed.

All **315 tests in ten planning groups** passed, including actual file/PostgreSQL cases, with zero skips.
Their original long macOS socket-path startup failure is retained beside the successful short-temporary-root run.
The live task plans ran on the deterministic graph; proposed model hierarchies were correctly refused without
effects. Thus the full C execution arm remains unmeasured. This small diagnostic does not replace the frozen TP-06
cases, improve its recorded 45/48 correctness, pass its 95% gate or establish general benefit.

## Records and limits

The complete local diagnostic is in `agent_runs/live_tests/planning_20261112_budget2/`, including the frozen
protocol, verified prices, script, individual results, cost/reservation events, full exchanges and `REPORT.md`.
TP-07 command logs, exit statuses, screenshots and JSON fixtures are in `agent_runs/live_tests/tp07_20261112/`.
Automatic test-run logs are under `training_records/test_runs/2026-11-12/`.
The permanent collection uses the existing append-only archive, with per-file SHA-256 manifests and entries in
[`training_records/INDEX.json`](../../training_records/INDEX.json). Raw captures stay local and out of training;
the development trajectory contains aggregate engineering findings only. The initial snapshot is
`training_records/snapshots/20261112T115806Z`; the final collection is `training_records/snapshots/20261112T162819Z`, including TP-07 evidence.

Remaining limits are explicit: execution is not enabled, general model-hierarchy dispatch is unavailable, the
promotion gate failed, the independent live sample is small, and complete browser workspace copies were skipped
by the resource threshold. The default-off preview supports the current binary adapter and existing host records.
No claim of production readiness follows from these checks.
