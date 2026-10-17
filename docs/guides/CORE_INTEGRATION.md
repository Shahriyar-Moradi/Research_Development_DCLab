# Adopting dclab_core in DCLab's main product

Package 17.8 of `docs/guides/PRODUCT_BUILD_PLAYBOOK.md`. This guide says where each `dclab_core` call goes in DCLab's
main product (github.com/Shahriyar-Moradi/DCLab), what the main product implements around it, what never crosses,
and the tests it adds on its side.

**Wiring it in is done in the main repository, as its own change, by its own process** (its ADRs, reviewers and gates).
Nothing here changes the main product, and nothing of the main product is copied into this repository: its contracts
were read with `gh` (read only) at commit `7fa23b977685634c9e103771a9a502c96830afde` (2026-10-09) and only their field
lists were recorded, in `dclab_core/adapters/main_product_fields.json`. The function and file names below are the ones
read at that commit; read them again before wiring.

## What the main product installs

- **The wheel.** `dclab-core` 0.6.0, pinned:
  `dclab-core @ git+https://github.com/Shahriyar-Moradi/Research_Development_DCLab.git@<tag or commit>#subdirectory=dclab_core`.
  It depends on numpy, pandas, scikit-learn and pydantic (2.10 or later) only, runs on Python 3.10 to 3.13, never
  imports `dclab_rnd`, never calls a model provider, never touches a database or the network. `make core-wheel-check`
  here builds it, installs it in a clean Python 3.12 with only those dependencies and runs a replica end to end
  (2026-10-13, 0.6.0: passed, numpy 2.5.4, pandas 3.0.6, scikit-learn 1.9.1, pydantic 2.14.1; the optional pdf extra is not a direct dependency). What changed between versions
  is in `dclab_core/CHANGELOG.md`.
- **The contracts.** `ProblemBrief`, `DataFindings`, `SolutionPlan` (of `Decision`s), `Review`, `SolutionReport`:
  frozen, versioned (`dclab_core.problem_brief.v1`, ...), read only with their version (`Versioned.read`), with JSON
  Schemas in `dclab_core/schemas/`. The main product stores them as JSON beside its own records if it wants the
  reasons and citations; it does not have to.
- **The adapter.** `dclab_core.adapters.main_product` turns each output into a plain dict shaped like the main
  product's contract, field by field: `{"contract", "main_product": {repository, commit, path, version}, "status",
  "decision_point", "fields", "notes", "excluded", "about"}`. `fields` has exactly the recorded field names; a field
  the core cannot fill is `None` with a note (never a guess); values use the main product's names (`pr_auc`, not
  `average_precision`; `group_disjoint`; `StratifiedGroupKFold`; the open-ingest family portfolio).

| Function | Main product contract (file at the commit read) | Status |
|---|---|---|
| `problem_spec(brief, plan=None)` | ProblemSpec content: `_content_payload` in `apps/api/app/services/problem_spec_service.py`, `ProblemSpecCreateRequest` in `apps/api/app/domain/workspace_identity.py` | proposal (`spec.objective`) |
| `problem_profile(findings)` | `ProblemProfile` in `apps/api/app/engine/modeling/problem_profile.py` | evidence beside the host's own profile |
| `metric_plan(plan)` | `MetricPlan` in `apps/api/app/engine/modeling/metric_planner.py` | proposal (`spec.objective`) |
| `holdout_plan(plan)` | `HoldoutPlan` in `apps/api/app/engine/modeling/holdout_planner.py` | proposal (`split.strategy`) |
| `validation_plan(plan)` | `ValidationPlan` in `apps/api/app/engine/modeling/validation_planner.py` | proposal (`split.strategy`) |
| `objective(plan, brief=None)` | `Objective` in `apps/api/app/engine/modeling/objective.py` | proposal (`spec.objective`) |
| `experiment_plan(plan, brief=None, findings=None)` | `ExperimentPlan` (and `PlanSplit`) in `apps/api/app/domain/run_plans.py`, one field group per `PLAN_FIELDS` key | proposal (an `ExperimentPlanProposal` payload) |
| `decision_points(plan)` | each plan decision with its point in `REGISTRY` (`apps/api/app/agents/governance/decision_points.py`), its cap, and the translated value, or why no point takes it | proposals |
| `decision_threshold(assess_result)` | the dict of `select_decision_threshold` in `apps/api/app/engine/modeling/objective.py` (binary only) | evidence / proposal |
| `experiment_review(review)` | `ExperimentReview` in `apps/api/app/agents/classes/experiment_critic.py` | advice (`experiment.review`) |
| `experiment_changes(review)` | `ExperimentChangeSet` in `apps/api/app/domain/experiment_changes.py` | proposal (`improve.next_action`) |
| `proposals(brief, findings, plan)` | all of the above that the inputs allow | |

Every mapped value is a **proposal**: the host adopts it through its own governed decision point, where its validator
re-checks it against its own data and a person accepts it at the point's level (`split.strategy`, `spec.objective`,
`target.column` and every exclusion are capped at L1). The adapter never applies anything.

## Where each call goes

The auto-train job runs its stages in this order (`run_auto_train_job` in `apps/api/app/services/auto_train_service.py`):
`run_load_profile` → `run_target_resolution` → `run_structural_cleaning` → `run_holdout_lock` →
`run_train_only_decisions` → `run_split_plan` → `run_column_roles` → `run_preprocessing_setup` → `run_training` →
`run_persistence` → `run_finalize`. The AI-before answers travel into a run in its `plan` (an accepted
`ExperimentPlanProposal`, `apps/api/app/services/run_plan_service.py`); the worker never waits on a model.

### 1. Understand: the problem-spec service and the wizard

- **Call:** `dclab_core.understand.understand(description, {file name: column names}, advisor=...)` →
  `ProblemBrief`; then `adapters.main_product.problem_spec(brief)`.
- **Where:** a new backend read beside `POST /v1/projects/{project_id}/problem-specs` (`create_problem_spec_v1` in
  `apps/api/app/api/v1.py`, which calls `create_problem_spec` in `services/problem_spec_service.py`) that takes the
  person's description and the dataset's column names and returns the mapped fields, their notes and the brief's
  open questions. The wizard (`apps/web/lib/application/studio-wizard.ts`: steps data, target, objective, train;
  `buildSpecBody`; `NewRunWizard.tsx`) pre-fills its target and objective steps from it, shows each note beside its
  field and asks the open questions; the person submits through the existing POST, which validates the objective
  (`parse_objective`) as it does today. The lead agent's `lead.propose_problem_spec` tool can call the same read.
- **Rules kept:** the spec is the person's intent: a field the description does not state is `None` with a note, an
  inferred one says so, and a person locks the spec (`status` is not content and is never set by the adapter).
  Agent-written specs still pass `_check_agent_spec_text`. The brief's constraints travel in `constraints` under a
  free-form key `brief_constraints` (`parse_objective` keeps unknown keys); stated costs of both errors on a binary
  task become `constraints.cost_matrix`; a review capacity has no structured key and stays text for a person.

### 2. Analyse: structure before the holdout lock, the target after it

- **Before `run_holdout_lock`:** only structure may inform the split (`split.strategy` takes evidence of partition
  `metadata`). Call `dclab_core.analyse.analyse({name: frame}, target=None, brief=None)` in `run_load_profile`
  (`apps/api/app/services/auto_train/load_profile.py`, beside `profile_frame` and `quality_report`): duplicates, keys,
  constant and impossible values, repeated entities and time columns, with no target statistic. **Pass no brief
  here**: given a brief that names a target column, `analyse` reads that target, and its statistics would include the
  rows the holdout is about to take (`tests/test_core_adapters.py` checks the call above reads no target). The
  entities the brief names (`brief.entities`) are read from the brief directly. Together, this is all a split
  proposal may rest on before the lock.
- **After `run_holdout_lock`:** in `run_train_only_decisions` (`apps/api/app/services/auto_train/decisions.py`),
  beside `plan_model_development(locked_train, ...)`, call
  `analyse({name: locked_train}, target=target.column, task=target.task_type, brief=brief,
  train_index=locked_train.index)` on **the locked training partition only** (the holdout rows are not passed at all;
  `train_index` tells the core these rows are a training partition, without which it warns that target statistics
  use every row and the adapter refuses to carry them: `problem_profile` then leaves the class balance and the
  target's distribution `None`, listed under `excluded`). `problem_profile(findings)` is evidence beside the
  `ProblemProfile` the host builds there, never a replacement: record it as an event or trace. The leakage findings
  (`features.forbidden`, `features.to_confirm`) go to the `feature.leakage_suspect` cross-check beside
  `resolve_leakage_point` (cap L1: a flag for a person; keeping a column out is an exclusion, which a person decides).
- **Known limit:** dclab_core groups folds by an entity when that entity tells the target out of fold, which needs
  the target. Before the lock the core can only see that rows repeat (its entity findings are then `info`), so its
  split proposal is weaker there than in the studio. A target-reading finding after the lock that would change the
  split never re-splits the locked holdout: it becomes a finding for a person and a proposal for the next run.

### 3. Plan: the experiment planner and the investigator, as proposals through governed points

- **Call:** `dclab_core.plan.plan(brief, findings)` → `SolutionPlan`; `validate(plan, findings)` must be empty; then
  `experiment_plan(plan, brief, findings)` and `decision_points(plan)`.
- **Where:** the experiment planner agent (`apps/api/app/agents/classes/experiment_planner.py`, points
  `spec.objective`, `split.strategy`, `training.families_budget`) outputs an `ExperimentPlanProposal` whose payload is
  exactly an `ExperimentPlan`. `experiment_plan(...)["fields"]` is such a payload, from rules: it can be the planner's
  deterministic proposal (or the context the agent is given), validated by `run_plan_service` (`plan_for_request`,
  `_check`) at request time and by `plan_points` (`resolve_target_point`, `resolve_objective_point`,
  `resolve_split_point`, `resolve_families_point`, `resolve_missing_point`) at job claim. Target, metric and split are
  L1 answers: only a plan a person accepted may carry them; a split tighter than the rule's stops the run at
  `needs_input` before `run_holdout_lock`. The dataset investigator agent
  (`apps/api/app/agents/classes/dataset_investigator.py`, `target.column`, `column.missing_value_action`) sees
  column metadata only; dclab_core's `data_investigator` role reads the same partition (names, kinds, missing and
  distinct shares, never values) and its proposals are tested by code before they count.
- **What has no point (said in each decision's note):** partial labels (`labels.coverage`: the main product treats an
  unlabelled row as a negative), a label file's join key (`labels.join`: the host builds the label column before the
  upload), secondary metrics, baselines (always evaluated there), feature directions (no point builds
  entity-history features), cleaning steps (the host's structural cleaning), a review budget for the threshold.
  Families the main product does not have (positive-unlabelled bagging, anomaly detectors, rules, a quantile model)
  are named in the note, not proposed: adding a family is the main product's own change. The families are mapped
  onto the *recorded* open-ingest portfolio (`gradient_boosting` becomes xgboost, lightgbm and catboost; `ridge`
  becomes linear_regression): the host intersects the list with its own `open_ingest_families(task)` before
  submitting, because `resolve_families_point` refuses the whole answer when one family is outside it.

### 4. Review and improve: the experiment review and the improve decision points

- **After a run**, on the locked winner's **out-of-fold training predictions**:
  `dclab_core.assess.assess(task, y_oof, p_oof, y_test, p_test, budget=..., costs=..., partial_labels=...)` (the
  three from `assess.threshold_inputs(plan_dict, rows, days)`, whose `note` says when a stated policy could not be
  used; `y_test`/`p_test` are the host's one final evaluation, read only after the threshold is chosen) and `dclab_core.review.review(task, y_oof, p_oof, metric=..., frame=train_frame, cap=..., entities=...)`.
- `decision_threshold(result)` shapes the binary threshold like `select_decision_threshold`'s record: evidence beside
  the host's own threshold, or, for a review budget, a proposal a person records as a chosen operating point
  (`operating_point_service.choose_operating_point`). Its holdout figures never cross (`excluded`).
- `experiment_review(review)` is advice for the experiment critic (`experiment.review`, AI-after, cap L1,
  `ExperimentReviewProposal`): weaknesses as findings with their statements; `verdict`, `cv_metrics` and `confidence`
  are left to the critic and a person (the critic's validator accepts only checks recorded on the reviewed run, so the
  host records the weaknesses as checks first).
- `experiment_changes(review)` is an `ExperimentChangeSet` for `improve.next_action` (cap L1). Only experiments with a
  change kind there cross (class weights); a median loss, a log target, clipping to the training range, entity
  history and merging tiny classes are named with why not. At the commit read, the improve loop (ADR 0007,
  `app/services/improve/`) was proposed and not built; until it is, a person starts a branch with the change set. The
  core's `review.adopt` (a change is kept only if its paired gain over the same folds beats the margin and the
  one-sided 90% t bound) should be compared with ADR 0007's comparison rule before either is used for the other.

### 5. The technical report

`build_technical_run_report` (`apps/api/app/services/technical_run_report.py`), called by `run_finalize`
(`apps/api/app/services/auto_train/finalize.py`) and stored with the run, is where the core's record goes: the brief,
the findings with their implications, the plan's decisions with reasons and the alternatives set aside, the
assessment's parts that have no contract (calibration, per-class figures, intervals and their coverage: listed by name
in `decision_threshold(...)["excluded"]["report_only"]`), and the limits (partial labels make every precision a lower
bound). `engine/reporting/report.py` (`render_markdown`) can render a section from them. dclab_core's own
`SolutionReport` (17.7) is not built yet; when it is, it gets its mapping here in the same package.

## The Advisor to implement over the main product's gateway

dclab_core asks a model only through an `Advisor` the host passes (`dclab_core/advisor.py`):
`ask(purpose: str, evidence: Mapping) -> Mapping | None`. `None` means the deterministic path, which always works; a
failure or an invalid answer is treated as no answer and reported to `on_error`. The roles are
`dclab_core.agents.roles()`: `problem_analyst` and `brief_checker` (understand), `data_investigator` (analyse).

The main product implements it once over `GatewayService.complete(db, CompletionRequest(...))`
(`apps/api/app/agents/gateway/service.py`, `contract.py`):

- **Prompt:** `dclab_core.prompts.text(purpose)` is the system prompt, the evidence (already checked to be a compact
  summary) the input. The prompts ship in the wheel and are pinned by `dclab_core/prompts/HASHES.json` (the first 16
  hex characters of each file's SHA-256). The main product's prompt releases (`apps/api/app/agents/prompt_releases.py`)
  expect `app/agents/prompts/<agent_key>/v<N>.md` and `v<N>.schema.json` with the full SHA-256 as `prompt_digest`: the
  release for a dclab_core role should be made from the installed wheel's file (and its digest checked against
  `HASHES.json`), not from an edited copy, so both products send the words that were measured (18.2, 18.3). The output
  schema of each role is the host's to write for now (the core validates answers with its parse functions); a
  published schema per role is an open item for the core.
- **Request:** a `CompletionRequest` with its required fields: `agent_role`, the role as `agent_key`, the purpose,
  `prompt_release_id` (the release made from the wheel's prompt file), `output_schema` (the host's pydantic model of
  the role's answer), `max_output_tokens`, a `budget` reservation (`GatewayService.reserve`) and the project for
  attribution; plus the decision point it serves (`target.column` or `spec.objective` for the analyst,
  `feature.leakage_suspect` for the investigator), `max_data_class="metadata"` and `outcome_scope="none"`. Commit
  before calling and hold no row locks (the gateway's caller contract).
- **Answer:** `response.output.model_dump()` when `response.ok`; `None` on any `Refusal` (the gateway has already
  written its ledger row). Never raise into the core.
- **Off:** with AI off (the gateway's switches), pass no advisor: outputs are identical to the rules'.

## What never crosses

- **Rows.** No DataFrame, array, long list or table of records reaches an advisor or a finding (`dclab_core.evidence`,
  checked before anything leaves). The adapter's outputs are aggregates, names and decisions.
- **The holdout.** Only the locked training partition enters `analyse` and `review`; `assess` chooses on out-of-fold
  training predictions and reads the holdout once, after the choice, as the host's one final evaluation; the
  adapter never carries a holdout figure into a field.
- **Secrets and credentials.** The core holds none and asks for none; the gateway owns keys and redaction.
- **The main product's code.** It is not copied here and the core never imports it; only its field lists, with the
  commit read.
- **Confidential data.** A replica is invented data; nothing of a person's confidential file is ever a replica or a
  test fixture.

## The tests the main product adds on its side

1. **Holdout isolation.** A spy on every dclab_core call in the auto-train stages asserts that no row index of the
   locked holdout reaches `analyse`, `review` or the choice side of `assess`; permuting the holdout labels changes no
   core output before the final evaluation; the target-free `analyse` before the lock is called with `target=None`.
2. **The mapping against the real models.** `ProblemSpecCreateRequest.model_validate` on `problem_spec(...)["fields"]`
   (with a status; `constraints` and `success_criteria` stay under `PROBLEM_SPEC_JSON_MAX_BYTES`, which the adapter
  checks), `ExperimentPlan.model_validate` on `experiment_plan(...)["fields"]` without the `None` fields,
   `HoldoutPlan.from_dict` and `ValidationPlan.from_dict` on theirs, `parse_objective` on the objective,
   `ExperimentChangeSet.model_validate` on the change set, and a drift test that compares the dataclass and model
   fields with `dclab_core.adapters.main_product.recorded()`: when either side changes a contract, the test names the
   field (then this repository records the new commit and moves the adapter).
3. **Governance.** Each proposal goes through `resolve_plan_point` and its ceilings like the agents' answers; an
   `applied` (L2) plan carrying an L1 answer is refused; an AI-off golden run with no advisor equals the rules.
4. **The benchmark harness.** `benchmarks/harness` (`run.py`, `compare.py`, `tasks.py`; `apps/api/tests/test_benchmark_harness.py`)
   gets an arm where the core's proposals are accepted, compared with the rule-only arm on the same pinned tasks by
   `compare.py`'s tolerance, plus the core's replicas (`dclab_core.replicas`: `delivery_fraud`, `ticket_routing`,
   `delivery_time`) with their acceptance decisions, run against the pinned wheel.
5. **The pin.** The lock file pins the wheel by commit or tag; upgrading it reads `CHANGELOG.md` and reruns 1 to 4.

## Experience: the record of a session (19.1)

The main product adds no table. `dclab_core.adapters.main_product_experience.events(agent_events=..., runs=...,
invocations=..., decisions=..., run_events=...)` reads its rows (plain dicts with the recorded column names) as
`ExperienceEvent`s; `dclab_core.experience.timeline`, `session_card` and `redact` then run on them as they do here.
Where it goes there: a read-only service next to the harness recorder that loads one thread's or one project's rows
and returns the timeline. What its side of 19.2 adds: the request id stored on `agent_events` and
`execution_requests` (additive columns), and the auth session id as the visit. Whether any of it is ever exported for
training is the owner's decision (its ADR 0008); `redact` keeps only what a use allows.

## Consent at write time (23.1)

Every `ExperienceEvent` (v8 since 23.11; v3 added the stamp) and `DecisionRecord` (v2) carries `consent`: the level the record was written under,
`operate` (kept to run the product, for support and audit; never leaves), `improve` or `train`
(`dclab_core.consent.level_of` on the workspace's and the project's switches). An export keeps a record only when its
stamp allows the use (`dclab_core.consent.exportable`) and the switches allow it now, so a later opt-in never
re-labels what was written before it; `consent.changed` events keep the history of the switches with the notice
version shown. Where it goes there: the adapter's `events(..., consent=)` takes the level the host reads from its
workspace settings (`CONSENT_MISSING` says it has none on its rows today); its side adds a `consent_level` column to
the tables it exports from (additive) and stamps it from the same rule when a row is written. A row mapped without a
level is `operate`.

## Sign-in and session events (23.3)

The kinds `auth.sign_in`, `auth.sign_in_failed`, `auth.sign_out`, `auth.session_expired`, `auth.token_created`,
`auth.token_revoked`, `auth.workspace_switched` and `auth.role_changed` are the contract, with
`dclab_core.experience.hash_user_agent` (sixteen hex characters) and `truncate_ip` (the address without its last
part); a token's name is the person's words (`person_text`), everything else meta; never a token, a password or a raw
address. Where it goes there: the sign-in service writes `auth.sign_in` and `auth.sign_in_failed` (a reason code, the
method), the session store writes sign-out and expiry once, the token service its create and revoke, the membership
service the role change with who made it; the rows live in its `auth_events` table (the document's package 2) and the
adapter maps them once that table is in the field lists (`KINDS_MISSING` names each until then).

## The browser's events (23.2)

The kinds `ui.page_viewed`, `ui.action`, `ui.upload_started`, `ui.upload_finished` and `ui.error_shown` are the
contract; `dclab_core.experience.ui_payload(kind, payload)` keeps only each kind's fields and refuses any value that is
not a route, an element's id or data name, a code or a count, so a form's contents, a query string or a message cannot
enter. Where it goes there: the web app batches them (twenty at a time or every two seconds, and on page hide) to a
route of its own under the signed-in person's session, which runs `ui_payload` and writes them with the request's
trace id; a signed-out page sends nothing. The adapter does not read them (`KINDS_MISSING`): they are rows that
route writes. Whether business users' pages are captured is decision D4 of the plan (route and action level, never
contents, is the proposal).

## The conformance gate, and what stays the main product's own (23.12)

`dclab_core.adapters.conformance.problems()` is the gate every package of phase 1 and 2 passes: every contract of the library has
an adapter (`CONTRACT_ADAPTERS`: five read the main product's rows into the library's record, five map the library's contract out
to the main product's own, such as a `ProblemSpec` proposal; `FROM_ROWS` says which) or says why it has none (a library output the
host stores or renders; a value is an existing `module:function` or a reason that starts with "library output"); every event kind is produced by an adapter or listed in `main_product_experience.KINDS_MISSING` with a
reason; every decision point has a name in the main product (`Point.main_product`), is read as one of its decision record types
(`main_product_ledger.DECISION_TYPE_POINTS`) or has a reason in `POINTS_NOT_MAPPED`; every id
of the record is filled or listed in `IDS_MISSING`; and the tables the adapters read are still in the field lists read at the
pinned commit (`main_product_fields.json`, commit `7fa23b9`). The studio's `tests/test_conformance.py` runs it and fails when a
package adds a contract, a kind, a point or an id without a mapping or a reason; the main product's repository runs the same
function in its own tests, and again after it re-reads its commit (the field lists are then refreshed and the lists above shrink).

What each package leaves to that repository, from the owner's integration plan (A.9 of the capture plan): its `auth_events`
table and one audit row per governed action (23.3); `feedback` and `error_events` tables and their routes (23.4, 23.6); the web
app's sender of the UI events (23.2); a `consent_level` column on its event tables (23.1); `trace_id`, `span_id` and `episode_id`
columns on `agent_events`, `llm_invocations`, `ml_run_events`, `project_decision_records` and `data_access_events`, and the
`traceparent` header on its edge (23.11); a per-column sensitivity class on `dataset_columns` (23.7); its own schedule for the
dataset builders and the rebuild after a deletion (23.10); the content vault with per-workspace keys and OpenTelemetry. None of
these is a change to the library: each is read through an adapter once the column exists and the field lists name it.

## Trace ids for a collector (23.11)

`dclab_core.tracing` reads and writes the W3C `traceparent` (`00-<trace id>-<parent span id>-<flags>`: 32 and 16 lower-case hex
characters) and derives a trace id from a request id (`trace_id_for`: the id itself when it already is 32 hex characters, else
the first 32 of its SHA-256, so one request id is always one trace). `Ids` carries `trace`, `span` and `parent_span` beside
`request`. Where it goes there: the edge accepts a `traceparent` (or derives the trace from `X-Request-ID`), binds the trace
and a span for the request, answers with a `traceparent` naming its span, and writes the ids on every event and log line; a job
carries the trace and the request's span and runs in a span of its own; a model request, an agent step and a tool call are
spans inside whatever ran. Here that is `observe.span`, the middleware in `agentic/server.py`, `jobs/worker._acting`, the gateway
and `experience/hooks.from_trace`; the log lines carry `trace_id`, `span_id` and `parent_span_id`, so an OpenTelemetry
collector joins them without a change. The records are read by trace (`trace=` on the log, the finder, `/api/experience/search`,
`search --trace`; PostgreSQL has an indexed `trace_id` column, migration 0019). The main product's rows with a request id are
stamped by `main_product_experience.with_trace(events, request_of)`.

## The lake's shape (23.10)

`dclab_core.datasets.episodes(events, records, text=…)` and `model_calls(events)` build two sets from exported events
(`consent.export_events` here; `main_product_experience.events` with the host's own redaction there) and the ledger's decision
records. An **episode** is a turn: from a person's message to the next, with the project's events of the same window joined by
the project id; its numbers (model calls, tokens, steps, moves, errors) are counts and sums of the events listed in
`event_ids`, its decisions are the records in the window, and without `text=True` it has no free text at all (the improve level).
A **model call** is one `model.request` whose prompt or reply the host kept (never a purpose that was shown cell values), joined
to its check (`model.checked`, same model request id) and to the feedback on the chat message that request produced. Every
`DatasetManifest` (v4) names `scrubber` (`redact.VERSION`), the builder's `commit` and its `files`. `columnar.write_parquet`
writes the same rows as columns (`pip install "dclab-core[parquet]"`; nested values as JSON text). Where it goes there: its
nightly job calls the builders over its adapted events, and a deletion marks the sets stale so the next run rebuilds them;
here `datasets.sweep_due` does both (the worker calls it every minute: stale after a deletion, else once a day where a switch
is on; `DCLAB_DATASETS_NIGHTLY=0` turns it off).

## Steps inside a stage (23.8)

`dclab_core.steps` records what a run did, one `Step` at a time: `with collect() as log:` around a run and `with step(name,
kind, params=…, shape_in=…, fold=…) as s: … s.out(shape=…, metric=…, value=…, produced=array)` around each action (a cleaning
action, a transform, a fold, a fit, a prediction, a metric). Nothing is recorded, and nothing costs, when no `collect()` is
active, so library code calls `step()` freely. `digest_of` hashes at most the first 5000 rows and 200 columns of what a step
produced, rounded to six decimals, so a faithful rerun gives the same digest and a table's values are never kept.
`RunRecord` (v2) carries `steps`; `lineage.compare(original, rerun)` returns `same_steps` and the first step that
differs; `steps.compare` does the same for two lists. Where it goes there: its pipeline code wraps the same actions; its
`ml_run_events` fold and candidate rows already are steps, and `main_product_experience.steps_from_run_events(rows)` reads them
that way (no digest: it keeps none). A recipe is one `transform` step here (its imputation and encoding are inside it, not
separate steps), a metric step's digest is that of its value, and a digest reads a bounded sample (six decimals) plus a checksum of the whole
array, so a change past the sample still moves it. The events `pipeline.step_started` and `pipeline.step_finished` carry each step under
its run's id so a collector can follow a stage while it runs.

## The catalog and the standard questions (23.9)

`dclab_core.catalog.build(emitted_by)` is the contract-derived half of a data catalog: every event kind with the class of
each field, what never leaves, the level an export needs, the decision points and the contracts with their versions. A host
adds where each kind is emitted, its tables and its retention and renders a page; here `python -m dclab_rnd.data_catalog`
writes `docs/DATA_CATALOG.md` and `docs/data_catalog.json` and `--check` (in `make rd-gates`) fails when they drift, so a
new kind, point, audit kind, builder or table cannot be added without being described. `dclab_core.queries` holds the
nine questions support and the owner ask most (what happened in a request, a project's story, where checks refuse most,
failed model calls, overridden decisions, whether approved ones turned out well, spend per week, what is ready for
training, what a deletion removes) as functions over lists of events, records, labels and outcomes: feed them
`main_product_experience.events(...)` and the ledger adapter's records and the answers are the same. A request id opens as
a timeline: the host filters events by `ids.request` (indexed here as `request_id`).

## Column sensitivity (23.7)

`dclab_core.sensitivity.classify(name, values)` gives a column one of `restricted` (a secret, a card number found by the
Luhn check, a national id's or an IBAN's shape), `pii` (an e-mail, a phone number, a name, an address, a date of birth,
an IP address), `sensitive` (health, religion, ethnicity, orientation, politics, income) or `none`, from the tokens of
its name (a count, a flag or a rate about such data is not that data: `email_opens`, `has_phone`) and the shape of up to
200 of its values, with the rule in words and never a value. `strip(doc, class)` removes what carries a column's values
from any description of it: examples, category names, prefixes and shapes for all three classes, and the distribution
(quantiles, histogram, time range, shares) for `restricted` and `pii`. `ColumnCard` carries `sensitivity` and
`sensitivity_why` (the data card is v3; v1 and v2 read as v3), and `data.analysed` carries the column-to-class map.
Where it goes there: the main product is expected to keep a per-column class (`dataset_columns.sensitivity_class`), but no
`dataset_*` table is in the field lists read at the pinned commit, so the mapping is not written and its names are unverified
(`main_product_experience.FIELDS_MISSING`, and a test that fails when the lists gain the table). Until 23.12 reads it, a host
runs `classify` over a table's columns where it profiles one, keeps the rule that a column that is not `none` reaches a model by
name only, and brings an older stored profile or card up with `upgrade_column` / `upgrade_card` (what `DataCard.read` does for a
v1 or v2 card).

## Feedback of every kind, tied to a version (23.6)

`dclab_core.feedback.check(body)` turns a request into the `feedback.given` payload: a `kind` (`thumbs`, `rating` with a
score of 1 to 5, `correction`, `edited_output` with the reply as the person would have it, `override` of a decision, a
proposal or a step), the target as `<kind>:<id>` (answer, message, step, proposal, decision, note or report) and a
reason; `verdict(feedback)` reads the feedback on one message for a training example (an edit is the reply, a thumb
down ends the example, a thumb up marks it). The host resolves the `version` of what was judged: here the release
(`core-<version>/prompts-<digest>`) stamped on the `chat.agent` event that carries the message's id, so a thumb is
tied to the prompts and rules that wrote it. `datasets.conversations` uses the verdicts as the label. Where it goes
there: the web app's buttons post to a route of its own that runs `check` and stamps the version of the message it
finds; `adapters.main_product_experience.feedback_events(project_decision_records rows)` maps a person's accept, reject
and revert of a proposal onto the same kind, so both products' feedback reads alike.

## Outcomes linked to decisions (23.5)

`dclab_core.outcomes.Outcome` (v1) is how a decision turned out: the record it follows, the metric, its value and
interval, the reference it was expected to meet, how it was observed (`holdout`, `promotion`, `business`) and the
verdict `judge(value, reference, higher_is_better)` gives (`well` when the number holds its reference within a
tolerance, `badly` otherwise, `unknown` without a reference). `join(records, outcomes)` gives the current outcome per
decision and `by_point` the counts the Decisions tab shows. Where it goes there: `adapters.main_product_outcomes
.from_experiment(row, decisions, metric, reference=…)` maps an `experiments` row's `result` onto the decisions whose
`experiment_id` is that run; a `model_versions` promotion is mapped once the field lists carry that table
(`FIELDS_MISSING`). The rows belong in a table of their own beside `project_decision_records` (append-only, deleted with
the project), and the decision export carries them as `outcomes` beside `labels`, without a person's note; an outcome is never a label.

## Errors with fingerprints (23.4)

`dclab_core.errors.from_exception(exc)` gives a caught exception its `code` (the type), its `frame` (the innermost
frame inside our own code, `module:function`) and a twelve-character `fingerprint` of the two, so the same failure is
one group wherever its message differs and nothing of a message, a path or a value enters it; a host passes its own
package prefixes (`own=("app", ...)`), or the fingerprint rests on the type alone. `error.raised` carries
`fingerprint`, `frame` and `status`, `stage.failed` carries `fingerprint` and `frame` (all meta). Where it goes there:
the exception handler of the API and the job runner call `from_exception` where they catch, and write the event under
the request's and the job's ids; a rate-limit refusal is an `error.raised` with `status` 429 and the limit's kind as
`where`. The grouping (the last days by fingerprint: count, first and last, the frame, up to three request ids) is a
query of the host's own event table; `dclab_rnd.experience.finder.errors` is the reference.

## The problem card (20.1)

`dclab_core.card.problem_card(brief, words=..., data=...)` returns a `ProblemCard`: the brief, the family, the label
situation, the questions still open (blocking or not) and `ready` with `blocking` reasons. Where it goes there: after
`understand` in the problem-spec flow; the card's `brief` maps onto `ProblemSpec` as before
(`adapters.main_product.problem_spec(card.brief)`); `family` is a proposal at its `spec.objective` decision point and
`outcome` at `target.column`; a spec is locked (`problem_spec_locked`) only from a ready card, and `blocking` is what
its assistant asks next.

## Lineage: where every number comes from (19.5)

The main product adds no table here either. `dclab_core.adapters.main_product_lineage` reads its `experiments`,
`pipeline_stage_runs` and `agent_runs` rows as `RunRecord`s (`dclab_core.lineage`): what a run read (dataset and split
plan ids, config, seed, an agent's context, policy and tool-catalog digests), its parameters, its numbers by path, its
duration and its code (`git_commit`, the runtime, the prompt release). `events(..., experiments=..., stage_runs=...)`
adds each finished run to the session record as `run.recorded` (event contract v2; v1 events still read). Where it goes
there: the technical run report (`services/technical_run_report.py`) can cite, for each number, the run and
`RunRecord.fingerprint`; a rerun of an experiment is checked with `lineage.compare` (same inputs, same code, same
number). What it cannot fill from these tables is listed in `INPUTS_MISSING` (the dataset version's hash, the problem
spec's lock), never guessed.

## The decision ledger (19.6)

`dclab_core.ledger.DecisionRecord` is one decision at a point (`POINTS`: the main product's governed points and Jev's
purposes among them), in a state of bands with no table value, with every answer and the value used;
`DecisionLabel` says what was right (benchmark, blind, measured, user_decision_exposed). The main product adds no
table: `adapters.main_product_ledger.from_semantic_answer(row)` reads a `semantic_decision_answers` row as a record and
its label, and `to_semantic_answer(record, ...)` writes a record back as that row's content, so a decision model
trained in this studio (Part 11) answers there through the same columns, as another answerer beside Jev, shadow first
and promoted only through its own R3 gates. `from_decision_record` reads `project_decision_records` at their points.
What its side adds for 19.7: the state itself (the bands its caller already builds for Jev, kept beside the digest).

## A decision model of this studio in the main product's R3 (22.7)

`adapters.main_product_r3` gives a DCLab answerer (a fitted classifier, `answerers.ClassifierAnswerer`) the shape the
main product's R3 run calls: `R3Answerer(model)(release, cases)` returns one `({"value": ...}, confidence)` per case (a
yes or no as the probability of yes, a choice as its label with its confidence, None when it does not answer), reading
each case's state as the main product sends it for the purpose (`project`: its release's subject and context fields).
Where it goes there: beside `ai_off`, `ScriptedAnswerer` and `port_answerer` in `r3_evaluation_service.py`, as the
answerer of an R3 run with a `dclab:<name>@<version>` candidate; promotion stays its own R3 gates (blind labels, level
caps). What its side changes before any row is written: a classifier makes no model call, and
`semantic_decision_answers.llm_invocation_id` is NOT NULL, so its answer rows need an invocation row of their own or
that column relaxed (ADR 0008). `run` reads an R3-shaped run here (by partition and label source, the ai_off
ablation, the digests as its report's) and `answer_rows` / `label_rows` write its rows' content;
`dclab_rnd.consent.export_main_product` is the only way labelled decisions leave this studio for it (the `train`
consent, Jev purposes only, its label format; DCLab's `measured` labels stay here). `make decision-r3` runs the reading
on generated cases.


## The problem map (24.1)

`dclab_core.problem_map.ProblemMap` (v1) is a problem as parts, sub-problems and how they combine. Each part has its question, unit
and moment; each sub-problem is shaped like a problem card (a family from `bands.FAMILIES`, the label situation, an outcome, what it
needs of the data by name, what it reads of the others, `predict_rank` and `known_rank` for when); a composition is ordered steps
(sequence, fusion, gate, budget), each on sub-problems and the outputs of earlier steps, with the key that joins two units. A small
problem is a map with one part and one sub-problem. `decompose(words, parts=...)` draws a map by rules (an explicit brief with parts
becomes the map's parts; a fraud problem, a routing asked as a team and then an urgency have templates; anything else is one part) and
`validate(map)` judges any map, whoever drew it, and returns every issue in words: an outcome read before it is known is a leak, a
read across units says its key, no cycle, a composition built in order that leaves nothing solved out, a stale fingerprint, the size.
The constructor checks shape, `validate` meaning: a missing unit or moment is a question for the map reviewer (24.2), not an error.

Where it goes there: the main product states one problem per `ProblemSpec` and relates no two.
`adapters.main_product_problem_map.problem_specs(map)` gives one `ProblemSpec` proposal per sub-problem (the task type where it has
one, a reason where it has none, `multiclass` kept) with the sub-problem's place in the map in `constraints["dclab_problem_map"]` (a
free-form key, as `brief_constraints` is) and the composition once; a map that is not sound maps onto nothing. What its side adds:
somewhere to keep the map beside the specs (a document on the project) and, for the composition to run, a pipeline that takes one
project's predictions as another's input (24.5). Whether the main product's `parse_objective` keeps the key was not read at the pinned
commit (the field list names no limit on it): its side checks it when it adopts the package.

## The understanding agents (24.2)

Three roles read what the person means, each in the pattern every role of the library follows (evidence of names and short lines, never a
value; a typed output behind a validator; the rules as the fallback; a prompt file pinned by its hash). The host implements
`dclab_core.advisor.Advisor` over its own gateway, sends `prompts.text(purpose)` as the system prompt and the evidence as the user message, and
the same roles run with the same words. `decomposer` draws a `ProblemMap` from the person's words: every part quotes them or names a rule
template, a unit and a moment are kept only as their words, and a map that `validate` refuses is dropped whole for the rules'. `map_reviewer`
asks what the map leaves unsaid, as questions that rest on the words, the map or the data card; a model's question never blocks (the rules'
do). `data_need` says which table and columns each sub-problem needs; a name the data card does not hold is dropped, and
`mapping.coverage` finds what the tables lack, in code. The rules alone are `problem_map.decompose`, `mapping.review_rules` and
`mapping.fill_needs`. A person changes a map with `add_part`, `remove_part` and `merge_parts`, each refused with the validator's reasons.

Where it goes there: the coordinator's walk gains the state `map` (`coordinator.situation`: `draw_map`, then `confirm_map` when the problem
has several parts; a host that draws no map leaves `Facts.map_size` at 'single' and is not held by it). Its assistant shows the map as the big
picture of the problem with the reviewer's questions; an edit is a move through the validator and a decision at `map.edit` (the drawing is
`map.size`), neither of which has a governed point of that name there (`conformance.POINTS_NOT_MAPPED` says so). A model serves a role only
after a sealed measurement against the rules (`dclab_rnd.agent_eval.complex`: 14 dev and 14 sealed problems, the gate `map_gate_v1`); the
measurement runs through the same `Advisor`, so the main product can run its own. What its side adds: somewhere to keep the map and its
confirmation on the project or its draft (a document), and the build gate (a problem of several parts is built from a map a person confirmed).
`adapters.main_product_problem_map.problem_specs` gives the confirmed map as one `ProblemSpec` proposal per sub-problem.

## The solution router (24.3)

`dclab_core.solutions` is the catalogue of approaches and the rules that route a sub-problem of a map to them. Each of its 16 entries serves one or
more of the 12 families and says what it needs (labels, a binary or many-class or numeric outcome, a long-text column, two columns of place, a time
column, an entity column, several tables, a treatment, a duration, a budget step) and which research track measured it; the entries are the studio's
problem kinds by key, and `review_budgeted` (the scores of other parts fused and cut to a team's budget) is the one that is a composition, not a model.
`route(sub, data_card, map, detected=)` judges every need of every approach for the family as met, unmet or unknown (never guessed: no data card or
no named outcome is unknown), ranks the approaches with none refuted first, chooses the first, and lists the others with what blocks them. A single
problem keeps the kind the host's own detection found. The role `solution_router` may choose among the candidates (a choice outside the catalogue or
one the data card refutes is refused, and a single problem is never asked). `Route` v1 is the stored record.

Where it goes there: the host passes its detected kind for a single problem and a data card of its tables (`datacard.card_of`); the route is a proposal
at the point `solution.route` (the main product has no governed point of that name: `conformance.POINTS_NOT_MAPPED` says so), kept by
`adapters.main_product_problem_map.problem_specs(map, routes=)` in each sub-problem's free-form place. What its side adds: the approach choice screen
(Home shows each part's approach with the others and what blocks them) and, for its own runs, the mapping from an approach to its model families.

## The solution map (24.4)

`dclab_core.solution_map.SolutionMap` (v1) is the problem map answered. It has a block per sub-problem (the approach the router chose, the solution workflow
(20.6) of that leaf and the split it uses), the composition's steps (a sequence, a fusion of scores on a key, a gate, a budget) and a last step that scores the
composed result, and one holdout design for the whole. The rules it holds: every leaf splits on the same keys and the same moment; in a map of several parts no
leaf scores the holdout on its own (each hands its predictions to the composition); the composition's last step reads the holdout once and nothing is chosen on it
(the fusion's weights, a gate's threshold and the budget's cut are chosen on validation before). `validate(map, problem_map)` refuses a leaf split by time beside one
split at random, a leaf that scores the holdout itself, a composition that reads the holdout early or twice, a workflow that does not hold and a step on what
nothing made. A part with no approach is a blocker the map names, not an unsound map. `split_for(data_card)` draws the one split from the card's entities and time.

Where it goes there: the one holdout is a `HoldoutPlan` proposal for every project of the programme
(`adapters.main_product_problem_map.holdout_plan`; its planner has no verified holdout for time and entity together and answers 'unsupported', so a person
decides). Each block's workflow maps as the plan's decisions do (`DECISIONS`), and the approach is a proposal in the sub-problem's place in the map. What its side
adds: the page that draws the solution beside the problem with each block linked to its part, and the run of the composition (24.5).

## The programme (24.5)

`dclab_core.programme.Programme` (v1) is a solution map run. It has a leaf for every block (what each chose on the validation rows, what it read, how many rows it used; names,
counts and scores, never a value of a row), the composition chosen on validation and, once, the holdout result: the composed score, every leaf's, and the composed score minus the
best leaf's, each with an interval from a bootstrap over clusters of rows (rows of one biker or one account move together). The best leaf it must beat is the one that did best
alone on *validation*, taken among every leaf that ran, never the one that did best on the holdout.

The rules it holds. No leaf scores the holdout: each predicts it for the composition. A leaf that reads another leaf's score reads it from the same protocol (fit on `fit`, judge
on `validation`; refit on `train`, predict the `holdout`), and a model learns a confirmed case only from the day it was confirmed. The split follows the design: a design that names
a moment is refused when the table cannot give days, never answered by a random split. The composition is chosen among every leaf alone and the ways to combine them, a tie to the
simpler one, so a combination that does not help is not chosen. `open_issues` keeps the holdout closed until every leaf that was to run has run, the composition rests on at least
five confirmed cases and nothing has read it before. A leaf that cannot run (no confirmed case, no table of accounts, an approach with no runner) is blocked with the reason and stays
a blocker until a person drops it with a reason; a drop is refused once any programme read the holdout, and `replan` carries that count by the split design, because the holdout is
the same rows whatever the routes. The host owns the rest of the rules: a second read of the holdout is a person's written decision and an agent's never; a replay or an export is
refused when anything the run read (the tables, the labels, the budget, the columns kept out) has changed; a table cannot be attached while the programme runs.

Where it goes there: each leaf that runs is an `ExperimentPlan` proposal with the programme's one split (`adapters.main_product_problem_map.programme_experiments`; a split by time
and entity together has no `PlanSplit` there and is left to a person). The main product has nothing that relates experiments, so the composition, the budget cut and the one read of
the holdout stay in the programme document its side keeps; its runs must not let a leaf score the holdout the programme shares. What its side adds: the store of programmes, the run
of the leaves as its own experiments, and the pipeline and code export of the composition. The report is `programme.report` (a `SolutionReport` whose numbers trace to the
programme document), and `programme.markdown` is the plain version.


## The fraud programme's replica and its acceptance (24.6)

`dclab_core.replicas.fraud_programme` is invented data shaped like the owner's complex example: rides, a table of accounts (phones and devices) and a partial, biased list of confirmed cases
with the date each was confirmed. Fake delivery marks are visible in the ride itself; fake orders come in rings of linked accounts and are visible only in the accounts table; some are both;
a family's shared phone, a short delivery next door and a cancelled ride are the honest look-alikes; a team can review 2% of the rides. Its `truth` (which rides were planted) is for tests only.
The acceptance (`tests/test_fraud_programme_acceptance.py`) is a case the main product's side can run against the library it adopts: on three seeds the first sentence becomes four parts, each is
routed to the right approach, the solution has one split and no leaf scores the holdout, no identifier or coordinate reaches a leaf, the composition beats the best leaf with its interval above
zero, and the holdout is read once. The written person that walks the same path through Home (`dclab_rnd/agent_eval/journeys/programme.py`) is this studio's.

## The capture of phase 2 (24.7)

A problem of several parts leaves, in the record a host keeps for the library, what was decided about it. Events (`experience_event` v9): `problem.mapped` when a map is drawn (its fingerprint,
size, the families of its sub-problems), `problem.part_changed` when a person adds, removes, merges, confirms or drops (and why, as person text), `solution.routed` for each sub-problem's
approach (with the candidates and the ones the data or the map refuted), `solution.composed` when the composition is chosen on validation and `programme.built` with the programme's numbers.
Decisions (`ledger`): `map.size`, `map.edit` and `solution.route` from phase 2's first packages, and `solution.compose`, whose measured label (`holdout_vs_best_leaf`) says whether the choice held
up on the holdout. Sets (`datasets`): `problem_maps` and `routes`, built only under the train consent, text redacted as every export, codes and counts for the routes.

Where it goes there: the main product has no map of parts, no routed approach as a row and no composition, so the five kinds are listed in `main_product_experience.KINDS_MISSING` with their
reasons and `solution.compose` in `conformance.POINTS_NOT_MAPPED`; the host that adopts the packages records the events where it draws the map, edits it, routes and runs the programme (the
payloads are in `experience.KINDS`), and builds the two sets with `datasets.problem_maps` and `datasets.routes` over items it assembles (the maps, the edits, the routes, the programme's
outcome). A route is recorded again when the tables change it; a programme's drop of a leaf is a `problem.part_changed`.

## Measurement and switch-on (24.8)

The roles of phase 2 (the decomposer, the map reviewer, the data-need agent and the solution router) are the library's; whether a model serves them is the host's decision, and the
library's rule for it is the one every role here follows: a model serves a role only after a measurement fixed before it ran, on people it was not tuned on, showed it right on at
least as many people as the rules and over the bar. The measurement is `dclab_rnd/agent_eval` in this studio: the decomposer against the rules on 28 written briefs (`complex`, the
sealed 14 run once), and the programme journeys (`journeys/programme.py`: six dev people, ten sealed people written by an agent that was given only the files' columns, the format and
the vocabulary), each person played from the first sentence to the one read of the holdout, or to the reason it stays closed, with two arms on the same people (the rules, and the
roles served by the model). Its parts and its bar are in code (`GATE`); a part that does not apply is not scored; a run needs a euro cap and a priced model, and an arm no model
answered is refused rather than scored against the rules.

Where it goes there: the main product can run the same people against the library it adopts (the sealed file and its hash are plain files; the journeys' `expect` is the vocabulary of
`solutions.CATALOGUE` and `bands.FAMILIES`), and keeps its own switch. In this studio the result of 2026-10-16 is that the decomposer passed its brief gate and the roles arm of the
journeys did not (2 of 10 against a bar of 85%), so `draft.map.roles_passed()` is false and the rules serve; the rules arm itself was also under the bar (2 of 10), and the misses are
listed as open work in the plan.


## The twin: the system and an AI-first agent on one step (25.1)

`dclab_core.twin` is the record of two approaches answering one step on one input: the system (rules, roles, validators) and an AI-first agent (a model solving the step its own
way). A `TwinStep` carries the input's fingerprint on the step and on both answers, so a difference is a difference of approach; `compare` gives each field's `Difference`, `judge`
the verdict against a truth (planted, a person's pick, measured, a label), and `effectiveness` a `TwinMatrix`: per step, each arm's accuracy with its interval, the agreement, the
cost and time, and the mode the evidence recommends by `twin_rule_v1` (software; software reviewed by the AI; the AI checked by the software; the AI; undecided below 10 judged
steps), fixed in the library before any run.

Where it goes there: the main product already has twin steps without the name. `semantic_decision_answers` keeps the rule's answer and Jev's on one question digest, and a label
row says which was right; `adapters.main_product_twin.twin_steps(rows, session)` reads them as `decision:<point>` steps (the rule the system arm, Jev the AI-first arm with its
model and release as the answerer), so `effectiveness` gives the main product the same matrix per governed point. What its rows do not hold (Jev's reasons, the cost, which is in
`llm_invocations`) is empty, never guessed (`NOT_MAPPED`). Running a second agent beside its own steps is that repository's change; the contract and the rule are the library's.

## Twin runs: the current system and Claude on one problem (25.2)

This package is the studio's: `dclab_rnd/twin` runs one problem in two lanes at once (the current system, and Claude as an AI-first agent with the same tools and validator),
each a workspace folder of its own, and writes the comparison as `dclab_core.twin` steps judged against the truth a blind seed plants. Nothing new enters the library. Where it goes
there: the main product can run the same comparison on its own projects, with its pipeline as the system lane and its agent loop as the AI-first lane, and hand each pair of answers
to `twin.step` with the fingerprint of what each lane was given; its own rule-and-Jev pairs already become twin steps through `adapters.main_product_twin` (25.1).

## The expert's record (25.3)

`dclab_core.expert.ExpertTrace` is how an AI-first agent, asked to act as a machine-learning expert, writes down its reading of a problem: the problem, its type, the data, the
exploration, the preparation, the problem's workflow, the algorithms, the evaluation and the solution's workflow, with the changes from version to version and the evidence behind
each. `read` takes a model's parsed answer and never refuses (what is missing is in `problems`); `workflow_problems` checks the workflow on the principles of
`solution_workflow`; `intents` puts its decisions on the ledger's points; `twin_steps` sets a version beside the system's answers. The prompt the studio uses is
`dclab_rnd/prompts/ai_first_expert.md`.

Where it goes there: the main product's agent can be asked for the same record before it acts on a project (through its own gateway, with the Advisor it implements), and keep it
beside its `ExperimentPlan`; `intents` gives the decisions at its governed points through the ledger's names (`POINTS[...].main_product`), so each sits beside the rule's and Jev's
on the same question. No row of the main product maps onto a trace (`CONTRACT_ADAPTERS` says so).

## Claude beside the system on Home, live (25.4)

The studio's: while its Home agent talks with a person, Claude, as the expert of 25.3, answers each Home step from the same conversation and data
in a thread of its own, behind the owner's switch and caps, and each pair of answers is a `dclab_core.twin` step in the twin folder. Where it goes
there: the main product's assistant can run the same arm beside its own wizard through its gateway, with the expert prompt and the twin step as the
record; the steps it writes are read by `twin.effectiveness` exactly as here.

## Side by side on Home, and the understanding panel (25.5)

The studio's page: Claude's lane beside the conversation, the differences per step with the person's pick (a twin step's `person_pick` truth), and
the understanding panel where the person corrects any field in their own words through the draft guard. Where it goes there: the pick is the
library's `Truth(source="person_pick")`; the main product's own page records it on its steps the same way, so its matrix and this studio's agree.

## A twin run from a project (25.6)

The studio's: a project's page starts a twin run of 25.2 and 25.3 as a job, both lanes on the project's table, and shows the comparison and the
expert's story; nothing of the project changes. Where it goes there: the main product can offer the same button on a project, with its pipeline as
one lane and its agent as the other, and keep each pair of answers as a twin step.

## The twin's training sets (25.7)

`dclab_core.datasets.twin_sets` builds, from twin steps and expert records, the sets of phase 3: every step (`twin.steps`), the preference pairs a
truth decided (`twin.preferences`), the expert's records with what became of them (`twin.expert`), and the teacher rows, one set per use
(`twin.teacher.workflow`: routing, scoring and choosing the next move inside the product; `twin.teacher.technical`: a column's role, a leak, the
split, the metric, the family), each with the rule's answer, the teacher's (Claude's) and the truth, and the state the rule decided on when a
ledger record was written before the step (`TEACHER_POINTS` maps a step's field to its point). Where it goes there: its own twin steps and its
`semantic_decision_answers` (through `adapters.main_product_twin`) feed the same builder; the teacher sets are the material for its decision
models of either kind, and a teacher's answer is never a label there either.

## Budgets: a person's month and a draft's Claude (25.10)

The studio's: a person's monthly model limit (15 by default, the owner's own per person), a draft's Claude budget raised within that room, each
change audited and in the session record. Where it goes there: the main product has its own per-person and per-workspace quotas; the rule worth
keeping is that a raise is bounded by the person's room this month and recorded with who, before and after.

## Trusted testers on their own machines (25.11)

The studio's: a tester's local workspace records everything, `make share` sends what changed (and the tables they agree to) to the owner's private
records repository, `make collect` rebuilds each tester's workspace on the owner's side, where the evaluation and the datasets read it. Where it goes
there: the main product serves its users from its own servers and keeps their records there; the rule worth keeping is the bundle with its manifest
(each file's SHA-256, checked before it is applied, never a path outside the person's folder) and the agreement per table.

## Live fixes: Claude's refusal, its thinking, the lane's state (25.12)

The studio's: the expert prompt asks for a solution design (not for a written record of how the model thinks, which
Claude refused), every thinking request asks for a summarized display, a refusal is said as declined, and the lane's
state is kept on the session. Where it goes there: any Claude call the main product makes should map
`stop_reason: "refusal"` to a visible state rather than an empty answer, and ask for `display: "summarized"` when it
keeps the reasoning; measured roles answer without thinking inside their call caps.

## Home that builds: "build it" to a running agent, the data reachable, a fuller EDA (25.13)

The studio's: "build it" (typed, the button, or the Home agent's `build_solution` tool on the person's word) accepts
the audit's proposal, checks the build's blockers, builds the project and hands it to the agent that plans, runs and
checks each stage; a blocker comes back with the step that clears it (confirm the map, bring data). Every table is a
CSV download and a page of rows for the person; the analysis shows every column; the data stage adds the outcome by
category or band of each input, on the training rows only. Where it goes there: the main product's chat should treat
"build" as an action through its own governed run path (never around its validator), and its EDA of outcome against
inputs should read only the rows its locked split leaves for training (`engine.outcome_by_feature` is the rule).

## The training lake (26.1, 26.2, 27.1, 27.3)

The library's: `dclab_core.lake`, the record shapes every DCLab model trains from (`DecisionSample`, `TrainingExample`, `GoldManifest`)
and the rules a gold set passes (`gold_problems`: one split per group, sealed rows only sealed, a teacher model's answer only in
`eval_only`, no holdout key). Where it goes there: the main product's `semantic_decision_answers` and agent records become
`DecisionSample` rows through `adapters/main_product_ledger` (its decision records) and the same builders, under its own consent
levels; its own public-table and synthetic sources may feed the same shapes. Training stays outside both products: the gold sets
and their manifests are what a training run reads.

## Decision models for the workflow (26.3)

The library's: `dclab_core.workflow_labels`, how a workflow decision is labelled from what a written case says is true (the family,
ready to build, the next question, the route of a sub-problem), and the reasons the other workflow points get no classifier; the
classifiers themselves are `decision_models.train` and `decision_gate.verdict`, as for the column points. Where it goes there: the main
product's governed workflow points (its `semantic_decision_answers` of a workflow kind, read through `adapters/main_product_ledger`)
can be labelled the same way from its own written acceptance cases; its R3 answerer (`adapters/main_product_r3`) serves a workflow
classifier like a column one, in shadow until it passes its own sealed gate. The studio's runner (`dclab_rnd/workflow_models.py`) and
its sealed journeys stay here.

## A company's own decision models (26.7)

The library's: `dclab_core.own_models.tune`, a decision point's own model for one company, stacked on DCLab's base classifier (its
answer read as features, never changed), chosen and calibrated by `decision_models.train` on folds grouped by the company's
projects, and judged by `decision_gate.verdict` on its own sealed projects. Where it goes there: the main product's tenant
exports its labelled `semantic_decision_answers` (through `adapters/main_product_ledger`) in the decision sets' shape, the tuned
model is the tenant's R3 answerer (`adapters/main_product_r3`) behind its own routing, in shadow first. The studio's runner
(`dclab_rnd/own_models.py`, `make own-models`) keeps every row in the workspace folder.

## Cleaning and reasoning for training data (27.5, 27.9)

The library's: `dclab_core.clean` (pure states, personal data, near duplicates, evaluation rows, quality, dataset cards) and
`dclab_core.reasoning` (a verified trace from an example's own inputs to its answer, rejected when it states a number or a column
the example does not hold, written as the `<think>` block). Where it goes there: any training set the main product builds from its
records passes the same cleaning before it leaves, and its reasoning traces are built by the same rules from its own records, never
from a model's text.

## Images in the engine (26.8)

The library's: `dclab_core.embeddings.pixel_columns` and `images_of` (an image table's rows as images, in pixel order),
`RandomFeatures` centred by `Centred` on the training images (a local baseline, not pretrained), `Pixels` for near copies, and
`calibrate(..., sources=)`. Where it goes there: the main product's image datasets (a table of pixel columns, or files it turns into
one) are split by their source (a scene, a camera, a patient) with `split`, encoded and scored with `choose` and `score`, and their
near copies across the split reported with `near_duplicates` on `Pixels` vectors; a pretrained encoder is its own injected
`Encoder` when weights are on its machines.

**Its decision points (26.8, dclab-core 0.65.0).** `dclab_core.image_decisions.rule_decisions(images, sources, labels, threshold=,
task=, totals=)` answers `image.near_copy`, `image.label_suspect` and `image.flip_safe` on the images it is given: the main product
gives its training images only, after its split, and writes each answer as a `DecisionRecord` of the point (subject `project`, the key
the rows' positions), with `totals` (counted before any cap) in what a person reads. A number label is asked about near copies only.
The three points have no name in the main product's `semantic_decision_answers` (`adapters.conformance.POINTS_NOT_MAPPED`): they are
read there as decision events until it governs them. `image_decisions.decide(replica, ...)` and the shapes replica's traps
(`shapes(..., reshot=, mislabel=, arrows=)`) build the benchmark sets (`make decisions-gen ARGS=--images`).


## Invented tables for an in-context tabular model (26.6b)

The library's: `dclab_core.priors.sample(seed, prior=None)` (one classification table from a random causal model, random trees or
a noisy linear model, the same table from the same seed) and `learnable(table)`. Where it goes there: nowhere in the main product's
request path; it is a data generator. A team that pretrains or evaluates an in-context tabular model calls it with its own seeds,
or reads DCLab's corpus (`make priors`, `lake/bronze/priors/priors_v1/`); `params.effects_shown` marks the tables whose shown
features include the outcome's effects, for a team that wants none.
