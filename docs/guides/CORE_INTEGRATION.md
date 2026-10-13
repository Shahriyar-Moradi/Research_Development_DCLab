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
