# Experience capture and the solving workflows

The owner, 2026-10-13: *first*, store everything a person does in DCLab to solve their problem (sessions, actions,
chat) and everything DCLab does in reply (agent steps, model calls, decisions, errors), organized so it can be found
and understood, and turned into training data (SFT) and evaluation cases. *Then*, make DCLab understand the problem,
understand the data, connect the two, analyse, build the solution workflow, the pipeline of workflows, and the code of
each step, for the common problems of machine learning. Both must be adoptable by DCLab's main product.

Why now (checked against the records, 2026-10-13): the owner uploaded the challenge's two halves into two separate
drafts: the orders (134,362 rows, no label) and the known frauds (100 rows, every one DELIVERED, a `reason` column with
"fake ride" and "same_biker_same_customer", `order_request_id` unique on every row: a label file, keyed to the
orders). The right solution, which DCLab solves on the replica (15.2–15.6), joins the frauds to the orders on
`order_request_id`, predicts a fake delivery with partial labels, splits by biker and judges by a review budget.
Instead, the target candidates were ranked "few distinct values first", `customer_id` (3 values) came first, the
answer "it is anomaly detection" named no column, and the build took the first candidate. The details of the owner's
own test that morning:  ("I want build anomaly detection", a 100-row delivery-events table) was
built into a project whose target was `customer_id`, task multiclass, and all five stages "completed": DCLab trained a
model to predict customer ids and objected to nothing. The labels question was asked but unanswered when it built.
Reconstructing that took six tables joined by hand, by time. Both halves of this plan answer that morning.

## Principles

1. **One contract, two hosts, no parallel system.** Every event, session, export and workflow is a versioned
   pydantic contract in `dclab_core` with a frozen JSON Schema (`make core-schemas`), pure Python, no storage, no
   network, no model call. The contract is the *exchange format*: each host keeps its own stores and provides two
   adapters, *rows → events* (to read a timeline) and *events → rows* (where it writes). Every builder, detector and
   export is a `dclab_core` function over events, so both hosts run the same code on their own data. The studio gains
   one log table because its fourteen stores share no ids; the main product gains none: its `agent_events`,
   `ml_run_events`, `llm_invocations` and `project_decision_records` already are that log.
2. **Capture is complete; use is governed.** Everything is recorded for the workspace that owns it. What leaves the
   workspace (a training set, a benchmark case) passes consent, redaction and a gate, every time, by code.
3. **Ids join everything.** Every event carries the same ids (workspace, user, visit session, conversation, project,
   run, request, model request) and a sequence number, so a session's timeline is a query, not an investigation.
4. **Classes of content, not trust.** Each field of an event is marked by what it holds: metadata, aggregates,
   person's text, agent's text, model prompt and reply, cell values, secrets. Rules act on the class: cell values and
   secrets never leave the workspace; free text leaves only redacted and consented.
5. **Measured, not assumed.** A failure found in a session becomes a test case; a capability is "done" when its
   acceptance runs on a replica of a real conversation, with the gate fixed before the run.

---

# Part 8 — Record everything, find anything, learn from it (packages 19.x)

Today (inventory, 2026-10-13): fourteen mechanisms record parts of a session: audit, transitions, activity, agent
traces, intern session documents, draft documents and events, project memory, lessons, model usage, output checks,
shadow logs, jobs, request logs, accounts. Gaps: no visit session id; no user id on transitions, activity, traces,
drafts, sessions or memory; the request id is never stored; usage rows cannot be joined to trace steps; model replies
are never kept; the project "ask" agent's answers are dropped; edits, abandonment and feedback are not events; there
is no outcome signal; consent exists for one export only; deletes leave orphans; timestamps are in seconds across
separate stores.

## Phase C1 — One record (build first)

### 19.1 The experience contract
- **Delivers:** `dclab_core/experience.py`: `ExperienceEvent` v1 (event id, `seq` per session, `at` in microseconds,
  the ids: workspace, user, visit session, conversation (draft or intern session), project, run, job, request, model
  request, parent event; actor person/agent/system/model; `kind` from a closed vocabulary in namespaces `visit.*`,
  `chat.*`, `draft.*`, `data.*`, `solution.*`, `stage.*`, `graph.*`, `agent.*`, `tool.*`, `model.*`, `decision.*`,
  `feedback.*`, `export.*`, `error.*`; a typed payload per kind; a content class per payload field), `Session` v1 (the
  summary of a visit: who, when, problem in a sentence, kind, outcome, counts, spend), and the redaction rules by
  class. JSON Schemas frozen; CHANGELOG entry.
- **Done when:** the schema round-trips every kind; a payload field without a content class is refused; an event
  carrying a cell value marked as metadata is refused by the validator.

### 19.2 The store and the writer
- **Delivers:** `dclab_rnd/experience/`: one `record(event)` function used by every writer; PostgreSQL table
  `experience_event` (append-only, partitioned by month, indexed by each id) and the file store
  `experience/YYYY/MM/DD/<session>.jsonl`; the visit session id minted at the first request and carried by a cookie
  or header, into jobs and the worker; the request id stored; a model request id returned by the gateway and stamped
  on the trace step that caused it; model replies stored (class "model reply", in the workspace only).
- **Done when:** one end-to-end flow (Home question → upload → build → stages → intern question → report) yields one
  ordered timeline with every id filled, and the same flow on the file store and on PostgreSQL gives the same events.

### 19.3 Every path emits
- **Delivers:** the existing writers emit into the log without losing their own stores: audit, transitions,
  activity, agent steps (with the assistant's own text), intern messages, draft chat and edits, memory, model usage
  and replies, output checks, jobs (start, stop, cancel by whom), exports and downloads, `/mcp` and team tool calls
  (traced today only as graph moves), errors with their request id. The person's edits become events: problem text,
  settings, solution fields before a project exists, workflow edits, overrides, stops.
- **Done when:** a test lists every route and agent tool and fails when one changes state without emitting an
  event; the owner's morning, replayed on a replica, shows the target chosen while the labels question was open.

> **Superseded from 19.4 on (2026-10-13):** the owner asked for a fuller plan that captures every decision and every
> processing step, and adds decision models (any model: open-source, classical, hosted). Part 8 from 19.4 and the new
> Part 11 are in `docs/guides/CAPTURE_AND_DECISION_DATA_PLAN.md`; the packages below are kept for their history.

## Phase C2 — Find and understand

### 19.4 Timeline, finder and session cards
- **Delivers:** `python -m dclab_rnd.experience sessions|show|search|export`; `/api/experience` routes; an Admin page
  "Sessions": a list (who, when, problem, kind, outcome, spend, flags) and a timeline per session (each chat turn, tool
  call, decision, stage, model call, error, with links to the project, draft and records). Session cards are the
  `Session` contract, written when a visit ends or goes quiet.
- **Done when:** finding "every session where the target was an identifier" is one search, and the timeline of the
  owner's morning reads top to bottom as what happened.

### 19.5 Signals of quality
- **Delivers:** feedback in the product (a helpful/not helpful mark with a reason on every agent answer, a "this is
  wrong" on every proposal) as `feedback.*` events; corrections derived by code (proposed vs saved solution, agent's
  choice vs person's override, questions re-asked, problem text rewritten); outcomes (report read, exported,
  approved; draft left unbuilt and where); failure detectors run on each session: an identifier or a constant as the
  target, a build with an unanswered blocking question, a task that contradicts the problem's words, repeated
  rephrasing, an error, a refusal, spend without a result.
- **Done when:** the owner's morning is flagged by two detectors (identifier target; build with the labels question
  open) and each detector has a test with a session it must flag and one it must not.

## Phase C3 — Learn from it

### 19.6 Consent, redaction and retention
- **Delivers:** workspace and project settings: what may be used for improving DCLab (off by default), what for
  training (off by default), by whom changed (audited); redaction of free text (names, e-mails, phone numbers,
  addresses, ids, numbers that look like amounts) before anything leaves; retention per content class; deletes that
  remove a project's sessions, traces and events together, with a tombstone event; the export refuses any class its
  consent does not allow.
- **Done when:** a test exports a session with consent off (nothing), on for improvement (redacted, no cells), on for
  training (redacted text, aggregates, no cells), and a planted e-mail and a planted cell value never pass.

### 19.7 Training and evaluation sets from real use
- **Delivers:** builders over the log, each a `dclab_core` function on the contract, so the main product can run
  them: (a) conversations of problem understanding (person ↔ Home agent) whose final card the person accepted, as
  SFT; (b) corrected turns as preference pairs (the agent's proposal, the person's correction); (c) tool trajectories
  (extending `studio/trajectories.py`); (d) workflow examples: problem card + data card → solution workflow →
  pipeline → code; (e) failure sessions as benchmark cases, through a reviewer, into a new benchmark version (never
  into a sealed set); (f) data samples: the data card (schema, roles, aggregates) and a synthetic table made from it,
  never the person's rows. Every builder: whole sessions held out, deduplicated, consent and redaction checked,
  manifest with counts and sources.
- **Done when:** the owner's morning, as a replica, comes out as one benchmark case (the target must not be an id;
  the labels question must be answered) and one preference pair, and `make sft-experience` builds a set whose
  manifest names every source session by id.

### 19.8 The organized archive
- **Delivers:** one layout, documented in `docs/guides/EXPERIENCE_ARCHIVE.md`: per workspace, events by month;
  session cards; exports by builder and version with manifests; the dev trajectories (`devlog.py`) and raw build
  transcripts beside them under the same index, so the owner's training sets come from one place with one catalog
  (`python -m dclab_rnd.experience catalog`).
- **Done when:** the catalog lists every set with its source, size, consent basis and date, and `make backup`
  includes the archive.

---

# Part 9 — From the problem to the pipeline: the solving workflows (packages 20.x)

Today: `dclab_core.understand` writes a problem brief from a description; `analyse` gives decision findings on
training rows; `plan` turns them into decisions; Home asks questions and draws a solution workflow from a pack's
template (WF-01 … WF-10); the engine runs five stages; problem kinds bring their own machines (16.15). What is missing
is visible in the owner's morning: no notion of the *family* of a problem when the words do not name a column (anomaly
detection without labels), nothing stops an identifier from becoming the target, a build can happen with a blocking
question open, the problem and the data are never checked against each other, and the workflow, the pipeline and the
code are not one connected object a data scientist can read, run and change.

## Phase W1 — Understand the problem

### 20.1 The problem card
- **Delivers:** `ProblemCard` v1 in `dclab_core` (extending the brief): the family (supervised classification,
  regression, anomaly detection without labels, with partial labels, forecasting, time to event, ranking,
  segmentation, uplift, entity tables, streams, text or images), the unit scored, the action, the moment, the label
  situation (a column, another file, none, partial), the success measure, the constraints, the open questions with the
  ones that block marked; the readiness rule in code: a project is built only when family, unit, label situation and
  moment are known and no blocking question is open.
- **Done when:** "I want build anomaly detection" with no label gives family = anomaly without labels and a blocking
  question about labels; the build is refused with that reason until it is answered.

### 20.2 The understanding conversation
- **Delivers:** the next question chosen by what it decides (the decision it unblocks, from the card), answers like
  "it is anomaly detection" mapped to the card by a model role with a validator and the rules as fallback,
  contradictions shown (the words say anomaly, the table has a label column), the card shown to the person to confirm.
  A model role at each step, measured against the rules before it is on by default.
- **Done when:** a set of written conversations (dev and sealed), one per family, reaches the right card in at most
  five questions, with the gate fixed before the run.

## Phase W2 — Understand the data, and connect it to the problem

### 20.3 The data card
- **Delivers:** `DataCard` v1: each column's role with its reason (identifier, time, entity, target candidate,
  feature, text, place, constant, leak suspect), the grain (what one row is), label files and their join keys,
  units, ranges, missingness; refusals in code: an identifier, a constant, a row number or a free-text column is
  never a target candidate.
- **Done when:** on every replica the roles match the replica's facts; `customer_id` can never be proposed as a target.
- **Status: done (2026-10-14).** The 19.11 card, moved to `dclab_core.data_card.v2` (dclab-core 0.14.0; v1 read as v2):
  each column's `purpose` with its rule in words (`why`) and the unit its name says; each table's grain in words, its
  key, the share of whole repeats, its target candidates and a labels table's join key; `card_of(..., target=)`
  judges the outcome the problem names as an outcome (a unique number is a regression target, a 30-class column a
  multiclass one) and refuses it, as a `target_refused` problem, only when its name says it is a key, it counts the
  rows, it is one value, free text, a time, a coordinate or a leak suspect; a column named as a key (`customer_id`,
  whatever its values) is never a candidate. Leak suspicion by name (words often written after an outcome: a
  suspicion to check, never a verdict) and, with the outcome, by data (empty exactly for one outcome both ways,
  naming it, separating it alone, tracking it, or a time that gives it); outcome words match the whole name or its
  form (`is_…`, `…_flag`), never a token inside a longer name; an entity whose rows share one outcome stays an entity
  ("split by it"). On all 16 replica tables the purposes match their facts and docstrings, and held-out realistic
  names get theirs (`tests/test_data_card_purposes.py`); two cannot be seen in a table and are listed there with why
  (`rain_mm` known only on the day; `last_order_date` written after a cutoff). A draft's card checks the outcome known
  when its data is analysed. Left open: an outcome named later is checked at the build by 20.4; the engine's own
  column roles (blocked, group) do not read the card yet; units only from names.

### 20.4 The problem meets the data
- **Delivers:** `check(problem_card, data_card)`: the stated target exists and fits the family; the unit equals the
  grain (or an aggregation is proposed); what the moment says is known is in the table; the label situation agrees
  with the files; each mismatch becomes a question or a blocking issue with its evidence.
- **Done when:** each planted mismatch in a mismatch suite (unit ≠ grain, target an id, label file not joined, moment
  needs a column that is absent) is found, and a matching pair raises nothing.

## Phase W3 — Analyse the data for the decisions

### 20.5 Analysis by family
- **Delivers:** `analyse` extended per family, on training rows only: without labels (the shape of normal, rule hits,
  per-entity behaviour, a planted-anomaly check of what detectors can find); partial labels; time; entities; places;
  each finding naming the decision it bears on.
- **Done when:** each family's replica yields the findings its decisions need, and none reads a holdout row.

## Phase W4 — The solution workflow

### 20.6 The solution workflow from the two cards
- **Delivers:** `solution_workflow(problem_card, data_card, findings)` in `dclab_core`: a graph of steps for this
  problem, each with purpose, inputs, outputs, the decision it makes, the check that proves it and the rule it rests
  on; built from the family's template and adjusted by the findings; validated by code (the safety blocks present,
  the gates in place); a model may propose changes, the validator decides.
- **Done when:** for every family the workflow has its safety blocks and family steps, and a planted invalid model
  proposal (holdout before the choice, no leakage audit) is refused.

### 20.7 Anomaly detection without labels, in the engine
- **Delivers:** the missing family as a kind with its own machine (16.15's pattern): detectors (robust scores per
  entity and time, isolation forest, nearest neighbours, rules from the findings), evaluation without labels (planted
  anomalies of the found shapes, stability across time, a review budget), with partial labels joining the rare-event
  kind; the review loop: confirmed cases become labels.
- **Done when:** the owner's morning on a replica (events of deliveries, no label) ends with a ranked review list,
  its evaluation on planted anomalies, and the honest limits stated; with a label file it moves to the rare-event kind.

## Phase W5 — The pipeline and its code

### 20.8 The pipeline of workflows
- **Delivers:** `Pipeline` v1: the problem workflow → data workflow → analysis → solution workflow → evaluation →
  report as one object with lineage (each step's inputs, outputs and record), resumable, its state on the graph.
- **Done when:** a project's pipeline lists every step that produced its numbers and reruns one step from its record.

### 20.9 Code for each step
- **Delivers:** each step renders its code (a notebook cell and a module), from the records, readable by a data
  scientist; the whole pipeline as a notebook and a package; a test that the exported code reproduces the recorded
  numbers on the replica.
- **Done when:** for every family supported in the engine, the exported pipeline runs in a clean environment and
  reproduces its holdout number.

## Phase W6 — Proof on common problems

### 20.10 The acceptance suite of conversations
- **Delivers:** for each family, a written person (problem words, answers, a replica table, files), run from the
  first sentence to the report; the owner's morning first. Scored: the card, the workflow, the pipeline, the holdout
  number against the replica's planted truth. Gate fixed before the run; failures from Part 8 added as cases.
- **Done when:** the gate passes on the sealed conversations, with every family listed and its interval.

---

---

# Part 10 — Agents that are managed, orchestrated and measured (packages 21.x)

The owner, 2026-10-13: better models, better agent management and orchestration, better harness engineering. Today
the Home agent, the intern, the team flow and the campaign each run their own loop on `agents.run()` or beside it;
`/mcp` and team tool calls are not traced; nothing coordinates the answers into one understanding (the Home agent
asked the labels question, and the build happened anyway); one model tier served every role that morning.

### 21.1 One harness for every agent
- **Delivers:** every agent (Home, intern, team roles, campaign, `/mcp` callers) runs through one harness: typed state,
  a step and spend budget, the tool registry and its scope guard, the validator before every write, retries with a
  reason, stop, and the Part 8 recorder on every step (the assistant's text included). No agent loop outside it.
- **Done when:** a test enumerates every agent entry point and fails if one runs a step the recorder did not see.

### 21.2 The coordinator
- **Delivers:** a lead agent over an explicit state machine: problem card → data cards of every file → their links →
  analysis → solution workflow → pipeline. It delegates to specialist roles (problem analyst, data investigator,
  file-relations reader, planner, critic, feature engineer, reporter) through typed hand-offs, and moves state only
  through deterministic gates (20.1's readiness, 20.4's checks). It never builds while a blocking issue is open, and
  says which one.
- **Done when:** the owner's morning, replayed, ends in the fraud project (labels joined, partial, biker groups), and
  a planted bad hand-off (a role naming a column that does not exist) is refused by the gate.

### 21.3 Files understood together
- **Delivers:** every file of a workspace's draft read together: a table of cases (each row one confirmed case, a
  type or reason column, a key unique per row that matches another table's key) recognized as a label file; its join
  key found and its coverage measured; files uploaded to separate drafts offered to be joined; duplicates of the key
  in the main table found and resolved before the join.
- **Done when:** the two challenge-shaped replica files, uploaded separately, are recognized (labels for orders, key
  `order_request_id`), and a table that merely shares a column is not taken for a label file.
- **Status: done (2026-10-14).** Built on 15.6's label review and cases detection: a file of known cases in one draft
  is offered the table it labels in another only when that table holds at least half of its keys (its coverage, and
  how many of its rows repeat a key, shown on Home); `POST /api/drafts/{id}/data/from-draft` brings that table into the
  cases' draft (the other draft untouched), where the two are read together and the cases join on their key; the
  key's repeats in the table are found before the join and reported with how they are resolved (every row of a
  labelled key is marked a case; a finer key both files share is named when one tells those rows apart). The fraud
  replica's two files renamed to `order_request_id`, uploaded to separate drafts, join on it; a table of bikers sharing
  `biker_id` is neither a file of cases nor a label file (`tests/test_files_together.py`). Left open: a finer key is
  named, not yet offered as a choice; files of one draft are still cleaned into one table.

### 21.4 Model management
- **Delivers:** each role's model and tier chosen per role (the routing of `models/routing.py`, with a reviewer and a
  reason), prompts versioned, a shadow model compared on live turns without being used, spend caps per role, a page
  that shows each role's model, prompt version, measured score and cost; a stronger model for the understanding roles
  when, and only when, it measures better on 20.10's conversations.
- **Done when:** switching a role's model is one audited change, and the conversation suite reports each role's
  score and cost by model.

### 21.5 Harness evaluation from real sessions
- **Delivers:** every role and the coordinator measured on the conversation suite (20.10) and on replays of recorded
  sessions (Part 8); each failure a regression case; a gate fixed before each run; traces graded step by step (the
  right question asked, the right tool, the right hand-off).
- **Done when:** the owner's morning is a permanent regression case, and a change to a prompt, a model or a gate runs
  the suite before it is on.

The main product has the same shape already: `agent_runs` with kinds assistant, lead, specialist and ops; a harness
recorder; one gateway with `prompt_releases`; governed decision points. The coordinator maps onto its lead kind, the
roles onto its specialists and decision points, and the evaluations onto its `r3_runs`.

## Adoption by the main product

Read at github.com/Shahriyar-Moradi/DCLab, commit 7fa23b9 (2026-10-09), file by file; never copied here. Its rules
that this plan obeys: no new parallel system for an existing concern; one LLM gateway, one storage port, one
decision ledger; append-only tables get no update or delete paths; migrations are additive; the workspace is the
tenant boundary; agents propose, deterministic services validate and execute; contract and schema changes regenerate
its truth artifacts in the same change.

What it has, and how each package maps onto it (wiring is that repository's own change; `dclab_core/adapters/
main_product.py` holds the field-by-field mapping, as for the earlier contracts):

| Package | Main product today | Its side of the adoption |
|---|---|---|
| 19.1 contract | `agent_events` (run, seq, type, redacted payload ≤ 16 KB, digest, `llm_invocation_id`), `ml_run_events` (workflow run, sequence, stage, event type), `llm_invocations` (one row per gateway call, ids of workspace, project, run, agent run, prompt release; redacted `safe_output`; no raw prompts), `project_decision_records` (actor human/rule/agent, rationale, evidence refs), `agent_proposals` (shadow → proposed → accepted/rejected/reverted), `execution_requests`, `data_access_events` | an adapter *rows → ExperienceEvent* over those tables; each event kind names the table and type it comes from; a kind it cannot fill is listed as missing, never invented |
| 19.2 store, ids | `X-Request-Id` bound per request but not stored; `auth_sessions` per sign-in; ids on every ledger | store the request id on `agent_events` and `execution_requests` (additive columns); the visit id is the auth session id; no new table |
| 19.3 every path emits | harness recorder, observability writer, decision record and proposal review services, gateway ledger | new event types by additive CHECK-list migrations; person edits as decision records |
| 19.4 timeline | chat rebuilt from `agent_events` | the same `timeline()` function from `dclab_core` over its adapter |
| 19.5 signals | no feedback table; proposal accept/reject/revert, experiment accepted/rejected, labels on `semantic_decision_answers` | feedback as `agent_events` of a new type, or as decision records; the detectors are library functions |
| 19.6 consent | sensitivity, LLM exposure and retention labels per column and dataset; prompts kept 12 months redacted; no training-data consent; ADR 0008 rejects fine-tuning its assistant on customer data | **owner decision**: whether the main product exports anything for training at all; the library's export refuses without an explicit consent record, so the default there is nothing |
| 19.7 sets | none | the builders run on its adapter's events, only if 19.6's decision allows |
| 20.1 problem card | `problem_specs` (task, target, unit, time column, horizon, metric, objective, constraints, success criteria, locked) | `ProblemCard` maps onto `ProblemSpec`, the family and readiness as proposals at its `target.column` / `spec.objective` decision points |
| 20.3 data card | decision point `column.semantic_role`, dataset policy labels | roles as `DatasetInvestigationProposal`s |
| 20.6 solution workflow | `ml_workflows`/`workflow_versions`, `pipeline_scientific_plans`, `ExperimentPlanProposal` | the workflow as an `ExperimentPlanProposal` payload; its validator next to its engine |
| 20.8–20.9 pipeline, code | `pipelines`/`pipeline_versions`, `workflow_runs`, `code_snapshots`, notebooks generated from `ModelBuildReproductionSpec` | the pipeline maps onto `pipeline_versions.graph_definition`; code per step onto `code_snapshots` |

Each package's "Done when" includes its mapping table and a test that runs the library on rows shaped like the main
product's, as `main_product_fields.json` does for the earlier contracts.

## Order and owner decisions

Order: first a guard against the morning's failure (an identifier, a constant, a row number or a key unique per row
is never a target; a build waits for the outcome to be named or confirmed; a table of cases is offered as a label
file), then 19.1 → 19.2 → 19.3 → 19.4 → 19.5 → 19.6 → 19.7 → 19.8, then 20.1 → 20.3 → 21.3 → 20.4 → 21.1 → 21.2 →
20.2 → 21.4 → 20.5 → 20.6 → 20.7 → 20.8 → 20.9 → 20.10 → 21.5. The failure detectors (19.5) and the problem card's readiness rule (20.1) are the two fastest
answers to the owner's morning; 20.1's readiness rule may be pulled forward as a guard while Part 8 is built.

Status:
- **20.0 the guard: done (2026-10-13).** No outcome is guessed anywhere (wizard, page, chat options, intern): the
  person names it, or the table names it (`target`, `label`, `outcome`); a key, a constant, a row number, a counter or
  free text is refused by the graph's `set_solution` check and by the draft's wizard; a list of known cases (every row
  in one state, a kind of case, a key per row, no outcome of its own) is recognised, pointed at the table in the
  workspace it labels, and not built alone unless the person names an outcome; the model may ask any open field of its
  draft once the outcome is raised. The owner's morning replayed on the replica (`tests/test_morning_guard.py`).
  Judgment suite unchanged (AEV-020 = AEV-019). Review: 10 findings, fixed; 12 planted faults caught.

- **19.1 the experience contract: done (2026-10-13).** `dclab_core.experience` (event and session, v1, 33 kinds,
  content classes, credentials refused) and `adapters.main_product_experience` (its agent events, invocations,
  decision records and run events read as events; every type mapped or listed). dclab-core 0.5.0. Review: 14
  findings, fixed; 18 planted faults caught.

- **19.2 and 19.3 the studio's record: done (2026-10-13).** `dclab_rnd/experience`: one append-only log per workspace
  (PostgreSQL `experience_event`, trigger-protected, indexed by every id; or `experience/YYYY/MM/DD/<visit>.jsonl`),
  a visit minted by the server and carried into jobs with the request id, and the existing writers emitting into it
  (audit, draft events, project activity, agent steps with the model's words, the gateway with a model-request id,
  intern turns, jobs, MCP calls, settings, edits, refusals, errors). `python -m dclab_rnd.experience sessions|show`.
  Every write route says what it records. Prompts and replies only with DCLAB_LOG_PROMPTS=1 and never for a purpose
  shown cell values. Open: partitioning by month; the Admin page and search (19.4).

- **20.1 the problem card: done (2026-10-13).** `dclab_core.card` (ProblemCard v1, 12 families, the label situation,
  blocking questions, readiness; `adapters.main_product.problem_card`), `dclab_rnd/draft/card.py`, and on Home a build
  only from a ready card (`GET /api/drafts/{id}/card`). Against the plan's text: the morning's sentence leaves rare
  events and anomalies in play (asked, not chosen); the unit and the action are open questions that never block (Home
  asks neither yet: 20.2); a ready anomaly card is refused plainly until 20.7. Only Home builds read the card; a
  project made on the project page or by the intern is checked by the solution's own rules. Review: 9 findings, fixed.

Owner decisions (answered 2026-10-13):
- **Prompts and replies: kept for training.** `DCLAB_LOG_PROMPTS=1` in the owner's `.env`: the usage log and the session
  record keep each model request's prompt and reply (class `model_text`, in the workspace). A purpose shown cell values
  (the parser's sample lines, the intern's tool results) still keeps none: those are cell values, and 19.6 decides them
  with the consent below.
- **Consent: yes, off by default for each workspace and each project, customer data included.** The record may be used
  for improving DCLab and for training once a workspace's owner switches it on, and per project; nothing is used
  before. This covers the main product too: its ADR 0008 (no fine-tuning on customer data) is to be amended there to
  "only with this consent"; until its own change, its exports stay off. 19.6 builds the switches, the audit of each
  change and the export's refusal without them.
- Still open: retention per class (proposed: events 24 months, model text 12 months); who may read session timelines
  (proposed: owner and admin roles).

Earlier note:
- The main product: its ADR 0008 rejects fine-tuning its assistant on customer data. Whether its sessions may ever
  feed a training set is the owner's call; until then the library's exports there produce nothing.
- Consent defaults: recorded for the workspace always; used for improving DCLab and for training only when switched
  on (proposed: off by default, per workspace and per project).
- Whether model replies and prompts are stored by default in the workspace (proposed: yes, class "model text",
  never exported without consent).
- Retention per class (proposed: events 24 months, model text 12 months, raw uploads as today).
- Who may read session timelines (proposed: owner and admin roles).
