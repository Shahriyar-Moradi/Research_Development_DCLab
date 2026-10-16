# Capture everything, then solve complex problems: the audit and the plan for phases 1 and 2

Written 2026-10-15 from the owner's two documents ("DCLab Data Capture Plan", 2026-10-13, and its summary) and a
line-by-line check of this repository as it is today (main = `5946784`). The two documents were written two days before
Part 8 (packages 19.1 to 19.13) and Part 9 (20.x, 21.x, 22.x) were built, so most of what they ask for exists now under
other names. This document says, for every item, whether it is **done**, **partly done**, **missing**, a **design
difference** to decide, or the **main product's** work (its repository, never edited from here). Then it plans what is
left of phase 1 (packages 23.x) and phase 2 (packages 24.x): the agentic system that understands a problem, breaks a
complex one into its parts, routes each part to a solution, composes the solutions, and leaves every step of that in
the record.

The two products, as the owner describes them: **DCLab** (github.com/Shahriyar-Moradi/DCLab) is the main product, for
data scientists, machine-learning engineers and business users. **The R&D studio** (this repository) experiments,
tests and measures, and is a product for data scientists and engineers in its own right; what it proves well is adopted
by the main product through `dclab_core` (a wheel) and `docs/guides/CORE_INTEGRATION.md`. Both products share one aim:
understand the problem, understand the data, connect the two, analyse, draw the solution workflow, run the pipeline,
and give the code of each step.

Two rules from the owner's documents hold throughout: **no parallel system for an existing concern** (every stream
extends the store that already has it), and **nothing enters training without consent**.

---

## Part A. The audit: the capture plan against the repository today

### A.1 The eleven streams

| # | Stream (the document) | The studio today | Verdict |
|---|---|---|---|
| 1 | Sign-in and sessions | `visit.started`/`visit.ended` events (surface, client) carry the visit id; the audit keeps `sign_in` and `role_change` rows. Missing as events: a failed sign-in with its reason, sign-out, session expiry, token create/revoke, a workspace switch. | **partly** → 23.3 |
| 2 | User actions | Server side, done: `draft.*`, `data.*`, `solution.*`, `stage.*`, `graph.move`, `export.made`, `job.*` events, each with the person; the audit (15 kinds) for every governed action. Missing: events from the browser (page viewed, element acted on, upload started/finished, error shown). | **partly** → 23.2 |
| 3 | Chats with agents | `chat.person` and `chat.agent` with the full text under its class (`person_text`, `agent_text`), one event per message, redaction and retention per class. The document's "content vault" is a separate object store; here the text lives in the event store under its class (see A.4). | **done** |
| 4 | Agent steps and tool calls | `agent.step` and `tool.call` with the person, the visit, the request, the run, the model request and the parent event (the document's complaint "no user id, no trace id" was true on 2026-10-13; 19.2 and 19.3 fixed it). Every agent runs in the harness (21.1): an unrecorded step fails a test. | **done** |
| 5 | Model calls | `model.request` with purpose, tier, model, tokens, cost, time, outcome, and the prompt and reply as `model_text` (kept under `DCLAB_LOG_PROMPTS`, retained 365 days by the owner's choice); the output check (`gateway.check`) and the shadow's answer are recorded (21.4). One writer: the gateway. | **done** |
| 6 | Workflow moves and decisions | `graph.move` with the person and the validator's reason; the decision ledger (45 points, 19.6–19.8) written where each decision is made, with its options, the answer and the labels; `decision.*` events. | **done** |
| 7 | Outcomes | Stage records hold the holdout numbers; `run.recorded` and lineage (19.5) trace every number to its run; `measured` labels tie a decision to a number (an apparent lift, an optimism gap). Missing: one outcome record per decision that names the holdout result of the stage it shaped, a model promoted or rejected, and a business metric typed later. | **partly** → 23.5 |
| 8 | Feedback | `feedback.given` (`POST /api/feedback`: what, rating, reason), decision overrides as `user_decision_exposed` labels, lesson reviews, memory corrections. Missing: the kinds *correction* and *edited output* tied to a message's version, and the buttons on chat messages and stage notes. | **partly** → 23.6 |
| 9 | Errors | `error.raised` and `stage.failed` with code and message; the failure detectors (19.9). Missing: a fingerprint (exception type plus the top frame of our code) to group by, a 429 as an event, a grouped view. | **partly** → 23.4 |
| 10 | Data and files | Uploads with SHA-256, data cards with each column's purpose (identifier, time, entity, text, leak suspect…), raw-upload retention. Missing: a sensitivity class per column (`pii`, `sensitive`, `restricted`) that gates what a model sees and what a dataset keeps. | **partly** → 23.7 |
| 11 | Our own development | 203 devlog records, `make sft-dev`, `make dev-archive`, the `development` builder in the catalog. | **done** |

### A.2 The id chain

The chain exists; the names differ. `dclab_core.experience.Ids` carries workspace, user, visit, conversation,
project, run, job, request, model_request and parent; `seq` orders equal times; the schema is versioned
(`experience_event.v2`). The map from the document's names:

| The document | The studio | Note |
|---|---|---|
| `workspace_id` | `ids.workspace` | |
| `actor_kind`, `actor_id` | `actor` (person, agent, system, model), `ids.user` | |
| `auth_session_id` | — | the visit stands in; a sign-in session id comes with 23.3 |
| `client_session_id` | `ids.visit` | one sitting of one browser |
| `trace_id` | `ids.request` | the request id the response carries back (`X-Request-ID`); 23.11 emits it as W3C `traceparent` |
| `span_id`, `parent_span_id` | `ids.parent`, `ids.model_request`, `ids.job` | the tree: turn, step, model call |
| `episode_id` | `ids.project` or `ids.conversation` | a draft, a project or an intern session; the builders group by it today |
| `run_id` | `ids.run` | a stage run, a pipeline, an agent run (19.5) |
| `event_id` (ULID) | `id` + `at` + `seq` | sortable by time within a visit |
| `schema_version` | `schema_version` | |
| `data_class` | `classes` (per field, not per event) | finer than the document's |
| `consent_level` | — | read at export time today; 23.1 stamps it at write time |
| `content_ref` | not needed here | the text is in the event under its class; a vault pointer appears only where a vault exists (the main product) |

**Verdict: done**, with two small additions (23.1, 23.11).

### A.3 The event contract and the pipeline stream

One versioned contract (`ExperienceEvent`, its JSON Schema published by `make core-schemas`), the kinds and their
payload classes in `KINDS`, `redact()` before every write, a test that refuses an unknown kind or an unclassed field.
The document wanted the schemas in a `contracts/events/` package in the main product, mirrored here "until dclab_core
can carry it as a wheel": it is carried as the wheel, and the main product reads its own tables through
`dclab_core/adapters/main_product_experience.py` (the rows it cannot map are listed, never invented). **Done.**

The pipeline stream: `run.recorded` (`RunRecord`: what a run read, was told, made and ran on, with its fingerprint)
per stage, data pipeline and agent run; `make lineage` traces a report's number to its run; `make replay` reruns a
stage in a scratch copy. Missing: the steps *inside* a stage (each fold, transform, fit and metric with its parameters,
shapes and digest). **Partly** → 23.8.

### A.4 The four stores

| Store | The document | The studio today | Verdict |
|---|---|---|---|
| Operational database | PostgreSQL 16, append-only where it already is | PostgreSQL 16 (or files for one person), append-only triggers on the record, the ledger and the audit (migration 0017), `make backup`/`restore` | **done** |
| Content vault | object storage, per-workspace key, 365 days, deleted on request | no object store: text lives in the event store under its class; model text retained 365 days (the owner's choice), person and agent text 730; deletion removes it with the rest | **design difference** (decision D2): for one person or one server the event store is the vault; a separate object store with residency and per-workspace keys is the main product's package |
| Ops telemetry | OpenTelemetry → Loki/Tempo/Prometheus, error groups | JSON logs with the request id, `/healthz`, `/readyz`, Prometheus counters on their own port; no traces, no error groups | **partly** → 23.4 (fingerprints) and 23.11 (`traceparent`); the collector and its stores are a deployment choice (decision D3) |
| Training lake | Parquet bronze/silver/gold, nightly, rebuilt from the database and the vault | `make datasets` writes `<workspace>/datasets/<builder>/<name>/` (`rows.jsonl`, `sealed.jsonl`, `manifest.json`), consent-gated, redacted, with a catalog and lineage back to the visits; rebuilt, never edited | **done in substance**; the shape (columnar files, an episodes view, a model-calls set, a nightly job) → 23.10 |

### A.5 Consent and privacy

The three levels are exactly the studio's: `operate` (both switches off, the default), `improve`, `train`, per
workspace and per project (`settings.consent`, `share_for_training` honoured by the exports); changes are the owner's
and audited; `export_events`/`export_decisions` are the one way out, redacted, never a cell value, a prompt or a
person's id. Deletion of a project, a draft or a session removes its events, decisions, labels and prompts and leaves
a tombstone; retention per class runs hourly in the worker with the owner's numbers (730/365 days).

Differences:

- **Consent at write time.** The document's rule 1 says each event stores the level it was written under, so a later
  opt-in never re-labels old data. Here the export reads the switch at export time: a workspace that opts in later
  exports its earlier events. The document's rule is the honest one for a contract. → 23.1 (decision D1).
- **Pseudonyms.** The document keeps a per-workspace pseudonym for the person; the studio drops the person's id from
  everything that leaves. Dropping is stricter and loses nothing the builders use (they group by project, draft and
  conversation). Kept as is.
- **Column sensitivity.** Not built. → 23.7.
- **Rebuilding datasets after a deletion.** A deletion removes the rows; the sets are rebuilt by the next
  `make datasets`, not by the deletion itself. → 23.10 makes the deletion schedule the rebuild.

### A.6 From records to training data

| The document's dataset | The studio's builder | Verdict |
|---|---|---|
| `sft_workflow` (WF-01…WF-10 single steps, real and synthetic mixed) | `workflows` (19.12) and the sample factory (`sft_factory`, SFT v3) | done |
| `sft_trajectories` (whole episodes with free text under `train`) | `trajectories` and `corpus_v4` (whole projects held out, nothing below 50 from 10 projects) | done |
| `sft_chat` (turn and reply with a thumbs up or an edit as the label) | `conversations`; the edit as a label waits for 23.6 | partly |
| `preference_pairs` (rejected/accepted, override, shadow vs live where the check disagreed) | `preferences` | done |
| `eval_cases` (turns with a verdict, held out by workspace and episode) | the harness suite's accepted sessions (21.5) and the journeys (20.10, four sealed sets) | done |
| `outcome_labels` (decisions joined to observed outcomes) | decision sets with `measured` labels; the outcome record itself → 23.5 | partly |
| `error_cases` (turns ending in an error or a refused move, with what followed) | `failures` (19.9 detectors, reviewed) | done |

The quality gates the document lists (`numbers_ok`, validator-allowed and not overridden, no duplicate
conversations, one split per workspace and episode, the frozen benchmark's briefs excluded, minimum counts, manifests
naming every source id) are the ones `datasets.py`, `corpus_v4` and the sample factory apply today.

### A.7 Finding things

| The document | The studio today | Verdict |
|---|---|---|
| `data_catalog` table, `make catalog`, `docs/DATA_CATALOG.md`, a freshness gate | the datasets catalog with lineage (19.13, the Admin Datasets tab); the layout written by hand in `EXPERIENCE_ARCHIVE.md`; no generated catalog of event kinds, ledger points, audit kinds and builders | **missing** → 23.9 |
| naming (`stream.verb_past`, id prefixes) | kinds are `stream.noun` (`chat.person`, `stage.failed`); ids have no prefix | kept: a rename of 40 kinds buys nothing; the generated catalog says what each is |
| the episode timeline page in both products | the Admin Sessions tab, `python -m dclab_rnd.experience show`, `agents.traces.replay`; the main product's is its own | done here |
| the standard queries in `queries/` | the finder (kinds, the groups `refused` and `errors`, search); cost in the usage log; the others answered by hand | **partly** → 23.9 |

### A.8 The document's seven packages

| The document's package | Status here |
|---|---|
| 1 Contract and id chain | done (19.1–19.3) |
| 2 Fill the gaps (auth events, audit rows, feedback, errors, UI events, `user_id`) | audit rows and `user_id` done; feedback partly; errors partly; auth events and UI events missing → 23.2, 23.3, 23.4, 23.6 |
| 3 Consent and the vault | consent and deletion done; the write-time stamp → 23.1; the vault: decision D2 |
| 4 Telemetry | logs and metrics done; fingerprints → 23.4; `traceparent` → 23.11; the stack: decision D3 |
| 5 Catalog and the timeline | the timeline done; the catalog and the queries → 23.9 |
| 6 The lake and the episode builder | the sets with manifests done; the shape → 23.10 |
| 7 Gold datasets | done under other names; outcome labels → 23.5; the chat edit label → 23.6 |

**Phases 1 to 3 of the document, "before the first paid pilot"**: here they are done except the consent stamp (23.1)
and the browser's events (23.2). Those two come first.

### A.9 What belongs to the main product's repository

The document's DCLab-side items are that repository's changes, under the owner's integration plan: an `auth_events`
table, one audit row per governed action, `feedback` and `error_events` tables and their routes, the web app's UI
events, the `trace_id`/`span_id`/`episode_id` columns on `agent_events`, `llm_invocations`, `ml_run_events`,
`project_decision_records` and `data_access_events`, the content vault with per-workspace keys, OpenTelemetry. From
here: `dclab_core` carries the contracts (experience, lineage, ledger, pipeline, data card, datasets), the adapters
read its rows (`main_product_experience.py`, `main_product_ledger.py`, `main_product_lineage.py`,
`main_product_pipeline.py`, `main_product_r3.py`), and `CORE_INTEGRATION.md` says where each call goes. 23.12 keeps
that side current as phase 2 adds contracts.

### A.10 Phase 2 against the repository today

What exists for "understand the problem, draw the solution, connect the data":

- the problem brief (`understand.py`; the analyst role, 18.2, measured), the problem card (20.1) and its readiness rule,
  the conversation (20.2; the reader role, measured on four sealed sets: 25 of 30 against the rules' 20, below the bar,
  off), the data card (20.3), the problem-against-data check (20.4), the analysis by family (20.5; the investigator
  role, measured), the plan (the planner role, 18.4), the solution workflow (20.6: steps validated on their content, the
  holdout read once, after the gate), the coordinator (21.2: problem → data → links → meet → analysis → workflow →
  build), the pipeline (20.8) and its code (20.9), the report (18.7), the feature engineer and the critic;
- the router that exists: 15 problem kinds with a detection order (`kinds.detect`, specificity first) and the domain
  packs (`draft/pack.py`); the decision ledger routes each decision point to its answerer (22.6, 22.9);
- the big picture on Home: the ten-block workflow WF-01…WF-10 (solution draft, lineage, split, train-only EDA, leakage,
  features, screen, optimisation, reliability, knowledge);
- challenge mode (16.13): a brief with several parts, each part a project, each independent, a coverage table.

What the owner's phase 2 asks for that does not exist:

1. **Decomposition.** One problem that holds several (a fraud programme: fake delivery marks, fake orders for the
   fare, rings of accounts and phones, few confirmed cases, a review budget) is today one card, one kind, one project.
   Challenge mode's parts are independent; nothing composes them.
2. **A router with several candidate solutions per problem, kept with their reasons.** `kinds.detect` picks one kind;
   the plan picks model families inside it. No catalogue of approaches with their preconditions and trade-offs, and no
   record of the alternatives considered for a sub-problem.
3. **The big picture of the problem beside the big picture of the solution**, with the link from each solution block
   to the part of the problem it answers, and the drill-down to the detail workflow and the code of each step.
4. **Several agents for understanding**, each with one responsibility (decompose, review the breakdown, name the data
   each part needs, route), orchestrated, each behind a validator with the rules as its fallback.
5. **Composition**: how the parts' outputs combine (a screen, then a classifier, then a ring detector, then a budget),
   evaluated honestly: one holdout design shared by every part, read once at the end of the composition.
6. **Capture of all of it** (the owner's requirement): the map, every part, every candidate and choice, the
   composition, each agent's step, and the datasets built from them.

Nothing in phase 2 conflicts with what is built: it extends the card into a map, `kinds.detect` into a router with
candidates, the solution workflow into a map of workflows, a project into a programme of projects. The principles of
`AGENTIC_FOUNDATION_PLAN.md` hold: rules first; a model role behind a validator, with the rules as its fallback; on only
after it measured better than the rules on people nobody tuned for; every step in the harness; every decision in the
ledger; the holdout read once.

---

## Part B. Phase 1, what is left: packages 23.x

Each package: what it delivers, when it is done, and its check. Each gets its devlog record and `make sft-dev` in the
same commit, a reviewer pass, and `make check-commit`; a milestone's end gets `check-all`. Order: 23.1 and 23.2 first
(the pilot's contract and the browser), then outcomes and feedback (what training needs most), then the rest.

### 23.1 Consent stamped at write time
- **Delivers:** `ExperienceEvent` v3 and `DecisionRecord` v2 carry `consent` (`operate`, `improve`, `train`), the
  workspace's (and the project's) level at the moment of writing; the exports read the stamp, never the current switch;
  a v2 event reads as v3 with `consent=operate`. The consent page counts rows by stamp. Changing the switch writes a
  `consent.changed` event with the old and new level and the notice version shown.
- **Done when:** an event written while a workspace was at `operate` never enters a dataset after the workspace opts
  in; a deletion still removes everything; the catalog shows rows per level.
- **Decision D1** (the owner): stamp at write time (recommended; it is what a customer contract can promise).
- **Status: done (2026-10-15).** dclab-core 0.38.0 (`consent.py`, `ExperienceEvent` v3, `DecisionRecord` v2, the kind
  `consent.changed`, the adapter's `events(..., consent=)`); the studio stamps at its two writers from the workspace's
  `consent.json` (read again on change) and the project's switches (cached ten seconds, invalidated by the API; a
  project the store does not know is `operate`; a writer with no store takes the workspace's level, and the export
  checks the project again); the exports require the stamp and the switch now; the consent page counts the record by
  level and shows the notice version; a switch change writes `consent.changed` for the workspace and for a project.
  The stores learn their workspace folder when opened, so the worker and a direct writer stamp as the server does.

### 23.2 The browser's events
- **Delivers:** `ui.page_viewed` (view, pane, route), `ui.action` (view, element name, target), `ui.upload_started`,
  `ui.upload_finished`, `ui.error_shown`, batched from `core.js` to `POST /api/experience/ui` with the visit; never a
  form's contents or a query string; off for a signed-out page; the browser test (`make browser-test`) checks a page
  view lands with its visit.
- **Done when:** opening Home, the wizard and a project leaves the page views and the actions in the visit's timeline.
- **Decision D4** (the owner, for the main product): capture at route and action level for business users too.
- **Status: done (2026-10-15).** dclab-core 0.39.0 (the five kinds, `UI_KINDS`, `ui_payload`); `POST /api/experience/ui`
  (any signed-in role; at most 50 a batch; a value that is not a route, an element's name or a code is refused, so a
  form's contents never enter); `core.js` notes a page view at every `show`, an action at every button, link or data
  element clicked (its id or data name, never its text), an error shown at every failed toast or load, and Home notes
  an upload's start and end; batched twenty at a time or every two seconds and on page hide (kept alive through the
  unload); `make browser-test` has the check `ui` (a click, the Learn page and the wizard in the browser's own
  visit's timeline). The reviewer's pass moved the contract to `experience_event.v4` (a kind vocabulary is part of
  a contract's shape; v1 to v3 read as v4), made a signed-out page record nothing (a quiet request never opens the
  sign-in box), and refused an infinite count. Open: a page-side rule, not a check, keeps element ids fixed words
  (never a column's or a file's name).

### 23.3 Sign-in and session events
- **Delivers:** `auth.sign_in`, `auth.sign_in_failed` (reason code), `auth.sign_out`, `auth.session_expired`,
  `auth.token_created`, `auth.token_revoked`, `auth.workspace_switched`, `auth.role_changed` as experience events (the
  audit keeps its rows; the record gets one event per audit row of those kinds) with a user-agent hash and the IP's last
  octet removed; a sign-in session id on the visit.
- **Done when:** a failed sign-in and a revoked token appear on the Sessions page with their reasons; nothing in them
  is a secret.
- **Status: done (2026-10-15).** dclab-core 0.40.0 (the eight kinds, `hash_user_agent`, `truncate_ip`); the accounts
  API writes a sign-in (password or identity provider) with its session id, a failed sign-in with its reason code
  (`bad_credentials`, `no_workspace`), a sign-out, a token's creation (its name as the person's words) and revocation,
  a workspace switch and a role change (by the owner or the identity provider's groups) into the user's workspace's
  record, as the person, under the visit where there is one; an expired session is said once, by the request whose
  delete removed it. A password, a token or a raw address never enters; the audit keeps its rows as before. The
  reviewer's pass: another workspace's record is opened whole from the pool so its stamp is its own (a bare log is
  stamped `operate`); a failed sign-in is one event a minute per address and browser; the IPv6 prefix never keeps a
  host group; the identity-provider sign-in carries its session id too; a bearer caller's sign-out writes no event.

### 23.4 Errors with fingerprints
- **Delivers:** every `error.raised` carries a fingerprint (the exception type and the top frame inside our code), the
  request, the job or run; a 429 from `limits.py` is an event; the Sessions page groups the last 30 days by fingerprint;
  `python -m dclab_rnd.experience errors`.
- **Done when:** an error in a worker job appears grouped by fingerprint with its request id within a minute.
- **Status: done (2026-10-15).** dclab-core 0.41.0 (`errors.fingerprint`, `errors.from_exception`; `error.raised` with
  `fingerprint`, `frame`, `status`, `stage.failed` with `fingerprint`, `frame`; the contract at v5). Written where
  the failure is caught: the API's exception handler (status 500), a limit hit (status 429, `where` `limit:<kind>`),
  a failed job (`where` `job:<kind>`, under the job's visit and request, before `job.stopped`), a failed stage (the
  stage record keeps the fingerprint too), a failed draft pipeline and an intern turn. `finder.errors(log, days)`
  groups them (count, first and last, codes, where, frame, up to three request ids and jobs, visits hit);
  `GET /api/experience/errors?days=` (timelines, the owner), `python -m dclab_rnd.experience errors [--days --json]`,
  and an "Errors, last 30 days" table on the Admin page.
- **Decision D3** (the owner): the telemetry stack (self-hosted Grafana: Loki, Tempo, Prometheus) is a deployment
  choice in `deploy/`; this repository emits what such a stack reads (23.11).

### 23.5 Outcomes linked to decisions
- **Delivers:** `dclab_core.ledger.Outcome` v1 (the decision it follows, the metric, its value and interval, when and
  how it was observed: a stage's holdout, a model promoted or rejected, a business number typed by a person);
  `outcome.observed` events; written by the final stage for every decision that shaped it, by a promotion, and by
  `POST /api/projects/{id}/outcome` for a business number; the `decisions` sets gain the outcome column; the
  `outcome_labels` view in the catalog.
- **Done when:** every decision of a finished project whose stage ran has its outcome row, and the Decisions tab shows
  "approved and turned out well/badly" per point.
- **Status: done (2026-10-15).** dclab-core 0.42.0 (`outcomes.Outcome` v1, `judge`, `join`, `by_point`; the event kind
  `outcome.observed`, the contract at v6; `datasets.decisions` rows carry `outcome`, `datasets.outcome_labels`;
  `adapters.main_product_outcomes`). Written by `ledger/outcomes.py` where the number is observed: the final stage's
  holdout (`from_final`: value, interval, the CV mean as the reference, the metric's direction), the final stage's
  approval (`from_approval`: a promotion, judged well) and `POST /api/projects/{id}/outcome` (a business number with
  its reference and note); every decision of the project gets its row (`ledger/outcomes/` or the `decision_outcome`
  table, migration 0018, append-only, removed with the project); one event per observation. The Decisions tab shows
  well / badly / no outcome per point (`/api/ledger/summary` `outcomes`); `make datasets` writes
  `outcomes/outcome_labels` and the decision sets' `outcome` column; the export drops a note. Open: there is no
  rejection in this product (a promotion is the final stage approved); the main product's `model_versions` waits for 23.12.

### 23.6 Feedback of every kind, tied to a version
- **Delivers:** `feedback.given` with `kind` (`thumbs`, `rating`, `correction`, `edited_output`, `override`) and the
  target (a message id with its version, a step, a proposal, a decision, a stage note); the buttons on chat messages and
  stage notes; an edited reply kept as the label beside the original; the `conversations` builder uses a thumbs-up or
  an edit as its label (the document's `sft_chat`).
- **Done when:** a thumbs on a message creates the event with the message's version; an edited reply becomes a
  conversation row whose label is the edit.
- **Status: done (2026-10-16).** dclab-core 0.43.0 (`feedback.check` and `verdict`; `feedback.given` gains `kind`,
  `version`, `score`, `edited`, `chat.agent` gains `message` and `version`; the contract at v7;
  `datasets.conversations` uses an edit as the reply, a thumb up as the mark and a thumb down as the end of the
  example, rows carry `label` and `edits`; `main_product_experience.feedback_events`). `POST /api/feedback` runs the
  core's check and stamps the version of the message judged (the release that wrote it: library version and prompts
  digest). The buttons: Home's thumbs and a new edit control on every agent reply, thumbs on a stage's explanation.
  A viewer's feedback is recorded with its role and does not count toward a set (an edit by a viewer is refused). Open: the intern's own chat page has no buttons yet (its answers carry the version, not yet a message id).

### 23.7 Column sensitivity
- **Delivers:** `DataCard` columns gain `sensitivity` (`pii`, `sensitive`, `restricted`, `none`) from name tokens and
  value shapes (e-mail, phone, card number by Luhn, national id shapes; never a value kept); a `pii` column reaches a
  model by name only (the data-limit tests enforce it per purpose) and a dataset never holds its values; the data card
  on Home shows the classes; `data.columns_profiled` carries them.
- **Done when:** a replica with an e-mail and a phone column marks both, no event or set shows a value of them, and the
  intern's `describe_data` gives them no example values.
- **Status: done (2026-10-16).** dclab-core 0.44.0 (`sensitivity.classify`, `strip`; `ColumnCard.sensitivity` and
  `sensitivity_why`, the data card at v3; `data.analysed` carries the class map, the event at v7, with 23.6). The profile that
  every caller reads keeps no `preview` for a column that is not `none`; the Home analysis drops its top values,
  histogram and range (all three classes lose the names, `pii` and `restricted` the whole distribution); the intern's
  `describe_data` names the class and gives no example; the data-limit tests plant an e-mail, a phone and a card column
  and require that no purpose sees a value of them; Home's analysis card lists them by name and class. Open: the
  main product's `dataset_columns.sensitivity_class` is not in the field lists read (23.12 refreshes them), so its
  mapping is described in CORE_INTEGRATION.md, not coded.

### 23.8 Steps inside a stage
- **Delivers:** `RunRecord.steps`: one entry per cleaning action, imputation, encoding, feature transform, fold, model
  fit and metric computation with its name, parameters, input and output shapes, the fold index, the metric and value,
  the time and the digest of what it produced; `pipeline.step_started/finished` events; `make replay` rebuilds a stage
  step by step and compares the digests.
- **Done when:** a decision can be traced to the exact numbers it was made on, and a replay reports the first step whose
  digest differs.
- **Status: done (2026-10-16).** dclab-core 0.46.0 (`steps`: `Step`, `collect`, `step`, `digest_of`, `compare`; `RunRecord` v2
  with `steps`; the events `pipeline.step_started` and `pipeline.step_finished`; `steps_from_run_events` for the main
  product). The runner records, when a stage collects, each recipe fit and transform, model fit, prediction, fold and fold
  metric; the engine adds the split, the holdout's metrics and its interval, and one step for a kind's machine. A stage's run
  record holds them (at most 600, the rest counted), each step is two events under the run, `GET /api/projects/{id}/steps?stage=`
  lists a stage's and `?decision=` the steps that carry the numbers a decision was made on, and a replay (`make replay`, the
  pipeline rerun) reports `same_steps` and the first step whose digest differs (the command fails on it). Open: the draft's
  cleaning actions are events of the draft, not steps of a run; a kind's machine takes one step, not one per fold; a recipe is one
  `transform` step (its imputation and encoding are inside it, not steps of their own).

### 23.9 The catalog and the standard queries
- **Delivers:** `make catalog` generates `docs/DATA_CATALOG.md` and `data_catalog.json` from the contracts: every event
  kind (payload fields and classes), ledger point, audit kind, dataset builder and store, with its owner module, schema
  version, retention, the consent it needs and an example query; the test gate fails on drift (as `INDEX.md` and
  `capability.json`); `dclab_rnd/experience/queries.py` with the standard questions, used by the pages and the CLI:
  what happened in this request, replay this project, where agents get rejected most, which model calls failed their
  check, which decisions were overridden and what was chosen instead, did approved decisions turn out well (23.5), what a
  workspace costs per week, what is ready for training, what a deletion must remove; the Sessions tab takes a request id.
- **Done when:** support can open any request id as a timeline; the catalog lists every kind the code emits and the
  gate fails when one is added without it.
- **Status: done (2026-10-16).** dclab-core 0.45.0 (`catalog.build`, `queries`: nine questions as pure functions).
  `python -m dclab_rnd.data_catalog` / `make catalog` writes `docs/DATA_CATALOG.md` and `docs/data_catalog.json` (every event
  kind with its fields' classes, what never leaves and where it is emitted by scanning the code; the decision points, audit
  kinds, dataset builders, tables with their file paths, contracts, retention and consent); `--check` is a gate of
  `make rd-gates`, and a table or a builder with no description fails it. `dclab_rnd/experience/queries.py` reads the log
  and the ledger for them: `python -m dclab_rnd.experience query NAME`, `GET /api/experience/queries/NAME` (owner). A
  request id is a filter of the log (files and PostgreSQL), `finder.search`, the route, `search --request` and a field on
  the Admin Sessions tab, which lists up to 200 of its events. Open: the Admin page has the request field but not a menu of
  the standard questions yet (the route and the command line have them).

### 23.10 The lake's shape
- **Delivers:** `make datasets` also writes columnar files (Parquet beside the JSONL, when pyarrow is present) and two
  more sets: `episodes` (one row per turn: the message, the steps, the model calls, the move, the decision, the feedback
  and the outcome, joined by the ids) and `model_calls` (prompt, reply, check verdict, feedback, under `train`); the
  worker rebuilds the sets nightly where consent allows, and a deletion schedules a rebuild of the sets that held the
  rows; each manifest names the scrubber version and the builder's commit.
- **Done when:** a seeded workspace at `train` produces episodes whose every number traces to event ids; at `improve`
  no free text; a deletion is followed by a rebuild that no longer holds the rows.
- **Status: done (2026-10-16).** dclab-core 0.47.0 (`datasets.episodes`, `datasets.model_calls`, `columnar`, `redact.VERSION`,
  the manifest at v4 with `scrubber`, `commit` and `files`, the event kind `model.checked`). `make datasets` writes
  `rows.parquet` beside `rows.jsonl` when pyarrow is installed, the episodes under train (with the person's words) and under
  improve alone (codes and counts only, no decision), and the model calls under train (the gateway keeps a prompt and a reply
  only when the workspace asks and the purpose is one allowed to keep it, `Purpose.keeps_text`: the Home agent, the schema
  simulator, the conversation reader and the two decision roles, not a purpose shown another project's lessons, a file's name or
  the evidence records; redacted again on the way out; the consent notice moved to 2026-10-16 and the Admin text says so). The worker
  rebuilds the sets right after a deletion or a withdrawal of consent (`forget` and the consent switches leave a `.stale` marker,
  `datasets.sweep_due` takes it under the folder's lock and clears it before the build, so a deletion during a build is not lost; a
  build that raises is tried again in an hour) and once a day where a switch is on (`DCLAB_DATASETS_NIGHTLY=0` turns off the daily
  one only). A turn's window is its own events up to the next message; a project's events that name no conversation belong to the
  one turn that started last before them. Open: a project with train off while the workspace has it on gets no episodes (the
  improve variant is built only when train is off for the whole workspace); the check is joined to its call by the last model
  request of the context; the timed retention sweeps do not mark the sets stale (the daily build covers them).

### 23.11 Trace ids for a collector
- **Delivers:** the request id accepted and emitted as W3C `traceparent`; agent steps, tool calls, model requests and
  jobs carry span ids beside `ids.parent`; the JSON log lines carry both, so an OpenTelemetry collector (decision D3)
  joins them without a change here.
- **Done when:** one request's log lines, events and model requests share one trace id end to end, through a job.
- **Status: done (2026-10-16).** dclab-core 0.48.0 (`tracing`: `parse_traceparent`, `format_traceparent`, `trace_id_for`; `Ids`
  gains `trace`, `span` and `parent_span`, the event at v8; `main_product_experience.with_trace`). The edge continues a
  `traceparent` or derives the trace from the request id (a request that offers only a trace is known by its first 16
  characters), binds a span for the request and answers with a `traceparent`; every event and log line of the request carries
  `trace_id`, `span_id` (and the parent), a queued job runs in the same trace in a span of its own with the request's span as
  its parent, and a model request, an agent step and a tool call are spans inside it. `read(trace=)`, `finder.search`,
  `GET /api/experience/search?trace=`, `search --trace` and an indexed `experience_event.trace_id` (migration 0019) find them.
  Open: the model-request table keeps no trace id of its own (its event does); work done in a thread the request did not start
  carries no trace.

### 23.12 The main product's side: the conformance gate of every package
- **Delivers:** the adapters and `CORE_INTEGRATION.md` extended for every contract phase 2 adds (the problem map, the
  solution map, outcomes, the new ledger points and event kinds); a conformance test that every kind the studio emits
  has a mapping or a listed reason in `main_product_experience.py`; the list of that repository's own packages (A.9) as
  the owner's integration plan states them.
- **Done when:** the conformance test passes on the field lists read at one commit of the main product.
- **Status: done for phase 1 (2026-10-16); refreshed at every package of phase 2.** dclab-core 0.51.0
  (`adapters.conformance`: `CONTRACT_ADAPTERS`, `POINTS_NOT_MAPPED`, `problems()`). Of 22 contracts 11 have an adapter (5 read the
  main product's rows, 6 map a library contract out to its own) and 11 are library outputs with their reason; of 52 event kinds 16
  are produced and 36 are listed as missing with a reason; of 47 decision points 6 are read as its decision records and 27 have no
  main-product name and say why; the ids the main product's rows do not carry (visit, request, parent, trace, span, parent span)
  are listed. The gate refuses a value that is neither an existing adapter nor a library reason (the review of 23.12 found that any
  word passed). `tests/test_conformance.py` fails when a package adds one without a mapping or a reason, and checks that every finished package
  (23.1 to 23.12, 24.1, 24.2) has its section in CORE_INTEGRATION.md. The commit of the field lists is still `7fa23b9`; the main product's head
  has not moved. A section lists what each package leaves to that repository.

---

## Part C. Phase 2: the problem-solving agentic system, packages 24.x

### The shape of it

```text
problem words ──► brief ──► problem card ──► PROBLEM MAP (24.1) ──► SOLUTION ROUTER (24.3) ──► SOLUTION MAP (24.4)
                                   │                │                      │                         │
                            data card (20.3)   parts, sub-problems,   candidates per           blocks per part, a
                            meet check (20.4)  composition, data      sub-problem, chosen       workflow per leaf (20.6),
                                               needs, risks           with reasons, the         composition steps, one
                                                                      alternatives kept         holdout design
                                                                                                         │
                                                       PROGRAMME (24.5): a project per leaf ◄────────────┘
                                                       sharing the data, the composition run once on the
                                                       holdout, one report; the pipeline (20.8) and its code (20.9)
```

A small problem is a map with one part and one leaf: nothing changes for it except that the router says so and
records why. A complex problem is the fraud programme of the owner's example.

Two agentic systems, as the owner describes them, each a set of roles in `dclab_core/agents` with the pattern every
role here follows (evidence of aggregates only, a typed output, a validator, the rules as the fallback, a prompt file,
measured against the rules before it serves):

- **Understanding:** the analyst (18.2), the reader (20.2), the brief checker, and new: the **decomposer** (parts and
  sub-problems), the **map reviewer** (what the map misses: risks, leaks, missing data, ambiguity; its output is
  questions for the person), the **data-need agent** (per sub-problem, the tables, columns and labels it needs, checked
  against the data card by `meet.check`).
- **Solving:** the planner (18.4), the feature engineer (18.5), the critic (18.6), the reporter (18.7), and new: the
  **solution router** (candidates per sub-problem from the catalogue, with preconditions and evidence), the **solution
  mapper** (the composition and the shared holdout design), the **composer** (runs the composition on the leaves'
  predictions).

### 24.1 The problem map contract and its rules
- **Delivers:** `dclab_core/problem_map.py`: `ProblemMap` v1: `parts` (the big sections: each with its question, unit,
  moment, outcome, data needs, risks, in the person's words or a rule's reason) → `sub_problems` (each shaped like a
  problem card: family, kind, labels situation, the outputs it needs from other sub-problems) → `composition` (how the
  outputs combine: sequence, fusion, gate, budget) and `size` (`single` or `complex`). The rule decomposer: templates
  per family and pack, seeded from the replicas (fraud: a screen without labels, a classifier on partial labels, an
  entity-ring detector, a review budget; ticket routing: the team, then the urgency; demand with promotions: one leaf
  plus the known-ahead columns); challenge mode's `read_brief` parts become a map's parts. `validate`: every sub-problem
  has a unit and a moment consistent with its part; no sub-problem reads another's outcome before its moment; a leaf is
  a kind the engine runs (or a library kind, said); a map with one leaf is a single problem.
- **Done when:** the fraud replica's brief yields a map with its four parts and the right families, the churn replica
  yields a single map, and a map whose leaf reads a later leaf's outcome is refused with the reason.
- **Status: done (2026-10-16).** dclab-core 0.50.0 (`problem_map`: `ProblemMap` v1, `decompose`, `validate`, `fingerprint`;
  `adapters.main_product_problem_map.problem_specs`). The fraud words give four parts (a screen with no labels, a classifier on the
  confirmed cases, the linked accounts, the review list) fused on the account of the unit and cut to the budget in the words; a
  denial of confirmed cases drops the classifier; a routing asked as a team and then an urgency is a sequence; a forecast with
  planned promotions stays one problem with its need named; the 15 dev journeys stay single. The review (9 findings) made the
  composition ordered and complete, the fingerprint cover units, moments, needs, keys and budget, a stale one refused, the templates
  need the card not to say another family and no label column to be stated, and the crash paths (a long budget phrase, 13 parts,
  a part that is not a record) clipped. Left open: the router (24.3) reads the map; the understanding agents (24.2) turn a missing
  unit or moment into a question and read what the rules cannot.

### 24.2 The understanding agents
- **Delivers:** the roles `decomposer`, `map_reviewer` and `data_need`, each with its evidence (the brief, the card,
  the data card's column names and purposes; never values), parse (every part and sub-problem quotes the person or
  names a rule template; a family from the card's vocabulary; a data need by column name), fallback (24.1's rules) and
  merge; the coordinator gains the state `map` between `problem` and `data`, skipped when the router says `single`;
  Home shows the map as the big picture of the problem with the reviewer's questions; the person can add, remove or
  merge parts (each a `graph.move` through the validator and a ledger decision).
- **Done when:** on the dev set of complex problems the roles never do worse than the rules on the map's parts and
  families, and every map on Home is editable with its reasons shown.
- **Status: done (2026-10-16).** dclab-core 0.51.0 (`mapping`: the roles `decomposer`, `map_reviewer` and `data_need` with their prompts and
  parsers, `review_rules`, `coverage`, `fill_needs`, the edits `add_part`, `remove_part`, `merge_parts`, `MapReview` v1; the coordinator's state
  `map`; the ledger points `map.size` and `map.edit`). The walk is `problem → map → data → …`: it draws the map, asks the person to confirm a
  problem of several parts, and a build of one waits for that confirmation (the coordinator and the build route say the same). Home shows the map
  in the summary: each part with its family, unit, labels, what it reads, the reason it is there, the way the results combine and the reviewer's
  questions; the person removes, merges, adds or draws again, each edit refused with the validator's reasons or recorded as a graph move and a
  decision; a single problem shows its questions and can be corrected too. A model's map keeps only what rests on the person's words or a template
  the rules found, takes no label, moment or known-outcome rank from the reply, and a table or column the data card does not hold (or suspects of
  leaking) is dropped. Measured on the 14 dev problems (live, MAP-005, €0.10): mean F1 0.94 with the roles against 0.74 with the rules, size right in
  13 cases against 10, no size regression, worse in two (CD-01, CD-03: the fraud template's linked-accounts part, which their words never mention);
  the plain arm reads the same reply and scored the same. "Never worse than the rules" is therefore read as the sealed gate `map_gate_v1` (no size
  regression, at most two cases worse, a better mean F1), fixed before any run; the sealed set runs once in 24.8 and the roles serve only after it
  passes (`DCLAB_MAP_ROLES` and `draft.map.roles_passed`). The review (9 findings) and nine problems found while building it are in the record.
  Live spend for the package, each run under €0.5: about €0.53 over five dev runs (two cut at their token limit, stored unscored) and two probes.
  The owner's morning (RG-01) now has two more steps, draw and confirm the map. Left open: the programme of linked projects (24.5) builds one
  project per part; until then the build below the map is for the problem the card names.

### 24.3 The solution router and its catalogue
- **Delivers:** `dclab_core/solutions.py`: the catalogue of approaches per family and kind (what each needs: labels,
  volume, a time column, several tables; its strengths and risks; the evidence track that measured it, from 16.14's
  records by kind), and `route(sub_problem, data_card) -> candidates` ranked with reasons, the chosen one, the
  alternatives kept; the ledger point `solution.route` (options = the candidates); a single problem routes to today's
  `kinds.detect` result (never replaced); the role `solution_router` behind a validator (a candidate must be in the
  catalogue and its preconditions met by the data card).
- **Done when:** for each replica the router's first candidate is the kind its acceptance runs, with the alternatives
  and their blocking preconditions listed; the owner's fraud example routes its parts to anomaly, rare event with
  partial labels, entity tables and a budgeted review.
- **Status: done (2026-10-16).** dclab-core 0.52.0 (`solutions`: the catalogue of 16 approaches for the 12 families, `check`, `route`, `routes`, the role
  `solution_router`, `Route` v1; the ledger point `solution.route`; `problem_specs(map, routes=)`). Each need of each approach is met, unmet or unknown, in
  names and counts (an outcome nobody named is unknown, never a guess from the table); the chosen approach is the first none refutes; the others are listed
  with what blocks them; a single problem the rules drew keeps the kind the detection found (and it is said when the data refutes it); a model may choose only a
  candidate the data does not refute and whose needs are met (or the rules' own choice). On each of the 15 kinds' replicas the first candidate is that kind, and the
  fraud words route to a screen with no labels, a rare event on the cases found, entity tables and a budgeted review. Home shows each part's approach with the others
  and what blocks them, and records a `solution.route` decision once, again only when a part's approach changes. Found by the review (10 findings) and kept:
  with only the rides and the file of cases uploaded the rings part has no approach (the entity kind needs a table of events beside the cases): the map says so
  instead of pretending. Left open: the router role serves only after a sealed measurement (24.8), as the other roles.

### 24.4 The solution map
- **Delivers:** `SolutionMap` v1: a block per part, a solution workflow (20.6) per leaf, the composition steps (join by
  entity and moment, fusion of scores, a gate, the budget), and one holdout design for the whole map: the same split
  keys and moment on every leaf, the holdout read once at the composition's end, nothing chosen on it; `validate`
  refuses leaves that split differently or a composition that reads a holdout before the gate; Home draws the big
  picture of the solution beside the big picture of the problem, each block linked to the part it answers, with
  drill-down to the leaf's workflow, its pipeline steps and the code of each step (20.8, 20.9).
- **Done when:** the fraud map's leaves share one split and the composition's holdout read is the only one; a leaf
  split by time beside one split at random is refused.
- **Status: done (2026-10-16).** dclab-core 0.53.0 (`solution_map`: `SolutionMap` v1, `solution_map`, `validate`, `split_for`;
  `adapters.main_product_problem_map.holdout_plan`). The fraud map has a block for each part, one split for all (by time and by entity from the data card), the
  composition's steps ending in the one step that reads the holdout; a leaf split at random beside leaves split by time, a leaf that scores the holdout itself, an
  early or second reading of the holdout, a broken workflow and a step on what nothing made are each refused with their reason. Home draws the solution beside the
  problem: each block linked to the part it answers, with its approach, its research track, whether it hands its predictions to the composition, and the steps of its
  workflow to drill into; and the one holdout in words. The review (6 findings) made the deferral a checked list of steps (every leaf's final score is the composition's to run and the holdout is read by one step in the whole map), the composition and its score checked, the split design coherent and pinned (share, seed) and related to each workflow, a finding that would lock the holdout again a blocker, and the fingerprint cover every step's content. Left open: the drill-down to a leaf's pipeline steps and the code of each step (20.8, 20.9) comes with the
  programme (24.5), when a project exists for each leaf; a part's unit is not yet tied to a split key.

### 24.5 Programmes: linked projects and the composition
- **Delivers:** a programme document owning the map: one project per leaf, the data attached by name (16.15 C), the
  leaves' predictions on the shared holdout kept apart; `dclab_core/compose.py` runs the composition (sequence,
  fusion, gate, budget) on the leaves' outputs, scores the composed system once on the holdout with its interval and
  against the best single leaf; the graph validator opens the holdout only when every leaf's model is chosen; the
  pipeline and the code export cover the programme; one report per programme with a section per leaf.
- **Done when:** the fraud programme's composed holdout number beats its best single leaf with the interval above
  zero, and the record shows one holdout read for the programme.
- **Status: done (2026-10-16).** dclab-core 0.54.0 (`compose`, `leaves`, `programme`: `Programme` v1; the adapter `programme_experiments`). A problem of several parts runs as a programme: a leaf per block of the
  solution map on one shared split (a screen with a detector chosen on planted anomalies, a classifier that reads the screen and learns a case only from the day it was confirmed, the links of
  accounts counted as of each ride), the composition chosen on validation among every leaf alone, the rank fusions and the gates (a tie to the simpler; the bar is the best leaf on validation), and the
  holdout read once with a cluster bootstrap and the paired difference. The graph's moves `run_programme` and `drop_leaf`, a job, `/api/projects/{id}/programme`, the pipeline steps and their replay, the
  code export (which reproduces the holdout number from `dclab_core` alone), the report (`programme.report`, every number traced) and the owner's sign-off on the plan cover it. On the fraud replica, seeds 0 to
  2 (calibrated on 10 to 15): composed 0.962, 0.967 and 1.000 against 0.679, 0.600 and 0.564 for the best leaf, differences +0.283 (0.109 to 0.495), +0.367 (0.205 to 0.557), +0.436 (0.236 to 0.685), one read each.
  The review (17 findings) made a second read a person's written decision counted by the split design and a drop refused after any read, a replay and an export refused on changed inputs, the project marked
  running while it reads its files, a design that names a moment never answered by another split, the forbidden columns and the data card's leak suspects kept out of every leaf, history counted to the ride's own
  moment, labels cut at their confirmation date, ties at the cut shared, and a holdout with too few cases not read. Left open: a project per leaf (the leaves are the core's runners on one split, not one engine
  run each); a page for the programme and an agent tool for it; the bootstrap clusters by one entity key.

### 24.6 The complex replica and its journey
- **Delivers:** the `delivery_fraud` replica extended (or a sibling) with planted sub-problems: fake delivery marks
  (GPS and timing patterns), fake orders for the fare (rings of phones and accounts), partial labels, a review budget;
  acceptance decisions in the `kinds/acceptance` style (the map finds the parts; the router chooses the right
  approaches; the composition beats the best leaf; the holdout is read once; no leak reaches a leaf); a journey written
  from the first sentence.
- **Done when:** the acceptance passes on three seeds with intervals, from the committed code only.
- **Status: done (2026-10-16).** `replicas.fraud_programme` (14 rings, 28 marker bikers, 80,000 rides, partial and biased confirmed cases with their dates, a family's shared phone, a short delivery next door, cancelled
  rides, a 2% review budget) and `tests/test_fraud_programme_acceptance.py`: the replica pinned (each kind of fraud visible to one leaf only), and on seeds 0, 1 and 2 the first sentence becomes the four parts, the
  router gives each its approach, one split by time and entity, every leaf ran, no identifier, label key or coordinate reached a leaf, the composition beats the best leaf with the interval above zero, the holdout is read
  once, and without the table of accounts the holdout stays closed. The journey (`agent_eval/journeys/programme.py`, `journeys_programme_dev.json`, JP-01) carries a written person from the first sentence (the map drawn
  and confirmed, the build, the accounts attached, the programme run, the pipeline whole and its replay the same document). Left open: the sealed programme journeys are 24.8's.

### 24.7 Capture of phase 2
- **Delivers:** experience kinds `problem.mapped`, `problem.part_changed` (added, removed, merged, by whom),
  `solution.routed`, `solution.composed`, `programme.built`; ledger points `problem.decompose` (single or parts, with
  the map's fingerprint), `problem.subproblem_family`, `solution.route` (the candidates), `solution.compose` (the
  method); every new role's steps recorded by the harness as today; `datasets` builders `problem_maps` (brief → map,
  with the person's edits as correction labels and 23.5's outcome: did the composed system beat the best leaf) and
  `routes` (a sub-problem's state → the chosen approach, the alternatives, the outcome); the catalog (23.9) lists them;
  the timeline shows a programme's map as a turn; the main product's adapters (23.12) read the two contracts.
- **Done when:** a journey through the fraud programme leaves every map, route and composition decision in the ledger
  with its reasons and every agent step in the record, and `make datasets` builds both sets from it.

### 24.8 Measurement and switch-on
- **Delivers:** journeys v5: the dev set (fifteen single problems and six complex ones, written beside the rules) and a
  sealed set written by an independent agent (as the four sets of 20.10) with the expected parts and families; the gate
  fixed before the run (the map's parts and families, the route, the composed holdout against the best leaf, the holdout
  read once); each new role compared with the rules on the same people; a role serves the product only after it is
  right on at least as many people as the rules and passes the bar.
- **Done when:** the sealed run is stored once, with both arms, and the switches follow its result.

---

## Part D. Adoptable by the main product: the rule for every package

The owner's requirement (2026-10-15): everything in phases 1 and 2 must be adoptable by the main product
(github.com/Shahriyar-Moradi/DCLab). Its head is still `7fa23b9`, the commit the adapters' field lists were read at
(`dclab_core/adapters/main_product_fields.json`), so the mapping below starts from current ground. The rule, for every
package of J and K, is the one `dclab_core` already follows, made part of each package's gate:

1. **The contract and the rules live in `dclab_core`.** Every new record shape is a versioned contract with its JSON
   Schema (`make core-schemas`), and every rule that produces or judges one (the decomposer's templates, the router's
   catalogue, the composition, the consent stamp, the sensitivity scan, the error fingerprint, the catalog builder, the
   standard queries, the dataset builders) is a pure function over contract objects: numpy, pandas, scikit-learn and
   pydantic only, never a store, a route or `dclab_rnd`. `make core-wheel-check` runs it in a clean Python 3.12.
2. **The studio's part is wiring only** (`dclab_rnd`): its stores, its routes, its pages, its jobs call the core.
3. **The main product's part is an adapter and a section**: `dclab_core/adapters/main_product_*.py` maps that
   product's rows (field names from `main_product_fields.json`, read at one commit, never copied) onto the contract, a
   row it cannot fill is `None` with a listed reason, and `docs/guides/CORE_INTEGRATION.md` gains a section saying
   where each call goes there. The wiring inside that repository is its own change, under the owner's integration plan.
4. **The gate.** A package of J or K is done only when: its contract is in the wheel and its schema published; the
   adapter's conformance test passes on the field lists (`tests/test_core_adapters.py` and its siblings); the
   integration section exists; and nothing in `dclab_core` imports the studio. 23.12 is that gate, run at every package
   and refreshed when the main product's commit moves.

What goes where, package by package:

| Package | In `dclab_core` (the wheel) | In the studio (`dclab_rnd`) | The main product: adapter here, change there |
|---|---|---|---|
| 23.1 consent stamp | `experience` v3 and `ledger` v2 with `consent`; `consent.level_for(workspace, project)` and `exportable(record, use)` | the stamp at every writer; the consent page's counts | adapter reads its workspace settings into the stamp; there: a `consent_level` column on its event tables (additive) |
| 23.2 UI events | the `ui.*` kinds and their payload classes; a validator that refuses form contents | `core.js` batching to `/api/experience/ui` | the kinds are the contract; there: the Next.js sender and a route |
| 23.3 sign-in events | the `auth.*` kinds; `experience.hash_user_agent`, `truncate_ip` | the audit's auth rows emitted as events | adapter maps `auth_sessions` and `service_tokens` rows; there: an `auth_events` table |
| 23.4 error fingerprints | `errors.fingerprint(exc_type, frames)`; the `error.raised` payload | the hook on every job and request; the Sessions grouping | adapter maps `status`/`failure_reason` rows; there: an `error_events` table |
| 23.5 outcomes | `ledger.Outcome` v1, `outcomes.join(decisions, outcomes)` | the final stage and the promotion write them; the route for a business number | adapter maps `experiments.result` and `model_versions` to outcomes |
| 23.6 feedback kinds | the `feedback.given` kinds and targets; `datasets.conversations` with an edit as the label | the buttons on messages and notes | adapter maps proposal accept/reject/revert and Jev labels to feedback events; there: a `feedback` table and route |
| 23.7 column sensitivity | `sensitivity.classify` (names and shapes, never values), `strip`, `upgrade_card`; the classes on `ColumnCard` | the data card page; the data-limit tests | expected `dataset_columns.sensitivity_class`, not in the field lists read (`FIELDS_MISSING`): mapped after 23.12 |
| 23.8 steps inside a stage | `lineage.RunRecord.steps` and `lineage.compare(replay, record)` | the engine writes each step; `make replay` compares | adapter maps `ml_run_events` and `pipeline_stage_runs` to steps |
| 23.9 catalog and queries | `catalog.build()` from the contracts (kinds, points, builders, audit kinds); `queries.*` as functions over events and decisions | `make catalog`, `docs/DATA_CATALOG.md`, the Sessions tab's request lookup | the catalog page is generated the same way there; the queries run on its adapted rows |
| 23.10 the lake's shape | `datasets.episodes` and `datasets.model_calls` builders; the columnar writer as the extra `dclab-core[parquet]` | the nightly job; the rebuild after a deletion | the builders run on its adapted events; there: its own schedule |
| 23.11 trace ids | `tracing.traceparent()` parse and format; span ids on `Ids` | the middleware and the worker | adapter joins its `request_id`; there: the `traceparent` header on its edge |
| 23.12 conformance | the `NOT_MAPPED` lists per adapter | — | the test on the field lists; a refresh when the main product's commit moves |
| 24.1 problem map | `problem_map.py`: `ProblemMap` v1, the rule decomposer, `validate` | Home's map page | adapter: a map from its `ProblemSpec` and wizard answers; there: a map document on its project |
| 24.2 understanding agents | the roles `decomposer`, `map_reviewer`, `data_need` with prompts, validators and fallbacks | the coordinator's `map` state; the edits as moves | the roles run over its gateway through the `Advisor` it implements (CORE_INTEGRATION.md) |
| 24.3 solution router | `solutions.py`: the catalogue, `route()`, the `solution_router` role; the ledger point `solution.route` | the Home proposal | adapter: candidates as proposals through its governed decision points |
| 24.4 solution map | `SolutionMap` v1, `validate` (one holdout design) | the big-picture page with drill-down | adapter: a map onto its `ExperimentPlan` and `HoldoutPlan`; there: the page |
| 24.5 programmes | `Programme` v1, `compose.py` (sequence, fusion, gate, budget; scored once) | the programme store, the graph rule, the report per programme | adapter: a programme over its projects; there: the composition run in its pipeline |
| 24.6 the fraud programme | the replica with its planted parts | the acceptance and the journey | the replica is the same test material there |
| 24.7 capture of phase 2 | the kinds, the points, the builders `problem_maps` and `routes` | the writers | adapter maps its map and route rows; `NOT_MAPPED` says what it lacks |
| 24.8 measurement | — | the journeys and the gate | its own sealed run through the adapter is its choice |

What never crosses, as today: a cell value, a prompt, a person's id, a key, a path of this machine.

## Part E. Order, effort and the decisions for the owner

**Order.** 23.1 → 23.2 → 23.5 → 23.6 → 23.4 → 23.3 → 23.7 → 23.9 → 23.8 → 23.10 → 23.11 → 23.12 (milestone J, its
`check-all`), then 24.1 → 24.2 → 24.3 → 24.4 → 24.5 → 24.6 → 24.7 → 24.8 (milestone K, its `check-all`). 23.12 is
kept current through K.

**Effort.** Phase 1's remainder about 25–35 hours of building; phase 2 about 45–60 hours, the fraud programme (24.5,
24.6) the largest. Live model use: the new roles' dev measurements and the sealed run of 24.8, each capped at €0.5.
Training waits for the owner's machine; the collection under the hood grows throughout.

**Decisions for the owner** (each with the proposal the plan assumes):

| # | Decision | Proposal |
|---|---|---|
| D1 | Consent stamped at write time (an event written under `operate` never becomes training data after a later opt-in) | yes (23.1) |
| D2 | A content vault in object storage with per-workspace keys and residency | not for the studio (the event store with its classes and retention is the vault here); a package of the main product |
| D3 | The telemetry stack (self-hosted Grafana: Loki, Tempo, Prometheus) | a `deploy/` choice, the owner's; the studio emits `traceparent` and fingerprints (23.4, 23.11) |
| D4 | UI events for business users of the main product | at route and action level, never form contents (23.2 does it here) |
| D5 | Default consent for pilots | `train`, written into the pilot contract (the document's assumption, unchanged) |
| D6 | Retention of `operate` events | 2 years (already the default, 730 days); decisions and the audit forever (already) |
| D7 | Phase 2's first complex case | the delivery-fraud programme (24.6); the owner may name another |

**What the plan does not do.** It does not rename the studio's event kinds, datasets or ids to the document's names
(the generated catalog maps them); it does not move text into a separate vault; it does not build a collector; it
does not edit the main product's repository. It does not retrain anything: the data is collected under the hood, and
the owner says when the machine is ready.
