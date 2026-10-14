# Capture everything, and turn it into training data for decision models

The owner, 2026-10-13: store **every decision and every step of processing**, everything that happens and is useful,
in DCLab and the R&D studio: user sessions, actions and chats, the agents' replies and logs, workflow runs, and the
data samples that become SFT training data, saved the correct way, organized and structured, and clean and clear to
find and understand. Then, **plan 2: fine-tune decision models** — classifiers that make the data-science and ML
decisions (is this column an identifier, does it leak, which split, which metric...) — and collect as much training
data for them as we can, under the hood. **A decision model is any model** (the owner, same day): an open-source model
fine-tuned for it (Laya, Qwen, Llama, Mistral, Phi or any other), a classical classifier, or a hosted typed-judgment
model like the main product's Jev. Jev is one example, not the target: the data and the interface are model-neutral.

**Two uses** (the owner, same day), and the ledger records the decisions of both:
1. **For data scientists and ML engineers** — the *technical* decisions of building a model, answered with a
   confidence and the evidence behind it, wherever they work (the notebook, the companion, the API, an MCP tool): is
   this column an identifier, a code, a date, free text; does it leak the outcome; which split; which metric; which
   feature recipe; which model families; accept this tuning; which threshold.
2. **Inside DCLab's workflow** — the *workflow* decisions that steer solving a problem: which family the problem is;
   which question to ask next and when the problem is ready to build; which workflow template and which steps; which
   step comes next, and when to go back (a leak sends the solution back, a model screen sends the features back); which
   tool an agent calls; which experiment to try next and when to stop; when a person must be asked.

This plan replaces Part 8 of `EXPERIENCE_AND_WORKFLOWS_PLAN.md` from 19.4 on, and adds Part 11 (decision models).
19.1 (the event contract), 19.2 and 19.3 (the studio's log, every writer emitting) are done and stay as they are.

## What was learned before writing it

**DCLab's main product and Jev** (read at commit 7fa23b9, read only):
- Jev is a hosted model (`jev-1.13.0`, TypeSafe) answering typed questions at five purposes: `column.is_identifier`
  (yes or no), `column.semantic_role` (numeric, categorical_code, identifier, datetime, free_text, other),
  `feature.leakage_suspect` (yes or no), `command.intent_route`, `proposal.completeness`. It sees **metadata only, as
  bands**: a column's name tokens, dtype, uniqueness band, nulls band, cardinality band, value pattern, availability
  (before prediction, after outcome, unknown), the target's name and the task — never a cell value.
- Its answers are stored in `semantic_decision_answers` beside the rule's answer, the agreement (agree, disagree,
  abstain, unavailable), the level (L0 shadow … L3) and the value used; labels are new rows superseding old ones, with
  a source: `blind` (a reviewer who saw neither answer), `benchmark` (planted truth), `user_decision_exposed` (a person
  who saw both: may demote, never promote). Promotion needs its R3 gates (sealed cases, blind cases, a Wilson lower
  bound, a margin over the rule, a cluster bootstrap by dataset).
- **Jev itself cannot be fine-tuned** ("TypeSafe does not fine-tune or adapt on customer data"). Its own design names
  the way: "custom trained decision models are a DCLab ModelType (classical now, SLM in Phase 9)". Plan 2 builds
  exactly that: DCLab's own decision classifiers, answering the same points with the same state vocabulary, so the
  main product can run one as another answerer (shadow first) and promote it through its own R3 gates.
- The agent-backed points (`target.column`, `spec.objective`, `split.strategy`, `column.missing_value_action`,
  `training.families_budget`, `experiment.review`, `improve.next_action`) have no stored feature state and no labels
  there today: this studio can supply both.

**This studio's decisions** (inventory 2026-10-13): about thirty decision points across problem understanding (task,
target, label coverage, the card's family and readiness, pack and kind, unit, moment, entities, the questions asked),
data understanding (column roles, category codes, risk flags, cleaning steps, findings, cases table, label join key),
leakage (the audit, the moment and the reviewer, the measured lift), split (strategy, folds, sampling, holdout reuse),
features (recipe, behaviour features, programs), models (family, review rounds, the critic), tuning, metric,
threshold, stage approvals and overrides, report claims, lessons, and the intern's tool choices. Most are made by
rules with recorded reasons; some by model roles, checked by code; some by a person. **They are recorded in pieces
(stage files, the plan, memory notes, the audit) but never as one labelled record of what was decided, from what,
by whom, and whether it was right.** The best labels already exist: planted truth in the replicas and the benchmark
cases, measured outcomes (a leak's apparent lift, a random split's optimism, a tuning gain, the holdout), and a
person's overrides.

## Principles

1. **Three layers, one set of ids.** (1) The *event log* (19.2, done): everything that happened, in order. (2) The
   *decision ledger*: every decision, with the state it was made from, every answer given (rule, model role,
   classifier, person), the value used, and its labels as they arrive. (3) *Datasets*: curated, versioned exports
   built from the ledger and the log, each with a manifest. Everything joins on the same ids (workspace, visit,
   conversation, project, run, decision).
2. **The state is metadata, in bands.** A decision's context is stored the way Jev sees it: names as untrusted text,
   types, bands (uniqueness, nulls, cardinality), value patterns, availability, aggregates. Never a cell value. So a
   decision record can always be shared once its workspace consents, and a classifier trained on it can run where Jev
   runs.
3. **A label says where it came from.** Planted truth (`benchmark`), a reviewer who saw neither answer (`blind`), a
   person who saw both (`user_decision_exposed`), a measurement (`measured`: a lift, an optimism gap, a holdout
   result). A new label supersedes, never overwrites. Training and evaluation each say which sources they use.
4. **Capture is complete; use is governed.** Everything is recorded in the workspace. Nothing leaves it for training
   unless the workspace's owner switched it on, per workspace and per project (owner's decision: off by default,
   customer data included once on). Cell values never leave. Exports refuse, by code, what their consent does not allow.
5. **Contracts in dclab_core.** The decision record, the label, the data card and the dataset manifest are versioned
   contracts with JSON Schemas, so the main product reads and writes the same shapes (mapped onto its own tables, no
   parallel system).
6. **Organized and findable.** One documented layout, one catalog, one page per layer (Sessions, Decisions, Datasets),
   and a command line for each. Every dataset can be traced back to the sessions and decisions it came from.
7. **Measured before trusted.** A decision model is on only after it beats the rule on sealed cases by a margin fixed
   before the run; until then it answers in shadow, logged, never used.

---

# Part 8 — Capture everything (19.4 – 19.12)

## Phase C2 — Find and understand what happened

### 19.4 Sessions: the timeline and the finder
- **Delivers:** an Admin page **Sessions**: the latest visits (who, when, the problem, its family, outcome, spend,
  refusals, errors), a search (by person, project, draft, kind of event, text in a problem, an error code, a date
  range), and a visit's timeline read top to bottom with links to its draft, project, records and jobs;
  `GET /api/experience/sessions`, `/sessions/{visit}`, `/search`; `make sessions` / `make session` already exist.
  Only owner and admin roles read timelines (proposed; the owner may change it).
- **Done when:** finding "every session where a build was refused" or "every visit that touched project P" is one
  search, and the owner's morning reads as what happened, on the page and on the command line.
- **Status: done (2026-10-13).** `dclab_rnd/experience/finder.py` (kinds, families, the groups `refused` and
  `errors`; words in what people and agents wrote; whole codes; a UTC window, `until` excluded),
  `agentic/pages/sessions.py` (the three routes, the owner only: the `timelines` permission), the Admin page's
  Sessions tab, `make session-search`. Left open: who else reads timelines (the owner's answer); a search reads the
  newest 20,000 events and says when it stopped there.

### 19.5 Workflow runs: lineage of every processing step
- **Delivers:** every stage run, pipeline job and agent run as a **run record** in the log: its inputs (the table's
  hash and version, the sampling rule, the settings, the solution's hash, the decisions it used), its parameters, its
  outputs (metrics with intervals, the chosen option), its duration, and the code that ran it (git commit, dclab-core
  version, the prompt files' hashes). A run can be found from any of its outputs and replayed from its record.
- **Done when:** for any number in a project's report, the run that made it, its inputs and its code version are one
  query away, and a stage rerun from its record reproduces its number.
- **Status: done (2026-10-14).** `dclab_core/lineage.py` (`RunRecord` v1, dclab-core 0.7.0; the event contract at v2
  with `run.recorded`, v1 events read as v2), `adapters/main_product_lineage.py` (experiments, stage runs, agent runs),
  `dclab_rnd/studio/lineage.py` (the stage's run in its record; `trace`, `replay` in a scratch copy, refused when the
  table, the solution or an earlier choice changed), every `agents.run()` and every draft data pipeline a run with its
  id on the events inside it; `GET /api/projects/{id}/lineage`, `make lineage`, `make replay`. Left open: a page for
  lineage (the report's numbers as links), job runs other than stages (synthetic data, exports), and stage records from
  before 19.5 (they say so; a rerun records its run).

## Phase C3 — The decision ledger

### 19.6 The decision record contract
- **Delivers:** in `dclab_core`: `DecisionRecord` v1 — the decision point (one shared vocabulary: the main product's
  points and Jev's purposes, plus this studio's points mapped onto them), the subject (a column, a project, a draft),
  the **state** (metadata in bands: Jev's own fields where Jev has them — name tokens, dtype, uniqueness, nulls,
  cardinality, value pattern, availability, target, task — and aggregates elsewhere), the options, every **answer**
  (rule, model role, classifier, person: value, confidence, reason, source), the value used, the level, and the ids;
  `DecisionLabel` v1 — the value, its source (`benchmark`, `blind`, `user_decision_exposed`, `measured`), who and when,
  what it supersedes, the evidence (a lift, a gap, a planted truth's id). Adapters: the main product's
  `semantic_decision_answers` and `project_decision_records` rows read as records and labels, field by field.
- **Done when:** a column's identifier decision, a split decision and a person's override round-trip; a state field
  carrying a cell value is refused; a main-product row becomes a record and back.
- **Status: done (2026-10-14).** `dclab_core/ledger.py` (dclab-core 0.8.0: 45 points, `STATE_FIELDS` with Jev's bands
  read at the recorded commit, records, labels), `adapters/main_product_ledger.py` (semantic answers both ways, decision
  records at their points). Left open: the studio writes no record yet (19.7); the precedence of labels and
  "an exposed label may demote, never promote" (19.8); a main-product answer's state (only its digest is stored there)
  and a blind label's sampling weight (kept with its draw there).

### 19.7 Capture every decision
- **Delivers:** each of the ~30 decision points of the inventory writes a decision record where it decides, with its
  state and every answer it had: problem understanding (task, target, labels, family, readiness, pack, kind, unit,
  moment, entities, which question was asked), data understanding (each column's role, identifier, time, text, group,
  category code; the cases table; the label join key; cleaning steps; findings), leakage (each column's audit flag,
  the moment rule, the reviewer model, the measured lift), split (strategy, folds, sampling, holdout reuse), features
  (recipe), models (family, review rounds, the critic), tuning, metric, threshold, stage approvals and overrides,
  report claims kept or dropped, lessons; and the **workflow decisions** (the second use): the problem's family and
  readiness, the next question asked or skipped, the workflow template and its steps, each move on the workflow graph
  (allowed, refused, waiting for a person) with the state it saw and the moves it had, each revisit, each tool an agent
  chose among the tools it was offered, each experiment chosen among the catalogue, each stop or go-on. A test lists
  the decision points of both kinds and fails when one decides without a record.
- **Done when:** one project, from Home to its report, leaves a record for every decision it made, each with its
  state and its answers, and the list of points is complete by the test.
- **Status: done (2026-10-14).** `dclab_rnd/ledger` (the stores, `decide`, `label`, `column_state` in Jev's bands) and
  `ledger/capture.py` at each site; 35 points captured, 10 said not decided here with why (`NOT_DECIDED`); migration
  0016 (`decision_record`, `decision_label`, append-only); a saved solution written in one batch; a studied sample's
  written solution as benchmark labels and the leakage stage's lift as measured labels. Left open: the research
  campaign's own experiment choice (its own endpoint), points for the unit, the moment, the entities and the data
  findings (no point in the vocabulary yet), labels from every other source (19.8).

### 19.8 Labels: planted truth, measurements, people
- **Delivers:** labels from every source, as superseding rows: **benchmark** labels written for every decision made on
  a replica or a benchmark case (the planted identifier, leak, time, group, label key, family); **measured** labels
  (a leak confirmed by its apparent lift, a split's optimism gap, a tuning gain, a feature recipe's ladder result, the
  holdout against cross-validation); **person** labels from overrides, corrections, accepted or rejected proposals
  (`user_decision_exposed`); and a **blind labelling** page: a reviewer shown only the state (never an answer) labels a
  sample of decisions drawn with known probabilities, over-sampling disagreements, so estimates can be weighted.
- **Done when:** every decision on a replica has its benchmark label; a person's override becomes a label that
  supersedes the rule's; a blind label carries its sampling weight; the precedence (blind, benchmark, measured,
  exposed) is one function.
- **Status: done (2026-10-14).** dclab-core 0.9.0 (`effective_label`, the blind draw); `dclab_rnd/ledger/labels.py`
  (benchmark labels from a replica's declared traps, written after every acceptance run that keeps a ledger; the blind
  draw and label); exposed labels where a person chose otherwise than the rule, measured labels from the leakage
  stage (a forbidden column's lift, a random split's optimism); the Admin page's Decisions tab (counts, blind
  labelling) behind `/api/ledger/*` (owner and reviewers); the blind draw is stratified by whether the answers agree
  (disagreements at 0.60, the rest at 0.10 to 0.30), only over points a reviewer can answer from what is shown.
  Left open: benchmark labels for the label key and the family (no replica declares them yet) and for the
  `agent_eval` cases (22.1 generates labelled cases at scale); measured labels from the tuning gain, the recipe ladder
  and the holdout against cross-validation; who labels blind (the owner's answer).

## Phase C4 — Quality, consent and data samples

### 19.9 Signals of quality
- **Delivers:** feedback in the product (helpful or not, with a reason, on every agent answer and proposal; "this is
  wrong" on a decision, which becomes a person label); corrections derived by code (proposal against saved solution,
  the agent's choice against the person's, questions re-asked, problems rewritten); outcomes (report read, exported,
  approved; draft left unbuilt and where); failure detectors run on every session (an identifier as the target, a
  build with a blocking question open, a task contradicting the words, repeated rephrasing, an error, spend without a
  result), each a flag on the session card.
- **Done when:** the owner's morning is flagged by two detectors, and each detector has a session it must flag and
  one it must not.
- **Status: done (2026-10-14).** `dclab_core/signals.py` (dclab-core 0.10.0: nine detectors, each with a session it
  flags and one it must not; the morning is flagged by three), flags on the Sessions page's cards and timelines and in
  `make sessions`; `POST /api/feedback` (every role; "wrong" on a yes/no decision is a correction label, once per person, only from
  those who may change or review the work; the near misses of Home's normal flow are tested as must-not cases), thumbs on the
  Home agent's answers; corrections from saved solutions and overrides are the ledger's exposed labels (19.8). Left
  open: feedback on proposals and decisions in the page (the route takes them), report read and export as outcomes.

### 19.10 Consent, redaction, retention, deletion
- **Delivers:** the switches the owner decided: use for improving DCLab, use for training, each off by default, per
  workspace and per project, changed only by an owner, audited; redaction of free text before anything leaves (names,
  e-mails, phone numbers, addresses, ids, amounts); retention per content class (to be decided: proposed events 24
  months, model text 12 months); deletion of a project or a session with everything it left, and a tombstone event;
  the export refusing any class or workspace its consent does not allow.
- **Done when:** with consent off nothing is exported; on for improvement, redacted text and no cells; on for
  training, the same plus decision states; a planted e-mail and a planted cell value never pass.
- **Status: done (2026-10-14).** `dclab_core/redact.py` (dclab-core 0.11.0); `dclab_rnd/consent.py`: the workspace's
  switches (`consent.json`, `GET`/`PUT /api/consent`, the owner's, audited as a policy and recorded) and each project's
  (`settings.consent` through `PATCH /api/projects/{id}`, the owner's; the earlier `share_for_training` counts as
  `train`), and the one export (`export_events`, `export_decisions`) that refuses a use the workspace does not allow,
  keeps only consenting projects, only meta, counts and redacted text, never cells, prompts, the person or the
  request; `experience/retention.py`: deleting a project or a draft removes its events, its decisions with their labels and
  the prompts the gateway's usage log kept for it; deleting a session (`DELETE /api/experience/sessions/{visit}`)
  removes its events (its decisions belong to their project or draft and go with them); PostgreSQL lets only these
  functions past the append-only triggers (migration 0017), files are rewritten under the folder's lock
  (`filelock.py`, which every appender holds); the audit row is the tombstone, with what could not be removed;
  a draft's Home conversation and decisions follow the consent of the project it became; every string of an
  exported event is redacted (a file's name leaves as its extension), and who labelled leaves by kind, never by person; retention per class (`DCLAB_RETENTION_EVENT_DAYS`,
  `DCLAB_RETENTION_MODEL_TEXT_DAYS`, off until the owner sets them) in the worker's hourly sweep and
  `python -m dclab_rnd.retention`; the switches and a session's deletion on the Admin page. Left open: the owner's
  retention days; the datasets that use the export (19.12); a tombstone inside an export already made.

### 19.11 Data samples: data cards and synthetic tables
- **Delivers:** a **data card** for every table that reaches DCLab (the schema, each column's role and bands, the
  aggregates the analysis found, the planted or found problems) as a `dclab_core` contract; a **synthetic table**
  generated from the card (the same columns, roles, bands and relations, invented values), labelled synthetic, so a
  training example can carry a table without carrying a person's rows.
- **Done when:** for the fraud replica, the card and a synthetic table from it reproduce the roles and the planted
  traps, and the synthetic table shares no row with the source.
- **Status: done (2026-10-14).** `dclab_core/datacard.py` (dclab-core 0.12.0): `card_of` and `synthesize`; on the
  fraud replica the card names all six planted traps (the inflating pair join, rows not in the table, `sender_phone`
  with two meanings, fares that disagree, frauds on a few bikers, no biker on a cancelled ride), a drawn pair of tables
  keeps every role and the same problems, and shares no row and no identifier with the source (numeric ids are drawn
  above the source's largest); `dclab_core/bands.py` the shared banding. A card keeps no row's value: quantiles at the
  1st to 99th percentiles, rounded to a hundredth of the spread; dates at the 1st and 99th percentile days over 20 rows
  or more; a number kept as a category only when every kept level has 5 rows. Drafts (`draft["data_card"]`, the table
  and its label file, whichever came first) and projects (`data.card`, built off the event loop) keep their table's
  card. Left open: relations across rows (a behaviour, a formula between numbers) are not drawn again;
  a rounded quantile still says roughly where values lie; 19.12 puts cards and drawn tables into training examples.

## Phase C5 — Training sets and the catalog

### 19.12 Training and evaluation sets from real use
- **Delivers:** builders over the ledger and the log, each a `dclab_core` function, each with a manifest (sources by
  id, counts, consent basis, date, version): (a) **decision datasets**, one per decision point: state, answers, label
  and its source, grouped by dataset and workspace for splits, a sealed partition kept apart; (b) **SFT conversations**
  of problem understanding whose card the person accepted; (c) **preference pairs** from corrections (the proposal,
  the person's choice); (d) **tool trajectories** (extending `studio/trajectories.py`); (e) **workflow examples**
  (problem card + data card → solution workflow → pipeline → code); (f) **failures as benchmark cases**, reviewed, into
  a new benchmark version (never a sealed set); (g) the **development records** (`devlog.py`) beside them.
- **Done when:** `make datasets` builds every set the consent allows, each manifest names every source, and a decision
  dataset for `column.is_identifier` holds its benchmark, measured and person labels apart.

### 19.13 The archive and its catalog
- **Delivers:** one documented layout (`docs/guides/EXPERIENCE_ARCHIVE.md`): per workspace, events by day and visit,
  decisions and labels, run records, data cards, and datasets by builder and version with their manifests; the
  catalog (`python -m dclab_rnd.experience catalog`, an Admin **Datasets** page): every set with its source, size,
  consent basis, version and date; `make backup` includes it all.
- **Done when:** the catalog lists every dataset with its lineage back to its sessions, and a restore from a backup
  gives the same catalog.

---

# Part 11 — Decision models, any model (22.1 – 22.7)

Plan 2. Models that make the data-science decisions, trained on the ledger, measured against the rules, used only when
they beat them. **Any model**: a classical classifier, an open-source model fine-tuned for decisions (Laya, Qwen,
Llama, Mistral, Phi, …), or a hosted typed-judgment model (Jev). They all answer through one interface and are judged
by the same gates; the ledger feeds all of them, each in its own training format.

### 22.0 One interface for every decision model
- **Delivers:** a `DecisionAnswerer` protocol in `dclab_core`: given a decision point, its state and its options, an
  answer with a confidence (or an abstention). Implementations: the rule (today's code), a classical classifier, an
  open-source model (local through Ollama or vLLM, or a Hugging Face checkpoint, through the gateway), a hosted model
  (Jev's typed questions, or any provider's), each registered with its name, version and the points it answers. The
  ledger records every answerer's answer beside the rule's, so models are compared on the same decisions.
- **Done when:** a scripted answerer, the rule and a classical classifier answer the same recorded decisions, and
  their agreement and accuracy come out of one function.

### 22.1 Decision datasets at scale: generated benchmarks
- **Delivers:** the largest source of labels, by construction: thousands of tables generated from the replicas and
  the public datasets with **planted** columns whose answer is known (identifiers, codes, dates, free text, leaks
  written after the outcome, a target copied, a sum identity, an entity repeated across rows, a time order), each
  also with **obfuscated names** (`f000`…) so a model learns from bands as well as names; every decision the studio's
  rules make on them recorded with its benchmark label. The same generator gives the main product R3-style cases.
- **Done when:** for each Jev purpose and each studio point with a planted truth, at least a few thousand labelled
  decisions across at least a hundred tables, grouped by table, with a sealed partition never used for tuning.

### 22.2 Classical decision classifiers
- **Delivers:** one classifier per decision point (gradient boosting or a calibrated logistic model on the state's
  bands and name tokens), trained on the decision datasets with grouped cross-validation, calibrated, each with its
  card: the point, the features, the training sources, the accuracy by source, the rule's accuracy beside it.
- **Done when:** for `column.is_identifier`, `column.semantic_role` and `feature.leakage_suspect`, each classifier's
  precision and recall on the sealed partition are reported with their intervals beside the rule's.

### 22.3 Gates fixed before the runs
- **Delivers:** the promotion gates, written before any result: as the main product's R3 (sealed cases, blind cases,
  a margin over the rule, a Wilson lower bound, a cluster bootstrap by dataset, demotion bars), applied to each
  classifier; the result stored as an evidence record, never rerun on the same sealed set.
- **Done when:** each classifier has one recorded verdict per gate version.

### 22.4 Shadow in the studio
- **Delivers:** a classifier as another answerer at its decision point, at level 0: its answer recorded in the
  decision record beside the rule's, never used; agreement and disagreement counted; disagreements go to the blind
  labelling page first.
- **Done when:** a week of studio use (or a replay of recorded sessions) gives each classifier's agreement and its
  accuracy on the labelled disagreements.

### 22.5 Open-source decision models: export, fine-tune, compare
- **Delivers:** the decision datasets exported in each family's format, from one builder: **tabular** (the state's
  bands and name tokens as columns, the label, the source and the group: parquet, for classical models and AutoML);
  **instruction and chat** (the state in words, the options, the answer and its reason: JSONL in the chat-template
  shape that Laya, Qwen, Llama, Mistral and Phi fine-tuning tools take, with a Hugging Face dataset card); **typed
  questions** (Jev's shape: the question, the state, the primitive, the answer, for hosted judgment models); fine-tuning
  recipes for a small open-source model (LoRA, on the owner's GPU or a rented one), each run recorded; the fine-tuned
  model registered as an answerer (22.0) and compared with the classical classifiers and the rules on the same sealed
  partition and gates.
- **Done when:** one decision dataset exports in the three formats from the same records, a fine-tuned open-source
  model answers through the interface, and its comparison is recorded; it is used only where it passes.

### 22.6 Choosing a model per decision
- **Delivers:** for each decision point, the answerers that passed their gates ranked by accuracy, cost and speed;
  the routing that picks one (as `models/routing.py` routes purposes: a reviewer and a reason), the rule as the
  fallback when a model abstains or fails, and demotion when its accuracy on new labels falls below its bar.
- **Done when:** switching a point's model is one audited change, and a demoted model falls back to the rule.

### 22.8 Decision models for data scientists (the first use)
- **Delivers:** the technical decision points as something a data scientist asks while they work: in the notebook
  companion and the copilot review ("this column looks like an identifier: 0.97, because …"), as an API and an MCP
  tool (`decide(point, state)`), and in the project page beside each stage's choice; each answer with its confidence,
  the answerer that gave it, the evidence records it rests on, and the rule's answer beside it; a disagreement is shown,
  never hidden, and the person's choice becomes a label.
- **Done when:** on the replicas, asking each technical point through the API, the MCP tool and the companion gives
  the same answer with its confidence and evidence, and a person's override is recorded as a label.

### 22.9 Decision models inside the workflow (the second use)
- **Delivers:** the workflow's own choices made with decision models where they passed their gates: the coordinator
  (21.2) asks which question next, whether the problem is ready, which step comes next and when to go back; the
  solution workflow (20.6) asks which template and steps fit; the review rounds ask which experiment next and when to
  stop; an agent's tool choice is checked against the model's. Every such answer goes through the validator first (a
  model never moves the graph on its own), the rule answers when the model abstains or fails, and each workflow
  decision is judged on the conversation suite (20.10) against the rules.
- **Done when:** on the conversation suite, the workflow with decision models reaches the right card, workflow and
  holdout number at least as often as with the rules, with no move the validator refuses, and each model's
  contribution is measured on its own.

### 22.7 Adoption by the main product
- **Delivers:** the classifiers as a DCLab ModelType answerer in the main product's harness (its design's own path for
  custom decision models): the same purposes and state, rows shaped like `semantic_decision_answers`, R3 runs on its
  side; the export of decision datasets in its label format (`blind`, `benchmark`, `user_decision_exposed`), under the
  consent decided (off by default, customer data once on; its ADR 0008 amended there).
- **Done when:** the mapping is a tested adapter in `dclab_core`, and an R3-shaped run on generated cases compares a
  classifier with the rule.

---

## How much data, from where

| Source | What it gives | Labels | Volume | Consent |
|---|---|---|---|---|
| Generated benchmarks (22.1) | every decision on planted tables | benchmark, by construction | as many as we generate: thousands per point | none needed: invented data |
| Replicas and benchmark cases (existing) | every decision on them | benchmark | hundreds | none needed |
| Public datasets (UCI catalog) | column roles, category codes | declared in the catalog | hundreds of columns | public |
| Studio sessions (the owner's, reviewers') | every real decision, every answer | measured, person (exposed), blind once labelled | grows with use | the workspace's switch |
| Blind labelling sessions (19.8) | a reviewer's label on sampled decisions | blind, weighted | as many as reviewers label | the reviewers' workspace |
| The main product (22.6) | its Jev answers and decisions | its labels | its traffic | its owner's consent and ADR change |

## Order and time

Order: 19.4 → 19.5 → 19.6 → 19.7 → 19.8 → 19.9 → 19.10 → 19.11 → 19.12 → 19.13, then 22.0 → 22.1 → 22.2 → 22.3 →
22.4 → 22.5 → 22.6 → 22.8 → 22.9 → 22.7. 22.9 needs the coordinator (21.2) and the solution workflow (20.6) of
`EXPERIENCE_AND_WORKFLOWS_PLAN.md`. Part 11 needs 19.6–19.8 (the ledger and its labels) and 19.10 (consent) before 22.4–22.7; 22.0–22.3 can
start right after 19.8, on generated data that needs no consent.

Estimate at the pace of 2026-10-13 (each package built, tested, reviewed, fixed and committed): Part 8, ten packages,
about 30–36 working hours; Part 11, eight packages, about 26–32 working hours (22.5's fine-tuning runs are GPU time,
not counted), plus 22.8 and 22.9, about 8–10 hours. Live model use: none in Part 8; Part 11 compares models offline.

Owner decisions still open: retention per class; who reads timelines and decisions (proposed: owner and admin roles);
whether blind labelling is done by the owner, reviewers, or both; a GPU for 22.5, and which open-source models to
fine-tune first (proposed: one small model such as Qwen 2.5 1.5B or Phi-3 mini beside the one the owner names, e.g. Laya).
