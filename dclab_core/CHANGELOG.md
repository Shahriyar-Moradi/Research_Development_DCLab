# Changelog of dclab-core

The library's version (`pyproject.toml` and `dclab_core.__version__`, kept equal by a test) moves when what a host
installs changes. Each contract keeps its own version (`dclab_core.problem_brief.v1`, ...): a contract's shape changes
only with its version, and its frozen schema is in `schemas/baseline/`. A host pins the library version it tested.

## 0.14.0 (2026-10-14)

Package 20.3: what each column is for. `DataCard` moves to v2 (`dclab_core.data_card.v2`); a v1 card is read as v2,
since v2 only adds optional fields.

- Each `ColumnCard` has a `purpose` (identifier, time, entity, target candidate, feature, text, place, constant, leak
  suspect), the rule that gave it (`why`) and the unit its name says. `purposes(frame, cards, target)`.
- Each `TableCard` has its `grain` in words, the `key` (the fewest columns that tell rows apart, after whole repeats),
  the share of rows repeated whole, its `target_candidates`, and, for a labels table, `joins_on`.
- `card_of(..., target=)`: the outcome the problem names, judged as an outcome. It sharpens leak suspicions: a column
  empty for exactly one outcome, naming it almost perfectly, separating it alone, tracking a number outcome, or a time
  that gives it with another time. It is refused, with a `target_refused` problem, when its name says it is a key, it
  counts the rows, it is one value, free text, a time, a coordinate or a leak suspect: `customer_id` is never an
  outcome. `outcome_named(name)`: an outcome by its whole name or its form, never by a token inside a longer name.

## 0.13.0 (2026-10-14)

Package 19.12: training and evaluation sets from real use. A new contract; no other contract changes shape.

- `datasets` (new): `DatasetManifest` (`dclab_core.dataset_manifest.v1`): a set's builder and version, when, on what
  consent, every source by id, its counts and how it was split. Builders over what a consented export let through:
  `decisions` (one set per decision point, each label source in its own column and the label that counts, whole groups
  split into train, dev and a sealed partition), `conversations` (Home conversations that ended in a build),
  `preferences` (a proposal and the person's choice used instead), `trajectories` (an agent run's steps, never a
  tool's arguments), `workflows` (problem card, data card, workflow, solution and pipeline), `failures` (flagged
  sessions as candidate benchmark cases, a case only once a reviewer accepts it). `group_of` (a dataset, else a
  project, a draft with the project it became) and `split_of(group)` with one salt: a group lands in one split in
  every set. Each row names its sources; `refit` recounts a manifest from the rows a host kept.

## 0.12.0 (2026-10-14)

Package 19.11: data cards and synthetic tables. A new contract; no other contract changes shape.

- `datacard` (new): `DataCard` (`dclab_core.data_card.v1`), the tables of one problem without their rows: each
  column's role, dtype and bands (the ledger's vocabulary), the aggregates needed to draw it again (rounded quantiles,
  category shares with invented codes unless kept, an identifier's shape, a date range), the relations between columns
  and tables (`key`, `join`, `disjoint`, `agrees`, `equals`, `null_when`, `one_to_one`), and the problems they show
  (a coarse join that inflates labels, rows not in the table, one name with two meanings, values that disagree,
  labels gathered on a few entities, a column empty for one category). `card_of(tables)`, `synthesize(card, seed)`:
  invented tables with the same roles, relations and problems, no row of the source, marked synthetic.
- `bands` (new): the state's vocabulary and `ratio_band`, `cardinality_band`, `name_tokens`, `rows_band`, moved out
  of `ledger` (which re-exports them) so a card and a decision's state band a column the same way.

## 0.11.0 (2026-10-14)

Package 19.10: redaction before anything leaves a workspace. A new module; no contract changes shape.

- `redact` (new): `redact_text(text)` replaces e-mail addresses, web and IP addresses, IBANs and card numbers, phone
  numbers, amounts of money, street addresses, long ids and probable names (two or three capitalised words outside the
  product's own words) with placeholders, and counts what it replaced. A rule, not a guarantee: that is why a table's
  values and model prompts never leave at all. A host redacts person and agent text with it on every export.

## 0.10.0 (2026-10-14)

Package 19.9: signals of quality. A new module; no contract changes shape.

- `signals` (new): failure detectors over one session's events, each a `Flag` (detector, severity, the events that
  show it, plain words): an outcome the analysis called an identifier (or, with no analysis, named like a key: medium),
  a build while a blocking question of its conversation had no answer, a draft's latest words that say anomaly (any
  task) or rare events (a plain multiclass or regression) against the solution saved for it, the problem rewritten or
  a message sent again before a new question, an error, model spend with nothing built, run, exported or answered, a
  build refused again, the same question (field and options) asked again, a draft left unbuilt. `detect(events, only=)`; `DETECTORS`. The main product runs them on its own sessions
  (read with `adapters.main_product_experience`).

## 0.9.0 (2026-10-14)

Package 19.8: which label counts, and blind labels. New functions; no contract changes shape.

- `ledger.effective_label(record, labels)`: the one label that says what was right: among the labels no later one
  supersedes, blind, then benchmark, then measured, then exposed (the latest within a source); an exposed label (a
  person who saw the answers) counts only when it says the value used was wrong: it may demote, never promote.
  `current_labels`, `PRECEDENCE`.
- Blind labels: `blind_probability` (fixed per stratum of agreement, disagreements over-sampled, `BLIND_RATES`),
  `blind_draw` (a hash of the record's id and a seed against that probability: reproducible, its probability known
  before anyone labels; a label's weight is 1 / it), `blind_view` (what a blind reviewer sees: the question, the
  subject, the state and the options; never an answer, the value used or who decided).

## 0.8.1 (2026-10-14)

Package 19.7: the ledger's vocabulary as the studio's capture needs it; no contract changes shape.

- `ledger`: `workflow.template` is decided on a draft (Home) or a project; a state may say whether a person gave a
  reason (`reason_given`, a flag: a holdout reuse).

## 0.8.0 (2026-10-14)

Package 19.6: the decision ledger. Two new contracts at v1; the others unchanged.

- `ledger` (new): `DecisionRecord` (`dclab_core.decision_record.v1`): one decision at a point of one shared vocabulary
  (`POINTS`: 45 points: 28 technical (what a data scientist decides) and 17 workflow (what steers solving, the main product's
  ops points among them), each naming the
  main product's governed point, Jev's purpose and the solution plan's key where it has them), about a subject (a
  column, a project, a draft, a conversation, the workflow, a proposal), in a **state** that holds no table value (only
  `STATE_FIELDS`, each a band from a closed set, Jev's own where Jev has the field, a name, name tokens, a flag, a
  count, a ratio, a measure or a code; anything else refused), with its options, every answer (rule, model role,
  classifier, person: value, the model's raw answer, confidence, reason, source; each checked against its point's answer
  shape, as are the options and the value used, so none carries a row's text), the value used, who it came from, whether
  it was held for review, the level (L0 to L3; a model's answer is used only from L2), the evidence partition and its ids;
  `agreement()` and `policy_outcome` as the main product says them. `DecisionLabel`
  (`dclab_core.decision_label.v1`): the right value from one source (benchmark, blind with its sampling weight,
  measured, user_decision_exposed), who and when, what it supersedes, its evidence; a blind label carries its sampling
  weight, or says where its source keeps it (`weight_note`), never an invented one.
- `adapters.main_product_ledger` (new): `from_semantic_answer` (a `semantic_decision_answers` row as a record and,
  a label row a label of the decision at the root of its chain, never a second decision; Jev's raw answer banded with
  each purpose's acting band, a leakage rule's words read as a flag, an unavailable answer as none, a score never in
  band), `to_semantic_answer` (a record back as that row's content, every recorded row shape unchanged),
  `from_decision_record` (`project_decision_records` at their points, the value from the facts each service writes,
  a rejected or proposed row as such; the types that are not decisions with options listed with why). Columns checked against the lists read at the recorded commit.

## 0.7.0 (2026-10-14)

Package 19.5: lineage. One new contract at v1 and the event contract at v2; the others unchanged.

- `lineage` (new): `RunRecord` (`dclab_core.run_record.v1`): one run (a stage, a pipeline step, an agent, a job) with
  what it read (`RunInputs`: the table's SHA-256 and size, the sampling rule, the solution's hash, the settings, the
  earlier runs' choices it used and their ids, the seed, and `refs` for ids and digests no field names), its
  parameters, what it made (`RunOutputs`: metric, value, interval, choice, and every number by its record path), its
  duration and its code (`RunCode`: commit, whether tracked code differed and the hash of that difference, this
  library's version, the runner, the prompts' and tools' hashes). `fingerprint` hashes the inputs and parameters (not
  when, not which run, not the code; fields at their default left out, so a field added later keeps stored records
  valid), is computed on validation, and a record edited after its run is refused. `numbers`, `flat`, `run_for_path`
  (a report path's first part names the stage), `compare` (same inputs, code, value, choice, and every number written,
  timings left out), `event_payload` (with the run's own id), `latest`.
- `experience`: `ExperienceEvent` is `dclab_core.experience_event.v2`: one kind added, `run.recorded` (a run's flat
  parts, meta and counts only). `ExperienceEvent.read` reads a v1 event as v2 (every v1 event is a valid v2 event).
- `adapters.main_product_lineage` (new): `from_experiment`, `from_stage_run`, `from_agent_run`; columns checked against
  the lists read at the recorded commit (`experiments` and `pipeline_stage_runs` added), a live row is not a run,
  `INPUTS_MISSING` names what the tables cannot fill; an experiment's `intent` (the person's text) and a stage's input
  summary (only its numbers kept) never become meta. `main_product_experience.events(..., experiments=, stage_runs=)`
  adds each finished run as `run.recorded`; `from_run_record`.

## 0.6.0 (2026-10-13)

Package 20.1: the problem card. One new contract at v1; the others unchanged.

- `card` (new): `ProblemCard` (`dclab_core.problem_card.v1`) holds the brief and adds the problem's family
  (classification, regression, rare events with labels (all or a sample), anomalies without labels, forecasting,
  time to event, ranking, segmentation, uplift, entity tables, streams, text or images; the last three named by a host,
  no word rules) with how it was decided (a family the person stated first, then the label situation, then a stated
  task, then the words), the families still in play when the words do not settle it,
  the label situation (a column, a file of known cases, nothing), the outcome, the questions still open (each marked
  blocking or not) and a readiness verdict with its reasons. `problem_card(brief, words=, data=)`: rules in code; words
  like "anomaly" or "fraud" with no label situation leave rare events and anomalies in play and raise a blocking
  question about confirmed cases; an outcome is a column or a joined file of known cases, never a file only spoken of;
  the outcome blocks for a family that learns from one; the moment blocks for every family scored at a moment; the
  unit and the action are open questions that never block. `adapters.main_product.problem_card(card)`: the ProblemSpec
  content, the family at `spec.objective`, the outcome at `target.column`, `lock` only for a ready card.

## 0.5.0 (2026-10-13)

Package 19.1: the record of what a person and DCLab did. Two new contracts, at v1; the six earlier ones unchanged.

- `experience` (new): `ExperienceEvent` (`dclab_core.experience_event.v1`): one event with the ids every event shares
  (workspace, user, visit, conversation, project, run, job, request, model request, parent), `seq`, the actor, a kind
  from a closed vocabulary (`KINDS`, 33 kinds in 15 namespaces) and a payload whose every field carries its declared
  content class (meta, count, person text, agent text, model text, cell values). Refused: a field recorded as another
  class, a field outside the vocabulary, a field without its class, anything nested or long in meta, a count that is
  not a number, a time without its zone (times are stored in UTC), and anything shaped like a credential in the
  payload, the ids or the source (API keys of the common providers, JWTs, private keys, bearer tokens, a password in a
  connection string), without refusing ordinary hyphenated words. A tool's arguments and result are cell values; an
  error's message and a refusal's reason are agent text. The vocabulary, the classes and the field-class pairing are
  in the JSON Schema too. `make`, `timeline` (UTC time, then `seq`, each event once), `redact` (the vocabulary's classes
  decide, never an event's own labels), `session_card` → `ExperienceSession` (`dclab_core.experience_session.v1`:
  the problem as first written, its kind, turns, model requests, spend by currency, refusals, errors, outcome; the
  last stage is this studio's `final` or the main product's `terminal`). The contracts register in `CONTRACTS` when
  `dclab_core.contracts` is imported.
- `adapters.main_product_experience` (new): the main product's own append-only rows read as events, no new table on
  its side: `agent_events` with their `agent_runs` (a thread is the conversation, each `lead` run a turn of it; the
  `message` of a user or assistant message; a tool call that is not ok, or denied, with its code; a validator's
  reasons), `llm_invocations` (cost in its own currency: its gateway writes USD), `project_decision_records` (an
  untrusted rationale left out), `ml_run_events` by the main product's own stage rules (a fold or a candidate is a step
  inside a stage; `terminal` closes the run). Every recorded type is mapped or listed with the reason it is left out;
  every contract kind is produced or listed in `KINDS_MISSING`; the tables not read are in `TABLES_NOT_READ`; a column
  not recorded at the read commit fails loudly; the ids it does not store today (a visit, a request) are in
  `IDS_MISSING`.

## 0.4.0 (2026-10-13)

Package 16.15: the parts a host needs to run a forecast or a time to event in its own stages, choice and holdout
apart. The six contracts are unchanged: all still at v1.

- `forecast.holdout_origin` (new): the last origin, whose horizon dates after the gap end the data: the holdout a host
  locks first. `forecast.origin_scores` (new): each model's scores at one origin, optionally over given series only, so
  models are compared on the same forecasts; each with its MASE per series and the series it forecast.
- `forecast.backtest` changed: each score says whether the model forecast every series the seasonal naive did
  (`covers_every_series`); `best` is chosen only among those, as a model that skips the short series is scored on fewer
  forecasts.
- `forecast.covariate_check` changed: a numeric column the same on every row of a series that follows each series'
  mean of the target (|correlation| at least 0.9 across series) is kept out and listed to confirm, as a summary
  computed from the target, perhaps over future dates, would be; declared known ahead, it is kept. Pass training rows.
- `survival.split`, `survival.choose` and `survival.score` (new): `evaluate` in three steps, so the holdout is split
  first, the model chosen on the validation cut, and the holdout scored once afterwards. `evaluate` composes them and
  returns what it returned before.

## 0.3.0 (2026-10-13)

Packages 16.2, 16.9, 15.9, 18.7, 18.8, 16.3 to 16.8 and 16.10 to 16.12. The six contracts are unchanged: all still at v1.

- `decide` (new): decisions on top of a model. `review_budget` chooses the share of rows to review from the value of
  a case found and the cost of a review, on out-of-fold scores, with a bootstrap interval over rows or entities, read
  once on the holdout. `price_curve` models acceptance by price inside folds and gives the expected margin at each
  price multiple of a reference, selling or paying, with the best multiple and a bootstrap interval.
  `price_variation` refuses to draw a curve when prices were not varied: too few levels, too little spread, or prices
  set by a formula of the other columns. Every result states its assumption (16.9).
- `forecast` (new): many series forecast a horizon ahead (16.3). `backtest` cuts rolling origins with an optional gap, fits
  each model on the target up to the origin only (each origin records the last target date it read, and every forecast
  is returned with its dates and values), and compares naive, seasonal naive, a moving average, one boosted model for
  all series and a smoothing model per series by MASE and sMAPE. Intervals by horizon step come from residuals scaled by
  each series' own error; their coverage and pinball loss are measured on the last origin from the earlier ones, for
  the boosted model and the seasonal naive. `covariate_check` keeps the covariates declared known ahead and those that
  describe a series, keeps the others out, and marks a declared one that moves with the target "to confirm".
- `survival` (new): time to event with right-censoring (16.4). `kaplan_meier` (also per group; None past the longest
  follow-up), `censoring_bias` (survival at each horizon honest, with censored rows dropped, and with them counted as
  events), `duration_check` (columns that move with the duration among rows still going and rows that ended, or are
  filled only when the event happens), `censoring_check`, and `evaluate`: rows split by start day, training censored
  at the holdout's first day, a model chosen on the latest training rows, then Kaplan-Meier, Cox (ridge, Breslow) and
  discrete-time logistic and boosted hazards scored once on the latest starts by C-index, the integrated Brier score
  and Brier and time-dependent AUC at the horizons the follow-up reaches, weighted by the scored rows' own censoring;
  the chosen model is also fitted with censored rows dropped and counted as events, to show the bias.
- `replicas.subscription_lifetimes` (new): signups over two years, Weibull lifetimes, right-censored where the data
  ends, two columns counted during the follow-up, the true survival and every true lifetime in its facts (for judging).
- `ranking` (new): what to show each user next (16.5). `evaluate` reads the interactions up to the recommendation
  moment and the items created by it; it chooses a model at a validation origin one window earlier, then scores every
  model once on the window after the moment: popularity, recent popularity, matrix factorisation (dense, bounded) and
  a pointwise boosted ranker on candidates (trained on the window before an origin, its features ending before that
  window; item columns that track interactions kept out of it) by NDCG@k and recall@k (each against popularity, with a
  paired 90% bootstrap interval over users), coverage and novelty, for users with and without a history. `item_check`
  finds item columns that know counts after the moment (beyond everything the log knew then, for any event, on the raw
  or log scale) and the items created after it. `choose` never picks the popularity baseline. Relevant events may be a set.
- `replicas.shop_interactions` (new): a shop's views and purchases with category tastes, items arriving later, and an
  item column of lifetime sales.
- `segment` (new): groups in a table with no outcome (16.6). `prepare` records every preprocessing choice (log scale
  for skewed positive numbers, median fills, standardising, one-hot scaled by 1/sqrt(levels) with a missing level,
  identifiers and constants left out); `segment` refits k-means and a regularised diagonal Gaussian mixture on bootstrap
  resamples for each k, keeps a k only when its stability (adjusted Rand against the full fit, the lower end of a 90%
  interval) holds and its silhouette rises above the one the same method reaches on data with no groups (one Gaussian
  with the data's spread, a flat box over its ranges), chooses by silhouette, leaves out columns no plausible grouping
  of a first pass uses and searches again, adds an HDBSCAN density view, profiles each segment in the data's columns,
  describes it with a named outcome (numbers or two levels in words; never used to form it), and states that there is
  no ground truth.
- `replicas.customer_behaviours` (new): three planted buying behaviours, a tenth of customers following none, two
  columns that carry no behaviour, the planted labels and a later outcome in its facts (for judging).
- `uplift` (new): who responds to an action (16.7). `binary` reads a two-valued column (1/0, yes/no, true/false) or
  refuses it by name; `overlap` (out-of-fold gradient-boosted propensities, their AUC, the share inside a band that
  allows a small control arm, imbalanced columns); `wording` (effects only with randomisation the columns do not
  contradict; effects under an assumption with balanced overlap; differences between unlike arms; associations
  otherwise, said in the record); `evaluate`: a T-learner, an X-learner and a transformed-outcome uplift tree against a
  response model and random targeting, the model chosen on a validation cut of the training rows by the Qini
  coefficient, then the Qini coefficient and the AUUC on a holdout split once by arm, each with a paired 90% interval,
  and the uplift by decile; arms weighted by inverse propensity when the action was not random.
- `replicas.campaign_uplift` (new): a discount campaign, randomised or chosen by the team, a planted responsive
  segment and every row's true uplift in its facts (for judging).
- `entities` (new): several tables related by keys (16.8). `check_join` refuses a join that repeats rows, counted by
  the join itself (missing keys included), with the numbers; `snapshot_rows` makes (entity, cutoff) rows for entities
  with an event strictly before the cutoff; `window_features` counts, sums, recency and trend from events strictly
  before each row's cutoff, recording the last event read; `window_target` reads [cutoff, cutoff + horizon), leaves a
  row unknown when its horizon passes the data's end, and records the last event it read; `entity_check` keeps out
  entity columns written after a row's cutoff (later dates; numbers that follow a whole-log total or last event time,
  over the entities that have both); `label_join` joins a label table by key, only before the label's time;
  `build` and `evaluate` (trained on earlier cutoffs whose horizon ends by the latest, scored once on the latest,
  against recency; the training targets' last read reported).
- `replicas.shop_customers` (new): customers, orders, tickets and addresses, an export-day snapshot in the customer
  table, churn in the 30 days after monthly cutoffs.
- `spatial` (new): places (16.10). `project` and `blocks` (square cells in km), `morans_i` (nearest-neighbour
  autocorrelation with a permutation p-value), `correlation_range` (the distance within which values stay alike),
  `neighbourhood_counts` (fitted rows within a radius, never a row itself; a host fits it per fold, the studio's engine
  does not use it yet), `split_optimism` (random folds against folds of whole blocks). Longitudes are unwrapped across
  the antimeridian; coordinates not in degrees give no blocks.
- `analyse`: `F-places` when the rows carry coordinates and nearby targets are alike, with the block size and how much a
  random split flatters on the training rows. `plan`: a `spatial_block` split for it instead of a random split (an
  entity group or rows in time come first, with the blocks named as the risk); `validate` refuses a random split while
  F-places stands, and an advisor may not replace the blocks by a random split.
- `replicas.city_prices` (new): flat prices over a city with a smooth district field.
- `streams` (new): readings per unit over time and rare incidents (16.11). `windows` (rolling time windows up to each
  end: mean, spread, last, slope, change against the unit's own level read before the window, none without a baseline;
  no window across a gap longer than itself), `label_windows` (an incident in the horizon after the end; downtime
  dropped), `incident_metrics` (incidents caught in their horizon, never before the previous repair ended; lead time;
  alerts and false alerts per unit-day), and `evaluate`: the holdout line from the windows alone, training labels
  complete and known (a label delay) before it, every method fitted on the first part of the training period with its
  threshold set out of sample on the second for an alert budget and chosen there, the holdout read once, unseen machines
  on the same periods, injected detectors (the studio's percentile rules and ECOD), and a future check that recomputes
  the windows with every later reading changed.
- `replicas.machine_sensors` (new): machines with hourly readings, rare failures and the warning signs before them.
- `embeddings` (new): images and long text through an injected encoder (16.12). `content_key` (SHA-256 of the content,
  tagged by type: arrays and images by their values, files by their bytes; anything else refused), `Cache` (vectors by
  encoder and content, written through a temp file of its own), `encode` (the cache first; widths checked),
  `calibrate` (the near-copy threshold for an encoder, from items against slightly changed copies of them),
  `near_duplicates` (pairs across sides), `evaluate` (a logistic or ridge baseline: whole sources held out once, folds by
  source, near copies counted across the holdout, folds by source and random folds, the holdout's predictions kept),
  and two local baselines that need no download and say they are not pretrained: `HashingText` and `RandomFeatures`.
- `replicas.shapes_and_notes` (new): shapes photographed several times per scene, and tickets by authors with a few
  pasted again by another author.
- `replicas.store_demand` (new): daily units per store and product, a planned promotion known ahead, the day's rain not.
- `replicas`: `subscription_churn` (binary), `product_reviews` (text + tabular) (16.2), and `ride_pricing` (fees
  offered to couriers, varied or set by a formula, with the true best multiple) (16.9).
- `report.REPORTER` and `report.write_prose` (18.7): a model's summary above a section's lines, labelled as a model's, each
  sentence kept only when the host's check and the report's own stricter rules pass it; nothing kept without a check.
- `plan`: rows that arrive over time get a time split even when no drift is measured (DCLAB-R02); a random split only when no time orders them (18.8).
- `understand.label_coverage`: "partial" when a sentence about the labels says unmarked rows are unknown (15.9).
- `report`: the decision, when a run made one, in "Results, with their intervals", every number traced to
  `final.evidence.decision`; a requirement asking for a price or a fee maps to it.

## 0.2.0 (2026-10-12)

Everything after the first release, built in packages 17.2 to 17.8 and 18.1 to 18.3 of
`docs/guides/PRODUCT_BUILD_PLAYBOOK.md`. The six contracts are unchanged: all still at v1.

- `understand`: a description and its files' column names give a `ProblemBrief`; narrow rules that give up rather than
  guess; the problem analyst (an advisor role) may fill only unknown fields, with quoted words that support the value;
  the brief checker turns a disagreement into an open question, never a changed field (17.2, 18.2).
- `analyse`: tables, a brief and the host's training rows give `DataFindings` for binary, multiclass and regression;
  target statistics on the training rows only, measured out of fold against chance; label files, keys that inflate
  labels, leaks by name, power, recomputation, a missing value that speaks and parts that add up to the target; the
  data investigator role reads names and kinds only (17.3, 18.3).
- `replicas`: seeded, invented data with planted traps and their acceptance decisions: `delivery_fraud`,
  `ticket_routing`, `delivery_time` (15.1, 17.3).
- `plan`: a `SolutionPlan` of decisions with reasons, cited findings and the alternatives set aside; `validate` and
  `override` (a person's override stands, a model's is checked) (17.4).
- `agents` and `prompts`: a role contract (`Role`, `run_role`, `ScriptedAdvisor`); prompts as files pinned by
  `prompts/HASHES.json` and shipped in the wheel (18.1).
- `assess`: what to check after the first model, per task: the threshold from a review budget or stated costs chosen
  on out-of-fold predictions, calibration, per-class figures, intervals and their coverage (17.5).
- `review`: weaknesses from out-of-fold errors, hypotheses that cite them, ranked next experiments, and the paired
  rule that keeps a change (`adopt`) (17.6).
- `behaviour`: fold-safe behaviour features of an entity (counts, repeats, shrunk rates, distances) (15.4).
- `adapters.main_product`: every output above mapped onto DCLab main product's contracts (`ProblemSpec` content,
  `ProblemProfile`, `MetricPlan`, `HoldoutPlan`, `ValidationPlan`, `Objective`, the decision threshold,
  `ExperimentPlan` per decision point, `ExperimentReview`, `ExperimentChangeSet`), field by field, from field lists
  recorded from its code at commit `7fa23b9` (`adapters/main_product_fields.json`); a field it cannot fill is `None`
  with a note. Proposals only: the main product adopts them through its governed decision points (17.8).
- `features`: a small feature language compiled into fold-safe transformers, and the feature engineer role (18.5).
- Roles: the solution planner (`plan.plan_with`, its proposals guarded and validated, 18.4) and the critic
  (`review.critique`, each hypothesis citing a weakness its experiment addresses, 18.6).
- `report`: the solution report from the records, every number traced to its record path; Markdown and HTML, and
  PDF with the `pdf` extra (17.7).
- `challenge`: a brief read into its parts and their kinds, requirements with rubric points, and confidentiality (16.13).
- Packaging: the wheel carries the adapters' field lists and this changelog; it is built and used from a clean Python
  3.12 environment by `make core-wheel-check`.

## 0.1.0 (2026-10-12)

The first release (package 17.1).

- `contracts`: six frozen, versioned pydantic contracts (`ProblemBrief`, `DataFindings`, `SolutionPlan`, `Decision`,
  `Review`, `SolutionReport`) that refuse unknown fields and NaN and are read only with their version
  (`Versioned.read`); typed facts (stated with a quote, inferred with a reason, unknown without a value).
- `schemas`: the published JSON Schemas (`make core-schemas`) and each version's frozen shape in `schemas/baseline/`.
- `evidence`: what may be called evidence, a compact summary, never rows.
- `advisor`: the `Advisor` protocol a host implements over its own model gateway; `consult` checks the evidence and
  returns a validated answer or nothing.
- Dependencies: numpy, pandas, scikit-learn and pydantic (2.10 or later) only; Python 3.10 or later.
