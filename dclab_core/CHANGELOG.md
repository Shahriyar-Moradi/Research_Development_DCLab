# Changelog of dclab-core

The library's version (`pyproject.toml` and `dclab_core.__version__`, kept equal by a test) moves when what a host
installs changes. Each contract keeps its own version (`dclab_core.problem_brief.v1`, ...): a contract's shape changes
only with its version, and its frozen schema is in `schemas/baseline/`. A host pins the library version it tested.

## 0.38.0 (2026-10-15)

Package 23.1: consent stamped at write time.

- `consent.py`: the ladder `operate < improve < train`, `level_of(workspace, project)`, `exportable(level, use)`,
  `changes(before, after)` and `NOTICE_VERSION`.
- `experience.ExperienceEvent` v3 (`consent`, default `operate`; v1 and v2 read as v3) and the kind
  `consent.changed`; `make(..., consent=)`.
- `ledger.DecisionRecord` v2 (`consent`; v1 reads as v2).
- `adapters.main_product_experience.events(..., consent=)` stamps every event with the host's level;
  `CONSENT_MISSING` says the main product has none on its rows today.

## 0.37.0 (2026-10-15)

The owner asked Claude to help label decisions.

- `ledger.DecisionLabel` v2 (`dclab_core.decision_label.v2`): the source `model`, a model's answer to a decision's blind
  view, named (`by="model:<name>"`) and weighted like a person's blind label. A v1 label reads as v2. `effective_label`
  never takes a model's label as the truth.
- `datasets`: the decision sets keep a `model` column beside each source's.
- `prompts/decision_labeler.md`: what the model is asked; `conversation_reader.md` names every family the card knows and
  tells cases to review from items to rank.

## 0.36.0 (2026-10-15)

Package 20.10 v2 and the open half of 16.15 C. No contract changes shape.

- `card`: the less common families read from people's own words: lifetimes with rows still waiting ("expected lifetime",
  "hasn't cancelled yet"), items a person is likely to buy, an offer that changes a mind, early warnings from logged
  readings, an outcome as no order in the next days, a model that reads the note itself. Time to event is tried before
  forecasting. A label situation said ("no history of confirmed fraud", "never confirmed any") is nothing labelled, and
  outweighs a coverage the words only suggest. `refine_family`: an answer that only gives the output's shape ("a yes or no
  per note") keeps the family the words name. `problem_card`'s `data` takes `text_only` (the table's only input is a long
  text: text_or_images); several tables with an outcome said in words is entity_tables.
- `conversation`: family answers for survival ("the chance they are still with us at 90 days"), alerts per machine,
  ordered lists of items and an offer's lift.
- `embeddings.near_duplicates`: blocks hold at most `BLOCK_CELLS` similarities; above `EXACT_UP_TO` items, candidates
  come from random-hyperplane hashing and are checked exactly (`method`, `candidates_compared`, `recall_at_threshold`
  from `hashed_recall`). Folds for a multiclass task are stratified by class and grouped by source.
- `replicas.shapes_and_notes.notes(outcomes=True)`: a ticket's topic (seven classes) and minutes to resolve; the default
  table is unchanged.

## 0.35.0 (2026-10-15)

Package 16.15 B: the library kinds run stage by stage in the engine. No contract changes shape, and every library
function's output is unchanged (checked on each replica).

- `uplift.split` (by arm only, with the holdout's fingerprint), `uplift.validate` (the training rows' outcomes only) and
  `uplift.score` (the holdout read once): `evaluate` is their composition.
- `ranking.setup`, `ranking.validate` (the log up to the moment) and `ranking.score` (the window after it): `evaluate`
  is their composition.
- `segment.search` (the two passes, never an outcome) and `segment.describe_outcome` (a later outcome per segment):
  `segment` and its report use them.
- `card.OUTCOME_AS_EVENT`: ranking's outcome, an event in the window after the list, may be said in words.

## 0.34.0 (2026-10-15)

Package 22.7: adoption by the main product. A new module; no contract changes shape.

- `adapters.main_product_ledger.to_semantic_answer`: a classifier's yes or no is written as the probability of yes
  (a confident no was written as its confidence, read there as a yes), a choice as its label; `in_acting_band` and
  `agreement` follow the main product's bands. `as_raw` is the reading.
- `adapters.main_product_r3` (new): the state as the main product sends it (`project`), a DCLab answerer as its R3
  `Answerer` (`R3Answerer`), the R3-shaped reading (`run`, an `r3_runs` row with its digests) and its answer and label
  rows; `main_product_fields.json` records the releases and the R3 fields read at the same commit.

## 0.33.0 (2026-10-15)

Package 21.5: what the owner's morning's other half found. No contract changes shape.

- `meet.check`: "anomaly" stated while known cases are here (a joined file of cases, a named outcome column) is a
  `labels_contradict` question, blocking while the family was stated and until the person says nothing is labelled.
- `card.LABEL_WORDS`: "the cases we found", "the frauds the team caught" are known cases, tried after the words for a
  column.
- `datasets.failures`: a candidate keeps the person's answers by question field, so a reviewed session can be replayed.
- `coordinator.FAMILY_NOT_RUN` (0.32.0, 20.10): the build's rule for a family no machine runs, said at the problem.

## 0.32.0 (2026-10-15)

Package 20.10: the acceptance suite of conversations found where the card misread people. No contract changes shape.

- `card.problem_card`: the words name three more families (streams: sensor readings, an alert before a failure;
  entity tables: an outcome from several related tables; text or images: an outcome read from text or images), and
  more of how people say time to event ("how long will each subscriber stay") and uplift ("people it would make
  buy"). `data["columns"]`: with a table, an outcome said in words counts only for a family whose outcome is computed
  from related tables (entity tables, streams); for the others it is a column or a file of cases.
- `conversation.read_answer`: the family answers for streams, more of time to event and uplift; an answer naming
  several columns names its outcome first (by where it names them, not the table's order).

## 0.31.0 (2026-10-15)

Package 22.8: decision models for data scientists. No contract changes shape.

- `answerers.compare`: a label superseded by a later one counts no more, wherever the later one is (a person's second
  answer on the same column is a new record that supersedes their first label).

## 0.30.0 (2026-10-15)

Package 22.5: open-source decision models. A new module; no contract changes shape.

- `decision_export` (new): the same decision rows as a table (`tabular`, one column per state field), chat
  conversations (`chat`, the shape fine-tuning tools take, with `card`, its Hugging Face dataset card) and typed
  questions (`typed`: the question, the state, the primitive, the answer); a sealed row is never written.
- `answerers.ModelAnswerer`: a language model as an answerer, asked through an injected `Advisor` (the host's
  `decision_answer` purpose) with the question, the options and the state's bands and name tokens; an answer outside
  the options, no answer or a failed request is an abstention (failures counted).
- `prompts/decision_answer.md`: its prompt.

## 0.29.0 (2026-10-15)

Package 22.4: the decision classifiers in shadow. No contract changes shape.

- `ledger`: an answer whose source is `shadow` (recorded beside the others, never used) is no answer of the record:
  `DecisionRecord.answer` and `agreement` set it aside, and `effective_label` never counts it among the answers a
  person was shown. `blind_view` keeps every band in the state (a band left out when it equals the answer would give
  the answer away); only a name equal to the answer is left out.
- `datacard`: `dtype_band` and `value_pattern`, the bands a decision's state and a data card share, made public.

## 0.28.0 (2026-10-15)

Package 20.8: the pipeline of workflows. A new contract, `pipeline.v1`.

- `pipeline` (new): `Pipeline` and `PipelineStep`, every step behind a project's numbers (the problem card, the
  table, the solution, the workflow, each stage, the report) in order, each with what it read and made by hash, its
  run and fingerprint, where the host keeps its record and how it is rerun; a step depends only on earlier steps.
  `stale_from` carries a stale, failed or unrun step forward to every step that read it; `produced` gives the steps
  behind a number's record path; `resume` the first step to run again; `graph_definition` and `content_digest` the
  graph and digest of the main product's `pipeline_versions` row (`adapters/main_product_pipeline.py`).

## 0.27.0 (2026-10-14)

Package 16.15 C (third part): long texts in the engine. `embeddings` gains what an engine runs stage by stage; no
contract changes shape; `evaluate` is unchanged.

- `embeddings.split`: whole sources held out, chosen from their names alone (each source's place is the SHA-256 of the
  seed and its name; more rows of the same sources never move one); the record keeps a fingerprint of the held-out set,
  not the names. `holdout_mask`. `choose`: the regularisation (C 0.1, 1 or 10) by folds grouped by source on the
  training rows, the chosen one also on random folds, near copies across both counted. `score`: fitted on the training
  rows only and scored once on the held-out sources, with the holdout's predictions.
- Every function reads a source as the same text (a date, a number or a string alike); `choose` uses folds grouped by
  source and stratified by class for a binary target and refuses folds that hold one class; `score` says, rather than
  scores, a holdout of one class (`why_no_score`).

## 0.26.0 (2026-10-14)

Package 16.15 C (second part): streams in the engine. `streams` splits into what an engine runs stage by stage; no
contract changes shape; `evaluate` is unchanged.

- `streams.split`: the holdout's start (`cut`, the latest 30 % of window ends) and the validation's (`validation_from`)
  from the windows' dates alone, never an incident. `known_by` keeps the readings before the cut and the incidents
  known by it (their label delay included); `training_view` builds and labels the fit and validation windows from
  those only. `choose` fits every method on the fit windows, sets each threshold on the validation windows for the
  alert budget and picks the best incident recall there; `score` builds the holdout's windows from every reading and
  scores the chosen method and the two that need no label once, and the chosen one again on machines left out.
- `split` cuts on the time axis (a machine down after a failure writes fewer windows and never moves the line);
  `choose` refuses a validation period with no scoreable incident; `score` leaves the machines-left-out measure empty
  (and says so) when the other machines have no validation window.
- `written_ahead`: sensors that follow the hours to the next known incident on windows far before it (more than twice
  the horizon), to be confirmed; `without` drops a sensor's features from a view; `wilson`, the interval of a share.

## 0.25.0 (2026-10-14)

Package 16.15 C (first part): several tables in the engine. `entities` splits into what an engine runs stage by stage;
no contract changes shape.

- `entities.split`: the holdout is the latest cutoff whose horizon the log reaches (from dates alone), the training
  cutoffs those whose horizon ends by it, the validation the latest of them; cutoffs past the data's end are left out
  and said. `clip` keeps the events before a moment; `training_view` builds every training row and target from the
  log clipped at the holdout. `choose` picks among `CANDIDATES` (boosted trees, logistic regression, recency alone)
  on the validation cutoff, fitted on the cutoffs whose horizon ends by it; `score` builds the holdout from the whole
  log at its cutoff and scores the chosen model once, beside recency. `build` takes `data_end`; `data_end_of`.

## 0.24.0 (2026-10-14)

Package 20.7: anomaly detection without labels. A new module; no contract changes shape.

- `anomaly` (new): `split` (the rows to review are the latest whole days, a validation period before them; without a
  time column a seeded share); `Features` (numbers robust-scaled, skewed ones logged, and an entity's behaviour without
  labels) and `with_history` (rows scored with the earlier rows as their history, nothing after their own day);
  detectors `rules`, `isolation_forest`, `nearest_neighbours`, `entity`; `plant` (three generic shapes: a value beyond
  its range, a correlated pair broken against its direction, a row's entity counts pushed high); `choose` (each
  detector fitted on the earlier rows and judged on anomalies planted in the validation period, with its stability
  across halves); `review` (the chosen detector ranks the rows to review into a list of the budget's size, measured on
  anomalies planted in a copy of them, with its limits); `evaluate`; `numeric_columns`; `budget_rows`.

## 0.23.0 (2026-10-14)

Package 20.6: the solution workflow from the two cards. A new module and a new contract, `dclab_core.solution_workflow.v1`.

- `solution_workflow(problem_card, data_card, findings)`: the steps of one problem's solution, each with its purpose,
  inputs and outputs (from a closed list of artefacts), the decision it makes (`analyse.AFFECTS`), its check, its rule,
  the findings it answers and the columns it acts on: the safety steps (lock the holdout, analyse the training rows,
  audit leakage, choose on validation, a person's gate, score the holdout once, report the limits), the family's
  steps, the data card's split and label join before the lock, the findings' steps (the leaks in one step naming their
  columns, per-entity features, censoring kept), and a second lock with the analysis read again when the analysis finds
  a split the cards did not name. A family not settled is refused, never guessed.
- `validate(workflow, current)`: on the steps' content: the safety and family steps as their template, blocks never
  going back, the holdout read by the final score alone, after the gate, only the choice on validation choosing, and
  after the leakage audit, nothing but reporting after the holdout is read, every finding answered by a step.
  `propose(current, proposed)`: a model's workflow used only when it validates.
- `analyse`: `F-behaviour-<entity>` names its column in its evidence.

## 0.22.0 (2026-10-14)

Package 20.5: analysis by family. No contract changes shape; `analyse` gains arguments and findings.

- `analyse(..., family, entity, time, treatment, event)`: a family's own findings, all on the training rows. Without
  labels (anomaly, streams, a rare event with no or partial labels): `F-normal` (each column's robust centre and
  spread), `F-rules` (rows beyond 3.5 robust deviations, and whether a few entities hold them), `F-behaviour-<entity>`
  (rows per entity, those far busier) and `F-detectable` (two generic shapes planted in a copy of the training rows: a
  value beyond the column's range, and a pair of columns broken against its direction; what an isolation forest finds
  among as many top rows as were planted). Time to event: `F-censoring` (or why the durations cannot be read) and
  `F-counted-<col>` (columns counted during follow-up). Forecasting: `F-series`, `F-season`. Uplift: `F-arms` (the
  propensity of the arms). Ranking: `F-popularity-<item>`. Segmentation: `F-scales`.
- `AFFECTS`: the decisions a finding may name, each with its decision-ledger point; a finding naming any other raises.
- The label join reads the training rows only (which rows the label file marks is a fact about the target);
  `F-labels-unmatched` (scope `all_rows`) counts the label rows that match no row of the table at all.

## 0.21.0 (2026-10-14)

Package 21.2: the coordinator's state machine. A new module and a new contract, `dclab_core.coordinator_situation.v1`.

- `coordinator` (new): `situation(facts)` walks the states problem → data → links → meet → analysis → workflow → build
  → built and returns the first one not satisfied, its blockers (each a question with its evidence) and the next
  action; a question a file can settle (a file of known cases, and the outcome while the draft holds only cases) waits
  for the data and links states; a file that could not be read stops the walk; from a table's draft, known cases in
  another draft are named; nothing is built while a question to the person is open or a build check refuses.
  `Handoff`, `gate(handoff, data_card)`: a role's hand-off naming a column the tables do not hold is refused whole;
  `investigator_handoff` reads the data investigator's raw answer as one (column names, never its words; an answer
  not in the hand-off's shape is refused as malformed); without a data card nothing is taken on trust.

## 0.20.0 (2026-10-14)

Package 22.3: the promotion gate of a decision model. A new module; no contract changes shape.

- `decision_gate` (new): `decision_gate_v1`, fixed before any sealed run, in the shape of the main product's R3: every
  sealed decision scored (an error counted, never skipped), lower bounds on accuracy (0.90), precision and recall (0.80)
  and a margin over the rule above zero, each from a bootstrap of whole table groups (`bounds`, `margin`); no class of
  30 or more decisions below recall 0.70; at most 2% of the rule's right decisions turned wrong. For a choice, recall is
  the mean over the classes in the truth and precision the mean over the classes answered. `DEMOTION` declares the bars
  a promoted model is demoted by; `verdict` returns the criteria, the bounds and the scores of the model and the rule.

## 0.19.0 (2026-10-14)

Package 22.2: classical decision classifiers. A new module and a new contract, `dclab_core.decision_model_card.v1`.

- `decision_models` (new): `train(point, train_rows, dev_rows, exclude)` chooses a logistic or gradient-boosted
  classifier and its strength by out-of-fold accuracy over folds grouped by table (a tie to the smaller strength, then
  the logistic model; a fold that cannot be fitted counts its rows wrong), refits it on every training row, calibrates
  it (Platt scaling on the same grouped folds) and measures it on the dev rows beside the rule; sealed rows are refused,
  in training and in dev. `metrics` gives accuracy (an abstention counts wrong); for a yes/no point the positive
  answer's precision and recall with Wilson intervals; for a choice, macro means over every class true or answered (a
  class never answered counts precision 0, as scikit-learn's `zero_division=0`) with a bootstrap interval clustered by
  table group, and each class's figures with Wilson intervals. `calibration_error` (expected calibration error over ten
  bins). `ModelCard` (`decision_model_card`): the point, the model's version (a hash of its candidate and its training
  rows), the family and strength, the features and the state fields they came from, the training rows, their
  fingerprint, groups and label sources, each candidate's out-of-fold accuracy, the calibration, the dev score by
  source, the rule's dev score, the calibration error and the limits.

## 0.18.0 (2026-10-14)

Package 22.1: generated decision benchmarks. A new module; no contract changes shape.

- `benchgen` (new): `plant(base, target, seed)` adds seventeen kinds of column to a table: eight positives (a unique
  identifier, a recurring key, a numeric category code, a date in time order, free text, a column written after the
  outcome, a copy, the sum of an exact identity) and nine negatives with the same signatures (so no single band gives
  a label away); `obfuscate(..., seed)` renames every column `f000`… in a shuffled order; `decide` runs the data card's
  rules on the planted columns and returns their `DecisionRecord`s (bands only) with benchmark `DecisionLabel`s from
  `TRUTH`, the record's `dataset` the group a table shares with its hidden-name twin.

## 0.17.0 (2026-10-14)

Package 22.0: one interface for every decision model. A new module; no contract changes shape.

- `answerers` (new): `DecisionAnswerer`, the protocol every decision model answers through (a point, its state in
  the ledger's bands, its options → an `Answer` with a confidence, or None to abstain), with a registry by
  `name@version` and the points each answers; `RecordedRule` (the rule's recorded answer, today's code),
  `ScriptedAnswerer`, `ClassifierAnswerer` (logistic regression on the state's bands and name tokens, abstaining below
  a confidence); `with_answers(record, answerers)`, every answer beside the rule's (a shadow, never used);
  `compare(records, labels, answerers)`, each answerer's coverage, agreement with the rule and accuracy against the
  labels that count, overall and by point. `features(state)`: a state as features, a column's name only by its tokens.

## 0.16.0 (2026-10-14)

Package 20.2: the understanding conversation. A new module; no contract changes shape. `card.WORDS` reads more ways
of saying time to an event ("the time until each customer cancels").

- `conversation` (new): `card_for(problem, said, columns, data, kinds=)`, the card of a conversation (a host passes
  the outcome column's kind from its table); `next_question(card, contradictions, asked)`, the question that unblocks
  the most (a contradiction first, then the files, the labels, the outcome before the kind of answer, the moment);
  `read_answer(field, text, columns)`, an answer read by rules ("it is anomaly detection" states the family whatever
  was asked); `converse(...)` plays a written conversation in at most five questions; `summary(card)` is what the
  person confirms; `READER`, the conversation reader role (pinned prompt `conversation_reader`), kept only behind its
  validator (a value from the field's vocabulary, quoted from the answer, a column of the table), the rules its fallback.

## 0.15.0 (2026-10-14)

Package 20.4: the problem meets the data. A new contract; no other contract changes shape.

- `meet` (new): `check(problem_card, data_card) -> ProblemDataCheck` (`dclab_core.problem_data_check.v1`): every
  mismatch between what the person said and what the tables are, each a `Mismatch` (code, blocking, a question to
  the person, evidence in column names and counts): the outcome named is not a column (`target_absent`) or may never
  be one (`target_refused`); it does not fit the family (`target_family`: blocking when the person stated the family,
  a question when it was read from words); one row is not what is predicted (`unit_grain`); a time is needed and
  absent (`time_absent`); a file of cases is not joined (`labels_not_joined`); the words say nothing is labelled but a
  column looks like the outcome (`labels_contradict`); a check that could not run (`not_checked`, set by a host).
  `ready` when nothing blocks. A unit blocks only when the person stated it, and matches an entity by its noun
  (`customer_id` for "a customer", never `delivery_zone_id` for "a delivery"); `check(..., joined=)` takes the host's
  own label review over the data card's guess. Inserted words are clipped: a check never fails on a long answer.

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
