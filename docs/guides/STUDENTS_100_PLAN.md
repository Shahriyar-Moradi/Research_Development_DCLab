# 100 students: gather the data, then the students (plan, 2026-10-23)

The owner, 2026-10-23: "let's gather and collect for 100 students", "we should make an agent to collect these data
more", and test the students on real tables in shadow, recorded and stored. This plan builds on
`PHASE7_STUDENTS_PLAN.md` (29.x: the register, the loop, the people's labels, active learning, the curriculum,
generation 2), all built. It does not replace it: it widens it from 17 students to 100, data first.

## Where it stands (counted 2026-10-23)

- **66 decision points** are defined (`dclab_core.ledger.POINTS`): 41 technical, 25 about the workflow.
- **17 have a student** (a small classifier per point: logistic regression or gradient-boosted trees, chosen on folds
  grouped by table; `dclab_core.decision_models`).
- **Sealed exams:** 2 of 13 passed (`column.semantic_role`, `feature.leakage_suspect`; `evidence/campaigns/decision_models_v1`).
  Every student only whispers (shadow, level 0).
- **Real tables:** 1,122 OpenML tables give real truths for 3 points (`real_v1`). There, a column-type student trained
  on real homework scores 99.6% on dev against the rule's 82.2%.
- **The owner's own ledger:** 294 real decisions, 3 labels; the students disagree with the rule on about 23.
- **Today's real-table shadow run** (`python -m dclab_rnd.real_shadow run|measure|silver`): DCLab's product path on
  every real table, the students whispering beside the rule, recorded in their own workspace
  (`agent_runs/live_tests/2026-10-23/real-shadow/`, consent "train"), Claude's silver opinion on a sample of the
  disagreements, kept apart.

## First results of the real-table shadow run (2026-10-23)

652 real OpenML tables (107 dev, measured only; 545 train, recorded as homework; the sealed ones left out), 50,967
decisions, none failed; Claude's silver opinion on 289 disagreements, EUR 0.64. Records in
`agent_runs/live_tests/2026-10-23-real-shadow/` (`eval/measure.json`, `eval/compare.json`, `eval/silver.jsonl`).

**The students trained only on invented tables are not better than the rules on real tables.** Where a student and the
rule disagree, Claude sides with the rule on leakage 95 to 5 (the leakage student, which passed its sealed exam on
invented tables, flags 709 of 2,642 real columns the rule keeps), on the time column 23 to 3, on the task 20 to 2, on
identifiers 25 to 14, and on roles 59 to 38.

**Real homework fixes that where real truths exist.** On the 2,638 real dev columns, against their declared truth:

| Point | Rule | Student, invented only | Student, invented and real |
|---|---|---|---|
| `column.semantic_role` | 93.6% | 94.0% | **99.7%** |
| `column.is_identifier` | **100%** | 98.9% | 99.2% |

A fresh retrain (`models_real_v2`) reproduces `models_real_v1`'s cards (real dev 99.6%, 99.2%, task 98.1%). The lesson for
every point: invented homework teaches the method, real homework teaches the skill. Leakage, time and group have no
real truth yet: their real labels must come from people (S5.1), measured outcomes (S3) or, as evaluation only, Claude.

## The role student's sealed exam on real tables, fixed before it is read (2026-10-25)

- **Student:** `column.semantic_role` from `models_real_v1` (invented and real homework, 2026-10-19), unchanged.
- **Exam:** `real_v1`'s sealed partition for `column.semantic_role` (4,585 decisions; its fingerprint committed on
  2026-10-19), read once with `python -m dclab_rnd.decision_gate_run --set real --points column.semantic_role --yes`.
- **Gate:** `decision_gate_v1`, its thresholds unchanged.
- **Scope, decided on the dev rows before the read:** columns whose truth is `identifier` are left out. Why: real tables
  declare almost no row identifiers (6 of 3,744 dev columns), the student never learned them (0 of 6), and DCLab asks
  `column.is_identifier` as its own question first. So this student is judged as the role of a column that is not an
  identifier. On the dev rows, under this scope: student 99.8%, rule 82.2%, macro precision 0.969, macro recall 0.992.
  The scope and its reason are kept in the result record. A pass lets the owner switch it on for non-identifier
  columns only; the identifier question stays with its own point.

**Result (DMG-014, 2026-10-25): fail.** On 4,581 sealed real decisions: accuracy 99.2% (lower bound 98.6%), the
rule 83.8%, a margin over it of at least 10.5 points, recall 97.3% (95.3%), no class missed, the rule's right answers
kept; **precision 79.85% (lower bound 79.53%), under the 80% bar.** The gate's precision is the mean over the classes
the student *answered*: it still answered `identifier` on a few non-identifier columns, a class with no truth in this
scope, so precision 0 there pulled the mean of five classes to about 0.8. The dev check before the read averaged over
the true classes only and missed it: the check, not the gate, was wrong. The sealed set is spent for this point. Next:
the student answers only the non-identifier options when it is asked inside this scope, and a fresh sealed set comes
from new real tables (the collector, S2).

## The rule this plan keeps

**A student is worth adding only where it has labelled examples and a rule to beat.** One hundred students without data
are one hundred guessers. So every package below gathers data first; a point's student is trained only when its
homework reaches the bar (a train set of at least 300 decisions from at least 30 groups, a dev set of at least 100), and
it stays in shadow until it passes its sealed exam and a person switches it on.

## Where each point's truth comes from

| Source | What it is | Points it serves | Its limits |
|---|---|---|---|
| **A. Construction** | Invented tables and pictures with planted traps (benchgen v1/v2, image sets) | Column and table checks, cleaning, splits, modifiers | Invented: proves the method, not real skill |
| **B. Real declarations** | Public tables that say their own truth (OpenML: row identifier, types, outcome) | Identifier, column role, task, outcome column, metric | The uploader's word, not checked |
| **C. Measured outcome** | The holdout or the folds say afterwards whether a choice helped | Experiment and tuning accepts, feature recipe, model family, threshold, promote | Needs many project runs |
| **D. Written journeys** | Conversations written with their right steps (journeys, the teams' walks) | Workflow points: next question, ready, team node, stop, move | Written by us |
| **E. People** | The "who's right?" clicks (29.10), the blind queue (29.8), testers' work (`make collect`) | Every point a person sees | Few, slow, the most valuable |
| **F. Silver** | Claude's opinion on real disagreements (evaluation, kept apart, capped) | Any point with a state | Never a label a promotion reads (the owner's rule, 2026-10-17) |

## The packages

### S1. The real-table shadow run (built and running today)
- **Delivers:** `dclab_rnd/real_shadow.py`; the run on all real tables; `measure.json` (agreement, disagreements, and
  accuracy where the dev rows declare a truth); `silver.jsonl` (Claude on up to 60 disagreements per point, capped).
- **Done when:** every table done or failed with its reason; the measurement and the silver opinions stored; the
  disagreements in the people's queue. **Capture:** the workspace's ledger (consent train), kept by the archive.

### S2. The collector agent ("an agent to collect these data more")
- **What:** a background agent (a job of kind `collect`, the worker's sweep, nightly by default) that, each run:
  1. reads the register and picks the points furthest from the bar (fewest labelled decisions, widest intervals);
  2. finds material for them: more real public tables (OpenML first; Kaggle when the owner's key is set and the licence
     allows), new invented tables for construction points (benchgen v2 generators), self-play projects (014ccf88);
  3. runs DCLab's own path on them in shadow (as S1) and records everything;
  4. queues the disagreements for people (E) and asks Claude for silver on a capped sample (F);
  5. writes a short report: what it gathered per point, what it spent, what is still missing.
- **Guards:** a euro cap per run and per month; the 80% resource rule; the store's consent; never a sealed set.
- **Done when:** two nightly runs grow the three neediest points' homework, inside the cap, each run reported.
- **Coordinate with:** the learning agent being built in another session (`dclab_rnd/learning`), so the two share one
  queue and one budget instead of competing.
- **Built (2026-10-25), not yet run live:** `dclab_rnd/collector.py`, `python -m dclab_rnd.collector plan|run|status|exam`;
  on when `DCLAB_COLLECTOR=1`; a run starts at night (00:00 to 07:00), with 3 GB free and the Mac under 80% busy, one at a
  time; caps EUR 0.5 a run and EUR 10 a month (`DCLAB_COLLECTOR_RUN_EUR`, `DCLAB_COLLECTOR_MONTHLY_EUR`), the run's cap
  counted before Claude is asked. OpenML only: a copy of a known table (its columns' names) is set aside, so no copy of a
  homework table sits in an exam. `real_v2`, the role student's second exam, is built from the collector's own tables
  once 50 of them are sealed, then frozen. Still open: picking the neediest points (OpenML tables feed every column point
  at once), benchgen and self-play as sources, one shared budget with the learning agent.

### S3. Fill the 49 points that have no student (data first)
Grouped by truth source, the work for each group:
- **Construction (A), about 16 points:** `column.missing_value_action`, `data.cleaning_keep`, `data.sampling`,
  `data.structure_format`, `data.synthetic_shape`, `labels.join_key`, `labels.situation`, `target.column`,
  `split.folds`, `metric.primary`, `holdout.reuse`, the five `modifier.*`: extend benchgen v2 to plant each one and
  record the rule's answer and the truth.
- **Measured outcome (C), about 10 points:** `experiment.accept`, `experiment.next`, `tuning.accept`, `features.recipe`,
  `model.family`, `improve.next_action`, `model.promote`, `model.review`, `threshold.operating_point`, `lesson.accept`:
  label from what the folds and the holdout said afterwards (29.3's measured labels), gathered by self-play runs.
- **Journeys and walks (D), about 15 points:** `agent.tool_choice` (52 real decisions already), `workflow.move`, `.stop`,
  `.stage_approval`, `.revisit`, `.template`, `.g4`, `team.next_node`, `map.edit`, `map.size`, `command.intent_route`,
  `critic.send_back`, `department.proposal`, `solution.compose`, `proposal.completeness`: label from the journeys' right
  steps and the written walks.
- **People and operations (E), about 8 points:** `ops.*`, `report.claim`, `evidence.records`, `problem.pack`,
  `data.fit_verdict`: gathered from people and testers; trained last.

**Measured before building (2026-10-25): most of the 16 "plantable" points do not need a new student.** A student is
worth adding only where it has a rule to beat and evidence the rule does not already use. On 120 planted tables
(each also with its names hidden):

| Point | Today's rule on planted tables | Verdict |
|---|---|---|
| `modifier.groups` | right 61% (50% with names hidden) | weak rule, but the split student already holds the skill (77% against the rule's 50% on dev): map its answer onto this point, no new student |
| `modifier.time_order` | right 72% (a birth date read as the row's time) | the same: the split student's answer |
| `data.cleaning_keep` | right 55% | the cleaning-step student holds it (92% against 73%): map its answer |
| `data.structure_format`, `labels.join_key`, `modifier.noisy_labels`, `modifier.review_capacity` | not measured: new generators needed | their states are what the rule reads; measure the rule on planted files first, a student only where it errs |
| `column.missing_value_action`, `labels.situation` | — | no student: DCLab does not make this choice today (`ledger.NOT_DECIDED`) |
| `target.column` | — | no student: no rule by design (DCLab never guesses the outcome; a key once became one) |
| `split.folds`, `data.sampling`, `metric.primary` | — | settings or a kind's default: nothing to learn |
| `holdout.reuse` | — | a rule of method: never a student's to decide |
| `modifier.rare_positives` | — | the rule is the definition (under 5%): a student could only copy it |
| `data.synthetic_shape` | — | its truth is the people's (whether the draft goes on) |

The measured group is closer than the plan thought: the 1,114 self-play runs (`lake/bronze/selfplay`) already hold
measured labels for `tuning.accept` (149, from a fresh draw: a real outcome, not the rule's own answer), `model.family`
(258) and `features.recipe` (103; both "clearly best on validation", close to the rule's own choice). All are under the
bar. `tuning.accept` is the one worth growing: about 13% of its decisions get a label, so about 2,000 more self-play runs
(about 1 GB) would bring it to 300 training and 100 dev labels.

The planted traps also show the cleaner changing real data in the product: every country code `NA` (Namibia) becomes
missing, and versions `1.10` become the number 1.1. Both were planted on purpose in 29.2, and a student only whispers,
so the product keeps changing the data. "NA" is truly ambiguous (R writes missing values as `NA`), so the owner decides
whether the cleaner keeps it as written in a column of two-letter codes and says so.

**The owner's answers (2026-10-25) and what followed:**

- **"NA" kept in a column of two-letter codes** (`dclab_rnd/studio/na_codes.py`): in every reader of a person's table
  (upload, project table, cleaner, exported code and notebook, the scoring service), said in the cleaning log. Versions
  like 1.10 are still read as numbers (not asked).
- **Part 1, mapped students** (`dclab_rnd/ledger/mapped.py`): the split student answers `modifier.groups` and
  `modifier.time_order`, the cleaning-step student answers `data.cleaning_keep`. Their homework and exam is
  `generated_v3`: seeds 10,000 and up (never drawn for generated_v2, where the base students trained), states written by
  production's own capture. Each is judged once on its sealed rows by `decision_gate_v1`, unchanged
  (`python -m dclab_rnd.decision_gate_run --set v3 --points modifier.groups --yes`: one point, its sealed set alone).
  **On dev (2,236 and 550 decisions; the sealed rows unread), checked with the gate's own function before any read:**
  `modifier.groups` 100% (lower bound 99.3%) against the rule's 64.6%, the gate passed; `modifier.time_order` 73.1%,
  exactly the rule (a birth date fools both: the state shows nothing about a date column's span or order), the gate
  failed on its margin; `data.cleaning_keep` 86.9% against 74.6%, the gate failed (accuracy's lower bound 85.4% under
  90%, and 17.6% of the rule's right answers turned wrong, the gate allows 2%). **The exam's scope, fixed here before
  the read:** only `modifier.groups` sits its sealed exam; the other two sealed sets stay unread, kept for a student
  with the evidence they lack, since reading them now would spend them on a known fail.
  **The review then stopped the groups exam before its read (2026-10-25, morning).** The 100% was not skill:
  benchgen v2's split tables always hold exactly three features and the outcome, so any further column that is neither a
  date nor an id *is* the planted entity, and the split student had learned to count columns (one formula on the
  columns fits every v2 and v3 row; adding one plain number to a table without an entity flips its answer). The split
  student's own 77% on v2 measured the same layout. Fixed in generated_v3: 0 to 5 plain columns drawn apart from the
  truth (`decisions_gen.with_plain_columns`), and the tables with both a row time and an entity left out of the groups
  set, since there the entity recurs over the period and the product's rule asks the person (no truth in the data).
  generated_v3 is rebuilt (its sealed rows were never read) and dev measured again before any exam. The other v2 sets
  have the same kind of risk (the cases tables' column counts differ by variant): every v2 student failed its sealed
  exam, so no serving student rests on them, but any new student needs homework where a column count says nothing.
  **On the rebuilt generated_v3 (dev, sealed unread):**

  | Point | Mapped student | Student trained on v3's homework | Rule | Gate on dev |
  |---|---|---|---|---|
  | `modifier.groups` | 42.8% (it answers yes wherever a column is added) | 82.6% | 80.9% | fail (under 90%) |
  | `modifier.time_order` | 73.1% | 73.1% | 73.1% | fail (no margin) |
  | `data.cleaning_keep` | 86.9% | 100% | 74.6% | the trained one passes, on a shortcut |

  The trained keep student's 100% is the dirty tables' own layout again: every "numbers" step they make wrong is either
  on a column of few values (a version) or on unique plain digits (an identifier), and they never plant unique plain
  integers written as text that *are* numbers, so "never convert unique digits" scores perfectly here and would be wrong
  on real tables. **No student sits an exam; all three sealed sets stay unread, and the mapped students are not loaded
  in shadow** (failing students would fill the people's queue with false disagreements). The next step for planted
  homework is a generator package in `dclab_core` (benchgen v3): layouts drawn apart from every truth and, for each
  trap, its counter-example (plain numbers as text, an entity with an id-like name that never repeats, a repeating
  category that is no entity), with a test that the state alone cannot name the truth by a simple formula.
- **Part 2, the rule measured on planted files, no student needed for any of the four:**

| Point | Today's rule | Verdict |
|---|---|---|
| `data.structure_format` | the right table from 11 of 12 tricky files (a tab file named .csv, Excel named .csv, parquet named .dat, semicolons with decimal commas) | no student; the one miss, a title line above the header, was a reader bug: fixed (12 of 12) |
| `labels.join_key` | right on clean files, a shared row number when the real key finds every label, an entity key (refused) | no student; a running number under a name the rule does not know ("line", "S.No", "ردیف") wins when a tenth of the labelled cases are missing from the export and marks the first rows as cases: a fix in `dclab_core.analyse` (both files' columns a running number of their own file), queued after the other session's 0.109.0; zero-padded text ids against integers find no key (said to the person) |
| `modifier.noisy_labels` | knows 13 source words (rule, cluster, expert, audit...) | no student: today's state holds nothing about a source column; "manual", "auto", "analyst" are missed, a vocabulary gap |
| `modifier.review_capacity` | asks the person on every screening problem | no student: its truth is what the person says about the team |

- **`tuning.accept`'s state** now holds what judges a gain on the folds (its size in units of the folds' spread, how
  many candidates were tried, the baseline's fold mean and spread, the family, the metric), so the self-play runs grown
  for it carry evidence a student can use beyond the rule's 0.001 margin.

### S4. Find the 34 new points (to reach 100)
- **First an inventory, not invention:** read where DCLab makes a choice today that no point records (the engine's
  encoders and imputers, class weights, calibration, feature selection, text vectorising, date parts, imbalance handling,
  stopping rules, ensemble choice, the report's sections, the notebook coder's cell kind, the Reporter's audience, the
  Data engineer's change checks, the "several tasks" splitter). Each found choice becomes a point with its question,
  options, state fields (never a value) and its capture line.
- **Done when:** 34 new points are recorded where they are made, each with a truth source from the table above.

### S5. Better homework (the owner's five, in order)
1. **Real labels:** the people's queue first; testers (Ali, Mehdi) through `make share` / `make collect`.
2. **Real tables:** S1 and S2; train each point's student on invented and real homework together, as `models_real_v1` did.
3. **Harder homework where students fail** (29.12's curriculum), confirmed on a fresh draw.
4. **Balance:** a cap per point and per class in every export (for example 3,000 per point), the tiny points oversampled.
5. **Generation 2** (one small language model for all points), after the data is balanced: a pilot on the owner's
   M1 Max with MLX (free), its own short prompt matching its answers, then a fair sealed exam.

### S7. More ability: the expert's choices, not only its traps (the owner, 2026-10-25)

The owner: "we are very focused on leakage; we want more ability; the student model must have more ability". The 17
students judge traps (leakage, identifiers, cleaning, splits), because traps can be planted. An ML expert's main skill is
different: looking at a table and knowing what will work on it before trying everything.

**Measured first (2026-10-25):** self-play's 1,116 model screens are a weak teacher for that: "always the generator's
usual winner" is right 70% of the time (logistic regression wins 99 of the 100 binary replicas, "naive" all 100
forecasting ones). A student would learn the generator, not the table. The honest homework is real tables, the truth
measured.

**The point, and why it is new:** DCLab screens every model family and keeps the measured best, so "which family wins"
has no rule to beat (the rule is the measurement). The ability is *before* the screen: `model.first`, "which family is
tried first?", today a fixed order (the kind's list). Truth: the family that then measures best on the training folds.
Value: the likely winners screened first, so a run is shorter, a larger search fits its budget, and the agentic solver
gets a prior. The state is the table's description in bands and counts (rows, columns by type, high-cardinality and
missing columns, the task, the class balance), never a value.

| Package | Delivers | Done when |
|---|---|---|
| G1 benchgen v3 (`dclab_core`) | craft tables whose winner varies by a planted structure (linear or interacting signal, a high-cardinality category, a seasonal date, a skewed outcome, informative missing values, noise), layouts drawn apart from every truth; the trap generators rebuilt with each trap's counter-example | a test: no formula on the layout names a truth; the winners vary within every structure |
| G2 screens on real tables | `model.first` and the state fields in the ledger; DCLab's engine on the OpenML tables (and the collector's), every family's score and the table's state recorded; resumable, under the 80% rule | train and dev screens from at least 300 real tables; sealed tables screened apart |
| G3 the student | a classifier for `model.first` from real screens (and craft ones), measured against the fixed order and "always the commonest winner" on dev | its sealed exam on real tables only, by `decision_gate_v1` unchanged, read once |
| G4 use | the engine records the student's guess beside the screen (shadow); later it screens the guessed top families first | the owner's switch after a passed exam |

**The review of S7 (2026-10-25), before any screen was used:** the acceptance harness runs the engine under its fast
*test* profile (ensembles cut to ten trees), so the first screens' winners were the test profile's, not the product's
(on one skewed table LightGBM's error was 347 there and 69 in a full run). Real screens now run the product's own
settings and say so (`profile: full`); the three written before are kept apart. Self-play's screens run the fast
profile too, so "logistic regression wins 99 of 100 binary replicas" is partly the ensembles' handicap. Also fixed: the
table at a glance now carries its task and class balance; craft tables carry no-effect decoys (a big category, a date),
so neither names a structure; the student's homework label is the screen's best mean (a clear winner is reported
apart); the baselines come from real training screens only; and two arms are reported, real only and real with craft.

**Screening real tables is also a robustness test of the product:** DCLab's engine crashed on two of the first three
(a column with no value, classes of one row); both fixed (cb93b9a9). Every failure is kept with its reason.

### S8. Decide, data and learn: is the rule's answer true on real tables? (the owner's map, layers 5, 2 and 6)

The owner's six-layer map (understand, data, build, trust, decide, learn) asks for ability beyond leakage. For three
layers DCLab already answers with a fixed rule, each built on replicas or simulated rows. Before a student is built for
any of them, each rule's answer is checked against what really happened on real OpenML tables of the train split (dev
and sealed tables untouched). Each plan was written before any result, in `agent_runs/live_tests/2026-10-25-*/PLAN.md`:

| Layer | The rule | The truth on real tables | What it decides |
|---|---|---|---|
| 5 decide | `threshold.operating_point`: the best point of the out-of-fold value curve (`decide.choose_operating_point`) | six ways of choosing (argmax, bagged, the top share of new rows by either, the cost-ratio cut with and without calibration), each acted on unseen rows, 8 cost statements per table, 60 two-class tables | a rule change if one way is best or within noise on 80% of the statements; a student only if a picker on the table's state beats the best single way by 2 points of attainable value |
| 2 data | `data.more_rows` (34.13): the learning curve's answer | the answer on half the rows against training on twice the rows, scored on unseen rows; 30 two-class and 30 regression tables | the rule holds at 80% for both "likely" and "unlikely"; needs work below 70% |
| 6 learn | `ops.retrain` (34.11): promote a challenger only when it wins on the latest window beyond its bound | three replays per table in time order; each decision judged on the rows after its window; 14 real sources | holds if at most 1 promotion in 20 is worse later; needs work if over 1 in 10, or over half of the kept champions had a better challenger |

**Found by the dry run, before any real table:** a threshold chosen on out-of-fold scores, applied to the scores of the
model refitted on all training rows (as the engine does), acted on 16.8% of new rows where the capacity allowed 20%:
the refitted model scores the top a little lower. Layer 5 therefore also measures "act on the chosen share".

**Layer 1 (understand), with the owner's $15 (2026-10-25):**
- **Homework:** 589 written people over the 15 kinds, written by Claude through the gateway (EUR 2.21), each person's
  kind and family set by the code (`dclab_rnd/people_writer.py`); a sealed set of 150, written by a separate agent.
- **Too easy, before:** from the first words alone the rules read the right family for 73% of the older people and 14%
  of the new ones; in the plays the rule's accuracy on the new people is 0.71 (family) and 0.71 (ready), against 0.95
  and 0.96 on the older ones.
- **A student that reads the words** (`dclab_rnd/understand_students.py`): before the fix below, the text student for
  `problem.family` scored 0.967 on dev against the rule's 0.825; the state-only student 0.50 (the state holds no words).
  It is retrained on plays of the fixed product before its one sealed exam.
- **Live with Claude** (45 dev people, EUR 3.63): the rules right on 11, Claude as the Home agent on 14. Fixed: a
  person's "I don't know" was recorded as their answer (46 moments, 33 outcomes, 31 families; 4 projects built with
  it); a table with nothing left to learn from crashed the model.
- **The largest gap found, the next package (S9):** the understood family does not reach the kind that is built. A
  forecast the card read right ran as a regression; most of the 30 people wrong in both arms are kinds whose Solution
  page needs fields (a forecast's series and horizon, a ranking's user and item, a segmentation's outcome) that the
  older written people typed themselves. S9: the Solution page proposes the kind and its fields from the card's family
  and the table, each kind's own way, and the same people are played again.

### S6. Fair exams for every student
- A fresh sealed set (v2) per point, drawn once and never read until a promotion; a gate runner that accepts any
  answerer (classifier, language model, rule) so generation 2 is judged like the students.
- **Done when:** a student, the rule and a language model are judged once on the same sealed rows under the gate fixed
  before the run.

## Order

| Order | Package | Needs | Cost |
|---|---|---|---|
| 1 | S1 real-table shadow run | running today | CPU; Claude silver capped at €3 |
| 2 | S5.1 label the waiting disagreements | the owner's 10 minutes | none |
| 3 | S2 collector agent | S1; coordinate with the learning agent | CPU; a monthly cap the owner sets |
| 4 | S3 construction group, then journeys group | S2 | CPU |
| 5 | S4 inventory of the new points | — | reading the code |
| 6 | S6 fair exams | S3 | CPU |
| 7 | S5.5 generation 2 pilot | balanced export | free on the M1 Max |
| 8 | S3 measured-outcome group | many self-play runs | CPU |

## Decisions for the owner

1. **The collector's monthly cap** for Claude's silver opinions (suggested: €10 a month to start).
2. **Kaggle:** add your key if the collector should use Kaggle tables (only those whose licences allow it).
3. **Who labels:** you and testers for now; a paid labeller later for public and invented tables.
4. **Promotion:** a student acts only after its sealed pass and your switch; keep this rule.
