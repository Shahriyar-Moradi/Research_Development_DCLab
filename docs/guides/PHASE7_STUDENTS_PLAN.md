# Phase 7: a student for every choice

Written 2026-10-17 from the owner's words on the "little students" (the decision models of 22.2, 26.3 and 26.8): *I really
love this idea; let's build it and make it better; I want it in all small processes; this is one of DCLab's important
goals; I do not care how much it costs; build it together and make it better.* Over the repository at `45bc427` with
phase 6 (`PHASE6_MULTI_AGENT_PLAN.md`) in progress: phase 6 is the layer that records the choices; phase 7 is the
student behind each one.

**The idea, in one line:** every small choice DCLab makes is a decision point in the ledger; each point gets a small model
(a student) trained on homework whose answers are known, examined once on a sealed set it never saw, kept in whisper
mode beside the rule, and allowed to decide only where it proved better; every real project is more homework.

---

## Part A. Where it stands today (counted, 2026-10-17)

| | Count |
|---|---|
| Choices DCLab records as decision points (`dclab_core.ledger.POINTS`) | **52** (31 technical, 21 workflow) |
| Of them, recorded somewhere in the product (`CAPTURED`) | 42 |
| Decided by nothing yet (`NOT_DECIDED`: the system does not make the choice, so there is nothing to learn) | 10 |
| With a homework set of known answers (generated sets with a sealed partition) | 9 (6 table, 3 image) |
| With a student (a fitted model and its card) | **13** (6 table, 4 workflow, 3 image) |
| Students that passed their sealed exam (may be switched on, after shadow measurement) | **2** (`column.semantic_role`, `feature.leakage_suspect`) |
| Serving in the product today | 0 (all whisper) |
| Choices with no student | **39** |

The 39 without a student, by where their truth can come from:

| Source of truth | Points | How |
|---|---|---|
| **By construction** (planted in invented tables and problems; the answer is known because we planted it) | `target.column`, `task.framing`, `labels.situation`, `labels.join_key`, `data.cases_table`, `split.strategy`, `split.folds`, `metric.primary`, `data.sampling`, `column.missing_value_action`, `data.cleaning_step`, `features.recipe`, `model.family`, `threshold.operating_point` | benchgen v2: replicas and 26.6b's prior tables with the truth in their facts (which column is the outcome, the task, the label situation, the join key, the moment and so the split, the cost that fixes the threshold, the structure that favours a family) |
| **By measured outcome** (the project's own later numbers say what was right) | `holdout.reuse`, `tuning.accept`, `experiment.accept`, `experiment.next`, `model.review`, `model.promote`, `improve.next_action`, `solution.compose`, `workflow.revisit`, `workflow.stop` | the measured labels of 19.8 extended: the validation and holdout numbers after the choice label it (a kept experiment that did not help; a promotion the holdout contradicted); self-play (27.3) gives thousands of verified solves |
| **By people and by the journeys** (a person's choice, exposed or blind; the written people's scripted truth) | `workflow.move`, `workflow.template`, `workflow.stage_approval`, `map.size`, `map.edit`, `problem.pack`, `agent.tool_choice`, `command.intent_route`, `proposal.completeness`, `report.claim`, `lesson.accept`, `ops.diagnose`, `ops.retrain`, `ops.release`, `ops.rollback` | the journeys suite (labels per step), the tester records (25.11), the blind draws (19.8), the ops loop's own outcomes (23.x); the four workflow students failed for lack of homework (154 to 936 rows): these points need the volume phase 6 and self-play produce |

And beyond the ledger: small processes that are not decision points yet (the structure reader's choice of header and
types, the cleaning pipeline's each step, the pack detector, the synthetic schema, the critic's verdict on a stage, the
retrieval's choice of records, phase 6's send-backs and next-node choices). Each becomes a point first (its state as
bands, its options), then a student.

---

## Part B. The loop, the same for every choice

1. **Record** the choice where it is made: a `DecisionRecord` with its point, the state as bands (never a value), the
   options, who answered (the rule, a model, a classifier, a person) and what was used.
2. **Truth**: a label from one of the three sources (construction, measured outcome, a person), kept apart by source.
3. **Homework**: a set per point with a sealed partition by group (table, project, draft), the same builder for every
   source (`dclab_core.datasets.decisions`).
4. **Train**: a student per point (`dclab_core.decision_models.train`: logistic or boosting chosen on grouped folds,
   calibrated), its card beside the rule's dev score.
5. **Exam**: the sealed set once, under the gate fixed before (`decision_gate_v1`: bounds by group, a margin over the
   rule, no class missed, the rule's right answers kept). A failed exam is a result, kept; the student stays in whisper.
6. **Whisper → flag → act**: the routing switch (22.6), one audited change by a person; promotion needs the sealed pass
   and enough labelled disagreements in shadow; demotion is DCLab's own when it slips.
7. **More homework**: every real project, every self-play solve, every journey, every tester's session adds rows; a
   scheduled retrain rebuilds the cards; the sealed sets are never retrained on.

**The capture rule (the owner's standing rule):** every package below records what it adds; a test reads it back.

---

## Part C. The packages (29.x)

### 29.1 The students' register
- **Delivers:** one page and one command (`make students`, Admin → Students) with a row per point: recorded decisions,
  labels by source, the set (rows, sealed rows), the student (family, dev vs rule), the sealed verdict, the routing
  level, the labelled disagreements still needed to act; the `NOT_DECIDED` reasons shown; `python -m dclab_rnd.students`
  prints the same.
- **Done when:** every one of the 52 points has a row, the numbers equal the files they are read from (a test), and the
  page shows the 2 passed students as "whisper, ready to act after N labels".

### 29.2 Truth by construction for the technical points (benchgen v2)
- **Delivers:** `dclab_core.benchgen` v2: the replicas' and the prior tables' facts become benchmark labels for
  `target.column`, `task.framing`, `labels.situation`, `labels.join_key`, `data.cases_table`, `split.strategy`,
  `split.folds`, `metric.primary`, `data.sampling`, `column.missing_value_action`, `data.cleaning_step`, `features.recipe`,
  `model.family`, `threshold.operating_point`; the rules' decisions on every table; `make decisions-gen ARGS=--v2` writes
  the sets with a sealed partition; the lake reads them as gold.
- **Done when:** each point's set has at least 2,000 decisions over at least 100 tables and both labels; the manifest is
  committed and reproducible (fingerprints); the rules' accuracy per point is in the manifest.

### 29.3 Truth by measured outcome
- **Delivers:** `ledger/labels.py` extended: a measured label for each outcome point from the numbers that came after it
  (the optimism gap and apparent lift as today; a kept experiment's validation against the next; a promotion against the
  holdout; a reuse of the holdout against the rule); self-play's verified solves labelled the same way; the sets built
  from every workspace's ledger under consent.
- **Done when:** the outcome points have sets with both labels from self-play alone, and a planted bad choice (a tuning
  accepted that the holdout contradicts) is labelled wrong by the rule and read into its set.

### 29.4 Train everything, one command
- **Delivers:** `make students ARGS=train`: every point with a set gets a student (or a `SKIPPED.json` with why: one
  class, too few rows, labels from a source that needs fields excluded); the cards committed, the fitted files hashed;
  `ARGS=compare` prints every student beside its rule on dev.
- **Done when:** the command runs end to end on the Mac in under an hour, and the register (29.1) shows every new card.

### 29.5 The exams
- **Delivers:** the sealed gate run once per new set (`make decision-gate` extended to v2 and the images), the verdicts
  in `evidence/campaigns/`, the passed students routed to shadow with their bar; the register shows each verdict.
- **Done when:** every student has a sealed verdict or a reason it has none; nothing serves without one.

### 29.6 Points for the small processes that have none
- **Delivers:** new points (with `CAPTURED` entries, states as bands, the main-product mapping reasons) for: the
  structure reader (header, types), each cleaning step accepted or not, the pack detector, the synthetic schema's shape,
  the critic's verdict, the retrieval's records, phase 6's send-back and next-node choices; each recorded where it is
  made, with its truth source named (construction for the readers and cleaning, people and outcome for the rest).
- **Done when:** the ledger's count grows and every new point appears in the register with recorded decisions from the
  test suite's own runs and self-play.

### 29.7 The loop made continuous
- **Delivers:** a scheduled job (the jobs table) that rebuilds the sets from new labels, retrains the students whose
  homework grew by a margin, re-runs dev comparisons, and writes the register; promotion and demotion through the
  existing switches with an audit row; the Students page shows the history per point.
- **Done when:** a new labelled decision in a test workspace retrains its student on the next run and the register
  shows the new card; a demotion happens when a planted slip crosses the bar.

### 29.8 The people's labels, made easy
- **Delivers:** the Admin Decisions tab's blind draws and corrections (19.8) widened to every point with a student, a
  queue of the disagreements worth a person's look (student against rule, highest confidence first), one click per label;
  testers' labels flow in through `make collect`.
- **Done when:** a person can label 30 disagreements of one point in a few minutes, and the register counts them toward
  the point's "needed to act".

### 29.9 Generation 2: one small language model for every point (26.4, extended)
- **Delivers:** the export for 26.4 covers every point with a set (the typed questions of 22.5), the recipe unchanged;
  the comparison on dev (`decision_model_compare`) per point against the students and the rules. The training run is
  the owner's (GPU).
- **Done when:** the export's manifest lists every point with its counts, and a run's adapter can be compared per point.

### 29.10 Every user is a teacher: disagreements shown where the work happens
- **Delivers:** on a project's pages and on Home, where a student whispered something other than the rule, a small note:
  "the rule said X, a student thinks Y; which is right?", one click to answer; the answer is a label (`user_decision_exposed`
  when the person saw both, as 19.8 counts it), never a change of what was decided; the person can turn the notes off.
- **Done when:** a disagreement on a replica project shows the note, the click writes the label, the register counts it,
  and nothing about the project's run changes.

### 29.11 Students ask for the labels they need most (active learning)
- **Delivers:** the label queue (29.8, 29.10) ranked by what a label would teach: disagreements first, then the student's
  least sure answers, then points furthest from their MIN_LABELS; a student's needed labels shrink faster.
- **Done when:** on the generated sets, a student trained with the queue's first 200 labels beats one trained with 200 drawn
  at random (dev, the same seeds), recorded as a measurement.

### 29.12 Harder homework where students fail (a curriculum)
- **Delivers:** the generators (benchgen, the image shapes, the priors) told which kinds of table each student gets wrong
  on dev, and make more of those (more look-alike scenes, subtler leaks, names in other languages); the sealed sets never
  change; each round's sets are a new version with its manifest.
- **Done when:** two rounds of curriculum on one point raise its dev accuracy with the sealed set untouched, and the rounds
  are in the register's history.

### 29.13 A student explains itself
- **Delivers:** each answer carries the two or three state fields that moved it most (the coefficients or the trees'
  contributions on the bands), shown beside the whisper and kept in the ledger; a person judges the reason, not only the
  answer.
- **Done when:** the leakage student's whisper on a planted leak names the timing band among its reasons, and the reasons are
  in the record.

### 29.14 Report cards over time
- **Delivers:** the register's history (`students/register.jsonl`) as a report card per student on the Students tab: dev
  score, labels, exam, level, by date; a student that slips is shown before demotion fires.
- **Done when:** two register runs with a retrain between them draw the change on the page.

### 29.15 Real homework from public data
- **Delivers:** OpenML's tables (already in the lake, 26.2) give truths for `target.column`, `task.framing` and
  `column.is_identifier` on real columns (the dataset's own declared target and ID columns); Kaggle (with the owner's key,
  27.4) adds human choices from public notebooks (dropped columns, the split used). Real-table sets get their own sealed
  partition.
- **Done when:** at least three points have a real-table set beside the invented one, and each student is scored on both.

### 29.16 A committee: the rule, the student, the other models
- **Delivers:** a point may be answered by a vote of the rule, its student, GLiNER (26.9) and a model role, each weighted by
  its measured accuracy on that point; the committee is one more answerer in the register, examined like any student.
  Claude's answers vote only in evaluation, never as homework (the owner's rule).
- **Done when:** the committee has a sealed verdict on at least the six table points, beside each member's.

### 29.17 Claude and GLiNER beside every student (the owner's $30, 2026-10-17)
- **The owner's decisions:** $30 "to use Claude to enhance the result, enhance everything"; Claude's answers are evaluation by
  default and, on real projects only (truth unknown), silver labels: lower weight, tagged, kept apart, removable; GLiNER
  (instead of Jev) joins as an examinee and committee member and its answers may train the students (silver, weighted by its
  measured accuracy per point) and evaluate; GLiNER itself is not fine-tuned.
- **Delivers:** `python -m dclab_rnd.students_exam` (plan printed; `--yes --cap-eur`): the dev decisions of every point with a
  generated set, the student's disagreements with the rule first, answered by the rule, the student (batched), Claude through
  `decision_answer` (six at a time, the gateway's cap) and GLiNER (its captured answers, read); each scored on the decisions all
  answered, and on the disagreements who was right; Claude's answers kept whole with their reasons
  (`lake/bronze/answers/claude/lake/<point>/dev.jsonl`, `use: eval_only`, never a label on homework); the result
  `evidence/campaigns/students_exam_v1/results/SEX-NNN.json`; the run's requests in `agent_runs/live_tests/2026-10-17_students_exam/`.
- **Status: run (2026-10-17), EUR 3.00 of the owner's $30:** SEX-001 (1,800 decisions) and SEX-002 (rescored, no request). The
  students lead on the points they were trained for (role 0.91, leakage 0.97, category code 0.85, time 0.90 on the hard-first sample);
  the rules are perfect on identifiers and groups; Claude is close on category codes (recall 1.0) and catches the group student's
  mistakes (9 of 9), but misses leaks seen only as bands (7 of 57 on the student's disagreements); GLiNER is behind on these points.
- **Next in the same package:** Claude as the judge of each disagreement with its reason, measured against the planted truth;
  the image and workflow points (their states through a purpose that may see them); the silver path for real projects; GLiNER
  and Claude as committee members (29.16).

---

## Part C2. What the owner can do (the levers only a person has)

1. **Label disagreements, blind, a few minutes a day** (Admin → Decisions). The two students that passed need 20 labelled
   disagreements each before they may act; nothing else moves a student from whisper to work as fast.
2. **Bring real problems.** Every real project solved in DCLab is homework with real answers; a few of the owner's own
   startup tables teach more than thousands of invented ones.
3. **Invite testers** (`make tester-setup`, 25.11): each tester's sessions and labels come back with `make collect`.
4. **Approve promotions.** When the register says "may act", one audited switch moves a student to flag, then to act.
5. **Say which choices matter most.** The order of 29.2 and 29.15 follows the choices whose mistakes cost the most (the
   leakage, the outcome column, the split come first unless the owner says otherwise).
6. **The Kaggle key and a GPU run** (27.4, 26.4): real human choices as homework, and the one model that answers every point.

---

## Part D. How phase 6 and phase 7 feed each other

The orchestration layer (28.x) is where the choices of the understanding and solution teams are made; each node's
choice is a point (29.6) and each run is homework (28.9). The students (29.x) are the answerers a node asks beside the
rule (28.10); where one has passed and measured, the node serves it. Neither phase waits for the other: 28.2 and 29.1 can
start side by side.

## Part E. Cost and the owner's rule

The students train on the Mac (minutes each; the whole register in under an hour, 29.4); the sets are generated locally;
self-play is free. Live model spend is only in the arms that compare a model against the students (29.9's dev comparison,
phase 6's measurements): planned, printed and capped before each run, reported after. The owner said cost is not the
constraint; the caps stay as a habit, set generously.

## Status

- **Design (2026-10-17):** written from the owner's words; widened the same day (29.10 to 29.16, Part C2) after the owner asked how to
  involve the idea more.
- **29.1 built (2026-10-17), the review's 5 findings fixed:** `dclab_rnd/students.py` (`read` shared by the command and the page,
  exams matched by `name@version`, workflow counts on their labelled rows, warnings instead of unmeasured counts, each run kept in
  `<home>/students/register.jsonl`), `GET /api/students` (owner), Admin → Students, `make students`. Today: 52 points, 13 with
  homework, 13 students, 2 passed, none serving.
