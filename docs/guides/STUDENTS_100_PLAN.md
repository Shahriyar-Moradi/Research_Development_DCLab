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
