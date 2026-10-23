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
