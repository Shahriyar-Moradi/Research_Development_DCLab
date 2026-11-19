# hp_gate_v2: held-out tables

**Written by:** an independent Claude Code session (Claude Opus 5.5), started by the developer's session on 2026-11-19,
at the owner's decision of 2026-11-19 that the planning exam's held-out material is written by someone who has not seen
the planning code. No model was run to write it, and no exam or gate was run.

**What was not read while writing it:** anything under `dclab_rnd/tasks/`, `dclab_core/task_catalog.py`,
`dclab_rnd/draft/`, `dclab_rnd/prompts/`, `dclab_rnd/agent_eval/plan_enforced/`; any result under `evidence/`; the
files `cases_dev3.json`, `answer_key_dev3.json`, `cases_nodata_held.json`, `answer_key_nodata_held.json`,
`cases_sealed.json`, and every file with "sealed" in its name; `.env`. Only the first case of
`dclab_rnd/agent_eval/task_planning_v4/cases_dev.json` was read, to learn the shape of a case.

**What is here:**

- `cases.json`: eight cases (schema `dclab_rnd.planning_gate_v2.held.v1`, split `held`, two seeds). Each case has the keys of
  a `task_planning_v4` case, plus `enforceable_traps`: the traps code can check in a built project
  (`late_column`: kept out of the inputs; `group_split`: the holdout shares no value of the column with training;
  `time_split`: the holdout is the latest rows). `replica` names the case's generator in `tables.py`.
- `tables.py`: `make_<case_id>(seed) -> {file name: DataFrame}`, numpy and pandas only, every table invented. The file
  names and columns are the ones `cases.json` lists. The main table of each case has 1,000 to 4,000 rows; a small
  lookup table (one case's machine list) has fewer.
- `cases.sha256`: the SHA-256 of `cases.json` and `tables.py`. A changed file is a new set.

**How the traps are planted:** a late column is computed from the outcome after it happened; a group column repeats
and its rows share a hidden effect on the outcome; a time column orders the rows and the outcome rate moves with it.
Several late columns have neutral names, and some honest columns have frightening names but are known at the
prediction moment, so they must stay in. One case asks for both a group split and a time split: its holdout is the
latest days of homes never seen in training.

**The rule:** only the exam run reads these files, once. Nobody who changes the planning code reads them, and no
result of a case is shown case by case to the developer before the run is over. Changing a case after the run makes a
new set, not a fix.
