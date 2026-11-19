# hp_gate_v3: held-out tables

**Written by:** an independent Claude Code session (Claude Opus 5.5), started by the developer's session on 2026-11-19,
at the owner's decision that each planning exam's held-out material is written by someone who has not seen the
planning code. No model was run to write it, and no exam or gate was run.

**What was not read while writing it:** anything under `dclab_rnd/tasks/`, `dclab_rnd/intern/`, `dclab_rnd/studio/`,
`dclab_rnd/expansion/`; `dclab_core/split_hints.py`, `dclab_core/task_catalog.py`, `dclab_core/leak_evidence.py`; the
practice files of `planning_gate_v2`; any result under `evidence/campaigns/planning_gate_v2/`; the day's commits; every
file with "sealed", "dev3" or "nodata_held" in its name; `.env`. Read for format only: v2's held folder (spent), the
fields `planning_gate_v2/run.py` reads from a case and its `load_held`.

**What is here:**

- `cases.json`: ten cases in ten domains new to the planning exams (schema `dclab_rnd.planning_gate_v3.held.v1`, split
  `held`, two seeds). Each case has the keys of a v2 held case: `enforceable_traps` lists the traps code can check in a
  built project (`late_column`: kept out of the inputs; `group_split`: the holdout shares no value of the column with
  training; `time_split`: the holdout is the latest rows by that column). A case whose rows repeat an entity that the
  person never asks to generalise past also carries `no_group_needed`, naming that column: no group trap is planted
  there, and a grouped holdout is not required. `replica` names the case's generator in `tables.py`.
- `tables.py`: `make_<case_id>(seed) -> {file name: DataFrame}`, numpy and pandas only, deterministic, every table
  invented. Each case has one table, which holds the target; it has 600 to 2,000 rows on both seeds.
- `cases.sha256`: the SHA-256 of `cases.json` and `tables.py`. A changed file is a new set.

**Counts:** five classification and five regression cases. Traps planted: 9 late columns, 8 time splits, 4 group
splits (21 in all). Three cases have two date columns, and the one known first on every row is the time trap's column.
Every group case's brief says plainly that the model must serve entities it has never seen, in different words each
time; in three of the four, the entity's column name shares no word with the brief's noun for it. Four more cases
repeat an entity without asking for new ones (`no_group_needed`). One case has no late column, and two have no time
trap. Seven cases carry honest columns with alarming names that are known at the prediction moment and must stay in.
Datetimes and ISO date strings are mixed.

**How the traps are planted:** a late column is computed from the outcome after it happened (counted, paid, booked or
sent later); a group column repeats over the whole period and its rows share a hidden effect on the outcome; a time
column orders the rows and the outcome moves with it (a market fall, a policy change, a season, a launch).

**How the traps were checked by code (both seeds, every case):** generation is deterministic; the files and columns
match `cases.json`; every trap column exists; each late column is far from chance (AUC at least 0.75 for a
classification outcome, from 0.78 to 0.99 here; |Spearman| at least 0.5 for a regression outcome, from 0.53 to 1.00);
each time column shows drift (the outcome's latest quarter against its first, at least 0.2 standard deviations apart,
from 0.39 to 1.34); each group column has at least 20 values (44 to 110), 80% or more of them present at both ends of
the period (all of them here), and a share of the outcome's variance explained by the group well above that of a
shuffled grouping (0.11 to 0.73 against 0.02 to 0.09).

**The rule:** only the exam run reads these files, once. Nobody who changes the planning code reads them, and no
result of a case is shown case by case to the developer before the run is over. Changing a case after the run makes a
new set, not a fix.
