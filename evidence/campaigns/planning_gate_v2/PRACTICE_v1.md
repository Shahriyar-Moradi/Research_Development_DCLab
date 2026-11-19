# Planning practice set v1: why the planner misses planted traps (2026-11-19)

12 invented cases written by an independent session that never read the planner's code
(`dclab_rnd/agent_eval/planning_gate_v2/practice/`), each built on 2 seeds with the rules arm on the exam's own builder
(`python -m dclab_rnd.agent_eval.planning_gate_v2.practice`, no model, EUR 0). Every answer may be read.

**Result: 18 of 48 planted traps enforced (37%), the same on both seeds** — the same level as the exam (29%).

| Trap kind | Planted | Enforced | Where |
|---|---|---|---|
| late column | 16 | 8 | 8 kept out before the plan (named `refund`, `final`, `outcome`, `chargeback`); 8 missed |
| time order | 16 | 10 | 10 by the plan; 6 missed |
| same entity on both sides | 16 | 0 | 16 missed |

## The four causes (each seen on both seeds)

1. **No entity detection from the data.** The rules plan adds a grouped split only when the problem card declares an
   entity key or a joined table repeats a key ending in `_id`. A main table where `store_no` or `meter_id` repeats 40
   times is never read as an entity. 16 of 16 missed.
2. **Two date columns: no time column, a random split, no question.** `booking_date` + `appt_date`, `ts` + `due_date`,
   `booking_date` + `arrival_date`: the standard plan picks neither, the plan catches no time trap, and the holdout is a
   random split. 6 of 6 time misses.
3. **Leaks found by name only.** `late_fee_charged`, `n_followups`, `days_to_resolve`, `rework_minutes` carry no
   after-outcome word, so leak evidence never weighs them; three were features of the final model.
4. **A named outcome lost to an ordinary word.** pr05's request says ``Predict `defect`.`` but its brief also uses the
   word "line", a column name; two names matched, so the build stopped asking for the outcome (both seeds).

## Caution about the test itself

4 of the 8 entity cases (pr01 customers, pr03 accounts, pr04 patients, pr11 cards) predict for entities that already
exist. There, a split by time with the same customer on both sides is the honest evaluation of real use (the lesson of
the end-to-end future set, 2026-11-19), so "missed" may be the right decision. The other 4 (pr05, pr08, pr09, pr12) say
in the brief that the model must work for new machines, buildings, students or stores: those are real misses. The
held-out exam may count the same kind of case, so part of its "one in three" can be the test's definition.

Rows: `agent_runs/practice/2026-11-19T123440-planning-practice/rows.json` (not committed).

## After fixes 1, 2 and 4 (the owner's choice; dclab-core 0.163.0)

Rerun `agent_runs/practice/2026-11-19T130846-planning-practice`: **24 of 48 enforced** (was 18), 0 builds failed, 0
stopped wrongly.

| Trap kind | Before | After | What changed |
|---|---|---|---|
| time order | 10 of 16 | 16 of 16 | of two dates, the one known first on every row orders the split, and the report says so |
| same entity | 0 of 16 | 0 applied, 2 asked | read only when the brief says new entities; applied only when the engine can split by time and entity at once, else asked |
| late column | 8 of 16 | 8 of 16 | not in scope (fix 3) |
| named outcome lost | 2 stops | 0 | a column named right after "predict" breaks a tie |

Why the entity traps are asked, not applied: in every practice table each entity appears in every period, so the
engine cannot hold out unseen entities in the latest period; applying it failed every one of those builds. pr05
("new presses", `machine_serial`) and pr08 ("buildings", `meter_id`) are not read at all: the noun names no column, and
the review showed a guess picks the wrong entity (a customer split for "the stores we open").

What this does not prove: the rules were written after reading this set, so the gain here is not evidence. A fresh
held-out exam (another session's cases) must judge it. Entity splits for new entities need an engine split that holds
out entities and scores their later rows; that is a separate change.

## After the engine's split for new entities

The engine's time-and-entity split used to refuse a table where every entity appears in every period. It now holds a
seeded quarter of the entities out as new (at least 20 entities, so at least five held out): their rows after a time
cut (the 0.75 quantile) are the test, the other entities' rows before the cut train, and the rest is used by neither,
which the person is told. Rerun `agent_runs/practice/2026-11-19T132745-planning-practice`: **28 of 48 enforced**,
0 builds failed; the entity split applied in pr09 and pr12 on both seeds.

The cost is real: pr12 leaves 460 of 1,200 rows unused, pr09 about half. The test set is small (80 to 160 rows), so its
interval is wide. pr05 and pr08 are still not read: their brief's noun names no column.
