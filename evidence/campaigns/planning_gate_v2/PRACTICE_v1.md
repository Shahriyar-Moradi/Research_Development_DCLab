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
