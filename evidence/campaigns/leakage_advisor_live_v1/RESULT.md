# Does the Leakage Advisor help? A live check (2026-11-19)

The planning practice set (12 open cases, seed 11): 8 planted late columns and 29 honest columns that only look
suspicious, both marked by the cases' independent author. The leakage stage ran twice per case on the same project:
the advisor's rules alone, and the advisor with `qwen/qwen3.8-flash` through the run's capped gateway. Run
`agent_runs/live_tests/2026-11-19T154923-leakage-advisor-live`, result `LAL-001_dev.json`.

| | Rules alone | With the model |
|---|---|---|
| Planted leaks flagged (block or ask) | 7 of 8 | 8 of 8 |
| Planted leaks blocked | 5 of 8 | 5 of 8 |
| Honest columns flagged | 0 of 29 | 2 of 29 (asked, none blocked) |

Cost EUR 0.09 for 12 cases (the run's gateway also served the stage notes); 54 of 54 calls kept their thinking.

## What it means

- The model adds little: one more leak raised as a question, two honest columns asked about. The rules part does most
  of the work at no cost.
- The striking finding is elsewhere: the advisor's rules, which read the data's statistics, flag 7 of 8 planted leaks,
  while the unattended build kept out only half of the practice set's late columns. The advisor only advises; nothing
  acts on it in an unattended build. Feeding its evidence-based blocks into the unattended plan is the most direct route
  to the largest gap of the planning exam (no plan caught a held-out late column).

## Limits

One seed, 12 cases, 8 leaks; the practice set was read by the fixer of the planner (not of the advisor). The model's
two extra asks and one extra flag are too few to call it better or worse.
