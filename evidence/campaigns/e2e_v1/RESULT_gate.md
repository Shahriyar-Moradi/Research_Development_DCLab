# End-to-end comparison: the held-out gate (`E2E-004_gate.json`, 2026-11-18)

**Verdict: `e2e_gate_v1` passed.** The gate file was the pinned one (sha256 checked), and the code that judges it was
committed before the run (cae3ffac). The 8 held-out cases (ticket routing and delivery time, seeds 101 to 104) were run
once, with Qwen 3.8 flash, capped at EUR 0.5. Spent: EUR 0.449. No case was skipped.

## The gate's checks (DCLab with the model)

| Check | Result | Bar | Met |
|---|---|---|---|
| Finished | 8 of 8 | 8 of 8 | yes |
| Leaks kept out | 8 of 8 | 8 of 8 | yes |
| Holdout read exactly once | 8 of 8 | 8 of 8 | yes |
| Real score inside its own interval | 7 of 8 | 6 of 8 | yes |
| Ticket routing, real score against the model alone (4 pairs) | mean difference -0.0019 | at least -0.02 | yes |
| Delivery time, real MAE against the model alone (4 pairs) | mean difference +0.066 min | at most +0.443 min | yes |

## By arm (means over 4 cases of each kind)

| Kind | Arm | Real | Claimed | Real inside interval | Leaks kept out | EUR | Minutes a case |
|---|---|---|---|---|---|---|---|
| ticket routing (macro F1) | rules | 0.279 | 0.941 | 0 of 4 | 0 of 4 | 0 | 0.1 |
| ticket routing | DCLab agent | 0.388 | 0.419 | 3 of 4 | 4 of 4 | 0.111 | 8.9 |
| ticket routing | plain | 0.390 | 0.435 | 1 of 4 | 4 of 4 | 0.103 | 10.1 |
| delivery time (MAE, min) | rules | 8.81 | 6.69 | 0 of 4 | 0 of 4 | 0 | 0.0 |
| delivery time | DCLab agent | 8.92 | 8.98 | 4 of 4 | 4 of 4 | 0.130 | 10.5 |
| delivery time | plain | 8.86 | 6.75 | 0 of 4 | 4 of 4 | 0.104 | 10.0 |

## What it means, in plain words

1. **The gate asks "not worse", and that is what it found.** DCLab with the model and the same model alone choose
   columns that score the same on the future set. DCLab is not better at the score.
2. **DCLab's number is honest; the model alone's is not.** On delivery time the model alone kept the leaks out but said
   about 6.7 minutes, and the future set gave about 8.9: 2.1 minutes too good, in all 4 cases. DCLab said 9.0 and got
   8.9. The likely cause (read in one case's code, not proven for all): the model alone checked itself with shuffled folds,
   while DCLab splits by time. The gate does not check the plain arm's honesty, so this is a finding, not a gate result.
3. **The rules alone keep the leaks.** They claim 0.94 on tickets and get 0.28. This is the unattended standard plan
   keeping name-suspected leak columns (a decision left to the owner, see RESULT_dev.md).
4. **The model alone wrote numbers nobody gave** in 6 of 8 cases (up to 4 a case). DCLab's report is built from records.

## Limits

- Two problem kinds, invented tables, 4 cases each: a pass on this gate, not a general claim.
- The real score refits one reference model on each arm's chosen columns, so it judges the choice of columns, not each
  arm's own model.
- One model (Qwen 3.8 flash). Pairs: 4 on each kind; no case was dropped for naming built features.
- The held-out set is now used. Any change judged again needs new seeds.

## Records

- Run folder: `agent_runs/live_tests/2026-11-18-e2e-E2E-004/` (not committed).
- Capture: 432 of 432 model exchanges kept whole, every one with its thinking.
- Archive snapshot: `training_records/snapshots/20261118T235221Z`.
