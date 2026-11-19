# hp_gate_v2, the planning exam: run once on 2026-11-19 — FAILED

Result: `results/HPG-001_gate.json` (counts and means only). People: `harness_ablation_v1/results/HAB-063_dev.json` (set
`plan_held`). Model `qwen/qwen3.8-flash` for both parts; people judged by `nvidia/nemotron-3.5-super-vl:free`. The gate's
terms were pinned before the run (`gate_v2.sha256`) and the pin held. Cost: tables EUR 0.2054, people EUR 0.1682, total
about EUR 0.37. Every model call kept its thinking: 22 of 22 planner calls (whole, in the exam vault, not in any training
folder) and 141 of 141 calls of the people run.

Nobody who changes planning read a held-out case, a person or a row; this note was written from the committed counts.

| Check | Result | Bar | Met |
|---|---|---|---|
| 1 obeyed | 69 of 73 caught traps enforced or asked (0.945); 22 of them asked, not applied; 8 projects stopped for their outcome | 1.0 | no |
| 2 nothing lost | 0 projects put a column back; 1 model plan dropped a column the rules' plan had caught | 0 and 0 | no |
| 3 people | planning +0.125 and understanding +0.25 for the lead agent over the model alone; 1 blocked run | gaps at least -0.5; 0 blocked | no (the blocked run only) |
| 4 zero events | 2 projects neither built nor stopped for their outcome; 0 holdouts read twice, 0 stages past a needed yes, 0 numbers nobody gave | all 0 | no |
| 5 model step | 16 pairs: planted traps enforced 0.29 (rules) vs 0.35 (model), +0.06, 95% interval [0.00, 0.16] | interval above 0 | decides: the rules serve |
| 6 cost | rules plan 0.019 s and EUR 0 a project (12 projects) | 300 s, EUR 0.03 | yes |

## What it means

- The model's plan was not shown to be better than the rules' plan: its lead of 0.06 has an interval that touches zero.
  The rules keep serving. That part is a decision, not a failure.
- Both arms enforce only about a third of the planted traps. The planner misses most of what the case authors planted.
  This is the biggest finding, and it holds for the rules and the model alike.
- The gate failed on four concrete counts: 4 caught traps neither enforced nor asked, 1 model plan that lost a rules
  catch, 2 projects that failed with nothing built, and 1 blocked conversation in the people run.
- On people, the lead agent was not worse than the model alone (it was a little better on both scores).

## Limits

- 8 held-out cases on 2 seeds, 8 people: small. The interval on check 5 is wide.
- The held-out material is now used. A next gate needs new cases and people written by another session.
- The causes of the four failing counts were not read, by the gate's own rule. Finding them needs either the run's rows
  (which would spend this held-out set for any future claim) or a dev set that reproduces them.
- The people run's conversations are kept in the run folder as dev runs are; the harness does not seal held-out people
  (a known gap of `harness_ablation`).
