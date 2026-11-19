# hp_gate_v3, the planning exam after the practice set's fixes: run once on 2026-11-19 — FAILED (3 of 8 checks)

Result: `results/HPG3-001_gate.json` (counts and means only). Held-out tables and people written by an independent
session (e824a90f), pinned before the run (feda6ccf); 10 cases on 2 seeds, 40 builds, and 8 people (HAB-064). Model
`qwen/qwen3.8-flash`; people judged by `nvidia/nemotron-3.5-super-vl:free`. Cost: tables EUR 0.313, people EUR 0.123.
Thinking kept on every call (34 of 34 planner calls in the exam vault, 123 of 123 in the people run). The first start
was stopped after one build when the Leakage Advisor leaked held-out columns to a model; it was fixed, moved to the
vault and restarted with the owner's agreement (`agent_runs/exam_vault/HPG3-001_aborted_start/README.md`).

| Check | v2 | v3 | Met |
|---|---|---|---|
| 1 obeyed | 69 of 73 | 108 of 108 (29 asked) | yes |
| 2 nothing lost | 1 model plan dropped a rules catch | 2 | no |
| 3 people (lead agent vs model alone) | +0.125 / +0.25, 1 blocked | 0.0 / 0.0, none blocked | yes |
| 4 zero events | 2 built nothing | none | yes |
| 5 model step | rules serve (+0.06, [0.00, 0.16]) | **the model serves**: 0.52 vs 0.37, +0.15 [0.03, 0.30] | decides |
| 6 cost of the serving plan | rules: 0.02 s | model: 561 s a project (bar 300), EUR 0.016 | no |
| 7 caught (new) | — | model 0.50 (bar 0.5), rules 0.38 | yes |
| 8 no needless entity split (new) | — | 8 of the 16 projects it applies to | no |

Planted traps enforced, by kind: time order 100% (both); same entity: rules 0%, model 62.5%; late columns: **0% for
both**.

## What it means

- The build now follows its own plan every time and never broke: the practice set's fixes held on unseen cases.
- For the first time the model's plan is measurably better than the rules' (its interval is above zero), mostly on
  entity splits. But it takes 9 minutes a project, three times the bar, and it splits by entity where the case says
  the same entities recur, half the time.
- The rules' new-entity words (fix 1) caught none of the four held-out entity cases: their wording differed from the
  word list. As warned, a word list fitted on the practice set did not carry over.
- No plan caught a single planted late column: leaks named neutrally are found by neither arm. Leaks found from the
  data (fix 3) is now the largest gap.

## Limits

10 cases, 8 people. The held-out material is now used. Rows were not read.
