# End-to-end comparison on development cases (`e2e_v1`)

dev cases only: no claim; the gate in `gate_v1.json` is for the held-out run.

## Live three-arm run (`E2E-002_dev.json`)

- Model: `qwen/qwen3.8-flash`
- Cases: `delivery_time-s11`, `ticket_routing-s11` (`not_run`: none)
- Cap: `0.5` EUR; total spent: `0.100322` EUR
- Run folder: `agent_runs/live_tests/2026-11-18-e2e-E2E-002`
- Archive snapshot: `training_records/snapshots/20261118T205250Z`

| Case | Arm | Finished | Leaks kept out | Claimed (interval) | Real | Real inside interval | Holdout reads | Numbers nobody gave | Euros | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| `delivery_time-s11` | `rules` | True | False (`courier_rating`) | 6.755406593640644 ([6.266952139441172, 7.10084858226776]) | 8.829994170532464 | False | 1 | n/a | 0.0 | 3.0 |
| `delivery_time-s11` | `dclab_agent` | True | True | 8.460902870496113 ([8.013594441962242, 9.267552373711268]) | 8.923342584220812 | True | 1 | n/a | 0.02364 | 516.0 |
| `delivery_time-s11` | `plain` | True | True | 6.5 ([6, 7.2]) | 10.5572039018691 | False | n/a | 5 | 0.022721 | 589.1 |
| `ticket_routing-s11` | `rules` | True | False (`first_response_minutes`, `resolved_by`) | 0.8934976229162905 ([0.7969010143805235, 0.9027635257297473]) | 0.28917315469145954 | False | 1 | n/a | 0.0 | 6.0 |
| `ticket_routing-s11` | `dclab_agent` | True | True | 0.38697927519525016 ([0.3590965262243677, 0.4158132054511912]) | 0.38959781155488016 | True | 1 | n/a | 0.03025 | 597.0 |
| `ticket_routing-s11` | `plain` | True | True | 0.43 ([0.36, 0.48]) | 0.38769758524437153 | True | n/a | 5 | 0.023711 | 800.9 |

## Free rules run on all six dev cases (`E2E-001_rules_dev.json`)

- Model: none (`DCLAB_NO_LIVE_MODELS=1`)
- Run folder: `agent_runs/live_tests/2026-11-18-e2e-E2E-001`

| Case | Arm | Finished | Leaks kept out | Claimed (interval) | Real | Real inside interval | Holdout reads | Numbers nobody gave | Euros | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| `ticket_routing-s11` | `rules` | True | False (`first_response_minutes`, `resolved_by`) | 0.8934976229162905 ([0.7974185136989408, 0.9056737993974836]) | 0.28917315469145954 | False | 1 | n/a | 0.0 | 70.2 |
| `delivery_time-s11` | `rules` | True | False (`courier_rating`) | 6.116295268694559 ([5.679326661674182, 6.567479778774579]) | 8.829994170532464 | False | 1 | n/a | 0.0 | 12.8 |
| `ticket_routing-s12` | `rules` | True | False (`first_response_minutes`, `resolved_by`) | 1.0 ([0.9166666666666666, 1.0]) | 0.2305673758743002 | False | 1 | n/a | 0.0 | 79.4 |
| `delivery_time-s12` | `rules` | True | False (`courier_rating`) | 6.3057959388097125 ([5.933964909235637, 6.748610858662923]) | 9.271879759487046 | False | 1 | n/a | 0.0 | 12.7 |
| `ticket_routing-s13` | `rules` | True | False (`first_response_minutes`, `resolved_by`) | 0.9166666666666666 ([0.8333333333333334, 0.9166666666666666]) | 0.3136645909945733 | False | 1 | n/a | 0.0 | 72.9 |
| `delivery_time-s13` | `rules` | True | False (`courier_rating`) | 5.643928181330363 ([5.211046238795916, 6.101070546619097]) | 9.66401748912539 | False | 1 | n/a | 0.0 | 12.9 |

## What each arm did wrong

- **`rules`:** When started blind without a model, the workflow lane left `forbidden=[]` (and `time_column=None` on `delivery_time`) even where its data notes flagged leak suspects, so it kept leak columns (`courier_rating` on delivery time; `first_response_minutes` and `resolved_by` on ticket routing) and produced over-optimistic claims outside the future-set intervals on all 6 cases.
- **`dclab_agent`:** It finished both cases cleanly, kept every planted leak column out, and put the real score inside its interval, though it still left the high-cardinality entity/customer id columns (`restaurant_id`, `customer_id`) in `feature` roles with `group_column=None`.
- **`plain`:** It caught the leaks in Python, but on `delivery_time-s11` it returned 32 engineered feature names (`distance`, `hour`, `te_rest`, ...) instead of raw table column names so only 3 features matched for reference scoring (real MAE 10.5572 outside [6, 7.2]), hit failing `run_python` scripts (6 of 23 on delivery time, 4 of 23 on ticket routing), and wrote 5 numbers in each final report not found verbatim in its tool outputs.

## Thinking capture note (`E2E-002`)

- `capture` on `agent_runs/live_tests/2026-11-18-e2e-E2E-002`: 48 exchanges (`plain`, purpose `e2e_plain_agent`), 48 `ok`, 48 `whole`, 48 `with_thinking`, 0 `without_thinking`.
- The twin's AI-first lane (`TwinGateway` / `ClaudeClient`) writes `transcripts/model_calls.jsonl` and `transcripts/thinking.jsonl` rather than `model_exchanges/**/*.jsonl`, so `capture` did not count twin calls; inspecting `model_calls.jsonl` showed all 49 `ok` calls (21 of 38 rows on `delivery_time-s11` and 28 of 55 rows on `ticket_routing-s11`; remaining rows are `check` records) kept non-empty `answer.thinking`.

## Limits

- The real score scores each arm's feature choices with one fixed scikit-learn gradient boosting reference model, not each arm's own fitted model (and only raw columns present in `case.table.columns` enter `reference.real_score`).
- Two problem kinds and two live dev cases give a first look, not a general claim.
- dev cases only: no claim; the gate in `gate_v1.json` is for the held-out run.

## Scored again with the fixed scorer (`E2E-003_rescore_dev.json`, 2026-11-18, no model asked)

The review of E2E-002 found three faults in the scoring, fixed before any held-out run (`e2e-scorer-fixes`); the gate
file is unchanged and E2E-002 itself is kept as it was.

- **The plain arm's real score on delivery time was not its model's.** It named 32 features it had built; 3 were table
  columns, and the reference model was fitted on those 3 (MAE 10.56). A name that is not a column now leaves the real
  score unknown: that case has no real score (29 names are not columns). The prompt and the finish tool now ask for the
  table's own columns.
- **"Numbers nobody gave" counted rounding as invention.** All 10 were outputs rounded (7.937 written 7.94) or settings
  in its own code (a 300-row fold, `max_iter=4000`). Now: 0 on delivery time, 1 on ticket routing (its own interval
  end, 0.36).
- **The capture missed the DCLab agent's calls.** It now reads the twin lane's own log: 97 of 97 answered calls kept
  their thinking (48 plain, 49 DCLab agent).

| Case | Arm | Real | Real inside interval | Numbers nobody gave |
|---|---|---|---|---|
| `delivery_time-s11` | `plain` | unknown (built features named) | unknown | 0 |
| `ticket_routing-s11` | `plain` | 0.3877 | True | 1 |

The rules and DCLab agent rows are unchanged. On ticket routing, the one case with both scored, DCLab with the model and
the model alone are level on the real score (0.3896 and 0.3877) and both honest about it.

## The rules arm after the leak evidence (`E2E-006_rules_dev.json`, 2026-11-19, no model asked)

An unattended build no longer leaves a column in only because nobody answered about its name. Each column suspected by
its name is weighed on the rows read before the holdout (`dclab_core.leak_evidence`): its timing, whether alone it beats
every other column together, and the score with and without it. `E2E-005` was the first rerun, before the review's
fixes; `E2E-006` is the final code. Both gave the same rows.

| Kind | Run | Claimed | Real | Suspects kept out |
|---|---|---|---|---|
| ticket routing (macro F1) | E2E-001 (before) | 0.937 | 0.278 | none |
| ticket routing | E2E-006 (after) | 0.464 | 0.302 | `resolved_by` (too good: alone 1.00 against 0.50 for the rest) |
| delivery time (MAE, min) | E2E-001 (before) | 6.02 | 9.26 | none |
| delivery time | E2E-006 (after) | 7.42 | 9.40 | `courier_rating` (too good: alone 7.4 against 8.7 for the rest) |

- The claims are much closer to the truth (tickets 0.94 to 0.46 claimed, against about 0.30 real).
- The real score is not better yet. Two leaks remain, and this change does not reach them:
  - `first_response_minutes` (tickets): its name does not sound late, so it is never a suspect. Without `resolved_by`
    the model leans on it, and on seed 11 the real score fell to 0.13.
  - Delivery time: the rules arm does not find `order_time` as the time column, so the holdout is random rows and the
    claim stays about 2 minutes too good. The report now says the rows were drawn at random.
- On delivery the real MAE is 0.15 minutes worse without `courier_rating` (9.26 to 9.40, 3 cases): small, and the
  reason is not checked.
