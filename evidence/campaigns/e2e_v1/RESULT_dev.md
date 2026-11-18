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
