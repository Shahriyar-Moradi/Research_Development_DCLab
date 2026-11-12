# Home intent: hypothetical preference follow-up

2026-11-12. A targeted correction following TP-06, not completion of the full R-01 routing audit or TP-07.

The English `which_will` signal treated any “which/what … would” question as a stated prediction. For example, asking “Which deliverable would help our operations staff?” opened the ML track without identifying a prediction task. The same false signal could force a conflict question after a valid model reading, or treat an unresolved follow-up question as the person's prediction choice.

`intent_rules_v6` removes `would` alone as sufficient prediction evidence. Without other signals or a model reading, the existing open-question path asks what help the person needs. Explicit future requests, requests naming a model, supported outcome words and validated model readings retain their existing paths. A confirmed user choice still takes precedence for all six tracks. “Plan” still means business intent.

## Verified scope

Only `dclab_rnd/draft/intent.py` changes runtime behavior: one signal and its recorded rule version. There is no prompt, API/schema, storage, frontend, capability or permission change. No existing draft is migrated or reinterpreted automatically. A conditional prediction with no other recognized evidence may now require clarification in no-model mode; the word `would` alone is intentionally insufficient.

Six new test methods in `tests/test_home_intent.py` cover three hypothetical preferences, genuine future/conditional predictions, a valid model reading without a spurious conflict, Home waiting without an ML workflow, all six explicit choices, and an ambiguous answer to the track question. Three regression methods reproduced the old behavior before the fix; the other cases protect existing routes.

All **109 distinct relevant application tests passed**, without skips:

- `tests.test_home_intent`: 49 tests passed after the fix, then the newly added `RuleTests.test_a_hypothetical_preference_does_not_answer_the_track_question` passed separately; 50 distinct tests in total.
- `tests.test_home_lang`, `tests.test_home_conversation`, `tests.test_home_routing_points`, and four selected `tests.test_case_home.CaseHomeTests` checks: 59 passed. The selected checks cover chip routing, a chip changed before work, a chip pressed in an existing chat and prediction-only workflow creation.
- Every application command used `.venv/bin/python -m unittest ... -q`, `DCLAB_NO_LIVE_MODELS=1` and `DCLAB_DECISION_SHADOW=0`. The integration group used a disposable temporary root with automatic cleanup.

Independent `dclab-reviewer` review passed with no correctness finding. `git diff --check` passed. The separate development record was added and its corpus rebuilt with `make sft-dev`; all four `tests.test_devlog` validation/freshness checks passed with no refused examples. No broad gate, live/sealed run, PostgreSQL or browser suite was needed for this rule-only correction.

## Evaluation and next step

The TP-06 frozen cases, gate, results and generated comparison reports are untouched. The new regression wording is independently written; it does not copy the frozen request. This correction is a regression repair, not an unseen evaluation or evidence that the planner now meets its gate. The old release recommendation remains **FAIL, not promoted**. The current source differs from the frozen source closure, so the old runner correctly refuses to record a new comparison under that freeze. A new measurement needs its own versioned protocol.

TP-07 explicitly allows a developer-only read-only task-progress preview while promotion has not passed. That is the next product-integration slice: task status and evidence visibility, with execution disabled. Enabling model-assisted execution remains blocked by C's proposed-graph admission gap, its missing real-model comparison and insufficient independent completion/recovery cases. No model spend or rollout is authorized by these tests. The broader R-01 audit remains pending.
