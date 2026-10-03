# Agentic R&D studio

**Status: implemented prototype; continuing evaluation required.** The tracked studio combines typed specialist roles, a LangGraph controller, a bounded experiment schema, and deterministic model execution. It is intended to turn a research question into traceable proposals, trials, critiques, and follow-up questions.

## Questions to answer

- Does the plan match the user's question and dataset constraints?
- Are proposals valid, useful, and meaningfully different?
- Does the system react correctly to measured evidence and failures?
- Are runs reproducible and do artifacts preserve provenance and limitations?
- Does the system avoid unsupported conclusions and unsafe promotions?

## Existing implementation and evaluation

- Runtime: [`../../dclab_rnd/agentic/`](../../dclab_rnd/agentic/)
- Agent tests: [`../../tests/test_agentic.py`](../../tests/test_agentic.py) and [`../../tests/test_agentic_worker.py`](../../tests/test_agentic_worker.py)
- Runtime source: [`../../dclab_rnd/agentic/engine.py`](../../dclab_rnd/agentic/engine.py) and [`../../dclab_rnd/agentic/worker.py`](../../dclab_rnd/agentic/worker.py)

## Next useful test

Build a blinded suite of research requests with expected constraints and acceptable plans. Score plan adherence, invalid-proposal rate, citation grounding, whether the agent selects a discriminating next experiment, and deterministic metric agreement. Include rate-limit/provider failures and resume behavior. A completed run is not itself proof of correctness.

Run instructions are in the [main README](../../README.md#dclab-rd-paths). Live runs send task context to the configured model provider, so use data that is authorized for that destination.
