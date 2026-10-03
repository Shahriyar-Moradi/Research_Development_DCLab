# Evaluation, reliability, and trust

**Status: automated software checks exist; end-to-end correctness and user value remain unproven.** No finite test suite can guarantee an agent is completely correct. Evaluation should state exactly which tasks and failure modes have been tested and where it remains uncertain.

## Layers to evaluate

1. Deterministic worker correctness: feature policy, folds, metrics, artifact fingerprints, and replay.
2. Agent behavior: plan quality, proposal validity, evidence citation, critique accuracy, safe stopping, and recovery from provider errors.
3. User value: whether ML engineers find and fix real issues faster with fewer errors.
4. Generalization: new datasets, notebooks, users, and problem families.
5. Operational safety: data handling, permissions, auditability, and rollback.

## Existing artifacts

- Agent and worker tests: [`../../tests/test_agentic.py`](../../tests/test_agentic.py) and [`../../tests/test_agentic_worker.py`](../../tests/test_agentic_worker.py)
- CI workflow: [`../../.github/workflows/`](../../.github/workflows/)

## Next useful study

Run a blinded, reviewer-labeled pilot on HyperAck and churn notebooks. Report denominators and per-severity false positives/false negatives, citation precision, unsupported advice, task success, review time, and confidence intervals where sample size allows. Publish limitations and failures beside the score. Software tests are necessary, but are not evidence of real-world usefulness by themselves.
