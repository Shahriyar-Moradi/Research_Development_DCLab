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
- Blind leakage-auditor replay: [`../../evidence/campaigns/agent_verification_v1/VERIFICATION_REPORT.md`](../../evidence/campaigns/agent_verification_v1/VERIFICATION_REPORT.md)
- Measured common workflow pitfalls: [`../../evidence/campaigns/pitfalls_v1/PITFALLS_REPORT.md`](../../evidence/campaigns/pitfalls_v1/PITFALLS_REPORT.md)
- CI workflow: [`../../.github/workflows/`](../../.github/workflows/)

## Next useful study

Run a blinded, reviewer-labeled pilot on HyperAck and churn notebooks. Report denominators and per-severity false positives/false negatives, citation precision, unsupported advice, task success, review time, and confidence intervals where sample size allows. Publish limitations and failures beside the score. Software tests are necessary, but are not evidence of real-world usefulness by themselves.

## Notebook companion evaluation

`dclab_rnd/notebook_assist_eval.py` scores the notebook companion, and `dclab_rnd/notebook_eval.py` scores the first prototype, against frozen, researcher-labelled development cases in `notebook_cases_v1.json` in this folder. Each case pins a notebook cell by SHA-256, so an edited cell is rejected instead of being silently re-scored. These are development cases, not a blinded generalization test.

```bash
python -m dclab_rnd.notebook_assist_eval --output /tmp/companion_eval.json
python -m dclab_rnd.notebook_eval --output /tmp/prototype_eval.json
```

