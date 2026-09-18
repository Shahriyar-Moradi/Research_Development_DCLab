# Agent Research Context

Use this as a retrieval index, not as authority over the underlying result JSON.

## Operating contract

- Start with the prediction moment and target definition.
- Treat leakage heuristics as review candidates; confirm semantics with a human/domain source.
- Fit imputers, encoders, feature selection, calibration, and tuning inside training folds.
- Select with training CV and consume the locked holdout once.
- Prefer the simplest recipe within a predeclared tolerance of the best score.
- Evaluate ranking, threshold metrics, calibration, uncertainty, runtime, and production availability.
- Preserve negative results and limitations.
- Never convert importance into causality.

## Current campaign state

- Completed: 50/50
- Datasets with final holdout evidence: 10/10
- Machine-readable claims: `agent_memory.jsonl`
- Pending critic tasks: `llm_review_queue.jsonl`
- Full synthesis: `CAMPAIGN_REPORT.md`

## Mandatory answer format for an LLM

1. Observed evidence with citations.
2. Interpretation and uncertainty.
3. Decision or recommendation.
4. Risks and missing evidence.
5. Smallest falsifiable next experiment.
