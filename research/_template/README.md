# {Track name}

**Status:** planned · active · paused · concluded
**Owner:** {name} · **Started:** {YYYY-MM-DD}

## Research question

One or two sentences. What do we want to know, and what decision will the answer change?

## Why it matters for DCLab

How the result reaches the product: a copilot detector, an agent tool, a workflow stage, a rule, or a model.

## Prediction contract

For modeling work, write this before looking at any score (rule DCLAB-R01):

| Item | Value |
|---|---|
| Unit of prediction | e.g. one order, one user-day, one graph node |
| Prediction moment | when the prediction is made |
| Target and window | what is predicted, over which period |
| Allowed information | what exists at the prediction moment |
| Forbidden information | post-outcome columns, future rows, label-derived aggregates |
| Primary metric | chosen from the real cost of errors |

## Experiments

| ID | Question | Status | Result file |
|---|---|---|---|
| {TRACK}-001 | | planned | `results/{TRACK}-001_<name>.json` |

## Layout

```
README.md          ← this file: question, contract, experiment log, conclusions
data.md            ← sources, licenses, SHA-256 of every input file
src/               ← reusable code for this track
notebooks/         ← exploration; promote anything reused into src/
results/           ← one JSON per experiment (schema below), committed
reports/           ← human-readable write-ups
```

## Result schema

Write results in the shape the rest of the repository already understands, so `dclab_rnd.evidence_index` can index them and the SFT builder can learn from them:

```json
{
  "schema_version": 1,
  "campaign_id": "{track}_v1",
  "experiment_id": "{TRACK}-001",
  "dataset": "dataset_key",
  "kind": "data_understanding | leakage_audit | feature_engineering | model_selection | optimization_reliability",
  "task_type": "binary | multiclass | regression | timeseries_regression | graph_node_classification | ...",
  "primary_metric": "roc_auc | average_precision | macro_f1 | mae | ...",
  "question": "...",
  "hypothesis": "...",
  "setup_summary": "1-3 sentences: exactly what was compared, with numbers",
  "evidence": {"selection_rule": "...", "...": "..."},
  "claims": [{"claim_id": "{TRACK}-001-C1", "kind": "fact | risk | decision | recommendation",
              "statement": "...", "evidence": ["evidence.<key>"], "limitations": ["..."]}],
  "provenance": "from dclab_rnd.provenance.capture_provenance(...)"
}
```

Results that follow the five standard stages can live under `campaigns/<track>_v1/results/EXP-*.json`, where the evidence index, the critic gate and the SFT builder pick them up automatically.

## Conclusions

Write what was learned, with the numbers and the result files that support it. Turn durable lessons into rules in `knowledge/model_building_rules.jsonl` only after they hold on more than one dataset.
