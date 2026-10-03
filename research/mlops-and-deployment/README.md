# MLOps and deployment

**Status:** planned

## Research question

What does it take for a validated model to stay valid in production: feature contracts, monitoring, drift detection, retraining and rollback?

## Why it matters for DCLab

Rule DCLAB-R22 says no candidate is production-ready without these gates; this track builds them.

## Leakage traps specific to this field

- **Training-serving skew:** features computed differently online than in the notebook.
- **Silent missing inputs:** a feature disappears in production and the model degrades quietly (missing-input stress tests in the Studio worker).
- **Drift on proxies** that stays invisible in overall accuracy.

## First experiments

| ID | Question |
|---|---|
| OPS-001 | Feature contract: declare availability time, units and null behavior for the HyperAck safe features, and test them |
| OPS-002 | Drift detection: PSI vs model-based drift on a time-split HyperAck replay |
| OPS-003 | Retrain policy: fixed schedule vs drift-triggered, measured on backtests |

## Candidate datasets

HyperAck and the expansion time-series data (reusing existing repository data).

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=mlops-and-deployment TITLE="MLOps and deployment" PREFIX=OPS
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
