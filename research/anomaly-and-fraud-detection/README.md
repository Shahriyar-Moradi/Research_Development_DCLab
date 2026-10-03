# Anomaly and fraud detection

**Status:** planned

## Research question

Which methods and metrics work when the event of interest is rare (well under 1%), and how should alert thresholds be chosen?

## Why it matters for DCLab

Fraud, abuse and failure detection are high-value DCLab use cases where accuracy is meaningless.

## Leakage traps specific to this field

- **Oversampling before the split:** inflated ROC-AUC by up to +0.30 in PIT-003.
- **Accuracy:** always predicting "not fraud" is 99.87% accurate on the card-fraud data (EXP-055).
- **Globally fitted transforms:** the public card-fraud PCA features were fitted on all rows, a look-ahead that cannot be undone.

## First experiments

| ID | Question |
|---|---|
| FRD-001 | Seed result, done: card fraud, ExtraTrees average precision 0.814; a threshold chosen on training data gave precision 0.963 and recall 0.693 (EXP-051..055) |
| FRD-002 | Supervised vs unsupervised (isolation forest, autoencoder) vs semi-supervised when only a few labels exist |
| FRD-003 | Cost-based threshold selection: expected cost per 1,000 transactions under different review budgets |

## Candidate datasets

Credit card fraud (ULB/Kaggle), IEEE-CIS fraud (Kaggle), Elliptic, NAB anomaly benchmark.

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=anomaly-and-fraud-detection TITLE="Anomaly and fraud detection" PREFIX=FRD
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
