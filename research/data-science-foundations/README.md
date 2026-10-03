# Data science foundations

**Status:** planned

## Research question

Which exploratory analysis, statistics and data-cleaning habits most reduce errors later in modeling, and how do we teach them?

## Why it matters for DCLab

DCLab's users range from beginners to experts; the agent needs a clear model of good first steps to coach them.

## Leakage traps specific to this field

- Summary statistics computed on all rows before the split shape later decisions (train-only EDA, rule DCLAB-R03).
- Duplicates and entity repeats make random splits optimistic.
- Imputing with statistics from all rows (measured as tiny for means in PIT-001, larger for learned imputers).

## First experiments

| ID | Question |
|---|---|
| DS-001 | Does a train-only EDA checklist change the features people choose, versus EDA on all rows? |
| DS-002 | Which data-quality checks (missingness patterns, duplicates, unit errors) catch the most real problems across the campaign datasets? |
| DS-003 | Turn `research/churn-prediction/logistic_regression/pandas_numpy_intro.ipynb` into a graded learning path the copilot can follow |

## Candidate datasets

All datasets already in `data/` and `external_data/`.

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=data-science-foundations TITLE="Data science foundations" PREFIX=DS
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
