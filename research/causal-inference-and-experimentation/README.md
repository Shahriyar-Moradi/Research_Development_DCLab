# Causal inference and experimentation

**Status:** planned

## Research question

When can models answer "what happens if we act?" rather than "what will happen?", and how do we design and analyze experiments that answer it?

## Why it matters for DCLab

Most DCLab users ultimately want a decision (who to call, what to discount), which is a causal question.

## Leakage traps specific to this field

- **Post-treatment variables** used as controls.
- **Confusing prediction with uplift:** high churn risk does not mean a retention offer will help.
- **Peeking** at running A/B tests.

## First experiments

| ID | Question |
|---|---|
| CAU-001 | Uplift models (T-learner, X-learner, causal forest) vs response models on a marketing dataset with randomized treatment |
| CAU-002 | Simulate post-treatment control bias and measure the error in estimated effect |
| CAU-003 | Sequential testing vs fixed-horizon A/B tests: false-positive rates under peeking |

## Candidate datasets

Hillstrom email marketing, Criteo uplift dataset, Lalonde.

## Start the track

When work begins, scaffold the standard layout (src/, notebooks/, results/, reports/, data.md) from the template:

```bash
make new-track NAME=causal-inference-and-experimentation TITLE="Causal inference and experimentation" PREFIX=CAU
```

This keeps the notes on this page and adds the experiment log, prediction-contract table and result schema.
