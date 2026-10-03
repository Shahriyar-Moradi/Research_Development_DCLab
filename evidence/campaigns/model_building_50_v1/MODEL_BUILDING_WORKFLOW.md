# Reusable Model-Building Workflow

These are the code flows that the campaign executes and tests. They are patterns, not copy/paste-only recipes.

```text
prediction contract
  → immutable source + provenance
  → locked holdout
  → train-only EDA and leakage review
  → fold-fitted feature ablation
  → stability-aware algorithm screen
  → conservative optimization
  → one final holdout evaluation
  → calibration/uncertainty/feature reliability
  → evidence claims + LLM critic queue
```

## Train-only feature fitting

```python
engineer.fit(X_fold_train, y_fold_train)
X_fit = engineer.transform(X_fold_train)
X_valid = engineer.transform(X_fold_valid)
model.fit(X_fit, y_fold_train)
```

## Feature promotion rule

```text
decision-time available
AND stable across validation folds
AND reproducibly generated
AND contractually obtainable in production
AND monitored for missingness and drift
AND improves a predeclared metric/cost objective
```

## Model selection rule

```text
Compare multiple families on identical folds.
Rank by validation evidence and stability, then runtime/cost.
Optimize only the selected family.
Use the final holdout once after all choices are locked.
```

Implementation: `dclab_rnd/science.py`; orchestration: `dclab_rnd/campaign.py`.
