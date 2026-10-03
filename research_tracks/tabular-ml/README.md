# Tabular data science and machine learning

**Status: existing evidence and executable pipelines.** This is the most mature research area in this repository. Its empirical results are scoped to their datasets, target definitions, split designs, and feature availability assumptions.

## Research questions

- What does the data represent, and what information exists at the intended prediction moment?
- Which apparent features leak future or target information?
- Which transformations improve performance without making features fragile or unavailable in production?
- Which algorithm family works best for a given data shape, metric, and operational constraint?
- Do optimization, calibration, and thresholds improve the actual decision rather than just a headline score?

## Existing material

- [`../../knowledge/MODEL_BUILDING_FIELD_GUIDE.md`](../../knowledge/MODEL_BUILDING_FIELD_GUIDE.md): EDA, leakage, feature reliability, model selection, and ten workflow blocks.
- [`../../campaigns/model_building_50_v1/`](../../campaigns/model_building_50_v1/): 10 public UCI datasets × 5 research stages.
- [`../../external_projects/`](../../external_projects/): per-dataset experiments, reports, and notebooks.
- [`../../hyperack_exp/`](../../hyperack_exp/), [`../../safe_leakage_exp/`](../../safe_leakage_exp/), [`../../optimized_safe_model/`](../../optimized_safe_model/): HyperAck feature/model comparisons, including deliberate unsafe-versus-safe analysis.
- [`../../churn_exp/`](../../churn_exp/): churn research.

## How to advance it

Start from a prediction contract, inspect the dataset card and leakage policy, choose an appropriate split, run a simple baseline, then change one factor at a time. Report paired development-fold metrics, uncertainty/stability, runtime, missing-input and slice behavior. Keep the final confirmation data untouched until choices are frozen.

Use the commands and warnings in the [main README](../../README.md#dclab-rd-paths). Do not infer that a winning algorithm or feature transfers to another dataset without testing it there.
