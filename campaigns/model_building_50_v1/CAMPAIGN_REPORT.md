# DCLab 50-Experiment Model-Building Campaign

Status: **50/50 completed**, **0 failed**, **0 pending**.

This campaign is intentionally organized as 10 real UCI datasets × 5 scientific questions. Model and feature choices use training-only CV; the final holdout is consumed once per dataset.

## Dataset decisions

| Dataset | Leakage exclusions | Feature recipe | CV model candidate | Final recipe | Holdout ROC-AUC | Brier | Evidence |
|---|---|---|---|---|---:|---:|---|
| adult | none declared | raw | lightgbm | lightgbm/lightgbm_C02 | 0.9097 | 0.0971 | `campaigns/model_building_50_v1/results/EXP-005_adult_optimization_reliability.json` |
| bank_marketing | duration | ratios | extra_trees | extra_trees/extra_trees_C02 | 0.7718 | 0.0870 | `campaigns/model_building_50_v1/results/EXP-010_bank_marketing_optimization_reliability.json` |
| breast_cancer | none declared | raw | logistic_regression | logistic_regression/baseline | 0.9960 | 0.0213 | `campaigns/model_building_50_v1/results/EXP-015_breast_cancer_optimization_reliability.json` |
| credit_default | none declared | raw | extra_trees | extra_trees/extra_trees_C02 | 0.7746 | 0.1371 | `campaigns/model_building_50_v1/results/EXP-025_credit_default_optimization_reliability.json` |
| german_credit | none declared | interactions | extra_trees | extra_trees/extra_trees_C01 | 0.7943 | 0.1605 | `campaigns/model_building_50_v1/results/EXP-030_german_credit_optimization_reliability.json` |
| heart_disease | none declared | raw | extra_trees | extra_trees/extra_trees_C02 | 0.9448 | 0.1070 | `campaigns/model_building_50_v1/results/EXP-020_heart_disease_optimization_reliability.json` |
| mushroom | none declared | raw | extra_trees | extra_trees/baseline | 1.0000 | 0.0000 | `campaigns/model_building_50_v1/results/EXP-035_mushroom_optimization_reliability.json` |
| online_shoppers | PageValues | raw | extra_trees | extra_trees/extra_trees_C02 | 0.7394 | 0.1182 | `campaigns/model_building_50_v1/results/EXP-045_online_shoppers_optimization_reliability.json` |
| spambase | none declared | raw | hist_gradient_boosting | hist_gradient_boosting/baseline | 0.9849 | 0.0382 | `campaigns/model_building_50_v1/results/EXP-040_spambase_optimization_reliability.json` |
| wine_quality | none declared | ratios | extra_trees | extra_trees/baseline | 0.8878 | 0.1319 | `campaigns/model_building_50_v1/results/EXP-050_wine_quality_optimization_reliability.json` |

## Data-understanding evidence

Profiles use the training partition for target-aware statements. Random-split PSI is only a smoke test; it does not replace temporal or operational drift validation.

| Dataset | Source rows | Analyzed rows | Features | Positive rate | Missing cells | Duplicate rows | Features needing reliability review |
|---|---:|---:|---:|---:|---:|---:|---:|
| adult | 48842 | 4000 | 14 | 0.239 | 0.000 | 0.000 | 0 |
| bank_marketing | 45211 | 4000 | 16 | 0.117 | 0.000 | 0.000 | 0 |
| breast_cancer | 569 | 569 | 30 | 0.374 | 0.000 | 0.000 | 1 |
| credit_default | 30000 | 4000 | 23 | 0.221 | 0.000 | 0.000 | 0 |
| german_credit | 1000 | 1000 | 20 | 0.300 | 0.000 | 0.000 | 0 |
| heart_disease | 303 | 303 | 13 | 0.459 | 0.001 | 0.000 | 3 |
| mushroom | 8124 | 4000 | 22 | 0.482 | 0.000 | 0.000 | 1 |
| online_shoppers | 12330 | 4000 | 17 | 0.155 | 0.000 | 0.005 | 0 |
| spambase | 4601 | 4000 | 57 | 0.394 | 0.000 | 0.071 | 0 |
| wine_quality | 6497 | 4000 | 11 | 0.633 | 0.000 | 0.099 | 0 |

## Leakage evidence

The apparent lift is measured on training CV only. A heuristic flag is a request for semantic review, not automatic proof of leakage.

| Dataset | Declared exclusions | Review candidates | Detector canary | Safe CV ROC-AUC | Unsafe CV ROC-AUC | Apparent leakage lift |
|---|---|---:|---|---:|---:|---:|
| adult | none | 0 | True | — | — | — |
| bank_marketing | duration | 1 | True | 0.7130 | 0.8972 | 0.1842 |
| breast_cancer | none | 0 | True | — | — | — |
| credit_default | none | 0 | True | — | — | — |
| german_credit | none | 0 | True | — | — | — |
| heart_disease | none | 0 | True | — | — | — |
| mushroom | none | 0 | True | — | — | — |
| online_shoppers | PageValues | 1 | True | 0.7156 | 0.9124 | 0.1968 |
| spambase | none | 0 | True | — | — | — |
| wine_quality | none | 0 | True | — | — | — |

## Feature-engineering evidence

The selection rule chooses the smallest matrix within 0.002 mean CV ROC-AUC of the best stage. This makes feature dilution a first-class negative result.

| Dataset | Raw CV ROC-AUC | Selected stage | Selected CV ROC-AUC | Selected − raw | Interpretation |
|---|---:|---|---:|---:|---|
| adult | 0.8955 | raw | 0.8955 | 0.0000 | raw was sufficient within tolerance |
| bank_marketing | 0.7130 | ratios | 0.7172 | 0.0042 | generated features helped |
| breast_cancer | 0.9907 | raw | 0.9907 | 0.0000 | raw was sufficient within tolerance |
| credit_default | 0.7317 | raw | 0.7317 | 0.0000 | raw was sufficient within tolerance |
| german_credit | 0.7527 | interactions | 0.7601 | 0.0073 | generated features helped |
| heart_disease | 0.8700 | raw | 0.8700 | 0.0000 | raw was sufficient within tolerance |
| mushroom | 1.0000 | raw | 1.0000 | 0.0000 | raw was sufficient within tolerance |
| online_shoppers | 0.7156 | raw | 0.7156 | 0.0000 | raw was sufficient within tolerance |
| spambase | 0.9869 | raw | 0.9869 | 0.0000 | raw was sufficient within tolerance |
| wine_quality | 0.8398 | ratios | 0.8429 | 0.0031 | generated features helped |

## Algorithm and optimization evidence

| Dataset | Selected family | Model-screen CV ROC-AUC | CV std | Accepted configuration | Accepted CV lift vs baseline | Holdout ROC-AUC | Bootstrap 95% interval | ECE |
|---|---|---:|---:|---|---:|---:|---|---:|
| adult | lightgbm | 0.8955 | 0.0065 | lightgbm_C02 | 0.0067 | 0.9097 | 0.8855–0.9297 | 0.0261 |
| bank_marketing | extra_trees | 0.7288 | 0.0295 | extra_trees_C02 | 0.0235 | 0.7718 | 0.7126–0.8290 | 0.0272 |
| breast_cancer | logistic_regression | 0.9955 | 0.0047 | baseline | 0.0000 | 0.9960 | 0.9858–1.0000 | 0.0302 |
| credit_default | extra_trees | 0.7363 | 0.0159 | extra_trees_C02 | 0.0189 | 0.7746 | 0.7326–0.8150 | 0.0462 |
| german_credit | extra_trees | 0.7769 | 0.0298 | extra_trees_C01 | 0.0099 | 0.7943 | 0.7180–0.8621 | 0.0824 |
| heart_disease | extra_trees | 0.8978 | 0.0244 | extra_trees_C02 | 0.0069 | 0.9448 | 0.8851–0.9872 | 0.1495 |
| mushroom | extra_trees | 1.0000 | 0.0000 | baseline | 0.0000 | 1.0000 | 1.0000–1.0000 | 0.0002 |
| online_shoppers | extra_trees | 0.7366 | 0.0072 | extra_trees_C02 | 0.0200 | 0.7394 | 0.6997–0.7828 | 0.0307 |
| spambase | hist_gradient_boosting | 0.9874 | 0.0021 | baseline | 0.0000 | 0.9849 | 0.9752–0.9925 | 0.0261 |
| wine_quality | extra_trees | 0.8609 | 0.0036 | baseline | 0.0000 | 0.8878 | 0.8610–0.9077 | 0.0331 |

## Feature-reliability evidence

These are importance-stability candidates, not automatically approved production features. Names still require decision-time, privacy, availability, and monitoring review.

| Dataset | Most stable CV features (top-10 frequency) |
|---|---|
| adult | `fnlwgt` (100%), `age` (100%), `hours-per-week` (100%), `occupation` (100%), `education-num` (100%) |
| bank_marketing | `poutcome` (100%), `log1p_poutcome` (100%), `month` (100%), `log1p_month` (100%), `housing` (100%) |
| breast_cancer | `texture3` (100%), `radius2` (100%), `symmetry3` (100%), `radius3` (100%), `area3` (100%) |
| credit_default | `X6` (100%), `X7` (100%), `X8` (100%), `X11` (100%), `X9` (100%) |
| german_credit | `Attribute1` (100%), `log1p_Attribute3` (100%), `Attribute3` (100%), `Attribute2` (100%), `log1p_Attribute2` (100%) |
| heart_disease | `thal` (100%), `ca` (100%), `exang` (100%), `cp` (100%), `sex` (100%) |
| mushroom | `gill-size` (100%), `odor` (100%), `spore-print-color` (100%), `bruises` (100%), `stalk-surface-below-ring` (100%) |
| online_shoppers | `ExitRates` (100%), `Month` (100%), `ProductRelated_Duration` (100%), `ProductRelated` (100%), `BounceRates` (100%) |
| spambase | importance unavailable |
| wine_quality | `alcohol` (100%), `log1p_volatile_acidity` (100%), `volatile_acidity` (100%), `density` (100%), `ratio_fixed_acidity_div_alcohol` (100%) |

## Cross-dataset evidence

The counts below describe this campaign only. They are empirical defaults to challenge on new data, not universal laws.

### Final model-family selections

- `extra_trees`: 7 dataset(s)
- `lightgbm`: 1 dataset(s)
- `logistic_regression`: 1 dataset(s)
- `hist_gradient_boosting`: 1 dataset(s)

### Selected feature recipes

- `raw`: 7 dataset(s)
- `ratios`: 2 dataset(s)
- `interactions`: 1 dataset(s)

## Interpretation boundaries

1. UCI datasets are real public data, but they are benchmarks—not substitutes for a production pilot's prediction-time contract.
2. Cached categoricals were factorized by the earlier downloader. Future ingestion should preserve raw categorical values and fit encoders on training data only.
3. Feature importance is not causality and not proof of production availability.
4. Small score differences require uncertainty and cost/latency interpretation before promotion.
5. The LLM review queue may challenge and propose; it may not rewrite observed metrics or approve deployment.

## Next evidence priorities

1. Re-run close decisions with repeated 5×5 CV and full available rows; this first completed pass analyzed up to 4,000 rows per dataset with 3-fold CV.
2. Re-ingest raw categorical values and compare train-fitted encoders or native categorical handling against the legacy globally factorized cache.
3. Add temporal, geographic, source-system, and out-of-domain holdouts where the data-generating process supports them.
4. Add explicit cost matrices and select thresholds on out-of-fold predictions, never on the final holdout.
5. Investigate calibration when ECE or Brier loss is operationally material; Heart Disease is an immediate calibration-review candidate in this pass.
6. Add subgroup/fairness reviews for demographic, credit, and health datasets before any real-world use.
7. Send the 50 bounded tasks in `llm_review_queue.jsonl` to a capable LLM critic, then store only cited, reviewed conclusions—not free-form chat—as durable memory.
