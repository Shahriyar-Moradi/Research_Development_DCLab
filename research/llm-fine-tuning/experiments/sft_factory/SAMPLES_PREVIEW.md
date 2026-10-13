# Training sample preview

One real sample per stage and decision, copied from `out/all_stages/train.jsonl`. The system prompt is the same in every sample and is shown once.

**Totals:** train 1,661 · val 258 · test 491

**System prompt:**

> You are the DCLab workflow agent. You build ML models step by step: data_understanding -> leakage_audit -> feature_engineering -> model_selection -> optimization_reliability. At each step you decide using only the evidence given, choose exactly one allowed action, explain why with the numbers, cite the rule, and say what happens next.

## data_understanding

### `deduplicate_before_split`  (mushroom, du_deduplicate_before_split)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 3,239 | FEATURES: 22 | POSITIVE RATE: 47.9%

PROFILE:
- rows: 3,239 | features: 22 | rows per feature: 147.2
- positive rate: 47.9% (minority class 47.9%)
- exact duplicate rows (features + target): 11.8%
- highest missing rates: `cap-shape` 0.0%, `cap-surface` 0.0%, `cap-color` 0.0%
- near-constant columns (one value in >99.5% of rows): 1

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: deduplicate_before_split
WHY: 11.8% of rows are exact duplicates, above the 1% threshold. If they are split randomly, copies of the same row land in both training and test folds, so the test score partly measures memory. Duplicates must be removed, or grouped so all copies stay in one fold, before any split.
RULE: DCLAB-R04 (duplicate/entity leakage) and DCLAB-R02 (split by the independence unit)
CONFIDENCE: high
NEXT: Deduplicate (or define groups), re-profile, then start the leakage audit.
```

### `impute_within_folds`  (credit_default, du_impute_within_folds)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 3,858 | FEATURES: 23 | POSITIVE RATE: 21.9%

PROFILE:
- rows: 3,858 | features: 23 | rows per feature: 167.7
- positive rate: 21.9% (minority class 21.9%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `X2` 57.2%, `X12` 55.1%, `X16` 14.5%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: impute_within_folds
WHY: `X2` is missing in 57.2% of rows, above the 5% threshold. Filling it with statistics from the whole dataset would leak test-fold information into training, so imputation (and a missing-value indicator, since absence may carry meaning) must be fitted inside each training fold.
RULE: DCLAB-R03 (fit imputers on training partitions only) and DCLAB-R12 (missingness as a state)
CONFIDENCE: high
NEXT: Add in-fold imputation and missing indicators to the pipeline, then start the leakage audit.
```

### `proceed`  (spambase, du_proceed)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 1,629 | FEATURES: 57 | POSITIVE RATE: 40.8%

PROFILE:
- rows: 1,629 | features: 57 | rows per feature: 28.6
- positive rate: 40.8% (minority class 40.8%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `word_freq_make` 0.0%, `word_freq_address` 0.0%, `word_freq_all` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: proceed
WHY: No threshold is crossed: duplicates 0.0%, minority class 40.8%, highest missing rate 0.0%, 1,629 rows (28.6 per feature). The data is ready for the leakage audit.
RULE: DCLAB-R01 (contract written, profile clean)
CONFIDENCE: high
NEXT: Write the profile to the decision ledger and start the leakage audit.
```

### `small_data_protocol`  (adult, du_small_data_protocol)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 313 | FEATURES: 14 | POSITIVE RATE: 23.3%

PROFILE:
- rows: 313 | features: 14 | rows per feature: 22.4
- positive rate: 23.3% (minority class 23.3%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `age` 0.0%, `workclass` 0.0%, `fnlwgt` 0.0%
- near-constant columns (one value in >99.5% of rows): 0

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: small_data_protocol
WHY: There are only 313 rows (22.4 per feature). Single splits on data this small swing widely, so use repeated stratified CV, start with simple regularized models, and report wide uncertainty instead of single scores.
RULE: DCLAB-R16 (paired, stable comparisons) and DCLAB-R13 (start from a transparent baseline)
CONFIDENCE: high
NEXT: Switch to repeated stratified CV with a linear baseline, then start the leakage audit.
```

### `use_imbalance_aware_evaluation`  (mushroom, du_use_imbalance_aware_evaluation)

**Input**

```
STAGE: data_understanding
Data understanding: profile the dataset and decide the first action before any split or model. Thresholds: duplicate rows above 1% | minority class below 10% | any column missing above 5% | fewer than 500 rows or fewer than 20 rows per feature.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
ROWS USED: 2,075 | FEATURES: 22 | POSITIVE RATE: 4.7%

PROFILE:
- rows: 2,075 | features: 22 | rows per feature: 94.3
- positive rate: 4.7% (minority class 4.7%)
- exact duplicate rows (features + target): 0.0%
- highest missing rates: `cap-shape` 0.0%, `cap-surface` 0.0%, `cap-color` 0.0%
- near-constant columns (one value in >99.5% of rows): 1

ALLOWED ACTIONS: proceed | deduplicate_before_split | use_imbalance_aware_evaluation | impute_within_folds | small_data_protocol
(proceed = no threshold crossed | deduplicate_before_split = remove or group duplicates before splitting | use_imbalance_aware_evaluation = stratified folds, PR-AUC and cost-based thresholds | impute_within_folds = fit imputation (plus missing indicators) inside training folds only | small_data_protocol = repeated stratified CV, simple models first, wide uncertainty)
```

**Answer**

```
DECISION: use_imbalance_aware_evaluation
WHY: The minority class is only 4.7% of rows, below the 10% threshold. Accuracy and even ROC-AUC can look good while the model misses most positives, so evaluation needs stratified folds, PR-AUC, and a threshold chosen from the cost of each error.
RULE: DCLAB-R18 (metrics and thresholds from the decision cost)
CONFIDENCE: high
NEXT: Fix the evaluation plan (stratified folds, PR-AUC, cost-based threshold) before the leakage audit.
```

## leakage_audit

### `exclude`  (spambase, date_after_decision)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS: 4,601 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 39.4% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.827

COLUMN: `last_status_update`
DATA DICTIONARY: Date of the most recent change to the record.
EVIDENCE:
- type: date (stored as days relative to the decision moment)
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 12.1% of rows
- univariate ROC-AUC: 0.727
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- values after the decision moment: 17.2%
- 3-fold CV ROC-AUC with the column: 0.993 | without: 0.988 | change: +0.004

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: exclude
WHY: 17% of `last_status_update` values fall after the decision moment, and those late dates are concentrated in one class (univariate ROC-AUC 0.727). At prediction time these dates do not exist yet, so the column leaks the future.
RULE: DCLAB-R04 (future leakage)
CONFIDENCE: high
NEXT: Remove `last_status_update` from the feature list, record the reason in the decision ledger, and continue the audit.
```

### `flag_for_review`  (spambase, strong_signal_unknown_origin)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS: 4,601 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 39.4% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.827

COLUMN: `tier_code`
DATA DICTIONARY: no entry
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 100.0% of rows
- univariate ROC-AUC: 0.978
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.997 | without: 0.988 | change: +0.009

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: flag_for_review
WHY: `tier_code` is unusually strong (univariate ROC-AUC 0.978 vs 0.827 for the strongest real feature) and raises CV ROC-AUC by +0.009, but there is no data-dictionary entry saying when it is recorded. A strong score alone does not prove leakage, and it does not prove availability either, so a person with the source lineage has to confirm it.
RULE: DCLAB-R05 (high AUC starts a review, it does not prove leakage)
CONFIDENCE: medium
NEXT: Keep `tier_code` out of all models until the data owner confirms when it is recorded; continue the audit meanwhile.
```

### `keep`  (spambase, date_before_decision)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS: 4,601 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 39.4% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.827

COLUMN: `updated_on`
DATA DICTIONARY: Date of the most recent change to the record.
EVIDENCE:
- type: date (stored as days relative to the decision moment)
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 34.6% of rows
- univariate ROC-AUC: 0.522
- identical to the target (or its inverse): no
- name matches the suspicious-word list: no
- values form a run of consecutive integers (row-number pattern): no
- values after the decision moment: 0.0%
- 3-fold CV ROC-AUC with the column: 0.987 | without: 0.988 | change: -0.001

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: keep
WHY: All `updated_on` values are before the decision moment (0% after), and the column carries no outcome signal (univariate ROC-AUC 0.522). It is safe to use; whether it is useful is a feature-engineering question, not a leakage one.
RULE: DCLAB-R04 (checked: no future leakage)
CONFIDENCE: high
NEXT: Keep `updated_on` and continue the audit with the next column.
```

### `rebuild_within_folds`  (spambase, full_data_target_encoding)

**Input**

```
STAGE: leakage_audit
Leakage audit: decide whether one column may be used by a model that predicts at the declared decision moment. A column must exist at that moment and must not be derived from the outcome or from information across train/test splits.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS: 4,601 (evidence computed on a stratified sample of 4,000) | POSITIVE RATE: 39.4% | STRONGEST REAL FEATURE (univariate ROC-AUC): 0.827

COLUMN: `word_freq_over_target_rate`
DATA DICTIONARY: Average outcome rate for each value of `word_freq_over`, computed over the full dataset before splitting.
EVIDENCE:
- type: numeric
- missing: 0.0% (positives 0.0%, negatives 0.0%)
- unique values: 0.9% of rows
- univariate ROC-AUC: 0.655
- identical to the target (or its inverse): no
- name matches the suspicious-word list: yes
- values form a run of consecutive integers (row-number pattern): no
- 3-fold CV ROC-AUC with the column: 0.988 | without: 0.988 | change: -0.000

ALLOWED ACTIONS: keep | exclude | flag_for_review | rebuild_within_folds
(keep = available at the decision moment, no leakage evidence | exclude = not available at the decision moment, a copy or consequence of the outcome, or an ordering/identifier artifact | flag_for_review = strong signal but availability cannot be confirmed from the evidence | rebuild_within_folds = valid idea, but computed using all rows; recompute inside each training fold)
```

**Answer**

```
DECISION: rebuild_within_folds
WHY: Encoding `word_freq_over` by its outcome rate is a legitimate idea, but this version was computed on all rows, so every evaluation row's own label is inside its feature value. Here the visible effect is small (CV change -0.000) because each group is large, but the method is still wrong and the same mistake on a high-cardinality column produces large fake gains. The fix is to recompute the rate inside each training fold, not to drop the idea.
RULE: DCLAB-R04 (cross-split aggregate leakage)
CONFIDENCE: high
NEXT: Drop this version of `word_freq_over_target_rate`, recompute it out-of-fold inside each training split, and re-audit it.
```

## feature_engineering

### `adopt_interactions`  (credit_default, fe_sim_interaction_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 23 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 23 | 0.6144 | 0.0135 | — | — |
| log_transforms | 29 | 0.6195 | 0.0129 | +0.0051 | 4/5 |
| ratios | 29 | 0.6116 | 0.0145 | -0.0027 | 2/5 |
| interactions | 29 | 0.6407 | 0.0164 | +0.0263 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_interactions
WHY: `interactions` improves on raw by +0.0263 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + interactions feature set into model selection and log each new feature's formula.
```

### `adopt_log_transforms`  (spambase, fe_sim_interaction_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 57 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 57 | 0.5374 | 0.0230 | — | — |
| log_transforms | 114 | 0.5494 | 0.0217 | +0.0120 | 5/5 |
| ratios | 63 | 0.5376 | 0.0242 | +0.0002 | 3/5 |
| interactions | 63 | 0.5422 | 0.0217 | +0.0048 | 5/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_log_transforms
WHY: `log_transforms` improves on raw by +0.0120 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 57 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + log_transforms feature set into model selection and log each new feature's formula.
```

### `adopt_ratios`  (adult, fe_sim_interaction_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 14 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 14 | 0.5755 | 0.0127 | — | — |
| log_transforms | 19 | 0.5775 | 0.0130 | +0.0020 | 3/5 |
| ratios | 20 | 0.5824 | 0.0150 | +0.0069 | 5/5 |
| interactions | 20 | 0.5874 | 0.0192 | +0.0119 | 4/5 |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: adopt_ratios
WHY: `ratios` improves on raw by +0.0069 mean paired ROC-AUC and wins on 5/5 folds, which clears the +0.005 bar consistently, not by luck on one fold. It adds 6 features with a recorded formula, so its lineage stays auditable.
RULE: DCLAB-R07 (promote the smallest defensible set within tolerance) and DCLAB-R16 (paired margin)
CONFIDENCE: high
NEXT: Carry the raw + ratios feature set into model selection and log each new feature's formula.
```

### `keep_raw`  (mushroom, fe_sim_interaction_logistic_regression)

**Input**

```
STAGE: feature_engineering
Feature engineering: compare engineered recipes against the raw baseline on the same folds with the same model. Adopt a recipe only if its mean paired gain over raw is at least +0.005 ROC-AUC AND it improves on every fold. If several qualify, take the largest gain; if two gains are within 0.002, take the one with fewer features.

DATASET: Mushroom (mushroom). Predicts whether a mushroom is poisonous.
DECISION MOMENT: Classification after recorded physical traits are observed; never an edibility/safety recommendation.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 3,000 | FEATURES: 22 | POSITIVE RATE: 50.0%

MODEL USED FOR THE COMPARISON: logistic_regression | 5 stratified folds, identical for every recipe
RESULTS:
| recipe | features | mean ROC-AUC | std | mean paired gain vs raw | folds improved |
|---|---|---|---|---|---|
| raw | 22 | 0.8248 | 0.0156 | — | — |

ALLOWED ACTIONS: keep_raw | adopt_log_transforms | adopt_ratios | adopt_interactions
(keep_raw = no recipe clears the bar | adopt_log_transforms / adopt_ratios / adopt_interactions = adopt that recipe for model selection)
```

**Answer**

```
DECISION: keep_raw
WHY: No engineered recipe clears the bar. Extra features without a real gain only add maintenance and drift risk, so the raw baseline stays.
RULE: DCLAB-R07 (start from a strong raw baseline) and DCLAB-R16 (demand a predeclared margin)
CONFIDENCE: high
NEXT: Carry the raw feature set into model selection.
```

## model_selection

### `extra_trees`  (bank_marketing, ms_real_300)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 300 | FEATURES: 15 | POSITIVE RATE: 11.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.6765 | 0.0443 | 0.6655 | 0.02 |
| random_forest | 0.6694 | 0.0582 | 0.6548 | 0.76 |
| extra_trees | 0.6966 | 0.0330 | 0.6884 | 0.67 |
| hist_gradient_boosting | 0.5952 | 0.0125 | 0.5920 | 0.22 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: extra_trees
WHY: `extra_trees` has the highest stability-adjusted score (0.6884: mean 0.6966, std 0.0330), ahead of `logistic_regression` at 0.6655. The linear baseline scores 0.6655, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `extra_trees` into optimization; keep `logistic_regression` as the paired comparison when tuning.
```

### `hist_gradient_boosting`  (bank_marketing, ms_sim_linear_full)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
NOTE: Training-variety scenario: the outcome was replaced by a simulated one built from the real columns; the mechanism is not disclosed.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 50.0%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9532 | 0.0056 | 0.9518 | 0.03 |
| random_forest | 0.9670 | 0.0053 | 0.9657 | 1.20 |
| extra_trees | 0.9426 | 0.0056 | 0.9412 | 1.00 |
| hist_gradient_boosting | 0.9663 | 0.0048 | 0.9651 | 0.72 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: hist_gradient_boosting
WHY: `hist_gradient_boosting` and `random_forest` are effectively tied on stability-adjusted score (0.9651 vs 0.9657, within 0.002), so the clearly faster one goes forward: `hist_gradient_boosting` takes 0.72s vs 1.20s. A score gap this small is noise, while a speed gap this large is real.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: medium
NEXT: Carry `hist_gradient_boosting` into optimization; keep `random_forest` as the paired comparison when tuning.
```

### `logistic_regression`  (breast_cancer, ms_real_300)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 300 | FEATURES: 30 | POSITIVE RATE: 38.3%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.9992 | 0.0006 | 0.9990 | 0.03 |
| random_forest | 0.9958 | 0.0004 | 0.9957 | 0.77 |
| extra_trees | 0.9963 | 0.0005 | 0.9962 | 0.61 |
| hist_gradient_boosting | 0.9925 | 0.0057 | 0.9911 | 0.26 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: logistic_regression
WHY: `logistic_regression` has the highest stability-adjusted score (0.9990: mean 0.9992, std 0.0006), ahead of `extra_trees` at 0.9962. The linear baseline wins here, which is exactly why it is always screened: no family is best on every dataset.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `logistic_regression` into optimization; keep `extra_trees` as the paired comparison when tuning.
```

### `random_forest`  (credit_default, ms_real_800)

**Input**

```
STAGE: model_selection
Model selection: screen a linear baseline and several nonlinear families on identical folds. Rank by stability-adjusted score = mean ROC-AUC − 0.25 × fold std. If the top two scores are within 0.002 and one is at least 1.5× faster, carry forward the faster one; if they are within 0.002 and similar in speed, keep the higher score but mark the choice as low confidence.

DATASET: Default of Credit Card Clients (credit_default). Predicts whether a client defaults on next month's credit-card payment.
DECISION MOMENT: End of the recorded billing period, predicting next-month default; confirm X1-X23 mapping and event times before deployment.
ROWS USED: 800 | FEATURES: 23 | POSITIVE RATE: 20.6%

SCREEN: 3 stratified folds, identical for every model, same raw feature set
| model | mean ROC-AUC | std | stability-adjusted score | fit+predict seconds |
|---|---|---|---|---|
| logistic_regression | 0.6932 | 0.0443 | 0.6822 | 0.04 |
| random_forest | 0.7308 | 0.0276 | 0.7239 | 0.91 |
| extra_trees | 0.7238 | 0.0229 | 0.7181 | 0.72 |
| hist_gradient_boosting | 0.6997 | 0.0334 | 0.6913 | 0.51 |

ALLOWED ACTIONS: logistic_regression | random_forest | extra_trees | hist_gradient_boosting
(choose the one model family to carry into optimization)
```

**Answer**

```
DECISION: random_forest
WHY: `random_forest` has the highest stability-adjusted score (0.7239: mean 0.7308, std 0.0276), ahead of `extra_trees` at 0.7181. The linear baseline scores 0.6822, so the nonlinear family earns its extra complexity.
RULE: DCLAB-R13 (no universal best family; screen a linear baseline) and DCLAB-R14 (stability and runtime, not ROC-AUC alone)
CONFIDENCE: high
NEXT: Carry `random_forest` into optimization; keep `extra_trees` as the paired comparison when tuning.
```

## optimization_reliability

### `accept_result`  (breast_cancer, opt_holdout_random)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 569 | FEATURES: 30 | POSITIVE RATE: 37.3%

MODEL: hist_gradient_boosting (locked) | training rows: 457 | holdout rows: 112 (holdout positive rate 43.8%)
- 3-fold CV ROC-AUC on training: 0.9890 (std 0.0073)
- holdout ROC-AUC (scored once): 0.9948
- difference, CV minus holdout: -0.0058
- largest feature shifts, training vs holdout (PSI): `symmetry2` 0.298, `fractal_dimension1` 0.233, `area3` 0.198

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: accept_result
WHY: CV ROC-AUC 0.9890 and holdout ROC-AUC 0.9948 differ by -0.0058, inside the ±0.02 band, so the CV estimate held up on untouched data. `symmetry2` does shift (PSI 0.298), so it goes on the monitoring list, but it did not hurt performance here.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Report CV and holdout together with uncertainty, and do not tune anything on the holdout.
```

### `calibrate_isotonic`  (spambase, opt_calibration_logistic_regression)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Spambase (spambase). Predicts whether an email is spam.
DECISION MOMENT: Email classification after message receipt. Sender-specific proxies and distribution shift remain risks.
ROWS USED: 900 | FEATURES: 57 | POSITIVE RATE: 36.4%

MODEL: logistic_regression | 3 stratified folds | training rows per fold: about 600
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0420 | 0.9550 | 0.0703 |
| sigmoid | 0.0744 | 0.9622 | 0.0735 |
| isotonic | 0.0260 | 0.9619 | 0.0626 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_isotonic
WHY: Raw `logistic_regression` probabilities are off by an ECE of 0.0420, above the 0.03 bar. Isotonic calibration fitted inside the training folds brings ECE to 0.0260 with ROC-AUC 0.9619 vs 0.9550 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add isotonic calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `calibrate_sigmoid`  (breast_cancer, opt_calibration_extra_trees)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Breast Cancer Wisconsin (Diagnostic) (breast_cancer). Predicts whether a tumour is malignant.
DECISION MOMENT: Research diagnosis after all cell-nuclei measurements are available. Not an early screening model or a clinical validation.
ROWS USED: 569 | FEATURES: 30 | POSITIVE RATE: 37.3%

MODEL: extra_trees | 3 stratified folds | training rows per fold: about 379
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0494 | 0.9939 | 0.0301 |
| sigmoid | 0.0279 | 0.9934 | 0.0261 |
| isotonic | 0.0101 | 0.9889 | 0.0258 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: calibrate_sigmoid
WHY: Raw `extra_trees` probabilities are off by an ECE of 0.0494, above the 0.03 bar. Sigmoid calibration fitted inside the training folds brings ECE to 0.0279 with ROC-AUC 0.9934 vs 0.9939 raw, so ranking is kept while probabilities become trustworthy.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Add sigmoid calibration inside the pipeline, lock it, and move to the final holdout check.
```

### `investigate_distribution_shift`  (adult, opt_holdout_shifted)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Adult (Census Income) (adult). Predicts whether a person earns more than $50K per year.
DECISION MOMENT: Retrospective census-income classification; sampling weight fnlwgt and sensitive attributes need separate review.
ROWS USED: 4,000 | FEATURES: 14 | POSITIVE RATE: 23.9%

MODEL: hist_gradient_boosting (locked) | training rows: 3,000 | holdout rows: 1,000 (holdout positive rate 36.3%)
- 3-fold CV ROC-AUC on training: 0.9082 (std 0.0151)
- holdout ROC-AUC (scored once): 0.8629
- difference, CV minus holdout: +0.0453
- largest feature shifts, training vs holdout (PSI): `age` 11.889, `marital-status` 1.010, `relationship` 0.706

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_distribution_shift
WHY: The holdout ROC-AUC (0.8629) is 0.0453 below CV (0.9082), outside the ±0.02 band, and `age` has a PSI of 11.889 (above 0.2): the holdout population is different from training. The CV number does not describe this population, so the shift must be explained before any claim is made.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Do not tune on the holdout. Slice results by the shifted feature, document the gap, and decide with the data owner.
```

### `investigate_gap_without_shift`  (bank_marketing, opt_holdout_random)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, final holdout: the holdout is scored once, after every choice is locked. Accept if |CV ROC-AUC − holdout ROC-AUC| ≤ 0.02. Otherwise, if any feature's population stability index (PSI) between training and holdout is above 0.2, investigate distribution shift; if not, investigate the gap without shift.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 11.7%

MODEL: hist_gradient_boosting (locked) | training rows: 3,031 | holdout rows: 969 (holdout positive rate 10.8%)
- 3-fold CV ROC-AUC on training: 0.7394 (std 0.0167)
- holdout ROC-AUC (scored once): 0.7177
- difference, CV minus holdout: +0.0217
- largest feature shifts, training vs holdout (PSI): `job` 0.021, `age` 0.016, `month` 0.010

ALLOWED ACTIONS: accept_result | investigate_distribution_shift | investigate_gap_without_shift
(accept_result | investigate_distribution_shift | investigate_gap_without_shift)
```

**Answer**

```
DECISION: investigate_gap_without_shift
WHY: CV (0.7394) and holdout (0.7177) differ by +0.0217, outside the ±0.02 band, yet no feature shifts much (highest PSI 0.021). The holdout scored lower than CV without a population change, which points to choices fitted to CV noise (optimistic CV) or an unstable estimate on 969 rows, so the gap needs explaining before results are reported.
RULE: DCLAB-R17 (holdout scored once, never tuned on) and DCLAB-R19 (check shift slices)
CONFIDENCE: high
NEXT: Do not tune on the holdout. Put a bootstrap interval on the holdout score, review which earlier choices were made on CV, and report both numbers with the explanation.
```

### `keep_uncalibrated`  (bank_marketing, opt_calibration_hist_gradient_boosting)

**Input**

```
STAGE: optimization_reliability
Optimization and reliability, probability calibration: compare expected calibration error (ECE, 10 bins) of the raw model with sigmoid and isotonic calibration fitted inside the training folds. Calibrate only if raw ECE is above 0.03 AND the better method cuts ECE by at least 25% without losing more than 0.002 ROC-AUC. Prefer sigmoid when the training folds have fewer than 1,000 rows or isotonic is better by less than 0.005 ECE.

DATASET: Bank Marketing (bank_marketing). Predicts whether a client subscribes to a term deposit.
DECISION MOMENT: Immediately before a marketing call. Duration is post-call and forbidden. Contact schedule and prior history are assumed known; campaign counter needs source verification.
ROWS USED: 4,000 | FEATURES: 15 | POSITIVE RATE: 11.7%

MODEL: hist_gradient_boosting | 3 stratified folds | training rows per fold: about 2,666
| probabilities | ECE | ROC-AUC | Brier |
|---|---|---|---|
| raw | 0.0506 | 0.7244 | 0.0934 |
| sigmoid | 0.1663 | 0.3309 | 0.1495 |
| isotonic | 0.0200 | 0.6915 | 0.0973 |

ALLOWED ACTIONS: keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic
(keep_uncalibrated | calibrate_sigmoid | calibrate_isotonic)
```

**Answer**

```
DECISION: keep_uncalibrated
WHY: Raw `hist_gradient_boosting` probabilities are already usable or calibration does not pay: raw ECE is 0.0506, and neither method cuts it by 25% without losing ROC-AUC (sigmoid 0.1663, isotonic 0.0200). Adding a calibration layer would add complexity for no measured benefit.
RULE: DCLAB-R14 (probability quality, not ROC-AUC alone) and DCLAB-R18 (thresholds need trustworthy probabilities)
CONFIDENCE: high
NEXT: Lock the uncalibrated model and move to the final holdout check.
```
