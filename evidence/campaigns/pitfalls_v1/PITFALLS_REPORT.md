# Pitfalls campaign: what common notebook mistakes really cost

Each mistake was run the wrong way and the right way on identical data. The difference is how much the reported score lies. These measurements back the severity and the proof shown by the notebook copilot.

| ID | Mistake | Measured effect |
|---|---|---|
| PIT-001 | preprocess before split | Fitting mean-imputation and scaling before the split changed holdout ROC-AUC by at most 0.0001 on average (german_credit). |
| PIT-002 | selection before cv | On pure noise with random labels, selecting 20 of 2000 features before CV produced ROC-AUC 0.859; doing it inside the folds gave 0.512 (chance is 0.5). |
| PIT-003 | oversample before split | Oversampling before the split inflated holdout ROC-AUC by up to +0.3003 on average (online_shoppers: 0.9019 reported vs 0.6016 honest). |
| PIT-004 | target encoding before split | Encoding customerID (unique per row) with the target before the split gave ROC-AUC 1.0000 on telco_churn versus 0.8160 honestly (inflation +0.1840). |
| PIT-005 | random split on time data | A random split reported ROC-AUC 0.9332; training on the past and testing on the last 20% of time gave 0.9371 (random minus time-ordered -0.0038). Here the random split was NOT optimistic: the process was stable enough over this window that a forward-in-time test scored as well or better. |
| PIT-006 | holdout reuse for selection | Choosing the best of 30 configurations by holdout score overstated ROC-AUC by +0.0072 on average and by +0.0177 on german_credit; on the untouched holdout the pick ranked 7.2 of 30 on average. |

## PIT-001 · preprocess before split

**Question.** How much does fitting an imputer and scaler on all rows before the train/test split inflate the holdout score?

**Setup.** Logistic regression with mean imputation + standard scaling on 6 datasets (≤8000 rows each), 3 random stratified 80/20 splits; wrong = preprocessing fitted on all rows, right = fitted on training rows only.

- **fact**: Fitting mean-imputation and scaling before the split changed holdout ROC-AUC by at most 0.0001 on average (german_credit).
- **recommendation**: Still fit every preprocessing step inside the training data (a Pipeline): the measured cost is small here, but the identical mistake with feature selection, oversampling or target encoding is large (PIT-002, PIT-003, PIT-004).

Limitations: Simple, unsupervised, row-wise transforms only; learned or target-aware transforms behave very differently.

Evidence: `evidence/campaigns/pitfalls_v1/results/PIT-001_preprocess_before_split.json`

## PIT-002 · selection before cv

**Question.** How much does selecting features on all rows before cross-validation inflate the CV score?

**Setup.** SelectKBest(f_classif) + logistic regression, 5-fold CV, 3 seeds. A pure-noise canary (random labels, 2000 features, 200 rows) plus three real datasets (≤4000 rows). Wrong = selection fitted once on all rows; right = selection inside each training fold.

- **fact**: On pure noise with random labels, selecting 20 of 2000 features before CV produced ROC-AUC 0.859; doing it inside the folds gave 0.512 (chance is 0.5).
- **fact**: On the real datasets (few features, thousands of rows) the inflation was at most +0.0041.
- **recommendation**: Put feature selection inside the cross-validated pipeline. The danger grows with the number of candidate features relative to rows, so wide data (text, genomics, many engineered features) is where this mistake fabricates results.

Limitations: Univariate filter selection only; wrapper/model-based selection can inflate more.

Evidence: `evidence/campaigns/pitfalls_v1/results/PIT-002_selection_before_cv.json`

## PIT-003 · oversample before split

**Question.** How much does oversampling the minority class before the split inflate the holdout score?

**Setup.** Random minority oversampling to a 50/50 balance, random forest (200 trees), 4 imbalanced datasets (≤6000 rows), 3 seeds. Wrong = oversample then split; right = split, oversample the training rows only.

- **fact**: Oversampling before the split inflated holdout ROC-AUC by up to +0.3003 on average (online_shoppers: 0.9019 reported vs 0.6016 honest).
- **recommendation**: Resample only inside the training data (imblearn Pipeline or after the split), and evaluate on untouched, naturally imbalanced rows with a metric that respects imbalance such as average precision.

Limitations: Random duplication oversampling; SMOTE interpolates new points and leaks through near-duplicates instead of exact copies.

Evidence: `evidence/campaigns/pitfalls_v1/results/PIT-003_oversample_before_split.json`

## PIT-004 · target encoding before split

**Question.** How much does computing target-mean encodings on all rows before the split inflate the holdout score?

**Setup.** Mean-target encoding (smoothing 1) of one categorical column plus a few numeric columns, histogram gradient boosting, 80/20 stratified split, 3 seeds. Wrong = encoding computed on all rows; right = out-of-fold encodings for training rows, training statistics applied to test rows.

- **fact**: Encoding customerID (unique per row) with the target before the split gave ROC-AUC 1.0000 on telco_churn versus 0.8160 honestly (inflation +0.1840).
- **fact**: For a low-cardinality column (deliverey_category_id on hyperack) the inflation was +0.0019.
- **recommendation**: Compute target encodings out-of-fold on training data only, and never target-encode identifiers; an identifier that looks predictive is memorizing labels. Even encoding training rows with their own label (without out-of-fold) teaches the model to trust a column that is useless on new rows.

Limitations: One encoding scheme and smoothing value; out-of-fold encoders reduce but do not remove the risk for rare categories.

Evidence: `evidence/campaigns/pitfalls_v1/results/PIT-004_target_encoding_before_split.json`

## PIT-005 · random split on time data

**Question.** On data that arrives over time, how different is a random holdout from a train-on-past, test-on-future holdout?

**Setup.** HyperAck orders (11,107 rows, 2022-06-29 to 2022-11-14), 11 leakage-safe features, LightGBM. Random = stratified 80/20 over 3 seeds; time-ordered = train on the first 80% by timestamp, test on the last 20% (from 2022-10-16).

- **fact**: A random split reported ROC-AUC 0.9332; training on the past and testing on the last 20% of time gave 0.9371 (random minus time-ordered -0.0038). Here the random split was NOT optimistic: the process was stable enough over this window that a forward-in-time test scored as well or better.
- **recommendation**: When rows have timestamps and the model will score future rows, validate forward in time (TimeSeriesSplit or a time cut-off) and report that number; a random split answers a different question. Do not assume the direction of the gap: measure it, because a stable process can show none.

Limitations: One dataset and one cut-off; the gap depends on how fast the process drifts.

Evidence: `evidence/campaigns/pitfalls_v1/results/PIT-005_random_split_on_time_data.json`

## PIT-006 · holdout reuse for selection

**Question.** If we pick the best of many configurations by their holdout score, how optimistic is that holdout score?

**Setup.** 30 random LightGBM configurations per run on a 60/20/20 split into train and two holdouts, 5 datasets (≤6000 rows), 3 seeds, both selection directions. Bias = the picked model's score drop from the selection holdout to the untouched one, minus the average drop across all configurations.

- **fact**: Choosing the best of 30 configurations by holdout score overstated ROC-AUC by +0.0072 on average and by +0.0177 on german_credit; on the untouched holdout the pick ranked 7.2 of 30 on average.
- **recommendation**: Tune with cross-validation on training data, then touch the holdout once. If the holdout was used to choose, the reported score is biased upward; confirm on a fresh split or use nested CV. The bias grows with the number of configurations tried and shrinks with holdout size.

Limitations: Configurations from one model family are highly correlated, which keeps the bias small; comparing many unrelated pipelines inflates more.

Evidence: `evidence/campaigns/pitfalls_v1/results/PIT-006_holdout_reuse_for_selection.json`

Re-run: `python -m dclab_rnd.pitfalls run --force`.
