# Campaign Report — expansion_v1

Task-type-aware extension of `model_building_50_v1` to four task types it lacked. Completed experiments: **20 / 20**; summed stage runtime **8.3 min**.

Each dataset ran five stages in order. Stages 1–4 select on training rows only (time-ordered, grouped, or stratified CV). Stage 5 consumes the locked holdout once. Every number below comes from a result JSON in `results/`; claims and evidence paths are in `agent_memory.jsonl`.

## Overview

| Dataset | Task | Primary metric | Leakage lift (CV) | Recipe | Model / config | Holdout (95% CI) | Naive baseline |
|---|---|---|---|---|---|---|---|
| credit_card_fraud | binary_imbalanced | average precision (PR-AUC) | -0.0042 (Time) | raw | extra_trees / extra_trees_baseline | 0.8139 (0.7317–0.8847) | random-ranking AP 0.0013; majority accuracy 0.9987 |
| letter_recognition | multiclass | macro-F1 | +0.0000 (none blocked) | raw | extra_trees / extra_trees_C01 | 0.9751 (0.9704–0.9798) | majority macro-F1 0.0030 |
| bike_sharing_daily | timeseries_regression | MAE | +413.6 (casual, registered) | raw | xgboost / xgboost_C01 | 680.0 (484.0–890.1) | lag-2 MAE 1,156; lag-7 1,203; mean 2,270 |
| ecommerce_clothing_reviews | text_tabular_binary | ROC-AUC | +0.0371 (Rating, Positive Feedback Count) | tfidf_word | logistic_regression / logistic_regression_C01 | 0.9459 (0.9391–0.9525) | tabular-only AUC 0.5480 |

## Credit Card Fraud Detection (ULB / Kaggle) — `credit_card_fraud` (binary_imbalanced)

Two days of September-2013 European card transactions (28 PCA components V1-V28, Amount, elapsed Time) used to predict whether a transaction is fraudulent (Class=1, 0.17% of rows).

*Decision-time contract:* Score each transaction at authorization time using only that transaction's own attributes (V1-V28, Amount, and time-of-day derived from its timestamp); the dataset-relative elapsed `Time` counter itself is not a production feature.

*Holdout policy:* Locked holdout = last 20% of rows in `Time` order; recipe/model/parameter choices use only expanding-window 3-fold time-ordered CV on the earlier rows. The holdout is consumed once, in optimization_reliability.

| Exp | Stage | What was compared | Key claim |
|---|---|---|---|
| EXP-051 | data_understanding | Profiled 227846 training rows × 30 raw columns of credit_card_fraud (holdout: last 56961 of 284807 rows by `Time`); computed target summary, per-column missingness/uniqueness/PSI, duplicates, time range; 6 columns flagged for review. | credit_card_fraud has 284807 source rows and 30 raw columns; the locked holdout is last 56961 of 284807 rows by `Time` (56961 rows), leaving 227846 training rows with a training positive rate of 0.0018 (417 positives vs 227429 negatives). |
| EXP-052 | leakage_audit | Scanned 60000 training rows with binary_imbalanced heuristics (univariate signal, exact proxy/identity, name, uniqueness) and synthetic canaries; compared lightgbm on the raw recipe with vs without blocked ['Time'] on expanding-window TimeSeriesSplit(3) on time-ordered training rows: average precision (PR-AUC) 0.7804±0.0192 safe vs 0.7763±0.0180 unsafe (apparent lift -0.0042). | Declared decision-time exclusions for credit_card_fraud: ['Time']. Including them moves training-CV average precision (PR-AUC) (lightgbm, raw recipe) from 0.7804±0.0192 to 0.7763±0.0180, i.e. no apparent lift (-0.0042); the exclusion rests on decision-time semantics, not on CV inflation. |
| EXP-053 | feature_engineering | 4 recipes compared with lightgbm on expanding-window TimeSeriesSplit(3) on time-ordered training rows: raw (29 feats, 0.7804±0.0192); log_amount (30 feats, 0.7804±0.0192); hour_of_day (32 feats, 0.7778±0.0212); log_amount_hour (33 feats, 0.7778±0.0212). Selected: raw. | Use the `raw` recipe (V1-V28 + Amount (absolute Time blocked)) for credit_card_fraud: 0.7804±0.0192 training-CV average precision (PR-AUC) with 29 features; the best recipe was `raw` at 0.7804. |
| EXP-054 | model_selection | 5 model families screened on the `raw` recipe with identical expanding-window TimeSeriesSplit(3) on time-ordered training rows, ranked by mean - 0.25×std average precision (PR-AUC): extra_trees 0.7858±0.0205 (5.6s); lightgbm 0.7804±0.0192 (2.6s); xgboost 0.7760±0.0138 (2.0s); hist_gradient_boosting 0.7462±0.0589 (1.1s); logistic_regression 0.7238±0.0924 (1.0s). Selected: extra_trees. | extra_trees is the training-CV model candidate for credit_card_fraud on the `raw` recipe (0.7858±0.0205 average precision (PR-AUC), adjusted 0.7807); runner-up lightgbm at 0.7804±0.0192. |
| EXP-055 | optimization_reliability | Compared default extra_trees vs 3 explicit candidates on the `raw` recipe (expanding-window TimeSeriesSplit(3) on time-ordered training rows): extra_trees_baseline 0.7858±0.0205; extra_trees_C01 0.7810±0.0169; extra_trees_C02 0.7810±0.0184; extra_trees_C03 0.7706±0.0284; tuning rejected (gain -0.0048 vs required 0.0050). Final fit on 227846 training rows, scored once on the holdout (last 56961 of 284807 rows by `Time`): average precision (PR-AUC) 0.8139 [0.7317, 0.8847]. | The preselected extra_trees/extra_trees_baseline on the `raw` recipe scored average precision (PR-AUC) 0.8139 on the once-consumed holdout (last 56961 of 284807 rows by `Time`; iid row bootstrap 95% interval 0.7317–0.8847); training-CV mean was 0.7858. |

**Holdout claims:**

- `EXP-055-C1` (fact): The preselected extra_trees/extra_trees_baseline on the `raw` recipe scored average precision (PR-AUC) 0.8139 on the once-consumed holdout (last 56961 of 284807 rows by `Time`; iid row bootstrap 95% interval 0.7317–0.8847); training-CV mean was 0.7858.
- `EXP-055-C2` (decision): Tuning rejected: best candidate extra_trees_C01 changed mean training-CV average precision (PR-AUC) by -0.0048 (required ≥ 0.0050), so extra_trees_baseline was used.
- `EXP-055-C3` (fact): Accuracy is uninformative on the holdout: the model's 0.9994 at threshold 0.5 compares with 0.9987 for always predicting 'not fraud', while average precision is 0.8139 vs 0.0013 for a random ranking (ROC-AUC 0.9891).
- `EXP-055-C4` (decision): A threshold of 0.2548 chosen on training out-of-fold scores to reach precision ≥ 0.90 (OOF recall 0.750) gives holdout precision 0.963 and recall 0.693 (52 of 75 frauds, 54 alerts); a holdout-chosen (oracle) threshold would reach recall 0.720 at precision ≥ 0.90. Precision at top-100 scores is 0.600.
- `EXP-055-C5` (recommendation): Do not promote features from importance alone; require decision-time availability, cross-fold stability, production availability, and monitoring.

**Limitations:**

- A cross-validation leader is not a universal algorithm winner and has not yet seen the locked holdout.
- CV models are fitted on part of the training rows while the final model uses all of them, so the holdout can legitimately beat the CV mean; a single holdout is one draw.
- Name, univariate-signal, identity, and uniqueness rules propose review; only decision-time semantics confirm leakage.
- Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search.
- PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability.
- Recipes were compared with a single default-parameter model; a different family could rank them differently.
- Strong signal alone is not leakage; e.g. legitimate lags or text sentiment are expected to be predictive.
- The size of this effect cannot be measured from the released data.
- V1-V28 are PCA components released already fitted on the full two-day extract (including the holdout period); this cannot be undone downstream.
- With only ~75 holdout frauds, precision/recall at a single threshold moves by several points per misclassified case.

## Letter Recognition (UCI via PMLB) — `letter_recognition` (multiclass)

20,000 distorted capital-letter glyph images summarized by 16 integer shape statistics, used to predict which of the 26 letters (A-Z) the glyph shows.

*Decision-time contract:* All 16 shape statistics are computed from the glyph image itself before classification, so every column is decision-time available; no column is blocked.

*Holdout policy:* Locked holdout = stratified random 20%; selection uses 3-fold stratified CV on training rows only. The holdout is consumed once, in optimization_reliability.

| Exp | Stage | What was compared | Key claim |
|---|---|---|---|
| EXP-056 | data_understanding | Profiled 16000 training rows × 16 raw columns of letter_recognition (holdout: stratified random 4000 of 20000 rows); computed target summary, per-column missingness/uniqueness/PSI, duplicates; 0 columns flagged for review. | letter_recognition has 20000 source rows and 16 raw columns; the locked holdout is stratified random 4000 of 20000 rows (4000 rows), leaving 16000 training rows with 26 classes with 587–650 training rows each (max/min ratio 1.11). |
| EXP-057 | leakage_audit | Scanned 16000 training rows with multiclass heuristics (univariate signal, exact proxy/identity, name, uniqueness) and synthetic canaries; no blocked columns; safe lightgbm raw baseline macro-F1 0.9592±0.0040. Measured exact duplicate feature-vector contamination across CV folds and train/holdout. | No column of letter_recognition is blocked under the decision-time contract; the safe lightgbm raw baseline scores 0.9592±0.0040 training-CV macro-F1, apparent lift 0 by construction. |
| EXP-058 | feature_engineering | 4 recipes compared with lightgbm on StratifiedKFold(3, shuffle, random_state=42) on training rows: raw (16 feats, 0.9592±0.0040); ratios (24 feats, 0.9605±0.0018); poly2 (136 feats, 0.9587±0.0027); selected_mi (10 feats, 0.9479±0.0025). Selected: raw. | Use the `raw` recipe (the 16 shape statistics) for letter_recognition: 0.9592±0.0040 training-CV macro-F1 with 16 features; the best recipe was `ratios` at 0.9605. |
| EXP-059 | model_selection | 5 model families screened on the `raw` recipe with identical StratifiedKFold(3, shuffle, random_state=42) on training rows, ranked by mean - 0.25×std macro-F1: extra_trees 0.9625±0.0018 (3.5s); lightgbm 0.9592±0.0040 (19.6s); hist_gradient_boosting 0.9538±0.0011 (14.4s); xgboost 0.9522±0.0010 (11.7s); logistic_regression 0.7672±0.0035 (2.9s). Selected: extra_trees. | extra_trees is the training-CV model candidate for letter_recognition on the `raw` recipe (0.9625±0.0018 macro-F1, adjusted 0.9621); runner-up lightgbm at 0.9592±0.0040. |
| EXP-060 | optimization_reliability | Compared default extra_trees vs 3 explicit candidates on the `raw` recipe (StratifiedKFold(3, shuffle, random_state=42) on training rows): extra_trees_baseline 0.9625±0.0018; extra_trees_C01 0.9672±0.0013; extra_trees_C02 0.9557±0.0013; extra_trees_C03 0.9649±0.0005; tuning accepted (gain +0.0047 vs required 0.0020). Final fit on 16000 training rows, scored once on the holdout (stratified random 4000 of 20000 rows): macro-F1 0.9751 [0.9704, 0.9798]. | The preselected extra_trees/extra_trees_C01 on the `raw` recipe scored macro-F1 0.9751 on the once-consumed holdout (stratified random 4000 of 20000 rows; iid row bootstrap 95% interval 0.9704–0.9798); training-CV mean was 0.9672. |

**Holdout claims:**

- `EXP-060-C1` (fact): The preselected extra_trees/extra_trees_C01 on the `raw` recipe scored macro-F1 0.9751 on the once-consumed holdout (stratified random 4000 of 20000 rows; iid row bootstrap 95% interval 0.9704–0.9798); training-CV mean was 0.9672.
- `EXP-060-C2` (decision): Tuning accepted: best candidate extra_trees_C01 changed mean training-CV macro-F1 by +0.0047 (required ≥ 0.0020), so extra_trees_C01 was used.
- `EXP-060-C3` (fact): Holdout balanced accuracy 0.9750, accuracy 0.9752, log loss 0.2592 (training-prior log loss 3.2577); majority-class macro-F1 would be 0.0030. Worst classes: class 1 (label 2) F1 0.942, class 17 (label 18) F1 0.951, class 4 (label 5) F1 0.955.
- `EXP-060-C4` (recommendation): Do not promote features from importance alone; require decision-time availability, cross-fold stability, production availability, and monitoring.

**Limitations:**

- A cross-validation leader is not a universal algorithm winner and has not yet seen the locked holdout.
- CV models are fitted on part of the training rows while the final model uses all of them, so the holdout can legitimately beat the CV mean; a single holdout is one draw.
- Identical integer summaries may come from different source images; this is contamination risk, not proven label leakage.
- Name, univariate-signal, identity, and uniqueness rules propose review; only decision-time semantics confirm leakage.
- Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search.
- PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability.
- Recipes were compared with a single default-parameter model; a different family could rank them differently.
- Source labels are PMLB's integer codes 1-26; mapping them to letters assumes PMLB kept alphabetical order (A=1), which was not verified.
- Strong signal alone is not leakage; e.g. legitimate lags or text sentiment are expected to be predictive.

## Bike Sharing (Capital Bikeshare, daily; UCI via interpretable-ml-book) — `bike_sharing_daily` (timeseries_regression)

728 days (2011-01-03 to 2012-12-31) of Washington D.C. Capital Bikeshare usage with calendar and weather attributes, used to predict the total daily rental count (cnt).

*Decision-time contract:* Forecast day d's total rentals at the end of day d-2 (the horizon implied by the provided `cnt_2d_bfr` lag): calendar fields, the day's weather (treated as a forecast proxy), and rental counts from day d-2 or earlier are allowed; same-day casual/registered counts are not.

*Holdout policy:* Locked holdout = last 20% of rows in `dteday` order; recipe/model/parameter choices use only expanding-window 3-fold time-ordered CV on the earlier rows. The holdout is consumed once, in optimization_reliability.

| Exp | Stage | What was compared | Key claim |
|---|---|---|---|
| EXP-061 | data_understanding | Profiled 582 training rows × 17 raw columns of bike_sharing_daily (holdout: last 146 of 728 rows by `dteday`); computed target summary, per-column missingness/uniqueness/PSI, duplicates, time range; 9 columns flagged for review. | bike_sharing_daily has 728 source rows and 17 raw columns; the locked holdout is last 146 of 728 rows by `dteday` (146 rows), leaving 582 training rows with a training target mean of 4176 (std 1780, range 431–8362). |
| EXP-062 | leakage_audit | Scanned 582 training rows with timeseries_regression heuristics (univariate signal, exact proxy/identity, name, uniqueness) and synthetic canaries; compared lightgbm on the raw recipe with vs without blocked ['casual', 'registered'] on expanding-window TimeSeriesSplit(3) on time-ordered training rows: MAE 833.1±460.6 safe vs 419.5±340.5 unsafe (apparent lift +413.6). Also compared random KFold(3) vs time-ordered CV for the safe recipe (MAE 430.0 vs 833.1). | Declared decision-time exclusions for bike_sharing_daily: ['casual', 'registered']. Including them moves training-CV MAE (lightgbm, raw recipe) from 833.1±460.6 to 419.5±340.5, an apparent lift of +413.6 that would not exist in production. Per column: casual +40.8, registered +263.5. The identity rule independently found that casual + registered == cnt exactly on every training row. |
| EXP-063 | feature_engineering | 4 recipes compared with lightgbm on expanding-window TimeSeriesSplit(3) on time-ordered training rows: raw (33 feats, 833.1±460.6); calendar (38 feats, 836.7±445.4); calendar_lags (46 feats, 895.5±562.3); lags_compact (21 feats, 920.5±578.2). Selected: raw. | Use the `raw` recipe (provided calendar/weather columns + days_since_2011 + cnt_2d_bfr (one-hot categoricals)) for bike_sharing_daily: 833.1±460.6 training-CV MAE with 33 features; the best recipe was `raw` at 833.1. |
| EXP-064 | model_selection | 5 model families screened on the `raw` recipe with identical expanding-window TimeSeriesSplit(3) on time-ordered training rows, ranked by mean + 0.25×std MAE: xgboost 787.4±406.3 (0.5s); ridge 861.9±144.7 (0.1s); lightgbm 833.1±460.6 (0.3s); hist_gradient_boosting 857.5±498.1 (0.4s); extra_trees 1,083±508.9 (1.4s). Selected: xgboost. | xgboost is the training-CV model candidate for bike_sharing_daily on the `raw` recipe (787.4±406.3 MAE, adjusted 888.9735); runner-up ridge at 861.9±144.7. |
| EXP-065 | optimization_reliability | Compared default xgboost vs 3 explicit candidates on the `raw` recipe (expanding-window TimeSeriesSplit(3) on time-ordered training rows): xgboost_baseline 787.4±406.3; xgboost_C01 749.9±399.0; xgboost_C02 797.7±426.5; xgboost_C03 786.2±428.4; tuning accepted (gain +37.5354 vs required 7.8740). Final fit on 582 training rows, scored once on the holdout (last 146 of 728 rows by `dteday`): MAE 680.0 [484.0, 890.1]. | The preselected xgboost/xgboost_C01 on the `raw` recipe scored MAE 680.0 on the once-consumed holdout (last 146 of 728 rows by `dteday`; moving-block bootstrap (block length 7) 95% interval 484.0–890.1); training-CV mean was 749.9. |

**Holdout claims:**

- `EXP-065-C1` (fact): The preselected xgboost/xgboost_C01 on the `raw` recipe scored MAE 680.0 on the once-consumed holdout (last 146 of 728 rows by `dteday`; moving-block bootstrap (block length 7) 95% interval 484.0–890.1); training-CV mean was 749.9.
- `EXP-065-C2` (decision): Tuning accepted: best candidate xgboost_C01 changed mean training-CV MAE by +37.5354 (required ≥ 7.8740), so xgboost_C01 was used.
- `EXP-065-C3` (fact): Holdout MAE 680.0 (RMSE 1017.5, MAPE 145.1%, bias +163.9) versus naive baselines: lag-2 (cnt_2d_bfr) MAE 1156.4, seasonal lag-7 1202.5, training mean 2269.5. Model minus lag-2 MAE = -476.5 (block-bootstrap 95% -731.8..-237.3).
- `EXP-065-C4` (recommendation): Do not promote features from importance alone; require decision-time availability, cross-fold stability, production availability, and monitoring.

**Limitations:**

- A cross-validation leader is not a universal algorithm winner and has not yet seen the locked holdout.
- CV models are fitted on part of the training rows while the final model uses all of them, so the holdout can legitimately beat the CV mean; a single holdout is one draw.
- Name, univariate-signal, identity, and uniqueness rules propose review; only decision-time semantics confirm leakage.
- Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search.
- PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability.
- Recipes were compared with a single default-parameter model; a different family could rank them differently.
- Strong signal alone is not leakage; e.g. legitimate lags or text sentiment are expected to be predictive.
- The holdout covers only ~5 months (late 2012, incl. the Hurricane Sandy period); one season is not a full-year test.
- The source `atemp` column is on an unusual scale (mean ~32) relative to `temp`; its derivation in this redistribution is undocumented.

## Women's E-Commerce Clothing Reviews (Kaggle) — `ecommerce_clothing_reviews` (text_tabular_binary)

23,486 customer reviews of women's clothing (title, free text, reviewer age, product taxonomy) used to predict whether the reviewer recommends the product (Recommended IND=1, 82% of rows).

*Decision-time contract:* Predict the recommendation flag from the review title/text, reviewer age, and product taxonomy at submission time, for products not seen in training; the star rating (written in the same form as the recommendation) and helpful-vote counts (accrued after publication) are not available.

*Holdout policy:* Locked holdout = one of 5 stratified group folds by `Clothing ID` (~20% of rows, no Clothing ID shared with training); selection uses 3-fold stratified group CV on training rows only. The holdout is consumed once, in optimization_reliability.

| Exp | Stage | What was compared | Key claim |
|---|---|---|---|
| EXP-066 | data_understanding | Profiled 18789 training rows × 10 raw columns of ecommerce_clothing_reviews (holdout: 4697 rows from 256 `Clothing ID` groups never seen in training (first of 5 stratified group folds)); computed target summary, per-column missingness/uniqueness/PSI, duplicates, text length stats, group concentration; 3 columns flagged for review. | ecommerce_clothing_reviews has 23486 source rows and 10 raw columns; the locked holdout is 4697 rows from 256 `Clothing ID` groups never seen in training (first of 5 stratified group folds) (4697 rows), leaving 18789 training rows with a training positive rate of 0.8224 (15452 positives vs 3337 negatives). |
| EXP-067 | leakage_audit | Scanned 18789 training rows with text_tabular_binary heuristics (univariate signal, exact proxy/identity, name, uniqueness) and synthetic canaries; compared logistic_regression on the combined recipe with vs without blocked ['Rating', 'Positive Feedback Count'] on StratifiedGroupKFold(3) by `Clothing ID` on training rows: ROC-AUC 0.9424±0.0033 safe vs 0.9795±0.0006 unsafe (apparent lift +0.0371). Also tabular-only logistic safe vs unsafe: ROC-AUC 0.5283 vs 0.9742. | Declared decision-time exclusions for ecommerce_clothing_reviews: ['Rating', 'Positive Feedback Count']. Including them moves training-CV ROC-AUC (logistic_regression, combined recipe) from 0.9424±0.0033 to 0.9795±0.0006, an apparent lift of +0.0371 that would not exist in production. Per column: Rating +0.0371, Positive Feedback Count +0.0001. |
| EXP-068 | feature_engineering | 5 recipes compared with logistic_regression on StratifiedGroupKFold(3) by `Clothing ID` on training rows: tabular (33 feats, 0.5283±0.0226); tfidf_word (30000 feats, 0.9431±0.0033); tfidf_word_char (60000 feats, 0.9359±0.0038); combined (30037 feats, 0.9424±0.0033); combined_char (60037 feats, 0.9350±0.0033). Selected: tfidf_word. | Use the `tfidf_word` recipe (word 1-2gram TF-IDF of Title + Review Text) for ecommerce_clothing_reviews: 0.9431±0.0033 training-CV ROC-AUC with 30000 features; the best recipe was `tfidf_word` at 0.9431. |
| EXP-069 | model_selection | 5 model families screened on the `tfidf_word` recipe with identical StratifiedGroupKFold(3) by `Clothing ID` on training rows, ranked by mean - 0.25×std ROC-AUC: logistic_regression 0.9431±0.0033 (6.7s); lightgbm 0.9433±0.0058 (13.2s); linear_svm 0.9406±0.0045 (6.6s); xgboost 0.9363±0.0055 (25.4s); complement_nb 0.9257±0.0024 (5.9s). Selected: logistic_regression. | logistic_regression is the training-CV model candidate for ecommerce_clothing_reviews on the `tfidf_word` recipe (0.9431±0.0033 ROC-AUC, adjusted 0.9423); runner-up lightgbm at 0.9433±0.0058. |
| EXP-070 | optimization_reliability | Compared default logistic_regression vs 3 explicit candidates on the `tfidf_word` recipe (StratifiedGroupKFold(3) by `Clothing ID` on training rows): logistic_regression_baseline 0.9431±0.0033; logistic_regression_C01 0.9466±0.0037; logistic_regression_C02 0.9405±0.0030; logistic_regression_C03 0.9409±0.0028; tuning accepted (gain +0.0034 vs required 0.0010). Final fit on 18789 training rows, scored once on the holdout (4697 rows from 256 `Clothing ID` groups never seen in training (first of 5 stratified group folds)): ROC-AUC 0.9459 [0.9391, 0.9525]. | The preselected logistic_regression/logistic_regression_C01 on the `tfidf_word` recipe scored ROC-AUC 0.9459 on the once-consumed holdout (4697 rows from 256 `Clothing ID` groups never seen in training (first of 5 stratified group folds); cluster bootstrap over 256 groups 95% interval 0.9391–0.9525); training-CV mean was 0.9466. |

**Holdout claims:**

- `EXP-070-C1` (fact): The preselected logistic_regression/logistic_regression_C01 on the `tfidf_word` recipe scored ROC-AUC 0.9459 on the once-consumed holdout (4697 rows from 256 `Clothing ID` groups never seen in training (first of 5 stratified group folds); cluster bootstrap over 256 groups 95% interval 0.9391–0.9525); training-CV mean was 0.9466.
- `EXP-070-C2` (decision): Tuning accepted: best candidate logistic_regression_C01 changed mean training-CV ROC-AUC by +0.0034 (required ≥ 0.0010), so logistic_regression_C01 was used.
- `EXP-070-C3` (fact): On unseen products the text-aware model's holdout ROC-AUC 0.9459 / average precision 0.9864 compares with 0.5480 / 0.8507 for the pre-declared tabular-only reference; AUC difference +0.3979 (cluster-bootstrap 95% +0.3498..+0.4328).
- `EXP-070-C4` (recommendation): Do not promote features from importance alone; require decision-time availability, cross-fold stability, production availability, and monitoring.

**Limitations:**

- A cross-validation leader is not a universal algorithm winner and has not yet seen the locked holdout.
- CV models are fitted on part of the training rows while the final model uses all of them, so the holdout can legitimately beat the CV mean; a single holdout is one draw.
- Name, univariate-signal, identity, and uniqueness rules propose review; only decision-time semantics confirm leakage.
- Only a handful of explicit candidates were tried; this is a sanity check, not an exhaustive search.
- PSI on feature distributions is a smoke check; it uses no holdout labels and does not prove production stability.
- Recipes were compared with a single default-parameter model; a different family could rank them differently.
- Strong signal alone is not leakage; e.g. legitimate lags or text sentiment are expected to be predictive.

## Campaign-wide limitations

- One dataset per task type and one holdout per dataset; results are evidence about these datasets, not universal rankings.
- 3-fold CV with a single repeat and small explicit parameter grids keep runtime low; fold standard deviations are coarse.
- Fraud model fits keep all positives but only 25% of negatives (seeded); validation and holdout scoring always use every row, and probabilities are prior-corrected.
- Heuristic leakage flags are review prompts. Only the written decision-time contracts declare exclusions.
- No LLM review has been run on these results yet.
