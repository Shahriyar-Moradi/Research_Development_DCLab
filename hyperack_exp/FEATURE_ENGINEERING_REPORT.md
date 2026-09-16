# HyperAck Experiments — Feature & Feature Engineering Report

**What this document is:** a simple explanation of *which features we used*, *why we chose them*, and *what each experiment changed*.

**What we predict:** `hyper_ack` (0 or 1) — whether a delivery is hyper-acknowledged.

**How we compare fairly:** every experiment uses the **same** cleaned data and the **same** train/test split (80/20, stratified, seed 42). Primary score is **ROC-AUC**. We also track F1, **recall**, accuracy, and average precision.

**Quick result (held-out test):**

| Rank | Experiment | ROC-AUC | Recall | Features used |
|------|------------|---------|--------|---------------|
| 1 | 09 Tuned LightGBM | 0.9793 | 0.9114 | Full engineered set (34) |
| 2 | 08 Tuned XGBoost | 0.9788 | 0.9104 | Full engineered set (34) |
| 3 | 15 Blend LGBM+XGB | 0.9781 | 0.9114 | Full engineered set (34) |
| … | … | … | … | … |
| 14 | 14 Leakage-safe | 0.9387 | 0.8025 | Safe set without final fares (25) |

Experiment **11 (TabPFN)** was skipped (license / token needed).

---

## 1. The raw data columns (what we start from)

These come directly from `hyper_ackt-dataset.csv` after dropping rows with missing `total_distance`.

| Feature | Simple meaning | Why it can matter for `hyper_ack` |
|---------|----------------|-----------------------------------|
| `deliverey_category_id` | Type of delivery | Different job types may be accepted more/less often |
| `weekday` | Day of week (1–7) | Weekdays vs weekends can change courier behavior |
| `time_bucket` | Time-of-day bucket | Busy vs quiet periods |
| `total_distance` | Trip distance | Longer trips may be harder to accept |
| `sum_product` | Product count / basket size | Bigger orders can change attractiveness |
| `source_latitude` / `source_longitude` | Pickup location | Area / city effects |
| `destination_latitude` / `destination_longitude` | Drop-off location | Route shape and area |
| `first_customer_fare` | First offered customer price | Early price signal (usually available early) |
| `final_customer_fare` | Final customer price | Strong signal, but may be **after** the decision |
| `final_biker_fare` | Final courier price | Strong signal, but may be **after** the decision |
| `created_date` / `first_created_at` | Timestamps | Used only to build time features; not fed as raw strings |

**Important caution:** `final_customer_fare` and `final_biker_fare` may not be known *before* the courier decides. Models that use them can look great in the lab but be weaker in real life. Experiment **14** measures that honest gap.

---

## 2. How we choose features (simple rules)

We did **not** pick features randomly. The rules were:

1. **Start simple** — use raw numeric columns first (experiments 01–02).
2. **Add one idea at a time** — pricing, then time, then geography, then interactions (03–06). That way we see *what helps*.
3. **Prefer features available early** — time and first fare are safer than final fares.
4. **Engineer human-meaningful signals** — e.g. fare-per-km is easier for a model than raw fare + distance separately.
5. **Fit transforms on train only** — clustering, bins, and feature selection never peek at the test set.
6. **Then tune / ensemble** — after features are good, improve the model (08–15).
7. **Always keep a leakage-safe check** — experiment 14.

---

## 3. Feature engineering families (the building blocks)

### A) Base features
Raw numeric columns listed in section 1 (with or without final fares).

### B) Pricing features (used in 03, 06–15, and partly in 14)
| New feature | How it is made | Why it is useful |
|-------------|----------------|------------------|
| `log_distance` | `log(1 + distance)` | Compresses long-trip outliers |
| `first_fare_per_km` | first fare ÷ distance | Price attractiveness per km |
| `final_customer_fare_per_km` | final customer fare ÷ distance | Same idea with final price |
| `customer_fare_delta` | final − first customer fare | Did the price go up/down? |
| `customer_fare_change_pct` | delta ÷ first fare | Relative change, not just amount |
| `biker_customer_gap` | biker fare − customer fare | Split / margin signal |
| `biker_fare_per_km` | biker fare ÷ distance | Courier pay intensity |
| `log_final_*` | log of final fares | Softens huge fare values |

**Why pricing?** Couriers often care about “is this pay worth this trip?” Ratios and changes express that better than raw money alone.

### C) Time features (used in 04, 06–15, and 14)
| New feature | How it is made | Why it is useful |
|-------------|----------------|------------------|
| `hour` | hour from `first_created_at` | Fine-grained time of day |
| `is_rush_hour` | 1 if hour in lunch/evening peaks | Busy windows change acceptance |
| `is_weekend` | 1 if weekend weekday codes | Weekend demand differs |
| `hour_sin` / `hour_cos` | circular encoding of hour | Hour 23 is close to 0 (not far) |
| `weekday_sin` / `weekday_cos` | circular encoding of weekday | Same idea for days |
| `day_of_month` | day number | Mild seasonality / payday patterns |

**Why cyclical sin/cos?** Trees and linear models treat hour `23` and hour `0` as far apart if we use a plain number. Sin/cos fix that.

### D) Geographic features (used in 05, 06–15, and 14)
| New feature | How it is made | Why it is useful |
|-------------|----------------|------------------|
| `haversine_km` | great-circle distance from pickup to drop-off | True route length on Earth |
| `latitude_delta` / `longitude_delta` | dest − source | Direction and span |
| `geo_bearing_sin` / `geo_bearing_cos` | trip direction angle | Northbound vs eastbound patterns |

**Why geo engineering?** Raw lat/lon are hard for models to read as “how long / which way is this trip?” Distance and bearing make that explicit.

### E) Interactions and bins (experiment 06)
| New feature | How it is made | Why it is useful |
|-------------|----------------|------------------|
| `distance_x_first_fare` | distance × first fare | Long + cheap vs short + expensive |
| `category_x_hour` | category × hour | Some categories may matter only at night |
| `total_distance_qbin` | quantile bin of distance (fit on train) | Soft cutoffs like “very long trip” |
| `first_customer_fare_qbin` | quantile bin of first fare (fit on train) | Soft cutoffs like “very low fare” |

---

## 4. Experiment-by-experiment (features, selection, importance)

### 01 — Current baseline
- **Goal:** recreate the original approach and set a floor score.
- **Features:** base columns + `geo_cluster` (KMeans with 3 clusters on lat/lon, **fit on train only**).
- **How selected:** keep what the original notebook used; do not invent new FE yet.
- **Why important:** proves later gains are real, not from a different split.
- **Model:** Logistic Regression, XGBoost, SVM → keep the best (XGBoost won).
- **Result:** ROC-AUC **0.9730**, recall **0.8931**, **13** features.

### 02 — LightGBM strong baseline
- **Goal:** strong modern tree model on **raw** features only.
- **Features:** base numeric columns (including final fares), no engineered extras.
- **How selected:** same raw set as a control; only the model changes.
- **Why important:** if LightGBM already beats 01, the model family matters more than FE so far.
- **Result:** ROC-AUC **0.9720**, recall **0.8805**, **12** features.

### 03 — Fare / pricing feature engineering
- **Goal:** test whether price ratios and price changes help.
- **Features:** base + pricing family (section 3B).
- **How selected:** business logic — acceptance often depends on pay vs effort. We add per-km and delta features instead of hoping the tree invents them.
- **Why important:** pricing was one of the strongest single FE lifts.
- **Result:** ROC-AUC **0.9767**, recall **0.8998**, **21** features.

### 04 — Time / cyclical feature engineering
- **Goal:** test whether time-of-day and weekend patterns help.
- **Features:** base + time family (section 3C).
- **How selected:** operational knowledge — rush hours and weekends change courier behavior. Sin/cos chosen so night wrap-around is handled correctly.
- **Why important:** usually available at prediction time (deployment-safe).
- **Result:** ROC-AUC **0.9740**, recall **0.8979**, **20** features.

### 05 — Geographic feature engineering
- **Goal:** test route distance and direction features.
- **Features:** base + geo family (section 3D).
- **How selected:** physics of delivery — distance and bearing are more meaningful than four raw coordinates.
- **Why important:** alone it did **not** beat the raw LightGBM baseline much (ROC-AUC 0.9682). Geo helps more when combined with pricing/time (see 06).
- **Result:** ROC-AUC **0.9682**, recall **0.8796**, **17** features.

### 06 — Interactions and bins
- **Goal:** combine pricing + time + geo, then add joint effects and soft cutoffs.
- **Features:** full FE (B+C+D) + interactions + train-fitted quantile bins.
- **How selected:**
  - Combine everything that individually made sense.
  - Add interactions for “two things together matter.”
  - Bins capture thresholds without hard-coding cut values.
- **Why important:** best pure FE experiment before tuning (ROC-AUC 0.9778).
- **Result:** ROC-AUC **0.9778**, recall **0.9085**, **38** features.

### 07 — Feature selection
- **Goal:** keep only the most informative engineered features.
- **Features:** full pricing+time+geo set, then **SelectKBest** with mutual information → top ~18 features inside a pipeline.
- **How selected:** statistical filter on **train only**. Mutual information asks: “how much does this feature tell us about `hyper_ack`?”
- **Why important:** tests whether fewer features are enough. Here selection **hurt** a bit vs using all features (0.9689 vs ~0.977+), so for this dataset keeping the full FE set is better.
- **Result:** ROC-AUC **0.9689**, recall **0.8786**.

### 08 — Tuned XGBoost
- **Goal:** keep full FE; improve the model with randomized hyperparameter search.
- **Features:** full pricing + time + geo (34 features). Same feature idea as later winners.
- **How selected:** features already validated in 03–06; search tunes depth, learning rate, subsample, etc. on **train CV only**.
- **Why important:** shows tuning on good features is valuable.
- **Result:** ROC-AUC **0.9788**, recall **0.9104**.

### 09 — Tuned LightGBM (winner)
- **Goal:** same as 08, but for LightGBM.
- **Features:** same full engineered set (34).
- **How selected:** same FE as 08; different model + search space.
- **Why important:** **best ROC-AUC overall**. Full FE + careful tuning beat fancy stacking here.
- **Result:** ROC-AUC **0.9793**, recall **0.9114**.

### 10 — sklearn HistGradientBoosting
- **Goal:** compare a built-in sklearn booster on the same full FE.
- **Features:** same full engineered set (34).
- **How selected:** feature set locked; only model family changes (fewer external dependencies).
- **Why important:** almost as strong as tuned trees, with strong F1/recall.
- **Result:** ROC-AUC **0.9776**, recall **0.9123**.

### 11 — TabPFN
- **Goal:** try a prior-fitted tabular model on a 1,024-row context.
- **Features:** planned full engineered set.
- **Status:** **skipped** in this run (needs PriorLabs license acceptance / valid token).
- **Why it was planned:** strong on medium tabular data without heavy tuning.

### 12 — Calibration + threshold
- **Goal:** keep ranking quality, improve decision cut-off for F1.
- **Features:** full engineered set (34).
- **How selected:** same FE as winners; change is post-processing (isotonic calibration + threshold chosen on an inner validation split, not on test).
- **Why important:** ROC-AUC stayed high (0.9780), but recall dropped a bit (0.8815) because the threshold moved toward precision/F1.
- **Result:** ROC-AUC **0.9780**, recall **0.8815**.

### 13 — Stacking ensemble
- **Goal:** combine Logistic Regression + LightGBM + XGBoost.
- **Features:** full engineered set (34) for all base models.
- **How selected:** features fixed; diversity comes from different model types. Meta-learner sees out-of-fold probabilities only.
- **Why important:** **best recall (0.9143)**, even though ROC-AUC is slightly below 09. Useful if catching more true positives matters most.
- **Result:** ROC-AUC **0.9737**, recall **0.9143**.

### 14 — Leakage-safe features (production reference)
- **Goal:** honest score **without** `final_customer_fare` / `final_biker_fare`.
- **Features:** pricing + time + geo, but pricing transforms that need final fares are removed/limited → **25** features.
- **How selected:** remove anything that may only exist after the decision. Keep first fare, distance, time, geo.
- **Why important:** this is the realistic number. Gap to the winner (~0.04 ROC-AUC) shows how much “lab performance” may depend on final fares.
- **Result:** ROC-AUC **0.9387**, recall **0.8025**.

### 15 — Full FE + probability blend
- **Goal:** blend strong LightGBM and XGBoost probabilities (0.60 / 0.40).
- **Features:** full engineered set (34).
- **How selected:** features from the winning FE path; blend is a simple ensemble without a stacking meta-model.
- **Why important:** near the top (ROC-AUC 0.9781), fast to train vs stacking, good recall.
- **Result:** ROC-AUC **0.9781**, recall **0.9114**.

---

## 5. What we learned about features (plain summary)

1. **Pricing features helped a lot** (experiment 03). Fare-per-km and fare change are strong signals.
2. **Time features helped a little** and are safe to deploy (experiment 04).
3. **Geo features alone were weak**, but useful as part of the full set (05 vs 06).
4. **Combining FE families** (06) beat any single FE idea.
5. **Automatic feature selection** (07) did not beat “keep the full engineered set” here.
6. **Tuning on full FE** (08/09) gave the best ROC-AUC.
7. **Final fares inflate scores** — compare 09 (0.979) vs 14 (0.939). For real products, trust 14 more, or retrain 09 without final fares.
8. **If recall matters most**, stacking (13) catches the most true positives among completed runs.

---

## 6. Recommended next steps

1. **Research / leaderboard winner:** experiment **09** (tuned LightGBM on full FE).
2. **Production candidate:** start from experiment **14**, or re-run the 09 tuning recipe **without final fares**.
3. **If you care more about recall than ROC-AUC:** consider experiment **13**, still without final fares for deployment.
4. **Optional:** finish experiment **11 (TabPFN)** after accepting the PriorLabs license, then re-run `99_final_benchmark.ipynb`.

---

## 7. Where the details live in code

- Shared feature builders: `hyperack_exp/shared/protocol.py`
- Each strategy notebook: `hyperack_exp/01_*.ipynb` … `15_*.ipynb`
- Scores: `hyperack_exp/results/benchmark_summary.csv`
- Charts: `hyperack_exp/99_final_benchmark.ipynb`

This report is meant to be read by humans first. For exact formulas, open `protocol.py`.
