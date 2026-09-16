# HyperAck Top Models — Complete Training Report & General Playbook

**Purpose:** document what we trained, why the top models won, and turn that into a **general rule** you can reuse for most tabular classification / prediction projects.

**Dataset:** `hyper_ackt-dataset.csv` (~11k rows after cleaning)  
**Target:** `hyper_ack` (binary)  
**Protocol:** same stratified 80/20 split for every experiment (`random_state=42`)  
**Primary metric:** ROC-AUC (also tracked F1, recall, accuracy, average precision)

---

## 1. What we did (end-to-end)

### 1.1 Shared training protocol
1. Clean data (drop missing `total_distance`, parse timestamps).
2. Lock one train/test split for all experiments.
3. Build features **only from train knowledge** (no test leakage in clustering, bins, selection, thresholds).
4. Train model(s) on train; score once on held-out test.
5. Save standardized metrics to `results/` and rank in `99_final_benchmark.ipynb`.

### 1.2 Experiment ladder (how we searched for better models)
We did **not** jump straight to ensembles. We climbed a ladder:

| Stage | Experiments | Question we asked |
|-------|-------------|-------------------|
| Baseline | 01–02 | How good is a simple / strong default model? |
| Feature engineering | 03–06 | Which feature ideas help? |
| Feature selection | 07 | Do fewer features help? |
| Model tuning | 08–10 | Does hyperparameter search help on good FE? |
| Advanced / special | 11–13 | TabPFN, calibration, stacking |
| Honesty check | 14 | What if we remove likely leaky final fares? |
| Final blend | 15 | Can a simple blend beat a single tuned model? |

This ladder is the core of the general rule below.

---

## 2. Top 10 trained models (held-out test)

| Rank | Model / experiment | ROC-AUC | Recall | F1 | Features | Fit time | What made it strong |
|------|--------------------|---------|--------|-----|----------|----------|---------------------|
| 1 | **09 Tuned LightGBM** | **0.9793** | 0.9114 | 0.9284 | 34 | ~195s | Full FE + randomized search |
| 2 | **08 Tuned XGBoost** | 0.9788 | 0.9104 | 0.9278 | 34 | ~20s | Same FE, tuned XGB |
| 3 | **15 LGBM+XGB blend** | 0.9781 | 0.9114 | 0.9288 | 34 | ~9s | 0.60/0.40 probability blend |
| 4 | 12 Calibrated LGBM | 0.9780 | 0.8815 | 0.9205 | 34 | ~13s | Same FE; threshold tuned for F1 |
| 5 | 06 Interactions + bins | 0.9778 | 0.9085 | 0.9281 | 38 | ~4s | Best pure FE (no heavy tuning) |
| 6 | 10 HistGradientBoosting | 0.9776 | 0.9123 | 0.9316 | 34 | ~4s | Strong sklearn booster on full FE |
| 7 | 03 Pricing FE | 0.9767 | 0.8998 | 0.9188 | 21 | ~3s | Fare-per-km / deltas alone helped a lot |
| 8 | 04 Time FE | 0.9740 | 0.8979 | 0.9205 | 20 | ~3s | Rush hour / cyclical time |
| 9 | 13 Stacking | 0.9737 | **0.9143** | 0.9299 | 34 | ~20s | Best recall, not best ROC-AUC |
| 10 | 01 Baseline XGB | 0.9730 | 0.8931 | 0.9115 | 13 | ~1s | Original raw + geo cluster floor |

**Production honesty check (not in top ROC-AUC, but critical):**  
Experiment **14 leakage-safe** → ROC-AUC **0.9387**, recall **0.8025** (no final fares).  
Gap vs winner ≈ **0.04 ROC-AUC** → much of the lab lift may depend on post-decision price fields.

---

## 3. How the top models were trained (details)

### 3.1 Feature set used by most top models (08, 09, 10, 12, 13, 15)
**Base raw:** category, weekday, time_bucket, distance, products, lat/lon, first fare, final fares  

**Pricing FE:** fare-per-km, fare deltas / % change, biker–customer gap, log transforms  

**Time FE:** hour, rush-hour, weekend, sin/cos hour & weekday, day_of_month  

**Geo FE:** Haversine km, lat/lon deltas, bearing sin/cos  

**Why this set won:** each family answers a real business question (pay vs effort, when, how far/which way). Combined FE beat any single family.

### 3.2 Winner recipe — experiment 09 (Tuned LightGBM)
1. Build full FE matrix above (34 features).
2. Pipeline: median impute → LightGBM.
3. `RandomizedSearchCV` (~20 trials, 4-fold stratified CV) maximizing ROC-AUC on **train only**.
4. Search space included: `n_estimators`, `learning_rate`, `num_leaves`, `max_depth`, `min_child_samples`, `subsample`, `colsample_bytree`, `reg_lambda`.
5. Refit best params; score once on locked test set.

### 3.3 Runner-up — experiment 08 (Tuned XGBoost)
Same FE and search discipline; different model family. Confirms the lift is not “LightGBM magic only.”

### 3.4 Near-winner — experiment 15 (Blend)
Train strong LGBM + XGB on the same FE; average probabilities with fixed weights (0.60 / 0.40). Simple, fast, almost as good as heavy tuning.

### 3.5 Best recall — experiment 13 (Stacking)
LR + LGBM + XGB → logistic meta-learner with out-of-fold probabilities. Slightly lower ROC-AUC than 09, highest recall. Use when catching more positives matters more than ranking score.

---

## 4. What actually mattered (lessons from this project)

1. **Good features > fancy models alone.** Pricing FE (03) already beat the old baseline before tuning.
2. **Combine FE families** (06) before spending days on ensembles.
3. **Automatic feature selection** (07) did not beat “keep the full useful set” here.
4. **Tuning on a strong FE set** (08/09) gave the top ROC-AUC.
5. **Stacking is not always worth it** for ROC-AUC; it helped recall.
6. **Calibration/threshold** changes decision quality (F1/recall tradeoff), not ranking as much.
7. **Leakage check is mandatory.** Final fares inflated scores. Always compare a safe feature set.
8. **One locked split + one primary metric** makes comparisons honest.

---

## 5. General rule — how to train top models for most tabular classification / prediction

Use this as a default playbook for churn, fraud, acceptance, lead scoring, demand class, etc.

### Rule A — Fix the game before you play
1. Define the prediction moment (“what is known at decision time?”).
2. Choose **one primary metric** (ROC-AUC / PR-AUC / RMSE / MAE) that matches the business cost.
3. Lock train/validation/test (or time-based split for time data). Never tune on the final test.
4. Build a **baseline** first (logistic / default LightGBM / simple mean for regression).

### Rule B — Feature engineering that works in most cases
Add features in **families**, one family at a time, measure lift:

| Family | Typical ideas | Why it often helps |
|--------|---------------|--------------------|
| Ratios | value/count, price/distance, amount/tenure | Models understand intensity better than raw pairs |
| Deltas / changes | current − previous, % change | Behavior change is often the signal |
| Time | hour, day, month, rush flags, sin/cos cycles | Seasonality & circular time |
| Aggregations | user history counts, averages, recency | “Who is this entity?” context |
| Interactions | important_A × important_B | Joint effects |
| Bins | quantile bins of key numerics | Soft thresholds |
| Domain risk flags | VIP, new user, long trip, high amount | Human priors |

**Selection rule:** keep a feature if it (a) is available at prediction time, (b) has a clear story, and (c) improves validation metric or is needed for monitoring. Drop leaky post-outcome fields.

### Rule C — Model training ladder (top-10 hunter)
Always climb in this order:

1. **Strong default GBDT** (LightGBM / XGBoost / HistGB) on raw + obvious FE  
2. **Better FE** (ratios → combine winners)  
3. **Tune the best model** with randomized / Bayesian search on train CV  
4. **Simple blend** of top 2 models (often enough)  
5. **Stacking** only if blend is not enough and you can afford complexity  
6. **Calibrate + threshold** if the product needs hard decisions (not only ranking)  
7. **Leakage-safe retrain** for deployment  

This is how we produced the HyperAck top 10, and it generalizes well.

### Rule D — Tuning that usually works
- Tune on CV of training data only.
- Search: learning rate, depth/leaves, row/column subsample, regularization, n_estimators.
- Prefer **randomized search / Optuna** over huge grids.
- Early: fewer trials to learn direction; later: more trials on the winning model family.
- Do not retune on the final test set.

### Rule E — Important factors beyond FE and tuning
- **Class imbalance:** use PR-AUC / F1 / class weights if positives are rare.
- **Time leakage:** for time-ordered data, split by time, not random rows.
- **Target leakage:** remove anything caused by the label or only known after the event.
- **Calibration:** needed when probabilities are shown to users or drive thresholds.
- **Stability:** prefer a slightly worse but stable model over a fragile winner.
- **Cost of errors:** if FN is expensive, optimize recall; if FP is expensive, optimize precision.
- **Speed / ops:** HistGB or untuned LGBM may be “good enough” for production.

### Rule F — How to declare a “top model”
A model is top-tier only if:
1. It beats a strong baseline on the locked validation/test metric.
2. It uses only features available at prediction time (or you explicitly mark research-only leakage).
3. Gains are large enough vs complexity (tuning/stacking cost).
4. You can explain the main feature families in plain language.

---

## 6. Copy-paste checklist for a new classification / prediction project

```text
[ ] Define decision time and remove leaky fields
[ ] Choose primary metric + secondary metrics (include recall if positives matter)
[ ] Lock split (random stratified OR time-based)
[ ] Train baseline (LR + default LightGBM)
[ ] Add FE family 1 (ratios/deltas) → measure
[ ] Add FE family 2 (time) → measure
[ ] Add FE family 3 (entity/geo/history) → measure
[ ] Combine winning FE families
[ ] Tune best GBDT on train CV
[ ] Try simple blend of top-2
[ ] Optional: stacking / calibration / threshold
[ ] Retrain leakage-safe version for production
[ ] Report actual vs predicted (confusion, recall, sample errors)
[ ] Write short “why these features” notes
```

---

## 7. Mapping HyperAck → general problems

| HyperAck idea | General equivalent |
|---------------|--------------------|
| Fare-per-km | Intensity ratio (amount/tenure, clicks/session) |
| Fare delta | Change features (this week − last week) |
| Rush hour / weekend | Seasonality & calendar effects |
| Haversine / bearing | Distance & direction / similarity features |
| Final fare leakage | Any post-outcome field (paid_at, closed_reason, etc.) |
| Tuned LightGBM winner | Default champion model class for tabular data |
| Stacking best recall | Ensemble when catching rare events matters |
| Leakage-safe model 14 | Mandatory production candidate |

---

## 8. Recommended default for your next project

1. **Research champion:** tuned LightGBM/XGBoost on full safe+useful FE (like experiment 09).  
2. **Production candidate:** same recipe with leaky fields removed (like experiment 14).  
3. **If recall is the KPI:** compare stacking/blend and threshold tuning (like 13/12), still leakage-safe.  
4. **Keep the ladder:** baseline → FE families → combine → tune → blend → honesty check.

---

## 9. Where to find the artifacts

| Artifact | Path |
|----------|------|
| Experiment notebooks | `hyperack_exp/01_*.ipynb` … `15_*.ipynb` |
| Shared FE + metrics code | `hyperack_exp/shared/protocol.py` |
| Leaderboard | `hyperack_exp/99_final_benchmark.ipynb` |
| Scores CSV | `hyperack_exp/results/benchmark_summary.csv` |
| Feature details report | `hyperack_exp/FEATURE_ENGINEERING_REPORT.md` |
| This training playbook | `hyperack_exp/TOP_MODELS_TRAINING_REPORT.md` |
| Cursor rule (auto guidance) | `.cursor/rules/tabular-classification-playbook.mdc` |

---

*This report is based on the completed HyperAck suite in this workspace. Treat experiment 14 as the honest deployment reference whenever final/outcome fields may not be available at prediction time.*
