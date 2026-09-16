# Safe vs Unsafe HyperAck Report

This report compares models trained **without final fares** (`safe_leakage_exp`) against the original suite that **allowed final fares** (`hyperack_exp`). Both use the same locked train/test split.

## What changed in the safe suite

- Dropped unsafe columns: `final_customer_fare`, `final_biker_fare`
- Kept safe signals: first fare, distance, time, geo, category, products
- Same experiment strategies / model families as before

## Headline results

- **Best safe model:** 09 - lightgbm_tuned (ROC-AUC=0.9450, recall=0.7707)
- **Best unsafe model:** 09 - lightgbm_tuned (ROC-AUC=0.9793, recall=0.9114)
- **Gap (best unsafe − best safe) ROC-AUC:** 0.0343
- **Average ROC-AUC drop when going safe:** 0.0336

## Per-experiment comparison (sorted by safe ROC-AUC)

| Exp | Name | Unsafe ROC-AUC | Safe ROC-AUC | Δ ROC-AUC | Unsafe Recall | Safe Recall | Δ Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| 09 | lightgbm_tuned | 0.9793 | 0.9450 | -0.0343 | 0.9114 | 0.7707 | -0.1407 |
| 08 | xgboost_tuned | 0.9788 | 0.9419 | -0.0369 | 0.9104 | 0.7755 | -0.1349 |
| 04 | time_cyclical_features | 0.9740 | 0.9419 | -0.0321 | 0.8979 | 0.8150 | -0.0829 |
| 10 | histgb_sklearn | 0.9776 | 0.9390 | -0.0385 | 0.9123 | 0.8131 | -0.0992 |
| 06 | interactions_bins | 0.9778 | 0.9389 | -0.0389 | 0.9085 | 0.7987 | -0.1098 |
| 07 | feature_selection | 0.9689 | 0.9387 | -0.0302 | 0.8786 | 0.8064 | -0.0723 |
| 14 | leakage_safe_features | 0.9387 | 0.9387 | +0.0000 | 0.8025 | 0.8025 | +0.0000 |
| 01 | baseline_current | 0.9730 | 0.9386 | -0.0344 | 0.8931 | 0.8025 | -0.0906 |
| 12 | threshold_calibration | 0.9780 | 0.9386 | -0.0394 | 0.8815 | 0.7659 | -0.1156 |
| 15 | full_fe_best_blend | 0.9781 | 0.9384 | -0.0397 | 0.9114 | 0.8054 | -0.1060 |
| 13 | stacking_ensemble | 0.9737 | 0.9367 | -0.0370 | 0.9143 | 0.8054 | -0.1089 |
| 02 | lightgbm_strong_baseline | 0.9720 | 0.9365 | -0.0356 | 0.8805 | 0.8035 | -0.0771 |
| 03 | fare_pricing_features | 0.9767 | 0.9358 | -0.0409 | 0.8998 | 0.7948 | -0.1050 |
| 05 | geo_distance_bearing | 0.9682 | 0.9353 | -0.0329 | 0.8796 | 0.7919 | -0.0877 |

## How to read this

- **Negative Δ** means the safe model is worse on that metric (expected if final fares were helpful but leaky).
- Prefer **safe** models for production if final fares are not known at prediction time.
- Unsafe scores are an optimistic lab ceiling, not a deployable guarantee.

## Files

- Comparison CSV: `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/safe_leakage_exp/results/safe_vs_unsafe_comparison.csv`
- Safe results: `safe_leakage_exp/results/`
- Unsafe results: `hyperack_exp/results/`

## Simple explanation

**Unsafe models** trained with final fares (may be future info) → higher lab scores (~0.97–0.98).

**Safe models** trained without final fares → honest scores (~0.93–0.95).

Dropping leakage features costs about **0.03–0.04 ROC-AUC** on average, but the safe models are the ones you can trust for real prediction.

### Best safe ranking (this run)

1. 09 tuned LightGBM — 0.9450
2. 08 tuned XGBoost — 0.9419
3. 04 time features — 0.9419
4. 10 HistGB — 0.9390
5. 06 interactions/bins — 0.9389

## توضیح ساده (فارسی)

- **مدل ناامن:** با قیمت نهایی آموزش دیده → امتیاز آزمایشگاهی بالاتر
- **مدل امن:** بدون قیمت نهایی → امتیاز واقعی‌تر برای استفاده عملی
- هزینه امن‌سازی: حدود **۰.۰۳ تا ۰.۰۴** کاهش ROC-AUC
- برای پروداکشن: مدل‌های **safe_leakage_exp** را انتخاب کنید (بهترین فعلاً: 09 LightGBM تیون‌شده امن)
