# Optimized Safe vs Unsafe Benchmark

Same locked test split. Optimized-safe models never use `final_customer_fare` / `final_biker_fare`.

## Champions
- **Best unsafe:** 09 lightgbm_tuned (ROC=0.9793, recall=0.9114)
- **Best previous safe:** 09 lightgbm_tuned (ROC=0.9450, recall=0.7707)
- **Best optimized safe:** 53 softvote_etbag_lgbmwinner_xgb (ROC=0.9455, recall=0.7784)
- **Avg ROC gap (matched 01–15):** unsafe ahead by 0.0354

## Matched ladder (optimized safe vs unsafe)

| Exp | Optimized safe | ROC opt | ROC unsafe | ΔROC | Recall opt | Recall unsafe |
|---|---|---:|---:|---:|---:|---:|
| 08 | xgboost_tuned_safe | 0.9410 | 0.9788 | -0.0378 | 0.7697 | 0.9104 |
| 09 | lightgbm_tuned_safe | 0.9409 | 0.9793 | -0.0383 | 0.7871 | 0.9114 |
| 01 | baseline_safe_balanced | 0.9381 | 0.9730 | -0.0350 | 0.8112 | 0.8931 |
| 12 | calibrated_recall_f1_safe | 0.9375 | 0.9780 | -0.0404 | 0.9374 | 0.8815 |
| 02 | lgbm_optimized_defaults | 0.9372 | 0.9720 | -0.0348 | 0.7977 | 0.8805 |
| 06 | full_optimized_safe_fe | 0.9371 | 0.9778 | -0.0407 | 0.8006 | 0.9085 |
| 15 | softvote_lgbm_xgb_hgb_safe | 0.9368 | 0.9781 | -0.0413 | 0.8025 | 0.9114 |
| 10 | histgb_balanced_safe | 0.9366 | 0.9776 | -0.0409 | 0.8083 | 0.9123 |
| 14 | safe_champion_lgbm | 0.9366 | 0.9387 | -0.0021 | 0.8015 | 0.8025 |
| 13 | stacking_4model_safe | 0.9364 | 0.9737 | -0.0373 | 0.8141 | 0.9143 |
| 04 | safe_time_emphasis | 0.9362 | 0.9740 | -0.0378 | 0.7987 | 0.8979 |
| 05 | safe_geo_emphasis | 0.9360 | 0.9682 | -0.0321 | 0.8025 | 0.8796 |
| 03 | safe_pricing_emphasis | 0.9360 | 0.9767 | -0.0407 | 0.7987 | 0.8998 |
| 07 | safe_feature_selection | 0.9333 | 0.9689 | -0.0356 | 0.8015 | 0.8786 |

## Plots
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_vs_unsafe_side_by_side.png
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_vs_unsafe_roc_gap.png
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_vs_unsafe_champions.png
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_top_vs_unsafe_ceiling.png
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_vs_unsafe_comparison.csv

