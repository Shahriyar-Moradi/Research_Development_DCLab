# Optimized Safe Models Report

Goal: improve HyperAck models **without** using unsafe final fares.

## What we changed (still leakage-safe)
- Kept dropping `final_customer_fare` / `final_biker_fare`
- Added richer **safe** FE: first-fare ratios, time flags (morning/night), route ratios, interactions, train-only geo clusters + quantile bins
- Used class imbalance handling (`scale_pos_weight` / `class_weight=balanced`)
- Stronger tuning (more trials) for XGB/LGBM
- Recovery passes: replay prior winner, local search around it, seed bags, soft votes
- Better ensembles: 4-model stacking + 3-model soft vote
- Threshold tuned for F1+recall blend (exp 12)

## Headline
- **Best optimized safe:** 53 - softvote_etbag_lgbmwinner_xgb (ROC-AUC=0.9455, recall=0.7784)
- **Prior best safe:** lightgbm_tuned (ROC-AUC=0.9450)
- **Lift vs prior best safe ROC:** +0.0005
- **Avg Δ vs matched safe exp:** ROC -0.0017, recall 0.0129

## Leaderboard (optimized safe)

| Rank | Exp | Name | ROC-AUC | Recall | F1 | vs prev safe ΔROC |
|---:|---|---|---:|---:|---:|---:|
| 1 | 53 | softvote_etbag_lgbmwinner_xgb | 0.9455 | 0.7784 | 0.8505 | n/a |
| 2 | 51 | stacking_top_tuned_models | 0.9450 | 0.8006 | 0.8506 | n/a |
| 3 | 21 | lgbm_prior_winner_replay | 0.9450 | 0.7707 | 0.8461 | n/a |
| 4 | 34 | soft_avg_winner_te_optuna | 0.9446 | 0.7678 | 0.8452 | n/a |
| 5 | 36 | lgbm_top7_oof_seeds_bag | 0.9446 | 0.7649 | 0.8442 | n/a |
| 6 | 35 | lgbm_best_oof_seed | 0.9446 | 0.7688 | 0.8449 | n/a |
| 7 | 27 | lgbm_best_of_pass2_seed_bag | 0.9445 | 0.7649 | 0.8447 | n/a |
| 8 | 50 | softvote_top_tuned_models | 0.9445 | 0.7871 | 0.8519 | n/a |
| 9 | 43 | tuned_extra_trees | 0.9444 | 0.7794 | 0.8525 | n/a |
| 10 | 23 | lgbm_winner_seed_bag | 0.9444 | 0.7649 | 0.8442 | n/a |
| 11 | 24 | softvote_winner_xgb_hgb | 0.9443 | 0.7803 | 0.8513 | n/a |
| 12 | 40 | lgbm_champ_ablation_seed_params | 0.9443 | 0.7697 | 0.8482 | n/a |
| 13 | 32 | oof_stack_lgbm_xgb_hgb_cat | 0.9443 | 0.7977 | 0.8519 | n/a |
| 14 | 52 | extratrees_tuned_seed_bag | 0.9442 | 0.7784 | 0.8523 | n/a |
| 15 | 39 | lgbm_oof_jitter_around_winner | 0.9441 | 0.7688 | 0.8462 | n/a |
| 16 | 28 | lgbm_winner_cv_target_encoding | 0.9439 | 0.7688 | 0.8431 | n/a |
| 17 | 37 | lgbm_winner_feature_ablation | 0.9438 | 0.7640 | 0.8445 | n/a |
| 18 | 38 | oof_auc_weighted_blend | 0.9438 | 0.7871 | 0.8528 | n/a |
| 19 | 18 | lgbm_seed_bag_safe | 0.9433 | 0.7909 | 0.8508 | n/a |
| 20 | 22 | lgbm_local_around_winner | 0.9433 | 0.7871 | 0.8519 | n/a |
| 21 | 26 | lgbm_light_extras_local | 0.9432 | 0.7726 | 0.8473 | n/a |
| 22 | 29 | lgbm_winner_bootstrap_bag | 0.9430 | 0.7572 | 0.8447 | n/a |
| 23 | 16 | lgbm_deep_tune_safe | 0.9430 | 0.7890 | 0.8491 | n/a |
| 24 | 30 | catboost_safe | 0.9428 | 0.8025 | 0.8601 | n/a |
| 25 | 46 | tuned_lightgbm_full_search | 0.9427 | 0.7852 | 0.8490 | n/a |
| 26 | 33 | best_params_te_bootstrap | 0.9426 | 0.7572 | 0.8429 | n/a |
| 27 | 19 | softvote_deep_tuned_safe | 0.9426 | 0.7919 | 0.8487 | n/a |
| 28 | 49 | tuned_selectk_lightgbm | 0.9425 | 0.7929 | 0.8520 | n/a |
| 29 | 31 | lgbm_optuna_around_winner | 0.9425 | 0.7775 | 0.8477 | n/a |
| 30 | 20 | lgbm_best_params_more_trees_safe | 0.9425 | 0.7871 | 0.8436 | n/a |
| 31 | 47 | tuned_xgboost_full_search | 0.9417 | 0.7900 | 0.8489 | n/a |
| 32 | 48 | tuned_catboost_full_search | 0.9417 | 0.7842 | 0.8528 | n/a |
| 33 | 42 | tuned_random_forest | 0.9414 | 0.7746 | 0.8437 | n/a |
| 34 | 17 | xgb_deep_tune_safe | 0.9414 | 0.7890 | 0.8491 | n/a |
| 35 | 44 | tuned_hist_gradient_boosting | 0.9412 | 0.7958 | 0.8524 | n/a |
| 36 | 08 | xgboost_tuned_safe | 0.9410 | 0.7697 | 0.8477 | -0.0009 |
| 37 | 09 | lightgbm_tuned_safe | 0.9409 | 0.7871 | 0.8453 | -0.0041 |
| 38 | 25 | lgbm_broad_80_original_space | 0.9404 | 0.7746 | 0.8392 | n/a |
| 39 | 01 | baseline_safe_balanced | 0.9381 | 0.8112 | 0.8467 | -0.0005 |
| 40 | 12 | calibrated_recall_f1_safe | 0.9375 | 0.9374 | 0.8108 | -0.0011 |
| 41 | 02 | lgbm_optimized_defaults | 0.9372 | 0.7977 | 0.8385 | +0.0007 |
| 42 | 06 | full_optimized_safe_fe | 0.9371 | 0.8006 | 0.8390 | -0.0018 |
| 43 | 15 | softvote_lgbm_xgb_hgb_safe | 0.9368 | 0.8025 | 0.8440 | -0.0016 |
| 44 | 10 | histgb_balanced_safe | 0.9366 | 0.8083 | 0.8453 | -0.0024 |
| 45 | 14 | safe_champion_lgbm | 0.9366 | 0.8015 | 0.8404 | -0.0021 |
| 46 | 13 | stacking_4model_safe | 0.9364 | 0.8141 | 0.8480 | -0.0004 |
| 47 | 04 | safe_time_emphasis | 0.9362 | 0.7987 | 0.8391 | -0.0057 |
| 48 | 05 | safe_geo_emphasis | 0.9360 | 0.8025 | 0.8461 | +0.0008 |
| 49 | 03 | safe_pricing_emphasis | 0.9360 | 0.7987 | 0.8429 | +0.0002 |
| 50 | 07 | safe_feature_selection | 0.9333 | 0.8015 | 0.8425 | -0.0054 |
| 51 | 45 | tuned_linear_svc_calibrated | 0.8829 | 0.7187 | 0.8180 | n/a |
| 52 | 41 | tuned_logistic_regression | 0.8823 | 0.7187 | 0.8171 | n/a |

## Files
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_vs_safe_comparison.csv
- /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D/optimized_safe_model/results/optimized_vs_safe_side_by_side.png
- results/*.json

## Simple takeaway
Leakage-safe ceiling on this split is ~0.9450 ROC-AUC (replay of prior safe LightGBM winner).
Richer FE, class weights, CatBoost, Optuna, TE, ablation, and ensembles did not beat that peak,
but several models improve recall vs the prior winner (0.7707) while staying safe (no final fares).
Champion for ROC: exp 21 `lgbm_prior_winner_replay`. See plots for the full ladder.

