# Category-codes campaign: encoding factorized codes as categories (DCLAB-R11)

Every arm runs on identical training folds; no locked holdout is read. Differences are fold-paired, shown as mean [Nadeau-Bengio corrected 95% interval]. Training-CV evidence, not production approval.

## CAT-001 · playbook ladder (FeatureEngineer + LightGBM)

**Setup.** 7 public datasets with declared codes; model_building_50_v1 protocol (4000 stratified rows, 80% training split, blocked columns removed); FeatureEngineer ladder (raw, logs, ratios, interactions, full_fe, selected) with LightGBM under three arms (legacy, numbers, one_hot), plus a raw-matrix screen of 5 families under numbers vs one_hot; RepeatedStratifiedKFold(3x3, seed 42) on training rows only, every arm on identical folds; paired fold differences with Nadeau-Bengio corrected 95% intervals. No holdout row is read.

| Dataset | Legacy stage, 3 folds → 9 folds | Legacy stage − raw | One-hot raw − legacy stage | LightGBM raw: one_hot − numbers | Logistic | Extra Trees | Reproduced |
|---|---|---|---|---|---|---|---|
| bank_marketing | ratios → raw | +0.0012 [-0.0185, +0.0209] | -0.0065 [-0.0305, +0.0175] | -0.0053 [-0.0247, +0.0141] | +0.0377 [+0.0085, +0.0669] | -0.0033 [-0.0322, +0.0256] | yes |
| german_credit | interactions → raw | +0.0006 [-0.0296, +0.0308] | +0.0178 [-0.0131, +0.0488] | +0.0184 [-0.0076, +0.0445] | +0.0058 [-0.0313, +0.0428] | -0.0012 [-0.0288, +0.0265] | yes |
| adult | raw → raw | +0.0000 [+0.0000, +0.0000] | +0.0040 [-0.0008, +0.0088] | +0.0040 [-0.0008, +0.0088] | +0.0395 [+0.0211, +0.0578] | -0.0141 [-0.0229, -0.0054] | yes |
| credit_default | raw → raw | +0.0000 [+0.0000, +0.0000] | -0.0011 [-0.0129, +0.0107] | -0.0011 [-0.0129, +0.0107] | +0.0176 [-0.0243, +0.0595] | -0.0210 [-0.0305, -0.0114] | yes |
| heart_disease | raw → raw | +0.0000 [+0.0000, +0.0000] | +0.0044 [-0.0154, +0.0241] | +0.0044 [-0.0154, +0.0241] | +0.0075 [-0.0073, +0.0224] | +0.0032 [-0.0192, +0.0255] | yes |
| online_shoppers | raw → raw | +0.0000 [+0.0000, +0.0000] | +0.0002 [-0.0148, +0.0152] | +0.0002 [-0.0148, +0.0152] | -0.0015 [-0.0269, +0.0238] | -0.0299 [-0.0536, -0.0062] | yes |
| mushroom | raw → raw | +0.0000 [+0.0000, +0.0000] | -0.0004 [-0.0024, +0.0017] | -0.0004 [-0.0024, +0.0017] | +0.0009 [-0.0002, +0.0019] | +0.0000 [+0.0000, +0.0000] | yes |

- **fact**: On the first repeat (the model_building_50_v1 folds) the legacy arm selects the derived stage EXP-008/EXP-028 recorded; over all 9 folds that lift is bank_marketing `ratios` minus raw +0.0012 [-0.0185, +0.0209], selected over 9 folds: legacy `raw`, numbers `raw`, one_hot `raw`; german_credit `interactions` minus raw +0.0006 [-0.0296, +0.0308], selected over 9 folds: legacy `raw`, numbers `raw`, one_hot `raw`. The lift that motivated this campaign is within fold noise.
- **fact**: The one-hot raw matrix against the legacy arm's recorded derived stage, LightGBM ROC-AUC over 9 paired folds: bank_marketing -0.0065 [-0.0305, +0.0175] (3/9 folds up); german_credit +0.0178 [-0.0131, +0.0488] (6/9 folds up). One-hot does not reliably recover what arithmetic on codes appeared to add.
- **fact**: Raw-matrix LightGBM, one_hot minus numbers ROC-AUC: bank_marketing -0.0053 (within noise, 2/9 folds up); german_credit +0.0184 (within noise, 8/9 folds up); adult +0.0040 (within noise, 9/9 folds up); credit_default -0.0011 (within noise, 3/9 folds up); heart_disease +0.0044 (within noise, 6/9 folds up); online_shoppers +0.0002 (within noise, 5/9 folds up); mushroom -0.0004 (within noise, 0/9 folds up).
- **fact**: Raw-matrix five-family screen, one_hot minus numbers ROC-AUC (better/worse = corrected 95% interval excludes zero): logistic_regression: better on bank_marketing +0.0377, adult +0.0395, worse on none, within noise on 5; extra_trees: better on none, worse on adult -0.0141, credit_default -0.0210, online_shoppers -0.0299, within noise on 4; hist_gradient_boosting: better on none, worse on none, within noise on 7; lightgbm: better on none, worse on none, within noise on 7; xgboost: better on none, worse on none, within noise on 7. Logistic regression, which can only use a code through its arbitrary order, is where the encoding pays.
- **fact**: The legacy arm reproduces the recorded model_building_50_v1 feature-stage ROC-AUC exactly on the first repeat's folds for 7 of 7 datasets (bank_marketing, german_credit, adult, credit_default, heart_disease, online_shoppers, mushroom).

## CAT-002 · DCLab notebook recipes

**Setup.** 7 R&D samples loaded into the DCLab notebook with the catalog's contract and code declaration (default settings: up to 20,000 rows, stratified 20% holdout locked and unread, 3 stratified training folds). Every registered recipe with the stage model, plus the raw recipe with every other notebook model family, under numbers (codes passed as numbers) and one_hot (codes one-hot per fit fold); identical folds; paired fold differences with Nadeau-Bengio corrected 95% intervals.

| Dataset | Metric | Raw LightGBM one_hot − numbers | Logistic | Extra Trees | Selected recipe numbers → one_hot |
|---|---|---|---|---|---|
| bank_marketing | roc_auc | -0.0015 [-0.0168, +0.0138] | +0.0311 [-0.0127, +0.0749] | -0.0054 [-0.0146, +0.0037] | raw → raw |
| german_credit | roc_auc | +0.0125 [-0.0802, +0.1051] | +0.0172 [-0.0197, +0.0540] | -0.0079 [-0.0545, +0.0388] | raw → raw |
| adult | roc_auc | +0.0004 [-0.0064, +0.0072] | +0.0563 [+0.0314, +0.0812] | -0.0101 [-0.0240, +0.0038] | raw → raw |
| credit_default | roc_auc | +0.0011 [-0.0091, +0.0114] | +0.0417 [-0.0052, +0.0886] | -0.0087 [-0.0164, -0.0010] | raw → raw |
| heart_disease | roc_auc | +0.0184 [-0.0386, +0.0754] | +0.0050 [-0.0783, +0.0882] | +0.0039 [-0.0234, +0.0312] | poly2 → raw |
| online_shoppers | roc_auc | -0.0104 [-0.0400, +0.0192] | +0.0694 [+0.0410, +0.0979] | -0.0041 [-0.0301, +0.0219] | raw → raw |
| mushroom | roc_auc | +0.0000 [+0.0000, +0.0000] | +0.0005 [-0.0003, +0.0013] | +0.0000 [+0.0000, +0.0000] | selected_mi → raw |

- **fact**: DCLab notebook raw recipe, one_hot minus numbers training-CV ROC-AUC on the engine's own folds: bank_marketing LightGBM -0.0015 (within noise), logistic +0.0311 (within noise); german_credit LightGBM +0.0125 (within noise), logistic +0.0172 (within noise); adult LightGBM +0.0004 (within noise), logistic +0.0563 (better); credit_default LightGBM +0.0011 (within noise), logistic +0.0417 (within noise); heart_disease LightGBM +0.0184 (within noise), logistic +0.0050 (within noise); online_shoppers LightGBM -0.0104 (within noise), logistic +0.0694 (better); mushroom LightGBM +0.0000 (within noise), logistic +0.0005 (within noise).
- **fact**: Raw recipe by model family, one_hot minus numbers: extra_trees mean -0.0046 across 7 datasets (0 better, 1 worse); hist_gradient_boosting mean +0.0034 across 7 datasets (0 better, 0 worse); lightgbm mean +0.0029 across 7 datasets (0 better, 0 worse); logistic_regression mean +0.0316 across 7 datasets (2 better, 0 worse); xgboost mean +0.0032 across 7 datasets (0 better, 0 worse).
- **fact**: The engine's recipe rule selects a different recipe after the change on heart_disease (poly2 -> raw), mushroom (selected_mi -> raw).
- **risk**: selected_mi under one_hot minus numbers: bank_marketing -0.0411; german_credit -0.0184; adult -0.0010; credit_default -0.0087; heart_disease +0.0020; mushroom -0.0054. Its k is 60% of the input columns but it now picks among the encoded columns, so it keeps fewer inputs. With one-hot codes the rule selected it on no dataset; sizing k on encoded columns is a separate change.

Limitations (CAT-001): Training-CV evidence on at most 3,200 rows per dataset; not holdout confirmation and not production approval. Repeated folds share training rows; even the corrected interval is approximate. The cached tables lost the category labels, so the one-hot columns carry codes; the encoding is equivalent, the interpretation is not.

Limitations (CAT-002): Three folds give wide intervals; a direction seen on several datasets is stronger evidence than any single interval. Training-CV evidence only; the notebook's holdout stays sealed and nothing here approves a model for production.

Re-run: `python -m dclab_rnd.code_encoding run --force` (results under `results/`, never edited by hand).
