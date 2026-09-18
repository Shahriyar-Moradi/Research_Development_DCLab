# DCLab LLM Review Report

Campaign: `model_building_50_v1`  
Reviews: **50**  
Model: `gpt-5.4-mini`

> LLM is critic/proposer only. Metrics in result JSON remain authoritative.

## Executive summary

| ID | Dataset | Stage | Confidence | Decision | Next experiment |
|---|---|---|---|---|---|
| EXP-001 | adult | data_understanding | medium | Do not proceed to modeling yet; first resolve representation and production-availability risks with a targeted validatio | Decision-time and raw-category audit for adult features |
| EXP-002 | adult | leakage_audit | medium | Insufficient evidence to conclude which columns are leakage or target proxies. The current package supports only a detec | Column-level decision-time contract audit for adult |
| EXP-003 | adult | feature_engineering | medium | Do not accept the claim that the raw/selected recipe should be used next based on this package alone. The smallest defen | Nested-CV rerun of the selection rule with fold-local feature fitting |
| EXP-004 | adult | model_selection | medium | Provisional selection only: lightgbm is the current training-CV leader under the provided rule, but the evidence is insu | Repeated stratified CV comparison of lightgbm vs hist_gradient_boosting with leakage audit |
| EXP-005 | adult | optimization_reliability | high | Insufficient evidence to conclude that optimization produced a meaningful CV gain. The once-consumed holdout performance | Minimal CV delta audit against baseline |
| EXP-006 | bank_marketing | data_understanding | medium | Do not approve modeling readiness yet. The dataset description is usable, but production-availability and leakage risks  | Decision-time availability and raw-category restoration audit |
| EXP-007 | bank_marketing | leakage_audit | medium | Do not approve duration for use until decision-time semantics and production availability are verified; treat it as a le | Decision-time availability audit for duration |
| EXP-008 | bank_marketing | feature_engineering | medium | Do not accept the recommendation as supported; request a minimal confirmatory experiment before choosing a recipe. | Paired repeated-CV comparison of raw vs ratios with leakage audit |
| EXP-009 | bank_marketing | model_selection | medium | Provisional training-CV leader only; do not treat as a validated algorithm winner. | Repeat stratified CV with leakage audit for extra_trees vs hist_gradient_boosting |
| EXP-010 | bank_marketing | optimization_reliability | medium | Do not conclude that optimization produced a meaningful CV gain from this package. Treat the holdout score as a consumed | Paired CV comparison of baseline vs tuned recipe on identical folds |
| EXP-011 | breast_cancer | data_understanding | medium | Do not proceed to modeling until production input representation and split strategy are verified. The current evidence i | Production-schema and decision-time availability audit |
| EXP-012 | breast_cancer | leakage_audit | medium | Insufficient evidence to approve or clear any columns. The package does not establish prediction-time availability, prod | Column-level prediction-time contract audit with decision-time availability matrix |
| EXP-013 | breast_cancer | feature_engineering | high | Do not accept the recommendation as supported. Require a rule-consistency check and a leakage/validation audit before an | Rule-consistency and leakage audit with one held-out evaluation |
| EXP-014 | breast_cancer | model_selection | medium | Provisional training-CV candidate only; not enough evidence to approve a final model choice or deployment decision. | Locked-holdout verification of the current CV leader with fold-safe preprocessing |
| EXP-015 | breast_cancer | optimization_reliability | medium | Do not claim a meaningful optimization gain from this package. Treat the holdout score as a consumed, one-time performan | Paired baseline-vs-tuned nested CV with untouched holdout lock |
| EXP-016 | heart_disease | data_understanding | medium | Do not proceed to modeling until decision-time availability, production-availability, and leakage checks are completed f | Decision-time and lineage audit for flagged heart_disease features |
| EXP-017 | heart_disease | leakage_audit | medium | Insufficient evidence to determine which columns violate the prediction-time contract or are suspicious target proxies.  | Column-level decision-time contract audit for heart_disease |
| EXP-018 | heart_disease | feature_engineering | medium | Do not treat the raw recipe as validated for production or as a proven improvement. At most, it is the current smallest  | Single held-out validation of raw vs logs under the same fold-wise pipeline |
| EXP-019 | heart_disease | model_selection | medium | Do not treat extra_trees as a confirmed winner. Use it only as a provisional CV leader pending a locked-holdout check an | Repeated nested CV comparison: extra_trees vs logistic_regression |
| EXP-020 | heart_disease | optimization_reliability | medium | Do not accept any claim that optimization produced a meaningful gain from this package alone. Treat the holdout score as | Baseline-vs-tuned CV audit with untouched evaluation split |
| EXP-021 | credit_default | data_understanding | medium | Do not proceed to modeling until production-availability and leakage checks are completed. | Decision-time and raw-category availability audit |
| EXP-022 | credit_default | leakage_audit | medium | Insufficient evidence to identify or clear any specific columns as contract-safe. No feature should be approved from thi | Column-level decision-time contract audit for candidate proxies |
| EXP-023 | credit_default | feature_engineering | medium | Do not accept the recommendation as stated; the evidence is insufficient to justify a feature choice beyond a provisiona | Nested CV comparison of raw vs selected recipe with fixed fold splits |
| EXP-024 | credit_default | model_selection | medium | Do not treat extra_trees as the final algorithm choice yet; treat it as the current CV leader only, pending repeat/outer | Repeat the raw-stage CV comparison with repeated stratified folds |
| EXP-025 | credit_default | optimization_reliability | medium | Do not accept the claim that optimization produced a meaningful CV gain on the basis of this package alone. Treat the ho | Baseline-vs-tuned CV audit with decision-time feature check |
| EXP-026 | german_credit | data_understanding | medium | Do not approve modeling yet. The dataset can be described, but the evidence is insufficient to clear production-availabi | Raw-schema and availability audit for all 20 features |
| EXP-027 | german_credit | leakage_audit | medium | Insufficient evidence to identify or approve any specific column as safe or unsafe. The current package is a leakage-aud | Column-level decision-time contract audit for german_credit |
| EXP-028 | german_credit | feature_engineering | medium | Do not accept the recommendation as written; the evidence is insufficient and internally inconsistent with the selection | Fold-local ablation of interactions vs selected under nested validation |
| EXP-029 | german_credit | model_selection | medium | Do not treat extra_trees as the final winner from this package alone. The evidence is sufficient only to nominate a prov | Rule-verification and leakage-check rerun on the top two families |
| EXP-030 | german_credit | optimization_reliability | medium | Do not claim a meaningful optimization gain from this package alone. Treat the holdout score as a finalized retrospectiv | Baseline-vs-tuned CV comparison on a fresh split with locked recipe |
| EXP-031 | mushroom | data_understanding | medium | Do not proceed to modeling yet; resolve data representation, leakage, and production-availability questions first. | Raw-category and decision-time availability audit for all 22 features |
| EXP-032 | mushroom | leakage_audit | medium | Insufficient evidence to identify or approve any real columns as safe or unsafe. The current package supports only that  | Column-level prediction-time contract audit with safe/unsafe ablation |
| EXP-033 | mushroom | feature_engineering | medium | Do not accept the recommendation to use the raw recipe as stated. The smallest defensible conclusion is that the selecte | Nested holdout comparison of raw vs selected feature recipe |
| EXP-034 | mushroom | model_selection | medium | Do not accept the claim that extra_trees is the strongest stable training-CV candidate from this package alone. The pack | Repeat stratified nested CV with locked holdout comparison for top two families |
| EXP-035 | mushroom | optimization_reliability | high | Do not conclude that optimization produced a meaningful CV gain from this package. Treat the holdout result as a complet | Baseline-vs-tuned nested CV comparison with leakage check |
| EXP-036 | spambase | data_understanding | medium | Do not proceed to modeling until decision-time availability, raw-source reconstruction, and leakage checks are completed | Decision-time feature availability and raw-semantic reconstruction audit |
| EXP-037 | spambase | leakage_audit | medium | Insufficient evidence to conclude which columns violate the prediction-time contract or are suspicious target proxies. | Column-level prediction-time availability audit for spambase |
| EXP-038 | spambase | feature_engineering | high | Do not accept the recommendation as written; the evidence is insufficient and internally inconsistent. | Nested fold-safe comparison of raw vs selected recipe |
| EXP-039 | spambase | model_selection | medium | Provisional training-CV candidate only; do not treat as final model choice without a locked holdout and repeated validat | Repeated stratified CV comparison of hist_gradient_boosting vs xgboost with a locked holdout |
| EXP-040 | spambase | optimization_reliability | medium | Hold the optimization claim as unverified; accept only the narrow factual statement that the preselected recipe achieved | Nested CV check of optimization gain with untouched final evaluation |
| EXP-041 | online_shoppers | data_understanding | medium | Do not proceed to modeling until decision-time feature availability and encoding semantics are verified, and until a tim | Time-aware schema and availability audit for all 17 features |
| EXP-042 | online_shoppers | leakage_audit | medium | Do not approve PageValues for use until decision-time semantics and production-availability are verified; treat it as a  | Decision-time availability check for PageValues |
| EXP-043 | online_shoppers | feature_engineering | medium | Do not accept the recommendation as stated. The smallest defensible next candidate from this package is the raw recipe,  | Single holdout validation of raw vs full_fe under the stated tolerance rule |
| EXP-044 | online_shoppers | model_selection | medium | Do not promote a universal algorithm-family winner from this package. At most, treat extra_trees as the provisional trai | Locked holdout verification for the current CV leader |
| EXP-045 | online_shoppers | optimization_reliability | medium | Do not accept the claim that optimization produced a meaningful CV gain from this package alone. Treat the holdout score | Baseline-vs-tuned CV audit with fold-wise deltas |
| EXP-046 | wine_quality | data_understanding | medium | Do not approve modeling yet. The dataset description is partially supported, but production-availability and leakage rev | Raw-schema and decision-time availability audit |
| EXP-047 | wine_quality | leakage_audit | medium | Insufficient evidence to identify or approve any column as safe or unsafe. A decision-time and production-availability r | Column-level decision-time availability audit with safe-vs-unsafe comparison |
| EXP-048 | wine_quality | feature_engineering | medium | Do not accept the recommendation as written. The ratios recipe is the best reported candidate for the next falsifiable s | Nested fold-wise comparison of raw vs ratios with leakage checks |
| EXP-049 | wine_quality | model_selection | medium | Provisional only: extra_trees can be treated as the current training-CV candidate, but not as a validated winner. Do not | Repeated leakage-safe CV comparison of extra_trees vs lightgbm |
| EXP-050 | wine_quality | optimization_reliability | medium | Hold. The holdout result can be reported descriptively, but the optimization-gain claim is unsupported without the missi | Baseline-vs-tuned CV delta audit with decision-time feature check |

## Detailed reviews

### EXP-001 · adult

- **Confidence:** medium
- **Decision:** Do not proceed to modeling yet; first resolve representation and production-availability risks with a targeted validation pass.
- **Interpretation:** This is a dataset-understanding package, not modeling evidence. It supports only a narrow factual summary: the dataset size, feature count, and class balance, plus a preliminary profile suggesting no obvious missingness or near-constant issues in the inspected fields. However, the package does not establish production readiness, because the categorical representation appears to have been altered by prior factorization and the shift check is explicitly random-split only. The evidence is insufficient to conclude that the dataset is safe for modeling without restoring raw category semantics and checking decision-time availability and leakage risks.
- **Next experiment:** Decision-time and raw-category audit for adult features — Pass only if every feature is mapped to a decision-time availability status, every categorical feature is shown in raw semantic form with a production encoder plan, and no feature is flagged as post-outcome, unavailable, or schema-incompatible.
- **Challenges:**
  - `EXP-001-C1` (medium): The statement is factual, but it omits whether the 14 features are raw, encoded, or cached representations; the bundle itself warns that categoricals were factorized before this campaign, which affects downstream production interpretation.
  - `EXP-001-C2` (high): The claim that 0 features triggered review rules is not sufficient to establish safety because the review rules shown are limited to missingness, identifier, near-constant, and random-split shift; they do not cover target leakage, decision-time availability, or production schema mismatch.

### EXP-002 · adult

- **Confidence:** medium
- **Decision:** Insufficient evidence to conclude which columns are leakage or target proxies. The current package supports only a detector sanity check, not a column-level leakage determination.
- **Interpretation:** The package shows only that the detector can flag an injected exact-target canary, but it does not identify any real columns that violate the prediction-time contract or behave as target proxies. The absence of declared leakage features is not evidence of absence, and the missing safe-vs-unsafe CV result leaves the audit unsupported for ranking or validating suspicious columns.
- **Next experiment:** Column-level decision-time contract audit for adult — For each column, provide decision-time availability status plus a minimal evidence table with name-based review, uniqueness, and univariate target association; flag any column that fails the contract or is a strong proxy candidate. Success is achieved if the audit produces a reproducible list of flagged columns with explicit rationale, or confirms none after review.
- **Challenges:**
  - `EXP-002-C1` (medium): The claim only demonstrates detector behavior on a synthetic canary. It does not establish that any real dataset column is a leakage source or proxy.
  - `EXP-002-C2` (high): The statement 'Declared decision-time exclusions for adult: none' is unsupported by the provided evidence because no decision-time semantics or production-availability review is included.

### EXP-003 · adult

- **Confidence:** medium
- **Decision:** Do not accept the claim that the raw/selected recipe should be used next based on this package alone. The smallest defensible action is to rerun the selection logic against the actual best stage and verify the criterion on an untouched validation split, with leakage checks and decision-time availability review before any feature approval.
- **Interpretation:** The evidence does not support the recommendation to use the selected/raw recipe as the next screen under the stated selection rule. Based on the reported means, the best validation evidence is the interactions stage, while the selected stage is materially worse on mean ROC-AUC and also does not satisfy the stated within-0.002 criterion. The package also does not establish that the chosen recipe avoids feature dilution in a decision-safe sense, because the feature-count reduction is paired with lower validation evidence and the stability table shows several engineered features appearing in only 1-2 folds, which raises selection-instability concerns rather than resolving them. This is predictive evidence only; it does not justify causal claims about the recipe itself.
- **Next experiment:** Nested-CV rerun of the selection rule with fold-local feature fitting — On an untouched outer validation split, the chosen recipe must satisfy the predeclared within-0.002 mean ROC-AUC rule relative to the best stage under the same protocol, and the selected matrix must not lose more than the predeclared tolerance versus the best stage; otherwise the claim fails.
- **Challenges:**
  - `EXP-003-C1` (high): The recommendation says to use the raw feature recipe, but the reported selection rule would require the smallest matrix within 0.002 of the best stage; the selected stage is not within that tolerance of the best reported mean ROC-AUC, and the raw stage is not the best stage by mean ROC-AUC.
  - `EXP-003-C1` (high): The claim implies the recipe improves validation evidence without feature dilution, but the selected stage has lower mean ROC-AUC than raw and the larger interaction/full_fe stages do not improve over raw either.
  - `EXP-003-C1` (high): The evidence does not show fold-local fitting for all engineered features, so leakage from train-fitted transformations or selection cannot be ruled out.
  - `EXP-003-C1` (medium): The feature stability table shows several engineered features with observed_folds below 3, which weakens the case that the recipe is stable enough to prefer.

### EXP-004 · adult

- **Confidence:** medium
- **Decision:** Provisional selection only: lightgbm is the current training-CV leader under the provided rule, but the evidence is insufficient to finalize an algorithm family choice for production or to claim stability beyond this single 3-fold run.
- **Interpretation:** Based on the stated selection rule, lightgbm is the top training-CV candidate, but the margin over hist_gradient_boosting is extremely small and may not be practically meaningful without repeated CV or a locked holdout. The evidence supports a ranking under the chosen heuristic, not a claim that lightgbm is universally best.
- **Next experiment:** Repeated stratified CV comparison of lightgbm vs hist_gradient_boosting with leakage audit — Run at least 10 repeated stratified CV evaluations with identical preprocessing inside folds; declare lightgbm preferred only if its mean ROC-AUC remains higher than hist_gradient_boosting and the paired fold-wise difference is consistently positive in a majority of repeats, with no leakage findings.
- **Challenges:**
  - `EXP-004-C1` (medium): The statement 'lightgbm is the training-CV model candidate' is supported only as a ranking under the stated heuristic, not as a robust winner; hist_gradient_boosting has nearly identical mean ROC-AUC and a different runtime profile.
  - `EXP-004-C1` (high): The evidence does not show repeated CV or a locked holdout, so the stability implied by 'strongest stable training-CV evidence' is not well established.
  - `EXP-004-C1` (high): No leakage audit is included for raw features, fold construction, or any preprocessing steps, so the validity of the CV comparison cannot be fully verified.
  - `EXP-004-C1` (high): The recommendation could be misread as a deployment approval, but there is no decision-time or production-availability review.

### EXP-005 · adult

- **Confidence:** high
- **Decision:** Insufficient evidence to conclude that optimization produced a meaningful CV gain. The once-consumed holdout performance is reported, but it should not be used to justify feature promotion or deployment without decision-time and production-availability review.
- **Interpretation:** The holdout score is a factual performance result, but the package does not provide enough evidence to answer whether optimization produced a meaningful CV gain. The only explicit optimization criterion is a 0.001 mean-CV improvement threshold, yet no baseline CV score, tuned CV score, or delta is shown. The feature evidence supports a cautionary recommendation about promotion criteria, but it does not establish that any feature should be promoted or that the listed importances are decision-safe.
- **Next experiment:** Minimal CV delta audit against baseline — Report baseline mean CV ROC-AUC, tuned mean CV ROC-AUC, and their difference on the same folds; success only if tuned minus baseline >= 0.001 and the fold-wise improvement is not driven by a single outlier fold.
- **Challenges:**
  - `EXP-005-C1` (high): The holdout ROC-AUC is reported correctly as a factual metric, but the package does not support any inference that optimization was meaningful because no baseline-versus-tuned CV comparison is included.
  - `EXP-005-C2` (medium): The recommendation is directionally sound, but the evidence does not demonstrate decision-time availability or production monitoring for the listed features; it is a policy statement rather than a validated result.

### EXP-006 · bank_marketing

- **Confidence:** medium
- **Decision:** Do not approve modeling readiness yet. The dataset description is usable, but production-availability and leakage risks remain unresolved.
- **Interpretation:** This evidence supports a narrow dataset-understanding summary: the dataset is a moderately sized binary classification table with 45,211 rows, 16 cached features, and an imbalanced positive class near 11.7%. The sample feature profiles suggest no obvious missingness in the inspected columns and no strong random-split distribution shift in the reported PSI values. However, the package does not establish that the current feature representation is production-ready, because the categorical variables were already factorized in the cached matrices and the evidence only checks random splits rather than decision-time or temporal availability. Therefore, the main unresolved issues are feature semantics restoration, production-time availability of each field, and leakage/validation design beyond random holdout.
- **Next experiment:** Decision-time availability and raw-category restoration audit — For every feature, document (1) whether it is available at decision time, (2) its raw source field, and (3) whether the cached factorized representation can be reproduced from production inputs; the experiment passes only if all 16 features are confirmed available and reproducible with no unresolved temporal or semantic mismatch.
- **Challenges:**
  - `EXP-006-C1` (medium): The row count, feature count, and positive rate are factual dataset descriptors, but the package does not show how the cached factorized features relate to raw production fields; the claim is incomplete as a basis for modeling readiness.
  - `EXP-006-C2` (high): The statement that 0 features triggered review rules is too strong without the full feature list and without decision-time/temporal checks; random-split PSI is not sufficient to rule out production shift or leakage.

### EXP-007 · bank_marketing

- **Confidence:** medium
- **Decision:** Do not approve duration for use until decision-time semantics and production availability are verified; treat it as a leakage-risk candidate, not a confirmed target proxy.
- **Interpretation:** The evidence supports that duration is a suspicious feature under the declared prediction-time contract because it is explicitly excluded and it materially changes cross-validated predictive performance when included. However, the package does not establish leakage by itself: the semantics of duration at decision time, its availability in production, and whether the CV protocol matches the intended deployment setting are not demonstrated. The canary result only validates the detector on a synthetic test, not the real column.
- **Next experiment:** Decision-time availability audit for duration — For duration, produce a signed decision-time data dictionary entry and serving-path check showing whether the value exists before scoring for at least one production-like request trace; if unavailable, mark it contract-violating. If available, rerun a strict ablation with identical preprocessing and a leakage-safe split, and require the AUC delta to persist under the same split before any further suspicion is escalated.
- **Challenges:**
  - `EXP-007-C1` (medium): The canary detection only shows the detector can catch a synthetic exact-target canary. It does not validate that the detector correctly identifies real leakage columns in this dataset.
  - `EXP-007-C2` (high): Calling duration a decision-time exclusion is plausible but not proven here; the evidence only shows a declared exclusion and a CV lift. The limitation text itself says only decision-time semantics can confirm leakage.
  - `EXP-007-C2` (high): The +0.1842 AUC lift is not sufficient to label a feature as leakage or proxy; predictive lift can arise from legitimate signal, correlation structure, or preprocessing artifacts.

### EXP-008 · bank_marketing

- **Confidence:** medium
- **Decision:** Do not accept the recommendation as supported; request a minimal confirmatory experiment before choosing a recipe.
- **Interpretation:** The evidence does not support the recommendation to use the ratios recipe as the next model screen under the stated selection rule. Ratios has the highest reported mean ROC-AUC among the shown stages, but the package does not demonstrate that this improvement is robust, statistically meaningful, or free of leakage/validation artifacts. The selected stage is smaller than full_fe, but it is not within 0.002 of the best stage by the reported means, so the stated rule is not satisfied by the provided numbers. The package also mixes feature-count comparisons with model-quality comparisons without showing a decision-time or production-availability review for the generated features.
- **Next experiment:** Paired repeated-CV comparison of raw vs ratios with leakage audit — Run repeated stratified CV with identical splits for raw and ratios, plus a train-only feature-generation audit. Accept only if ratios beats raw by a positive median fold-wise ROC-AUC difference and the lower bound of the paired fold-wise difference is above 0, while no leakage is detected in the audit.
- **Challenges:**
  - `EXP-008-C1` (high): The recommendation says to use the ratios recipe for the next model screen because it achieved 0.7172 mean training-CV ROC-AUC with about 38 features. The reported mean ROC-AUC and feature_count_mean are present, but the claim is a recommendation that is not justified by a robustness check, and the stated selection rule is not satisfied by the provided means because the selected stage mean ROC-AUC is 0.7028679428349656, not within 0.002 of the best stage mean 0.7171968586626486.
  - `EXP-008-C1` (high): The claim implicitly treats higher training-CV ROC-AUC as sufficient evidence to prefer the ratios recipe. This is unsupported because no holdout, repeated CV, or statistical test is provided, so the improvement may be within sampling noise.
  - `EXP-008-C1` (medium): The claim does not address decision-time and production-availability constraints for generated ratio features, despite the stated limitation that generic mathematical feature generation may lack domain meaning.

### EXP-009 · bank_marketing

- **Confidence:** medium
- **Decision:** Provisional training-CV leader only; do not treat as a validated algorithm winner.
- **Interpretation:** Based on the provided training-CV evidence and the stated ranking rule, extra_trees is the leading candidate by the package’s own criterion. However, the evidence is only weakly stable: the ROC-AUC spread across folds is non-trivial, the sample is only 3 folds with 1 repeat, and the package does not show a locked holdout or any decision-time/production-availability review. The question asks for the strongest stable training-CV evidence, but the current package supports only a provisional training-CV preference, not a robust winner claim.
- **Next experiment:** Repeat stratified CV with leakage audit for extra_trees vs hist_gradient_boosting — Run 5x repeated stratified CV with the same preprocessing and a documented feature-availability audit; success is met only if extra_trees remains top-ranked by the predeclared score and its mean ROC-AUC exceeds the runner-up by at least one pooled fold standard error, with no leakage findings.
- **Challenges:**
  - `EXP-009-C1` (medium): The statement says extra_trees is the training-CV model candidate, but the evidence bundle’s model_results entry for extra_trees is actually model=extra_trees with mean ROC-AUC 0.7288407757459535; the claim is directionally supported, yet the package does not demonstrate that this is the strongest stable evidence under a validated protocol because only 3 folds and 1 repeat are shown.
  - `EXP-009-C1` (high): The claim uses a recommendation framing, but the evidence lacks a locked holdout and any production-availability review, so it overreaches if interpreted as a deployable choice.
  - `EXP-009-C1` (high): The claim references ratios features, but the package does not provide a leakage check for engineered ratio features or the source timestamps/availability of inputs.
  - `EXP-009-C1` (medium): The phrase 'strongest stable training-CV evidence' is not fully justified because the stability signal is based on only 3 folds and the reported fold standard deviation remains material relative to the mean.

### EXP-010 · bank_marketing

- **Confidence:** medium
- **Decision:** Do not conclude that optimization produced a meaningful CV gain from this package. Treat the holdout score as a consumed, descriptive result only. Require a direct baseline-vs-tuned CV comparison, with the same folds and metric, before any claim about optimization benefit. Do not approve feature promotion without separate decision-time and production-availability review.
- **Interpretation:** The holdout result is a single post-selection estimate with a bootstrap interval, so it supports only a performance description, not a causal or durable claim about optimization benefit. The evidence package does not include the baseline CV ROC-AUC, the tuned candidate CV ROC-AUC, or their uncertainty, so the question of whether optimization produced a meaningful CV gain is not answered from the provided evidence alone. The feature-stability and production-rule evidence appropriately cautions against promoting features from importance alone, but it does not by itself validate any feature for deployment.
- **Next experiment:** Paired CV comparison of baseline vs tuned recipe on identical folds — Run baseline and tuned models on the same CV splits; accept optimization only if mean ROC-AUC(tuned) - mean ROC-AUC(baseline) >= 0.001 and the fold-wise difference is directionally consistent enough to rule out a trivial fluctuation. Otherwise keep the baseline.
- **Challenges:**
  - `EXP-010-C1` (high): The holdout ROC-AUC is reported correctly as a factual result, but the package does not provide the baseline CV score or tuned CV score needed to answer whether optimization produced a meaningful CV gain.
  - `EXP-010-C1` (medium): Because the holdout is marked consumed, any stronger interpretation about model selection or future performance would be unsupported by the evidence provided.
  - `EXP-010-C2` (medium): The recommendation is directionally sound, but it is not backed by per-feature decision-time availability or production-contract evidence in this package.
  - `EXP-010-C2` (low): Cross-fold stability is only shown over 3 folds, which is limited evidence for a strong stability claim.

### EXP-011 · breast_cancer

- **Confidence:** medium
- **Decision:** Do not proceed to modeling until production input representation and split strategy are verified. The current evidence is sufficient for a preliminary inventory, but insufficient to clear leakage, encoding, and production-availability risks.
- **Interpretation:** This is a compact data-understanding snapshot, not a modeling validation result. The evidence supports that the dataset is small, tabular, and numerically encoded in the cached form, with no missingness shown in the displayed feature profiles. However, the package does not establish whether the current feature representation matches production-time inputs, whether any preprocessing was fit without leakage, or whether the observed random-split shift is relevant to the actual deployment distribution. The single flagged feature suggests at least one distribution-shift review item remains unresolved, but the package does not quantify how many total features were reviewed or whether the rule was applied to all 30 features.
- **Next experiment:** Production-schema and decision-time availability audit — For every one of the 30 features, document source, availability at decision time, transformation lineage, and production population method; the experiment passes only if all features are confirmed available at prediction time with no post-outcome leakage and no unresolved schema mismatch.
- **Challenges:**
  - `EXP-011-C1` (medium): The claim is a factual inventory, but it does not distinguish cached representation from raw production inputs. The statement may be incomplete for modeling readiness because feature semantics in production are not verified.
  - `EXP-011-C2` (medium): The claim says 1 feature triggered review rules, but the evidence only displays one flagged feature among ten shown profiles; it is unclear whether this is exactly one of thirty or just one in the excerpt.
  - `EXP-011-C2` (high): The cited random-split PSI is not sufficient to support any production-stability inference, and the package itself acknowledges this limitation.

### EXP-012 · breast_cancer

- **Confidence:** medium
- **Decision:** Insufficient evidence to approve or clear any columns. The package does not establish prediction-time availability, production-time availability, or target-proxy status for any real feature.
- **Interpretation:** The package supports only a very limited conclusion: the synthetic canary was detected, and no features were formally declared as leakage. It does not provide enough evidence to identify which real columns violate the prediction-time contract or act as target proxies. The absence of declared leakage features is not evidence of absence, and the missing safe_vs_unsafe_training_cv result leaves the audit incomplete.
- **Next experiment:** Column-level prediction-time contract audit with decision-time availability matrix — For every column, produce a decision-time availability label and a production-availability label; flag any column with unavailable-at-prediction-time semantics or with strong proxy indicators (e.g., near-deterministic target association, post-outcome timing, or identity-like uniqueness). The experiment succeeds if the audit yields a non-empty, reproducible set of flagged columns or else a documented empty set with explicit decision-time justification for each column.
- **Challenges:**
  - `EXP-012-C1` (medium): The canary result only shows the detector can catch a synthetic exact-target feature; it does not support any claim about real columns or the overall absence of leakage.
  - `EXP-012-C2` (high): The statement 'Declared decision-time exclusions for breast_cancer: none' is unsupported as a decision because the package provides no decision-time semantics or production-availability review for the actual columns.

### EXP-013 · breast_cancer

- **Confidence:** high
- **Decision:** Do not accept the recommendation as supported. Require a rule-consistency check and a leakage/validation audit before any feature recipe is selected for the next screen.
- **Interpretation:** The package does not support the recommendation to use the raw recipe as the next screen if the stated goal is to improve validation evidence without feature dilution. By the package's own selection rule, the selected stage should be the smallest matrix within 0.002 of the best mean training-CV ROC-AUC, but the reported selected stage is both smaller and materially worse on mean ROC-AUC than raw. The evidence instead suggests a mismatch between the selection rule and the reported chosen stage, or a missing comparison stage / missing evaluation detail. The package also does not establish that any feature recipe improves validation evidence without dilution in a decision-ready sense, because only 3-fold CV is shown and no decision-time or production-availability review is provided.
- **Next experiment:** Rule-consistency and leakage audit with one held-out evaluation — Pre-register the exact selection rule, rerun the stage comparison with one untouched hold-out split, and accept only if the chosen stage is the smallest matrix within 0.002 of the best mean CV ROC-AUC and its hold-out ROC-AUC is not worse than raw by more than a predeclared tolerance.
- **Challenges:**
  - `EXP-013-C1` (high): The recommendation says to use the raw recipe, but the stated selection rule would require the smallest matrix within 0.002 of the best mean training-CV ROC-AUC. The reported selected stage is smaller than raw yet has lower mean ROC-AUC, so the recommendation is not supported by the rule as written.
  - `EXP-013-C1` (high): The claim frames 0.9907 mean training-CV ROC-AUC as evidence that raw should be used next, but the package does not show a direct comparison against a held-out validation or test set, so the recommendation is not decision-grade.
  - `EXP-013-C1` (medium): The claim mentions 'about 30 features,' but the selected stage reports 25 features and the raw stage reports 30 features; the wording is ambiguous and may obscure the actual feature dilution tradeoff.
  - `EXP-013-C1` (medium): The limitation about domain semantics is acknowledged, but there is no evidence of decision-time availability or production freshness review for the engineered features.

### EXP-014 · breast_cancer

- **Confidence:** medium
- **Decision:** Provisional training-CV candidate only; not enough evidence to approve a final model choice or deployment decision.
- **Interpretation:** On the provided training-CV evidence alone, logistic_regression is the strongest candidate by the package’s own ranking criterion because it has the highest mean ROC-AUC and the lowest runtime among the listed models. However, the evidence is not sufficient to conclude it is the best algorithm family in a deployment sense, because the package only reports training CV, uses just 3 folds, and does not include a locked holdout or any decision-time/production-availability review. The apparent margin over the other models is small relative to the uncertainty implied by the fold variability and the limited number of folds.
- **Next experiment:** Locked-holdout verification of the current CV leader with fold-safe preprocessing — Run a single locked-holdout evaluation after retraining with the exact same CV protocol and fold-safe preprocessing; success is defined as logistic_regression remaining within the top tier on holdout ROC-AUC and not showing a materially worse calibration profile than the best competing family, with no evidence of leakage in the pipeline audit.
- **Challenges:**
  - `EXP-014-C1` (medium): The statement that logistic_regression is the training-CV model candidate is supported only within the package’s own selection rule and only on training CV; it is not supported as a universal winner or as a final model choice.
  - `EXP-014-C1` (high): The reported advantage over other models is small and based on only 3 folds, so the claim of strongest stable evidence is underpowered without repeated or nested validation.
  - `EXP-014-C1` (high): No locked holdout evidence is present, so the recommendation cannot be validated against unseen data.
  - `EXP-014-C1` (high): The evidence bundle does not document fold-safe preprocessing, so the reported ROC-AUC could be inflated by leakage if any transformations were fit outside the CV loop.

### EXP-015 · breast_cancer

- **Confidence:** medium
- **Decision:** Do not claim a meaningful optimization gain from this package. Treat the holdout score as a consumed, one-time performance estimate only. Before any feature or recipe promotion, require a pre-registered comparison against the baseline on the same CV protocol and a decision-time/production-availability audit.
- **Interpretation:** The holdout result is strong predictive evidence for this dataset, but it does not by itself establish that optimization produced a meaningful gain. The package does not provide the baseline CV score, the tuned CV score, or their uncertainty, so the core question about optimization benefit remains unsupported. The feature-ranking note is directionally cautious, but any promotion decision still needs explicit decision-time and production-availability review; importance and 3-fold stability alone are insufficient to justify feature use.
- **Next experiment:** Paired baseline-vs-tuned nested CV with untouched holdout lock — Pre-register the comparison; run paired nested or repeated CV on baseline and tuned candidate; declare optimization beneficial only if mean ROC-AUC improvement is >= 0.001 and the paired fold-wise difference is consistently positive across resamples. Do not touch the consumed holdout.
- **Challenges:**
  - `EXP-015-C1` (high): The claim reports a strong holdout ROC-AUC, but the experiment question asks whether optimization produces a meaningful CV gain; the package does not include the baseline CV score, tuned CV score, or their uncertainty, so the optimization-gain part is unsupported.
  - `EXP-015-C1` (high): Because the holdout is marked consumed, the reported holdout score should not be used as evidence for further model selection or iterative tuning without a fresh untouched evaluation set.
  - `EXP-015-C2` (medium): The recommendation is sensible, but it is not backed by explicit decision-time availability or production-availability evidence for the listed features in this package.
  - `EXP-015-C2` (medium): Cross-fold stability is based on only 3 observed folds, which is too sparse to support a strong stability conclusion.

### EXP-016 · heart_disease

- **Confidence:** medium
- **Decision:** Do not proceed to modeling until decision-time availability, production-availability, and leakage checks are completed for the flagged features and target definition.
- **Interpretation:** This package supports a basic dataset inventory and a preliminary quality screen, but it does not yet establish production readiness. The strongest evidence is descriptive: small sample size, moderate class balance, and several features flagged for distribution-shift review. The package also hints at a preprocessing-history risk for categorical semantics, but that concern is only stated in the claim limitations and is not directly evidenced in the feature profiles. The random-split PSI values are only a smoke test and should not be treated as evidence of deployment stability.
- **Next experiment:** Decision-time and lineage audit for flagged heart_disease features — For each flagged feature and the target, produce a lineage report showing raw source, extraction timestamp, decision-time availability, preprocessing steps, and split-specific fit/transform boundaries; the experiment succeeds only if no feature is derived from post-outcome information and all flagged features are available with unchanged semantics in the intended production context.
- **Challenges:**
  - `EXP-016-C1` (medium): The row count, feature count, and positive rate are descriptive facts, but the package does not show how the positive rate was computed or whether the training split was stratified; the metric is therefore not fully auditable from the provided evidence paths alone.
  - `EXP-016-C2` (medium): The statement that 3 features triggered review rules is only partially supported by the visible feature profiles; the bundle excerpt shows at least age, trestbps, and chol flagged for split_distribution_shift, but it does not expose the full rule evaluation for all 13 features or the exact thresholding logic.
  - `EXP-016-C2` (high): The limitation about factorized categoricals is an unsupported preprocessing-history assertion unless backed by lineage or raw schema evidence; it should not be treated as established fact.
  - `EXP-016-C2` (high): Using random_split_psi as a production-risk indicator is a weak proxy; it does not test temporal drift, decision-time availability, or post-deployment feature generation stability.

### EXP-017 · heart_disease

- **Confidence:** medium
- **Decision:** Insufficient evidence to determine which columns violate the prediction-time contract or are suspicious target proxies. No feature should be approved or excluded solely from this package.
- **Interpretation:** The package only establishes that the detector can recognize an injected exact-target canary and that no leakage features were declared. It does not identify which real columns violate the prediction-time contract or behave as target proxies. Because the key validation artifact comparing safe versus unsafe training is null, the evidence is insufficient to support any decision about feature safety or exclusion beyond a request for further review.
- **Next experiment:** Column-level decision-time contract audit for heart_disease — For every column, provide a decision-time availability verdict and a rationale; flag any column with post-outcome timing, direct target encoding, or near-deterministic target association. The experiment succeeds only if the audit produces a non-null safe_vs_unsafe comparison and a reproducible list of candidate proxy columns with supporting evidence paths.
- **Challenges:**
  - `EXP-017-C1` (medium): This only shows the detector caught a synthetic canary; it does not demonstrate detection of any real leakage column in heart_disease.
  - `EXP-017-C2` (high): The statement 'Declared decision-time exclusions for heart_disease: none' is unsupported as a decision because the package provides no decision-time semantics review and no column-level evidence.

### EXP-018 · heart_disease

- **Confidence:** medium
- **Decision:** Do not treat the raw recipe as validated for production or as a proven improvement. At most, it is the current smallest candidate that matches the best reported training-CV ROC-AUC under the stated rule, pending independent validation and feature availability review.
- **Interpretation:** The evidence supports a narrow, non-causal inference: among the reported stages, the raw recipe is the smallest matrix and its mean training-CV ROC-AUC is tied for the best reported value, so it satisfies the stated selection rule better than the larger recipes. However, the package does not establish that this recipe improves validation evidence in a deployment-relevant sense, because only training-CV metrics are shown and no independent validation/test or decision-time availability checks are provided.
- **Next experiment:** Single held-out validation of raw vs logs under the same fold-wise pipeline — Run one pre-registered evaluation with a fixed train/validation split or nested CV: raw and logs only, same model and seed, fold-wise feature fitting, and report held-out ROC-AUC plus calibration. Success requires raw to be within 0.002 of logs on held-out ROC-AUC and to use fewer features, with no leakage detected in the pipeline audit.
- **Challenges:**
  - `EXP-018-C1` (high): The statement says 'improves validation evidence,' but the cited evidence only reports mean training-CV ROC-AUC and feature count. No independent validation metric is present.
  - `EXP-018-C1` (medium): The recommendation to use the raw recipe is based on a size-vs-training-CV tradeoff, but the raw and logs stages have the same mean ROC-AUC, so the evidence supports 'smallest tied-best' rather than 'better'.
  - `EXP-018-C1` (high): The package does not show decision-time or production-availability review for the raw features, so the recommendation is incomplete for deployment use.
  - `EXP-018-C1` (high): Because multiple engineered stages were compared, the best mean training-CV ROC-AUC may be subject to selection bias; the 0.002 rule is not enough to rule out overfitting without a held-out check.

### EXP-019 · heart_disease

- **Confidence:** medium
- **Decision:** Do not treat extra_trees as a confirmed winner. Use it only as a provisional CV leader pending a locked-holdout check and a repeated/nested validation comparison against logistic_regression.
- **Interpretation:** Within this training-CV-only package, extra_trees is the top raw-stage candidate by the stated ranking criterion, but the evidence is not strong enough to claim it is the strongest stable family overall. The margin over logistic_regression is small, the sample of folds is only 3, and the package does not include a locked holdout or repeated/nested validation. The most defensible conclusion is that extra_trees is the current CV leader under the specified rule, while logistic_regression remains a plausible near-tie on stability/runtime grounds.
- **Next experiment:** Repeated nested CV comparison: extra_trees vs logistic_regression — Run repeated nested CV with identical preprocessing inside folds for extra_trees and logistic_regression. Declare extra_trees the provisional winner only if its mean outer-fold ROC-AUC exceeds logistic_regression by at least 0.01 and the result is directionally consistent in a majority of repeats; otherwise, treat the two as tied and defer to simpler/faster model choice criteria.
- **Challenges:**
  - `EXP-019-C1` (medium): The statement that extra_trees is the training-CV model candidate is supported only as a provisional ranking outcome, not as a robust winner. The ROC-AUC advantage over logistic_regression is small, and logistic_regression has lower variance and much lower runtime under the same raw-stage setting.
  - `EXP-019-C1` (high): The evidence does not include a locked holdout, so the claim cannot be extended beyond training-CV selection.
  - `EXP-019-C1` (high): The package does not show nested CV or an untouched validation layer, so the ranking may be biased by model selection on the same folds used for evaluation.

### EXP-020 · heart_disease

- **Confidence:** medium
- **Decision:** Do not accept any claim that optimization produced a meaningful gain from this package alone. Treat the holdout score as a completed evaluation result, but require the missing baseline CV comparison and decision-time/production-availability review before any feature or recipe promotion.
- **Interpretation:** The package supports a single holdout performance result for the preselected recipe and provides a weakly informative feature-stability summary. It does not, by itself, establish that optimization produced a meaningful CV gain because the actual baseline-versus-tuned CV scores are not shown here, only the selection rule. The holdout result is promising but remains a one-shot estimate on a once-consumed holdout, so it should be treated as evidence of predictive performance, not causal impact or deployment readiness.
- **Next experiment:** Baseline-vs-tuned CV audit with untouched evaluation split — Report baseline mean CV ROC-AUC and tuned mean CV ROC-AUC on identical folds; accept optimization only if tuned - baseline >= 0.001. Then evaluate the selected recipe once on a fresh untouched split and require the holdout ROC-AUC to be reported without any further tuning on that split.
- **Challenges:**
  - `EXP-020-C1` (high): The holdout ROC-AUC is reported correctly, but the question asks whether optimization produced a meaningful CV gain; this package does not include the baseline CV score or tuned CV score needed to verify the selection rule threshold.
  - `EXP-020-C1` (medium): Because the holdout is once-consumed, the result should not be generalized as evidence of robust out-of-sample superiority without a fresh, untouched evaluation or nested validation.
  - `EXP-020-C2` (medium): The recommendation is directionally sound, but the evidence bundle does not show decision-time availability, contractual obtainability, or monitoring status for the listed features; therefore the recommendation is not fully evidenced here.
  - `EXP-020-C2` (low): Cross-fold stability is based on only 3 observed folds, which is limited support for a strong stability requirement.

### EXP-021 · credit_default

- **Confidence:** medium
- **Decision:** Do not proceed to modeling until production-availability and leakage checks are completed.
- **Interpretation:** This is a dataset-understanding package, not modeling evidence. It supports only a narrow description of the cached table and some basic profile statistics. It does not establish that the features are production-ready, temporally stable, or safe for modeling without additional review. The strongest unresolved issue is that the data appear to come from cached, factorized UCI matrices, which may not preserve raw category semantics needed for production encoders. The random split checks are insufficient to rule out leakage or distribution shift in a real deployment setting.
- **Next experiment:** Decision-time and raw-category availability audit — For all 23 features, document source system, event timestamp, decision-time availability, and raw-to-cached value mapping; success requires zero unresolved features and no feature marked as post-outcome or unavailable at scoring time.
- **Challenges:**
  - `EXP-021-C1` (medium): The row count, feature count, and positive rate are descriptive facts, but they do not by themselves establish that the dataset is suitable for modeling or that the target is correctly aligned with decision time.
  - `EXP-021-C2` (high): The statement that 0 features triggered review rules is not fully supported by the excerpt because only a subset of feature profiles is shown, and the review criteria are not fully auditable here.
  - `EXP-021-C2` (high): Random-split PSI cannot support a claim about production stability; it is explicitly acknowledged as a smoke check and does not address temporal drift or availability at scoring time.
  - `EXP-021-C2` (high): The note about factorized categoricals is a material production-risk warning: without raw category semantics and encoder design, the current cached representation may not be usable as-is.

### EXP-022 · credit_default

- **Confidence:** medium
- **Decision:** Insufficient evidence to identify or clear any specific columns as contract-safe. No feature should be approved from this package alone.
- **Interpretation:** This package provides only a minimal audit signal: the detector can recognize an injected exact-target canary, but there is no evidence here that any real column was tested, ranked, or validated against decision-time availability. The empty declared leakage list is not sufficient to conclude absence of leakage or absence of target proxies, especially because the comparative safe_vs_unsafe_training_cv result is missing.
- **Next experiment:** Column-level decision-time contract audit for candidate proxies — For every column in the dataset, provide a decision-time availability label and a minimal evidence trace; any column with post-outcome timing, target-derived naming, or unusually strong univariate association must be explicitly reviewed and either justified as available at prediction time or marked excluded.
- **Challenges:**
  - `EXP-022-C1` (medium): The canary passing is a narrow detector sanity check, not evidence about real feature leakage or proxy behavior. It cannot support any conclusion about the dataset columns under audit.
  - `EXP-022-C2` (high): The statement 'Declared decision-time exclusions for credit_default: none' is not justified as a decision. The package lacks decision-time semantics and production-availability review for actual columns, so 'none' is unsupported.

### EXP-023 · credit_default

- **Confidence:** medium
- **Decision:** Do not accept the recommendation as stated; the evidence is insufficient to justify a feature choice beyond a provisional screening preference for raw under this specific CV summary.
- **Interpretation:** The evidence does not support the recommendation to use the raw recipe as the next screen under the stated selection rule, because the raw stage is the best by mean ROC-AUC and also the smallest among the stages shown. However, the package only provides 3-fold training-CV evidence, so the apparent ranking is still vulnerable to fold variance and selection bias. The phrase 'without feature dilution' is not operationalized beyond feature counts, and there is no decision-time or production-availability review for the generated features.
- **Next experiment:** Nested CV comparison of raw vs selected recipe with fixed fold splits — Using identical outer folds and a pre-registered selection rule, the raw recipe must achieve outer-fold mean ROC-AUC no worse than the selected recipe by more than 0.002, while using fewer or equal features; otherwise the recommendation is not supported.
- **Challenges:**
  - `EXP-023-C1` (high): The recommendation says to use the raw feature recipe, but the stated selection rule would require the smallest matrix within 0.002 of the best mean ROC-AUC; the raw stage is best by mean ROC-AUC and also smallest among the listed stages, so the recommendation is not justified as a 'next model screen' improvement over alternatives, only as the least-diluted top performer in this summary.
  - `EXP-023-C1` (high): The claim implies improved validation evidence, but the selected stage has lower mean ROC-AUC than raw and more features; the evidence does not show that feature engineering improved validation without dilution.
  - `EXP-023-C1` (high): The claim relies on training-CV ROC-AUC only; without nested validation or a held-out test, the stage comparison may be biased by feature selection.
  - `EXP-023-C1` (medium): The limitation about generic mathematical feature generation is acknowledged, but there is no documented decision-time or production-availability review, so the recommendation is incomplete for deployment-oriented use.

### EXP-024 · credit_default

- **Confidence:** medium
- **Decision:** Do not treat extra_trees as the final algorithm choice yet; treat it as the current CV leader only, pending repeat/outer validation and leakage checks.
- **Interpretation:** The evidence supports a narrow training-CV ranking claim: under the stated selection rule, extra_trees is the top candidate on this raw-feature CV run. However, the package does not establish that extra_trees is the strongest stable family in a broader sense, because the comparison is limited to one 3-fold run, one stage, and one metric-weighted rule. Stability is only partially evidenced by modest fold-to-fold ROC-AUC spread, and the gap to other tree-based families is not obviously large enough to rule out ranking instability under different splits or repeats.
- **Next experiment:** Repeat the raw-stage CV comparison with repeated stratified folds — Run at least 10 repeated stratified CV resamples on the same raw-stage model set; success requires extra_trees to rank first in at least 70% of repeats and to retain a positive margin over the next-best family under the same mean ROC-AUC minus 0.25×std rule.
- **Challenges:**
  - `EXP-024-C1` (high): The statement that extra_trees is the training-CV model candidate is supported only within the stated selection rule and raw-stage CV run; it is not evidence that extra_trees is the best algorithm family overall or on unseen data.
  - `EXP-024-C1` (high): The claim implies stability, but only one 3-fold run is shown. That is insufficient to establish ranking stability across resamples.
  - `EXP-024-C1` (high): The evidence package does not show fold construction details, so leakage or dependence between folds cannot be ruled out.
  - `EXP-024-C1` (medium): Runtime is used as a tie-breaker in the selection rule, but the package does not justify that runtime differences are relevant to the modeling objective or deployment constraints.

### EXP-025 · credit_default

- **Confidence:** medium
- **Decision:** Do not accept the claim that optimization produced a meaningful CV gain on the basis of this package alone. Treat the holdout score as a completed descriptive evaluation, not as evidence of superiority or deployability. Require the missing baseline-vs-tuned CV comparison and a decision-time/production-availability review before any feature or recipe promotion.
- **Interpretation:** The package supports a descriptive statement that the selected recipe achieved a moderate holdout ROC-AUC and that some features appear repeatedly important across the reported CV folds. It does not, by itself, establish that optimization produced a meaningful CV gain because the actual baseline and tuned CV scores are not shown here, only the selection threshold. The holdout result is informative but remains a single once-consumed estimate; the bootstrap interval suggests uncertainty, and no decision-time or production-availability review is evidenced for any feature or recipe.
- **Next experiment:** Baseline-vs-tuned CV audit with decision-time feature check — Report baseline and tuned mean CV ROC-AUC on the same folds; accept the optimization claim only if tuned - baseline >= 0.001. Separately, for each promoted feature, provide decision-time availability and production-obtainability evidence; otherwise exclude it.
- **Challenges:**
  - `EXP-025-C1` (high): The statement reports a holdout ROC-AUC, but the question also asks whether optimization produced a meaningful CV gain. The package does not include the tuned and baseline CV scores needed to verify the stated selection rule or any improvement magnitude.
  - `EXP-025-C1` (medium): Because the holdout is marked consumed, any further use of this holdout for recipe selection would be a validation risk; the package does not show safeguards preventing that.
  - `EXP-025-C2` (medium): The recommendation is directionally sound, but it is not supported by explicit evidence of decision-time availability, contractual obtainability, or monitoring readiness for the listed features.
  - `EXP-025-C2` (medium): Cross-fold stability is based on only 3 folds, which is limited evidence for a strong stability claim.

### EXP-026 · german_credit

- **Confidence:** medium
- **Decision:** Do not approve modeling yet. The dataset can be described, but the evidence is insufficient to clear production-availability and leakage-risk checks.
- **Interpretation:** Observed evidence supports a narrow data-understanding summary: the dataset is small, tabular, and appears to have no missing values in the sampled feature profiles. However, the package does not establish that the cached features preserve raw category semantics, nor does it establish production-time availability, decision-time availability, or temporal stability. The claim that no features triggered review rules is only partially supported because the evidence shown is a sample of 10 feature profiles, not the full 20-feature set, and one feature lacks a PSI value. The strongest unresolved issue is potential preprocessing leakage from factorized categorical encodings and the absence of a production-availability review for each feature.
- **Next experiment:** Raw-schema and availability audit for all 20 features — For all 20 features, provide raw source column names, decision-time availability status, reversible categorical mapping status, missingness, identifier/near-constant flags, and a non-null shift statistic; the experiment passes only if no feature is missing any of these fields and no feature is flagged as unavailable or unrecoverable.
- **Challenges:**
  - `EXP-026-C1` (medium): The fact claim is plausible from the cited fields, but the package does not show the underlying label definition or whether the 0.300 positive rate is computed on a leakage-free training split. Without split construction details, the rate is descriptive only.
  - `EXP-026-C2` (high): The statement that 0 features triggered review rules is not fully supported by the evidence excerpt because only 10 feature profiles are visible, not all 20 cached features. Also, null PSI for one feature means the shift review is incomplete for the shown sample.
  - `EXP-026-C2` (high): The limitation about factorized categoricals is a valid risk note, but the evidence bundle does not show the raw source schema or a reversible mapping, so the production-encoder requirement remains unverified.

### EXP-027 · german_credit

- **Confidence:** medium
- **Decision:** Insufficient evidence to identify or approve any specific column as safe or unsafe. The current package is a leakage-audit stub, not a completed column-level contract review.
- **Interpretation:** The package supports only a narrow factual conclusion: the canary test succeeded and no leakage features were declared. It does not establish which columns violate the prediction-time contract, because there is no column-level evidence, no decision-time semantics review, and no safe-vs-unsafe validation result. The claim that there are no declared decision-time exclusions is not evidence that no leakage or proxy columns exist; it only indicates none were declared in the package.
- **Next experiment:** Column-level decision-time contract audit for german_credit — Produce a column-by-column table with: feature name, decision-time availability verdict, production-availability verdict, and heuristic flags (e.g., target proxy, post-outcome, identifier-like). The experiment succeeds only if every feature has an explicit verdict and any flagged feature is supported by a documented decision-time rationale; if no features are flagged, the audit must still show complete coverage with no missing verdicts.
- **Challenges:**
  - `EXP-027-C1` (medium): The canary result is a detector sanity check, not evidence about real dataset columns. It cannot support any claim about actual leakage or proxy behavior in german_credit.
  - `EXP-027-C2` (high): The statement 'Declared decision-time exclusions for german_credit: none' is unsupported as a decision about leakage. It only reflects that no exclusions were declared, while the package explicitly notes that only decision-time semantics can confirm leakage.

### EXP-028 · german_credit

- **Confidence:** medium
- **Decision:** Do not accept the recommendation as written; the evidence is insufficient and internally inconsistent with the selection rule.
- **Interpretation:** The evidence supports only a narrow descriptive conclusion: among the tested recipes, the interactions stage has the highest reported mean training-CV ROC-AUC. However, the package does not support the recommendation to use it as the next screen under the stated selection rule, because the selected stage is not within the 0.002 ROC-AUC tolerance of the best stage. The evidence also does not establish that the interactions recipe improves validation evidence without feature dilution in a decision-ready sense, because the reported metrics are training-CV only, the fold count is small, and the selected stage is actually lower-scoring than the best stage while being smaller. The package therefore contains an internal inconsistency between the selection rule and the stated recommendation.
- **Next experiment:** Fold-local ablation of interactions vs selected under nested validation — Run nested CV with feature engineering fit only on each training fold. Declare success only if interactions exceeds the selected recipe by at least 0.002 mean outer-CV ROC-AUC and does not increase feature_count_mean by more than a predeclared dilution budget.
- **Challenges:**
  - `EXP-028-C1` (high): The recommendation says to use the interactions recipe, but the stated selection rule would instead choose the smallest matrix within 0.002 of the best stage; the selected stage is not within that tolerance of the best reported mean ROC-AUC.
  - `EXP-028-C1` (high): The claim implies validation improvement, but the evidence only reports training-CV metrics and does not include an independent holdout or outer CV estimate.
  - `EXP-028-C1` (medium): The claim suggests the interactions recipe is preferable, but the evidence does not quantify the tradeoff against feature dilution using a defined metric beyond raw feature count.
  - `EXP-028-C1` (high): The package does not show fold-wise generation details for the derived features, so leakage from global fitting of feature engineering cannot be ruled out.

### EXP-029 · german_credit

- **Confidence:** medium
- **Decision:** Do not treat extra_trees as the final winner from this package alone. The evidence is sufficient only to nominate a provisional training-CV candidate, not to approve a family choice or deployment decision.
- **Interpretation:** The evidence supports only a narrow training-CV conclusion: extra_trees is the strongest mean ROC-AUC performer among the reported models, while xgboost may be the strongest under the stated mean-minus-0.25×std selection rule. However, the package does not provide enough detail to verify that the selection rule was applied consistently, nor does it establish that any model is the best stable choice beyond this CV setup. The stability signal is weakly informative because it is based on only 3 folds and mixed feature-stability reporting, so the safest interpretation is that the ranking is provisional and should be rechecked with a locked evaluation and a more explicit stability protocol.
- **Next experiment:** Rule-verification and leakage-check rerun on the top two families — Run repeated stratified CV with the exact same pipeline for extra_trees and xgboost, report per-fold mean ROC-AUC, fold std, and the explicit selection score for each run; the success condition is that the same family wins in at least 2 independent repeats and the ranking score margin is positive in every repeat.
- **Challenges:**
  - `EXP-029-C1` (high): The statement says extra_trees is the training-CV model candidate, but the stated selection rule is mean ROC-AUC minus 0.25×fold std, and the package does not report the computed score for each model. Based on the provided mean/std fields, xgboost appears potentially competitive or better under that rule, so the candidate claim is not fully supported as written.
  - `EXP-029-C1` (high): The claim presents extra_trees as the candidate using interactions features, but the evidence bundle does not document whether the interaction-generation pipeline was confined to each training fold. This leaves leakage risk unresolved.
  - `EXP-029-C1` (medium): The claim implies stability from 3-fold CV, but three folds are too few to support a strong stability conclusion without repeated CV or split-sensitivity analysis.

### EXP-030 · german_credit

- **Confidence:** medium
- **Decision:** Do not claim a meaningful optimization gain from this package alone. Treat the holdout score as a finalized retrospective estimate only, and require an explicit baseline-vs-tuned CV comparison plus a fresh untouched evaluation before any stronger conclusion.
- **Interpretation:** The package supports a single held-out performance estimate for the preselected recipe, but it does not establish that optimization produced a meaningful CV gain because the actual baseline-versus-tuned CV comparison is not shown. The holdout result is informative but limited because the holdout has already been consumed, so it should not be used for further model selection. The feature evidence is consistent with a cautious feature-governance stance, but it does not by itself validate any feature for production use without decision-time and availability checks.
- **Next experiment:** Baseline-vs-tuned CV comparison on a fresh split with locked recipe — Pre-register baseline and tuned candidates, evaluate both on the same fresh CV split, and accept optimization only if tuned mean ROC-AUC - baseline mean ROC-AUC >= 0.001; otherwise keep baseline. The holdout must remain untouched for this experiment.
- **Challenges:**
  - `EXP-030-C1` (high): The holdout performance is stated correctly, but the question also asks whether optimization produced a meaningful CV gain; the package does not include the baseline CV score or the tuned CV score needed to verify the improvement threshold.
  - `EXP-030-C2` (medium): The recommendation is directionally sound, but it is not supported by explicit evidence of decision-time availability or contractual obtainability for the named features; it should be treated as a governance rule, not a validated conclusion about these features.

### EXP-031 · mushroom

- **Confidence:** medium
- **Decision:** Do not proceed to modeling yet; resolve data representation, leakage, and production-availability questions first.
- **Interpretation:** This evidence supports a basic dataset inventory and suggests the data are mostly low-cardinality categorical fields with no missingness in the sampled profiles. However, the package does not establish production readiness: the features are described as cached/factorized, so raw category semantics and the intended production encoder path remain unresolved. The random-split PSI values are only a within-sample smoke test and do not address temporal, source, or decision-time stability. The evidence is insufficient to conclude that the dataset is safe for modeling without leakage, schema, and availability checks.
- **Next experiment:** Raw-category and decision-time availability audit for all 22 features — For every feature, provide raw value semantics, missingness, cardinality, and a decision-time availability flag; fail if any feature lacks a production-available source, is derived from post-decision information, or cannot be mapped from the cached encoding back to a stable production encoder.
- **Challenges:**
  - `EXP-031-C1` (medium): The claim is numerically supported by the evidence bundle, but it implicitly treats cached features as the final modeling representation. That is unsupported because the package does not show raw categorical values or the production encoder path.
  - `EXP-031-C2` (high): The statement that 1 feature triggered review rules is not verifiable from the provided sample because only a subset of feature profiles is shown and the triggering rule thresholds are not included.
  - `EXP-031-C2` (medium): The limitation about random-split PSI correctly warns against overinterpretation, but the evidence does not show any temporal or production-simulation test to support stability claims.

### EXP-032 · mushroom

- **Confidence:** medium
- **Decision:** Insufficient evidence to identify or approve any real columns as safe or unsafe. The current package supports only that the detector is not completely broken on a synthetic canary, not that the mushroom features satisfy prediction-time availability or are free of target proxies.
- **Interpretation:** The package provides only a minimal sanity check that the detector can flag an injected exact-target canary. It does not provide column-level evidence for which real columns violate the prediction-time contract or act as target proxies. The absence of declared leakage features is not evidence of absence, and the null safe_vs_unsafe_training_cv field leaves the main leakage-risk question unresolved.
- **Next experiment:** Column-level prediction-time contract audit with safe/unsafe ablation — For each column, provide a decision-time availability verdict and a reason; then run a pre-registered safe-vs-unsafe ablation. Success is achieved only if at least one column is flagged by decision-time review and the ablation shows a reproducible validation difference between the full and contract-compliant feature sets under the same split protocol.
- **Challenges:**
  - `EXP-032-C1` (low): The canary pass is a detector sanity check, not evidence about any real dataset column. It cannot support conclusions about actual leakage or proxy columns.
  - `EXP-032-C2` (high): The statement 'Declared decision-time exclusions for mushroom: none' is unsupported as a decision about feature safety; the evidence only shows an empty declared_leakage_features list and a null safe_vs_unsafe_training_cv field. That does not establish prediction-time compliance.

### EXP-033 · mushroom

- **Confidence:** medium
- **Decision:** Do not accept the recommendation to use the raw recipe as stated. The smallest defensible conclusion is that the selected stage is the smallest recipe within the stated ROC-AUC tolerance of the best stage, but the package is insufficient to claim a production-ready improvement or to prefer raw over selected without an external validation check.
- **Interpretation:** The evidence supports a narrow ranking claim: under the stated selection rule, the selected feature matrix is the smallest recipe within 0.002 mean training-CV ROC-AUC of the best observed stage. However, the package does not justify the stronger recommendation to use the raw recipe, because the selected stage has the best reported mean ROC-AUC and fewer features than the full_fe stage. The evidence also does not establish that any recipe improves validation evidence in a production-relevant sense, because only training-CV is shown and no decision-time or production-availability review is provided.
- **Next experiment:** Nested holdout comparison of raw vs selected feature recipe — On an outer holdout or outer CV split, the selected recipe must achieve ROC-AUC no worse than raw by at most 0.002, with all feature engineering fitted only on the inner training folds and with no leakage detected in a fold-by-fold audit.
- **Challenges:**
  - `EXP-033-C1` (high): The recommendation says to use the raw recipe, but the selected stage has the highest reported mean ROC-AUC and a smaller feature count than full_fe; the evidence does not support preferring raw over selected.
  - `EXP-033-C1` (high): The phrase "improves validation evidence" is unsupported because only training-CV metrics are reported and no external validation or nested selection result is provided.
  - `EXP-033-C1` (high): The evidence bundle does not demonstrate that the feature recipe was fitted without leakage inside each fold; the train-fitted requirement is asserted by the question but not verified by the evidence paths shown.
  - `EXP-033-C1` (medium): The limitation about generic mathematical feature generation lacking domain meaning is acknowledged, but no decision-time or production-availability review is provided, so the recommendation is premature.

### EXP-034 · mushroom

- **Confidence:** medium
- **Decision:** Do not accept the claim that extra_trees is the strongest stable training-CV candidate from this package alone. The package is internally inconsistent with its own selection rule and lacks locked-holdout evidence, repeated CV, and leakage checks.
- **Interpretation:** The evidence supports that several algorithm families are near-ceiling on training CV for this dataset profile, but it does not uniquely establish extra_trees as the strongest stable family under the stated ranking rule. Based on the reported mean ROC-AUC and fold variability alone, xgboost appears numerically strongest on the provided training-CV summary, while extra_trees is perfectly tied on mean ROC-AUC but not on the stability criterion as written because its reported std is 0.0 despite one fold being below 1.0. This inconsistency makes the selection claim under-specified and potentially miscomputed. The evidence is also insufficient to answer any holdout/generalization question.
- **Next experiment:** Repeat stratified nested CV with locked holdout comparison for top two families — Run repeated stratified CV with at least 10 repeats for xgboost and extra_trees using identical preprocessing and a locked holdout; accept the current candidate only if it remains top-ranked by the predeclared rule on the repeated-CV summary and its holdout ROC-AUC is not worse than the runner-up by more than a prespecified margin, with all fold-std calculations matching the raw fold metrics.
- **Challenges:**
  - `EXP-034-C1` (high): The claim that extra_trees is the training-CV candidate is not supported by the stated ranking rule because xgboost has the highest reported mean ROC-AUC, and extra_trees' reported std is inconsistent with its fold metrics.
  - `EXP-034-C1` (high): The claim implies a winner for the dataset profile, but only training CV is shown and no locked holdout is available; this is not enough to compare algorithm families for generalization.
  - `EXP-034-C1` (medium): The phrase 'strongest stable' is unsupported because stability is not consistently reported across models and the extra_trees std appears miscomputed or misreported.

### EXP-035 · mushroom

- **Confidence:** high
- **Decision:** Do not conclude that optimization produced a meaningful CV gain from this package. Treat the holdout result as a completed evaluation of the preselected recipe only, and require a direct baseline-vs-tuned CV comparison plus decision-time/production-availability review before any feature or recipe promotion.
- **Interpretation:** The evidence supports that the chosen recipe performs perfectly on the consumed holdout, but this does not by itself establish that optimization produced a meaningful generalization gain. The selection rule is conservative, yet the package does not show the baseline CV score, the tuned CV score, or their difference, so the central optimization question remains unresolved. The feature evidence is directionally consistent with stability, but it is not sufficient to approve feature promotion without decision-time and production-availability checks.
- **Next experiment:** Baseline-vs-tuned nested CV comparison with leakage check — Run nested CV on the training data only, report baseline mean ROC-AUC, tuned mean ROC-AUC, and their difference with identical folds and preprocessing. Success requires tuned minus baseline >= 0.001 and no evidence of target leakage or fold contamination.
- **Challenges:**
  - `EXP-035-C1` (medium): The claim about holdout performance is supported by the reported metrics, but it should not be generalized beyond this consumed holdout. The package does not provide evidence that the result is robust to a fresh unseen test set or that no leakage occurred.
  - `EXP-035-C1` (high): The question asks whether optimization produces a meaningful CV gain, but no baseline CV score, tuned CV score, or delta is shown. The evidence is insufficient to support any claim about optimization gain.
  - `EXP-035-C2` (medium): The recommendation is directionally sound, but the package does not actually provide decision-time availability or production availability evidence for the listed features, so the recommendation cannot be operationally validated here.

### EXP-036 · spambase

- **Confidence:** medium
- **Decision:** Do not proceed to modeling until decision-time availability, raw-source reconstruction, and leakage checks are completed.
- **Interpretation:** This is a dataset-understanding package, not a modeling result. The evidence supports a basic description of the cached table shape and class balance, and it suggests no obvious missingness in the sampled feature profiles. However, the package does not establish production readiness: the data source, label timing, feature availability at decision time, and whether the cached representation preserves raw semantics are unresolved. The random-split PSI values are not sufficient to argue stability in any real deployment setting.
- **Next experiment:** Decision-time feature availability and raw-semantic reconstruction audit — For all 57 features, document raw source column, transformation, availability timestamp relative to prediction time, and production encoder path; fail if any feature is unavailable at decision time, cannot be reconstructed from raw inputs, or has ambiguous semantics.
- **Challenges:**
  - `EXP-036-C1` (medium): The row count, feature count, and positive rate are factual for the cached dataset, but the claim does not specify whether these are from the full raw source, a filtered subset, or a post-processing cache. That distinction matters for reproducibility and production mapping.
  - `EXP-036-C2` (high): The statement that 0 features triggered review rules is not fully supported by the excerpt because only 10 feature profiles are visible here, not all 57. Also, random-split PSI is not evidence against temporal or operational shift.
  - `EXP-036-C2` (high): The limitation about factorized categoricals indicates a representation risk, but the package does not show which columns were factorized, how they map back to raw semantics, or whether production encoders can reproduce them.

### EXP-037 · spambase

- **Confidence:** medium
- **Decision:** Insufficient evidence to conclude which columns violate the prediction-time contract or are suspicious target proxies.
- **Interpretation:** The package provides only weak evidence about prediction-time contract violations. A passing synthetic canary supports that the detector is wired correctly, but it does not identify any real columns as leakage or target proxies. The absence of declared leakage features is not evidence that no columns violate the contract; it only shows none were declared. Because the key validation field is null, the package does not substantiate the decision claim that there are no decision-time exclusions for spambase.
- **Next experiment:** Column-level prediction-time availability audit for spambase — For every column, provide a decision-time availability label, source timestamp/provenance, and a leakage/proxy flag; success is achieved only if all columns are either explicitly justified as available at prediction time or flagged with a concrete reason for exclusion.
- **Challenges:**
  - `EXP-037-C1` (medium): A synthetic canary passing only demonstrates detector plumbing, not that the audit found or ruled out real leakage in spambase.
  - `EXP-037-C2` (high): The statement that there are no declared decision-time exclusions is unsupported as a substantive leakage conclusion; the evidence bundle contains no column-level semantics and the validation field is null.

### EXP-038 · spambase

- **Confidence:** high
- **Decision:** Do not accept the recommendation as written; the evidence is insufficient and internally inconsistent.
- **Interpretation:** The evidence does not support the recommendation to use the raw recipe as the next screen if the stated objective is to improve validation evidence without feature dilution. By the package's own selection rule, the raw stage is not the smallest matrix within 0.002 of the best mean ROC-AUC; the selected stage is much smaller but also much worse on mean ROC-AUC. The package therefore contains an internal inconsistency between the stated selection criterion and the recommendation. The strongest validation evidence in the bundle appears to come from the full_fe and interactions stages, but the package does not establish whether those gains are robust, leakage-free, or available at decision time.
- **Next experiment:** Nested fold-safe comparison of raw vs selected recipe — Run nested CV with identical model settings for raw, full_fe, interactions, and selected recipes; pre-register the rule 'choose the smallest matrix within 0.002 mean inner-CV ROC-AUC of the best inner-CV stage'; success is achieved only if the chosen recipe satisfies that rule and its outer-fold mean ROC-AUC is not worse than the raw recipe by more than 0.002.
- **Challenges:**
  - `EXP-038-C1` (high): The statement 'Use the raw feature recipe for the next model screen' is not supported by the selection rule, because the raw stage is not the smallest matrix and the selected stage is not within 0.002 mean ROC-AUC of the best stage.
  - `EXP-038-C1` (high): The claim implies improvement in validation evidence, but the raw stage's mean ROC-AUC is lower than the full_fe and interactions stages reported in the same bundle.
  - `EXP-038-C1` (high): The claim does not establish leakage-free feature fitting inside folds for the engineered stages, so the validation evidence may be inflated.
  - `EXP-038-C1` (medium): The claim ignores decision-time and production-availability review for engineered features, which is required before approving any feature recipe.

### EXP-039 · spambase

- **Confidence:** medium
- **Decision:** Provisional training-CV candidate only; do not treat as final model choice without a locked holdout and repeated validation review.
- **Interpretation:** Within this training-CV-only package, hist_gradient_boosting is the top mean ROC-AUC model and is also very stable across the 3 folds. However, the evidence is not strong enough to conclude it is the strongest algorithm family for the dataset in a deployment sense, because the ranking is based on a single 3-fold CV run, the lead over xgboost is small, and no locked holdout or repeated/nested validation is shown. The safest inference is that hist_gradient_boosting is the current training-CV leader under the stated selection rule, not a proven overall winner.
- **Next experiment:** Repeated stratified CV comparison of hist_gradient_boosting vs xgboost with a locked holdout — Run at least 10 repeated stratified CV splits with identical preprocessing inside folds, then evaluate the top two families on one untouched locked holdout. Success requires hist_gradient_boosting to retain the best mean ROC-AUC across repeats and to remain within a small predeclared margin of xgboost on the holdout; if the ranking flips or the gap is not robust, the current claim is not supported.
- **Challenges:**
  - `EXP-039-C1` (medium): The statement says hist_gradient_boosting is the training-CV candidate, but the evidence bundle's model entry for the top ROC-AUC result is labeled model=hist_gradient_boosting while the package also lacks a holdout check; this supports only a provisional CV leader, not a final algorithm-family winner.
  - `EXP-039-C1` (low): The claim highlights 0.9874±0.0021 ROC-AUC, but the evidence does not show the exact computation method for the uncertainty interval beyond fold std/CI fields, so the summary should not be treated as independently validated outside the package.
  - `EXP-039-C1` (high): The claim implies stability from 3-fold CV, but three folds with one repeat is thin evidence for stability and may miss sensitivity to split choice.
  - `EXP-039-C1` (medium): The selection rule ranks by mean ROC-AUC minus 0.25×fold std and then runtime, which can favor small differences in mean score; the package does not show whether the ranking is robust to alternative reasonable selection rules.

### EXP-040 · spambase

- **Confidence:** medium
- **Decision:** Hold the optimization claim as unverified; accept only the narrow factual statement that the preselected recipe achieved the reported consumed-holdout ROC-AUC. Do not promote any feature based on this package. Require a new falsifiable validation step before any stronger claim about optimization gain or feature readiness.
- **Interpretation:** The evidence supports that the selected recipe performs strongly on the consumed holdout, but it does not by itself establish that optimization produced a meaningful CV gain. The package also does not provide the CV numbers needed to verify the 0.001 improvement rule was actually met. Because the holdout is already consumed, any further model-selection claims should be treated cautiously unless they are based on a fresh, untouched evaluation set or a strictly nested validation design. The feature-related recommendation is directionally reasonable, but the provided evidence does not show any actual feature candidates or stability analysis to substantiate a specific promotion decision.
- **Next experiment:** Nested CV check of optimization gain with untouched final evaluation — Report baseline and tuned mean outer-fold ROC-AUC with fold-level intervals; declare optimization gain only if tuned minus baseline >= 0.001 on the outer folds. If a fresh untouched evaluation set exists, evaluate the pre-registered winner once and do not use it for model selection.
- **Challenges:**
  - `EXP-040-C1` (high): The holdout performance is reported, but the question also asks whether optimization produced a meaningful CV gain; no CV baseline/tuned metrics are included, so the optimization part of the claim is unsupported.
  - `EXP-040-C2` (medium): This is a recommendation rather than an evidence-backed result, and the package provides no actual feature list or stability evidence to validate or operationalize the recommendation in this experiment.

### EXP-041 · online_shoppers

- **Confidence:** medium
- **Decision:** Do not proceed to modeling until decision-time feature availability and encoding semantics are verified, and until a time-aware validation check replaces the random split smoke test.
- **Interpretation:** This evidence package supports only a narrow data-understanding summary: the dataset is small-to-moderate in row count, has 17 cached features, and the shown feature subset has no missingness or obvious random-split shift flags. It does not yet establish production readiness, because the package itself flags unresolved categorical-semantic restoration and explicitly warns that random-split PSI is insufficient for temporal stability. The evidence shown is also incomplete relative to the stated question: only 10 feature profiles are visible, so the claim about all 17 features having no review-rule triggers is not fully substantiated by the excerpt provided.
- **Next experiment:** Time-aware schema and availability audit for all 17 features — For every one of the 17 features: confirm decision-time availability, confirm raw semantic restoration or a production-compatible encoder, and show no leakage/shift flag under a forward-chaining or time-based split; if any feature fails any check, modeling is blocked until resolved.
- **Challenges:**
  - `EXP-041-C1` (medium): The row count, feature count, and positive rate are factual if the cited fields are trusted, but the package does not show provenance for how the cached features were derived or whether the 17 features are production-available at decision time.
  - `EXP-041-C2` (high): The statement that 0 features triggered review rules is not fully supported by the excerpt because only 10 feature profiles are visible here, not all 17 features referenced elsewhere in the package.
  - `EXP-041-C2` (high): The absence of random-split PSI flags does not rule out temporal drift, seasonality, or leakage; the package itself acknowledges this limitation.
  - `EXP-041-C2` (high): The note about factorized categoricals indicates a production-encoding risk, but the package does not show that raw category semantics can be restored or that the production encoder will match training semantics.

### EXP-042 · online_shoppers

- **Confidence:** medium
- **Decision:** Do not approve PageValues for use until decision-time semantics and production-availability are verified; treat it as a high-priority leakage/proxy review candidate, not a confirmed violation.
- **Interpretation:** The evidence supports that PageValues is a strong suspicious proxy candidate because it was explicitly declared a decision-time exclusion and it materially changes cross-validated training performance when included. However, the package does not establish that PageValues violates the prediction-time contract in production; that requires decision-time semantics and availability review, which are only partially referenced in the limitations text. The canary result only validates the detector on a synthetic exact-target case, not on real leakage patterns.
- **Next experiment:** Decision-time availability check for PageValues — For PageValues, document a concrete decision-time data lineage showing whether the value exists before the prediction timestamp for every row in a held-out audit sample; if any row lacks a valid pre-decision value, classify it as a contract violation.
- **Challenges:**
  - `EXP-042-C1` (medium): The canary passing only shows the detector can catch a synthetic exact-target canary; it does not demonstrate that the detector would catch realistic leakage or that the real dataset is free of leakage.
  - `EXP-042-C2` (high): Declaring PageValues as a decision-time exclusion is not the same as proving a prediction-time contract violation. The evidence lacks explicit decision-time semantics and production-availability review.
  - `EXP-042-C2` (high): The +0.1968 apparent AUC lift is a predictive comparison, not proof of leakage or proxying. It may reflect legitimate signal, leakage, or confounding; the evidence does not distinguish these.

### EXP-043 · online_shoppers

- **Confidence:** medium
- **Decision:** Do not accept the recommendation as stated. The smallest defensible next candidate from this package is the raw recipe, but only as a hypothesis for further validation, not as a production-approved feature set.
- **Interpretation:** The evidence supports a narrow factual conclusion: among the reported stages, raw is the smallest feature matrix that is within the stated 0.002 mean training-CV ROC-AUC tolerance of the best reported stage. However, the package does not support the recommendation to use the selected stage as written, because the selected stage is not within the stated tolerance and the rule appears inconsistently applied. The evidence is also limited to training-CV summaries, so it cannot establish generalization improvement or production value.
- **Next experiment:** Single holdout validation of raw vs full_fe under the stated tolerance rule — On one pre-specified untouched holdout split, raw must achieve ROC-AUC no more than 0.002 below full_fe, and the split must be fixed before any feature selection or model comparison.
- **Challenges:**
  - `EXP-043-C1` (high): The recommendation says to use the raw recipe for the next model screen, but the stated selection rule would not select the reported 'selected' stage, and the raw stage is only within tolerance relative to the best mean ROC-AUC, not proven superior. The evidence supports at most a candidate for further screening, not a recommendation.
  - `EXP-043-C1` (high): The phrase 'improves validation evidence' is not supported because all reported metrics are training-CV summaries; no independent validation evidence is shown.
  - `EXP-043-C1` (medium): The phrase 'without feature dilution' is not validated as a decision criterion; only feature counts are reported, and no analysis shows that smaller matrices preserve performance under a pre-registered dilution threshold.

### EXP-044 · online_shoppers

- **Confidence:** medium
- **Decision:** Do not promote a universal algorithm-family winner from this package. At most, treat extra_trees as the provisional training-CV candidate pending a locked holdout and a leakage/robustness check.
- **Interpretation:** The evidence supports only a narrow training-CV preference for extra_trees under the package’s own ranking rule, but not a general claim that it is the strongest algorithm family for this dataset profile. The reported ROC-AUC differences are small, the CV design is only 3 folds with no repeats, and the package does not include a locked holdout or any decision-time/production-availability review. The claim should therefore be treated as a provisional model-selection hypothesis, not a validated winner.
- **Next experiment:** Locked holdout verification for the current CV leader — Pre-register the split, train only on the existing training partition, and evaluate extra_trees plus the nearest competitor on the locked holdout; success requires extra_trees to remain top-ranked on ROC-AUC and not be materially worse on calibration metrics (log_loss and ece_10) than the competitor.
- **Challenges:**
  - `EXP-044-C1` (high): The statement elevates extra_trees as the candidate winner, but the evidence bundle only supports a training-CV ranking under a custom heuristic; it does not justify a universal algorithm-family conclusion or any production-facing recommendation.
  - `EXP-044-C1` (high): The package lacks a locked holdout and any decision-time/production-availability review, so the recommendation is premature.
  - `EXP-044-C1` (medium): With only 3 folds and 1 repeat, the stability claim is underpowered; fold variance alone is not enough to establish robust superiority.
  - `EXP-044-C1` (high): The evidence bundle’s summary claim and the detailed fold metrics should be recomputed/verified for consistency before ranking models.

### EXP-045 · online_shoppers

- **Confidence:** medium
- **Decision:** Do not accept the claim that optimization produced a meaningful CV gain from this package alone. Treat the holdout score as a completed single-split estimate, not as evidence of deployment readiness. Require an explicit baseline-vs-tuned CV comparison and a decision-time/production-availability review before any feature promotion or model selection conclusion.
- **Interpretation:** The holdout result shows a moderate ranking signal, but the package does not establish that optimization produced a meaningful CV gain because the actual baseline and tuned CV scores are not provided here, only the selection threshold. The holdout performance is a single once-consumed estimate with a bootstrap interval, so it can support uncertainty-aware reporting but not a claim of generalization improvement beyond the evaluated split. The feature evidence supports a cautious feature-candidate screen, but not feature promotion on importance alone.
- **Next experiment:** Baseline-vs-tuned CV audit with fold-wise deltas — Report baseline and tuned mean CV ROC-AUC, fold-wise ROC-AUC for both, and the mean delta; success requires tuned minus baseline mean CV ROC-AUC >= 0.001 and no single fold accounting for the entire gain.
- **Challenges:**
  - `EXP-045-C1` (high): The claim reports a holdout ROC-AUC, but the question also asks whether optimization produced a meaningful CV gain. The package does not include the baseline CV score, tuned CV score, or fold-level comparison needed to verify that optimization improved CV by the stated rule.
  - `EXP-045-C1` (medium): A once-consumed holdout score can summarize one evaluation, but it does not justify broader claims about model superiority or generalization improvement without the missing CV comparison and protocol details.
  - `EXP-045-C2` (medium): The recommendation is directionally sound, but the evidence bundle does not show decision-time availability, contractual obtainability, or monitoring status for the named features, so the recommendation is not yet evidenced for specific features.
  - `EXP-045-C2` (medium): Cross-fold stability is based on only 3 folds, which is limited evidence for feature robustness and may not generalize to a different split or time period.

### EXP-046 · wine_quality

- **Confidence:** medium
- **Decision:** Do not approve modeling yet. The dataset description is partially supported, but production-availability and leakage review are incomplete.
- **Interpretation:** This evidence supports a narrow dataset summary: a 6497-row, 11-feature table with no observed missingness in the supplied profiles and a moderately imbalanced training target. However, the package does not establish production readiness or rule out leakage/availability issues. In particular, the evidence only shows random-split stability smoke checks and cached float64 profiles; it does not demonstrate that the raw source schema, categorical semantics, target definition, or decision-time feature availability are preserved for production use.
- **Next experiment:** Raw-schema and decision-time availability audit — For every feature and the target, provide raw-source lineage, timestamp/availability relative to prediction time, and encoder requirements; success is achieved only if all features are available at decision time and no label or post-outcome information is used in feature construction.
- **Challenges:**
  - `EXP-046-C1` (medium): The row count, feature count, and positive rate are supported by the evidence, but the claim implicitly suggests a stable dataset summary without showing raw-source provenance or target construction details.
  - `EXP-046-C2` (high): The statement that 0 features triggered missingness, identifier, near-constant, or random-split shift review rules is only a limited screening result. It does not rule out temporal drift, leakage, or production-availability problems, and the package itself notes that random train/holdout PSI is only a smoke check.

### EXP-047 · wine_quality

- **Confidence:** medium
- **Decision:** Insufficient evidence to identify or approve any column as safe or unsafe. A decision-time and production-availability review is still required before any feature can be treated as non-leaky.
- **Interpretation:** The package only establishes that the detector can recognize an injected exact-target canary and that no leakage features were declared. It does not establish which real columns violate the prediction-time contract, because the key evidence needed to compare safe versus unsafe training behavior is missing. The absence of declared leakage features is not evidence of absence of leakage; it is only an absence of declaration.
- **Next experiment:** Column-level decision-time availability audit with safe-vs-unsafe comparison — For every column, provide a decision-time availability label and a production-availability label; flag any column that is unavailable at scoring time or whose univariate association materially exceeds the rest under the same validation split. If no columns are flagged, the audit must still include the full column-by-column evidence table and the validation protocol.
- **Challenges:**
  - `EXP-047-C1` (medium): The canary pass only shows the detector can catch an injected exact-target proxy; it does not imply the detector will catch real-world leakage patterns or that the audit is complete.
  - `EXP-047-C2` (high): The statement 'Declared decision-time exclusions for wine_quality: none' is unsupported as a decision because the evidence bundle provides no column-level decision-time availability review and the safe_vs_unsafe_training_cv field is null.

### EXP-048 · wine_quality

- **Confidence:** medium
- **Decision:** Do not accept the recommendation as written. The ratios recipe is the best reported candidate for the next falsifiable screen, but the current selection claim is unsupported because the selected stage is not within 0.002 mean ROC-AUC of the best stage and is materially worse than the ratios stage.
- **Interpretation:** The evidence supports only a narrow inference: among the reported stages, the ratios recipe has the highest mean training-CV ROC-AUC and a moderate feature count, so it is the strongest candidate for further screening. However, the stated selection rule and the reported selected stage are internally inconsistent with the best-stage criterion, and the package does not demonstrate that the chosen recipe improves validation evidence without feature dilution. The evidence is also limited to training-CV summaries from 3 folds, so it cannot establish robustness, leakage resistance, or production suitability.
- **Next experiment:** Nested fold-wise comparison of raw vs ratios with leakage checks — Run a nested CV comparison using the same outer folds for raw and ratios, with all feature construction fit strictly inside each training fold. Accept the ratios recipe only if its outer-fold mean ROC-AUC exceeds raw by at least 0.002 and the fold-wise improvement is consistent in at least 2 of 3 folds; otherwise reject the claim.
- **Challenges:**
  - `EXP-048-C1` (high): The recommendation says to use the ratios recipe for the next model screen because it achieved 0.8429 mean training-CV ROC-AUC with about 32 features, but the evidence package also contains a selected stage with 25 features and much lower mean ROC-AUC, which conflicts with the stated selection rule and weakens the recommendation logic.
  - `EXP-048-C1` (high): The phrase 'improves validation evidence without feature dilution' is not supported by an independent validation set; the evidence is only training-CV and cannot establish validation improvement beyond the reported CV summary.
  - `EXP-048-C1` (medium): The recommendation implicitly treats a 0.002 ROC-AUC margin as meaningful, but no uncertainty-aware comparison or paired fold-level test is provided to show that the ratios stage is reliably better than nearby alternatives.
  - `EXP-048-C1` (medium): The generic limitation about domain semantics is acknowledged, but there is no decision-time or production-availability review for the ratio features, so the feature recipe should not be approved for use.

### EXP-049 · wine_quality

- **Confidence:** medium
- **Decision:** Provisional only: extra_trees can be treated as the current training-CV candidate, but not as a validated winner. Do not finalize algorithm-family choice until a leakage-safe repeated CV comparison and locked holdout check are completed.
- **Interpretation:** On the provided training-CV evidence, extra_trees is a plausible top candidate by the stated ranking rule, but the package does not establish that it is the strongest stable family in a decision-grade sense. The evidence is limited to 3-fold training CV on one feature stage, with no locked holdout, no repeated CV, and no explicit leakage audit. The apparent tie between extra_trees and lightgbm in mean ROC-AUC is especially important because the selection rule also uses fold variability and runtime, but the package does not provide the computed composite scores needed to verify the ranking outcome directly.
- **Next experiment:** Repeated leakage-safe CV comparison of extra_trees vs lightgbm — Run 5x repeated stratified CV with identical fold splits for extra_trees and lightgbm, compute the stated composite score for each fold, and require that the winner is unchanged in at least 4 of 5 repeats with no fold showing evidence of preprocessing leakage; otherwise the current ranking is not stable.
- **Challenges:**
  - `EXP-049-C1` (medium): The statement that extra_trees is the training-CV model candidate is not directly verified by the package because the composite selection score is not reported. The mean ROC-AUC shown for extra_trees matches lightgbm exactly, so the ranking depends on unreported variability/runtime tie-breaking.
  - `EXP-049-C1` (high): Calling extra_trees the strongest stable evidence overstates certainty: stability is based on only 3 folds and 1 repeat, which is too little to support a robust family-level conclusion.
  - `EXP-049-C1` (high): The claim is limited to training CV, but the wording can be misread as a general recommendation. There is no locked holdout or production review, so the evidence does not justify broader approval.

### EXP-050 · wine_quality

- **Confidence:** medium
- **Decision:** Hold. The holdout result can be reported descriptively, but the optimization-gain claim is unsupported without the missing CV comparison, and no feature promotion decision should be made from importance alone.
- **Interpretation:** The package supports a descriptive statement that the preselected recipe achieved a strong holdout ROC-AUC on the consumed holdout, but it does not by itself establish that optimization produced a meaningful CV gain. The evidence bundle includes a decision rule for accepting tuned candidates, yet it does not expose the baseline CV score, the tuned CV score, or their difference, so the gain question is unanswerable from the provided evidence alone. The feature evidence is consistent with a cautious feature-governance stance, but it is not sufficient to justify promotion of any feature because decision-time availability and production availability are not demonstrated here.
- **Next experiment:** Baseline-vs-tuned CV delta audit with decision-time feature check — Report baseline mean CV ROC-AUC, tuned mean CV ROC-AUC, and their delta on the same folds; accept optimization only if delta >= 0.001. Separately, for each candidate feature, provide a pass/fail decision-time availability record before any promotion discussion.
- **Challenges:**
  - `EXP-050-C1` (high): The holdout ROC-AUC and bootstrap interval are factual, but the wording can be misread as evidence of model superiority or optimization success. The package does not provide the baseline CV score or the tuned CV score needed to assess whether optimization produced a meaningful gain under the stated rule.
  - `EXP-050-C2` (medium): The recommendation is directionally sound, but the evidence bundle does not actually demonstrate decision-time availability, contractual obtainability, or monitoring implementation for the listed features; it only states a rule.

