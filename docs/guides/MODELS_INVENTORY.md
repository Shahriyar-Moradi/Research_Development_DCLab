# Every model in DCLab R&D: what it is, how it was trained, which features it reads

Written 2026-11-03 for the owner, from the records in this repository (model cards, campaign results, the students'
register, the code that trains them). Every number below is copied from a card or a result file; where I did not open
the source, the text says so. Dev figures are measured on rows the model never trained on, but they are **not** a sealed
verdict, and the planted tables are invented (see "What the numbers do not show").

## 0. The groups, in one table

| Group | What | Trained by DCLab? | Status |
|---|---|---|---|
| A | Small decision models ("students"), one per decision point | Yes (scikit-learn) | 17 points have one; 2 passed their sealed exam; 0 serve above shadow |
| B | The workflow and understanding models (family, ready, next question, route, intent) | Yes (scikit-learn) | Shadow only; most do not beat the rules |
| C | Language models: Claude Sonnet 5.5 (product), GLiNER2.5-Decide, local Qwen | No, DCLab only uses them | Claude serves; the owner's own fine-tuned model is **not trained yet** |
| D | Learned AutoML (picks a model family from table meta-features) | Yes | Measured on 179 invented problems; not better than "most frequent winner" |
| E | Retrieval and curators | Partly | Retrieval measured; model curator in shadow |
| F | Research track champions (the ML models on real datasets: HyperAck, Telco, Adult ...) | Yes | The "main models" of the R&D tracks |
| G | The models the engine fits on a user's data (per problem kind) | Yes, at run time | Not inventoried here (see 7) |

## 1. How every group-A and group-B model is trained (one recipe)

Code: `dclab_core/decision_models.py` (`train`), features in `dclab_core/answerers.py` (`features`).

1. **Rows.** A decision row is a *state* (bands and counts, never a cell value), the label that counts, and a *group*
   (the table or the written person it came from). Sealed rows are refused by the trainer.
2. **Features from the state** (`features()`): a true/false field becomes `field=True`; a number stays a number
   (scaled for the logistic model); a text band becomes a one-hot `field=band`; a list becomes one feature per token
   (at most 16). The fields `column` and `target` are never used whole, only through their word pieces
   (`name_tokens`), so a model cannot learn a project's name.
3. **Two families, three strengths** (`FAMILIES`, `STRENGTHS`):
   - logistic regression, `C` in {0.1, 1, 10}, features scaled by `MaxAbsScaler`;
   - gradient-boosted trees (`HistGradientBoostingClassifier`), depth = 2 + the whole part of the strength
     (so depth 2, 3 or 12).
4. **Choice.** Out-of-fold accuracy over 5 folds **grouped by table** (a table never sits in two folds). Ties go to the
   smaller strength, then to logistic. The winner is refitted on all training rows.
5. **Calibration.** Platt scaling (sigmoid) on the same table-grouped folds. The card reports the calibration error on dev.
6. **Card.** `ModelCard` keeps the point, family, strength, number of features, the fields read, rows and tables
   trained on, label sources, every candidate's out-of-fold accuracy, dev score beside the rule's, and the limits.
7. **Gate.** `decision_gate_v1` judges the model **once** on a sealed set, with bounds fixed before the run. A model
   that fails stays in shadow beside the rule (`ledger/routing.py`).

## 2. Group A: the column and table students (invented tables)

Training data: invented tables from `dclab_core/benchgen*.py` (planted traps, known truth). Label source for every
row: `benchmark`. Folder: `research/llm-fine-tuning/experiments/decisions/models_v*`.

**Features of the column points** (`models_v1`, all six): `cardinality`, `dtype`, `name_tokens`, `nulls`, `uniqueness`,
`value_pattern`, plus the `column` and `target` fields (read only through name pieces). 109 to 156 features after
one-hot.

| Point | Model | Trained on (rows / tables) | Dev accuracy | Rule on same rows | Sealed exam |
|---|---|---:|---:|---:|---|
| column.semantic_role | boosting, depth 2 | 4,320 / 144 | 0.936 | 0.835 | **pass** |
| feature.leakage_suspect | boosting, depth 3 | 4,896 / 144 | 0.971 | 0.921 | **pass** |
| column.category_code | boosting, depth 3 | 2,880 / 144 | 0.919 | 0.803 | fail |
| column.time | boosting, depth 2 | 4,896 / 144 | 0.973 | 0.941 | fail |
| column.is_identifier | boosting, depth 2 | 4,896 / 144 | 0.991 | 1.000 | fail |
| column.group | boosting, depth 2 | 4,896 / 144 | 0.992 | 1.000 | fail |

**Features of the table points** (`models_v2`):

| Point | Fields read | Model | Rows / tables | Dev | Rule | Sealed |
|---|---|---|---:|---:|---:|---|
| data.cases_table | columns, key_unique, name_says_cases, reason_like, rows, status_constant | boosting, depth 2 | 1,406 / 703 | 0.883 | 0.542 | fail |
| data.cleaning_step | cardinality, column, dtype, name_tokens, nulls, rows, uniqueness, value_pattern | logistic, C 0.1 | 8,472 / 706 | 0.922 | 0.726 | fail |
| split.strategy | columns, date_columns, has_groups, has_time, id_columns, repeat_share, rows | logistic, C 1 | 1,406 / 703 | 0.772 | 0.504 | fail |
| task.framing | cardinality, classes, column, columns, dtype, integer_valued, name_tokens, nulls, rows, uniqueness, value_pattern | boosting, depth 2 | 1,424 / 712 | 0.916 | 0.790 | fail |

`models_v3` (new seeds): `data.cleaning_keep` logistic dev 0.869 vs rule 0.746; `modifier.groups` logistic dev **0.428
against the rule's 0.809** (worse than the rule); `modifier.time_order` dev 0.731 = the rule.

**Image points** (`models_images_v1`): `image.near_copy` (fields images, margin,
near_threshold, same_label, similarity; logistic) dev 0.809 vs rule 0.810; `image.label_suspect` (copies,
copies_agree, images, neighbours_agree; boosting) 0.996 vs 0.993; `image.flip_safe` (classes, flip_accuracy, flip_drop,
image_side, images, plain_accuracy; logistic) 1.000 vs 1.000 on only 80 training rows. All three failed their exam.

### The "improved" versions: trained with real tables

`models_real_v1/v2` add real OpenML tables (their declared truths) to the invented ones.

| Point | Model | Rows / tables | Features | Dev | Rule |
|---|---|---:|---:|---:|---:|
| column.semantic_role | logistic, C 10 | 22,341 / 931 | 6,618 | 0.996 | 0.822 |
| column.is_identifier | boosting, depth 3 | 5,628 / 163 | 608 | 0.992 | 1.000 |
| task.framing | boosting, depth 2 | 1,943 / 1,231 | 239 | 0.981 | 0.971 |

Why they exist: the first students scored much lower on real tables. The students' register shows the invented-table
student on real tables: `column.is_identifier` 0.989, `column.semantic_role` 0.803 (invented 0.936),
`task.framing` 0.748 (invented 0.916). The real-table versions are dev scores only; I found no sealed verdict for them.

## 3. Group B: workflow, understanding and intent models

Training data: written people played through the Home conversation (journeys); the state holds counts, flags and codes,
never the person's words. Label: the journey's truth. Folder: `workflow_models_v1`, `understand_models_v1/v2`.

| Point | Fields read | Model | Rows / groups | Dev | Rule | Sealed |
|---|---|---|---:|---:|---:|---|
| problem.family | labels, open_questions, outcome_known | logistic, C 1 | 160 / 80 | 0.526 | 0.895 | fail |
| problem.ready | family, labels, open_questions, outcome_known | logistic, C 0.1 | 196 / 98 | 0.905 | 0.905 | fail |
| solution.route | family, has_text, has_time, labels, parts, related_tables, sub_problem, tables | boosting, depth 2 | 154 / 66 | 0.909 | 0.879 | fail |
| workflow.next_question | labels, moment_known, open_questions, outcome_known, pack, step, steps_left | logistic, C 0.1 | 936 / 107 | 0.995 | 0.979 | fail |

With 600 more written people (`understand_models_v1`), the same four points: family 0.498 (rule 0.821), ready 0.887
(rule 0.827), route 0.820 (rule 0.826), next_question 0.876 (rule 0.855). Conclusion: a state-only model **cannot read
the problem family**, because the state has no words.

**Text students** read the words instead (character 2-to-5-grams, tf-idf, balanced logistic regression, `C` 4):
- `problem.family_from_words`: trained on 719 people, dev on 461: accuracy **0.898** (the rule 0.158 on these, because it
  reads the family on only a third of first messages). It is 0.838 on writer A and 0.971 on writer B, so it partly
  learned a writing style. The sealed exam is **not read**: the gate needs a lower bound of 0.90 and dev predicts a fail.
- `request.intent` (the intent classifier, shadow only): 871 written requests from 3 writers, 22,882 features, dev
  0.851; trained on one writer and tested on another, 0.776 to 0.866. It answers nothing in the product yet.

## 4. Group C: language models

| Model | Role | Trained by DCLab? | Evidence |
|---|---|---|---|
| Claude Sonnet 5.5 (`claude-sonnet-5-5`) | The product's model for every role (lead agent, explanations, decisions asked in `decision_answer`) | No | Called through the gateway; thinking summaries are kept |
| GLiNER2.5-Decide (open weights, Apache 2.0) | Examinee and committee member; answers captured whole | No | Used beside every student in `students_exam_v1` |
| Qwen2.5-coder 1.5B (local, Ollama) | One comparison: `DLM-001`, 240 dev decisions, 23 model errors | No | `decision_llm_v1` |
| Qwen3-class 8B solver, decision LLM | The owner's own model (LoRA) | **Not trained** | Recipe and data exporter exist; `finetune/` holds only a README, no run |

Claude against the students (`SEX-001`, 1,800 dev decisions, 300 per point, cap EUR 6): Claude answers from bands and
name tokens only. 36 requests failed.

The six model types the store collects data for (m1 students, m2 workflow decisions, m3 ML decisions, m4 workflow LLM,
m5 solver LLM, m6 step-by-step) are **data sets**, not trained models, apart from m1 to m3's scikit-learn students above.

## 5. Group D to E

- **Learned AutoML** (`automl_v1`): a meta-model picks the model family from a table's meta-features. `AML-002` on 179
  invented self-play problems: mean regret 0.465 (interval 0.403 to 0.529) against always-XGBoost 0.786; "most frequent
  winner" scored 0.395, which is better. I did not open the meta-feature list in the code.
- **Evidence search** (`dclab_rnd/retrieval`): BM25, LSA (tf-idf reduced by SVD) and a hybrid, measured on 60 questions;
  the product uses the hybrid only if it is not worse on any group.
- **Curators** (`dclab_rnd/store/curators`): rules decide; a model curator (gateway purpose `store_curator`) answers in
  shadow only.

## 6. Group F: the research tracks' champions (the main ML models)

Best leakage-safe result per dataset from the evidence registry (`research/<track>/INDEX.md`). Development CV or one
holdout, not independent confirmation.

| Dataset | Champion | Metric | Score |
|---|---|---|---:|
| HyperAck (delivery fraud) | Soft3: soft vote of an ExtraTrees seed-bag (0.40), a LightGBM winner (0.40) and a tuned XGBoost (0.20) | ROC-AUC | 0.9455 (precision 0.937, recall 0.778) |
| Telco churn | CHURN-014: logistic regression with charge features | ROC-AUC | 0.8499 (AP 0.671, fold sd 0.004) |
| Telco churn | TabPFN (deep-model comparison: logistic 0.8448) | ROC-AUC | 0.8496 |
| Adult | CatBoost | ROC-AUC | 0.9274 |
| German credit | CatBoost | ROC-AUC | 0.8112 |
| Credit default | stacking ensemble | ROC-AUC | 0.8005 |
| Bank marketing | LightGBM | ROC-AUC | 0.8029 |
| Spambase | LightGBM | ROC-AUC | 0.9878 |
| Credit-card fraud | ExtraTrees | average precision | 0.8139 (0.7317 to 0.8847) |
| Bike sharing (daily) | XGBoost | MAE | 680 (484 to 890) |
| Clothing reviews | logistic regression | ROC-AUC | 0.9459 |

**HyperAck Soft3 features** (25, leakage-safe: `final_customer_fare` and `final_biker_fare` are dropped): the shared
protocol's pricing, time and geo features, plus `log_first_fare`, `sqrt_distance`, `products_per_km`, `fare_x_distance`,
`fare_x_products`, `distance_x_products`, `is_morning`, `is_night`, `fare_x_hour`, `distance_x_hour`, `category_x_hour`,
`category_x_weekend`, `route_vs_reported`, `abs_lat_delta`, `abs_lon_delta`, `source_lat_abs`, `source_lon_abs`
(`research/tabular-classification/experiments/optimized_safe_model/features.py`). I did not list the other base features.
Note: the TabPFN HyperAck score of 0.981 is flagged in the registry as including the leaky final fares and is not valid.

**Telco CHURN-014**: "charge features" is the experiment's title; I did not open its exact feature list.

## 7. Group G: models the engine fits for a user

The engine runs every problem kind except images (binary, multiclass, regression, rare events, forecasting, time to
event, uplift, ranking, segmentation, streams, entity tables, long text). It chooses the model on validation only and
reads the holdout once. Each kind has its own candidate models (for example logistic regression, boosted trees and
recency for entity tables; an uplift tree for uplift; k-means for segmentation). This inventory does not list them
kind by kind; `dclab_rnd/kinds/capability.json` and `docs/guides/PRODUCT_BUILD_PLAYBOOK.md` (16.15) do.

## 8. What the numbers do not show (read this before trusting a score)

1. **Invented tables.** Group A dev scores come from planted traps. Every table plants the same kinds, so intervals are
   narrow and say "stable across these tables", not "good on other tables". On real tables the first students dropped
   (task.framing 0.916 to 0.748).
2. **Dev is not sealed.** Only 2 of 17 students (semantic_role, leakage_suspect) passed their sealed exam; 0 are above
   shadow. The real-table versions have no sealed verdict I could find.
3. **Several students lose to the rule** on the points where the rule is already near perfect (is_identifier,
   group): a model at 0.99 against a rule at 1.00 adds risk, not value.
4. **Two groups are worse than a rule on dev:** `modifier.groups` (0.428 vs 0.809) and the family student with state
   only (0.526 vs 0.895).
5. **Text students learn style.** The family text student is 0.838 on one writer and 0.971 on another.
6. **The owner's own LLM does not exist yet.** What exists: the data exporters, the LoRA recipe and the serving
   switches. No training run is recorded.
7. **Champions are development numbers** (CV or one holdout), not independent confirmation, and none is production-ready.

## 9. Where to look

| Need | File |
|---|---|
| The register of all 77 decision points and their students | `python -m dclab_rnd.students` |
| A student's card | `research/llm-fine-tuning/experiments/decisions/models_*/<point>.json` |
| A sealed verdict | `evidence/campaigns/decision_models_v1`, `workflow_models_v1` |
| Claude and GLiNER beside the students | `evidence/campaigns/students_exam_v1` |
| Learned AutoML | `evidence/campaigns/automl_v1` |
| Track champions | `research/<track>/INDEX.md` |
