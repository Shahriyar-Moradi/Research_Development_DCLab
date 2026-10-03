# expansion_v1 — task-type-aware model-building campaign

`model_building_50_v1` ran 50 experiments (10 datasets × 5 stages) on binary
classification only. `expansion_v1` keeps its evidence discipline but adds the
four task types it lacked, using one real public dataset each:

| Experiments | Dataset | Task type | Primary metric | Locked holdout |
|---|---|---|---|---|
| EXP-051..055 | `credit_card_fraud` (ULB/Kaggle, 284,807 rows) | `binary_imbalanced` | average precision | last 20% of rows by `Time` |
| EXP-056..060 | `letter_recognition` (UCI via PMLB, 20,000 rows) | `multiclass` (26 classes) | macro-F1 | stratified 20% |
| EXP-061..065 | `bike_sharing_daily` (UCI via interpretable-ml-book, 728 days) | `timeseries_regression` | MAE | last 20% of days |
| EXP-066..070 | `ecommerce_clothing_reviews` (Kaggle, 23,486 reviews) | `text_tabular_binary` | ROC-AUC (+ AP) | unseen `Clothing ID` groups (~20%) |

Every dataset runs the same five stages, in order:

1. **data_understanding**: target distribution, missingness, duplicates, identifiers, time range, text lengths, and group concentration.
2. **leakage_audit**: a written decision-time contract with blocked columns and their rationale. Task-specific heuristics only *propose* review candidates: univariate AUC, NMI or binned R², exact proxies, `a + b == target` identities, and uniqueness. Synthetic canaries check the detector itself. The stage also measures the safe-vs-unsafe CV lift (`evidence.apparent_lift`, oriented so that positive means it looks better). For bike sharing it also measures the gap between random KFold and time-ordered CV.
3. **feature_engineering**: 4–5 train-fitted recipes. The stage picks the smallest recipe whose CV score is within a declared tolerance of the best.
4. **model_selection**: five model families on identical folds, ranked by mean ∓ 0.25×fold std.
5. **optimization_reliability**: three explicit parameter candidates, accepted only if they beat the default by a declared margin. Then **one** holdout evaluation with a bootstrap CI: iid for letters, moving-block for bike, and cluster-by-product for reviews. The holdout result is compared with naive baselines. For fraud, a threshold chosen on training out-of-fold scores is applied to the holdout.

Stages 1–4 use training rows only. CV is time-ordered for fraud and bike, grouped by product for reviews, and stratified for letters. The holdout is consumed once, in stage 5 (`holdout_consumed: true`).

## Files

- `results/EXP-0NN_<dataset>_<kind>.json`: one result per experiment. These use the same top-level schema as `model_building_50_v1`, plus `task_type`, `primary_metric` and `setup_summary`, and no `llm_review` key yet.
- `manifest.json`: the 20 experiments, each with its question, hypothesis and status.
- `CAMPAIGN_REPORT.md`: a human-readable summary for each dataset, with limitations.
- `agent_memory.jsonl`: one line per claim. It has the same shape as the original campaign's file, without the provenance block.

## Rerun

Raw files download to `data/external/<dataset>/` on first use. That folder is not committed. Their SHA-256 hashes are pinned in `dclab_rnd/expansion/datasets.py`, and a mismatch stops the run.

```bash
python3 -m dclab_rnd.expansion run                      # resumable: skips completed results
python3 -m dclab_rnd.expansion run --dataset bike_sharing_daily --force
python3 -m dclab_rnd.expansion run --dataset credit_card_fraud --stage optimization_reliability --force
python3 -m dclab_rnd.expansion status
python3 -m dclab_rnd.expansion report                   # regenerate manifest/report/memory
python3 -m unittest tests.test_expansion                # offline, synthetic data only
```

`random_state=42` is used everywhere. The full campaign takes about 10–15 minutes on 4 CPU cores.
These results are kept separate from the `model_building_50_v1` registry on purpose.
