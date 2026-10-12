# SFT sample factory: the whole DCLab workflow

Generates supervised fine-tuning (SFT) samples for **every step** of the DCLab workflow agent, built on the 10 real datasets in `external_data/`:

```
data_understanding → leakage_audit → feature_engineering → model_selection → optimization_reliability
```

Each sample is **one decision at one step**: the input shows the situation and computed evidence, and the output is the decision, the reasoning (citing only numbers shown in the input), the rule, a confidence level and the next step.

## How the right answer is known

Two methods, both producing labels you can trust:

| Method | Used for | How |
|---|---|---|
| **Planted problems** | data understanding, leakage audit | Add one problem whose truth we control (duplicate rows, a copy of the target, a value recorded after the outcome...). A checker confirms the computed evidence actually shows it, and that nothing else does |
| **Measure, then apply your rule** | feature engineering, model selection, optimization | Actually train and evaluate every option on paired folds of real data, then apply the repo's own rule (R07, R13, R14, R16, R17, R18, R19) to the numbers shown |

Some feature-engineering and model-selection scenarios replace the outcome with a **simulated one built from the real columns** (an interaction, a ratio, a log effect, a linear or a threshold mechanism). This adds variety the real targets don't have. The dataset card states it openly, and the label still comes from the measured results, never from the mechanism.

## The five stages

| Stage | Decisions | What the evidence shows | Rules cited |
|---|---|---|---|
| `data_understanding` | proceed, deduplicate_before_split, use_imbalance_aware_evaluation, impute_within_folds, small_data_protocol | rows, rows per feature, positive rate, duplicate rate, top missing columns, near-constant columns | R01, R02, R03, R04, R12, R13, R16, R18 |
| `leakage_audit` | keep, exclude, flag_for_review, rebuild_within_folds | per column: missingness by class, unique ratio, univariate ROC-AUC, target-copy check, suspicious name, row-number pattern, dates after the decision moment, CV with vs. without | R04, R05 |
| `feature_engineering` | keep_raw, adopt_log_transforms, adopt_ratios, adopt_interactions | raw vs. each recipe on the same 5 folds: mean, std, paired gain, folds improved | R07, R16 |
| `model_selection` | logistic_regression, random_forest, extra_trees, hist_gradient_boosting | 4 families on the same folds: mean, std, stability-adjusted score, runtime | R13, R14 |
| `optimization_reliability` | calibration: keep_uncalibrated / calibrate_sigmoid / calibrate_isotonic. Final holdout: accept_result / investigate_distribution_shift / investigate_gap_without_shift | ECE, ROC-AUC and Brier before/after calibration; CV vs. holdout ROC-AUC, the difference, top feature shifts (PSI) | R14, R17, R18, R19 |

The decision thresholds are written in each stage's card inside the input, for example "adopt only if the paired gain is at least +0.005 and it improves on every fold." That way the model learns to apply a stated rule, not to guess a hidden one.

## Checks every sample passes

`check_and_merge.py` runs these on all stages and rejects failures:

1. The decision is one of the actions allowed in that input.
2. **Every number in the reasoning appears in the input.** No invented evidence. (This caught a real bug during the build: a CV-minus-holdout difference that wasn't printed. It's now part of the evidence.)
3. No duplicate inputs.
4. Each dataset is in exactly one split. Validation is `wine_quality`; test is `online_shoppers` and `german_credit`, so test scores measure skill on unseen data.
5. The answer word never appears in the evidence (model selection excepted, since the candidates are the options).

The leakage stage has two more checks:

- Column-name pools are shared across labels, so the name alone can't give the answer away. A name-only guesser scores about 53% on test (always answering "exclude" scores about 46%).
- A hand-written evidence-only rule agrees with about 96% of labels.

See `out/all_stages/report.json` for counts per stage and per decision.

## Run it

```bash
pip install pandas pyarrow scikit-learn
python sft_factory/leakage_audit_samples.py --variants 8                 # ~10 min
python sft_factory/workflow_samples.py --variants 16                     # ~45 min on 2 CPUs
python sft_factory/workflow_samples.py --stages feature_engineering_balance --variants 9
python sft_factory/check_and_merge.py                                    # checks + merged set
```

The merged training set is in `sft_factory/out/all_stages/{train,val,test}.jsonl`, and each stage also has its own folder.

## Train on it (LoRA, loss on the answer only)

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

ds = load_dataset("json", data_files={"train": "sft_factory/out/all_stages/train.jsonl",
                                      "validation": "sft_factory/out/all_stages/val.jsonl"})
# prompt/completion form: TRL computes the loss on the completion (the answer) only
ds = ds.map(lambda r: {"prompt": r["messages"][:2], "completion": r["messages"][2:]},
            remove_columns=["messages", "metadata"])

trainer = SFTTrainer(
    model="Qwen/Qwen3-8B",
    train_dataset=ds["train"], eval_dataset=ds["validation"],
    args=SFTConfig(output_dir="out/dclab-workflow-lora", num_train_epochs=3, learning_rate=1e-4,
                   per_device_train_batch_size=4, gradient_accumulation_steps=4,
                   max_length=1536, eval_strategy="epoch"),
    peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
)
trainer.train()
```

**What to measure on `test.jsonl`, per stage:**
- Accuracy of the `DECISION:` line, overall and per decision type
- Decisions outside the allowed list (should be 0)
- Numbers in `WHY` that aren't in the input (should be 0)

Accuracy per decision type matters most where one decision dominates. In feature engineering, "keep raw" is about 70% of samples, which mirrors your registry (raw wins most of the time). So check that "adopt" decisions are actually learned, and if not, upsample them in training.

## Known limits

- The datasets are UCI's numeric caches: categoricals were factorized upstream, so "category" semantics are thin.
- Each sample is a single decision. Full multi-step trajectories (stage results feeding the next stage) are the next layer to build on top of these.
- The planted ratio scenario rarely clears the feature-engineering bar, so `adopt_ratios` stays rare. The label is honest; the scenario could be made stronger.

## Next steps

1. **Trajectory samples:** chain one dataset's five decisions into one sequence, so the model learns the order and hand-offs.
2. **Mistake → fix samples:** show a wrong decision and train the correction.
3. **Teacher-model samples:** a strong model answers the same inputs, and only answers matching these labels are kept. This adds varied phrasing.
4. **New datasets:** add a folder in `external_data/` and an entry in `dclab_rnd/agentic/catalog.py`.
