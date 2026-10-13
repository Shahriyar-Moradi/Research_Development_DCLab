# SFT sample factory: the whole DCLab workflow

Generates supervised fine-tuning (SFT) samples for **every step** of the DCLab workflow agent, built on the 10 real datasets in `data/public/`:

```
problem_understanding → contract_check → data_analysis → solution_design     (the front: problem → contract → findings → plan)
data_understanding → leakage_audit → feature_engineering → model_selection → optimization_reliability
```

The front stages are also joined into **connected samples** (problem text → contract → findings → plan), and the
whole workflow into **end-to-end trajectories**: one conversation per table that runs every step, each step reading
what the earlier ones decided, ending with the locked holdout scored once.

Each sample is **one decision at one step**: the input shows the situation and computed evidence, and the output is the decision, the reasoning (citing only numbers shown in the input), the rule, a confidence level and the next step.

## The front of the workflow: problem → contract → findings → plan

Built by `problem_to_plan_samples.py` on six replicas from `dclab_core.replicas` (subscription churn, ticket routing,
delivery time, city prices, store demand, product reviews) and the ten public datasets, each with written problem
descriptions assembled from sentences whose meaning is known.

| Stage | Input | Output | Where the answer comes from |
|---|---|---|---|
| `problem_understanding` | problem text + file headers | the problem contract (`ProblemBrief`): each field stated (with the person's quote), inferred (with the reason) or unknown, plus the open questions | built by construction; checked with `ProblemBrief`, `understand.quoted` and `understand.supports`; open questions from `understand._questions` |
| `data_analysis` | the contract + measurements on the training partition | the findings: severity, statement, implication, decisions affected | `dclab_core.analyse.analyse`, run with the gold contract |
| `solution_design` | the contract + the findings | the plan (decisions, reasons, citations, alternatives, risks) laid on WF-01…WF-10 with its gates | `dclab_core.plan.plan`, every plan passing `plan.validate`; steps from `dclab_rnd.studio.graph` |
| `contract_check` | the contract + a profile of the real table (kind, distinct values, blanks per column) | the checks (target present, fits the task, labelled, varies; time column and entities present), the decision (proceed / fix_contract / ask_owner), contract changes and questions | one mismatch planted per sample (target renamed, wrong inferred task, blank labels, constant target, missing time column, missing entity column) or none; the checks are computed and a sample is kept only if exactly the planted checks fail. A contradicted STATED fact means ask the owner; a contradicted INFERRED fact is corrected |
| `connected_problem_to_plan` | problem understanding, data analysis and solution design in one conversation | all three answers | as above |
| `end_to_end` | the whole workflow in one conversation (9 assistant turns) | contract → contract check (proceed) → findings → plan → leakage audit (JSON, one decision per column) → feature engineering → model selection → calibration → final holdout | `end_to_end_samples.py`; see below |

Why the contract comes first: with no target in the contract, `analyse` cannot test a column against the outcome. On
the churn replica it then misses the planted leak `final_refund_amount` and the plan reports a wrong positive rate.
With the contract, it finds the leak (critical), the real rate, the identifier and the time span, and the plan
forbids the leak and chooses a time split.

**Benchmark safety:** the problem analyst's briefs (`dclab_rnd/agent_eval/analyst`, dev and sealed) are never used.
Each generated description is compared with all of them and dropped at difflib similarity 0.5 or more; the highest
seen is about 0.37.

**What the rule-based reader gets wrong** (diagnostic in `out/problem_to_plan_report.json`, never used as labels):
`dclab_core.understand` often takes the action or target sentence as the goal, and takes `day_of_week`, `Month`
and even the target `defaulted_next_month` as the time column.

## End-to-end trajectories

Built by `end_to_end_samples.py` on the binary scenarios (the churn replica and the ten public datasets; three data
seeds each). One table, one conversation, and every step uses what the earlier steps fixed:

- the **training partition** the plan chose (time-ordered for churn, random otherwise) is the only data used for the
  audit, feature engineering, model selection and calibration; the rest is the **locked holdout**, scored once at the end;
- the **leakage audit** applies the plan: a column it forbids, a column the declared decision moment rules out (for
  example `duration` in bank marketing, the finished-session counts in online shoppers), and an identifier are
  excluded; a column held for confirmation is flagged; the strongest remaining columns and one more are kept, all with
  computed evidence. The plan's own checks miss the moment-only leaks, so the audit is where they are caught;
- **model selection** screens the plan's model families;
- feature engineering, model selection, calibration and the holdout call the same rule functions as the single-step
  stages (`fe_decide`, `ms_decide`, `cal_decide`, `holdout_decide`), so a trajectory and a single sample never disagree;
- only descriptions whose plan picks ROC-AUC as the primary metric are used, because the downstream rules are written
  in ROC-AUC;
- every assistant turn after the contract passes the number check against everything before it.

They are long (median about 6.5k tokens, longest about 13k; see `out/end_to_end_report.json`). 96 trajectories: train 69, val 9, test 18. Train them with loss on **every** assistant
turn and `max_length` 8192 or more, or leave them out of a short-context run.

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
4. Each dataset is in exactly one split. Validation is `wine_quality`; test is `online_shoppers`, `german_credit` and the `product_reviews` and `city_prices` replicas, so test scores measure skill on unseen data.
5. The answer word never appears in the evidence (model selection excepted, since the candidates are the options).

The leakage stage has two more checks:

- Column-name pools are shared across labels, so the name alone can't give the answer away. A name-only guesser scores about 53% on test (always answering "exclude" scores about 46%).
- A hand-written evidence-only rule agrees with about 96% of labels.

See `out/all_stages/report.json` for counts per stage and per decision.

## Run it

```bash
pip install pandas pyarrow scikit-learn
python research/llm-fine-tuning/experiments/sft_factory/problem_to_plan_samples.py                            # ~2 min
python research/llm-fine-tuning/experiments/sft_factory/contract_check_samples.py                             # ~1 min
python research/llm-fine-tuning/experiments/sft_factory/end_to_end_samples.py --variants 3                    # ~10 min
python research/llm-fine-tuning/experiments/sft_factory/leakage_audit_samples.py --variants 8                 # ~10 min
python research/llm-fine-tuning/experiments/sft_factory/workflow_samples.py --variants 16                     # ~45 min on 2 CPUs
python research/llm-fine-tuning/experiments/sft_factory/workflow_samples.py --stages feature_engineering_balance --variants 9
python research/llm-fine-tuning/experiments/sft_factory/check_and_merge.py                                    # checks + merged set
```

The merged training set is in `research/llm-fine-tuning/experiments/sft_factory/out/all_stages/{train,val,test}.jsonl`, and each stage also has its own folder.

## Train on it (LoRA, loss on the answer only)

```python
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig

ds = load_dataset("json", data_files={"train": "research/llm-fine-tuning/experiments/sft_factory/out/all_stages/train.jsonl",
                                      "validation": "research/llm-fine-tuning/experiments/sft_factory/out/all_stages/val.jsonl"})
# prompt/completion form: TRL computes the loss on the completion (the answer) only.
# Multi-turn samples (connected_problem_to_plan, end_to_end) need loss on EVERY assistant turn: keep them as
# "messages" and use assistant_only_loss=True (needs a chat template that marks assistant turns).
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
- End-to-end trajectories cover binary tasks only; the downstream stage rules are written for a yes/no outcome in ROC-AUC.
- In a trajectory the contract check always proceeds (clean data). The mismatch cases live in the single-step `contract_check` samples.
- The planted ratio scenario rarely clears the feature-engineering bar, so `adopt_ratios` stays rare. The label is honest; the scenario could be made stronger.

## Next steps

1. **Trajectories with a detour:** a contract check that fixes the contract or asks the owner, then the run continues from the corrected contract.
2. **Mistake → fix samples:** show a wrong decision and train the correction.
3. **Teacher-model samples:** a strong model answers the same inputs, and only answers matching these labels are kept. This adds varied phrasing.
4. **New datasets:** add a folder in `data/public/` and an entry in `dclab_rnd/agentic/catalog.py`.
