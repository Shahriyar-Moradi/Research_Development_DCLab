# SFT sample factory: the DCLab workflow agent

Supervised fine-tuning data for a model that takes an ML problem through the **whole DCLab workflow**, and the
pipeline that trains **Qwen3.5-9B** on it on an M1 Max (`mac/`, see `mac/RUNBOOK.md`).

## One workflow: WF-01…WF-10

Every sample belongs to one step of the repository's workflow graph (`dclab_rnd.studio.graph`). Each input starts
with its step, and each answer ends with a **route**: continue to the next step, go back to the step the evidence
reopens (only the revisits the graph allows), or ask the owner.

| Step | Stage(s) | What the step decides |
|---|---|---|
| WF-01 Solution draft | `problem_understanding` (+ `solution_revisit` in trajectories) | the problem contract: each field stated (with the quote), inferred (with the reason) or unknown; open questions |
| WF-02 Source and lineage | `contract_check` | does the real table match the contract? proceed / fix_contract (back to WF-01) / ask_owner |
| WF-03 Split design | `data_understanding` | pre-split checks on the whole table: deduplicate first, small-data protocol, imbalance-aware evaluation, in-fold imputation |
| WF-04 Train-only EDA | `data_analysis`, then `solution_design` | findings on the training partition; the plan (one decision per key) laid on WF-01…WF-10; the solution gate |
| WF-05 Leakage audit | `leakage_audit` | keep / exclude / flag_for_review / rebuild_within_folds per column; a leak the plan missed goes back to WF-01 |
| WF-06 Feature ladder | `feature_engineering` | keep raw or adopt a recipe, by paired folds (R07, R16) |
| WF-07 Algorithm screen | `model_selection` | the family, by stability-adjusted score and runtime (R13, R14) |
| WF-08 Conservative optimization | calibration | keep uncalibrated, sigmoid or isotonic (R14, R18) |
| WF-09 Reliability challenge | holdout | scored once: accept, investigate a shift (back to WF-03), or investigate a gap without shift |
| WF-10 Knowledge capture | `report` (in trajectories) | the final report, every number taken from the run |

Routes are added in one place (`workflow_map.py`) for single-step samples and computed with their context in the
trajectories. The single-step format is unchanged: one decision at one step; the input shows the situation and the
computed evidence; the answer gives the decision, the reasoning (citing only numbers shown in the input), the rule, a
confidence level and the next step, which now carries the route, for example `NEXT: [continue → WF-07] …`.

## The sample families

| Family | Files | What it teaches | Where the right answer comes from |
|---|---|---|---|
| Single-step decisions | `problem_to_plan_samples.py`, `contract_check_samples.py`, `leakage_audit_samples.py`, `workflow_samples.py` | each step's decision with its rule | planted problems (known by construction, kept only if the computed evidence shows them) or measure-then-rule (every option trained and scored on paired folds, then the repo's rule applied) |
| Connected front | `problem_to_plan_samples.py` | problem → contract → findings → plan in one conversation | `dclab_core` (`understand` rules, `analyse`, `plan`, `plan.validate`) with the gold contract |
| End-to-end trajectories | `end_to_end_samples.py` | the whole run, WF-01 to WF-10, with going back where the evidence says so | the same rule functions as the single steps, on the partition the plan fixed; the holdout touched once |
| Code, then decide | `code_samples.py` | write the pandas / scikit-learn code for a step's evidence, read its real output, decide | the code is executed on the real data; the decision is the rule applied to the printed numbers |
| Code review | `code_samples.py` | find leakage in code (preprocessing on all rows, selection or oversampling before CV, target encoding on all rows, tuning on the test set, deduplicating after the split, a random split on time-ordered rows) and fix it | the bug is planted; both versions are executed and must run |
| Replay | `replay_tasks.py` + `mac/make_replay.py` | general coding and ML arithmetic, so general skill does not drift | Qwen's own answers, kept only when the tests pass or the number is right |

**Trajectories** (`end_to_end_samples.py`) cover the binary scenarios (the churn replica and the ten public
datasets, three data seeds). Variants: a clean run, and on every third seed a run whose labels are partly blank, so
the contract check corrects the contract and the work goes back to WF-01 before continuing on the labelled rows.
The leakage audit applies the plan and the declared decision moment, so it catches leaks the plan's automatic
checks miss (bank `duration`, online shoppers' finished-session counts) and sends the work back to WF-01 to record
them. A holdout that shows a shift ends the run with a route back to WF-03.

## Checks every sample passes

`check_and_merge.py` applies the routes and checks every stage before merging into `out/all_stages/`:

1. every input starts with its workflow step, and every answer carries its route (except a code turn and the final report);
2. the decision is one of the actions allowed in its own input;
3. **every number in the reasoning appears in the input** (the generators check every turn of a trajectory against
   everything before it);
4. JSON answers parse; code answers compile (and the code generators also ran them);
5. no duplicate conversations; each dataset or scenario in exactly one split (`splits.py`): validation is
   `wine_quality`, `heart_disease` and the `delivery_time` replica; test is `online_shoppers`, `german_credit` and the
   `product_reviews` and `city_prices` replicas;
6. the answer word never appears in the evidence (model selection excepted, since the candidates are the options).

The leakage stage has two more: column names are shared across labels (a name-only guesser scores about 53%), and a
hand-written evidence-only rule agrees with about 96% of labels.

**Benchmark safety:** the problem analyst's briefs (`dclab_rnd/agent_eval/analyst`, dev and sealed) are never used;
every generated description is compared with them and dropped at difflib similarity 0.5 or more.

**What the rule-based reader gets wrong** (diagnostic only, never a label): `dclab_core.understand` often takes the
action or target sentence as the goal, and takes `day_of_week`, `Month` and even the target `defaulted_next_month` as
the time column. See `out/problem_to_plan_report.json`.

## Counts

See `out/all_stages/report.json` (per stage and per decision) and `out/end_to_end_report.json`. One real sample per
stage and decision is in `SAMPLES_PREVIEW.md`.

## Run it

```bash
pip install pandas pyarrow scikit-learn
bash research/llm-fine-tuning/experiments/sft_factory/regenerate_all.sh      # every stage, checks, merge, preview (~3 h on 2 CPUs)
```

Then train on the Mac: `mac/RUNBOOK.md`.

## Training (summary of `mac/`)

| Step | Script | Notes |
|---|---|---|
| Training items | `build_training_mix.py` | one item per answer, earlier answers as context without thinking (Qwen's rule); about 70% thinking and 30% non-thinking items; rare decisions and small stages repeated (at most 4×); a few turns per trajectory |
| `<think>` reasoning | `mac/make_think_traces.py` | Qwen3.5 reasons freely; a trace is kept only if it reaches the gold decision and route with no invented number; long structured answers get a reasoning written towards the gold answer, same checks |
| Replay | `mac/make_replay.py` | Qwen's own correct answers to checkable tasks |
| Train | `mac/train_mlx.py` | LoRA rank 16 on every layer; loss only on what the model writes; logits only where needed; Qwen3.5's linear attention trains chunk by chunk (`mac/gated_delta_chunked.py`, tested equal to MLX's own rule); exact resume |
| Choose and test | `mac/eval_mlx.py` | generation-based: decision and route accuracy per stage and decision, format, invented numbers in answer and thinking, replay pass rate; `--pick` chooses the epoch |

## Known limits

- The public datasets are UCI's numeric caches: categoricals were factorized upstream, so category meaning is thin.
- Trajectories cover binary tasks only (the downstream stage rules are written for a yes/no outcome in ROC-AUC).
- In trajectories the feature ladder almost always keeps the raw columns, as the real data warrants; the
  single-step feature-engineering samples (with simulated mechanisms) carry the adopt decisions.
- Free-text ML reasoning (diagnosing symptoms, critiquing a plan, trade-offs) is not in this set yet: its answers
  cannot be checked as strictly, so it waits for a teacher model with a checked rubric.
- The Mac pipeline was tested end to end on a tiny model with Qwen3.5's architecture and chat template; the hours
  for the 9B model in the runbook are estimates until step 3 of the runbook measures them.

## Next steps

1. Run the Mac pipeline (`mac/RUNBOOK.md`) and compare the chosen epoch with the base model on the test set.
2. Agent format: turn the trajectories into tool calls (the model asks for each step's evidence and receives the
   tool's output), add recovery samples (a failed check, a tool error), and a runtime that checks every move with
   the graph's validator.
3. Reinforcement learning on whole runs with checkable rewards (rules followed, no leakage, honest holdout, no
   invented numbers).
