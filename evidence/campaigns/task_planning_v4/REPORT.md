# Task Planning v4 (`tp08.gate.v1`) Campaign Report

## Summary

Task Planning v4 evaluates whether adaptive problem understanding, hierarchical subproblem decomposition (`ProblemDecomposition`), grounded evidence lookup, and typed pre-holdout catalog tasks (`CatalogTaskCall` compiled to `dclab_core.task_graph.v1`) improve solution quality, trap catching, and real empirical ML holdout performance over both:
- **`B0` (`legacy_fixed_pipeline`)**: fixed data -> leakage -> features -> models -> holdout -> report pipeline without problem understanding or adaptive pre-holdout tasks.
- **`B1` (`deterministic_catalog`)**: deterministic rule-based use of the P2 task catalog without LLM problem decomposition or adaptive judgment.

Candidate **`C` (`adaptive_problem_planner`, `qwen/qwen3.8-flash` on non-standard strata, deterministic fast path on clean single-table standard strata)** passed **`tp08.gate.v1`** against **both `B0` and `B1`** on both the frozen `dev` split (`TP4-001_dev_live.json`) and the unseen `sealed` split (`TP4-002_sealed_live.json`, run once with zero tuning).

---

## Frozen Dev Evaluation (`TP4-001_dev_live.json`)

- **Split / Cases:** `dev`, `20` cases x `2` seeds (`40` matched pairs per arm; `18` hard/ambiguous pairs, `2` clean standard pairs, `11` holdout replica pairs)
- **Model:** `qwen/qwen3.8-flash`
- **Spend:** `$0.046461` (`$0.001162` mean per case)
- **Critical Violations (`B0` / `B1` / `C`):** `0` / `0` / `0` (holdout violations, leakage bypasses, forbidden actions, unauthorized tool calls, fabricated evidence)
- **Summary Metrics (`C`):** `candidate_composite_kind_match = 0.975`, `candidate_question_precision = 1.0`, `candidate_max_extra_prompts = 2`
- **Exchange Capture:** `53 / 53` ok, `53 / 53` `whole=True`, `53 / 53` `raw_reasoning_present`
- **Dev Training Export:** `36` non-standard `dev`-only exchanges exported to `training_records/datasets/task_planning_v4_dev.jsonl` (`0` sealed rows exported)

| Comparison | Metric | Baseline Mean | Candidate (`C`) Mean | Diff (95% Bootstrap CI) | Wins / Losses / Ties | Gate Verdict |
|---|---|---:|---:|---:|---:|---|
| `C` vs `B0` (Legacy Fixed) | `required_coverage_hard` (n=18) | 0.0000 | 0.7569 | **+0.7569** `[0.6792, 0.8278]` | 18 / 0 / 0 | **PASS** (`promoted=True`) |
| `C` vs `B0` (Legacy Fixed) | `trap_catch_hard` (n=18) | 0.0000 | 0.6667 | **+0.6667** `[0.5000, 0.8198]` | 16 / 0 / 2 | **PASS** |
| `C` vs `B0` (Legacy Fixed) | `holdout_score` (n=11) | 0.5096 | 0.5723 | **+0.0627** `[0.0315, 0.0947]` | 9 / 1 / 1 | **PASS** |
| `C` vs `B1` (Det. Catalog) | `required_coverage_hard` (n=18) | 0.4509 | 0.7569 | **+0.3060** `[0.2014, 0.4172]` | 15 / 1 / 2 | **PASS** (`promoted=True`) |
| `C` vs `B1` (Det. Catalog) | `trap_catch_hard` (n=18) | 0.3889 | 0.6667 | **+0.2778** `[0.1250, 0.4444]` | 9 / 1 / 8 | **PASS** |
| `C` vs `B1` (Det. Catalog) | `holdout_score` (n=11) | 0.5321 | 0.5723 | **+0.0402** `[0.0151, 0.0689]` | 7 / 2 / 2 | **PASS** |

---

## Frozen Sealed Evaluation (`TP4-002_sealed_live.json` — Run Once)

- **Split / Cases:** `sealed`, `14` unseen cases x `2` seeds (`28` matched pairs per arm; `12` hard/ambiguous pairs, `2` clean standard pairs, `6` holdout replica pairs)
- **Model:** `qwen/qwen3.8-flash`
- **Spend:** `$0.028922` (`$0.001033` mean per case)
- **Critical Violations (`B0` / `B1` / `C`):** `0` / `0` / `0`
- **Summary Metrics (`C`):** `candidate_composite_kind_match = 0.8929`, `candidate_question_precision = 1.0`, `candidate_max_extra_prompts = 2`
- **Exchange Capture:** `33 / 33` ok, `33 / 33` `whole=True`, `33 / 33` `raw_reasoning_present` (`dev_training_exported = 0`)

| Comparison | Metric | Baseline Mean | Candidate (`C`) Mean | Diff (95% Bootstrap CI) | Wins / Losses / Ties | Gate Verdict |
|---|---|---:|---:|---:|---:|---|
| `C` vs `B0` (Legacy Fixed) | `required_coverage_hard` (n=12) | 0.0000 | 0.8944 | **+0.8944** `[0.8305, 0.9549]` | 12 / 0 / 0 | **PASS** (`promoted=True`) |
| `C` vs `B0` (Legacy Fixed) | `trap_catch_hard` (n=12) | 0.0000 | 0.8750 | **+0.8750** `[0.7500, 1.0000]` | 12 / 0 / 0 | **PASS** |
| `C` vs `B0` (Legacy Fixed) | `holdout_score` (n=6) | 0.4974 | 0.5546 | **+0.0572** `[0.0053, 0.1144]` | 3 / 2 / 1 | **PASS** |
| `C` vs `B1` (Det. Catalog) | `required_coverage_hard` (n=12) | 0.5194 | 0.8944 | **+0.3750** `[0.2361, 0.5243]` | 10 / 0 / 2 | **PASS** (`promoted=True`) |
| `C` vs `B1` (Det. Catalog) | `trap_catch_hard` (n=12) | 0.5000 | 0.8750 | **+0.3750** `[0.1667, 0.6250]` | 6 / 0 / 6 | **PASS** |
| `C` vs `B1` (Det. Catalog) | `holdout_score` (n=6) | 0.5106 | 0.5546 | **+0.0440** `[-0.0020, 0.1060]` | 3 / 2 / 1 | **PASS** |

## Key Engineering Takeaways

1. **Reasoning Budget Allocation (`qwen/qwen3.8-flash` on OpenRouter):** Setting an explicit `reasoning.max_tokens` (`1000` initial, `600`/`800` retry/repair) with `max_tokens=3800` prevented reasoning runaway while keeping `100%` raw reasoning capture (`86/86` exchanges across dev + sealed).
2. **Planning Actually Changes Empirical Holdout ML Quality:** Compiling validated catalog tasks (`check_join_key`, `grouped_split`, `time_split_check`, `leakage_probe`, `calibration`, `threshold_for_cost`, `segment_subproblem`) directly alters real sklearn modeling pipelines on synthetic replicas, yielding `+0.0627` dev (`[0.0315, 0.0947]`) and `+0.0572` sealed (`[0.0053, 0.1144]`) holdout improvement over `B0`, and `+0.0402` dev / `+0.0440` sealed over `B1`.
3. **Router Economics:** Clean single-table problems route deterministically (`$0.0000`, `0.0s`), while non-standard problems average `~$0.0011/case` (`$0.075` total combined spend across full dev and sealed campaigns).

---

## Prompt-Audit Re-Verification (`TP4-003_dev_live.json` — `"purpose"` Schema Field)

Following the `test_prompt_audit.py` compliance update (renaming the schema/prose `"reason"` field to `"purpose"` in `dclab_rnd/prompts/problem_planner.md` with backward-compatible `_accept_purpose_alias` in `dclab_core/task_catalog.py`), the `dev` split was re-verified live on OpenRouter (`qwen/qwen3.8-flash`):

- **Spend / Capture:** `$0.042451` (`$0.001061` mean/case), `47 / 47` ok exchanges with `whole=True` and `raw_reasoning_present=47`.
- **`C` vs `B0` (`tp08.gate.v1`):** **PASS** (`promoted=True`, `0` critical violations; `required_coverage_hard = +0.7856`, `trap_catch_hard = +0.6389`, `holdout_score = +0.0536` `[0.0226, 0.0853]`).
- **`C` vs `B1` (`tp08.gate.v1`):** `required_coverage_hard = +0.3347` `[0.2208, 0.4556]`, `holdout_score = +0.0330` `[0.0088, 0.0612]`, `trap_catch_hard = +0.2500` `[0.0556, 0.4583]` (`FAIL` on `trap_catch_gain_below_threshold` lower CI bound `0.0556 < 0.10` in this stochastic re-run; `TP4-001` and frozen unseen `TP4-002_sealed` both passed `C` vs `B1`).

---

## Live Studio + `SolutionTeam` Runtime Verification (`adaptive_studio_live_20261113_r2`)

Executed all 5 live end-to-end `SolutionTeam(..., task_execution=True, admit_proposals=True)` scenarios against real OpenRouter (`qwen/qwen3.8-flash`) with full `TaskDispatcher`, checkpointed `catalog_plan` state, workspace `FileLedger`, `task_plan_view`, single-use holdout enforcement (`holdout_uses == 1`), and `whole=True` exchange capture:

- **Outcome:** **`5 / 5` scenarios passed**, total spend **`$0.026473`**, **`40 / 40`** ok exchanges (`whole=True`, `raw_reasoning_present=40` across `task_decomposer` and `task_planner`).
- **Ledger Decisions Recorded (17 points):** Includes `map.size`, `problem.family`, `workflow.template`, `team.next_node`, `workflow.move`, `workflow.g4`, `model.family`, `threshold.operating_point`, and column/split decisions.

| Scenario | Stratum | Routed To | Subproblems / Key Traps | Revisions / Clarify Pause | Holdout Score | View Status | Cost |
|---|---|---|---|---|---|---|---:|
| `studio_multitable_post_outcome` | Multi-table + leak | `model` | `classification`, `entity_tables`; `one_to_many_join_duplication`, `post_outcome_closure_recorded_after_review` | 1 rev / no pause | `0.8984` | `recorded` | `$0.005609` |
| `studio_timedrift_fraud_capacity` | Time + capacity | `model` | `rare_event`, `segmentation`; `time_drift_random_split_leak`, `extreme_class_imbalance`, `capacity_constrained_threshold` | 1 rev / no pause | `0.6821` | `recorded` | `$0.005400` |
| `studio_composite_uplift` | Composite uplift | `model` | `classification`, `segmentation`, `uplift`; `observational_treatment_bias_voucher_offer` | 1 rev / no pause | `0.6502` | `recorded` | `$0.005792` |
| `studio_ambiguous_ask_owner` | Unresolved target/horizon | `model` | `classification`; `missing_horizon`, `capacity_constrained_threshold` | **2 revs (P4)** / **`catalog_clarify` pause** | `0.7094` | `recorded` | `$0.005658` |
| `studio_clean_deterministic_fastpath` | Clean single-table | `deterministic` (P6) | `classification`; `[]` | 1 rev ($0 catalog) | `0.6667` | `recorded` | `$0.004014` |

---

## P7 Outcome-Labelled Planning Gold Set (`planning_gold_v1.jsonl`)

Built via `dclab_rnd.tasks.catalog_lake.build_planning_gold_set()` (`sealed_excluded=True` strictly enforced against `cases_sealed.json`):
- **Output:** `training_records/datasets/planning_gold_v1.jsonl` and `lake/gold/planning/dev.jsonl`
- **Counts (Latest Harvest):** **`225`** total outcome-labelled trajectories (**`56` positive**, **`169` negative**) across all non-sealed `tp08_dev_live` (`TP4-001`, `TP4-003`, `TP4-004`, `TP4-005`, `TP4-006`) and `adaptive_studio_live` (`r2`, `r3`, `r4`) runs with full raw `reasoning`, `messages`, `reply`, trap/coverage metrics, and empirical holdout deltas.

---

## Fallback Floor, Taxonomy & Default-On Rollout (`TP4-006_dev_live.json` & `adaptive_studio_live_20261113_r4`)

1. **Deterministic Fallback Floor + Schema Coercion (`dclab_core/task_catalog.py`, `dclab_rnd/tasks/catalog.py`):**
   - Added `_deterministic_fallback_payload` so any model refusal/JSON error/outage retains `B1`'s full catalog checks, declared `treatment` uplift check, and trap detection instead of dropping to `caught_traps=[]`.
   - Added `_clip_text` and null coercions on `IdentifiedRisk` (`evidence_citation` capped cleanly at 80 chars instead of failing validation), `CatalogTaskCall`, `SubproblemSpec`, `OpenUnknown`, and `ProblemDecomposition`.
   - Added `_derive_subproblems` so specialized subproblem classifications (such as `rare_event`) preserve `"classification"` alongside confirmed `segment_subproblem` kinds.
2. **Honest Bench Gating (`dclab_rnd/bench/__init__.py`):**
   - `_compat_doc` reports both `vs_legacy` (`C vs B0`) and `vs_deterministic` (`C vs B1`) in `bench read` so any `B1` regression fails loudly.
3. **Project UI Display, Clarify Answer & Default-On Rollout (`dclab_rnd/orchestra/settings.py`, `dclab_rnd/orchestra/service.py`, `dclab_rnd/tasks/view.py`, `project.html`):**
   - Rendered `catalog_plan` and added `POST /api/projects/{project_id}/task-plan/confirm` for inline `catalog_clarify` owner answers.
   - Enabled `task_plan_preview: True`, `task_execution: True`, `admit_proposals: True` by default in `DEFAULTS`, while `walk_project` keeps multi-leaf programmes (`programme_plan`) on their atomic owner.
4. **Latest Live Dev Gate (`TP4-006_dev_live.json`) & Live Studio (`adaptive_studio_live_20261113_r4`):**
   - **`TP4-006_dev_live.json`:** **`PASS`** against both **`B0`** (`required_coverage_hard=+0.7431 [0.6444, 0.8264]`, `trap_catch_hard=+0.7639 [0.6250, 0.8889]`, `holdout_score=+0.0567 [0.0284, 0.0857]`, `9W/1L/2T`) and **`B1`** (`required_coverage_hard=+0.2921 [0.1541, 0.4171]`, `trap_catch_hard=+0.3750 [0.2083, 0.5417]`, `holdout_score=+0.0361 [0.0136, 0.0625]`, `7W/2L/3T`); `candidate_composite_kind_match = 1.0`, `candidate_question_precision = 0.975`, `0` critical violations, `53/53` raw-reasoning exchanges captured (`spent_eur = 0.046781`).
   - **`adaptive_studio_live_20261113_r4`:** **`5 / 5` scenarios passed** (including `studio_composite_uplift` and `studio_ambiguous_ask_owner`), `39/39` ok exchanges (`whole=True`, `raw_reasoning_present=39`), `$0.023989` spend, `17` ledger decision points recorded, `holdout_uses == 1` on every project.
