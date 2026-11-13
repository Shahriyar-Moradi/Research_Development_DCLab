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
