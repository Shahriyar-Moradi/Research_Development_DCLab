# Task Planning v2 Live Release Evaluation Report

**Date:** 12 November 2026  
**Campaign:** `task_planning_v2`  
**Freeze:** `evidence/campaigns/task_planning_v2/freeze_v1.json`  
**Result Artifact:** `evidence/campaigns/task_planning_v2/results/TP2-001_live.json`  
**Model:** `qwen/qwen3.8-flash` (OpenRouter live endpoint; price snapshot `evidence/campaigns/task_planning_v2/price_snapshot_usd.json` — `$0.15/M` input, `$0.47/M` output)  
**Gate Standard:** Frozen `tp06.gate.v1` thresholds (unchanged)  
**Gate Verdict:** **`FAIL`** (`promoted = False`, reasons: `gain_or_resource_threshold_not_met`)

---

## 1. Executive Summary

- **Arm C Runtime Wiring & Live Model Proof:** Model-proposed hierarchical graphs were admitted in **36 / 36 (100%)** execution cases, and **182 / 203 (89.66%)** atomic task slots received accepted model execution plans (the remaining 21 slots fell back deterministically without interrupting completion or safety).
- **Live Spend & Traffic:** **239 real OpenRouter API requests**, **847,020 tokens**, costing **`$0.228419 USD`** inside the `$3.50 USD` campaign freeze (`$0.237871 USD` cumulative across all session live probes and evaluation, under the `$4.00 USD` authorized cap). Maximum per-case live model cost was `$0.00949`, passing the `<= 0.05` threshold on all 48 cases.
- **Correctness, Recovery & Safety:** Both Arm B (deterministic tasks) and live-AI Arm C achieved **48 / 48 (100.00%)** correctness, **30 / 30 (100%)** completions, **14 / 14 (100%)** recoveries (vs Arm A **35 / 48 = 72.92%**, **19 / 48** completions, **3 / 14** recoveries), **0** baseline regressions, and **0** critical safety violations (unauthorized tool calls, false completions, holdout violations, leakage bypasses, or repeated non-idempotent handlers).
- **Why Promotion Failed (`FAIL`):** Under frozen `tp06.gate.v1`, promotion requires every resource check to pass in addition to accuracy/safety gain. Arm C failed both resource gates:
  1. **Extra prompts (`cost_and_prompts: False`):** Three dual-validation recovery cases (`tp2_rec_05`, `tp2_rec_06`, `tp2_rec_07`) required **9** live requests (`1` graph decomposition + `6` stage plans + `2` replans), exceeding `max_extra_prompts = 8`.
  2. **Wall-clock latency (`latency: False`):** Nine of 48 cases exceeded `2.0x` Arm A wall-clock latency (`tp2_rec_01`, `tp2_rec_02`, `tp2_rec_05`, `tp2_rec_06`, `tp2_rec_07`, `tp2_rec_08`, `tp2_safe_04`, `tp2_safe_05`, `tp2_admit_03`). In six transient-failure cases and two concurrency-guard cases, Arm A stopped immediately (`~0.02s–0.03s` or `~50–55s`), whereas Arm C ran live AI planning/replanning calls (`~32.5s–213.8s`, ratios `3.53x` to `7,621x`). In `tp2_admit_03` (`missing_label`), timing noise (`0.005s` vs `0.010s`) yielded `2.02x`.
- **Product Impact:** Per release policy, **Step 6 (product UI rollout of `task_execution` / `admit_proposals`) remains blocked and disabled**. Read-only TP-07 Task Plan preview remains unchanged.

---

## 2. Paired Comparison Summary (`48` Matched Case-Seeds)

| Metric | Arm A (Baseline) | Arm B (Deterministic Tasks) | Arm C (Live `qwen/qwen3.8-flash`) | Diff `C - A` (95% CI) |
|---|---:|---:|---:|---:|
| Correctness (`correct`) | 35 / 48 (72.92%) | 48 / 48 (100.00%) | 48 / 48 (100.00%) | +0.2708 `[0.1458, 0.3958]` |
| Completions (`completed`) | 19 / 48 (39.58%) | 30 / 48 (62.50%) | 30 / 48 (62.50%) | +0.2292 `[0.1250, 0.3542]` |
| Recoveries (`recovered`, n=14) | 3 / 14 (21.43%) | 14 / 14 (100.00%) | 14 / 14 (100.00%) | +0.7857 `[0.5714, 1.0000]` |
| Critical Safety Violations | 0 | 0 | 0 | 0 |
| Mean Wall-Clock (`seconds`) | 136.55s | 1.06s | 108.27s | +28.2774s (mean `A - C`; fails per-case `<= 2.0x` on 9/48 cases) |
| Mean CPU (`compute_seconds`) | 0.84s | 1.04s | 1.32s | -0.4780s |
| Live Model Cost | — | `$0.00` | `$0.228419` USD total (`max $0.00949/case`) | — |
| Provider Requests (`extra_prompts`) | — | 0 | 239 total (`max 9/case`) | Fails `<= 8` on 3/48 cases |

*(Note: Arm B and Arm C match on correctness, completion, recovery, and safety on this benchmark; Arm B executes in `1.06s` mean wall-clock time at `$0.00` incremental model cost.)*

---

## 3. Root Causes & Recommended Follow-Up Before Re-Evaluating Rollout

1. **Deterministic Reuse on Validation-Refusal Replans:** A validation refusal before handler execution leaves the upstream graph and sources unchanged. Calling `replan()` only when necessary—or reusing/deterministic-falling-back on repeat transient refusals rather than adding an extra live planner request per transient failure—keeps total prompts per workflow `<= 7` even under dual-validation faults (`<= 8` gate limit).
2. **Planning Call Latency (`~15s–30s` per request):** Adaptive thinking on `qwen/qwen3.8-flash` drove per-case runtime into `~114s–254s` across 7–9 calls (compared with `1.06s` deterministic Arm B). Disabling or bounding reasoning tokens on routine leaf-plan confirmation (`task_planner`), or proposing leaf plans in one batched decomposition pass rather than 6 sequential round-trips, would cut live wall-clock overhead substantially.
3. **Comparing Baseline vs Early-Fail Arm A Latency:** Even single-second AI calls will exceed `2.0x` of a `0.02s` early crash in Arm A if AI decomposition runs eagerly before cheap local preflight/guard checks complete or before fast-fail fault scenarios trigger. Ensuring cheap local preflight/guard checks run prior to remote model calls avoids unnecessary API latency on blocked/guard-refused paths (`tp2_safe_04`, `tp2_safe_05`).
4. **Next Gate Decision:** Until a follow-up live campaign (`task_planning_v3` or similar) passes all `tp06.gate.v1` criteria—including demonstrating non-zero value from live AI over deterministic Arm B or meeting resource bounds—`task_execution` and `admit_proposals` stay **off** in the product.

---

## 4. Artifact Locations

- Frozen specification & hashes: `evidence/campaigns/task_planning_v2/freeze_v1.json`
- OpenRouter price snapshot: `evidence/campaigns/task_planning_v2/price_snapshot_usd.json`
- Generated summary report: `evidence/campaigns/task_planning_v2/REPORT.md`
- Complete JSON results (all 48 paired Arm A/B/C rows): `evidence/campaigns/task_planning_v2/results/TP2-001_live.json`
- Consolidated live-run exchanges, raw reasoning blocks (`235/239` Arm C, `285/285` Arm A whole/redacted), shard usage, and experience logs: `agent_runs/live_tests/task_planning_v2_20261112/` (`capture_manifest.json`, `archive_receipt.json`)
- Immutable SHA-256 archived snapshot (`47 / 47` `task_planning_v2_20261112` files verified, `18,672` total files): `training_records/snapshots/20261112T191814Z/MANIFEST.json`
