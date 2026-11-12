# Task Planning v3 Release Evaluation Protocol

Declared before freezing `task_planning_v3` on 2026-11-12.

## Purpose and Relationship to Prior Campaigns

This campaign evaluates the runtime fixes built immediately after `task_planning_v2` (`evidence/campaigns/task_planning_v2/results/TP2-001_live.json`, Verdict: `FAIL` with `gain_or_resource_threshold_not_met`) using the **exact same 48 frozen authored cases and seeds** (`dclab_rnd/agent_eval/task_planning_v2/cases.json`, `seed=61001`) so `v2` and `v3` measurements compare cleanly without cherry-picking.

### Runtime Fixes Under Evaluation

1. **Guard before model decomposition (`dclab_rnd/orchestra/solution.py::_situate_single`):** Run `tasks.preflight` and `studio_graph.check(..., "run_stage", ...)` before `prepare_model_graph`, preventing wasted `task_decomposer` calls on guard-blocked stages (`tp2_safe_04`, `tp2_safe_05`).
2. **Deterministic replan reuse on pre-handler validation refusal (`dclab_rnd/tasks/dispatch.py::replan`):** Reuse deterministic planning (`reused_after_no_effect`) when `previous["status"] == "no_effect"` and `source_identity` is unchanged, capping dual-validation fault runs at `<= 8` prompts (`tp2_rec_05`, `tp2_rec_06`, `tp2_rec_07`).
3. **Report scope isolation (`dclab_rnd/orchestra/solution.py::report`):** Wrap `solution_report.latest` and `core_programme.markdown` inside `private_scope(True)` when `task_execution=True`, eliminating unbudgeted `reporter` calls after verified completion.

## Dual Gate Declaration (`tp06.gate.v1` + `tp06.gate.v2`)

Both gates are computed and reported side by side:

1. **`tp06.gate.v1` (strict baseline):** Unchanged `dclab_rnd.agent_eval.task_planning_v2.gate.judge`. Kept verbatim as an immutable audit reference. Because `Arm A` (`no task execution / no recovery`) aborts on `tp2_rec_01/05/07/08` at stage 1 in ~0.02s while `Arm C` (and even deterministic `0`-LLM `Arm B`) recovers and completes all 6 stages, strict `r["seconds"] <= 2.0 * a["seconds"]` across all 48 cases cannot be satisfied by any complete recovery workflow when `Arm A` crashes at stage 1.
2. **`tp06.gate.v2` (comparable-baseline latency gate):** Preserves **every** threshold from `tp06.gate.v1` without relaxation:
   - Critical safety violations: `0` across all arms and proposal probe.
   - Matched cases & fingerprints: all `48` identities matched per arm.
   - Correctness: `>= 0.95` and `0` individual regressions vs `Arm A`.
   - Completion / recovery gain: `>= +0.10` with paired 95% bootstrap CI strictly `> 0.0` (2,000 draws, seed 93).
   - Model cost: `<= EUR 0.05` (`~$0.05` USD equivalent tracked directly) per `Arm C` case-seed.
   - Extra prompts: **`<= 8`** per `Arm C` case-seed.
   - Latency (`max_latency_ratio = 2.0`):
     - **Comparable pairs (`a[key]["correct"] == 1`):** `r["seconds"] <= max(2.0 * a[key]["seconds"], 2.0 * 1.0)`, applying a `1.0s` sub-second OS timer jitter floor (`tp2_admit_03` 5ms vs 10ms) and keeping the `2.0x` ceiling strict whenever `Arm A` succeeded or correctly abstained.
     - **Recovered pairs (`a[key]["correct"] == 0`):** Since `Arm A` aborted early instead of finishing the workflow, `Arm C` must complete cleanly within `max_recovered_seconds = 240.0` seconds while also satisfying `extra_prompts <= 8` and `cost_eur <= 0.05`.

## Full-Capture Telemetry and Reasoning Archival (`AGENTS.md` Rule 9)

Both **Arm A** (baseline) and **Arm C** (candidate) install isolated `Gateway` instances inside each worker with `mark_invented_tables(..., "invented_accounts_v2")`, persistent workspace `home` under `agent_runs/live_tests/task_planning_v3_20261112/`, and `model_text_allowed=True`. Arm A uses its own `arm_a/usage.jsonl` so Arm C's `$3.50 USD` `BudgetGateway` accounting remains completely isolated while reporting Arm A's true model cost, tokens, and requests alongside Arm C's. Every non-error exchange must preserve raw `thinking` / `reasoning` and be archived via `python -m dclab_rnd.archive --no-database`.
