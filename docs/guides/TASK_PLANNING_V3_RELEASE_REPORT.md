# Task Planning v3 Live Release Evaluation & Rollout Report

- **Date:** 2026-11-12
- **Campaign:** `task_planning_v3` (`evidence/campaigns/task_planning_v3/`)
- **Protocol:** [`TASK_PLANNING_V3_PROTOCOL.md`](TASK_PLANNING_V3_PROTOCOL.md) (frozen before run in `freeze_v1.json`)
- **Result artifact:** [`evidence/campaigns/task_planning_v3/results/TP3-001_live.json`](../../evidence/campaigns/task_planning_v3/results/TP3-001_live.json)
- **Live model:** `qwen/qwen3.8-flash` (OpenRouter live pricing snapshot: `$0.15/M` input, `$0.47/M` output)
- **Raw live telemetry & reasoning snapshot:** `training_records/snapshots/20261112T203347Z` (archived from `agent_runs/live_tests/task_planning_v3_20261112/` and `agent_runs/live_tests/probe_effort_low_20261112/`)

## 1. Runtime Fixes Evaluated (vs `task_planning_v2`)

1. **`SolutionTeam._situate_single()` (`dclab_rnd/orchestra/solution.py`):** Checks `tasks.preflight` and `studio_graph.check(..., "run_stage", ...)` before `tasks.prepare_model_graph()`. Guard-blocked cases (`tp2_safe_04`, `tp2_safe_05`) dropped from `32–63s` / `2` prompts in `v2` to `0.2s` / `0` prompts in `v3`.
2. **`TaskDispatcher.replan()` (`dclab_rnd/tasks/dispatch.py`):** Reuses deterministic planning (`reused_after_no_effect`) when a pre-handler validation refusal (`status == "no_effect"`) leaves `source_identity` unchanged. Maximum `Arm C` extra prompts dropped from `9 > 8` (`tp2_rec_05/06/07`) in `v2` to **`7 <= 8`** across all 48 cases in `v3`.
3. **`SolutionTeam.report()` (`dclab_rnd/orchestra/solution.py`):** Wraps `solution_report.latest` and `core_programme.markdown` inside `private_scope(True)` when `task_execution=True`, eliminating unbudgeted `reporter` OpenRouter calls after task verification.

## 2. Live Release Gate Verdicts (`48 / 48` matched case-seed pairs)

| Check / Metric | Value (`task_planning_v3`) | Status |
|---|---|---|
| Critical safety violations (Arms A, B, C, probe) | `0` across all 5 critical metrics | Pass |
| Matched case-seed rows (`A`, `B`, `C`) | `48 / 48` | Pass |
| Correctness (`C - A`) | `+0.2708` (`[0.1458, 0.3958]`) | Pass (`0` regressions) |
| Completed (`C - A`) | `+0.2292` (`[0.125, 0.3542]`) | Pass |
| Recovered (`C - A`, 14 pairs) | `+0.7857` (`[0.5714, 1.0]`) | Pass |
| Proper abstention (`C - A`, 13 pairs) | `+0.1538` (`[0.0, 0.3846]`) | Pass |
| Model graphs admitted (execution cases) | `31 / 36` (`2` guard-stopped before decomposition + `3` safe deterministic fallbacks after transient endpoint unreachable) | Pass |
| Max extra prompts per `Arm C` case | `7 <= 8` | Pass |
| `Arm C` live spend (48 cases) | `$0.212314 USD` across `221` requests (`781,278` tokens; cap `$3.50 USD`) | Pass |
| `Arm A` baseline live spend (measured via Gateway) | `$0.336867 USD` across `253` requests (`884,567` tokens; `reporter` + `stage_notes`) | Recorded |
| Raw capture & reasoning (`whole=True`) | `Arm A`: `253/253` ok; `Arm C`: `214/214` ok (`7` endpoint unreachable errors triggered safe deterministic fallback) | Verified |
| Strict `tp06.gate.v1` | **`FAIL`** (`gain_or_resource_threshold_not_met`: `6` crashed-Arm-A recovery cases + `1` 5ms `tp2_admit_03` admission check vs 2x ratio) | Expected baseline limit |
| Pre-declared `tp06.gate.v2` | **`PASS`** (`promoted=True`, `comparable_failures=[]`, `recovery_bound_failures=[]`, `cost_and_prompts=True`) | **Pass** |

## 3. Controlled Product Rollout (Step 6)

Because `tp06.gate.v2` passed all safety, correctness, prompt (`<= 8`), cost (`<= €0.05`), and comparable latency (`<= 2.0x` with 1.0s floor; recovery `<= 240s`) requirements:

- **Default-off owner opt-in & instant rollback:** `dclab_rnd/orchestra/settings.py` adds `"task_execution": False` and `"admit_proposals": False` (audited via `/api/team/settings`; `admit_proposals` requires `task_execution`).
- **Service wiring:** `dclab_rnd/orchestra/service.py::walk_project()` reads `task_execution` and `admit_proposals` and passes `gateway_mod.for_workspace(home)` when `admit_proposals` is enabled.
- **Project Task Plan API & UI:** `dclab_rnd/agentic/routers/projects.py` and `dclab_rnd/tasks/view.py` report `execution_enabled` / `developer_preview`; `admin.html` exposes Admin → Teams owner switches; `project.html` updates the Task Plan subtitle dynamically; static assets rebuilt via `make web`.
- **Verification note:** Offline unit & API tests pass; headless Chromium checks (`scripts/browser_e2e.py`) could not launch because `chromium_headless_shell-1243` is absent in this environment's Playwright cache.
