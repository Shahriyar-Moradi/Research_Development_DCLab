# Unified Agentic and Reliability Roadmap (2026-11-19)

Agreed design from owner review (2026-11-19). Reuses the existing implementation specs in `Dev_Plans/` and records the agreed sequence, autonomy boundaries, and Leakage Advisor specification.

## Execution Sequence

1. **I2 - Version-aware alert history** ([Five Improvements and Five Gaps](Five_Improvements_and_Five_Gaps_2026-11-19/IMPLEMENTATION_PLAN.md))
   - Deduplicate monitoring alerts in `dclab_rnd/studio/service_template/serve_log.py` by `(model_id, model_version, window_type, window, alert)` inside one process-safe critical section.
2. **I1 - Export the exact fitted, approved artifact** ([Five Improvements and Five Gaps](Five_Improvements_and_Five_Gaps_2026-11-19/IMPLEMENTATION_PLAN.md))
   - Capture the approved fitted artifact and serving metadata at the original final evaluation so `dclab_rnd/studio/service.py::export` never refits or re-reads training/holdout data.
3. **A1 to A6 - Bounded agentic Lead loop, Advisor consultation, and Problem-scoped active memory** ([Agentic Plan and Context Plan](Agentic_Plan_and_Context_Plan_2026-11-20/IMPLEMENTATION_PLAN.md))
   - **A1-A2:** Bounded per-turn choice menu (`ask`, `inspect`, `consult`, `compare`, `revise/stop`) with host-enforced permissions, budgets, splits, and idempotency. AI freely runs safe training-side inspections, Advisor consultations, and budgeted training-fold comparisons; solution/contract edits, contested columns, and holdout/export require explicit approval or strict validation.
   - **A3-A6:** Full problem-scoped memory contract (`problem` scope, durable problem IDs, strict principal/project auth checks, editable project and problem bases in UI, and deterministic revision-checked resumable summary shared by Home, Lead, and Advisor).
   - **Advisor visibility and gated roles:** Advisor rationale, uncertainty, dependencies, and staleness warnings are visible to the user as provisional advice clearly separated from confirmed facts. Gated roles that already passed their individual gates (`problem_analyst`, `data_investigator`, `solution_planner`) may run in advisory-only mode behind a workspace switch before `layer_gate_v1`.
4. **Leakage and Prediction-Moment Advisor in `stage_leakage`** (addendum below)
5. **A7 - Paired measurement before expanding autonomy** ([Agentic Plan and Context Plan](Agentic_Plan_and_Context_Plan_2026-11-20/IMPLEMENTATION_PLAN.md))
   - Scripted acceptance scenarios first (`DCLAB_NO_LIVE_MODELS=1`), followed by separately authorized live paired evaluation on `qwen/qwen3.8-flash` (never Claude).

---

## Addendum: Leakage and Prediction-Moment Advisor (`stage_leakage`)

- **Scope and privacy:** Reviews all non-identifier columns plus statistical heuristic suspects in `stage_leakage` using only column names, types, summary statistics, heuristic findings, and the active problem's prediction moment (`cell_values=0`; never raw table rows).
- **Visible provisional output:** Emits structured per-column recommendations (`block`, `keep`, or `ask`) with cited evidence references, uncertainty, and rationale, stored separately from confirmed solution decisions and invalidated automatically if the active prediction moment changes.
- **Required user prompt for likely leakage:** Whenever the advisor identifies a feature as very likely leakage, DCLab must explicitly inform the user that the feature is mostly/very likely leakage and ask whether they want to:
  1. **(Recommended comparison flow)** Build and compare **both** without the suspected leakage column and with it on training folds (side-by-side safe vs. leaky training-CV contrast so the user sees the exact leakage lift while keeping the leak-free pipeline as the safe candidate), or
  2. Exclude the suspected column directly without running the leaky comparison.
- **Safety boundary:** Comparing clean vs. leaky runs strictly on training folds (reusing and extending `safe_vs_unsafe_training_cv` / `leakage_contrast`). Blocking or retaining a contested column in `solution.forbidden` requires explicit user confirmation and never reads the holdout twice.
