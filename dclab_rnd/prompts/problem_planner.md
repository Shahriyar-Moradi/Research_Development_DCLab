# Adaptive Goal Decomposition & Task Catalog Planner — tp08.v1

Keep internal reasoning VERY BRIEF (<100 words). Do NOT list or evaluate tasks you do not pick. Output ONLY raw JSON immediately starting with `{`.

Inspect `tables`, `known_facts`, and `evidence_sources`, then return ONE compact JSON object:
1. `decomposition`: `ProblemDecomposition` (`decision`, `target`, `unit`, `horizon_days`, `success_metric`, `subproblems` from `available_families`, `risks`, `unknowns`).
2. `calls`: 3 to 6 pre-holdout tasks from `catalog_specs` (stages `data..report` run automatically unless needed). Keep reasons concise (<12 words).

Rules:
- Subproblems (`kind` MUST come strictly from `available_families`):
  - Use `"entity_tables"` when outcomes/features come from aggregating child/event tables (`orders`, `sessions`, `addresses`, etc.) rather than one flat row.
  - Use `"rare_event"` when positives are rare (<5%) or accuracy hides misses; use `"streams"` for repeated sensor/telemetry readings.
  - Use `"uplift"` when optimizing who to treat/contact/offer rather than raw propensity; pair with `"segmentation"` when personas/segments drive the action.
  - Otherwise use `"classification"` or `"regression"`. Emit only essential families.
- Params: use ONLY keys listed in `catalog_specs[task_type]` (`req` = required, `opt` = optional). Never invent keys.
- Grounding: cite ONLY tables/columns in `tables` (or `"cutoff_date"` for snapshot `time_split_check`), and cite valid IDs from `evidence_sources`.
- Ambiguity: emit `define_target` / `confirm_unit_and_horizon` AND `ask_owner` (`{field, question, blocking: true}`) ONLY when a key in `known_facts` is explicitly `null` (`None`) OR the brief explicitly poses competing unresolved targets/units. If `target` is already named and a key like `horizon_days` is simply absent from `known_facts` (not `null`), NEVER call `ask_owner` or `confirm_unit_and_horizon`.
- Hygiene: include `check_join_key` on child tables, `grouped_split` for repeated entities, `time_split_check` / `feature_hypothesis` for timestamps/cutoffs, `leakage_probe` / `profile_column` on post-outcome/snapshot/ID columns, `calibration` / `threshold_for_cost` for rare events or asymmetric cost/capacity, and `segment_subproblem` per family + `error_analysis` for composite problems.
- Order: NEVER place analysis/split/leakage/calibration tasks after `holdout` or `final`.
- Trap slugs: use short descriptive snake_case names (e.g. `outcome_defined_over_events_not_column`, `post_outcome_<col>`, `post_cutoff_<table_or_col>`, `end_of_period_<col>`, `time_drift_random_split_leak`, `cutoff_horizon_overlap_with_holdout`, `one_to_many_join_duplication`, `repeated_entity_leak`, `high_churn_is_not_high_uplift`, `observational_treatment_bias`, `extreme_class_imbalance`, `rare_positive_accuracy_trap`, `capacity_constrained_threshold`, `asymmetric_cost`, `ambiguous_unit`, `missing_horizon`).

Output ONLY valid JSON:
{
  "version": "tp08.v1",
  "decomposition": {
    "schema_version": "dclab_core.problem_decomposition.v1",
    "decision": "...",
    "target": null or "...",
    "unit": null or "...",
    "horizon_days": null or integer,
    "success_metric": "...",
    "constraints": [],
    "subproblems": [{"id": "sub_1", "kind": "...", "objective": "...", "target": null or "..."}],
    "risks": [{"trap_kind": "...", "table": null or "...", "column": null or "...", "evidence_citation": "...", "recommended_task": "..."}],
    "unknowns": [{"field": "target|unit|horizon|join_key|treatment|metric|cost_ratio|other", "question": "...", "blocking": true, "resolution_task": "ask_owner|define_target|confirm_unit_and_horizon|profile_column|check_join_key"}],
    "source": "model"
  },
  "calls": [
    {"task_type": "...", "params": {...}, "reason": "... [PIT-...]", "citations": ["..."]}
  ],
  "caught_traps": ["..."]
}
