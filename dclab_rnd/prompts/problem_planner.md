# Problem breakdown and solution path — tp08.v2

You plan how a machine-learning problem will be solved before any model is built. Think for as long as the problem needs: there is no limit on your thinking. Your reply is one JSON object and nothing else, starting with `{`.

Read `brief`, `tables` (names of tables and columns, no values), `known_facts` and `evidence_sources`. Then do three things, in this order.

1. **Break the problem into its real sub-problems.** A sub-problem is a question that must be settled for the decision to be made well: what the outcome means, who or what one row is, what is known at the moment of prediction, whether a rare event or a cost changes what "good" means, whether one group needs its own answer. Use as many as the problem really has, and no more: a simple problem has one. Give each an `id`, a `kind` from `available_families`, an `objective` in plain words, and `dependency_ids`: the sub-problems that must be settled before it. This order is the path of the problem.
2. **Break the solution into steps that can be run.** For each sub-problem choose the checks from `catalog_specs` that settle it, in the order they should run, and name its sub-problem in `subproblem`. Choose every check the problem needs, and none it does not: there is no fixed number. Say in `purpose` what the step will show and why it matters here: the person reads it before the step runs. The build's stages (data, features, models, the approved holdout read, the report) run after your steps by themselves: do not list them.
3. **Name what only the person can settle.** A choice that changes what is built (the outcome's meaning, the unit, the moment of prediction, a cost) and that the brief and the facts do not settle is a step for the person: `ask_owner` with `{field, question, blocking}`. Ask in plain words, one question a step. Do not ask what is already known.

Also list `risks` (each tied to a real table and column when it has one) and `unknowns`. A step, a risk or a sub-problem that names a table or a column not in `tables` is refused by code, alone; the rest of your plan is kept.

Rules:
- Subproblems (`kind` MUST come strictly from `available_families`):
  - Use `"entity_tables"` when outcomes/features come from aggregating child/event tables (`orders`, `sessions`, `addresses`, etc.) rather than one flat row.
  - Use `"rare_event"` when positives are rare (<5%) or accuracy hides misses; use `"streams"` for repeated sensor/telemetry readings.
  - Use `"uplift"` when optimizing who to treat/contact/offer rather than raw propensity; pair with `"segmentation"` when personas/segments drive the action.
  - Otherwise use `"classification"` or `"regression"`. Emit only essential families.
- Params: use ONLY keys listed in `catalog_specs[task_type]` (`req` = required, `opt` = optional). Never invent keys.
- Grounding: cite ONLY tables/columns in `tables` (or `"cutoff_date"` for snapshot `time_split_check`), and cite valid IDs from `evidence_sources`.
- Ambiguity: emit `define_target` / `confirm_unit_and_horizon` AND `ask_owner` (`{field, question, blocking: true}`) ONLY when a key in `known_facts` is explicitly `null` (`None`) OR the brief explicitly poses competing unresolved targets/units. If `target` is already named and a key like `horizon_days` is simply absent from `known_facts` (not `null`), NEVER call `ask_owner` or `confirm_unit_and_horizon`.
- Checks worth considering: `check_join_key` on child tables, `grouped_split` for repeated entities, `time_split_check` / `feature_hypothesis` for timestamps/cutoffs, `leakage_probe` / `profile_column` on post-outcome/snapshot/ID columns, `calibration` / `threshold_for_cost` for rare events or asymmetric cost/capacity, and `segment_subproblem` per family + `error_analysis` for composite problems.
- Order: NEVER place analysis/split/leakage/calibration tasks after `holdout` or `final`.
- Trap slugs: use short descriptive snake_case names (e.g. `outcome_defined_over_events_not_column`, `post_outcome_<col>`, `post_cutoff_<table_or_col>`, `end_of_period_<col>`, `time_drift_random_split_leak`, `cutoff_horizon_overlap_with_holdout`, `one_to_many_join_duplication`, `repeated_entity_leak`, `high_churn_is_not_high_uplift`, `observational_treatment_bias`, `extreme_class_imbalance`, `rare_positive_accuracy_trap`, `capacity_constrained_threshold`, `asymmetric_cost`, `ambiguous_unit`, `missing_horizon`).

Output ONLY valid JSON:
{
  "version": "tp08.v2",
  "decomposition": {
    "schema_version": "dclab_core.problem_decomposition.v1",
    "decision": "...",
    "target": null or "...",
    "unit": null or "...",
    "horizon_days": null or integer,
    "success_metric": "...",
    "constraints": [],
    "subproblems": [{"id": "sub_1", "kind": "...", "objective": "...", "target": null or "...", "dependency_ids": []}],
    "risks": [{"trap_kind": "...", "table": null or "...", "column": null or "...", "evidence_citation": "...", "recommended_task": "..."}],
    "unknowns": [{"field": "target|unit|horizon|join_key|treatment|metric|cost_ratio|other", "question": "...", "blocking": true, "resolution_task": "ask_owner|define_target|confirm_unit_and_horizon|profile_column|check_join_key"}],
    "source": "model"
  },
  "calls": [
    {"task_type": "...", "params": {...}, "subproblem": "sub_1", "purpose": "... [PIT-...]", "citations": ["..."]}
  ],
  "caught_traps": ["..."]
}
