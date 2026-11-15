Person's problem description (no data uploaded yet):
${problem}

Initial family reading: ${family}
The person's own figures (state a number only if it is one of these):
${figures}

Initial virtual schema floor (you may refine columns, roles, grain, and leak_suspects to fit this exact domain):
${schema}

Allowed workflow step IDs: WF-01..WF-10. Allowed tools: describe_data, check_leakage, query, run_stage, get_results, ask_project.
Remember: holdout (`final` stage / WF-09) is read once at the very end, AFTER the `leakage` audit stage (WF-05); every preprocessing step that learns from data must set `inside_folds: true`.
Keep lists concise (<= 8 columns, <= 4 EDA checks, <= 3 preprocessing steps, <= 6 solution_workflow steps) so the full valid JSON object fits comfortably.
