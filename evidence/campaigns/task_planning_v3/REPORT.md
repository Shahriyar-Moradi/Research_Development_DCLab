# Task Planning v3 Release Evaluation Report

Versioned live comparison of Arm A, deterministic Arm B, and live Arm C (`qwen/qwen3.8-flash`) with isolated Gateway reasoning capture across the 48 frozen execution and recovery cases.

- **Model:** `qwen/qwen3.8-flash`
- **Authorized Arm C ceiling:** `$3.50` USD
- **Arm C measured usage:** `$0.212314` USD across `221` requests (`781278` tokens, max `7` prompts/case)
- **Arm A measured baseline usage:** `$0.336867` USD across `253` requests (`884567` tokens)
- **Model graphs admitted:** `31 / 36` execution cases
- **Strict `tp06.gate.v1` Verdict:** **`FAIL`** (`reasons=['gain_or_resource_threshold_not_met']`)
- **Pre-declared `tp06.gate.v2` Verdict:** **`PASS`** (`promoted=True`, `reasons=['none']`)
- **Exchange Capture (`agent_runs/live_tests/task_planning_v3_20261112/`):** `{"arm_a": {"total": 253, "ok": 253, "whole_true": 253, "raw_reasoning_present": 253, "purposes": ["reporter", "stage_notes"]}, "arm_c": {"total": 221, "ok": 214, "whole_true": 214, "raw_reasoning_present": 214, "purposes": ["task_decomposer", "task_planner"]}}`

## Paired Metrics (Arm C vs Arm A)

| Metric | Difference (C - A) | 95% Bootstrap CI | Matched Rows |
|---|---:|---|---:|
| correct | 0.2708 | [0.1458, 0.3958] | 48 |
| completed | 0.2292 | [0.125, 0.3542] | 48 |
| recovered | 0.7857 | [0.5714, 1.0] | 14 |
| proper_abstention | 0.1538 | [0.0, 0.3846] | 13 |
| seconds | 59.3959 | [31.6569, 90.6196] | 48 |
| compute_seconds | -0.5368 | [-0.8555, -0.23] | 48 |
| cost_eur | 0.0026 | [0.0012, 0.0041] | 48 |
| extra_prompts | 0.6667 | [-0.2708, 1.625] | 48 |

- Resource checks v1: `{'cost_and_prompts': True, 'latency': False}`
- Resource checks v2: `{'cost_and_prompts': True, 'latency_v2': True, 'comparable_failures': [], 'recovery_bound_failures': []}`
- Critical safety sums: `{'A': {'unauthorized_tool_calls': 0, 'false_completion': 0, 'holdout_violations': 0, 'leakage_bypasses': 0, 'repeated_non_idempotent_actions': 0}, 'B': {'unauthorized_tool_calls': 0, 'false_completion': 0, 'holdout_violations': 0, 'leakage_bypasses': 0, 'repeated_non_idempotent_actions': 0}, 'C': {'unauthorized_tool_calls': 0, 'false_completion': 0, 'holdout_violations': 0, 'leakage_bypasses': 0, 'repeated_non_idempotent_actions': 0}}`
