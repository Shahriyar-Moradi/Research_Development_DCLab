# Task Planning v2 Release Evaluation Report

- **Model:** `qwen/qwen3.8-flash`
- **Authorized evaluation cap:** `$3.50` USD
- **Measured live cost:** `$0.228419` USD across `239` requests (`847020` tokens)
- **Model graphs admitted:** `36 / 36` execution cases
- **Gate Verdict:** **`FAIL`** (`promoted=False`)
- **Reasons:** `gain_or_resource_threshold_not_met`

## Paired Metrics (Arm C vs Arm A)

| Metric | Difference (C - A) | 95% Bootstrap CI | Matched Rows |
|---|---:|---|---:|
| correct | 0.2708 | [0.1458, 0.3958] | 48 |
| completed | 0.2292 | [0.125, 0.3542] | 48 |
| recovered | 0.7857 | [0.5714, 1.0] | 14 |
| proper_abstention | 0.1538 | [0.0, 0.3846] | 13 |
| seconds | 28.2774 | [0.9099, 57.8413] | 48 |
| compute_seconds | -0.478 | [-0.7617, -0.2014] | 48 |
| cost_eur | -0.0048 | [-0.0057, -0.0038] | 48 |
| extra_prompts | -4.9792 | [-5.9375, -3.9792] | 48 |

- Resource checks: `{'cost_and_prompts': False, 'latency': False}`
- Critical safety sums: `{'A': {'unauthorized_tool_calls': 0, 'false_completion': 0, 'holdout_violations': 0, 'leakage_bypasses': 0, 'repeated_non_idempotent_actions': 0}, 'B': {'unauthorized_tool_calls': 0, 'false_completion': 0, 'holdout_violations': 0, 'leakage_bypasses': 0, 'repeated_non_idempotent_actions': 0}, 'C': {'unauthorized_tool_calls': 0, 'false_completion': 0, 'holdout_violations': 0, 'leakage_bypasses': 0, 'repeated_non_idempotent_actions': 0}}`
