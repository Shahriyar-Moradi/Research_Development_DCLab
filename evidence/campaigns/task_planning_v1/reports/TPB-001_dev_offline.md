# TP-06 DEV task planning comparison

FAIL

Invented DEV cases only. These measurements do not establish production readiness.

| Arm | Judged | Correct | Verified completion | Safety failures |
|---|---:|---:|---:|---:|
| A | 48 | 40 | 13 | 0 |
| B | 47 | 44 | 14 | 0 |
| C | 0 | 0 | 0 | not measured |

The C diagnostic did not establish its refusal/capture protocol; inspect its recorded failures.
The scripted proposal probe does not measure model quality or a complete C execution.

## Per-stratum outcomes

| Stratum | A correct | B correct | C judged |
|---|---:|---:|---:|
| clear_model | 3/3 | 3/3 | 0 |
| ambiguous_intent | 0/3 | 0/3 | 0 |
| explicit_home_chip | 3/3 | 3/3 | 0 |
| churn_binary | 1/3 | 2/3 | 0 |
| segmentation_reporting_ambiguity | 3/3 | 3/3 | 0 |
| multi_table_fraud_dependency | 0/3 | 3/3 | 0 |
| missing_label | 3/3 | 3/3 | 0 |
| incomplete_data | 3/3 | 3/3 | 0 |
| contradictory_request | 3/3 | 3/3 | 0 |
| unsupported_approach | 3/3 | 3/3 | 0 |
| human_approval | 3/3 | 3/3 | 0 |
| holdout_already_used | 3/3 | 3/3 | 0 |
| guard_denied | 3/3 | 3/3 | 0 |
| transient_failure | 3/3 | 3/3 | 0 |
| stale_input | 3/3 | 3/3 | 0 |
| resume_retry | 3/3 | 3/3 | 0 |

## Paired uncertainty

Seed repetitions are clustered by authored case; they do not increase the independent sample size.

```json
{
  "B": {
    "correct": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.8438,
      "mean_b": 0.9375,
      "difference": 0.0938,
      "interval": [
        0.0,
        0.2188
      ],
      "wins_b": 2,
      "losses_b": 0,
      "ties": 14,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "completed": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.2812,
      "mean_b": 0.3125,
      "difference": 0.0312,
      "interval": [
        0.0,
        0.0938
      ],
      "wins_b": 1,
      "losses_b": 0,
      "ties": 15,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "recovered": {
      "n": 2,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 1.0,
      "mean_b": 1.0,
      "difference": 0.0,
      "interval": [
        0.0,
        0.0
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 2,
      "higher_is_better": true,
      "mcnemar_p": 1.0,
      "matched_case_seed_rows": 6,
      "left_out_a": 42,
      "left_out_b": 42
    },
    "proper_abstention": {
      "n": 10,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.8,
      "mean_b": 0.9,
      "difference": 0.1,
      "interval": [
        0.0,
        0.3
      ],
      "wins_b": 1,
      "losses_b": 0,
      "ties": 9,
      "higher_is_better": true,
      "mcnemar_p": 1.0,
      "matched_case_seed_rows": 30,
      "left_out_a": 18,
      "left_out_b": 18
    },
    "task_coverage_rate": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "valid_graph_rate": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "valid_plan_rate": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "seconds": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.2442,
      "mean_b": 0.4039,
      "difference": -0.1596,
      "interval": [
        -0.2915,
        -0.0507
      ],
      "wins_b": 5,
      "losses_b": 11,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "compute_seconds": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.2355,
      "mean_b": 0.3976,
      "difference": -0.1621,
      "interval": [
        -0.2886,
        -0.0582
      ],
      "wins_b": 5,
      "losses_b": 11,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "tool_calls": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 1.9688,
      "mean_b": 2.3125,
      "difference": -0.3438,
      "interval": [
        -0.9062,
        0.0625
      ],
      "wins_b": 2,
      "losses_b": 5,
      "ties": 9,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "confirmations": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.3438,
      "mean_b": 0.375,
      "difference": -0.0312,
      "interval": [
        -0.0938,
        0.0
      ],
      "wins_b": 0,
      "losses_b": 1,
      "ties": 15,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "tokens": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.0,
      "mean_b": 0.0,
      "difference": 0.0,
      "interval": [
        0.0,
        0.0
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 16,
      "higher_is_better": false,
      "mcnemar_p": 1.0,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "cost_eur": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.0,
      "mean_b": 0.0,
      "difference": 0.0,
      "interval": [
        0.0,
        0.0
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 16,
      "higher_is_better": false,
      "mcnemar_p": 1.0,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "extra_prompts": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.0,
      "mean_b": 0.0,
      "difference": 0.0,
      "interval": [
        0.0,
        0.0
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 16,
      "higher_is_better": false,
      "mcnemar_p": 1.0,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    },
    "reexecution_avoided": {
      "n": 16,
      "only_a": 0,
      "only_b": 0,
      "mean_a": 0.125,
      "mean_b": 0.125,
      "difference": 0.0,
      "interval": [
        0.0,
        0.0
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 16,
      "higher_is_better": true,
      "mcnemar_p": 1.0,
      "matched_case_seed_rows": 47,
      "left_out_a": 1,
      "left_out_b": 1
    }
  },
  "C": {
    "correct": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "completed": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "recovered": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "proper_abstention": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "task_coverage_rate": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "valid_graph_rate": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "valid_plan_rate": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "seconds": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "compute_seconds": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "tool_calls": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "confirmations": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "tokens": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "cost_eur": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "extra_prompts": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": false,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    },
    "reexecution_avoided": {
      "n": 0,
      "only_a": 0,
      "only_b": 0,
      "mean_a": null,
      "mean_b": null,
      "difference": null,
      "interval": [
        null,
        null
      ],
      "wins_b": 0,
      "losses_b": 0,
      "ties": 0,
      "higher_is_better": true,
      "mcnemar_p": null,
      "matched_case_seed_rows": 0,
      "left_out_a": 48,
      "left_out_b": 48
    }
  }
}
```

## Limits

- Model-assisted execution is unavailable and unscored; no live-model request was sent.
- Admission and routing strata are boundary checks, not completed scientific workflows.
- Independent completion/recovery sample requirements are intentionally larger than this DEV smoke.
- No defaults change. No sealed data, real-user outcome or general production claim is included.

Reasons: evaluation_runtime_error, incomplete_B, incomplete_C, insufficient_independent_completion_or_recovery_cases, model_graph_not_admitted_by_runtime, no_configured_model_evidence, probe_summary_text
