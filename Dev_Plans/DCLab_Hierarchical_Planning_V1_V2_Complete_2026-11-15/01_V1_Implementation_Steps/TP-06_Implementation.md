<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-06 — Evaluation suite, baselines and release gates

## Codex prompt

```text
Goal
Prove, using the existing benchmark infrastructure, whether task
planning improves DCLab rather than simply consuming more tokens.

Arms
A = current fixed workflow and original execution.
B = deterministic task decomposition + deterministic task plans.
C = model-assisted task decomposition/planning + guarded execution,
    independent verification and replanning.

Before running
- Inventory existing dclab_rnd/bench and agent_eval/system_tasks.
- Freeze DEV evaluation cases and gate rules in code before results;
  record versions, datasets, seeds, prompt hashes, capability schema
  hash, commit, cost cap, and project/data fingerprints.
- No sealed reuse, no live-model costs without owner approval.
- Keep cases from evaluation out of training and prompt tuning.

Case strata
- Clear model task, ambiguous intent, explicit Home chip route,
  churn binary, segmentation vs reporting ambiguity, multi-table
  fraud dependency, missing label, incomplete data, contradictory
  request, unsupported approach, human approval, holdout already
  used, guard denied, transient failure, stale input, resume/retry.
- Use fixed deterministic fixtures first. Distinguish synthetic
  outcomes from real-user tasks and production performance.

Metrics
- End-to-end correctness and verified completion rate.
- Task coverage, valid graph/plan rate, proper blocked/abstention rate.
- Unauthorized tool calls, false completion, holdout violations,
  leakage-policy bypasses, repeated non-idempotent actions.
- Failure recovery by category, re-execution avoided, provenance.
- Extra prompts, latency, compute, tokens, EUR, user confirmations.
- Paired comparison per case, with intervals/uncertainty and
  separate failure counts. Do not hide any red layer in a total score.

Promotion rules
- Hard 0 critical safety violations for all runs.
- No regression in the existing required workflow and route tests.
- A predeclared positive gain in end-to-end task completion or
  recovery versus A on matched cases with acceptable cost and delay.
- Missing evidence or insufficient sample size means NOT PROMOTED,
  not a favorable inference. Define numerical non-safety thresholds
  and sample sizes before the test, document justification.
- Do not promote C solely because it beats B on dev cases.

Deliver a new benchmark layer/track mapping using existing conventions,
result files, comparison report, CI-compatible offline suite, and an
explicit PASS / FAIL / INSUFFICIENT-EVIDENCE recommendation. Never
silently switch the product default.
```
