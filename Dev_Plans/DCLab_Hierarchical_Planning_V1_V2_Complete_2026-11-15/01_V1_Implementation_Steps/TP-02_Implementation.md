<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-02 — Bounded model-assisted task decomposition

## Codex prompt

```text
Goal
Add one model-assisted task decomposer that turns an already
validated solution workflow (and its deterministic TP-01 adapter
result) into an improved executable task hierarchy, under deterministic
validation. Preserve the rule-generated graph as fallback.

Prerequisite
TP-01 acceptance report explicitly passes. No default rollout.

Implementation
- Suggested dclab_rnd/tasks/decomposer.py and a narrowly scoped
  prompt in the existing dclab_rnd/prompts/ registration convention.
- Discover whether the existing solution_planner/decomposer roles and
  the gateway can own this role. Prefer extending a measured existing
  role only if its semantics fit; otherwise register one new role with
  harness purpose, prompt hash and model routing. Do not build an
  independent LLM client.
- Input context contains confirmed goal, sanitized data-card metadata,
  validated workflow, task-capability map, budgets, unresolved
  questions, deterministic graph and authorized tool names. Never raw
  rows or untrusted instructions promoted to system policy.
- Model proposes TASK BOUNDARIES/dependencies only; cannot alter
  mandatory solution steps, signed decisions, safety guards or run
  arbitrary tools. Do not let it execute while planning.
- Validate the proposed TaskGraph against TP-01 checks. If invalid,
  retain the deterministic graph and log refusal reasons. A bounded
  one-repair attempt is optional only if measured.
- Limit node count, depth, model calls, tokens, cost, and planning
  time; reuse existing budget conventions. Do not split each tool
  argument or micro-operation into a separate task.
- Require grounding of all proposed subtask objectives in source
  workflow steps, confirmed case facts or registered capabilities.
- Preserve source workflow fingerprints and plan provenance.
- Explicitly report unsupported tasks as blockers with reasons.

Evaluate in scripted/no-model mode using honest cases
- One step/task, multi-step churn, branch dependencies, missing
  capability, contradictory model response, hallucinated tool,
  attempts to remove safety steps, missing required data, and change
  of objective after confirmation.
- Check fallback output equals the deterministic TP-01 graph on
  invalid or unavailable LLM responses.
- Benchmark decomposition coverage and invalid proposal rate against
  deterministic adapter on dev examples.

Deliver code, tests, tracing, a prompt version, docs and a recorded
DEV result. The runtime must not execute decomposed tasks yet.
```
