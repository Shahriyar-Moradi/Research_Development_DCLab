<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-03 — Per-task execution planner and plan gate

## Codex prompt

```text
Goal
Plan HOW to execute one dependency-ready leaf task. Produce a typed
TaskExecutionPlan with authorized calls, concrete expected results,
verification and recovery guidance. Planning never calls a write tool.

Read dclab_rnd/agents/registry.py, intern/tools.py, agents/runtime.py,
models/gateway and the TaskExecutionPlan contract at HEAD.

Design
1. Input is one task plus current project state, declared artifact
   hashes, allowed registry tool schemas for that scope, remaining
   budget and past failures. It cannot see unrelated workspace data.
2. Registry metadata is the single authority for tool names, JSON
   argument schemas, read/write effects, scopes and project moves.
3. Create bounded ordered action intents and explicit artifact
   bindings to earlier outputs. Unknown capabilities are blockers.
4. Validate all static arguments through the existing registry.check
   or a pure schema snapshot. Revalidate resolved arguments immediately
   before runtime execution. Do not execute here.
5. Prohibit side effects from trying to 'preview' a write. Read-only
   tools may be used only when already permitted by agent policy.
6. The output requires preconditions, acceptance checks, budget,
   selected steps, failure classifications and stop/escalation rules.
7. An existing task may have no tool action because it is a gate or
   verification. Never falsely mark it as completed.
8. Scripted fallback produces deterministic plans for existing five
   studio stages. Model planning is behind an off-by-default flag.
9. The existing workflow and programme guards remain authoritative.
10. Track model/tool schema hashes in the plan so stale plans cannot
    run after changed permissions or tool definitions.

Tests
- Good plan for each supported stage, missing required tool args,
  arbitrary tool, wrong scope, write without move, forbidden data,
  missing artifact, invalid binding, stale capability hash, timeout,
  overbudget, model refusal, changed user intent.
- Confirm 0 write-tool calls during planning and no extra holdout
  accesses. Confirm fallback if model is down.

Integrate with the shared harness and gateway, not another agent loop.
Deliver code, tests and a DEV plan validity comparison. No scheduler.
```
