<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-01C — Deterministic graph validator, planning invariants and fingerprints

## Codex prompt

```text
Goal
Implement pure graph and plan validation with reproducible structural
fingerprints. No scheduler, tool execution or model calls in this step.

Prerequisite
Contracts and schema baselines from TP-01B must be green.

Suggested owner
- dclab_core/task_graph.py and task_execution_plan.py or small
  pure companion modules if necessary.
- tests/test_task_graph.py, tests/test_task_execution_plan.py.

Functions to define after matching repository conventions
- validate_graph(graph, source_workflow=None, source_solution_map=None)
  -> list of actionable problems.
- ready_tasks(graph, observed_state) or equivalent PURE dependency
  traversal. Observed_state is supplied by the host, never fetched.
- graph_fingerprint(graph) and plan_fingerprint(plan), stable hashes
  over canonical semantic content, excluding volatile timestamps,
  transient statuses, raw user data, and the hash field itself.
- validate_execution_plan(plan, graph, known_capabilities) -> issues;
  the core receives a pure capability snapshot, not a runtime registry.

Requirements
1. Reject duplicate IDs, absent parents, parent cycles, dependency
   cycles, self-dependencies, dependencies on unknown/invalid nodes.
2. Reject executable group nodes and dependencies that cannot be
   satisfied with the requested outputs/artifacts.
3. Preserve source-workflow provenance. Mandatory steps must be
   accounted for by action, guard, explicit gate, evidence check or
   blocked node. The adapter may use a justified many-to-one mapping
   but must never silently omit a mandatory safety step.
4. Enforce temporal constraints from existing SolutionWorkflow
   validation, including leakage-before-model-selection, human gate,
   single final holdout read, and reporting after final evaluation.
5. Prohibit model execution intents before approval and tools outside
   the provided capability snapshot/scope. A core validator should
   never invoke a tool or import studio/agents.
6. Prevent multiple nodes from scheduling the same non-idempotent
   engine effect, especially final score and programme run.
7. Deterministic ordering when multiple nodes are ready; no parallel
   execution in v1. No node is marked complete by validation.
8. Clearly distinguish a changed revision from a changed input artifact.
9. Fingerprints must be stable across dictionary insertion order,
   restarts, file vs PostgreSQL storage, and round-trip serialization.
10. Fingerprint mismatch must be explicit and refused; never silently
    rewrite the old graph or plan. Preserve old revision for audit.

Red-team tests
- cycle of length 2 and 3, missing parent, dangling dependency,
  duplicate executable action, final-before-approval, holdout twice,
  missing leakage, cross-scope tool, missing capability, inconsistent
  input artifact reference, stale fingerprint, benign reordering.
- A valid binary case passes and has deterministic topological order.

Document exactly which conditions are enforced by the pure planning
validator and which must be rechecked by the authoritative runtime
at execution time to avoid time-of-check/time-of-use vulnerabilities.

Run focused tests and schema checks, report precise failures.
```
