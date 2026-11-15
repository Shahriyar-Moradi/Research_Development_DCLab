<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-01D — Deterministic solution-workflow-to-task adapter

## Codex prompt

```text
Goal
Convert one validated binary-classification SolutionWorkflow into an
accurate, non-executing TaskGraph. Prove real integration without any
new ML computation or new project writes.

Prerequisites
TP-01A capability map, TP-01B contracts and TP-01C validator.

Suggested file
- dclab_rnd/tasks/adapter.py (and __init__.py)
- tests/test_task_adapter.py
- fixture based on the existing Telco churn or another verified
  binary-classification sample in the repository.

Inputs
- Validated existing SolutionWorkflow, source SolutionPlan,
  current project/data/solution metadata and, when applicable,
  verified capability snapshot from the live registry.
- Return TaskGraph + structured mapping report, no side effects.

Approach
1. Build a deterministic mapping from existing Step.id, Step.block,
   Step.kind, Step.inputs/outputs, decision/check/gate and required
   family steps to group/action/gate/check nodes.
2. Tie each actually runnable unit to an EXISTING tool or stage. For
   example the stage owner may be run_stage(data, leakage, features,
   models, final) but resolve the exact schema from the registry.
   Treat split locking as existing guarded engine behavior, not a
   second 'lock_split' operation. Use the precise actual call mapping.
3. Represent many-to-one coverage when one studio engine run
   implements multiple abstract workflow steps. One physical
   side effect = one action task with several source step IDs.
4. If a required capability is missing, represent a blocked task
   naming why; do not invent tools, mark complete, or guess data.
5. Carry artifact lineage and stable hashes, not full table rows.
6. Do not open holdout, fit, query raw values, or set approvals.
7. Use deterministic topological order for v1, with no concurrency.
8. Keep Programme execution as an atomic existing capability when
   implementing complex cases later. Do not decompose its leaves into
   new independently scheduled writes here.

Positive scenario
- Generate a valid current classification workflow from the
  existing core builder.
- Run the adapter twice and compare semantic fingerprints.
- Validate that mandatory steps, guards and gates are represented.
- Show the resulting task tree and mapping without any execution.

Negative scenarios
- Missing dataset/solution or unknown family blocks appropriately.
- An unconfirmed label/outcome never becomes a runnable modeling task.
- A planted invalid workflow is refused before adaptation.
- No phantom registered tools; no duplicate final-stage action.
- Changing the data or solution changes the relevant source hash.
- Do not mutate the original workflow or project.

Add a simple read-only CLI/developer inspection path ONLY if there is
already an appropriate pattern. Otherwise keep the adapter callable in
Python and exercised by tests. Do not add product API or UI here.

Deliver example JSON in tests/fixtures or docs, source mapping summary,
tests, exact commands and output. No runtime defaults changed.
```
