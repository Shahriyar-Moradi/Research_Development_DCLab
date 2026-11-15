<!-- Extracted verbatim from DCLab_Hierarchical_Planning_Codex_Pack.md, prepared 2026-11-04. Refer to the master pack for shared operating rules and current-HEAD audit instructions. -->

# TP-01B — Minimal typed task contracts and frozen schemas

## Codex prompt

```text
Goal
Add the smallest public, versioned, deterministic contracts needed
for task decomposition and per-task execution planning. Implement
only contracts and schema registration in this package.

Prerequisite
Read TASK_PLANNING_ARCHITECTURE.md and TASK_CAPABILITY_MATRIX.md.
Refuse to proceed if the matrix has unresolved fundamental ownership
or holdout-authorization conflicts. Inspect current dclab_core
contract/versioning/schema generation patterns first.

Suggested files (adapt to actual current conventions)
- dclab_core/task_graph.py
- dclab_core/task_execution_plan.py
- dclab_core/contracts.py (registration, minimal additions)
- dclab_core/schemas/task_graph.json and
  dclab_core/schemas/task_execution_plan.json
- frozen versioned baseline schemas using the current schema command
- dclab_core/README.md, CHANGELOG.md
- tests/test_task_graph.py, tests/test_task_execution_plan.py

TaskGraph v1 should express an immutable, structural plan
- schema_version, graph_id, revision, objective/source references,
  workflow/solution fingerprints, nodes, edges, creation provenance.
- Task node contract with id, parent id (optional), title/objective,
  source_step_ids, kind (group, action, gate, verification, blocked),
  needs, produces, dependencies, acceptance checks, capability binding,
  scope, and blocker reason where unsupported.
- Use stable bounded IDs and known enums when appropriate. Do not
  invent arbitrary executable Python actions or arbitrary URLs.
- Distinguish logical grouping edges from execution-dependency edges.
- Group nodes are not executable. Any kind other than 'action' must
  have clearly specified interpretation and never run arbitrary code.
- A task may be human-gated or check-only without inventing a tool.
- A declared action must point to a capability in existing registry,
  not copy that capability's schema.

TaskExecutionPlan v1
- graph/task/revision identity, immutable plan revision and fingerprint,
  expected input fingerprints and artifact IDs, selected authorized
  capability names, ordered bounded actions, controlled argument refs,
  acceptance/evidence requirements, compute/tool budgets, stop and
  compensation/escalation policy, plan provenance and status.
- Allow deferred arguments only as typed references to already
  declared outputs; do not permit raw model-supplied expressions/eval.
- Plans must support no-op and explicit blocker as distinct from
  successful task completion.
- Choose a minimal model based on measured adapter needs rather than
  fields added just for theoretical completeness.

Schema/version rules
- Reuse dclab_core.contracts.Contract/Versioned/_version and
  dclab_core.contracts.CONTRACTS registration conventions.
- Freeze JSON schemas at v1. Never modify a frozen version after
  publication. Verify unknown fields and missing version are refused.
- Keep core import-safe and without network, agent runtime, database
  access or circular imports. Inspect the installed wheel behavior.
- No task execution, graph traversal or model request here.

Tests
- JSON round-trip, version rejection, unknown field rejection,
  wrong enum/ID rejection, duplicate IDs, empty objective,
  untrusted argument expressions, bounded lists and schema freshness.
- Check that unchanged schemas remain byte-compatible.
- Include a minimal valid example and an unsupported-capability node.
- No network/model calls.

Deliverables
New typed contracts, frozen schemas, source documentation, failing-
before/passing-after tests. Run focused core tests and the applicable
schema/wheel checks. Report the schema diff and compatibility risks.
```
