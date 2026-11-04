# TP-01C — pure graph and plan validation

Status: **complete**, verified on 2026-11-04. All TP-01C acceptance criteria and the applicable package gate pass.
Library version: **0.157.0**. TaskGraph and TaskExecutionPlan remain **v1**.

## Scope and architectural decisions

TP-01C adds deterministic checks to the proposal contracts from TP-01B. It neither builds a graph from a workflow
nor runs a task. Existing SolutionWorkflow and SolutionMap validators are called before source coverage is
accepted. The original studio graph, engine, programme, registry, jobs and storage remain the execution owners.
No model call, dependency, runtime import, persisted task store, Home route or default setting is added.

Companion modules keep canonical identities and graph semantics separate from frozen document shapes:

- `dclab_core/task_fingerprints.py`: graph, plan, source workflow and capability identities.
- `dclab_core/task_validation.py`: graph admission, sequential readiness and execution-plan checks.
- `tests/task_validation_cases.py`: handwritten metadata fixtures, not a workflow adapter.
- `tests/test_task_validation.py`: positive, negative, purity and review regression tests.

Existing `solution_workflow.validate` checks mandatory workflow content, family templates, criteria and order.
Existing `solution_map.validate` checks same-split composition, deferred leaf scoring and one composition holdout.
The existing SolutionMap fingerprint remains authoritative for a typed map reference. The new full workflow hash
separately identifies its ordered workflows/composition. Mutable Programme results never define graph identity.
The main-product adapter inventory has no task fields, so its library-only conformance declarations remain valid.

## Public functions

```python
from dclab_core.task_fingerprints import (
    graph_fingerprint, plan_fingerprint, workflow_fingerprint, capability_fingerprint,
)
from dclab_core.task_validation import validate_graph, ready_tasks, validate_execution_plan

issues = validate_graph(
    graph, source_workflow=workflow,
    known_capabilities=capabilities, external_artifacts=current_artifact_ids,
)
next_action_ids = ready_tasks(graph, observed_state, source_workflow=workflow)
issues = validate_execution_plan(
    plan, graph, capabilities, source_workflow=workflow, observed_state=observed_state,
)
```

Use `source_solution_map=solution_map` instead of `source_workflow` for a complex programme. Exactly one admitted
source is required. Missing/malformed sources are refused. `validate_graph` and `validate_execution_plan` return
sorted, deduplicated actionable issues, capped at 100. They never repair documents. Readiness raises ValueError
for invalid/stale state; it returns an empty list when waiting, blocked, finished or in flight. An empty list is not
a completion verdict. A valid non-executing `noop`/`blocked` plan is not a verified outcome.

Without a capability snapshot, graph validation checks structure, sources and recognized source-bound effects.
Full capability/effect admission requires the snapshot. Readiness and executing-plan validation require it.
Capabilities and observed state are privately copied through JSON; live callbacks/runtime objects are refused.

## Host snapshot format and trust boundary

These dictionaries are function inputs, not new persisted/versioned contracts. They must be made by trusted host
code from current authoritative records. A model must never provide them. Core cannot authenticate a receipt or
prove an approval happened merely because a dictionary says so. Omit raw table values, secrets, prompts and model
text. Only safe primitive control metadata, opaque references, types and fingerprints belong here.

`known_capabilities` maps tool name to a descriptor with `name`, `scope`, `effect` (`read` or `write`), `move`,
`allowed`, `replay_safe` and the registry's JSON `parameters` schema. The host computes each descriptor identity
with `capability_fingerprint`. Task/plan `schema_fingerprint` binds this entire descriptor, including permissions
and replay policy, not just parameter names. Guarded run_stage/run_programme/export_notebook bindings must retain
their actual write effects and moves. A snapshot grants no dispatch permission.

`observed_state` supplies:

| Key | Required meaning |
|---|---|
| `graph_id`, `graph_revision`, `graph_fingerprint` | Exact current admitted graph identity; revision is an integer, not a boolean. |
| `project_id`, `solution_fingerprint`, `workflow_fingerprint` | Current project and source identities. |
| `capabilities` | Current JSON capability descriptor mapping used by readiness. |
| `artifacts` | Mapping of artifact ID to `fingerprint` (full SHA-256), `value_type`, and, for produced artifacts, `producer_task`. |
| `approval` | Current approved source record, as specified below. |
| `tasks` | Current host attempt/verifier observations, as specified below. |
| `holdout_approval`, `holdout_uses` | Current human approval and exact unused counter before a final/programme action. |

Artifact type is `artifact`, `string`, `integer`, `number` or `boolean`. An `artifact` has no embedded value. Safe
primitive controls, such as project ID or stage name, may include `value`; its type and canonical metadata hash
must match. Arbitrary table rows are not supported. Runtime registry schema validation still owns actual arguments.

An approval has strict `approved: true`, `project_id`, `solution_fingerprint`, `workflow_fingerprint`, opaque
`reference_id`, owner `fingerprint`, `sources` and `inputs`. Its sources exactly cover graph source references
except the approval reference itself. Inputs bind dataset IDs and any further approved artifacts to their current
hashes. The normal approval must match the graph's typed approval source reference. A holdout approval also needs
`used: false`, with `holdout_uses: 0`. Its input bindings must cover the human gate's prerequisites. After final
consumption, the historical gate receipt remains valid, but no unused final permission can be inferred.

Each `tasks[task_id]` is either pending, in flight, blocked/failed/stale, or independently verified. A verified
receipt needs `status: verified`, an opaque `verification_ref`, exact `graph_revision`, `graph_fingerprint`,
`project_id`, and `inputs`/`outputs` mappings of artifact IDs to current hashes. Inputs cover all task needs;
outputs exactly match declared artifacts and their producer task. All dependencies need matching verified receipts.
A human gate also needs `approval_ref: {id, fingerprint}` matching the host's holdout approval. Boolean flags,
`completed` claims, missing/unknown outcomes, stale inputs and unverified outputs cannot unlock another action.

Absent/pending attempts mean the host has established that execution has not begun. Running, failed, stale,
ambiguous or unreconciled attempts stop progression. Existing output artifacts without a verified receipt are
refused. Recovery and durable crash reconciliation remain TP-04/TP-05 responsibilities; do not infer that a lost
receipt makes a write safe to repeat.

## Safety mapping and readiness

Parent and dependency cycles, dangling/self/invalid dependencies, inconsistent grouping/dependency edges, duplicate
IDs, executable groups, ambiguous artifact producers and unavailable artifacts are rejected. A needed artifact
must be supplied externally or produced by a valid ancestor. Source steps must be covered exactly once by an
action, gate, evidence node or explicit blocker; groups cannot silently cover them. Missing source steps refuse
admission. A blocked source is preserved and gives no ready action.

For a flat workflow, dependency reachability must preserve holdout locking, training-only analysis, leakage audit,
validation selection, human approval, one final evaluation and reporting. All WF-10 post-final evidence must wait
for final scoring. A known engine operation may cover several source steps when the existing engine owns that
atomic boundary: lock and training-only analysis inside data; validation-selected threshold or post-final evidence
inside final. This does not permit a second final action. Features must follow leakage. A stage plan must resolve
its fingerprinted stage metadata to the operation declared by that task; `run_all` and deferred guesses are refused.

Duplicate non-idempotent stage effects and programme runs are refused. Extra plan actions must be explicitly
replay-safe reads; the primary capability occurs exactly once. Non-idempotent actions cannot carry automatic retry
allowances. Runtime parameter names and required arguments are checked against the supplied schema; full JSON
Schema value validation remains the registry's job.

Complex leaf workflows reuse unqualified source step IDs. Flattening them would invent ambiguous coverage and
separate holdout reads. Until a later scoped representation is reviewed, the safe supported form is one map-level
`run_programme` action behind an explicit human gate. Its map fingerprint, approved source list, composition and
holdout policy remain bound. Leaf execution stays entirely within the existing programme owner. This is a
documented conservative boundary, not a second programme executor.

Among eligible actions, readiness chooses the lexicographically first ID and returns at most one. Gates/evidence
nodes wait for host verification. No group, validator call or plan status marks a task complete. Current project,
approved sources, prerequisite receipts and input hashes are checked again by plan validation.

## Canonical identity rules

Hashes are full lowercase SHA-256 of UTF-8 JSON with sorted object keys, compact separators, preserved Unicode and
non-finite numbers refused. Helpers validate document shapes on private copies. A builder may omit the hash before
computing it. A persisted hash mismatch is explicit refusal; validation never overwrites it.

| Identity | Retained content | Canonical ordering / excluded content |
|---|---|---|
| Graph | ID, revision, objective, source identities, node semantics, edges, needs, outputs and acceptance checks | Sort nodes by ID, edges by kind/source/target, sources by kind/ID, set-like references and checks. Exclude hash and creation provenance. |
| Plan | Graph/task/plan IDs and revisions, ordered actions, typed refs, inputs/outputs, checks, budgets, stop/retry policy and reason | Keep action order. Sort input/output IDs, capability scope/name, argument names and check/evidence refs. Exclude hash, creation provenance and transient status. Status is checked separately. |
| Workflow | Complete ordered SolutionWorkflow content; for a map, ordered block workflows/deferrals, composition steps and holdout policy | Preserve source step order and meaningful titles/criteria. Map's existing owner fingerprint is additionally checked as its own source reference. |
| Capability | Name, scope, permission, effect, move, replay policy and complete parameter schema | Sort object keys; preserve arrays and every schema property, including properties named title or description. |

A graph revision change produces a new identity and invalidates old plans/receipts. An input artifact change is
independently detected from current metadata even when the graph revision stays the same. Neither change is
silently rebased. Benign dictionary and set-like graph ordering, JSON roundtrips and restarts preserve identity.
File and PostgreSQL serialization have no backend-specific fields in these identities. Backend roundtrip behavior
is tested as JSON serialization here; this package adds no task persistence backend.

## Pure checks versus execution-time checks

| Condition | Pure planning check | Authoritative runtime must recheck |
|---|---|---|
| Graph/source identity | Declared hash, revision, source coverage, source validator and approved source bindings | Reload current sources and compare inside the dispatch boundary; preserve older revisions. |
| Tools and arguments | Descriptor identity, allowed scope, required/name refs, guarded source operation | Real registry membership, principal authorization, full argument schema and scope guard. |
| Inputs and evidence | Current supplied hashes/types, dependency receipts, producer ownership, expected plan inputs | Read through authoritative storage, verify actual bytes/lineage and authenticate verifier records. |
| Human approval | Current metadata binding, gate receipt, declared unused permission | Authenticate human record and atomically consume the current permission. |
| Scientific safety | Source ordering, no duplicate scheduled effects, atomic programme boundary | Original splits, leakage rules, selection, programme guards and holdout-once counter. |
| Attempts and budgets | Refuse in-flight/failed/unknown/stale or orphan outputs; bounded plan policy and no unsafe retries | Durable attempt state, idempotency/reconciliation, concurrency lock, Stop, spend and time/tool/model caps. |

The state may change between validation and execution. Pure validation cannot close this time gap. A later
dispatcher must recheck under its authoritative locking/transaction boundary and fail closed on ambiguity.

## Verification and acceptance

Initial tests failed before implementation. Additional tests reproduced and fixed approval/source binding gaps,
failed-attempt retry, duplicate effects, weak evidence flags and live callback acceptance. Review caught schema
properties named description being dropped from capability identity, training analysis before holdout locking,
and WF-10 evidence being allowed too early. Each has a passing regression. Existing outputs without receipts also
have a red-first regression and now refuse scheduling.

Current focused result: **63 tests passed**, including **40 TP-01C tests** and **23 TP-01B tests**. The broader focused
selection ran **187 tests across 10 files in 23.4 seconds**: 186 passed, one opt-in wheel unittest skipped. The real
standalone wheel check passed separately on isolated Python 3.12.13 with core 0.157.0, four direct dependencies and
no unexpected distributions. Canonical helper roundtrips and source refusal were exercised inside the wheel.

The first focused compatibility run failed one corpus freshness test: generated `train.alpaca.jsonl` contained
16,384 NUL bytes. The writer that introduced them was not established. Rebuilding with `make sft-dev` removed the
damage; the complete focused rerun passed. This was a generated-file problem, not a task validator test failure.
No generated file was repaired by hand. The data catalog was regenerated for the new core version. All **262**
schema files were byte-identical after `make core-schemas`; contract registration remains **107**. All **10**
`make rd-gates` preflight checks passed.

Final `make check-quick JOBS=3` result: **passed**.

- File backend: **5,265 tests**, 450 files / 679 execution units, 2,288.5 seconds, **11 skips**, no failures.
- PostgreSQL: **827 tests**, 58 files / 112 units, 611.2 seconds, **2 skips**, no failures.
- All **10** evidence/freshness gates and **5** product flows passed: upload, upload-team, no-data, synthetic, log-file.
- All 40 new semantic tests ran successfully; none skipped. Final adversarial source review: **PASS**.

The file pass skips five PostgreSQL-specific cases, three notebook cases without their frozen manifest, two NOOA
cases belonging to the isolated agent environment and the opt-in wheel unittest. The PostgreSQL pass covers its
backend-specific cases and skips two file-specific cases. The opt-in wheel skip is covered by the actual isolated
wheel run. The notebook/NOOA skips remain existing coverage limits, not passed tests. No live-model or sealed
benchmark run was performed. No source/runtime changes were made while the full gate ran; completion records and
their generated development corpus are refreshed afterward and checked separately.

| TP-01C acceptance | Status |
|---|---|
| Graph references/cycles/groups/artifact satisfiability | Pass |
| Mandatory source coverage and temporal safety | Pass |
| Approval, capability scope, one effect and stale input refusal | Pass |
| Deterministic sequential readiness without implicit completion | Pass |
| Canonical identities and explicit stale-hash refusal | Pass |
| Revision/input distinction; no mutation or host imports | Pass |
| Required adversarial and valid binary cases | Pass |
| Runtime boundary and conservative programme mapping documented | Done |
| Unchanged schemas, installed wheel and adapter compatibility | Pass |
| Full applicable package gate | Pass: files, PostgreSQL, gates and flows |
| Final adversarial source review | PASS; no unresolved correctness finding |

TP-01D remains **not started**. Its task is to build the deterministic source-to-graph adapter using these rules,
current registry metadata and existing effect boundaries. Handwritten fixtures must not be promoted into an app
integration. No visible app behavior changes until later packages wire the validated proposals into guarded flows.

## Files in this package

New source and tests: `dclab_core/task_fingerprints.py`, `dclab_core/task_validation.py`,
`tests/task_validation_cases.py`, `tests/test_task_validation.py`.

Updated distribution and verification: `dclab_core/__init__.py`, `dclab_core/pyproject.toml`,
`dclab_core/CHANGELOG.md`, `scripts/core_wheel_check.py`. Generated version catalog:
`docs/DATA_CATALOG.md`, `docs/data_catalog.json` (through `make catalog`).

Documentation and progress: this guide, `dclab_core/README.md`, `docs/guides/CORE_INTEGRATION.md`,
`engineering/planning/README.md`. Development record:
`research/llm-fine-tuning/experiments/dev_trajectories/records/tp-01c.json`; generated corpus files under
`research/llm-fine-tuning/experiments/sft/out_dev/` are rebuilt with `make sft-dev`.

Previously uncommitted TP-01A/TP-01B artifacts are preserved. The task v1 contract modules, their schema registration,
all 262 schemas and main-product adapter conformance declarations are unchanged by TP-01C. No execution owner is
modified. The package handoff proposed `Validate hierarchical task proposals before guarded execution` and left
the changes in the working tree on 2026-11-04.

The package prompt requested a proposed review commit; no actual commit or `make rd-check` result was claimed in
that handoff. On 2026-11-05 the owner explicitly requested committing and pushing all changes, including the
repository agent instructions and reviewer setup. The required commit checks are separate from the package results
above. TP-01D is ready to start in a separate package; it must use current registry metadata and preserve these
conservative execution boundaries.

Commit gate on 2026-11-05: `make rd-check JOBS=3` passed all 10 evidence/freshness gates and ran 5,305 tests across
463 files / 692 units in 2,198.2 seconds, with 14 existing skips and no failures. These are the full-suite results,
separate from the affected-suite package gate above. No new semantic test skipped. The clean-checkout commit gate
is run after creating the commit and before pushing.

The full suite's three additional skips are existing placeholder evaluation tests whose modules are not committed:
replay_compare, system_eval and agentic.trace_eval. They are recorded as coverage gaps, not successful evaluations.
