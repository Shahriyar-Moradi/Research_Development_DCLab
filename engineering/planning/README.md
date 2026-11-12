# README

# DCLab Hierarchical Planning

## Overview

This directory contains the architecture, implementation roadmap, and Codex instructions for DCLab's hierarchical task planning system.

The goal is to extend DCLab's existing planning capabilities so it can decompose validated solutions into executable tasks, plan each task, execute through existing tools, verify results, and recover safely from failures.

This initiative must extend the current architecture rather than introduce a parallel orchestration framework.

## Project goals

- Convert validated solution workflows into hierarchical task graphs.
- Represent task dependencies, expected outputs, and acceptance criteria.
- Generate detailed execution plans for individual tasks.
- Execute tasks through the existing guarded tool registry.
- Verify task completion using actual evidence.
- Support bounded retries and dynamic replanning.
- Preserve completed work when dependencies and inputs remain unchanged.
- Maintain existing leakage, approval, and holdout protections.
- Evaluate the new planner against the existing workflow before rollout.

## Directory structure

```text
engineering/planning/
├── README.md
├── DCLab_Hierarchical_Planning_Codex_Pack.md
├── R_01.txt
├── TP_01A.txt
├── TP_01B.txt
├── TP_01C.txt
├── TP_01D.txt
├── TP_01E.txt
├── TP_02.txt
├── TP_03.txt
├── TP_04.txt
├── TP_05.txt
├── TP_06.txt
├── TP_07.txt
└── TP_01A_EVIDENCE.json  # static source/capability audit, not a runtime contract
```

## Implementation roadmap

| Package | Description | Status |
|---|---|---|
| R-01 | Audit and stabilize Home routing | Pending verification |
| TP-01A | Architecture and capability mapping | Complete — source coverage, 360 focused tests, check-quick and adversarial review passed (2026-11-04) |
| TP-01B | Task contracts and frozen schemas | Complete — core 0.156.0, 23 new tests, schema/wheel/compatibility checks, check-quick and review passed (2026-11-04) |
| TP-01C | Graph validation and fingerprints | Complete — core 0.157.0, 40 new tests, unchanged schemas, wheel/review and full check-quick passed (2026-11-04) |
| TP-01D | Deterministic workflow adapter | Complete — 41 new tests, generated Telco preview, wheel/compatibility and applicable check-quick/review passed; corpus freshness repaired and verified (2026-11-05) |
| TP-01E | Regression tests and foundation acceptance | Complete — all nine criteria satisfied; full check-all (5,348 file / 5,350 PostgreSQL tests), final review and completion corpus checks passed; existing skips documented (2026-11-05) |
| TP-02 | Model-assisted task decomposition | Complete — proposal only; 44 task tests, 9/9 scripted DEV, 175 focused tests and independent review passed; owner-directed focused validation (2026-11-11) |
| TP-03 | Task-level execution planning | Complete — proposal only; 28 planner tests, 125 relevant tests, 24/24 scripted DEV and independent review passed (2026-11-11) |
| TP-04 | Guarded execution and task progression | Complete — explicit opt-in; 74 relevant checks, file/PostgreSQL real-tool and crash/resume tests, independent review passed (2026-11-12) |
| TP-05 | Verification, recovery, and replanning | Implemented — 115 focused checks and isolated wheel passed; final independent review pending after provider usage limit (2026-11-12) |
| TP-06 | Evaluation and release gates | Not started |
| TP-07 | Product integration and rollout | Not started |

Update this table only when implementation and required validation are complete.

## TP-01A review artifacts

- [Architecture and decisions](../../docs/guides/TASK_PLANNING_ARCHITECTURE.md)
- [Workflow and registered tool mapping](../../docs/guides/TASK_CAPABILITY_MATRIX.md)
- [Acceptance, actual verification and next-package handoff](../../docs/guides/TASK_PLANNING_ACCEPTANCE.md)
- [Expanded source audit](TP_01A_EVIDENCE.json): all mandatory family/step pairs and exact registry schemas at the reviewed commit.

The supplied prompts are flat in this directory. References to a `codex/` subdirectory in the original instructions resolve to these files. TP-01A is design only; no runtime behavior changes. R-01 remains independent. TP-01B through TP-01E are complete. TP-02 through TP-04 are complete under the owner-directed focused validation scope. TP-05 is implemented with final independent review pending. TP-06 and TP-07 have not started.

## TP-01B contracts and verification

[Task contracts acceptance](../../docs/guides/TASK_CONTRACTS_ACCEPTANCE.md) records the new v1 contracts, frozen schemas, compatibility checks, review fixes, exact test results and handoff. This package adds typed document validation; graph generation, semantic/hash validation and execution remain later work.

## TP-01C validation and verification

[Task validation acceptance](../../docs/guides/TASK_VALIDATION_ACCEPTANCE.md) records the pure validation APIs,
host metadata requirements, canonical fingerprints, review findings and execution-time checks. TP-01C does not
generate graphs or execute tasks. Verification passed: 63 combined task tests; 187 focused tests (one optional
wheel skip covered by the real wheel check); check-quick with 5,265 file tests (11 skips), 827 PostgreSQL tests
(two skips), all 10 gates and all five product flows. Final review passed. See the acceptance guide for skip reasons
and runtime limits. TP-01D is ready for a separate session.

The owner requested committing and pushing all changes on 2026-11-05, including repository agent instructions.
The required full `make rd-check JOBS=3` passed: 5,305 tests across 463 files, 14 existing skips, no failures, and all
10 evidence/freshness gates. This commit verification is separate from each package's recorded acceptance results.

## TP-01D workflow preview

[Binary workflow adapter acceptance](../../docs/guides/TASK_ADAPTER_ACCEPTANCE.md) records the source/stage mapping,
the metadata-only Python API, the Telco example, unsupported inputs, review fixes and actual verification results.
The adapter generates a task preview only. No product route, UI, workflow execution or approval writer is added.
Verification passed: 41 new adapter tests, 104 combined foundation tests, 249 focused compatibility tests, isolated
core wheel, unchanged schemas and final independent review. Applicable check-quick passed 264 file tests (one
existing NOOA skip), 55 PostgreSQL tests, all 10 gates and all five product flows. The adapter does not execute tasks.
TP-01E now verifies the combined foundation; it adds no task execution or product rollout.

## TP-01E foundation acceptance

[Foundation acceptance report](../../docs/guides/TASK_PLANNING_TP01_REPORT.md) records all nine master checklist
items, the current ownership and execution boundaries, schema/wheel compatibility, actual offline and lean tests,
skip reasons and outstanding runtime limits. All nine criteria are satisfied. Full check-all passed: 5,348 file
tests (14 existing skips) and 5,350 PostgreSQL tests (11 existing skips), all ten gates and all five product flows.
The real lean agent/import tests, installed core wheel, frozen schemas, final independent review and all four
completion-corpus validation/freshness checks passed. No task or new boundary test skipped. Core remains 0.157.0;
Home, Lead Agent and execution defaults are unchanged by the foundation. TP-02 is now complete as the separate
proposal-only package recorded below. Task execution, durable recovery and rollout remain later work.

## TP-02 proposal acceptance

[Task decomposition acceptance](../../docs/guides/TASK_DECOMPOSITION_ACCEPTANCE.md) records the versioned prompt,
source-preserving boundary compiler, exact deterministic fallback, request limits, privacy and API thinking capture.
The final relevant suite passed 175 tests in six files, including all 44 task-decomposer tests. All nine scripted
DEV scenarios passed at zero live cost and zero task execution. Final focused review passed. The OpenRouter
reasoning allowance is preserved and included in the planner's token ceiling and euro reservation.

The owner explicitly stopped broad testing on 2026-11-11 and requested only relevant tests and test groups.
Acceptance follows that instruction; no final full repository or full PostgreSQL pass is claimed. The authorized
replay compatibility repair requires fresh permission and keeps its reason in the workspace audit. TP-03 acceptance follows below.

## TP-03 task execution planning acceptance

[Task execution planning acceptance](../../docs/guides/TASK_EXECUTION_PLANNING_ACCEPTANCE.md) records typed plans
for the five existing stages and export, source/permission/schema/budget checks, explicit gate no-ops, bounded
off-by-default model proposals, exact fallback, privacy and saved API thinking summaries. Planning calls no tools.
All 125 relevant tests and 24/24 scripted DEV scenarios passed. Independent review passed after a canonical-order
comparison fix. Core contracts and schemas are unchanged; there is no scheduler or product rollout.

Only affected tests ran under the owner's standing instruction. The DEV result pins the new prompt and relevant
code; unchanged global prompt/harness snapshots are stale for the new prompt, and no broad gate pass is claimed.
TP-04 acceptance follows below.

## TP-04 guarded execution acceptance

[Task execution acceptance](../../docs/guides/TASK_EXECUTION_ACCEPTANCE.md) records the explicit task mode of the
existing SolutionTeam, durable pre-effect attempts in its current saver, project locks, real registry/engine
dispatch, owner-record reconciliation, unknown-effect stops and exactly one final holdout read. All 74 relevant
checks passed, including 24 new PostgreSQL cases and selected existing workflow/storage/lineage checks. Independent
review passed; its pool, ownership and recovered-budget findings are fixed. Checkpoints retain source hashes,
not private source class values. Scientific acceptance remains pending for TP-05. No product default is enabled.

The owner authorized continuing to TP-05 after this package was committed. Its implementation and review limit follow below.

## TP-05 verification and recovery

[Verification and recovery acceptance](../../docs/guides/TASK_RECOVERY_ACCEPTANCE.md) records measured scientific
proof, independent status/reason/evidence results, trusted registry boundary observation, bounded replacement
plans, explicit source-bound revisions and preserved verification history. All 115 distinct relevant application
checks passed, including file/PostgreSQL recovery, human choices, old approval rejection and one holdout use.
The focused installed core wheel passed; schemas are unchanged. Four development-corpus checks run before commit.
The reviewer findings are fixed, but the final independent review could not finish after a provider usage limit.
No final review pass is claimed. TP-06 has not begun.

## Implementation rules

1. Read the complete master specification before implementing a package.
2. Inspect the current repository before making changes.
3. Reuse existing contracts, tools, execution engines, validators, and checkpoints.
4. Implement only one package per Codex session.
5. Preserve backward compatibility and existing safety invariants.
6. Keep model-generated proposals separate from deterministic validation.
7. Never treat an agent's claim of completion as sufficient verification.
8. Run applicable tests after every package.
9. Record failures and incomplete requirements rather than hiding them.
10. Do not proceed automatically to the next package.

## Existing architecture to preserve

- `dclab_core/problem_map.py`
- `dclab_core/plan.py`
- `dclab_core/solution_workflow.py`
- `dclab_core/solution_map.py`
- `dclab_core/workflow_run.py`
- `dclab_core/programme.py`
- `dclab_rnd/agents/registry.py`
- `dclab_rnd/agents/harness.py`
- `dclab_rnd/intern/loop.py`
- `dclab_rnd/orchestra/solution.py`
- `dclab_rnd/studio/graph.py`
- `dclab_rnd/studio/engine.py`

These components remain authoritative for their existing responsibilities.

## How to run with Codex

Start by reading the master specification and the selected package prompt.

For the first planning package, give Codex the following instruction.

```text
Read engineering/planning/README.md.

Read the master specification at
engineering/planning/DCLab_Hierarchical_Planning_Codex_Pack.md.

Then read
engineering/planning/TP_01A.txt.

Inspect the current repository and implement TP-01A only.

Run the relevant tests, report the actual results,
and document architectural decisions.

Do not continue to TP-01B automatically.
```

After TP-01A passes its acceptance requirements, move to TP-01B and continue in dependency order.

R-01 is an independent routing verification package. It does not block the deterministic task planning foundation unless its findings reveal a shared integration issue.

## Definition of done

A package is complete only when

- Its required functionality is implemented.
- Its acceptance criteria pass.
- Existing safety protections remain intact.
- Relevant regression tests pass.
- Source and contract changes are documented.
- Known limitations and blockers are recorded.
- The implementation is ready for review.

A package with unresolved critical failures remains incomplete.

## Final objective

DCLab should be able to take a validated solution, break it into meaningful tasks, create safe execution plans, run tasks using existing capabilities, verify their outputs, and replan when necessary.

The result must be more reliable and useful than the current workflow, not merely more complicated.
