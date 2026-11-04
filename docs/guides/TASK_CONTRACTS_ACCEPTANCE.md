# Task contracts acceptance — TP-01B

Package: [TP_01B.txt](../../engineering/planning/TP_01B.txt). Reviewed checkout before this package:
`45344f45b3816605c0e0bcea19f9ce9b957ec65b`, 2026-11-04. This is an additive contract package only.
TP-01C and later work is not included. Status: **complete — all TP-01B acceptance criteria satisfied**.

During validation the checkout advanced to `5320b46eedc3758714a433f3d5a231d3ea27dc0a` with UI-only changes. The core, registry and execution owners remain unchanged by that commit; its files are preserved. The final gate runs on that checkout.

## Scope and prerequisite

TP-01A [architecture](TASK_PLANNING_ARCHITECTURE.md) and [matrix](TASK_CAPABILITY_MATRIX.md) remain the authority for
integration boundaries. The only intervening committed change from that review was the product's default appearance;
core contracts, registry, engine/programme, approval owners and storage did not change. There is no unresolved
ownership decision: the existing owners keep authorization and scientific effects. Recorded approval/recovery gaps
remain execution blockers for future packages. This package neither resolves nor bypasses them.

The earlier TP-01A files and owner's untracked workspace files were preserved. Its source audit remains a historical
snapshot of 0.155.0; new package version/schema registrations intentionally differ and do not rewrite that evidence.

## Delivered contracts

| Contract | Identity and structure | Local guarantees | Deferred checks |
|---|---|---|---|
| `task_graph.TaskGraph` v1 | Graph ID/revision, business objective/source refs, workflow/solution fingerprints, structural SHA-256 declaration, nodes, typed grouping/dependency edges, provenance | Explicit version, unknown fields refused, bounded typed IDs/text/tuples, no blank objectives, duplicate IDs/references refused; only actions have capability bindings; blocked/gate/verification rules | Canonical hash computation/authenticity, graph traversal/cycles/reachability, coverage, source approval, actual capability membership/permissions and edge/dependency consistency |
| `task_execution_plan.TaskExecutionPlan` v1 | Graph/task/revision and plan identities, declared hashes, expected input identities/types/fingerprints, selected capability refs, bounded ordered actions and outputs, checks, budget/policy/provenance/status | No raw argument values/expressions; inputs declared; deferred output ref must exist, match type and come from earlier action; duplicate identities refused; conservative unknown=block; bounded retry/tool/time/cost limits | Real registry argument schema validation and authorization, actual artifact resolution, source approval freshness, safe per-tool retry rules, evidence verification and execution |

Every nested collection is a tuple; nested objects are frozen `Contract` models. This goes beyond shallow outer
`frozen=True` without changing the shared base. JSON serialization still uses arrays/objects. New documents require
an explicit supported `schema_version` in constructors, `model_validate`, `.read` and JSON schemas. Existing models
retain their default-construction behavior and reader/version rules.

An action binds a canonical capability name, scope and schema fingerprint, without copying the tool's schema.
The supported fixture names the real project `run_stage` capability. This contract does not certify a supplied
schema fingerprint or grant authority: a later host must match it against the actual registry, principal, project,
approval and source inputs. No registry/runtime is imported into core.

Plan status `ready` means local shape-ready, never permission to execute. `noop` and `blocked` require a reason and
contain no actions, outputs or selected capabilities. There is no `completed`, `verified` or `success` plan status.
Attempts, measured outcomes and unknown completion remain outside the immutable structural graph.

The first adapter slice needs only the existing stage calls: host-supplied project/stage input references and one
stage record output per physical action. It needs no literal/evaluation language. Limits leave room for all mandatory
workflow steps while bounding proposals: graph 256 nodes/1024 edges, plan 32 actions, 128 inputs/outputs, 32
capabilities/checks, 32 arguments per action. Tool allowance includes the declared retry maximum; it does not grant
retry permission for non-idempotent/final actions. Host budgets may be stricter. These are capacity bounds, not a
benchmark effectiveness claim.

## Schema and wheel compatibility

Core release advances **0.155.0 → 0.156.0**; the version moves together in pyproject, module and changelog. Direct
dependencies remain numpy, pandas, scikit-learn and pydantic; Python ≥3.10 remains supported. The only schema additions:

- `dclab_core/schemas/task_graph.json`
- `dclab_core/schemas/task_execution_plan.json`
- `dclab_core/schemas/baseline/dclab_core.task_graph.v1.json`
- `dclab_core/schemas/baseline/dclab_core.task_execution_plan.v1.json`

All **258 pre-existing current/historical baseline schema files remain byte-identical**. The registered schema count
is 107, previously 105. Generated files were created with `make core-schemas`; no existing baseline was edited.
The new unpublished provisional baselines were regenerated after the review fix before finalizing the v1 shape.
Once published, either v1 shape must never change without a new contract version.

`schemas.py` imports both modules using the existing explicit registration pattern. Each self-registers in
`contracts.CONTRACTS`. Both fresh-process import orders are tested. No global Contract/Versioned behavior or
contracts.py import graph was changed.

The JSON schemas express local text/kind/scope/status rules as well as type and collection limits. JSON Schema
cannot compare arbitrary IDs/types across records or calculate a budget from the action/retry counts; use `.read`
for Python local cross-reference checks and equivalent host semantic validation where Python is not used. Neither
schema validation nor `.read` replaces the later graph/capability/approval validator.

The pinned main-product inventory has no fields for these objects. `adapters.conformance` lists both as library-only
outputs with explicit reasons. There is no invented ExperimentPlan property or new adapter mapping. Existing adapter
contract tests and conformance still apply. `CORE_INTEGRATION.md` documents the deferred host adoption requirements.

## Quick local test

From the repository root:

```bash
.venv/bin/python -m unittest tests.test_task_graph tests.test_task_execution_plan -q
```

Expected: 23 tests and `OK`. This reads and validates invented task documents; it runs no model and calls no tool.
Grouping edges run parent → child; dependency edges run prerequisite → dependent. These meanings are structural;
edge consistency, cycles and canonical hashes are TP-01C work.

## Tests and observed review

Tests were written before implementation. The first run failed with two missing-module import errors. After initial
implementation, review found JSON Schema accepting callable groups, blank text and blocked plans with actions even
though Python rejected them. Two new negative schema tests exposed **15 failing subcases before the fix**. The fix
added nonblank patterns and node-kind/scope/plan-status conditionals. The corrected task suite passes; the reviewer
returned PASS with no remaining correctness finding. A small ambiguous README sentence was also corrected.

| Check | Result | Limits / failure classification |
|---|---|---|
| New task contract tests | PASS: 23 tests | Includes JSON roundtrip, both version readers, unknown fields, invalid enums/IDs, duplicate IDs, blank objectives, bounds, immutable collections, typed earlier-output refs, no-op/blocked states, Python/JSONSchema negative cases and import orders |
| Focused core/adapter/conformance/wheel-metadata tests | PASS: 91 tests run, 90 passed, 1 opt-in wheel test skipped; 6 files, 20.2 seconds | The standalone installed-wheel command passed separately; 23 new task tests are included |
| `make core-schemas` and old schema byte comparison | PASS: four additive files; 258 previous files unchanged | No existing contract version/shape changed |
| `make core-wheel-check` | PASS: installed 0.156.0 wheel, isolated Python 3.12.13, both task v1 documents and frozen schemas | Four declared direct dependencies; no unexpected distributions or host imports. Initial sandbox cache-access failure was environmental; final approved run passed |
| Initial `make check-quick JOBS=3` | FAILED: 5,225 tests ran across 449 files; one stale-catalog test failure | Introduced by new version/contract inventory. All five end-to-end flows passed. PG phase did not run because file-test recipe failed |
| Catalog correction and `make rd-gates` | PASS: 17 catalog tests and all 10 gates after `make catalog` | Generated diff contains only version/count/new contract rows; no runtime fix |
| Final `make check-quick JOBS=4` | **PASS**: 5,225 file-backed tests run (11 skipped), 827 PostgreSQL tests run (2 skipped); all 10 rd-gates and all five product flows passed | 449 file-backed files / 678 units in 3,044.4 seconds; 58 PG files / 112 units in 652.7 seconds. No unresolved failures |
| Final `make sft-dev` | PASS: 6,573 examples from 632 packages | Final development trajectory validated; generated corpus rebuilt, no hand edits |
| Final `.venv/bin/python -m unittest tests.test_devlog -q` | PASS: 4 tests, 3.872 seconds | Run after the final record/corpus update |
| Adversarial TP-01B review | PASS after fixing one schema-parity finding | Confirmed all 23 new tests; no remaining code correctness finding |
| Full `make rd-check`, `make check-all`, live model calls and sealed benchmarks | Not run for TP-01B | No actual commit or push is claimed; full rd-check remains a repository prerequisite to a commit. No execution/storage change or milestone closure; no task effectiveness claim |

The `.venv-agent` environment already lacks core dependencies (numpy/pandas/scikit-learn); its baseline import fails.
No compatibility shim or dependency was added. The isolated installed wheel with its declared four dependencies is
the relevant portability test, alongside existing source boundary/Python 3.10 syntax tests.

## Acceptance checklist

- [x] Rechecked current ownership and TP-01A prerequisite; no new authorization owner.
- [x] Added only immutable public task contracts and supporting local shape rules; no execution, graph traversal or model request.
- [x] Bounded IDs, known kinds/scopes, grouping/dependency distinction, human/check/blocked interpretation.
- [x] Controlled typed input/output references, budgets, conservative unknown handling, distinct noncompletion states.
- [x] New current/baseline schemas; unknown fields and missing/unsupported versions rejected.
- [x] Preserved all previous schema bytes, dependencies and adapter outputs.
- [x] New positive/adversarial tests written first; reviewer finding fixed with failing-before/passing-after tests.
- [x] Final installed wheel, focused suite and applicable repository gate passed and recorded.
- [x] Final development record/corpus, tracker and handoff reflect verified status.

Skip accounting: the file run skipped five PostgreSQL-only cases (covered by the PG run), three notebook cases
whose frozen manifest is absent, two NOOA cases belonging to the isolated agent environment, and the opt-in wheel
unittest (covered by the standalone installed-wheel command). The PG run skipped two file-store-specific cases
covered in the file run. No new task-contract test was skipped. These environment/fixture exclusions are existing
limits, not new test failures. Live-model and sealed benchmark commands were not run.

## Files and handoff

New: `dclab_core/task_graph.py`, `task_execution_plan.py`, four generated schema files,
`tests/test_task_graph.py`, `test_task_execution_plan.py`, this guide and development trajectory `tp-01b.json`.
Updated: schema registration, adapter conformance reasons, core README/CHANGELOG/version metadata,
`scripts/core_wheel_check.py` (installed task contract/schema smoke), `CORE_INTEGRATION.md`, planning README tracker
and generated SFT development corpus plus `docs/DATA_CATALOG.md` / `docs/data_catalog.json` (rebuilt with `make catalog`). Host runtime, Home/Lead Agent/default settings, dependencies and persistence
remain unchanged. Earlier TP-01A work is preserved separately in the working tree.

**TP-01B is complete. TP-01C is ready to begin as a separate package and has not started.**

TP-01C will add the semantic graph validator and canonical fingerprint rules. It must check current approved-source
identity, registry capability snapshots, edge/dependency semantics, coverage and stale-source handling. Supplied
fingerprints, capability names and `ready` documents in this package are untrusted declarations until those checks.
No automatic progression into TP-01C occurs here.

Proposed reviewable commit: `feat(core): add immutable task proposal contracts for TP-01B`. Changes remain uncommitted
until the repository's commit gate is satisfied; preserve the owner and TP-01A working-tree changes.
