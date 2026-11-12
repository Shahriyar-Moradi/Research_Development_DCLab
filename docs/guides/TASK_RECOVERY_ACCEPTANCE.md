# TP-05 acceptance: measured verification and bounded recovery

Implemented on 2026-11-12. The explicit task path remains off by default. Implementation and focused validation
are complete; final independent review remains pending because the reviewer run ended at a provider usage limit.
Its six concrete findings were addressed and covered by focused regressions. No final review pass is claimed.

## Scope and owners

The existing SolutionTeam, guarded registry, engine, project stores, holdout receipt and checkpoint saver still own
execution. `dclab_rnd/tasks/verifier.py` reads their records independently. It invokes no tool, model, fit or table-row
analysis. `dclab_core/task_verification.py` contains pure measured-proof checks; `task_recovery.py` contains bounded
failure decisions, dependency invalidation and explicit rebasing of unchanged verification evidence.

Core is 0.158.0. No frozen contract or published schema changes. Verification dictionaries are internal host metadata,
not a new interoperable TaskVerification contract. Existing `agent.step` events record verifier/recovery results.
Their complete structured results and evidence references are retained in the existing task saver. The verifier also
checks the current solution signature and semantic source context itself, independently of dispatcher preflight.

## Evidence and decisions

| Criterion | Measured owner evidence |
|---|---|
| Class balance and locked split | Actual `TaskBundle.train_idx/test_idx` split identity, disjoint partitions, training class-slot counts, per-fold fit/validation counts and index digests. No class values are copied into task traces. |
| Training scope | Fold coverage, stratified allocation, matching training proof across stages, and owner scope manifest. Candidate metrics must be finite and measured on the same folds; blocked columns cannot enter selected candidate fits. |
| Leakage | Existing canaries, declared exclusions, identifiers and prediction-time contract. Strong unresolved findings stop before features/models/final. The leakage stage itself can finish so the existing team can ask the person. |
| Human exceptions | An explicit kept-suspect decision requires the current team's persisted audit and later human-pause result, tied to the current audit inputs and verdict. A ledger flag alone cannot transfer across inputs. Valid human feature/model choices retain their existing transition provenance. |
| Threshold | Complete, unique, finite training OOF scores, split/CV/configuration/policy fingerprints. Missing proof blocks acceptance after scoring; it never permits another final read. |
| Final result | Actual consumed receipt, exact split, one use, matching owner run and finite metric/interval. A business-success failure does not authorize holdout tuning. |
| Report | The guarded export additionally registers a canonical SolutionReport JSON in task mode. Its displayed metric and interval paths, limits, citations, current context and exact deterministic content are checked. Reading the report never reanalyses rows or calls a model. |

The split fix is material: the old engine checked nonexistent `train_indices/holdout_indices` fields and could bind
a binary final receipt to empty index lists. New evidence uses the actual bundle fields. Old incompatible receipts
remain blocked; no score is repeated to repair them. Optional proof capture cannot break a legacy engine run;
unavailable proof is recorded as unavailable and task acceptance stays blocked.

Verification returns `verified`, `failed`, `blocked` or `stale`, with code-owned reasons, violated criteria, plan/graph
and source identities, input/output fingerprints and evidence links. Operation completion and acceptance are separate.
A successful tool reply, an agent's claim, a journal flag or an incomplete report is insufficient.

## Recovery timeline

1. Save the validated plan, original input fingerprints, reserved budget and attempt identity before an effect.
2. The registry observes validation, guard and actual handler entry in a scoped in-process context. A tool reply cannot
   supply this evidence. An exception before the actual handler may prove no scientific effect; handler entry remains
   ambiguous until an exact owner completion record exists.
3. Reconcile completed owner records first, including a late completion after an earlier unknown result. Preserve the
   original attempt; do not invoke the handler again. Recovered work retains at least its reservation or known duration.
4. `TaskDispatcher.replan(project_id, task_id)` creates a new deterministic plan only for trusted no-handler-entry
   evidence and no conflicting owner result. It passes the same planner/action validator. It never calls a tool.
   A subsequent explicit dispatch uses that revision. Original plans, attempts and charges remain in history.
5. Each active task has at most two attempts. At most two replacement plans and two source revisions are admitted.
   Reusing one retry authorization, repeating a replan or exhausting the budget stops transparently.
6. Policy refusal, missing capability/approval, domain failure, uncertain effects and consumed/intended holdout stop.
   There is no compensation, automatic safety-step rewrite or automatic holdout reuse.

For changed data, schema, objective or solution, dispatch first reports stale inputs. Source adoption is explicit:

```python
# Called by the authenticated owner through the existing approval operation, after reviewing current inputs.
revision = dispatcher.source_revision(project_id)
graph.approve_gate(store, project_id, "solution", by="owner", reason="Task source revision " + revision)
result = dispatcher.adopt_sources(project_id)  # metadata only; never signs or clears owner stages itself
```

The confirmation binds the full semantic context used by the adapter, solution, table/schema and related inputs.
An approval identity cannot be consumed twice. The existing owner must clear scientifically stale stages through
its normal operation before adoption. Only affected dependency descendants are invalidated. Unchanged verified
work is rechecked and gets a new verification identity with `reused_from`; prior proofs are never relabelled.
Grouping alone does not invalidate an independent sibling. Budgets never reset. Prior task-version/code checkpoints
are retained and blocked rather than silently migrated or replayed.

The current physical binary graph shares its table, solution and execution context across stages. Such a source
change legitimately invalidates that full scientific chain. Independent verification branches are covered by the
pure dependency test; this package adds no parallel scheduler or new scientific branch executor.

## Focused checks

115 distinct application tests passed across focused runs, with no skips:

- 55 TP-05 tests: seven pure tests and 24 integration cases on each of file/SQLite and PostgreSQL. These include
  full measured execution/export, absent/invalid proof, actual-byte/goal/solution changes, revoked approval,
  related tables, strong leakage and human decisions, absent threshold proof, hidden report figures, source-bound
  confirmation, old unused holdout permissions, retry exhaustion, late completion and saver reopen/timeline.
- All 22 existing file task-dispatch tests, plus four selected PostgreSQL crash/resume, overrun and exclusivity cases.
- 19 core-boundary/schema, holdout crash-safety and tool-recording checks.
- 15 existing registry/caller, binary assessment and final-replay compatibility checks.

The initial pure tests failed with missing-module imports before implementation. Added regressions exposed a lost
public verdict and a late-completion time charge below its reservation; the final targeted cases pass. An initially
mistyped existing test name caused one loader error; the correct overrun cases passed afterward.

Commands used the no-live environment, `.venv/bin/python -m unittest`, and the affected classes/cases in
`tests/test_task_verifier.py`, `test_task_recovery.py`, `test_task_dispatch.py`, `test_agents_runtime.py`,
`test_core_contracts.py`, `test_holdout_crash_safety.py`, `test_agent_harness.py`, `test_core_assess.py` and
`test_lineage.py`. Existing passing cases were repeated only after relevant behavior changed. The task-specific
cases were added and run in focused batches rather than repeatedly running the whole growing group.

`python scripts/task_core_wheel_check.py` reuses the existing isolated wheel build/install runner with a TP-05-only
smoke: Python 3.12, only the four declared direct dependencies, installed proof/recovery functions and adapter
conformance. It passed. Published schemas remain byte-compatible. `git diff --check` passed.
The development corpus is rebuilt with `make sft-dev` and its four validation/freshness tests before commit.

No broad repository or full PostgreSQL suite, live model, sealed benchmark, browser flow or prompt evaluation ran.
No production-readiness or rollout claim follows from these invented-data checks.

## Review and remaining limits

The reviewer identified incomplete source invalidation/confirmation binding, unresolved strong leakage, insufficient
report display proof, loss of measured public verification results and reuse of an old model attempt's permissions.
These are fixed. Confirmation tests also caught circular hashing through approval-derived source-plan prose; the
semantic confirmation excludes its own signature while the adopted graph retains the actual source references.
The final independent recheck could not finish because the reviewer hit a provider usage limit. TP-06 has not begun.

Semantic feature availability still relies on the signed prediction-time contract and explicit human exceptions;
statistical screening cannot prove it. Unsupported/grouped/temporal partitions may have insufficient proof and
remain blocked in task mode. Related-table/programme execution stays with its existing owner. Legacy engine mode
continues unchanged as the default. Existing task-version journals require explicit future migration design, and
consumed holdout projects require their existing governed handling rather than automatic recovery.

The local `dev` branch already contains a commit absent from `main`; it was left intact rather than reset or
merged into this package. TP-05 is committed on `main`.
