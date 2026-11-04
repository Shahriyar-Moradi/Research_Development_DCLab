# Task planning acceptance and handoff — TP-01A

Scope: the architecture/capability package only, reviewed on 2026-11-04 against `81e8a9009174031c64148ebd22ef035500fccd08`. The task planning foundation and the roadmap as a whole are **not implemented**. No public contract or execution behavior changes in this package.

Required outputs: [architecture](TASK_PLANNING_ARCHITECTURE.md), [capability matrix](TASK_CAPABILITY_MATRIX.md), and this acceptance record. [TP_01A_EVIDENCE.json](../../engineering/planning/TP_01A_EVIDENCE.json) is the expanded source audit attachment. [Planning README](../../engineering/planning/README.md) tracks package status. Source mismatch after this baseline requires reinspection.

## TP-01A acceptance criteria

| Criterion | Evidence | Status |
|---|---|---|
| Read README, full master and every supplied prompt before edits | Twelve flat prompt files and master recorded in evidence source hashes; actual paths corrected in README | Satisfied |
| Review existing understanding, planning, execution, tools, guards, schemas, persistence, evaluation and tests | Architecture sections 1–6 and source citations; inspected focused test files, Makefile, `scripts/affected_tests.py`, `scripts/core_wheel_check.py`, `docs/guides/BENCHMARK.md` | Satisfied |
| Map every mandatory SolutionWorkflow step with every requested field | Evidence `mandatory_steps`: exact family/step set, 105 entries, 28 IDs; full-field readable shared definitions in matrix | Satisfied — exact metadata cross-check passed |
| Independent inventory from actual registry, not inferred tool names | Evidence `registered_tools`: all 105 tools, exact schemas, aliases, scopes/effects/moves, handler lines, guards, permission/repeat policies | Satisfied — exact metadata cross-check passed |
| Explain shared execution boundaries and complex versus flat | Single data/models/final producer; physical features predecessor; 12-step WorkflowRun readback; atomic programme with deferred leaf finals | Satisfied |
| Inventory file/PG project, artifact, trace, event and checkpoint behavior | Architecture storage table and revision/recovery limitations; no second datastore | Satisfied |
| Settle seven requested architectural decisions | Architecture sections 1–4: object separation, structural/mutable ownership, authorities, approved admission, minimal v1/deferred work, real bindings, wheel/host compatibility | Satisfied |
| First supported binary Telco slice with producer/check/evidence links | Architecture section 7, `studio.data.SAMPLE_META["telco_churn"]`, target Churn, ID exclusion, all five stage producers and human gate | Satisfied; design only, no Telco execution claimed |
| Threat model, hard gates, benchmark plan, exclusions and discrepancies | Architecture sections 3/6/8; gates and benchmark below | Satisfied |
| No runtime/public contract/default/dependency/new-agent changes; preserve owner work | Restricted diff; prior routing work saved outside checkout; no later package implementation | Satisfied — runtime/public contract diff is empty |
| Applicable checks actually executed, failure classification, adversarial review | Verification ledger below; review record in development trajectory | Satisfied — coverage, focused regressions, package gate and review passed |

All TP-01A acceptance rows above pass. The package status does not mark later cross-phase gates passed. Open execution gaps below are explicit blockers for later execution packages, not claims that new runtime features exist.

## Cross-phase hard gates

These gates belong to the named future packages. Listing them does not mark them passed.

| Package / boundary | Required evidence before acceptance | Current status |
|---|---|---|
| TP-01B contracts | Additive frozen/versioned TaskGraph/TaskExecutionPlan with bounded IDs/fields, explicit source and provenance, blockers, no raw rows; unchanged existing payload roundtrips; generated schema baselines; core wheel and existing main-product adapters pass | Not started |
| TP-01C graph validator | Cycles/dangling edges/invalid groups rejected; stable canonical semantic hashes; changed dependencies, inputs, approvals or capabilities invalidate affected work; no current source bypass | Not started |
| TP-01D adapter | Pure/no handler/no model/no table load; all required workflow steps covered or blocked; many-to-one stage binding; no duplicate effect; first Telco slice; unsupported families remain explicit | Not started |
| TP-01E foundation regression | Offline fixtures, imports/schema/wheel/adapter compatibility, deterministic hashes, no default path change; coverage of negative/blocker cases | Not started |
| TP-02 proposal boundary | Gateway-injected model proposes only; semantic validation, fallback and human decision preservation; new purpose stores tested/redacted API thinking summary, never prompted hidden reasoning | Not started |
| TP-03 action plan boundary | Exact real tool/schema/scope binding, approved-source admission, bounded sequential plan, typed input refs, spend/stop limits, independent checks; no calls while planning | Not started |
| TP-04 dispatch boundary | Original registry/guard/engine rechecks, principal/context binding, current human approval, durable before-effect intent and owner receipt, project concurrency safety in file/PG, cancellation, no repeated non-idempotent action | Not started; existing final ambiguity blocks automatic retry |
| TP-05 completion/recovery | Verified/failed/blocked/stale/unknown outcomes from artifact/run/input proof; model self-report insufficient; crash reconciliation and exact approval/data revision binding; changed-input descendants invalidated; final unknown never retried | Not started |
| TP-06 promotion | Frozen paired A/B/C benchmark; hard zero safety violations, no regression, predeclared gain/cost/latency/sample thresholds and intervals; insufficient evidence fails promotion | Not started |
| TP-07 rollout | Explicit opt-in, existing route/role/no-data behavior retained, human gate visible, rollback and observability; no implicit default switch | Not started |

Every boundary must preserve leakage-safe folds, protected labels/holdout-once, person decisions, permission checks, provenance and stop/budget controls. “Zero safety violations” is a hard gate, not a weighted score that faster runs can offset.

## Threat model and required negative cases

| Failure / adversarial case | Deterministic protection to require | Existing owner / future proof |
|---|---|---|
| Task text/prompt names arbitrary tool, file, URL or code | Strict real capability/schema/scope/allowlist; unresolved capability blocked; no arbitrary execution | Registry and host authorization; TP-01C/03/04 rejection tests |
| Model drops human gate or invents confirmed budget/intent | Required coverage and approved-source provenance, current human signature/input identity; preserved corrections | Workflow/map validators, coordinator, graph approval API; TP-01D/02/04 tests |
| Valid-looking graph runs before a solution is approved | Admission requires explicit current approved source, independent of legacy optional flags | New opt-in host boundary, original execution guard recheck; TP-03/04 negative fixture |
| Multiple conceptual steps repeat one stage/holdout read | Canonical effect binding and receipts; many-to-one coverage; one atomic programme | Engine/Programme; TP-01D/04 duplicate-action tests |
| Success claim without current output or report | Inspect actual artifact hash, run and input references, criterion result; missing/partial evidence blocks | Existing stage records/RunRecord/catalog; TP-05 verifier tests |
| Crash after scoring before status/count save | Persist owner-level intent/receipt and reconcile; unknown blocks, never blindly retry | Existing engine/programme gap; TP-04/05 fault injection around every write boundary |
| Checkpoint replays completed write | Reconcile attempt/effect receipt before dispatch; duplicate plan call cannot repeat side effect | Existing checkpoints plus future receipt semantics; file/PG resume tests |
| Changed table, related table, solution, approval, prompt/capability or upstream output | New revision; reject stale artifact reuse; invalidate descendants with exact lineage | Source hashes/RunRecord plus TP-01C/05 tests |
| Two workers race same task or project | Atomic revision/claim/receipt checks across processes, not process-local walk lock | Existing file/PG storage extended later; TP-04/05 concurrent fixture |
| Read tool leaks rows or silently writes while planning | Metadata-only pure adapter; strict allowlisted task event fields; no handler discovery | Registry metadata; gateway redaction/consent/traces; TP-01D/privacy tests |
| Sealed data enters prompts/training or gate is tuned after score | Use existing dev/sealed separation and once-only runner; freeze cases/rules before runs | Existing benchmark/consent/export owners; TP-06 review and manifest tests |
| Unsupported near-copy isolation, multi-entity split, target transform or report | Explicit blocker, never substitute detection/summary/WIP file for stronger proof | Capability matrix partial guarantees; TP-01D/05 abstention fixtures |

## Comparative benchmark plan (TP-06, not run here)

Reuse `dclab_rnd/bench`, current campaign/result conventions in [BENCHMARK.md](BENCHMARK.md), and `dclab_rnd/agent_eval/system_tasks`. Reuse existing no-model/scripted transports, journeys, replica safety traps and the harness; do not create another evaluation framework. Verify actual owners again at TP-06 because the repository may advance.

Arms are fixed by the master package:

- **A:** current fixed workflow and original execution.
- **B:** deterministic task decomposition and deterministic task plans, with original guarded execution.
- **C:** model-assisted proposals/plans, original guarded execution, independent verification and bounded replanning.

Run matched cases with the same problem/data/capability snapshot, seeds and budgets. Freeze DEV case IDs, split manifest, gate rules, sample size, metric definitions, prompts/hashes, capability-schema hash, commit and source/input fingerprints **before** results. Predeclare what positive completion/recovery gain and acceptable extra latency/cost mean numerically, with justification, in TP-06 code. Those non-safety numbers are not invented in this architecture package. Maintain whole-case separation from training and prompt tuning. Sealed evaluation remains once-only under its existing owner; live model spend and sealed runs need explicit authorization.

Case strata: clear binary model task, ambiguous intent, explicit Home chip, segmentation/reporting ambiguity, multi-table fraud, missing labels, incomplete tables, contradictory instructions, unsupported approach, unapproved solution, stale/revoked approval, used holdout, denied guard, transient failure, final completion ambiguity, changed input, resume/retry and concurrency. Mark fixture/synthetic evidence separately from real user outcomes. R-01 route evidence is independent; do not change routing to make a task benchmark pass.

Record paired end-to-end correctness/verified completion, required-step coverage, graph/plan validity, correct abstention/blocker handling, false completion, authorization violations, holdout/leakage violations, duplicate effects, provenance completeness and recovery by failure class. Also record avoided re-execution, prompt/tool counts, tokens, latency, compute, EUR and human confirmations. Report per-stratum failure counts and paired differences with intervals; do not hide a red safety stratum behind an average.

Promotion requires zero critical safety violations across all runs, existing required workflow/route regressions passing, and a predeclared positive completion or recovery gain over A at acceptable cost/delay. C beating B on development cases alone is insufficient. Missing evidence or inadequate sample size yields **INSUFFICIENT-EVIDENCE**, not promotion. Deliver versioned result artifacts and PASS/FAIL/INSUFFICIENT-EVIDENCE through the existing bench conventions. No benchmark silently changes a default.

## Verification ledger

The initial acceptance check failed because all three required documents were absent. TP-01A fixes that documentation gap. It introduces no executable behavior for a new runtime test to exercise. Source/matrix checks inspect metadata only; they never invoke registered handlers, open a table or score a holdout.

| Command / check | Actual result | Interpretation |
|---|---|---|
| `.venv/bin/python tests/run_parallel.py --jobs 3 test_core_contracts.py test_solution_workflow.py test_problem_map.py test_mapping.py test_solution_map.py test_graph.py test_agents_runtime.py` | **PASS: 256 tests, 7 files, 26.0 seconds, no skips** | Existing core/map/workflow/guard/runtime regression tests; scripted/offline, not a new planner benchmark |
| Final source/matrix/registry/schema coverage check | **PASS: 105 pairs, 18 audit columns, 105 exact tools, 105 contracts; source hashes and links valid** | Must check exact family/step set, all required fields plus template checks, full tool schemas/handlers/guards, source hashes/paths and current schema count without invoking tools |
| `.venv/bin/python tests/run_parallel.py --jobs 3 test_core_adapters.py test_workflow_run.py test_programme.py test_orchestra_solution.py` | **PASS: 104 tests, 4 files, 72 execution units, 274.5 seconds, no skips** | Existing adapter, evidence readback, programme and SolutionTeam fixtures; no new wheel/core API |
| `make sft-dev` | **PASS: 6,559 examples from 631 packages** | Development trajectory validated and generated SFT corpus rebuilt; generated files were not edited by hand |
| `.venv/bin/python -m unittest tests.test_devlog -q` after the final record update | **PASS: 4 tests, 3.855 seconds** | Final development record/corpus freshness verification |
| `make check-quick JOBS=3` | **PASS: 23 file-backed tests, 10 PostgreSQL tests, no skips; all 10 rd-gates passed** | Four affected files on file storage; one on local dclab_test PG; product flows correctly reported “not reached by the change” |
| TP-01A adversarial reviewer | **PASS: no correctness findings** | Independently matched all 105 step/family pairs, 105 tool schemas, 54 source hashes and 105 contracts; confirmed authority and recovery limits |
| `make core-schemas`, `make core-wheel-check` | Not run for TP-01A | No contract/schema/wheel changes. Current registrations/baselines are inspected, focused contract/adapter tests run. Mandatory once contracts change in TP-01B/E |
| `make check-all`, full `make rd-check`, browser/product flows | Not run for TP-01A | No runtime/storage/frontend change or milestone close. Full `rd-check` is required before an actual commit under repository rules; this handoff proposes a single commit until that gate passes |
| Live model calls, Telco training/scoring, dev comparative benchmark, any sealed suite | Not run | Not required for this design-only package; no effectiveness or production-readiness claim |

The first additional-test attempt failed before tests ran because the sandbox blocked the multiprocessing local socket (`PermissionError`). The identical command passed when run with approved local test permissions. This was an environment limitation, not an application regression. No test failure remains unresolved in the executed TP-01A checks.

Superseded R-01 runs and an interrupted broad suite are not TP-01A passes. Only the executed results above belong to this acceptance record. Documentation corrections discovered by source checks are introduced mapping defects and must be fixed here. Approval/counter/replay shortcomings found in unchanged runtime code are pre-existing execution limitations; they are recorded and must block unsafe future bindings.

## Files created or modified

| Files | Change |
|---|---|
| `docs/guides/TASK_PLANNING_ARCHITECTURE.md` | New architecture, boundaries, authority/storage/recovery decisions, threat limits and Telco slice |
| `docs/guides/TASK_CAPABILITY_MATRIX.md` | New full-field shared step definitions, family/conditional coverage and complete registered tool inventory |
| `docs/guides/TASK_PLANNING_ACCEPTANCE.md` | New package acceptance, cross-phase gates, benchmark plan, actual results and handoff |
| `engineering/planning/TP_01A_EVIDENCE.json` | New static expanded step/tool/schema/source audit; no runtime registration |
| `engineering/planning/README.md` | Correct actual input paths, link artifacts, mark only TP-01A verified complete |
| `research/llm-fine-tuning/experiments/dev_trajectories/records/tp-01a.json` | New development trajectory with findings, open existing risks, tests and reviewer verdict |
| `research/llm-fine-tuning/experiments/sft/out_dev/MANIFEST.json`, `train.alpaca.jsonl`, `train.chat.jsonl` | Rebuilt by `make sft-dev` |

The master and twelve prompt files under `engineering/planning/` were supplied by the owner and read unchanged. They remain part of the proposed self-contained planning-package commit. No source/test/dependency/schema/migration/UI file changed. Owner `.agents/`, `.codex/` and `AGENTS.md` remain untouched and outside this package.

## Completion and readiness

**TP-01A: complete — all package acceptance criteria satisfied.** No TP-01B work is included. Runtime capability gaps remain recorded as later-package blockers, not solved features.

Readiness for TP-01B is limited to additive pure contract work following the completed TP-01A checks. Existing authority owners and capability boundaries are resolved. Exact contract field names, enum choices and schema versions belong to TP-01B. Its contracts must represent blocked/unsupported/stale/unknown work and current-approved source references; they must not imply that durable execution receipts already exist.

Execution/recovery readiness is **not established**. Final score crash ambiguity, source-bound approvals, split proof and atomic revision/receipt/concurrency semantics remain open work for their owners before TP-04/05 can pass. These limits do not require inventing a new agent, datastore or engine. Stop after TP-01A and hand off to TP-01B; do not automatically proceed.

Proposed single reviewable commit: `docs(planning): map guarded task planning boundaries for TP-01A`. Include the supplied planning package, three guides, static source audit, tracker, development trajectory and generated SFT corpus. Exclude owner `.agents/`, `.codex/`, `AGENTS.md` and the separately preserved R-01 work. No commit or push is claimed by this document.

## Reproduce the metadata coverage check

Run from the repository root with `.venv/bin/python`. This inspects contracts and registry metadata; it does not call a tool handler or access a dataset. Source hashes intentionally fail if a later checkout changes an audited file.

```python
import hashlib
import inspect
import json
from pathlib import Path
from dclab_core.schemas import generate
from dclab_core.solution_workflow import SAFETY, FAMILY_STEPS
from dclab_rnd.agents import default_registry

audit = json.loads(Path("engineering/planning/TP_01A_EVIDENCE.json").read_text())
expected = {
    (family, step.id): step
    for family, steps in FAMILY_STEPS.items()
    for step in (*SAFETY, *steps)
}
rows = {(row["family"], row["step_id"]): row for row in audit["mandatory_steps"]}
assert len(rows) == len(audit["mandatory_steps"]) == 105
assert rows.keys() == expected.keys()
fields = {
    "step_id", "family", "wf_block", "prerequisites", "artifacts_consumed",
    "expected_outputs", "parent_business_goal", "suggested_task_group",
    "existing_authority", "actual_callable_action", "guard", "approval",
    "verification_record", "dependency", "failure_meaning", "replay_safety",
    "runnable_classification", "template_check",
}
for key, row in rows.items():
    assert fields <= row.keys()
    assert all(row[field] is not None and row[field] != "" for field in fields)
    assert row["wf_block"] == expected[key].block
    assert row["template_check"] == expected[key].check
registry = default_registry()
tools = {tool["name"]: tool for tool in audit["registered_tools"]}
assert len(tools) == 105 and tools.keys() == registry.tools.keys()
for name, tool in registry.tools.items():
    row = tools[name]
    assert (row["parameters"], row["scope"], row["effect"], row["move"], row["aliases"]) == (
        tool.parameters, tool.scope, tool.effect, tool.move, list(tool.aliases)
    )
    assert row["handler"] == tool.handler.__module__ + ":" + tool.handler.__qualname__
    assert row["line"] == inspect.getsourcelines(tool.handler)[1]
    guard = registry.guards.get(tool.scope)
    name_of_guard = type(guard).__module__ + ":" + type(guard).__qualname__ if guard else None
    assert row["guard"] == name_of_guard
    assert row["permissions_profile"] and row["repeat_safety"]
for path, expected_hash in audit["source_sha256"].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected_hash, path
assert len(generate()) == 105
print("PASS: exact workflow/tool/schema coverage and audited source hashes")
```
