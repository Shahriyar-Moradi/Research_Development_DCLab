# TP-01 foundation acceptance report

Review date: 2026-11-05. Implementation baseline: `08c7bac622db6355a59e376eabc02f9489085d08`, plus the
completed TP-01D working-tree changes and TP-01E verification changes. This report closes only the architecture,
contracts, pure validation and deterministic binary adapter foundation. It does not authorize task execution.

**Final verdict: TP-01E and the TP-01 foundation are complete. All nine checklist items are satisfied.** Code
gates, final independent review and completion-corpus checks passed. TP-02 may begin as a separate package; it has
not started. This accepts the planning foundation, not a task executor or product rollout.

## Architecture reviewed and reused

The detailed [architecture and authority decisions](TASK_PLANNING_ARCHITECTURE.md),
[capability matrix](TASK_CAPABILITY_MATRIX.md) and [expanded source audit](../../engineering/planning/TP_01A_EVIDENCE.json)
map the existing system. TP-01E rechecked all 105 canonical registry tools, including schemas, aliases, scope,
effect and move, and all 105 mandatory family/step pairs against the current checkout. Those inventories still match.
The original audit's commit, version and contract count remain historical; they have not been rewritten.

| Component | Responsibility and integration boundary |
|---|---|
| ProblemBrief / ProblemCard / ProblemMap; conversation, mapping and coordinator | Understand the business problem, identify sub-problems and blockers, confirm the current source. A task graph references that source; it does not redefine it. |
| SolutionPlan / signed studio Solution | Propose decisions and preserve their human adoption. The existing Solution signature stays compatible; separately consumed execution settings receive separate fingerprints. |
| SolutionWorkflow / SolutionMap | Authoritative scientific ordering, mandatory checks and one shared split/composition holdout. Their existing validators run before new graph admission. |
| TaskGraph v1 | Immutable task hierarchy, dependencies, capability references, source coverage and evidence obligations. Groups organize tasks; they never run. |
| TaskExecutionPlan v1 | Bounded proposal for one task, typed input references, actions, checks and limits. Its shape-ready status grants no execution permission. |
| Registry / Policy / ProjectGuard / studio.graph | Authorize actual tool calls, validate their schemas and workflow moves, enforce human gates. Registry membership alone grants no permission. |
| Studio engine / kind machines / Programme | Own training, preprocessing inside folds, validation selection, final evaluation and holdout accounting. Task planning adds no parallel engine. Programme leaves remain inside their existing atomic owner. |
| WorkflowRun | Derived progress/evidence readback built from stage records. It is not an execution controller; an inferred report-done state does not prove an export artifact exists. |
| Stage/programme records / SolutionTeam / existing checkpointers / jobs | Own mutable execution progress, orchestration, checkpoints and cancellation. New structural contracts do not replace these records or persist task attempts. |
| Storage / RunRecord / experience / ledger / traces / gateway | Own project truth, scientific evidence, provenance, decisions, observations and model exchange capture. Task declarations are obligations, not measured completion. |

| Authority | Existing owner retained |
|---|---|
| Current table bytes and input truth | studio.data.data_path and storage hash checks; supplied metadata must be current |
| Execution state and mutable run status | Stage/programme records, jobs and existing orchestration/checkpoints; WorkflowRun only reads derived evidence |
| Principal, workspace and tool authorization | API/router roles, bound runtime context, Policy, registry and scope guard |
| Human decisions and approval provenance | Signed Solution/brief, studio.graph approval records and original human API path |
| Holdout count and irreversible evaluation | Studio final/Programme and their original project counters and approvals |
| Spend and stop | Runtime Policy, model gateway/live guard/control limits and job checkpoints |
| Artifact provenance and completion evidence | Existing stage records, artifact hashes, RunRecord, ledger and experience writers |

## Physical execution boundaries

The [TP-01D adapter guide](TASK_ADAPTER_ACCEPTANCE.md) documents the exact metadata-only API and Telco fixture.
The fixture is authored metadata, with explicitly synthetic digests and signoff. It is not a real table run or a real
approval. The graph contains 12 nodes: a root, four groups, six action tasks and one human gate. Nine source steps
map to seven non-group tasks without duplicating physical effects.

| Task / source coverage | Classification | Existing physical owner / limit |
|---|---|---|
| data: class_balance, lock_holdout, train_only_analysis | Physically callable stage with implicit scientific work | One run_stage(data). Split/preprocessing and analysis stay inside the engine. Per-fold class-count proof remains a verification gap. |
| leakage: leakage_audit | Physically callable stage | One run_stage(leakage), before models. Human corrections retain their provenance. |
| features: no abstract source step | Required physical predecessor | One run_stage(features), because the current engine requires it. No fabricated source step. |
| models: choose_on_validation | Physically callable stage | One run_stage(models); selection uses training/validation evidence. |
| holdout: holdout_gate | Human gate; no tool binding | Original approval owner must supply current authenticated evidence. The planner cannot approve itself. |
| final: threshold and final_once | One physically callable stage with implicit threshold selection | One run_stage(final). Threshold uses OOF training predictions; holdout scoring occurs once. Optional threshold assessment is not guaranteed evidence. |
| report: report_limits | Physically callable legacy export | export_notebook, move capture. Existing report.md/notebook.ipynb; not a claim of canonical SolutionReport completeness. |
| Other families, programmes, extra decisions/steps, missing identities or capabilities | Explicitly blocked | No invented tools, source references, table reads or leaf schedulers. Some identity failures return no graph; diagnostic blocked graphs are not admitted for execution. |

These are existing callable owners. **No task dispatcher or product route invokes them through this new graph.**
The adapter returns execution_authorized=false. Missing proof, stale inputs, unsupported schemas and missing
capabilities remain blockers. A logical artifact name never creates another split, table file or scoring action.

## Contracts, schemas and dependency boundary

Core remains **0.157.0**. TP-01B introduced the contracts in 0.156.0; TP-01C added pure semantic checks and hashes
in 0.157.0. TP-01D/E do not change the portable public shapes or add dependencies.

| Contract | Published current schema | Frozen baseline |
|---|---|---|
| dclab_core.task_graph.v1 | dclab_core/schemas/task_graph.json | dclab_core/schemas/baseline/dclab_core.task_graph.v1.json |
| dclab_core.task_execution_plan.v1 | dclab_core/schemas/task_execution_plan.json | dclab_core/schemas/baseline/dclab_core.task_execution_plan.v1.json |

There are 107 registered contracts after loading the schema registry and 262 current/historical JSON schema files.
make core-schemas regenerated them; all remained byte-identical to HEAD. The release number and changelog agree.
Nested task data are frozen typed contracts and tuples, with explicit supported versions and bounded fields.
Canonical SHA-256 identities cover the declared graph/plan/workflow/capability content. Current host snapshots,
approved source identities and verified receipts remain required; a hash is not authorization.

The isolated wheel installed on cached Python 3.12.13 with exactly four direct dependencies: numpy, pandas,
scikit-learn and pydantic. It exposed both new contracts and exercised canonical roundtrips and unadmitted-source
refusal, with no unexpected distribution or host runtime import. The host adapter stays in dclab_rnd.tasks.
Main-product integration continues through the [existing adapter/conformance boundary](CORE_INTEGRATION.md).

The lean agent environment is Python 3.13.7 and has no numpy, pandas or sklearn. New fresh-process tests forbid
even attempted eager imports of dclab_core, numpy, pandas, scipy and sklearn. Harness imports and in-memory metrics
still work. Negative controls reproduce the old eager-core import failure. No production import refactor was needed:
agent_metrics already imports bench lazily inside eval_rows.

## TP-01 checklist

The TP-01E prompt calls this a seven-item checklist. The actual master contains **nine** items. All nine are retained.

| Master criterion | Evidence | Status |
|---|---|---|
| Complete Workflow/Map/WorkflowRun/stage/Programme mapping; no duplicate effect | Architecture, matrix, 105-pair audit recheck, adapter many-to-one source coverage | Satisfied |
| Explicit input/state/authorization/approval/holdout/budget/provenance ownership | Authority tables above and architecture sections 3–6 | Satisfied |
| Typed, versioned, registered and schema-frozen TaskGraph/TaskExecutionPlan v1 | Contract tests, generated schemas/baselines and installed wheel | Satisfied |
| Deterministic dependencies/cycles/coverage/scope/safety validation | task_validation tests, including adversarial graphs and conservative readiness | Satisfied |
| Stable canonical fingerprints and stale-input refusal | Serialization/reordering, changed data/settings/capability/source/receipt regressions | Satisfied |
| One binary workflow maps without a model call or holdout read | Real builder/registry fixture and explicit zero-call spies, including stored handlers and both persistence owners | Satisfied |
| Negative graphs and lean agent compatibility pass | Task suite, actual lean interpreter and import-blocker negative controls | Satisfied |
| Honest existing regression/schema/wheel results | Verification ledger below; complete file/PG milestone gate passed, with every skip classified | Satisfied |
| Reviewable acceptance report deciding TP-02 readiness | Final reviewer PASS, completion corpus verified, all nine criteria covered and later runtime limits explicit | Satisfied |

## Verification ledger

All tests use offline/scripted transports. No live model, real sealed benchmark run or external API is authorized
or executed. Unit tests of holdout/once-only rules use their temporary test fixtures.

| Command / check | Actual result |
|---|---|
| .venv/bin/python -m unittest tests.test_task_adapter tests.test_agent_import_boundary -q | PASS: 43 tests in 3.685s, no skips. |
| .venv-agent/bin/python -m unittest tests.test_agent_import_boundary -v | PASS: 2 tests in 0.843s, no skips. Same 2 tests also passed in .venv in 1.986s. |
| Deliberate swallowed-write fault injected into the adapter's metadata helper | PASS: strengthened purity test fails as expected on the spy's call count. No real write occurred. |
| Offline .venv-agent/bin/python -m pytest tests/test_agentic.py -q | PASS: 12 passed, 1 skipped in 15.61s. Skipped test_local_api_security_and_missing_key requires pandas/product server; not claimed passed here. One existing Starlette deprecation warning. |
| Offline .venv-agent/bin/python -m unittest tests.test_live_eval -v | PASS: 3 pure grader tests in 0.001s, no skips. These grade supplied synthetic answers; no live model request or evaluation run occurred. |
| Current registry/workflow inventory versus TP-01A audit | PASS: all 105 tool identities/schemas and 105 mandatory family/step pairs match. |
| make core-schemas; unchanged schema/changelog diff check | PASS: 262 schema files unchanged, 107 registered contracts after schema-module loading. |
| UV_OFFLINE=1 make core-wheel-check | PASS: core 0.157.0, Python 3.12.13, both task contracts, hash/admission smoke, four direct dependencies, no unexpected distribution. |
| Focused existing/new foundation regressions (exact invocation below) | PASS: 370 tests in 15 files / 40 units in 306.5s, no skips. |
| make check-quick JOBS=3 | PASS: file backend 266 tests / 23 files in 143.0s (1 existing NOOA skip); PostgreSQL 96 tests / 6 files in 67.4s (no skips); all ten evidence/freshness gates and all five product flows passed. |
| make check-all JOBS=3 | PASS. File pass: 5,348 tests / 465 files / 529 units in 6,470.3s, 14 existing skips, no failures. Nested make test-pg: 5,350 tests / 465 files / 728 units in 2,892.6s, 11 existing skips, no failures. All ten evidence/freshness gates and all five product flows passed. The nested lean campaign suite passed 12 tests with 1 server-only skip in 9.06s. |
| Final independent dclab-reviewer review | PASS, no correctness finding. Independently ran 43 .venv tests and 2 lean tests. Report ownership finding fixed and confirmed. Final report/trajectory results, skip reasons, all nine criteria and bounded TP-02 handoff verified. |
| Development trajectory and generated corpus validation | PASS. All five staged generator files parsed with zero NUL bytes and preserved hashes after publication. Interim four-test check passed in 5.488s; reviewed completion refresh passed all four in 3.860s; final acceptance-record refresh passed all four in 3.741s. No skipped cases. |

Focused invocation:

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python tests/run_parallel.py --jobs 3 \
  test_task_graph test_task_execution_plan test_task_validation test_task_adapter \
  test_agent_import_boundary test_core_contracts test_core_experience test_solution_workflow \
  test_programme test_orchestra_solution test_graph test_agent_harness test_intern \
  test_home_orchestrator test_storage_contract
```

The focused storage contract cases used actual temporary file stores and isolated PostgreSQL worker databases.
The package gate's file-environment NOOA skip is not claimed passed there: the actual lean campaign suite above
covers that path. Its different, server-only skip is covered by existing ML-environment server/API regressions.

The milestone gate ran every rd-check component (the ten rd-gates and complete file suite) and invoked the actual
make test-pg target. Standalone make rd-check and a second standalone make test-pg were not additionally repeated.
The stored file and PostgreSQL run records both report ok=true, zero failed units and zero process-signal retries.
The long runtimes above are observations from this machine, not measurements of planner or product performance.
Existing macOS malloc-logging and dependency/deprecation warnings did not cause a test failure.

The complete file pass recorded these 14 pre-existing skips, confirmed from its stored run artifact and current
test conditions. No new task or import-boundary case skipped:

| Skipped file cases | Count | Reason / separate verification |
|---|---|---|
| test_agentic, test_live_eval | 2 | NOOA belongs to .venv-agent, not the ML interpreter. The campaign suite ran separately there; the pure live-evaluation graders are checked there separately without a live request. |
| test_core_wheel | 1 | Optional opt-in unit wrapper; the actual isolated wheel check passed separately. |
| test_consent, test_multi_instance, test_orchestra, test_store_lost | 5 | PostgreSQL-specific cases intentionally skip the file-mode pass; the PostgreSQL pass executed them successfully. |
| test_notebook_assist, test_notebook_eval | 3 | Existing frozen notebook manifest is not committed/present. These cases are not claimed passed or repaired by TP-01. |
| test_replay_compare, test_system_eval, test_trace_eval | 3 | Existing optional modules dclab_rnd.replay_compare, dclab_rnd.system_eval and dclab_rnd.agentic.trace_eval are absent. TP-01E does not implement unrelated legacy evaluation modules. |

The PostgreSQL pass recorded 11 existing skips: the same two NOOA module wrappers, one optional wheel wrapper,
three absent-manifest cases and three absent-module wrappers (nine total), plus the file-rewrite lock case in
test_consent and the file-store-specific route case in test_steps. Those two file-mode cases passed in the file
suite. The PostgreSQL-specific file-pass skips are therefore covered without weakening their guards.

The reviewer corrected one report error before acceptance: WorkflowRun had been grouped with mutable execution
owners. It is now explicitly documented as derived evidence readback. No production change was required.

The focused runner accepts basenames (test_task_graph), not tests/ paths. The first invocation was rejected before
any test ran; it was corrected without a source change. An initial registry inspection omitted the explicit schema
registration imports and observed only the base registry; loading dclab_core.schemas confirmed all 107 registrations.
The first sandboxed PostgreSQL availability probe lacked socket access; the permitted read-only probe confirmed
the existing dclab_test database. None of these command/environment errors was a product regression.

## Changes, limits and next-package handoff

| TP-01E file | Change |
|---|---|
| tests/test_agent_import_boundary.py | Two fresh-process boundary/negative-control tests; runs in both supplied Python environments. |
| tests/test_task_adapter.py | Strengthen two existing purity tests with mapped-result checks and explicit zero-call assertions on actual owners and stored handlers. |
| docs/guides/TASK_PLANNING_TP01_REPORT.md | This combined foundation report and verified acceptance decision. |
| engineering/planning/README.md | Verified package status and report handoff. |
| research/llm-fine-tuning/experiments/dev_trajectories/records/tp-01e.json | Decisions, findings/fixes, actual measurements, review and outcome. |
| research/llm-fine-tuning/experiments/sft/out_dev/ | Rebuild all five generated manifest/chat/alpaca files with the existing producer and verified publication. |

No runtime behavior, provider, prompt, dependency, Home/Lead Agent routing or default setting changes in TP-01E.
The completed TP-01D adapter, fixtures and documentation are retained and verified together; its guide lists those
files separately. Adapter/fixture hashes remained unchanged through both broad passes. The unrelated pre-existing
training_records/STORE_CATALOG.json edit is preserved and excluded from the package's review commit.

Future execution still needs durable attempts/effect receipts, crash reconciliation, current authenticated approval
and principal binding, current bytes and ambient runner/routing identity, concurrency control, cancellation and
evidence verification. An ambiguous final attempt must block rather than retry. Existing tabular records lack some
per-fold/row-membership proofs; legacy exports do not promise every source report criterion. These limits are explicit
and do not become solved by graph validation. Unsupported scientific families and programmes stay blocked.

The prior package observed unexplained damage to generated development-corpus files after regeneration. TP-01E
uses the existing devlog generator's supported --out option, validates JSON/JSONL bytes and replaces each published
file atomically before rerunning freshness tests. This is the same producer invoked by make sft-dev, with staged
output instead of direct publication. The damage origin remains unconfirmed; the generator is not redesigned here.

**TP-02 readiness decision: ready to begin separately, with the limits above retained.** Its scope is a model proposal
boundary using this deterministic foundation and fallback. A new model purpose must use the existing gateway,
enable thinking with display="summarized" for Codex, save the actual redacted API thinking summary beside the final
answer and prove capture with a test. It must preserve source/human decisions and
introduce no dispatcher, new execution framework or default rollout. TP-02 is not implemented in this package.

Review commit unit: **Keep workflow task previews read-only and verify the planning foundation**. It contains the
completed TP-01D metadata adapter and TP-01E verification, documentation and generated development records only.
