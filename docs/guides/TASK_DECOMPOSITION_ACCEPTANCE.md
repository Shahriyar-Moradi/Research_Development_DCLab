# TP-02 — bounded task-boundary proposals

Status: implementation, scripted DEV comparison and independent TP-02 review complete; final gates and the owner-authorized replay compatibility repair are being verified.
Scope: TP-02 plus the replay compatibility repair explicitly authorized by the owner on 2026-11-11 after a required gate failure. TP-03 planning, task execution, persistence, recovery and product rollout remain later work.
Original working base: `602c82ad692c6b4af03e0332dcd292e6f6ce555d`. Implementation is now committed as
`c49b6790dcfaeb0c84e5055834165647a829a19c`; completion resumed on `69fcb92de5049b0e4397f965ada057cb2b8bcaa9`
on 2026-11-11 after other repository work. Unrelated changes and untracked files are preserved.

## Behavior and ownership

`dclab_rnd.tasks.decomposer.decompose` is an explicit, off-by-default Python API. It accepts the already validated
workflow, source plan, current project metadata, explicit confirmed goal, capability snapshot and deterministic
TP-01 graph. It recompiles the adapter before requesting a proposal. Changes to the goal, dataset, solution,
signoff, settings, source plan or capability metadata refuse the old graph before any model request.

The model proposes group membership and dependencies through a narrow `tp02.v1` JSON protocol. Code copies every
atomic task, preserving its objective, source steps, artifacts, capability, scope, checks, gates and blocker. Every
atomic task must occur exactly once. Group objectives are derived from their descendant tasks. Model-written
objectives, tool names, arguments and completion claims are not accepted fields. No source decision can be erased.
The dependency closure must retain every adapter prerequisite. The compiled TaskGraph then passes TP-01's full
validator. One physical engine operation still equals one action. Model grouping never splits the final stage.

For example, the model may group data, leakage, features and model selection together, with approval, final score
and reporting in another group. The physical chain remains data → leakage → features → models → approval → final
→ report. Branch or join edges may add prerequisites. They cannot remove a required dependency or introduce a cycle.
This package does not prove general branching workflows: the admitted TP-01 adapter remains basic binary only.

An unavailable model, provider error, invalid JSON, unknown field, stale context, unsafe dependency, ungrounded
group or exhausted budget returns the supplied deterministic graph unchanged. Missing source data may return the
adapter's `None` with an explicit blocker. A stale or blocked graph is inspectable, never execution permission.
Unsupported capabilities retain the adapter's blocker reasons. There are no repairs or retries.

## Integration

```python
from dclab_rnd.tasks.decomposer import DecompositionBudget, decompose

candidate, report = decompose(
    workflow, source_plan, current_project_metadata,
    deterministic_graph=adapter_graph,
    capabilities=capability_snapshot,
    confirmed_goal=confirmed_goal,
    enabled=True,  # explicit developer preview only; default is False
    gateway=workspace_gateway,
    budget=DecompositionBudget(cap_eur=0.2),
)
```

Callers must obtain metadata, confirmation and permissions from current authenticated host records. Model output
cannot supply any of these inputs. No product route or existing agent calls this API. Core remains 0.157.0; frozen
TaskGraph and TaskExecutionPlan v1 schemas, the portable wheel and main-product adapters are unchanged.
Model proposals require the normal gateway's `Bound` client, which implements the strict request controls.
Legacy adapters such as `TwinGateway` return the exact fallback before requesting: accepting similarly named
options is not proof that an adapter enforces their budgets.

The existing business `decomposer` builds ProblemMaps. `solution_planner` proposes solution decisions. Neither
owns execution task boundaries, so this package registers the separate `task_decomposer` purpose and harness role.
It uses the existing standard tier, routing, workspace privacy checks, usage ledger and gateway exchange capture.
`prompts/task_decomposer.md` is prompt version `tp02.v1`; normal prompt hash gates cover it. The request offers no
tools and promotes no source text into system policy.

One harness role step records the outcome, code-owned refusal codes, elapsed seconds and model request ID. Model
provenance points to that request. The gateway keeps the returned API thinking summary beside the reply where
workspace policy permits model text. It is labeled **API thinking summary**, requested with adaptive thinking and
`display: "summarized"` on Anthropic. No thinking or raw model prose enters the task graph or role step. The new
purpose has a dedicated persistence test and a zero-cell privacy test.

## Bounds and limits

Default bounds are 32 nodes, depth 4 (root included), 48,000 UTF-8 input bytes, 16,000 reply bytes, 1,800 requested
reply tokens, 9,800 total output/thinking tokens, one provider attempt, 30 seconds and a €0.20 reservation cap.
Host bounds have fixed upper limits. Unknown prices fail closed; the gateway reserves conservatively using input
UTF-8 bytes and the transport's full output limit, including thinking headroom and endpoint token floors.

Optional gateway request controls leave existing callers' retries and shadow behavior unchanged. TP-02 opts into
one attempt, which disables gateway retries, provider compatibility retries and shadow requests. Its request timeout
overrides a tier timeout floor without mutating the transport. A POSIX real-time alarm interrupts synchronous model
planning at its deadline and restores the caller's signal handler afterward. A sleeping transport is interrupted;
no background request continues after return. A caller with an active alarm, a worker-thread caller or a platform
without this deadline facility gets deterministic fallback before a model request. This restriction is explicit for
the developer preview. Product integration needs its own cancellable host before enabling model proposals there.
Budget cleanup is owned before reservation acquisition. A deadline during a slow cap callback, reservation handoff
or the tiny hold update releases that call's funds. Releases are idempotent and call-local. A Python-level critical
section defers expiration until the in-memory lease transaction commits, then raises immediately. This also handles
process alarms delivered through another OS thread. Callbacks and model requests remain interruptible.

Input metadata deliberately excludes table rows, profiles, previews, category values, target class values, paths,
credentials and unrelated project fields. Column names and confirmed goal/workflow text are allowed source metadata
and are treated as untrusted content. They must not be used as a channel for raw table values. Unresolved questions
block the request, so the model cannot guess the missing answer. No sealed set or live endpoint is used here.

## DEV comparison

Command:

```bash
DCLAB_NO_LIVE_MODELS=1 .venv/bin/python -m dclab_rnd.tasks.evaluate --out evaluation/task_decomposition_tp02_dev.json
```

The recorded result uses one authored Telco metadata fixture in nine scenarios. The fixture's dataset hash and
human signoff are test declarations, not real file evidence. Three valid scripted proposals are accepted. Three
deliberately invalid replies are refused. Missing capability, missing data and changed objective block before a
model call. Six of six non-proposal outputs equal the deterministic result, including `None` for missing data.

Structural source coverage matches the deterministic adapter in every scenario: nine of nine mandatory steps
where a graph exists, zero of nine for missing data. Explicit blocked nodes count as represented source work,
not executable work or completion. The deliberately injected invalid-proposal rate is 3/6 (50%). It measures the
checker under these scripts, not a real model's failure rate. The sample has one independent source fixture, so
no general quality estimate or model superiority claim is justified. Cost is zero, live calls zero, tasks run zero.
The gate was declared in the runner before recording the result. TP-06 promotion is not evaluated.

## Verification ledger

- Test-first evidence: `python -m unittest tests.test_task_decomposer -v` failed before implementation because
  the decomposer module did not exist. The initial seven tests then exercised success and exact fallback.
- The expanded task-decomposer suite passed 30 tests before review; deadline and later-revision regressions were
  added after review.
- A first 264-test focused run passed nine files and failed the purpose inventory in the data-limit suite.
  A real marked-cell test was added for the new purpose; the inventory gate was retained.
- An earlier direct multi-file unittest invocation failed because shared environment state affected model
  settings. The repository's isolated test runner is used for verification; that failed invocation is not a pass.
- Independent review found that a valid revision-2 baseline was wrongly compared with revision 1, and that an SDK
  timeout did not enforce a whole-call deadline. A failing revision regression reproduced the first finding.
  Freshness now recompiles at the supplied baseline revision. An interrupting deadline, thread refusal and caller
  alarm preservation tests cover the second. The completion review below covers the subsequent fixes too.
- The re-check found a reservation leak when the deadline interrupted acquisition/handoff. A warmed regression
  reproduced the leaked hold before the fix. Call-local cleanup now owns the lease before money is reserved;
  three regressions cover acquisition, handoff and an alarm during the hold update. All 38 task tests and all 41
  gateway tests pass together. The first broad gate was deliberately interrupted and will be restarted on this fix.
- A further independent check found that masking main's OS signal alone missed process alarms delivered through
  another thread. A real-timer regression reproduced the leak. The deadline handler now recognizes a Python-level
  budget critical section and raises immediately after its transaction commits. The same real-timer regression
  verifies that no money remains held and no provider call runs. This final finding is fixed and tested locally.
- The broad `make check-quick JOBS=3` pass ran 4,545 tests across 393 files (604 units, 2,278.8 seconds) and found one
  missing registration: `store/sets/llm.py` had no stage for `task_decomposer`. Its inventory test was retained.
  The purpose now maps explicitly to `plan`, with a regression that carries the actual summary through shaping
  and set construction. All ten evidence/freshness gates and all five product flows passed in that run. The
  PostgreSQL stage did not start because the file pass failed. Ten existing skips were reported in the file pass.
- A subsequent full scan found three existing standard-tier override names used by the now-staged DEV runner
  missing from `.env.example`. Their optional empty defaults and provider inheritance are now documented; the
  unchanged configuration-inventory test passes. The full gate is restarted on the complete staged source set.
- After the 2026-11-11 restart, all 39 original task tests still passed. The resumed checkout exposed a newer Twin
  OpenRouter constructor outside the shared gateway factory and stale generated development data. The constructor
  now uses the existing shared factory, with a failing-before delegation regression. `make sft-dev` refreshes the
  corpus from the committed records; unrelated source records are retained.
- The final review reproduced a budget bypass through the real legacy `TwinGateway`: a single reported call could
  make two unpriced provider attempts at an effective 20,000-token limit. The new regression failed before the fix.
  Unsupported client implementations now return `bounded_client_unavailable` before any request. The refreshed
  focused run passes 190 tests in ten files, including all 40 task tests and summary-retention checks, with no skips.
- Final independent source review: **PASS**, no remaining correctness findings. The reviewer independently ran
  all 40 task tests, both gateway-construction checks and the marked-cell privacy check. The current harness
  evaluation (`HRN-114`) passed all 17 cases at zero cost. The nine-case DEV comparison passed again, with zero
  live requests or task execution. Broad package and repository gates still need their final recorded results.
- The resumed full `make rd-check JOBS=6` ran 5,447 tests in 476 files (733 units, 1,986.9 seconds), with 14 existing
  skips and two failures. All ten evidence/freshness gates passed. An existing routing override lacked configuration
  documentation; its empty default is now documented and the unchanged inventory passes. The other failure was the
  existing implicit final replay conflicting with the newer consumed-holdout receipt guard.
- The owner explicitly authorized the replay compatibility repair outside the original TP-02 scope. Final replay
  now needs a fresh, nonblank `reuse_reason`, even for older records without receipts or a previously authorized run.
  The existing CLI accepts `--reuse-reason`. Code never invents permission from the old run. The copied receipt and
  split-drift guard remain intact; the original project, receipt and count do not change. A newly authorized replay
  preserves the numerical comparison but honestly reports `holdout_reuse` as a changed parameter. Replay regression
  tests failed before their corresponding fixes. The audited focused checks passed 93 tests in five files with no skips.
- Replay review found that the fresh permission reason disappeared with the scratch copy. The existing workspace
  audit now records a bounded, redacted reason and the original run before scoring, followed by a linked completed
  or failed outcome. The regression failed before the fix. It verifies the reason survives, an email is redacted,
  default refusal writes no permission entry, and split-drift refusal is recorded honestly. The audit keeps its
  existing best-effort semantics: failures are reported, and a script-created store without a workspace audit
  writes no audit. Durable permission provenance therefore requires the normal audited workspace host. Free text
  does not enter the run metadata or task graph.
- Final independent re-check: **PASS**, no correctness findings. The reviewer independently passed 37 tests with
  no skips: 28 lineage, six holdout safety, both gateway-construction checks and the unsupported-client regression.
- Final package/repository gates and isolated committed gate: pending.

## Files and remaining work

New files: `tasks/decomposer.py`, `tasks/evaluate.py`, `prompts/task_decomposer.md`, `tests/test_task_decomposer.py`,
the DEV JSON, this guide and the package development trajectory. Existing files register the role/purpose and its
training stage, add opt-in gateway/transport bounds and the privacy/summary-retention tests. Required evaluation
and corpus files are generated by commands. The authorized compatibility repair also changes `studio/lineage.py`,
`experience/__main__.py` and `tests/test_lineage.py`; `.env.example` documents an existing route override.

Task execution remains in the existing guarded registry, Studio engine, graph and Programme authorities. TP-02
does not grant approvals, change splits, read holdout, fit a model, write projects, expose UI/API or change defaults.
Live decomposition quality, useful hierarchy choices on other problem families, real latency and cost, durable
recovery and independent completion verification remain unmeasured and outside this package.
