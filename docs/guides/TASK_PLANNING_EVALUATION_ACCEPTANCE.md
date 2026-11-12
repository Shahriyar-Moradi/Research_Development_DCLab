# TP-06 evaluation and release-gate acceptance

2026-11-12. Evaluation implementation and final independent review passed; promotion rejected.

## Recorded recommendation

**FAIL — not promoted.** The unchanged `tp06.gate.v1` rejects deterministic B's correctness of 45/48 (93.75%), below 95%. The shared ambiguous-intent case routed instead of clarifying on all three seeds. B regressed on no case A answered correctly, but that does not waive the absolute correctness requirement. C has no measured execution, and the independently authored completion/recovery sample floors are unmet. Nothing enables a new product default.

The authoritative comparison is [TPB-002 JSON](../../evidence/campaigns/task_planning_v1/results/TPB-002_dev_offline.json), with its [generated report](../../evidence/campaigns/task_planning_v1/reports/TPB-002_dev_offline.md) and [freeze v2](../../evidence/campaigns/task_planning_v1/freeze_v2.json). The [pre-run protocol](TASK_PLANNING_EVALUATION_PROTOCOL.md) gives the frozen rules, numerical rationale, existing ownership and reproduction command.

| Measurement | A: original team | B: deterministic tasks | C: model-assisted |
|---|---:|---:|---|
| Judged case-seed observations | 48 | 48 | 0; unavailable |
| Authored cases | 16 | 16 | Unmeasured |
| Correct observations | 42/48 | 45/48 | Unmeasured |
| Verified scientific completions | 15 | 15 | Unmeasured |
| Recovery observations successful | 6/6 | 6/6 | Unmeasured |
| Critical safety violations | 0 measured | 0 measured | No execution measurement |
| Runtime errors | 0 | 0 | No execution measurement |
| Live requests / EUR | 0 / 0 | 0 / 0 | No live requests |

Only five authored cases request scientific completion; their three seeds account for the fifteen completions in each arm. Two authored recovery cases account for the six successful recovery observations per arm. Routing and admission checks are not completed scientific projects. Seeds are repetitions, not new independent problems.

B's authored-case correctness difference against A is 0.0625, with paired clustered 95% interval [0.0, 0.1875]. Its completion difference is 0.0 [0.0, 0.0]; recovery difference is 0.0 [0.0, 0.0] on only two authored cases. The apparent correctness benefit comes entirely from B refusing an unsupported related-table dependency that A's prepare-only admission check accepts. It is not evidence about solving multi-table fraud. Both arms fail the ambiguous-intent case, which stays in the frozen set.

Measured mean elapsed time across authored cases is 0.3564 seconds for A and 0.532 seconds for B; the stored signed improvement is -0.1756 seconds [-0.3243, -0.0535]. These are local, instrumented, tiny synthetic trials and include boundary-only cases; they are not user-facing latency estimates. The generated report retains all paired metrics, safety categories and per-stratum outcomes rather than combining them into one favorable score. The existing Bench reader exposes the latest result as red.

## C integration limit

The scripted probe gets valid decomposition and planning proposals through the existing gateway. The dispatcher refuses the plan because its immutable identity belongs to the proposed graph, while the dispatcher admits only its reconstructed deterministic graph. It performs zero attempts, zero stage writes and zero holdout reads. Both scripted API thinking summaries are captured and explicitly labeled; no tool schemas are offered.

That diagnostic passes its protocol. It establishes neither real-model quality nor integrated execution. C's safety, correctness, cost and latency remain unmeasured. A future implementation must admit a validated proposal without weakening source or approval checks, obtain explicit live-model authorization and measure it against A on sufficient independently authored cases. Beating B on DEV alone cannot promote C.

## Preserved first run and harness corrections

[TPB-001](../../evidence/campaigns/task_planning_v1/results/TPB-001_dev_offline.json), its report and freeze v1 remain unchanged with `FAIL`. It recorded SQLite `OperationalError` in three observations while the disk had only 135 MiB free. The affected case/seed passed when rerun separately; no scientific engine or checkpoint error was suppressed or reclassified. Only this session's identified completed store-keeper test artifact was removed before the second run, recovering about 90 MiB. Low disk is a plausible resource cause, not a proven diagnosis from the retained exception type alone.

The first run also showed why unit tests had missed a provider leak: outside unittest, the owner's OpenRouter settings could be read from `.env`, changing the scripted summaries' labels. A new regression reproduces inherited provider settings; the offline environment now blanks the OpenRouter and legacy model keys explicitly. Freeze v2 records that harness change and the protocol amendment. Both freezes have exactly the same case hash, gate source hash and numerical gate values. The second run has no runtime errors and its probe validation passes. The observed routing failure remains scored as a failure.

Pre-run independent review found and resolved two further issues. The probe now isolates/restores the process-installed decision ledger and rejects an active workspace-services context before any effects. Dashboard summaries now expose observation and authored-case counts as numbers, avoiding automatic Wilson intervals that would count 48 correlated seed observations as independent trials. Both issues have failing-before regressions. The affected storage-export tests also exposed the missing explicit `task_planner` stage mapping; the planner now belongs to `plan`, with its recorded summary preserved by the same tested exporter as decomposition.

## Implementation and verification

New evaluation code is under `dclab_rnd/agent_eval/task_planning/`: immutable case definitions, promotion gate, real A/B owner operations, isolated scripted probe, freeze/result/report writer and DEV-only CLI. `make task-planning-eval` is the offline entry point. Existing Bench layer mapping and campaign keeper/LLM contamination guards exclude this evaluation from training. A small decomposer compatibility correction reads the owner's forbidden-column records as column metadata without leaking their freeform reasons.

No public contract, frozen schema, portable core dependency, API route, product flag or product default changes. Core's existing paired statistics and BenchRow reader are reused; there is no new main-product adapter contract. See the TP-06 section of [CORE_INTEGRATION](CORE_INTEGRATION.md).

Relevant application checks: **150 distinct tests passed**, with no skips, after fixes. The commands were:

- `DCLAB_NO_LIVE_MODELS=1 DCLAB_DECISION_SHADOW=0 .venv/bin/python -m unittest tests.test_task_planning_eval -q` initially passed 24 tests. Targeted reruns of changed `MeasuredRunnerTests` and `EvaluationRecordTests`, plus three added isolation/provider regressions, establish all 27 distinct final evaluation tests. The new ledger, embedded-context, provider and dashboard regressions failed before their fixes.
- `DCLAB_NO_LIVE_MODELS=1 DCLAB_DECISION_SHADOW=0 .venv/bin/python -m unittest tests.test_task_decomposer tests.test_bench tests.test_store_sets_llm tests.test_store_keepers tests.test_home_intent tests.test_home_routing_points -q` ran 120 tests: 119 passed initially and the purpose-stage completeness test failed. After adding the missing planner mapping, that test and the affected summary-export test both passed. Unchanged passing groups were not repeated.
- `DCLAB_NO_LIVE_MODELS=1 DCLAB_DECISION_SHADOW=0 .venv/bin/python -m unittest tests.test_orchestra_solution.SingleProblemTests tests.test_orchestra_solution.TeamTests.test_a_declined_holdout_reads_nothing tests.test_orchestra_solution.TeamTests.test_a_start_while_a_pause_waits_returns_the_pause_and_redoes_nothing -q` passed all three existing workflow/approval regressions.
- `make task-planning-eval` wrote two append-only comparisons. Both exit nonzero with `FAIL`; the second is a complete A/B measurement with a rejected gate, not a test harness crash.
- Result inspection confirmed 48 matched A/B rows, zero runtime errors in the second run, unchanged case/gate definitions across freezes, a passing probe protocol and a red BenchRow verdict. `git diff --check` passed.

The development trajectory is recorded and its generated corpus was rebuilt with `make sft-dev`. All four `tests.test_devlog` validation/freshness checks passed, with no refused examples in the corpus. Its initial no-new-number failure was corrected by putting the authored-case count into the measurement input, without weakening the corpus gate. Final independent review passed after recomputing the saved verdict and report, comparing the frozen hashes and checking these acceptance claims.

No broad gate, full PostgreSQL suite, live-model run, sealed benchmark, wheel rebuild or unrelated model evaluation was run. No portable core/storage schema changed; existing task-runtime file/PostgreSQL checks remain those of TP-04/05. The new benchmark executes real guarded owner operations on disposable file workspaces, not a new PostgreSQL acceptance run.

## Handoff

TP-06 delivers a functioning conservative measurement and rejection gate. It does not establish the gain required for rollout. Correct shared routing in a separately scoped change with its own relevant tests; never rewrite these outcomes. Then integrate and measure C under an appropriately versioned, independently authored protocol with explicit model authorization. TP-07 and automatic rollout remain blocked by the rejected promotion recommendation.

Work stays on `main` with a local commit and no push. The existing `dev` branch already has divergent history, so a fast-forward update is unavailable; this package does not reset or merge that unrelated branch.
