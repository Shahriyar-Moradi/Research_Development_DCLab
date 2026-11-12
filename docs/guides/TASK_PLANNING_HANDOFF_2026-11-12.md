# DCLab hierarchical planning — continuation handoff

**Updated:** 12 November 2026, Asia/Dubai  
**Purpose:** Give another AI enough context to continue from the current implementation without repeating completed work or bypassing release requirements.

## 1. Executive status

The planning foundation, guarded deterministic task execution, verification, recovery, and read-only developer preview (TP-07) are implemented. **TP-08 is now complete: validated model-proposed task hierarchies can be admitted into guarded execution under double opt-in (`enabled=True`, `admit_proposals=True`).**

**Execution rollout remains disabled by default.** While the dispatcher can now execute admitted model-proposed task hierarchies, default product execution rollout requires passing the independent promotion gate evaluation.

Earlier chat statements that TP-07 was “not started” or “uncommitted” are superseded by this report. The checkout advanced while the handoff was being prepared. Do not recreate the preview or treat completed package tooling as a passed release gate.

## 2. Repository snapshot

```text
Repository:
  /Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D

Branch:
  main

HEAD:
  4c990288063d005ea03379f4cc7499e7744458c6

Commit:
  Expose verified task progress without enabling planner execution

Status before adding this report:
  Clean working tree.
  main matched its local origin/main tracking reference.
  No fresh remote fetch was performed for this handoff.
```

This handoff and `TASK_PLANNING_NEXT_STEPS.md` were committed on top of that snapshot (2026-11-12). An earlier chat draft described an older checkout (HEAD `4e28771c`, TP-07 uncommitted, tracker "Not started"); that draft is superseded by this file.

Creating this handoff adds a documentation file after that snapshot. Recheck HEAD and working-tree ownership before continuing; another session may advance the repository.

Recent relevant commits:

| Commit | Work |
|---|---|
| `4c990288` | TP-07 read-only Task Plan preview |
| `4e28771c` | Hypothetical-preference Home routing correction |
| `aa574ed0` | TP-06 evaluation and release gates |
| `d2185987` | TP-05 verification/recovery follow-up |
| `25bd1b3a` | TP-05 verification and bounded recovery |
| `f64f011c` | TP-04 guarded task execution |
| `7d6849c7` | TP-03 task execution planning |

## 3. Completed packages

| Package | Delivered | Boundary |
|---|---|---|
| TP-01A–E | Architecture, task contracts, frozen schemas, fingerprints, graph validation, deterministic adapter and foundation acceptance | Current adapter supports the basic binary workflow |
| TP-02 | Bounded model-assisted task decomposition, deterministic validation and exact fallback | Produces proposals; no execution during planning |
| TP-03 | Atomic execution plans, capability/source/permission/budget checks and optional model proposals | Planning calls no tools |
| TP-04 | Explicit opt-in execution through the existing guarded registry/engine, durable attempts, project locks and crash/resume handling | Uses the deterministic task graph |
| TP-05 | Independent scientific verification, bounded recovery, source-bound revisions and preserved verification history | No default execution rollout |
| TP-06 | Frozen evaluation harness, comparisons, reports and promotion gate | Tooling complete; promotion failed |
| TP-07 | Real Task Plan panel, read-only project API, owner preview setting, responsive UI and rollback | Preview complete; execution rollout blocked |
| TP-08 | Validated model-proposed task hierarchy admission into guarded execution, structural leaf checks, hierarchy bounds, tamper receipts, re-grounding across source updates | Complete; admission works under double opt-in (`admit_proposals=True`), execution remains default-off |
| TP-09 | Cancellable product hosting for model planning in worker threads and background jobs, PlanningHost context, jobs.worker Stopped exception preservation | Complete; 9 host tests, 366 total planning tests pass |

The separate **R-01 Home routing audit remains pending verification**. A focused routing correction is complete; that is not completion of the broader audit.

Existing owners remain authoritative: SolutionTeam, the tool registry, studio graph, scientific engine, project records and checkpoint storage. Extend them rather than create another orchestration framework.

## 4. What the user can see now

Enable the preview through the existing owner settings:

1. Open **Admin → Teams**.
2. Select **Read-only Task Plan developer preview · no execution**.
3. Save.
4. Open a real project at **Workflow → Overview → Task Plan**, or use Home’s Task Plan link in the built-project context.

The setting is `task_plan_preview`; its default is false. Clear it and save to roll back. No migration or project rewrite is required.

The panel shows task groups, dependencies, planned/running/completed/blocked/stale status, approvals, evidence links, capabilities, source workflow steps, revisions, preserved work and recorded execution budget. Execution budget figures are distinct from model-planning request costs.

The API is:

```text
GET /api/projects/{project_id}/task-plan
```

It uses existing project access checks and storage-enforced read-only access. It does not start, retry, reconcile, verify or approve work. Refreshing it cannot execute tasks or create checkpoint storage.

A completion label requires matching saved independent verification and current owner evidence. Missing proof blocks descendants; changed evidence makes progress stale. Existing projects without a task journal receive a metadata-only preview without invented completion claims.

Implementation entry points:

```text
dclab_rnd/tasks/view.py
dclab_rnd/agentic/routers/projects.py
dclab_rnd/orchestra/settings.py
dclab_rnd/agentic/web/src/views/project.html
dclab_rnd/agentic/web/src/views/home.html
dclab_rnd/agentic/web/src/views/admin.html
dclab_rnd/agentic/web/src/look.css
tests/test_task_plan_view.py
scripts/browser_e2e.py
```

## 5. Validation already recorded

These are existing recorded results. No application tests were rerun just to write this handoff. Counts overlap across packages and must not be summed into a unique total.

- **Planning regression diagnostic:** 315 tests across ten planning groups passed, with zero failures/skips, including actual file/PostgreSQL cases.
- **TP-07:** 57 distinct application checks passed: 33 presenter/API checks and 24 existing team/web checks.
- **Browser:** Four final checks passed: Developer and Data science modes, each at 1280 and 375 pixels.
- **Frontend:** `make web` passed; generated assets and the API client were rebuilt.
- **Development corpus:** `make sft-dev` and all four validation/freshness checks passed, with zero refused examples.
- **Independent TP-07 review:** Passed after proof-propagation and adjacent-revision fixes; the final focused regression passed.
- **TP-08 graph admission:** 36 dedicated tests passed in `test_task_admission.py` (FileStore and PostgreSQL), 357 tests passed across all 11 task planning test files. Adversarial review resolved 4 confirmed findings (unvalidated reconstruction, unrooted trees, unhandled Blocked exceptions, provenance tampering detection). Zero spend on live models. All 10 `rd-gates` passed.
- **TP-09 cancellable planning host:** 9 dedicated tests passed in `test_task_host.py`, 366 tests passed across all 12 task planning test files. Supports worker threads and `jobs.worker` Stop exceptions without swallowing. Zero live model spend.

Recorded regressions include missing upstream proof leaving descendants completed, incorrect preserved-work summaries, incorrect adjacent-revision comparisons, and task status rendered through the wrong UI helper. Failing-before records were retained.

Evidence locations:

```text
docs/guides/TASK_PLAN_PREVIEW_ACCEPTANCE.md
agent_runs/live_tests/tp07_20261112/checks.jsonl
agent_runs/live_tests/tp07_20261112/screenshots/status-fixed-dev/
agent_runs/live_tests/tp07_20261112/screenshots/status-fixed-ds/
training_records/test_runs/2026-11-12/
```

Resource limits are documented. The browser keeper refused complete temporary-workspace copies below its existing 2 GB free-disk threshold. Actual screenshots and explicit invented project/checkpoint/owner JSON fixtures were retained separately. An initial missing Chromium revision and an initial long macOS socket-path failure were also retained beside successful reruns.

## 6. Completed bounded live-model diagnostic

Report:

```text
agent_runs/live_tests/planning_20261112_budget2/REPORT.md
```

The separately authorized ceiling was **USD 2**. The recorded model was `qwen/qwen3.8-flash`.

- Four fresh invented binary scenarios.
- 27 actual requests: four accepted decomposition proposals and 23 accepted task-plan proposals.
- Three guarded workflows completed, each with one holdout read.
- The changed-contract scenario stopped after data, with zero holdout reads.
- No unauthorized, repeated or leaking handler was observed.
- Planning offered no tool schemas.
- All 27 whole replies and returned reasoning records were retained.
- Provider-catalog token-priced cost: **USD 0.025211**.
- Permanently reserved worst-case exposure: **USD 0.08684772**.

**The task plans executed on the deterministic graph. Model-proposed hierarchies were refused without effects. The full C execution arm is still unmeasured.**

Run-local verified prices and an independent USD reservation counter enforced the cap. USD values occupied the gateway’s EUR-named fields only inside this diagnostic. Application-wide prices were not changed. These are token-accounting figures, not an invoice.

The diagnostic is finished. Do not rerun it unnecessarily or assume its previous spending authorization covers a new campaign. Raw exchanges remain in the opted-out evaluation archive; the development trajectory contains aggregate engineering findings.

## 7. Permanent records

Initial diagnostic snapshot:

```text
training_records/snapshots/20261112T115806Z/
```

Final collection including TP-07 evidence:

```text
training_records/snapshots/20261112T162819Z/
training_records/snapshots/20261112T162819Z/MANIFEST.json
training_records/INDEX.json
```

The final snapshot and index entry were present at inspection. Records use the existing append-only collection with per-file hashes. Preserve failed runs and resource limitations alongside successful results.

## 8. Remaining blockers

### A. Admit validated model-proposed hierarchies

The dispatcher reconstructs the deterministic graph and compares it with the stored graph. A valid proposal with different hierarchy/dependencies therefore cannot execute.

Inspect:

```text
dclab_rnd/tasks/dispatch.py
  TaskDispatcher._initial
  TaskDispatcher._sources

dclab_rnd/tasks/decomposer.py
dclab_rnd/tasks/planner.py
dclab_core/task_validation.py
```

Implement admission without weakening immutable atomic operations, mandatory checks, source bindings, approvals, capability restrictions, budgets, independent verification or crash/resume safety. Preserve deterministic fallback. Do not “fix” admission by simply deleting the source/graph equality guard.

### B. Provide cancellable product hosting

The current decomposer deadline supports synchronous POSIX main-thread callers and refuses unsupported thread/deadline contexts. Product integration needs an appropriate cancellable host using existing jobs, workers, cancellation and budget mechanisms.

The read-only preview does not solve this execution integration requirement.

### C. Obtain passing release evidence

The authoritative frozen result remains:

```text
evidence/campaigns/task_planning_v1/results/TPB-002_dev_offline.json
```

**Verdict: FAIL — not promoted.**

The deterministic candidate recorded **45/48 correctness**, below the required **95%**. The full model-assisted arm remains unmeasured. The sixteen authored DEV cases include only five completion cases and two recovery cases; repeating seeds does not create independent cases.

The later Home routing correction passed 109 relevant tests. It does not rewrite the frozen result or establish unseen improvement. Current source differs from the frozen source closure; a new measurement requires an appropriately versioned protocol/freeze.

Existing release requirements include:

- Zero measured critical safety violations.
- At least 95% candidate correctness, with no regression on a correct baseline outcome.
- At least 24 independently authored completion cases or 12 recovery cases.
- At least 0.10 absolute completion or recovery gain against the original workflow, with the paired 95% interval strictly above zero.
- For C, elapsed time at most twice the baseline on each matched case-seed, at most eight extra requests, and at most EUR 0.05 model cost per case-seed.
- Adequate configured-model evidence and complete safety/provenance instrumentation.

These requirements are not authorization to spend. Read the full protocol before designing the next evaluation; do not weaken thresholds or redefine success after seeing outcomes.

### D. Complete controlled execution rollout

The delivered preview switch authorizes visibility only. Execution needs reviewed integration and passing promotion evidence before any rollout. Preserve explicit owner control, auditability, applicable human confirmations and straightforward rollback.

### E. Complete the separate R-01 routing audit

The hypothetical-preference correction is scoped and complete. Broader routing stabilization/verification remains separately tracked. Do not mark R-01 complete from the targeted fix.

## 9. Recommended continuation order

1. **Confirm the current snapshot.** Read AGENTS.md, HEAD, working-tree changes, tracker and acceptance records. TP-07 is committed; do not rebuild it from scratch.
2. **Define and implement a focused proposed-graph admission follow-up.** Add failing-before regressions for valid/invalid proposals, stale sources, preserved gates, approvals, checkpoint/resume, unknown effects and duplicate work.
3. **Integrate a cancellable model-planning host.** Reuse current jobs/workers and model budget controls.
4. **Complete applicable routing verification.** Preserve the completed narrow correction and accurately scope outstanding R-01 work.
5. **Freeze a new independent release evaluation.** Use sufficient independently authored cases, preserve earlier failures, and obtain explicit authorization plus a total cap before additional live calls. Do not run sealed data without explicit authorization.
6. **Evaluate and inspect actual evidence.** If the gate fails, record it and improve the implementation. A passing unit suite cannot substitute for the release comparison.
7. **Roll out execution only after its gate passes.** Finish audited controls, rollback and documentation, and verify the affected complete user journeys.

Use separate reviewable packages rather than combine all follow-ups into an unbounded change.

## 10. Standing instructions

- The owner wants the complete feature and clear English reporting.
- Follow the owner’s focused-test instruction: run only affected tests and safety checks. Broad repository/full PostgreSQL gates require explicit owner instruction.
- Do not repeat passing tests unless changed code, a failure or a new concern justifies it.
- Never edit old results. Append new freezes/results. Never tune on sealed evaluation data.
- Rebuild generated API clients, static assets, schemas/corpora and other generated files through their owners.
- Keep preprocessing/selection inside training folds. Preserve the single holdout read and existing human approval boundaries.
- Completion requires current measured evidence, not an agent’s assertion.
- Preserve human choices, provenance and the ability to stop/resume safely.
- Keep raw rows, secrets and hidden reasoning out of task/product responses. Preserve required whole model exchanges through the gateway with applicable redaction and consent.
- Keep portable contracts/rules in dclab_core and host wiring in dclab_rnd; apply integration/adapter requirements where relevant.
- Apply the dclab-package skill and required independent review for package implementation. The owner’s targeted-test instruction supersedes older blanket broad-gate requirements.
- Do not push, deploy or enable execution merely because this handoff lists them as future work. Check the current authorized scope.
- Do not claim production readiness from unit tests or small invented diagnostics.

## 11. Reading list

Start with:

- [Planning tracker](../../engineering/planning/README.md)
- [Master specification](../../engineering/planning/DCLab_Hierarchical_Planning_Codex_Pack.md)
- [TP-07 package instructions](../../engineering/planning/TP_07.txt)
- [TP-07 preview acceptance, enablement and rollback](TASK_PLAN_PREVIEW_ACCEPTANCE.md)
- [Evaluation acceptance and failed promotion](TASK_PLANNING_EVALUATION_ACCEPTANCE.md)
- [Frozen release protocol](TASK_PLANNING_EVALUATION_PROTOCOL.md)
- [Targeted Home routing correction](HOME_INTENT_HYPOTHETICAL_FOLLOWUP.md)
- [Guarded execution acceptance](TASK_EXECUTION_ACCEPTANCE.md)
- [Verification and recovery acceptance](TASK_RECOVERY_ACCEPTANCE.md)
- [Task execution planning acceptance](TASK_EXECUTION_PLANNING_ACCEPTANCE.md)
- [Core integration](CORE_INTEGRATION.md)

Relevant implementation/tests:

```text
dclab_core/task_graph.py
dclab_core/task_execution_plan.py
dclab_core/task_fingerprints.py
dclab_core/task_validation.py
dclab_core/task_verification.py
dclab_core/task_recovery.py

dclab_rnd/tasks/adapter.py
dclab_rnd/tasks/decomposer.py
dclab_rnd/tasks/planner.py
dclab_rnd/tasks/dispatch.py
dclab_rnd/tasks/verifier.py
dclab_rnd/tasks/view.py
dclab_rnd/orchestra/solution.py
dclab_rnd/agent_eval/task_planning/

tests/test_task_graph.py
tests/test_task_execution_plan.py
tests/test_task_validation.py
tests/test_task_adapter.py
tests/test_task_decomposer.py
tests/test_task_planner.py
tests/test_task_dispatch.py
tests/test_task_verifier.py
tests/test_task_recovery.py
tests/test_task_planning_eval.py
tests/test_task_plan_view.py
```

The existing `make task-planning-eval` is the old frozen offline DEV comparison/probe entry point. It is not a ready-made full C promotion run and must not be forced past a source-freeze mismatch.

## 12. Copyable instruction for the next AI

> Continue the DCLab hierarchical planning work from the actual current checkout. Read AGENTS.md and this report, then verify HEAD and the working tree. TP-01 through TP-06 tooling and TP-07’s default-off read-only preview are implemented; TP-07 is committed as 4c990288. Preserve the completed work. The next engineering gap is safe admission and cancellable hosting of model-proposed task graphs. Full C execution and passing independent promotion evidence are missing. Implement focused, reviewed follow-ups with relevant tests only; retain historical failures, do not spend or run sealed evaluations without explicit authorization, and keep execution rollout blocked until its release gate passes.
