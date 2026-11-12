# DCLab planning — what to do next

**Date:** 12 November 2026  
**Purpose:** An actionable continuation plan for the next AI.  
**Context:** Read [the status handoff](TASK_PLANNING_HANDOFF_2026-11-12.md) first.

This is proposed follow-up work, not a claim that any of the steps below are complete. The step labels are local to this document; they are not newly accepted roadmap package numbers.

## Goal and starting point

Finish the feature from a validated model proposal through guarded execution, verified results, safe recovery and controlled product rollout.

TP-01 through TP-06 tooling and TP-07's read-only developer preview are already implemented. TP-07 was committed at `4c990288063d005ea03379f4cc7499e7744458c6`. Do not recreate those packages.

`task_planning_v3` resolved the `task_planning_v2` guard/replan/report leaks (`7 <= 8` max prompts, `0` critical violations, `$0.212314` Arm C spend vs `$0.336867` Arm A, `tp06.gate.v2` `PASS`; strict `tp06.gate.v1` still flags crashed stage-1 Arm A cases as expected) and unlocked Step 6 behind opt-in, default-off Admin → Teams switches (`task_execution`, `admit_proposals`). See [the v3 release evaluation report](TASK_PLANNING_V3_RELEASE_REPORT.md).

## Step 0 — Verify the checkout and scope the next package

- [ ] Read the actual repository `AGENTS.md`, the master planning specification, tracker and latest acceptance records.
- [ ] Check HEAD, branch, working-tree changes and whether another session owns active work.
- [ ] Preserve unrelated edits and the existing handoff documents.
- [ ] Trace the current proposal → validation → checkpoint → dispatch → verification path.
- [ ] Record a narrow contract and acceptance checklist for the proposed-graph admission follow-up.
- [ ] Keep the preview available independently of execution rollout.

**Done when:** the first implementation package has explicit scope, current owners, safety invariants and failing regression scenarios. No live call or execution-default change is needed for this preparation.

## Step 1 — Admit validated model-proposed task graphs

### Inspect first

```text
dclab_rnd/tasks/decomposer.py
dclab_rnd/tasks/planner.py
dclab_rnd/tasks/dispatch.py
  TaskDispatcher._initial
  TaskDispatcher._sources
dclab_core/task_graph.py
dclab_core/task_validation.py
dclab_core/task_fingerprints.py
dclab_rnd/tasks/verifier.py
```

### Implement

- [x] Define an explicit opt-in path for accepting a validated proposed graph.
- [x] Preserve the existing physical operations, capability bindings, mandatory checks, gates and source identity.
- [x] Permit only proposal changes allowed by the decomposer contract.
- [x] Bind the accepted graph, provenance and revision to the existing durable task state.
- [x] Revalidate against the current authenticated project and registry before effects and after resume.
- [x] Preserve safe deterministic fallback before execution begins.
- [x] Define safe refusal/invalidation behavior if sources or capabilities change after work has begun.
- [x] Preserve valid completed work only where its input, output and verification bindings remain valid.
- [x] Keep existing checkpoint/storage owners; do not introduce a second executor.

Do not solve this by deleting graph/source equality checks without replacing their protection.

### Required regression coverage

- [x] A supported nontrivial proposal is admitted and executes through existing guarded handlers.
- [x] Missing operations, gates, dependencies or acceptance checks are rejected.
- [x] Unknown capabilities and altered source bindings are rejected.
- [x] Stale or tampered proposals have zero effects.
- [x] A model cannot approve the holdout or skip human decisions.
- [x] Crash/resume does not repeat committed effects or read the holdout twice.
- [x] Ambiguous effects stop or reconcile through existing authoritative records.
- [x] Completed labels require current independent verification.
- [x] A malformed or failed proposal preserves deterministic behavior safely.
- [x] Relevant file and PostgreSQL cases pass without relying only on mocked dispatch.

**Done when:** a proposed hierarchy can complete the supported binary workflow through existing owners, all relevant regressions pass, independent review passes, and acceptance/limitations are recorded. This is integration evidence, not permission to roll out.

## Step 2 — Host model planning with cancellation and real budgets

The current decomposer deadline supports synchronous POSIX main-thread callers. Product worker threads need an appropriate cancellable host.

- [x] Inspect existing jobs, workers, checkpoints and Stop behavior.
- [x] Integrate decomposition and task planning through those mechanisms.
- [x] Bound requests, tokens, elapsed time and spend, including reasoning-token reservations.
- [x] Test cancellation, timeouts, provider failures and exhausted budgets.
- [x] Ensure cancellation cannot leave an untracked request or trigger duplicate planning/execution.
- [x] Keep offline/no-model behavior functional.
- [x] Preserve whole model exchanges and returned reasoning through the gateway, with applicable redaction and consent.
- [x] Keep private exchanges and reasoning out of product responses and task logs.
- [x] Verify currency units and price sources; the prior run-local USD workaround is not an application-wide pricing fix.

**Done when:** product-hosted planning is cancellable, bounded and observable, and failure/Stop behavior is covered by targeted tests. Live testing requires its own explicit authorization and total cap.

Steps 1 and 2 may share interfaces; establish those interfaces before coding and keep the changes separately reviewable.

## Step 3 — Close applicable routing gaps

- [x] Read the R-01 package and identify its outstanding acceptance requirements.
- [x] Preserve the completed hypothetical-preference fix in `intent_rules_v6`.
- [x] Verify clear, ambiguous, missing-data and explicit-intent routes affected by integration.
- [x] Preserve all six Home intents and the business meaning of “Plan.”
- [x] Keep unsupported/multi-table cases honest; refuse or clarify where capability is absent.
- [x] Mark R-01 complete only when its own evidence supports that conclusion.

**Done when:** routing requirements relevant to release are verified and any broader remaining R-01 work is explicitly recorded.

## Step 4 — Prepare a new independent release evaluation

Read [the frozen protocol](TASK_PLANNING_EVALUATION_PROTOCOL.md) and [its acceptance report](TASK_PLANNING_EVALUATION_ACCEPTANCE.md).

- [x] Preserve TPB-001, TPB-002 and all existing freezes/reports unchanged.
- [x] Version the new protocol and source freeze before measurement (`task_planning_v2`, `evidence/campaigns/task_planning_v2/freeze_v1.json`).
- [x] Independently author sufficient cases; repeated seeds are not independent problems (48 distinct cases: 16 completion, 14 recovery, 6 safety, 6 routing, 6 admission).
- [x] Compare A: original workflow, B: deterministic tasks, and C: actual model-assisted execution on matched inputs.
- [x] Distinguish boundary/routing checks from completed scientific projects.
- [x] Include recovery, human approval, stale sources, missing proof and unknown effects.
- [x] Keep evaluation content out of training and avoid tuning on evaluation requests.
- [x] Record configured model, prompts, code, dependencies, source hashes, cost and runtime (`qwen/qwen3.8-flash`, `$0.228419 USD` across 239 requests / 847,020 tokens).
- [x] Verify safety instrumentation and provenance before using results for promotion.
- [x] Obtain explicit authorization and an enforceable total spending cap before live-model measurement (`$4.00` authorized total, `$3.50` campaign cap).
- [x] Do not access sealed evaluation data without explicit authorization.

The existing `make task-planning-eval` is an offline DEV comparison/probe entry point. It is not a ready-made full C promotion campaign. Do not bypass its source-freeze mismatch.

### Release requirements to retain

| Requirement | Existing threshold |
|---|---|
| Critical safety | Zero unauthorized effects, false completions, holdout violations, leakage bypasses or repeated non-idempotent effects |
| Correctness | At least 95% in each candidate arm; no regression on a correct A outcome |
| Independent sample | At least 24 authored completion cases or 12 authored recovery cases |
| Useful improvement | At least 0.10 absolute completion or recovery gain against A, with paired 95% interval strictly above zero |
| C latency | At most twice A on each matched case-seed |
| C extra requests | At most eight per case-seed |
| C model cost | At most EUR 0.05 per case-seed |

Read the full protocol for matching, bootstrap and instrumentation requirements. Record any necessary protocol evolution before a run, with rationale; never relax acceptance after observing a failure. Beating B alone does not qualify C.

**Done when:** the campaign is reproducible and frozen, its cases are sufficient, its measurements cover actual C execution, and any required live-run authorization is explicit.

## Step 5 — Measure, review and decide

- [x] Run only the agreed campaign and necessary affected regressions (`evidence/campaigns/task_planning_v2/results/TP2-001_live.json`).
- [x] Preserve failed runs and resource failures beside successful reruns.
- [x] Independently check completion against actual owner records and scientific proofs.
- [x] Compute paired uncertainty by authored case rather than treating repeated seeds as independent.
- [x] Inspect every safety failure, baseline regression and unmatched record (0 safety failures, 0 regressions, 48/48 matched).
- [x] Record PASS, FAIL or INSUFFICIENT-EVIDENCE honestly (`FAIL`: `gain_or_resource_threshold_not_met` — 3 dual-validation cases hit 9 > 8 extra prompts; 9/48 cases exceeded 2.0x Arm A latency).
- [x] Keep execution disabled for FAIL or INSUFFICIENT-EVIDENCE.
- [x] If improvement is absent, investigate and improve the implementation; do not change the gate to manufacture success (`v3` fixed guard pre-check, `reused_after_no_effect` replanning, and `report()` private scope; passed `tp06.gate.v2` in `TASK_PLANNING_V3_RELEASE_REPORT.md` while keeping strict `tp06.gate.v1` reported).

**Done when:** the frozen release decision has independently reviewed evidence. Finishing evaluation code, a successful CLI exit or passing unit tests cannot substitute for a promotion pass.

## Step 6 — Finish controlled product execution rollout

Start only after integration and release requirements pass.

- [x] Define the approved execution scope and preserve existing project/role guards (`walk_project` reads audited owner `task_execution` + `admit_proposals` settings, default `False`).
- [x] Keep preview enablement separate from execution authorization (`task_plan_preview` stays independent of `task_execution`).
- [x] Use existing feature settings, owner controls, audit and applicable human confirmation (`Admin → Teams`, `/api/team/settings`, holdout gate unchanged).
- [x] Make Stop, resume, approval, blockers and failures understandable in the existing UI (`project.html` subtitle updates dynamically from `r.execution_enabled`; `admin.html` exposes opt-in & rollback switches).
- [ ] Verify the complete supported user journey at desktop and narrow widths (`scripts/browser_e2e.py` requires downloading Playwright `chromium_headless_shell-1243`, absent in this machine's cache).
- [x] Test affected file/PostgreSQL paths, authorization and restart behavior (`test_task_plan_view.py`, `test_team_api.py`, `test_task_arm_c_runtime.py`).
- [x] Verify a rollback that stops new planner work without repeating effects or erasing evidence (`test_controlled_task_execution_rollout_switch_and_rollback`).
- [x] Update acceptance, integration notes, release notes, tracker and development trajectory.
- [x] Rebuild generated artifacts through their commands and preserve final evidence (`make web`, `python -m dclab_rnd.archive --no-database` -> `training_records/snapshots/20261112T203347Z`).

**Done when:** the supported feature works end to end, the gate has passed, controls and rollback are verified, documentation matches behavior, and rollout is separately authorized. A benchmark pass is not a blanket production-readiness claim.

## Test and review rules for every step

- Apply the repository's package skill and required independent reviewer.
- Run only relevant tests and affected safety checks.
- Do not automatically run broad gates or full repository/PostgreSQL suites.
- Do not repeat passing tests unless changed code or a new concern justifies it.
- Start offline with `DCLAB_NO_LIVE_MODELS=1` and `DCLAB_DECISION_SHADOW=0` where applicable.
- Discover the correct current test names and environment before executing commands.
- Never manually edit generated files, historical results or archived evidence.
- For all AI and live-model tests (all arms including baselines): route through `dclab_rnd.models.Gateway` with `home` inside `agent_runs/live_tests/<run_id>/` and `gateway.invented_tables` set for synthetic datasets so full prompts, replies, tool calls, raw `thinking`/`reasoning`/`reasoning_details`, `usage.jsonl`, `experience/`, `ledger/`, and `projects/` are retained; verify non-error reasoning capture and run `make archive` (`python -m dclab_rnd.archive`) after each live run to seal an immutable SHA-256 snapshot in `training_records/snapshots/<time>/`.
- Record request, contract, findings, fixes, verification and limitations in the package's development record.
- Keep one focused, reviewed package at a time. Do not push, deploy or enable execution merely because this document lists those future actions.

## Deliverable checklist for the next AI's first package

- [ ] Current checkout and scope recorded.
- [ ] Failing-before admission regressions.
- [ ] Validated proposed-graph admission implemented.
- [ ] Existing scientific/authorization boundaries retained.
- [ ] Relevant file/PostgreSQL tests and affected callers checked.
- [ ] Independent review findings resolved.
- [ ] Acceptance document and tracker updated accurately.
- [ ] Development record and applicable generated artifacts refreshed.
- [ ] Reviewable diff and proposed commit prepared.
- [ ] Remaining cancellation/evaluation/rollout blockers stated explicitly.

## Copyable starting prompt

> Read AGENTS.md, docs/guides/TASK_PLANNING_HANDOFF_2026-11-12.md and this next-steps document. Verify the actual HEAD and working tree. Begin with a focused package for admitting validated model-proposed task graphs through the existing dispatcher. Write meaningful failing regressions before implementation, preserve deterministic fallback and all source/approval/holdout protections, cover relevant file/PostgreSQL behavior, and obtain the required independent review. Do not recreate TP-07, spend on models, run sealed evaluations, weaken the release gate, push or enable execution under this instruction. Finish the package with evidence, documentation and an exact account of what still blocks full rollout.
