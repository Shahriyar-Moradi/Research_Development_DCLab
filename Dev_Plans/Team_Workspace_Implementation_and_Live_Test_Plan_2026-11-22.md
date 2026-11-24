# Team workspace implementation and live test plan

**Proposed engineering work · 22 November 2026 · Baseline reviewed at `7e33b994`**

> **Status (2026-11-24): parked.** W1, W2 and W3 were built (as T1 to T3), and so was the approval part of W5 (only the
> decision's owner approves, T5a). W4 and the rest of W5 are not started. W6 was replaced by a three-way comparison of
> rules, an AI helper and an AI agent (built; first race run). The work is parked for the next plan. What was built,
> measured and left:
> [Team_Workspace_AI_Arms_2026-11-24/STATUS.md](Team_Workspace_AI_Arms_2026-11-24/STATUS.md).

A Business member should be able to open a project, ask a question, get a defensible answer from the right teams and record what the organization decided. The acceptance journey ends when a later constraint or observed outcome reaches that same conversation. The existing Teams page, private threads, delivery queue, action log, scoped memory and report revisions are the starting point.

The November 22 review reproduced four defects on temporary workspaces and found a fifth gap in the decision-approval entry point. The Teams browser walkthrough passed with seeded inputs. This plan covers those findings and the remaining distance from fixtures to an uploaded project, independent commercial branches and measured agent judgment. Broader opportunities have their own [development roadmap](DCLab_Agentic_Expansion_Roadmap_2026-11-22.md).

## Build order

| Package | Responsible area | Dependency | Demonstrable result |
|---|---|---|---|
| W1. Align the question and label | Data Science and problem contracts | Existing feasibility flow | A confirmed cancellation definition on renewal-only data returns a precise blocker. |
| W2. Recover work and retract corrections | Team runtime and storage | Existing score and revision records | A crash resumes one score. An explicit zero delay restores the initial ETA through a new report. |
| W3. Close commercial handoffs | Team messaging, Sales and Marketing | Existing directory and outbox | Both teams can answer Business independently through their own signed-in conversations. |
| W4. Connect real project evidence | Studio data, ML Engineering and model adapters | W1 and W2 | A regular upload yields a checked contract, measured baseline, evaluated version and linked report. |
| W5. Carry memory through a human decision | Scoped memory, decision store and Teams UI | W3 and W4 | An owner approves a revision, sees a correction, approves the revised proposal and logs an outcome. |
| W6. Evaluate one agent choice | Agent runtime and evaluation | W1–W5 for the full journey | A paired evaluation quantifies whether choosing the next action improves case resolution. |

W1 and W2 can begin independently. W3 can ship an immediate capability check before the full commercial workflow. Each package ends with its own targeted regression checks and a separately reviewable change.

## W1. Make a feasibility claim mean what Business asked

- Persist the confirmed target event, entity, prediction time, horizon, label source, maturity rule and brief revision. Keep an ambiguous business term open until evidence or its owner resolves it.
- Have Data Science compare that contract with the authorized data card and actual label definition. A column called “churn” is insufficient. Return the specific missing event, why it matters, who can supply it and which task resumes when it arrives.
- Pass the same confirmed revision to ML Engineering. If Business changes the target after feasibility, mark dependent results stale and revalidate before running.
- **Regression pair:** On one renewal table, “failure to renew by the contractual date” may pass given mature outcomes. “Explicit cancellation only” must block until a suitable cancellation record exists. Add paraphrases, conflicting answers, immature outcomes and a late feature. Verify the outcome in the Business report as well as in the stored contract.

Reuse the [problem–data check](../dclab_core/meet.py), [team feasibility flow](../dclab_rnd/teamspace/agents.py) and [acceptance journey](../tests/test_team_acceptance.py).

## W2. Make retries and corrections safe to trust

- For each score, distinguish pending, running, terminal success, refusal and failure. Reuse the persisted attempt identity on retry. Reconcile a recoverable result by its run ID before repeating expensive work. A report may claim only a terminal result with a permitted evidence reference.
- Treat an absent correction, an explicit zero, and a repeated delivery of a previous correction as separate cases. Include the source revision in the deduplication key. Reject a late correction against an obsolete base or ask its owner to reconcile it.
- Recompute only affected tasks. A kitchen update can change ETA while the availability snapshot for the initial courier offer remains valid. A **new offer** constitutes a new acceptance task and snapshot.
- **Recovery tests:** interrupt before scoring, after result creation, before the terminal write and after reporting but before acknowledgement. Restart the worker, deliver messages twice and reverse independent branches. Require one visible reply per revision, no stranded accepted task, and no successful report for an unsuccessful score.
- **Correction tests:** original ETA 18:32, +8 minutes → 18:40, withdrawal to zero → 18:32, subsequent +8 at a newer revision → 18:40. Repeat one event and deliver an older one late. Verify report ancestry, current Business brief and the unaffected acceptance version on both supported storage backends.

Start in the [model-score binding](../dclab_rnd/teamspace/tools.py), [team runtime](../dclab_rnd/teamspace/agents.py), [delivery](../dclab_rnd/teamspace/delivery.py) and [storage](../dclab_rnd/teamspace/store.py).

## W3. Give Sales and Marketing a working return path

1. Reconcile the UI recipient list with the executable capability directory. Until a recipient can answer, return an immediately visible unsupported result and a next owner. A successfully queued envelope does not establish that a team accepted work.
2. Give each request a stable correlation ID, originating thread and expected output. Multiple open requests in the same problem must be addressable separately. A reply button selects its task. Ambiguous free text asks which task the member means.
3. Sales reads dated exclusions, asks its own team for capacity and returns a capped, restricted shortlist or a blocker. Marketing asks its owner for dated offer terms, eligibility and cost, then returns the confirmed terms or names what is missing. Each team may reply while the other is offline.
4. Business can inspect partial results, see who owes what and continue from the same conversation. Protect team notes and customer-level artifacts according to the recipient. Support a member contributing a constraint without giving them deployment or decision-approval rights.

**Acceptance:** Open both branches from the same Business conversation. Answer Marketing first, then Sales; reverse the order on a second case. Interleave two projects and two tasks owned by the same person. Retry an answer and revoke a membership before delivery. Verify exactly one visible result per task, correct return thread, truthful outstanding work and scoped artifact reads.

Extend the [thread API](../dclab_rnd/agentic/routers/team_threads.py), [team contracts](../dclab_core/team_work.py), [team screens](../dclab_rnd/agentic/web/src/views/team.html) and [thread tests](../tests/test_teamspace_threads.py).

## W4. Reach an evaluated model from an ordinary upload

- Resolve permitted inputs from the active project, problem and authenticated principal. Load the table through the studio's data accessor and validate its fingerprint. Preserve explicit fixture references and simulated labels for the demo. Existing projects without team input declarations need an authorized resolution path or a specific missing-input answer.
- Data Science produces the verified contract and a baseline specification. ML Engineering submits an authorized studio job within the existing budget. Its report returns the run ID, data and feature revisions, baseline, measured comparison, uncertainty, limitations and versioned artifact. A failed job returns a blocker and a resume point.
- Freeze the prediction contract before looking at scores. Keep feature selection, encoding and fitting inside training folds. Select on validation, lock the candidate and evaluate the sealed holdout once. An already consumed holdout cannot silently become another selection input.
- For delivery, expose versioned ETA and first-offer acceptance adapters only for supported snapshots. Absent inputs yield a partial or blocked report. Preserve the distinction between an arrival estimate, probability of acceptance and observed outcome.
- **Acceptance:** From the product upload flow, use an invented but non-fixture customer event table. Verify that the Business report traces to this upload, a real executed baseline and the selected run. Repeat with no dates, a post-outcome column, a mismatched version, a cancelled job and a stale brief. Require zero fabricated scores.

Reuse [studio data](../dclab_rnd/studio/data.py), [workflow graph](../dclab_rnd/studio/graph.py), [run lineage](../dclab_rnd/studio/lineage.py) and [Lead input resolution](../dclab_rnd/teamspace/lead.py).

## W5. Connect scoped memory, approval and follow-up

Build each turn from the signed-in actor, team, project, problem, task and relevant model version. Record the source and scope of each decision-relevant fact. Before using a cached summary or forwarding a result, recheck membership, artifact visibility and current revisions. A summary can be rebuilt from authorized current records after restart.

Wire the existing team decision writer to an authenticated product command and an approval control on the proposal in its originating thread. The human confirms the exact report and constraint revisions. Use an expected revision and request identity to handle stale clicks, concurrent approvals and retries. Persist the decision and publish the receipt durably. A later outcome records what happened, when, under which policy and how it was measured.

**Acceptance:** Business approves 50 contacts. Sales corrects capacity to 30 in its own thread. Only the affected proposal, list and approval become stale when capacity was applied after scoring. A second approval cites 30, the prior decision remains traceable and another workspace sees neither. Exercise reload, private notes, signed-out and revoked users, an agent attempting human approval and duplicate outcome submissions. In the delivery case record an actual arrival and an actual first-offer outcome separately from predictions.

Use [team decision storage](../dclab_rnd/teamspace/store.py), [memory routes](../dclab_rnd/agentic/routers/memory.py) and [existing correction tests](../tests/test_team_memory_corrections.py).

## W6. Let the agent choose one useful next action

Start with the choice to answer, clarify a definition, inspect an authorized source, delegate to a capable team, or return a blocker. Every proposal identifies the task, owner, input and brief revision, reason, evidence references, budget and stopping condition. The existing runtime and guard validate eligibility and execute the authorized tool. Store the chosen action, attempted tool, terminal result, report and next owner separately. Persist a switch for this decision so new cases can return to rules and in-flight tasks can resume from their recorded state.

A control case has a complete, current answer in memory. The preferred action is to answer from that source. A cancellation request on renewal-only data requires clarification or a different source. A missing Marketing answer may warrant a partial report. These are distinct decisions even when the user uses the same word, “churn”.

## Verification matrix and release evidence

| Layer | Exercise | Required observation |
|---|---|---|
| Contract / unit | Target agreement, zero corrections, terminal-action claims, allowed recipient, scoped approval and report revision | Direct assertions on contract, action, status and evidence. |
| Storage / integration | File and isolated PostgreSQL stores, restart, ordering, concurrent receipt, retry, legacy rows and revoked membership | One terminal result per task revision, no lost branches or cross-scope data. Apply forward migration to a copy of an existing workspace if storage changes. |
| Product / browser | Start from signed-in Home or Teams with a normal upload. Use separate sessions for contributors and the approver. Exercise churn, a missing dispatch snapshot and a corrected decision. | The originating member sees answer, supporting report, actions taken, limitations, next owner and what changed. Refresh and reopen the deep link at desktop and narrow widths. |
| Scientific | Frozen prediction contract, mature labels, train-only features, baseline, one selected version and sealed evaluation | Values in the Business report resolve to run records. Simulated results are labeled. No claim of outreach lift from churn risk alone. |
| Live judgment | Pair actual model-backed next-action decisions with the deterministic baseline on identical cases | Score tool execution and final task state. Record task-level uncertainty, costs, abstentions and failures. |
| Rollback | Disable the new decision policy with open tasks and an undelivered reply | The pending task resumes once, earlier history remains readable and existing permissions still apply. |

For the current repository, targeted starting commands are below. New acceptance cases belong alongside the closest existing group. Run affected PostgreSQL cases against the designated disposable test database. The browser command currently uses a seeded project; W4/W5 must extend it to cover upload and approval.

```sh
PYTHONPATH=tests:. .venv/bin/python -m unittest test_team_acceptance.FileTeamAcceptanceTests
PYTHONPATH=tests:. .venv/bin/python -m unittest test_teamspace_agents.FileTeamAgentsTests
DCLAB_BROWSER_CHANNEL=chrome .venv/bin/python scripts/browser_e2e.py --only teamwork
```

### Live evaluation runbook

1. **Freeze the comparison.** Have an independent author write cases and expected outcomes before model trials. Proposed pilot: 24 distinct cases spread across target mismatch, commercial routing, parallel replies, missing evidence, revisions and unsupported requests. Include straightforward controls, paraphrases and two simultaneous problems. Reserve a fresh sealed set and score it once after development. Keep related paraphrases in the same split.
2. **Run paired arms.** Rules and model receive identical state, evidence, tools, permissions and budgets. Keep Home and the test user fixed while varying the specialist chooser. First run shadow proposals. Then execute in isolated invented workspaces to measure actual resolution. Use repeated development runs to reveal unstable choices. Record abstentions, timeouts and fallbacks in the model arm. A rules-only turn records zero model calls.
3. **Use the real test gateway.** Existing [live scenarios](../scripts/live_lead_scenarios.py) exercise Home-to-Lead and are a smoke test. Extend the harness to invoke W6 and assert from the ledger that the candidate chose the specialist action. Respect the runner's provider restrictions. Set a total spend cap and per-case step/time limits before starting; stop on exhaustion and report incomplete pairs.
4. **Keep auditable evidence.** Isolate each arm under the run's workspace in `agent_runs/live_tests/<run_id>/`. Route every model call through the project gateway, enable test-workspace text capture, mark invented tables, verify required gateway records for successful exchanges, and retain usage, task ledger, tool outputs, project versions and frozen expectations. Archive with `python -m dclab_rnd.archive`, verify the SHA-256 receipt and retain evaluation-only markings on sealed material. Follow the repository's [live-test and archive requirements](../AGENTS.md).
5. **Score the user outcome.** Primary metric is the fraction of all cases reaching the correct answer **or justified blocker** with valid evidence and owner. Also report wrong-target completions, unnecessary questions, missed branches, duplicate replies, latency, cost per resolved case and model-to-rules fallbacks. Compute paired wins/losses/ties with uncertainty grouped by case family. Have an independent reviewer adjudicate ambiguous findings from anonymized artifacts.
6. **Make the decision before release.** Proposed promotion gate: zero unauthorized reads, false completed actions, ungrounded scores or holdout violations; all deterministic regression cases pass; the paired lower 95% confidence bound for the resolution gain is above zero within the predeclared cost and latency budgets. A small pilot can be inconclusive. Keep rules serving in that event, diagnose on development cases and obtain a fresh sealed set for a changed candidate. Roll out the winning capability in shadow, then to opted-in workspaces, with the same rollback switch.

**Definition of done:** the churn and delivery journeys pass through the normal product, every important claim traces to an authorized revision or run, and Business can see what happened after the decision. Enabling model-selected actions is a separate release decision supported by paired results. The tests and live runs above are planned work.
