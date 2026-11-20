# Handover: collaborative team workspace (2026-11-20)

This is for the next AI agent that continues this work. Read it first, then the two plans below.

## The goal

The owner's plan is `Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md`. Each company team (Business, Data
Science, ML Engineering, Sales, Marketing) talks to its own agent. The agents hand work to each other on a shared
problem.

The owner's most important point: **each agent knows its own tasks, the actions it may take, its responsibility and
the report it owes.**

The approved implementation plan (packages W1 to W8, decisions, verification) is
`/Users/shahriar/.claude/plans/users-shahriar-downloads-dclab-harness-eventual-planet.md`. It is outside the repo, so
the parts that matter are copied below. The scripted prototypes are in `docs/ui-demos/team-workspace/`. They are the
owner's files: do not commit or change them.

## Decisions already taken (do not reopen)

- **Scope.** Release 1, plus the actions-and-reports layer, plus the food-delivery join (ETA and HyperAck branches)
  with fixture specialists.
- **Teams.** A team is a new entity beside the four permission roles. A role says what a person may do; a team says
  whom an agent speaks for.
- **Agents are deterministic now.** AI team agents come later, behind an off-by-default switch, and are used only
  after a live paired test on people written by an independent session. Scripted tests prove plumbing, never judgment.
- **Contracts.**
  - `TeamTask`, `TeamAction`, `WorkReport`, `TeamDecision` and `TeamAgentConfig` live in `dclab_core/team_work.py`.
  - A report is built only from the action log and child reports.
  - It never claims an action without a terminal record, never invents a missing child result, and never carries a
    private team note into a shared report.

## Done (all pushed to `main` and `dev`)

| Package | What | Commit |
|---|---|---|
| W1 | Contracts and rules, dclab-core 0.164.0 | 4ea722c7 |
| W4 | Task store, action log, delivery outbox, migration 0026 | c970ffdb |
| W2 | Teams, membership, permissions `contribute`, `approve_decision`, `contact_customers`; migration 0027 | e847bb8b |
| W3 + W5 | Team threads and the shared handoff history (migration 0028); agent configurations, TeamGuard, deterministic agents, `job_description`, rules ledger | e06a49e6 |
| W6 + W8 | Team memory scope (memory_item v3, dclab-core 0.165.0) and corrections; the team screens (Teams tab, Team inbox, task, report, history, job description) | 3db2cbbb |
| W6 follow-ups | Team memory routes (members only), the reconcile route for a failed correction fix-up, steady delivery timing tests (`failed_at`) | 6b089e8a |
| W7 | Experience v18 team events and the ledger point `team.task_owner` (dclab-core 0.166.0); rename and remove teams (a team with open work is refused); `tests/test_team_acceptance.py`, the seven checks on files and PostgreSQL | 92527bda |

**Reviews.** Every package had an independent review (`dclab-reviewer`). Each review found 2 to 10 issues, and all
were fixed with tests. W6 had 7 findings, W8 had 2.

**Proven by tests, on files and on PostgreSQL:**
- Two signed-in people finish the renewal steps 1 to 4 through the routes.
- The food-delivery branches join in either order.
- A denied action is reported.
- A kitchen correction (+8 minutes) supersedes the ETA report.
- A restart delivers once.
- Another team, project or workspace cannot read a thread.
- No agent records a person's answer or approves for the owner.
- A correction (50 → 30) marks dependent tasks, reports and decisions stale down the chain and alerts the decision
  owner.

## Left to do (in this order)

**Update 2026-11-21:** items 1 and 2 are done (6b089e8a, 92527bda; W7's review found 4 issues, all fixed). Left: item 3
(the browser test needs Chromium: ask the owner first), item 4 (the final gate) and item 5 (later releases).

1. **W6 open items** (from its fixer):
   - No route yet exposes the team memory base.
   - No route lets someone retry a failed correction fix-up. Today it shows as `team_work.error` in the route's result,
     and running the correction again finishes the work.
   - **Flaky timing tests.** In `tests/test_teamspace_delivery.py` (W4), `PgDeliveryTests` has tight time limits:
     the backoff wait must be between 0.05 and 0.2 s, and the hung-recipient cut-off must take under 0.9 s.
     - They failed (wait −0.16 s; 0.93 s) while another session's gate loaded the machine.
     - They pass when run alone.
     - Make them robust: measure from the recorded failure time instead of from the clock after the call, and give the
       cut-off more margin.
2. **W7 capture and acceptance:**
   - Experience event kinds `team.task.*`, `team.action.*` and `team.report.*` in `dclab_core.experience`. This is a
     dclab-core contract bump: 0.166.0, with the CHANGELOG, schemas (`make core-schemas`), `CORE_INTEGRATION.md` and
     `make core-wheel-check`.
   - Emit them where tasks, actions and reports are written (`dclab_rnd/teamspace/store.py`, `delivery.py`,
     `agents.py`).
   - A ledger point for each routing choice (which team owns a task) via `dclab_rnd/ledger/capture.py`.
   - W2 follow-ups:
     - team events (team created, member added or removed);
     - rename and delete team routes, owner only, in `guard.SPECIAL`.
   - The owner's acceptance checks as one end-to-end test file (see "Verification" below), on files and with `--pg`.
   - A development-trajectory record in `research/llm-fine-tuning/experiments/dev_trajectories/records/`, then
     `make sft-dev`.
   - An independent review with `dclab-reviewer`, and fix everything it finds with tests.
3. **Browser test of the team screens** (`make browser-test`). It needs Chromium, which is not installed. Do not
   download it without the owner's yes.
4. **Final gate.** First check that no other gate is running (`pgrep -fl "make check-"`). Then run `make check-all`,
   fix what fails, and push.
5. **Later releases, not now:**
   - R2: Sales and Marketing with real capacity and offer tasks.
   - R3: Data Science ↔ ML Engineering through the studio engine and its leak checks.
   - R4: AI team agents behind the switch, with a live paired test.

## Verification (the owner's acceptance checks)

1. **Business ↔ Data Science round trip.** One answer and one linked report. Every claimed action has a terminal
   record, an actor and permitted evidence. The handoff history reads like the owner's example (`1. Business → Data
   Science · question`).
2. **Two delivery branches finished in opposite orders.** One dispatch brief holds both results exactly once. While
   one is out, the parent says which result is outstanding.
3. **A specialist action fails or is denied.** The report names the failed attempt and the next owner. No invented
   score appears, and a retry adds no duplicate.
4. **The kitchen adds 8 minutes after completion.** The revised ETA report links to the superseded one. The acceptance
   report keeps its snapshot.
5. **Restart, another team, another project, revoked member.**
   - A restart delivers the answer once.
   - Other teams see the history only.
   - Another project or workspace sees nothing.
   - A revoked member's delivery is refused.
6. **Authority.** No agent approves for the decision owner, and no role contacts customers by default.
7. **The four questions.** A Business member can answer them from the thread alone: what did we decide, why, who acts
   next, what changed since yesterday.

## Key files

- **Contracts and rules:** `dclab_core/team_work.py`, `dclab_core/memory.py` (v3 team scope).
- **Wiring:** `dclab_rnd/teamspace/`, with `store.py`, `delivery.py`, `threads.py`, `stream.py`, `agents.py`,
  `tools.py` and `fixtures.py`.
- **Teams:** `dclab_rnd/storage/teams.py`.
- **Memory:** `dclab_rnd/memory/scope.py` (`TEAM_STAMP`, `team_chain`), `bases.py` (`correct_base_fact`, stale marks).
- **Routes:** `dclab_rnd/agentic/routers/team_threads.py`, the teams router, `routers/memory.py`.
- **Screens:** `dclab_rnd/agentic/web/src/views/team.html` and `core.js`. Rebuild with `make web`; never edit
  `static/app/**` by hand.
- **Migrations:** 0026 (team work), 0027 (teams), 0028 (threads).
- **Tests:** `tests/test_core_team_work.py`, `test_teams.py`, `test_teamspace_*.py`,
  `test_team_memory_corrections.py`, `test_team_screens.py`.

## Rules that bite here

- **Tests.** Always run with `DCLAB_NO_LIVE_MODELS=1` and `tests/run_parallel.py`, plus `--pg` for anything touching
  storage.
- **Git.**
  - Never `git stash`. Stage explicit paths only.
  - Other sessions keep uncommitted work in this tree. Never stage these:
    - `.claude/launch.json`, `.env.example`;
    - `dclab_rnd/agent_eval/e2e/__main__.py`, `dclab_rnd/agentic/pages/lab.py`;
    - `dclab_rnd/bench/__init__.py`, `dclab_rnd/twin/__init__.py`;
    - `tests/test_agent_eval.py`;
    - `evidence/campaigns/agent_eval_v1/results/AEV-130_*`;
    - the `gate-old-failures-2026-11-20.json` record;
    - `Dev_Plans/Collaborative_Agentic_Workspace_2026-11-20.md`, `docs/ui-demos/`.
  - Push with `C=$(git rev-parse HEAD); git push origin "${C}:refs/heads/main" "${C}:refs/heads/dev"; git branch -f dev HEAD`.
  - Commit messages explain why, and end with the co-author line.
- **Models.** No Claude model in any test, arm, person or judge. A live run costs at most €0.5.
- **Generated files** are rebuilt by commands, never edited:
  - `make web`;
  - `make catalog` (`docs/data_catalog.json`, `docs/DATA_CATALOG.md`);
  - `make core-schemas`;
  - `make sft-dev`.
