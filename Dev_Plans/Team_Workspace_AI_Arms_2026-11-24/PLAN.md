# Plan: Team workspace, more room for AI, compared three ways

> **Status (2026-11-24): PARKED.** Parts A and B are built and pushed; Parts C and D are not started. The full record
> of what was built, what was measured and what remains is [STATUS.md](STATUS.md). This file is the plan as approved on
> 2026-11-23, copied unchanged below from the session's plan file.

---


Updated 2026-11-23 after the owner's steer: "give more room to AI and involve it more; test rules, AI as advisor and AI
with more room against each other; I prefer more AI where it performs better."

## Context

The owner's plan is `Dev_Plans/Team_Workspace_Implementation_and_Live_Test_Plan_2026-11-22.md` (packages W1 to W6,
called **T1 to T6** here, because W1 to W8 already name the finished earlier work). The first version of this plan
ended with an honest limit: after all of it, the team agents would still follow fixed rules, and the model would only
make one small choice in shadow. The owner does not want that. This version makes AI involvement the main thread and
turns the question "how much AI?" into a measured comparison.

The owner's prototype `docs/ui-demos/retention-simulation/` (`GUIDE.md`) shows the target experience. It is the
owner's file and is never edited here.

## What our own results already say

| Date | What was measured | Result | Lesson |
|---|---|---|---|
| 11-15 | The same model alone, and inside DCLab's tight harness (planning score, 8 cases) | 4.0 alone, 1.3 inside | The harness set the ceiling, not the model |
| 11-15 | Numbers nobody gave | 0 inside the harness, 76 for a free agent | Code must own the numbers |
| 11-16 | "Think first": the Lead writes its whole plan before acting | 2.75 → 3.88, accepted 3 of 8 → 7 of 8 | Room to think helps |
| 11-16 | The same change on a weaker model (Nemotron Super) | 2.00 against 2.88 alone | Room helps a strong model more |
| 11-19 | A loop where the model picks one small step and rules check each step | 0.4 to 1.4 against 3.75 with it off | Small bounded choices hurt |

**What this means for the design:**
- Give the model room to **understand, break down, plan and decide**. Keep code strict about **what gets executed and
  what is claimed**.
- Do not build "the model chooses one next action and rules validate it". That was T6 in the first version of this
  plan, and it is the shape that failed on 11-19. **T6 is dropped and replaced by Part B below.**
- Test with a strong open model as well as the cheap one. A weak model hides what room can do.

## The design: two layers and one switch

**The strict layer (the same for every arm; code only).** This is what makes any amount of AI safe:
- who may do what: the action directory guard, permissions, only the owner approves, nobody contacts customers;
- the task state machine, delivery once, retries, corrections with revisions;
- truth: a report is built from the action log; no action is claimed without a terminal record; every number comes
  from a tool result or a run, never from the model's words;
- data science rules: splits, metrics, the holdout read once, leakage checks;
- budgets (steps, seconds, euros per task) and privacy (team notes stay in the team).

**The free layer (where the arms differ).** For each team agent (Lead, Business, Data Science, ML Engineering, Sales,
Marketing), one audited switch picks its "brain":

| Arm | Who decides | What the model does | Example: Business asks "who should we call before renewal?" |
|---|---|---|---|
| **R. Rules** (today) | Code | Nothing | A keyword rule picks the teams. Each agent runs its fixed steps. |
| **H. Helper** | Code decides; the model helps at named seats | Reads the person's words when the keyword rule is unsure; suggests which question to ask and how to word it; explains the report in plain words; lists risks it sees. Its output is used only where a code check passes. | The rule cannot read "customers who quietly leave". The model reads it as "non-renewal" with its reason; code checks the label supports it and continues without a needless question. |
| **A. Agent with room** | The model decides inside the strict layer | Reads the task, the thread, the memory and the evidence. Writes its own plan first (kept whole). Chooses its actions, their order, what to ask and whom to hand work to. The Lead breaks the request into team tasks itself. Writes the report's findings and recommendation; code attaches the numbers. | The Lead plans: "Data Science confirms the outcome and checks the label; in parallel Sales gives its capacity and Marketing its budget; then ML Engineering ranks; I join the result". It opens those tasks through the guarded tools. |

- Every arm uses the same tools, guard, store and report builder. Only the brain changes.
- If a model call fails or times out, that turn falls back to the rules, and the fallback is recorded.
- The switch is per role, so the product can end up mixed (for example an agent Lead with a rules Sales).
- Default stays "rules" for each role until that role's comparison says otherwise.
- Every model call keeps its API thinking summary; every decision of every arm is recorded with who answered, so the
  runs also become training data for our own models later.

## The comparison

- **Same everything:** the same cases, data, tools, permissions and budgets for the three arms.
- **Two models:** `qwen/qwen3.8-flash` (cheap) and the strongest open model available on OpenRouter (from the
  2026-11-16 list: Qwen 27b, Nemotron Ultra). Never a Claude model in a test.
- **Cases are written by an independent session**, never by the builder, in families:
  - controls, where today's rules are right;
  - free wording and paraphrases;
  - unclear or mismatched targets (cancellation against renewal);
  - several teams with limits (capacity, budget) and answers in any order;
  - missing evidence;
  - a correction in the middle;
  - traps: an order to approve, to contact customers, a request for a number nobody measured, a private note;
  - requests no rule covers.
- **Two sets:** a dev set to improve on (about 40 cases), and a sealed set (about 70) that is run once.
- **What is scored:**
  - main score: the share of cases that end with the right answer **or** a justified blocker, with valid evidence
    and the right next owner (checked by code from the records);
  - quality of understanding, plan and report: a neutral judge model (the free Nemotron judge), 0 to 4, as in our
    earlier exams;
  - hard zeros, in any arm: an invented number, an action claimed but not done, an unauthorised read, an agent
    approving, a holdout read twice;
  - also: needless questions, missed teams, duplicate replies, time, cost, fallbacks to rules.
- **How the winner is chosen, per role:** the arm with the best main score, with zero hard violations, inside the time
  and cost limits fixed before the run. On a tie, the simpler arm. The terms are pinned by hash before any sealed run.
- **Cheap ladder:** free trace counts first, then a 3-case quick set, then the dev set, then the sealed set once.
- **Budget (default):** at most EUR 0.5 per run and about EUR 10 for the whole comparison; spend is reported after
  each run.

## Build order

Each package: a failing test first, the change, its tests on files and PostgreSQL, an independent `dclab-reviewer`
review, fixes, a development record, commit and push. The full gate runs at night and at the end.

### Part A. The strict layer first (every arm needs it)

1. **T5a. Only the decision owner can approve** (small). Today one person can record another's approval
   (`teamspace/store.py:895-923`). `TeamDecision` v2 with a revision and a request identity; a rule
   `decision_write`; the `approve_decision` permission enforced; `DCLAB_AUTH=none` fixed (`memory/scope.py:54`).
2. **T2. A crash resumes once; a zero correction restores the first answer** (small). `_model`
   (`teamspace/agents.py:399-405`) re-scores an unfinished attempt; three truthiness tests become `is not None`; a
   correction names the revision it is based on, so a late older one is refused.
3. **T1. Feasibility means what Business asked** (medium). New `dclab_core/target_contract.py` (`TargetContract`,
   `LabelDeclaration`, `agreement`); the task carries its confirmed target; a mismatch blocks with what is missing
   and who supplies it. The keyword reader `read_event` returns "unsure" instead of guessing: that is exactly the
   seat where arms H and A let the model read.
4. **T3. Sales and Marketing can answer** (medium). New `teamspace/commercial.py`; a reply answers its own task
   (`task_id`); a recipient with no agent says "unsupported" at once; the recipient list comes from the teams that
   can really answer.

### Part B. AI involvement (replaces T6)

5. **B1. The brain switch** (medium). One interface for a team agent's brain with three implementations: rules,
   helper, agent. One audited switch per role (the pattern of `ledger/routing.py` `switch`); the rules fallback; the
   new model purposes (`team_helper`, `team_agent`) with prompt files, harness entries and tests that their thinking
   is stored; the `team` tool scope added to the prompt gate.
6. **B2. Helper arm** (medium). The seats: read the person's words (T1's "unsure"); choose and word the clarifying
   question; explain the report (checked by `dclab_rnd/cited.py`: cites its source, numbers in it); a risk list
   (advisory, checked by `critic_gate.py`).
7. **B3. Agent arm** (large). A model loop on `agents/runtime.py` `run()` with the team tools: plan first, kept
   whole; free choice of actions, questions and hand-overs; the Lead breaks the request into team tasks itself.
   Reports: the model writes findings and a recommendation; code attaches numbers and refuses any number that does
   not trace to an action.
8. **B4. The comparison harness and the first dev comparison** (medium). New `dclab_rnd/agent_eval/team_arms/`,
   copying `planning_gate_v3` (pinned terms, run-once marker, no Claude, caps). First comparison for the Lead and
   Data Science on the renewal and delivery cases. This gives the first real evidence early.

### Part C. Real evidence and decisions

9. **T4a. Team work reads the project's real tables** (medium). A governed move `set_team_inputs`; new
   `teamspace/evidence.py`; the Data Science check runs on the real data card (`dclab_core/meet.py`).
10. **T5b. Approve, correct, re-approve and log the outcome in the thread** (medium). Routes and the approval control
    in `team.html`.
11. **T4b. ML Engineering runs real stages** (large). The team tool `team_run_stage` starts a studio job through the
    graph check and the holdout-once rule; the report is built from lineage. In arm A the model designs the
    experiment (inputs, candidates); the engine runs it with its leak-safe pipeline.

### Part D. The full comparison and the decision

12. **B5. Arms for every role** (medium): ML Engineering, Sales, Marketing, Business.
13. **B6. Dev comparison on the full journeys**, including a real upload; improve the arms on the dev set only.
14. **B7. The sealed comparison, once**; then the switch is set per role from the result, with the owner's decision
    where the result is unclear.

**Out of scope:** real ETA and first-offer acceptance models (the code has only invented fixtures for them).

## Verification

- **Per package:** its test files on files and PostgreSQL; `make core-schemas` and `make core-wheel-check` when the
  core changed; `make web` and `make catalog` when routes or pages changed; `make agent-eval` and `make harness-eval`
  when a prompt, a tool scope or the graph changed; `make sft-dev`.
- **Journeys end to end:** renewal against cancellation; ETA 18:32 → 18:40 → 18:32 → 18:40; Marketing and Sales in
  either order; an invented uploaded table to a report traced to its run; approve 50, correct to 30, re-approve.
- **Browser:** `DCLAB_BROWSER_CHANNEL=chrome .venv/bin/python scripts/browser_e2e.py --only teamwork` (the installed
  Chrome; no download).
- **Live:** every comparison run goes through the gateway under `agent_runs/live_tests/<run>/`; every answered call
  keeps its thinking; records archived with `make archive`.
- **At the end:** `make check-all`.

## Risks, said plainly

- **Invented numbers come back with freedom** (76 against 0 on 11-15). The strict layer must refuse them, and they
  are a hard zero in the comparison.
- **The cheap model may make the agent arm look worse than it is.** That is why a strong open model runs too.
- **Cases written for the rules favour the rules; cases no rule covers favour the model.** Both families are in the
  set, and results are reported per family.
- **More AI costs time.** Think first added about 40 seconds a conversation. Time limits are fixed before the run.
- **The result may be mixed:** the agent arm may win for the Lead and Data Science and lose for Sales. The per-role
  switch is built for that.
- **Size:** this is about 14 packages. The first real comparison (step 8) comes after 8 of them.
