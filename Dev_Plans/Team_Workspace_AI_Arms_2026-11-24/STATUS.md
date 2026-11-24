# Team workspace with more room for AI: what was built, what was measured, what remains

**Status report · 24 November 2026 · Code at `0386af3e` (dclab-core 0.173.2) · Work PARKED**

This document closes the current round of work on the team workspace. It says what was built, what the first
measurements showed, and what is left. It is written so that the next person (or session) can pick the work up
without reading the conversation it came from.

Related plans:

- The approved plan for this work: [PLAN.md](PLAN.md) (parts A to D, packages T1 to T5b and B1 to B7).
- The owner's earlier plan it builds on: [Team_Workspace_Implementation_and_Live_Test_Plan_2026-11-22.md](../Team_Workspace_Implementation_and_Live_Test_Plan_2026-11-22.md)
  (packages W1 to W6, called T1 to T6 here; T6 was dropped and replaced by part B).
- The earlier handover: [HANDOVER_team_workspace_2026-11-20.md](../HANDOVER_team_workspace_2026-11-20.md).

**Decision (the owner, 2026-11-24):** this plan is parked. The next plan starts fresh. Every team role stays on the
rules until a comparison shows that more AI performs better.

---

## 1. The question this work answers

DCLab has a team of agents that answer a business question together. Example: Sara, in Business, asks "Which
customers might not renew, so Sales can call them?". The Lead plans the work, Data Science checks the data and the
target, ML Engineering ranks the customers, Sales says how many calls it can make, and Marketing says what offer it can
give.

The owner's question (2026-11-23): **who should do the thinking for each team role?** The owner asked for more AI, but
only where it really performs better. So the work built three ways ("arms") for each role, behind one switch, and a fair
race between them:

| Arm | Who decides | What the model does |
|---|---|---|
| **R. Rules** (the default) | Code | Nothing. Fixed steps, the same every time. |
| **H. Helper** | Code decides; a model helps at five named seats | Reads the person's words when the keyword reader is unsure, suggests teams, words a question, explains a report, lists risks. Each answer is used only where a code check passes. |
| **A. Agent with room** | The model decides, inside the strict layer | Writes its own plan first, then chooses its actions, questions and hand-overs through the same guarded tools. |

**The strict layer is the same in every arm and is code only:**

- who may do what;
- the task state machine;
- every number in a report comes from a tool or a person, never from the model's words;
- no action is claimed without its record;
- only the owner approves;
- nobody contacts customers.

This is what makes any amount of AI safe.

---

## 2. What was built (all pushed to `main` and `dev`)

### Part A. The strict layer (the owner's T1 to T5a)

| Package | Commit | What it does, in plain words |
|---|---|---|
| **T5a** Only the decision's owner can approve | `49be4ad8` | Before: anyone who could write to a problem could record another person's approval. Now only the owner, in person and holding the permission, approves; never an agent. An approved decision cannot change; a change is a new decision that names the old one. dclab-core 0.169.0. |
| **T1** Feasibility follows what Business asked | `7c063db0` | Before: Data Science said "feasible" for a renewal question when the data's label was cancellations. Now the task carries its confirmed target (`dclab_core/target_contract.py`); a mismatch blocks with what is missing and who must supply it. The keyword reader says "unsure" instead of guessing. |
| **T2** Corrections name their base | `7c063db0` | Before: a crash re-scored an unfinished attempt, and a correction to 0 was lost. Now a correction names the revision it is based on; a late, older one is refused; 0 is a real value. |
| **T3** Sales and Marketing can answer | `7c063db0` | Before: Sales and Marketing had no agents, so their tasks waited forever. Now `teamspace/commercial.py` answers capacity and budget from the members' own words; a team with no agent says "unsupported" at once. dclab-core 0.170.0. |

### Part B. AI involvement and the race (replaces the old T6)

| Package | Commit | What it does, in plain words |
|---|---|---|
| **B1** The brain switch | `c88bf65f` | One owner-only, audited switch per role: rules, helper or agent. Default: rules. `DCLAB_TEAM_BRAIN` overrides it for comparison runs only. New model purposes `team_helper` and `team_agent`; every call keeps its thinking summary. dclab-core 0.171.0. |
| **B2** The helper arm | `9115e36d` | A model answers at five seats (`read_target`, `route_teams`, `word_question`, `explain_report`, `list_risks`). A refused, late or invalid answer falls back to the rules, and the reason is recorded. dclab-core 0.172.0. |
| **B3** The agent arm | `25c6e830` | The model plans first (kept whole), then acts through the guarded tools. Retries never repeat an action (a lease on the delivery, each write counted before it runs). The rules take over only before the first write. Four review rounds found 15 problems, among them the model recording a person's answer that was never given. dclab-core 0.173.0. |
| **B4** The race track | `ba5a714f` | `dclab_rnd/agent_eval/team_arms/`. It plays each written case through the real team runtime, once per arm and model, and scores it **from the records only**, never from what the model says. The winner rule is pinned by hash before any run. Cases were written by an independent case author that never saw the arms: 40 practice (dev) cases and 70 secret (sealed) cases, in 11 families. |
| **Smoke-run fixes** | `f2514ea4` | The first tiny live run (3 cases, EUR 0.05) showed that 3 of the agent's 4 "hard zeros" were the scorer misreading ordinary text ("(1) (2)" read as invented numbers, "T-7" read as -7). It also found one real bug: an agent Lead could hand work to Business, so the case hung. All fixed. dclab-core 0.173.1. |
| **B4.1** Round-1 fixes | `34fe927c` | Fixes from the first dev race (section 3). dclab-core 0.173.2. Details below. |
| **B4.2** Choices as a list | `0386af3e` | The values a question offers the person now travel as a separate list, never guessed from the words. Details below. |

**B4.1 in detail:**

- **The race stops when the AI credit runs out.** Before, a refused call made the arm quietly use the rules, which
  made those games invalid. Now such a game is kept aside to replay; after two refused tries it is final and scored
  "incomplete".
- **The scorer no longer gives unfair zeros.** It used to read "an approved offer" as an approval and "seven days
  before renewal" as a claim that scoring was done. The product's own claim check (`dclab_core/team_brain.py`) was
  fixed in the same way.
- **The agent arm:**
  - it uses the real team names (7 of 10 hand-overs had bounced on a wrong name);
  - it does not ask again what the person already said;
  - it never blocks a task while its question is open;
  - its questions and hand-overs are checked for numbers and claimed actions before they are sent.
- **The rules arm:** a request about an offer, discount, budget or campaign also goes to Sales and Marketing. The
  rules had forgotten Marketing.
- **The cases:** they were revised by their independent author, and new hashes are pinned in `cases/MANIFEST.json`.

**B4.2 in detail:**

- **Why it was needed.** B4.1 needed six review rounds to guess, from a question's words, which numbers were choices
  for the person ("7 or 14 days?") and which were facts. Each round found a new wording that let an invented number
  through, for example "Is the churn rate 12% or 15%?".
- **The new interface.**
  - `team_clarify` takes `options`: 2 to 6 short lines.
  - Code writes them under the question in one fixed form, "Choices for you to pick: (a) ... · (b) ...".
  - Code reads them back exactly (`tools.read_choices`).
- **The checks.**
  - The question's own words are all treated as facts: every number in them must be on record.
  - An option may carry a new number only when it is a value to pick, such as a window or a threshold.
  - An option is never free when it is a count of the data ("812 rows have no date"), a measured result ("Holdout AUC:
    0.81"), or sits under a question that checks a fact.
- **No store change.** The text travels whole through the relays.

**Quality checks on everything above:**

- Every package was reviewed by an independent reviewer, and the findings were fixed with tests. In total: more than
  60 problems found and fixed before the first race, and 8 more in B4.1 and B4.2.
- The latest test run: 417 tests in the 12 affected files pass, both on files and on PostgreSQL.
- `make agent-eval` (AEV-138) and `make harness-eval` (HRN-148, 17 of 17) pass after the prompt change.
- `make core-wheel-check` passed on dclab-core 0.173.2.
- Development records for training: `research/llm-fine-tuning/experiments/dev_trajectories/records/team-b41-dev-round-1.json`
  and `team-b42-clarify-options.json`, plus the earlier packages' records. `make sft-dev` has been rebuilt.

---

## 3. What was measured

**Ground rules for every run:**

- No Claude model, in any arm or judge.
- Model: `qwen/qwen3.8-flash` through OpenRouter.
- Every exchange is kept with its thinking summary under `agent_runs/live_tests/`.
- At most EUR 0.5 per run.

### Smoke run (`agent_runs/live_tests/2026-11-23-233112-team-arms-quick`, EUR 0.054)

3 cases × 3 arms. Rules 3 of 3, Helper 2 of 3, Agent 0 of 3. Most of the agent's hard zeros were scorer mistakes, fixed
in `f2514ea4`.

### First dev race, round 0 (`agent_runs/live_tests/2026-11-24-020928-team-arms-dev`, EUR 0.49)

19 valid dev cases. A second sitting was **invalid**: the OpenRouter account ran out of credit ($20.19 of $20), every
call got a 402 refusal, and the AI arms silently used the rules. Those plays were thrown away.

| | Rules | AI helper | AI agent |
|---|---|---|---|
| Cases passed | **9 of 19** | 8 of 19 | 2 of 19 |
| Real safety breaks (hard zeros) | 0 | 0 | 0 |
| Mean time per case | 0.4 s | 38 s | 374 s (limit 300) |
| Mean cost per case | EUR 0 | EUR 0.002 | EUR 0.024 |

An independent analyst checked every failure. All main-score failures were real. All 10 hard zeros were scorer false
positives, now fixed: rescoring all 114 round-0 plays read-only under the latest scorer gives 0 invented numbers.

**Why each arm failed:**

- **AI agent:**
  - It got stuck after the person answered: 9 attempts to pass the answer on, wrong each time, then out of time.
  - It used wrong team names, so 7 of 10 hand-overs bounced.
  - It asked again what the person had already said ("7 days before renewal"), then blocked the task, so the answer
    could not arrive.
  - It was too slow.
- **Rules:**
  - They asked the churn question even on a delivery-time request.
  - They forgot Marketing when the person asked for an offer.
- **AI helper:** about as good as the rules. Once it guessed the person's meaning instead of asking, and guessed wrong.

**What it means:**

- More freedom did **not** make the AI better yet: the more the model decided, the worse it did.
- The failures are specific and fixable, and B4.1 fixes them.
- This was one small race on a cheap model. It is not enough to choose a winner.
- **Decision for now: every role stays on rules.**

A simple 5-page report for non-technical readers: `output/DCLab_Team_AI_Race_Simple_Report.pdf`.

---

## 4. What remains

### Runs (blocked on AI credit; nothing to build)

| Step | What | Cost (estimate) | Blocked by |
|---|---|---|---|
| Round-1 dev race | The 40 revised dev cases, all three arms, on the fixed code (B4.1 and B4.2). This is the first test of the new options field with a live model. | about EUR 1.5 | OpenRouter credit; the owner declined a top-up on 2026-11-24 |
| Strong-model race | The same dev race on the strongest open model on OpenRouter (for example Qwen 27b or Nemotron Ultra). A weak model may hide what room can do. | a few euros | credit |
| Sealed race, once | The 70 secret cases, run once and whole, under the pinned gate (`gate_v1.sha256` = `198b8bee…`). Never open or tune on the sealed cases. | a few euros | credit, and the dev races first |
| Archive | `make archive` once per whole comparison (about 4 GB per snapshot; keep 2 GB free; the ADATA drive was not mounted on 2026-11-24). | free | disk space |

**How to run (when credit exists):**

```bash
.venv/bin/python -m dclab_rnd.agent_eval.team_arms plan --set dev
.venv/bin/python -m dclab_rnd.agent_eval.team_arms run --set dev --arms R,H,A --yes --cap-eur 0.5
.venv/bin/python -m dclab_rnd.agent_eval.team_arms report --set dev
```

`plan` prints what a run would cost and does not spend. A run cut by the cap goes on with `--resume <run folder>`.
Start a fresh run; the round-0 run is never resumed (resume refuses changed code by design). After the run, verify
that the answered exchanges kept their thinking, then `make archive`.

### Packages to build (parts C and D of the plan; none started)

| Package | Size | What it adds | Example |
|---|---|---|---|
| **T4a** Team work reads the project's real tables | medium | A governed move `set_team_inputs`, new `teamspace/evidence.py`; Data Science checks the real data card (`dclab_core/meet.py`), not a fixed sample. | Sara uploads her renewal table; Data Science finds that "renewed_by_date" is missing and blocks with that. |
| **T5b** Approve, correct, re-approve and log the outcome in the thread | medium | Routes and the approval control in `team.html`; the missing route for `confirm_outcome` (open since T3). | Approve "call 50", correct to 30, approve again, all in the thread. |
| **T4b** ML Engineering runs real stages | large, the riskiest | The team tool `team_run_stage` starts a studio job through the graph check and the holdout-once rule; the report is built from lineage. In the agent arm the model designs the experiment and the engine runs it. | The ranking of customers comes from a real trained model, not a fixture. |
| **B5** Arms for every role | medium | Helper and agent arms for Business, ML Engineering, Sales and Marketing. Today the Sales and Marketing agent arm cannot use the commercial tools (they need the member's own words). | Marketing's agent proposes an offer within the budget its member gave. |
| **B6** Dev race on full journeys | run + fixes | Full journeys including a real upload; improve the arms on dev only. | |
| **B7** Sealed race, once, then set the switches | run | Each role is set to its winner; the owner decides where the result is unclear. | |

Suggested order: T4a → T5b → T4b → B5, each reviewed, tested, committed and pushed. Then B6 and B7 when credit
exists.

### Small open items

- T2: after a crash, no "applied" event is written for the interrupted write.
- B3: the claim check in the product is still a word-pattern check; the race scores claims from the action log, which
  is the real guard.
- The helper and rules arms' questions are still read by the old word rule (`split_choices`) in the scorer, so they
  stay comparable with round 0; only the agent arm uses the options list.
- Deciding whether an option is "a value to pick" is still a word rule; it errs toward treating a number as a fact.
- A browser check of the team page (`DCLAB_BROWSER_CHANNEL=chrome .venv/bin/python scripts/browser_e2e.py --only teamwork`).
- The full gate `make check-all` has not run on `0386af3e`; only the affected test files did. Run it at the next
  nightly gate.
- `.env.example` lines `DCLAB_LEAD_BOUNDED_CHOICE` and others belong to another session; a clean checkout fails
  `test_config` until that session commits them.

---

## 5. Risks, said plainly

- **Invented numbers come back with freedom.** On 2026-11-15 a free agent made 76 numbers nobody gave, against 0
  inside the harness. The strict layer refuses them, and the race scores them as a hard zero. Every new arm must keep
  this.
- **A cheap model can make the agent arm look worse than it is.** The strong-model race is needed before any choice.
- **The cases can favour one arm.** Cases written for the rules favour the rules; cases no rule covers favour the model.
  Both families are in the sets, and results must be reported per family.
- **More AI costs time.** The agent took 374 s per case against a 300 s limit.
- **Word rules have limits.** The claim and choice checks are word rules, and new wordings may find gaps. B4.2 replaced
  the worst one with a structured field. The same idea (the model states its intent in a field code reads exactly)
  should be preferred over new word rules.

---

## 6. Where everything is

| What | Where |
|---|---|
| Brain switch, helper seats, agent arm | `dclab_rnd/teamspace/brains.py`, `seats.py`, `agent_arm.py`, `tools.py` |
| Core contracts and rules | `dclab_core/team_brain.py`, `team_work.py`, `target_contract.py` (0.173.2) |
| Race track (harness, scorer, gate, cases) | `dclab_rnd/agent_eval/team_arms/` |
| Prompts | `dclab_rnd/prompts/team_agent.md`, `team_helper.md` and the five `team_helper_<seat>.md` |
| Tests | `tests/test_team_arms.py`, `test_team_agent.py`, `test_team_helper.py`, `test_teamspace_agents.py`, `test_core_team_work.py` |
| Live run records (gitignored) | `agent_runs/live_tests/2026-11-23-233112-team-arms-quick`, `2026-11-24-020928-team-arms-dev` |
| Development records | `research/llm-fine-tuning/experiments/dev_trajectories/records/team-*.json` |
| Simple report | `output/DCLab_Team_AI_Race_Simple_Report.pdf` (and `.html`) |
