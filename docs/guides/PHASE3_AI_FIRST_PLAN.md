# Phase 3: Claude beside the system, on every step, and a Home that understands

Written 2026-10-16 from the owner's request for phase 3 of the five phases, over the repository at `790fe58`
(phases 1 and 2 built: capture, the problem map, the router, the solution map, programmes).

The owner's words, in short: *Home does not understand problems well; I want much more interaction and intelligence.
At every step (the Home conversation, the important decisions, the solution draft, the project's stages) I want to see,
at the same time, how Claude solves the problem as an AI-first agent and how the current system handles it, side by
side, with the differences shown; everything both approaches do kept in a separate, organised, findable folder, and
turned into training data under the hood; and then to know where Claude is most effective, where the combination of AI
and deterministic software is best, and where software alone does better, so I can make DCLab more agentic in the
right places.*

The owner's answers (2026-10-16): Claude may see the data card **and a small redacted sample** (about 20 rows) for this
arm only, off by default per workspace; the arm runs on **every session behind one owner switch** per workspace (off by
default); it may spend **$1.5 a session** (prices here are entered in dollars and shown as euros, so the cap is 1.5), with
a monthly workspace cap on top.

---

## Part A. What exists and what is missing

| Need | Today | Missing |
|---|---|---|
| A Claude answer beside the system's | `models/shadow.py` shadows one request with a **local** tier only (`gateway.py` drops a remote shadow) and compares the first tool call; `ledger/shadow.py` adds fitted classifiers as shadow answers; `answerers.ModelAnswerer` answers one decision from its state | an agent that solves a whole step the way Claude would (its own reasoning, its own tools, the data sample), not a copy of the system's request |
| The same inputs for both arms | the Home agent's `context(draft)` (names and aggregates), the data card, the decision states (bands) | one recorded *input* per step, its fingerprint, so a difference is a difference of approach and never of input |
| The differences | `same_tool` / `same_arguments` per request; ledger agreement per point | a typed difference per step (fields, sets, order, text), scored against the truth when the truth is known |
| A separate, organised store | the experience log, the ledger, `agent_steps`, `model_shadow` (spread out, by design) | one folder per session with both arms side by side, an index, a README, and the training samples built from it |
| Where Claude, software or both are best | the role switches (one bar per role, rules vs model) | a matrix per step: each arm's accuracy, agreement, cost and time, and the recommended mode by a rule fixed before the runs |
| Home understanding | a deterministic order of questions (outcome, moment, action, pack fields), a guard on every record; the Home agent with a model is constrained to the same tools; the coordinator runs only in the journeys; the `plan` event is not drawn; the prompt says at most 4 questions and the code allows 5 | a live understanding panel, questions grounded in the data with their reason, a "what I understood" summary, the person able to correct any field, and the AI arm's measured strengths adopted where they win |

## Part B. The shape

```text
            person's message / data ready / a decision / a stage
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          SYSTEM ARM (unchanged)     AI-FIRST ARM (Claude, 25.3)
          rules, roles, guard,       its own loop: reads the same input,
          the coordinator            the data card, a redacted sample,
                 │                   asks aggregate questions of the table
                 │                   through read-only tools, answers in
                 │                   the step's typed shape; never writes
                 │                   to the system's state
                 ▼                         ▼
              TwinStep (25.1): the input fingerprint, both answers, the differences,
              cost and time of each arm, the truth when known
                              │
        ┌─────────────────────┼──────────────────────────┐
        ▼                     ▼                          ▼
   twin/ folder (25.2)   side by side on Home (25.5)  effectiveness matrix (25.8):
   sessions, both arms,  and on the project (25.6)    where Claude, software, or the
   steps, index, SFT                                   combination is best
```

**Rules that hold in every package.**

1. The system arm is never changed by the AI arm: the AI arm reads a copy of the inputs, and its tools are read-only.
   What the person sees from the system is what they would see with the switch off.
2. Both arms get **the same input** for a step; the step records its fingerprint. The AI arm may additionally use its
   own tools (that is its approach), and each call is recorded.
3. The sample sent to Claude is at most 20 rows, redacted (`dclab_core.redact`: identifiers, e-mails, phone numbers,
   free text over a length cut to its shape), never a column the person marked sensitive (23.7), only with the
   workspace's sample switch on. The rows are never stored in the twin folder: its fingerprint is.
4. Spend is capped per session (1.5) and per month per workspace (default 30), through the gateway's `run_limit_eur`
   and caps; the arm stops and says so when a cap is reached. Tests use scripted transports only.
5. The truth decides who was right, never agreement: planted truth on replicas and journeys, the person's pick when
   they choose between the two, measured outcomes (a holdout number, a leak confirmed by its lift) when they exist.
6. Training use follows consent (19.10): the folder keeps everything for the workspace; samples leave only through
   `consent.export_*` with the train consent.

## Part C. Packages 25.x

Two inputs from the owner shaped this part after its first draft (2026-10-16): the twin runs of packages A6.6 and A6.7,
built on the branch `sft-factory-all-stages` (commits `4934ad0`, `94800da`), already ran the current system and Claude
on one problem in two lanes, and watched how Claude designs a solution and its workflow; and the owner wants **as much
as can be had from Claude**: how it solves the problem, how it designs the solution, how it designs the workflow,
monitored and stored. That branch left main 114 commits before `790fe58` and rests on the A6.5 decision samples, which
main replaced with the ledger (19.6, 19.7); it is brought onto main here, not merged: its lanes, its design monitor and
its folder are kept, its decision samples become ledger records, and Claude is reached through the gateway's native
Messages transport (which returns Claude's thinking, which the OpenAI-compatible endpoint did not).

### 25.1 The twin contract
- **Delivers:** `dclab_core/twin.py`: `TwinStep` v1 (`step` from a closed list of steps, the input and its
  fingerprint, `system` and `ai_first` as `ArmAnswer` (value in the step's typed shape, text said, tools used, the
  validator's verdict, tokens, cost, seconds, status), `differences`, `truth` and its source, `verdict`), the step
  catalogue (`STEPS`, plus `decision:<point>` for every ledger point; each field compared as equal, set, ranked,
  numeric, text by key terms, or mapping), `compare`, `judge`, and `effectiveness(steps)` into a `TwinMatrix` v1: per
  step, each arm's accuracy with its interval, agreement, what one arm got right that the other missed, median cost and
  time, and the mode by `twin_rule_v1`, fixed before any run: `software`, `software_reviewed` (software proposes, the
  AI reviews), `ai_checked` (the AI proposes, the software checks), `ai`, `undecided` below 10 judged steps; ties go to
  the software. Adapter: the main product's `semantic_decision_answers` (the rule beside Jev, a label row the truth) as
  twin steps.
- **Done when:** equal answers have no differences; a set field reports what each arm alone gave; the rule gives each
  mode on constructed steps; a step whose two inputs differ is refused; a rule-and-Jev pair with its label is a judged
  step.
- **Status: done (2026-10-16).** dclab-core 0.56.0 (`twin`: `TwinStep` v1, `TwinMatrix` v1, `STEPS`, `compare`, `judge`, `effectiveness`,
  `twin_rule_v1`; `adapters.main_product_twin`). Left open: nothing writes twin steps yet (25.2 and 25.4 do).

### 25.2 Twin runs on main: the whole problem in two lanes
- **Delivers:** `dclab_rnd/twin/` from A6.6, on main: one problem (a sample, a project, a draft's conversation), two
  lanes at the same time in two threads, each a complete workspace folder under `<workspace>/twin/runs/<run>/`
  (`workflow`: the current system, the standard plan through the graph validator; `ai_first`: Claude as the agent with
  the same tools and the same validator, and Claude serving every role); the purpose `ai_first` and the agent `twin` in
  the harness; every request and answer kept verbatim in `transcripts/`; each lane's decisions as ledger records in its
  own folder (the A6.5 samples replaced); `compare.json`, `COMPARE.md`, and the comparison as `TwinStep`s
  (`steps.jsonl`) on the solution, the stage choices and the holdout; caps on requests, euros (1.5 a run by default),
  tool calls and minutes; nothing sent without `--yes`, refused under `DCLAB_NO_LIVE_MODELS`.
- **Done when:** a run on a scripted transport writes both lanes, the comparison and the steps; the workflow lane's
  stage records equal a plain run's; the main workspace is never written; a run over its cap stops with the reason.

### 25.3 Claude the expert: how it understands, analyses and designs, monitored and kept
- **Why:** the owner (2026-10-16): store how Claude understands the problem and the data, how it analyses and explores
  the data (EDA), how it would prepare it, how it decides the type of the problem, the kind of machine-learning problem
  and the algorithms, how it draws the problem's workflow (its parts) and the solution's workflow; Claude treated as a
  machine-learning expert of twenty years who has solved every kind of problem; then put those capabilities into
  DCLab's agents.
- **Delivers:** `dclab_core/expert.py`: `ExpertTrace` v1, the expert's solution as typed sections, each beside the
  system's own record of the same thing so the two compare field by field: `problem` (restatement, the decision it
  serves, the unit, the moment, the outcome and how it is known, success, constraints; against the problem card),
  `problem_type` (family, kind, task, labels situation, why, the alternatives it set aside; against the card and the
  map's families), `data` (what each table is, its grain, the roles of the columns, identifiers, times, groups, the
  leak suspects with why; against the data card), `eda` (each question it asked of the table through its tools, the
  aggregate answer, and what it concluded; against the data findings), `preprocessing` (each step, why, fitted inside
  the folds or not; against the engine's recipe), `problem_workflow` (the parts and sub-problems and how they compose;
  against the problem map), `algorithms` (families ranked with why, the baseline, what would change the choice; against
  the router and the models stage), `evaluation` (split, metric, the holdout read once, the threshold; against the
  solution map), `solution_workflow` (steps on WF-01…WF-10 with the tool, the why, the evidence, the decision rule, the
  dependencies and the routes back; checked by `solution_workflow.validate`), `risks`, `questions` (what it would ask
  the person); `compare_expert(trace, system)` into twin steps. `prompts/ai_first_expert.md` (the twenty-year expert
  persona, the sections, the evidence rule: aggregates and the redacted sample only, every claim tied to a tool result).
  In the twin runs (A6.7 on main): the design before acting, the monitor's timeline (each call: what Claude said, the
  design step it named, in order, out of order, unplanned, refused), the revisions after the checkpoint stages with
  every change and the number behind it, the retrospective, `adherence.json`, `DESIGN.md`; **Claude's thinking** (the
  Messages API's summarised thinking, `DCLAB_TWIN_THINKING`) kept beside each reply in `thinking.jsonl` and threaded
  back as the API requires; the expert's decisions written as ledger shadow answers (`answerer="ai_first@expert"`).
- **Done when:** a scripted run writes the trace, the design files and the thinking; each section compares with the
  system's record as twin steps; a trace whose solution workflow fails `solution_workflow.validate` is kept with the
  reasons and never used.

### 25.4 The twin on Home, live
- **Delivers:** the workspace switch `twin.json` (on, sample, session cap 1.5, monthly cap 30), owner only and audited
  (`/api/twin/settings`); with it on, every Home event (start, data ready, each message, the map, the routes, the
  workflow, the readiness to build) also runs Claude in the background, on the same input, with the data card and the
  redacted sample, and writes a `TwinStep` per step (`home.understanding`, `home.next_question`, `home.pack`,
  `home.map`, `home.route`, `home.workflow`, `home.ready`) under `<workspace>/twin/sessions/YYYY/MM/DD/<session>/`
  (`session.json`, `system.jsonl`, `ai_first.jsonl`, `thinking.jsonl`, `steps.jsonl`), an `index.jsonl` and a
  generated `README.md`; Claude keeps its own conversation: the person writes to both (default), to the system only, or
  to Claude only; SSE events `twin`; experience kinds `twin.session` and `twin.step` pointing into the folder; deleting
  a draft removes its twin sessions.
- **Done when:** with the switch off nothing is written and no model asked; with it on a scripted conversation leaves
  the same system state as with it off and a session folder whose steps cover each Home step reached; a failing Claude
  never fails the system's turn; a member who is not the owner cannot change the switch.

### 25.5 Side by side on Home, and the understanding panel
- **Delivers:** Home with two lanes when the switch is on (the system, Claude), a differences strip under each step,
  a choice per difference ("this one is right") stored as the step's truth `person_pick`; Claude's design and thinking
  readable from its lane; the live understanding panel in both modes: the problem card's fields filling as the
  conversation goes, each with its quote and status, editable by the person (an edit is a `record` with the person's
  words, through the guard); the `plan` events drawn; the reason of each question shown.
- **Done when:** the browser test drives a conversation with both lanes at 1280 and 375 px; a pick is stored as the
  step's truth; an edit in the panel changes the card through the guard.

### 25.6 The twin on a project: start a twin run from the page
- **Delivers:** "Run with Claude beside" on a project (owner switch on): a twin run of 25.2 and 25.3 seeded from the
  project as a job, its progress and its `COMPARE.md`, `DESIGN.md` and the step differences on the project page; the
  run's stage choices written to the project's ledger as shadow answers (`ai_first@v1`), never as the project's own.
- **Done when:** a job run on a scripted transport leaves the project's records unchanged and its decisions with an
  `ai_first` shadow answer; the page shows both.

### 25.7 Capture into training data: every decision Claude makes, for the decision models
- **Why:** the owner (2026-10-16): capture and store every decision and every step Claude makes, as data for the
  decision models of the next phase. These are **different models for different uses**: (1) **workflow decision
  models** that route, score, classify and decide inside DCLab's workflows (the pack, the next question, the route of
  a part, the tool, whether to stop, a stage's approval); (2) **data-science decision models**, classifiers in the
  manner of the main product's Jev, for machine-learning work (a column's role, an identifier, a leak, the split, the
  metric, the family, the threshold). Jev is one example; an open model the owner chooses can be tuned on the same sets,
  or a model built and trained from scratch.
- **Delivers:** every Claude answer in the twin (the AI-first lane's moves and approvals, the Home twin's steps, the
  expert trace's sections) written to the ledger at its point, with the same state the rule saw (bands, names, codes;
  never a row), as an answer `by="model_role"`, `answerer="claude:<model>@<role>"`, `source="shadow"`, so the rule's
  answer, Claude's, the person's and the truth sit on one record; a label source `teacher` for Claude's answers used as
  labels, kept apart from a person's and never outranking a planted or measured truth; the dataset builders
  `twin_trajectories` (each lane as a trajectory), `twin_designs` (Claude's designs and expert traces with what became
  of them), `twin_preferences` (pairs where the truth chose one lane), and `decision_teacher` (per point: state,
  options, the rule, Claude, the person, the truth), split by use (`workflow` points and `technical` points, the
  ledger's two kinds) with the sealed partition kept; consent-gated, in the catalog and `EXPERIENCE_ARCHIVE.md`; the
  twin folder's README says where each kind of record lives.
- **Done when:** a twin run on a scripted transport leaves Claude's answers on the lane's ledger records beside the
  rule's; a workspace without train consent builds nothing; with it, the sets are written per use without a row of the
  sample, and a teacher label never replaces a planted one.

### 25.8 Where Claude, software, or both: the measurement
- **Delivers:** `make twin-eval`: both lanes on the replicas and the dev journeys, where the truth is planted, step by
  step, capped (≤ 0.5 a run); the matrix of 25.1 in `evidence/campaigns/twin_v1/`; the same matrix from the live
  sessions (person picks, measured outcomes); the Admin page "Claude and the system" (per step: each arm's accuracy,
  agreement, cost, time, the mode); a written analysis.
- **Done when:** the dev run is stored with its matrix and every step has a mode or "undecided" with its count.

### 25.9 DCLab learns from the expert: adopt what won
- **Delivers:** from 25.8's matrix, per section of the expert trace and per Home step: where the mode is `ai_checked`,
  Home and the studio use Claude's answer, accepted only after the step's validator (the rules' when refused), one
  audited switch per step as the role switches; where the mode is `software_reviewed`, Claude's disagreement is shown
  to the person as a second opinion; the expert's sections that win become roles of `dclab_core` with their prompts
  (understanding, EDA questions, problem type, algorithms, workflows), so the main product gets them through the
  wheel; the Home prompt fixed (its question cap); questions with options grounded in the data; the expert traces
  that the truth vindicated become the SFT set for the owner's own model (25.7).
- **Done when:** a step switched on is measured not worse on the journeys (`make journeys`, `make harness-eval`
  17/17), and a refused answer falls back to the rules.

## The next phase: fine-tuning the decision models

Phase 3 collects the data; the next phase (its own plan, written when the sets of 25.7 hold enough) fine-tunes on it:
one family of models for the workflow decisions inside DCLab (routing, scoring, classifying, deciding the next move)
and one for the data-science decisions (Jev-like classifiers of columns, leaks, splits, metrics and families), each
an open model the owner picks or one trained from scratch, each measured against the rule and Claude on the sealed
partition before it serves (the ledger's routing levels, 22.6). The recipe of 22.5 (`decisions/finetune/README.md`)
is the starting point; the run is the owner's.

## Part D. For the main product

| Package | In `dclab_core` | Here | There |
|---|---|---|---|
| 25.1 | `twin.py`: `TwinStep`, `STEPS`, `compare`, `judge`, `effectiveness`, `twin_rule_v1` | — | adapter: its governed decision points as twin steps |
| 25.3 | `expert.py`: `ExpertTrace`, `compare_expert`; the expert prompt | the design monitor, thinking | the expert runs over its gateway; its trace compares with its own records |
| 25.9 | the roles that won, with their prompts | the switches | the roles through the wheel |
| 25.2–25.7 | — | the twin runs, the design monitor, the folder, the switch, Home, the project page, the builders | the AI-first lane runs over its gateway; its rule-and-Jev pairs are twin steps through the 25.1 adapter |
| 25.8 | the matrix | the measurement | its own run through the adapter |

## Part E. Order, effort, spend

Order: 25.1 → 25.2 → 25.3 → 25.4 → 25.5 → 25.6 → 25.7 → 25.8 → 25.9 (milestone L, its `check-all`). Live use while
building: the dev measurement of 25.8, each run capped at 0.5; the product's arm is the owner's switch with its 1.5 cap.
