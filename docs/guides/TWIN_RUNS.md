# Twin runs: the current system and an AI-first Claude agent, side by side

Package 25.2 (`dclab_rnd/twin`), brought onto main from A6.6 of the branch `sft-factory-all-stages`, in phase 3
(`docs/guides/PHASE3_AI_FIRST_PLAN.md`). One problem, two lanes, at the same time, everything from both kept in one
folder, a comparison, and the comparison as `dclab_core.twin` steps judged against the truth a seed plants. The question
it answers: on the same table, with the same tools and the same validator, what does an AI-first agent (Claude) decide
that the current system (rules, validator, standard plan) does not, and what does each cost?

## What runs

| Lane | Who decides | Models |
|---|---|---|
| `workflow` | the current system: the intern's standard plan (describe → propose → set the solution → the five stages → report), every move through the graph validator | none by default; `--workflow-models configured` gives it the workspace's configured model, as the product has it |
| `ai_first` | Claude as the agent: the same task, the same tools (describe, understand, propose, set the solution, run and approve each stage, get results, finish), the same validator; it chooses the order and the options | Claude for everything: the agent, the stage notes, the leakage reviewer and every role of dclab_core |

Claude is reached through the gateway package's native Messages transport (`models/anthropic_client.py`), with
`ANTHROPIC_API_KEY`; `DCLAB_TWIN_MODEL` names the model (default `claude-sonnet-5-5`). Both lanes run in two threads;
each runs inside its own workspace services (`context.using`), so the gateway, the ledger, the session record and the
audit each lane's code reaches are the lane's own.

## Seeds

- `--sample <key>`: a studied sample (`python -m dclab_rnd.twin samples`); `--blind` strips the R&D's known solution
  from both lanes, so each must find the leak itself. A blind run is judged against it (the target and the columns the
  R&D solution blocks, a time or group column left out); without `--blind` both lanes were given it, so nothing is judged.
- `--project <id>`: a project of the workspace (`--source-home` for another): its table and known solution are copied
  into each lane, never its stages or approvals.
- `--draft <id>`: a draft's chat: the person's turns are replayed in each lane, the standard questions against Claude.

The sealed benchmark is never a seed.

## Where everything is

`<workspace>/twin/runs/<run_id>/`:

| Path | What it holds |
|---|---|
| `run.json` | the manifest: seed, task, model, caps, each lane's status, start and end, model usage, the differences, the planted truth |
| `compare.json`, `COMPARE.md` | the lanes side by side: solution fields, stage choices, holdout, moves, tool calls, usage, time, the final reports, who was right where the truth is planted |
| `steps.jsonl` | the comparison as twin steps (`dclab_core.twin`, 25.1): `solution.columns`, `stage.data`, `stage.features`, `stage.models`, each with both answers, the differences and the verdict |
| `README.md` | this table, for the folder |
| `<lane>/projects/<id>/` | the project: `project.json`, `stages/<stage>.json` (full records), `transitions.jsonl` (every move and verdict), `activity.jsonl` |
| `<lane>/intern/<session>.json` | the agent's whole conversation: system prompt, task, every model reply, every tool result, the final report |
| `<lane>/ledger/decisions/`, `ledger/labels/` | every decision the lane made, by day (19.7): the rule's answer, the agent's, the value used; on main these replace the branch's A6.5 decision samples |
| `<lane>/experience/` | the lane's session record (19.2) |
| `<lane>/agent_steps.jsonl` | one row per tool call |
| `<lane>/audit.jsonl` | the audit trail |
| `<lane>/model_requests.jsonl` | one usage row per model request |
| `<lane>/transcripts/model_calls.jsonl` | every request and answer, verbatim: messages, tools, the reply, tokens, seconds, cost; refusals and provider errors too |
| `<lane>/summary.json` | what the comparison read from the lane |

Each lane is a complete workspace folder; the main workspace is never written. A twin run's files hold prompts, model
answers, column names and the intern session's tool results (which carry three example values per column, as the
product's own sessions do); the transcript's tool results go through the trace's scrub (the keys that carry cells
removed) unless `--capture-cells`. Research material, kept on the machine that made it.

Limits: each lane keeps files only (its stores, ledger, log, audit and usage), also when the workspace is on
PostgreSQL, where the source project or draft is read from; a project marked private is refused unless
`--allow-private`; a cap is checked before a request, so the last request can pass the euro cap by its own cost;
`--roles all` (the default) turns every dclab_core role on for the run, which reaches only the AI-first lane because
the workflow lane has no model.

## Running

```bash
python -m dclab_rnd.twin run --sample bank_marketing                 # the plan only; nothing is sent
python -m dclab_rnd.twin run --sample bank_marketing --yes --max-requests 40 --cap-eur 0.5
python -m dclab_rnd.twin run --project <id> --blind --yes --cap-eur 1.5
python -m dclab_rnd.twin run --draft <id> --yes --cap-eur 0.5
python -m dclab_rnd.twin run --sample bank_marketing --blind --yes --cap-eur 0.5 --design-checkpoints leakage --no-steer  # the record watched, one revision
python -m dclab_rnd.twin run --sample bank_marketing --yes --no-design                                                  # the AI-first lane from the task alone
python -m dclab_rnd.twin list
python -m dclab_rnd.twin compare <run_dir>                            # rebuild the comparison and the steps from the files
```

Caps: a number of requests (always), euros (1.5 a run by default, reserved before each request by its upper-bound cost,
and the workspace's monthly cap with this month's spend on top), the intern's tool calls and minutes. A lane stopped
by a cap is `stopped` in the manifest, with the reason. Without `--yes` nothing is sent;
under `DCLAB_NO_LIVE_MODELS=1` nothing can be.

## Reading a comparison

`COMPARE.md` says where the lanes differ: a solution field, a stage choice, the holdout score (one holdout each,
scored once; a difference inside the intervals is noise), the moves the validator refused, the tool calls and their
order, the requests and euros, the seconds; and, where the seed planted a truth, which lane was right on it. The
matrix of many runs is `dclab_core.twin.effectiveness` over their `steps.jsonl` (25.8).

## How Claude, as the expert, understands and designs (25.3)

On a sample or project seed the AI-first lane is watched as well as run. Before the agent starts, Claude is asked, as a
machine-learning expert of twenty years (`prompts/ai_first_expert.md`, purpose `ai_first`), for its record of the
problem: the problem and how the outcome becomes known, what kind of problem it is and why (with the alternatives set
aside), each table's grain and the columns' roles with the leak suspects and the test that settles each, the questions
it would ask of the data, the preparation (and which steps are fitted inside the folds), how the problem breaks into
parts, the algorithms ranked with why, the evaluation, and the solution's workflow on WF-01…WF-10 with the routes back.
Each version is an `ExpertTrace` (`dclab_core.expert`); the record goes to the agent with its task, and the agent is
asked to name the workflow step before each tool call. A monitor on the agent's client writes the timeline; after the
checkpoint stages (`data`, `leakage`, `models` by default; `--design-checkpoints`) Claude revises the record from that
stage's compact record (what the evidence answered goes into its exploration; every change with the number behind it),
handed to the agent unless `--no-steer`; at the end a final version with a retrospective.

| `ai_first/design/` | What it holds |
|---|---|
| `DESIGN.md` | the story: every version, what changed and why, the timeline, the plan against the act |
| `design_vNN.json`, `design_vNN.md`, `expert.jsonl` | each version as an `ExpertTrace`, as a page, and one line each |
| `designs.jsonl` | one row per request: checkpoint, version, what was wrong in the answer, Claude's raw answer and its thinking |
| `timeline.jsonl` | one row per tool call (what the agent said, the step it named, the step it matches: in order, out of order, not in the record, refused) and per design event |
| `adherence.json` | the planned workflow against the calls |
| `intents.jsonl` | the record's decisions on the ledger's points (`target.column:-`, `feature.leakage_suspect:<col>`, `split.strategy:-`, `model.family:-`, …), per version |
| `sft_candidates.jsonl` | each version as a chat sample with its outcome; Claude's words, `verified: false` |
| `summary.json` | what `COMPARE.md` reads |

The run folder's `expert_steps.jsonl` sets every version beside the current system's answers as twin steps (judged on a
blind seed), and `COMPARE.md` has **How Claude, as the expert, designed it**. Claude's thinking (`DCLAB_TWIN_THINKING`:
`adaptive` by default, `enabled:<budget>`, `off`) comes back through the native transport and is kept in
`ai_first/transcripts/thinking.jsonl`, beside each reply; when the API refuses thinking, the request is asked once more
without it and the run goes on without it. What the expert never sees: an example value, the target's values, and
before the end a holdout score. The agent of the AI-first lane does see the intern's tool results, which carry three
example values per column as the product's sessions do; its thinking and its words are free text, kept whole and never
scrubbed (a key-based scrub cannot reach free text), so `thinking.jsonl` and `model_calls.jsonl` may quote a value of
the table: they stay on the machine that made them, and a run on a confidential table stays local.

## On Home, live (25.4, 25.5)

The same expert answers beside the Home agent on every conversation when the owner's switch is on
(`/api/twin/settings`, `twin.json`: `on`, `sample`, `session_eur` 1.5, `monthly_eur` 30). After each turn of the Home
agent, Claude reads the same conversation, the draft's data card, the findings and, with `sample` on, up to 20 redacted
rows (a column that holds a person's data left out), and answers every Home step (`home.understanding`,
`home.next_question`, `home.pack`, `home.map`, `home.workflow`, `home.ready`) with its reply to the person; each step is
a twin step beside the system's state after the same turn, in `twin/sessions/…/steps.jsonl`. The page shows Claude's
lane beside the conversation (its conversation, the differences per step with "DCLab is right" and "Claude is right",
its reading), a message can go to both, to DCLab only or to Claude only, and the understanding panel lets the person
correct any field in their own words. A pick is the step's truth (`person_pick`).
