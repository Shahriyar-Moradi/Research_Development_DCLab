# Twin runs: the current system and an AI-first Claude agent, side by side

Package A6.6 (`dclab_rnd/twin`), R&D only. One problem, two lanes, at the same time, everything from both kept in
one folder, and a comparison. The question it answers: on the same table, with the same tools and the same
validator, what does an AI-first agent (Claude) decide that the current system (rules, validator, standard plan)
does not, and what does each cost?

## What runs

| Lane | Who decides | Models |
|---|---|---|
| `workflow` | the current system: the intern's standard plan (describe → propose → set the solution → the five stages → report), every move through the graph validator | none by default; `--workflow-models configured` gives it the workspace's configured model, as the product has it |
| `ai_first` | Claude as the agent: the same task, the same tools (describe, understand, propose, set the solution, run and approve each stage, get results, finish), the same validator; it chooses the order and the options | Claude for everything: the agent, the stage notes, the leakage reviewer and the seven roles of dclab_core |

Claude is reached through Anthropic's OpenAI-compatible endpoint (`https://api.anthropic.com/v1/`) with the
studio's own client, so nothing in the gateway changes. `ANTHROPIC_API_KEY` is the key; `DCLAB_TWIN_MODEL` the
model (default `claude-sonnet-5-5`).

## Seeds

- `--sample <key>`: a studied sample (`python -m dclab_rnd.twin samples`); `--blind` strips the R&D's known solution
  from both lanes, so each must find the leak itself.
- `--project <id>`: a project of the workspace (`--source-home` for another): its table and known solution are copied
  into each lane, never its stages or approvals.
- `--draft <id>`: a draft's chat: the person's turns are replayed in each lane, the standard questions against Claude.

The sealed benchmark is never a seed.

## Where everything is

`<workspace>/twin/<run_id>/`:

| Path | What it holds |
|---|---|
| `run.json` | the manifest: seed, task, model, caps, each lane's status, start and end, model usage, the differences |
| `compare.json`, `COMPARE.md` | the lanes side by side: solution fields, stage choices, holdout, moves, tool calls, usage, time, the final reports |
| `README.md` | this table, for the folder |
| `<lane>/projects/<id>/` | the project: `project.json`, `stages/<stage>.json` (full records), `transitions.jsonl` (every move and verdict), `activity.jsonl` |
| `<lane>/intern/<session>.json` | the agent's whole conversation: system prompt, task, every model reply, every tool result, the final report |
| `<lane>/agent_steps.jsonl` | one row per tool call |
| `<lane>/audit.jsonl` | the audit trail |
| `<lane>/model_requests.jsonl` | one usage row per model request |
| `<lane>/transcripts/model_calls.jsonl` | every request and answer, verbatim: messages, tools, the reply, tokens, seconds, cost; refusals and provider errors too |
| `<lane>/decision_samples.jsonl`, `decision_labels.jsonl` | the typed decisions with their evidence (A6.5); the agent's approvals and overrides as labels of source `agent`, kept apart from a person's |
| `<lane>/summary.json` | what the comparison read from the lane |

Each lane is a complete workspace folder; the main workspace is never written. A twin run's files hold prompts,
model answers, column names and the intern session's tool results (which carry three example values per column, as
the product's own sessions do); the transcript's tool results go through the trace's scrub (the keys that carry cells
removed) unless `--capture-cells`. The holdout score is in the files when the agent asked for the final results;
`summary.json` says how many times the holdout was scored. Research material, kept on the machine that made it.

Limits: a twin run keeps files only (`DCLAB_DATABASE_URL` must be unset for the run); a project marked private is
refused unless `--allow-private`; the euro cap applies only when the model has a price, and a cap is checked before a
request, so the last request can pass it by its own cost; `--roles all` (the default) turns every dclab_core role on
for the run, which reaches only the AI-first lane because the workflow lane has no model (with
`--workflow-models configured` the roles are on there too).

## Running

```bash
python -m dclab_rnd.twin run --sample bank_marketing                 # the plan only; nothing is sent
python -m dclab_rnd.twin run --sample bank_marketing --yes --max-requests 80 --max-steps 30
python -m dclab_rnd.twin run --project <id> --blind --yes --cap-eur 2  # the euro cap applies when the model has a price (DCLAB_PRICES_FILE)
python -m dclab_rnd.twin run --draft <id> --yes
python -m dclab_rnd.twin list
python -m dclab_rnd.twin compare <run_dir>                            # rebuild the comparison from the files
```

Caps: a number of requests (always), euros (when priced), the intern's tool calls and minutes. Without `--yes`
nothing is sent; under `DCLAB_NO_LIVE_MODELS=1` nothing can be.

## Reading a comparison

`COMPARE.md` says where the lanes differ: a solution field (target, forbidden columns, identifiers, split), a stage
choice (recipe, family, optimization), the holdout score (one holdout each, scored once; a difference inside the
intervals is noise), the moves the validator refused, the tool calls and their order, the requests and euros, the
seconds. It never says which lane was right: the planted traps and the sealed holdout do that, and the files let a
reader see why each lane chose what it chose.

## What it feeds

Every lane writes decision samples (A6.5), so a twin run is also training data: the AI-first lane's approvals and
overrides are labels of source `agent` beside the rule's answer, with the evidence both saw (a judge must read
`labels[].source == "agent"`; the export's `label` field holds human labels only). With enough runs the plan's judge
(plan 3 in the model roadmap) has what it needs: where the agent and the rule disagree, and which one the holdout
favoured.
