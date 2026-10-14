# Twin runs: the current system and an AI-first Claude agent, side by side

Packages A6.6 and A6.7 (`dclab_rnd/twin`), R&D only. One problem, two lanes, at the same time, everything from both kept in
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
| `ai_first/design/` | how Claude designed the solution and its workflow, and what the agent then did with it (A6.7, below) |

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

## How Claude designs the solution (A6.7)

On a project seed the AI-first lane is watched as well as run. Before the agent starts, Claude is asked for a solution
design (purpose `twin_designer`): its understanding of the problem (task, target, prediction moment, unit, success),
the data checks it plans, its leakage hypotheses (each with why, the test that settles it and an action), the
identifiers, the split, the feature direction, the families, the evaluation, and the **workflow**: steps `S1, S2, …`,
each on a WF-01…WF-10 step, with the tool it calls, the stage, what it does, why it comes there, the evidence it needs
and the rule it will choose by, plus routes back ("from S9 to S4 when a leak is confirmed"). The design goes to the
agent with its task, and the agent is asked to name the design step before each tool call and to say when it departs.

A monitor sits on the agent's client. Before each request it reads the calls that finished and writes them to the
timeline: what the agent said, the step it named, the design step the call carries out (same tool, same stage),
and whether it was in order, out of order or not in the design. After each checkpoint stage is approved (`data`,
`leakage`, `models` by default; `--design-checkpoints`) Claude revises the design from that stage's compact record and
lists every change with the number that made it; the revision is handed to the agent as a message in its
conversation (`--no-steer` keeps it out, so the design is only watched). When the run ends, the last calls are read
from the session and Claude writes a final revision with a retrospective (what worked, what changed, what it would do
differently).

| `ai_first/design/` | What it holds |
|---|---|
| `DESIGN.md` | the story: every version, what changed and why, the timeline, the plan against the act |
| `design_vNN.json`, `design_vNN.md` | each version as Claude wrote it, and as a page |
| `designs.jsonl` | one row per design request: checkpoint, version, parsed or not, the problems found (a missing key, a tool that does not exist, a route to a step that is not there), Claude's raw answer |
| `timeline.jsonl` | the monitoring log: one row per tool call and per design event (`design`, `design_handed_to_agent`, `design_failed`) |
| `adherence.json` | the planned workflow against the calls: steps done in order, out of order, skipped; calls not in the design (and which of them write) |
| `intents.jsonl` | the design's decisions keyed like the decision samples (`contract.target_column:-`, `column.leakage:<col>`, `split.strategy:-`, `model.families:-`, …), per version |
| `sft_candidates.jsonl` | each parsed design as a chat sample (system, request, design) with its outcome; Claude's words, `verified: false`: candidates for the sample factory, not samples |
| `summary.json` | what `COMPARE.md` reads |

`COMPARE.md` gets a section **How Claude designed it**: the versions, the first workflow, the plan against the act,
the current system's own plan (the workflow lane's standard plan), and one table per decision point: Claude's first
design, its last, the rule's answer (the workflow lane's decision samples), and what each lane finally did, with
`≠ rule` where Claude's first design disagreed with the rule.

What the designer sees: the project's name and goal, the table's profile without example values (always, whatever
`--capture-cells`), the known solution unless `--blind`, the workflow's steps, the tool names, its own design, and at
a checkpoint the calls made (results through the trace's scrub, never the agent's words) and that stage's compact
record (target values removed: its positive label, classes and range, and the data stage's sentences, which spell the target out). No revision before the end sees a holdout
score (the data stage's drift note, which the agent reads too, compares feature distributions with the holdout); the
final one sees the final stage's record, as the agent's report does, when nothing can be tuned any more. A checkpoint
revision never takes one of the run's last three requests (they are kept for the agent), and the time Claude spends on
designs is left out of the lane's seconds in `COMPARE.md`; it does count against the agent's `--max-minutes`, because a
revision runs between two of the agent's requests. Each design request counts against the run's caps; a refused or
failed one is a timeline row, never a failure of the run. Claude's hidden reasoning is not returned by the
OpenAI-compatible endpoint; what is kept is the reasoning Claude writes down (each step's why, evidence and rule,
each change's because and evidence, what the agent says before each call). The design runs on project seeds; a draft
seed has the Home agent's own drawn workflow in the draft summary instead. `--no-design` turns it off.

## Running

```bash
python -m dclab_rnd.twin run --sample bank_marketing                 # the plan only; nothing is sent
python -m dclab_rnd.twin run --sample bank_marketing --yes --max-requests 80 --max-steps 30
python -m dclab_rnd.twin run --project <id> --blind --yes --cap-eur 2  # the euro cap applies when the model has a price (DCLAB_PRICES_FILE)
python -m dclab_rnd.twin run --draft <id> --yes
python -m dclab_rnd.twin run --sample bank_marketing --yes --design-checkpoints leakage --no-steer  # the design watched, one revision
python -m dclab_rnd.twin run --sample bank_marketing --yes --no-design                            # the AI-first lane from the task alone
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

The design's files feed two more: `intents.jsonl` sets Claude's planned decisions beside the rule's answers on the
same keys, and `sft_candidates.jsonl` holds each design with its outcome, for a builder that keeps the designs the
holdout and the traps vindicated.

Every lane writes decision samples (A6.5), so a twin run is also training data: the AI-first lane's approvals and
overrides are labels of source `agent` beside the rule's answer, with the evidence both saw (a judge must read
`labels[].source == "agent"`; the export's `label` field holds human labels only). With enough runs the plan's judge
(plan 3 in the model roadmap) has what it needs: where the agent and the rule disagree, and which one the holdout
favoured.
