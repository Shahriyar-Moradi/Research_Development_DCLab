# Harness against the model alone: the plan as built (campaign `harness_ablation_v1`, HAB)

**Recorded:** 2026-11-15. **Status:** the pilot ran live once (HAB-002, section 9). The free diagnosis (HAB-000) is stored.
The owner asked for the answer key to be rewritten to reward thinking and then for the run; the cases were frozen after
that rewrite. The lead agent's code in the run is commit `22ad1bf8`.

## 1. The question

Does DCLab's harness improve, keep or hold back the model it runs? Two documents of 2026-11-15 ("Harness Evaluation and
Model-Only Comparison", "Intelligence Recovery") and a reviewer's comment propose one explanation: DCLab binds a problem
to a fixed type before the model understands it, so the model can only plan inside that frame. They ask for a controlled
comparison first, and an architecture change only if traces show where the model's good ideas are lost.

This plan keeps that discipline. It corrects the two documents where the code differs from what they assumed.

## 2. What the code does (read from the working tree, not from GitHub)

| The documents assumed | What the code does | Where |
|---|---|---|
| The lead agent picks a problem family early | It never picks a family. It picks one of 6 kinds of work ("tracks") and 7 tools | `dclab_rnd/prompts/case_agent.md`, `dclab_rnd/agents/case_agent.py` `_start` |
| An unknown problem falls back to `classification` | True in `dclab_core/problem_map.py` and `dclab_rnd/kinds` (`binary`), but Home reaches it only through the no-data blueprint. At build time the problem card refuses instead of defaulting | `dclab_rnd/draft/no_data.py`, `dclab_core/card.py` |
| The coordinator's gates reject model proposals | The coordinator is not on the Home turn path, and its gate refuses only on data grounds | `dclab_rnd/draft/coordinator.py` |
| A new runner is needed for model-only against DCLab | The conversation exam already runs the model alone (`plain`) against the lead agent (`case`) with a blind judge. The "19 of 19 against 15 of 19" result is its run CEX-017 | `dclab_rnd/agent_eval/conversation_exam/` |

The places where code does act on the model in the lead agent, each with its reason code:

| Reason code | What the code does |
|---|---|
| `start_refused_not_ready` | `_start` refuses work until the decision and the user are known |
| `propose_refused_not_ready` | `_propose` refuses options until the case may propose |
| `route_overruled` | `route_doubt`: the person's words (a pattern) or a second small model name another kind of work; the agent is told to read again |
| `options_dropped` | `check_options` drops an option (same kind as another, a column that is absent, a number with no source) |
| `guard_refused` | `CaseGuard`: no new options once work has started |
| `closed_list` | a kind of work or a tool outside the list: refused by the tool's schema, or a pair DCLab cannot run |
| `route_change_refused` | a new route after work started, without the person's words asking for it |
| `no_data_redirect` | no table: the agent is sent to the no-data blueprint |
| `other_refusal` | any other refusal of a move that decides what the person gets, kept with its words |
| `advice_refused` | advice refused when the words ask for data work |
| `route_set_by_code` | `_worked`: figures were shown, so code writes the route as the agent's |
| `chip_bound`, `needs_table` | the person's own pick on Home; a table is needed. These protect the person's choice or the data |
| `budget_exhausted` | a turn ended by its budget (14 steps, 8 replies or the time limit) |

When no code is seen to intervene, the record says what the conversation shows: `no_gate` (nothing to explain),
`no_route`, `model_error` or `picked_wrong_option` (only when the route is not the person's), `arm_error`.

Not readable from what a run keeps, so never claimed: a tool result cut at 4,000 characters, a reply rewritten by the
number check, evidence missing from the context.

## 3. What the kept runs already show (HAB-000, no model call)

`python -m dclab_rnd.agent_eval.harness_ablation --recorded-dev` reads every kept dev run of the conversation exam that
played both `plain` and `case` (never a sealed one): 11 runs, all on Claude Sonnet 5.5, between 2026-10-30 and 2026-11-02.

- The lead agent lost the right route to the plain model on **8** conversations (7 different people) and won it on
  **12** (8 different people). On dev, the harness is not simply worse. Dev was tuned on, and some people were replayed
  across runs, so this is a description, not a result.
- First divergence of the 8 losses:

| First reason | Pairs | Reading |
|---|---|---|
| `picked_wrong_option` | 3 | No refusal was recorded. The agent offered options and the simulated person picked a wrong one |
| `start_refused_not_ready` | 2 | **The model asked to start the right work** (`preparation/recipe`, both times). The readiness gate refused because the decision and the user were not known. The agent then offered options and another route was chosen |
| `model_error` | 1 | No refusal was recorded. The model started advice for a prediction request (no table was shared, so prediction could not start) |
| `arm_error` | 1 | The run broke |
| `work_done_no_route` | 1 | The figures and a dashboard were shown; no route was recorded. A scoring artefact, since fixed by `_worked` |

So two of eight losses carry the signature the documents describe: a valid proposal, refused by a gate that asks for the
product's shape (a decision and a user for a plain cleaning request), not by a safety rule. In four of eight no refusal
was recorded. This is the first trace-backed evidence, and it points at one gate, not at the architecture.

Limits:
- Only refused or changed tool calls are visible. "No refusal recorded" is not proof that no code shaped the conversation.
- Five of the eight lost conversations are from CEX-009, which kept no `case.changed` event: there a route set by code
  cannot be seen at all.
- Old code (the gates changed since), a different model from today's (Qwen), a simulated person, and an answer key that
  is itself one of the six tracks.

## 4. The pilot

**Arms.** One model for all of them (the product's: `qwen/qwen3.8-flash` on OpenRouter, the owner's choice).

| Arm | What it is | What a comparison with it shows |
|---|---|---|
| `model_only` (A) | One request, no tools, the table's column view | Understanding and plan only. Never compared on work done |
| `open` (B) | One loop, read tools, neutral prompt, no list of kinds of work | The minimal agent |
| `plain` | The exam's arm: the same loop, told the six kinds of work | `plain` against `open` = the closed list alone |
| `case` (C) | The lead agent as the product runs it, unchanged | `case` against `plain` = the case file, gates and budgets |

The exam's `plain` arm is told the same six tracks as the lead agent. It is therefore not a "no forced type" control,
which is why `open` was added.

**Cases.** Eight written cases in `cases_dev.json`; their answer key apart in `answer_key_dev.json`, never sent to an
arm. Four have a right answer that is not a model; two lie outside the six tracks (an experiment's design, a routing
optimisation); one is a plain prediction request, a control the harness should win or tie. Tables are the exam's three
invented ones. A live run is refused until `--freeze` pins the reviewed cases.

**The person is a script.** The opening holds everything a cooperative person would say. When an arm shows option
cards, the script clicks the first one whose kind of work the key accepts, as the product's person would. Otherwise an
arm that asks is told "choose what you judge best and go ahead". At most four assistant turns, so the lead agent's own
turn budget can be reached. This removes the simulated person as a source of noise (three of the eight recorded losses
came from its picks). Each conversation records `picked` and `options_unpicked`, so a refusal the script caused can be
told from one the gate caused.

**Judge.** Claude Opus on Anthropic's own endpoint, never the arms' model (a test checks the model of the kept judge
exchange). It reads the conversation and the key, never the arm's name, and scores four dimensions from 0 to 4:
understanding, solution design, planning, evidence discipline. A fifth, adaptation, is scored only where a case has an
adaptation check; **none of the eight cases has one yet, so adaptation is not measured in this pilot.** Each dimension
is reported by itself. Numbers are left to code: it counts the numbers nobody gave and no tool computed, and the rows
listed. The judge's answer is shape-checked and never repaired.

**Trace.** For every `case` conversation, `first_divergence.py` reads the events the run kept beside it and records the
first intervention of code, what the model wanted there, and whether that was right.

**Records.** Every call goes through the gateway with the run's own folder,
`agent_runs/live_tests/<date>-harness-ablation-HAB-NNN/`: `manifest.json` (commit, models, prompt and tool-schema hashes,
flags, the cases' hash), each conversation with its events and decisions, `scorecard.csv`, the model exchanges with their
API thinking summaries. After the run: `make archive`, then `--receipt <run> --snapshot <folder>` writes
`archive_receipt.json` and says "archived" only when every file is in the snapshot with the same hash.

**Before anything is sent** the runner refuses when: the cases are not frozen; the lead agent's code has uncommitted
edits; the owner switched live calls off or the workspace's month would pass its cap; no model serves an arm; no
Anthropic key serves the judge. A conversation the cap stops is kept on disk and left out of the scores. The run's spend
is in the run's own usage log and in its result, not in the workspace's monthly log.

**Cost.** The estimate is about EUR 2.7 for each case once in all four arms (most of it the judge), a guess until the
first run measures it. The default is one repeat; `--repeats 2` costs about twice.

## 5. Known weak points

1. Eight cases prove nothing statistically. The product is traces and reason codes. No arm is called better.
2. The cases were written by the same author as the runner, who knows the harness. The owner's review is the check. A
   30-case study needs an independent writer.
3. The judge cannot be fully blind: option cards and drawn charts show which kind of system spoke. On CEX-013 the Opus
   judge agreed with the owner at 0.66 (bar 0.70). Judge scores count only after the owner labels ten conversations
   blind and agreement reaches 0.70.
4. The `model_only` and `open` prompts ask for the problem, the approaches and a plan. The lead agent's prompt asks for
   a conversation. A judge that scores plans will favour the first two on the first turn. That is partly the question
   (what does the person receive), and it must be read with the first-divergence record, not alone.
5. The arms run on Qwen flash. A weaker model may need a harness more. A result here may not hold for a strong model.
6. Tables are limited to the exam's three; there is no multi-table case yet.
7. Most openings state two or three of the key's "must understand" points almost in the key's words, the planted trap
   included. The share of points understood will therefore reward restating the opening. Read the design and planning
   scores and the forbidden actions, not that share. This is the first thing to fix in the owner's review of the cases.
8. No case has an adaptation check (a planted contradiction or a failed tool), so recovery is not tested.

## 6. Reading the result

| What the pilot shows | Next step |
|---|---|
| `open` good, `case` poor, first reason `closed_list`, `route_set_by_code` or `route_overruled` | Build the late-binding arm D behind a flag |
| First reason `start_refused_not_ready`, `propose_refused_not_ready`, `options_dropped`, `guard_refused` | Fix that gate: tell safety from shape, make the refusal recoverable. Keep safety gates |
| `plain` poor against `open` | The closed list itself costs quality: test a prompt without it before any gate change |
| `budget_exhausted` | Budgets, not architecture |
| All arms poor | Model or evidence limit: repeat on Claude Sonnet 5.5 before any other change |
| `case` matches or beats `open` | Keep the harness; the hypothesis is not supported on this model |

No architecture change, no new AI feature and no fine-tuning before this table is filled from traces.

## 7. Commands

```bash
make harness-ablation                               # the plan and the estimate; nothing is sent
make harness-ablation ARGS=--recorded-dev           # the free diagnosis (already stored as HAB-000)
make harness-ablation ARGS=--freeze                 # after the owner's review of the cases
make harness-ablation ARGS="--yes --cap-eur 3"      # the pilot, each case once
make archive
python -m dclab_rnd.agent_eval.harness_ablation --receipt agent_runs/live_tests/<run folder> --snapshot <the snapshot folder>
```

## 8. Files

- `dclab_rnd/agent_eval/harness_ablation/`: `__init__.py` (runner), `arms.py` (`model_only`, `open`), `judge.py`,
  `first_divergence.py`, `cases_dev.json`, `answer_key_dev.json`
- `dclab_rnd/prompts/ablation_model_only.md`, `ablation_minimal_agent.md`, `ablation_judge.md`
- `tests/test_harness_ablation.py`
- `evidence/campaigns/harness_ablation_v1/results/HAB-000_recorded_dev.json`

## 9. The first live run: HAB-002 (2026-11-15)

Eight cases, four arms, each once: 32 conversations on `qwen/qwen3.8-flash`. Arms EUR 0.19; judge EUR 0.14
(`qwen/qwen3.8-27b` on OpenRouter, the owner's choice, because the Anthropic account had no credit). Every one of the 215
model calls of the arms kept its thinking. Result: `evidence/campaigns/harness_ablation_v1/results/HAB-002_dev.json`.

**Scores (mean, 0 to 4).**

| Arm | Understanding | Solution design | Planning | Evidence discipline | Accepted |
|---|---|---|---|---|---|
| `model_only` | 2.9 | 2.6 | 2.4 | 3.4 | 5 of 8 |
| `open` | 2.6 | 2.4 | 2.6 | 3.0 | 6 of 8 |
| `plain` | 2.8 | 2.5 | 2.1 | 2.6 | 6 of 8 |
| `case` (current DCLab) | 2.6 | 2.5 | 2.1 | 3.4 | 5 of 8 |

No arm is better. Every paired difference has an interval that crosses zero.

**Where the lead agent's code stepped in (the main refusals, 8 conversations).**

| Reason | Times | Example |
|---|---|---|
| `query_cannot_express` | 11 | a sum over one quarter is refused (a filter is allowed on a count or a share, not on a sum); the first and last order date cannot be asked (dates are text) |
| `query_cap` | 10 | after 4 questions of the table in a turn, or 3 refusals, the agent must stop and write |
| `other_refusal` | 7 | a build without the person's yes (4, a safety rule); a group by a price column; malformed arguments |
| `route_overruled` | 3 | hab-07: the second reading named advice; advice was then refused as data work; analysis was refused for lack of a table; no route was set |
| `options_dropped` | 3 | hab-02: the option "Randomised pilot brief", the key's right answer, was removed for a number nobody gave (23%) |
| `budget_exhausted` | 3 | a turn ended by its step or reply limit |
| `start_refused_not_ready` | 1 | a start of prediction on hab-01 (not the right work there) |

**Reading.**
- No evidence here that binding to a kind of work held the model back. The lead agent reached a sensible kind of work on
  7 of 8 cases, including the two outside the six kinds.
- The brakes that the traces show are the table tool (what it cannot express, and its per-turn caps), the second reading
  of the route working against the other gates, and the number check removing a correct option.
- The lead agent is slower: hab-01 took 4 turns and 8 minutes; the other arms answered in one turn.
- The number check cuts both ways: the lead agent stated 0 numbers nobody gave, `plain` stated 52; many of those 52 are
  plan suggestions ("read 200 tickets"), not false facts.
- By the table in section 6, the next step is a focused repair of the query tool and its caps, not late binding.

**What weakens this run.**
- 5 of the 24 baseline conversations returned an empty or cut answer (the runner's limit of five model calls a turn and
  its token limit): `plain` hab-01 and hab-04, `open` hab-05, `model_only` hab-04 and hab-05. They were scored as given,
  which lowers the baselines. A rerun should raise those limits.
- The judge is from the same family as the arms.
- Eight cases, one repeat, cases written by the runner's author.
- Twice the lead agent "drew a dashboard" for a person who shared no table (hab-06, hab-07): to be looked at.

