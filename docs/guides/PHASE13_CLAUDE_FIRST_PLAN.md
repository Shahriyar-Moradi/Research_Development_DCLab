# Phase 13: Claude-first DCLab agents

Oct 29, 2026 · @Shahriar

## Why

DCLab runs on Claude but behaves like a much weaker model, because the structure around Claude makes it worse. On Home's second sealed set (HOME-013), the rules read the kind of help right on 21 of 36 requests, Home with Claude on 31, and Claude alone, with nothing of DCLab around it, on 36 of 36. The owner, an AI engineer, judges the understanding, the routing and the help as far below what Claude can do.

The box is visible in today's prompts and code:

- **Told not to think.** `home_agent.md` says "call decide\_point instead of reasoning it out", "Reply in plain English, 1-3 sentences" and "at most 4 short questions".
- **Understanding as a form.** `home_intent.md` asks for one JSON object with fixed fields; every field must rest on an exact quote, and a request must fit one of six fixed tracks.
- **Order set by code.** The coordinator (21.2) is a fixed state machine: problem, data, links, analysis, workflow, build, whatever the problem is.
- **Buttons first, questions capped.** An unclear request gets the six kinds of help as buttons on the first reply (`UNDERSTAND_TURNS = 2`, `MAX_QUESTIONS = 5`), and the rules can overwrite the model's reading.
- **No call sees the whole case.** The problem card, data card, memory, evidence, ledger and past cases are stored, but each call gets a slice.

These limits came from real risks (leakage, invented numbers, wrong builds). The mistake was putting the guards in the middle of the thinking instead of at its edges. Phase 13 moves them.

## What we decided (29 and 30 October 2026)

The owner and Claude settled these in conversation (29 and 30 October); each answer is a rule for every package below.

| Question | The owner's answer | The decision it sets |
| --- | --- | --- |
| What is wrong today? | The level of understanding, intelligence, routing and help is low; DCLab uses Claude but acts "like a child of Claude", trapped in a deterministic way, never reasoning about the sentence. | Claude reasons first, with the whole case in view; rules become a fallback, not the first reader. |
| How should Home talk? | Understand before asking many questions; gather context; at least about three natural exchanges, like Claude; options only after that, better and based on context. | Options come after understanding, written for this person; the rule is "enough understanding", with about three exchanges as the normal minimum. |
| Bigger agent: one strong agent, more agents, or both? Only understanding, or all the way? | Both: more thinking to help, guide and route. | One lead agent holds the case from the first sentence to the build, and sends specialists when they help. |
| What must stay with code? | Most of the internal process: the holdout, splits, numbers, building. | A guard layer in code owns these; the agent reaches them only through tools that validate. |
| What may it cost? | Working well matters more than budget for now; Claude to advise. | Sonnet 5.5 with thinking; a soft cap of about $5 per case that stops and asks; every case's real cost recorded. |
| Implement now? | No: design first, ready to implement. | Design first, ready to implement; nothing is built until the owner approves it. |
| Persian in the exam? | Skip it for now. | The exam is English only; Persian returns later. |
| What should be captured for training? | All data, as much as possible: conversations, actions, decisions, context, for every model type; the thinking and reasoning too, within Claude's and Anthropic's limits; in every test; and the code, to prevent forgetting. | A recorder in code writes every event; an outcome follower and a curator complete it; summarized thinking on every call, never reasoning asked in the reply; every test kept and marked; sealed exams kept whole in an eval-only vault; code from every source kept and replayed against forgetting (packages 37.12 to 37.18). |
| Keep sealed exams whole? | Yes, accepted. | Sealed exams are kept whole in an eval-only vault that no training set can read. |
| Capture code too? | Yes, of course: to prevent forgetting. | Code from every source is kept with its run verdict and replayed in every training round; a frozen code exam guards each round (37.18). |

## Principles

1. **Claude thinks first.** Understanding, conversation, reasoning about data, routing, solution design and explanation are the model's, with adaptive thinking on and the whole case in its context.
2. **Code guards the internal process, at the edges.** Splits, the holdout read once, metrics, every number shown, building, deleting, spend and privacy stay deterministic. A guard checks what the agent proposes; it never rewrites the agent's thinking.
3. **Understand before options.** Listen, say back what was understood, ask about the person's situation; options only once the case file is rich enough.
4. **Enough understanding, not a fixed count.** About three exchanges is the normal minimum; a person who arrives clear is not made to answer questions they already answered.
5. **One case file.** Every agent reads and writes the same living record of the case; nothing important lives only in one prompt.
6. **Tracks are labels, not boxes.** The six kinds of help stay as words the agent may use; any problem may be taken, combined or split.
7. **Measure against Claude alone.** If DCLab with Claude is not clearly better than Claude alone on the same people, the structure is in the way.
8. **Everything is training data.** Every case file, turn, summarized thought, tool call, guard verdict and outcome is recorded by code, in every test as well as in real use, and sorted for the model types it can teach; reasoning is read from Claude's summarized thinking, never asked for in the reply.
9. **The person always has the last word.** Nothing is built, removed or sent without their yes; they can answer in their own words at any point.

## The architecture

One lead agent holds each case from the first sentence to the build, around a shared case file, with specialists it can send and a guard layer in code between it and everything that must be exact.

&#91;embedded content: Phase 13 architecture · lead agent, case file, guards\]

Everything inside the dashed harness is Claude's to reason about; everything below the guard layer is code's, reached only through tools that check. The recorder inside the harness writes every event to the store.

| Part | What it does | Built on (today's code) |
| --- | --- | --- |
| **Lead agent** (`case_agent`) | Talks with the person, keeps the case file, decides what to find out next, calls tools, sends specialists, proposes the route and the solution. One long session per case, adaptive thinking on. | `agents.run()` and `runtime.run` (21.1); replaces the Home agent's fixed flow and the coordinator's order |
| **Case file** (`dclab_core.case`) | The living record: the person's words, goal, the decision it serves, who it is for, the data and what it cannot answer, constraints, assumptions with their status, unknowns ranked by what they block, options offered and chosen, the route and its reasons. Versioned; every change says who made it and why. | Absorbs the problem card (20.1), track briefs, problem map (24.1) as views |
| **Context builder** | Assembles what the lead agent sees each turn: the case file whole, the data card, memory items for this person and project (35.x), similar past cases, evidence and lessons, recent turns; compacts older turns into the case file. | `memory` (phase 11), `evidence_index`, `lessons.py`, the data card (19.11) |
| **Tools** | Read and profile data, query it (`dclab_core.query`), search evidence and past cases, recall and propose memory, ask the person, update the case file, propose a step, propose the route, request a build. | `default_registry()`, draft and project scopes |
| **Specialists** | Sub-agents with their own context, sent by the lead agent when a question needs depth: data investigator, domain expert, leakage auditor, solution designer, critic. Each returns findings into the case file, never to the person directly. | Phase 6 team nodes, map roles (24.2), leakage review |
| **Guard layer** | Deterministic code between the agents and the internal process: the workflow graph's validator, splits, the holdout read once, metrics, numbers traced to data (`cited.py`), consent and redaction, spend caps. A refusal comes back to the agent with its reason, so it can reason again. | `studio/graph.py`, `critic_gate.py`, `cited.py`, `limits.py`, the engine |
| **Harness** | Runs long sessions: checkpoints after every turn, pause and resume, a budget per case with a soft cap, context compaction, full traces with thinking; every event written by the recorder (37.12). | `agents/harness.py`, jobs, the gateway, the experience log |
| **Rules** | No longer the first reader. They serve as a fallback when no model is available (the open edition), as a shadow answer recorded beside the agent's, and as part of guards where the rule is exact. | `intents`, `conversation.py`, the coordinator |

## The conversation

The lead agent moves through four stages, judged by the case file's state, not by a turn count; options appear only in the fourth.

1. **Listen.** Read the first sentence with thinking: what the person seems to want, the decision behind it, what they may not have said. Reply in natural language, no buttons.
2. **Reflect.** Say back the understanding in the person's terms, with what is still unclear, so they can correct it early.
3. **Explore.** Ask the one question that unblocks the most, about their situation rather than an ML field. Read the table when there is one, so questions are informed by the data ("your table has no cancellation date; how do you know a customer left?").
4. **Propose.** When the case file is ready, offer two or three options written for this person, plus their own words.

**When the case file is ready to propose.** The goal and the decision it serves are stated; who uses the result is known; the data is known or its absence is; no unknown blocks every option. Usually after about three exchanges; sooner only when the person's first message already settles these.

**How an option is written.** Each one says what DCLab would do, what the person gets, the trade-off and its rough effort, in plain words. Options differ in kind (an analysis now, a prediction later; groups found, or groups they already have), never three wordings of one plan. Code checks every option before it is shown: columns named exist, numbers come from the person or the data, and the route is one DCLab can run.

**Questions.** One at a time, open by default; buttons only for a closed choice (yes or no, which of two tables). No fixed limit: the guard is that a question must reduce an unknown in the case file, and the same field is never asked twice.

**Language.** The reply follows the person's language, including Persian, and the person's level: plain business words for a business owner, technical terms for a data scientist.

## What stays deterministic

The owner's rule: the internal process stays with code, however capable the agent becomes. The agent proposes through a tool; code checks, runs, and answers with a result or a reason.

| Owned by code | How the agent reaches it | What a refusal returns |
| --- | --- | --- |
| Splits and the holdout read once | Proposes a split in the solution design; the engine makes it | The rule broken ("a time column exists; a random split would leak") |
| Metrics and selection | Names the business goal; code maps it to a metric and selects | The metric chosen and why |
| Every number shown to the person | Asks a query tool; code computes and writes the numbers | Refused numbers it did not compute (`cited.py`) |
| Leakage audit before any score | Can raise a suspicion; the audit decides | The audit's finding with its evidence |
| Building, removing rows, deleting | Requests it; the person's yes and the graph validator allow it | The open question that blocks it |
| Workflow moves | Proposes a move; the validator checks it (`studio/graph.py`) | The move's rule |
| Spend, consent, privacy | Sees its remaining budget; code stops at the cap | The cap reached; the agent asks the person |

Nothing in this table is new code; Phase 13 changes who calls it and how a refusal is read, not the rules themselves.

## What changes in today's code

The new path is built beside the old one behind a switch, and the old path is retired only after the new one passes its exam.

| Today | In Phase 13 | Fate |
| --- | --- | --- |
| `home_intent.md`: one JSON object, six tracks, quotes on every field | The lead agent's reading in the case file, with reasons; tracks as labels | Replaced; kept as the shadow answer and for the open edition |
| `home_agent.md`: 1–3 sentences, at most 4 questions, `decide_point` instead of reasoning | A new prompt for the lead agent: think, reflect, explore, propose | Replaced |
| `UNDERSTAND_TURNS`, `MAX_QUESTIONS`, buttons on the first reply | The readiness rule of the case file | Retired |
| Coordinator state machine (21.2) | Guards keep its hand-off checks (never build with a blocker open); the order becomes the agent's | Its gates kept, its order retired |
| Problem card, track briefs, problem map | Views of the case file | Kept as views; their contracts stay for the main product |
| Rules-first readers (intents, conversation) | Fallback with no model; shadow answers beside the agent | Kept |
| Team graph (phase 6), map roles (24.x) | Specialists the lead agent sends | Reused |
| Sealed sets that count questions and fields | The conversation exam (next section) | Old sets kept for history; new gates |
| Ledger points for routing and intent | Recorded from the agent's proposals, labelled by the person's choice | Kept |

## Measurement

Phase 13 is judged on whole conversations against Claude alone, with the gate written before any run, as every DCLab gate is.

**The people.** Written people, each with a hidden truth: their real goal, the decision behind it, their data, what they know and their level (business owner, analyst, data scientist), and how they answer (vague, detailed, impatient); English only for now. A simulated person, played by a separate model, answers only from that truth. Dev: about 40 people; sealed: about 40, written by an independent agent, pinned by hash, run once.

**Three arms on the same people.** Today's Home; Claude alone with a plain prompt and the same data tools; the Phase 13 agent.

**The rubric** (a judge model, blind to the arm, plus the owner's labels on a sample):

- understood the real goal and the decision behind it;
- each question was useful, and none was asked whose answer was already given;
- options fit this person and differ in kind;
- the route chosen is right for the hidden truth;
- safety: no invented number, no row exposed, no build without a yes, no guard refused and ignored.

**Proposed gate** (for the owner to fix before the first run): the Phase 13 agent beats Claude alone on understanding and on the route; no safety failure; it is not worse than today's Home on any trap. Turns and cost are reported, not gated.

## Capturing everything for training

Every case, test and run is recorded whole by code, sorted for the model types it can teach, and kept with Claude's summarized thinking, within Anthropic's rules. The owner's words (30 October): "we need all data as much as we can", from conversations, actions, decisions and context, for every model we train or fine-tune.

**Who captures.** Recording is code, so nothing depends on a model remembering to log; a model only sorts and labels afterwards.

| Helper | Kind | What it does |
| --- | --- | --- |
| Recorder | Code, inside the harness | Writes every event as it happens, under one case id and one trace id: each turn as sent and received, the thinking, every tool call and result, every guard verdict and its reason, every case-file change with who and why, every option shown and the one chosen, costs and timings |
| Outcome follower | Code, on a schedule | Links what happens later to the decision that caused it: a build's holdout score, a call list's results, an answer the person edited or threw away, a case abandoned |
| Curator | An agent (Claude, capped), after a case ends | Sorts the record into the model types' sets, writes labels with their source (person, test, measurement, or Claude as silver), lists gaps; never edits the record |

**What is captured, and which models it feeds.** The model types are the store's (`dclab_core.store.MODEL_TYPES`): m1 students, m2 workflow decisions, m3 ML decisions, m4 workflow LLM, m5 solver LLM, m6 stepwise decisions, m7 judges, m7 routing, m8 retrieval.

| Source | Captured | Feeds |
| --- | --- | --- |
| Conversation turns | The person's words, the agent's reply, summarized thinking, stage, language | m4, m7 routing, m7 judges (corrections) |
| Case file versions | Every field change, who made it and why | m4, m1 |
| Routing and options | The reading, the options offered, the one chosen or the person's own words instead | m7 routing, m2 |
| Tool calls | Arguments, result, refusal and its reason, the retry | m5, m6 |
| Guard verdicts | The rule, what was refused or allowed, what the agent did next | m7 judges, m3, m1 |
| Technical choices | Ledger records of columns, leakage, split and metric, with every answerer | m3, m1 |
| Specialists | The brief, the findings, whether the lead agent used them | m4, m7 judges |
| Builds and runs | Pipeline, code, scores, the holdout read once | m5, m3, m6 |
| Outcomes | Later results linked to their decisions | m7 judges (rewards), m6 |
| Retrieval | The query, what came back, what was used | m8 |
| Costs and timings | Tokens, billed thinking tokens, euros, seconds | Reports; budget choices later |

**Every test, captured.**

- **Unit tests with scripted models:** kept as today (`training_records/test_runs`), marked synthetic.
- **Dev evaluations, live tests, twin runs, self-play, journeys, the exam's dev runs:** kept whole, thinking included, marked by split.
- **Simulated people and the judge:** their calls kept too, marked as simulation and judge, never mixed with real people.
- **Sealed exams:** today only their shape is kept, so an exam never becomes training data. Decided (the owner, 30 October): they are kept whole in an eval-only vault that no training set can read.

**Code, kept against forgetting.** A model fine-tuned only on new conversations loses what it knew about writing code, so code is captured from everywhere and mixed into every training round (the owner, 30 October).

| Code source | Kept with it | Feeds |
| --- | --- | --- |
| Pipelines exported as code (`studio/code.py`) | The problem, and whether a clean Python reproduced the recorded number | m5, m4 |
| Notebook cells run in a project | The cell, its output's shape, errors, whether the person kept it | m5, m4, m7 judges |
| Code the agent writes or proposes | The request, the code, the guard's verdict, the run's result | m5, m7 judges |
| DCLab Code changes | The step, the person's allow or deny, kept or undone later | m5, m7 judges, m2 |
| This repository's commits (`store/keepers/commits.py`) | The patch, its message, the failure it fixed | m4 |

- **Verified code only as a good example.** A row teaches as a good example only when its code re-ran to its recorded result (`store/sets/code.py`); code that failed is kept too, labelled failed, for the judges.
- **No invented reasoning.** Code the engine wrote carries no reasoning; only a model's own thinking summary is kept beside code it wrote (the owner's rule of 28 October).
- **A replay share in every round.** Each fine-tune round adds a share of every earlier round's rows and the verified code to its new rows (`store/replay.py`), so new skills don't push out old ones.
- **A frozen code exam.** Scored before and after every round; a drop beyond the margin the owner fixes stops the round's model from serving.

**Thinking, within Anthropic's limits.**

- **Ask for the summary explicitly.** On Sonnet 5.5 and Opus 5.5 the default is `display: "omitted"`; the gateway already sends `display: "summarized"` (`models/anthropic_client.py`), and every new purpose must too.
- **A summary, not the raw chain of thought.** Another model writes it; it is stored as `thinking_summary`, never as the model's own words. No setting returns the raw thinking.
- **Never ask for reasoning in the reply text.** Anthropic says to read the thinking blocks instead, and Sonnet 5.5 may refuse such a request (`stop_details.category: "reasoning_extraction"`). Every prompt is audited for "explain your reasoning" asks; a short reason meant for the person (why an option fits them) stays.
- **Blocks kept unmodified.** Thinking and `redacted_thinking` blocks are stored with their signatures and passed back unchanged inside a tool loop; a filter on `thinking` alone drops the redacted ones.
- **Billed for the full thinking.** The recorder keeps the billed thinking tokens beside the summary.
- **Training on Claude's outputs.** Anthropic's Commercial Terms let the customer own its outputs but bar using the Services "to train competing AI models" except as approved. Every Claude output carries its provenance, so each model type can include or exclude it; today's owner rule keeps Claude's outputs as silver, apart. Whether DCLab's own models count as competing is for the owner to settle (this is not legal advice).

**Consent and privacy stay as they are.** Workspace and project consent decide what may train and what may leave; cell values and secrets never enter a training set; text is redacted as today (`consent.py`).

**Where it goes, and how it is shaped.** Phase 13's data goes into the store that already exists, in its shape; no second store is built.

- **Place:** the owner's machine keeps dated, read-only snapshots in `training_records/snapshots/<time>/` (13 by 30 October; the latest 3.4 GB, each with a SHA-256 manifest). The store itself is the private `dclab-records` repository; this repository commits only the catalog and the index.
- **Structure:** sets per model type (m1 to m8), versioned, each row with its split, origin and use; `capture_rules_v2` covers 22 record kinds; duplicates, broken records and consent refusals are quarantined; sealed rows are counted, never read.
- **What Phase 13 adds:** `CaseEvent` as a record kind in the capture rules, the case id on every row, and sets for case files, options and guard verdicts, each placed under its model types.

**What the store shows today, and what Phase 13 must close.** From the store's catalog rebuilt on 30 October (commit `87da8884`):

| Gap | Today | Closed by |
| --- | --- | --- |
| Model-made examples with no reasoning | About 15,000 across m1, m2, m3 and m6 (the other 95% there are rule- or code-made, which carry no thinking) | 37.14: summarized thinking on every call. Past examples stay marked "no reasoning"; nothing is filled in afterwards |
| Routing readings without thinking | 1,278 of 1,434 in m7 routing, mostly from before 29 October | 37.14 |
| Solver examples with thinking | 77 of 1,917 in m5 | 37.14, 37.18 |
| One kind of help dominates | 91% of m1 comes from prediction; the advice track has nothing to judge yet; m8 retrieval has 9 examples | 37.13: coverage targets per track and per model type; 37.17 |
| Outcomes | About 5% of m1 and m3 examples say what happened afterwards | 37.16 |
| No agent campaign recorded | 0 runs | 37.15 |
| A stale rule | `training_records/README.md` says Claude's outputs are evaluation only; the owner's rule of 25 October lets them train as silver, apart | 37.14 |

Sources: [Building with extended thinking](https://platform.claude.com/docs/en/build-with-claude/thinking) · [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms)

## Budget and models

Sonnet 5.5 stays the product's model, with adaptive thinking, at about $2 to $5 for a whole case; a soft cap of $5 per case stops and asks the person.

- **Price:** $2 per million input tokens and $10 per million output tokens; thinking is billed as output; cached input costs far less ([LangCopilot](https://langcopilot.com/llm-pricing/anthropic/claude-sonnet-5-5); the same figures as `models/prices.py`).
- **Measured so far:** 36 Home dev cases at effort `xhigh` cost EUR 0.99 in total (HOME-024). Opus 5.5 scored the same at about 1.6 times the cost (HOME-023), so it stays a measured arm, not the default.

| Unit | Estimate (assumptions below) |
| --- | --- |
| One turn with thinking | $0.03 to $0.08 |
| Understanding one problem, about 8 to 12 turns | $0.30 to $0.80 |
| A whole case: understanding, data, specialists, design | $2 to $5 |
| 100 cases a month | $200 to $500 |

The estimate assumes about 20,000 tokens of context a turn, mostly cached, and 3,000 to 5,000 tokens of thinking and reply. Package 37.0 replaces it with measured costs; every case records its real spend.

## The packages

Nineteen packages, each built the house way: a failing test first, the change, the `dclab-reviewer` and its fixes, a development record, `make sft-dev`, `make check-quick`, a commit pushed. Contracts go in `dclab_core`, wiring in `dclab_rnd`.

| Id | Package | Delivers | Done when | Needs | Spend |
| --- | --- | --- | --- | --- | --- |
| 37.0 | The exam first | Written dev people with hidden truths, the simulated person, the judge and rubric; today's Home and Claude alone measured on dev | Both arms scored on dev; the gap measured; the gate written and approved by the owner | — | about $5 (two arms on dev) |
| 37.1 | The case file | `dclab_core.case`: contract, versions, who changed what and why, readiness rule, views onto the problem card and briefs; JSON Schema | Contract tests; the problem card rebuilt from a case file unchanged | — | none |
| 37.2 | The context builder | One function that assembles a turn's context (case file, data card, memory, past cases, evidence, recent turns) with a token budget and compaction | A replay of a long draft fits the budget and keeps every case-file fact | 37.1 | none |
| 37.3 | The lead agent | `case_agent`: a long session on `agents.run()`, adaptive thinking, the new prompt, the case-file tools; behind `DCLAB_CASE_AGENT` | Plays every dev person end to end with scripted transports in tests; one live dev pass | 37.1, 37.2 | about $3 |
| 37.4 | The understanding conversation | Listen, reflect, explore, propose; readiness from the case file; the person's language and level | Beats Claude alone on understanding on dev | 37.3 | about $5 |
| 37.5 | Context-built options | Two or three options from the case file, each checked by code before showing | Every option on dev passes the checks; none repeats another in different words | 37.4 | in 37.4 |
| 37.6 | Guards as tools | The guard layer's tools with refusals that carry reasons; the agent reasons again after a refusal | Every planted trap on dev refused, and the agent recovers without breaking a rule | 37.3 | none |
| 37.7 | Specialists | Data investigator, domain expert, leakage auditor, solution designer, critic as sub-agents writing into the case file | Each one sent on dev only when its trigger holds; findings traced in the case file | 37.3 | about $5 |
| 37.8 | The harness grows | Checkpoint and resume per turn, the per-case budget and soft cap, compaction, traces with thinking | Kill and resume mid-case loses nothing; a case at the cap stops and asks | 37.3 | none |
| 37.9 | Home, conversation first | The page shows the conversation and the case file as it fills; buttons only for closed choices; options as cards | `make browser-test` passes at 1280 and 375 px | 37.4, 37.5 | none |
| 37.10 | The sealed exam | 40 sealed people by an independent writer, pinned; the three arms run once | The owner's gate read once: pass or fail recorded | 37.4 to 37.9 | about $10 |
| 37.11 | Switch-on | The case agent serves Home; the old flow becomes the fallback; prompts and gates retired as listed above | 37.10 passed; the harness suite and journeys pass on the new path | 37.10 | none |
| 37.12 | The recorder | Every event of a case written by the harness under a case id and a trace id; one event contract in `dclab_core` (`CaseEvent`), feeding the experience log, the ledger and the store | A case rebuilds from its events alone; a kill mid-turn loses no event | 37.3 | none |
| 37.13 | The capture map | The sources × model types table in code; a coverage report per model type and per source, with the gaps | `make capture-coverage` maps every source and feeds every model type, or names why not | 37.12 | none |
| 37.14 | Thinking within Anthropic's rules | `display: "summarized"` on every purpose; a prompt audit for reasoning asked in the reply; redacted blocks kept; provenance on every Claude output; billed thinking tokens recorded | A test fails on a purpose without summarized thinking or a prompt that asks for reasoning in the reply | 37.12 | none |
| 37.15 | Every test captured | Dev, live, twin, self-play, journeys, simulated people and the judge kept whole and marked; sealed exams kept whole in the eval-only vault | Every run type shows in the coverage report; no sealed row in any train split | 37.12, 37.13 | none |
| 37.16 | The outcome follower | Late outcomes linked to the decisions that caused them, on a schedule | A call list's results land on the decisions that made it | 37.12 | none |
| 37.17 | The curator | An agent sorts finished cases into the model types' sets, labels with their source, never edits the record | Sets built from dev cases; every label traced to its source; spend within its cap | 37.13 | about $3 |
| 37.18 | Code kept against forgetting | Every code source captured with its problem and its run verdict; every fine-tune round built with the replay share and verified code; a frozen code exam scored before and after each round | A round's set holds code rows from every source; a tuned model's code-exam score falls no more than the margin the owner fixes before the first round | 37.12, 37.13 | none |

Spend figures are estimates for live model runs on dev or sealed people; each run asks for a cap first, under the owner's rule.

## Implementation notes

Where each package lands in the repository, so a build session can start from its row. New files are marked new; everything else extends what exists.

| Id | Files | Notes |
| --- | --- | --- |
| 37.0 | `dclab_rnd/agent_eval/conversation_exam/` (new): `people_dev.json`, the simulated person, the judge, the gate; prompts in `dclab_rnd/prompts/` | Three arms in one runner; dev only here; a sealed run is refused until 37.10 |
| 37.1 | `dclab_core/case.py` (new): `CaseFile`, `CaseChange`, readiness; `make core-schemas`; `CHANGELOG.md`; a `CORE_INTEGRATION.md` section | The problem card, track briefs and problem map read as views of it |
| 37.2 | `dclab_rnd/agents/context.py` (new) | Reads `memory/`, `evidence_index.py`, `lessons.py`, the data card; a token budget per turn |
| 37.3 | `dclab_rnd/agents/case_agent.py` (new), `dclab_rnd/prompts/case_agent.md` (new), tools in `agents/registry.py`, the switch in `settings.py` | Runs on `agents.run()` in `agents/harness.py`; Home's draft route chooses it when `DCLAB_CASE_AGENT` is on |
| 37.4–37.5 | `agents/case_agent.py`; option checks in `dclab_core/case.py` | Readiness and option checks are code; the conversation is the model's |
| 37.6 | Guard tools in `agents/registry.py`, wrapping `studio/graph.py`, `cited.py`, the engine, `limits.py` | A refusal returns its rule and reason as the tool result |
| 37.7 | Specialists from `orchestra/` (team nodes) and `core_advisor.py` (map roles), sent through the registry | Findings written into the case file only |
| 37.8 | `agents/harness.py`, `orchestra/checkpoints.py`, `jobs/` | Checkpoint per turn; the soft cap from the gateway's spend |
| 37.9 | `dclab_rnd/agentic/web/src/` pages; `make web`; `make browser-test` | The five screens on the Phase 13 Home designs canvas are the reference |
| 37.10–37.11 | `conversation_exam/` sealed set and its `.sha256`; `models/routing.py` switch | Old prompts `home_intent.md` and `home_agent.md` stay as the fallback |
| 37.12 | `CaseEvent` in `dclab_core/experience.py`; `dclab_core/store_rules.py` (capture rules v3); writers in `agents/harness.py` | Every event also reaches the experience log and the ledger as today |
| 37.13 | `dclab_rnd/store/coverage.py`, `store/catalog.py`; a `capture-coverage` target in the `Makefile` | Coverage per model type, per source and per track, with targets |
| 37.14 | `models/anthropic_client.py` (already sends `display: "summarized"`), `tests/test_thinking_capture.py`, a prompt audit test over `dclab_rnd/prompts/`, `store/provider_outputs.py`, `training_records/README.md` | The README's "evaluation only" line corrected to the owner's rule of 25 October |
| 37.15 | `experience.sealed_run` (keep whole into the vault instead of shape only), `store/exams.py`, `store/layout.py` | The vault is outside every set's train split; a test fails if a sealed row reaches one |
| 37.16 | `dclab_rnd/store/outcomes.py` (new) as a job in `jobs/`; labels through `ledger/labels.py` | Outcomes are measured labels, never guesses |
| 37.17 | `store/curators/`, `prompts/store_curator.md` | Extends the curator of 32.9; never edits a record |
| 37.18 | `store/sets/code.py`, `store/replay.py`, `store/keepers/commits.py`; a frozen code exam beside `store/exams.py` | Builds on 36.5 and 36.6 |

Every package also gets its development record (`dev_trajectories/records/`), `make sft-dev`, the `dclab-reviewer` and `make check-quick`; a package that changes `dclab_core` also runs `make core-wheel-check`.

## Order, effort, risks

The exam comes first, so every later package is measured against the gap it is meant to close; the sealed run comes last, once.

&#91;embedded content: build order · 37.0 to 37.12\]

The four foundation packages run in a line; the six after 37.3 can run side by side; 37.12 to 37.18, the capture, start right after 37.3 and run beside everything else, so every later dev run is recorded; they do not hold up the sealed exam.

**Effort.** Roughly 3 to 4 weeks of build sessions for 37.0 to 37.18, and about $35 of live runs across dev and the sealed exam. This is an estimate; 37.0 makes the first measurement.

**Risks and what we do about them.**

- **The agent invents or overreaches.** Every number and build still goes through code; a refusal returns its reason; 37.6 plants traps on dev.
- **Longer conversations annoy clear people.** The readiness rule lets a clear request go straight to options; the exam includes impatient people.
- **The judge flatters the agent.** The judge is blind to the arm, a different model from the simulated person, and checked against the owner's labels on a sample.
- **Cost grows unseen.** A per-case soft cap and every case's recorded spend; the monthly cap in `limits.py` still holds.
- **Two Home flows drift apart.** One switch, the old flow frozen as the fallback, retired in 37.11.
- **Overfitting to dev people.** The sealed people are written by an independent agent and run once, as every DCLab sealed set is.
- **The open edition has no model.** The rules remain its path, unchanged.

## Questions still open for the owner

One question shapes the agent's voice and has no answer yet; the rest have a proposal that holds unless the owner changes it.

| Question | Proposal until answered |
| --- | --- |
| Who usually types into DCLab: a business owner, an analyst, a data scientist? | All three; the agent reads the person's level from their words and adapts; the exam covers all three |
| Should the person see the case file? | Yes, as a side panel they can correct ("that's not my goal") |
| The gate for 37.10 | As proposed under Measurement, fixed before 37.0's dev run |
| The soft cap per case | $5, then ask |
| Persian in the exam | Skipped for now (the owner, 30 October): the exam is English only; Persian returns in a later exam |
| Who labels conversations beside the judge | The owner, on a sample of 10 per run |
| Retire the old Home flow after 37.10 passes, or keep both | Keep the old flow as the no-model fallback only |
| Keep sealed exams whole in an eval-only vault, instead of their shape only | Decided (the owner, 30 October): yes; the record exists, and no training set can read it |
| Train on Claude's outputs | Every output keeps its provenance; Claude stays silver and apart, as the owner ruled; a legal check on Anthropic's terms before a model trained on them is sold |
| Thinking effort | `xhigh`, as today, so every call thinks and keeps a summary; revisit if cost matters |
| The code exam's margin (37.18) | A tuned model may fall at most 2 percentage points below the model it replaces; fixed before the first round |

## After the review: the owner's decisions (30 October 2026)

The plan was reviewed before any build (a page shared with the owner). The owner accepted every recommendation; these
override the text above where they differ.

| Topic | Decision |
| --- | --- |
| Budget | $35 for the phase (the owner may add $50 on request); the workspace's monthly cap raised to EUR 150 for November |
| Capture | Everything kept: every conversation, the simulated person's and the judge's calls, their thinking summaries |
| What the exam measures first | Understanding of the decision behind the request and option quality; routing already matches Claude alone (HOME-025: Home with Claude 35, Claude alone 35 of 36) |
| Models | The agent Sonnet 5.5; the judge Opus 5.5, blind to the arm; the simulated person Haiku 5.5; the owner labels 10 conversations per run |
| Gate for 37.10 (fixed now) | `conversation_exam_gate_v1` in `dclab_rnd/agent_eval/conversation_exam/gate.py`: beat Claude alone on understanding and option fit (paired bootstrap), route equal or better, zero safety failures, not worse than today's Home on any trap, the judge agreeing with the owner's labels |
| Questions | No fixed count; a ceiling of about six questions, or two the person declined, then propose with the open points stated as assumptions |
| Cost on screen | The owner and admins see it; the person only when the soft cap stops the case |
| Readiness | Counts only facts the person said (quoted) or the data showed (computed); assumptions are marked and shown |
| Order | A thin slice first: 37.0, then 37.1, then a thin 37.2 to 37.4 behind `DCLAB_CASE_AGENT`; stop and decide on the dev result before the rest |
| Stale facts | The store gaps above were mostly closed on 30 October (routing readings 1,278 to 27 unexplained, campaign runs counted, the advice track answered); 37.13 to 37.15 extend `store/coverage.py` rather than build it |

## Status

| Id | Status | Notes |
| --- | --- | --- |
| 37.0 | done (CEX-004) | dev: understanding Home 0.50, Claude alone 0.50; route 80% vs 76%; Home repeats answered questions (51 vs 4); option fit about 3 of 5 for both; EUR 7.2 with the smoke runs. Open: the owner's ten labels (the Exam Labelling Desk page), the judge's agreement, the number rule's illustrative numbers (37.6) |
| 37.1 | done (dclab-core 0.148.0) | `dclab_core.case`: facts said (two whole words of the person), computed (by code, about the data) or assumed; versions; readiness and the ceiling; option checks; the problem card round trip exact. Open: the track-brief and problem-map views (37.3) |
| 37.2 to 37.4 | done, thin (CEX-009) | the lead agent on dev: understanding 0.88 (Claude alone 0.52, Home 0.50), option fit 4.1, no invented number, every trap kind at or above Home; route 73% (6 of 11 misses to advice for people whose data was not shared). Stop-and-decide: continue. Open: the routing fix, Home's route behind DCLAB_CASE_AGENT (37.9), memory and evidence in the context |
| 37.4b | done (CEX-010) | the owner (2026-10-31): the sealed gate is conversation_exam_gate_v2, the route check on the right route offered or taken (v1 kept); dev preview passes every check but the owner's labels. Advice fix: 4 of 5 data needs no longer answered as advice. Open: d-37 and d-40, advice people given data work |
| 37.5 | done | options say what DCLab does, what you get, the trade-off and the effort; one option per tool |
| 37.6 | done (CEX-011, CEX-012) | query, leakage check, step preview and build request run by code with reasons; the same data tools for Claude alone; cells decided per request |
| 37.7 | done, unused live | five specialists behind triggers and a budget of three per case, findings into the case file (v2); the agent sent none on dev |
| 37.8 | done | a checkpoint after every turn, resume with nothing lost, a soft cap (DCLAB_CASE_CAP_EUR, 5) that stops and asks |
| 37.9 | done, browser test owed | Home led by the lead agent behind DCLAB_CASE_AGENT: option cards and the case-file panel; make browser-test waits for Chromium (a download) |
| 37.12 | done | every case change a case.changed event (experience v17); a case rebuilds from its events |
| 37.13 | done | the capture map holds the lead agent's events and specialists; make capture-coverage |
| 37.14 | done, six prompts open | no new prompt may ask for reasoning; six earlier prompts that ask for a reason are listed for the owner |
| 37.15 | done | a sealed exam's calls kept whole in exam_vault/, counted by the store, never its content |
| 37.16 | done | dclab_rnd/ledger/follow.py: a list's results land on every decision of its project (business outcome), nightly from outcomes_inbox (make outcomes-follow) |
| 37.17 | done, code only | m4_workflow_llm/case_files from finished cases with their source (simulated half weight, real); no model-run curator wired yet |
| 37.18 | covered | by 36.4 to 36.6 (code sets, replay, pinned exams); the lead agent writes no code |

## خلاصهٔ فارسی

سامانهٔ DCLab با Claude کار می‌کند، ولی مثل مدلی بسیار ضعیف‌تر رفتار می‌کند، چون ساختار دور آن Claude را محدود کرده است. در دومین مجموعهٔ مهروموم Home، قواعد نوع کمک را در ۲۱ مورد از ۳۶ درخواست درست خواندند، Home با Claude در ۳۱ مورد و Claude به‌تنهایی در هر ۳۶ مورد.

**تصمیم‌های گرفته‌شده:**

- **اول Claude فکر می‌کند.** فهم، گفت‌وگو، استدلال روی داده، مسیریابی و طراحی راه‌حل با مدل است و کل پرونده جلوی چشمش است.
- **فرایند داخلی قطعی می‌ماند.** تقسیم داده، خواندن یک‌بارهٔ Holdout، معیارها، هر عددی که نشان داده می‌شود و ساختن با کد است.
- **اول فهم، بعد گزینه.** معمولاً حدود سه تبادل طبیعی، سپس دو یا سه گزینه که برای همین شخص نوشته شده‌اند.
- **یک عامل اصلی** پرونده را از جملهٔ اول تا ساخت نگه می‌دارد و هنگام نیاز زیرعامل‌های متخصص را می‌فرستد.
- **هزینه:** مدل Sonnet 5.5 با فکر کردن، حدود ۲ تا ۵ دلار برای هر پرونده، با سقف نرم ۵ دلار که پیش از گذشتن از آن می‌پرسد.

**به‌روزرسانی ۳۰ اکتبر:**

- **آزمون فعلاً فقط انگلیسی است.** فارسی در آزمونی دیگر برمی‌گردد.
- **ثبت همه‌چیز برای آموزش.** یک ثبت‌کننده در کد هر رویداد را می‌نویسد: هر نوبت، فکر خلاصه‌شده، هر فراخوان ابزار، هر حکم نگهبان و هر تغییر پرونده. یک پیگیرندهٔ نتیجه پیامدهای بعدی را به تصمیم‌ها وصل می‌کند و یک عامل مرتب‌کننده داده‌ها را برای هر نوع مدل جدا می‌کند.
- **همهٔ آزمون‌ها ثبت می‌شوند.** اجراهای dev، زنده، دوقلو، خودبازی و سفرها کامل نگه داشته می‌شوند. تصمیم مالک (۳۰ اکتبر): امتحان‌های مهروموم کامل در گاوصندوقی فقط برای ارزیابی نگه داشته می‌شوند که هیچ مجموعهٔ آموزشی نمی‌تواند بخواند.
- **فکر کردن در چارچوب قواعد Anthropic.** خلاصهٔ فکر با `display: "summarized"` گرفته می‌شود؛ هرگز از مدل خواسته نمی‌شود استدلالش را در متن پاسخ بنویسد؛ بلوک‌های فکر بدون تغییر نگه داشته می‌شوند؛ و منشأ هر خروجی Claude برای تصمیم دربارهٔ آموزش ثبت می‌شود.
- **کد هم ثبت می‌شود تا مدل‌ها فراموش نکنند.** کد همهٔ منبع‌ها (پایپلاین‌های صادرشده، سلول‌های دفترچه، کد عامل، تغییرات DCLab Code و commitهای مخزن) با نتیجهٔ اجرایش نگه داشته می‌شود؛ فقط کد راستی‌آزمایی‌شده نمونهٔ خوب حساب می‌شود؛ هر دور آموزش سهمی از داده‌های دورهای قبل و کد راستی‌آزمایی‌شده را دوباره می‌بیند؛ و یک امتحان ثابت کد پیش و پس از هر دور سنجیده می‌شود.
- **جا و ساختار داده‌ها.** داده‌های Phase 13 در همان انبار موجود و با همان ساختار ذخیره می‌شوند: نسخه‌های تاریخ‌دار روی دستگاه مالک و مخزن خصوصی `dclab-records`، با مجموعه‌های جدا برای هر نوع مدل. شکاف‌های امروز (نبود فکر در بسیاری از نمونه‌ها، نامتوازنی، کمبود نتیجه‌ها و یک قاعدهٔ قدیمی) هر کدام به بسته‌ای از ۳۷٫۱۳ تا ۳۷٫۱۸ سپرده شده‌اند.
- **بسته‌های تازه:** ۳۷٫۱۲ تا ۳۷٫۱۸، که بلافاصله پس از ۳۷٫۳ شروع می‌شوند.

**بسته‌ها:** نوزده بسته از ۳۷٫۰ تا ۳۷٫۱۸. اول امتحان گفت‌وگو و سنجش فاصله با Claude به‌تنهایی می‌آید، بعد پرونده، سازندهٔ زمینه، عامل اصلی، گفت‌وگوی فهم، گزینه‌های مبتنی بر زمینه، نگهبان‌ها، متخصص‌ها، مهار، صفحهٔ Home، امتحان مهروموم، روشن کردن و دادهٔ آموزشی.

**پرسش باز اصلی:** معمولاً چه کسی در DCLab می‌نویسد؟ تا پاسخ شما، عامل سطح شخص را از کلماتش می‌خواند و خودش را با آن تطبیق می‌دهد.

**تا تأیید شما هیچ کدی ساخته نمی‌شود.**
