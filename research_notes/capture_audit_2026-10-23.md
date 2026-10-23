# Training-data capture audit (2026-10-23)

Read-only audit. Nothing was edited, run live, committed or stashed. Counts are from disk on 2026-10-23 (store last built
2026-10-23 04:52 UTC; the private repo `dclab-records` last pushed 2026-10-22 09:51 +04:00, branch `owner/main`).
"Store" = `training_records/store/` (raw → records → sets). "Lake" = `lake/` (gitignored). Paths are relative to the
repository root.

Two gaps were known before this audit and are confirmed here, not repeated below unless new evidence was found:
(1) model thinking is almost never stored in training sets; (2) `pipeline.py` code exports are never produced or kept
(`find agent_runs -name pipeline.py` → 0; `notebook.ipynb` and `report.md` exist 60 times, only in twin-lane projects).

---

## (a) Inventory: one row per source

Legend. Produced: does it exist on disk, how much. Whole: are inputs, outputs, reasoning, state, reason and verdict
recorded, or cut. Stored: kept by the store (kind) and pushed to `dclab-records`. Set: which model set reads it and how many
rows. Missing → who loses.

| Source | Produced? | Whole? | Stored? | In a training set? | Missing → which model |
|---|---|---|---|---|---|
| **Home agent** (draft chat, `draft/chat.py`) | yes: 53 `draft.json` files in agent_runs (68 MB drafts), 493 model requests | turns whole in `draft.json` (messages, card, data_card, problem_map, workflow, brief, plan); model exchanges whole only for 33 of 493 requests (only since 32.10, 2026-10-20, and only with `DCLAB_LOG_PROMPTS=1`); production thinking: not asked for | yes, `conversations` + `model_requests`; pushed | M4 `understand_problem`: 221 home conversations + 33 exchanges (train). Text turns only | `from_draft` (`store/sets/llm.py:164`) keeps text only: the card, map, workflow the agent produced are never the target → **M4** loses "conversation → problem card / map / workflow" examples; tool calls of the turn dropped → **M5/M6** |
| **Home orchestrator / team chat** (`home_orchestrator/`) | 6 files (124 KB), 126 requests, 35 whole | turns whole | yes, pushed | M4 `plan`: 35 (stage label "plan", not its own) | none of 8 recorded thinkings have text (see gap 3) → M4 |
| **Intern LLM loop** (`intern/loop.py`) | 66 session files (most in twin lanes), 3 435 requests | session file now whole (`step_record`, loop.py:80, since 58ccfc48): messages incl. assistant text, tool args, whole results. Exchange log: shape only (`intern` has no `keeps_text`). Trace (`agent_steps.jsonl`, 600 rows): result cut to 360 chars (`agents/traces.py:31`) | yes, `conversations`; pushed | M5 `trajectories/studio/v2`: 7 train / 1 dev / 2 sealed; `silver_frontier`: 7/2/2 (consent "improve", not for training) | the assistant's own words before each tool call (e.g. "I expect ca and thal to be flagged for review") are in `session["messages"]` but the builder sets "the intern's step has no thought of its own" (`llm.py:407`) → **M5** gets no thoughts at all; older solves of the same table are dropped ("superseded", `newest_per_table`) → failed attempts lost for **M5/M7** |
| **Intern standard plan** (no model) | yes (same sessions, mode standard) | steps whole; no thought by design | yes | in M5 studio only when holdout + table known | fine as rules data |
| **Team/orchestra roles** (critic/auditor, reporter, data investigator, feature engineer, solution planner, challenger, debate judge, problem analyst…) | 2 000+ requests (e.g. solution_planner 472, data_investigator 646, problem_analyst 421, solution_router 278, solution_challenger 260, debate_judge 52, reporter 15, critic 16) | **shape only**: none of these purposes has `keeps_text` (`models/settings.py`); boards keep node/what/handoffs/decisions, not the role's prompt, reply or reason | boards/walks yes; prompts/replies never existed | M6 `team/orchestra` 42 train, `team/solution` 4 (state = step index + move code only) | the role's text is gone at capture → **M4** (workflow LLM per role) and **M7** (critic verdicts with reasons) lose the most |
| **Coordinator** (`draft/coordinator.py`, `dclab_core/coordinator.py`) | yes, inside drafts (`coordinator` key) and walks (8 `walks.jsonl` lines) | state machine steps whole in draft | yes | M6 `walk` 570 train | M6 step `state` is `{"step": i, "move": code}` only (`others.py:40`), action/outcome cut to 200 chars → **M6** learns sequences without states |
| **Conductor / store curator** | 57 conductor agent steps; 24 store_curator requests (shape only) | runs whole in `store/runs/` | runs yes | no | small; M2-internal |
| **Campaign LangGraph loop** (`agentic run`) | 7 runs, 201 events in `agent_runs/research.sqlite3` (Sep 2026); 272 campaign_review + 20 campaign requests | events whole in sqlite; requests shape only | **no**: the file matches `TABLE_FILES` (`.sqlite3`) and is kept as counts only (`keepers/standard.py`; confirmed in `raw/other/dev/…counts.json`) | no | 7 whole research-planning runs (planner, critique, decisions) never reach the store → **M4/M5** |
| **Twin: workflow lane** | 49 runs, 620 MB | intern session whole; trace 360 chars | yes, pushed | M5 studio / silver | as intern |
| **Twin: ai_first lane (Claude)** | 17 runs with `transcripts/thinking.jsonl` + `model_calls.jsonl` (thinking text present, e.g. heart-disease plan), 16 `expert_steps.jsonl`, 16 `compare.json` | whole, **with thinking** | yes (`conversations`, `twin/`); pushed | **no**: the M5 builder never reads `transcripts/`; `expert_steps.jsonl` unread; only `steps.jsonl` is read, for planted truths | the only Claude thinking DCLab owns is unused → **M5** (thought per step), **M4** (expert design record = problem/data/algorithms), **M7** (compare verdicts) |
| **Solver runs** (twin/solver_runs.py) | 16 (2026-10-21) + rules re-runs 2026-10-22 | as intern | yes | M5 studio/silver (above) | as intern |
| **MCP calls** (`mcp_server.py` → `mcp/calls.jsonl`) | **0 files** (never used) | designed whole | would be | no | nothing to lose yet; unverified in practice |
| **Workspace chat** (`workspace_chat/`) | **0 files** | purpose shown cells → shape only | — | no | — |
| **notebook_assist / copilot review / VS Code companion** | used, not captured | **no capture at all**: no `experience`, `ledger` or `harness` call in `notebook_assist.py` or `copilot/*.py`; `notebook_review` purpose is shape only | no | no | notebook reviews (findings on real notebooks) → **M7**, **M4 code stage** |
| **decide / choose** (`decide.py`) | API path exists; 2 person labels (`user_decision_exposed`) on disk | whole decision + label | yes | M1/M3 (as decisions) | fine; little data |
| **Notebook cells in the Jupyter kernel** (`code_runs/`) | 36 cells (2026-10-20); 18 `notebook_cell` requests, 8 whole | code, stdout (tables removed), stderr, ok; model's code whole when kept | yes (`tests`/`code_runs`), pushed | **no** set reads `code_runs/`; 8 exchanges land in M4 `plan` (no "code" stage) | agent-written code with its run result → **M5** (code + execution), **M7 code judge** |
| **Ledger decisions** (56 CAPTURED points, 10 NOT_DECIDED, `ledger/__init__.py:42,101`) | 154 003 decision versions (10 401 outside self-play), 136 552 outcomes, 5 719 labels, 23 corrections | state bands + options + answers; **reason almost never**: 0 records with a reason for 35 of 45 points seen (e.g. `column.is_identifier` 1 917, `column.semantic_role` 1 927) | yes, `decisions`; pushed | M1 62 sets (98 962 train), M2 14 sets (5 664), M3 63 sets (89 213) | **95 % of examples have no label** (M1 154 046 'none' of 160 872; M3 140 235 of 145 237) → **M1/M3** learn imitation of the rule, not truth. `agent.tool_choice` (765), `workflow.stop` (115), `evidence.records` (2) have no group → **dropped from every set** (`sets/decisions.py:77`) → **M2/M6**. Points captured but with **0 records**: `critic.send_back`, `department.proposal`, `holdout.reuse`, `lesson.accept`, `solution.compose` |
| **Labels: measured / benchmark / person** | benchmark 3 530, measured 1 397, model 790, person 2 | whole | yes | as above | person labels ≈ 0 (no users yet) |
| **People's corrections** (`ledger/corrections/`) | 14 unique (8 notebook_cell, 6 solution) | before/after whole; solution corrections have no reason | yes | M7 `preferences` 4 train; M7 `correction` holds 6 person rows | the preference **prompt is a fixed sentence** ("Propose the solution for this case.", `others.py:167`): no data card, no problem, no options → the pairs cannot teach **M7/M4** anything |
| **Approvals** (`workflow.stage_approval`) | 122 | decision whole, no reason | yes | M2 `stage_approval` 72 train | approver's reason → M2 |
| **Team interrupts** (questions to the person) | 47 lines | question whole; answer in board | yes | no | small → M2 next_question |
| **Experience events** (`experience/`) | 25 268 events in agent_runs (e.g. 805 `agent.step`, 788 `tool.call`, 98 `chat.agent`, 79 `chat.person`) | `agent.step` payload is shape only: agent, n, state, tool, verdict, seconds — **no arguments, no result** | yes, `sessions`; pushed | **no** set reads `sessions` (only `service.py` aggregates) | Home/orchestra/role steps exist only as shapes → **M5/M6** |
| **Agent traces** (`agent_steps.jsonl`, `agents/traces.py`) | 600 rows (intern only in dev; 66 more in twin lanes) | result cut to 360 chars | yes, `actions` | only M8 reads `actions` (search tools) | whole result only in intern sessions; other agents' whole results nowhere |
| **Model requests / usage log** (`model_requests.jsonl`) | 15 961 lines | shape + cost; prompt only with `DCLAB_LOG_PROMPTS=1` and **cut to 20 000 chars** (`gateway.py:501`) | yes | no (M4 reads only `model_exchanges/`) | the cut violates "never truncate before saving" where the exchange is not kept elsewhere |
| **Whole model exchanges** (`model_exchanges/`, 32.10) | 168 lines in the store (dev, 2026-10-20/21); only `20.jsonl` still on disk (21.jsonl in the store only) | 82 whole, 86 shape; **tool schemas never stored** (only `prompt_sha256` of the system prompt, `gateway.py:483`) | yes | M4: 33 + 35 + 8 + 3 + 2 | all requests before 2026-10-20 are lost as text (conversation_reader 1 994, decision_answer 2 160, decision_labeler 4 068, decomposer 224, understanding_critic 370) |
| **Teachers** (`judgments/`, reasoning teacher) | 11 kept versions; 6 reasoning_teacher requests | whole | yes | M4 via `teacher_reasonings` | teacher skips any exchange whose `thinking` list is non-empty even when every block's text is empty (`reasoning.py:114`) → see gap 3 |
| **Committee / Claude labels** (`lake/bronze/answers/claude/`) | 4 381 rows (3.3 MB), committee sealed sets | value + short reason + prompt **hash** | **no** (lake not walked by keepers) | no (eval only by rule) | single machine copy (+ local snapshots) → **M1 eval** at risk |
| **GLiNER2.5-Decide answers** (`lake/bronze/answers/gliner2.5-decide/`) | 86 MB | whole | **no** | lake export only | single copy → M1 committee/students |
| **Rules / classifiers / people answers in lake** | dclab-rules 31 MB, classifiers 408 KB, people 32 KB | whole | **no** | lake export | single copy (rebuildable for rules) |
| **Lake priors / sessions / decisions** (`lake/bronze/{priors,sessions,decisions}`) | 36 MB / 33 MB / 23 MB | whole | **no** | lake | single copy |
| **Lake exports** (`lake/export/decisions` 37 057 train / 9 908 val; `lake/export/solver` 785 / 158) | yes | whole | **no** (gitignored; only MANIFEST in git) | these *are* training sets, outside the store | a second, parallel set pipeline the store does not version |
| **Generated decision sets** (benchgen, `research/llm-fine-tuning/experiments/decisions/generated_v1, v2, real_v1, workflow_v1, …`) | ≈ 350 000 rows (generated_v1 89 280, export_v2 163 140, real_v1 28 425, workflow_v1 4 636 …) + fitted `models_*/*.joblib` | whole, labelled by the rules | **no** (rows gitignored; manifests in git; copies only in local `training_records/snapshots/`) | lake gold → lake export; **not in store M1/M3** | the main labelled data for students is outside the store and the repo → **M1/M3** |
| **Self-play** (`lake/bronze/selfplay/`, the only lake part the store walks) | 1 114 solves in `done.jsonl` (1 052 verified, 62 failed); 1 108 ledgers; `examples.jsonl` 10 380 chat examples (36 MB); `screens.jsonl` 1 024; `library_solves.jsonl` 2 | ledgers whole (state bands, options, labels) | yes (`decisions` + `other`), pushed | M5 `trajectories/selfplay` 533/189/304; `preferences/selfplay` 220/81/48; ledgers → M1/M3 | failed solves never become trajectories (only pairs) → **M5/M7** lose negatives; `examples.jsonl`, `screens.jsonl`, `library_solves.jsonl` stored but read by **no** store builder; 7 kinds have only 2 solves (anomaly, entity_tables, matching, metric_change, ranking, segmentation, streams) vs 100 for binary → **M5/M3** imbalance |
| **Journeys** (`agent_eval/journeys`) and graders | 27 result files (`evidence/campaigns/journeys_v1`), each with per-check verdicts | rows whole; the play itself (drafts, ledger, events) runs in a `TemporaryDirectory` and is **deleted** unless `--keep` (only live runs pass it, `journeys/__init__.py:322,367`) | results in git; plays not | workflow_v1 (lake), not store | rules-arm plays (expected vs actual questions/steps) vanish → **M2/M6**; grader verdicts never reach **M7** |
| **Harness eval** (`agent_eval/regression`, 21.5) | 39 result files, e.g. RG-01 with 6 graded steps ("expected exactly [...]") | rows whole | git | **no** | step-by-step expected/actual pairs → **M6**, **M7 step judge** |
| **agent_eval suite, benchmark v2–v4, analyst, investigator, conversation, committee, students_exam** | 52 / 2–3 / … result files in `evidence/campaigns/` | rows whole | git (pushed with the main repo) | **no** (only campaigns' model-selection candidates → M3 `experiments/*`) | judged cases → **M7**; frozen benchmarks for **no-forgetting** exist but no set links them to a model version |
| **Evidence campaigns: model selection** | 44 campaign folders, 18 MB | whole | git | M3 `experiments/*` (via `campaign_records`) | fine |
| **Project stage records / RunRecords / lineage** | 79 `project.json`, 356 stage files with `run` (RunRecord with fingerprint) | whole | yes, `workflows` | M6 `stages` 453 train; M3 experiments; M7 `result` (G4) 701 | stage `explanation`/`notes` text, `claims`, evidence not used as targets → M4 report stage |
| **Data cards / problem cards** | data_card in 76 `project.json`, 9 drafts; problem card in drafts | whole (bands) | yes | no set uses them as input or target | → **M4** (understand_data / understand_problem targets), **M3** (state for decisions) |
| **Solution workflows / problem maps / routes / programme** | 6 `problem.mapped`, 6 `solution.routed`; `solution.compose` 0; no programme record on disk | whole | yes | M2 `map.size` 6, `map.edit` 1, `solution.route` 7 | programme (24.5) never run → nothing |
| **Code exports** (`studio/code.py`) | `notebook.ipynb` 60, `report.md` 60 (twin lanes only); `pipeline.py` 0 | notebook kept without outputs | yes (`workflows`) | **no** set reads `exports/` | (known gap) → **M5** code |
| **Test runs and gates** (`training_records/test_runs/`) | 346 runs indexed, 16 gates; store `tests` 2 028 versions (133 MB) | whole | yes (`test_runs` keeper), pushed | M7 `code` 73 train (only failing→passing fixes) | other test runs unused → fine |
| **Fixes / commits** | 688 versions (patches + accepted fixes) | whole patch | yes, pushed | M7 `code` | fine |
| **Development-trajectory records** (`dev_trajectories/records/`, 370) | yes, in git | whole by design | git, not store (the `fixes` keeper reads `dev_trajectories/fixes` only) | SFT `out_dev` (3 069 / 801, in git); lake marks 2 628 `dev_record` rows `eval_only` | not in any store set → fine if `out_dev` is the route |
| **Claude Code sessions** (this repository) | 1 553 kept versions (826 MB in store); `dev_trajectories/raw` 597 MB copy | transcript whole in raw; in the set: **thinking text empty** for 6 595 of 8 185 blocks in the largest transcript; only 776 of 21 120 steps have a thought; later user turns dropped | yes, `agents`; pushed (up to 10/22) | M5 `trajectories/development/v2`: **52 train, 0 dev, 306 sealed** | see gap 1 and 5 → **M5** |
| **SFT corpora** | `out_v3` 264/66, `out_dev` 3 069/801, `out` (old); **`out_v4` never built** (needs ≥ 50 trajectories from 10 projects; studio has 15) | whole | git | they are sets, outside the store | v4 blocked by too few trajectories → M4/M5 |
| **Students / own models** | students kind 78 versions; `own_models/` 5 files; `decisions/models_*/*.joblib` local only | whole | students kind yes; research joblibs **no** | M1 pairs (teacher vs student) | fitted research classifiers single copy |
| **Audit, deletions, curation, tasks (jobs)** | 129 / 92 / 31 / 134 versions | whole | yes | tasks → M6 `job` 3 train; others none | fine |
| **Agent metrics, searches** | 55 / 12 files | whole | yes, `actions` | M8 `evidence` 4 train | → **M8** has 11 examples total |
| **Testers' records** (`testers/`) | **none collected** | — | — | — | no outside users yet |
| **Phase 9 loop records** (31.17, planned) | not built (`dclab_core/loop_record.py` absent) | plan says whole | plan writes "into the lake's bronze layer and the archive" (`PHASE9_LEARNING_LOOP_PLAN.md:391`) — **not the store** | — | as planned, would be single-copy like the rest of the lake |

---

## (b) Gaps ranked by value for the six models

Ranked by (data lost or locked) × (how central to the model) ÷ (size of the fix).

### 1. 300 Claude Code trajectories are locked in M5's sealed split by a redaction bug (M5)
- **Evidence.** `m5_solver_llm/trajectories/development/v1`: 334 rows, all sealed, every id `m5_solver_llm:claude_code:[ID]` — the
  PII redactor replaced the session UUID, so all sessions shared one key and `split_of("[ID]")` is "sealed". v2 fixed the id,
  but `keep_sealed` (`store/sets/llm.py:433`) keeps every example sealed that touches a raw entry v1 sealed: v2 has 306 sealed,
  **302 of them hash to train** by their own session id; dev = 0; train = 52.
- **Example.** Session `0058c91c-…` (20 steps) is sealed only because v1 sealed its raw entry under `[ID]`.
- **Smallest fix.** Treat v1 of `trajectories/development` as void (its sealing came from a bug, not a rule): exclude v1 from
  `keep_sealed` for this folder and cut v3. Also add a test that no two examples of a set share one id.

### 2. The intern's visible reasoning is in the session but never reaches M5 (M5, M4)
- **Evidence.** Intern session `messages` hold the assistant's words before each tool call (twin run `20261021-144909-90078b`,
  session `d101e16099c6`: 33 messages, e.g. "S2 of the design: propose the solution and review leakage … I expect ca and thal to be
  flagged"). `intern_trajectories` reads only `steps` and writes "the intern's step has no thought of its own" (`llm.py:407`).
- **Smallest fix.** In `intern_trajectories`, pair each assistant message's `content` with the tool calls it issued (by
  `tool_call_id`) and use it as `thought` with `thought_source="said"`. No capture change needed.

### 3. Thinking is billed but its text is empty, and then nothing fills the hole (M4)
- **Evidence.** All 32 whole exchanges that carry thinking blocks (home_agent 20, home_orchestrator 8, notebook_cell 2,
  synthetic_schema 1, reasoning_teacher 1) have **empty thinking text**. M4 has 0 examples with reasoning
  (`grep reasoning_source":"recorded"` → 0 in every M4 split). The teacher skips them because the list is non-empty
  (`store/reasoning.py:114`, `not r.body.get("thinking")`), and `from_exchange` finds no text, so they get neither.
  Likely cause (inferred, not verified with a live call): the model thinks by default but the client asks for
  `display: "summarized"` only when a thinking setting is passed (`anthropic_client.py:101-113`), which only the twin does.
- **Smallest fix.** (a) In the gateway, send `{"type":"adaptive","display":"summarized"}` for every purpose with
  `keeps_text`; (b) in `reasoning.py:114` and `llm.py:150`, test for thinking *text*, not a non-empty list.

### 4. Twin ai_first transcripts — the only Claude thinking DCLab owns — feed no set (M5, M4, M7)
- **Evidence.** 17 runs have `ai_first/transcripts/thinking.jsonl` and `model_calls.jsonl` with real thinking text; 16 have
  `expert_steps.jsonl` (25.3 expert trace) and `compare.json`. The M5 builder reads only `steps.jsonl` (for truths); nothing
  reads `transcripts/` or `expert_steps.jsonl`.
- **Smallest fix.** Join `thinking.jsonl` to the ai_first intern session by request number and fill `thought` for its steps
  (silver_frontier, weight 0.5, still "improve" until the terms are checked). Add an M4 `design_solution` example from each
  `expert_steps.jsonl` row (input: the brief; target: the expert record).

### 5. 95 % of decision examples have no label; agentic decision points are dropped (M1, M3, M2, M6)
- **Evidence.** Label sources: M1 154 046 none / 4 170 benchmark / 1 403 measured / 1 249 teacher / 4 person; M3 140 235
  none. `agent.tool_choice` (765 records), `workflow.stop` (115), `evidence.records` (2) have `group: null` and are dropped
  ("no group: it cannot be sealed", `sets/decisions.py:77`). Five captured points have 0 records (`critic.send_back`,
  `department.proposal`, `holdout.reuse`, `lesson.accept`, `solution.compose`).
- **Example.** The Home agent's choice `record` among 10 tools at step 1 of conversation `197d337059b9` is in the ledger and in
  no set.
- **Smallest fix.** Give conversation/run-scoped decisions a group (`conversation:<id>` or `run:<id>`) in the shaper's
  `ctx.group`; join self-play and benchgen planted truths as `benchmark` labels for the same states (they already exist in
  the lake's gold sets).

### 6. The labelled student data lives outside the store and outside any off-machine copy (M1, M3)
- **Evidence.** Generated decision sets (≈ 350 000 rows), lake bronze answers (GLiNER 86 MB, Claude committee 4 381 rows,
  rules 31 MB), priors, lake exports (37 057 + 785 train rows) and the research classifiers' `.joblib` files are gitignored
  and not walked by any keeper (keepers read `agent_runs/` and `lake/bronze/selfplay` only). Their only other copies are the
  local `training_records/snapshots/` on the same disk, and 6 of 9 snapshots listed in `INDEX.json` are already gone
  (only 20261018T214105Z, 20261021T202042Z, 20261022T055431Z remain). Disk: 3.5 GB free.
- **Smallest fix.** One keeper for `lake/bronze/{answers,priors,sessions,decisions}` (kind `teachers` for answers) and one for
  the generated sets' rows (kind `decisions`, source `benchgen/…`). Rules output is rebuildable, but GLiNER and Claude answers
  are not (time, money).

### 7. Role agents' prompts and replies are never kept (M4, M7)
- **Evidence.** 24 purposes are shape-only, including every team role: solution_planner 472, data_investigator 646, problem_analyst
  421, leakage_review 418, stage_notes 417, solution_router 278, solution_challenger 260, debate_judge 52, critic 16, reporter 15,
  intern 3 435. Many are "never rows or cell values" by their own description, so the cell rule does not forbid keeping them.
- **Smallest fix.** Set `keeps_text=True` on the purposes whose `may_see` already excludes cells (planner, router, challenger,
  critic, reporter, debate_judge, problem_analyst, data_need, brief_checker, stage_notes); keep shape-only for purposes shown
  cells. Add `tools` (the schemas sent) to the whole-exchange doc (`gateway.py:483-488`): today an M4 tool-calling example has
  `tool_calls` but no tool definitions.

### 8. M6 episodes have no state, and their actions are cut (M6)
- **Evidence.** `EpisodeStep(state={"step": i, "move": code}, action=action[:200], outcome=…[:200])` (`sets/others.py:40`).
  The records hold the real state (board version, open questions, blockers, graph status, validator reason, stage evidence).
- **Smallest fix.** Put the record's own state (board without values; graph move's from/to/status/reason; stage status and
  chosen model) into `state`, and drop the 200-char cut (the raw is whole, the set should be too).

### 9. Corrections become preference pairs with no context (M7, M4)
- **Evidence.** Every M7 preference prompt is `"Propose the <what> for this case."` (`sets/others.py:167`); the data card, the
  problem and the options are absent. 14 unique corrections exist (8 notebook_cell, 6 solution); solution corrections carry
  no reason.
- **Smallest fix.** Build the prompt from the project's data card + problem card at the correction's time (both stored), and
  ask for an optional reason in the UI when a person changes a proposed field.

### 10. Rules-arm journeys and harness plays are deleted after grading (M2, M6, M7)
- **Evidence.** `journeys.play` runs in a `TemporaryDirectory` and copies it only with `--keep` (live runs only). The harness
  rows keep step verdicts ("expected exactly [...]", e.g. RG-01) but no set reads `evidence/campaigns/*/results`.
- **Smallest fix.** Default `--keep` to `agent_runs/live_tests/<date>_journeys_<split>/` for every run (rules runs are cheap
  and small), and add an M7 `step` set from harness/journey checks (candidate = actual steps, evidence = the grader).

### 11. Failed runs are kept in raw but not in sets (M5, M7)
- **Evidence.** 62 failed self-play solves are only preference pairs, never trajectories; older solves of a table by the
  same solver are dropped as "superseded"; M5 studio has 15 trajectories. No M7 set judges a failed trajectory.
- **Smallest fix.** Write failed solves to `trajectories/*/failed` (score + missed checks), and an M7 `trajectory` set
  (passed vs failed solve of the same kind and seed).

### 12. The PII redactor corrupts code and technical text in M4/M5 sets (M5, M4)
- **Evidence.** In M5 development v2 train (52 rows): 5 531 `[ID]`, 2 533 `[URL]`, 1 966 `[NAME]`. Tested on
  `dclab_core.redact.redact_text`: `hashlib.sha256(raw)` → `hashlib.[ID](raw)`; `git show 98ec55b8` → `git show [ID]`;
  `Claude Code ran …` → `[NAME] ran …`. Raw stays whole; the damage is in records and sets.
- **Smallest fix.** For the owner's own sources (`agents`, `fixes`, consent train, workspace `rnd`/`dev`), apply only the
  secrets redaction (`keepers/agents.SECRET`), not person-PII redaction; keep PII redaction for testers and people's text.

### 13. Uncaptured agents (M4, M7)
- **Evidence.** `notebook_assist.py` and `copilot/*.py` write no experience, ledger or harness record; the campaign loop's
  `research.sqlite3` (7 runs, 201 events) is counted, not kept (matches `TABLE_FILES`).
- **Smallest fix.** Wrap `notebook_assist` and the copilot review in `harness.session("notebook_assist")`; add
  `research\.sqlite3` to `NOT_TABLES` in `keepers/standard.py` (as `checkpoints.sqlite3` already is) — or export its events
  as JSONL.

### 14. Cards, maps and workflows are never targets (M4)
- **Evidence.** Drafts hold `problem`, `data_card`, `problem_map`, `workflow`, `brief`, `plan`; projects hold `data_card` (76). No
  builder uses them. `from_draft` keeps text turns only.
- **Smallest fix.** One M4 example per built draft: input = the conversation turns and the data card; target = the problem card,
  map and workflow JSON (stage `understand_problem` / `design_solution`).

### 15. Agent-written code and kernel runs feed nothing (M5, M7)
- **Evidence.** 36 kernel cells in `code_runs/` (code, stdout, stderr, ok) and 8 whole `notebook_cell` exchanges; no builder
  reads `code_runs/`; the 8 exchanges are filed as stage "plan" (`STAGES` has no `notebook_cell`).
- **Smallest fix.** Map `notebook_cell` → stage `code` in `STAGES`; add an M7 `code` example per cell (code, ok/err as the verdict).

### 16. Frozen exams and versioned manifests for "no forgetting" (all models)
- **Evidence.** All 159 set manifests have `commit: null` (the run context has the commit; `write_set` is never given it).
  `sealed.jsonl` is written once, at the first build; later sealed rows wait in `sealed.pending.jsonl` (gitignored) until a new
  version, so frozen exams stay at first-build size (M1 6 770 sealed vs 25 201 pending; `column.is_identifier` 34 vs 7 298).
  Sets with no sealed rows at all: 9 in M1, 5 in M2, 5 in M3, M4 `plan`, M6 `team/solution`; M5 development has 0 dev.
  `train.jsonl`/`dev.jsonl` are rewritten in place on every build and never pushed. No general-skill evaluation set
  (instruction following, coding, general reasoning) is stored for checking forgetting after a fine-tune.
- **Smallest fix.** Pass the run's commit and the redactor version into `write_set`; on each training run, freeze a copy of the
  exact train/dev rows (or their ids + sha) as `sets/<model>/<set>/<version>/trained/<run>.ids.jsonl` and push it; cut v2 of
  the sealed sets on a schedule so pending rows become exam rows; add a small general-skill eval set per LLM family.

### 17. Single-copy data since the last push (all)
- **Evidence.** `dclab-records` working tree: 332 modified + 120 untracked paths since 2026-10-22 09:51 (Claude Code transcripts
  of 10/22–23, agent metrics, quarantine). `model_exchanges/2026/10/21.jsonl` is no longer in `agent_runs` and exists only in
  the store.
- **Smallest fix.** Push the store after each conductor run (or nightly), not only per milestone.

### 18. Claude Code dev trajectories drop the owner's later turns and most thinking (M5)
- **Evidence.** `_trajectory` (`llm.py:203`) keeps only the first user text as `problem` and tool results; later user
  messages (corrections, new instructions) are not steps. In the largest transcript 6 595 of 8 185 thinking blocks have empty
  text (the client's display setting; cannot be fixed after the fact). Images: 10 counted, never kept.
- **Smallest fix.** Add each later user text as a step (`tool=None`, `actor=person`), which also gives M7 natural
  correction pairs ("the agent did X, the owner said Y").

---

## Cross-cutting checks (short answers)

- **Thinking per agent.** Asked for only in the twin lane (`twin/claude.py`, `twin/home.py`, `twin/design.py`). Elsewhere
  blocks may arrive with empty text (gap 3). Kept: twin transcripts and whole exchanges. Used: never.
- **Code.** Kernel cells kept, unused (gap 15); `pipeline.py` never produced (known); fixes/commits kept and used by M7 `code`;
  code text in sets is damaged by redaction (gap 12).
- **Tool schemas / system prompts.** System prompt kept inside whole exchanges plus `prompt_sha256`; tool schemas never (gap 7).
- **Person corrections as preferences.** 4 train pairs, no context (gap 9).
- **Failed runs.** Kept in raw; mostly dropped from sets (gap 11).
- **Consent.** Every raw entry is `train` (23 791); no silent drop found. Counts-only: table folders (`data/`,
  `clean_exports/`, `datasets/*.parquet`), images, and `research.sqlite3` (gap 13). `consent.json` of agent_runs: train true.
- **Single copy.** Lake (all but self-play), generated decision rows, research `.joblib` models, local snapshots, unpushed store
  changes (gaps 6, 17).
- **Split coverage.** See gap 16; M5 development broken by gap 1.
- **Not verified.** Whether the Home agent's own tool steps are written anywhere whole (only `agent.step` shapes were found);
  the cause of empty thinking text (inferred from the client code, no live call made); MCP and workspace-chat capture in
  practice (0 records); whether `make datasets` output (`agent_runs/datasets/*.parquet`, counted only) holds anything not
  rebuildable.
