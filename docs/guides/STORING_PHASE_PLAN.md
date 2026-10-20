# The storing phase: capture, record and store everything, under the hood (plan)

Written 2026-10-20 from the owner's words: *before start phase 9 i call it storing phase: i want to add another multi
agents system that capture record and store all data we need … each agent responsible for each tasks in this process like
session, action, decision, fixes, conversations, tasks, workflows, everything that we can … for collecting training data
and fine tuning data … also for improving the system … works under the hood … store all things in the repo and locally
first and then we move it to server … my trusted friends clone and use the R&D app … store all they did and when they
pushed all store and save in git repos in their branches … make guideline for them.*

Later the same day the owner added four things:

- *Also store other agents' actions, decisions and tasks … capture data and working on student models also under the
  hood; maybe we should make another agent that helps us do this.*
- *Keep data even they remove project.*
- *Storing must be organized, clean, clear and structured, not only the directory `training_records/store/`: inside it,
  structured folders, each related part in a separate folder, and data with a structure good enough to use in the future
  for training and fine-tuning; `dclab_rnd/store/` organized and structured too.*
- *We have six types of model that we should collect and store data for.*
- *We should store all teacher and student data.*
- *What else can we store that is useful … this agent must collect information so that in the future a service helps
  businesses build these six types of model for themselves.*
- *We need to collect data and knowledge of the correct way to build these models for businesses.*
- *The orchestrator or each agent should have the correct rule to capture the right data for each task; guardrails and
  validation for both input and output; the agents using Claude (Sonnet 5.5) should handle this; but the guardrails must
  not prevent us from capturing and storing the right data. We should take care that the data we store is correct.*
- *The right way we must capture data for all the models, and collect the knowledge for learning how to give services to
  our businesses.*
- *Yes, add all* (2026-10-20): the more data of Part F, the two more model types M7 and M8, and the service's Part G.
- *All this capturing, collecting, recording and storing should involve an agentic system to capture and store the right
  data for each model and save it in the right positions; and all LLM models need a reasoning and thinking part, to make
  them better at solving.*

**Status: a plan; built after Phase 8 (Part G included) and before Phase 9.** Packages 32.1 to 32.31. Phase 9 stays a
plan; it learns from what this phase keeps.

## The eight model types

The owner named six (M1 to M6) and accepted two more (M7, M8) on 2026-10-20. Everything kept is for these eight. Each
has its own folder in the store, its own record shape and its own builder.

| # | Folder | Model | What it learns | Its examples |
|---|---|---|---|---|
| M1 | `m1_students` | **Student models**: small, many, one or more per decision of every other model | to answer one small choice as well as its teacher (the rule, Claude, a person), cheaply | a decision example (state, options, answer, label, its source and weight) per point |
| M2 | `m2_workflow_decisions` | **Workflow decision models** for DCLab, the R&D system and internal automation | the internal workflow: the next question, ready or not, the route, the next node, a stop, a gate, a tool | a decision example per workflow point, by system (`dclab`, `rnd`, `internal`) |
| M3 | `m3_ml_decisions` | **Data science and machine learning decision models** | the technical choices on a table: an identifier, a leak, the target, the framing (classification and the others), the split, the metric, a cleaning step | a decision example per technical point, by task family |
| M4 | `m4_workflow_llm` | **An LLM that knows the machine-learning workflow** | from understanding the problem and the solution to the plan and the code | chat examples by stage: `understand_problem`, `understand_data`, `design_solution`, `plan`, `code`, `report`; preference pairs |
| M5 | `m5_solver_llm` | **An LLM that knows many machine-learning problems, solutions and code**, to solve MLE-bench and lead it, beside M4 | to solve a whole problem as an agent: read, explore, build, submit | whole trajectories (each step's thought, tool call and result) and problem → solution → code cases, by problem kind |
| M6 | `m6_stepwise_decisions` | **A step-by-step decision model** | to decide a sequence: at each step, given what happened so far, the next action and its expected outcome | episodes: steps of (state, options, action, outcome), the episode's final outcome |

| M7 | `m7_judges` | **A judge or reward model** | It scores a plan, a solution, a report or a step. It is what lets M4, M5 and M6 learn from their own attempts (preference training, reinforcement, search), and what the curators and the service use to say "this is good". | a candidate, the score or the preferred one, and who judged it (a person, a measured outcome, Claude); the twin's verdicts, the critic's, the reviewer's, the gates' |
| M8 | `m8_retrieval` | **A retriever and reranker** | It finds the evidence, the business example, the pitfall or the earlier case that fits. The roles and the LLMs are only as good as what they read; the search today is lexical and measured weak on plain words (30.18). | a query, the passages offered, the ones used or cited, and whether the answer passed |

M1 to M3 and M6 are decision models (classifiers or small sequence models). M4 and M5 are language models. M7 is a
scoring model, a small classifier or a language model as a judge. M8 is an embedding and reranking model. M7 and M8 are
needed by the others: M7 lets M4 to M6 learn from their own attempts, and M8 decides what every role reads. A source
usually feeds more than one: a team's walk is an M2 example at each hand-off, an M6 episode as a whole, and an M5
trajectory when the agent is Claude.

## What exists, and what it misses

DCLab already records a great deal, each kind in its own store:

- the experience log (`dclab_rnd/experience`: visits, chat, stages, graph moves, agent steps, tool calls, model requests,
  decisions, feedback, outcomes, jobs, errors, runs);
- the decision ledger (`dclab_rnd/ledger`), the audit (`audit.py`) and the agent traces (`agents/traces.py`);
- model requests, the twin's lanes, the orchestra's boards and walks, the students' register;
- test runs and gates (`training_records/test_runs`), the development records (`devlog.py`) and the raw Claude Code
  transcripts (`make dev-archive`);
- the lake (`lake harvest`, `make lake`), the datasets (`make datasets`) and snapshots (`make archive`).

They are scattered across the workspace folder, the lake, `training_records/` and `research/`. Each has its own shape,
and none is organised by the model it trains.

What it misses, found on 2026-10-20 (the gaps the phase closes):

| Kind | Where it lives now | What is lost |
|---|---|---|
| Tasks (jobs) | `jobs.json` or the `job` table, rewritten in place | every state but the start and the stop |
| Intern sessions | `<home>/intern/<id>.json`, rewritten | every version but the last |
| Team and orchestra runs | `orchestra/` (boards, interrupts, walks, checkpoints) | never harvested, never shared by a tester |
| Workflows | spread over the ledger, graph moves and journeys | no record of one run of a workflow as a whole |
| Fixes | hand-written development records and git | the link from a failing test to the change that fixed it |
| Other agents | Claude Code transcripts of the main checkout only | the worktrees' sessions; testers' Claude Code sessions |
| Students, judgments, own models | the workspace folder | not harvested; not shared by a tester |
| Everything in the experience log | 730 days of events, 365 of model text (`experience/retention.py`) | anything older, unless a snapshot kept it |
| A deleted project, draft or session | the workspace, until deleted | everything, the moment it is deleted |
| Testers' records | one shared branch, a folder per tester | a tester can write into another's folder |

## Principles

1. **Keep whole, then shape.** A keeper copies a source whole, compressed, with its hash, before anything summarises it
   (the owner's rule: never truncate before saving). Nothing is rewritten in place: a changed source is a new version
   beside the old.
2. **Three tiers, one direction.** `raw/` holds what was kept, whole. `records/` holds the same data normalised into one
   record shape per kind. `sets/` holds training sets per model type, built from `records/` only. A tier is rebuilt from
   the tier before it; `raw/` is never rebuilt, only added to.
3. **One folder for one thing.** Every kind of data, every model type, every point and every task has its own folder.
   The path says what a file is (see *The layout*). A file never mixes kinds.
4. **A shape good enough to train on, years from now.** Every record and every set is a versioned `dclab_core` contract
   with a JSON Schema (`make core-schemas`). Each record carries its provenance (the source, the raw entry and its hash,
   the commit, the consent, the person or agent) and its split. Each set has a manifest and a dataset card. A file can be
   read without the code that wrote it.
5. **One keeper per kind of data, one conductor.** Each keeper knows its sources, what is new since its last run, and
   the model types its kind feeds. The conductor runs them, records each run, and never lets two keepers write the same
   path.
6. **Under the hood.** Keepers, shapers and builders run as jobs of the worker (`python -m dclab_rnd.worker`), on a
   schedule and after events (a gate, a finished run, a share, a deletion), never in a request's path. A failure is
   recorded and never blocks the product.
7. **The store keeps what it kept** (the owner, 2026-10-20: "keep data even they remove project"):
   - Deleting a project, a draft or a session, or the retention sweep, removes it from the workspace, never from the
     store. The keepers copy first.
   - The deletion is recorded beside the data (`raw/deletions/`), so a later use can tell a removed project from a live
     one.
   - A keeper copies only what its workspace's consent allows when it copies (`consent.py`).
   - Only the owner's audited command (`python -m dclab_rnd.store forget ...`) removes from the store. It writes a
     tombstone and removes the data from every tier.
8. **Never a key or a secret.** The same scrub as the tester share (`testers.NEVER`) and `dclab_core.redact` run before
   `records/`. A table's rows are kept only where a person said yes to that table. A row of a table never enters
   `records/` or `sets/` (aggregates, bands and names only, as in the ledger).
9. **Sealed is sealed.** A set's sealed partition is written once, by a rule fixed before it (the hash of its group key),
   and never read by a builder, a tutor or a test that tunes anything.
10. **Agents choose, code checks.** Keeping is mechanical: the keepers copy everything, so nothing is lost to a wrong
    choice. Choosing what each model learns from is agentic. The curators (32.9) decide which records belong to which
    model, where they go and how good they are. Every write goes through one guarded tool that checks the path, the
    contract, the split and consent. Every choice a curator makes is itself a recorded decision.
11. **Every LLM example reasons.** An M4 or M5 example has its reasoning before its answer, and each step of a trajectory
    has its thought before its action. The reasoning is labelled by where it came from (32.10) and checked like any
    model-written text.
12. **Locally and in a repository now, a server later.** The store is a folder (`training_records/store/`) pushed to the
    owner's private records repository, one branch per person (the owner and each tester). The R&D repository keeps the
    store's catalog (counts, kinds, hashes, schemas), not its bulk. A server is a later backend behind the same
    interface.

## The layout

### The store on disk: `training_records/store/`

```text
training_records/store/
  README.md                        what is here and how to read it (generated, with the counts)
  CATALOG.json                     every kind, record type and set: counts, schemas, hashes, last run (generated)
  schemas/                         the JSON Schema of every record and set contract, by version
  raw/                             tier 1: kept whole, append-only, never rebuilt
    sessions/                      the experience log's events, per workspace and day
    actions/                       agent steps and tool calls
    decisions/                     the ledger: records/, labels/, outcomes/
    conversations/                 home/, intern/, workspace_chat/, twin/, mcp/
    tasks/                         every job's state changes; the development sessions' task lists
    workflows/                     each walk from a draft to a report; team and orchestra boards, walks, interrupts
    fixes/                         a failing test, the passing run after it, the commits between (proposed and accepted)
    tests/                         test runs and gates
    model_requests/                every model request with its shadows and checks
    agents/                        other agents: claude_code/ (main checkout and worktrees), testers/<id>/
    teachers/                      every teacher's answer, whole: <teacher>/<point or task>/ (Claude, GLiNER, the rules,
                                   people, any served or shadow model), with its prompt, its thinking, its cost and its version
    students/                      every student, whole: <point>/<student>/<version>/ holds the fitted model, its
                                   training-set snapshot (ids and hashes), its card, every measurement, every shadow answer,
                                   and its routing level and history
    audit/                         the audit trail
    deletions/                     each deletion in the product, recorded beside what it removed
    curation/                      every curator's choice: where each record went, its quality, why, and its failed answers
  quarantine/                      records kept out of a set: <reason>/ with the failed check; released by a person or by the rule
    <each kind>/<workspace>/<yyyy>/<mm>/<dd>/<source>-<entry>.jsonl.gz, with index.jsonl (entry, hash, bytes, version)
  records/                         tier 2: one normalised record per line, one shape per kind
    <kind>/<workspace>/<yyyy-mm>.jsonl.gz
  sets/                            tier 3: training sets, one folder per model type
    m1_students/<point>/<version>/
    m2_workflow_decisions/{dclab,rnd,internal}/<point>/<version>/
    m3_ml_decisions/<family>/<point>/<version>/          family: classification, regression, forecast, ranking, ...
    m4_workflow_llm/<stage>/<version>/                   stage: understand_problem, understand_data, design_solution, plan, code, report
    m5_solver_llm/{trajectories,cases}/<problem_kind>/<version>/
    m6_stepwise_decisions/<process>/<version>/           process: the walk, a stage, the intern's session, a team's walk
    m7_judges/<what is judged>/<version>/                plans, solutions, reports, steps, code: candidates with scores or pairs
    m8_retrieval/<corpus>/<version>/                     evidence, business examples, pitfalls, cases: queries, offered, used
    each <version>/ holds: train.jsonl, dev.jsonl, sealed.jsonl (written once), MANIFEST.json, CARD.md
  runs/                            each conductor, shaper, builder and tutor run: what it read, wrote, skipped and why
  forgotten/                       the owner's tombstones (what was removed, when, why), never the data
```

### The package: `dclab_rnd/store/`

```text
dclab_rnd/store/
  __init__.py          the public interface: Store, conduct, shape, build, catalog, forget
  __main__.py          python -m dclab_rnd.store run|shape|build|status|catalog|forget
  layout.py            every path of the layout above, in one place; nothing else joins store paths
  store.py             append-only writing, versions, hashes, the index, the read-back check
  conductor.py         runs the keepers, then the shapers, then the curators, then the builders; one run record each; a harness session
  curators/            the agentic layer, one agent per model type (m1 … m8) and the librarian that routes to them;
                       write.py, the one guarded write tool (path, contract, split, consent checked)
  reasoning.py         the reasoning of an LLM example: kept, drawn from its record, or written by a teacher and checked
  keepers/             tier 1, one module per kind: sessions, actions, decisions, conversations, tasks, workflows,
                       fixes, tests, model_requests, agents, students, audit, deletions
  shapers/             tier 2, one module per kind: raw entries into records (scrubbed, with provenance)
  sets/                tier 3, one module per model type: m1_students, m2_workflow_decisions, m3_ml_decisions,
                       m4_workflow_llm, m5_solver_llm, m6_stepwise_decisions, m7_judges, m8_retrieval;
                       splits.py (the one split rule)
  tutor.py             the students' agent (32.14)
  catalog.py           CATALOG.json and README.md
  forget.py            the owner's audited removal
  share.py             a branch per tester (32.15)
```

The contracts live in `dclab_core/store.py`, as every package's do. The studio's side is `dclab_rnd/store/`. The main
product would get an adapter and a `CORE_INTEGRATION.md` section.

### The record and set shapes (`dclab_core/store.py`)

- **`StoreEntry` v1:** one kept source. Kind, source, workspace, version, sha256, bytes, captured_at, consent, the
  keeper's run.
- **`Provenance` v1:** carried by every record and every example. The raw entry and its hash, the source, the commit,
  the workspace, the project or draft (deleted or not), the person or agent, the consent, and when it was made.
- **`StoreRecord` v1:** one normalised record of a kind, its kind-specific body typed by kind:
  - sessions: `SessionRecord`
  - actions: `ActionRecord`
  - decisions: `DecisionRecord`, the ledger's own
  - conversations: `ConversationTurn`
  - tasks: `TaskRecord`
  - workflows: `WorkflowRecord`
  - fixes: `FixRecord`
  - tests: `TestRunRecord`
  - model requests: `ModelRequestRecord`
  - other agents: `AgentSessionRecord`
- **`DecisionExample` v1 (M1, M2, M3):** point, system or family, state (bands, codes and counts; never a value), the
  options, each answerer's answer, the label with its source and weight, the split, provenance.
- **`ChatExample` v1 (M4, M5):** the messages (system, user, assistant, tool) as the model is trained. Every assistant
  turn has its `reasoning` (the thinking before the answer) and its `reasoning_source` (`recorded`, `from_record` or
  `teacher`). It also carries the task and stage, the answer's check (cited, grounded, passed), the split and its
  provenance. **`PreferencePair` v1:** a prompt, the chosen
  and the rejected answers, and who decided.
- **`Trajectory` v1 (M5):** a problem, then every step: what the agent read, its thought (with its reasoning source),
  its tool call and the result. Then the submission and its score against the problem's own metric.
- **`CurationDecision` v1:** one curator's choice for one record. The model type, where it went (or why not), its
  quality (keep, keep as silver with a weight, reject), the reason, and the curator (a rule or a model).
- **`JudgedExample` v1 (M7)** and **`RetrievalExample` v1 (M8)**, as in 32.13.
- **`Episode` v1 (M6):** a process, then steps of (state, options, action, outcome), and the final outcome.
- **`SetManifest` v1:** the model type, the set's name and version, the record shape and its schema version, the
  sources with their hashes, the split rule and the counts per split, the label sources and weights, the consent, the
  decontamination against every sealed set and benchmark, and the commit.

## Capture rules, guardrails and validation

The challenge has two sides. Store the wrong data and every model learns the wrong thing. Let a guardrail throw data
away and it is lost for good, along with the evidence of what went wrong. Both are solved by one design: **validation
labels and routes data, it never deletes it.**

### Never lose, always label

| Where | What is checked | When a check fails |
|---|---|---|
| **Capture** (the keepers, tier 1) | almost nothing: the bytes are kept whole first | nothing is ever refused. A source a keeper cannot read is kept as bytes, with the error beside it |
| **Input to an agent** (a curator, a shaper, the teacher) | the record parses as its contract; its provenance is complete; its consent allows the use; no secret or table row is in what the model will read | the record is not sent to the model. It goes to `quarantine/<reason>/` with the failed check, and the rules handle it or a person does. It stays in `raw/` |
| **Output of an agent** (Claude's answer, a rule's) | it parses as the expected shape; every choice is one the layout, the contract and the split allow; every number is in its record (`cited.py`); it does not contradict the record (a label the record refutes, a time after the decision) | the answer is not written to `sets/`. It is kept in `raw/curation/` as a failed answer, a training example for M7 and for the curator's own student, and the rules' answer is used |
| **Into a set** (tier 3) | the capture rule of its task is met (below); no duplicate; no near copy of a sealed case; the label's source and weight are set | the example is kept out of that set, filed under `quarantine/` with why, and counted in the catalog. It can be released by a person, or by the rule once the missing part arrives |

Three consequences:

- `raw/` is complete whatever a check says. A check only decides what each set uses.
- A failure is data too. A quarantined record, a refused write and a wrong answer by a curator are each kept with their
  reason, and train M7 (what is wrong) and the curators' students.
- The one thing removed is a secret's value (a key, a password, a token). It is replaced by a marker that says what it
  was, and the record around it is kept.

### A capture rule for every task

`dclab_core/store_rules.py` holds one `CaptureRule` v1 per task. Tasks include a stage run, a Home conversation, an intern
session, a team's walk, a job, a test run, a model request, a correction, a build and a curator's choice. Each rule says:

- **what must be captured:** the required fields, with where each comes from. A stage run must carry its input card, its
  choices with their reasons, every fold's score, the holdout read count, the seconds and the cost;
- **when:** at the moment the decision is made, never reconstructed afterwards. A decision's state is what was known then,
  so the state is checked against the record's own time (nothing written after it);
- **which model types it feeds,** and the example each one gets from it;
- **what makes it correct:** its checks. For a label: its source, and whether a person or a measured outcome confirmed it.
  For a build: its verdict (32.26). For a number: that it traces to its run;
- **what is sealed:** the group key whose hash decides the split.

How the rules are used:

- **The conductor runs the rules after every capture.** A task that is missing a required field is flagged at once:
  - the keeper that should have caught it is named;
  - the flag shows on the Admin page and in `runs/`;
  - the gap goes to the code's owner as a finding, so capture is fixed where it happens, not patched later.
- **Rule changes are versioned.** A record says which rule version it was checked by, so a stricter rule later can
  re-check old records without losing them.

### The agents that use Claude

Claude Sonnet 5.5, the product's model, serves the cases rules cannot decide. Each curator and the librarian is a model
loop in the harness (`runtime.run`, always traced), with:

- **typed tools only:** read a record, propose a placement, propose a quality, write through `curators/write.py`. Claude
  never writes a file or a path itself;
- **its input guarded:** a record is shown in its scrubbed shape (bands, codes, names, the person's words redacted), and
  never a row, a key or a contact;
- **its output validated** as in the table above, before anything is written. A refused answer falls back to the rules,
  and both are kept;
- **its thinking kept whole** (32.10);
- **its cost capped** per run and per month (the role caps of 21.4), with the cap on the command line for a backfill;
- **shadow first:** it acts only where it agrees with a person's review of a sample better than the rules do, measured
  and switched by the audited switch. Until then its answers are recorded, compared and never used.

The rules stay the floor. Claude, once measured better, decides above it. Code checks both.

## The right data for each model

What each model needs, written as its capture rules (`store_rules.py`) and checked by its curator. Every example of
every model carries its provenance, its split and its consent (principle 4); the table adds what each model needs on top.

| Model | An example must hold | It is correct when | It is wrong, and kept out, when | Where it comes from |
|---|---|---|---|---|
| **M1 students** | the point, the state as it was at the decision, the options, the teacher's answer, the label with its source and weight | the label is a person's, a measured outcome's or a benchmark's; a teacher's answer is labelled as silver, apart | the state holds anything written after the decision; the label is the student's own answer fed back; the row is in a sealed partition | the ledger, every teacher's answers (`raw/teachers/`), the tutor's refits |
| **M2 workflow decisions** | the system (`dclab`, `rnd`, `internal`), the point, the situation (what was open, asked, answered, blocked), the options offered, the move taken, what followed | the move's outcome is recorded: the person went on, came back, corrected it, or the walk was built | the move was refused by the validator and the refusal is missing; the outcome was assumed, not observed | graph moves, the coordinator, the Home agent's next question, the teams' hand-offs, jobs, gates, the conductor |
| **M3 ML decisions** | the point, the column's or table's profile in bands, the family, the answer, the label | the label comes from a planted trap (benchmark), a person's choice, or a measured result (a leak found by the audit) | the profile holds a cell value; the label is the rule's answer and no one confirmed it; the table is in a sealed set | the ledger's technical points, the replicas' planted traps, benchgen, OpenML |
| **M4 workflow LLM** | the stage, the context the model would have (the cards, the evidence), the reasoning, the answer, its check | the answer passed its check (cited, grounded, the gate) or a person accepted it; the reasoning is labelled by its source | the answer has a number its context does not hold; the context holds data the model may not read; the stage is unknown | Home, the intern, the explanations and reports, the case trajectories, the code exports, the development records |
| **M5 solver LLM** | the problem (its description and data card), every step's thought, tool call and result, the submission, its score on the problem's metric | the score is measured by the problem's own metric on a split the agent never saw | the agent saw the holdout or a test label; the problem is an MLE-bench task or a sealed case in a train split | the intern, the teams, Claude's AI-first runs, the campaigns, the real problems, the replicas, OpenML, self-play |
| **M6 step-by-step** | the process, each step's (state, options, action, outcome), the final outcome | every step's outcome is observed from the record, in order | a step is missing, so the sequence has a gap that is not said; the final outcome is written by hand | graph transitions, stages' experiments, intern sessions, team walks, conductor runs |
| **M7 judges** | the candidate, its score or the preferred one, the judge and the kind of evidence | the evidence is a measurement, a test, a build's verdict or a person's choice; Claude's judgment is labelled as a model's | a model judged its own answer with no other evidence; a pair's two candidates answer different questions | twin verdicts, gates, the critic, the reviewer, corrections, build verdicts |
| **M8 retrieval** | the query, the passages offered, the ones used and cited, the answer's result | the answer that used the passages passed its check | the cited passage is not among those offered; the query is a sealed retrieval question | every evidence search, the 60 questions, the practice set |

Every model shares two rules:

- **The state is what was known then.** A decision's or a step's state is captured at the moment it is made, with its
  time, and checked against the record's own time. This is the leakage rule (CLAUDE.md rule 1) applied to our own data.
- **The label says who knows.** Every label carries its source: a person, a measurement, a benchmark, a teacher. The
  sets weigh them by source and never mix a model's opinion with a measured truth.

## Part A: the store and its keepers

### 32.1 The layout, the contracts, the capture rules and the store

- The shapes above in `dclab_core/store.py`, with their schemas and baselines.
- `dclab_core/store_rules.py`: a `CaptureRule` per task, and `check(record, rule)`, which returns what is missing, late
  (written after its decision) or unconfirmed, never a bare pass or fail.
- `dclab_rnd/store/` with `layout.py`, `store.py` and the CLI.
- **Done when:**
  - every path is built by `layout.py` (a test reads the package for a joined store path);
  - a kept source reads back byte for byte by its hash;
  - a second run keeps nothing new, and a changed source is a new version beside the old;
  - every contract is in `CONTRACTS` with its schema;
  - every task the studio runs has a capture rule (a test reads the harness's agents and the ledger's points);
  - a record missing a required field is kept and flagged with the field and the keeper; one whose state was written
    after its decision is flagged late;
  - no check ever removes a record from `raw/` (a test plants a failing record at every check and finds it kept);
  - `README.md` and `CATALOG.json` are generated and current.

### 32.2 The conductor

Runs the keepers, then the shapers, then the builders, as a worker job. Each run is a harness session
(`harness.session("conductor")`) and one record in `runs/`.

- **Done when:** a run records what each step read, wrote, skipped and why; two keepers on one path are refused; a
  failing keeper is recorded and the others still run.

### 32.3 Keepers for what is already recorded

Sessions, actions, decisions, conversations, model requests, the audit, test runs and gates, live-test records.

- **Done when:** for each kind, a scripted session's records appear in `raw/<kind>/` exactly once with their hashes; a
  workspace without consent keeps nothing but counts.

### 32.4 Keepers for what is lost today

- **Tasks:** every job's state change is kept as a version, and the Claude Code task lists are read from the transcripts.
- **Intern sessions:** each write is a new version.
- **Team and orchestra runs:** boards, interrupts, walks and checkpoints.
- **Teachers and students, all of it** (the owner, 2026-10-20: "store all teacher and student data"):
  - every teacher's answer, kept whole with its prompt, its thinking, its cost and the teacher's version (`raw/teachers/`);
  - every student version, kept whole: the fitted model file, the training set it was fitted on (record ids and hashes),
    its model card, every measurement, every shadow answer, and its routing level with its history (`raw/students/`);
  - judgments and own models alike.
  A student version is never overwritten; a refit is a new version beside the old.
- **Deletions and retention:** every delete (a project, a draft, a session) and the retention sweep run the keepers first,
  and the deletion keeper records each deletion.
- **Done when:**
  - a job run through three states has three versions;
  - an intern session written five times has five;
  - a team's walk is in the store with its board;
  - a project deleted after a run is still whole in the store, with its deletion recorded beside it;
  - an event past retention is still in the store.

### 32.5 Workflows as records

A `WorkflowRecord` for every walk from a draft to a report. It carries each step, its input card, its decision and its
gate, with links to the ledger's decisions and the agent steps inside it (on the existing `dclab_core.workflow_run`).

- **Done when:** a scripted walk yields one workflow record whose steps link to every decision and step it made.

### 32.6 Fixes, linked

The fix keeper joins a failing test run to the next passing run of the same test and the commits between them (the
diffs' file names and messages, never a secret). It proposes a draft development record for a person to accept.

- **Done when:** a planted failure fixed by one commit yields one proposed fix; nothing reaches
  `dev_trajectories/records/` without a person's yes.

### 32.7 Other agents

- **Development sessions:** every Claude Code transcript of this repository, worktrees included, kept whole under
  `raw/agents/claude_code/`.
- **Testers' Claude Code sessions:** kept under `raw/agents/testers/<id>/`, only with their yes.
- **MCP calls:** kept as their own source.
- **Done when:** a worktree's transcript is kept; a tester's goes only with their yes; an MCP call's request and result
  are in the store.

## Part B: records, the curators, and the eight sets

### 32.8 Shapers: raw into records

One shaper per kind turns raw entries into `StoreRecord`s, scrubbed, each with its provenance.

- **Done when:**
  - every raw entry has its records or a recorded reason why not;
  - no record holds a secret, a table's row or a person's contact (a test plants each);
  - the records rebuild the same from the same raw entries, byte for byte.

### 32.9 The curators: an agent per model type

The agentic system that puts the right data in the right place. It has a librarian and eight curators.

- **The librarian** reads each new record and sends it to the curators whose model it could teach. One record often
  goes to several: a team's walk goes to M2, M6 and M5.
- **Each curator (M1 to M8)** knows its model's needs: the record shapes, the folders, what a good example is and what
  is missing. For each record it decides:
  - whether it belongs to its model;
  - where it goes (the point, the system, the family, the stage or the problem kind);
  - its quality: keep, keep as silver with a weight, or keep out of this set with the reason (a duplicate, a broken
    record, a label no one confirmed, a near copy of a sealed case). A record kept out goes to `quarantine/`, never away.
  - It also asks for what its model lacks (the tutor and the replicas read these gaps).
- **One guarded write tool** (`curators/write.py`), registered once in the agents' registry under scope "store", is the
  only way a curator writes. It checks the path against `layout.py`, the example against its contract, the split
  against `splits.py`, and the consent. A refused write is recorded with why.
- **Rules first, Claude where it measures better.** Each curator runs on its rules by default, with no spend. Served by
  Claude, it follows *The agents that use Claude* above: guarded input, validated output, shadow first.
- **Curation is data too.** Every choice is a `CurationDecision`, in `raw/curation/` and in the ledger (point
  `store.curate`), so curation has its own student (M1) and its own workflow examples (M2, system `internal`).
- **Done when:**
  - a scripted day's records reach every model type they fit, in the right folders, and nothing else;
  - a planted duplicate, a broken record and a near copy of a sealed case are each kept out of the set with their
    reason, and are still in `raw/` and in `quarantine/`;
  - a Claude answer that names a path past the layout, or a number not in its record, is not written; the rules' answer
    is used, and both are kept;
  - a write past the layout or the contract is refused;
  - every choice is in the ledger;
  - a model-served curator never writes while in shadow.

### 32.10 The reasoning of every LLM example

Every M4 and M5 example carries its reasoning, from one of three sources, always labelled:

- **`recorded`:** the model's own thinking, kept whole where it was made. Claude's extended thinking through the gateway
  (the gateway keeps it beside the answer, never cut), the AI-first expert's thinking (25.3), and the development
  sessions' reasoning in their transcripts.
- **`from_record`:** a rule's or a person's choice, explained from its own record: the evidence it read, the options it
  weighed, the reason it gave, the check that passed. It is written by code, so it holds no claim beyond the record.
- **`teacher`:** written afterwards by a teacher model (Claude through the gateway, capped, planned and reported), only
  where no reasoning was kept. It is checked by `cited.py`: every number from the record, no claim past it. A rejected
  reasoning leaves the example without one, never with an unchecked one. Teacher reasoning is kept apart, with a lower
  weight, as for silver labels.

- **Done when:**
  - every LLM example has a reasoning or a recorded reason why not;
  - a reasoning with a number its record does not hold is rejected (planted);
  - the gateway keeps a model's thinking whole beside its answer;
  - the teacher spends only within a cap given on the command line.

### 32.11 The decision sets: M1, M2, M3, and the teacher–student pairs

From the ledger's records and labels:

- M2 is per workflow point and system: `dclab` (the product's points), `rnd` (the studio's walk, the teams, the
  conductor) and `internal` (the development and the automation points: a gate, a fix, a review, a job's next state).
- M3 is per technical point and task family.
- M1 is per student: a point of any model with its teacher's answers as the labels, Claude's and GLiNER's kept apart as
  silver with their weights.
- **Teacher–student pairs:** for every decision a student answered, its answer beside each teacher's and the label. They
  feed distillation, disagreement study and the next refit (`sets/m1_students/<point>/pairs/`).
- All three use the one split rule (`sets/splits.py`: by the group key's hash, the project or the table), with sealed
  partitions written once. These sets supersede the lake's per-point decision sets, which become a view of them.
- **Done when:**
  - every ledger point with labels has its folder under the right model type, with its manifest and card;
  - a point's sealed rows never appear in its train or dev;
  - a label source's weight is in every example;
  - the lake's sets agree with the store's on the same records.

### 32.12 The LLM sets: M4 and M5

- **M4:** chat examples by stage, from:
  - the Home agent's and the intern's conversations;
  - the model-written explanations and reports that passed `cited.py`;
  - the case trajectories (30.19);
  - the development records (`sft/out_dev`);
  - the code exports (problem → pipeline code).
  Preference pairs from the twin runs and from people's choices.
- **M5:** whole trajectories from the agents that solve (the intern, the teams, Claude as the AI-first expert, the
  research campaign), each with its problem, steps, submission and score. Problem → solution → code cases from the
  real problems, the replicas, OpenML and the business examples. Everything is filed by problem kind, and decontaminated
  against MLE-bench's tasks and every sealed set.
- Every example has its reasoning (32.10) in the format it trains in: a thinking block before the answer, and a thought
  before each tool call of a trajectory.
- **Done when:**
  - every example parses as its contract and renders in the chat format it trains in, its reasoning included;
  - no MLE-bench task or sealed person appears in a train split (a planted one is caught);
  - a trajectory's score is the problem's own metric, from its record.

### 32.13 The step-by-step, judge and retrieval sets: M6, M7, M8

**M6.** Episodes from every process that decides step by step: the walk's moves (the graph's transitions), a stage's
experiments, the intern's session, a team's walk, the conductor's own runs. Each step is (state, options, action,
outcome), and the episode ends with its final outcome (built, refused, the holdout's number).

**M7.** `JudgedExample` v1: a candidate (a plan, a solution, a report, a step, a piece of code) with its score, or a
pair with the preferred one. Each carries who judged it and how: a measured outcome (the holdout's number, a test that
passed, a build's verdict, 32.26), a person's choice, the critic's or the reviewer's verdict, or Claude's as a teacher,
kept apart. The verdicts come from:

- the twin's verdicts and the gates;
- the critic and the reviewer;
- people's choices and corrections (32.19).

**M8.** `RetrievalExample` v1: a query, the passages offered, the ones read and cited, and whether the answer that used
them passed. It is built from 32.23's search records and from the retrieval questions (60 and the practice set), with
hard negatives (offered, not used, and the answer failed).

- **Done when:**
  - a scripted walk yields one episode whose steps are its graph moves in order with their outcomes, and an episode's
    outcome comes from its record, never written by hand;
  - every judged example names its judge and its kind of evidence, with measured outcomes and people kept apart from a
    model's opinion;
  - every retrieval example has its passages and the answer's result, and no sealed question is in a train split.

## Part C: the students' agent

### 32.14 The tutor

A dedicated agent on the existing loop (`students_loop.py`). After each conductor run it:

- reads the new examples in `sets/`;
- refits the students with enough new rows;
- measures every one in shadow against the rule and the served answerer;
- demotes a routed student that slipped;
- builds a first student for each point with data but none (the owner's goal: a student for every small choice).

The tutor works for M1, M2, M3 and M6 alike. Promotion stays the owner's audited switch: the tutor proposes, with its
numbers.

- **Done when:** new labels for a point refit its student into a new version under `raw/students/` (the model, its
  training snapshot, its card, its measurement), the old version kept; a point with
  data and no student gets one in shadow; nothing is promoted by the tutor; every tutor run is a harness session in
  `runs/`.

## Part D: testers on their own branches

### 32.15 A branch per tester

- `make share` pushes the tester's store to `tester/<id>` in the records repository, never to another tester's branch.
- `make collect` fetches every `tester/*` branch and replays each into `raw/agents/testers/<id>/` and the tester's
  workspace, as today.
- The owner's own store goes to `owner/main`.
- **Done when:** two testers sharing at once land on their own branches with no conflict; a share cannot write another
  tester's branch; collect rebuilds both.

### 32.16 What a tester shares

The share includes the store's kinds the tester agreed to: orchestra, students, judgments and, with their yes, their
Claude Code sessions. A first-run notice lists every kind, and says that deleting a project in the app does not remove
what was kept or shared.

- **Done when:** a share holds every kind on the notice and nothing else; a kind turned off is absent.

### 32.17 Testing together

`docs/guides/TESTERS.md` is rewritten as a step-by-step guide, in English and Persian:

- clone, set up, solve your own problem, share, what is recorded, how to stop;
- that a deleted project stays in the store, and removal is asked of the owner;
- how the owner and the testers test together: a weekly problem, a shared checklist, bugs filed with the share's id;
- a page for the owner on the records repository and its branch rules.

- **Done when:** a fresh clone, followed only by the guide, shares to its branch (run in a scratch clone by the gate).

## Part E: closing

### 32.18 The phase's check

- `make store-check` runs a scripted day and captures it end to end through all three tiers, the curators and the
  reasoning, into all eight sets: a draft
  built, a stage run, an intern turn, a team walk, a test failing then passing, a job, a deletion, two testers sharing.
- It checks that every kind and model type is present, nothing is kept twice, nothing goes past consent, there is no
  secret, and sealed stays sealed.
- `make capture` runs the conductor before the lake and the archive.
- **Done when:** the check passes, and the catalog and schemas in the repository are current.

## Part F: more data worth keeping

Each is kept by its own keeper into `raw/`, shaped into `records/` and curated into the sets it fits.

### 32.19 Corrections and failures

- **What a person changed:** every edit to a card, a solution, a map, a report, a plan or a cell, as before and after
  with the reason when given. The strongest signal there is for M2, M4, M6 and M7: it says exactly where DCLab was wrong.
- **Failures:** refused moves, failed runs and errors with their fingerprints, and wrong answers with their corrections.
  They become counterexamples and preference pairs.
- **Done when:** an edit to a card yields a correction record and a preference pair; a refused move yields a failure
  record linked to the decision it refused.

### 32.20 The experiments themselves (meta-learning)

Every model fitted anywhere: the data card it was fitted on, the kind, the features chosen, the algorithm and its
settings, every fold's score, the holdout's, the time and the cost. Across hundreds of tables this says which approach
wins on which kind of data: knowledge for M3, M5 and the service.

- **Done when:** a stage run yields one experiment record per model fitted, keyed to its table's data card, with no row.

### 32.21 Code with its execution

Every piece of code written, run or exported, kept whole:

- the pipeline exports, the notebook cells, the intern's code;
- the development diffs;
- for each run, its output, its errors and the fix that followed.

M4 and M5 learn code from what ran and what failed, not from code alone.

- **Done when:** a cell that fails and is fixed yields one record with both versions, the error and the passing output.

### 32.22 Cost, time and context

Each step's seconds, tokens, euros and CPU, and the context it ran in:

- the code commit;
- the library versions;
- the model and its prompt's hash;
- the machine.

They feed the cost estimators, the service's prices, and the replay of any example.

- **Done when:** every record carries its context, and the catalog totals the cost per model type.

### 32.23 Search and reading

What each agent searched for, what it was offered, what it read, what it cited, and whether its answer passed. This is
M8's data, and it shows where the knowledge base is thin.

- **Done when:** an evidence search by a role yields a retrieval record with the offered and cited passages.

## Part G: for the service (building these models for other businesses)

The owner's aim: a service that helps a business build its own versions of these models. A business will ask:

- what can be built from its data;
- how many labels it needs;
- how long the build takes and what it costs;
- what will go wrong.

The answers come only from builds kept whole. The agents collect them now. Each business's data stays its own: what
crosses between businesses is aggregates, never a row or a name.

### 32.24 The build record

A `ModelBuild` v1 for every model built anywhere: our own students and decision models, a project's model, a tester's.
It records:

- **the business context, in bands:** the industry, the size, the problem kind;
- **the data:** its card (roles, rows band, label situation, problems found);
- **the model type (M1 to M8) and the recipe:** every choice with its reason;
- **what was spent:** the people's time, the labels asked, the compute, the euros;
- **the result:** its measured result against its bar;
- **what went wrong and what a person changed;**
- **after the build:** whether it was used, and what it did after.

- **Done when:** a built project, a refitted student and a tester's build each yield one build record with every part
  filled or marked unknown with why.

### 32.25 What it takes: learning curves and the service's catalog

- **Learning curves:** for each student and each kind, the measured score at each count of labels and rows (from the
  tutor's refits and the experiments). From them the service can tell a business "about this many labels for this score".
- **The catalog:** per model type and problem kind, what was built, how often it met its bar, its median cost and time,
  the common failures, and the data it needed. Read from the build records, never from one business alone.
- **Done when:** the catalog answers, for a kind, the labels needed for a score, with its interval, from at least the
  builds it counts; a kind with too few builds says so.

## Part H: the knowledge of building correctly

Records say what was done; the service must also know what is *right*. A record of a wrong build is a lesson only if it
is known to be wrong, and a practice is knowledge only with its evidence and the conditions it holds under. DCLab
already holds much of this, scattered:

- the rules (`CLAUDE.md`, the playbook);
- the evidence index and the field guide (`evidence/knowledge/`);
- the pitfalls (`pitfalls.py`, `pitfalls_by_kind.py`);
- the lessons (`lessons.py`);
- the certified recipes and modifiers (30.10 to 30.12, `certify.py`);
- the gates G1 to G4;
- the reviewer's findings in the development records.

Part H turns them into knowledge of building each model type correctly, checked and kept.

### 32.26 Every build judged correct or not

Each build record (32.24) gets a `BuildVerdict` v1, computed by code from the build's own records. A check that cannot
be read is "unknown, with why", never assumed passed. The checks:

- the prediction contract was written before any score;
- no column unknown at the prediction moment was used (the leakage audit);
- every fitting step stayed inside the training folds;
- the holdout was read once;
- the result beats its baselines with its interval;
- the export reproduces the number;
- the uncertainty and the limits are said;
- the bar agreed before building was met.

What follows from the verdict:

- correct builds are the gold examples of how to build;
- incorrect ones are counterexamples, each with exactly what was wrong;
- an M4, M5 or M7 example built from a build carries its verdict.

- **Done when:** a planted build that read its holdout twice, and one that used a leaking column, are each judged
  incorrect with the check that failed; a clean build is judged correct; a build missing its contract is "unknown".

### 32.27 The building knowledge base

A `KnowledgeItem` v1 per practice, rule or pitfall. Each item holds:

- the model type (M1 to M8) and the stage (problem, data, labels, split, features, model, evaluation, deployment,
  monitoring);
- the practice, in one sentence, and why it holds;
- when it holds: its scope, in data-card bands (rows, label situation, time, groups, rarity);
- its evidence: our campaigns by id with their numbers and intervals, or an external source with its citation;
- its counterexamples;
- its status: `measured here`, `from the literature` or `hypothesis`.

How items are made:

- A knowledge curator (one of the curators, 32.9) proposes items from the sources above, from the correct and incorrect
  builds, and from the experiments' meta-data (32.20). For example: "with fewer than 50 confirmed positives a ranking
  metric's interval spans …", measured.
- A person accepts each item.
- An item is never promoted past `hypothesis` without its evidence.

Its uses:

- the roles and the LLMs read it (through M8);
- it becomes training data for M4, M5 and M7, with its reasoning `from_record`;
- it is the service's own knowledge.

- **Done when:**
  - every rule of `CLAUDE.md` and every pitfall is an item with its evidence or marked `hypothesis`;
  - an item whose evidence a later campaign contradicts is flagged for review, never silently changed;
  - an item is never written without a person's yes.

### 32.28 Knowledge from outside, collected carefully

A research agent gathers how others build these models correctly. Its sources:

- papers;
- library documentation;
- public competition write-ups;
- MLE-bench's published analyses.

How it collects:

- It runs only with the owner's yes, within a cap, through the gateway.
- Each finding is kept as a summary in our own words with its citation, link and licence; the source's text is never
  copied whole.
- An external item stays `from the literature` until a DCLab experiment tests it. A test that confirms it makes it
  `measured here`; one that does not is kept as a counterexample.
- MLE-bench's tasks and their solutions stay out of every training split (32.12).

- **Done when:** a planted copied paragraph is refused; every external item has its citation; an external item tested
  here links to its campaign.

### 32.29 A building guide per model type

For each model type (M1 to M8) and each problem kind, a guide generated from the accepted knowledge and the judged
builds. It covers:

- what data and how many labels are needed (from the learning curves, 32.25);
- the steps in order, and the checks at each step;
- the common mistakes, with their real cases;
- the typical numbers, cost and time, with their intervals.

It is the service's playbook for a business, regenerated as the store grows, and an M4 source.

- **Done when:** each guide cites an accepted item or a build for every claim (checked by `cited.py`), and a kind with too
  few builds says so instead of guessing.

## Part I: the knowledge of giving the service

Building a model correctly (Part H) is half of a service. The other half is serving a business well:

- understanding what it really needs;
- agreeing what success means;
- explaining the numbers and their limits;
- delivering, and staying useful afterwards.

DCLab learns this from every engagement it has: the owner's, the testers', and later its customers'.

### 32.30 The engagement record

A `ServiceEngagement` v1 for every piece of work done for someone. It holds:

- **the request:** the person's own first words, redacted;
- **the conversation that clarified it:** the questions asked, the ones that changed the problem, and the ones that were
  wasted;
- **the agreement:** the success number agreed before building (the brief's bar), and the brief they approved;
- **the expectation, against what was delivered:** time, cost and result;
- **the explanation:** what was said about the result and its limits, what the person asked next, and what they did not
  understand;
- **the handover:** the code, the report, the service, and whether it was used;
- **what the person said:** feedback, corrections, complaints, a request for more, a stop.

How it is kept:

- It is filed by problem kind and business context, in bands.
- Its parts link to the build record (32.24) and its verdict (32.26).
- **Done when:** a tester's project yields one engagement record whose every part is filled or marked unknown, with the
  person's words redacted; a stopped engagement records where it stopped and why.

### 32.31 The service's playbook

From the engagement records, and only from them and the judged builds, `SERVICE_PLAYBOOK` items. Each holds:

- **the evidence it rests on:** the engagements and builds counted, never one business's figures alone;
- **the questions that most often changed the problem,** by kind;
- **the expectations businesses get wrong** (labels they think they have, accuracy they expect, a moment they forget);
- **the explanations that worked,** and those that confused;
- **the failure points:** where engagements stall, and what moved them on;
- **the typical time and cost of each step,** with intervals;
- **what businesses ask for after the first model.**

How it is used:

- It is accepted item by item by the owner, as the knowledge base is (32.27).
- It feeds M4 (understanding a business's problem and talking about results), M2 (the service's own workflow) and M7
  (what a good answer to a business looks like).
- **Done when:** each playbook item cites the engagements it rests on, and a kind with too few engagements says so.

## Part J: people (added 2026-10-20)

### 32.32 Each person's account, sessions and work

The owner, 2026-10-20: store each user, the sessions of their account, and everything they do (actions, decisions,
logs), for training and fine-tuning. With accounts on (`DCLAB_AUTH`), every event already carries `ids.user`; the store
keeps no account table today (`keepers/database.NEVER`).

- **A people keeper.** Per workspace (on PostgreSQL), it keeps each person and each of their sessions:
  - **a person:** the user id (stable, already a pseudonym), their role per workspace, how they sign in, when they
    joined, and whether they are active;
  - **a session:** its start, last-seen and end times and its workspace.
  - **Never kept:**
    - a password or its hash;
    - a session or API token, or a token's digest;
    - an email (only its hash, to recognise the same person);
    - a raw address.
  - The store is pushed to GitHub.
- **Every record says whose it is.** The shapers lift `ids.user` onto every record (`body.user`, `provenance.by`). That
  covers:
  - visits, page views and clicks;
  - chat turns;
  - decisions and corrections;
  - graph moves;
  - stage runs;
  - intern and agent steps;
  - model requests and their cost;
  - feedback, uploads and errors;
  - logs.
- **A per-person record.** `service/people.json`: each person's sessions, actions, decisions and corrections, projects,
  spend, and where they stalled. Aggregates only across people, as the catalog.
- **Splits.** Training sets keep splitting by project, so one person's work never crosses train and sealed.
- **Consent.** A workspace's consent governs, as for every record. A friend given their own workspace (Admin, Members,
  "their own workspace") is governed by their own consent.
- **Done when:**
  - a signed-in person's session and actions are kept and joined to them;
  - no password, token, digest or email is in `raw/` or `records/` (the store check gains these plants);
  - a person removed from the product stays in the store until the owner's audited forget.

## The owner's answers (2026-10-20)

- **The store's bulk:** a separate private GitHub repository, `dclab-records`, with one branch per person (`owner/main`,
  `tester/<id>`) and Git LFS for large files. The R&D repository keeps the catalog and the schemas.
- **The repository:** created private by Claude with `gh`; `DCLAB_TESTER_RECORDS` points at it.
- **Claude's budget for this phase:** up to $15, and more from the remaining phases' budget if needed. Every run is
  capped and reported.
- **Backfill:** everything that already exists, compressed (every workspace, the live tests, the test runs, the Claude
  Code transcripts with their worktrees). It stops with a report if free disk falls below 2 GB.

## Decisions taken by default (the owner may change them)

| # | Decision | Default and why |
|---|---|---|
| S1 | Where the bulk goes | The private records repository, one branch per person. The R&D repository keeps the catalog and the schemas, so clones stay small. |
| S2 | Raw transcripts in git | Compressed, on the owner's branch of the records repository; never in the R&D repository. |
| S3 | Deletion and retention against "keep everything" | The owner's choice (2026-10-20): the store keeps everything it kept when a project, draft or session is deleted or retention runs. Only the owner's audited `store forget` removes from it. |
| S4 | The tutor's power | It builds, refits, measures and demotes; it never promotes. |
| S11 | M7 and M8 | Accepted by the owner (2026-10-20): full model types, with their folders, records, curators and sets like the others. |
| S13 | What "correct" means | DCLab's own rules and gates, checked by code on the build's records: the leakage contract, folds, the holdout once, baselines with intervals, reproduction, the bar. A person's judgment adds to it, never replaces it. |
| S14 | Outside knowledge | Summaries in our own words, with citations and licences; never a source's text copied whole; untrusted until tested here. |
| S15 | Guardrails against "capture everything" | A check never deletes. Capture keeps everything; a failed input or output check only routes the record to `quarantine/` with why. The one removal is a secret's value, replaced by a marker. |
| S16 | Claude in the curators | Sonnet 5.5, through the gateway, with typed tools, scrubbed input, validated output, kept thinking, a cap, and shadow first. The rules are the floor. |
| S12 | Data between businesses | A business's records stay in its workspace's store. The service's catalog reads aggregates across builds (bands, counts, scores), never a row, a name or one business's figures alone. |
| S10 | Teachers and students | Everything kept, every version: each teacher's answers whole, and each student's fitted model, training snapshot, card, measurements and shadow answers. Fitted files are not committed to the R&D repository; they go to the records repository, or to its large-file storage when big. |
| S5 | Testers' Claude Code sessions | Off unless the tester says yes on the notice. |
| S6 | The lake and the store | The store's `sets/` become the one place training sets are built; the lake's gold sets become a view of them, and its bronze inputs a source of `raw/`. |
| S7 | File format | JSON Lines, gzip in `raw/` and `records/`. In `sets/`, plain `.jsonl` (parquet beside it for the decision sets), with a schema version on every line. |
| S8 | Reasoning sources | Recorded thinking first. Then reasoning from the record, written by code. Then a teacher's, only where neither exists, checked and weighted lower. Never an unlabelled one. |
| S9 | Curators served by a model | Off until a curator measures better than its rules on a person's review of a sample; shadow first, by the audited switch. |

## Order

32.1 → 32.2 → 32.3 → 32.4 → 32.5 → 32.6 → 32.7 → 32.8 → 32.9 → 32.10 → 32.11 → 32.12 → 32.13 → 32.14 → 32.15 → 32.16 →
32.17 → 32.18, then 32.19 → 32.20 → 32.21 → 32.22 → 32.23 → 32.24 → 32.25, then 32.26 → 32.27 → 32.28 → 32.29, then 32.30 → 32.31 (the
check, 32.18, is extended by each).

Every package follows the usual steps: a test first, the reviewer, its development record and `make sft-dev`, and the
gate. Model spend:

- none for the keepers, the shapers, the rules' curators and the tutor (small classifiers, trained locally);
- the teacher's reasoning (32.10) and a model-served curator's shadow run, planned and capped on the command line, and
  reported.


## Status

- **32.1 to 32.6 built (2026-10-20), two reviews' 29 findings fixed (dclab-core 0.98.0).**
  - **The store:**
    - three tiers, every path through `layout.py`;
    - the contracts of eight model types, with labelled reasoning and states that never hold a value;
    - a capture rule per task that flags and never deletes;
    - an append-only core with versions, hashes, tails cut at a line break, and an index cache;
    - its root a clone of the private `dclab-records` on `owner/main`.
  - **The conductor:** a worker job at most once an hour and a harness session.
  - **One inventory per run:** a dry run over the development workspaces placed 15,024 files from 800 workspaces in 4.8 s,
    none twice, nothing left out, 3,673 tables counted only.
  - **Never kept as content:**
    - cell-quoting keys;
    - notebook outputs;
    - tables anywhere;
    - secret file names.
  - **PostgreSQL:** workspaces exported table by table.
  - **Kept first:**
    - every job state and every intern save;
    - the teams' boards, students and judgments;
    - a deleted project, draft or session, with its deletion recorded;
    - events past retention.
  - **Walks:** each walk is one `WorkflowRecord`.
  - **Fixes:** failing tests are linked to their fixes (33 proposed, 21 open in two days of runs).
  - **Left open:**
    - the first push of the store's bulk (about 500 MB, compressed) to the records repository, after Git LFS is installed;
    - the keepers named in the rules for 32.7 and 32.9.
- **32.7 to 32.11 built (2026-10-20), a review's 17 findings fixed.**
  - **Other agents:** every Claude Code session on this repository is kept, the main checkout and every worktree.
    - Credentials of every common shape are removed at capture.
    - A tester's machine reads none of its own sessions unless they are named explicitly.
    - Screenshots are counted only.
    - Each MCP call is kept whole, whether it worked or not.
  - **Shapers:** each entry is shaped once, streamed into chunks under 20 MB.
    - A tool's printed table and every contact in a transcript's words are removed; ids, hashes and times are left as they are.
    - Database rows are read too.
    - Labels are sealed with their decisions, and drafts with the projects they became.
    - A rebuild is made beside the old records and swapped in.
  - **Curators:**
    - a rule curator per model type;
    - a guarded write;
    - Claude in shadow only, shown each record's shape and never its content.
  - **Reasoning:**
    - the gateway keeps each exchange whole, thinking included, where the workspace keeps model text;
    - reasoning can be written from the record;
    - a teacher's reasoning is checked against its record.
  - **Decision sets:** M1, M2 and M3 by point, system and family.
    - The best label wins; a teacher's weighs 0.5.
    - The sealed split is written once.
    - Teacher-student pairs are kept.
  - **Left open:** the lake's per-point sets as a view of these; live runs of the curator and the teacher (within the budget).
- **32.12 to 32.23 built (2026-10-20), a review's 20 findings fixed.**
  - **The LLM sets.** M4 by stage, from every exchange kept whole and every Home conversation. M5 from every Claude Code
    and intern session, streamed one transcript at a time. Every turn and step carries its labelled reasoning. Sealed
    people and MLE-bench Lite tasks are kept out of train and dev (one compiled pattern, matched on the texts).
  - **M6, M7 and M8.**
    - M6: walks, moves and jobs as episodes.
    - M7: corrections, fixes, G4 and refused moves as judgments, and corrections as preference pairs.
    - M8: searches with what they offered and what the answer cited.
  - **The tutor.** It fits a student on confirmed labels and measures it on confirmed dev labels beside the rule. Every
    version is kept whole and listed in shadow; it never promotes.
  - **Testers.** Each has their own orphan branch, and a share pushes only there. Collect reads every tester branch in a
    worktree of its own. The share carries the store's kinds, and the tester's own Claude Code sessions only with their
    yes. The guide is in English and Persian.
  - **The check.** `make store-check` runs one scripted day and passes 26 of 26. It found that a key in a plain log was
    kept, so every keeper now removes credentials.
  - **Corrections, experiments, code, context, searches.**
    - Corrections are kept whole as before and after.
    - Every fitted model is an experiment record against its table's shape.
    - Code runs are paired with their fixes; notebooks keep their errors; every commit is kept as its patch.
    - Each run's context (commit, libraries, a hash of the machine) is carried on its records, and the spend is totalled.
    - Every search is kept with what it offered and what was cited.
  - **The review's leakage fixes.**
    - Once sealed, always sealed.
    - One group for every record of a project, across all eight model types.
    - The tutor folds on the true table group (`DecisionExample` v2).
    - Consent and the curators' refusals hold in every builder.
  - **Left open:**
    - the experiments of the kinds with their own machine (their candidates are in their own shapes);
    - CPU time per step;
    - a workspace reading the store's shadow students;
    - live runs of the curator and the teacher.
- **32.24 to 32.31 built (2026-10-20), a review's 17 findings fixed (dclab-core 0.99.0).**
  - **Builds.** Every project the store kept is a `ModelBuild`: file or PostgreSQL workspaces, a deleted project, a
    tester's project. So is every student the tutor refitted.
    - Each build holds its context and data in bands, its recipe with reasons, people's active time (gaps capped at 30
      minutes), its spend, its result against its signed bar, what went wrong and what a person changed.
    - An outcome counts as use only when it was observed in the business or in a promotion.
    - Only each source's current version is read, so a rewritten log or a walk kept again is never counted twice.
    - Builds are keyed by workspace and project.
    - Only records whose workspace allows improving DCLab are read.
  - **Verdicts.** Code judges each build on eight checks, from the engine's real records:
    - a declared leakage column used as a feature;
    - a holdout read twice;
    - an interval that does not clear every baseline on the metric's own side;
    - the code export's rerun.
    - A fact its records cannot say stays unknown; a missing contract is unknown, not incorrect.
  - **Curves and the catalog.**
    - Curves per student point, and per problem kind by rows band.
    - The catalog of kinds gives each met share with a Wilson interval, and gives a figure only from three builds in
      three workspaces.
  - **Knowledge.**
    - Items are proposed from the evidence index, CLAUDE.md's rules and the outside findings.
    - An item is `measured_here` only with a campaign behind it, and is accepted only by a person.
    - An accepted item whose source changed is flagged.
    - An outside finding is refused when it copies its source, even in a short sentence; `tested()` links it to its
      campaign or records the counterexample.
    - The guides keep only cited sentences.
  - **The service.**
    - Each walk's newest version is an engagement; what its records cannot say is listed.
    - The playbook gives a figure only from three engagements in three workspaces.
    - All of it is rebuilt into `service/` once per whole run (`python -m dclab_rnd.store
      service|knowledge|know-accept|guides`).
  - **Left open:**
    - the research agent that finds outside findings (needs the owner's yes and a cap);
  - **The backfill (2026-10-20).**
    - **What it kept:** every workspace (800), Claude Code session, test run and commit, in 21 minutes with no error:
      about 15,500 sources, 771 MB compressed (6 GB of disk left).
    - **What came out of it:** 476,539 records; 384,191 curators' choices; the sets of every model type; 874 walks.
    - **What the real records showed:**
      - the engine writes the holdout interval as a dict, so it is now read that way;
      - walks need a server, so without one they are now read from each workspace's folder, with nothing written there.
    - **Verdicts on 715 real builds:**
      - 116 are incorrect, every one because its interval does not clear the best baseline (a forecast against seasonal
        naive, uplift against the response model, anomaly recall against its own change);
      - 599 are unknown: no build has a signed bar or a checked export in these development workspaces.
    - **Engagements:** 159 of 874 stopped in the draft.
    - **The tutor refitted no student:** no point's dev split has 5 confirmed labels yet (the rule's labels are not confirmed).
  - **The model curator, live in shadow (2026-10-20, EUR 0.26 of the $15).**
    - **Result:** on 40 records the rules placed, it named one of the rules' model types on 26 of 34 answers, agreed on
      quality 2 times in 26, left out 21 of the rules' 47 placements, and gave no answer for 6 records. It stays in
      shadow.
    - **Next:** a person reviews the disagreements.
    - **Fixed on the way:** a command-line tool's requests now reach the usage log (`models.for_tools`), and a batch
      has room for its whole reply.
    - **The teacher had nothing to do:** no whole exchange without thinking is kept yet. `DCLAB_LOG_PROMPTS=1` keeps
      them from now on.
  - **Labelled rows counted apart (32.25):** a build's labelled rows are its final record's training and holdout rows. The
    kind's curve is drawn by them, and the catalog gives their bands where the bar was met.
  - **Stage and team-walk episodes (32.13):**
    - **New episodes:** a project's stages in the order they ran, and a team's walk node by node.
    - **Read from current records:** M6, M7 and M8 now use them, per workspace (a walk kept twice is one episode, and
      PostgreSQL moves are read).
  - **Engagements read whole (32.30):** the person's first words, the questions asked and answered, the brief, the final
    record's notes, limits and interval, and the feedback given.
  - **The research campaigns' experiments (32.20):** the 14 model-selection results on public datasets are M3 experiment
    records, split by dataset. The engine's experiments are now read per workspace and project, train consent only.
  - **The store in its repository (2026-10-20).**
    - **In Git LFS:** every kept source (`raw/**/*.gz`) and each set's sealed split.
    - **Plain files:** the indexes, the catalog and the schemas, the service records, the runs, and each set's manifest
      and card.
    - **Kept out:** `records/`, the train, dev and pending splits, and the curators' running state. They are rebuilt
      from `raw/` on every run, so pushing them would fill the LFS quota with copies.
  - **The MLE-bench list checked** against openai/mle-bench: the Lite split matched (22 of 22), and the rest of its 75
    competitions and its development task were added (76 names).

---

## خلاصهٔ فارسی

این مرحله پیش از فاز ۹ ساخته می‌شود و هدفش ثبت همه‌چیز است، بی‌آنکه کسی متوجه شود. داده‌ها مرتب و ساختاریافته ذخیره می‌شوند
تا سال‌ها بعد هم برای آموزش به کار بیایند.

**هشت نوع مدل.** هر کدام پوشهٔ خودش را دارد و شکل دادهٔ خودش را:

۱. مدل‌های دانش‌آموز
۲. مدل‌های تصمیم گردش‌کار برای DCLab، سامانهٔ R&D و خودکارسازی داخلی
۳. مدل‌های تصمیم علم داده و یادگیری ماشین
۴. یک مدل زبانی که گردش‌کار یادگیری ماشین را از فهم مسئله تا راه‌حل و کد می‌داند
۵. یک مدل زبانی که مسئله‌ها، راه‌حل‌ها و کدهای فراوان یادگیری ماشین را می‌داند تا در MLE-bench اول شود
۶. مدل تصمیم گام‌به‌گام
۷. مدل داور (پاداش)
۸. مدل بازیابی و رتبه‌بندی

**سه لایهٔ ذخیره:**

- `raw/`: نسخهٔ کامل و دست‌نخورده
- `records/`: رکوردهای یکدست، یک شکل برای هر نوع داده
- `sets/`: مجموعه‌های آموزشی، یک پوشه برای هر نوع مدل

هر رکورد منشأ خودش را دارد و هر مجموعه فهرست و کارت خودش را.

**ساختار کد.** بستهٔ `dclab_rnd/store/` هم مرتب است: یک ماژول برای هر نوع داده، هر لایه و هر نوع مدل.

**ارکستر ثبت.** برای هر نوع داده یک عامل نگهبان هست. یک رهبر نگهبان‌ها را اجرا می‌کند و هر اجرا را ثبت می‌کند.

**حذف.** حذف یک پروژه، پیش‌نویس یا جلسه، یا پاک‌سازی خودکار، چیزی را از انبار پاک نمی‌کند. فقط فرمان خودِ مالک
(`store forget`) چیزی را برمی‌دارد، و آن هم ثبت می‌شود. هیچ کلید یا رمزی ذخیره نمی‌شود.

**سامانهٔ عامل‌محور.** برای هر نوع مدل یک عامل گزینشگر هست و یک کتابدار رکوردها را به آن‌ها می‌سپارد. هر گزینشگر
تصمیم می‌گیرد کدام داده برای مدلش مناسب است، کجا ذخیره شود و کیفیتش چقدر است. هر نوشتن از یک ابزار کنترل‌شده می‌گذرد
که مسیر، قرارداد، تقسیم‌بندی و رضایت را بررسی می‌کند، و خودِ هر انتخاب هم ثبت می‌شود.

**استدلال.** همهٔ نمونه‌های مدل‌های زبانی (نوع ۴ و ۵) پیش از پاسخ، بخش استدلال و تفکر دارند. منشأ هر استدلال برچسب
می‌خورد:

- تفکری که خودِ مدل در لحظه کرده و کامل ثبت شده؛
- توضیحی که کد از روی خودِ رکورد می‌نویسد؛
- یا استدلالی که بعداً یک مدل معلم می‌نویسد. این آخری بررسی می‌شود و وزن کمتری می‌گیرد.

**معلم‌ها و دانش‌آموزها.** همهٔ داده‌هایشان نگه داشته می‌شود:

- هر پاسخ هر معلم (Claude، GLiNER، قاعده‌ها، آدم‌ها و هر مدل دیگر)، کامل، با پرسش، تفکر و هزینه‌اش؛
- هر نسخهٔ هر دانش‌آموز: مدل آموزش‌دیده، دادهٔ آموزشی‌اش، کارت مدل، همهٔ سنجش‌ها و پاسخ‌های سایه‌اش.

هیچ نسخه‌ای بازنویسی نمی‌شود.

**دادهٔ بیشتر:**

- اصلاح‌هایی که آدم‌ها انجام می‌دهند، و شکست‌ها؛
- خودِ آزمایش‌ها: کدام روش روی کدام نوع داده برنده است؛
- کد همراه با اجرا، خطا و اصلاحش؛
- هزینه، زمان و زمینهٔ هر گام؛
- جست‌وجوها: چه پیدا شد و چه به کار رفت.

**دو مدل پیشنهادی دیگر:**

- مدل داور (پاداش)، برای اینکه مدل‌های ۴ تا ۶ از تلاش‌های خودشان یاد بگیرند؛
- مدل بازیابی و رتبه‌بندی.

ساختن یا نساختنشان تصمیم شماست.

**هشت نوع مدل.** دو مدلی که پذیرفتید هم اضافه شدند:

- مدل داور (۷)، که از نتیجه‌های سنجیده، انتخاب آدم‌ها و داوری‌ها می‌آموزد؛
- مدل بازیابی (۸)، که از جست‌وجوها، آنچه پیشنهاد شد و آنچه به کار رفت می‌آموزد.

**قاعده، نگهبان و اعتبارسنجی:**

- **قاعدهٔ ثبت:** هر کار یک قاعدهٔ ثبت دارد: چه چیزی باید ثبت شود، در چه لحظه‌ای، برای کدام مدل، چه چیزی آن را درست
  می‌کند و چه چیزی مُهروموم است.
- **ورودی و خروجی:** ورودی و خروجی هر عامل، از جمله عامل‌هایی که با Claude Sonnet 5.5 کار می‌کنند، بررسی می‌شود.
- **اصل اصلی:** اعتبارسنجی برچسب می‌زند و مسیر را تعیین می‌کند، اما هرگز داده را پاک نمی‌کند. همه‌چیز کامل در `raw/`
  می‌ماند. داده‌ای که از بررسی رد شود فقط با دلیلش به `quarantine/` می‌رود، و خودِ خطا هم داده‌ای برای آموزش است.
- **تنها استثنا:** تنها چیزی که برداشته می‌شود مقدار یک رمز یا کلید است، که با یک نشانه جایگزین می‌شود.

**دادهٔ درست برای هر مدل.** برای هر مدل مشخص است:

- هر نمونه چه چیزی باید داشته باشد؛
- چه وقت درست است؛
- چه وقت نادرست است، که آن وقت با دلیلش کنار گذاشته می‌شود؛
- از کجا می‌آید.

دو قاعده برای همهٔ مدل‌ها مشترک است:

- وضعیتِ هر تصمیم همان چیزی است که در همان لحظه معلوم بود؛
- هر برچسب می‌گوید چه کسی آن را می‌داند: یک آدم، یک سنجش، یک محک یا یک معلم.

**دانش ارائهٔ خدمت:**

- **پروندهٔ همکاری:** برای هر کاری که برای کسی انجام می‌شود یک پروندهٔ همکاری ثبت می‌شود: درخواست، گفت‌وگو، توافق بر
  معیار موفقیت، انتظار در برابر نتیجه، توضیح، تحویل و بازخورد.
- **دفترچهٔ راهنمای خدمت:** از این پرونده‌ها ساخته می‌شود: کدام پرسش‌ها مسئله را عوض کردند، کسب‌وکارها کجا اشتباه
  انتظار دارند، کدام توضیح‌ها کار کردند و کار کجا متوقف می‌شود.

**دانش ساختِ درست:**

- **داوری هر ساخت:** کد هر ساخت را می‌سنجد و درست یا نادرست اعلام می‌کند: قرارداد پیش از امتیاز نوشته شده، نشت داده
  نبوده، همه‌چیز داخل foldها برازش شده، داده‌های آزمون فقط یک بار خوانده شده، و از مبنا بهتر است. ساخت‌های درست نمونهٔ
  طلایی‌اند و نادرست‌ها نمونهٔ ضد، با خطای دقیقشان.
- **پایگاه دانش:** هر قاعده و دام با شاهدش، شرایطی که در آن درست است، نمونه‌های نقضش و وضعیتش نگه داشته می‌شود: «سنجیده‌شده
  در DCLab»، «از منابع» یا «فرضیه». هر مورد با تأیید یک آدم وارد می‌شود.
- **دانش بیرونی:** با اجازهٔ شما و سقف هزینه، خلاصه به زبان خودمان و با ارجاع ثبت می‌شود و تا وقتی در DCLab آزموده نشود
  قابل اعتماد شمرده نمی‌شود.
- **راهنمای ساخت:** برای هر نوع مدل راهنمایی ساخته می‌شود: چه داده‌ای، چند برچسب، چه گام‌هایی، چه خطاهای رایجی.

**برای سرویس آینده:** برای هر مدلی که ساخته می‌شود یک «پروندهٔ ساخت» نگه داشته می‌شود: زمینهٔ کسب‌وکار، داده، دستورالعمل،
هزینه، زمان، نتیجه و خطاها. از این پرونده‌ها منحنی‌های یادگیری و فهرست خدمات ساخته می‌شود، تا بتوانیم به هر کسب‌وکار
بگوییم چه چیزی با چند برچسب و چه هزینه‌ای ساختنی است. دادهٔ هر کسب‌وکار مال خودش می‌ماند و فقط آمار کلی بین آن‌ها
جابه‌جا می‌شود.

**عامل معلم.** دانش‌آموزها را در حالت سایه مدام دوباره آموزش می‌دهد و می‌سنجد، اما هرگز خودش چیزی را ارتقا نمی‌دهد.

**آزمایش‌کنندگان.** هر دوست شاخهٔ خودش را دارد و راهنمای گام‌به‌گام به انگلیسی و فارسی آماده می‌شود.
