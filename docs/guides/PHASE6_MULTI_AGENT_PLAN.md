# Phase 6: a multi-agent system for understanding and solving ML problems, on an orchestration layer

Written 2026-10-17 from the owner's request for the next phase, over the repository at `45bc427` (phases 1 to 5 built;
the GLiNER capture finishing in the background). Approved by the owner on 2026-10-17 (Part F). **This phase is the
infrastructure of the phase after it.**

The owner's words, in short: *build a framework with multiple agents for problem understanding and multiple agents for
the solution; not from scratch, on the best existing framework (LangGraph is one example); help me choose the best
framework and the best architecture, where these agents work under an orchestration layer to understand the problem
and the data and solve the ML problem together.*

---

## Part A. What exists and what is missing

DCLab already has most of the *agents*. What it lacks is the *orchestration* that lets them work as a team.

| Need | Today | Missing |
|---|---|---|
| Agents that understand a problem | the problem analyst (18.2), the conversation reader, the data investigator (18.3), the decomposer, the map reviewer, the data-need agent (24.1 to 24.2), the Home agent (A3, 25.13) | they run one at a time, called by hand-written code; none sees another's doubt; no second opinion on the card before the person confirms it |
| Agents that solve | the solution router (24.3), the solution planner and the expert design (25.3), the critic rounds, the leakage reviewer, the reporter, the intern | the same: a fixed order, no parallel work on the parts of a complex problem, no auditor that can send a step back |
| A control plan | `dclab_core.coordinator` (problem → map → data → links → meet → analysis → workflow → build), `Situation`, `gate()`; `draft/coordinator.py` walks it | the walk is a loop in Python that calls one action per state; it cannot pause for a person inside a step, resume after a crash, or run two things at once |
| A graph framework | LangGraph 1.2 is in the agent environment and runs the research campaign loop (`agentic/engine.py`: a `StateGraph`, SQLite checkpoints) | it is used for one loop only; the product's agents do not run on it |
| One harness for every agent | `agents/harness.py` (`AGENTS`, `ROLES`, `MODEL_USES`), every step traced, every tool through the registry and the validator | the harness knows agents, not teams: no record of who handed what to whom |
| Model routing | the gateway, a model per role, shadows, 27.8's tuned models | per role, not per node of a graph; no way to run two models on one node and keep both answers |
| Measurement | the journeys suite (dev and sealed), the twin matrix (25.1), the benchmark, harness-eval | no arm for a team of agents; no measure of what a second agent adds beyond its cost |
| Training data | the lake, the ledger, self-play, the experience log | the hand-offs between agents are not recorded as a trajectory a solver model could learn from |

The conclusion of Part A, and the owner's rule for the phase (2026-10-17): **enhance, never replace.** The orchestration
layer is the phase; the agents, the coordinator, the router, the validator and the engine exist and stay. The phase wraps
them as nodes, adds the few nodes that are missing (an understanding critic, an auditor with the power to send work
back, a composer for parallel parts), and measures what the team adds. Nothing in the understanding path or the
solution path is rebuilt beside what is there: a node that would duplicate existing code is a defect of the design.

---

## Part B. Principles (the repository's rules, applied to a team of agents)

0. **Enhance, never replace (the owner's rule).** Every node names the existing function it calls (D2 and D3 list them);
   the coordinator's walk stays as the path with no framework and no model, serving whenever the team is off, over
   budget or failing; the rules keep serving until the team passes its gate. The phase adds a layer over the system,
   not a second system beside it. The measurement arms (Part E, 28.7) compare the same system with the layer on and
   off; they are not two implementations.
1. **Deterministic code owns the control flow.** The graph's transitions are the coordinator's state machine and the
   graph validator; an agent proposes, a gate decides. No agent chooses the next state (CLAUDE.md rule 5).
2. **The framework is a dependency, not the architecture.** Contracts (the board, the hand-off, the team's record) live
   in `dclab_core` as versioned pydantic models with no framework import; the wiring lives in `dclab_rnd/orchestra/`.
   Swapping the framework is a change in one folder.
3. **One model door.** Every agent's request goes through `dclab_rnd.models` (the gateway): Claude Sonnet 5.5 today, a
   tuned Qwen tomorrow (27.8), a scripted transport in tests. The framework never holds an API key or a model client.
4. **One harness.** Every node is a `harness.session` or `runtime.run`; `tests/test_agent_harness.py` keeps failing on an
   unrecorded step. Every tool comes from `default_registry()`; a write tool declares its move.
5. **Every decision in the ledger, every hand-off in the experience log**, under the workspace's consent (19.10). The
   team's work is training data for Model 1 (the solver LLM) from the first run.
6. **Measured before trusted.** A team arm runs beside the rules arm and the single-agent arm on the same journeys; the
   gate is fixed before the run; the sealed set is read once; every live run is capped (the owner's budget rules).
7. **A person is a node, not an interruption.** Questions and approvals are typed interrupts of the graph, resumed from
   a checkpoint; the Home page and the coordinator's `person` are the same interface.
8. **Parallel where the problem is parallel.** The parts of a complex problem (24.x) are solved at once and composed
   once; nothing else runs in parallel just because it can (cost and the 8 GB Mac).

---

## Part C. The framework: comparison and the decision

Criteria, in the order they matter for DCLab: (1) model-agnostic through our gateway (a must: Claude now, our own
models later, scripted in tests); (2) explicit graph with typed state, so deterministic code owns the control flow;
(3) durable checkpoints and typed interrupts for a person; (4) parallel branches with a join; (5) tracing hooks that
our harness can own (no mandatory cloud); (6) Python, a permissive licence, maturity; (7) fit with the code we have.

| Framework | Model-agnostic | Explicit graph + typed state | Checkpoints + interrupts | Parallel + join | Tracing we own | Licence, maturity | Fit with DCLab |
|---|---|---|---|---|---|---|---|
| **LangGraph** (LangChain) | yes: any callable; our gateway as a thin chat-model adapter | yes: `StateGraph`, typed state with reducers, subgraphs, conditional edges | yes: SQLite and PostgreSQL checkpointers, `interrupt()` and resume by thread | yes: fan-out with `Send`, join on the reducer | yes: callbacks are optional; LangSmith not needed | MIT; 1.x since 2025, 1.2 in our lock file | **already here**: the campaign loop runs on it; PostgreSQL is our store |
| PydanticAI (+ pydantic-graph) | yes | yes, typed on pydantic (as our contracts are); the graph part is younger | durable execution through Temporal or DBOS integrations, not built in | limited | yes | MIT; 1.x since 2025 | good types; a second dependency with less for orchestration |
| OpenAI Agents SDK | partly: other providers through LiteLLM; built around OpenAI's shapes | no: hand-offs between agents, the flow is in the agents | sessions, not graph checkpoints; no typed interrupt | hand-offs, not fan-out | its tracing is OpenAI's by default | MIT; 2025 | would put the control flow in the agents, against principle 1 |
| Claude Agent SDK (Anthropic) | **no**: Claude only | no: a tool loop, not a graph | sessions | no | Anthropic's | 2025 | excellent for coding agents; it would lock DCLab's own models out |
| Microsoft Agent Framework (AutoGen + Semantic Kernel) | yes | yes: workflows as graphs, agents as nodes | checkpoints; human-in-the-loop as request/response | yes | OpenTelemetry | MIT; new in late 2025, moving | credible second choice; Azure-leaning docs; no code here uses it |
| CrewAI | yes | partly: crews and flows; the control is in roles and tasks | flows persist state | yes | its own | MIT; popular | opinionated roles; harder to keep the validator in charge |
| Google ADK | partly (LiteLLM) | yes: sequential, parallel, loop agents | sessions; events | yes | OpenTelemetry | Apache 2; 2025 | Google-leaning; no fit with our code |
| LlamaIndex Workflows | yes | event-driven steps | checkpoints | yes | its own | MIT | built for retrieval pipelines; smaller for agent teams |
| Temporal / DBOS (durable execution) | n/a | no: a substrate, not an agent framework | the best durability there is | yes | yes | open source | an option *under* a framework later, not instead of one |

**The decision: LangGraph, used thinly.**

- The graph, its typed state, its checkpoints (PostgreSQL in the workspace's database, SQLite in file workspaces), its
  `interrupt()` for the person and `Send` for the parts of a complex problem.
- **Not** LangChain's agents, tools or model classes: nodes are our functions, tools are our registry, the model is our
  gateway behind a thin adapter. We take the orchestration and leave the ecosystem.
- Why not the others, in one line each: PydanticAI's graph is younger and its durability is borrowed; the OpenAI and
  Claude SDKs put the control in the agents and (the Claude one) in one vendor; the Microsoft framework is the credible
  alternative but is newer, moving, and absent from our code; CrewAI's roles fight the validator.
- The exit is kept open: the board and the hand-off contracts are framework-free (principle 2); a spike on the Microsoft
  framework is a package the owner can ask for later (28.11, optional).

---

## Part D. The architecture

```
                     ┌──────────────────────────── the person (Home page, API, CLI) ────────────────────────────┐
                     │   questions ↑ answers ↓        gates ↑ approvals ↓          "build it"                   │
┌────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────┴───┐
│  ORCHESTRATION LAYER  (dclab_rnd/orchestra, on LangGraph)                                                       │
│  the coordinator's state machine as the graph's router · typed interrupts · checkpoints (PostgreSQL / SQLite)   │
│  budgets per run and per node · retries · model routing per node (gateway) · the harness on every node          │
│                                                                                                                 │
│   ┌──────────────── UNDERSTANDING TEAM (subgraph) ────────────────┐   ┌──────────── SOLUTION TEAM (subgraph) ───────────┐
│   │ interviewer · problem analyst · data investigator · linker   │   │ router · designer · builder · experimenter       │
│   │ decomposer + map reviewer + data-need · understanding critic │──▶│ auditor · composer · reporter                     │
│   │   → problem brief, data card, problem map (person confirms)  │   │   → solution map, workflow, built project, report │
│   └──────────────────────────────────────────────────────────────┘   └───────────────────────────────────────────────────┘
│                                   ▲ reads / writes                                    ▲ reads / writes               │
│   ┌──────────────────────────── THE BOARD (dclab_core.orchestra.Board v1, framework-free) ──────────────────────┐  │
│   │ brief · data card · map · routes · workflow · open questions · blockers · budget · hand-offs · decision ids │  │
│   └──────────────────────────────────────────────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
          │ every node                     │ every decision                 │ every hand-off               │ every stage
          ▼                                ▼                                ▼                              ▼
   agents/harness (traces)      ledger (DecisionRecord)         experience log (events)        engine + graph validator
          └───────────────── the lake: trajectories for Model 1, decisions for Model 2 (consent) ─────────────────────┘
```

### D1. The orchestration layer (what it owns, and nothing else)

- **Control flow:** the top-level graph has one node per coordinator state; its conditional edges call
  `coordinator.situation()` (deterministic) to choose the next state. An agent never returns "go to build"; it returns
  its artefact, and the situation decides.
- **Interrupts:** a question to the person, a gate (map confirmed, build approved, holdout read), a blocker only more
  data clears. Each is a typed `Interrupt` contract; the graph pauses at a checkpoint and resumes with the answer. The
  draft's chat and the coordinator's `person` object become two ways to answer the same interrupt.
- **Checkpoints:** LangGraph's PostgreSQL checkpointer in the workspace's database (file workspaces: SQLite in the
  workspace folder), keyed by the draft or project id as the thread. A crash, a restart, or a person who answers
  tomorrow resumes from the last node.
- **Budget:** a cap per run (the owner's rule: live runs capped, €0.5 by default) and per node; a node over its budget
  returns an abstention and the rules' path serves; the board records the spend.
- **Model routing per node:** each node names its role; the gateway's routing gives the role its model (Claude today,
  a tuned model in shadow then served, 27.8). A node can ask two answerers (a model and the rules) and keep both: that
  is how the twin matrix (25.1) is filled without a second system.
- **The harness on every node:** a node is a `harness.session` (or `runtime.run` for a model loop); `AGENTS` gains one
  entry per node; the team's hand-offs are events (`team.handoff`) with the board's version before and after.

### D2. The understanding team

Every node is existing code called from a graph node; the "What it adds" column is the team behaviour, not new logic for
the task itself. Only the understanding critic is a new agent, and it reads and asks; it never writes the brief.

| Node | Exists as | What it adds in the team |
|---|---|---|
| Interviewer | the Home agent's questions (`dclab_core.conversation`: the next question by what it unblocks) | asks only what no other node could settle; one question at a time, the person's words kept |
| Problem analyst | `agent_eval/analyst`, the problem card | reads the first sentence and every answer into the brief |
| Data investigator | 18.3, the data card (19.11) | runs **in parallel** with the analyst once a table is there |
| Linker | `dclab_core.meet.check`, label joins, related tables | every mismatch between the problem and its tables becomes a question or a blocker |
| Decomposer, map reviewer, data-need | 24.1 to 24.2 | unchanged, as nodes; the map's confirmation is an interrupt |
| **Understanding critic** (new) | the brief checker role, `dclab_core.cited` | reads the brief against the data card and the leakage and moment rules; can **send back** to the interviewer with the question it could not answer itself; never edits the brief |
| Confirmation | the person | a gate: the brief and the map as the person sees them, confirmed or corrected |

The team's output is what the solution team may read: the brief, the data card, the map, the person's confirmations.
The person's words are in the brief; a table's values are not.

### D3. The solution team

The same rule: the router, the designer's checks, the builder, the experimenter, the composer and the reporter are the
existing modules; the auditor is the existing leakage review and critic gate given the power to send a stage back.

| Node | Exists as | What it adds in the team |
|---|---|---|
| Router | 24.3 (16 approaches, needs met/unmet/unknown) | per part, **in parallel** (`Send`) for a complex problem |
| Designer | the solution planner, the expert design (25.3), `solution_workflow` | draws the workflow of each part; ranks algorithms with reasons; the validator checks every step's content |
| Builder | `build_project`, the engine's stages through the graph validator, the registry's tools | deterministic; the only node that touches data and numbers; an agent may call it, never bypass it |
| Experimenter | the critic rounds, `experiment.next` | proposes the next experiment from the records; the rules' first stays first unless it measured better |
| **Auditor** (new power, existing checks) | `leakage_review`, `critic_gate`, the holdout-once policy | reads every stage record; can **send back** a stage to the designer or stop the run with a named blocker; never changes a number |
| Composer | `compose`, `programme` (24.5) | joins the parts on validation, scores the holdout once |
| Reporter | `solution_report`, `cited.py` | every number traced to its record; the person reads it |

### D4. The board (the shared state)

`dclab_core.orchestra.Board` v1, a pydantic contract with a JSON schema (`make core-schemas`): the brief, the data
card, the map, the routes, the workflow, the open questions, the blockers, the budget spent, the hand-offs
(`from`, `to`, `what`, board version), the ledger ids of every decision. Reducers (framework-side) merge parallel
parts by part id. A table's value never enters the board (the same rule as the ledger's states).

### D5. Patterns the team may use, each measured before kept

- **Second opinion:** two readers of the first sentence (the rules and a model), a judge node that keeps the agreed
  fields and asks about the rest. Kept only where it beats one reader on the journeys.
- **Send-back:** the critic and the auditor can return work with a named reason; at most two rounds; the third stops with
  a blocker the person sees.
- **Debate** (two designers, a judge): a package of its own (28.6), run as an experiment, not assumed useful.
- **Specialist by kind:** a designer per problem kind (fraud, demand, routing…) reading `dclab_rnd/kinds`; same contract,
  different prompt file.

### D6. The person in the loop

Three interrupts and nothing else: a **question** (one field, the person's words), a **confirmation** (the brief, the
map, the solution map), an **approval** (build, read the holdout, promote). The Home page shows the board's state and the
open interrupt; the API resumes the thread; a tester's CLI does the same. The "build it" of 25.13 becomes an approval.

### D7. Memory, evidence and tools

Project memory (`studio/memory.py`), lessons, the evidence index and the retrieval that measured best are tools in the
registry, with scope "draft" or "project"; a node declares which it may call. Nothing new is written for this phase.

### D8. What is recorded, and what trains

Every node: a harness trace (what it read, what it returned, cost, time). Every decision: a ledger record with its
point. Every hand-off: an experience event. Every run: a trajectory in the lake's `solution_design` and `read_results`
families, with the board before and after each node, when the workspace's train switch is on (27.2). Claude's outputs
stay evaluation-only (the owner's rule); the rules' and the person's are gold.

---

## Part E. The packages (28.x)

Each package: a failing test first, the smallest change, `make check-quick`, the reviewer, its development record,
`make sft-dev`, a commit. Live runs capped; the owner's budget rules apply.

**The capture rule, in every package:** a package is not done until what it adds is recorded for training: every node
in the harness (inputs, outputs, cost, time), every decision in the ledger with its point, every hand-off and interrupt
as an experience event with the board before and after, every run's trajectory in the lake under consent, the files
archived by `make archive`. A test in each package reads back what it recorded. 28.9 reads it all into the lake's
families; nothing is captured for the first time there.

### 28.1 The framework decision made concrete: the thin adapter and a spike
- **Delivers:** `dclab_rnd/orchestra/` with (a) the gateway as a chat-model adapter LangGraph nodes can call (no
  LangChain model classes, no key in the framework), (b) a checkpointer bound to the workspace store (PostgreSQL when
  `DCLAB_DATABASE_URL` is set, SQLite in the workspace folder otherwise), (c) one spike graph of three nodes (interviewer
  → analyst → confirmation interrupt) on a dev journey, resumed after a simulated crash; the harness records every node.
- **Done when:** the spike runs on a scripted transport in tests with no live model, resumes from its checkpoint, and
  `test_agent_harness` passes with the nodes added to `AGENTS`; the lock file pins `langgraph` and `langgraph-checkpoint-postgres`;
  **capture:** every node's trace and the interrupt are in the harness and the experience log, read back by a test.
- **Decision recorded:** Part C's table and the reasons, in the package's record.
- **Status: built (2026-10-17).** `dclab_rnd/orchestra/` (`model.complete` through `installed().client`; `checkpoints.open_checkpointer`:
  PostgreSQL when `DCLAB_DATABASE_URL` is set, else `<home>/orchestra/checkpoints.sqlite3`; `spike.Spike`: interviewer → analyst →
  confirm on `dclab_core.conversation`, `interrupt()` for a question and a confirmation, `run(draft_id, problem | resume=)` inside
  `harness.session("orchestra")`, a pause recorded as a step and a run ended `stopped`). Tests: the probe, resume from the file in
  another process, PostgreSQL checkpoints, the adapter on a scripted gateway. The pins in `base.txt`, `product.txt`, `agent.txt`.

### 28.2 The board and the hand-off contracts
- **Delivers:** `dclab_core.orchestra` (`Board` v1, `Handoff` v1, `Interrupt` v1, `TeamRecord` v1) with schemas and the
  conformance rule that no table value enters a board; reducers in `dclab_rnd/orchestra/state.py`; the experience event
  `team.handoff`; `CORE_INTEGRATION.md` section.
- **Done when:** the schemas are frozen (`make core-schemas`), `core-wheel-check` passes, a board with a cell value is
  refused, and a hand-off round-trips through the experience log; **capture:** `team.handoff` and `team.interrupt` events
  carry the board's version before and after, and `make archive` keeps them.

### 28.3 The orchestrator: the coordinator's machine as the graph
- **Delivers:** the top-level graph, one node per coordinator state, edges by `situation()`; the three interrupts; the
  budget per run and per node; the graph's nodes call the same `Coordinator.act()` functions the loop calls today (no
  action is rewritten); `draft/coordinator.py` walks the graph when the workspace's team switch is on and its loop
  otherwise, permanently: the loop is the fallback, not a transition; resume from any state.
- **Done when:** the 15 dev journeys carry to the same result through the graph as through the loop (the rules arm
  15 of 15), with the walk resumable at every state and every step in the harness; **capture:** each run's node
  sequence, interrupts and spend as one `TeamRecord` in the experience log, and every decision in the ledger as today.

### 28.4 The understanding team
- **Delivers:** the nodes of D2 as a subgraph; the analyst and the investigator in parallel; the understanding critic
  (new, prompt file `prompts/understanding_critic.md`, checked by `cited.py`) with send-back (two rounds); the map
  confirmation as an interrupt.
- **Done when:** on the dev journeys the team's brief is at least as right as the single Home agent's on every field
  (the harness-eval grading), with the critic's send-backs counted and each one's reason recorded; nothing is asked twice;
  **capture:** the critic's questions and the person's answers as conversation examples, the brief before and after each
  node as `solution_design` inputs.

### 28.5 The solution team
- **Delivers:** the nodes of D3 as a subgraph; parts of a complex problem routed and designed in parallel (`Send`) and
  composed once; the auditor with send-back and stop; the builder only through the validator and the registry.
- **Done when:** the programme journeys (24.8, dev) build through the team; the holdout is read once per project in every
  run (a test counts); an auditor send-back is exercised on a planted leakage; the code export reproduces the number;
  **capture:** each part's route, design, audit verdict and send-back as ledger records and `solution_design` /
  `read_results` examples; self-play (27.3) solves through the team so verified solves carry the hand-offs.

### 28.6 Patterns as experiments: second opinion and debate
- **Delivers:** the second-opinion reader with a judge, and a two-designer debate, each as an arm on the dev journeys
  with a gate fixed before the run (better on the field it targets, no worse elsewhere, cost within 1.5× the single
  agent); kept only where it passes. **Budget:** about €2 per pattern, planned and capped before the run.
- **Done when:** each pattern has its result file in `evidence/campaigns/orchestra_v1/` and its switch (on where it
  passed, off elsewhere), audited.

### 28.7 The measurement: the team beside the rules and the single agent
- **Delivers:** a `team` arm in the journeys suite and in the twin matrix: accuracy per step, agreement, cost, time; the
  gate (fixed before any run): the team at least equals the rules on the journeys' four parts and beats the single
  agent on understanding, within a cost cap per journey; the sealed set once, after the dev pass.
- **Done when:** the dev pass is stored (`JRN-…`), the sealed verdict is in `evidence/campaigns/`, and the plan's
  status table says where the team serves (behind `draft.team.passed()`, false until then: the rules serve).
- **Budget:** about €4 per full dev pass (approved 2026-10-17; two live arms × 15 journeys with room for send-backs),
  planned, printed and capped before any request; the spend reported after.

### 28.8 Home and the product
- **Delivers:** the Home page shows the board (which node works, what it found, the open interrupt), answers resume the
  thread, "build it" is an approval; the Admin trace view shows a run's nodes, hand-offs, cost; the API
  (`/api/drafts/{id}/team`) for testers' CLIs.
- **Done when:** the product e2e flows pass through the graph; a browser test drives a question, a confirmation and a
  build; the page needs no page-specific model call.

### 28.9 The team's work as training data
- **Delivers:** trajectories from runs (the board before and after each node, the hand-offs, the person's answers)
  into the lake under consent, as `solution_design` and `read_results` examples; self-play (27.3) runs through the team
  so the verified solves carry the team's hand-offs; Claude's nodes evaluation-only.
- **Done when:** `make lake` reads them, the cleaning (27.5) passes, and the export for Model 1 holds the new families.

### 28.10 DCLab's own models as nodes
- **Delivers:** per-node model routing through 27.8: a tuned model shadows a node, is measured on the node's own
  decisions, promoted where it measured better, retired in one change; the tabular decision models (22.2, 26.3, 26.8)
  as answerers inside the nodes that ask their points.
- **Done when:** one node runs with a shadowed local model on the dev journeys and its shadow answers are measured in
  the evaluation board; nothing serves without the sealed gate.

### 28.11 (optional, the owner's call) A spike on the second-choice framework
- A three-node spike of 28.1 on the Microsoft Agent Framework, timed and compared on the same test; a note in the
  plan. Only if the owner wants the exit measured, not assumed.

---

## Part F. The owner's decisions (2026-10-17)

1. **The framework:** LangGraph, used thinly (Part C). Decided; no spike on an alternative first.
2. **Where the team runs first:** Home (drafts), then projects. 28.3 and 28.4 on drafts; 28.5 on projects.
3. **The person's place:** three interrupts (question, confirmation, approval); the rules' path serves while one is open.
4. **Budget for measurement:** about **€4 per dev pass** (28.7) and about **€2 per pattern experiment** (28.6); the sealed
   pass once. Each run is planned, printed and capped before any request (the owner's standing rule); the spend is
   reported after it. The rate seen in 26.5's passes was about €0.12 a journey per live arm, so €4 covers two live arms
   on the 15 dev journeys with room for send-backs.
5. **Parallelism on the Mac:** builds (engines) run one at a time; only Claude model calls run in parallel.
6. **The teams, exactly:** understanding = the Home agent's questions, the problem analyst, the data investigator,
   meet/links, the decomposer, the map reviewer, data-need; solution = the router, the solution planner and expert
   design, the engine through the validator, the critic rounds, compose/programme, the report. Nothing else is in a
   team until it is measured in (28.6).
7. **Capture everything, always (the owner's standing rule, restated for this phase):** every process and procedure of
   every agent and of the whole system records and stores its data for training and fine-tuning, as much as can be
   collected, under the workspace's consent. In this phase that is Part D8 and the rule in Part E that **every package's
   "Done when" includes its capture**; 28.9 only adds the lake's reading of what the earlier packages already wrote.
8. **This phase is infrastructure.** The board, the interrupts, the checkpoints, the per-node routing and the capture are
   the base of the next phase (the owner: related to this one). Each is a contract or a module with its own test, not
   a feature of one page.

## Part G. Risks and what keeps them small

| Risk | Kept small by |
|---|---|
| More agents, more cost, no gain | every pattern is an arm with a gate and a cost cap; the rules serve until the team passes (28.7) |
| The framework decides the flow | the coordinator's situation chooses every edge; agents return artefacts, not states (28.3) |
| Lock-in | framework-free contracts in `dclab_core` (28.2); the wiring in one folder |
| Memory on the laptop | builds serial locally; parallel only model calls; the test suite's memory rule unchanged |
| A person waits on a crashed run | checkpoints per node; resume by thread id (28.1, 28.3) |
| Training data polluted by Claude | nodes' answers tagged by answerer; Claude evaluation-only; the lake's rule unchanged (28.9) |
| Prompt drift | prompt files with hashes, `make agent-eval` on change, harness-eval after a prompt or gate change |

## Part H2. The phase after this one

`PHASE7_STUDENTS_PLAN.md` (the owner, 2026-10-17): a student for every choice. Phase 6 records the choices of every
node; phase 7 trains the student behind each one and lets it serve only where it proved better. They run side by side.

## Part H. Not in this phase

A second implementation of problem understanding or of solving beside the existing one (the owner's rule: the agentic
system enhances what the repository has); a rewrite of the Home agent, the coordinator, the router, the validator or
the engine.

Training any model (the owner's GPU runs); a pretrained image backbone; new problem kinds; the LangGraph Platform or
LangSmith (not needed: our harness traces, our store checkpoints); a UI redesign beyond the board's view.

## Status

- **Design (2026-10-17):** written and approved by the owner the same day (Part F); the teams fixed as in F6; LangGraph
  thinly; Home first; three interrupts; €4 a dev pass, €2 a pattern; serial builds; capture in every package. Nothing
  implemented yet; 28.1 is next.
