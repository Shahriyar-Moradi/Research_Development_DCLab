# The agentic foundation: design, phases, and prompts

This is the design for DCLab's AI and agent layer, written as phases with small plans and a prompt for each. It
extends Part 2 of `PRODUCT_BUILD_PLAYBOOK.md` (packages 11.1 to 11.4) and uses its shared preamble.
Read the playbook's "Shared preamble" first and paste it before any prompt below.

**خلاصه فارسی.** این سند طراحی لایه‌ی هوش مصنوعی و ایجنت DCLab است: اصل‌ها، معماری، شش فاز (A1 تا A6) و برای هر
بسته یک پرامپت آماده. هسته‌ی ایده: **کد تصمیم می‌گیرد، مدل پیشنهاد می‌دهد.** هیچ مدلی تقسیم داده یا معیار را
محاسبه نمی‌کند؛ هر حرکت ایجنت از اعتبارسنج گراف می‌گذرد؛ همه‌ی درخواست‌های مدل از یک دروازه عبور می‌کنند؛ و هر ایجنت
با مجموعه‌ی آزمونی سنجیده می‌شود که تله‌های واقعی (نشت) دارد.

## Status

| Package | Status | Notes |
|---|---|---|
| A1.1 | Done | Every model request goes through `dclab_rnd/models` (the intern too). NOOA's research clients cannot be swapped and report each request to the usage log. Tests cannot reach a live model (`DCLAB_NO_LIVE_MODELS=1` in the make targets). |
| A1.2 | Done | Costs from a dated price table (`models/prices.py` or `DCLAB_PRICES_FILE`; none is guessed, local is free). A request whose upper-bound cost could pass a run, project or workspace cap is refused before it is sent, under a lock. Open: no price is configured yet, so remote requests are counted in tokens until prices are added. |

## 1. What exists today

Three agents, built separately, plus the tools they share.

| Piece | File | What it does | Gap |
|---|---|---|---|
| Home agent | `draft/chat.py` (492 lines) | Up to 4 questions, outcome and workflow, simulation, streamed replies; a scripted fallback | Own loop, own tool list; no trace of its steps |
| Intern | `intern/loop.py` (247), `intern/tools.py` (294, 20 tools) | Builds a project through the validator, with a step and minute budget; standard plan with no model | A third loop; budget in steps and minutes only |
| Campaign agent | `agentic/agents.py`, `worker.py` (NOOA, LangGraph) | Research campaigns that fill `evidence/campaigns/` | Separate runtime and model client; web UI trigger lives in `/classic` |
| Notebook copilot | `copilot/`, `notebook_assist.py` | Reviews a notebook against the rules | Deterministic; no model |
| Model client | `intern/llm.py` | OpenAI-compatible, streaming, safe failure reasons, pause after a hard refusal | Created in several places; no budget, no usage log, no tiers |
| Validator | `studio/graph.py` (364) | WF-01…10, every move checked and logged | The one thing that must stay in front of every state change |
| Evidence | `evidence_index.py`, `tools.py`, `critic_gate.py` | Search over 137 records; critique checked against numbers | Retrieval is keyword (BM25) only |
| Evaluation | `notebook_eval.py`, `system_eval.py`, `trace_eval.py`, `replay_compare.py`, `agentic/live_eval.py` | Pilot scorecards, trace grading | Not one suite; not run per agent change |
| SFT | `studio/sft.py`, `research/llm-fine-tuning` | 330-example corpus; project-derived examples | No trajectory data from real runs yet |

## 2. Principles (they do not change between phases)

1. **Code decides, a model advises.** A model proposes JSON (a parse pattern, a workflow, a synthetic schema, a plan step). Deterministic code validates and applies it. A model never computes a split, a metric or a selection.
2. **Every state change goes through the workflow validator.** No agent writes a record, a solution, an approval or a split except through `studio.graph.check`. A refused move is logged and returned to the agent as a reason it can read.
3. **Aggregates, not rows.** An agent may read column summaries and descriptive findings. Cell values stay on the machine. The single exception (sample lines for an unreadable file) is disclosed and can be switched off.
4. **No model, no problem.** Every agent has a deterministic path that finishes the job. A model improves the conversation; it is never required for correctness.
5. **Every request is counted.** One gateway, one usage log, one budget. No module creates its own client.
6. **Every step is replayable.** What the agent saw, which tool it called with which arguments, what the validator said and what came back, in order, per run.
7. **Evidence cited or absent.** A claim cites a record that exists, with numbers that appear in it, or it is dropped.
8. **Unsafe is measured.** Planted traps (a post-outcome column, an identifier that predicts the target, a random split on timestamped rows) are the test of every agent, scripted in CI and live on request.

## 3. The architecture

```text
   user / MCP client / API
          │
   ┌──────▼───────┐     ┌────────────────┐     ┌───────────────┐
   │ Agent runtime │────▶│  Tool registry │────▶│ Workflow       │
   │ (one loop)    │     │  (schemas,     │     │ validator      │──▶ stores (PostgreSQL)
   │ budget, trace │     │   effects)     │     │ studio.graph   │
   └──────┬───────┘     └───────┬────────┘     └───────────────┘
          │                     │ read-only tools: evidence, profile, records
   ┌──────▼───────┐     ┌───────▼────────┐
   │ Model gateway │     │ Evidence index │
   │ tiers, budget,│     │ (+ later:      │
   │ usage log,    │     │  embeddings)   │
   │ pause, retry  │     └────────────────┘
   └──────┬───────┘
          │ OpenAI-compatible endpoints: strong · standard · cheap (a local model can serve cheap)
```

- **Agent runtime** (`dclab_rnd/agents/`): one loop. A *Tool* has a name, a JSON schema, a handler and a flag saying whether it changes state. A *Policy* is a system prompt plus a tool list plus a budget. The Home agent, the intern and the campaign agent become three policies on this loop.
- **Model gateway** (`dclab_rnd/models/gateway.py`): `complete()` and `stream()` take a *purpose* and a *tier*. It owns retries, timeouts, the safe failure reasons and the pause, counts tokens, prices them from one table, checks the project and workspace budget, and writes a usage row.
- **Trace store**: each runtime step is a row (state summary, tool, arguments, verdict, result summary, tokens, time). The trace feeds the Compute page, the audit trail and the training data.

## 4. Phases

| Phase | Goal | Packages | Depends on |
|---|---|---|---|
| A1 | One gateway | A1.1 to A1.3 | Part 2: 9.2 |
| A2 | One runtime | A2.1 to A2.4 | A1 |
| A3 | Make the agents better at the real job | A3.1 to A3.4 | A2 |
| A4 | Prove it: evaluation | A4.1 to A4.3 | A2 |
| A5 | Retrieval and memory | A5.1 to A5.3 | A2 |
| A6 | Learn from runs: the policy model | A6.1 to A6.4 | A3, A4 |

Order: A1, A2, then A3 and A4 together (every A3 change must pass A4), then A5, then A6.

---

## Phase A1: One gateway

### A1.1 The gateway and its usage log

- **Delivers:** `dclab_rnd/models/gateway.py`, a `model_request` table (migration), tiers in settings, and every current caller moved onto it.
- **Done when:** a search finds no `openai.OpenAI(` or `ChatClient(` outside the gateway; the usage table fills during `make product-e2e` with a model configured; `make rd-check` passes with scripted clients.

```text
Package A1.1. Build the model gateway. Today ChatClient (dclab_rnd/intern/llm.py) is created in the Home agent,
the intern, the evidence answers and the synthetic-data path, and studio/agent.py and the campaign worker call
OpenAI directly. Write dclab_rnd/models/gateway.py with complete(messages, tools, purpose, tier, project_id) and
stream(...). Settings map each tier (strong, standard, cheap) to an OpenAI-compatible endpoint and model; an
unset tier falls back to the standard one. Keep what llm.py already does (streaming, the safe failure reason,
the pause after a hard refusal) and add: per-purpose timeouts, retry with backoff on 429 and 5xx, and a
model_request row for every call (purpose, tier, model, input and output tokens, latency, outcome, project;
the prompt text only when DCLAB_LOG_PROMPTS=1). Add the table through an Alembic migration. Move every caller
onto the gateway. Tests use scripted clients; none may touch a live endpoint.
```

### A1.2 Budgets and prices

- **Delivers:** one price table, a per-project and per-workspace monthly cap, a refusal with a clear message when a request would pass it, and real spend on the Compute and Admin pages.
- **Done when:** a scripted run over its cap is refused before the request is sent; the pages show euros from the usage table, and "not metered" is gone only where it is true.

```text
Package A1.2. Add money to the gateway. Keep prices per model in one file (dclab_rnd/models/prices.py, euros per
million tokens, with a date) and compute a request's cost from its token counts. Add caps: per project per run,
per project per month, per workspace per month; the wizard's euro budget becomes a real cap. Check before
sending; refuse with "this project has used its budget" and say how much is left. Show spend on the Compute and
Admin pages from the usage table and remove "not metered" only where it is now true (local models cost zero:
say so). Never guess a price: an unknown model is "price not configured" and is counted in tokens only.
```

### A1.3 Tiers by purpose

- **Delivers:** a table of which purpose uses which tier, a local small model for cheap purposes, and a check that a purpose never sees more than its allowed data.
- **Done when:** with a local model on the cheap tier, parse patterns and evidence answers run on it and the usage log says so.

```text
Package A1.3. Assign tiers by purpose in one table: parse_pattern and evidence_answer on cheap, home_agent and
synthetic_schema on standard, intern and campaign on strong. Make each tier configurable from the environment
(for example DCLAB_TIER_CHEAP_BASE_URL=http://127.0.0.1:11434/v1, a local model). For each purpose write down in
the table what data it may be shown (problem text, answers, column summaries, findings, evidence text, raw lines)
and test that the gateway's callers never exceed it. A model that fails a purpose's output check twice is
reported on the Admin page, with the purpose and tier.
```

---

## Phase A2: One runtime

### A2.1 The runtime and the tool registry

- **Delivers:** `dclab_rnd/agents/` with `Tool`, `Registry`, `Policy` and `run()`; the 20 intern tools and the Home agent's tools registered once; the MCP server reads the same registry.
- **Done when:** adding a tool is one registration; the intern, the Home agent and `/mcp` all list it; `tests/test_mcp.py` and `tests/test_intern.py` pass unchanged.

```text
Package A2.1. Extract one agent runtime. Write dclab_rnd/agents/: Tool (name, description, JSON schema, handler,
effect: "read" or "write"), Registry, Policy (system prompt, tool names, budget) and run(policy, state, gateway).
run() asks the gateway, validates the arguments against the tool's schema (a bad call is returned to the model
as an error it can read, not raised), executes the tool, and loops until a final answer, the step budget or the
time budget. Register the 20 intern tools (dclab_rnd/intern/tools.py) and the Home agent's tools once. Make
dclab_rnd/mcp_server.py list the registry. Do not change any tool's behaviour or name; keep the legacy names.
```

### A2.2 The validator in front of every write

- **Delivers:** every `write` tool goes through `studio.graph.check` inside the runtime; a refusal becomes the tool result with the rule it broke.
- **Done when:** a test calls every write tool with a move the validator refuses and gets the reason and a logged transition; no write tool bypasses the check.

```text
Package A2.2. Make the validator impossible to skip. In the runtime, every tool with effect "write" declares the
workflow move it makes (run_stage, set_solution, approve_gate, approve_stage, capture). Before the handler runs,
the runtime calls studio.graph.check for that move with the actor "agent"; a refusal is logged by graph.log and
returned to the model as the tool result with the verdict's reason and the rules it cites. Add a test that
enumerates the registry and fails if a write tool has no declared move. The Home agent's writes (record,
set_pack, propose_workflow, simulate_data) change a draft, not a project: give them their own checks (workflow
validation, pack list, one simulation at a time).
```

### A2.3 Traces

- **Delivers:** an `agent_step` table; every runtime step stored with what the agent saw (a summary), the tool and arguments, the verdict, the result summary, tokens and time; a trace view on the Intern page.
- **Done when:** any finished run can be replayed from its trace with a scripted model and gives the same moves.

```text
Package A2.3. Store a trace for every run. Add an agent_step table (run id, step number, state summary, tool,
arguments, verdict, result summary, tokens, seconds) through an Alembic migration. The runtime writes a row per
step. Add replay(run_id): feed the stored assistant messages back through the runtime with a scripted gateway
and check that the same moves result. Show the trace on the Intern page with the existing step list. Never store
cell values: arguments and results are the summaries the tools already return.
```

### A2.4 The three agents as policies

- **Delivers:** the Home agent, the intern and the campaign agent run on the runtime with their behaviour unchanged.
- **Done when:** `make product-e2e` and the agent tests pass; `draft/chat.py`'s own loop and `intern/loop.py`'s loop are gone; the campaign agent runs one campaign on the same runtime.

```text
Package A2.4. Rebuild the three agents as policies on the runtime, one at a time, with the existing tests as the
safety net. Home agent: keep the scripted fallback, the one-turn-per-draft lock and the streamed replies. Intern:
keep the standard plan when no model is configured. Campaign agent (dclab_rnd/agentic/worker.py, NOOA and
LangGraph): put it behind the same runtime and gateway, or if NOOA cannot be wrapped, keep it as an adapter that
reports its requests to the gateway. Delete a loop only when its tests pass on the runtime.
```

---

## Phase A3: Make the agents better at the real job

### A3.1 A planner for the Home agent

- **Delivers:** the agent keeps a short plan (what is known, what is unknown, what to ask next, why) as state; the questions come from the plan, not from a fixed order.
- **Done when:** with a scripted model, a problem sentence that already states the outcome and the moment is not asked those again; the question count falls without losing a field.

```text
Package A3.1. Today the Home agent asks the same four things in order. Give it a plan in the draft: for each of
outcome, prediction moment, action and cost, and data, the status (stated, inferred, unknown) and where it came
from. After the problem sentence and after each answer, update the plan from the text (a model proposes, code
checks that the quoted words are in the user's text). Ask only unknown fields, in the order that matters most for
leakage (the prediction moment before the cost). Show the plan as the existing "DCLab understood" sheet. The
script path does the same with keyword rules.
```

### A3.2 A leakage reviewer

- **Delivers:** a reviewer step in the wizard and for the intern: it reads the column audit, the prediction moment and the column names, and proposes which columns are unavailable at that moment, each with a reason and a record; code decides what is applied.
- **Done when:** on the studied samples the reviewer's proposals include the R&D's own forbidden columns (for example `duration` on bank_marketing); a column it proposes without a reason is dropped.

```text
Package A3.2. Add a leakage reviewer. Input: the prediction moment text, the column names, kinds and summaries,
and the column audit's flagged list. Output (JSON, validated): per column, "available" or "after the moment", a
one-sentence reason that quotes the moment, and the evidence record IDs that support it (they must exist in the
index). Code drops any item with no reason or a missing record and never un-forbids a column the audit
flagged; the user ticks what to forbid. Without a model, the existing audit and name rules answer. Measure it on
the 16 studied samples: how many of the R&D's forbidden columns it finds, and how many it adds that the R&D did
not (those are shown to the user, never applied).
```

### A3.3 Explanations with citations

- **Delivers:** the notebook, reliability and brief pages can show a short model-written explanation of each result, with every sentence cited and every number checked against the stage record.
- **Done when:** a sentence with a number that is not in the cited record is removed; the page labels the text "written by a model from these records".

```text
Package A3.3. Reuse the Evidence library's answer check (dclab_rnd/agentic/pages/evidence.py: keep only sentences
that cite a retrieved record and whose numbers appear in it) for stage explanations. Input: one stage record
(claims, evidence, notes). Output: at most 120 words, each sentence citing a claim id or a record id. Drop the
rest. Show it on the Notebook and Reliability pages and in the brief, labelled as written by a model from the
stage record, with the deterministic notes unchanged beside it. Never let the explanation say "production-ready".
```

### A3.4 A reviewer for the user's own notebooks

- **Delivers:** the copilot reviews an uploaded notebook with a model for the explanations and code for the findings.
- **Done when:** the leaky demo notebook gives the same findings as today, with a model-written fix per finding that cites the rule.

```text
Package A3.4. The notebook copilot (dclab_rnd/copilot, notebook_assist.py) finds problems deterministically.
Keep every finding exactly as it is. For each finding, ask the model for a fix in two sentences that cites the
rule and the matching pitfall record, and show it under the finding. A finding never depends on the model. Test
with the leaky bank-marketing example notebook and a scripted client.
```

---

## Phase A4: Prove it

### A4.1 The judgment suite

- **Delivers:** `dclab_rnd/agent_eval/` with at least 20 cases, each a small table with a planted trap and the expected behaviour; the scripted run is part of `make rd-check`.
- **Done when:** the standard plan scores on every case; a deliberately broken policy (one that ignores the audit) fails the cases it should.

```text
Package A4.1. Build the judgment suite the Benchmark page describes. Each case: a generated table (seeded), the
trap, the prediction moment, and the expected outcome. Traps: a post-outcome column; a copy of the target with
noise; an identifier that predicts the target; a time column with a random split; group leakage; a duplicated
row set across the split; a column that is a unit conversion of the target; a constant column; a high-missing
column with informative missingness (not a leak). For each case score: leak caught (yes/no), false alarm (yes/no),
move refused by the validator when it should be, and citations that exist. Run the standard plan and a scripted
"bad" policy on all cases in make rd-check, and expect the bad one to fail the right ones.
```

### A4.2 Live runs with a budget

- **Delivers:** `make agent-eval-live`, which needs an explicit flag and prints requests, tokens and cost before it starts and refuses to exceed a stated cap; scores with intervals over repeated runs.
- **Done when:** a run on a local model prints per-case results and a table of scores with their spread; no run starts without the flag.

```text
Package A4.2. Add a live evaluation of a configured model on the A4.1 cases. It must show the estimated number
of requests and the cap and ask for --yes, run each case several times (default 5) because a model's answer
varies, and report each score as a mean with a 95% interval, plus tokens and cost from the gateway. Store the
result under evidence/campaigns with a run id and the model name so it can be compared later. Say in the output
that scores on these cases are evidence about this model on these traps, not about production readiness.
```

### A4.3 A regression gate for prompt and tool changes

- **Delivers:** a change to a policy prompt, a tool schema or a validator rule must pass the scripted suite; the live results of the last release are the baseline to compare.
- **Done when:** a test fails when a prompt file changes without its recorded hash being updated by the suite run.

```text
Package A4.3. Treat prompts as code with tests. Keep each policy's system prompt and tool schemas in versioned
files. Record a hash of each next to the latest scripted-suite result; make rd-check fails when a prompt or
schema changed and the suite has not been re-run for that hash. When a live baseline exists, print the change
in score per case after a run. No prompt text is edited by a model.
```

---

## Phase A5: Retrieval and memory

### A5.1 Better evidence search

- **Delivers:** hybrid retrieval (the current keyword search plus embeddings), measured on a set of real questions with known answers.
- **Done when:** recall at 5 on the question set is reported for keyword only and for hybrid; hybrid is used only if it is not worse on any question group.

```text
Package A5.1. The evidence index uses BM25 only. Write 60 questions with the record IDs that answer them (from
the rules, pitfalls and experiments; include paraphrases that share no words with the record). Measure recall at
5 for BM25. Add an embedding retriever (a local sentence-embedding model or an OpenAI-compatible embeddings
endpoint through the gateway) stored in PostgreSQL (pgvector) or a file when pgvector is absent, and a hybrid
that merges both lists. Report recall at 5 for each and use hybrid only if it is not worse on any group of
questions. The index stays generated: rebuild it with make knowledge.
```

### A5.2 Project memory

- **Delivers:** what a project has learned (decisions, why, user corrections) kept as short notes the agents read, each tied to the move that produced it.
- **Done when:** a second intern session on the same project reads the first session's decisions instead of asking again.

```text
Package A5.2. Give each project a memory: short notes (decision, reason, who, which move, date), written when a
stage is approved, a solution is saved, a gate is approved or the user corrects the agent. Agents read the notes
as part of their state (the latest 20, summarised). A note never holds a cell value. Show the notes on the
Solution & data page. A note can be removed by a person; the audit trail keeps the removal.
```

### A5.3 Workspace lessons

- **Delivers:** lessons from finished projects, reviewed by a person, that can join the evidence index with their scope and the project they came from.
- **Done when:** a lesson accepted by a reviewer appears in the evidence index with its scope and is cited by an agent.

```text
Package A5.3. Build the "From projects" flow the Evidence library says is not built. After a project's final
stage, the agent proposes up to three lessons, each with its claim, its scope (this dataset, this task, this size),
what argues against it and the next test, citing the stage records. A reviewer accepts, edits or rejects each
(role: reviewer). Accepted lessons are stored in the workspace's lessons table and searched together with the
repository's records, labelled as workspace lessons; they never change evidence/knowledge. A lesson from synthetic
data is labelled synthetic and is not offered as evidence.
```

---

## Phase A6: Learn from runs

### A6.1 Trajectory export

- **Delivers:** finished runs exported as training records: state, move, validator verdict, evidence, outcome; only from runs the owner opted in.
- **Done when:** `python -m dclab_rnd.studio.sft --trajectories` writes records for opted-in projects and none for others; no cell value appears in any record.

```text
Package A6.1. Turn the traces (A2.3) and the transition logs into trajectory examples: (state summary, allowed
tools, chosen move, validator verdict, cited records, what happened next). Export only projects whose owner opted
in (a setting on the project; off by default; says so on the Policy model page). Remove anything that could hold
a value: tool arguments are kept only as the schema's non-free-text fields. Write to a new folder, never into
out_v3. Add a test that scans an export of a project built on the Telco CSV for any cell value.
```

### A6.2 The data recipe and the held-out split

- **Delivers:** a corpus v4 recipe that mixes declarative examples (v3) with trajectories, held out by whole projects, with the number of held-out projects shown against the threshold.
- **Done when:** the Policy model page shows "N of 50 trajectories" from the real count; stage 2 is marked ready only at the threshold.

```text
Package A6.2. Write the recipe for corpus v4: the 330 declarative examples of v3 plus trajectory examples,
split by whole project (about one in five held out), with the same quality gates as v3 (no benchmark
contamination, no near-duplicate prompts across the split). Do not generate anything until at least 50
trajectories from at least 10 different projects exist; until then print how many exist. Keep generated files out
of version control as the repository does for out_v3.
```

### A6.3 Train and evaluate a small policy model

- **Delivers:** a LoRA run on a small model with the existing scripts, evaluated on the A4.1 suite against the standard plan and the untuned base model.
- **Done when:** the Benchmark page shows the three side by side with intervals; the tuned model is promoted only if it is not worse on any trap class.

```text
Package A6.3. Only when A6.2's threshold is met and I approve the compute and cost. Run research/llm-fine-tuning's
train_lora.py on the v4 corpus, then evaluate the tuned model, the base model and the standard plan on the A4.1
cases with A4.2's protocol. Report per trap class with intervals. Do not call the tuned model better unless it
is not worse on every class and better on the total. Register it as a tier in the gateway behind a switch, off by
default.
```

### A6.4 Serve it as a tier, with a safety net

- **Delivers:** the tuned model as an optional cheap or standard tier, with a shadow mode that runs it beside the current model and logs disagreements without acting on them.
- **Done when:** shadow mode logs disagreements for a week of scripted traffic and the validator still sees only the production model's moves.

```text
Package A6.4. Add shadow mode to the gateway: a purpose can name a shadow model that receives the same request;
its answer is stored and compared (same tool chosen? same validated output?) but never used. Show agreement
rates on the Admin page. Switching a purpose to the tuned model requires a reviewer's approval, recorded in the
audit trail, and can be undone in one setting.
```

---

## 5. How a package is checked

Every package in this document is finished only when all of these hold:

1. `make rd-check` passes on files, and `make test-pg` passes on PostgreSQL.
2. `make product-e2e` passes (a model path is also run once with a scripted client).
3. A unit test fails without the change (it was written first or shown to fail).
4. No model request left the machine in a test (the gateway refuses a live endpoint under test).
5. The Admin page's statement about what a model may see is still true; if the package changes it, the page and `.env.example` change in the same commit.

## 6. What this plan does not do

- It does not make a model's judgment trustworthy by itself. Scores here are evidence about a model on these traps; the validator and the user's approvals are what protect a project.
- It does not train anything without enough real trajectories (A6.2) and your approval of the cost (A6.3).
- It does not send raw rows to a model. Sample lines for an unreadable file are the one disclosed exception, and `DCLAB_MODEL_READS_SAMPLE_LINES=0` turns them off.
