# DCLab's in-house agent harness: an audit from the code (2026-10-22, HEAD aee3f458)

Sources are files in the repository `/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D` (paths relative to it, with line numbers at HEAD), commits (`git show <hash>`), and development records read with `git show HEAD:research/llm-fine-tuning/experiments/dev_trajectories/records/<name>.json` (the folder is not on disk). No model was called and no test was run for this audit; everything comes from reading code and history.

## Which of the 11 needs does the harness meet well, partly, or not at all?

### Takeaway
The harness is strong where DCLab is special: guarded tool calls, a workflow validator, money caps, a scripted-model test culture, and very wide capture. It is weaker on generic agent-runtime features: no durable pause inside the main loop, sequential tool calls only, no real OpenTelemetry export, extended thinking used only in the twin lane, and capture that is whole in some places but cut or shape-only in others (the intern and campaign exchanges are not kept whole; the shared trace row and `tool.call` event still hold a 360-character result summary).

Scorecard (my judgment from the evidence below): (1) tool calling + guards: **well**; (2) full trace for SFT: **partly**; (3) deterministic replay/testing: **well for testing, partly for replay**; (4) HITL with durable pause/resume: **partly** (durable only in the LangGraph "orchestra" layer, which is off by default); (5) cost caps: **well**; (6) multi-agent: **partly** (built on LangGraph, off by default); (7) MCP: **partly** (server yes, client no); (8) providers: **mostly well** (Anthropic + OpenAI-compatible, streaming, caching; thinking only in the twin lane; tool calls run one by one); (9) observability: **partly** (in-house spans and Prometheus counters, no OTel export); (10) size/maintenance: **heavy**; (11) known bugs: a recurring pattern of live keys reaching tests, and capture cuts fixed one place at a time.

### Cited Findings

**(1) Tool calling with validators/guards on every call**
- Each tool is registered once with a JSON schema; a write tool must declare its `move` or registration fails; parameters must be an object schema — [dclab_rnd/agents/registry.py:59-70](dclab_rnd/agents/registry.py)
- Every call is checked against the schema with `jsonschema` Draft 2020-12 before the handler runs; bad calls return `{"error": ...}` the model can read, never an exception — [dclab_rnd/agents/registry.py:85-133](dclab_rnd/agents/registry.py)
- A write in a scope without a guard raises and never runs; writes run only through `guard(tool, context, arguments, proceed)` — [dclab_rnd/agents/registry.py:125-133](dclab_rnd/agents/registry.py)
- Six guards exist: project (`studio.graph.check` as actor "agent", refusals logged as transitions), draft, home, campaign, session — [dclab_rnd/intern/tools.py:95-128](dclab_rnd/intern/tools.py), [dclab_rnd/draft/chat.py:1503](dclab_rnd/draft/chat.py), [dclab_rnd/draft/orchestrator.py:1406](dclab_rnd/draft/orchestrator.py), [dclab_rnd/agentic/campaign_tools.py:84](dclab_rnd/agentic/campaign_tools.py), [dclab_rnd/intern/loop.py:497-507](dclab_rnd/intern/loop.py)
- The session guard is a pass-through (`return proceed()`), there "so that every write in the registry passes one" — [dclab_rnd/intern/loop.py:497-502](dclab_rnd/intern/loop.py)
- The validator is plain code: 10 workflow nodes WF-01…WF-10, verdicts `allowed | blocked | needs_approval`; only a person can approve a gate — [dclab_rnd/studio/graph.py:1-18, 196-221, 313-323](dclab_rnd/studio/graph.py)
- The Home draft guard checks content too: "build" only when the person's last message is an order to build; a proposed workflow is validated and refused with the problems — [dclab_rnd/draft/chat.py:1480-1500](dclab_rnd/draft/chat.py)
- Weak spots: MCP clients and the intern's Toolbox call path use `check="names"` (only unknown/missing argument names, no type check); the comment says this lasts "until A2.4 moves the intern onto run()", which already happened, so the comment is stale — [dclab_rnd/intern/tools.py:213-215](dclab_rnd/intern/tools.py), [dclab_rnd/mcp_server.py](dclab_rnd/mcp_server.py) (calls `toolbox.call`). Code-initiated Home tool calls use `check="none"` — [dclab_rnd/draft/chat.py:1208](dclab_rnd/draft/chat.py)
- An AST test fails if any tool is called or a handler invoked outside the recording functions — [tests/test_agent_harness.py:99-115](tests/test_agent_harness.py)

**(2) Full trace capture for SFT**
- The shared trace row (`agent_step` table or `agent_steps.jsonl`) stores the tool result only as a summary: `RESULT_CHARS = 360`, with cell-bearing keys removed first; arguments are cut above 8,000 characters — [dclab_rnd/agents/traces.py:28-33, 59-69, 216-219](dclab_rnd/agents/traces.py)
- The session record's `tool.call` event copies that same `row["result"]` (the 360-char summary) — [dclab_rnd/experience/hooks.py:164-184](dclab_rnd/experience/hooks.py)
- The "RESULT_CHARS" problem: commit 58ccfc48 (2026-10-21) found the M5 solver set had 4 train rows, one clean, partly because "each step kept only a 360-character summary"; the fix made the *intern session* keep each result whole (cells removed) — [git show 58ccfc48](.), [dclab_rnd/intern/loop.py:80-86](dclab_rnd/intern/loop.py). The fix is local to the intern; the shared tracer still cuts.
- The original 360-char design (A2.3) was chosen so that the trace, "the training source for A6", would not carry table cells — [records/A2.3.json](research/llm-fine-tuning/experiments/dev_trajectories/records/A2.3.json) (via `git show HEAD:`)
- Gateway exchange capture (32.10, commit 5b55f4a7): every request appended to `<workspace>/model_exchanges/YYYY/MM/DD.jsonl` with messages, reply, tool calls and thinking — but only when the purpose has `keeps_text=True` and `cell_values == 0` and the workspace allows model text; otherwise only sizes and a flag — [dclab_rnd/models/gateway.py:465-498](dclab_rnd/models/gateway.py)
- Consequence: the `intern` purpose (`cell_values=600`, no `keeps_text`) and the `campaign` purpose (no `keeps_text`) get shape-only exchange records — [dclab_rnd/models/settings.py:142-146](dclab_rnd/models/settings.py)
- Tool schemas are not stored in the exchange record (the doc has messages, reply, tool_calls, thinking, usage, a 16-char hash of the system prompt, no `tools`) — [dclab_rnd/models/gateway.py:482-491](dclab_rnd/models/gateway.py)
- The `model.request` event keeps the prompt only from string-content messages, cut at 20,000 characters, and only where text may be kept — [dclab_rnd/experience/hooks.py:187-204](dclab_rnd/experience/hooks.py); the usage log keeps a prompt only with `DCLAB_LOG_PROMPTS=1`, also cut at 20,000 — [dclab_rnd/models/gateway.py:268, 501](dclab_rnd/models/gateway.py)
- Places that do keep tool steps whole: intern sessions (above), Home orchestrator turn records (`result: s.result`), workspace chat records, MCP calls (`mcp/calls.jsonl`, whole) — [dclab_rnd/draft/orchestrator.py:1793-1796](dclab_rnd/draft/orchestrator.py), [dclab_rnd/workspace_chat.py:155-158](dclab_rnd/workspace_chat.py), [dclab_rnd/mcp_server.py](dclab_rnd/mcp_server.py) (`_keep_call`)
- The twin lane's `ClaudeClient` keeps every request's messages and tools and every answer, plus thinking, in its own transcript files — [dclab_rnd/twin/claude.py:1-20](dclab_rnd/twin/claude.py)
- The tool-message size the model sees is a separate limit: 4,000 chars by default, 6,000 for the intern — [dclab_rnd/agents/runtime.py:40, 77-79, 216](dclab_rnd/agents/runtime.py), [dclab_rnd/intern/tools.py:35](dclab_rnd/intern/tools.py)

**(3) Deterministic replay and testing with scripted models**
- `replay(rows, policy, registry)` re-feeds the stored tool calls through `run()` with a scripted model; tools answer from the trace; it checks the same tools, order and verdicts — [dclab_rnd/agents/traces.py:254-322](dclab_rnd/agents/traces.py)
- Replay refuses traces with cut arguments; tools answer with the 360-char summary (`{"replayed": row["result"]}`), so replay checks the runtime's moves, not the model's behaviour on full results — [dclab_rnd/agents/traces.py:276-297](dclab_rnd/agents/traces.py)
- The gateway takes an injected `transport`, `sleep`, `clock`, `shadow_runner`, so tests run with scripted transports — [dclab_rnd/models/gateway.py:89-100](dclab_rnd/models/gateway.py)
- Harness-related test files total about 4,670 lines (runtime 343, traces 282, validator 213, gateway 709, Anthropic transport 299, harness 353, orchestra ×4 1,011, MCP 91, intern 325, Home orchestrator 1,046) — `wc -l tests/...`
- The harness regression suite: scripted runs 17/17 in HRN-025…039; live runs HRN-022 10/17 and HRN-024 14/17 — [evidence/campaigns/harness_v1/results/](evidence/campaigns/harness_v1/results/)

**(4) Human-in-the-loop approvals with durable pause/resume**
- In the main loop, an approval is a state, not a pause: the validator returns `needs_approval`; the agent gets an error result; a person approves through the API (`approve_gate` requires actor "human"); the agent's later call passes — [dclab_rnd/studio/graph.py:215-221, 266-273, 313-323](dclab_rnd/studio/graph.py). Holdout and solution sign-off approvals are off by default (`DEFAULT_POLICY` both False) — [dclab_rnd/studio/graph.py:59](dclab_rnd/studio/graph.py)
- `runtime.run` has no checkpointer; a turn ends on `stop_after` (e.g. a question to the user) and continues in a later `run()` call with `steps_used`; the intern saves its session (messages) after every step and calls `checkpoint()` so a person's Stop ends the turn there — [dclab_rnd/agents/runtime.py:28-41, 95-128](dclab_rnd/agents/runtime.py), [dclab_rnd/intern/loop.py:289-308](dclab_rnd/intern/loop.py)
- Jobs: a heartbeat; a dead worker's job is marked interrupted and settled; a person retries it — not resumed mid-turn — [dclab_rnd/jobs/worker.py:1-13](dclab_rnd/jobs/worker.py)
- Durable pause/resume exists in the orchestra layer: LangGraph checkpoints on PostgreSQL or SQLite, `interrupt` for question/confirmation/approval, `Command(resume=...)` — [dclab_rnd/orchestra/checkpoints.py:1-51](dclab_rnd/orchestra/checkpoints.py), [dclab_rnd/orchestra/__init__.py:1-12](dclab_rnd/orchestra/__init__.py), [dclab_rnd/orchestra/walk.py:108-115](dclab_rnd/orchestra/walk.py), [dclab_rnd/orchestra/solution.py:459-487](dclab_rnd/orchestra/solution.py)
- The team switch is off by default (`"on": False`) — [dclab_rnd/orchestra/settings.py:22](dclab_rnd/orchestra/settings.py); its sealed gate passed in the v6 set (JRN-025 `team_live`, `passed: True`) after failing in v5 (JRN-019) — [evidence/campaigns/journeys_v1/results/](evidence/campaigns/journeys_v1/results/)
- The research campaign also uses LangGraph with `AsyncSqliteSaver` — [dclab_rnd/agentic/engine.py:1-17](dclab_rnd/agentic/engine.py)

**(5) Cost/budget caps**
- Before each request the gateway estimates an upper-bound cost and, under one lock, refuses it if it could pass a run, project, role (purpose), workspace or person cap; the estimate is held until the request ends so parallel requests cannot jointly pass — [dclab_rnd/models/gateway.py:114-169](dclab_rnd/models/gateway.py)
- A refused request reaches no provider and raises `BudgetExceeded`; the intern turns it into `budget_exhausted` — [dclab_rnd/models/gateway.py:427-444](dclab_rnd/models/gateway.py), [dclab_rnd/intern/loop.py:313-319](dclab_rnd/intern/loop.py)
- Limits: a model without a price "cannot be stopped in euros" (only an already-reached cap refuses) — [dclab_rnd/models/gateway.py:118-123](dclab_rnd/models/gateway.py). The held amounts live in a process-local dict (`self._held`), see [dclab_rnd/models/gateway.py:101](dclab_rnd/models/gateway.py)
- Step and time budgets per policy (`max_steps`, `max_seconds`, `max_replies`) are checked before each call, not just each reply — [dclab_rnd/agents/runtime.py:148-159, 185-195](dclab_rnd/agents/runtime.py)
- Clients outside the gateway (the campaign's NOOA client) must call `gateway.record` and `check_external` themselves — [dclab_rnd/models/gateway.py:239-244, 531-537](dclab_rnd/models/gateway.py)
- The twin lane's own cap is checked before a request, and "the last request can pass the euro cap by its own cost" — [dclab_rnd/twin/claude.py:12-15](dclab_rnd/twin/claude.py)
- A dev record found a live run "would have sent up to 30 uncapped, uncounted model requests" because only some roles used the team's advisor — [records/18.8.json](research/llm-fine-tuning/experiments/dev_trajectories/records/18.8.json)

**(6) Multi-agent orchestration**
- 17 named agents (`AGENTS`), 15 roles, 17 single-use model purposes; a test checks every purpose is classified exactly once — [dclab_rnd/agents/harness.py:41-85](dclab_rnd/agents/harness.py), [tests/test_agent_harness.py:71-97](tests/test_agent_harness.py)
- Multi-agent graphs (28.x) are on LangGraph "used thinly": graph, typed state, checkpoints, `interrupt`, `Send` for parallel parts; DCLab's own nodes, tools and gateway — [dclab_rnd/orchestra/__init__.py:1-12](dclab_rnd/orchestra/__init__.py), [dclab_rnd/orchestra/solution.py:210](dclab_rnd/orchestra/solution.py)
- The Home orchestrator is a single `run()` with team tools labelled per specialist (`[{TEAM[WHO[name]]}]` in each description), max 6 steps, 5 replies — [dclab_rnd/draft/orchestrator.py:1395-1406, 1772-1796](dclab_rnd/draft/orchestrator.py)
- Roles are one-shot JSON answers with a deterministic fallback (`dclab_core.agents.run_role`) — [dclab_core/agents/__init__.py:1-48](dclab_core/agents/__init__.py)

**(7) MCP support**
- DCLab is an MCP server (low-level `mcp` server, streamable HTTP, stateless) exposing the project tools; each call is a harness `session("mcp")` step and is kept whole — [dclab_rnd/mcp_server.py](dclab_rnd/mcp_server.py)
- No MCP client was found: `grep -r "mcp.client|ClientSession|streamablehttp_client" dclab_rnd` hits only `storage/models.py` and `accounts/store.py` (unrelated names) — search in [dclab_rnd/](dclab_rnd/)

**(8) Providers, streaming, prompt caching, thinking, parallel tool calls**
- Two transports behind one shape: OpenAI-compatible `ChatClient` and Anthropic `AnthropicClient` (OpenAI-style messages translated both ways) — [dclab_rnd/models/gateway.py:43-48](dclab_rnd/models/gateway.py), [dclab_rnd/models/anthropic_client.py:1-25, 50-98](dclab_rnd/models/anthropic_client.py)
- Streaming on both (retry only before the first word reached the user) — [dclab_rnd/models/gateway.py:402-425](dclab_rnd/models/gateway.py), [dclab_rnd/models/anthropic_client.py:263-274](dclab_rnd/models/anthropic_client.py)
- Prompt caching: last tool and system prompt always marked; automatic breakpoint once a conversation is under way; cache read/write tokens priced and logged — [dclab_rnd/models/anthropic_client.py:16-22, 237-246, 183-190](dclab_rnd/models/anthropic_client.py)
- Extended thinking is supported by the Anthropic client (adaptive or budgeted, thinking blocks sent back during tool use) — [dclab_rnd/models/anthropic_client.py:10-14, 70-71, 101-113](dclab_rnd/models/anthropic_client.py) — but the gateway's transport builds `AnthropicClient` without `thinking`, and the only caller that turns it on is the twin lane — [dclab_rnd/models/gateway.py:44-47](dclab_rnd/models/gateway.py), [dclab_rnd/twin/claude.py:36-58](dclab_rnd/twin/claude.py). The OpenAI-compatible client has no reasoning capture; it only sets `reasoning_effort: "none"` when a model refuses tools otherwise — [dclab_rnd/models/client.py:76-143](dclab_rnd/models/client.py)
- Parallel tool calls: several calls in one reply are accepted, but run one after another in a `for` loop — [dclab_rnd/agents/runtime.py:185-221](dclab_rnd/agents/runtime.py); `tool_choice` is `auto` with no parallel setting — [dclab_rnd/models/anthropic_client.py:244](dclab_rnd/models/anthropic_client.py)
- Structured outputs: `response_format` JSON schema maps to Anthropic `output_config.format`; passed through as-is to OpenAI-compatible endpoints — [dclab_rnd/models/anthropic_client.py:139-145](dclab_rnd/models/anthropic_client.py), [dclab_rnd/models/client.py:218-219](dclab_rnd/models/client.py)
- Small-model robustness: detects tool calls written as text (found with qwen2.5-coder in Ollama) and nudges once — [dclab_rnd/agents/runtime.py:64-74, 170-184](dclab_rnd/agents/runtime.py)
- The runtime's `RunResult.usage` keeps only input/output tokens (cache tokens are in the usage log, not in the run result) — [dclab_rnd/agents/runtime.py:61, 165-167](dclab_rnd/agents/runtime.py)

**(9) Observability**
- In-house spans: a trace id and span ids in a context variable, put into JSON log lines and events; no exporter — [dclab_rnd/observe.py:31-36, 109-121](dclab_rnd/observe.py)
- Prometheus-format counters on a separate port (HTTP, jobs, model requests, tokens, cost) — [dclab_rnd/observe.py:1-15](dclab_rnd/observe.py)
- An agent-metrics ledger (`request_ok`, `output_check`, `step_ok`, …) with a scorecard and an Admin Agents tab — [dclab_rnd/agent_metrics.py:1-25](dclab_rnd/agent_metrics.py), commit 0ff6dc10
- OpenTelemetry packages appear only as transitive dependencies (via `mcp` and `openinference-instrumentation-litellm`) in the agent lock file — [requirements/agent.lock.txt:178-208](requirements/agent.lock.txt). The campaign deliberately disables NOOA's OTLP auto-probe — [dclab_rnd/agentic/agents.py:10-13](dclab_rnd/agentic/agents.py)

**(10) Code size and maintenance burden** (`wc -l`, HEAD)
- Core harness: `agents/` 958 lines (harness 203, runtime 224, registry 159, traces 322); `models/` 2,471 (gateway 579, anthropic_client 277, client 245, routing 259, usage 286, settings 268, …)
- Loops and their agents: `intern/` 1,174; `draft/chat.py` 1,534; `draft/orchestrator.py` 2,004; `draft/coordinator.py` 278; `workspace_chat.py` 195; `mcp_server.py` 146
- Orchestration: `orchestra/` 2,043; campaign `agentic/engine.py`+`agents.py`+`campaign_tools.py` 497; `studio/graph.py` 588 + `studio/team.py` 336; `dclab_core` roles/advisor/orchestra 440
- Capture and operations: `experience/` 1,704; `ledger/capture.py` 872; `observe.py`+`agent_metrics.py` 515; `jobs/` 1,359
- Total of these modules: about 17,100 lines (much of `draft/chat.py` and `draft/orchestrator.py` is domain logic, not harness). Tests listed above: about 4,670 lines.

**(11) Known bugs and weaknesses in history**
- Live keys reaching tests, repeatedly: record 0.1 (intern API test hit OpenAI); 16.2 (acceptance run could send stage records to a live model); 17.2 ("made about 50 live requests" because `DCLAB_NO_LIVE_MODELS=1` was set only by make targets); commit bae85e29 (2026-10-22): `check-quick` never exported the switch and the runner blanked only Anthropic settings, so the owner's OpenAI key from `.env` let a live model plan an intern turn (24 tool calls) and stage explanations "passed silently, spending on the owner's key" — [records 0.1, 16.2, 17.2](research/llm-fine-tuning/experiments/dev_trajectories/records/), [git show bae85e29](.)
- The only hard stop in the gateway is an environment check `DCLAB_NO_LIVE_MODELS == "1"` — [dclab_rnd/models/gateway.py:192-197](dclab_rnd/models/gateway.py)
- Capture holes found and fixed one by one: Home agent made no tracer without a store (19.2); steps never held the model's words (19.2); a tool call that raised left no step (21.1); a role step could name an earlier request (21.1); the campaign's record disagreed with its store (21.1); a failed tool call recorded as ok (19.1); 360-char summaries starving the M5 set (58ccfc48); the gateway kept no thinking and cut prompts at 20,000 chars before 32.10 (5b55f4a7) — [records 19.1, 19.2, 21.1](research/llm-fine-tuning/experiments/dev_trajectories/records/), [git show 5b55f4a7](.)
- Provider quirks handled by hand: a model refusing tools unless reasoning is off, with the retry invisible to the usage log (14.2); Sonnet 5.5 refusing `temperature` (handled by a process-wide `NO_SAMPLING` set) — [records/14.2.json](research/llm-fine-tuning/experiments/dev_trajectories/records/14.2.json), [dclab_rnd/models/anthropic_client.py:116-125, 249-261](dclab_rnd/models/anthropic_client.py)
- A cut-short reply counted as the model declining, which would have scored a broken run and used up the sealed set (18.2) — [records/18.2.json](research/llm-fine-tuning/experiments/dev_trajectories/records/18.2.json)
- Commit 58ccfc48 notes a test failure in `test_agents_traces` under load ("a timing race in the test") — [git show 58ccfc48](.)

### Inferences
- The guard design is sound and unusually strict: a write cannot run without a declared move and a scope guard. This is the harness's best part.
- The capture story is "wide but uneven". There are at least six places that keep agent steps (trace rows, session events, intern session files, Home orchestrator records, workspace-chat records, MCP calls, gateway exchanges, twin transcripts), each with a different cut rule. For SFT this means a builder must know which source is whole for which agent. The intern — the main tool-using agent — has whole tool results in its session file but no whole model exchange (shape only), so its exact prompts as sent (system + tool schemas) are not in one record.
- The cell-value privacy rule is the reason for most cuts. That is a deliberate trade-off, not a bug, but it collides with the owner's "never truncate before saving" rule.
- The repeated live-key incidents show the safety switch is opt-in (an env var) rather than opt-out (tests could only reach a model through an explicit test fixture). An external framework would not fix this by itself.
- Because held reservations are per process, two workers or app instances could together pass a monthly cap by up to one estimate each. This is an inference from `self._held` being an instance attribute; I did not test it.

### Gaps
- I did not run the tests or the replay, so I cannot confirm replay works on current traces end to end.
- I did not measure how often the intern or Home agent receives several tool calls in one reply, so the cost of sequential execution is unknown.
- I could not see whether `model_exchanges/` files exist in real workspaces or how complete they are in practice.

## What would be hard to replace if DCLab adopted an external framework?

### Takeaway
The hard-to-replace parts are not the loop. They are the domain checks and the accounting around it: the workflow validator, the scope guards with declared moves, the gateway's caps/routing/shadowing, the decision ledger capture, and the experience/consent capture rules. The loop itself (`runtime.run`, 224 lines) is small and replaceable.

### Cited Findings
- Workflow validator: 10 nodes, gate logic, holdout approval, transition log as audit trail and training data — [dclab_rnd/studio/graph.py:1-18, 196-273](dclab_rnd/studio/graph.py)
- Project guard wires the validator into every write and logs refusals — [dclab_rnd/intern/tools.py:95-119](dclab_rnd/intern/tools.py)
- Gateway caps across run/project/role/workspace/person with reservation — [dclab_rnd/models/gateway.py:114-190](dclab_rnd/models/gateway.py)
- Routing with an approved reviewer, fallback to the purpose's own tier, and a local shadow whose answer is logged and never used — [dclab_rnd/models/gateway.py:199-219, 350-379](dclab_rnd/models/gateway.py)
- Private-data scope: no model offered to any purpose for confidential projects — [dclab_rnd/models/gateway.py:75-86, 230-237](dclab_rnd/models/gateway.py)
- Ledger capture at the loop: each tool chosen among those offered, and why a run stopped — [dclab_rnd/agents/runtime.py:124-128, 213-215](dclab_rnd/agents/runtime.py), [dclab_rnd/ledger/capture.py](dclab_rnd/ledger/capture.py) (872 lines)
- Purpose classification with per-purpose data limits (`may_see`, `cell_values`, `keeps_text`) tested by `tests/test_model_data_limits.py` — [dclab_rnd/models/settings.py:30-40](dclab_rnd/models/settings.py)
- The orchestra layer already shows the pattern of adopting a framework thinly: LangGraph for graph/checkpoints/interrupts, DCLab for tools, models and records — [dclab_rnd/orchestra/__init__.py:1-12](dclab_rnd/orchestra/__init__.py), [dclab_rnd/orchestra/model.py](dclab_rnd/orchestra/model.py)

### Inferences
- Any external framework would need its model calls routed through DCLab's gateway (or a re-implementation of caps, routing, shadow and capture), and its tool execution routed through `Registry.call` so guards still run. Frameworks that own the model client (as NOOA does in the campaign) break this; the campaign already needed `record`/`check_external` patches.
- The AST tests that forbid tool calls and model clients outside the harness are a strong, cheap protection and would need to be kept and adapted to any framework.

### Gaps
- No measurement exists in the repository of how much effort the orchestra's LangGraph adoption cost compared with building it in-house.

## What is missing that an external framework would give for free?

### Takeaway
Mainly: durable checkpoints inside every agent loop, real OpenTelemetry export, concurrent tool execution, first-class extended-thinking use across agents, an MCP client, and a single standard trace format with full fidelity. Structured outputs and streaming are already present.

### Cited Findings
- No checkpointer in `runtime.run`; a run can only stop and be called again — [dclab_rnd/agents/runtime.py:95-224](dclab_rnd/agents/runtime.py)
- The Home agent rebuilds its messages each turn from the last 12 text messages of the draft; earlier tool calls and results are not sent again — [dclab_rnd/draft/chat.py:1084-1088](dclab_rnd/draft/chat.py)
- Spans are in-house context variables; no OTel exporter — [dclab_rnd/observe.py:109-121](dclab_rnd/observe.py)
- Tool calls in one reply run sequentially — [dclab_rnd/agents/runtime.py:185-221](dclab_rnd/agents/runtime.py)
- Thinking is enabled only in the twin lane — [dclab_rnd/twin/claude.py:36-58](dclab_rnd/twin/claude.py), [dclab_rnd/models/gateway.py:44-47](dclab_rnd/models/gateway.py)
- No MCP client — search in [dclab_rnd/](dclab_rnd/)
- Durable pause/resume and parallel fan-out already come from LangGraph where it is used (orchestra, campaign) — [dclab_rnd/orchestra/checkpoints.py](dclab_rnd/orchestra/checkpoints.py), [dclab_rnd/agentic/engine.py:1-17](dclab_rnd/agentic/engine.py)
- Structured outputs and streaming are already implemented — [dclab_rnd/models/anthropic_client.py:139-145, 263-274](dclab_rnd/models/anthropic_client.py)

### Inferences
- Several of the "missing" features are already available in the repository through LangGraph; the gap is that the main loops (Home, intern, orchestrator chat, workspace chat) do not use it.
- Concurrent tool execution matters little today: Home orchestrator runs at most 6 steps per turn, and most tools are fast reads; heavy stage runs go through jobs.

### Gaps
- I did not compare against a specific external framework's feature list (out of scope; other researchers cover those).

## Is there more than one agent loop, and how consistent are they?

### Takeaway
There is one shared model loop (`runtime.run`) used by four agents, but there are several other ways agents run: harness `session()` scripts, one-shot roles, two LangGraph graphs (orchestra and campaign), a third-party framework (NOOA) in the campaign, and a parallel client in the twin lane. They share the registry, the gateway and the session record, but each keeps its own record of steps with different cut rules.

### Cited Findings
- `runtime.run` callers: Home agent, Home orchestrator, intern, workspace chat — [dclab_rnd/draft/chat.py:1100](dclab_rnd/draft/chat.py), [dclab_rnd/draft/orchestrator.py:1796](dclab_rnd/draft/orchestrator.py), [dclab_rnd/intern/loop.py:310](dclab_rnd/intern/loop.py), [dclab_rnd/workspace_chat.py:156](dclab_rnd/workspace_chat.py)
- Non-loop agents run inside `harness.session` (Home script, intern standard plan, team, campaign, MCP, orchestra spike/walk/solution, tutor, conductor, coordinator, twin) — [dclab_rnd/agents/harness.py:41-61, 154-180](dclab_rnd/agents/harness.py)
- One-shot roles go through `dclab_core.agents.run_role` with a fallback; recorded as `role_step` — [dclab_core/agents/__init__.py](dclab_core/agents/__init__.py), [dclab_rnd/agents/harness.py:191-200](dclab_rnd/agents/harness.py)
- The campaign uses NOOA agents with OpenAI's `ResponsesClient`, outside the gateway's client, so it "cannot run on agents.run()" and reports usage through `gateway.record` — [dclab_rnd/agentic/agents.py:1-13](dclab_rnd/agentic/agents.py), [dclab_rnd/agentic/campaign_tools.py:1-10](dclab_rnd/agentic/campaign_tools.py), [dclab_rnd/models/gateway.py:239-244](dclab_rnd/models/gateway.py)
- The twin lane's `ClaudeClient` and `TwinGateway` replace the gateway for a lane, with their own capture and caps — [dclab_rnd/twin/claude.py:1-20, 251-300](dclab_rnd/twin/claude.py)
- Inconsistencies: Home agent and Home orchestrator both serve chat on Home but have different policies, prompts and records (draft text history vs. a turn record with whole steps); the intern sets `must_act` and a 6,000-char result limit, others use the 4,000 default; the intern's per-session euro cap vs. the orchestrator's `CAP_EUR` default vs. workspace chat's optional `cap_eur` — [dclab_rnd/intern/loop.py:282-287](dclab_rnd/intern/loop.py), [dclab_rnd/draft/orchestrator.py:1778-1780](dclab_rnd/draft/orchestrator.py), [dclab_rnd/workspace_chat.py:146-150](dclab_rnd/workspace_chat.py)
- Validation levels differ by entry point: `schema` in `run()`, `names` for MCP/Toolbox, `none` for Home code calls — [dclab_rnd/agents/registry.py:85-110](dclab_rnd/agents/registry.py), [dclab_rnd/intern/tools.py:213-215](dclab_rnd/intern/tools.py), [dclab_rnd/draft/chat.py:1208](dclab_rnd/draft/chat.py)

### Inferences
- The consistency that matters most (every model call through the gateway; every write through a guard; every step in the session record) is enforced by tests and mostly holds. The exceptions are the campaign (NOOA) and the twin lane, both documented.
- The inconsistency that hurts most for training data is capture: each loop keeps its steps in a different file with different cuts. A single "whole step record" written by `runtime.run` itself (with the cell scrub) would remove most of this duplication.

### Gaps
- I did not read `studio/team.py`, `draft/coordinator.py` or the orchestra nodes line by line; their consistency is judged from their docstrings and harness entries only.
