# Vendor agent SDKs as candidate harnesses for DCLab (Claude Agent SDK, OpenAI Agents SDK, Google ADK, Microsoft Agent Framework), state on 2026-10-22

Method note: I took the version numbers, release dates, stars, licenses, open-issue counts and package layouts directly from the GitHub API and PyPI JSON on 2026-10-22. Feature claims come from official docs pages fetched the same day, from release notes, or from source files in the repos. Independent practitioner write-ups were thin, especially for Microsoft Agent Framework. Those gaps are marked.

## Current versions, maturity and activity

### Takeaway
All four are active. Each shipped a release in the last 10 days. Only Microsoft Agent Framework (MAF) declares a stable 1.x core, and Google ADK is on 2.x after a major runtime rewrite. OpenAI's SDK is still 0.x, where each minor version may break APIs. The Claude Agent SDK is 0.2.x, tagged "Alpha" on PyPI, and ships a new release almost every time the Claude Code CLI it bundles changes.

### Cited Findings
- **Claude Agent SDK (Python)**: latest v0.2.172 (2026-10-22). Repo created 2025-06-11. About 8.3k stars, MIT license on the repo, 580 open issues+PRs (228 open issues). Seven releases between 2026-10-08 and 2026-10-22 — [GitHub releases](https://github.com/anthropics/claude-agent-sdk-python/releases)
- PyPI `claude-agent-sdk` 0.2.172 needs Python >=3.10 and is classified "Development Status :: 3 - Alpha" — [PyPI](https://pypi.org/project/claude-agent-sdk/)
- Almost every Claude Agent SDK release is mostly "Updated bundled Claude CLI to version 2.1.3xx". For example, v0.2.172 bundles CLI 2.1.308 and v0.2.168 bundles CLI 2.1.301 — [GitHub releases](https://github.com/anthropics/claude-agent-sdk-python/releases)
- **OpenAI Agents SDK**: latest v0.24.0 (2026-10-15), with v0.23.0/0.23.1 on 2026-10-02 and v0.22.0 on 2026-08-19. About 30.2k stars, MIT, only 15 open issues (27 issues+PRs) — [GitHub releases](https://github.com/openai/openai-agents-python/releases); [PyPI](https://pypi.org/project/openai-agents/)
- OpenAI's versioning is "a slightly modified version of semantic versioning using the form `0.Y.Z`". The project says the leading 0 means the SDK "is still evolving rapidly", and a minor (`Y`) bump means "breaking changes to any public interfaces that are not marked as beta" — [release.md](https://openai.github.io/openai-agents-python/release/)
- **Google ADK (Python)**: latest v2.12.0 (2026-10-13). v2.0.0 went GA on 2026-05-19 (alphas from 2026-03-18). A 1.x line is still maintained in parallel, with v1.39.1 on 2026-08-27. About 22k stars, Apache-2.0, 185 open issues. Requires Python >=3.11 — [GitHub releases](https://github.com/google/adk-python/releases); [PyPI](https://pypi.org/project/google-adk/)
- ADK 2.0 GA dates: Python 2026-05-19, Go 2026-06-30, TypeScript 2026-08-21 — [ADK 2.0 docs](https://adk.dev/2.0/)
- **Microsoft Agent Framework**: Python 1.0.0 shipped 2026-04-02 (RCs from 2026-02-20), and the latest is python-1.23.0 (2026-10-22). Python and .NET release weekly as separate lines. About 14.3k stars, MIT, 793 open issues+PRs (603 open issues) — [GitHub releases](https://github.com/microsoft/agent-framework/releases)
- Microsoft announced MAF 1.0 GA for .NET and Python on 2026-04-03, with "a commitment to backward compatibility going forward" — [Microsoft devblog](https://devblogs.microsoft.com/agent-framework/microsoft-agent-framework-version-1-0/); [Visual Studio Magazine](https://visualstudiomagazine.com/articles/2026/04/06/microsoft-ships-production-ready-agent-framework-1-0-for-net-and-python.aspx)
- On PyPI, `agent-framework` and `agent-framework-core` 1.23.0 are "5 - Production/Stable". The connectors DCLab would need are still beta: `agent-framework-anthropic` 1.0.0b261022, `agent-framework-ollama` 1.0.0b261022 and `agent-framework-claude` 1.0.0b261015, all "4 - Beta". `agent-framework-openai` 1.17.0 is Stable — [PyPI agent-framework-anthropic](https://pypi.org/project/agent-framework-anthropic/); [PyPI agent-framework-ollama](https://pypi.org/project/agent-framework-ollama/); [PyPI agent-framework](https://pypi.org/project/agent-framework/)

### Inferences
- Measured by declared API stability, MAF core is the most mature. But for DCLab's two model paths, Anthropic and Ollama, MAF's connectors are beta. So MAF's "GA" does not cover the parts DCLab would use.
- The OpenAI SDK has the healthiest issue hygiene (15 open issues against 30k stars). Its release notes now include formal "release readiness reviews".

### Gaps
- I did not compute download counts or contributor counts.
- I did not verify .NET/Go/TypeScript parity. It is not relevant for Python-only DCLab.

## Model support: Anthropic Claude natively AND local/OpenAI-compatible models?

### Takeaway
ADK and MAF both have a native Anthropic client and a route to Ollama/OpenAI-compatible servers. OpenAI's SDK reaches Claude only through beta adapters (LiteLLM or any-llm), or through an OpenAI-compatible endpoint. The Claude Agent SDK is built for Claude. It can reach a local model only by pointing the bundled Claude Code CLI at an Anthropic-compatible server, such as Ollama ≥0.14. Nobody has verified that path for the SDK at production quality.

### Cited Findings
- **Claude Agent SDK**: options are `model`, `fallback_model`, `effort`, `thinking` and `betas`. `ModelUsage.provider` lists `bedrock`, `vertex`, `foundry` and `gateway`. The docs mention `ANTHROPIC_BASE_URL` only for a gateway. No non-Anthropic model is documented — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- The Claude Agent SDK runs the Claude Code CLI as a subprocess. The CLI is bundled, and you can set `cli_path` or a custom `Transport` — [README](https://github.com/anthropics/claude-agent-sdk-python); [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- Ollama has supported the Anthropic Messages API since v0.14.0, so Claude Code can run on local models. Setup: set the base URL to `http://localhost:11434` and use any auth token, or run `ollama launch claude`. Ollama recommends models such as qwen3-coder and a context of at least 32K — [Ollama docs](https://docs.ollama.com/api/anthropic-compatibility); [Ollama blog](https://registry.ollama.ai/blog/claude)
- A third-party comparison calls the Claude Agent SDK "Claude-only" — [Composio comparison](https://composio.dev/blog/claude-agents-sdk-vs-openai-agents-sdk-vs-google-adk)
- **OpenAI Agents SDK**: it uses the Responses API by default. Its docs say "many other LLM providers still do not support it", so you switch to Chat Completions or use `OpenAIChatCompletionsModel` with a custom `base_url`, which is the route for Ollama or vLLM. `LitellmModel` and `AnyLLMModel` are "best-effort, beta". On Chat Completions the SDK may silently drop Responses-only fields unless `strict_feature_validation=True`. Some providers lack `json_schema` structured output, and some LiteLLM providers report no usage unless `include_usage=True` — [OpenAI models docs](https://openai.github.io/openai-agents-python/models/)
- The OpenAI SDK ships these adapter files: `src/agents/extensions/models/litellm_model.py` and `any_llm_model.py`. There is no first-party Anthropic model class — [repo tree](https://github.com/openai/openai-agents-python/tree/main/src/agents/extensions/models)
- **Google ADK**: `src/google/adk/models/` contains `anthropic_llm.py`, whose `class AnthropicLlm(BaseLlm)` uses `anthropic.AsyncAnthropic` and `AsyncAnthropicVertex`, so it works with the direct API or Vertex. It also has `lite_llm.py`, which has Ollama-specific handling (`ollama_chat`, "Ollama via LiteLLM"), and `gemma_llm.py` — [ADK models source](https://github.com/google/adk-python/tree/main/src/google/adk/models)
- ADK docs list connectors for LiteLLM, Ollama, vLLM, LiteRT-LM and Apigee, plus a model router with automatic failover — [ADK models docs](https://adk.dev/agents/models/). Automatic model failover was added in v2.9.0 — [ADK v2.9.0 release](https://github.com/google/adk-python/releases/tag/v2.9.0)
- ADK v2.2.0 BREAKING: the `LlmAgent` default model changed from `gemini-2.5-flash` to `gemini-3-flash-preview`, a preview model — [ADK releases](https://github.com/google/adk-python/releases)
- **MAF**: the Python packages include `anthropic`, `ollama`, `openai`, `gemini`, `bedrock`, `mistral`, `foundry_local` and `claude` — [repo tree](https://github.com/microsoft/agent-framework/tree/main/python/packages). `agent-framework-anthropic` talks to the Anthropic API, adds Foundry/Bedrock/Vertex transports, and supports native system blocks for prompt-cache `cache_control` — [anthropic README](https://github.com/microsoft/agent-framework/tree/main/python/packages/anthropic)
- `agent-framework-claude` wraps the Claude Agent SDK as an Agent Framework agent — [claude README](https://github.com/microsoft/agent-framework/tree/main/python/packages/claude)

### Inferences
- DCLab needs Claude Sonnet 5.5 plus Ollama plus a future fine-tuned Qwen ~8B. Today it already runs through its own `dclab_rnd.models` gateway. ADK (AnthropicLlm + LiteLlm) and MAF (anthropic + ollama/openai connectors) both cover that pair. OpenAI's SDK covers Claude only through beta adapters. The Claude Agent SDK covers local models only through Ollama's Anthropic shim, and that path drives the whole Claude Code agent loop and its prompts, which assume a frontier model.
- A fine-tuned 8B Qwen served by MLX or vLLM through an OpenAI-compatible endpoint fits OpenAI's SDK, ADK (LiteLLM) and MAF (openai connector with a base_url) naturally. It fits the Claude Agent SDK only if the server also speaks the Anthropic protocol.

### Gaps
- I found no official Anthropic statement that the Agent SDK supports non-Anthropic backends. Ollama documents its Anthropic compatibility for Claude Code, not for the SDK. No source measured how well tool calling works with small local models through any of the four.
- MAF's Ollama connector feature limits (tool calling, structured output) were not checked.

## Tool calling and per-call guards (could a state-machine validator refuse a move?)

### Takeaway
All four let custom code inspect a tool call's arguments and refuse it before it runs. OpenAI has tool input/output guardrails. ADK has `before_tool_callback`. MAF has function middleware. The Claude Agent SDK has the `PreToolUse` hook. So DCLab's graph validator (`studio/graph.py`) could sit in that seat in any of them. The Claude Agent SDK carries a trap: `can_use_tool` is NOT called for pre-approved calls, so the guard must be a hook.

### Cited Findings
- **Claude Agent SDK**: you declare tools with `@tool(name, desc, schema)` and serve them through an in-process MCP server built by `create_sdk_mcp_server`. `can_use_tool` "runs only when the permission flow falls through to a prompt" and is not called for calls approved by `allowed_tools`, allow rules or the permission mode. The docs therefore point to a `PreToolUse` hook to gate every tool call. Permission modes are default, acceptEdits, plan, dontAsk, bypassPermissions and auto — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- v0.2.166 added typed hook-specific output classes — [release v0.2.166](https://github.com/anthropics/claude-agent-sdk-python/releases/tag/v0.2.166)
- **OpenAI Agents SDK**: tool guardrails "wrap FunctionTool instances". Input tool guardrails run before the tool and "can skip the call, replace the output with a message, or raise a tripwire". Output tool guardrails run after it. Agent-level input/output guardrails also exist, and MCP servers can attach server-wide guardrails (v0.22.1) — [guardrails docs](https://openai.github.io/openai-agents-python/guardrails/); [v0.22.1 release](https://github.com/openai/openai-agents-python/releases/tag/v0.22.1)
- v0.23.0 now rejects "silently discarded kwargs tool arguments" and enforces explicit strict-tool parameters — [v0.23.0 release](https://github.com/openai/openai-agents-python/releases/tag/v0.23.0)
- **Google ADK**: `before_tool_callback` runs before a tool. Returning a dict skips the tool and hands that dict to the model, while returning None lets the tool run — [Ravi Chaganti, ADK callbacks](https://ravichaganti.com/blog/google-adk-callbacks/) (secondary source; consistent with ADK docs referenced there). In 2.0, overriding legacy `_run_async_impl()` is ignored by the graph engine — [ADK 2.0 docs](https://adk.dev/2.0/)
- **MAF**: function middleware receives a `FunctionInvocationContext` (`function`, `arguments` (validated), `session`, `metadata`, `result`, `kwargs`). It can stop execution by not calling `call_next()` or by raising `MiddlewareTermination` with a result. Chat middleware runs for every model call. "Agent Hooks" give a "fail-closed control boundary" — [MAF middleware docs](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/)
- In MAF, ending the function loop "might result in your chat history being left in an inconsistent state", meaning a function call with no result — [MAF middleware docs](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/)

### Inferences
- For DCLab's "a write tool runs only through its scope's guard" rule, the cleanest seats are OpenAI's tool input guardrail (`reject_content`) and MAF function middleware. With a refusal, both return a message to the model rather than crash the run, which matches how the graph validator refuses a move. ADK's dict-return is equivalent.
- With the Claude Agent SDK, built-in tools (Bash, Edit, etc.) must be disabled or denied explicitly. Otherwise the agent can act outside DCLab's registry.

### Gaps
- I did not test whether a refusal keeps the transcript valid (paired call/result) in each SDK under parallel tool calls.

## Tracing: what is recorded per step, and can full traces be exported for SFT? OpenTelemetry?

### Takeaway
MAF and ADK emit OpenTelemetry GenAI-convention spans natively. MAF records full prompts, tool arguments and results only when `ENABLE_SENSITIVE_DATA=true`. OpenAI's SDK has its own tracing model, which uploads to OpenAI by default, with pluggable processors and no native OTel. The Claude Agent SDK gives full message streams, session transcripts, cost and thinking blocks through its own types and session store. Its OTel comes from the bundled CLI or third-party instrumentors. None of them exports "SFT-ready trajectories" out of the box, so DCLab would still need its own exporter.

### Cited Findings
- **Claude Agent SDK**: `ResultMessage` includes `total_cost_usd`, `usage` (input/output/cache tokens, main loop only, excluding subagents), `model_usage` per model with `costUSD`, `num_turns`, `permission_denials`, `structured_output` and `stop_reason`. Cost is a client-side estimate. `session_store` mirrors transcripts to external storage, and `include_hook_events` streams hook events — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- v0.2.170 fixed parallel tool calls missing from `get_session_messages`/`get_session_messages_from_store`. Before that fix, "only" some calls were returned — [release v0.2.170](https://github.com/anthropics/claude-agent-sdk-python/releases/tag/v0.2.170)
- The Claude Agent SDK's OTel comes from the bundled Claude Code CLI (`CLAUDE_CODE_ENABLE_TELEMETRY`, `OTEL_EXPORTER_OTLP_ENDPOINT`) — [OpenObserve blog](https://openobserve.ai/blog/claude-agent-sdk-observability-opentelemetry.md). Another OpenObserve page says the SDK has no built-in OTel instrumentor and uses manual spans, which partly contradicts the blog — [OpenObserve docs](https://openobserve.ai/docs/integration/ai/frameworks/claude-agent-sdk.md). Third-party OpenInference instrumentation exists — [orq.ai](https://docs.orq.ai/docs/proxy/frameworks/claude-agent-sdk)
- **OpenAI Agents SDK**: spans are agent, generation (it "stores the inputs/outputs of the LLM generation"), function (tool inputs/outputs), guardrail, handoff, task, turn and custom. `trace_include_sensitive_data` defaults to True. The default exporter sends to OpenAI's Traces dashboard. `add_trace_processor()` adds a sink, and `set_trace_processors()` replaces the default. The page does not mention OpenTelemetry — [OpenAI tracing docs](https://openai.github.io/openai-agents-python/tracing/)
- With non-OpenAI models, the OpenAI SDK may error trying to upload traces without an OpenAI key. The fixes are `set_tracing_disabled(True)` or a custom processor — [OpenAI models docs](https://openai.github.io/openai-agents-python/models/)
- The OpenAI SDK's usage object has per-request `request_usage_entries` with cached/cache-write tokens. Cost in currency is not computed — [usage docs](https://openai.github.io/openai-agents-python/usage/)
- **MAF**: emits traces, logs and metrics per the OTel GenAI semantic conventions. The spans are `invoke_agent`, `chat <model>` (prompt and response when sensitive data is on) and `execute_tool <fn>` (args and result when sensitive data is on). The metrics are `gen_ai.client.token.usage` and duration. `ENABLE_SENSITIVE_DATA` defaults to false. MCP trace context is propagated in `params._meta` — [MAF observability docs](https://learn.microsoft.com/en-us/agent-framework/agents/observability)
- Python 1.15.0 [BREAKING] consolidated the OTel semconv modes — [MAF releases](https://github.com/microsoft/agent-framework/releases)
- **Google ADK**: the `src/google/adk/telemetry/` package includes `_genai_instrumentor.py`, `_stable_semconv.py`, `_experimental_semconv.py`, `_token_usage.py`, `node_tracing.py`, `google_cloud.py` and `sqlite_span_exporter.py` — [ADK telemetry source](https://github.com/google/adk-python/tree/main/src/google/adk/telemetry). ADK 2.0 docs say the engine "catches exceptions to support automatic retries, telemetry, and HITL pauses" — [ADK 2.0 docs](https://adk.dev/2.0/)

### Inferences
- For DCLab's "capture everything whole" rule and its SFT corpora, OTel spans are a poor primary record. Content capture is optional, and attribute sizes can be truncated by exporters. The more faithful source is each SDK's own run/session object: OpenAI's `RunResult`/`RunState`, the Claude Agent SDK's message stream and session store, ADK's session events, and MAF's session plus checkpoint state. DCLab's own trace/ledger would still be the canonical store.
- Only the Claude Agent SDK reports a currency cost natively. The others report tokens, so DCLab's `models/prices.py` stays necessary.

### Gaps
- I could not verify which of the four capture extended thinking or reasoning content in traces, except the Claude SDK's `thinking` config. MAF and ADK handling of Anthropic thinking blocks in spans was not confirmed.
- I did not verify whether ADK records full tool results in spans by default.

## Deterministic replay and testing with scripted/fake models

### Takeaway
OpenAI's SDK is clearly ahead here. Since v0.21.0 it has a public, frozen `agents.testing` module with `ScriptedModel`, FIFO steps, call inspection, failure injection and drift detection. MAF checkpoints are documented as "fully replayable" from input and HITL responses (≥1.13.0). ADK 2.0 can resume workflows, with retry semantics. The Claude Agent SDK has no scripted-model utility. Its `testing` package only checks session-store conformance.

### Cited Findings
- OpenAI `agents.testing`: "deterministic, provider-neutral testing utilities... run in memory, make no model ... requests". It includes `ScriptedModel`, `assistant_message()`, `function_call()`, `ModelStep.raise_error()`, `assert_complete()` for workflow drift, and `calls`/`last_call` inspection — [OpenAI testing docs](https://openai.github.io/openai-agents-python/testing/); added in v0.21.0 with "freeze public testing module contracts" — [v0.21.0 release](https://github.com/openai/openai-agents-python/releases/tag/v0.21.0)
- OpenAI RunState resume: v0.24.0 release notes say "Preserve completed work without promising automatic replay". RunState snapshots now use schema 1.21, and older same-name agent graphs should be finished "on their original SDK" — [v0.24.0 release](https://github.com/openai/openai-agents-python/releases/tag/v0.24.0)
- Claude Agent SDK `claude_agent_sdk.testing` exports only `run_session_store_conformance` ("Test utilities for SDK extension authors") — [source](https://github.com/anthropics/claude-agent-sdk-python/tree/main/src/claude_agent_sdk/testing). There is session `resume`, `fork_session` and `resume_session_at` (resume up to a message UUID) — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- MAF: since Python 1.13.0, workflows create entry checkpoints for the input and for delivered HITL responses, which "make the complete workflow run replayable" (a minor breaking change). Storage options are InMemory, File, and Cosmos. A rehydrated workflow "must preserve the topology and executor identities". File/Cosmos storage uses a restricted unpickler — [MAF checkpoints docs](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints)
- ADK v2.9.0 changed resume behavior: a failed node now re-runs on resume, where before it replayed as completed — [search summary of ADK 2.9.0 changelog](https://newreleases.io/project/github/google/adk-python/release/v2.9.0) (secondary). ADK has an `evaluation/` package (eval sets, `llm_as_judge`, `multi_turn_task_success`, `hallucinations_v1`, `final_response_match_v2`) — [ADK evaluation source](https://github.com/google/adk-python/tree/main/src/google/adk/evaluation)

### Inferences
- DCLab's tests already use scripted transports at its gateway. Only the OpenAI SDK offers a public, supported equivalent. In ADK and MAF a fake can be built by subclassing `BaseLlm` or a chat client, which is straightforward but not a frozen contract. The Claude Agent SDK can only be faked at the CLI/transport boundary (`Transport`), which is heavier.
- "Replay" in all four means resuming from saved state. None re-executes a past trace deterministically against recorded model outputs, the kind of replay DCLab's `make replay` does for stages.

### Gaps
- I did not confirm whether ADK ships an official mock/scripted LLM for unit tests.

## Human-in-the-loop: pause for approval, resume later (durable)

### Takeaway
The OpenAI SDK (`needs_approval` → interruptions → serializable `RunState`) and MAF (tool `approval_mode` plus workflow request/response with durable checkpoints) have the most complete durable HITL. ADK's tool confirmation is still marked experimental and does not support `DatabaseSessionService`. The Claude Agent SDK pauses through permission callbacks and `deferred_tool_use`, and resumes through sessions.

### Cited Findings
- OpenAI: `needs_approval=True` or an async policy (it receives the context, parsed params and call ID). Approval works on function tools, `Agent.as_tool`, ShellTool, ApplyPatchTool and local MCP. You call `result.to_state()`, then `state.approve/reject`, then `Runner.run(agent, state)`. Sticky `always_approve`/`always_reject` is available. Approvals are run-wide across handoffs and nested agents — [OpenAI HITL docs](https://openai.github.io/openai-agents-python/human_in_the_loop/)
- v0.24.0 persists deferred approval history across resumes, and automatic compaction waits for approvals — [v0.24.0 release](https://github.com/openai/openai-agents-python/releases/tag/v0.24.0)
- MAF: the `@tool(approval_mode=...)` samples say "Use 'always_require' in production" — [MAF middleware docs](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/). Workflow checkpoints capture "pending requests and responses" and survive restarts with File or Cosmos storage — [MAF checkpoints docs](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints)
- ADK: `FunctionTool(..., require_confirmation=True|callable)` or `tool_context.request_confirmation()`. The caller answers with an `adk_request_confirmation` FunctionResponse. The docs say "The Tool Confirmation feature is experimental". "DatabaseSessionService is not supported", "VertexAiSessionService is not supported", and resume needs `invocation_id` — [ADK tool confirmation docs](https://adk.dev/tools-custom/confirmation/)
- Claude Agent SDK: the `ResultMessage` has `deferred_tool_use` and `permission_denials`. Sessions resume with `resume`/`session_id`, and transcripts can be mirrored with `session_store` — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)

### Inferences
- DCLab already has a job table with `checkpoint()` and PostgreSQL storage. An SDK's HITL would have to be bridged into that store. The OpenAI RunState (JSON-serializable) and MAF's `CheckpointStorage` protocol (custom store possible) are the easiest to put into PostgreSQL.

### Gaps
- I did not verify exactly how the Claude Agent SDK pauses for a person mid-run and resumes in another process (`deferred_tool_use` semantics).

## Cost/budget caps and rate limits

### Takeaway
Only the Claude Agent SDK has a built-in currency cap (`max_budget_usd`, client-side estimate). The others cap turns or LLM calls: OpenAI `max_turns` (default 10) and ADK `max_llm_calls` (default 500). MAF has no documented cap, so a cap there would be middleware. DCLab's own per-run EUR caps and `limits.py` would remain the authority under any SDK.

### Cited Findings
- Claude Agent SDK: `max_budget_usd` "stops the query when the client-side cost estimate reaches this value". The docs advise leaving headroom because the estimate can overshoot. There are also `max_turns` and `task_budget` (API-side token budget) — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- OpenAI: `DEFAULT_MAX_TURNS = 10` in `run_config.py`. Usage is tracked per request, with no currency cap found — [run_config.py](https://github.com/openai/openai-agents-python/blob/main/src/agents/run_config.py); [usage docs](https://openai.github.io/openai-agents-python/usage/)
- ADK: `_DEFAULT_MAX_LLM_CALLS = 500`, overridable with the `ADK_MAX_LLM_CALLS` env var — [run_config.py](https://github.com/google/adk-python/blob/main/src/google/adk/agents/run_config.py)
- MAF: rate limiting is given as a use case for middleware termination ("security checks, rate limiting, or validation failures") — [MAF middleware docs](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/)

### Inferences
- Chat middleware in MAF, a model wrapper in OpenAI/ADK, or a hook in the Claude SDK could all call DCLab's spend ledger before each model call.

### Gaps
- I found no built-in provider rate-limit handling documentation beyond retries. ADK has an `_AnthropicRateLimitError` class in `anthropic_llm.py` ([source](https://github.com/google/adk-python/blob/main/src/google/adk/models/anthropic_llm.py)), but its behavior was not read.

## Multi-agent orchestration

### Takeaway
All four support sub-agents. ADK 2.0 and MAF are the graph-workflow engines. OpenAI keeps two primitives, handoffs and agents-as-tools, and leaves orchestration to Python. The Claude Agent SDK has subagents defined by a prompt and a tool list.

### Cited Findings
- ADK 2.0 Workflow Runtime: "from a hierarchical agent executor to a graph-based execution engine". Agents, tools and functions are nodes, with routing, fan-out/fan-in, loops, retry, dynamic nodes, HITL and nested workflows. YAML graph authoring arrived in v2.9.0 — [ADK 2.0 docs](https://adk.dev/2.0/); [v2.9.0 release](https://github.com/google/adk-python/releases/tag/v2.9.0). The `workflow/` source has `_graph.py`, `_join_node.py`, `_tool_node.py` and `_retry_config.py` — [source](https://github.com/google/adk-python/tree/main/src/google/adk/workflow)
- In ADK 2.0 TypeScript, `SequentialAgent`, `ParallelAgent` and `LoopAgent` are deprecated — [ADK 2.0 docs](https://adk.dev/2.0/)
- MAF: graph workflows built with `WorkflowBuilder` run in supersteps. The orchestration builders are Concurrent, Sequential, GroupChat, Handoff and Magentic — [MAF checkpoints docs](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints)
- OpenAI: handoffs and `Agent.as_tool()`. One write-up says it "leaves orchestration to standard Python" — [OpenAI HITL docs](https://openai.github.io/openai-agents-python/human_in_the_loop/); [Composio comparison](https://composio.dev/blog/claude-agents-sdk-vs-openai-agents-sdk-vs-google-adk)
- Claude Agent SDK: `agents: dict[str, AgentDefinition]` with description, prompt, tools, model, maxTurns, background, permissionMode, etc. `usage` excludes subagents, so use `model_usage` instead — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)

### Inferences
- DCLab's coordinator state machine (`dclab_core/coordinator.py`) and the WF-01…WF-10 graph are already deterministic code. Moving them into ADK or MAF graphs would duplicate validated logic. Using an SDK only for the inner model-plus-tools loop of each role keeps DCLab's graph as the authority.

### Gaps
- I found no independent benchmark comparing multi-agent reliability across the four.

## MCP support (client/server)

### Takeaway
All four are MCP clients. MAF ships a dedicated MCP hosting package and ADK has an `mcp_server` module. The Claude Agent SDK's own custom tools are in-process MCP servers. OpenAI adds guardrails and approvals to MCP tools.

### Cited Findings
- Claude Agent SDK: `mcp_servers`, `strict_mcp_config` and `create_sdk_mcp_server` (in-process) — [Agent SDK Python reference](https://code.claude.com/docs/en/agent-sdk/python)
- OpenAI: MCP docs page, server-wide MCP guardrails (v0.22.1), MCP approval policies and `mcp_annotations` (v0.24.0) — [docs index](https://github.com/openai/openai-agents-python/tree/main/docs); [v0.22.1](https://github.com/openai/openai-agents-python/releases/tag/v0.22.1); [v0.24.0](https://github.com/openai/openai-agents-python/releases/tag/v0.24.0)
- ADK: `tools/mcp_tool`, `tools/mcp_server` and `tools/mcp_resource_tool`. v2.9.0 adds MCP SDK 2.x compatibility — [ADK tools source](https://github.com/google/adk-python/tree/main/src/google/adk/tools); [v2.9.0](https://github.com/google/adk-python/releases/tag/v2.9.0)
- MAF: `MCPStdioTool`, `MCPStreamableHTTPTool` and `MCPWebsocketTool` (client) with trace propagation, plus a `hosting-mcp` package. 1.19.0 [BREAKING] deprecated the MCP sampling callback — [MAF observability docs](https://learn.microsoft.com/en-us/agent-framework/agents/observability); [repo packages](https://github.com/microsoft/agent-framework/tree/main/python/packages); [MAF releases](https://github.com/microsoft/agent-framework/releases)

### Inferences
- DCLab already serves its toolbox at `/mcp`. Any of the four could consume it as-is, which gives a low-cost way to trial an SDK without moving tools.

### Gaps
- I did not verify the exact version of the MCP specification each SDK supports.

## Python quality, license, lock-in and API churn

### Takeaway
All four are permissively licensed (MIT or Apache-2.0). Churn is high everywhere. MAF ships "[BREAKING]" items in most weekly minors despite its 1.0 compatibility promise, though many are in hosting, AG-UI or experimental parts. ADK had a full 2.0 rewrite in May 2026, plus later breaking changes. OpenAI breaks on minors by policy. The Claude Agent SDK is Alpha and coupled to the fast-moving Claude Code CLI. Lock-in is highest with the Claude Agent SDK (one model vendor plus a CLI runtime), and next with ADK, whose defaults and integrations lean toward Gemini and Google Cloud.

### Cited Findings
- Licenses: Claude Agent SDK MIT, OpenAI MIT, ADK Apache-2.0, MAF MIT — [GitHub API repo metadata for each repo](https://github.com/anthropics/claude-agent-sdk-python), [openai](https://github.com/openai/openai-agents-python), [google](https://github.com/google/adk-python), [microsoft](https://github.com/microsoft/agent-framework)
- MAF Python [BREAKING] entries appear in 1.13.0 (replayable checkpoints), 1.14.0, 1.15.0 (OTel semconv), 1.17.0 (middleware inputs), 1.18.0, 1.19.0 (MCP), 1.20.0, 1.22.0 and 1.23.0 (AG-UI auth scope, Docker shell args) — [MAF releases](https://github.com/microsoft/agent-framework/releases)
- ADK breaking items after 2.0: v2.2.0 (default model changed to a preview Gemini; GenAI SDK v2 "turns"→"steps" rename), v2.5.0 (Skill Registry endpoint) and v2.7.0 (pyarrow moved out of the `gcp` extra). The 2.0 migration changed the event schema (`node_info`/`output`), so 2.0 sessions are unreadable by ADK <1.28 — [ADK releases](https://github.com/google/adk-python/releases); [ADK 2.0 docs](https://adk.dev/2.0/)
- OpenAI: v0.22.0 breaking (`OpenAIProvider` arg conflicts) and v0.24.0 compatibility changes for error callbacks and saved agent graphs (RunState schema 1.21) — [release.md](https://openai.github.io/openai-agents-python/release/); [v0.24.0](https://github.com/openai/openai-agents-python/releases/tag/v0.24.0)
- v0.21.0 moved to `openai>=3.0.0,<4` (HTTPX2) — [v0.21.0](https://github.com/openai/openai-agents-python/releases/tag/v0.21.0)
- Claude Agent SDK v0.2.129 breaking: skill-name validation (an `--allowedTools` injection fix). The repo's own CI had to pin models after "the CLI's new default model" changed (v0.2.159, v0.2.163) — [Claude Agent SDK releases](https://github.com/anthropics/claude-agent-sdk-python/releases)
- Python floors: Claude/OpenAI/MAF >=3.10, ADK >=3.11 — [PyPI](https://pypi.org/project/google-adk/)

### Inferences
- DCLab's `.venv-agent` is Python 3.12/3.13, which is compatible with all four.
- Under any SDK, DCLab should pin exact versions and wrap the SDK behind its existing `agents/harness.py` interface. Churn in all four is weekly.

### Gaps
- I found no source on how the Claude Agent SDK's bundled CLI is licensed separately from the MIT SDK wrapper. This should be checked before shipping it inside a product.

## Notable weaknesses and practitioner complaints

### Takeaway
Independent, recent, critical practitioner write-ups are scarce. Most "comparisons" are vendor or blog marketing. The concrete complaints found are: the Claude Agent SDK's subprocess fragility (startup latency, init timeouts, zombie processes tied to CLI versions); OpenAI's tracing-to-OpenAI default and beta non-OpenAI adapters; ADK's experimental HITL and breaking 2.0 migration; and MAF's large open-issue count and beta non-Azure connectors.

### Cited Findings
- Claude Agent SDK: 80+ second startup on Windows, blamed on CLI subprocess start — [claudeissues.com #16005](https://claudeissues.com/issue/16005-claude-agent-sdk-80-second-startup-time-on-windows). SDK 0.1.18 with CLI 2.1.9 hit "Control request timeout: initialize", fixed by downgrading the CLI. Zombie processes used 60–70% CPU — [#18667](https://claudeissues.com/issue/18667-2-1-9-breaks-sdk-subprocess-initialization-control-request-timeout-initialize); [#18666](https://claudeissues.com/issue/18666-sdk-subprocess-hangs-on-initialize-leaving-zombie-processes-that-block-subsequen). An intermittent 3+ minute silent hang was reported in May 2026 — [#58503](https://claudeissues.com/issue/58503-claude-cli-subprocess-silent-hang-zero-stdout-for-3-minutes-intermittent). v0.2.111 fixed orphaned child processes on cancellation and NDJSON lines over 64 KiB — [newreleases v0.2.111](https://newreleases.io/project/github/anthropics/claude-agent-sdk-python/release/v0.2.111). (These are third-party issue mirrors; some may be closed or fixed in current versions.)
- Claude Agent SDK, still in Oct 2026: a fix for follow-up turns failing after background subagents (v0.2.160) and a fix for parallel tool calls missing from session messages (v0.2.170) — [releases](https://github.com/anthropics/claude-agent-sdk-python/releases)
- OpenAI: traces go to OpenAI by default and sensitive data is on by default. Non-OpenAI adapters are "best-effort, beta", and Chat Completions can silently drop fields — [tracing docs](https://openai.github.io/openai-agents-python/tracing/); [models docs](https://openai.github.io/openai-agents-python/models/). Its local sandbox should not be described "as complete host or network isolation" — [v0.24.0 release](https://github.com/openai/openai-agents-python/releases/tag/v0.24.0)
- ADK: tool confirmation is experimental and unsupported on DB/Vertex session services — [ADK confirmation docs](https://adk.dev/tools-custom/confirmation/). A silent default-model switch to a preview model came in v2.2.0 — [ADK releases](https://github.com/google/adk-python/releases)
- MAF: 603 open issues. The Anthropic, Ollama and Claude connectors are still beta six months after GA. Its docs and samples are Azure/Foundry-first (`FoundryChatClient`, `AzureCliCredential` in most samples), and pickle-based checkpoint storage is called "a trust boundary" — [PyPI](https://pypi.org/project/agent-framework-anthropic/); [MAF middleware docs](https://learn.microsoft.com/en-us/agent-framework/agents/middleware/); [MAF checkpoints docs](https://learn.microsoft.com/en-us/agent-framework/workflows/checkpoints)
- One survey argues the frameworks are similar ReAct-style loops. In its view the real differences are the host provider and integration depth — [morphllm](https://www.morphllm.com/ai-agent-framework) (secondary, via search summary)

### Inferences
- For DCLab specifically: a Claude-primary product with local and fine-tuned models, a strict validator on every move, whole-trajectory capture for SFT, and scripted-model tests. On those needs, no vendor SDK clearly beats the in-house harness. The OpenAI Agents SDK fits testing and HITL best but is weakest on native Claude. MAF fits OTel, middleware and durable workflows best, but its Anthropic and Ollama connectors are beta. ADK has native Claude plus LiteLLM but experimental HITL and Gemini-leaning defaults. The Claude Agent SDK gives the strongest Claude-native tooling and native cost caps, but adds a CLI subprocess, offers no scripted-model testing, and supports local models only through Ollama's Anthropic shim.
- A low-risk experiment would be to wrap one DCLab role (for example the intern) with each candidate behind `agents/harness.py`, consuming DCLab's existing `/mcp` tools. Then compare captured trajectories on the dev journeys, without replacing the harness.

### Gaps
- I found no independent 2026 practitioner review of MAF Python in production. I found no head-to-head reliability or cost benchmark of the four on the same tasks. I did not check Reddit or Hacker News threads for late-2026 sentiment.
