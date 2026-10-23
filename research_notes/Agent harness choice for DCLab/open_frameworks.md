# Open-source agent frameworks as candidate harnesses for DCLab (as of 2026-10-22)

Scope: LangGraph (LangChain), Pydantic AI, Hugging Face smolagents, CrewAI, DSPy, Letta, Mastra.
Method: GitHub API snapshots taken on 2026-10-22 (stars, licence, last push, releases), official docs fetched the same day, plus a few practitioner write-ups. Anything not checked this session is flagged under Gaps. Repo stats come from `gh api repos/<owner>/<repo>` on 2026-10-22.

## 1. Current versions, maturity and community health

### Takeaway
LangGraph (1.2.x), Pydantic AI (2.x, with 1.x still patched), CrewAI (1.15.x), DSPy (3.4.x) and Mastra (core 1.79) all ship releases weekly or monthly. smolagents has had no release since May 2026; since then its commits are mostly CI and docs. The Python Letta server has been **retired**: active work moved to the TypeScript `letta-code`. So smolagents is a weak choice for a long-lived harness, and the Python Letta server is not an option at all.

### Cited Findings
- **LangGraph**: MIT, ~43.7k stars, 7.5k forks, 508 open issues, pushed 2026-10-22. Latest `1.2.16` (2026-10-16), and `prebuilt==1.1.1` (2026-10-15). Stable `1.0.0` came out 2025-10-17, and it has released about weekly since (1.2.12 on 09-21 through 1.2.16 on 10-16) — [GitHub langchain-ai/langgraph releases](https://github.com/langchain-ai/langgraph/releases)
- **Pydantic AI**: MIT, ~20.8k stars, 1,556 open issues (the highest open-issue count here), pushed 2026-10-22. `v2.59.0` (2026-10-21) and `v1.107.9` (2026-10-21) are both maintained. V2 went stable on 2026-06-23 "after seven betas". It leans into "a harness-first design with capabilities as a core primitive" — [Pydantic AI v2.0.0 release](https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0)
- Pydantic AI ships a new minor version almost daily (v2.57.0 on 10-16, v2.58.0 on 10-20, v2.59.0 on 10-21). That means fast change, so pin exact versions — [GitHub releases](https://github.com/pydantic/pydantic-ai/releases)
- **smolagents**: Apache-2.0, ~30k stars, 992 open issues. Last release `v1.26.0` (2026-05-29), before that v1.25.0 (2026-05-14) and v1.24.0 (2026-01-16). Only 13 commits since 2026-05-30. The recent ones are "Pin GitHub Actions to commit SHAs", "harden GitHub Actions workflows", "Update security policy" and doc fixes. No feature work since May — [GitHub huggingface/smolagents commits](https://github.com/huggingface/smolagents/commits/main)
- **CrewAI**: MIT, ~60k stars (the most here), 827 open issues. `1.15.28` (2026-10-14), with about 5 patch releases in one week (1.15.23–1.15.28, 09-28 to 10-14). `1.0.0` came out 2025-10-20 — [GitHub crewAIInc/crewAI releases](https://github.com/crewAIInc/crewAI/releases)
- **DSPy**: MIT, ~38.7k stars, 823 open issues. `3.4.1` (2026-10-12), `3.4.0` (2026-09-25), `3.0.0` (2025-08-12) — [GitHub stanfordnlp/dspy releases](https://github.com/stanfordnlp/dspy/releases)
- **Letta**: the `letta-ai/letta` README now says the current source "lives in `letta-ai/letta-code`". It adds that the `archive` branch "contains the retired Letta V1 API server", and "active projects should use the current source". The last Python release is `0.16.8` (2026-05-14), a security fix: "use JSON instead of pickle for sandbox->server tool result transport". Since then only README and policy commits — [letta-ai/letta README](https://github.com/letta-ai/letta)
- **letta-code**: TypeScript, Apache-2.0, ~3.7k stars, created 2025-10-25. Very active (v0.34.21 on 2026-10-22). It installs with `npm install -g @letta-ai/letta-code`. It offers a TUI, an App Server, channels (Slack/Telegram/Discord), a TypeScript "Letta Agent SDK", and Letta Cloud — [letta-ai/letta-code](https://github.com/letta-ai/letta-code)
- **Mastra**: ~29k stars, 605 open issues, `@mastra/core@1.79.0` (2026-10-21), about 6 minor versions in 3 weeks. Licence: Apache-2.0 except directories named `ee/` (e.g. `@mastra/core/auth/ee`, `@mastra/core/agent-builder/ee`, `@mastra/editor/ee`), which have their own `ee/LICENSE` (an open-core split) — [mastra LICENSE.md](https://github.com/mastra-ai/mastra/blob/main/LICENSE.md)
- Mastra is TypeScript/JavaScript only. A search found no Python SDK. Its docs mark some workflow pages as legacy/v0 and the current ones as v1 — [Mastra suspend & resume docs](https://mastra.ai/docs/workflows/suspend-and-resume.md)

### Inferences
- For a Python platform that plans to run for years, the "alive and Python" set is LangGraph, Pydantic AI, CrewAI and DSPy. Letta (now TypeScript) and Mastra (TypeScript) would need a second runtime and a cross-language boundary (HTTP/MCP). smolagents is close to maintenance mode.
- Pydantic AI's near-daily minor versions and parallel v1/v2 lines mean a lot of upkeep. On the other hand, the maintainers have kept an old major line patched for months, which is a good sign.

### Gaps
- I did not measure how fast issues get answered or closed per repo (it needs issue-level sampling). Open-issue counts alone say little: Letta shows 0 because its repo is archived in practice.
- I could not confirm what `ee/LICENSE` in Mastra actually allows (I did not fetch it).

## 2. Model support: Anthropic Claude natively and local / OpenAI-compatible models

### Takeaway
Pydantic AI, CrewAI, smolagents and LangGraph (via LangChain integrations) all speak to Claude and to OpenAI-compatible or local endpoints. Pydantic AI and CrewAI have native Anthropic adapters in their own repos. DSPy is in the middle of an "LM transition" (3.4 → 3.5) that breaks custom-LM integrations.

### Cited Findings
- Pydantic AI has a native `models/anthropic.py`, an Ollama provider (`providers/ollama.py`), `models/test.py` (TestModel) and `models/function.py` (FunctionModel). All were checked in the repo — [pydantic-ai repo, pydantic_ai_slim/pydantic_ai/models](https://github.com/pydantic/pydantic-ai/tree/main/pydantic_ai_slim/pydantic_ai/models)
- Pydantic AI's V2 release fixed an Anthropic-specific bug ("Fix `AnthropicModel.count_tokens` divergence on `ToolSearchTool` replay history"). It also handles thinking/reasoning per provider (Groq qwen3, Cerebras, DeepSeek `reasoning_content`) — [v2.0.0 release](https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0)
- CrewAI has a native Anthropic provider at `lib/crewai/src/crewai/llms/providers/anthropic/completion.py` — [crewAI repo](https://github.com/crewAIInc/crewAI/tree/main/lib/crewai/src/crewai/llms/providers)
- smolagents model classes: `VLLMModel`, `MLXModel` (Apple silicon), `TransformersModel`, `LiteLLMModel`, `LiteLLMRouterModel`, `InferenceClientModel`, `OpenAIModel`, `AzureOpenAIModel`, `AmazonBedrockModel`. Claude goes through LiteLLM or Bedrock; there is no native Anthropic class — [smolagents models.py](https://github.com/huggingface/smolagents/blob/main/src/smolagents/models.py)
- DSPy 3.4 "introduces native LM engines". It says: "3.4 is the LM transition release; 3.5 is the migration deadline". Legacy custom-LM integrations and OpenAI-style `messages=` calls still work in 3.4 but give deprecation warnings. The `dspy/clients` folder now has `engines`, `lm_local.py`, `_litellm.py`, `costs.py`, `openai.py` — [DSPy 3.4.0 release](https://github.com/stanfordnlp/dspy/releases/tag/3.4.0)
- DSPy 3.4 also adds "Jev integration through TypeSafe": experimental `Noul`/`Choice`/`Score` decision types with probabilities and confidence, needing `TYPESAFE_API_KEY` — [DSPy 3.4.0 release](https://github.com/stanfordnlp/dspy/releases/tag/3.4.0)

### Inferences
- Only smolagents has an MLX model class. That matters for running the planned ~8B Qwen fine-tune on an M1 Max inside the harness. With any other framework, the same model would be served through Ollama or an OpenAI-compatible server (e.g. mlx-lm's server), which they all support.
- DCLab already sends every request through its own gateway (`dclab_rnd.models`), with prompt caching, prices and shadowing. Any framework would need to plug into that gateway as a custom model, not replace it. Pydantic AI (`FunctionModel` / custom `Model`) and LangChain (`BaseChatModel`) both allow this. DSPy's custom-LM API is changing right now, which is a risk.

### Gaps
- I did not re-verify LangChain's Anthropic and Ollama integrations this session: they are no longer at `libs/partners/anthropic` / `libs/partners/ollama` in the langchain monorepo, so they have probably moved to separate repos. The `langchain-anthropic` package is widely used, but its current home and version are unverified.
- I did not verify whether CrewAI's native Anthropic provider supports prompt-cache control, or whether DSPy's new engines do.

## 3. Tool calling: declaration, a guard/validator before every call, typed outputs

### Takeaway
Pydantic AI has the finest-grained, documented pre-execution hooks: validation and execution are separate phases, with typed "skip / retry / fail / defer" exceptions. That fits DCLab's "graph validator refuses a move" rule best. LangChain v1 middleware (`wrap_tool_call`) and CrewAI tool hooks can also block calls. smolagents only has step callbacks and final-answer checks, with no documented pre-tool guard.

### Cited Findings
- Pydantic AI tool hooks: `before_tool_validate`, `wrap_tool_validate`, `after_tool_validate`, `on_tool_validate_error`, `before_tool_execute`, `wrap_tool_execute`, `after_tool_execute`, `on_tool_execute_error`. They can be filtered with `tools=` and fire for function tools only — [Pydantic AI hooks docs](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- What those hooks can do before a call runs:
  - Raise `SkipToolExecution(result)` to supply a replacement result.
  - Raise `ModelRetry` to send a retry prompt (counts against `max_retries`).
  - Raise `ToolFailed` to return a definitive failure (no retry budget used).
  - Raise `ApprovalRequired` / `CallDeferred` to pause for approval or outside execution.
  - Use a wrap hook that never calls its handler to skip the inner lifecycle.

  — [Pydantic AI hooks docs](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- Pydantic AI V2 "capabilities" bundle tools, hooks, instructions and model settings. The built-ins include Guardrails (Harness: "Validate/block/redact user input, tool calls, tool results, and output"), Tool approval, Prepare Tools, Process History and Subagents — [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- LangChain v1 middleware hooks: node-style `before_agent`, `before_model`, `after_model`, `after_agent`, and wrap-style `wrap_model_call`, `wrap_tool_call`. With a wrap hook, "you decide if the handler is called zero times (short-circuit), once…, or multiple times". `wrap_tool_call` returns `ToolMessage | Command`. Node hooks can `jump_to` `'end'|'tools'|'model'` — [LangChain custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom)
- LangChain built-in middleware includes `HumanInTheLoopMiddleware`, `ToolCallLimitMiddleware`, `ModelCallLimitMiddleware`, `ToolRetryMiddleware`, `ModelFallbackMiddleware`, `SummarizationMiddleware`, `PIIMiddleware`, `ContextEditingMiddleware`, `SubAgentMiddleware`, `RubricMiddleware` (beta), and `LLMToolEmulator` ("Replaces tool execution with LLM-generated responses for testing") — [LangChain built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- CrewAI tool hooks: `@before_tool_call` / `@after_tool_call` (legacy), and the recommended `@on(InterceptionPoint.PRE_TOOL_CALL)` / `POST_TOOL_CALL`. A before hook that returns `False` blocks the call, and the agent gets "Tool execution blocked by hook". `HookAborted` does the same but carries a reason. `request_human_input(...)` collects approval from the terminal — [CrewAI tool hooks](https://docs.crewai.com/en/learn/tool-hooks)
- smolagents `MultiStepAgent` options: `step_callbacks` (per step, optionally per `MemoryStep` type), `final_answer_checks` ("validation functions to run before accepting a final answer"), and `max_steps` (default 20) — [smolagents agents.py](https://github.com/huggingface/smolagents/blob/main/src/smolagents/agents.py)

### Inferences
- DCLab's rule is that "a write tool declares its move and runs only through its scope's guard". That maps onto Pydantic AI's `before_tool_execute` raising `ToolFailed` (refused move, reason shown to the model) or `ModelRetry` (try again). In LangChain it maps onto `wrap_tool_call` returning a refusal `ToolMessage`. In CrewAI it maps onto `HookAborted`.
- smolagents' main mode is `CodeAgent`, which writes Python that calls tools. A refusal would have to live inside each tool's own code. That is a weak fit for a central validator.

### Gaps
- I found no doc example of `wrap_tool_call` returning a refusal `ToolMessage` without running the tool (the signature allows it).
- I did not check tool guards for DSPy `ReAct` / the new "async ReActV2" (3.4); its docs were not fetched.
- I did not check structured/typed output for each framework this session. From prior knowledge only (unverified): Pydantic AI `output_type`, LangChain `response_format`, CrewAI `output_pydantic`, DSPy typed signatures.

## 4. Tracing and trace export (per-step messages, tool args/results, reasoning, tokens, cost); OpenTelemetry; paid-service dependence; use for SFT

### Takeaway
All the Python frameworks can emit OpenTelemetry (OTel, an open standard for traces). None strictly needs a paid service. But each vendor steers you to its own: Logfire for Pydantic AI, LangSmith for LangGraph, CrewAI's own telemetry endpoint. None ships a "trajectory → SFT JSONL" exporter. DCLab would keep its own recorder (ledger / experience log) and read the framework's message history or spans. CrewAI's telemetry-on-by-default is a real privacy and egress risk.

### Cited Findings
- Pydantic AI's Instrumentation capability emits "OpenTelemetry GenAI spans for every model and tool call; the raw material for Logfire traces" — [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- Pydantic AI `RequestUsage` has `cost: Decimal | None` ("Best-effort cost in USD, or None if no cost could be determined"). Usage includes the tokens and cost of responses a `FallbackModel` rejected — [pydantic-ai usage.py](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/usage.py)
- LangGraph/LangChain OTel tracing goes through `langsmith[otel]` (>=0.3.18, 0.4.25+ recommended). For self-hosted LangSmith the endpoint is `<host>/api/v1/otel`. Fan-out to other backends works with an OTel Collector — [LangSmith trace with OpenTelemetry](https://docs.langchain.com/langsmith/trace-with-opentelemetry)
- smolagents supports OTel via `openinference.instrumentation.smolagents.SmolagentsInstrumentor`, with Arize Phoenix (local server), Langfuse (cloud or self-hosted) and MLflow `mlflow.smolagents.autolog()` (spans, inputs/outputs, token usage) — [smolagents inspect runs](https://huggingface.co/docs/smolagents/main/en/tutorials/inspect_runs)
- CrewAI sends anonymous telemetry by default. You turn it off with `CREWAI_DISABLE_TELEMETRY` or `OTEL_SDK_DISABLED`. Expanded data is sent when `share_crew` is enabled — [CrewAI telemetry docs v1.15.28](https://docs.crewai.com/v1.15.28/en/telemetry.md)
- CrewAI users report a timeout to `telemetry.crewai.com:4319/v1/traces` on corporate networks even after setting both variables. One said they removed the code from the source; another said the `.env` setting alone did not work. One called the default "deceptive" — [CrewAI community: Disable Telemetry for production](https://community.crewai.com/t/disable-telemetry-for-production/4989); [community: telemetry.crewai.com](https://community.crewai.com/t/telemetry-crewai-com/6828)
- DSPy 3.4's `clients/` has `costs.py` and `call_result.py`, which suggests per-call cost accounting — [dspy/clients](https://github.com/stanfordnlp/dspy/tree/main/dspy/clients)
- A third-party comparison (June 2026, not vendor docs) recommends Langfuse for open-source, self-hosted, framework-agnostic OTel — [inference.net Langfuse vs LangSmith](https://inference.net/content/langfuse-vs-langsmith)

### Inferences
- For SFT of agent trajectories, OTel GenAI spans are a lossy carrier: attributes may be cut short, and reasoning blocks are often left out. DCLab's standing rule is "record full input+output first". So the safer design is to read the framework's in-memory message list per step (Pydantic AI `all_messages()`, LangGraph state/checkpoints, smolagents `agent.memory`) into DCLab's own store, and treat OTel as observability only.
- CrewAI's default outbound telemetry conflicts with DCLab's consent model (records leave only with consent). Adopting it would need a pinned, audited opt-out.

### Gaps
- I did not verify whether Pydantic AI or LangGraph spans include Anthropic thinking blocks and full tool results by default, or whether they are cut short.
- No framework docs I fetched describe a built-in export to SFT/chat-JSONL format.
- I did not check DSPy tracing (MLflow autolog / `dspy.inspect_history`) this session.

## 5. Deterministic replay, checkpoints, testing against a scripted model; human-in-the-loop with durable pause/resume

### Takeaway
LangGraph is still the reference for checkpoints and resume (thread-scoped checkpointers on SQLite or Postgres, time travel). Pydantic AI V2 has caught up on durable execution through outside engines (Temporal, DBOS, Prefect and more) and has first-class fake models (`TestModel`, `FunctionModel`). CrewAI's human input is terminal-based in the hooks doc. Mastra has durable suspend/resume, but in TypeScript.

### Cited Findings
- LangGraph checkpointers: `InMemorySaver` (lost on restart), `SqliteSaver` (development), and `PostgresSaver` / `AsyncPostgresSaver` (production; `thread_id` under 255 characters). They support "conversation continuity, human-in-the-loop workflows, time travel, and fault tolerance". Checkpoints pile up and need pruning. Each subgraph has its own checkpoint namespace — [LangGraph persistence docs](https://docs.langchain.com/oss/python/langgraph/durable-execution)
- LangChain `HumanInTheLoopMiddleware(interrupt_on={...})` pauses before named tools. Inside a LangGraph `StateGraph` it "travels with the agent node" — [LangChain middleware overview](https://docs.langchain.com/oss/python/langchain/middleware/overview)
- Pydantic AI durable execution supports Temporal, DBOS and Prefect, plus Restate and Airflow. The Harness package adds AWS Lambda, Absurd, pgtask, Render Workflows and Step Persistence. "Tool approval" and "Handle Deferred Tool Calls" cover human-in-the-loop — [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- Pydantic AI V2 maps AG-UI interrupts to `DeferredTools` — [v2.0.0 release](https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0)
- Pydantic AI has `TestModel` (`models/test.py`) and `FunctionModel` (`models/function.py`), the latter used in the hooks docs — [pydantic-ai repo](https://github.com/pydantic/pydantic-ai/tree/main/pydantic_ai_slim/pydantic_ai/models); [hooks docs](https://pydantic.dev/docs/ai/core-concepts/hooks/)
- LangChain ships `LLMToolEmulator`, which fakes tool results with an LLM for testing — [LangChain built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- Mastra workflows `suspend()` at a step and `resume()` with `resumeData`. Snapshots live in the configured storage and "persist across deployments and application restarts" — [Mastra suspend & resume](https://mastra.ai/docs/workflows/suspend-and-resume.md)
- CrewAI's documented approval path is `request_human_input`, which "collects input from the terminal" — [CrewAI tool hooks](https://docs.crewai.com/en/learn/tool-hooks)

### Inferences
- DCLab already has scripted transports in tests and a job table with `checkpoint()`. Pydantic AI's `FunctionModel` fits that "no live key in tests" rule directly. LangGraph needs a fake `BaseChatModel` (LangChain has fake chat models, not verified this session).
- Pydantic AI's durable execution needs an outside engine (Temporal/DBOS/…). DCLab already uses PostgreSQL, so DBOS (Postgres-backed) or pgtask would be the natural choice; adding Temporal would be a new service. This is my inference, not tested.
- No framework promises fully deterministic replay of LLM outputs. Replay means rerunning from a saved state with recorded model answers. Only LangGraph documents "time travel" from a checkpoint.

### Gaps
- I did not fetch LangGraph's `interrupt()` / `Command(resume=...)` and durability-mode docs (the persistence page did not cover them).
- I did not verify CrewAI Flows persistence (`@persist`) or a non-terminal human-in-the-loop mode this session.
- I did not check DSPy for checkpoint/resume or human-in-the-loop (it is mainly an optimization and programming framework, not a runtime).

## 6. Cost/budget caps, multi-agent orchestration, MCP

### Takeaway
Pydantic AI has the richest built-in budget controls (`UsageLimits` plus a Spend Limits harness capability with USD budgets per model and per tenant). LangChain has call-count limits only. Multi-agent: LangGraph (graphs/subgraphs, `SubAgentMiddleware`), Pydantic AI (Subagents capability, Dynamic Workflow), CrewAI (crews/flows), smolagents (`managed_agents`). MCP is supported in Pydantic AI, LangChain and CrewAI.

### Cited Findings
- Pydantic AI `UsageLimits.request_limit` is in `usage.py` — [usage.py](https://github.com/pydantic/pydantic-ai/blob/main/pydantic_ai_slim/pydantic_ai/usage.py)
- Pydantic AI "Spend Limits (Harness)": "Cross-window USD/token budgets and per-response cost tracking, per model and per tenant". "Dynamic Workflow (Harness)" has hard `max_agent_calls` budgets — [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- The Pydantic AI MCP capability takes any `MCPToolset`, runs locally by default, and uses provider-native MCP with `native=True` — [Pydantic AI capabilities](https://pydantic.dev/docs/ai/core-concepts/capabilities/)
- LangChain: `ModelCallLimitMiddleware` (per run or thread) and `ToolCallLimitMiddleware` (global or per tool). There is no built-in money cap in the middleware list. Multi-agent via `SubAgentMiddleware` (a `task` tool, from Deep Agents) — [LangChain built-in middleware](https://docs.langchain.com/oss/python/langchain/middleware/built-in)
- smolagents multi-agent: `CodeAgent(managed_agents=[ToolCallingAgent(...)])` — [smolagents inspect runs](https://huggingface.co/docs/smolagents/main/en/tutorials/inspect_runs)
- A practitioner comparison (opinion, 2026) says CrewAI has native MCP and A2A, while LangGraph "relies on community integrations". It also says teams hit "a control-flow ceiling at 5+ agents" in CrewAI and rewrite in LangGraph (anecdotal) — [ecorpit 2026 comparison](https://ecorpit.com/ai-agent-framework-production-langgraph-crewai-microsoft-pydantic-2026/); [morphllm LangGraph vs CrewAI](https://www.morphllm.com/comparisons/langgraph-vs-crewai)

### Inferences
- DCLab already prices requests in `models/prices.py` and enforces caps (`limits.py`, `--cap-eur`). It would keep its own caps whatever framework it picks. Pydantic AI's `cost` field could feed them but should not replace them, because cost is "best-effort" and may be `None`.

### Gaps
- The claim that LangGraph has no native MCP may be out of date (LangChain has `langchain-mcp-adapters`; not verified this session).
- I did not check MCP support in smolagents (`ToolCollection.from_mcp`, from prior knowledge, unverified) or in DSPy.

## 7. DSPy prompt optimization (GEPA, MIPRO) for a learning loop that evolves prompts/playbooks from traces

### Takeaway
GEPA fits DCLab's planned learning loop closely. It reflects on execution traces plus written feedback, keeps a Pareto set of candidate texts, and can evolve "textual components of arbitrary systems". So it can be used outside DSPy, through the standalone `gepa` package. Using it does not require adopting DSPy as the agent harness.

### Cited Findings
- `dspy.GEPA` (Genetic-Pareto) "adaptively evolves textual components (such as prompts) of arbitrary systems". An LLM reflects on execution traces (inputs, outputs, failures, feedback) and proposes new instructions for a target module. A Pareto frontier picks which candidate to change next — [DSPy GEPA overview](https://dspy.ai/current/api/optimizers/GEPA/overview/)
- The metric returns `dspy.Prediction(score=..., feedback=...)` and receives `gold`, `pred`, `trace`, `pred_name`, `pred_trace`, `program_trace`. Without written feedback, GEPA falls back to "This trajectory got a score of {score}". `frontier_type="objective"` uses named `objective_scores` — [DSPy GEPA overview](https://dspy.ai/current/api/optimizers/GEPA/overview/)
- The core evolution pipeline is a separate GitHub `gepa` package, used by `dspy.GEPA`. DSPy installs it only on Python < 3.15; without it, `dspy.GEPA` raises ImportError — [DSPy GEPA overview](https://dspy.ai/current/api/optimizers/GEPA/overview/)
- Paper: Agrawal et al., 2025, "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning", arXiv:2507.19457 — [DSPy GEPA overview](https://dspy.ai/current/api/optimizers/GEPA/overview/)
- DSPy 3.4 includes "ReAnchor", which "calibrates those [Jev] decisions against your program's metric" (experimental) — [DSPy 3.4.0 release](https://github.com/stanfordnlp/dspy/releases/tag/3.4.0)

### Inferences
- DCLab's prompts already live as files with hashes checked by a gate (`dclab_rnd/prompts/`, `make agent-eval`). That is close to the shape GEPA needs: a text component plus a scored, feedback-giving evaluator. The journeys and agent-eval graders could act as the metric with feedback. The sealed sets must never be the GEPA validation set (DCLab's own rule).
- Playbook evolution (many named sections) fits GEPA's "arbitrary textual components" better than MIPRO's instruction-plus-few-shot search. This is an inference; I did not compare them directly here.

### Gaps
- I could not confirm the paper's headline numbers or its venue from the fetched page. Its claims against MIPROv2/GRPO need the paper itself.
- I did not fetch MIPROv2's current docs; there is no direct GEPA vs MIPRO comparison here.
- I did not verify a standalone `gepa` API such as `optimize_anything`.

## 8. Abstraction weight, lock-in, breaking changes, and practitioner complaints

### Takeaway
- **Lightest, most explicit Python option with strong guards:** Pydantic AI. Risks: very fast release pace, a recent V1 → V2 jump, and soft pull toward Logfire.
- **Most proven for durable, stateful graphs:** LangGraph. Risks: heavier code, an ecosystem pull toward LangSmith, and a LangChain integration surface that keeps moving.
- **CrewAI:** quick to prototype. Weak points: control-flow limits reported at scale, and opt-out telemetry.
- **DSPy:** best used as an optimizer, not a runtime, and it is mid-migration on its LM layer.
- **smolagents:** stalled.
- **Letta (Python):** retired.
- **Mastra:** a language mismatch.

### Cited Findings
- DSPy breaking change: 3.4 replaces the experimental LM types from 3.3. "3.5 is the migration deadline" for legacy custom LMs. RLM's interpreter API changed (`interpreter_factory=` keyword-only, now a reserved input name) — [DSPy 3.4.0 release](https://github.com/stanfordnlp/dspy/releases/tag/3.4.0)
- Pydantic AI V1 → V2 (2026-06-23) was a major change with an upgrade guide; V1 is still patched in parallel (v1.107.9 on 2026-10-21) — [v2.0.0 release](https://github.com/pydantic/pydantic-ai/releases/tag/v2.0.0); [releases](https://github.com/pydantic/pydantic-ai/releases)
- Letta replaced its whole runtime: the V1 Python API server is retired, and the current source is TypeScript `letta-code` — [letta-ai/letta README](https://github.com/letta-ai/letta)
- One author's line count for the same agent (single test, weigh lightly): LangGraph ≈280, PydanticAI ≈160, CrewAI ≈420. The same source calls Pydantic AI's community "smaller, though unusually technically engaged" and says Pydantic AI lacks LangGraph's multi-agent orchestration out of the box (written before or regardless of V2 Subagents?) — [ecorpit 2026 comparison](https://ecorpit.com/ai-agent-framework-production-langgraph-crewai-microsoft-pydantic-2026/)
- The search summary found "no independent, quantitative production reliability benchmarks". Most comparisons are vendor or consultancy blogs — [dev.to 2026 matrix](https://dev.to/kunal_d6a8fea2309e1571ee7/langgraph-vs-crewai-vs-autogen-vs-pydanticai-2026-matrix-24gn); [redwerk LangGraph vs CrewAI](https://redwerk.com/blog/langgraph-vs-crewai/)
- CrewAI's telemetry opt-out has been unreliable for some users — [CrewAI community](https://community.crewai.com/t/disable-telemetry-for-production/4989)

### Inferences
- Lock-in, roughly from low to high:
  1. Pydantic AI (plain functions and Pydantic models; capabilities are its own concept).
  2. DSPy as an optimizer only (the output is text).
  3. LangGraph (graph state shape and checkpointer format are its own; the LangChain message types spread through the code).
  4. CrewAI (role/crew abstractions own the loop).
  5. Mastra / letta-code (different language).
- DCLab already has a harness that does several things the frameworks do not: one registry of tools with a move validator, ledger capture, consent-gated export, a gateway with prices and shadows. Any framework would mainly replace the loop and the step plumbing. The honest question is whether that saves more than it costs to wire DCLab's guards and recorders into it.

### Gaps
- I did not systematically collect GitHub issue themes (e.g. the most-upvoted complaints per repo). The complaints above are from the CrewAI forum and opinion blogs only.
- No independent benchmark of framework overhead or reliability was found.
- I did not verify the LangGraph 0.x → 1.0 migration details, or LangChain v1's move of old APIs (`langchain_classic` was not found at the path I tried).
