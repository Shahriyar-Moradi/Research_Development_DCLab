# Evaluation harnesses and agent-trajectory capture (for DCLab, as of 2026-10-22)

Scope: the difference between an evaluation harness and an agent framework; inspect-ai; mini-swe-agent / SWE-agent / OpenHands and how their trajectories train models; what makes measurement reliable; how to capture whole trajectories for SFT of a tool-calling model (MLX-LM, TRL, Qwen templates); OpenTelemetry GenAI conventions; pitfalls.

Research budget note: 16 tool calls. Several items are supported only by secondary sources or by reasoning; they are marked. Nothing here was run or tested locally.

## 1. Evaluation harness vs agent framework, and how to combine them

### Takeaway
An agent framework decides *how the agent acts* (loop, tools, memory). An evaluation harness decides *how the agent is measured* (task set, sandbox, repeats, graders, logs). They are separate layers. The cleanest setup keeps your own agent and plugs it into a harness as one "solver/agent", with model calls routed through the harness so every call is logged.

### Cited Findings
- inspect-ai splits an evaluation into **tasks** (which combine datasets, solvers and scorers), **datasets** (samples), **solvers** (prompting / elicitation), **scorers** (grade output and aggregate into metrics), plus **sandboxes**, **agents**, **approval** policies, **limits** (time, message, token, cost), **eval logs** and **eval sets** — [Inspect llms.txt index](https://inspect.aisi.org.uk/llms.txt)
- inspect-ai explicitly supports running agents "built with third-party frameworks or custom code" through an **agent bridge**, routing their model calls through Inspect's current model provider — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- mini-swe-agent's authors present its minimal design as a baseline that keeps "the model, not the scaffold" central, and avoids tying research to one agent scaffold — [mini-swe-agent README](https://github.com/SWE-agent/mini-swe-agent)

### Inferences
- Simple picture: the agent framework is the *player*; the harness is the *exam hall* (the room, the clock, the paper, the marker). A team should not let the player mark its own exam.
- For DCLab: the in-house harness (`agents/harness.py`) is already the agent framework *and* part of a harness (tracing, replay, scripted transports, sealed sets, gate). The open question is only whether to move the *measurement* layer onto inspect-ai, not the agent loop.
- Recommended combination: (a) keep DCLab's agent loop; (b) wrap it as an Inspect agent via `agent_bridge()` (in-process, Python); (c) keep DCLab's deterministic graders and write them as Inspect scorers; (d) keep DCLab's own trace as the training record, and use the Inspect log as the measurement record. Two records of one run are fine if they share a run id.

### Gaps
- No primary source found that defines "evaluation harness" vs "agent framework" formally; the distinction above is a synthesis.

## 2. inspect-ai: model, providers, wrapping an in-house agent, strengths and limits

### Takeaway
inspect-ai (UK AISI) is a mature, open-source Python eval framework with tasks/solvers/scorers, sandboxes, repeated epochs with reducers, and rich logs. It supports Anthropic and local models (Ollama, vLLM, llama-cpp-python, any OpenAI-compatible server). An in-house Python agent can be wrapped with `agent_bridge()`, which intercepts its OpenAI/Anthropic/Google API calls. MLX is not a built-in provider.

### Cited Findings
- **Core model:** tasks, datasets, solvers, scorers, epochs (repeated runs of samples, with reduction), sandboxes, agents (incl. a built-in ReAct agent), approval, limits, eval logs, eval sets, plus pages on Deep Agent, Multi Agent, Custom Agents, Checkpointing, Intervention, Human agent (`human_cli()` for human baselines) — [Inspect llms.txt](https://inspect.aisi.org.uk/llms.txt)
- **Epochs and reducers:** per-epoch sample scores are averaged by default; an `Epochs` object takes a count and one or more reducers (e.g. `"mode"`), each producing its own metrics — [Inspect Scoring Metrics](https://inspect.aisi.org.uk/metrics.html)
- **Unscored samples:** a scorer can return `Score.unscored()`; such samples are skipped by metrics and reducers and counted separately rather than as zeros — [Inspect Custom Scorers](https://inspect.aisi.org.uk/custom-scorers.html)
- **Scorers can inspect the sandbox** with `sandbox()` and `read_file()` / `exec()` (useful for grading files an agent wrote) — [Inspect Multiple Scorers](https://inspect.aisi.org.uk/multiple-scorers.html)
- **Providers:** built-in lab APIs include OpenAI, Anthropic, Google, Grok, Mistral, DeepSeek, etc.; local: Hugging Face, vLLM, Ollama, llama-cpp-python, SGLang, TransformerLens, nnterp. Anthropic uses `anthropic/<model>` with `ANTHROPIC_API_KEY`; Ollama uses `ollama/<model>` with default `http://localhost:11434/v1` (`OLLAMA_BASE_URL`); any OpenAI-compatible server via `openai-api/<provider>/<model>` with `<PROVIDER>_BASE_URL` and `<PROVIDER>_API_KEY`; `-M emulate_tools=true` for models without native tool calling — [Inspect Providers](https://inspect.aisi.org.uk/providers.html)
- **MLX is not listed** as a built-in provider — [Inspect Providers](https://inspect.aisi.org.uk/providers.html)
- **Agent bridge (in-process):** `agent_bridge()` runs a Python agent in the same process; the agent's client uses `model="inspect"` (or `inspect/<provider>/<model>`); the bridge state tracks `messages` and `output`. Agents are wrapped with the `@agent` decorator — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- **Agent bridge (sandboxed):** `sandbox_agent_bridge()` runs a proxy inside the sandbox container (port 13131), so the agent can be in any language; point `OPENAI_BASE_URL` / `ANTHROPIC_BASE_URL` / `GOOGLE_GEMINI_BASE_URL` at it — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- **Intercepted APIs:** OpenAI Completions, OpenAI Responses, Anthropic, Google Gemini. All model calls appear in the transcript; `transcript()` adds custom content; a model event's `requested_model` records the name the agent asked for — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- **Bridge limits:** client `max_tokens`/`temperature` are dropped unless `forward_generation_config=True` (the eval config wins); `previous_response_id` refused (400); only first choice used (`n=1` recommended); provider tools (`web_search`, `code_execution`, remote MCP) withheld unless granted; only `Accept-Encoding` header forwarded by default; bridged tool calls executed once per model proposal — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- An Inspect harness is used by third parties (e.g. a Hugging Face OpenEnv space ships an `inspect_harness.py`) — [HF OpenEnv inspect_harness.py](https://huggingface.co/spaces/openenv/chat_env/blob/main/src/core/evals/inspect_harness.py)

### Inferences
- DCLab routes every model call through its own gateway (`dclab_rnd.models`). To bridge, the gateway's Anthropic/OpenAI client would point at Inspect's `model="inspect"` (in-process) during an eval. Then Inspect owns generation config (temperature, max tokens) — good for repeatable measurement, but it means DCLab's per-purpose settings are overridden during evals. Decide this deliberately.
- DCLab's scripted transports (tests without a live model) have no direct Inspect equivalent in what I read; Inspect does have a mock/scripted model concept in my memory but I did not verify it (see Gaps).
- For the ~8B Qwen model on MLX: run `mlx_lm.server` (OpenAI-compatible) and use `openai-api/mlx/<model>`. Inference: should work, not verified.
- Strengths: repeat-runs and reducers built in; good log viewer; human-baseline agent; government-lab backing and wide use. Limits: a second logging system beside DCLab's ledger; bridge overrides generation config; MLX only via the generic OpenAI-compatible path.

### Gaps
- Could not verify the current inspect-ai version number or the `.eval` vs `.json` log format details (search did not reach the eval-logs page). From memory (unverified): `.eval` is a compressed binary default format, readable with `read_eval_log`.
- Did not verify whether Inspect has a built-in mock/scripted model for tests.
- Did not verify inspect's built-in confidence intervals (stderr metric exists from memory; bootstrap/cluster CIs not verified).

## 3. mini-swe-agent / SWE-agent / OpenHands: what the minimal design teaches, and training on their trajectories

### Takeaway
mini-swe-agent shows that a ~100-line agent with one tool (bash), a linear message history and stateless commands scores >74% on SWE-bench Verified with a strong model. Its key lesson for training: when the trajectory *is* exactly what the model saw, it can be trained on directly. SWE-smith and SWE-Gym show the standard recipe: run a strong model in the harness, keep only verified-successful trajectories (rejection sampling), fine-tune a smaller open model on them.

### Cited Findings
- mini-swe-agent: agent class ~100 lines of Python; only tool is bash; it does **not** use the LM's tool-calling interface, so it works with essentially any model; each step appends to one message list that is also the model's input ("trajectory and model input are identical"); actions run with `subprocess.run` (no persistent shell), and swapping to `docker exec` sandboxes it; >74% on SWE-bench Verified; ships a trajectory browser (inspector) — [mini-swe-agent README](https://github.com/SWE-agent/mini-swe-agent)
- **SWE-smith** recipe: generate expert trajectories with SWE-agent and a strong model (e.g. Claude 3.7 Sonnet, GPT-4o) on SWE-smith task instances, then train a student only on trajectories for resolved instances (rejection sampling fine-tuning); full fine-tuning with torchtune, LR 5e-5, up to 3 epochs, 32,768-token context — [SWE-smith paper (arXiv 2504.21798)](https://arxiv.org/pdf/2504.21798)
- **SWE-agent-LM-32B**: Qwen 2.5 Coder Instruct base, fine-tuned on ~5,000 trajectories generated with SWE-agent + Claude 3.7 Sonnet; reported 40.2% pass@1 on SWE-bench Verified — [model card mirror, featherless.ai](https://featherless.ai/models/SWE-bench/SWE-agent-LM-32B); the 40.2% figure also in a newsletter summary [Turing AGI Advance](https://www.turing.com/blog/agi-advance-newsletter-14). (Secondary sources; check the paper/model card for exact numbers.)
- **SWE-Gym** (ICML 2025): 2,438 real Python tasks with executable runtimes and unit tests; fine-tuning OpenHands agents on SWE-Gym trajectories gave up to +19 points; arXiv v2 reports +13.6 to 20.6% on Verified and +12.3 to 15.3% on Lite; a verifier (outcome reward model) trained on success/failure-labelled trajectories plus best-of-n sampling reached 32.0% Verified / 26.0% Lite; gains grow with more trajectories; trajectories released — [SWE-Gym, PMLR v267](https://proceedings.mlr.press/v267/pan25g.html); [arXiv 2412.21139v2](https://arxiv.org/pdf/2412.21139v2)
- **Learning from failed trajectories:** JetBrains Research's "Step Rejection Fine-Tuning" keeps useful steps of failed SWE-smith trajectories and masks loss on erroneous steps using a critic model (fine-tuning Qwen2.5-Coder-32B-Instruct on the SWE-agent scaffold) — [JetBrains Research blog, 2026-06](https://blog.jetbrains.com/research/2026/06/step-rejection-fine-tuning/) (only seen via search snippet; not read in full)

### Inferences
- Lesson 1 (record = input): store exactly the messages the model received, not a reconstructed summary. Then SFT data is a straight copy. DCLab's "never truncate before saving" rule matches this.
- Lesson 2 (outcome labels): every trajectory needs a deterministic verdict (pass/fail from a grader). That enables rejection sampling (SFT on passes), verifier training (success/failure), and step-level masking. DCLab's sealed task sets and gates are exactly the needed labeller.
- Lesson 3 (stateless tools, sandbox): independent actions make runs reproducible and easy to parallelise.
- Caveat for DCLab: these results are for code-repair tasks with unit tests. DCLab's tasks (leakage-safe tabular ML) need their own graders; transfer of the numbers is not guaranteed.
- Scale caveat: SWE-smith used ~5,000 successful trajectories and a 32B model with 32k context; an 8B model on a Mac with far fewer trajectories should be expected to gain less.

### Gaps
- Did not read SWE-agent's or OpenHands' trajectory file schemas directly (SWE-agent `.traj` JSON and OpenHands event stream are known from memory only; unverified field names).
- AgentTuning (Zeng et al., 2023) not checked in this pass.

## 4. What makes a harness reliable for measurement

### Takeaway
Reliability comes from: isolation (sandbox), fixed inputs and model settings, repeated runs with a declared reducer, intervals, unscored-vs-failed separation, a sealed set used once, and contamination checks. inspect-ai gives several of these out of the box; contamination and sealing are the team's discipline.

### Cited Findings
- Sandboxes isolate model-generated code and set up task environments — [Inspect llms.txt](https://inspect.aisi.org.uk/llms.txt)
- Epochs repeat each sample; reducers (mean by default, or mode, etc.) combine them — [Inspect Scoring Metrics](https://inspect.aisi.org.uk/metrics.html)
- Unscored samples are not counted as zeros — [Inspect Custom Scorers](https://inspect.aisi.org.uk/custom-scorers.html)
- Limits on time, messages, tokens and cost are first-class — [Inspect llms.txt](https://inspect.aisi.org.uk/llms.txt)
- With the agent bridge, the eval's generation config overrides the agent's (`forward_generation_config=False` by default) — keeps settings fixed across agents — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- SWE-Gym reports roughly log-linear gains from more sampled candidates — i.e. a pass@k number is not comparable to pass@1 — [SWE-Gym, PMLR v267](https://proceedings.mlr.press/v267/pan25g.html)

### Inferences
- Checklist for DCLab:
  1. Seeds: fix data seeds and sandbox state; API models are not fully deterministic even at temperature 0, so repeat (≥3 epochs) and report the spread.
  2. Intervals: report a bootstrap interval clustered by task (not by epoch), because repeats of one task are not independent.
  3. Separate "failed" from "could not run" (Inspect's `unscored`).
  4. Sealed set run once per gate; a dev set for tuning (DCLab already does this).
  5. Contamination: if training on trajectories, hold out whole *tasks* (and their replicas/templates), not just turns; check n-gram overlap of training data with sealed task text.
  6. Cost and time caps per sample.
  7. Store the harness version, model id, prompt hash and grader version with every score.

### Gaps
- No primary source found on agent-eval contamination checks specifically (e.g. tooling in Inspect for it). Not verified.

## 5. Trajectory capture for SFT of a tool-calling model (MLX-LM, TRL, Qwen)

### Takeaway
Capture the raw, whole record once (system prompt, tool schemas, every assistant turn with its reasoning and tool calls, every tool result in full, the verdict), then *render* it into the trainer's format at export time. Both MLX-LM and TRL accept the OpenAI-style shape: `{"messages": [...], "tools": [...]}` with assistant `tool_calls` and `role: "tool"` results. Truncate long tool results only when building a training set, never at capture. Match the training chat template to the serving template, especially how Qwen handles past reasoning.

### Cited Findings
- **MLX-LM formats** (`mlx_lm.lora`): `chat` (`messages` list), `tools` (`messages` + top-level `tools`), `completions` (`prompt`/`completion`), `text`; one JSON object per line; auto-detected; unknown keys ignored — [MLX-LM LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- MLX-LM tool calls: assistant message with `tool_calls`, each with `id`, `"type": "function"`, `function.name`, `function.arguments`; "the format for the `arguments` field in a function varies for different models" (JSON string vs dict) — [MLX-LM LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- MLX-LM `--mask-prompt` (loss on completion only) is documented for `chat` and `completion` datasets only; there is no documented max-length truncation, the advice is to split long examples when building the JSONL; memory tips: QLoRA (quantise with `-q`), smaller `--batch-size` with `--grad-accumulation-steps`, fewer `--num-layers`, `--grad-checkpoint` — [MLX-LM LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- **TRL tool-calling SFT:** `{"messages": [...], "tools": [json_schema]}`; assistant turns carry `tool_calls` (`type: function`, `function.name`, `function.arguments` as an object); tool results are `{"role": "tool", "name": ..., "content": ...}`; schemas can be generated with `transformers.utils.get_json_schema`; with `datasets` ≥ 4.7.0 use the `Json()` feature for arbitrary arguments, else store `tools` as a JSON string — [TRL dataset formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
- TRL's `SFTTrainer` expects language-modeling (`messages`) or prompt-completion data; DPO expects preference pairs; KTO unpaired preference with a `label` — [TRL dataset formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
- Reasoning as a separate field: TRL documents a `"thinking"` key on assistant messages for the Harmony (gpt-oss) format, rendered into the `analysis` channel — [TRL dataset formats](https://huggingface.co/docs/trl/main/en/dataset_formats)
- **Qwen past-reasoning handling:** a report says Qwen3.6's template drops prior-turn `<think>` blocks unless `preserve_thinking=true` is passed in `chat_template_kwargs`; in that report tool calls started arriving with empty arguments after 2–3 turns of an agent loop — [Hermes Agent issue #56004](https://github.com/NousResearch/hermes-agent/issues/56004) (community issue, single report)
- A community Qwen 3.x template notes the official template fails when tool arguments arrive as JSON strings instead of mappings — [Qwen3 chat template fixes (community)](https://huggingface.co/nrrso/Qwen3-Chat-Template-vLLM-Fixes)
- Older Qwen3 models (up to the 2507 release) use a template where the model writes its own think tags; vLLM's `qwen3` reasoning parser handles both styles — [vLLM Qwen3 reasoning parser](https://docs.vllm.ai/api/vllm/reasoning/qwen3_reasoning_parser.html)
- SWE-smith trained with a 32,768-token context — [SWE-smith paper](https://arxiv.org/pdf/2504.21798)

### Inferences
- What to store per trajectory (capture schema, raw layer):
  - run id, task id, split (dev/sealed), harness + prompt + tool-registry versions, model id and settings, seed;
  - the exact system prompt and the exact tool schemas sent;
  - each assistant turn: visible text, reasoning/thinking text (if the provider returns it), each tool call (id, name, arguments exactly as emitted, including malformed ones);
  - each tool result whole, with byte size, timing, error flag;
  - token usage per call, stop reason;
  - the final verdict from the deterministic grader, plus per-step labels when available.
- Export layer (derived, rebuildable): render to `{"messages", "tools"}` JSONL for MLX-LM and TRL; normalise `arguments` to what the target Qwen template expects (dict vs string); choose a reasoning policy (keep only the last turn's thinking, or all with `preserve_thinking`) that matches how the model will be served.
- Long tool results: at export, cap each tool message (e.g. head + tail + "[N bytes omitted]") and drop or split examples above the max sequence length. Keep a log of what was cut. This matches "truncate at training time, never at capture".
- Loss masking: train on assistant turns only. MLX-LM's `--mask-prompt` is documented for `chat`, not `tools`, so check whether the installed version masks tool messages; if not, multi-turn trajectories may train on tool output tokens too (Inference, unverified).
- Multi-turn split: a long trajectory can be cut into several examples, each ending in an assistant turn, with the earlier context kept — so the model learns every step, not only the last.
- Filtering: SFT on graded-pass trajectories (rejection sampling); keep failures for a verifier or step-masked training (SWE-Gym, JetBrains SRFT).

### Gaps
- Could not verify the current official Qwen3 / Qwen3.5 / "Qwen3.6" chat template text or which Qwen version names exist as of late 2026; the `preserve_thinking` flag is from a single community issue.
- Did not check Axolotl's tool-calling format (from memory it supports `chat_template` datasets with `tools` and per-role training masks; unverified).
- Did not verify MLX-LM version, or whether `--mask-prompt` now covers the `tools` format.

## 6. OpenTelemetry GenAI semantic conventions: maturity, full content, privacy

### Takeaway
The OTel GenAI conventions are still "Development" (not stable) in 2026 and have moved to their own repository. They define agent and tool spans (`invoke_agent`, `execute_tool`, etc.) and content attributes (`gen_ai.input.messages`, `gen_ai.output.messages`, `gen_ai.system_instructions`), but content capture is opt-in and off by default. Good for observability and interop; not on their own a training-data format.

### Cited Findings
- The GenAI conventions moved from opentelemetry.io to a dedicated repository, `open-telemetry/semantic-conventions-genai`, covering spans, metrics and events for GenAI clients, MCP and provider-specific conventions; its schema URL section reads "TODO" — [OTel GenAI moved notice](https://opentelemetry.io/docs/specs/semconv/gen-ai/); [semantic-conventions-genai repo](https://github.com/open-telemetry/semantic-conventions-genai)
- Status: GenAI and MCP conventions are in **Development** with no public stabilisation timeline (May 2026) — [GreptimeDB blog, 2026-05-09](https://www.greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions) (secondary); OTel's own blog calls them "already in use today and under active development" — [OTel blog: Inside the LLM Call](https://opentelemetry.io/blog/2026/genai-observability/)
- Operations include `create_agent`, `invoke_agent`, `invoke_workflow`, `plan`, `execute_tool`; span names follow `execute_tool {tool_name}` and `invoke_agent {agent_name}` — [TrueFoundry](https://www.truefoundry.com/blog/opentelemetry-genai-semantic-conventions); [Honeycomb docs](https://docs.honeycomb.io/send-data/agents)
- v1.41 changes reportedly: execute_tool span naming requires tool name; reasoning tokens; `invoke_workflow`; streaming metrics; `invoke_agent` split into CLIENT (remote agent services) and INTERNAL (in-process frameworks) — [GreptimeDB blog](https://www.greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions) (secondary; another source dates the invoke_agent kind guidance to v1.38 — conflict noted)
- Content attributes: `gen_ai.system_instructions`, `gen_ai.input.messages`, `gen_ai.output.messages`; content capture also records tool schemas, arguments and results; content capture is **off by default** and only metadata (model, token counts, durations) is recorded because content "can contain sensitive data" — [OTel blog](https://opentelemetry.io/blog/2026/genai-observability/)
- Content capture is opt-in; implementation support varies by language — [TrueFoundry](https://www.truefoundry.com/blog/opentelemetry-genai-semantic-conventions)

### Inferences
- For DCLab: emit OTel GenAI spans as an *export view* of its own trace (useful for Phoenix/Langfuse/Honeycomb-type viewers and for buyers who want standard telemetry), but keep the training record in DCLab's own store. Reasons: the spec is still moving; content is opt-in and varies by SDK; span attributes can hit size limits in collectors; a training set needs exact message order and the grader verdict, which OTel does not define.
- Privacy: DCLab's consent and redaction rules (`consent.py`, never cell values) should apply before any OTel export with content on.

### Gaps
- Could not read the spec file itself (raw URL returned 404 after the move); exact attribute list (e.g. `gen_ai.tool.definitions`, `gen_ai.tool.call.arguments`/`result`), the opt-in env var (`OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` and `OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT` from memory) and any external-storage ("upload hook") guidance are **unverified**.
- **OpenInference (Arize)** was not researched in this pass (budget). From memory only: it is an OTel-based convention with span kinds (LLM, TOOL, AGENT, CHAIN, RETRIEVER…) and `llm.input_messages`/`llm.output_messages` attributes, used by Phoenix. Treat as unverified.

## 7. Known pitfalls

### Takeaway
Main traps: trajectories longer than the student's context; reasoning not returned by some providers (or stripped by templates); template mismatch between training and serving; provider terms on training with outputs; and evaluation drift when the harness overrides settings.

### Cited Findings
- **Provider terms:** Anthropic's commercial terms (Section D.4) prohibit using the Services to build a competing product, "including to train competing AI models", without Anthropic's approval; customers own their Outputs — [Anthropic Commercial Terms](https://www.anthropic.com/legal/commercial-terms) (search result; which version is current not confirmed)
- **Template strips past reasoning:** Qwen template drops prior-turn thinking unless `preserve_thinking=true`; tool-call arguments degraded after a few turns in one report — [Hermes Agent issue](https://github.com/NousResearch/hermes-agent/issues/56004)
- **Argument format mismatch:** official Qwen template fails when arguments are JSON strings rather than mappings — [community Qwen3 template](https://huggingface.co/nrrso/Qwen3-Chat-Template-vLLM-Fixes); MLX-LM notes argument format varies by model — [MLX-LM LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- **Long sequences:** MLX-LM advises splitting long examples; SWE-smith used 32k context — [MLX-LM LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md); [SWE-smith](https://arxiv.org/pdf/2504.21798)
- **Harness overrides:** Inspect's bridge drops the agent's generation settings by default and uses only the first choice — [Inspect Agent Bridge](https://inspect.aisi.org.uk/agent-bridge.html)
- **Telemetry content off by default** — [OTel blog](https://opentelemetry.io/blog/2026/genai-observability/)

### Inferences
- Anthropic terms: DCLab plans to fine-tune a Qwen model on Claude-generated trajectories. Whether an in-house, narrow ML-workflow model counts as a "competing AI model" is a legal question. Flag it to the owner; do not assume it is fine. (Not legal advice.)
- Reasoning exposure: Claude's extended thinking may be returned summarised or redacted depending on model and settings; if only summaries are available, the student learns from summaries, not true reasoning. Verify what DCLab's gateway actually receives before claiming "reasoning captured". (Unverified for Claude Sonnet 5.5 specifically.)
- Distribution shift: a student trained on teacher trajectories never sees its own mistakes; SWE-Gym/SRFT-style use of failures, or on-policy correction (see [arXiv 2512.14895, imitation learning with on-policy expert corrections](https://arxiv.org/pdf/2512.14895), only seen as a search title), addresses this.
- Context: an 8B model on a Mac at 32k tokens is memory-heavy; plan for 8k–16k training windows, so per-step splitting and tool-result capping at export matter.

### Gaps
- Did not verify Anthropic's current wording on training with outputs (fetched only search-result copies) or any carve-outs.
- Did not verify how Claude's API returns thinking content for the current product model (summarised vs full).
- "Context as a Tool" (arXiv 2512.22087) on context management for long-horizon SWE agents appeared in search but was not read.
