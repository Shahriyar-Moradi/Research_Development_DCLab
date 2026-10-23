# Frameworks that train a model inside an existing agent harness (as of 2026-10-23)

Scope: Agent Lightning, OpenPipe ART (+RULER), rLLM, verl / verl-agent, SkyRL, OpenRLHF agent mode, TRL agent/tool training, Apple-silicon options (mlx-lm, mlx-lm-lora), plus Tinker as a notable managed addition. Read for DCLab: SFT of a ~8B Qwen on whole trajectories now, preferences/RL from a deterministic grader later, own Python harness kept.

Note on sources: most facts come from the projects' own READMEs/docs fetched on 2026-10-23. Several READMEs name models (Qwen3.5, Qwen3.6, "Beam 501B") and releases dated after mid-2026; these are reported as the pages state them and were not independently verified. Vendor claims are marked as such.

## 1. How each framework attaches to an existing agent, and what the harness must expose

### Takeaway
Three attachment patterns exist: (a) an OpenAI-compatible proxy/gateway that records every model call while the agent runs unchanged (Agent Lightning, rLLM SDK, ART client); (b) a Gym-style environment with reset()/step() that the trainer drives (SkyRL-gym, OpenRLHF, verl-agent, TRL `environment_factory`); (c) a plain dataset of finished conversations (SFT in TRL / mlx-lm). DCLab's gateway already sits where pattern (a) needs a hook, so the cheapest path is to point the gateway's local-model route at a trainer's proxy and add a reward per finished trajectory.

### Cited Findings
- Agent Lightning v1.0 (open-sourced Aug 2026): agents reach the model through the "Agent Lightning v1.0 proxy", which the README says needs "ZERO changes" to the agent; tools, context, control flow and environments stay in the agent; an API Gateway proxies model requests and captures training data; a Rollout Controller runs agents locally or as Kubernetes Jobs; core is about 3,500 lines of Python; MIT license — [GitHub README](https://github.com/microsoft/agent-lightning), [docs](https://microsoft.github.io/agent-lightning/stable/)
- Agent Lightning paper (arXiv 2508.03680, 5 Aug 2025): "Training-Agent Disaggregation"; agent execution modelled as an MDP with a unified data interface that turns any agent's trajectory into training transitions; works with LangChain, OpenAI Agents SDK, AutoGen or from-scratch agents "with almost no code changes"; uses agent observability (tracing) as the capture interface — [arXiv](https://arxiv.org/abs/2508.03680)
- OpenPipe ART: an OpenAI-compatible client in your code plus a separate GPU server; your code runs rollouts through the client, each message is stored in a trajectory, your code assigns a reward to each finished trajectory, trajectories are grouped and sent to the server, which pauses inference, trains a LoRA, loads it back into vLLM and repeats; LangGraph integration exists; `pip install openpipe-art`; Apache-2.0 — [GitHub](https://github.com/OpenPipe/ART)
- ART RULER needs trajectories organised into `art.TrajectoryGroup` (4–8 per group recommended) and a system prompt that clearly states the goal, because the judge relies on it — [ART docs: RULER](https://art.openpipe.ai/fundamentals/ruler)
- rLLM: an agent is a function decorated `@rllm.rollout` (Task + AgentConfig → Episode → Trajectories → Steps, one step per LLM call); rewards from `@rllm.evaluator`; existing agents (LangGraph, OpenAI Agents SDK, raw `openai.OpenAI`, 10+ CLI harnesses such as Claude Code, Codex, mini-swe-agent) are wrapped without rewriting; during training the agent's base URL points to a gateway that captures token IDs and logprobs, so the same code runs for eval and training; "rLLM SDK: training agentic programs without code changes" (Dec 2025); Apache-2.0 — [GitHub](https://github.com/rllm-org/rllm)
- SkyRL: skyrl-gym wraps tasks in the Gymnasium API (math, coding, search, SQL); skyrl-agent for long-horizon agents; Harbor integration for terminal agents (Feb 2026); the README gives no step-by-step for plugging in an external harness — [GitHub](https://github.com/NovaSky-AI/SkyRL)
- OpenRLHF: agent-based execution; write an agent by subclassing `AgentInstanceBase` (optionally wrapped in `MultiTurnAgentExecutor`) with reset()/step(), passed via `--train.agent_func_path`; every mode emits a token-level trajectory consumed by one loss layer; docs rate this interface "Advanced" — [OpenRLHF docs: agent paradigm](https://openrlhf.readthedocs.io/en/latest/agent_paradigm.html), [agent training](https://openrlhf.readthedocs.io/en/latest/agent_training.html)
- TRL GRPOTrainer: `tools=[...]` (plain typed Python functions with docstrings) or `environment_factory` (one stateful instance per rollout; public methods exposed as tools; `reset()` required, `get_reward()` optional); marked experimental; needs `transformers>=5.2.0`; a `rollout_func` hook (experimental) lets you return your own `prompt_ids`, `completion_ids`, `logprobs` — the route for driving an external harness — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- verl-agent: step-independent multi-turn rollouts (each step's input built separately, not by concatenating history) and a pluggable memory module choosing what history each step sees — [GitHub](https://github.com/langfengQ/verl-agent)
- verl: multi-turn training with tool calling, an agent-loop example, search-tool and sandbox-fusion guides — [GitHub](https://github.com/volcengine/verl)

### Inferences
- What DCLab's harness must expose for any RL path: (1) an OpenAI-compatible base URL it can be pointed at (DCLab's gateway already routes local models via OpenAI-compatible endpoints, so this is a config change); (2) a task/episode boundary (start, end, id); (3) a scalar reward per finished trajectory (DCLab's deterministic grader); (4) the ability to run several rollouts of the same task in parallel (GRPO needs groups); (5) token-level logprobs for on-policy RL — the proxy captures these, which DCLab's own trace capture does not need to.
- Proxy-style tools (Agent Lightning, rLLM SDK, ART) fit DCLab best because the tool registry, validator and harness stay as they are. Gym-style tools (OpenRLHF, SkyRL-gym, TRL `environment_factory`) mean re-expressing the harness as reset()/step(), which duplicates logic DCLab already has.
- One risk with "zero code change": the proxy sees text and token IDs, not DCLab's structured decision records. Re-tokenisation drift (chat template applied by proxy vs. by model server) can silently break on-policy RL; rLLM explicitly captures token IDs at the gateway to avoid this.

### Gaps
- Agent Lightning v1.0's exact reward API, whether its proxy is fully OpenAI-compatible (tool_calls), and whether earlier v0.x features (`agl.emit_*`, tracer, LightningStore, APO prompt optimisation, an Unsloth SFT example) survive in v1.0: the fetched v1.0 docs page did not mention them. My recollection of v0.x having APO and an SFT example is unverified here.
- No source gives the exact SkyRL steps for attaching an external harness.

## 2. Training methods supported (SFT, preferences, GRPO/PPO, multi-turn tool use, credit assignment)

### Takeaway
For supervised training on whole tool-calling trajectories, TRL SFTTrainer is the most complete and documented (tools column, assistant-only loss, auto-patched Qwen3 templates). mlx-lm-lora covers SFT + DPO/ORPO + GRPO on the Mac but has no documented tool-call training. All RL frameworks centre on GRPO-family methods; real multi-turn credit assignment (per step, not per episode) is offered by Agent Lightning (LightningRL) and verl-agent (GiGPO).

### Cited Findings
- TRL SFTTrainer: "fully supports fine-tuning models with tool calling" — each example holds messages with `tool_calls` and `tool` messages plus a `tools` column of JSON schemas; `assistant_only_loss=True` trains only on assistant turns and needs `{% generation %}` tags, which TRL auto-patches for known families such as Qwen3; also packing, PEFT/LoRA, `loss_type="dft"`; default `max_length` is 1024 (must be raised for trajectories) — [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- TRL GRPO with tools needs a "prefix-preserving" chat template; TRL swaps in patched training templates for Qwen3 and DeepSeek-V3; tested model list includes Qwen2.5, Qwen3, Qwen3.5, Qwen3.6, Llama 3.1/3.2, GPT-OSS, Gemma4, GLM-4.x; integrations OpenEnv, OpenReward, Harbor; `max_tool_calling_iterations` caps turns — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- TRL scripts list includes `grpo_agent.py` "to fine-tune a model for agentic usage" (docs v1.7–v1.9) — [TRL example overview v1.9.1](https://huggingface.co/docs/trl/v1.9.1/example_overview)
- ART: GRPO core; RULER (LLM-judge relative rewards); AutoRL (no labelled data); SFT (distillation, from a dataset, SFT warm-up then RL) — [GitHub](https://github.com/OpenPipe/ART)
- RULER: judge scores N trajectories of one scenario 0–1 together; only relative ranks matter because GRPO normalises within groups; any LiteLLM model can judge (examples default to `openai/o3`; "cheaper models like Qwen3 32B often work well"); vendor claim: in 3 of 4 launch tasks RULER-trained models slightly beat hand-crafted rewards — [ART docs: RULER](https://art.openpipe.ai/fundamentals/ruler)
- rLLM: GRPO, REINFORCE, RLOO, SFT, on-policy distillation; backends verl (multi-GPU), tinker (single machine/managed), fireworks — [GitHub](https://github.com/rllm-org/rllm)
- verl: PPO, GRPO, GSPO, ReMax, REINFORCE++, RLOO, PRIME, DAPO, DrGRPO, KL_Cov/Clip_Cov; SFT listed; DPO not a core algorithm (only via VeRL-Omni and an extension guide); SPPO recipe — [GitHub](https://github.com/volcengine/verl)
- verl-agent: GiGPO (critic-free, episode-level plus step-level grouping for finer credit assignment), GRPO, PPO, DAPO, GSPO, RLOO, REINFORCE++; GiGPO accepted at NeurIPS 2025 — [GitHub](https://github.com/langfengQ/verl-agent)
- Agent Lightning: LightningRL, "a hierarchical RL algorithm with a credit assignment module" that decomposes trajectories into transitions, which the authors say handles multi-agent and dynamic workflows — [arXiv 2508.03680](https://arxiv.org/abs/2508.03680); v1.0 trainer runs verl with vLLM; Megatron + R3 for MoE — [GitHub](https://github.com/microsoft/agent-lightning)
- SkyRL: multi-turn tool-use RL, fully async RL with in-flight weight updates, on-policy distillation, Tinker API backend (Feb 2026), v0.3.0 (16 Jul 2026) — [GitHub](https://github.com/NovaSky-AI/SkyRL)
- OpenRLHF: single-turn and multi-turn agent execution share one token-level loss; 0.10 added multi-turn VLM RL — [OpenRLHF docs](https://openrlhf.readthedocs.io/en/latest/agent_paradigm.html)
- mlx-lm-lora: SFT; DPO, CPO, ORPO, DSLA, FTPO; GRPO, GSPO, Dr. GRPO, DAPO, KLPO; Online DPO, XPO, RLHF-Reinforce, PPO; LoRA/DoRA/full/QLoRA (4/6/8-bit); tool-call training not mentioned — [GitHub](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- mlx-lm (official): data formats `chat`, `tools`, `completions`, `text`; the `tools` format holds assistant `tool_calls` plus tool definitions; `--mask-prompt` (loss on completion only; for chat only the final message counts as completion); LoRA, DoRA, full — [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)

### Inferences
- mlx-lm's `--mask-prompt` treats only the final message as the completion. For a multi-turn trajectory this either trains on everything (tool outputs and user turns too) or only the last turn. To train every assistant turn on the Mac, split each trajectory into one example per assistant turn (prefix → next assistant turn). This is a real difference from TRL's `assistant_only_loss`.
- DCLab's ~220 preference pairs fit DPO/ORPO directly (TRL DPOTrainer on GPU; mlx-lm-lora DPO/ORPO on the Mac).
- For a deterministic grader, GRPO with group sizes of 4–8 is the common default across ART, TRL, verl, rLLM; a deterministic grader removes the need for RULER's judge cost.

### Gaps
- TRL's DPO docs were not fetched; DPO support with tool-call conversations was not verified.
- Whether mlx-lm-lora's GRPO supports multi-turn tool environments is not documented on its README.

## 3. Hardware: Apple silicon vs NVIDIA; what fits in 32 vs 64 GB

### Takeaway
On Apple silicon, only the MLX family runs today: SFT/LoRA/QLoRA is practical; DPO/ORPO is practical; GRPO exists in mlx-lm-lora but is very slow (≈0.02 it/s even on a 0.6B model). Every agent-RL framework (Agent Lightning, ART server, rLLM-verl, verl, verl-agent, SkyRL, OpenRLHF) assumes CUDA GPUs with vLLM/SGLang. An 8B model with 4-bit QLoRA fits on 32 GB; bf16 LoRA with long trajectories is the case that wants 64 GB. The 8 GB Air is for inference/tests of small models only.

### Cited Findings
- mlx-lm-lora approximate benchmarks (M4 Pro): Qwen3-0.6B SFT ≈4.7 it/s using ≈2 GB; ORPO ≈4.5 it/s using 2–4 GB; GRPO ≈0.02 it/s using 9–20 GB; Qwen3-8B (4-bit) SFT ≈4.1 it/s using 6–10 GB; the page calls the figures approximate — [GitHub](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- mlx-lm: an M1 Max 32 GB runs Mistral-7B LoRA at about 250 tokens/s with batch size 1 and 4 LoRA layers (old setting); memory levers are `--batch-size`, `--grad-accumulation-steps`, `--num-layers` (default 16), `--grad-checkpoint`, shorter examples, quantized base (QLoRA) — [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- Planning estimate (not a measurement): LoRA on 8B ≈20–24 GB, QLoRA ≈10–14 GB at batch 1, 2K tokens, gradient checkpointing — [InventiveHQ fine-tuning VRAM calculator](https://inventivehq.com/tools/developer/fine-tuning-vram-calculator); another guide says 14B LoRA needs ≈14–18 GB and is "comfortable on a 32GB Mac" — [InsiderLLM guide](https://insiderllm.com/guides/fine-tuning-mac-lora-mlx/) (unverified secondary sources; they disagree in scale)
- mlx-lm GGUF export from `fuse` is limited to Mistral/Mixtral/Llama-style models in fp16 — [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- verl supports NVIDIA, AMD ROCm (MI300X/MI325X/MI355X), Ascend; Apple not mentioned — [GitHub](https://github.com/volcengine/verl)
- verl-agent: with LoRA, 7B models train on 2 H100s — [GitHub](https://github.com/langfengQ/verl-agent)
- ART server "runs on any machine with a GPU"; client runs anywhere; serverless option via W&B Training (vendor claims: 40% lower cost, 28% faster) — [GitHub](https://github.com/OpenPipe/ART)
- Agent Lightning install example targets a CUDA 13.0 machine; the Calc-X example needs one GPU — [GitHub](https://github.com/microsoft/agent-lightning)
- Managed option: Tinker (Thinking Machines) exposes `forward_backward`, `optim_step` and sampling for LoRA on open models, priced per token; one third-party guide reports Qwen3-8B training at $0.40/M tokens rising to $0.44 after 17 Jul 2026, and notes RL costs more because rollouts are billed at the sample rate — [Beam blog (third party)](https://www.beam.cloud/blog/tinker-model-pricing), [Tinker official page](https://www.thinkingmachines.ai/tinker); rLLM and SkyRL both offer a Tinker backend — [rLLM](https://github.com/rllm-org/rllm), [SkyRL](https://github.com/NovaSky-AI/SkyRL)

### Inferences
- Rough arithmetic (not measured): an 8B model is ~16 GB in bf16 and ~4.5–5 GB in 4-bit. Activations grow with sequence length; agent trajectories of 8–16K tokens make activations, not weights, the main cost.
  - 32 GB M1 Max: 4-bit QLoRA of Qwen 8B, batch 1, grad checkpointing, sequences up to ~4–8K tokens should fit. bf16 LoRA is tight even at 2K.
  - 64 GB M1 Max: bf16 LoRA at 4–8K, or QLoRA at longer contexts, is plausible.
  - Measure first: run 50 steps on the longest trajectories and watch memory pressure (the owner's rule already stops jobs at red pressure).
- QLoRA trains against a quantised base; for serving through Ollama the adapter must be fused and converted. mlx-lm's own GGUF export does not cover Qwen, so the route is fuse → HF safetensors → llama.cpp's converter (or serve with `mlx_lm.server`, which is OpenAI-compatible).
- GRPO on the Mac is not realistic for 8B with multi-turn rollouts: 0.02 it/s on a 0.6B model implies days per useful run. RL belongs on rented GPUs (1–2 × 80 GB for 8B LoRA, per verl-agent's 2 × H100 data point) or a managed API (Tinker, W&B Serverless RL).

### Gaps
- No measured memory figures for Qwen 8B LoRA on an M1 Max with 8–16K-token sequences were found.
- Whether the owner's M1 Max has 32 or 64 GB is unconfirmed (the brief says so).
- Current official Tinker prices were not verified on the Tinker site.

## 4. Reported results on agent tasks and data sizes

### Takeaway
Published gains from agent RL are large but come from hundreds to thousands of tasks with automatic graders, on GPU clusters. The smallest reported sizes are SkyRL-SQL (653 samples) and Agent Lightning's MoE coding agent (1.8K). DCLab's few hundred to ~1,000 trajectories are in the range where SFT is reasonable and a small RL run could work only if tasks are re-sampled with a grader (RL needs tasks, not stored trajectories).

### Cited Findings
- Agent Lightning: coding agent on Qwen3.5-9B with 6K samples, SWE-bench Verified 41.8% → 56.4% (+14.6); MoE coding agent Qwen3.5-35B-A3B with 1.8K samples, pure RL, 47.8% → 61.6% (+13.8) — [GitHub README](https://github.com/microsoft/agent-lightning) (self-reported)
- Agent Lightning paper: "stable, continuous improvement" on text-to-SQL, RAG and math tool use; abstract gives no numbers — [arXiv 2508.03680](https://arxiv.org/abs/2508.03680)
- SkyRL-SQL-7B: "trained on just 653 samples", claimed to beat GPT-4o and o4-mini on text-to-SQL (no scores on the README) — [SkyRL GitHub](https://github.com/NovaSky-AI/SkyRL)
- verl-agent / GiGPO (paper version): ALFWorld 86.7% (Qwen2.5-1.5B), 90.8% (7B); WebShop 67.4% / 75.2%; Search-R1-style QA 42.1 (3B), 47.2 (7B); results shifted slightly after the 2025-06-03 update — [GitHub](https://github.com/langfengQ/verl-agent)
- rLLM: DeepSWE-32B ("state of the art among open-source coding agents", Jul 2025), DeepCoder-14B (claims o3-mini level, Apr 2025), rLLM-FinQA-4B (claims to beat a 235B model, Feb 2026); scores are in linked posts, not the README — [GitHub](https://github.com/rllm-org/rllm)
- ART: ART·E email agent (Qwen 2.5 14B) reportedly beat o3 at email retrieval; numbers not on the README — [GitHub](https://github.com/OpenPipe/ART)
- RULER: 3 of 4 launch tasks slightly better than hand-crafted rewards; tasks and numbers not given on the docs page — [ART docs](https://art.openpipe.ai/fundamentals/ruler)

### Inferences
- These results come from benchmarks with clean automatic graders (tests pass, SQL executes, task completes). DCLab's deterministic grader (leakage traps, holdout rules, validator) is the same kind of signal — that is a strength.
- SFT data need: DCLab trajectories are mostly rule-based and self-play. SFT on them teaches the model to imitate the rules; it cannot exceed the rules. RL or preference data is where the model could improve past them.
- Minimum before training is worth it (judgement, not a sourced number): SFT on a few hundred diverse, validated trajectories can teach format and tool use; a meaningful judgement-quality gain needs a held-out set the model has never seen (DCLab's sealed sets) and probably ≥1,000 trajectories spread across task kinds. For GRPO, what matters is the number of distinct tasks with a grader (hundreds) and 4–8 rollouts per task.

### Gaps
- ART·E numbers (accuracy, training cost, GPU) could not be fetched: the OpenPipe blog pages returned only a title. My recollection (Qwen2.5-14B beating o3, trained on one GPU for under ~$100) is unverified.
- No source reports agent SFT/RL results on tabular ML / data-science agent tasks at DCLab's data scale.

## 5. Maturity, license, maintenance, model support

### Takeaway
All candidates are permissively licensed (MIT or Apache-2.0) and active in 2026. verl is the most widely used RL backbone (~24k stars) and several others (Agent Lightning, rLLM, verl-agent) build on it. TRL is the most stable for SFT/DPO. Agent Lightning v1.0 and TRL's `environment_factory` are new (2026) and partly experimental.

### Cited Findings
- Agent Lightning: MIT; v1.0 open-sourced Aug 2026; v1.0 technical report Aug 2026; MoE coding example Sep 2026 — [GitHub](https://github.com/microsoft/agent-lightning)
- ART: Apache-2.0; ~10.8k stars, 1,548 commits; Qwen-centred examples (Qwen2.5, Qwen3, Qwen3.x), GPT-OSS, Llama; Gemma 3 not supported; built on Unsloth, vLLM, trl, torchtune — [GitHub](https://github.com/OpenPipe/ART)
- rLLM: Apache-2.0; ~5.9k stars; Python ≥3.11; 60+ integrated benchmarks; UI and on-policy distillation (Mar 2026) — [GitHub](https://github.com/rllm-org/rllm)
- verl: Apache-2.0; ~24k stars, 4.8k forks; moved to the verl-project org Jan 2026; recipes split into verl-recipe; VeRL-Tinker (Aug 2026) runs SFT, RL and distillation on verl-managed GPUs; uni-agent (May 2026) — [GitHub](https://github.com/volcengine/verl)
- verl-agent: Apache-2.0; ~2.4k stars; models Qwen3, Qwen3-VL, Qwen2.5(-VL), Llama 3.2; last stated full upstream merge June 2025 — [GitHub](https://github.com/langfengQ/verl-agent)
- SkyRL: Apache-2.0; v0.3.0 Jul 2026; Tinker API (Feb 2026); APEX-Agents long-horizon recipe (Sep 2026) — [GitHub](https://github.com/NovaSky-AI/SkyRL)
- OpenRLHF: 0.10 added multi-turn VLM RL — [OpenRLHF docs](https://openrlhf.readthedocs.io/en/latest/agent_paradigm.html)
- TRL: `environment_factory` experimental, requires transformers ≥5.2.0; tool training tested on Qwen2.5/3/3.5/3.6 — [TRL GRPO docs](https://huggingface.co/docs/trl/main/en/grpo_trainer)
- mlx-lm-lora: Apache-2.0; ~451 stars, 505 commits; single main maintainer project; includes an MCP server to start and monitor runs — [GitHub](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- PyTorch Conference EU 2026 talk on multi-turn, multi-environment RL with TRL (OpenEnv) says the hard part becomes rollout orchestration: environments, tools, masking, batching, rewards — [slides](https://hosted-files.sched.co/pytorchconferenceeu2026/c3/From%20Responses%20To%20Trajectories_%20Multi-Turn%20and%20Multi-Environment%20Reinforcement%20Learning.pdf)

### Inferences
- verl-agent may lag upstream verl (last stated full merge mid-2025); prefer frameworks on current verl (Agent Lightning v1.0, rLLM) or TRL.
- mlx-lm-lora is useful but has a small maintainer base (bus-factor risk); the official mlx-lm is safer for plain SFT.
- Qwen chat templates with tool calls are best supported in TRL (auto-patched training templates for Qwen3). For mlx-lm, verify the rendered template on a few trajectories before training.

### Gaps
- Release tags/dates were not visible on most GitHub pages (empty Releases sections in fetched content).

## 6. Recommendation for DCLab's stage

### Takeaway
Now: SFT (QLoRA) of Qwen ~8B on the M1 Max with mlx-lm, using DCLab's own trajectory export, one example per assistant turn; DPO/ORPO on the 220 pairs with mlx-lm-lora as a small second experiment. Later, on rented GPUs with the owner's approval: GRPO with the deterministic grader through a proxy-style framework (Agent Lightning v1.0 or rLLM SDK), or TRL GRPO with `rollout_func` if a smaller dependency is preferred. Do not try agent RL on the Mac.

### Cited Findings
- mlx-lm supports `tools` and `chat` formats, QLoRA via a quantized model, LoRA/DoRA/full — [mlx-lm LORA.md](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/LORA.md)
- mlx-lm-lora supports DPO/ORPO on the Mac at small memory (ORPO 2–4 GB on 0.6B) — [GitHub](https://github.com/Goekdeniz-Guelmez/mlx-lm-lora)
- TRL SFT supports tool-calling datasets and assistant-only loss with auto-patched Qwen3 templates — [TRL SFT docs](https://huggingface.co/docs/trl/main/en/sft_trainer)
- Proxy-style capture without agent rewrites: Agent Lightning v1.0 — [GitHub](https://github.com/microsoft/agent-lightning); rLLM SDK — [GitHub](https://github.com/rllm-org/rllm); ART client — [GitHub](https://github.com/OpenPipe/ART)
- Managed RL without owning GPUs: Tinker backend in rLLM/SkyRL; W&B Serverless RL in ART (vendor claims) — [rLLM](https://github.com/rllm-org/rllm), [ART](https://github.com/OpenPipe/ART)

### Inferences
- Step plan (judgement):
  1. Export trajectories to the chat+tools JSONL shape that both mlx-lm and TRL read (`messages` with `tool_calls`, `tool` messages, `tools` schemas). Same file works on the Mac now and on GPUs later.
  2. Hold out whole projects/tasks (DCLab's sealed sets) before any training.
  3. QLoRA SFT on the Mac; compare against the base model and the rules on the dev set with the existing gates.
  4. If SFT clearly helps, DPO on the preference pairs (Mac).
  5. Only then rent GPUs for GRPO with the deterministic grader as reward; start with a proxy framework so the harness is unchanged.
- Main risk: SFT on rule-based and self-play trajectories copies the rules, including their mistakes, and the gain may be zero against the rules themselves. Measure against the rules, not just the base model.
- Second risk: 8B context budget. Long trajectories may need truncation or per-step context (verl-agent's step-independent idea), which changes what the model sees versus inference.

### Gaps
- No published comparison of these frameworks head-to-head on the same agent task was found.
- Minimum data thresholds above are reasoned, not sourced.
