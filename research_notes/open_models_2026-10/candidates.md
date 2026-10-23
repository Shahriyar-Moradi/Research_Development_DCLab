# Open-weight models DCLab could fine-tune (state on 2026-10-23)

Purpose: choose bases for DCLab's own models: (1) the agentic **solver** (calls DCLab's tools over many turns), (2) the **workflow** model (step-by-step ML reasoning plus Python), (3) **students** of 1–4B (quick decisions). The license must allow fine-tuning **and** a paid hosted service. Training: Apple M1 Max (32 or 64 GB) with MLX. Serving: Ollama, MLX or vLLM. Persian matters.

How this was made: five research passes (one per group of families) read Hugging Face model cards, LICENSE files, `config.json` and `chat_template.jinja` files directly, plus vendor blogs, ollama.com library pages, mlx-community listings and GitHub issues, on 2026-10-23. The key facts of the top candidates (repo exists, creation date, license tag, parameter count; the Qwen 3.6/3.8 `preserve_thinking` defaults; Gemma 4's history rule) were re-checked against the Hugging Face API and the raw template files in a second pass.

Legend: **(secondary)** = a non-official source. **UNVERIFIED** = not confirmed from a primary source. All benchmark numbers are the vendors' own, on their own harnesses, unless marked. They are not independent and often not comparable across vendors (example: Qwen3.6-35B-A3B SWE-bench Verified is 73.4 on Qwen's card but 70.12 in NVIDIA's comparison table). License notes are readings of the license text, not legal advice.

---

## 1. Summary: the strongest candidates for DCLab

| # | Model (HF repo) | Size total / active | Context | Released | License (fine-tune + paid hosting?) | Tools / thinking | Coding & agent (vendor) | Persian evidence | Mac/serving | Best role |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **Qwen3.6-35B-A3B** (`Qwen/Qwen3.6-35B-A3B`) | 35B / 3B MoE (+vision) | 262K | 2026-04-16 | Apache-2.0: yes | Qwen3-Coder XML; `enable_thinking`; `preserve_thinking` **off** by default | SWE-V 73.4, TAU3 67.2, MCP-Atlas 62.8, LCB v6 80.4 | Qwen3 listed Persian; 3.5+ list unpublished; no eval of 3.6 | Ollama `qwen3.6:35b`, mlx-community 4/8-bit; bf16 LoRA ≈ 74 GB (Unsloth) | Solver |
| 2 | **Qwen3.8-27B** (`Qwen/Qwen3.8-27B`) | 27B dense (+vision) | 262K (1M YaRN) | 2026-08 | Apache-2.0: yes | Qwen3-Coder XML; `reasoning_effort` xhigh/medium/low; `preserve_thinking` **on** by default | SWE-Pro 61.7, LCB v6 90.3, Terminal-Bench 2.1 73.0 (no SWE-V published) | Open Persian LLM Leaderboard (OPLL) avg 76.1 (#2 open) | Ollama `qwen3.8:27b`, mlx-community; bf16 LoRA ≈ 56 GB | Workflow (and strong solver) |
| 3 | **Gemma 4 26B-A4B** (`google/gemma-4-26B-A4B-it`) | 25.2B / 3.8B MoE | 256K | 2026-03-31 | Apache-2.0: yes (first Apache Gemma) | own tokens `<\|tool_call>call:…`; `<\|think\|>` / `enable_thinking`; history thinking **dropped**, kept in tool turns | τ2 avg 68.2, LCB v6 77.1 | OPLL avg 70.7 | Ollama `gemma4:26b`, mlx-community 4-bit; MLX LoRA UNVERIFIED | Solver or workflow with Persian |
| 4 | **Gemma 4 31B** (`google/gemma-4-31B-it`) | 30.7B dense | 256K | 2026-03-31 | Apache-2.0: yes | same as above | τ2 avg 76.9, LCB v6 80.0 | **OPLL avg 78.6 (#1 open)** | Ollama `gemma4:31b`; dense → slow training | Workflow / Persian teacher |
| 5 | **GLM-4.7-Flash** (`zai-org/GLM-4.7-Flash`) | 31B / ~3B MoE | 198K | 2026-01-19 | MIT: yes | `glm47` parser; preserved thinking (keep all, unmodified) | SWE-V 59.2, τ² 79.5, LCB v6 64.0 | en/zh only listed | Ollama `glm-4.7-flash`, mlx-community; MLX LoRA UNVERIFIED | Solver |
| 6 | **gpt-oss-20b** (`openai/gpt-oss-20b`) | 20.9B / 3.6B MoE (MXFP4) | 131K | 2025-08-05 | Apache-2.0 + "comply with law" policy: yes | harmony format; effort low/med/high, no off; drop CoT after final, keep in tool chain | SWE-V 60.7, τ-Retail 54.8 | not evaluated; knowledge cutoff 2024-06 | Ollama `gpt-oss:20b`, mlx-community; past MLX fine-tune bugs | Solver baseline (most battle-tested) |
| 7 | **Muse Glimmer 30B** (`meta-models/Muse-Glimmer-30B`) | 29.8B dense (+vision) | 131K+ | 2026-08 | Apache-2.0: yes; separate `USAGE_POLICY.md` (binding? UNVERIFIED) | own ATEM format; reasoning strength low…xhigh; template keeps all supplied reasoning | SWE-V 76.0, MCP-Atlas 75.5 | ">100 languages", Persian UNVERIFIED | Ollama `muse-glimmer:30b`, mlx-community; open vLLM structured-output bugs | Solver / workflow (if dense training is OK) |
| 8 | **Nemotron 3.5 Lightning 30B-A3B** (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`) | ~31.6B / 3B Mamba-MoE hybrid | up to 1M | 2026-08-11 | **OpenMDW-1.1**: yes, the cleanest terms | Qwen3-Coder XML; `enable_thinking`; history thinking dropped, kept in tool chain | SWE-V 51.6 (weaker coder) | not listed (EN/ES/FR/DE/IT/JA) | Ollama `nemotron-3.5-lightning:30b`, mlx-community; MLX LoRA on Mamba UNVERIFIED | Solver (license-first choice) |
| 9 | **Devstral Small 2 24B** (`mistralai/Devstral-Small-2-24B-Instruct-2512`) | 24B dense | 256K | 2025-12 | Apache-2.0: yes | Mistral format; **no thinking** | SWE-V 68.0 (HF card; Ollama page 65.8) | not stated | Ollama `devstral-small-2:24b`, mlx-community 4-bit | Coding-heavy solver |
| 10 | **MiMo-V2.6-Distill-Qwen-9B** (`XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B`) | 9.4B dense (Qwen3.5-9B base) | 262K (base) | 2026-09-21 | MIT on Apache-2.0 base: yes | Qwen3-Coder XML; `enable_thinking`; template keeps history thinking | SWE-V 61.1, SWE-Pro 44.6, Toolathlon 35.2 | via Qwen3.5, UNVERIFIED | HF only; no MLX/Ollama build; "research SFT checkpoint" | Mid-size solver (fits bf16 LoRA ≈ 22 GB) |
| S1 | **Qwen3.5-4B / 2B / 0.8B** (`Qwen/Qwen3.5-4B` …) | 4.7B / 2B / 0.8B dense | 262K | 2026-03-02 | Apache-2.0: yes | XML tools; 2B/0.8B non-thinking by default | BFCL-V4 4B 50.3, 2B 43.6; TAU2 4B 79.9 | Qwen3-4B OPLL 50.5 (older gen) | Ollama, mlx-community; bf16 LoRA 5–10 GB | Students |
| S2 | **Gemma 4 E4B / E2B** (`google/gemma-4-E4B-it`) | 4.5B eff. (8B with embeddings) / 2.3B eff. | 128K | 2026-03-31 | Apache-2.0: yes | as Gemma 4 | τ2 E4B 42.2, LCB 52 | **OPLL E4B 59.0, E2B 50.5** (best small Persian) | Ollama `gemma4:e4b`; community MLX LoRA worked, fused model reported broken | Students (Persian-facing) |
| S3 | **Granite 4.2-3B** (`ibm-granite/granite-4.2-3b`) | 3.66B dense | 128K | 2026-08 | Apache-2.0: yes | Qwen3-Coder XML; thinking on/off/low; history thinking stripped by default | BFCL v4 52.4, LCB v6 69.7 | not listed; multilingual weak (MMLU-ProX 27.8) | **official IBM MLX builds**, Ollama `granite4.2:3b` | Students (English decisions) |

Also worth a look, but not in the top group: **Ministral 3 3B/8B/14B** (Apache-2.0, 256K, Base+Instruct+Reasoning, Ollama + MLX; Persian not listed), **Magistral Small 1.2 24B** (Apache-2.0; the only small model whose card names **Farsi**; no MLX build found), **Nemotron 3 Nano 4B** (commercial OK, BFCL v3 61.1, English only), **Hermes 4 14B** (Apache-2.0 via Qwen3-14B, Hermes `<tool_call>` JSON, 40K context, low-refusal by design), **Apriel-1.6-15B-Thinker** (MIT, BFCL v3 63.5, always thinks, English-centric), **North Mini Code 1.0 30B-A3B** (Cohere, Apache-2.0, SWE-V 61.0, keep-all-thinking rule), **Seed-OSS-36B** (Apache-2.0, 512K, SWE-V 56; just over the size limit), **Qwen3-4B-Instruct-2507** (Apache-2.0, plain transformer: the most mature tooling for a student fallback).

### Suggested shortlist per role (my judgement from the evidence)
- **Solver:** Qwen3.6-35B-A3B first; Gemma 4 26B-A4B as the Persian-strong alternative; GLM-4.7-Flash or gpt-oss-20b as a second architecture to compare. Nemotron 3.5 Lightning if license simplicity outweighs coding strength.
- **Workflow:** Qwen3.8-27B (strongest code + good Persian) or Gemma 4 31B (best Persian). Both are dense, so training is slow on an M1 Max.
- **Students:** Qwen3.5-2B/4B and Gemma 4 E2B/E4B, measured against each other on DCLab's own decision sets and a Persian probe; Granite 4.2-3B for English-only decisions with the smoothest MLX path.
- **Teachers / labellers (not fine-tuned here, self-hosted or rented GPUs):** DeepSeek-V4-Flash-0731 (MIT, card allows distillation), GLM-5.x (MIT), Mistral Small 4 (Apache-2.0, lists `fa`), Command A+ 05-2026 (Apache-2.0, lists `fa`).

---

## 2. Key caveats (read before choosing)

1. **Persian is the biggest unknown.** No commercial-OK model publishes an official Persian evaluation. The only comparative data is the Open Persian LLM Leaderboard (Part AI + Amirkabir; averages computed by us from its data file, so (secondary)). The newest rows (Gemma 4, Qwen3.8) look like a changed protocol (e.g. one law task jumps to ~98) and one row is mislabelled. Treat it as a rough ranking. Dedicated Persian fine-tunes (Dorna/Dorna2, PersianMind, Maral) score at or below the general bases; PersianMind is CC-BY-NC. DCLab must run its own Persian probe before choosing.
2. **Earlier turns' thinking: every family has a different rule.** SFT data must be rendered with the exact template of the base being trained, and served with the same one.

   | Base | Earlier user turns' thinking | Inside the current tool chain | Switch |
   |---|---|---|---|
   | Qwen3.8 | **kept** by default (`preserve_thinking` on; writes empty `<think></think>` when none) | kept | `preserve_thinking: false` to drop |
   | Qwen3.6 | dropped by default | kept | `preserve_thinking: true` to keep |
   | Qwen3.5 (small) | dropped | kept | no option |
   | Gemma 4 | dropped ("historical model output should only include the final response") | kept (tool-call turns) | — |
   | gpt-oss (harmony) | dropped after a `final` message | kept | — |
   | Nemotron 3 / 3.5 | dropped (`truncate_history_thinking` default True) | kept | kwarg |
   | Granite 4.2 | dropped (`truncate_history_thinking=True`) | UNVERIFIED | kwarg |
   | GLM-4.7(-Flash)/5.x | **kept, unmodified** (preserved thinking; `clear_thinking=false`) | kept | `clear_thinking` |
   | MiniMax M2 | **kept** ("Do not remove the `<think>` part") | kept | — |
   | Muse Glimmer, Seed-OSS, MiMo distill | template renders whatever reasoning you pass | kept | caller decides |
   | Cohere North Mini Code | **keep all** (Cohere's advice) | kept | — |
   | Apriel 1.6, Hermes 4 | dropped (Hermes: `keep_cots=True` keeps) | Hermes: kept in turn | — |
   | DeepSeek V4 | dropped (`drop_thinking=True`), but **kept for every turn when tools are present** | kept | encoder flag |

   Template details that break SFT silently: Qwen3.8 and Muse Glimmer require tool-call `arguments` as **objects**, not JSON strings (Muse's template raises an error). Mistral's `[THINK]` must be special tokens. DeepSeek V4 has no Jinja template (Python encoder, "DSML" tool format), so its traces must be converted before training a Qwen/Gemma student.
3. **License traps (verified from LICENSE texts unless noted).**
   - **Qwen3.8-Flash-Next** (Qwen Community License 1.0): a "Model as a Service" or "AI Work Assistant" business must get a separate license before commercial use. A hosted DCLab could plausibly fall under it. All other Qwen models above are plain Apache-2.0.
   - **Cohere** Command A, R7B, Aya Expanse, Tiny Aya: CC-BY-NC (the best-documented Persian models, but not sellable). Cohere's new North models and Command A+ are Apache-2.0.
   - **LG EXAONE 4.0/4.5:** "EXAONE AI Model License 1.2 - NC": research only, no commercial use, outputs may not build competing models, names must start with "EXAONE". Excluded.
   - **Tencent Hunyuan 0.5–7B, A13B:** does not apply in the EU, UK, South Korea; bans using outputs to improve other models (distillation included); AUP bans "high-stakes automated decisions" (credit, employment, insurance, housing). Excluded.
   - **MiniMax:** M2.7 non-commercial; M2.5 bans "fully automated decision-making that adversely affects an individual's legal rights"; M3/M3.1-Flash need "Built with MiniMax …" and a notice, and authorization above $20M/year. No small MiniMax model exists anyway.
   - **Liquid LFM2.5:** LFM Open License v1.0, commercial rights end at **$10M annual revenue**.
   - **Mistral Medium 3.5, Devstral 2 (123B):** Modified MIT, rights end above $20M **monthly** revenue. (Devstral Small 2, Ministral 3, Magistral Small, Mistral Small 4 are Apache-2.0.)
   - **Llama 4:** "Built with Llama", derived model names must start with "Llama", 700M MAU cap, multimodal rights withheld from EU-domiciled companies. Too big anyway. No Llama 5 / 4.x exists; Meta's new open line is Muse.
   - **xAI Grok 2:** revocable commercial grant, bans using outputs to train other models. Not usable.
   - **NVIDIA Nemotron Nano 9B/12B v2** (older NVIDIA Open Model License): rights end if a fine-tune weakens safety guardrails. Newer Nemotron 3 (Nemotron Open Model License) and 3.5 (OpenMDW-1.1) do not have that clause.
   - **Teachers through APIs:** Z.ai's and Kimi's API terms ban using API outputs to train competing/external models. Distil from self-hosted MIT weights, not their APIs. DeepSeek API terms: UNVERIFIED.
   - **Muse Glimmer** ships a Meta-style `USAGE_POLICY.md` beside the Apache license; whether it binds is UNVERIFIED (ask a lawyer).
4. **Training on the M1 Max is the practical limit.**
   - Qwen3.5/3.6/3.8 use Gated DeltaNet layers; in mlx-lm the LoRA path does not use the fast Metal kernel (`use_kernel=not self.training`), so speed is UNVERIFIED. Unsloth (secondary) advises **against 4-bit QLoRA on any Qwen3.5-family model**; its bf16 LoRA memory: 4B 10 GB, 9B 22 GB, 27B 56 GB, 35B-A3B 74 GB. So on 64 GB the 27B/35B fit only as 4-bit QLoRA, the warned-against path. Measure quality before trusting it.
   - MLX LoRA is UNVERIFIED for Gemma 4 26B-A4B, Nemotron-H (Mamba), GLM-4.7-Flash, Muse Glimmer, Ling 3.0 and MiMo distill. Run a 10-step LoRA smoke test and a fused-vs-adapter equality check per base.
   - Students (≤4B) and 9B fit comfortably in bf16 LoRA.
5. **Serving bugs to test around:** mlx-lm #1293 (open): `mlx_lm.server` returns no `tool_calls` for Qwen3.5/3.6. Ollama #14493: Qwen3.5 renderer used the wrong tool format and left `</think>` open (partly fixed). gpt-oss harmony parsing: vLLM #58825, Ollama #12203/#12187/#12884 (open). Gemma 4's non-JSON tool tokens break layers that assume OpenAI JSON (LiteLLM, `anyOf`+null). Muse Glimmer: vLLM structured outputs not enforced with its reasoning parser (#53469). LFM's Pythonic tool calls can mis-parse Python-heavy answers. Test tool calling on each runtime before trusting any evaluation.
6. **Names that do not exist (as of 2026-10-23):** Qwen3.7 open weights, small Qwen3.6/3.8 sizes, DeepSeek "V4 Lite" or small V4 models, Llama 5 / Llama 4.x, gpt-oss v2, Phi-5, Granite 5 LLMs, Olmo 4, LFM3, SmolLM4, Hermes 5, Grok 3 weights. Mistral Large 4 weights were promised "by the end of the month" (announced 2026-10-06); license unpublished.

---

## 3. Detailed notes per family

The sections below are the full research notes, with sources inline.


---

## Open-weight candidates: Alibaba Qwen and DeepSeek (checked 2026-10-23)

Method: the Hugging Face Hub listings of the `Qwen` and `deepseek-ai` organisations (sorted by creation date), the model cards, `LICENSE` files and `chat_template.jinja` files read directly from the repos, the ollama.com library pages, the mlx-community listings, the mlx-lm source on GitHub, and GitHub/HF issue threads. "(secondary)" marks a non-official source. "UNVERIFIED" marks what I could not confirm.

**Which names exist (Oct 2026):**
- Qwen open weights: Qwen3 / Qwen3-2507 (2025), Qwen3-Coder-Next (Feb 2026), **Qwen3.5** (Feb–Mar 2026), **Qwen3.6** (Apr 2026: only 35B-A3B and 27B), **Qwen3.8** (Aug 2026: 27B dense, Flash-Next 125B-A6B, 2.4T-A95B). Source: [QwenLM/Qwen3.5 README news](https://github.com/QwenLM/Qwen3.5), [HF Qwen org](https://huggingface.co/Qwen).
- **Qwen3.7 has no open weights** that I could find. "Qwen3.7-Plus" appears only as a comparison column (an API model) on the Qwen3.8-27B card.
- **There is no small Qwen3.6 or Qwen3.8 (no 0.8B–9B, no 3.8 35B-A3B).** The newest small Qwen models are still Qwen3.5-0.8B/2B/4B/9B.
- DeepSeek open weights: V3.x (2025), V3.2 (Dec 2025), **V4-Pro / V4-Flash** (Apr 2026, "preview"), **V4-Flash-0731** (the official V4-Flash), **V4-Pro-0813**, **V4-Flash-Vision-Exp** (Aug 2026), **V4.1-Flash** (Sep 2026) and **V4.1-Flash-1017** (Oct 2026). [HF deepseek-ai org](https://huggingface.co/deepseek-ai).
- **"DeepSeek V4 Lite" does not exist.** There are **no small DeepSeek V4 models** and no V4 distills. The only small DeepSeek LMs are the 2025 R1 distills (on Qwen2.5 / Llama / Qwen3-8B). DeepSeek's June 2026 `dspark_*` / `eagle3_*` / `dflash_*` repos for Qwen3-4B/8B/14B and Gemma4-12B are speculative-decoding draft heads, not standalone chat models.

---

### Alibaba Qwen

#### Licence summary (read this first)
- **Apache-2.0** (verified in each repo's `LICENSE` / card metadata): Qwen3, Qwen3-2507, Qwen3-Coder-Next, all Qwen3.5 sizes, Qwen3.6-35B-A3B, Qwen3.6-27B, **Qwen3.8-27B** ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/LICENSE) is the standard Apache License 2.0 text). Apache-2.0 allows fine-tuning, derivatives and commercial hosted use. It has no user caps, no naming rule and no ban on training other models. You must keep the licence and NOTICE files and mark modified files.
- **Qwen Community License 1.0** (new in Aug 2026): **Qwen3.8-Flash-Next** ([LICENSE](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)). The text says use is free, including the right to "sell, deploy, host, fine-tune, and create derivative works", subject to two conditions:
  1. Above 100,000,000 MAU or US$20,000,000 monthly revenue, "respective model name must be prominently displayed on the user interface".
  2. "If the licensee or any of its affiliates conducts a Model as a Service or AI Work Assistant business, the licensee shall obtain a separate license from Qwen before Using the Software or its derivative works for any commercial purpose." The licence defines "Model as a Service" as giving third parties inference or fine-tuning access "in a manner that allows such third parties to exercise meaningful control over the inputs, parameters, or training data". It defines "AI Work Assistant" as a product "primarily designed for AI-assisted coding or office productivity".
  - **Risk for DCLab:** a hosted ML-building service whose users type the prompts, or a hosted agent that writes Python, could plausibly count as MaaS or an "AI Work Assistant". Treat Flash-Next as **not safe for DCLab's hosted use without a separate licence or legal advice**. It is also too big for the M1 anyway.
- **Qwen3.8-2.4T-A95B:** `license_name: qwen3.8-max`, licence text not read (the model is far too large). UNVERIFIED terms.

#### Qwen3.8-27B (dense, the newest mid-size Qwen) — best "workflow" and "solver" base in this family
- Repo [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (HF created 2026-08-05; GitHub news says released 2026-08-14). FP8: `Qwen/Qwen3.8-27B-FP8`. **Licence: Apache-2.0** (verified).
- 27B dense. Hybrid layout `16 × (3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN))`, so 3 of 4 layers are linear attention. Native vision-language model (image and video). MTP head included. **Context 262,144 tokens natively**, up to 1,000,000 with YaRN.
- **Thinking:** on by default. Off per request with `chat_template_kwargs: {"enable_thinking": false}` (template pre-fills an empty `<think>\n\n</think>\n\n`). `reasoning_effort` ∈ `xhigh` (default) | `medium` | `low`. The template turns this into a system-prompt sentence. A value outside these raises an exception. `/think` and `/no_think` soft switches are not supported in 3.5/3.6, and the 3.8 card does not mention them.
- **Earlier turns' thinking: `preserve_thinking` is ON by default in 3.8.** The card says: "By default, Qwen3.8 retains thinking blocks from all historical messages". Set `preserve_thinking: false` to keep only the thinking after the latest user message. I read [chat_template.jinja](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/chat_template.jinja). An assistant turn renders `<think>\n{reasoning_content}\n</think>\n\n{content}` when `preserve_thinking` is undefined/true or the turn is after the last real user query. Otherwise only `{content}` is rendered. Tool responses do not count as user queries. **So interleaved thinking inside a tool loop is always kept**, and thinking from older user turns is kept only with preserve on. With preserve on and an empty `reasoning_content`, the template still writes an empty `<think>\n\n</think>` block. SFT data must match this exactly.
- **Tool calling:** XML-in-text, the "Qwen3-Coder" format, not Hermes JSON. Tools are listed as JSON inside `<tools>…</tools>` in the system prompt. The call format is `<tool_call>\n<function=NAME>\n<parameter=ARG>\nvalue\n</parameter>\n</function>\n</tool_call>`. Results go back as `<tool_response>…</tool_response>` inside a user turn. The model may write reasoning text before the call, "but NOT after". Template detail: `tool_call.arguments|items` needs arguments as a **dict**, not a JSON string. Store SFT arguments as objects. No official BFCL/τ-bench on this card. No native JSON-schema "structured output" mode in the model; use the serving engine's guided decoding.
- **Coding/agent (official card; Claude Code harness for SWE runs):** SWE-bench Pro 61.7 (Qwen3.6-27B 53.5), Terminal-Bench 2.1 73.0, NL2Repo 42.3, LiveCodeBench v6 90.3, IFBench 79.5, GPQA-D 89.2, CoWorkBench 70.7. **No SWE-bench Verified number is published for 3.8-27B.** Several of these are in-house benchmarks.
- **Languages:** the 3.8 card gives no language list. It is built on Qwen3.5 (201 languages, see below). Persian: see the Qwen3.5 section.
- **Serving / availability:** vLLM, SGLang and TokenSpeed recipes are linked from the card. Ollama: `qwen3.8:27b` (18 GB) and `qwen3.8:27b-mlx`, tagged vision/tools/thinking ([ollama.com/library/qwen3.8](https://ollama.com/library/qwen3.8)). MLX: `mlx-community/Qwen3.8-27B-4bit`, `-8bit`, `-bf16`, `-mxfp4`, `-nvfp4`, `-MTP-4bit` (and others) ([HF search](https://huggingface.co/mlx-community?search_models=Qwen3.8)).
- **Known issues:** community reports say long agentic runs still have frequent tool-call failures under llama.cpp. The fix offered is a community chat template; there is no official reply ([HF discussion #68](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/68)). One report of over-thinking even at low effort in vLLM FP8 ([#82](https://huggingface.co/Qwen/Qwen3.8-27B/discussions/82)). Community fixed templates exist: [froggeric / eemin Qwen-Fixed-Chat-Templates](https://huggingface.co/eemin/Qwen-Fixed-Chat-Templates) and [nrrso vLLM fixes](https://huggingface.co/nrrso/Qwen3-Chat-Template-vLLM-Fixes) (secondary; nrrso says use `--tool-call-parser qwen3_xml`). A vendor blog reports Ollama `/v1/chat/completions` hanging for qwen3.8:27b ([betterclaw](https://www.betterclaw.io/blog/qwen-3-8-27b-tool-calling-fails-agents-fix), secondary, UNVERIFIED). Which vLLM tool parser the official 3.8 recipe uses: UNVERIFIED (the 3.6 card says `qwen3_coder`).

#### Qwen3.6-35B-A3B (MoE, small active) and Qwen3.6-27B (dense)
- Repos [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) (released 2026-04-16) and [Qwen/Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) (2026-04-22), plus `-FP8` versions. **Apache-2.0** (verified).
- 35B-A3B: 35B total / **3B active**. 256 experts, 8 routed + 1 shared. 40 layers, same Gated DeltaNet hybrid. 27B: dense. Both are vision-language. **Context 262,144** native, ~1,010,000 with YaRN.
- **Thinking:** on by default. Off with `enable_thinking: false`. "Qwen3.6 does not officially support the soft switch of Qwen3, i.e., `/think` and `/nothink`."
- **`preserve_thinking` exists in 3.6 but is OFF by default** (the opposite of 3.8). The card says: "By default, only the thinking blocks generated in handling the latest user message is retained … Qwen3.6 has been additionally trained to preserve and leverage thinking traces from historical messages." Turn it on with `chat_template_kwargs: {"preserve_thinking": true}`.
- **Tool calling:** same XML Qwen3-Coder format. The official serving flags are `--reasoning-parser qwen3 --tool-call-parser qwen3_coder` (vLLM also `--enable-auto-tool-choice`); vLLM ≥0.19.0 and SGLang ≥0.5.10 are recommended.
- **Official numbers (35B-A3B / 27B):** SWE-bench Verified **73.4 / 77.2**, SWE-bench Multilingual 67.2 / 71.3, SWE-bench Pro 49.5 / 53.5, Terminal-Bench 2.0 51.5 / 59.3, LiveCodeBench v6 80.4 / 83.9. 35B-A3B also: TAU3-Bench 67.2, MCPMark 37.0, MCP-Atlas 62.8, Tool Decathlon 26.9. These come from an internal agent scaffold.
- **Availability:** Ollama `qwen3.6:35b` (23–24 GB), `qwen3.6:27b` (18–19 GB), and `-mlx` variants ([ollama.com/library/qwen3.6](https://ollama.com/library/qwen3.6)). MLX: `mlx-community/Qwen3.6-35B-A3B-4bit`/`-8bit`/`-bf16`/`-4bit-DWQ`, `mlx-community/Qwen3.6-27B-4bit`/`-8bit`/`-bf16`.
- **Known issues:** [mlx-lm #1293](https://github.com/ml-explore/mlx-lm/issues/1293) (open since 2026-05-21): `mlx_lm.server` returns no `tool_calls` for Qwen3.5-35B-A3B and Qwen3.6-27B. Its tool-parser auto-detection only matches Qwen3-Coder templates. **This matters for MLX serving: parse the XML yourself or use another server.** Community template fixes target tool-call history pruning, whitespace around think blocks and KV-cache invalidation (secondary, see above).

#### Qwen3.5 small: 0.8B, 2B, 4B, 9B (the newest small Qwen) — the "student" candidates
- Repos `Qwen/Qwen3.5-0.8B`, `-2B`, `-4B`, `-9B` and `-Base` versions, released 2026-03-02. **Apache-2.0** (verified). All are vision-language, use the Gated DeltaNet hybrid and have **262,144 context**. Card: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B).
- **Thinking default differs by size:** "Qwen3.5-0.8B operates in non-thinking mode by default" and so does 2B ([2B card](https://huggingface.co/Qwen/Qwen3.5-2B)). Both warn the model "is more prone to entering thinking loops" in thinking mode. 4B and 9B think by default. The switch is `enable_thinking`; no `/think` soft switch. **No `preserve_thinking` in 3.5** (it was introduced as new in 3.6).
- **Official agent/tool numbers** (4B card table): BFCL-V4: 9B **66.1**, 4B **50.3**, 2B 43.6, 0.8B 25.3. TAU2-Bench: 9B 79.1, 4B 79.9, 2B 48.8, 0.8B 11.6. LiveCodeBench v6: 9B 65.6, 4B 55.8. IFEval: 9B 91.5, 4B 89.8. Multilingual: MMMLU 9B 81.2 / 4B 76.1, MMLU-ProX (29 languages) 76.3 / 71.5, WMT24++ (55 languages) 72.6 / 66.6.
- 2B/0.8B card: "the intended use cases are prototyping, task-specific fine-tuning, and other research or development purposes". This is what students are.
- **Languages:** "Expanded support to 201 languages and dialects." **The full list is not published** in the cards or the GitHub README. Whether Persian is named in Qwen3.5's list is **UNVERIFIED**. The predecessor Qwen3 (119 languages) explicitly lists **Persian, Dari and Tajik** ([Qwen3 blog](https://qwenlm.github.io/blog/qwen3/)). I found **no official Persian eval for any Qwen 3.5/3.6/3.8 model**. Check the [Open Persian LLM Leaderboard](https://huggingface.co/spaces/opll-org/Open-Persian-LLM-Leaderboard) (secondary) or run DCLab's own Persian set.
- **Availability:** Ollama `qwen3.5:0.8b` (1.2 GB), `:2b`, `:4b`, `:9b`, and `-mlx` variants ([ollama.com/library/qwen3.5](https://ollama.com/library/qwen3.5)). MLX: `mlx-community/Qwen3.5-4B-MLX-4bit`/`-8bit`/`-bf16`, `mlx-community/Qwen3.5-2B-MLX-4bit`, `mlx-community/Qwen3.5-0.8B-MLX-4bit`, `mlx-community/Qwen3.5-9B-MLX-4bit`.
- **Known issues:** [ollama #14493](https://github.com/ollama/ollama/issues/14493) (open, comments to Sep 2026) says Ollama's `qwen3.5` renderer/parser used Hermes JSON instead of the model's Qwen3-Coder XML, never closed `</think>`, and silently ignored repetition/presence penalties. Fixes were partly shipped (v0.17.3 parser `</think>` fix). Users still report failures, and a maintainer disputes "completely non-functional". **Do not trust Ollama for tool-calling evaluation of Qwen3.5 without testing it.**

#### Older but still relevant Qwen
- **Qwen3-4B-Instruct-2507 / Qwen3-4B-Thinking-2507** ([Instruct](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507), [Thinking](https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507)), Aug 2025, **Apache-2.0**, 4.0B, 262,144 context. These are **separate checkpoints**: Instruct "supports only non-thinking mode"; Thinking "supports only thinking mode" (the template auto-adds `<think>`). Tool format is **Hermes-style JSON in `<tool_call>`**, unlike 3.5+'s XML. BFCL-v3: Instruct 61.9, Thinking 71.2. TAU2-Retail 40.4 / 53.5. LiveCodeBench v6 35.1 / 55.2. Text-only, plain transformer (no DeltaNet), so it is the **most mature tooling target** (MLX, llama.cpp, Unsloth all mature). The Qwen3 language list includes Persian (verified above).
- **Qwen3-Coder-Next** ([card](https://huggingface.co/Qwen/Qwen3-Coder-Next)), ~2026-02-03, **Apache-2.0**. **80B total / 3B active** (qwen3_next arch), 262,144 context. **Non-thinking only.** Tool parser `qwen3_coder`. SWE-bench Verified 70.6, SWE-bench Pro 44.3, Terminal-Bench 2.0 36.2. 80B total weights do not fit a 64 GB M1 for training. It is beaten by Qwen3.6-35B-A3B (73.4 Verified) at the same active size.
- Bigger Qwen (mention only): Qwen3.5-122B-A10B, Qwen3.5-397B-A17B (Apache-2.0); Qwen3.8-Flash-Next 125B-A6B + 51B n-gram embeddings (Qwen Community License, preview of the "Qwen4" architecture, `qwen4_exp`; not in mlx-lm); Qwen3.8-2.4T-A95B.

#### Fine-tuning Qwen3.5/3.6/3.8 on an M1 Max with MLX — what I verified
- mlx-lm (latest v0.32.2, 2026-10-16) has `qwen3_5.py`, `qwen3_5_moe.py` and `qwen3_next.py`. In `qwen3_5.py` the Gated DeltaNet calls `use_kernel=not self.training`. **LoRA training runs on a slower op-by-op path, not the fast Metal kernel.** So training is supported by the code, but its speed on M1 is UNVERIFIED. mlx-lm loads the language model and drops the vision tower.
- [Unsloth's Qwen3.5 guide](https://unsloth.ai/docs/models/qwen3.5/fine-tune) (secondary, but a major trainer) says: "It is not recommended to do QLoRA (4-bit) training on the Qwen3.5 models, no matter MoE or dense", "due to higher than normal quantization differences". Its bf16 LoRA memory figures: 0.8B 3 GB, 2B 5 GB, 4B 10 GB, 9B 22 GB, 27B 56 GB, 35B-A3B 74 GB.
- **Consequence:** on 64 GB, 27B and 35B-A3B fit only as 4-bit QLoRA, which Unsloth warns against for this architecture. bf16 LoRA is comfortable up to 9B. Whether MLX's own 4-bit QLoRA degrades these models is UNVERIFIED; measure it. MoE router fine-tuning: Unsloth disables it by default "for stability".
- Unsloth: if a tuned model behaves worse in another runtime, the usual cause is "wrong chat template / EOS token at inference time". Train with the exact template you serve with.

---

### DeepSeek

#### Licence
- V3-0324 onward, V3.1, V3.2, V4-Pro, V4-Flash (all versions) and V4.1-Flash: **MIT**. I read the [V4.1-Flash LICENSE](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/LICENSE) and it is the plain MIT text. The cards say "This repository and the model weights are licensed under the MIT License." MIT allows fine-tuning, derivatives and commercial hosting, with no caps or naming rules. Keep the copyright notice.
- R1 distills: the card says the repo is MIT and "DeepSeek-R1 series (including Base and Chat) supports commercial use and distillation" ([R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B)). The distills also inherit their base models' licences: Qwen2.5 / Qwen3 Apache-2.0 and Llama 3.x community licence for the Llama distills. The base-licence part is from general knowledge; the 0528 card does not mention it, so treat it as UNVERIFIED for each distill.
- Using DeepSeek outputs to train DCLab's own models: MIT does not forbid it, and the R1 card explicitly allows distillation. The terms of the DeepSeek *API* (as opposed to the weights) were not checked: UNVERIFIED.

#### DeepSeek-V4-Flash family (284B total / 13B active) — too big to train locally; useful as a teacher or hosted judge
- Repos: [deepseek-ai/DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) (2026-04-22, "preview"), `-Base`, `-DSpark` (2026-06-27, with a speculative-decoding module), [DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731) ("the official release of DeepSeek-V4-Flash, superseding the preview version, with substantially enhanced agentic capabilities"), and `DeepSeek-V4-Flash-Vision-Exp` (2026-08-31). The tech report is arXiv 2606.19348.
- **284B total, 13B active. 1M context.** FP4 experts + FP8 rest, about 160 GB (secondary). Recommended sampling: temperature 1.0, top_p 1.0.
- **Thinking modes:** Non-Think, Think High, Think Max. Think Max adds a special prefix (`reasoning_effort="max"`) and needs ≥384K context.
- **No Jinja chat template.** DeepSeek ships a Python encoder (`encoding/encoding_dsv4.py`, [README](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/blob/main/encoding/README.md)). `thinking_mode="chat"` closes `</think>` at once; `"thinking"` opens `<think>`. **Earlier turns' reasoning: `drop_thinking=True` by default strips reasoning before the last user message. When tools are present, drop_thinking is automatically disabled and every turn keeps its reasoning.**
- **Tool calling: "DSML" format**, XML-like with special tokens: `<｜DSML｜tool_calls><｜DSML｜invoke name="…"><｜DSML｜parameter name="…" string="true|false">…</｜DSML｜parameter>…`. Results go in `<tool_result>` inside a user turn. Roles include `latest_reminder` and an internal `developer` role. The README warns the parser "does not attempt to correct or recover from malformed output".
- **Official numbers (preview V4-Flash, Max effort):** SWE-bench Verified **79.0**, LiveCodeBench 91.6, Terminal-Bench 2.0 56.9, MCPAtlas 69.0, SWE Multilingual 73.3. V4-Flash-0731: Terminal-Bench 2.1 82.7, NL2Repo 54.2, Toolathlon-Verified 70.3, DeepSWE 54.4 (DeepSeek Harness, max effort). No BFCL or τ-bench numbers are published.
- **Availability:** vLLM/SGLang per the card. The Ollama page [deepseek-v4-flash](https://ollama.com/library/deepseek-v4-flash) says it "was retired on August 27, 2026"; no local tags. mlx-lm has `deepseek_v32.py` and `deepseek_v41.py` but no `deepseek_v4.py` file (checked on GitHub main); V4 MLX support is UNVERIFIED.

#### DeepSeek-V4.1-Flash (Sep 2026) and V4.1-Flash-1017 (Oct 2026)
- [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash), **MIT**. Multimodal (image+text). "552B backbone parameters" plus 196B "Engram" memory (HF metadata 763B). It activates "8B parameters per token during prefill and 16B during decode". 1M context. Its causal encoder–decoder design uses about 1/4 of V4-Flash's KV cache.
- "Continuously controllable reasoning effort setting (integer 1–100)". Unlike V4, this repo ships a `chat_template.jinja` (15 KB, not read in detail) plus an `encoding` folder. Whether DSML is unchanged: UNVERIFIED.
- Numbers: Terminal-Bench 2.1 90.6 (max effort), DeepSWE v1.1 74.2, GPQA-D 90.9, HLE 36.8. No SWE-bench Verified, BFCL or τ-bench.
- Far too large for M1 training or local serving. Mention only.

#### DeepSeek-V4-Pro (mention only)
- [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) (Apr 2026) and V4-Pro-0813, MIT. Flagship; SWE-bench Verified 80.6 (Max). Data-centre only.

#### Small DeepSeek models (only the R1 distills)
- `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, `-7B`, `-14B`, `-32B` (Qwen2.5 bases), `-Llama-8B`, `-Llama-70B` (Jan 2025), and [DeepSeek-R1-0528-Qwen3-8B](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B) (May 2025). The last scores AIME24 86.0, LiveCodeBench (2408–2505) 60.5, GPQA-D 61.1. It uses DeepSeek-R1-0528's tokenizer config, not Qwen3's. Function calling is not confirmed for the distill.
- Ollama: `deepseek-r1:1.5b` / `7b` / `8b` (= R1-0528-Qwen3-8B) / `14b` / `32b` / `70b` ([ollama.com/library/deepseek-r1](https://ollama.com/library/deepseek-r1)). The page is inconsistent about the 8b base model.
- Verdict: reasoning-only, 2025-era, weak tool use, and older bases than Qwen3.5. **Not recommended as DCLab bases** in Oct 2026; Qwen3.5/3.6/3.8 dominate them.

---

### Fit to DCLab's three roles (my judgement, from the facts above)

| Role | Best pick here | Why | Main caveat |
|---|---|---|---|
| (1) Agentic tool-calling solver | **Qwen3.6-35B-A3B** (fast, 3B active) or **Qwen3.8-27B** (stronger, dense) | Apache-2.0, XML tool format trained for long agent loops, preserve_thinking for multi-turn | Training on 64 GB only fits as 4-bit QLoRA (Unsloth warns against QLoRA for 3.5-family); tool-parsing bugs in mlx_lm.server and Ollama |
| (2) Workflow / step-by-step + Python | **Qwen3.8-27B** | Highest coding numbers in reach (SWE-bench Pro 61.7, LCB 90.3), reasoning_effort control | No SWE-bench Verified published; slow MLX training path for DeltaNet layers; 27B bf16 LoRA needs ~56 GB |
| (3) Students 1–4B | **Qwen3.5-4B** / **Qwen3.5-2B** (non-thinking default) / 0.8B; fallback **Qwen3-4B-Instruct-2507** | Apache-2.0, 262K context, bf16 LoRA 5–10 GB, "task-specific fine-tuning" is the stated use | Qwen3.5 Persian coverage unverified; small sizes loop in thinking mode, so train non-thinking; 2507 is the safer tooling target |
| Teacher / judge (not trained) | DeepSeek-V4-Flash-0731 (MIT) | Strong agentic numbers and the most permissive licence; outputs may be used to train | 284B: hosted only; DSML format differs from Qwen's XML, so convert traces |

---

## Open-weight models, 2026-10-23: Gemma 4, Phi, Granite, OLMo, LFM, SmolLM

Research date: 2026-10-23. Sources were read on that date. "(secondary)" means the fact comes from a non-official source. UNVERIFIED means I could not confirm it from an official page in this pass.

DCLab's roles: (1) agentic solver, many tool turns; (2) workflow model, step-by-step ML reasoning plus Python; (3) students, 1–4B, quick classification-like decisions. Constraints: fine-tuning and commercial hosted use must be allowed; MLX LoRA on M1 Max; Persian matters.

**Persian across all six families:** no model card in these families lists Persian/Farsi as a supported language, and none reports a Persian evaluation. Gemma 4 is the best bet: it is "pre-trained on 140+ languages". But it is only "out-of-the-box" for 35+, and that list is not published. Persian must be measured by DCLab itself.

---

### Google Gemma 4

Source pages: model card https://ai.google.dev/gemma/docs/core/model_card_4 ; HF cards https://huggingface.co/google/gemma-4-E4B-it , https://huggingface.co/google/gemma-4-26B-A4B-it , https://huggingface.co/google/gemma-4-12B-it ; release log https://ai.google.dev/gemma/docs/releases?hl=en ; technical report arXiv 2607.02770 (v1 2026-07-02, v2 2026-07-24) https://arxiv.org/abs/2607.02770

#### Licence (whole family): Apache-2.0. This is a real change.
- Google's open-source blog (2026-04-02): "Gemma 4 models are the first in the Gemmaverse to be released under the OSI-approved Apache 2.0 license." https://opensource.googleblog.com/2026/03/gemma-4-expanding-the-gemmaverse-with-apache-20.html
- The HF metadata says `license: apache-2.0`. The licence link https://ai.google.dev/gemma/docs/gemma_4_license holds the standard Apache 2.0 text with no added terms.
- The **Gemma Prohibited Use Policy is not attached to Gemma 4 weights**. This reading is (secondary): https://vorplabs.com/models/gemma-license . The docs site still links "Prohibited use" and "Terms of use" pages in its navigation, but the licence text itself has no extra clauses.
- Result: fine-tuning, derivatives, commercial hosted use and redistribution are allowed. There is no MAU or revenue cap and no clause against training other models. You must keep the Apache NOTICE and attribution, and the licence gives no trademark rights, so do not name a product "Gemma".
- **Not covered:** FunctionGemma (270M, 2025-12-18), TranslateGemma and the other Gemma 3-based variants stay under the custom **Gemma Terms of Use**. The `google/functiongemma-270m-it` HF metadata says `license: gemma`.

#### Sizes and release dates
| Model | Params | Context | Modalities | Released |
|---|---|---|---|---|
| E2B | 2.3B effective (5.1B with embeddings), 35 layers | 128K | text, image, audio | 2026-03-31 per the release log; blog dated 2026-04-02 |
| E4B | 4.5B effective (8B with embeddings), 42 layers | 128K | text, image, audio | same |
| 12B "Unified" | 11.95B dense, 48 layers, no separate encoders | 256K | text, image, audio, video frames | 2026-06-03 |
| 26B A4B (MoE) | 25.2B total, 3.8B active; 128 experts, 8 active + 1 shared; 30 layers | 256K | text, image | 2026-03-31 |
| 31B dense | 30.7B, 60 layers | 256K | text, image | 2026-03-31 |

- MTP (multi-token prediction / "assistant" draft) checkpoints came out on 2026-04-16 (release log). On HF they are the `google/gemma-4-*-assistant` repos.
- QAT checkpoints exist: `-qat-q4_0-unquantized`, `-qat-q4_0-gguf`, `-qat-w4a16-ct`, `-qat-mobile`.
- Some third-party pages list "12B" or "27B A4B" in the March lineup. That is wrong: the 12B came in June, and the MoE is 26B A4B.

#### Tool / function calling
- Native support. The format is **not JSON**; it uses its own tokens (https://ai.google.dev/gemma/docs/capabilities/function-calling):
  - Tool declarations sit in the system turn, inside `<|tool>declaration:...<tool|>`.
  - The model calls a tool with `<|tool_call>call:NAME{key:<|"|>value<|"|>}<tool_call|>`.
  - Results go back as `<|tool_response>response:NAME{...}<tool_response|>`.
  - Turns use `<|turn>role ... <turn|>`.
- In `apply_chat_template(tools=...)`, JSON schemas or Python functions are converted for you. `processor.parse_response()` splits the output into thinking, content and tool calls.
- Official numbers: **τ2-bench (average of 3 domains)**: 31B 76.9, 26B A4B 68.2, 12B 69.0, E4B 42.2, E2B 24.5. Per domain for the 31B (tech report): airline 75.0, retail 86.4, telecom 69.3.
- **No official BFCL number** for any size.
- Structured JSON output: no dedicated mode is documented. Use constrained decoding in the server. UNVERIFIED beyond that.

#### Thinking mode
- **On:** put `<|think|>` at the start of the system prompt, or pass `enable_thinking=True` to the chat template. **Off:** remove the token.
- Output shape: `<|channel>thought\n…<channel|>` then the answer. With thinking off, every size except E2B and E4B still emits an empty thought block.
- **Earlier turns** (HF card, exact words): "In multi-turn conversations, the historical model output should only include the final response". Thoughts from earlier model turns are dropped, "with the exception of tool call turns where thinking content should be preserved."
  - So thinking is stripped at each new user turn, and kept between the tool calls inside one turn. That is interleaved thinking within the agent loop.
  - Caveat: the tech report's Table 11 multi-turn example shows a past thought block in the history. This conflicts with the card's rule. Follow the card and the chat template.
- For SFT data: train examples in exactly this shape. Strip thinking from past user-turn answers; keep it inside the tool-call chain.

#### Coding and reasoning (official, instruction-tuned, thinking on)
| | 31B | 26B A4B | 12B | E4B | E2B |
|---|---|---|---|---|---|
| LiveCodeBench v6 | 80.0 | 77.1 | 72.0 | 52.0 | 44.0 |
| Codeforces ELO | 2150 | 1718 | 1659 | 940 | 633 |
| MMLU-Pro | 85.2 | 82.6 | 77.2 | 69.4 | 60.0 |
| GPQA Diamond | 84.3 | 82.3 | 78.8 | 58.6 | 43.4 |
| MMMLU (multilingual) | 88.4 | 86.3 | 83.4 | 76.6 | 67.4 |
| BBEH | 74.4 | 64.8 | 53.0 | 33.1 | 21.9 |

#### Multilingual and Persian
- The card says: "Out-of-the-box support for 35+ languages, pre-trained on 140+ languages." The list of 35 is not published.
- The tech report names no Persian/Farsi. Its per-language tables cover en, ko, ja, de, fr, hi, es, it, pt-br, ru, ar and zh only.
- **Persian is not in any stated list and has no eval: UNVERIFIED.**
- Weak hint: the Gemma 3-based TranslateGemma appears to include Persian (fa_IR). This comes from a demo's language list (secondary): https://huggingface.co/spaces/webml-community/TranslateGemma-WebGPU/blob/main/src/constants.ts

#### Availability
- HF: `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it`, `google/gemma-4-12B-it`, `google/gemma-4-26B-A4B-it`, `google/gemma-4-31B-it`, plus the base and QAT repos.
- Ollama (https://ollama.com/library/gemma4): `gemma4:e2b`, `e4b`, `12b`, `26b`, `31b`, and `-mlx` builds of each (e.g. `gemma4:26b-mlx`).
- mlx-community: `mlx-community/gemma-4-e2b-it-4bit`, `gemma-4-e4b-it-4bit`, `gemma-4-12B-it-qat-4bit`, `gemma-4-26b-a4b-it-4bit`, `gemma-4-31b-it-4bit`, `gemma-4-26B-A4B-it-qat-OptiQ-4bit`.
  - These are tagged `mlx-vlm` (multimodal). Text-only LoRA in `mlx_lm` is reported to work for E4B by a community adapter (secondary: https://huggingface.co/Prashanth-24/gemma-4-e4b-python-lora ).
  - A fused E4B MLX checkpoint was reported broken, and an mlx-vlm LoRA scaling bug was fixed in PR #846 (secondary: https://discuss.huggingface.co/t/gemma4-e4b-adaptors-fuse-after-training-how/176429 ).
  - MLX LoRA on 26B A4B: no confirmation found. UNVERIFIED.

#### Known template and tool bugs (2026)
- Early GGUF and Ollama pulls carried a stale or broken chat template. It caused "unused token" errors and ignored multi-turn context. Fix: re-download, and update llama.cpp and Unsloth. Sources: Unsloth changelog https://unsloth.ai/docs/new/changelog (secondary) and https://aiproductivity.ai/news/gemma-4-gguf-chat-template-fix/ (secondary).
- Tool-call syntax is not JSON. Layers that assume OpenAI JSON (LiteLLM, streaming parsers) can corrupt arguments, numbers or booleans. Reported problems include `anyOf` with `null` rendering as empty types, and tool arguments passed as a JSON *string* getting double braces. Sources: https://discuss.huggingface.co/t/gemma-4-bug-fixes-and-research-request/176979 (secondary), https://gist.github.com/jscott3201/ad69c4ffbd79f18b11a0f6a94c94fadf (secondary).
- Fix PRs on Google's E2B and E4B repos cover null handling, keeping reasoning, and rejecting string-typed arguments. Merge status UNVERIFIED: https://huggingface.co/google/gemma-4-E4B-it/discussions/36 , https://huggingface.co/google/gemma-4-E2B-it/discussions/35
- vLLM: reasoning was dropped during tool calls when the "recommended" template was forced. Reported patched (vLLM PR #39027 / issue #39885, cited by a forum bot, so (secondary)): https://discuss.vllm.ai/t/what-is-the-correct-chat-template-when-serving-gemma4/2606
- Training bugs Unsloth reports as affecting all frameworks:
  - Loss spikes to 300–400 with gradient accumulation.
  - IndexError on 26B and 31B.
  - E2B/E4B output gibberish with `use_cache=False`.
  - Source: https://unsloth.ai/docs/models/gemma-4/train . Unsloth uses `gemma-4` and `gemma-4-thinking` templates.
- Ollama with LiteLLM: infinite tool loops reported (secondary, thin evidence).

---

### Microsoft Phi

**There is no Phi-5.** None of Microsoft's HF repos (https://huggingface.co/microsoft) carries that name, and no announcement was found. The newest Phi LLM is **Phi-4-reasoning-vision-15B** (HF repo created 2026-01-23; card says released 2026-03-04). Licence for all of the models below: **MIT**. MIT allows fine-tuning, derivatives and commercial hosting, with no caps.

#### Phi-4-mini-instruct (3.8B): https://huggingface.co/microsoft/Phi-4-mini-instruct
- 3.8B dense, 128K context, released Feb 2025. Vocabulary 200,064.
- Tool calling: a JSON tool list inside `<|tool|>…<|/tool|>` in the system message. Chat format is `<|system|>…<|end|><|user|>…<|end|><|assistant|>`.
  - vLLM parser: `--tool-call-parser phi4_mini_json`. Older vLLM read only `arguments`; newer versions also accept `parameters`. https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/phi4mini_tool_parser/
  - Ollama `phi4-mini` has the `tools` capability and needs Ollama ≥ 0.5.13.
- The card lists BFCL among its evals but **gives no score**. It warns the model "may hallucinate function names or URLs".
- No thinking mode. Multilingual MMLU 49.3, MGSM 63.9.
- Languages: 23 (ar, zh, cs, da, nl, en, fi, fr, de, he, hu, it, ja, ko, no, pl, pt, ru, es, sv, th, tr, uk). **Persian not listed.** The card says it is mainly English.
- Verdict: dated (Feb 2025). It is weaker than Granite 4.2-3B and Gemma 4 E4B on every axis DCLab cares about.

#### Phi-4-mini-reasoning / Phi-4-mini-flash-reasoning (3.8B)
- Both are math-only by design. Flash: "designed and tested for math reasoning only". It uses the SambaY hybrid (SSM + attention), 64K context, released June 2025.
- Flash needs `trust_remote_code`, flash-attn and mamba-ssm. No MLX or llama.cpp support is mentioned. No tool calling.
- Sources: https://huggingface.co/microsoft/Phi-4-mini-flash-reasoning , https://huggingface.co/microsoft/Phi-4-mini-reasoning
- Not suitable for DCLab.

#### Phi-4 (14B), Phi-4-reasoning / -plus (14B)
- Phi-4-reasoning: 32K context, released 2025-04-30.
  - It needs a fixed ChatML system prompt with `<think>{…}</think>{solution}`. Multi-turn rules are not stated. No tool calling is documented.
  - Primarily English: "not intended to support multilingual use".
  - LiveCodeBench (8/24–2/25) 53.8; HumanEval+ 92.9.
  - https://huggingface.co/microsoft/Phi-4-reasoning
- Phi-4 base context: 16K. UNVERIFIED in this pass.

#### Phi-4-reasoning-vision-15B: https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B
- 15B, built on Phi-4-reasoning plus SigLIP-2. Only 16,384 context.
- Hybrid thinking: `<think>…</think>` or `<nothink>`. Force a mode by appending `<think>` or `<nothink>` to the generation prompt; otherwise the model chooses.
- No tool calling or coding benchmarks. English only. MIT.
- Not suitable for the solver role.

#### Others
- `microsoft/Phi-tiny-MoE-instruct` and `Phi-mini-MoE-instruct` (2025-06, MIT, English) are research distillations, not agentic.
- **Phi overall:** MIT is the cleanest licence here, but the models are English-centred, short-context (14B), and have no official BFCL or τ-bench scores. Low priority for DCLab.

---

### IBM Granite

**Granite 5 LLMs do not exist yet.** Only `granite-speech-5.0-*` speech models carry the 5.0 name (Aug–Oct 2026). The current LLM line is **Granite 4.2** (2026-08-25). All the LLMs below are **Apache-2.0**.

#### Granite 4.2: 3B / 8B / 30B, dense, with reasoning. Strongest small-model candidate in this report.
Sources: https://huggingface.co/ibm-granite/granite-4.2-3b , https://huggingface.co/ibm-granite/granite-4.2-30b , blog https://research.ibm.com/blog/introducing-granite-4-2 , docs https://www.ibm.com/granite/docs/models/granite4-2

- Dense decoder-only transformer (GraniteForCausalLM). It is *not* hybrid Mamba; 4.1 and 4.2 went back to dense.
  - The 3B is really 3.66B (HF metadata). The 30B is about 29B.
  - Base model: `granite-4.1-3b-base`.
- Context: 128K native, extendable to 512K.
- **Thinking**, with three modes:
  - `enable_thinking=True` (the default) writes reasoning in `<think>…</think>`.
  - `enable_thinking=False` gives an empty `<think></think>`.
  - `enable_thinking=True, low_effort=True` appends `{reasoning effort: low}` to the user message.
- **Multi-turn:** `truncate_history_thinking=True` is the **default** and **strips thinking from earlier assistant turns**. Set it to `False` to keep it. Whether tool-call turns are exempt was not stated in what I read: UNVERIFIED.
- **Tool calling:** OpenAI function schema in; out comes `<tool_call>` with `<function=name>` and `<parameter=…>` (Qwen3-Coder-style XML).
  - vLLM: `--tool-call-parser qwen3_coder`. SGLang: `auto`.
  - IBM says it is "reasoning-augmented" (it thinks before each call).
- JSON output is mentioned on the Ollama description. Not verified on the card.
- **Official benchmarks:**

| | 3B | 8B | 30B |
|---|---|---|---|
| BFCL v4 | 52.41 | 52.39 | 61.39 |
| τ³-bench avg | 45.78 | 58.06 | 62.00 |
| LiveCodeBench v6 | 69.71 | 73.24 | 75.77 |
| MMLU-Pro | 67.84 | 74.04 | 77.60 |
| MMLU-ProX lite (multilingual) | 27.78 | 61.06 | 66.64 |
| IFBench (prompt) | 74.33 | 79.33 | 77.17 |

  The 3B also scores AIME25 78.33 and GPQA 54.80. SWE-bench Verified 57.0 for the 30B comes from IBM's own testing, as reported by a third party (secondary): https://www.eesel.ai/blog/granite-4-2
- Warning: the 3B's multilingual score (MMLU-ProX lite 27.78) is very low. Multilingual ability is weak at 3B.
- Languages: en, de, es, fr, ja, pt, ar, cs, it, ko, nl, zh. **Persian not listed.** The card says other languages may work but are untested.
- Availability:
  - HF: `ibm-granite/granite-4.2-{3b,8b,30b}`, plus `-GGUF`, `-fp8`, `-nvfp4` and `-mxfp4`.
  - **Official IBM MLX builds**: `ibm-granite/granite-4.2-3b-q4-mlx`, `-q8-mlx`, `-bf16-mlx` (same for 8b and 30b).
  - Ollama: `granite4.2:3b`, `:8b`, `:30b` (the listing shows no capability tags, but the description mentions tools and thinking) https://ollama.com/search?q=granite

#### Granite 4.1: 3B / 8B / 30B, dense, no reasoning (2026-04-29/30)
- Apache-2.0, 128K context for instruct models. Ollama `granite4.1`, tools tag.
- IBM says the 8B matches the earlier 32B MoE (secondary): https://rits.shanghai.nyu.edu/ai/ibm-releases-granite-4-1-dense-8b-matches-prior-32b-moe-flagship/
- **Granite Switch 4.1** (8B and 30B, 2026-10-14) is a modular model: Granite 4.1 plus a "decision" adapter, a guardian adapter and a speech encoder, switched on by control tokens. https://huggingface.co/ibm-granite/granite-switch-4.1-8b
  - Interesting as a design pattern for DCLab's "decision" heads.
  - Thinking and tools are not documented on its card.

#### Granite 4.0 (Oct 2025): hybrid Mamba-2 / transformer, and Nano
- **4.0-H-Small:** 32B total / 9B active hybrid MoE (4 attention + 36 Mamba2 layers), 128K context.
  - Tool calls are JSON in `<tool_call>` (Hermes-style); tools go in `<tools>`.
  - BFCL v3 64.69, HumanEval 88, IFEval 87.55. No thinking.
  - https://huggingface.co/ibm-granite/granite-4.0-h-small
- Also in 4.0: H-Tiny (7B / 1B active), H-Micro (3B), Micro (3B).
- **Nano (2025-10-28)**, from https://huggingface.co/ibm-granite/granite-4.0-h-1b :

| | 350M | H-350M | 1B (1.6B) | H-1B (1.5B) |
|---|---|---|---|---|
| Context | 32K | 32K | 128K | 128K |
| BFCL v3 | 39.32 | 43.32 | 54.82 | 50.21 |
| HumanEval | 39 | 38 | 74 | 73 |
| IFEval avg | 55.4 | 61.63 | 77.38 | 78.53 |

- Same 12 languages; Persian not listed. Apache-2.0.
- Ollama: `granite4:350m`, `granite4:1b`, `granite4:3b` and `-h` variants (https://ollama.com/library/granite4).
- Hybrid Mamba (`granitemoehybrid`): MLX and llama.cpp support it, but fine-tuning tooling for hybrids is less mature than for dense models. UNVERIFIED for mlx_lm LoRA.

---

### AI2 OLMo

**No Olmo 4 and no "Olmo 3.2" model.** Only a `olmo-3.2-tokenizer-think-dev` tokenizer repo exists (2026-02). There is no "Olmo agent" model. Ai2's agent line is **SERA** (coding agents, 2026-01-27), a separate family.

All Olmo models are **Apache-2.0**. The cards add "intended for research and educational use in accordance with Ai2's Responsible Use Guidelines". That is guidance, not a licence term; Apache itself allows commercial use. Ask a lawyer if this matters.

#### Olmo 3 / 3.1: 7B and 32B
Sources: https://huggingface.co/allenai/Olmo-3-7B-Instruct , https://huggingface.co/allenai/Olmo-3.1-32B-Think ; paper arXiv 2512.13961

- Released 2025-11-19 (Olmo 3). Olmo 3.1 32B Think and Instruct followed on 2025-12-10.
- Context: about 65K. UNVERIFIED for Olmo 3; the Olmo Hybrid card states 65,536. Recommended `max_tokens` 32768.
- ChatML template (`<|im_start|>`). The default system prompt is a "function-calling AI assistant" with `<functions></functions>`.
- The tool-call output format is not documented on the card. Ollama `olmo-3` and `olmo-3.1` are tagged Tools + Thinking (https://ollama.com/search?q=olmo).
- Olmo 3 7B Instruct: BFCL 49.8, HumanEval+ 77.2, LiveCodeBench v3 29.5, IFEval 85.6, MMLU 69.1.
- Olmo 3.1 32B Think: LiveCodeBench v3 83.3, AIME25 78.1, IFEval 93.8.
  - Separate Think and Instruct models (no switch). Think writes `<think>` first. Multi-turn thinking rules are not documented.
- **English only.**
- Strength: fully open data, including the Dolci SFT and RL sets with a tool-use SFT set (`allenai/Dolci-Instruct-SFT-Tool-Use-SA`). Useful as **training data** even if the model is not used.

#### Olmo Hybrid 7B (2026-01/03): https://huggingface.co/allenai/Olmo-Hybrid-7B
- 75% gated-DeltaNet layers and 25% attention, 65,536 context.
- Only SFT/DPO Instruct and Think-SFT checkpoints. Needs transformers ≥ 5.3. No llama.cpp or MLX mentioned.
- A research model; not recommended for DCLab.

#### SERA (Ai2 Open Coding Agents)
- 8B and 32B (secondary: SWE-bench Verified about 55% for 32B, 29.4% for 8B).
- Ai2 says a repo-specific agent can be trained for about $400 (secondary).
- Sources: https://www.geekwire.com/2026/ai2-cooks-up-open-source-coding-agents-with-tech-equivalent-of-a-hot-plate-and-a-frying-pan/ , https://siliconangle.com/2026/01/27/ai2-launches-family-open-source-ai-developer-agents-adapt-codebase/
- The base models, licence and template were not verified in this pass: UNVERIFIED. The *method* (soft-verified agent trajectories) is relevant to DCLab's self-play data.

---

### Liquid AI LFM2 / LFM2.5

**No LFM3 found.** The current line is **LFM2.5**. Sources: https://huggingface.co/LiquidAI ; blog for the 24B https://www.liquid.ai/blog/lfm2-5-24b-a2b (2026-10-13).

#### Licence: "LFM Open License v1.0" (`lfm1.0`). This is **not** Apache.
Source: https://huggingface.co/LiquidAI/LFM2.5-2.6B/blob/main/LICENSE

- It is Apache-like, with a **commercial cap**. Exact text: "'Threshold' shall mean annual revenue of 10 million United States dollars ($10,000,000) or more."
  - Commercial rights are "conditioned upon You or Your Legal Entity not exceeding the Threshold".
  - Above the threshold, commercial use "is not licensed under this Agreement".
- Derivatives and fine-tunes are allowed (Section 2). Modified files must say they were changed (4b). Keep notices (4c). No trademark rights (Section 7).
- No acceptable-use policy and no clause on training other models with outputs.
- For DCLab: fine while revenue is under $10M per year. A commercial licence from Liquid is needed above that. **This is a real business risk** for a hosted service that hopes to grow.

#### Models
- **LFM2.5-1.2B-Instruct**: https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct
  - 1.17B, 32K context. Released Jan 2026: 2026-01-05 or 01-20 (secondary; dates conflict).
  - Languages: en, ar, zh, fr, de, ja, ko, es.
  - GPQA 38.89, MMLU-Pro 44.35, IFEval 86.23, Multi-IF 60.98, **BFCLv3 49.12**.
  - The card says it is "not recommended for knowledge-intensive tasks and programming". Recommended for agentic tasks, extraction and RAG.
  - Variant: **LFM2.5-1.2B-Thinking** (Ollama `lfm2.5-thinking:1.2b`).
- **LFM2.5-2.6B** (Aug 2026): https://huggingface.co/LiquidAI/LFM2.5-2.6B
  - 2.69B; 22 short-convolution + 8 GQA blocks. 128K context.
  - A "pure reasoning model that always thinks". The template adds `<think>`. No off switch is documented.
  - BFCLv4 56.88, ToolSandbox 77.83, LiveCodeBench v6 59.41, IFBench 59.17, Multi-IF 80.07, IFStruct 85.49, AIME25 51.87.
  - 16 languages (en, ar, zh, fr, de, it, ja, ko, pt, es, vi, th, id, hi, ru, pl). **No Persian.**
- **LFM2.5-8B-A1B** (MoE, about 1B active, 2026). Details not fetched: UNVERIFIED.
- **LFM2.5-24B-A2B** (2026-10-13): https://huggingface.co/LiquidAI/LFM2.5-24B-A2B
  - 24.0B total / 2.45B active; 64 experts, top-4. 128K context. Always thinks.
  - BFCLv4 65.01, ToolSandbox 80.71, IFEval 96.75, Multi-IF 84.38, GPQA-D 74.42, SWE-bench Verified (OpenCode) 51.0.
  - Liquid says 184 tok/s on an M5 Max (vendor claim).
  - It is fast enough to be a solver candidate, but the licence cap applies.
- **LFM2.5-VL-3B** (Aug 2026): vision-language. Not needed.
- Older **LFM2** (2025): 350M, 700M, 1.2B, 2.6B, 8B-A1B, 24B-A2B. Superseded.

#### Tool calling and templates
- By default the model writes **Pythonic** calls between `<|tool_call_start|>` and `<|tool_call_end|>`. You can ask for JSON in the system prompt. Results go back in the `tool` role.
- vLLM: `--tool-call-parser lfm2` https://docs.vllm.ai/en/latest/api/vllm/tool_parsers/lfm2_tool_parser/
- llama.cpp needs `--jinja` and a recent build https://docs.liquid.ai/deployment/on-device/llama-cpp/function-calling
- Known bug (secondary): with the 8B-A1B GGUF via Ollama, the first tool call works. A later answer with bracket-heavy *code* then fails with a 500 "Failed to parse input", because the parser mistakes the code for a Pythonic tool call (cites llama.cpp issue #23838). https://ollama.com/pierreprudh/lfm2.5-8b-a1b
  - **This matters for DCLab, whose models write Python.**
  - A community M1 Max test also reports the 1.2B-JP hallucinated tool names (secondary): https://lilting.ch/en/articles/lfm25-12b-jp-local-test-plan
- Availability:
  - Official MLX repos from Liquid: `LiquidAI/LFM2.5-24B-A2B-MLX-4bit`, `-5bit`, `-6bit`, `-8bit`, `-bf16`; `LiquidAI/LFM2.5-2.6B-MLX-*`.
  - GGUF, ONNX and DSpark speculative-draft models.
  - Ollama `lfm2.5` (8b, 24b) and `lfm2.5-thinking:1.2b`.
  - Fine-tuning notebooks via Unsloth and TRL (SFT/DPO/GRPO LoRA).

---

### Hugging Face SmolLM

**No SmolLM4 found.** The latest is **SmolLM3-3B** (2025-07-08; base 2025-06-19). Source: https://huggingface.co/HuggingFaceTB/SmolLM3-3B

- 3B dense. 64K native context, 128K with YaRN ×2.
- Languages: en, fr, es, de, it, pt natively, plus some ar, zh and ru. **No Persian.**
- Thinking is on by default. Turn it off with `/no_think` in the system prompt or `enable_thinking=False`; the system flag wins over the argument. Multi-turn thinking rules are not documented.
- Tool calling: `xml_tools` (JSON inside `<tool_call>`) or `python_tools` (calls inside `<code>`).
- Benchmarks (instruct, no-think / think):
  - "BFCL" 92.3 / 88.8. This is far above every other small model here, so it is probably a BFCL subset; treat it as not comparable (my judgement).
  - LiveCodeBench v4 15.2 / 30.0. IFEval 76.7 / 71.2. Global MMLU 53.5 / 64.1.
- Licence: **Apache-2.0**. Fully open recipe and data.
- Availability: HF `HuggingFaceTB/SmolLM3-3B`, `-Base`, `-ONNX`. No `smollm3` in Ollama's library search (only `smollm2`). MLX conversions not checked: UNVERIFIED.
- Verdict: older (2025) and weaker at coding than Granite 4.2-3B or LFM2.5-2.6B. Useful mainly as a fully open baseline.

---

### Fit for DCLab (my judgement, from the facts above)

| Role | Best from these families | Why | Main caveat |
|---|---|---|---|
| Student 1–4B | **Granite 4.2-3B** (Apache-2.0) | Thinking with an off switch and low effort; tool calls; official MLX builds; LiveCodeBench 69.7, BFCL v4 52.4 | Weak multilingual (MMLU-ProX 27.8); no Persian |
| Student 1–4B | **Gemma 4 E2B / E4B** (Apache-2.0) | 140+ pretraining languages (best Persian chance); thinking switch; Ollama and MLX | Non-JSON tool format; template bugs; E4B is 8B with embeddings in memory; MLX LoRA only community-tested |
| Student <1.5B | **Granite 4.0 1B / H-1B** (Apache) or LFM2.5-1.2B (cap) | BFCL v3 about 50–55, IFEval about 77–86 | No thinking (Granite 4.0); LFM's licence cap |
| Workflow (reasoning + Python) | **Gemma 4 12B** or **26B A4B** (Apache) | LiveCodeBench 72 / 77, 256K context, 3.8B active for the MoE | Template and parser care needed; MoE LoRA on MLX unproven |
| Solver (multi-turn tools) | **Gemma 4 26B A4B** (τ2 68.2) or **Granite 4.2-30B** (BFCL v4 61.4, τ³ 62.0, dense) | Interleaved thinking kept across tool calls (Gemma); qwen3_coder-style calls (Granite) | Granite's 30B dense is slow on an M1; LFM2.5-24B-A2B is fast but has the $10M cap |

Skip for DCLab: all of Phi (English-centred, no tool numbers; MIT is nice), Olmo (English; use its open data instead), and SmolLM3 (older).

### Open checks before committing to a base model
1. Persian: run a small Persian probe on Gemma 4 E4B and 12B and on Granite 4.2-3B and 8B (understanding, plus answering DCLab questions in Persian). No official evidence exists for any of them.
2. Gemma 4 SFT format: the past-turn thinking rule (drop at user turns, keep in tool chains) must match DCLab's trajectory exporter exactly. Check the template in the exact repo revision used.
3. Granite 4.2: confirm whether `truncate_history_thinking` keeps thinking inside tool-call chains. Read the chat template.
4. MLX LoRA: run a smoke test of `mlx_lm.lora` on `ibm-granite/granite-4.2-3b-q4-mlx` and on Gemma 4 E4B (text-only), and check that a fused model gives the same output as base + adapter.
5. LFM licence: if LFM is ever used, record the $10M threshold in DCLab's licence register. Pythonic tool calls also clash with Python-writing outputs.

---

## Open-weight LLMs for DCLab — Nemotron, Meta (Llama/Muse), gpt-oss, Cohere, xAI

Checked on 2026-10-23. Primary sources: Hugging Face model cards, license files and chat templates read directly (`huggingface.co/<repo>/raw/main/...`), the Hugging Face API, ollama.com library pages, GitHub issue search. "(secondary)" = a news or third-party page. "UNVERIFIED" = not confirmed from a primary source.

DCLab needs: (1) an agentic **solver** (many tool turns), (2) a **workflow** model (step-by-step reasoning plus Python), (3) **students** of 1–4B. Fine-tune with MLX LoRA/QLoRA on an M1 Max (32–64 GB). Serve with Ollama, MLX or vLLM. The license must allow fine-tuning **and** selling a hosted service. Persian matters.

Important corrections to the brief:
- **"Nemotron 3.5 Lightning" exists.** It is `NVIDIA-Nemotron-3.5-Lightning-30B-A3B`, released 2026-08-11.
- **The newer NVIDIA releases changed license.** Nemotron 3.5 and Nemotron 3 Ultra use **OpenMDW-1.1**, a short permissive license. Nemotron 3 Nano and Super still use the **NVIDIA Nemotron Open Model License**.
- **Meta has no Llama 5 and no Llama 4.x on Hugging Face.** The `meta-llama` org's newest LLM is still Llama 4 (April 2025). Meta's new open line is **Muse**. `meta-models/Muse-Glimmer-30B` (Aug 2026) is **Apache-2.0**.
- **There is no gpt-oss v2.** The only newer OpenAI open weights are the `gpt-oss-safeguard` classifiers (Sep 2025) and `privacy-filter` (Apr 2026).
- **Cohere now has Apache-2.0 models.** These are the "North" line and Command A+ (05-2026). Command A, Command R7B, Aya Expanse and Tiny Aya stay **CC-BY-NC**.
- **Only Grok 1 and Grok 2 weights are on Hugging Face.** Grok 3 weights are not on HF as of today.

---

### NVIDIA Nemotron

#### Licenses (read directly)
- **OpenMDW-1.1** covers Nemotron 3.5 Lightning, 3.5 Bolt, 3.5 Super VL and 3 Ultra. License file: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16/blob/main/LICENSE
  - Grant: "permission is hereby granted, free of charge, to deal in the Model Materials without restriction".
  - On distribution: keep the agreement and the notices.
  - Patent/copyright suit: rights end if you sue claiming the Model Materials infringe.
  - Outputs: "does not impose any restrictions … with respect to … outputs".
  - There is **no** acceptable-use clause, **no** naming rule, **no** user cap and **no** clause against training other models.
  - Fine-tuning, derivatives and commercial hosting: **allowed**.
- **NVIDIA Nemotron Open Model License** (v. Dec 15 2025) covers Nemotron 3 Nano 30B, Nano 4B and Super 120B. https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-nemotron-open-model-license/
  - Grant: "Works are commercially usable". "You are free to create and distribute Derivative Works". The grant is perpetual and irrevocable.
  - Attribution: keep the notices. If a NOTICE file exists, include "Licensed by NVIDIA Corporation under the NVIDIA Nemotron Model License".
  - Rights end on patent/copyright litigation. No trademark rights.
  - The page as fetched has **no** guardrail clause, **no** AUP, **no** limit on using outputs to train other models, and **no** user cap.
  - Commercial hosting: **allowed**.
- **Older NVIDIA Open Model License** (v. Oct 24 2025) covers Nemotron Nano 9B v2 and 12B v2. https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
  - Commercial use and derivatives are allowed.
  - **But** Section 2.1 ends your rights automatically if you "bypass, disable, reduce the efficacy of, or circumvent any technical limitation, safety guardrail…" without a substantially similar guardrail.
  - Section 2.3 requires use "consistent with NVIDIA's Trustworthy AI terms".
  - **Risk for fine-tuning:** SFT can weaken built-in refusals. Avoid this license where a newer equivalent exists.

#### Nemotron 3.5 Lightning 30B-A3B — strongest Nemotron fit for the solver/workflow roles
- **Repos:**
  - `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (the customization reference)
  - `…-NVFP4` (for deployment), `…-Base-BF16`
  - Speculative drafters `…-NVFP4-DSpark` and `…-NVFP4-DFlash`
  - GGUF: `ggml-org/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-GGUF`
  - Card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16
- **Size:** 30B total / 3B active. Hybrid of Mamba-2, MoE and a few attention layers, plus MTP heads.
- **Context:** up to 1M tokens (256K validated on one H100).
- **Release:** 2026-08-11. Pre-training data cutoff Sep 2025; post-training cutoff May 2026.
- **License:** OpenMDW-1.1. The card says "ready for commercial use".
- **Tool calling:** XML-style Qwen3-Coder format, `<tool_call><function=NAME><parameter=P>value</parameter></function></tool_call>`.
  - vLLM flags: `--tool-call-parser qwen3_coder --reasoning-parser nemotron_v3 --enable-auto-tool-choice`.
  - For coding agents the card advises `chat_template_kwargs: {"force_nonempty_content": True}`.
  - Structured output was in SFT/RL ("structured outputs" environments).
  - BFCL is not reported. τ³-bench Banking 9.28 and PinchBench 85.37 are NVIDIA's numbers on its own harness.
- **Thinking:** `enable_thinking=True/False` in chat-template kwargs (default True). Recommended sampling: T=1.0, top_p=0.95.
- **Thinking in earlier turns (verified in `chat_template.jinja`):** `truncate_history_thinking` defaults to True.
  - Reasoning is **dropped** for assistant turns before the last user message.
  - Reasoning is **kept** for assistant turns after it. So it survives inside the current tool-call chain (interleaved thinking with tools).
  - Prior reasoning is passed as `reasoning_content` or inline `<think>…</think>`.
- **Coding (NVIDIA numbers):** SWE-bench Verified 51.56; SWE-bench Multilingual 39.33; Terminal-Bench 2.1 24.58; SciCode 32.60; MMLU-Pro 81.94; GPQA-D 75.44.
  - Its own table shows Qwen 3.6 35B-A3B ahead on most rows (SWE-V 70.12).
  - Lightning is **not** the strongest coder in its class.
- **Languages:** stated as English, Spanish, French, German, Italian, Japanese. Pre-training covered 19 other languages; post-training added Chinese.
  - **Persian is not listed.** No Persian eval.
  - The 15 non-English Common Crawl languages listed for the Nano 3 recipe do not include Persian (nano30 card).
- **Availability:**
  - Ollama `nemotron-3.5-lightning:30b` and `:30b-mlx` (tools, thinking badges). https://ollama.com/library/nemotron-3.5-lightning
  - mlx-community: `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-4bit`, `-8bit`, `-bf16`, `-mxfp4`, `-mxfp8`, `-OptiQ-4bit`.
- **Known issues:**
  - llama.cpp #27718: Q4_0 GGUF with zero KV-head metadata misclassified attention as SSM (closed). https://github.com/ggml-org/llama.cpp/issues/27718
  - llama.cpp #30894: Nemotron 3/3.5 GGUFs tagged "pixtral" with the wrong pre-tokenizer (closed). https://github.com/ggml-org/llama.cpp/issues/30894
  - Ollama #18352: context capped at 256K. https://github.com/ollama/ollama/issues/18352
  - Ollama #17735: `/save` fails for the `nemotron_h_moe` arch.
  - vLLM: several open hybrid-Mamba / speculative-decoding bugs, e.g. #55894 (Mamba+MTP corruption) and #52317 (prefix caching + DSpark crash).
  - mlx-lm: #980 prefix-cache reuse broken for hybrid/SSM models (closed); #1403 no batched generation for Nemotron-H (open).
- **MLX LoRA on `nemotron_h` (Mamba layers): UNVERIFIED.** I found no mlx-lm issue that confirms or denies it. NVIDIA documents LoRA via NeMo Megatron-Bridge / AutoModel and Unsloth (https://docs.nvidia.com/nemo/megatron-bridge/0.6.0/models/nemotron/nemotron3-nano.html). Test a 10-step LoRA run before committing.

#### Nemotron 3 Nano 30B-A3B (Dec 2025)
- **Repos:** `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`, `-FP8`, `-NVFP4`, `-Base-BF16`. Card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16
- **Size:** 31.6B total / 3.5B active (23 Mamba-2+MoE layers plus 6 attention layers; 128 experts, 6 active).
- **Context:** 1M (256K by default).
- **Release:** 2025-12-15.
- **License:** NVIDIA Nemotron Open Model License (commercial OK).
- **Tools and thinking:** Qwen3-Coder XML tool format. `enable_thinking` flag. Same `truncate_history_thinking` template logic (verified). It also has a `reasoning_budget` client pattern.
- **Benchmarks (NVIDIA):**
  - LiveCodeBench v6 68.3; SWE-Bench (OpenHands) 38.8
  - τ²-bench average 49.0 (Airline 48.0 / Retail 56.9 / Telecom 42.2)
  - BFCL v4 53.8; IFBench 71.5
  - MMLU-ProX 59.5, weak for multilingual
- **Languages:** EN, DE, ES, FR, IT, JA. **No Persian.**
- **Availability:** Ollama `nemotron-3-nano:30b`, `:30b-mlx`. mlx-community `NVIDIA-Nemotron-3-Nano-30B-A3B-4bit`, `-NVFP4`.
- **Known issue:** mlx-lm #682, tokenizer class error with the 4bit conversion (closed). https://github.com/ml-explore/mlx-lm/issues/682
- **Verdict:** superseded by 3.5 Lightning (same size, better scores, simpler license).

#### Nemotron 3 Nano 4B (Mar 2026) — student candidate
- **Repos:** `nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16`, `-FP8`, `-GGUF`. Card: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16
- **Size:** 3.97B dense hybrid Mamba-2 (4 attention layers), pruned from Nano 9B v2 with Nemotron Elastic.
- **Context:** 262K.
- **Release:** 2026-03-16.
- **License:** NVIDIA Nemotron Open Model License (commercial OK).
- **Tools:** Qwen3-Coder format; vLLM `--tool-call-parser qwen3_coder` plus a custom `nano_v3` reasoning parser plugin.
- **Thinking:** `enable_thinking` flag (default on). The same template keeps thinking only after the last user turn. Recommended T=0.6 for tool calling.
- **Benchmarks (NVIDIA):** BFCL v3 61.1; τ²-bench Retail 34.8 (reasoning off) / 39.8 (on); IFEval-prompt 82.8; LiveCodeBench 51.8 (reasoning on); RULER-128k 91.1.
- **Languages:** **English only.**
- **Availability:** Ollama `nemotron-3-nano:4b`, `:4b-mlx`. mlx-community `NVIDIA-Nemotron-3-Nano-4B-4bit`, `-BF16`, `-OptiQ-4bit`.
- **Verdict:** good English-only decision student with native tool calls. It is weak for Persian.

#### Nemotron 3.5 Bolt 74B-A9B 24GB (Oct 2026)
- **Repo:** `nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB` (plus GGUF and LiLiCorr drafters). https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
- **What it is:** a VLM compressed from 3.5 Super VL; 74B total / 9B active; 256K context.
- **Release:** HF 2026-10-15. License OpenMDW-1.1.
- **Quantization:** native 2-bit (experts at IQ2-XXS) to fit 24 GB. **That makes it a serving artifact, not a LoRA base.**
- **Languages:** EN, FR, ES, IT, DE, JA, HI, KO, pt-BR, ZH. **No Persian.**
- **Tools:** Qwen3-Coder format; "Supports JSON output format, reasoning … and tool calling".
- **Availability:** Ollama `nemotron-3.5-bolt:74b-a9b`, `:74b-a9b-iq2` (tools, thinking, vision).
- A smaller "Bolt E4B 4GB" was reported as due at the end of October (secondary: digitaltoday.co.kr, cloud.tencent.com). **UNVERIFIED**: no such repo on HF today.

#### Flagships (brief)
- **Nemotron 3 Super 120B-A12B** (2026-03-11): NVIDIA Nemotron Open Model License; 1M context; EN, FR, DE, IT, JA, ES, ZH. mlx-community 4bit/6bit exist. Too big to fine-tune on an M1 Max.
- **Nemotron 3.5 Super VL 120B-A12B** (Oct 2026, OpenMDW-1.1): SWE-bench Verified 71.10; τ²-bench airline 76.50.
- **Nemotron 3 Ultra 550B-A55B** (2026-06-04, OpenMDW-1.1): languages incl. HI, KO, pt-BR, ZH; SWE-V 70.7; PinchBench 90.0.
- **Nemotron Nano 9B v2 / 12B v2** (Aug 2025): older NVIDIA Open Model License with the guardrail termination clause. Superseded.

---

### Meta (Llama line and the new Muse line)

#### Status
- On HF, `meta-llama` has **no LLM newer than Llama 4** (2025-04-05). https://huggingface.co/meta-llama
- "Llama 5" release pages exist only on secondary/aggregator sites and contradict each other (techjacksolutions, swfte, orcarouter). **No official Llama 5 weights.**
- Meta's open line in 2026 is **Muse**, on `meta-models` (https://huggingface.co/meta-models):
  - `Muse-Glimmer-30B` (Aug 2026, Apache-2.0)
  - `Muse-Spark-Open-816B-A42B` (Sep 2026; gated, `license:other`; terms not read, **UNVERIFIED**; product listing calls it a "Muse Model License", secondary)

#### Muse Glimmer 30B (Aug 2026) — strongest candidate in these families for the solver
- **Repos:** `meta-models/Muse-Glimmer-30B` (BF16, not gated), `-GGUF`, `-ExecuTorch-PTE`, `-assistant` (DFlash drafter). Card: https://huggingface.co/meta-models/Muse-Glimmer-30B
- **Size:** ~29.6B **dense** (52 layers; 3 local + 1 global attention, window 2048; GQA 32/2), plus a ~1.8B ViT-G perception encoder (image input).
- **Context:** "131,072+". Knowledge cutoff 2026-01-04.
- **License:** **Apache-2.0** (the LICENSE file is plain Apache 2.0). The card says "intended for commercial and research use". Fine-tuning, derivatives and hosting are allowed.
  - **Caveat:** the repo also ships `USAGE_POLICY.md`, a Meta-style prohibited-use list (e.g. no unlicensed professional practice, no inferring sensitive personal data; not for under-18s).
  - The Apache text does not incorporate it. Whether it binds is legally unclear. **UNVERIFIED**, ask a lawyer.
  - There is **no** Llama-style 700M-MAU cap, naming rule or "Built with" rule.
- **Tool calling:** its own **ATEM** format (verified in `chat_template.jinja`). Tools are declared as JSON schemas in the system block. A call is a separate assistant message `to=<tool>` with `<atem:function_calls><atem:invoke name=…><atem:parameter name=…>`. Results come back as a `tool <name>` message.
  - The template **raises an error if `arguments` is a JSON string** rather than a dict. SFT data must store arguments as objects.
- **Thinking:** "Reasoning strength: low|medium|high|xhigh" in the system prompt (template kwarg `reasoning_strength`, default `high`). Reasoning is an assistant message `to=self`.
- **Thinking in earlier turns:** the template renders `reasoning_content` for **every** assistant turn that carries it. There is no truncation, so the caller decides. No rule on dropping old reasoning is stated. Keeping it is consistent with interleaved tool use. Drop-vs-keep guidance: **UNVERIFIED**.
- **Benchmarks (Meta, "High reasoning"):**
  - MCP-Atlas 75.5; τ³-Banking 23.5
  - SWE-bench Verified 76.0; SWE-bench Pro 51.2; Terminal-Bench 2.1 51.7; SciCode 43.6
  - IFBench 77.0; AA-LCR 80.0
  - Meta's table puts Qwen3.6-27B ahead on SWE-V (77.2) and Terminal-Bench (60.7).
- **Local speed (Meta):** M4 Max 23.7 tok/s, 37.8 with DFlash, at the 17 GB K-quant.
- **Languages:** "trained on data from more than 100 languages". No supported list. The card warns that performance "may degrade on languages outside the strongly supported set".
  - **Persian: UNVERIFIED.** No list, no eval found.
- **Availability:**
  - Ollama `muse-glimmer:30b`, `:30b-mlx` (tools, thinking, vision). https://ollama.com/library/muse-glimmer
  - mlx-community `Muse-Glimmer-30B-4bit`, `-5bit`, `-6bit`, `-8bit`, `-bf16`, `-mxfp4`, `-mxfp8`, `-nvfp4`, `-OptiQ-4bit`.
  - The model type is `muse_glimmer`. mlx-lm LoRA support for it is **UNVERIFIED** (no issues found either way).
- **Known issues (open unless noted):**
  - vLLM #53469: structured outputs never enforced with `--reasoning-parser muse_glimmer`. https://github.com/vllm-project/vllm/issues/53469
  - vLLM #52594: schema silently skipped with reasoning.
  - vLLM #54453: quadratic decoding under structured output.
  - vLLM #55395: tool parser leaks incomplete ATEM markers. https://github.com/vllm-project/vllm/issues/55395
  - Ollama #17656: `30b-mlx` manifest built from nvfp4-dflash layers. https://github.com/ollama/ollama/issues/17656
  - Ollama #18269: `30b-mlx` stuck in "Stopping".
  - Ollama #18808: GGUF broken (closed).
- **For M1 Max fine-tuning:** a dense 30B is far slower to train per token than a 3B-active MoE. QLoRA at 4-bit (~17–20 GB of weights) is realistic on 64 GB and tight on 32 GB. This is an estimate, not measured.

#### Llama 4 Scout / Maverick (Apr 2025)
- **Repos:** `meta-llama/Llama-4-Scout-17B-16E-Instruct` (109B total / 17B active, 10M context) and `meta-llama/Llama-4-Maverick-17B-128E-Instruct` (400B / 17B, 1M). Both gated. Card: https://github.com/meta-llama/llama-models/blob/main/models/llama4/MODEL_CARD.md
- **Languages:** 12 supported: Arabic, English, French, German, Hindi, Indonesian, Italian, Portuguese, Spanish, Tagalog, Thai, Vietnamese. **Persian is not supported.** Pre-training covered 200 languages. Fine-tuning for other languages is allowed under the license.
- **Benchmarks:** LiveCodeBench (10/24–02/25) Scout 32.8 / Maverick 43.4 (instruct). No native thinking mode.
- **Tools:** zero-shot "pythonic" function calls (`[func(a=1)]`), parallel calls supported (prompt_format.md in llama-models). vLLM parser `llama4_pythonic`.
  - vLLM #46863: Scout tool calls left in content. https://github.com/vllm-project/vllm/issues/46863
  - vLLM #61954: pythonic parsers break on a raw newline in a string.
  - vLLM #56840: streaming vs non-streaming mismatch.
- **License:** Llama 4 Community License (read from https://github.com/meta-llama/llama-models/blob/main/models/llama4/LICENSE):
  - Fine-tuning and derivatives are allowed.
  - You must "prominently display 'Built with Llama'".
  - A distributed model trained with Llama outputs or materials must "include 'Llama' at the beginning of any such AI model name".
  - A licensee with **>700M MAU** on the release date needs a separate license.
  - The AUP applies. The AUP (https://github.com/meta-llama/llama-models/blob/main/models/llama4/USE_POLICY.md) withholds Section 1(a) rights for **multimodal** Llama 4 models from EU-domiciled individuals and EU-based companies. End users of a product are exempt.
  - **Llama 4 is natively multimodal, so an EU-based DCLab could not license it.**
- **Availability:** Ollama `llama4:16x17b`, `:128x17b`. mlx-community `Llama-4-Scout-17B-16E-Instruct-4bit`, `-6bit`, `-8bit`.
- **Verdict for DCLab:** not recommended. It is too big to fine-tune on an M1 Max, has no thinking mode, no Persian, and the strictest license here.

#### Small Meta models
- No new small Llama exists. Llama 3.2 1B/3B (2024) is the last; Llama 3.2 license; EN, DE, FR, IT, PT, HI, ES, TH; **no Persian**.
- `facebook/MobileMoE-*` (Jun 2026) uses the **FAIR non-commercial research license**, so it is not usable.

---

### OpenAI gpt-oss

#### gpt-oss-20b and gpt-oss-120b (2025-08-05; no newer generation)
- **Repos:** `openai/gpt-oss-20b`, `openai/gpt-oss-120b`. https://huggingface.co/openai/gpt-oss-20b ; model card paper https://arxiv.org/abs/2508.10925
  - Also `openai/gpt-oss-safeguard-20b/120b` (Sep 2025): policy classifiers, not general models.
- **Size:** 20.9B total / 3.61B active (32 experts, top-4, 24 layers) and 116.8B / 5.13B. MoE weights ship in MXFP4 (20b checkpoint 12.8 GiB; "run within 16GB").
- **Context:** 131,072 (YaRN). Knowledge cutoff June 2024. Text only.
- **License:** **Apache-2.0** plus the gpt-oss usage policy. The HF `USAGE_POLICY` file reads only: "By using OpenAI gpt-oss-20b, you agree to comply with all applicable law."
  - Fine-tuning, derivatives and commercial hosting are allowed.
  - No naming, MAU or anti-distillation clauses.
- **Format:** the **harmony** format is mandatory ("should only be used with the harmony format as it will not work correctly otherwise").
  - Channels: `analysis` (CoT), `commentary` (function calls: "Calls to these tools must go to the commentary channel: 'functions'"), `final`.
  - Role hierarchy: System > Developer > User > Assistant > Tool.
  - Guide: https://developers.openai.com/cookbook/articles/openai-harmony
- **Thinking:** cannot be switched off. Reasoning effort is set in the system message, "Reasoning: low|medium|high" (default medium). `reasoning_effort` in OpenAI-compatible servers maps onto it.
- **Thinking in earlier turns (verified in the harmony guide):**
  - "you should drop any previous CoT content on subsequent sampling" once a turn ends with a `final` message.
  - "The exception for this is tool/function calling": within a tool chain "we should pass the previous chain-of-thought back in as input".
  - The model card adds "in multi-turn conversations the reasoning traces from past assistant turns should be removed".
  - Interleaved CoT ↔ tool calls is supported. **The SFT data must follow the same rule.**
- **Structured output:** "support Structured Outputs" (response formats in the developer message). The prompt alone "doesn't guarantee the full adherence"; use grammar-constrained decoding.
- **Benchmarks (paper Table 3, high effort, 20b / 120b):**
  - SWE-bench Verified 60.7 / 62.4
  - τ-bench Retail 54.8 / 67.8; τ-bench Airline 38.0 / 49.2
  - Aider Polyglot 34.2 / 44.4
  - Codeforces Elo with tools 2516 / 2622
  - AIME25 no-tools 91.7 / 92.5; GPQA-D 71.5 / 80.1
  - BFCL not reported.
- **Languages:** text-only data "with a focus on STEM, coding, and general knowledge". MMMLU covers 14 languages (AR, BN, ZH, FR, DE, HI, ID, IT, JA, KO, PT, ES, SW, YO); 20b high scores 75.7 on average.
  - **Persian is not evaluated.** Tokenizer is o200k_harmony (201k vocab). Persian quality is **UNVERIFIED**.
- **Availability:**
  - Ollama `gpt-oss:20b`, `:120b` (tools, thinking). https://ollama.com/library/gpt-oss
  - mlx-community `gpt-oss-20b-MXFP4-Q8` (168k downloads), `-MXFP4-Q4`, `-mxfp4-bf16`, `-OptiQ-4bit`.
- **Known issues:**
  - vLLM #58825 (open): stray `commentary to=assistant` returned as a tool call named "assistant…". https://github.com/vllm-project/vllm/issues/58825
  - vLLM #26083: tool calls failed in stream mode (closed).
  - vLLM, several open Responses-API / harmony bugs: #45570 constrained final output treated as an MCP call; #53626; #49493.
  - Ollama #12203: thinking returned in tool calls (open). https://github.com/ollama/ollama/issues/12203
  - Ollama #12187: not completing tool calls (open).
  - Ollama #12884: tool-call JSON parse errors (open).
  - Ollama #12159: no parallel tool calls (closed).
  - llama.cpp #19520: arguments escaped twice (closed). https://github.com/ggml-org/llama.cpp/issues/19520
  - mlx-lm #613: tool call issue with gpt-oss-20b-MXFP4-Q4 (closed). https://github.com/ml-explore/mlx-lm/issues/613
  - mlx-lm #361: NaN during fine-tuning (closed). https://github.com/ml-explore/mlx-lm/issues/361
  - mlx-lm #1994 (open): `sanitize()` breaks on BF16 checkpoints in the current transformers layout. https://github.com/ml-explore/mlx-lm/issues/1994
  - **MLX LoRA on gpt-oss works in principle, but has had numeric and conversion problems.** Pin versions and test.
- **Verdict:** gpt-oss-20b is the most battle-tested agentic MoE here. Its license is clean and it fits an M1 Max easily. Its weak spots are Persian (unknown), the dated knowledge (2024), and harmony-format fragility in servers.

---

### Cohere

#### License picture (HF API tags, 2026-10-23)
- **CC-BY-NC-4.0 plus Cohere Labs Acceptable Use Policy (non-commercial):**
  - Command A 03-2025 (111B), Command A Reasoning 08-2025, Command A Vision 07-2025, Command A Translate 08-2025
  - Command R7B 12-2024 (8.0B), Command R7B Arabic
  - Aya Expanse 8B/32B, Aya Vision 8B/32B, Aya 23
  - **Tiny Aya** (3.35B; global/earth/fire/water, thinkers), North-Small-Translate-1.0
  - The Tiny Aya card says: "This model is governed by a CC-BY-NC License (Non-Commercial) with an acceptable use addendum … If you are interested in commercial use, please contact Cohere's Sales team." https://huggingface.co/CohereLabs/tiny-aya-global
  - **You may fine-tune these models, but you cannot sell a hosted service on them without a separate commercial deal.**
- **Apache-2.0 (commercial OK):** `CohereLabs/North-Mini-Code-1.0` (30B-A3B), `CohereLabs/command-a-plus-05-2026-*` (218B-A25B), `CohereLabs/North-Micro-Vision-Instruct` (2.4B VLM).

#### North Mini Code 1.0 (Jun 2026) — Apache-2.0 coding agent
- **Repos:** `CohereLabs/North-Mini-Code-1.0`, `-fp8`, `-w4a16`, `-eagle` (drafter). Card: https://huggingface.co/CohereLabs/North-Mini-Code-1.0 ; blog https://huggingface.co/blog/CohereLabs/introducing-north-mini-code (2026-06-09)
- **Size:** 30.5B total / 3B active (128 experts, 8 active). Attention is 3 sliding-window + 1 global, the global layers with no positional embedding.
- **Context:** 256K, with 64K max output.
- **License:** Apache 2.0 ("open weights research release", but the license is Apache).
- **Tools:** JSON-schema tools via the HF chat template. vLLM main needs `cohere_melody>=0.9.0`; flags `--tool-call-parser cohere_command4 --reasoning-parser cohere_command4`.
- **Thinking:** "supports interleaved thinking and works best when turned on". Cohere is **"strongly encouraged to pass on all the model-generated thinking contents to future agentic steps, and chat turns"**. That is the opposite of harmony's drop rule. Store it as `reasoning` on the assistant message.
- **Benchmarks (Cohere):** SWE-bench Verified 61.0 pass@1 (mini-SWE-agent); 80.2 pass@10 (SFT model); Terminal-Bench v2 55.1 pass@10; Artificial Analysis Coding Index 33.4. BFCL/τ-bench not reported.
- **Languages:** not stated. The repo has no language tags. **Persian: UNVERIFIED.** Cohere's other models do include Persian, but that cannot be assumed here.
- **Availability:** Ollama `north-mini-code-1.0` (bf16, q4_K_M, q8_0, mlx-bf16, mlx-mxfp8, mlx-nvfp4; tools, thinking). mlx-community `North-Mini-Code-1.0-4bit`, `-5bit`, `-6bit`, `-bf16` (model type `cohere2_moe`).
- **Known issues:** none found in mlx-lm or vLLM searches. The model is young, which makes this lack of reports weak evidence.

#### Command A+ 05-2026 (flagship, brief)
- `CohereLabs/command-a-plus-05-2026-bf16`, `-fp8`, `-w4a4`. 218B total / 25B active; vision input; 128K context. **Apache-2.0.**
- Language tags list 48 languages **including `fa` (Persian)**.
- Too big for M1 Max fine-tuning (W4A4 still needs 2×H100). Possible teacher or labeller for Persian data. https://huggingface.co/CohereLabs/command-a-plus-05-2026-bf16

#### Command R7B 12-2024 and Aya Expanse 8B/32B (CC-BY-NC — not sellable)
- **Command R7B:** ~8B, 128K context. 23 languages **including Persian**. Native tool use and RAG via the HF chat template. CC-BY-NC plus AUP. Ollama `command-r7b:7b`. https://huggingface.co/CohereLabs/c4ai-command-r7b-12-2024
- **Aya Expanse:** 8B (8K context) and 32B. 23 languages **including Persian**; tool use not stated; CC-BY-NC plus AUP. Ollama `aya-expanse:8b/32b`. https://huggingface.co/CohereLabs/aya-expanse-8b
- **Tiny Aya (Feb 2026):** 3.35B, 8K context, 70+ languages incl. Persian. CC-BY-NC. No tool calling stated.
- **Aya is the best-documented Persian coverage in these families.** The license blocks DCLab's commercial plan.

#### North Micro Vision Instruct (Aug 2026)
- 2.4B (2B LM), Apache-2.0, 128K LM context.
- Languages: EN, DE, FR, ES, IT, PT, HI, JA, KO, ZH, AR "and more". Persian not listed.
- **"Tool calling and agentic workflows are not supported."** A possible student base only for pure classification. https://huggingface.co/CohereLabs/North-Micro-Vision-Instruct

---

### xAI Grok

- HF `xai-org` has only **grok-1** (Mar 2024, Apache-2.0, 314B) and **grok-2** (2025-08-22, ~500 GB, TP=8, SGLang). https://huggingface.co/xai-org
- News called the August 2025 release "Grok 2.5" (secondary: opensourceforu, peoplematters). The HF repo is named Grok 2.
- **Grok 3:** Musk confirmed plans (Feb 2026, secondary: dataconomy). **No Grok 3 weights on HF as of 2026-10-23.**
- **Grok 2 license, "xAI Community License Agreement" (Last Updated Nov 4 2025), read from https://huggingface.co/xai-org/grok-2/blob/main/LICENSE:**
  - Commercial use is allowed "solely if you and your affiliates abide by all of the guardrails provided in xAI's Acceptable Use Policy". The license is **revocable**.
  - "You may not use the Materials, derivatives, or outputs (including generated data) to train, create, or improve any foundational, large language, or general-purpose AI models, except for modifications or fine-tuning of xAI Materials". **This bans distilling it into DCLab's own models.**
  - You must display "Powered by xAI" plus a notice. Texas law applies.
- No tool-calling, thinking or Persian documentation on the card.
- **Verdict:** not usable. Too large for DCLab hardware, the anti-training clause blocks the students plan, and the commercial grant is revocable.

---

### Comparison for DCLab's roles

| Model | Total/active | License (commercial hosted?) | Tools | Thinking control / history rule | Persian | M1 Max LoRA |
|---|---|---|---|---|---|---|
| Nemotron 3.5 Lightning | 30B / 3B MoE-Mamba | OpenMDW-1.1 (yes, minimal terms) | Qwen3-Coder XML | `enable_thinking`; template drops thinking before last user turn, keeps within tool chain | not listed | MLX conversions exist; LoRA on Mamba UNVERIFIED |
| Muse Glimmer 30B | 29.6B dense (+vision) | Apache-2.0 (+ usage-policy file, status unclear) | ATEM XML | "Reasoning strength" low…xhigh; template keeps all supplied reasoning | ">100 languages", UNVERIFIED | dense 30B: slow; QLoRA fits 64 GB (estimate) |
| gpt-oss-20b | 20.9B / 3.6B MoE | Apache-2.0 + light usage policy (yes) | harmony `commentary` | effort low/med/high (no off); drop CoT after final, keep in tool chains | not evaluated | works in mlx-lm; past NaN/convert bugs |
| North Mini Code 1.0 | 30.5B / 3B MoE | Apache-2.0 (yes) | JSON schema, `cohere_command4` parser | interleaved; Cohere says keep all thinking | not stated | mlx-community conversions exist; LoRA untested |
| Nemotron 3 Nano 4B | 4B dense hybrid | Nemotron Open Model License (yes) | Qwen3-Coder XML; BFCL v3 61.1 | `enable_thinking`; same truncation | English only | MLX conversions exist; LoRA on Mamba UNVERIFIED |
| Command R7B / Aya / Tiny Aya | 3–32B | **CC-BY-NC (no)** | R7B yes | — | **yes** | n/a for a commercial service |
| Llama 4 Scout | 109B / 17B | Llama 4 CL (yes with Built-with-Llama, "Llama…" name, 700M MAU, EU block on multimodal) | pythonic | none | no | too big |
| Grok 2 | not stated on card (~500 GB checkpoint, TP=8) | xAI CL (revocable; bans training other models) | — | — | — | too big |

---

## Open-weight models: Mistral, Zhipu/Z.ai GLM, MiniMax, Moonshot Kimi (state on 2026-10-23)

Research for DCLab's three fine-tuning roles: (1) agentic **solver** (many tool turns), (2) **workflow** model (step-by-step ML reasoning + Python), (3) **students** 1–4B. Training target: Apple M1 Max 32–64 GB with MLX LoRA/QLoRA. Serving: Ollama/MLX or vLLM. Sold as a hosted service, so license must allow fine-tuning + commercial hosted use.

How this was checked: Hugging Face org listings and model cards/LICENSE files read directly through the HF Hub (today), vendor blogs/docs, Ollama library pages, vLLM recipes. "(secondary)" = not the vendor. "UNVERIFIED" = not confirmed from a primary source today. HF "Created" dates are repo creation dates and can differ from public release dates.

Names asked about that do not exist / are different from expected:
- "Mistral Small 3.x/4": exists, but **Mistral Small 4 is a 119B MoE (6.5B active)**, not a ~24B model.
- "Devstral 2" = **Devstral-2-123B** (not Apache; Modified MIT with revenue cap). "Devstral Small 2" = 24B, Apache-2.0.
- No "GLM-5 small" text model in 1–15B. Smallest new GLM text model is GLM-4.7-Flash (30B-A3B). GLM-5.3-Flash is 320B-A18B.
- No small MiniMax LLM (all M-series are 229B+). "MiniMax M3.1-Flash" is ~428B-A23B.
- Newer than asked: GLM-5.1/5.2/5.3, GLM-5.3-Flash; MiniMax M2.7, M3, M3.1-Flash; Kimi K2.6, K2.7-Code, K3; Mistral Medium 3.5, Mistral Large 4 (weights not yet out).

---

### Mistral AI

Org listing: https://huggingface.co/mistralai · News index (2026 posts): https://mistral.ai/news

General notes for all Mistral models:
- Native tool calling with Mistral's own format; vLLM flags `--tool-call-parser mistral --enable-auto-tool-choice`; reasoning models use `--reasoning-parser mistral`. Mistral says **mistral-common is the source of truth** for tokenization/templates (Magistral card: "Please make sure to use mistral-common as the source of truth"). https://huggingface.co/mistralai/Magistral-Small-2509
- Reasoning uses special tokens `[THINK]` … `[/THINK]` that "**must** be encoded as such" (special tokens, not text). Same card.
- Mistral API terms: only found a clause (secondary, ConductAtlas) banning use of **image** outputs to train competing image products; no general text-output training ban found. Weights under Apache-2.0 carry no output restriction. https://conductatlas.com/platform/mistral-ai/mistral-ai-terms-of-service/prohibited-use-of-image-outputs-for-competing-products/ (secondary)
- Template/tooling gotchas (verified): Ministral 3 needs transformers v5 (5.0.0rc0/main) + mistral-common ≥1.8.6 (`MistralCommonBackend`); vLLM needs `--tokenizer_mode mistral --config_format mistral --load_format mistral`. https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512 · Devstral Small 2 in llama.cpp needs changes from PR 17945. https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512 · Mistral Medium 3.5's original Transformers config "caused long-context performance degradation"; GGUFs made before the fix commit are affected. https://huggingface.co/mistralai/Mistral-Medium-3.5-128B
- UNVERIFIED (from memory, not re-checked): mistral-common requires 9-character alphanumeric tool-call IDs; mismatches between HF `chat_template` and mistral-common have caused tool-call bugs in third-party runtimes.

#### Ministral 3 — 3B / 8B / 14B (Base, Instruct, Reasoning) — **best student candidate**
- Params: 3B, 8B, 14B dense (14B = 13.5B LM + 0.4B vision encoder). Vision input. Context **256k**. Released Dec 2025 ("2512"; HF repos dated 31 Oct 2025; paper arXiv 2601.08584). https://huggingface.co/mistralai/Ministral-3-14B-Instruct-2512
- Variants per size: Base (BF16), Instruct (FP8 + BF16 copies), Reasoning (BF16), plus GGUF repos.
- Tool calling / JSON: card claims "native function calling and JSON outputting"; recommends temperature <0.1 in production and few tools. No BFCL/τ-bench published.
- Thinking: Reasoning variants are separate checkpoints (not a switch). Multi-turn rule: card "highly recommend[s] keeping the reasoning traces in context". https://huggingface.co/mistralai/Ministral-3-8B-Reasoning-2512
- Official scores (card): 14B Reasoning AIME25 0.850, GPQA-D 0.712, LiveCodeBench 0.646; 3B Reasoning AIME25 0.721, LCB 0.548; 3B Instruct Arena Hard 0.305 (Qwen3-VL-4B 0.438). No SWE-bench for Ministral.
- Languages: 11 listed (en, fr, es, de, it, pt, nl, zh, ja, ko, ar). **Persian not listed; not evaluated.**
- License: **Apache-2.0** (card: "allowing usage and modification for both commercial and non-commercial purposes"). Fine-tuning, derivatives, commercial hosting allowed. No caps.
- Availability: `mistralai/Ministral-3-{3B,8B,14B}-{Base,Instruct,Reasoning}-2512`; Ollama `ministral-3:3b|8b|14b` (3.0/6.0/9.1 GB, Apache-2.0) https://ollama.com/library/ministral-3 ; MLX: `mlx-community/Ministral-3-{3B,8B,14B}-Instruct-2512(-4bit|-5bit|-6bit)`, `mlx-community/Ministral-3-3B-Reasoning-2512-{4,5,6,8}bit`. https://huggingface.co/mlx-community/Ministral-3-3B-Instruct-2512-4bit
- Caveat: architecture is `mistral3` (vision wrapper). MLX-LM LoRA on these conversions: UNVERIFIED (conversions load as text-generation in mlx; training path not tested here). Mistral also fine-tuned the 3B base into Shieldstral-1.0-3B (safety) and Voxtral, a sign the 3B base fine-tunes well (Apache-2.0). https://huggingface.co/mistralai/Shieldstral-1.0-3B

#### Devstral Small 2 (24B) — solver/coding candidate
- 24B dense, based on Mistral-Small-3.1-24B-Base; vision; context 256k; Dec 2025 (2512). https://huggingface.co/mistralai/Devstral-Small-2-24B-Instruct-2512
- Coding (card): **SWE-bench Verified 68.0%**, SWE-bench Multilingual 55.7%, Terminal Bench 2 22.5%. Ollama page shows 65.8% / 51.6% / 32.0% — numbers disagree; trust the HF card but note it. https://ollama.com/library/devstral-small-2
- Tool calling: built for agentic coding scaffolds (OpenHands, SWE-agent, Cline, Claude Code, Mistral Vibe). Parser `mistral`. No thinking mode (instruct model).
- Fits "a single RTX 4090 or a Mac with 32GB RAM" (card) for inference. QLoRA of 24B dense on 32 GB is tight; 64 GB comfortable (my estimate, not tested).
- Languages: no natural-language list on card; Persian not stated.
- License: **Apache-2.0**.
- Availability: Ollama `devstral-small-2:24b` (15 GB, 384K ctx, needs Ollama ≥0.13.3); MLX `mlx-community/Devstral-Small-2-24B-Instruct-2512-4bit`. Known fix: llama.cpp PR 17945.

#### Magistral Small 1.2 (Magistral-Small-2509, 24B) — workflow (reasoning) candidate
- 24B dense, built on Mistral Small 3.2 with SFT from Magistral Medium traces + RL; vision; context 128k ("might degrade past 40k"); Sept 2025. https://huggingface.co/mistralai/Magistral-Small-2509
- Reasoning: `[THINK]…[/THINK]` special tokens; reasoning prompt goes in the system prompt. Multi-turn: Mistral lets you choose "between keeping reasoning traces during multi-turn interactions or keeping only the final assistant response."
- Scores (card): AIME24 86.14%, AIME25 77.34%, GPQA-D 70.07%, LiveCodeBench v5 70.88%. No SWE-bench/BFCL.
- Languages: card lists 24 incl. **"Farsi"** explicitly. No Persian evaluation.
- License: **Apache-2.0**. Fine-tuning guides: Axolotl, Unsloth.
- Availability: GGUF `mistralai/Magistral-Small-2509-GGUF`; Ollama `magistral` UNVERIFIED (not re-checked today); no `mlx-community` Magistral-Small-2509 conversion found today (conversion via `mlx_lm.convert` likely possible, UNVERIFIED).

#### Mistral Small 3.2 (24B Instruct 2506)
- 24B dense, vision, 128k context (from 3.1 card; UNVERIFIED today). Improved function-calling template vs 3.1 (card links the mistral-common change). https://huggingface.co/mistralai/Mistral-Small-3.2-24B-Instruct-2506
- Languages tags include **fa**. License Apache-2.0. Ollama `mistral-small3.2:24b`. Superseded in practice by Magistral 1.2 (reasoning) and Devstral Small 2 (code), and by Small 4 at the top.

#### Mistral Small 4 (119B-A6.5B, 2603) — too big to train on the Mac; good teacher/server
- MoE 128 experts / 4 active, **119B total, 6.5B active** (blog: 6B active, 8B incl. embeddings), 256k context, vision, released **16 Mar 2026**. Unifies Instruct + Magistral + Devstral. https://huggingface.co/mistralai/Mistral-Small-4-119B-2603 · https://mistral.ai/news/mistral-small-4
- Thinking: per-request `reasoning_effort="none"|"high"`. Multi-turn rule not stated (blog/card).
- Tool calling + JSON "native"; no BFCL/SWE-bench published (only AA LCR 0.72, LiveCodeBench vs GPT-OSS-120B).
- Languages tag list includes **fa**. License **Apache-2.0**.
- Availability: HF + NVFP4 + EAGLE head; Unsloth GGUF; LM Studio. No `mlx-community` conversion found today. Training on M1 Max: 4-bit weights alone ≈ 60 GB → not practical (estimate).

#### Mistral Medium 3.5 (128B dense) and Devstral 2 (123B) — license warning
- Medium 3.5: dense 128B, 256k, reasoning_effort switch, SWE-bench Verified **77.6%**, τ³-Telecom 91.4% (card). Languages incl. **fa**. Released with Vibe remote agents, 22 May 2026. https://huggingface.co/mistralai/Mistral-Medium-3.5-128B
- Devstral 2: 123B, SWE-bench Verified 72.2%. https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512
- License (both): **Modified MIT** — quote: "You are not authorized to exercise any rights under this license if the global consolidated monthly revenue of your company (or that of your employer) exceeds $20 million … This restriction … applies to the Model and any derivatives". Fine for DCLab today; becomes a blocker if DCLab grows past $20M/month. https://huggingface.co/mistralai/Mistral-Medium-3.5-128B/blob/main/LICENSE
- Not trainable on the Mac. Usable as teacher (outputs carry no restriction in the license text).

#### Mistral Large 3 (675B) and Mistral Large 4 (~1T)
- Large 3: `mistralai/Mistral-Large-3-675B-Instruct-2512`, Apache-2.0 (HF tag). Active params UNVERIFIED (not re-read today).
- Large 4: announced **6 Oct 2026**, 1T total / 52B active (blog; Ollama lists 1.05T/52B), multimodal hybrid instruct+reasoning, "natively fluent in 160+ languages". Blog: "We will release the weights by the end of the month." **Weights and license not yet published (UNVERIFIED).** https://mistral.ai/news/mistral-large-4 · Ollama `mistral-large-4` (cloud).

Other 2026 Mistral open releases not relevant to DCLab's roles: Timestral (time-series), Leanstral 1.5 119B-A6B (Lean proofs, Apache-2.0), Shieldstral-1.0-3B (safety classifier, Apache-2.0), Voxtral speech models.

---

### Zhipu / Z.ai GLM

Org listing: https://huggingface.co/zai-org

General notes for GLM:
- Tool calling native; vLLM/SGLang: `--tool-call-parser glm47 --reasoning-parser glm45` for GLM-4.7(-Flash) (verified, GLM-4.7-Flash card). Tool-call text format is XML-like (`<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`) — UNVERIFIED from memory.
- **Thinking rules (verified, Z.ai docs):** switch with `"thinking": {"type": "enabled"|"disabled"}`; **interleaved thinking** on by default ("thinking blocks should be explicitly preserved and returned together with the tool results"); **preserved thinking** (since GLM-4.7) keeps earlier turns' `reasoning_content`, controlled by `clear_thinking` (false = preserved); when on, "you must return the complete, unmodified reasoning_content", blocks must match the original order. **Turn-level thinking** (GLM-4.7) lets each turn enable/disable. GLM-5.3 and GLM-5.3-Flash "use forced thinking and cannot be disabled". https://docs.z.ai/guides/capabilities/thinking-mode
- GLM-4.7-Flash card: for multi-turn agentic tasks (τ²-Bench, Terminal Bench 2) "turn on Preserved Thinking mode".
- Languages: every GLM card lists only **en, zh**. Persian not stated, not evaluated.
- **API output rule (important for distillation):** Z.ai API terms forbid "utilization of platform models or exported data to develop, train, or enhance algorithms or models that compete with us" and bar using model-generated content to train external models (Additional Terms for API Services §1.f.xii; Terms §III.4(f)). So distil from **self-hosted MIT weights**, not from the Z.ai API. https://docs.z.ai/legal-agreement/terms-of-use

#### GLM-4.7-Flash (30B-A3B MoE) — **best solver candidate in these families**
- 30B total / ~3B active, MoE (`glm4_moe_lite`). HF repo 19 Jan 2026. Context 198K (Ollama page). https://huggingface.co/zai-org/GLM-4.7-Flash
- Official scores (card, thinking): **SWE-bench Verified 59.2**, **τ²-Bench 79.5**, BrowseComp 42.8, AIME25 91.6, GPQA 75.2, LCB v6 64.0 (vs Qwen3-30B-A3B-Thinking-2507: SWE-V 22.0, τ² 49.0). No BFCL.
- Thinking: on by default; preserved thinking recommended for multi-turn agent work (see above).
- License: **MIT** (no caps, no attribution clause beyond MIT notice). Fine-tune, derive, host commercially: allowed.
- Availability: Ollama `glm-4.7-flash` (q4_K_M 19 GB, q8_0 32 GB, bf16 60 GB; needed Ollama 0.14.3) https://ollama.com/library/glm-4.7-flash ; MLX `mlx-community/GLM-4.7-Flash-{4bit,5bit,6bit,8bit,bf16,4bit-mxfp4,8bit-mxfp8}` https://huggingface.co/mlx-community/GLM-4.7-Flash-4bit
- Known bugs: llama.cpp had the expert gating/scoring function set to softmax instead of sigmoid → looping/poor output; fixed ~21 Jan 2026, re-download GGUFs (secondary: Unsloth). Unsloth advised **not** using its GGUF with Ollama "due to potential chat template compatibility issues"; an HF discussion gives a custom Ollama Modelfile/parser for working tool calls (secondary). https://unsloth.ai/docs/models/glm-4.7-flash · https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF/discussions/23 . vLLM/SGLang needed main branches at release.
- M1 Max: 4-bit ≈ 17–19 GB → LoRA on 64 GB realistic; 32 GB tight (estimate). MoE LoRA in MLX: UNVERIFIED for this arch.

#### GLM-4.7 (355B-A32B) / GLM-4.6 / GLM-4.5 / GLM-4.5-Air — teachers
- GLM-4.7 (22 Dec 2025): SWE-bench Verified **73.8**, SWE-bench Multilingual 66.7, Terminal Bench 2.0 41.0, τ²-Bench 87.4. Introduced Preserved + Turn-level Thinking. MIT. https://huggingface.co/zai-org/GLM-4.7 . Total/active 355B/32B from GLM-4.5 family — UNVERIFIED today.
- GLM-4.5-Air: ~106B total / 12B active (HF shows 110.5B incl. MTP), MIT. https://huggingface.co/zai-org/GLM-4.5-Air — too big to train on the Mac.
- GLM-4.6 (Sep 2025): SWE-V 68.0 (per GLM-4.7 table). MIT.

#### GLM-5 / 5.1 / 5.2 (≈744–753B MoE) — teachers, MIT
- GLM-5: 11 Feb 2026, ~744B total / 40B active (secondary: HF community blog, NYU Shanghai RITS). MIT on HF. https://huggingface.co/zai-org/GLM-5 · https://huggingface.co/blog/mlabonne/glm-5 (secondary)
- GLM-5.1 (Apr 2026) and **GLM-5.2 (16 Jun 2026): MIT** (HF tags). https://huggingface.co/zai-org/GLM-5.2
- Thinking on by default, preserved thinking via `clear_thinking` (Z.ai docs).

#### GLM-5.3 (Aug 2026) — new custom license
- Same base as GLM-5.2, post-training only. reasoning_effort `low|high|max` (default max), forced thinking; chat template `clear_thinking` **defaults to false** (= keeps earlier thinking); for chat pass `clear_thinking=true`. https://huggingface.co/zai-org/GLM-5.3
- License **"GLM-5.3 License"**: MIT-style grant incl. "run, deploy, fine-tune … create derivative works", plus: a "Model as a Service" operator whose group revenue exceeds **US$10 billion over 12 months** must pass Z.ai's security review before commercial use. Irrelevant for DCLab now. No output/training-other-models ban in the license. https://huggingface.co/zai-org/GLM-5.3/blob/main/LICENSE

#### GLM-5.3-Flash (320B-A18B, MIT)
- First natively multimodal GLM-5; hybrid sparse+linear attention; 1M context (Ollama); forced thinking; MIT. Ollama only `glm-5.3-flash:cloud`. https://huggingface.co/zai-org/GLM-5.3-Flash · https://ollama.com/library/glm-5.3-flash . Teacher candidate (MIT, cheap active params), not trainable on the Mac.

#### Small GLMs (older): GLM-4-9B-0414, GLM-Z1-9B-0414 (reasoning), GLM-4-32B-0414, GLM-4.6V-Flash (10B vision)
- 9.4B / 32.6B dense, MIT, en/zh, April 2025. https://huggingface.co/zai-org/GLM-4-9B-0414 · https://huggingface.co/zai-org/GLM-Z1-9B-0414
- GLM-4.6V-Flash: 10.3B vision model, MIT (Dec 2025). https://huggingface.co/zai-org/GLM-4.6V-Flash
- Ollama `glm4:9b`. Older than Ministral 3 / Qwen-class rivals; no 1–4B GLM text model exists. Tool-calling quality UNVERIFIED.

---

### MiniMax

Org listing: https://huggingface.co/MiniMaxAI . **No small model** (all LLMs ≥229B). Relevance for DCLab = teacher only, and the licenses got **stricter** over time.

General notes:
- vLLM recipe: `--tool-call-parser minimax_m2 --reasoning-parser minimax_m2 --enable-auto-tool-choice`. https://docs.vllm.ai/projects/recipes/en/latest/MiniMax/MiniMax-M2.html . (A `minimax_m2_append_think` variant that keeps think text in content exists — UNVERIFIED.)
- **Thinking history rule (verified for M2):** interleaved thinking model; card: "you must ensure that the historical content is passed back in its original format" and "Do not remove the `<think>...</think>` part, otherwise, the model's performance will be negatively affected." https://huggingface.co/MiniMaxAI/MiniMax-M2 . M2.5 card did not restate it (assume same family behaviour; UNVERIFIED).
- Languages: no natural-language list on cards; Persian not stated.
- API terms on output use: not found in MiniMax's current ToS (searched). Older MiniMax-Text-01 license reportedly banned using outputs "to improve any other large language model" (secondary, HF discussion). https://huggingface.co/MiniMaxAI/MiniMax-Text-01/discussions/2

#### MiniMax-M1 (456B-A45.9B hybrid attention, Jun 2025)
- License **Apache-2.0** (HF tags). `MiniMaxAI/MiniMax-M1-40k`, `-80k` (+ `-hf`). Older; params from memory UNVERIFIED.

#### MiniMax-M2 (230B-A10B, Oct 2025) — permissive
- 230B total / 10B active; SWE-bench Verified **69.4**, τ²-Bench 77.2 (card). https://huggingface.co/MiniMaxAI/MiniMax-M2
- License: MIT + one modification: if used in commercial products with ">100 million monthly active users, or more than 30 million US dollars … in annual recurring revenue, you shall prominently display "MiniMax M2" on the user interface". https://github.com/MiniMax-AI/MiniMax-M2/blob/main/LICENSE

#### MiniMax-M2.1 (229B, Dec 2025)
- SWE-bench Verified **74.0**, SWE-bench Multilingual 72.5 (card). License "Modified-MIT" per card link https://github.com/MiniMax-AI/MiniMax-M2.1/blob/main/LICENSE (text not read today; assume M2-style attribution — UNVERIFIED). https://huggingface.co/MiniMaxAI/MiniMax-M2.1

#### MiniMax-M2.5 (229B, 12–13 Feb 2026) — custom "MiniMax Model License"
- SWE-bench Verified 80.2% (announcement; eval widget shows 75.8*), Multi-SWE-Bench 51.3%. https://huggingface.co/MiniMaxAI/MiniMax-M2.5
- Repo file is **LICENSE-MODEL "MINIMAX MODEL LICENSE"** (the card's tag/link says "Modified-MIT" — inconsistent; the file governs). Grants use, derivatives, modification; distribution of the model, derivatives "or a product or service that uses any of them, including another AI model" must ship the Agreement + "Notice" attribution; **Prohibited Uses** include military use and "**Conduct fully automated decision-making that adversely affects an individual's legal rights**" — relevant to DCLab's business-decision models (credit/fraud/churn) if MiniMax-derived. Singapore law/SIAC. https://huggingface.co/MiniMaxAI/MiniMax-M2.5/blob/main/LICENSE-MODEL

#### MiniMax-M2.7 (229B, Apr 2026) — **non-commercial**
- LICENSE: "NON-COMMERCIAL LICENSE … commercial use requires prior written authorization." Commercial use explicitly includes hosted APIs and "deployment … of the Software or its derivatives that have been subjected to post-training, fine-tuning". **Not usable for DCLab's service without a deal.** https://huggingface.co/MiniMaxAI/MiniMax-M2.7/blob/main/LICENSE

#### MiniMax-M3 (Jun 2026) and MiniMax-M3.1-Flash (17 Oct 2026, ~428B-A23B, 1M ctx, text+image+video)
- M3.1-Flash: DeepSWE v1.1 73.7%, BankerToolBench 76.2, OSWorld V2 61.7 (card). https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash
- License **"MINIMAX COMMUNITY LICENSE"** (both): grant is "for non-commercial purposes"; for Commercial Use you must "prominently display 'Built with MiniMax M3.1-Flash'" and either send a one-time notice to api@minimax.io or, above **US$20M yearly revenue**, get prior written authorization. Commercial Use includes hosted APIs and fine-tuned deployments. Prohibited-use appendix (military, etc.). https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/main/LICENSE · https://huggingface.co/MiniMaxAI/MiniMax-M3/blob/main/LICENSE
- Ollama: `minimax-m3.1-flash`, `minimax-m3`, `minimax-m2.7` are **cloud** tags; community `frob/minimax-m2.5` 230b local. https://ollama.com/search?q=minimax

---

### Moonshot Kimi

Org listing: https://huggingface.co/moonshotai

General notes:
- vLLM: `--tool-call-parser kimi_k2 --reasoning-parser kimi_k2 --enable-auto-tool-choice --trust-remote-code`. https://docs.vllm.ai/projects/recipes/en/latest/moonshotai/Kimi-K2-Think.html
- Thinking history: K2-Thinking is trained to interleave reasoning with tool calls and sustain "200–300 sequential tool calls" (card). Kimi platform guide: "Include the entire reasoning content from the context (the reasoning_content field) in your input." https://platform.kimi.ai/docs/guide/use-kimi-k2-thinking-model . K2.6: `preserve_thinking` **off by default**, enable with `thinking: {'type':'enabled','keep':'all'}`; instant mode via `thinking: {'type':'disabled'}` (API) or `chat_template_kwargs {"thinking": False}` (vLLM). K2.7-Code: thinking and preserve_thinking **forced on, can't be disabled**. K3: always thinking (`reasoning_effort` low/high/max), trained in "preserved thinking history mode": pass back full assistant message incl. `reasoning_content` and `tool_calls`.
- Languages: not stated on any card. Persian not stated.
- **API terms:** Kimi platform §3.2(5) forbids "developing, serving, or creating applications, products, Services, or models that have potential competitive possibilities" without authorization. So distil from **self-hosted weights** (license has no output clause), not from the Kimi API. https://platform.kimi.ai/docs/agreement/modeluse . Databricks' reseller terms explicitly ban using Kimi outputs to "build, train, distill" competing models (secondary, reseller).

#### Kimi-Linear-48B-A3B (Base, Instruct; Oct 2025) — the only "small-active" Kimi LLM
- 48B total / 3B active, **1M context**, hybrid linear attention, trained on 5.7T tokens; MMLU-Pro 51.0, RULER@128k 84.3 (card). **Tool calling and thinking not stated.** License **MIT**. https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct
- MLX: `mlx-community/Kimi-Linear-48B-A3B-Instruct-{3,4,5,6,8}bit`, `-mlx-bf16`. Ollama: none official.
- Verdict: a research architecture, weak general scores, no stated tool use → **not recommended** for any DCLab role.

#### Kimi-VL-A3B (Instruct, Thinking, Thinking-2506) and Moonlight-16B-A3B
- 16B-A3B vision-language (MIT, Apr–Jun 2025). https://huggingface.co/moonshotai/Kimi-VL-A3B-Thinking-2506 . Not a fit for text-agent work.

#### Kimi K2 / K2-0905 / K2-Thinking / K2.5 / K2.6 / K2.7-Code (1T-A32B) — teachers
- All 1T total / 32B active, 256K context. K2-Thinking: native INT4 QAT, SWE-bench Verified **71.3**. K2.5 (late Jan 2026, multimodal): SWE-V **76.8**. K2.6 (Apr 2026): SWE-V **80.2**. K2.7-Code (Jun 2026, built on K2.6): Kimi Code Bench v2 62.0, MCP Atlas 76.0 (no SWE-V). https://huggingface.co/moonshotai/Kimi-K2-Thinking · https://huggingface.co/moonshotai/Kimi-K2.5 · https://huggingface.co/moonshotai/Kimi-K2.6 · https://huggingface.co/moonshotai/Kimi-K2.7-Code
- License: **Modified MIT** — "if the Software (or any derivative works thereof) is used for any of your commercial products or services that have more than 100 million monthly active users, or more than 20 million US dollars … in monthly revenue, you shall prominently display "Kimi K2" [or "Kimi K2.7 Code"] on the user interface". Verified for K2-Thinking and K2.7-Code LICENSE files; K2.5/K2.6 labelled Modified MIT on cards (text not re-read). No ban on training other models with outputs.
- Ollama: `kimi-k2.6`, `kimi-k2.7-code` are cloud tags.

#### Kimi K3 (Jun–Jul 2026; 2.8T-A104B, 1M ctx) — new license
- "Kimi K3 License": MIT-style grant incl. fine-tune/derivatives, plus (a) a **"Model as a Service"** business with group revenue > **US$20M over 12 months** must sign a separate agreement before commercial use; (b) display "Kimi K3" above 100M MAU or $20M monthly revenue. MaaS excludes "end-user products with model capabilities solely embedded within specific features or harnesses" — DCLab's product likely falls in that exclusion, but a raw "fine-tune-as-a-service" offer would not (my reading, not legal advice). https://huggingface.co/moonshotai/Kimi-K3/blob/main/LICENSE · https://huggingface.co/moonshotai/Kimi-K3

---

### Cross-family takeaways for DCLab

License quick table (weights; verified from LICENSE files unless noted):

| Model | License | Fine-tune + hosted commercial | Catches |
|---|---|---|---|
| Ministral 3 3B/8B/14B, Devstral Small 2, Magistral Small 1.2, Mistral Small 3.2 / 4 | Apache-2.0 | Yes | none |
| Mistral Medium 3.5, Devstral 2 123B | Modified MIT | Yes until revenue > $20M/month | rights end above cap, incl. derivatives |
| GLM-4.x, GLM-4.7-Flash, GLM-5/5.1/5.2, GLM-5.3-Flash | MIT | Yes | Z.ai **API** ToS bans training on API outputs |
| GLM-5.3 | GLM-5.3 License | Yes | MaaS + > $10B revenue → security review |
| Kimi K2.x, K2-Thinking, K2.7-Code | Modified MIT | Yes | name display > 100M MAU / $20M monthly revenue; Kimi **API** ToS bars competing models |
| Kimi K3 | Kimi K3 License | Yes (with conditions) | MaaS > $20M/12 mo → separate agreement |
| Kimi-Linear, Kimi-VL, GLM-4-9B | MIT | Yes | — |
| MiniMax M1 | Apache-2.0 | Yes | old |
| MiniMax M2 | MIT + attribution | Yes | display "MiniMax M2" > 100M MAU / $30M ARR |
| MiniMax M2.5 | MiniMax Model License | Yes | notice files; prohibited "fully automated decision-making that adversely affects an individual's legal rights" |
| MiniMax M2.7 | Non-commercial | **No** without written authorization | — |
| MiniMax M3 / M3.1-Flash | MiniMax Community | Only with "Built with MiniMax …" display + notice; > $20M/yr needs authorization | fine-tuned deployments count as commercial |

Persian: only Mistral states it — Magistral Small 1.2 lists "Farsi"; Mistral Small 3.2 / 4 / Medium 3.5 carry the `fa` tag. Ministral 3 (11 languages), all GLM (en/zh), Kimi and MiniMax (none listed) do not. I found **no official Persian evaluation** for any of these models; the Open Persian LLM Leaderboard / MELAC benchmark is where to test (secondary). https://preview.aclanthology.org/setup/2025.ijcnlp-long.105

Thinking-format implication for SFT: every strong teacher here (GLM-4.7+/5.x preserved thinking, MiniMax M2 keep `<think>`, Kimi K2.7/K3 forced preserve) keeps earlier turns' reasoning in context. Student templates differ: Ministral 3 Reasoning also recommends keeping traces; Magistral lets you choose. Decide one rule for DCLab's trajectories and render teacher traces into the student's own special tokens (`[THINK]` for Mistral, `reasoning_content`/`<think>` for GLM) — mixing formats is a common source of silent template bugs.

---

## Open-weight LLMs: Hermes, Apriel, EXAONE, others, and Persian (checked 2026-10-23)

Scope: candidates for DCLab's (1) agentic solver, (2) workflow/code model, (3) 1–4B students.
Needs: fine-tuning + commercial hosted use allowed; ≤ ~35B total (MoE with small active preferred); MLX on M1 Max; Ollama/MLX/vLLM serving.

Method: model cards, LICENSE files, configs and chat templates were read directly from Hugging Face (hf.co) on 2026-10-23.
"(secondary)" marks third-party pages. "UNVERIFIED" marks anything not confirmed in a primary source.
Benchmarks below are the vendors' own numbers unless said otherwise. They are not independent.

---

### Nous Research Hermes

Hermes models are fine-tunes. **The base model's license is the one that matters.**
No Hermes 5 exists on the NousResearch Hugging Face org as of 2026-10-23 (newest repos: research MoE checkpoint Mar 2026, NousCoder-14B Dec 2025, Hermes 4.3 Nov 2025). Source: https://hf.co/NousResearch (org listing, sorted by date).
Nous's 2026 product is the Hermes Agent framework (MIT), not a new model (secondary: https://www.heyuan110.com/posts/ai/2026-04-14-hermes-agent-guide).

#### Hermes 4 14B (base: Qwen3-14B)
- Params: 14B dense. Context: `max_position_embeddings` 40,960 in config (Qwen3 base). Released Aug 2025.
  Sources: https://hf.co/NousResearch/Hermes-4-14B , config.json in same repo.
- Tool calling: Hermes format. Tools in `<tools>…</tools>` in the system prompt, or passed as `tools=` and rendered by the template. Calls come out as `<tool_call>{json}</tool_call>` (added tokens, easy to stream-parse); results go back as `<tool_response>`. vLLM parser `hermes`, SGLang `qwen25`. ChatML template.
- Structured output: card says trained for JSON-schema adherence and repairing malformed JSON.
- Thinking: hybrid. Turn on with chat-template flag `thinking=True` or a fixed "deep thinking" system prompt. Output in `<think>…</think>`. Earlier turns' thinking is **dropped by default**; template flag `keep_cots=True` keeps it. Card says tool calls are produced "after its reasoning" within one assistant turn, and after `<tool_response>` the model "continues reasoning inside <think>" → interleaved thinking with tools is supported.
- Coding: card shows only an image of benchmarks; points to the tech report (https://arxiv.org/abs/2508.18255). No code numbers verified here.
- Multilingual / Persian: card lists `en` only. Persian ability comes from Qwen3 base. On the Open Persian LLM Leaderboard, Qwen3-14B averages 57.8 (see Persian section). Hermes 4 itself is not on that board. Note: Hermes-3-Llama-3.1-8B scores 50.4 vs Llama-3.1-8B-Instruct 50.3 → the Hermes recipe neither helped nor hurt Persian there.
- License: `apache-2.0` (card metadata). Inherits Qwen3 Apache-2.0. Fine-tuning and commercial hosting allowed; keep notices.
- Availability: HF `NousResearch/Hermes-4-14B`, `-FP8`; MLX `mlx-community/Hermes-4-14B-4bit`; GGUF by LM Studio (per card). Ollama: no official tag; only community uploads (https://ollama.com/search?q=hermes4).
- Known issues: no Hermes-4-specific template bug found. Tool-call failures in practice usually come from the wrong parser (must be `hermes`) (secondary, search summary). Card explicitly brands "low refusal / neutral alignment" (RefusalBench) → you must add your own safety layer for a hosted product.

#### Hermes 4.3 36B (base: ByteDance Seed-OSS-36B-Base)
- Params: 36B dense, 64 layers. Context: config `max_position_embeddings` 524,288 (Seed-OSS native 512K). Released Nov 2025. First Hermes trained decentralised over Psyche.
  Sources: https://hf.co/NousResearch/Hermes-4.3-36B , config.json.
- Tool calling: same Hermes `<tool_call>` format. **But prompt format is Llama-3-Chat style** (`<|start_header_id|>…<|eot_id|>`) per the card, not Seed's native `<seed:…>` tokens. NousResearch also published `ByteDance-Seed-OSS-36B-Alternate-Tokenizer`.
- Thinking: same as Hermes 4 (`thinking=True`, `<think>`, `keep_cots`).
- Benchmarks (card): AIME25 69.3, GPQA-D 65.5, MMLU-Pro 80.7, IFEval 77.9. No code benchmark listed.
- License: `apache-2.0` (card). Inherits Seed-OSS Apache-2.0.
- Availability: `NousResearch/Hermes-4.3-36B`, `NousResearch/Hermes-4.3-36B-GGUF` (4/5/6/8-bit). No mlx-community conversion found. Ollama: community only.
- Fit: 36B dense is slow to train on an M1 Max (QLoRA possible on 64 GB, but slow). Over the 35B target.

#### Hermes 4 70B / 405B (base: Llama 3.1)
- License: Llama 3.1 Community License (base). Commercial use allowed with Llama terms (700M MAU cap, "Built with Llama" naming, AUP). Too large for DCLab. Skip.

#### NousCoder-14B (base: Qwen3-14B)
- RL for competitive programming. LiveCodeBench v6 Pass@1 67.87 vs 60.79 for Qwen3-14B. `apache-2.0`. Dec 2025.
  Source: https://hf.co/NousResearch/NousCoder-14B
- Not a chat/agent model; competitive-programming only. Possible reference for code RL, not a base.

---

### ServiceNow Apriel

All Apriel releases are `license: mit` on their cards. Source org: https://hf.co/ServiceNow-AI

#### Apriel-1.6-15B-Thinker (current; Nov 2025)
- Params: ~15B dense (Mistral-style text decoder, 48 layers) + Pixtral vision; `llava` arch. Context: config 262,400 positions; vLLM example uses 131,072; Ollama page shows 256K. AA-LCR 50 needs DCA (36 without).
  Sources: https://hf.co/ServiceNow-AI/Apriel-1.6-15b-Thinker , config.json , https://ollama.com/ServiceNow-AI/Apriel-1.6-15b-Thinker
- Tool calling: native. Special tokens `<tool_calls>`, `</tool_calls>`, `[BEGIN FINAL RESPONSE]`, `<|end|>`. Pass `tools=` to the template. vLLM needs parsers `--tool-call-parser apriel --reasoning-parser apriel`; card says the upstream PR was **not merged** and points to a custom Docker image (`docker.io/amant555/vllm_apriel:latest`) — check current vLLM before relying on it.
- Function-calling scores (card, vendor): BFCL v3 63.5, Tau2 Telecom 69, Retail 66.7, Airline 58, ComplexFuncBench 33.2, IFBench 69.
- Thinking: **always on**. Template forces the answer to start with "Here are my reasoning steps:"; no documented off switch. RL trained to use ~30% fewer reasoning tokens than 1.5. **Multi-turn rule: earlier assistant turns must contain only the final response, no reasoning** (card, Usage Guidelines §4). So reasoning is dropped from history.
- Coding (card): LiveCodeBench 81, SciCode 37, SWE-bench Verified 23, Terminal-Bench 14, Aider Polyglot 37.7, StructEval 79. → good at short code, weak at long agentic coding.
- Multilingual: card says "Strongest performance is in English. Output quality may degrade in underrepresented languages." Persian: no data. UNVERIFIED.
- License: MIT (card). Fine-tuning and commercial hosting allowed. Base lineage: Apriel 1.5 was built from Pixtral-12B (Apache-2.0) — UNVERIFIED on the 1.6 card itself.
- Availability: HF `ServiceNow-AI/Apriel-1.6-15b-Thinker`, `…-GGUF`; Ollama `ServiceNow-AI/Apriel-1.6-15b-Thinker:Q4_K_M` (9.7 GB; badges tools, thinking, vision); Together AI. mlx-community has only **1.5** conversions (`mlx-community/Apriel-1.5-15b-Thinker-4bit` etc.), none for 1.6 found.
- Known issues: custom vLLM image needed (above). Parsing depends on `[BEGIN FINAL RESPONSE]` markers; generic parsers can break. Always-on thinking costs tokens for small decisions.

#### Apriel-1.5-15B-Thinker (Sep 2025)
- Predecessor. Same size, MIT. BFCL v3 51.9, Tau2 Telecom 57.8 (from 1.6 card). Superseded by 1.6.

#### SuperApriel-15B-Instruct / -Base (Apr 2026)
- 15B "token-mixer supernet": each of 48 layers has 4 mixers (full attention, sliding window, Gated DeltaNet, Kimi Delta Attention); 8 presets give 1.0×–10.7× decode speed at 32K from one checkpoint. Context 262K. MIT. English best.
  Source: https://hf.co/ServiceNow-AI/SuperApriel-15B-Instruct
- Serving needs a vLLM plugin from a Fast-LLM feature branch. Tau2 drops from 56.7 (all-attention) to 30.7 at the 2.85× preset. **Not suitable**: exotic architecture, no MLX/Ollama path, agent quality falls fast with speed.

#### Apriel-Nemotron-15B-Thinker (May 2025), Apriel-H1-15B (Oct 2025), Apriel-5B (Apr 2025)
- All MIT. Superseded by 1.6, or hybrid architectures (H1) without a clear MLX path. Skip.

---

### LG EXAONE

**Unsuitable for DCLab: the small and mid models are non-commercial.**

#### EXAONE 4.0 (1.2B, 32B; Jul 2025), EXAONE 4.0.1-32B, EXAONE 4.5-33B (Apr 2026, VLM)
- EXAONE 4.5: 33B (31.7B LM + 1.29B vision), dense, 262,144 context, `enable_thinking=True` by default (set False to turn off), vLLM `--tool-call-parser hermes --reasoning-parser qwen3`. Languages: en, ko, es, de, ja, vi. No Persian.
  Source: https://hf.co/LGAI-EXAONE/EXAONE-4.5-33B
- Benchmarks (card, reasoning mode): LiveCodeBench v6 81.4, τ²-Retail 77.9, Airline 56.5, Telecom 73.0, IFEval 89.6.
- License: **"EXAONE AI Model License Agreement 1.2 - NC"** (LICENSE file in the 4.5-33B and 4.0-1.2B repos). Verified quotes:
  - §2.1(a): use "solely for research and educational purposes".
  - §3.1: "expressly prohibited from using the Model, Derivatives, or Output for any commercial purposes … Any commercial exploitation … requires a separate commercial license".
  - §3.1: shall not use the Model, Derivatives or Output "to develop or improve any models that compete with the Licensor's models".
  - §2.1(c): a modified model "must include 'EXAONE' at the beginning of its name".
  - Revocable; governed by Korean law (KCAB arbitration, Seoul).
- Verdict: cannot be fine-tuned for a paid hosted service without a separate deal with LG. Even training data distilled from its outputs is restricted (anti-competition clause).

#### K-EXAONE-236B-A23B (Dec 2025) and K-EXAONE-2.0-750B-A37B (Jul 2026)
- K-EXAONE 2.0 card says `license: apache-2.0`. (K-EXAONE 236B repo shows `license: other`; terms not read.) Languages listed: en, ko, es, de, ja, vi, fr, it, pl, pt — no Persian.
  Source: https://hf.co/LGAI-EXAONE/K-EXAONE-2.0-750B-A37B
- 750B total: far too large for DCLab. Mentioned only because it is LG's one permissive release.

---

### ByteDance Seed-OSS-36B

#### Seed-OSS-36B-Instruct / -Base / -Base-woSyn (Aug 2025)
- Params: 36B dense, 64 layers, GQA 80/8, vocab 155K. Context: **512K native**. Trained on 12T tokens.
  Source: https://hf.co/ByteDance-Seed/Seed-OSS-36B-Instruct
- Tool calling: native XML-style format: `<seed:tool_call><function=NAME><parameter=ARG>value</parameter></function></seed:tool_call>`. vLLM `--tool-call-parser seed_oss`. Optional `use_json_tooldef` renders tools as JSON schema; default renders them as Python signatures.
- Thinking: **thinking budget** via template kwarg `thinking_budget` (−1 unlimited default; 0 = answer directly; best in multiples of 512). The model emits `<seed:cot_budget_reflect>` progress notes inside `<seed:think>`. No clean on/off flag; budget 0 is the off switch.
- Multi-turn thinking (from chat_template.jinja): if an earlier assistant message carries `reasoning_content`, the template **re-renders it** (`<seed:think>…</seed:think>`), including on tool-call turns. So history thinking is kept if you pass it, dropped if you do not. Interleaved thinking + tool calls in one assistant turn is supported by the template.
- Benchmarks (card): LiveCodeBench v6 67.4, SWE-bench Verified 56 (OpenHands), TAU1-Retail 70.4, Airline 46, MMMLU 78.4, RULER-128K 94.6, IFEval 85.8.
- Multilingual: "primarily optimized for international (i18n) use cases" (card). MMMLU 78.4. Persian: no data. UNVERIFIED.
- License: Apache-2.0 (card + LICENSE). Fine-tuning + commercial hosting allowed.
- Availability: HF `ByteDance-Seed/Seed-OSS-36B-Instruct`; MLX `mlx-community/Seed-OSS-36B-Instruct-4bit` and `-8bit`; Ollama community only (`milkey/Seed-OSS-36B-Instruct`).
- Template quirks I saw in chat_template.jinja (my reading, not a reported bug): default system prompt with tools is "You are Doubao…"; tool section has Chinese labels (`[必填]`/`[选填]`, format hint in Chinese); `number` types render as `int`; non-string tool-call arguments are rendered with Jinja `| string` (Python-style repr, not JSON) in history. These matter when you build SFT data in its format.
- Known bugs: llama.cpp had a Seed-OSS grammar bug (unlimited newlines → runaway output) that was fixed (secondary: https://app.semanticdiff.com/gh/ngxson/llama.cpp/commit/1920345c3bcec451421bb6abc4981678cc721154).
- Fit: strong agent/code base, but 36B dense = slow on M1 Max; over the 35B target. Hermes 4.3 is a ready-made Hermes-format fine-tune of it.

#### Other ByteDance Seed releases (2026)
- Stable-DiffCoder-8B (Jan 2026, MIT, diffusion code model), Cola-DLM (May 2026, research). Not chat/agent models. Skip.

---

### Tencent Hunyuan / Hy

**Licensing changed in 2026.** The 2025 small models use a restrictive community license. The 2026 "Hy" models are Apache-2.0 (per card metadata).

#### Hunyuan dense 0.5B / 1.8B / 4B / 7B Instruct (Jul–Aug 2025)
- 256K context; hybrid fast/slow thinking (`enable_thinking=False` or `/no_think`, `/think`); output in `<think>…</think><answer>…</answer>`. Agent scores (7B): BFCL v3 70.8, τ-Bench 35.3, C3-Bench 68.5. LiveCodeBench 57 (7B), 49.4 (4B).
  Source: https://hf.co/tencent/Hunyuan-7B-Instruct
- License: **TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT** (LICENSE in repo). Verified terms:
  - "THIS LICENSE AGREEMENT DOES NOT APPLY IN THE EUROPEAN UNION, UNITED KINGDOM AND SOUTH KOREA"; Territory = worldwide excluding EU, UK, South Korea; use outside the Territory is "unlicensed".
  - >100M MAU → must request a license.
  - §5(b): "must not use the Tencent Hunyuan Works or any Output … to improve any other AI model" (other than Hunyuan derivatives).
  - "Model Derivatives" explicitly includes distillation and synthetic-data training.
  - AUP item 14 bans "high-stakes automated decisions" in credit, employment, housing, insurance, essential services, etc. **This directly hits typical DCLab business-table use cases** (credit, churn-to-pricing, HR).
  - Hosted service allowed inside the Territory, with notices and AUP flow-down.
- Verdict: **avoid** for DCLab (EU/UK exclusion + no-training-other-models + high-stakes-decision ban).

#### Hunyuan-A13B (80B total / 13B active; 2025)
- Same Tencent community license (`license: other`). Too large and same restrictions. Skip.

#### Hy3 (295B-A21B, Jul 2026), Hy4-preview (770B-A49B, Aug 2026)
- Cards say `license: apache-2.0`. Hy3: 256K context, `reasoning_effort` = `no_think` (default) / `low` / `high`. Hy4-preview: 1M context.
  Sources: https://hf.co/tencent/Hy3 , https://hf.co/tencent/Hy4-preview
- Far too large for DCLab. LICENSE file text not read: confirm it is plain Apache-2.0 with no added territory clause before use. UNVERIFIED.

#### Hy-Memory-30B-A3B (Oct 15 2026) and Hy-MT2 (May 2026)
- Hy-Memory-30B-A3B: `apache-2.0`, en/zh, ~3B active. A **memory-writer** model for Mem0/MemOS-type frameworks, not a general agent. GGUF and FP8 available. Source: https://hf.co/tencent/Hy-Memory-30B-A3B
- Hy-MT2 1.8B / 7B / 30B-A3B: translation-only models, `apache-2.0`, 36 languages **including `fa` (Persian)**. Source: https://hf.co/tencent/Hy-MT2-1.8B
  → Useful as a cheap Persian↔English translation tool inside DCLab. Not a base for the solver. LICENSE text not opened; card metadata only.

---

### Xiaomi MiMo

All MiMo repos are `license: mit` on their cards. Source org: https://hf.co/XiaomiMiMo

#### MiMo-V2.6-Distill-Qwen-9B (Sep 21 2026) — notable new candidate
- Params: ~9.7B (base **Qwen3.5-9B**, Apache-2.0; hybrid linear-attention + full attention; multimodal `qwen3_5`). SFT on 77.4B tokens (27.2B loss-bearing) of MiMo-V2.6-generated agent data: code, cyber, general agent, visual.
  Source: https://hf.co/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
- Card results (vendor, vs Qwen3.5-9B): SWE-bench Verified 61.1 (vs 60.0), SWE-Pro 44.6 (vs 32.0), Terminal-Bench 2.1 37.1 (vs 27.0), Toolathlon-Verified 35.2 (vs 25.9), AutomationBench 30.3 (vs 5.0), OfficeQA 19.5 (vs 9.0).
- Tool calling (from chat_template.jinja): tools listed as JSON inside `<tools>…</tools>` in the system turn; calls rendered as `<tool_call><function=NAME><parameter=ARG>value</parameter></function></tool_call>` (Qwen3-Coder-style XML). ChatML roles.
- Thinking: `enable_thinking` kwarg (template inserts empty `<think></think>` when False). SGLang `--reasoning-parser mimo`.
- Multi-turn thinking (template): every earlier assistant turn is rendered as `<think>{reasoning_content}</think>{content}{tool_calls}` → **history thinking is kept when provided**; interleaved thinking + tool calls is the native shape.
- Persian: no data on this model. Base Qwen3.5 family — see Persian section (Qwen3.x ranks well). UNVERIFIED for this fine-tune.
- License: MIT (fine-tune) on top of Apache-2.0 (base). Both allow fine-tuning and commercial hosting. The card describes it as an "SFT checkpoint as a starting point for open research in agentic RL" — release quality, not a polished product.
- Availability: HF only. No mlx-community or Ollama build found. MLX support for `qwen3_5` text+vision is UNVERIFIED (mlx-lm vs mlx-vlm). Serving documented for SGLang.

#### MiMo-V2.6-Flash / Pro (Sep 2026), MiMo-V2.5 (Apr 2026), MiMo-V2-Flash (Dec 2025)
- MiMo-V2.6-Flash: 309B total / 15B active, 1M context, omnimodal. MIT. Too large. Source: https://hf.co/XiaomiMiMo/MiMo-V2.6-Flash-RL
- MiMo-V2-Flash: large MoE (mlx-community 4-bit exists). Too large for 32–64 GB. Skip.

#### MiMo-7B (RL-0530, SFT; 2025)
- 7.8B, MIT, reasoning/math/code focused; mlx-community 4-bit builds exist. Older; superseded by the 9B distill. Skip.

---

### Arcee

#### AFM-4.5B (Jul 2025)
- 4.6B dense, `apache-2.0` (card metadata; earlier it had an Arcee license — now Apache), languages en, es, fr, de, it, pt, ru, ar, hi, ko, zh (no Persian listed). Source: https://hf.co/arcee-ai/AFM-4.5B
- Possible student base, but no Persian and no clear tool-calling story. Low priority.

#### Trinity Mini (26B total / 3B active) and Trinity Nano Preview (~6B total; Dec 2025–Mar 2026)
- Trinity Mini: `afmoe` MoE, 128 experts (8 active + 1 shared), 128K context, 10T tokens. vLLM `--tool-call-parser hermes --reasoning-parser deepseek_r1`. Recommended temp 0.15. Source: https://hf.co/arcee-ai/Trinity-Mini
- License: **OpenMDW-1.1** (LICENSE file read). Permissive: "permission is hereby granted, free of charge, to deal in the Model Materials without restriction"; keep the license and notices on redistribution; patent-suit termination; **no restrictions on outputs**. Fine-tuning + commercial hosting allowed.
- Languages: en, es, fr, de, it, pt, ru, ar, hi, ko, zh. No Persian.
- Availability: GGUF (`arcee-ai/Trinity-Mini-GGUF`, llama.cpp ≥ b7061), FP8/W4A16/NVFP4; Trinity-Nano has Arcee-made MLX 5/6/8-bit (`arcee-ai/Trinity-Nano-Preview-MLX-8bit`). Custom `afmoe` arch needs recent transformers or `trust_remote_code`.
- Trinity Large (Thinking/Preview): too large. Skip.

---

### Ant Group inclusionAI Ling / Ring

All Ling/Ring repos are `license: mit` on their cards. Source org: https://hf.co/inclusionAI

#### Ling-3.0-tiny (Aug 10 2026) — notable new candidate for students/fast agent
- Params: **7.9B total / 1.3B active** MoE (128 experts, 8 + 1 shared). Hybrid linear attention: 3 Kimi Delta Attention layers + 1 MLA layer per block. 256K context with YaRN (128K original).
  Source: https://hf.co/inclusionAI/Ling-3.0-tiny
- Thinking: on by default; per request `chat_template_kwargs: {"enable_thinking": false}`. Parsers `--reasoning-parser ling3 --tool-call-parser ling3` (SGLang/vLLM). Tool-call format details are in the SGLang cookbook, not the card. UNVERIFIED.
- Benchmarks: Artificial Analysis Intelligence Index 25, Agentic Index 16 (vendor quoting AA). Main table is an image (not read).
- Apple Silicon: card says validated on MacBook / Mac mini; FP8 ≈ 86–90 tok/s on M4 Pro, ~8.3 GiB peak at 8K. **Ollama support only via an unmerged PR (#17643) built from source, MLX backend.** No mlx-community conversion found; MLX **training** support for the `bailing_hybrid` architecture is UNVERIFIED.
- Languages: zh/en tags only. Persian: no data. UNVERIFIED.
- License: MIT.
- Risk: very new, custom architecture (`custom_code`), immature tooling. Good speed profile for "student" decisions if MLX training works.

#### Ling-3.0-flash (~127B total, Aug 2026), Ling-3.1-flash (~572B, Oct 2026), Ring-2.5/2.6-1T
- MIT, but far too large. Ling-3.0-flash-Fin is a finance/agent fine-tune of flash. Skip.

#### Ling-mini-2.0 (16B total / 1.4B active; Sep 2025) and Ring-mini-2.0 (thinking)
- MIT. mlx-community builds: `mlx-community/Ling-mini-2.0-4bit`, `-6bit`; GGUF for Ring-mini. Older generation; Ling-3.0-tiny replaces it.

#### LemonDecider 4B (Oct 5 2026) — directly relevant to DCLab's "students"
- A **typed decision engine** on Qwen3.5-4B: one forward pass returns a probability over up to 255 options (choice / yes-no / ordinal score). No text generation; 256K input, never truncated. SFT on ~805k public decision/QA rows; two runs weight-merged. `apache-2.0`.
  Source: https://hf.co/inclusionAI/LemonDecider
- Uncalibrated (temperature 1.0 per card). Custom runtime (`trust_remote_code`, transformers 5.17, flash-linear-attention). No MLX path. Use it as a **design reference** (readout-head classifier over a small LLM) for DCLab's decision students, not as a dependency.

---

### StepFun

- Step-3.5-Flash (~199B, Feb 2026), Step-3.7-Flash (~201B VLM, May 2026): `apache-2.0`. Too large.
  Sources: https://hf.co/stepfun-ai/Step-3.5-Flash , https://hf.co/stepfun-ai/Step-3.7-Flash
- Step-5-Preview (~604B, Sep 2026): `license: other`, terms not read (UNVERIFIED). Too large.
- Step3-VL-10B (Jan 2026, Apache-2.0): vision model, not a tool-agent base. Skip.

---

### Baidu ERNIE 4.5 (open)

#### ERNIE-4.5-21B-A3B-Thinking (Sep 2025)
- 21B total / 3B active MoE (64 experts, 6 active + 2 shared), 131,072 context. Function calling supported (FastDeploy `--tool-call-parser ernie_x1`; vLLM parser needs main branch per card). Thinking model with long thinking ("strongly recommend … highly complex reasoning tasks").
  Source: https://hf.co/baidu/ERNIE-4.5-21B-A3B-Thinking
- Languages: en, zh only (card). Persian: no data.
- License: Apache-2.0. Fine-tuning + hosting allowed.
- Availability: transformers weights; mlx-community has the non-thinking `ERNIE-4.5-21B-A3B-PT` (4/5/6/8-bit, bf16), not the Thinking variant.
- Fit: OK size, but en/zh focus and weaker agent ecosystem than Qwen-based options. Medium-low priority.
- ERNIE-4.5-0.3B: Apache-2.0, very small; possible tiny student, en/zh only.
- ERNIE 5 open weights: none found on https://hf.co/baidu as of 2026-10-23.

---

### Meituan LongCat

- LongCat-Flash-Lite (~69B, Jan 2026), LongCat-Flash-Thinking-2601, LongCat-2.0 (~1.78T, Jul 2026): all MIT, all too large for DCLab. Source: https://hf.co/meituan-longcat
- Skip.

---

### Other new open-weight releases seen (Jul–Oct 2026), one line each
- **Qwen3.6 / Qwen3.8** (e.g. `Qwen/Qwen3.6-35B-A3B`, `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.8-27B`) and **Gemma 4** (`google/gemma-4-31B-it`, `-26B-A4B-it`, `-E4B-it`, `-E2B-it`) exist (seen as base_model tags and on the Persian leaderboard). Covered by other research; they matter here because they lead on Persian.
- **tencent ContextPilot-8B/14B** (Qwen3) and **-E4B** (Gemma 4): context-management agent fine-tunes, `license: other` (not read). Aug 2026.
- **Hy-Memory-30B-A3B** (Tencent, Apache-2.0): memory writer, above.
- **MiMo-V2.6-Distill-Qwen-9B**, **Ling-3.0-tiny**, **LemonDecider**: above.

---

### Persian-specific models and leaderboards

#### Open Persian LLM Leaderboard (OPLL; Part AI + Amirkabir University LLM Lab)
- Space: https://hf.co/spaces/opll-org/Open-Persian-LLM-Leaderboard (newer data file, 85 rows) and the older https://hf.co/spaces/PartAI/open-persian-llm-leaderboard . Paper: MELAC (https://preview.aclanthology.org/setup/2025.ijcnlp-long.105).
- 21 Persian tasks (general knowledge, GSM8K-fa, homographs, PiQA, proverbs, Winogrande, ARC, Feqh, Iranian law, Parsi literature, BoolQA, reading comprehension, MMLU-Pro, Iranian social norms, truthfulness…).
- I computed the plain average over the 21 columns from `leaderboard_data.jsonl` (same as the app's "Average Accuracy"). Top open models:

| Model | Params | Avg |
|---|---|---|
| gemma-4-31B-it | 31B | 78.6 |
| Qwen3.8-27B | 27B | 76.1 |
| gemma-4-26B-A4B-it | 26B (4B active) | 70.7 |
| Llama-4-Scout-17B-16E | 109B | 69.0 |
| Llama-3.3-70B-Instruct | 70B | 68.5 |
| gemma-3-27b-it | 27B | 64.7 |
| Qwen3-32B | 33B | 62.8 |
| gemma-3-12b-it | 12B | 61.7 |
| gemma-4-E4B-it | ~4B effective | 59.0 |
| Qwen3-14B | 15B | 57.8 |
| Qwen3-30B-A3B | 31B (3B active) | 55.2 |
| Qwen3-8B | 8B | 55.0 |
| gemma-4-E2B-it | ~2B effective | 50.5 |
| Qwen3-4B | 4B | 50.5 |
| Hermes-3-Llama-3.1-8B | 8B | 50.4 |
| Dorna2-Llama3.1-8B (PartAI) | 8B | 48.8 |
| Qwen3-1.7B | 2B | 41.3 |
| PersianMind-v1.0 | 7B | 37.1 |
| Maral-7B-alpha-1 | 7B | 33.4 |

  (For reference: gpt-4o-2024-08-06 76.3, gpt-4.1 74.4.)
- **Caveats on this data (important):**
  - The newest rows (Gemma 4, Qwen3.8) have "Precision: unknown" and some scores jump in ways that suggest a changed protocol (e.g. Feqh 98.2 for Qwen3.8-27B vs ≤56.6 for every older model incl. GPT-4o). Compare new rows with old rows carefully.
  - Data errors exist: the "gpt-oss-120b" row points to `openai/gpt-oss-20b` and lists 1.8B params.
  - None of my families (Hermes 4, Apriel, Seed-OSS, EXAONE, MiMo, Ling, Trinity, ERNIE) are on the board.
- **What it says:** Gemma 4 and Qwen 3.x bases lead on Persian. The dedicated Persian fine-tunes (Dorna, Dorna2, PersianMind, Maral) score **below** their own bases or below newer general bases. A Persian-only fine-tune of an old base is not worth it; start from a strong multilingual base and add Persian task data.

#### Persian models
- **PartAI Dorna / Dorna2** (Llama 3 / 3.1 8B; Llama licenses). No newer Dorna release found on https://hf.co/PartAI as of 2026-10-23. PartAI's other Persian assets: TookaBERT, Tooka-SBERT(-V2) embeddings (Apache-2.0 on TookaBERT/Tooka-SBERT) — useful for Persian retrieval.
- **PersianMind-v1.0** (University of Tehran): `cc-by-nc-sa-4.0` → **non-commercial, unusable**. https://hf.co/universitytehran/PersianMind-v1.0
- **Maral-7B-alpha-1**: MIT, but weak (33.4 avg).
- **Hormoz-8B** (mann-e): MIT, 53.1 avg — near Aya-Expanse-8B, its likely base (UNVERIFIED).
- Community 2026 LoRAs (Qwen3-4B, Qwen3-1.7B, Gemma-3-4B, BitNet-2B Persian) exist but are tiny hobby projects with no evaluations. `alphaedge-ai/Qwen3.5-2B-fas-16384` is a Persian vocabulary-trimmed Qwen3.5-2B (Apache-2.0) — untested.
- **Tencent Hy-MT2** (1.8B/7B/30B-A3B) lists Persian among 36 languages, Apache-2.0 → a practical translation helper.

#### Other Persian benchmarks (secondary/papers)
- MatinaAI Persian LLM leaderboard: https://hf.co/spaces/MatinaAI/persian_llm_leaderboard (not read).
- ParsBench toolkit (PyPI `parsbench`); no current results table found.
- PARSE (2026, reasoning QA): Qwen2.5-72B best zero-shot; Gemma-2-27B good few-shot (secondary: https://arxiv.org/html/2602.01246v1).
- 2510.12807 (2025): Gemma 2 led open models on Persian NLP tasks (secondary: https://arxiv.org/html/2510.12807v1).
- Open Persian LLM **Alignment** Leaderboard (safety/fairness): https://mcilab-llm-alignment-evaluation.hf.space

---

### Quick verdict table (my families only)

| Model | Size (total/active) | License for paid hosting + fine-tune | Agent/tools | Persian | MLX path | Verdict |
|---|---|---|---|---|---|---|
| Hermes 4 14B | 14B dense | Yes (Apache-2.0, Qwen3) | Hermes `<tool_call>`, hybrid think | via Qwen3 (~58 avg for base) | mlx-community 4-bit | Good solver/workflow base |
| MiMo-V2.6-Distill-Qwen-9B | ~9.7B | Yes (MIT on Apache-2.0) | XML tool calls, think kept in history; strong agent SFT | via Qwen3.5 (UNVERIFIED) | UNVERIFIED | Best agent prior per size; verify MLX |
| Apriel-1.6-15B | 15B dense | Yes (MIT) | Good function-calling scores; always thinks | Weak/unknown | 1.5 only | Second choice; English-centric |
| Seed-OSS-36B / Hermes 4.3 36B | 36B dense | Yes (Apache-2.0) | Strong (SWE-V 56, 512K) | i18n, unknown | mlx-community 4/8-bit | Too big/slow for M1 training |
| Ling-3.0-tiny | 7.9B / 1.3B | Yes (MIT) | ling3 parser; think on/off | Unknown | Not yet (PR only) | Watch for students |
| Trinity Mini | 26B / 3B | Yes (OpenMDW-1.1) | hermes parser | No | GGUF; Nano has MLX | Possible, no Persian |
| ERNIE-4.5-21B-A3B-Thinking | 21B / 3B | Yes (Apache-2.0) | ernie_x1 parser | No (en/zh) | non-thinking only | Low priority |
| EXAONE 4.0/4.5 | 1.2B / 32–33B | **No (NC)** | — | No | — | Excluded |
| Hunyuan 0.5–7B, A13B | ≤80B | **No for DCLab** (EU/UK/KR excluded, no training other models, high-stakes ban) | — | — | — | Excluded |
