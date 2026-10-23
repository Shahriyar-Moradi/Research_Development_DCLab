# Fine-tuning open models on an M1 Max (32 GB / 64 GB) for agent trajectories — late October 2026

Research date: 2026-10-23. Goal: LoRA/QLoRA SFT, then DPO, on whole agent runs of 8K–32K tokens
(multi-turn, tool calls, thinking), with MLX on Apple silicon.

Labels used below:
- **[M]** measured by someone and published (source given).
- **[C]** community report (GitHub issue by a user, blog, Reddit). Real, but one person's setup.
- **[Code]** read directly from the source code at `main` on 2026-10-23.
- **[E]** my estimate. The arithmetic is shown. Treat ±30–50% as normal.

Nothing here was run on an M1 Max. The local machine is an 8 GB M1, too small for a useful test, and
its `.venv` mlx-lm does not import (transformers / huggingface-hub version clash).

---

## 0. Bottom line (short)

1. **The Mac can train, but it is slow.** At 16K tokens per example, one epoch over 1,000 examples
   (16.4M tokens) is about **1.5 days for a 4B, 2.5–3 days for an 8–9B, 5 days for a 14B, ~2 days
   (very uncertain) for a 30B-A3B MoE, and ~8 days for a 27B dense** on a 32-core M1 Max [E].
   DPO costs ~2–2.5× more per pair. Several epochs plus tuning runs means weeks.
2. **32 GB:** the practical ceiling is **an 8–9B dense model with QLoRA at 8K**, or **a 4B at 16K**.
   16K on an 8B is borderline (est. ~25 GB against ~21 GB the GPU may use by default). 32K does not fit
   on any useful model. A 30B MoE only loads; it cannot train long sequences.
3. **64 GB:** the practical ceiling is **a 30B-A3B MoE (Qwen3.6-35B-A3B, Qwen3-30B-A3B, Nemotron 3.5
   Lightning) with QLoRA at 8K–16K**, or **a 14B dense at 16K**, or **an 8–9B dense at 32K**. A 27–31B
   dense fits at 8K but is too slow (~8 days/epoch).
4. **Architecture matters more than size on Apple silicon right now.** Two big October 2026 changes
   made long-sequence training possible at all (MLX 0.32.4: fused attention backward and a gated-delta
   backward). But they only cover *causal* attention. **Gemma 4 and gpt-oss use sliding-window
   attention, which still falls back to a full T×T attention matrix in training** → impractical above
   ~8K (Gemma 4) or ~4K (gpt-oss) [Code + E]. **Qwen3.5/3.6/3.8 need mlx-lm-lora**, because stock
   mlx-lm still runs their linear-attention layers on a slow per-token path [M][Code].
5. **No trainer does per-assistant-turn masking out of the box.** `--mask-prompt` trains only the
   *last* message [Code]. For agent runs you need a ~60-line custom loss (mlx-lm's `train()` accepts
   `loss=` and `iterate_batches=` hooks).
6. **Sweet spot (my judgment):** develop and pilot on the Mac with a **4B (Qwen3.5-4B or Qwen3-4B)**;
   do the real runs on rented GPUs or Tinker. A 30B-A3B MoE costs about **$7–20 per epoch of 1,000 ×
   16K on Tinker** [E from list prices], versus ~2 days of a 64 GB Mac.

---

## 1. What changed in Sept–Oct 2026 (why older advice is wrong)

| Change | Date | Effect | Label |
|---|---|---|---|
| MLX PR #4563 "VJP for scaled dot product attention" | merged 2026-09-29, in **mlx 0.32.4** (PyPI 2026-10-14) | Before: the attention backward allocated a full `[T × T]` matrix. After: tiled; "up to 24x" less memory ("24.63 GiB to 1 GiB" at the longest shape, M5 Ultra) and up to 2.8× faster for causal. | [M] by the MLX team |
| MLX PR #4565 "VJP for gated delta net" | merged 2026-09-29, in 0.32.4 | A Metal backward for Qwen3.5-style linear attention. "Chunked kernel is 1.4x to 6.3x faster than the sequential one and 9x to 79x faster than the fallback." The chunked kernel uses the Neural Accelerator (NAX, M5 only). **An M1 gets the sequential kernel.** | [M] |
| mlx-lm still calls `gated_delta_update(..., use_kernel=not self.training)` | open issue #1956 (2026-10-08) | Stock `mlx_lm.lora` does **not** use that kernel in training. | [Code][C] |
| mlx-lm-lora PR #102 "fast VJP for gated-delta layers" | merged 2026-10-01 | mlx-lm-lora patches mlx-lm at runtime to call `mx.fast.gated_delta_update` when MLX has it. | [Code] |

Before these, long sequences were hopeless. Example: mlx-lm PR #47 (2025), Llama-3.2-1B bf16 at 4096
tokens peaked at **69.8 GB** [M]. And for Qwen3.5 linear layers, PR #1870 measured the old path at
"About 8 MB per token per layer": 31.4 GB for one layer at 2048 tokens, 108 GB at 4096 (M4 Pro) [M].

**Practical rule:** use `mlx>=0.32.4`, `mlx-lm>=0.32.2` and `mlx-lm-lora>=5.10.0` (released 2026-10-23).

---

## 2. Tools: mlx-lm vs mlx-lm-lora

### mlx-lm 0.32.2 (`mlx_lm.lora`)
- LoRA, DoRA, full. **QLoRA = point `--model` at a quantized model** (for example an mlx-community 4-bit repo) [Code/docs].
- Data formats: `chat`, `tools`, `completions`, `text` (JSONL). The `tools` format passes `tools=` to the
  model's chat template, so tool definitions and `tool_calls` render the model's own way [Code].
- `--grad-checkpoint` exists. Default `max_seq_length` is **2048**, and longer examples are **truncated
  with a warning** [Code]. Set `--max-seq-length` explicitly.
- **Masking limitation [Code]:** with `--mask-prompt`, loss is computed only on the **final message**.
  The offset comes from rendering `messages[:-1]`. The batch carries one `(offset, length)` pair per
  example. So one contiguous span is trained. There is no per-turn mask. Without `--mask-prompt`, every
  token is trained, including tool outputs and the system prompt. That teaches the model to *invent tool
  results*, which is bad for an agent.
- Hooks: `train(model, optimizer, train_ds, val_ds, args, loss=..., iterate_batches=...)` [Code]. A custom
  `iterate_batches` can yield a per-token 0/1 mask, and a custom `loss` can multiply by it. That is the
  clean fix for per-assistant-turn masking (about 60 lines).
- No chat-template kwargs (for example `preserve_thinking`, `enable_thinking`) on the CLI [Code]. To
  control them: pre-render the text yourself and use the `text` format with your own mask, or edit the
  model folder's `chat_template.jinja`.
- **No gradient clipping and no NaN guard** (issue #1658, open): "one bad batch loses the whole run" [C].
  This matters for multi-day runs. Save adapters often (`--save-every`).
- Open PR #1983 `--loss-chunk-size` (not merged): the logits `[T × vocab]` are the biggest single tensor.
  Measured Qwen3-4B-4bit, 4096 tokens, checkpointing on: **6.49 GB → 5.37 GB** [M].
- MoE: LoRA wraps expert layers (`SwitchLinear`) with **per-expert** A/B matrices `(num_experts, r, dim)`,
  stored in float32 [Code]. On a 128–256-expert model this is large. A Gemma 4 26B-A4B run showed
  "Trainable parameters: 1.414% (356.708M/25233.053M)" [C]. At 16 bytes per trainable parameter
  (fp32 weight + grad + 2 Adam states), 357M ≈ 5.7 GB [E]. Restrict `keys` to attention and shared-expert
  projections.

### mlx-lm-lora 5.10.0 (`mlx_lm_lora.train`, by Gökdeniz Gülmez)
- SFT, DPO, CPO, ORPO, GRPO/GSPO/Dr.GRPO/DAPO, online DPO, XPO, PPO, KLPO, and more [docs].
- `--load-in-4bits/6bits/8bits` QLoRA; `--grad-checkpoint`; `--sft-loss-type chunked_nll` (memory-bounded
  loss, 256-token chunks) [Code].
- Recurrent layers (Qwen3.5/Next, Mamba, nemotron_h, granitemoehybrid, …) are detected. It installs the
  fast gated-delta VJP and checkpointed recurrence blocks [Code].
- **Same masking limitation:** `--mask-prompt` = everything up to the last message is masked [Code].
- `--seq-step-size` / "efficient long context" splits a long example into chunks with a KV cache. Two
  warnings. (1) Gradients do not flow back through the cache to earlier chunks, so it is an approximation
  [Code]. (2) Issue #84: it "silently corrupts SFT loss above seq 512 and goes NaN at seq >= 2048". It was
  closed as fixed, but the reporter said the symptom was "still present" after the fix [C].
  **Do not use it without checking the loss against a normal run.**
- **DPO loads a second full copy of the base model as the reference** (`load_reference_model` → `load(args.model)`) [Code].
  That adds the weights' size again. A 30B-A3B 4-bit adds ~18–20 GB. ORPO/CPO are reference-free.
- Its README benchmark table (M4 Pro 24 GB, "context 4096"): Qwen3-8B 4-bit SFT "~4.1 it/s, ~6-10 GB".
  **I do not trust this for long sequences.** At 4096 tokens per step that would be ~16.8K tokens/s,
  about 0.8 PFLOP/s of training compute on a laptop chip. The samples were very likely much shorter than 4096.

### Other risks seen in open issues
- #1185 (open): Qwen3.5-27B/35B-A3B LoRA crashes after 60–220 iterations with
  "[metal::malloc] Resource limit (499000) exceeded", a buffer-handle leak, not out-of-memory [C].
  Multi-day runs need resume (`--resume-adapter-file`) and a watchdog.
- #1206 (open): Qwen3.5-9B-4bit crashes on the first backward pass on an M5 Max [C]. It may be M5-specific.
- #1348 (open): Qwen3-8B-4bit, 16K max length, rank 16 on 7 modules: hangs after the first validation pass (M5 Max) [C].

---

## 3. Architecture support for *training at long context* on an M1 (no NAX)

The fused attention backward is used only for causal attention without an array mask, and only for head
dims {64, 72, 80, 96, 128, 192, 256} (512 only on NAX GPUs) [Code: `mlx/backend/metal/scaled_dot_product_attention.cpp`, `has_fused_kernel`].
mlx-lm builds an **array mask for sliding-window layers whenever the sequence is longer than the window**
(`create_attention_mask`, `base.py`) [Code]. So in training those layers fall back to a materialized
T×T matrix.

| Family (MLX module) | LoRA backward works? | Long-T attention on M1 | Verdict for 16K agent runs |
|---|---|---|---|
| **Qwen3 classic** dense 4B/8B/14B/32B (`qwen3`), **Qwen3-30B-A3B-2507** (`qwen3_moe`) | Yes. MoE routing fix merged 2026-08-28 (#1787) | Fused (head dim 128, causal, all layers) | **Best-supported path.** Native context 32K (dense) / 256K (2507 MoE). |
| **Qwen3.5 / 3.6 / 3.8** (`qwen3_5`, `qwen3_5_moe`): 0.8B–9B, 27B, 35B-A3B | Yes, but use **mlx-lm-lora** (fast gated-delta VJP). Stock mlx-lm is the slow path (#1956: 0.073 it/s at ~600 tokens with 16 layers on an M4 Pro) | Full-attention layers fused (head 256). The 3 of 4 layers that are linear attention use MLX's sequential VJP kernel on M1. **Its M1 speed and memory at 16K are unpublished** | Good models; tooling only 3 weeks old. Pilot first. Unsloth "does not recommend QLoRA (4-bit) training on the Qwen3.5 models" due to "higher than normal quantization differences" (bitsandbytes; MLX 4-bit untested on this point). Prefer 8-bit base on 64 GB. |
| **Gemma 4** E2B/E4B/12B/31B (`gemma4_text`), 26B-A4B MoE | Yes. MoE routing fix in place (#1897 closed 2026-09-16) | **Sliding layers (most layers) fall back to T×T. On M1 the D512 global layers also fall back** (the blocked D512 path excludes training) | ≤8K practical. 16K only for E4B on 64 GB. Also #1363 (open): per-layer projection (E2B/E4B) not in the default LoRA targets. |
| **gpt-oss-20b** (`gpt_oss`) | Routing fix merged 2026-09-04 (#1843) | Sliding layers fall back (64 heads!). Full layers use attention sinks, and the fused VJP says "NYI: attention sinks" [Code] → may error, unverified | **Not practical above ~4K on any Mac.** |
| **Nemotron 3.5 Lightning 30B-A3B** (`nemotron_h`): 52 blocks = 23 Mamba-2, 23 MoE, **6 attention** (2 KV heads) | Code has the MoE `stop_gradient`. Mamba trains through the chunked `ssm_attn` ops [Code] | Attention fused (head 128, causal) | Promising on paper (tiny KV, few attention layers). **No public MLX LoRA run found.** License OpenMDW-1.1 (not Apache). Released 2026-08-11. |
| **Ministral 3** 3B/8B/14B, **Devstral Small 2** 24B (`ministral3`) | Standard dense | Fused (head 128, no sliding window in config) | Works like Qwen3 classic. Devstral 24B too slow on M1. |
| **Granite 4.1** 3B/8B/30B (`granite`) | Standard dense | Fused | Fine. Lower agent scores than Qwen. |
| **GLM-4.7-Flash** 30B (`glm4_moe_lite`, MLA) | Code has `stop_gradient` | Not checked (MLA head dims) | Unverified. |
| Phi-4 14B / mini, Llama 3.x | Standard | Fused | Older. Phi-4 context 16K. |

---

## 4. Memory: arithmetic and estimates

### Budget
- macOS lets the GPU use only part of RAM. Commonly logged `recommendedMaxWorkingSetSize`: ~21.8 GB
  (2/3) on 32 GB and ~48–51 GB (75%) on 64 GB machines [C]. It can be raised with
  `sudo sysctl iogpu.wired_limit_mb=…` (not persistent, unsupported, a system-setting change the owner
  must decide on).
- mlx-lm sets the wired limit to the recommended maximum at start (`maybe_set_recommended_wired_limit`) [Code].
- **Usable for training: ~20 GB on 32 GB; ~45 GB on 64 GB** (default). ~26 / ~56 GB if raised.

### Model (batch 1, gradient checkpointing on, MLX ≥ 0.32.4, chunked loss) [E]
`peak ≈ W + LoRA + checkpoints + one-layer working set + logits + 1.5 GB`
- **W** = real size of the mlx-community repo (measured file sizes, table below).
- **LoRA** = trainable params × 16 bytes (fp32 weight, grad, Adam m and v). Rank 16 on all 7 projections
  of every layer. Check: Qwen3-8B rank 16, 7 modules = **43.6M params**, matching issue #1348's log
  "43.647M" [C]. 43.6M × 16 B = 0.7 GB.
- **Checkpoints** = layers × T × hidden × 2 bytes. Example Qwen3.5-9B, 16K: 32 × 16384 × 4096 × 2 = 4.3 GB.
- **Working set** = 36 bytes × T × (hidden + q_dim + 2·kv_dim + ffn_width). The 36 is **calibrated** on
  PR #1983's two measurements: Qwen3-4B-4bit 4096×1: 5.369 − 2.26 (W) − 0.12 (LoRA) − 0.34 (ckpt) =
  2.65 GB over 4096 tokens × 18,432 width = **35 B**. Qwen3-0.6B-bf16 4096×2: 4.042 − 1.19 − 0.04 −
  0.27 = 2.54 GB over 8192 × 8192 = **38 B**. For MoE, ffn_width = top_k × expert_width + shared +
  top_k × hidden (the router copies each token top_k times).
- **Logits without chunking** = 1.8 bytes × T × vocab (the saving PR #1983 measured: 1.12 GB for 4096 ×
  151,936). For Qwen3.5 (vocab 248,320) at 16K that is **+7.3 GB**. So **always use a chunked loss**
  (`--sft-loss-type chunked_nll` in mlx-lm-lora).
- **Sliding/fallback attention** (Gemma 4, gpt-oss) adds ≈ 3 copies × heads × T² × 2 bytes for one layer:

| | 4K | 8K | 16K | 32K |
|---|---|---|---|---|
| Gemma-4-E4B (8 heads) | 0.8 GB | 3.2 GB | 12.9 GB | 51.5 GB |
| Gemma-4-12B / 26B-A4B (16 heads) | 1.6 | 6.4 | 25.8 | 103 |
| Gemma-4-31B (32 heads) | 3.2 | 12.9 | 51.5 | 206 |
| gpt-oss-20b (64 heads) | 6.4 | 25.8 | 103 | 412 |

### Measured 4-bit weight sizes (mlx-community repos, sum of .safetensors, 2026-10-23) [M]
Qwen3.5-4B 3.03 GB · Qwen3-8B 4.61 · Qwen3.5-9B 5.95 · Gemma-4-E4B 5.15 · Gemma-4-12B 6.74 · Qwen3-14B 8.31 ·
Ministral-3-14B 8.42 · gpt-oss-20b (MXFP4-Q8) 12.08 · Devstral-Small-2-24B 15.10 · Gemma-4-26B-A4B 15.34 ·
Qwen3.5/3.6/3.8-27B 16.05 · GLM-4.7-Flash 16.85 · Qwen3-30B-A3B 17.17 · Nemotron-3.5-Lightning-30B-A3B 17.78 ·
Gemma-4-31B 18.41 · Qwen3.6-35B-A3B 20.40. bf16: Qwen3.6-35B-A3B 70.2, Nemotron Lightning 63.2.

### Estimated peak memory, QLoRA 4-bit, rank 16, chunked loss [E] (GB; add the fallback table for Gemma/gpt-oss)

| Model | 8K | 16K | 32K | bf16 LoRA 16K |
|---|---|---|---|---|
| Qwen3.5-4B | 13 | 21 | 36 | 27 |
| Gemma-4-E4B | 14 (+3) | 20 (+13) | 33 (+52) | 31 (+13) |
| Qwen3-8B | 16 | 25 | 43 | 37 |
| Qwen3.5-9B | 18 | 28 | 48 | 42 |
| Gemma-4-12B | 21 (+6) | 32 (+26) | 54 (+103) | 49 (+26) |
| Qwen3-14B / Ministral-3-14B | 23 | 35 | 58–60 | 54–57 |
| gpt-oss-20b | 28 (+26) | 41 (+103) | — | 71 |
| Gemma-4-26B-A4B | 31 (+6) | 44 (+26) | — | 81 |
| Qwen3.6-35B-A3B | 33 | 44 | 66 | 94 |
| Nemotron-3.5-Lightning 30B-A3B | 34 | 49 | 78 | 94 |
| Qwen3.6/3.8-27B dense | 36 | 53 | 85 | 92 |
| Gemma-4-31B | 41 (+13) | 59 (+52) | 95 | 103 |

Not in the table (unknown, so the Qwen3.5/3.6 rows may be too low): the gated-delta backward's own state
memory on M1. The kernel's memory use was not published. PR #1870's chunked version needed 4.73 GB for one
layer at 4096 tokens [M].

---

## 5. Speed: arithmetic and estimates

### Anchors
- **[M] mlx-lm docs:** Mistral-7B (bf16), `--batch-size 1 --num-layers 4`, wikisql (short examples),
  M1 Max 32 GB: "about 250 tokens-per-second". Compute: forward 2 × 7.2B = 14.5 GFLOP/token, plus a
  backward through 4/32 layers ≈ 1.8 → ~16.3 GFLOP × 250 = **~4.1 TFLOPS effective**. (This was measured
  with older MLX.)
- **[M] llama.cpp #4167**, Llama-2-7B pp512 on an M1 Max 32-core GPU: F16 599.53 t/s, Q4_0 530.06 t/s →
  ~7–8 TFLOPS for forward-only matmuls. The 24-core M1 Max: Q4_0 400.26 t/s (**~25% slower**). A 32 GB
  M1 Max can have either GPU. The 64 GB model always has 32 cores.
- Peak FP16/FP32 of the 32-core M1 Max ≈ 10.4 TFLOPS (no double-rate FP16).

### Formula [E]
`tokens/s = 4 TFLOPS / (6 × N_active + 8 × T × attn_width × n_attn_layers)`
- 6N = forward + checkpoint recompute + backward to activations. LoRA weight grads are tiny.
- The attention term = causal forward 2·T·width per layer per token, ×4 for training.
- MoE: ×0.6 efficiency for gather-based expert matmuls. **This is my assumption, not measured.**
- Range: 3–5 TFLOPS → divide or multiply the hours by ~1.3.

Example, Qwen3-8B at 16K: 6 × 7.6e9 = 45.6 GFLOP. Attention 8 × 16384 × 4096 × 36 layers = 19.3 GFLOP.
Total 65 GFLOP/token → 4e12 / 65e9 ≈ **62 tok/s** → 16.4M tokens / 62 ≈ 264,000 s ≈ **74 h**.

### Time for 1,000 examples × 16K tokens (16.4M tokens), one SFT epoch, 32-core M1 Max [E]

| Model | tok/s (16K) | hours/epoch | days |
|---|---|---|---|
| Qwen3.5-4B / Gemma-4-E4B | ~130 | ~34–35 | ~1.5 |
| Qwen3-8B / Qwen3.5-9B | ~62–75 | ~61–74 | ~2.5–3 |
| Gemma-4-12B | ~54 | ~85 | ~3.5 (and memory-limited to 8K) |
| Qwen3-14B / Ministral-3-14B | ~36–40 | ~112–126 | ~5 |
| 30B-A3B MoE (Qwen3.6-35B-A3B, Nemotron Lightning) | ~100–110 | ~43–44 | ~2 (×0.5–2 uncertainty) |
| Qwen3.6/3.8-27B dense | ~24 | ~190 | ~8 |
| Gemma-4-31B | ~20 | ~230 | ~9.5 |

- 8K examples: about half the tokens, so roughly half the time (a little less, since attention shrinks).
- 32K: about 2× the time per example, plus 10–25% more for attention.
- **DPO** on full-length pairs: chosen + rejected (2×) + a reference forward (~0.3×) ≈ **2.3× the SFT
  time per pair** [E], plus the reference model's memory.
- Laptops throttle under multi-day load. A Mac Studio holds clocks better. This is unmeasured here.

---

## 6. Ready MLX weights and chat templates (checked 2026-10-23)

mlx-community has 4-bit / 8-bit / bf16 for: Qwen3.5 (0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B),
Qwen3.6 (27B, 35B-A3B), **Qwen3.8-27B** (2026-08-14, Apache-2.0, same `qwen3_5` architecture as
3.5/3.6-27B), Gemma 4 (E2B, E4B, 12B, 26B-A4B, 31B; also QAT 4-bit), NVIDIA-Nemotron-3.5-Lightning-30B-A3B
(4bit, 8bit, bf16, mxfp4/8), Nemotron-3-Nano-4B, gpt-oss-20b (MXFP4-Q8), Ministral-3 (3B/8B/14B,
Instruct, Reasoning and Base), Devstral-Small-2-24B, Granite 4.1 (3B/8B/30B), GLM-4.7-Flash, Qwen3
classic. Qwen3.8-Flash-Next is 180B, too big.

Licences: Qwen, Gemma 4, gpt-oss, Ministral, Granite are Apache-2.0. GLM-4.7-Flash is MIT. Nemotron 3.5 is **OpenMDW-1.1**.

Thinking in history (read from each `chat_template.jinja`) [Code]:
- **Qwen3.5:** keeps `<think>` only for assistant turns **after the last real user message** (tool
  responses do not count as user). No switch.
- **Qwen3.6:** same, but `preserve_thinking=true` keeps all thinking. **Default off.**
- **Qwen3.8:** `preserve_thinking` **defaults on** ("is undefined or … true").
- **Gemma 4:** thinking is rendered only after the last user turn, or when `preserve_thinking` and the
  turn has `tool_calls`. Default `enable_thinking=false`, `preserve_thinking=false`. `strip_thinking()` is
  applied to model content.
- **Nemotron 3.5 Lightning:** `truncate_history_thinking` defaults **True** (drops thinking before the
  last user message). `enable_thinking` defaults True.

What this means for training:
1. Render training text with **exactly the template and kwargs used at serving time**. Otherwise the model
   learns a format it never sees.
2. In a run with one user message and many tool turns, all thinking is kept → one training sequence is fine.
3. In a run where the user speaks again mid-run (DCLab's Home agent asks questions), earlier thinking is
   dropped from the history. Then the full rendering is **not** a prefix-extension of the earlier turns.
   **Split the run at each user message** (one example per segment, earlier thinking stripped as the
   template does), or turn `preserve_thinking` on at both training and serving (Qwen3.6/3.8, Gemma 4).
4. Neither trainer passes these kwargs, so pre-render (text format plus your own mask) or edit the template file.

---

## 7. When the Mac is too small: alternatives and costs

Prices change daily. Check before spending.

| Option | Price | What 1,000 × 16K tokens (16.4M) costs per epoch | Notes |
|---|---|---|---|
| **Tinker** (Thinking Machines) [docs table, 2026-10-23] | per 1M train tokens: Nemotron-3.5-Lightning-30B-A3B $0.44; GPT-OSS-20B $0.396; Qwen3-8B $0.44 (32K max); Qwen3.5-4B $0.737; Qwen3.6-35B-A3B $1.177; Qwen3.5-9B $1.463; Qwen3.8-27B $4.103 (64K tiers; 256K tiers ~1.8×) | Nemotron $7.2 · gpt-oss-20b $6.5 · Qwen3.5-4B $12.1 · **Qwen3.6-35B-A3B $19.3** · Qwen3.5-9B $24.0 · Qwen3.8-27B $67.3 [E = 16.4 × price] | LoRA training ("Model-agnostic LoRA training … dense and MoE"). SFT, RL, DPO guides. Checkpoint export pages exist ("Merge into a HuggingFace Model", "Build a PEFT LoRA Adapter"). Qwen3.6-27B and Nemotron-3-Nano retire 2026-10-23. Not verified: whether the train meter counts masked prompt tokens (assumed yes). |
| **Rented H100 80GB + Unsloth** | ~$2.0–3.3/h on RunPod/Lambda, ~$1.5–2.2 on Vast [C, aggregator sites] | dense 8–9B: ~1 h → ~$2–4 · 30B-A3B MoE: ~1–3 h → ~$3–10 · 27B dense: ~3 h → ~$6–10 [E] | Arithmetic: H100 BF16 dense ≈ 989 TFLOPS spec; at ~30% effective ≈ 300 TFLOPS → 8B at 16K ≈ 300e12 / 55e9 ≈ 5,400 tok/s → 16.4M in ~50 min. MoE efficiency is lower; I assumed 3–4× lower. Unsloth measured: gpt-oss up to "81K context length with QLoRA and 60K context with BF16 LoRA" on one H100 [M by Unsloth]. Unsloth bf16 LoRA VRAM for Qwen3.5: 4B 10 GB, 9B 22 GB, 27B 56 GB, 35B-A3B 74 GB (at short context) [M by Unsloth]. |
| Rented A100 80GB | ~$1.2–2.8/h [C] | roughly 2–3× slower than H100 → similar total cost | Fine for ≤14B at 16K. |
| RTX 5090 32 GB | ~$0.45–0.95/h [C, one source] | — | Same memory as a 32 GB Mac, but CUDA kernels. Good for 4–9B. |

**Honest comparison:** a 64 GB M1 Max spends **~2 days** per epoch that Tinker does for **~$7–20** and an
H100 does in **~1–3 hours**. Hyper-parameter search (3–5 runs) and DPO make the gap bigger.

---

## 8. Recommendation

**32 GB M1 Max**
- Largest *practical*: **8–9B dense with QLoRA at 8K** (est. 16–18 GB). Qwen3-8B (best-supported path)
  or Qwen3.5-9B (newer, mlx-lm-lora only).
- 16K: only a **4B** (Qwen3.5-4B est. ~21 GB, or Qwen3-4B). An 8B at 16K (~25 GB) needs a raised wired limit.
- Time: 4B ~1.5 days per 1,000 × 16K epoch. 8B at 8K ~1.5 days per 1,000 × 8K.
- DPO: only 4B (the reference copy plus two sequences).

**64 GB M1 Max (always the 32-core GPU)**
- Largest *practical*: **30B-A3B MoE with QLoRA at 8K–16K** (Qwen3.6-35B-A3B est. 33/44 GB;
  Qwen3-30B-A3B-2507 similar and on the better-tested path; Nemotron 3.5 Lightning est. 34/49 GB,
  unverified on MLX). Restrict LoRA to attention and shared experts.
- Also **14B dense at 16K** (~35 GB, ~5 days/epoch) or **8–9B at 32K** (~43–48 GB).
- 27B/31B dense fit at 8K but take ~8–10 days per epoch. Not practical.
- DPO on a 30B MoE does not fit with mlx-lm-lora's reference copy (+18–20 GB). Use ORPO/CPO, or precompute reference log-probs.

**Sweet spot (quality vs time)**
- On the Mac: **a 4B for development and pilots**, and at most **an 8–9B for a real run**. Agent quality
  on SWE-bench-type tasks rises a lot from 4B to 30B-A3B: the Nemotron card's harness puts Qwen3.6-35B-A3B
  at 70.12 SWE-bench Verified vs gpt-oss-20b at 52.44 [M by NVIDIA]. But the Mac cannot iterate on a 30B fast enough.
- For the model DCLab will actually ship: **rent or use Tinker** for a 30B-A3B MoE (Qwen3.6-35B-A3B, or
  Nemotron 3.5 Lightning if the licence suits). Bring the adapter back to the Mac for **inference**
  (4-bit 30B-A3B runs well there).
- Avoid on Apple silicon for long agent runs: Gemma 4 (sliding-window fallback) and gpt-oss (sliding plus sinks).

**Catastrophic forgetting**
- LoRA (not full) at a low learning rate (~1e-5 to 1e-4). Rank 16–32. Few epochs.
- Mix in replay: 20–30% general instruction and tool-use data from the base model's own domain. Unsloth
  advises keeping "a minimum of 75% reasoning" examples to preserve Qwen3.5's thinking.
- Before and after each run, evaluate on held-out general benchmarks and on DCLab's sealed sets.
- Keep adapters separate (do not fuse) so you can roll back.

---

## 9. First things to measure (before any decision)

1. On the target Mac, a **100-step pilot** of Qwen3.5-4B-4bit and Qwen3-4B-4bit at 8K and 16K
   (mlx-lm-lora, `--grad-checkpoint --sft-loss-type chunked_nll`). Log `Tokens/sec` and `Peak mem`.
   This replaces sections 4–5 with real numbers.
2. Write the per-turn mask loss and check it: loss on tool outputs and user text must be 0, and loss on
   every assistant turn (thinking + tool call) must count.
3. Compare the loss curve of a 100-example run on the Mac vs Tinker on the same data. That checks MLX
   4-bit QLoRA quality against bf16 LoRA (Unsloth's Qwen3.5 QLoRA warning).
4. Only then pick the size.

---

## 10. Sources

Code read at `main`, 2026-10-23:
- mlx-lm `mlx_lm/LORA.md`, `tuner/datasets.py`, `tuner/trainer.py`, `tuner/utils.py`, `tuner/lora.py`, `lora.py`, `models/{qwen3_5,qwen3_5_moe,qwen3_next,gemma4_text,gpt_oss,nemotron_h,ssm,base}.py`, `utils.py` — https://github.com/ml-explore/mlx-lm
- MLX `mlx/fast.cpp`, `mlx/backend/metal/scaled_dot_product_attention.cpp`, `python/src/fast.cpp` — https://github.com/ml-explore/mlx
- mlx-lm-lora README, `trainer/{datasets,sft_trainer,long_context,fast_vjp}.py`, `recurrent_patch.py`, `train.py` — https://github.com/Goekdeniz-Guelmez/mlx-lm-lora
- PyPI versions: mlx 0.32.4 (2026-10-14), mlx-lm 0.32.2 (2026-10-16), mlx-lm-lora 5.10.0 (2026-10-23)

MLX PRs / issues:
- SDPA VJP https://github.com/ml-explore/mlx/pull/4563 · gated delta VJP https://github.com/ml-explore/mlx/pull/4565 · D512 memory https://github.com/ml-explore/mlx/pull/4518
- mlx-lm: #1956 https://github.com/ml-explore/mlx-lm/issues/1956 · #1870 https://github.com/ml-explore/mlx-lm/pull/1870 · #1983 https://github.com/ml-explore/mlx-lm/pull/1983 · #47 https://github.com/ml-explore/mlx-lm/pull/47 · #1658 https://github.com/ml-explore/mlx-lm/issues/1658 · #1185 https://github.com/ml-explore/mlx-lm/issues/1185 · #1206 https://github.com/ml-explore/mlx-lm/issues/1206 · #1348 https://github.com/ml-explore/mlx-lm/issues/1348 · #1786/#1787 (Qwen3 MoE) https://github.com/ml-explore/mlx-lm/issues/1786 · #1814/#1843 (gpt-oss) https://github.com/ml-explore/mlx-lm/pull/1843 · #1897 (Gemma 4 MoE) https://github.com/ml-explore/mlx-lm/issues/1897 · #1363 (Gemma 4 PLE targets) https://github.com/ml-explore/mlx-lm/pull/1363 · #1172 (gpt-oss fuse) https://github.com/ml-explore/mlx-lm/issues/1172
- mlx-lm-lora #84 (long-context bug) https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/issues/84 · #102 https://github.com/Goekdeniz-Guelmez/mlx-lm-lora/pull/102

Benchmarks / docs:
- llama.cpp Apple silicon table https://github.com/ggml-org/llama.cpp/discussions/4167
- Model configs and chat templates: Hugging Face `Qwen/Qwen3.5-*`, `Qwen/Qwen3.6-*`, `Qwen/Qwen3.8-27B`, `google/gemma-4-*`, `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` (card: benchmarks, OpenMDW-1.1, release 2026-08-11), `openai/gpt-oss-20b`, `mistralai/Ministral-3-14B-Instruct-2512`, `ibm-granite/granite-4.1-*`, `zai-org/GLM-4.7-Flash`; mlx-community repo file sizes via the HF API
- Tinker models & pricing https://tinker-docs.thinkingmachines.ai/tinker/models/ · overview https://tinker-docs.thinkingmachines.ai/
- Unsloth Qwen3.5 fine-tune https://unsloth.ai/docs/models/qwen3.5/fine-tune · gpt-oss long context https://unsloth.ai/docs/models/gpt-oss-how-to-run-and-fine-tune/long-context-gpt-oss-training
- GPU prices (aggregators, unverified): https://intuitionlabs.ai/articles/h100-rental-prices-cloud-comparison · https://tech-insider.org/runpod-vs-lambda-vs-vast-ai-2026/ · https://www.promptquorum.com/local-llms/cloud-gpu-rental-comparison-2026
- macOS GPU memory cap: https://developer.apple.com/forums/thread/732035 · https://github.com/ivanopcode/devnote-override-macos-metal-vram-cap

Scratch scripts used for the estimates: `est.py`, `cfg.py`, `sizes.py` in this session's scratchpad (not kept).
