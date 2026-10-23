# Fine-tuning an open-source decision model (package 22.5)

A small open-source model learns to answer DCLab's decisions from the exported decision sets, then answers through the
same interface as the rules and the classifiers (`dclab_core.answerers.ModelAnswerer`, the gateway's `decision_answer`
purpose) and is compared with them on the same decisions. Which model, and on which GPU, is the owner's decision
(`docs/guides/ROADMAP.md`, "Decisions waiting on the owner"): nothing here downloads a model or runs on its own.

## 1. The data

```bash
make decisions-gen       # the generated decision sets (22.1), if they are not here
make decisions-export    # export_v2/ (every point with a set, 29.9): train.chat.jsonl, val.chat.jsonl (dev), README.md (dataset card), tabular.parquet, typed.jsonl
```

The sealed partition is never exported: a tuned model must not see it.

## 2. A LoRA run

On a CUDA GPU (owned or rented), with `pip install "trl>=0.12" "peft>=0.13" "transformers>=4.46" datasets accelerate`:

```bash
python research/llm-fine-tuning/experiments/sft/train_lora.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir research/llm-fine-tuning/experiments/decisions/export_v2 \
  --out research/llm-fine-tuning/experiments/decisions/finetune/runs/qwen2.5-1.5b-r16 \
  --rank 16 --epochs 2 --max-length 1024
```

On Apple silicon, `mlx-lm` takes the same chat files (`mlx_lm.lora --model <a local MLX model> --train --data
research/llm-fine-tuning/experiments/decisions/export_v2`; MLX reads `train.jsonl` and `valid.jsonl`, so link or copy
the two chat files under those names first).

Record each run as `runs/<name>.json`: the base model and its revision, the data's `MANIFEST.json` counts, the
hyper-parameters, the wall time, the hardware and the training and validation loss: a run nobody can repeat is not
evidence.

## 3. Serve it and compare it

Serve the adapter behind an OpenAI-compatible endpoint (vLLM, llama.cpp's server, Ollama with an imported model) and
point the tuned tier at it (`DCLAB_TIER_TUNED_BASE_URL`, `DCLAB_TIER_TUNED_MODEL`), then:

```bash
python -m dclab_rnd.decision_model_compare --tuned --per-point 200 --write
```

That compares it with the recorded rules and the fitted classifiers on the dev partition (`evidence/campaigns/
decision_llm_v1`), asking it exactly as it was trained (`decision_export.request` and the `decision_answer` prompt). A
remote tuned endpoint is asked only with `--cap-eur`; a local one costs nothing.

It is used nowhere until it passes a promotion gate on a sealed set it never saw. That is not possible yet: every
sealed set of `generated_v1` was read once by 22.3's gate, and the gate runner takes fitted classifiers only. A sealed
verdict for a tuned model needs a new generated set (v2) and a runner that takes any `DecisionAnswerer`: both open.

## 4. The agent sets, in the model's own chat shape (D2 / P5, 2026-10-23)

The M4 and M5 sets export as chat JSONL for tool-calling SFT: `tools` (the schemas the model was offered), `messages`
with `reasoning_content` on every assistant turn whose reasoning was recorded (never a visible, a rule's or a
teacher's), tool calls with their arguments as objects, `tool_call_id` on tool turns; a sealed row is never exported
and silver_frontier is refused by its manifest:

```bash
python -m dclab_rnd.store.export_chat --model-type m5_solver_llm --folder trajectories/reasoning --out sft/out_m5
```

Before any training run, render the export with the model's own chat template and the same setting the gateway sends
at run time (`DCLAB_TIER_<TIER>_CHAT_TEMPLATE_KWARGS`, Qwen's `preserve_thinking`): every earlier turn's reasoning and
every tool call must survive the rendering, or the model learns one habit and is used with another:

```bash
python -m dclab_rnd.store.template_check sft/out_m5/train.jsonl --tokenizer ~/models/Qwen3.8-27B --kwargs '{"preserve_thinking": true}'
```

The export's MANIFEST.json says how many lines call tools without a `tools` list (a set that kept no schemas teaches a
model to call tools it was never shown), where the set's files disagree with its manifest, and its longest line.

Then train every assistant turn and only them: the stock MLX trainer trains every token (user and tool turns included)
unless `--mask-prompt` is given, and with it only the last message, so a per-turn loss mask is a custom piece (about 60
lines; `research_notes/open_models_2026-10/m1max_finetuning.md`, the fine-tuning notes of 2026-10-23). Raise
`max_seq_length` above the export's longest line: its default of 2048 tokens cuts silently.
