# Training the DCLab workflow model on the M1 Max

What this produces: a LoRA adapter for **Qwen3.5-9B** that runs the DCLab workflow WF-01…WF-10 with its own
`<think>` reasoning, keeps Qwen's general coding and reasoning, and is chosen by measured results, not by loss.

Every command runs from `research/llm-fine-tuning/experiments/sft_factory/`. Every long step can be stopped and
started again: it continues where it stopped.

## Why these choices

| Choice | Why |
|---|---|
| Qwen3.5-9B | The strongest Qwen model of this size (MMLU-Pro 82.5, GPQA Diamond 81.7 on its model card), thinking mode by default, 262k context. Its hybrid architecture (3 of every 4 layers are linear attention) makes long inputs cheap. |
| LoRA, all layers, rank 16 | Learns the formats and rules; forgets far less than full fine-tuning (which also cannot fit). |
| 8-bit or 16-bit base, not 4-bit | No rounding loss worth mentioning; 4-bit is the last resort. |
| Qwen writes its own `<think>` | The model keeps its own way of reasoning; a trace is kept only if it reaches the right answer with no invented numbers. |
| One answer per training item, earlier answers without thinking | Exactly how the model is used (Qwen's rule: no thinking in the history). |
| Replay (Qwen's own correct answers to coding and ML-arithmetic tasks) | Guards general skill while the workflow is learned. |
| Chunked gated delta rule (`gated_delta_chunked.py`) | MLX trains Qwen3.5's linear attention token by token by default, which needs 20+ GB per layer for a 12k-token item; the chunked form needs about 64× less and is tested equal (values, state, gradients). |
| Choose the epoch with `eval_mlx.py --pick` | The best decision accuracy among checkpoints that keep general skill and honesty at the base model's level. |

## 0. Set up (once)

```bash
python3 -m venv ~/.venvs/mlx && source ~/.venvs/mlx/bin/activate
pip install -U mlx mlx-lm pyyaml numpy
sudo sysctl iogpu.wired_limit_mb=58000      # lets the GPU use 58 GB; resets on reboot
```

Plug the Mac in, close heavy apps, and keep it awake during long steps: run them under `caffeinate -dims`.

## 0b. Build the training items (a minute, standard library only)

```bash
python build_training_mix.py     # out/all_stages → out/mix/items_*.jsonl (one item per answer, weights, trace plan)
python replay_tasks.py           # out/mix/replay_tasks.jsonl (checkable coding and ML-arithmetic tasks)
```

## 1. Check the mechanics on a tiny model (15 minutes)

A random tiny model with Qwen3.5's architecture and chat template; its answers are noise, only the pipeline is tested.

```bash
python mac/gated_delta_chunked.py                         # the chunked rule equals MLX's own (prints ok)
python mac/make_tiny_model.py --out models/tiny-qwen35
python mac/train_mlx.py --config mac/config.yaml --set model=models/tiny-qwen35 adapter_path=adapters/tiny epochs=1 max_seq_length=4096 --stop-after 20
python mac/train_mlx.py --config mac/config.yaml --set model=models/tiny-qwen35 adapter_path=adapters/tiny epochs=1 max_seq_length=4096 --resume --stop-after 40
```

Step 1 needs `out/mix/final_train.jsonl`; if traces are not made yet, run `python mac/export_mlx.py` first (every
item is then non-thinking, which is fine for this check).

## 2. Get the model (about 20 GB download)

```bash
mlx_lm.convert --hf-path Qwen/Qwen3.5-9B --mlx-path models/qwen35-9b-bf16                 # 16-bit, ~19 GB
mlx_lm.convert --hf-path Qwen/Qwen3.5-9B --mlx-path models/qwen35-9b-8bit -q --q-bits 8   # 8-bit, ~10 GB
```

## 3. Measure memory and speed before the long runs (30 minutes)

```bash
python mac/train_mlx.py --config mac/config.yaml --set model=models/qwen35-9b-bf16 --bench 2048 4096 8192 12288 16384
python mac/train_mlx.py --config mac/config.yaml --set model=models/qwen35-9b-8bit --bench 2048 4096 8192 12288 16384
```

Pick the precision whose peak memory stays under about 52 GB at 16384 (or at the longest length you keep), and the
faster of the two if both fit. If neither fits at 16384, set `max_seq_length` to the longest length that fits: the
first line of training says how many items that skips.

On an M1, 16-bit (bfloat16) math may be slower than on newer chips; if `bf16` is much slower, use the 8-bit model.

## 4. Baseline: how good is Qwen3.5-9B before training (2–4 hours)

```bash
python mac/export_mlx.py
caffeinate -dims python mac/eval_mlx.py --model models/qwen35-9b-8bit --split val --out eval/base_val.json
```

## 5. Qwen writes the reasoning (`<think>`) — the longest step (roughly 10–30 hours, resumable)

```bash
caffeinate -dims python mac/make_think_traces.py --model models/qwen35-9b-8bit --splits train val
caffeinate -dims python mac/make_replay.py --model models/qwen35-9b-8bit
python mac/export_mlx.py
```

Start with `--limit 20` and read `out/mix/traces.jsonl`: the reasoning should sound like a careful engineer, cite
only shown numbers, and never say it was given the answer. `final_report.json` counts how many items got a sampled
trace, a rationalized one, or none (those train without thinking).

## 6. Train (roughly 25–45 hours for 3 epochs, resumable)

```bash
caffeinate -dims python mac/train_mlx.py --config mac/config.yaml
# stopped? the same command with --resume continues from the last checkpoint (every 100 steps)
caffeinate -dims python mac/train_mlx.py --config mac/config.yaml --resume
```

Watch `adapters/dclab-qwen35-9b/train_log.jsonl`: training loss should fall, validation loss on `workflow` should
fall, and on `replay` it should stay flat. A rising replay loss means general skill is drifting: lower the learning
rate to 5e-5 or add replay data.

## 7. Choose the epoch, then test once

```bash
for e in 1 2 3; do caffeinate -dims python mac/eval_mlx.py --model models/qwen35-9b-8bit --adapter adapters/dclab-qwen35-9b/epoch_$e --split val --out eval/e${e}_val.json; done
python mac/eval_mlx.py --compare eval/base_val.json eval/e1_val.json eval/e2_val.json eval/e3_val.json --pick
caffeinate -dims python mac/eval_mlx.py --model models/qwen35-9b-8bit --adapter adapters/dclab-qwen35-9b/epoch_2 --split test --out eval/final_test.json
```

The test set (online shoppers, German credit and two replicas the model never saw) is scored once, for the chosen
epoch only. Never tune anything on it.

## 8. Use it

```bash
mlx_lm.fuse --model models/qwen35-9b-8bit --adapter-path adapters/dclab-qwen35-9b/epoch_2 --save-path models/dclab-qwen35-9b
mlx_lm.server --model models/dclab-qwen35-9b      # an OpenAI-compatible API on localhost:8080
```

Use the same system prompt as in training (`SYSTEM_PROMPT` in `leakage_audit_samples.py`), thinking on, and the
sampling Qwen recommends for thinking mode (temperature 0.6, top_p 0.95, top_k 20).

## Times are estimates

All hours above are estimates for an M1 Max; step 3 measures the real speed. Every long step saves its progress,
so a stop costs minutes, not a day.
