# Fine-tuning an open-source decision model (package 22.5)

A small open-source model learns to answer DCLab's decisions from the exported decision sets, then answers through the
same interface as the rules and the classifiers (`dclab_core.answerers.ModelAnswerer`, the gateway's `decision_answer`
purpose) and is compared with them on the same decisions. Which model, and on which GPU, is the owner's decision
(`docs/guides/ROADMAP.md`, "Decisions waiting on the owner"): nothing here downloads a model or runs on its own.

## 1. The data

```bash
make decisions-gen       # the generated decision sets (22.1), if they are not here
make decisions-export    # export_v1/: train.chat.jsonl, val.chat.jsonl (dev), README.md (dataset card), tabular.parquet, typed.jsonl
```

The sealed partition is never exported: a tuned model must not see it.

## 2. A LoRA run

On a CUDA GPU (owned or rented), with `pip install "trl>=0.12" "peft>=0.13" "transformers>=4.46" datasets accelerate`:

```bash
python research/llm-fine-tuning/experiments/sft/train_lora.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir research/llm-fine-tuning/experiments/decisions/export_v1 \
  --out research/llm-fine-tuning/experiments/decisions/finetune/runs/qwen2.5-1.5b-r16 \
  --rank 16 --epochs 2 --max-length 1024
```

On Apple silicon, `mlx-lm` takes the same chat files (`mlx_lm.lora --model <a local MLX model> --train --data
research/llm-fine-tuning/experiments/decisions/export_v1`; MLX reads `train.jsonl` and `valid.jsonl`, so link or copy
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
