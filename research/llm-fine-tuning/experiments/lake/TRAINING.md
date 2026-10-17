# Training DCLab's own models from the lake (the owner's runbook)

Phases 4 and 5 (`docs/guides/PHASES_4_5_MODELS_PLAN.md`). DCLab prepares the data, the files and the checks; the
training run is yours, on a GPU (your decision of 2026-10-16). Nothing here downloads a model or spends money by itself.

## 1. Build the data

```bash
python -m dclab_rnd.lake openml          # 26.2: real tables from OpenML (resumable; only bands kept)
python -m dclab_rnd.lake selfplay        # 27.3: invented problems solved and verified by the engine
make datasets                            # each workspace's sets under its train consent (nothing while consent is off)
make lake                                # bronze, silver, gold; the committed MANIFEST.json beside this file
python -m dclab_rnd.lake export --target decisions
python -m dclab_rnd.lake export --target solver --weights leakage_judgment=2,split_design=2
```

The files land in `lake/export/<target>/` (not committed): `train.chat.jsonl`, `val.chat.jsonl`, the same as `train.jsonl`
and `valid.jsonl` for MLX, and `MANIFEST.json` (counts by set and origin, the mix weights, what was refused). A sealed row
and an evaluation-only row (Claude's answers, DCLab's build history) are never in them.

## 2. Two models, two runs

| | The decision model (26.4) | The ML-solver model (27.7) |
|---|---|---|
| What it learns | every typed decision: a column's role, identifier, leak, the task, the outcome column | the solver's task families: problem type, leakage judgment, split design, algorithm choice, reading results, … |
| Base (open weights, commercial use allowed) | 1.5B to 4B instruction model (Qwen2.5 or Qwen3 class) | 7B to 8B for quality; 3B to 4B to start |
| Hardware | one GPU of 16 to 24 GB, or a 32 GB Mac with MLX | one GPU of 24 GB (LoRA), 48 GB+ for a full 8B fine-tune |
| Recipe | LoRA rank 16, 2 epochs, max length 1,024 | LoRA rank 32, 2 to 3 epochs, max length 4,096 |

On a CUDA GPU:

```bash
pip install "trl>=0.12" "peft>=0.13" "transformers>=4.46" datasets accelerate
python research/llm-fine-tuning/experiments/sft/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct \
  --data-dir lake/export/decisions --out lake/runs/decisions-qwen1.5b-r16 --rank 16 --epochs 2 --max-length 1024
python research/llm-fine-tuning/experiments/sft/train_lora.py --model Qwen/Qwen2.5-7B-Instruct \
  --data-dir lake/export/solver --out lake/runs/solver-qwen7b-r32 --rank 32 --epochs 3 --max-length 4096
```

On Apple silicon: `mlx_lm.lora --model <a local MLX model> --train --data lake/export/<target>`.

Record every run as `lake/runs/<name>.json`: the base model and its revision, the export's `MANIFEST.json`, the
hyper-parameters, the hardware, the wall time, and the training and validation loss. A run nobody can repeat is not
evidence.

## 3. Judge it before it serves

1. Serve the adapter behind an OpenAI-compatible endpoint (vLLM, llama.cpp's server, Ollama) and point the tuned tier at
   it (`DCLAB_TIER_TUNED_BASE_URL`, `DCLAB_TIER_TUNED_MODEL`).
2. The decision model: `python -m dclab_rnd.decision_model_compare --tuned --per-point 200 --write` compares it with the rules
   and the classifiers on the dev partition, asked exactly as it was trained.
3. Then the sealed sets, once. The generated sets' sealed partitions are spent (read by the classifiers' gate in 22.3);
   the fresh sealed sets are OpenML's and self-play's, by group.
4. It serves only through the switches that exist (22.6 for a decision point, 21.4 for a role), in shadow first, and only
   where it measured better on the same cases.
5. A role, step by step (27.8): `python -m dclab_rnd.models.tuned shadow reporter` (it answers beside the serving model,
   logged, never used), `... plan reporter` (what is still missing), `make harness-eval` with it serving the role, then
   `... promote reporter --reason "..." [--measured dclab-solver=0.87,claude-sonnet-5-5=0.83]` (an understanding role
   needs the measurement); `... retire reporter --reason "..."` puts the earlier model back.

## 4. What it may not claim

A model trained here is evidence about these sources: invented tables, OpenML's public tables, verified invented
problems and the R&D's own records. It is not production-ready because it scores well on them; real tables and
blind labels from people decide that (26.2, decision 4 of the plan).
