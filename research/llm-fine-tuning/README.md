# LLM fine-tuning (a DCLab small language model)

**Status:** active · data ready, no training run yet

## Research question

Can a small open model, fine-tuned on DCLab's evidence, reason about model building as well as a large hosted model that is given the same retrieved evidence, at lower cost and latency?

## What is here

`sft/` contains:

| File | Purpose |
|---|---|
| `build_sft_dataset_v3.py` | Builds self-contained, evidence-grounded (RAFT-style) examples from every campaign. It gates out critic claims the numbers disprove and holds out whole datasets for validation |
| `out_v3/` | 326 examples in chat, Alpaca and ShareGPT formats, plus `MANIFEST.json` |
| `train_lora.py` | LoRA fine-tuning with Hugging Face TRL |
| `eval_sft.py` | Deterministic scoring on held-out datasets (no LLM judge) |
| `build_sft_dataset.py`, `out/`, `samples/` | The first (v1) corpus, kept for comparison |
| `DCLab_RD_Review_and_Roadmap.*` | The original small-model roadmap review |

The full method is in [`docs/guides/SFT_DATA_GUIDE.md`](../../docs/guides/SFT_DATA_GUIDE.md).

## How to run

```bash
make sft-v3                      # rebuild the corpus from current evidence
python research/llm-fine-tuning/sft/eval_sft.py score --reference   # sanity check: 1.0

# On a GPU machine
pip install "trl>=0.12" "peft>=0.13" "transformers>=4.46" datasets accelerate
python research/llm-fine-tuning/sft/train_lora.py --model Qwen/Qwen2.5-1.5B-Instruct --out research/llm-fine-tuning/sft/runs/qwen15b
python research/llm-fine-tuning/sft/eval_sft.py generate --model Qwen/Qwen2.5-1.5B-Instruct --adapter research/llm-fine-tuning/sft/runs/qwen15b --out preds.jsonl
python research/llm-fine-tuning/sft/eval_sft.py score --predictions preds.jsonl
```

## Next experiments

| ID | Question |
|---|---|
| SLM-001 | Score the base model (no fine-tuning) on the held-out datasets |
| SLM-002 | Fine-tune with LoRA and compare against SLM-001 on the same held-out datasets |
| SLM-003 | Compare the fine-tuned small model against a large hosted model given the same retrieved records |

## Evidence, retrieval and SLM learning: charter

**Status: SFT preparation exists; retrieval quality and training outcomes are not yet established.** This track connects R&D outputs to DCLab's assistant. Its central rule is that a proposal, an observed result, an interpretation, and a reusable principle are different kinds of records.

### Research questions

- Which evidence record is relevant to a question about dataset, workflow stage, or symptom?
- Can the assistant show the evidence, scope, counterexample, and uncertainty behind a recommendation?
- Which records are verified enough for retrieval or eventual training?
- Does retrieved context improve answers over no retrieval and metadata-filtered baselines?

### Existing material

- Rules, workflow blocks, and campaign records: [`../../evidence/knowledge/`](../../evidence/knowledge/) and [`../../evidence/campaigns/model_building_50_v1/agent_memory.jsonl`](../../evidence/campaigns/model_building_50_v1/agent_memory.jsonl)
- Experiment-level claim memory: [`../../evidence/campaigns/model_building_50_v1/agent_memory.jsonl`](../../evidence/campaigns/model_building_50_v1/agent_memory.jsonl)
- SFT preparation: [`../llm-fine-tuning/sft/README.md`](../llm-fine-tuning/sft/README.md) and [`../llm-fine-tuning/sft/build_sft_dataset.py`](../llm-fine-tuning/sft/build_sft_dataset.py)

### Suggested experiment

First compare a base model with no context, exact metadata-filtered evidence, and retrieved evidence on questions with known evidence IDs. Score citation correctness, answer correctness, scope fidelity, and unsupported-claim rate. Keep rejected and negative evidence retrievable. Only use reviewer-approved examples for training; split by dataset/source and workflow to prevent near-duplicate leakage.

For local commands, see the [main README](../../README.md#how-to-use-it). Retrieval or fine-tuning should not be described as reliable until evaluated against a fixed benchmark.

## Focused workflow model: related track

A small model that follows one narrow workflow deeply is researched in its own track: [workflow-model](../workflow-model/README.md). It uses the corpus and evaluator in this folder.
