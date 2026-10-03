# `research/llm-fine-tuning/experiments/sft/` — SLM roadmap pack and SFT corpus builder

This folder is an **independent review + SFT (supervised fine-tuning) dataset builder**.
Nothing here deletes Research Studio archives or campaign results.

## What is in this folder

| Path | Role |
|---|---|
| `DCLab_RD_Review_and_Roadmap.md` / `.pdf` | Plain-language review of DCLab R&D and the small-language-model roadmap |
| `Claude_SFT_sample.pdf` | Annotated sample of the SFT format |
| `build_sft_dataset.py` | Builds chat-format `sft_train.jsonl` / `sft_val.jsonl` |
| `samples/sft_train_sample.jsonl`, `samples/sft_val_sample.jsonl` | Frozen **sample** from the first 104-example build (rules + workflows + campaign claims) |
| `out/` | Latest committed build (`sft_train.jsonl`, `sft_val.jsonl`, `MANIFEST.json`) |

## Two complementary corpora

```
A — Knowledge / campaign (this pack's original job)
    evidence/knowledge/model_building_rules.jsonl
    evidence/knowledge/workflow_blocks.json
    evidence/campaigns/model_building_50_v1/agent_memory.jsonl
    → "what DCLab believes / found in the 50-exp campaign"

B — Agentic Research Studio (additive)
    agent_runs/clean_exports/sft_chat.jsonl
    → "how the live agent plans / proposes / critiques / synthesizes"
    Refresh:  make agent-export-clean
```

Use **both** for SLM fine-tuning. Keep raw `trial-*/result.json` and `oof_predictions.jsonl`
for analysis. They are **not** dumped into SFT text.

## Build the combined corpus

From the repository root:

```bash
# 1) Refresh Studio clean + SFT traces (optional but recommended)
make agent-export-clean

# 2) Merge knowledge + studio into one train/val split (writes research/llm-fine-tuning/experiments/sft/out/)
make sft-build
# equivalent: .venv/bin/python research/llm-fine-tuning/experiments/sft/build_sft_dataset.py --out-dir research/llm-fine-tuning/experiments/sft/out
```

Knowledge-only (old behaviour):

```bash
.venv/bin/python research/llm-fine-tuning/experiments/sft/build_sft_dataset.py --no-studio --out-dir research/llm-fine-tuning/experiments/sft/out
```

Outputs land in `research/llm-fine-tuning/experiments/sft/out/`: `sft_train.jsonl`, `sft_val.jsonl`, `MANIFEST.json`.

## Reading order for humans

1. `DCLab_RD_Review_and_Roadmap.md` — why this corpus exists
2. `agent_runs/clean_exports/README.md` — Studio clean layer (generated locally)
3. `<run_id>/clean/run_card.md` — one research run at a glance
4. This README — how to merge for fine-tuning
