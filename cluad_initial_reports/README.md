# `cluad_initial_reports/` — SLM roadmap pack (kept) + how it fits Studio cleans

This folder is an **independent review + first SFT builder**. Nothing here deletes Research Studio archives or campaign results.

## What’s in this folder

| File | Role |
|---|---|
| `DCLab_RD_Review_and_Roadmap.md` | Plain-language review of DCLab R&D and SLM roadmap |
| `build_sft_dataset.py` | Builds chat-format `sft_train.jsonl` / `sft_val.jsonl` |
| `sft_train_sample.jsonl` / `sft_val_sample.jsonl` | Frozen **sample** from the first 104-example build (rules + workflows + campaign claims) |

## Two complementary corpora

```
A — Knowledge / campaign (this pack’s original job)
    rules + workflow blocks + agent_memory claims
    → “what DCLab believes / found in the 50-exp campaign”

B — Agentic Research Studio (additive)
    agent_runs/clean_exports/sft_chat.jsonl
    → “how the live agent plans / proposes / critiques / synthesizes”
    Refresh:  make agent-export-clean
```

Use **both** for SLM fine-tuning. Keep raw `trial-*/result.json` and `oof_predictions.jsonl` for analysis — they are **not** dumped into SFT text.

## Build the combined corpus

From repo root:

```bash
# 1) Refresh Studio clean + SFT traces (optional but recommended)
.venv-agent/bin/python -m dclab_rnd.agentic export-clean

# 2) Merge knowledge + studio into one train/val split
.venv/bin/python cluad_initial_reports/build_sft_dataset.py --out-dir sft_out
```

Knowledge-only (old behavior):

```bash
.venv/bin/python cluad_initial_reports/build_sft_dataset.py --no-studio --out-dir sft_out
```

Outputs land in `sft_out/` (gitignored if you prefer): `sft_train.jsonl`, `sft_val.jsonl`, `MANIFEST.json`.

## Reading order for humans

1. `DCLab_RD_Review_and_Roadmap.md` — why this corpus exists  
2. `agent_runs/clean_exports/README.md` — Studio clean layer  
3. `<run_id>/clean/run_card.md` — one research run at a glance  
4. This README — how to merge for fine-tuning  
