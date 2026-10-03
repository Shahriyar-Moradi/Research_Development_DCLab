# CLAUDE.md — working in the DCLab R&D repository

This file gives every Claude Code session (cloud or local) the context it needs. Keep it short and current.

## What this repository is

The research arm of DCLab. It measures how to build ML models that are *right* (leakage-safe, honestly evaluated) and turns the evidence into tools: an evidence index, a notebook copilot, agent tools, and SFT data for a future small model. The owner prefers answers in clear English **and** Persian.

## Map

- `data/` — all datasets: `project/` (HyperAck, Telco), `public/` (10 UCI parquet), `downloads/` (gitignored).
- `research/<track>/` — one folder per research idea; each has a README with question, prediction contract, how to run, conclusions. Index: `research/README.md`. New idea: `make new-track NAME=... TITLE="..." PREFIX=...`.
- `evidence/campaigns/` — immutable experiment results (one JSON each). `evidence/knowledge/` — GENERATED; never edit by hand.
- `dclab_rnd/` — shared package: control plane (`python -m dclab_rnd`), `evidence_index.py`, `critic_gate.py`, `pitfalls.py`, `tools.py`, `copilot/`, `expansion/`, `agentic/` (Research Studio).
- `general_pipeline/` — shared tabular pipeline imported by research scripts and `dclab_rnd`.
- `docs/guides/` — field notes, integration plan, agent architecture, copilot, SFT guide.

## Commands

```bash
make help        # every target
make rd-check    # tests + evidence validation + freshness of all generated files (~10 s); must pass before committing
make rd-sync     # rebuild the registry and knowledge base after new results
make knowledge   # rebuild the evidence index and SFT v3 corpus after new results
python -m dclab_rnd.copilot review NOTEBOOK.ipynb   # methodology review of a notebook
```

Environments: `.venv` from `requirements/base.txt` (ML work); `.venv-agent` (Python 3.12/3.13) from `requirements/agent.lock.txt` for the Research Studio; CI uses `requirements/ci.txt`.

## Rules

1. **Leakage is the top risk.** Never use information unavailable at the prediction moment (`final_customer_fare`, `final_biker_fare`, `duration`, `casual`/`registered`, `Rating`, IDs). Write the prediction contract before looking at scores.
2. Fit every preprocessing, selection, resampling and encoding step inside training folds only. Use the holdout once.
3. Do not edit or rename files under any `results/` folder; the registry finds them by path. Rerun into new files instead.
4. Generated files (`evidence/knowledge/**`, `research/llm-fine-tuning/sft/out_v3/**`, campaign reports and `agent_memory.jsonl`) are rebuilt by commands, not edited.
5. LLM critique is advisory. Deterministic code owns splits, metrics and selection rules (`dclab_rnd/critic_gate.py` checks critiques against the numbers).
6. Report results with their uncertainty and limits. Never call a model production-ready from benchmark evidence alone.
7. Shared code stays at the repository root so `python -m dclab_rnd` and `import general_pipeline` work without installation.

## Git

Work on `dev`, merge to `main` when `make rd-check` passes. Commit messages explain why, not only what.
