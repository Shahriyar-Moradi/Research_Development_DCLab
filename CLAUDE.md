# CLAUDE.md — working in the DCLab R&D repository

This file gives every Claude Code session (cloud or local) the context it needs. Keep it short and current.

## What this repository is

The research arm of DCLab. It measures how to build ML models that are *right* (leakage-safe, honestly evaluated) and turns the evidence into tools: an evidence index, a notebook copilot, agent tools, and SFT data for a future small model. The owner prefers answers in clear English **and** Persian.

**Naming:** when the owner says "DCLab" in a conversation about this repository, it means the DCLab R&D project and the product growing out of it (the agentic notebook, the intern, the evidence library, the policy model). It does not mean DCLab's main product. The product vision and a clickable demo of the final product live in `docs/product-demo/`.

## Map

- `data/` — all datasets: `project/` (HyperAck, Telco), `public/` (10 UCI parquet), `downloads/` (gitignored).
- `research/<track>/` — one folder per research idea, same shape everywhere: `README.md` (idea, contract, conclusions), generated `INDEX.md` (champion, experiments, notebooks, evaluation, reports), `experiments/`, `notebooks/`, `evaluation/`, `reports/`. Index: `research/README.md`. New idea: `make new-track NAME=... TITLE="..." PREFIX=...`.
- `evidence/campaigns/` — immutable experiment results (one JSON each). `evidence/knowledge/` — GENERATED; never edit by hand.
- `dclab_rnd/` — shared package: control plane (`python -m dclab_rnd`), `evidence_index.py`, `critic_gate.py`, `pitfalls.py`, `tools.py`, `copilot/`, `notebook_assist.py` (cell-level companion), `expansion/`, `research_map.py` (track records → INDEX.md and the map page), `studio/` (the DCLab notebook: projects, the solution, five-stage engine on user data, `graph.py` the WF-01…WF-10 workflow graph whose validator checks every move and logs it to `transitions.jsonl`, evidence-cited notes, .ipynb export), `storage/` (the interface every caller uses for projects, drafts and intern sessions; file stores, or PostgreSQL when `DCLAB_DATABASE_URL` is set: `python -m dclab_rnd.storage upgrade` creates the schema; tests use the `dclab_test` database), `draft/` (Home before a project exists: a draft with a background pipeline structure → clean → analyze, the Home agent that asks a few questions and draws the solution workflow, pack detection, synthetic data, and `build_project`), `connectors/` (Kaggle, Hugging Face, databases, S3/GCS; credentials stay on the server), `agentic/` (API server; `agentic/pages/` holds the read-mostly routes of the product's workspace, lab, learn, platform and evidence pages; the product frontend in `agentic/web/src/`, built into `agentic/static/app/` and served at `/`; the earlier UI in `agentic/static/{css,js}` at `/classic`; and the LLM campaign loop), `intern/` (chat mode with tools and a budget over the notebook; standard plan when no model is configured), `studio/sft.py` (finished projects → SFT examples), `mcp_server.py` (the toolbox as an MCP server at `/mcp`, for Hugging Face Chat UI / ML Intern).
- `general_pipeline/` — shared tabular pipeline imported by research scripts and `dclab_rnd`.
- `docs/guides/` — field notes, integration plan, agent architecture, copilot, SFT guide, `PRODUCT_BUILD_PLAYBOOK.md` (the product plan and the software foundation as small packages, each with its check, its status and a prompt), and `AGENTIC_FOUNDATION_PLAN.md` (the AI and agent layer: principles, architecture, phases A1 to A6).

## Commands

```bash
make help        # every target
make rd-check    # tests + evidence validation + freshness of all generated files (~3 min); must pass before committing
make check-all   # the gate for a plan package: rd-check, the suite on PostgreSQL (test-pg) and the end-to-end flows (~8 min)
make rd-sync     # rebuild the registry and knowledge base after new results
make knowledge   # rebuild the evidence index and SFT v3 corpus after new results
make research-index   # regenerate every research/<track>/INDEX.md (champions, file maps)
make notebook         # the web UI (DCLab notebook, intern, research map, knowledge) + the MCP endpoint /mcp
make chat-ui          # Hugging Face Chat UI locally with DCLab as an MCP server (make chat-ui-intern: ML Intern mode)
make product-demo     # rebuild docs/product-demo/index.html: demo v1, the frozen UI reference (tag demo-v1; do not change it)
make web              # rebuild the product frontend (agentic/web/src → agentic/static/app); same UI as demo v1, wired to the API
make test-pg          # the whole suite on PostgreSQL (the local dclab_test database, emptied by the tests)
make product-e2e      # the product's main flows on a temporary server (upload, no data, synthetic, log file; ~90 s, no model)
python -m dclab_rnd.copilot review NOTEBOOK.ipynb   # methodology review of a notebook
```

Environments: `.venv` from `requirements/base.txt` (ML work); `.venv-agent` (Python 3.12/3.13) from `requirements/agent.lock.txt` for the Research Studio; CI uses `requirements/ci.txt`.

## Rules

1. **Leakage is the top risk.** Never use information unavailable at the prediction moment (`final_customer_fare`, `final_biker_fare`, `duration`, `casual`/`registered`, `Rating`, IDs). Write the prediction contract before looking at scores.
2. Fit every preprocessing, selection, resampling and encoding step inside training folds only. Use the holdout once.
3. Do not edit or rename files under any `results/` folder; the registry finds them by path. Rerun into new files instead.
4. Generated files (`evidence/knowledge/**`, `research/*/INDEX.md`, `research/llm-fine-tuning/experiments/sft/out_v3/**`, `docs/product-demo/index.html`, `dclab_rnd/agentic/static/app/**`, campaign reports and `agent_memory.jsonl`) are rebuilt by commands, not edited.
5. LLM critique is advisory. Deterministic code owns splits, metrics and selection rules (`dclab_rnd/critic_gate.py` checks critiques against the numbers).
6. Report results with their uncertainty and limits. Never call a model production-ready from benchmark evidence alone.
7. Shared code stays at the repository root so `python -m dclab_rnd` and `import general_pipeline` work without installation.

## Working on the plan

The product and its foundation are built package by package from `docs/guides/PRODUCT_BUILD_PLAYBOOK.md` and
`docs/guides/AGENTIC_FOUNDATION_PLAN.md`. Use the `dclab-package` skill for one package and the `dclab-reviewer`
subagent to review its diff before committing. When compacting, keep the package in progress, the files changed and
the last `make check-all` result.

## Gotchas

- Never `git stash`: the owner keeps uncommitted work in this tree (for example `docs/recap/`).
- `dclab_rnd/agentic/static/app/**` is generated from `dclab_rnd/agentic/web/src` by `make web`.
- `DCLAB_DATABASE_URL` switches projects, drafts and sessions to PostgreSQL; tests use (and empty) `dclab_test`.
- Every model request goes through `dclab_rnd.models` (the gateway); tests use scripted transports, never a live key.
- Every agent tool is registered once in `dclab_rnd.agents` (`default_registry()`): scope "project" for the intern and `/mcp`, "draft" for the Home agent. Never keep a private tool list.
- `.venv-agent` has no pandas, so the studio tests skip there; run the suite with `.venv`.

## Git

Work on `main` (keep `dev` fast-forwarded to it) and commit only when `make rd-check` passes. Commit messages explain why, not only what.
