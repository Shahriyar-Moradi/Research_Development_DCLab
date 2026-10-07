# CLAUDE.md — working in the DCLab R&D repository

This file gives every Claude Code session (cloud or local) the context it needs. Keep it short and current.

## What this repository is

The research arm of DCLab. It measures how to build ML models that are *right* (leakage-safe, honestly evaluated) and turns the evidence into tools: an evidence index, a notebook copilot, agent tools, and SFT data for a future small model. The owner prefers answers in clear English **and** Persian.

**Naming:** when the owner says "DCLab" in a conversation about this repository, it means the DCLab R&D project and the product growing out of it (the agentic notebook, the intern, the evidence library, the policy model). It does not mean DCLab's main product. The product vision and a clickable demo of the final product live in `docs/product-demo/`.

## Map

- `data/` — all datasets: `project/` (HyperAck, Telco), `public/` (10 UCI parquet), `downloads/` (gitignored).
- `research/<track>/` — one folder per research idea, same shape everywhere: `README.md` (idea, contract, conclusions), generated `INDEX.md` (champion, experiments, notebooks, evaluation, reports), `experiments/`, `notebooks/`, `evaluation/`, `reports/`. Index: `research/README.md`. New idea: `make new-track NAME=... TITLE="..." PREFIX=...`.
- `evidence/campaigns/` — immutable experiment results (one JSON each). `evidence/knowledge/` — GENERATED; never edit by hand.
- `dclab_rnd/` — shared package: control plane (`python -m dclab_rnd`), `evidence_index.py`, `cited.py` (the one check for model-written prose: cites a given source, numbers in it, no production-ready), `critic_gate.py`, `pitfalls.py`, `tools.py`, `lessons.py` (lessons from finished projects: proposed after the final stage, accepted by a reviewer, searched with the evidence as workspace lessons; never written into `evidence/knowledge`), `copilot/`, `notebook_assist.py` (cell-level companion), `expansion/`, `prompts/` (every prompt sent to a model, as a file; a change needs `make agent-eval`, the test gate checks the hashes), `retrieval/` (evidence search measured on 60 questions; the product uses what the measurement chose, `make retrieval-eval`), `agent_eval/` (the judgment suite: seeded planted traps, scripted policies, `make agent-eval`), `research_map.py` (track records → INDEX.md and the map page), `studio/` (the DCLab notebook: projects, the solution, five-stage engine on user data, `graph.py` the WF-01…WF-10 workflow graph whose validator checks every move and logs it to `transitions.jsonl`, evidence-cited notes, `memory.py` project memory (decisions and corrections the agents read), `explain.py` stage explanations written by a model and checked by `cited.py`, `leakage_review.py`, .ipynb export), `storage/` (the interface every caller uses for projects, drafts and intern sessions; file stores, or PostgreSQL when `DCLAB_DATABASE_URL` is set: `python -m dclab_rnd.storage upgrade` creates the schema, `… migrate SOURCE --to TARGET` moves a file workspace into it; `make backup` / `make restore` (`storage/backup.py`) dump and restore the database and the workspace folder; tests use the `dclab_test` database), `draft/` (Home before a project exists: a draft with a background pipeline structure → clean → analyze, the Home agent that asks a few questions and draws the solution workflow, pack detection, synthetic data, and `build_project`), `connectors/` (Kaggle, Hugging Face, databases, S3/GCS; credentials stay on the server), `agentic/` (API server: `server.py` only builds the app; routes in `agentic/routers/` and `draft/api.py` and `agentic/pages/`, typed with `agentic/api_models.py`, sharing `agentic/services.py` through `Depends`; settings in `dclab_rnd/settings.py`; `agentic/pages/` holds the read-mostly routes of the product's workspace, lab, learn, platform and evidence pages; the product frontend in `agentic/web/src/`, built into `agentic/static/app/` and served at `/`; the earlier UI in `agentic/static/{css,js}` at `/classic`; and the LLM campaign loop), `intern/` (chat mode with tools and a budget over the notebook; standard plan when no model is configured), `studio/sft.py` (finished projects → SFT examples; `--trajectories` exports opted-in projects' moves and agent steps via `studio/trajectories.py`, no cell values or free text; `studio/corpus_v4.py`, `make sft-v4`: v3 plus trajectories, whole projects held out, nothing below 50 trajectories from 10 projects; `sft/out_v4/` is not committed), `mcp_server.py` (the toolbox as an MCP server at `/mcp`, for Hugging Face Chat UI / ML Intern), `jobs/` (stage runs, data pipelines, synthetic data and intern turns as rows in a job table; a worker claims them, `checkpoint()` between steps makes Stop work; `python -m dclab_rnd.worker`), `audit.py` (one append-only audit trail per workspace: every governed action is written where it happens; the Admin audit tab reads it), `accounts/` (sign-in only when `DCLAB_AUTH` is set: sessions, API tokens, the four roles checked on every route by the shared Router; `python -m dclab_rnd.accounts add` makes the first owner), `observe.py` (JSON logs with a request id, /healthz and /readyz, Prometheus counters on DCLAB_METRICS_PORT), `retention.py` (DCLAB_RETENTION_RAW_DAYS: a draft's raw upload deleted after N days, the cleaned table and the records kept), `limits.py` (with accounts on: requests a minute, bytes a day, jobs at once and model spend a month, per person and per workspace; 429 when reached).
- `deploy/` — the cloud deployment as OpenTofu/Terraform (12.6): `aws/` and `gcp/`, each a module with `staging/` and `production/`; README says what the owner decides and runs. Never apply it from a session.
- `general_pipeline/` — shared tabular pipeline imported by research scripts and `dclab_rnd`.
- `docs/guides/` — field notes, integration plan, agent architecture, copilot, SFT guide, `PRODUCT_BUILD_PLAYBOOK.md` (the product plan and the software foundation as small packages, each with its check, its status and a prompt), and `AGENTIC_FOUNDATION_PLAN.md` (the AI and agent layer: principles, architecture, phases A1 to A6).

## Commands

```bash
make help        # every target
make rd-check    # tests + evidence validation + freshness of all generated files (~2 min); must pass before committing
make check-all   # the gate for a plan package: rd-check, the suite on PostgreSQL (test-pg) and the end-to-end flows (~7 min; longer when the machine is short of memory)
make test        # the suite in parallel (tests/run_parallel.py; JOBS=N to choose, make test-serial for one process)
make rd-sync     # rebuild the registry and knowledge base after new results
make knowledge   # rebuild the evidence index and SFT v3 corpus after new results
make research-index   # regenerate every research/<track>/INDEX.md (champions, file maps)
make dev              # local development on PostgreSQL: dclab_dev created and migrated, the API with reload, the frontend rebuilt on change (make db-reset CONFIRM=yes, make test-db)
make notebook         # the web UI (DCLab notebook, intern, research map, knowledge) + the MCP endpoint /mcp
make chat-ui          # Hugging Face Chat UI locally with DCLab as an MCP server (make chat-ui-intern: ML Intern mode)
make product-demo     # rebuild docs/product-demo/index.html: demo v1, the frozen UI reference (tag demo-v1; do not change it)
make web              # rebuild the product frontend (agentic/web/src → agentic/static/app); same UI as demo v1, wired to the API
make test-pg          # the whole suite on PostgreSQL (the local dclab_test database, emptied by the tests)
make agent-eval       # the scripted judgment suite: planted traps, the standard plan and reference policies (stores an AEV result)
make agent-eval-live  # the same suite with the configured model: prints the plan; ARGS="--yes --cap-eur 2" runs it with caps and repeats
make product-e2e      # the product's main flows on a temporary server (upload, no data, synthetic, log file; ~90 s, no model)
make model-paths      # every model path once against a real model on made-up data (ARGS="--ollama qwen2.5-coder:1.5b" free, or "--configured --cap-eur 2")
make deploy-check     # the cloud code in deploy/ (AWS and Google Cloud, staging and production): tofu fmt and validate; nothing is ever planned or applied by tooling
make browser-test     # Chromium drives the real page: Home, the wizard, a run, the brief, stats against the API, 19 pages at 1280 and 375 px (pip install -r requirements/browser.txt; CI runs it on the container)
python -m dclab_rnd.copilot review NOTEBOOK.ipynb   # methodology review of a notebook
```

Environments: `.venv` from `requirements/base.txt` (ML work); `.venv-agent` (Python 3.12/3.13) from `requirements/agent.lock.txt` for the Research Studio; CI uses `requirements/ci.txt`.

## Rules

1. **Leakage is the top risk.** Never use information unavailable at the prediction moment (`final_customer_fare`, `final_biker_fare`, `duration`, `casual`/`registered`, `Rating`, IDs). Write the prediction contract before looking at scores.
2. Fit every preprocessing, selection, resampling and encoding step inside training folds only. Use the holdout once.
3. Do not edit or rename files under any `results/` folder; the registry finds them by path. Rerun into new files instead.
4. Generated files (`evidence/knowledge/**`, `research/*/INDEX.md`, `research/llm-fine-tuning/experiments/sft/out_v3/**`, `docs/product-demo/index.html`, `dclab_rnd/agentic/static/app/**`, `dclab_rnd/agentic/web/src/api.js` (the API client, from the OpenAPI schema), campaign reports and `agent_memory.jsonl`) are rebuilt by commands, not edited.
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
- `dclab_rnd/agentic/static/app/**` is generated from `dclab_rnd/agentic/web/src` by `make web`, and so is `src/api.js` (`DC.client`, from the routes): pages call `DC.client.<route>()`, never a path string, and show `DC.states` for loading, empty and failed.
- `DCLAB_DATABASE_URL` switches projects, drafts and sessions to PostgreSQL; tests use (and empty) `dclab_test`. A project's table is read through `studio.data.data_path` (checked against its recorded SHA-256; `storage/files.py`), never by joining `data_dir` and the file name; a workspace folder keeps its identity in `.dclab_workspace`.
- Every model request goes through `dclab_rnd.models` (the gateway); tests use scripted transports, never a live key. A purpose can be moved to another tier (a reviewer and a reason) or shadowed by a local one (`models/routing.py`, `models/shadow.py`): a shadow's answer is logged, never used.
- Every agent tool is registered once in `dclab_rnd.agents` (`default_registry()`): scope "project" for the intern and `/mcp`, "draft" for the Home agent, "session" for the intern's plan and report, "campaign" for the research campaign's experiment. Never keep a private tool list. A write tool declares its move and runs only through its scope's guard (the graph validator for projects). The Home agent and the intern run on `agents.run()`; the campaign's LangGraph phases are traced and its experiment goes through the registry.
- `.venv-agent` has no pandas, so the studio tests skip there; run the suite with `.venv`.
- The suite runs in parallel: one process per test file, each with its own TMPDIR and, for PostgreSQL, its own database `dclab_test_p<N>` (`tests/run_parallel.py`). A test must not depend on another file having run; a failing file's output is printed whole.

## Git

Work on `main` (keep `dev` fast-forwarded to it) and commit only when `make rd-check` passes. Commit messages explain why, not only what.
