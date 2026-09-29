# AGENTS.md — DCLab R&D Project

## Project overview

Python ML/R&D project: tabular classification research with a FastAPI web UI
called the "Agentic Research Studio." The UI lets users set research goals,
browse datasets, and run LLM-guided experiment loops.

## Architecture

- **Web UI**: FastAPI app at `dclab_rnd/agentic/server.py`, served by uvicorn.
  Static files in `dclab_rnd/agentic/static/` (vanilla HTML/JS/CSS, no build step).
- **Agent environment** (`.venv-agent`): Python 3.13 + `requirements-agent.lock.txt`
  (fastapi, uvicorn, langgraph, nooa, openai). Runs the web server and LLM agent loop.
- **ML worker environment** (`.venv`): Python 3.13 + numpy, pandas, scikit-learn,
  lightgbm, xgboost. Spawned as a subprocess by the agent engine for deterministic
  experiment execution. No LLM/network access in the worker.
- **Store**: SQLite at `agent_runs/research.sqlite3` (gitignored).

## Running in Base44

```bash
docker compose -f docker-compose.base44.yml up -d --build
```

- Web UI on port 3000.
- Dependencies install into named volumes on first start (agent + ML venvs).
- Uvicorn `--reload` watches `dclab_rnd/` for live code reload.
- Healthcheck: `GET /api/config` returns JSON with app configuration.

## Key modifications for Base44 preview

- `server.py`: `TrustedHostMiddleware` set to `allowed_hosts=["*"]` (was loopback-only)
  so the preview proxy can reach the app.
- Uvicorn binds `0.0.0.0:3000` (was `127.0.0.1:8765`).

## Secrets

- `OPENAI_API_KEY`: Required to run research experiments (LLM agent loop). The UI
  loads without it. Set via the Base44 secrets dashboard. A dev placeholder is in
  `.env.base44-defaults` so the app boots; the real value in `/run/base44/app.env`
  overrides it.

## Useful commands

```bash
# Run tests
.venv/bin/python -m unittest discover -s tests -v

# R&D sync (rebuild evidence registry)
.venv/bin/python -m dclab_rnd sync

# Agent tests
.venv-agent/bin/python -m pytest tests/test_agentic.py -q
```
