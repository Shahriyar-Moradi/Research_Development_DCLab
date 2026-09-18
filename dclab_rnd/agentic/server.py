"""Loopback-only research UI. API keys stay in the server environment."""
import asyncio
from contextlib import asynccontextmanager
import json
import importlib.metadata
import os
from pathlib import Path
import secrets
import uuid

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .catalog import ROOT, catalog
from .projects import project_catalog
from .schemas import RunRequest, DEFAULT_GOAL
from .store import Store
from .engine import run_research

load_dotenv(ROOT / ".env", override=False)
STATIC = Path(__file__).with_name("static")

def create_app(home=None):
    store = Store(Path(home or os.environ.get("DCLAB_AGENT_HOME", ROOT / "agent_runs")))
    tasks = {}
    csrf = secrets.token_urlsafe(32)
    @asynccontextmanager
    async def lifespan(app):
        for run in store.list():
            if run["status"] in ("queued", "running", "pausing"):
                store.update(run["id"], status="interrupted", phase="Server restarted; resume explicitly")
        yield
        active = list(tasks.values())
        for task in active: task.cancel()
        if active: await asyncio.gather(*active, return_exceptions=True)
    app = FastAPI(title="DCLab Research Studio", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.store, app.state.tasks = store, tasks
    @app.middleware("http")
    async def protect(request: Request, call_next):
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            token = request.headers.get("x-dclab-token", "")
            if not secrets.compare_digest(token, csrf):
                return JSONResponse({"detail": "Missing local UI request token"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        return response
    def get(run_id):
        try: return store.get(run_id)
        except KeyError: raise HTTPException(404, "Research run not found")
    def launch(run_id, resume=False):
        if any(not task.done() for task in tasks.values()):
            raise HTTPException(409, "One active research run at a time; pause it before starting another")
        task = asyncio.create_task(run_research(store, run_id, resume))
        tasks[run_id] = task
        task.add_done_callback(lambda _: tasks.pop(run_id, None))
    @app.get("/api/config")
    async def configuration():
        return {"csrf": csrf, "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")), "ml_python_available": Path(os.environ.get("DCLAB_ML_PYTHON", ROOT / ".venv/bin/python")).exists(), "default_goal": DEFAULT_GOAL, "default_model": os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"), "default_project": "general", "projects": project_catalog(), "datasets": catalog(), "frameworks": [f"NOOA {importlib.metadata.version('nooa')} · typed Predict specialists", f"LangGraph {importlib.metadata.version('langgraph')} · durable research loop", "OpenAI · Responses API · store=false"], "commands": {"serve": ".venv-agent/bin/python -m dclab_rnd.agentic serve", "hyperack": ".venv-agent/bin/python -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4", "churn_campaign": ".venv/bin/python -m dclab_rnd.churn_suite run", "churn_agent": ".venv-agent/bin/python -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4"}, "privacy": "Aggregate data profiles and scientific evidence are sent to OpenAI. Raw rows and API keys are not included in agent context. store=false; provider policies still apply."}
    @app.get("/api/runs")
    async def runs(): return store.list()
    @app.post("/api/runs", status_code=201)
    async def start(request: RunRequest):
        if not os.environ.get("OPENAI_API_KEY"): raise HTTPException(409, "Set OPENAI_API_KEY in the server environment or local .env, then restart. Do not paste it in a research goal.")
        if any(not task.done() for task in tasks.values()): raise HTTPException(409, "A research run is already active")
        run_id = uuid.uuid4().hex
        run = store.create(run_id, request.model_dump())
        launch(run_id)
        return run
    @app.get("/api/runs/{run_id}")
    async def detail(run_id: str):
        run = get(run_id)
        events = [e for e in store.events(run_id) if e["kind"] != "llm_request"]
        return {**run, "events": events}
    @app.post("/api/runs/{run_id}/pause")
    async def pause(run_id: str):
        get(run_id)
        task = tasks.get(run_id)
        if not task or task.done(): raise HTTPException(409, "Run is not active")
        store.update(run_id, status="pausing")
        task.cancel()
        return {"status": "pausing"}
    @app.post("/api/runs/{run_id}/resume")
    async def resume(run_id: str):
        run = get(run_id)
        if run["status"] not in ("paused", "interrupted", "failed"): raise HTTPException(409, "This run cannot be resumed")
        if not os.environ.get("OPENAI_API_KEY"): raise HTTPException(409, "OPENAI_API_KEY is missing")
        launch(run_id, True)
        return {"status": "running"}
    @app.get("/api/runs/{run_id}/export")
    async def export(run_id: str):
        get(run_id)
        return Response(json.dumps(store.export(run_id), indent=2), media_type="application/json", headers={"Content-Disposition": f'attachment; filename="dclab-{run_id}-trajectory.json"'})
    @app.get("/api/runs/{run_id}/trials/{trial_id}/{filename}")
    async def artifact(run_id: str, trial_id: str, filename: str):
        get(run_id)
        import re
        if not re.fullmatch(r"trial-\d{3}", trial_id) or filename not in ("result.json", "recipe.json", "oof_predictions.jsonl"):
            raise HTTPException(404, "Unknown artifact")
        path = store.home / run_id / trial_id / filename
        if not path.is_file(): raise HTTPException(404, "Artifact is not available")
        return FileResponse(path, filename=filename)
    @app.get("/api/knowledge")
    async def knowledge(): return store.knowledge()
    @app.get("/guide")
    async def guide():
        path = ROOT / "knowledge" / "MODEL_BUILDING_FIELD_GUIDE.html"
        if not path.is_file(): raise HTTPException(404, "Build the field guide with: .venv/bin/python -m dclab_rnd.master_review build")
        return FileResponse(path, media_type="text/html")
    @app.get("/")
    async def index(): return FileResponse(STATIC / "index.html")
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon(): return Response(status_code=204)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DCLAB_PORT", "8765")))
