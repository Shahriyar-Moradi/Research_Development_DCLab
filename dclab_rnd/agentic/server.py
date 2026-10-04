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
from .. import research_map
from ..studio import ProjectStore, agent as studio_agent, contract as studio_contract, data as studio_data, engine as studio_engine, export as studio_export

load_dotenv(ROOT / ".env", override=False)
STATIC = Path(__file__).with_name("static")

def version(package):
    try: return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: return "not installed"

def create_app(home=None):
    store = Store(Path(home or os.environ.get("DCLAB_AGENT_HOME", ROOT / "agent_runs")))
    projects = ProjectStore(Path(os.environ.get("DCLAB_STUDIO_HOME") or (store.home / "projects")))
    tasks = {}
    jobs = {}
    csrf = secrets.token_urlsafe(32)
    @asynccontextmanager
    async def lifespan(app):
        for run in store.list():
            if run["status"] in ("queued", "running", "pausing"):
                store.update(run["id"], status="interrupted", phase="Server restarted; resume explicitly")
        yield
        active = list(tasks.values()) + list(jobs.values())
        for task in active: task.cancel()
        if active: await asyncio.gather(*active, return_exceptions=True)
    app = FastAPI(title="DCLab Research Studio", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.store, app.state.tasks, app.state.projects = store, tasks, projects
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
        from .engine import run_research  # imported on first run: the map and history work without the agent stack
        task = asyncio.create_task(run_research(store, run_id, resume))
        tasks[run_id] = task
        task.add_done_callback(lambda _: tasks.pop(run_id, None))
    @app.get("/api/config")
    async def configuration():
        return {"csrf": csrf, "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")), "ml_python_available": Path(os.environ.get("DCLAB_ML_PYTHON", ROOT / ".venv/bin/python")).exists(), "default_goal": DEFAULT_GOAL, "default_model": os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"), "default_project": "general", "projects": project_catalog(), "datasets": catalog(), "frameworks": [f"NOOA {version('nooa')} · typed Predict specialists", f"LangGraph {version('langgraph')} · durable research loop", "OpenAI · Responses API · store=false"], "commands": {"serve": ".venv-agent/bin/python -m dclab_rnd.agentic serve", "hyperack": ".venv-agent/bin/python -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4", "churn_campaign": ".venv/bin/python -m dclab_rnd.churn_suite run", "churn_agent": ".venv-agent/bin/python -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4"}, "privacy": "Aggregate data profiles and scientific evidence are sent to OpenAI. Raw rows and API keys are not included in agent context. store=false; provider policies still apply."}
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
    @app.get("/api/research")
    async def research():
        """Every research idea with its champion, experiments, notebooks, evaluation, reports and relations."""
        return research_map.research_map()
    @app.get("/api/research/file")
    async def research_file(path: str):
        try: return research_map.preview(path)
        except ValueError as error: raise HTTPException(404, str(error))

    # ------------------------------------------------------------------ projects: the DCLab notebook
    def project(project_id):
        try: return projects.get(project_id)
        except KeyError: raise HTTPException(404, "Project not found")
    def with_records(p):
        return {**p, "records": projects.records(p["id"]), "activity": projects.activity(p["id"]), "stage_meta": studio_engine.STAGES}
    def load_frame(p):
        if not p.get("data"): raise HTTPException(409, "Upload data first")
        return studio_data.load_table(projects.data_dir(p["id"]) / p["data"]["filename"])
    def attach_data(p, filename):
        path = projects.data_dir(p["id"]) / filename
        try: frame = studio_data.load_table(path)
        except Exception as exc: path.unlink(missing_ok=True); raise HTTPException(400, f"Could not read the file as a table: {type(exc).__name__}")
        if frame.shape[1] < 2 or len(frame) < 30: path.unlink(missing_ok=True); raise HTTPException(400, "The table needs at least 2 columns and 30 rows")
        for old in projects.data_dir(p["id"]).iterdir():
            if old.name != filename: old.unlink()
        p = projects.get(p["id"])
        p["data"] = {"filename": filename, "rows": int(len(frame)), "columns": [str(c) for c in frame.columns], "sha256": studio_data.sha256(path), "profile": studio_data.profile_table(frame)}
        p["contract"], p["proposal"] = None, None
        projects.save(p)
        projects.clear_stages(p["id"])
        projects.log(p["id"], "data_attached", {"filename": filename, "rows": p["data"]["rows"], "columns": len(p["data"]["columns"])})
        return projects.get(p["id"])
    async def run_job(project_id, stages):
        def work():
            for stage in stages:
                studio_engine.execute(projects, project_id, stage)
        try: await asyncio.to_thread(work)
        except Exception: pass  # the failure is recorded on the project by the engine
        finally:
            jobs.pop(project_id, None)
            p = projects.get(project_id)
            for stage in stages:
                if p["stages"][stage].get("status") == "queued": p["stages"][stage] = {"status": "pending"}
            p["running"] = None
            projects.save(p)
    def start_job(project_id, stages, wait):
        if project_id in jobs and not jobs[project_id].done(): raise HTTPException(409, "This project is already running a stage")
        p = projects.get(project_id)
        for stage in stages: p["stages"][stage] = {"status": "queued"}
        p["running"] = stages[0]
        projects.save(p)
        job = asyncio.create_task(run_job(project_id, stages))
        jobs[project_id] = job
        return job
    @app.get("/api/studio")
    async def studio_status():
        return {**studio_engine.capabilities(), "industries": list(studio_data.INDUSTRIES) if hasattr(studio_data, "INDUSTRIES") else [], "stages": studio_engine.STAGES}
    @app.get("/api/samples")
    async def samples(): return studio_data.sample_catalog()
    @app.get("/api/projects")
    async def list_projects():
        return [{k: v for k, v in p.items() if k not in ("proposal",)} | {"data": p["data"] and {k: v for k, v in p["data"].items() if k != "profile"}} for p in projects.list()]
    @app.post("/api/projects", status_code=201)
    async def create_project(request: Request):
        body = await request.json()
        if not isinstance(body, dict) or not str(body.get("name", "")).strip(): raise HTTPException(422, "A project needs a name")
        p = projects.create(str(body.get("name", "")), str(body.get("industry", "general")), str(body.get("goal", "")))
        return with_records(p)
    @app.get("/api/projects/{project_id}")
    async def get_project(project_id: str): return with_records(project(project_id))
    @app.patch("/api/projects/{project_id}")
    async def patch_project(project_id: str, request: Request):
        p = project(project_id)
        body = await request.json()
        for key in ("name", "goal"):
            if key in body: p[key] = str(body[key]).strip()[:4000 if key == "goal" else 120]
        if body.get("industry") in studio_data.INDUSTRIES if hasattr(studio_data, "INDUSTRIES") else False: p["industry"] = body["industry"]
        if isinstance(body.get("settings"), dict):
            settings = body["settings"]
            if "max_rows" in settings: p["settings"]["max_rows"] = max(200, min(int(settings["max_rows"]), 200000))
            if "quick" in settings: p["settings"]["quick"] = bool(settings["quick"])
        projects.save(p)
        return with_records(projects.get(project_id))
    @app.delete("/api/projects/{project_id}", status_code=204)
    async def delete_project(project_id: str):
        project(project_id)
        if project_id in jobs and not jobs[project_id].done(): raise HTTPException(409, "Stop the running stage first")
        projects.delete(project_id)
        return Response(status_code=204)
    @app.put("/api/projects/{project_id}/data")
    async def upload_data(project_id: str, request: Request, filename: str = "data.csv"):
        p = project(project_id)
        body = await request.body()
        if len(body) > 200 * 1024 * 1024: raise HTTPException(413, "Files above 200 MB are not supported in the local notebook")
        if not body: raise HTTPException(400, "Empty file")
        name = studio_data.safe_name(filename) if hasattr(studio_data, "safe_name") else filename
        (projects.data_dir(project_id) / name).write_bytes(body)
        return with_records(attach_data(p, name))
    @app.post("/api/projects/{project_id}/data/sample")
    async def use_sample(project_id: str, request: Request):
        p = project(project_id)
        key = (await request.json()).get("key")
        try: frame, suggestion = studio_data.load_sample(key)
        except KeyError: raise HTTPException(404, "Unknown sample dataset")
        name = f"{key}.csv"
        frame.to_csv(projects.data_dir(project_id) / name, index=False)
        p = attach_data(p, name)
        p["suggestion"] = {**suggestion, "sample": key}
        projects.save(p)
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/contract/proposal")
    async def contract_proposal(project_id: str, request: Request):
        p = project(project_id)
        body = await request.json()
        frame = load_frame(p)
        try: proposal = await asyncio.to_thread(studio_contract.propose, frame, p["data"]["profile"], str(body.get("target", "")), body.get("task"))
        except ValueError as exc: raise HTTPException(422, str(exc))
        if p.get("suggestion"):
            known = {f["column"] for f in proposal["forbidden"]}
            proposal["forbidden"] = [{**f, "proof": ["DCLAB-R01"]} for f in p["suggestion"]["forbidden"] if f["column"] not in known] + proposal["forbidden"]
            proposal["identifiers"] = sorted(set(proposal["identifiers"]) | set(p["suggestion"].get("identifiers", [])))
            proposal["prediction_moment_hint"] = p["suggestion"]["prediction_moment"]
            if p["suggestion"].get("time_column"): proposal["time_candidates"] = [p["suggestion"]["time_column"]] + [c for c in proposal["time_candidates"] if c != p["suggestion"]["time_column"]]
        p["proposal"] = proposal
        projects.save(p)
        return proposal
    @app.put("/api/projects/{project_id}/contract")
    async def save_contract(project_id: str, request: Request):
        p = project(project_id)
        body = await request.json()
        try:
            contract = studio_contract.Contract(**body)
            contract.check_columns(p["data"]["columns"] if p.get("data") else [])
        except Exception as exc: raise HTTPException(422, str(exc).split("\n")[0][:400] if "validation error" not in str(exc) else "; ".join(line.strip() for line in str(exc).split("\n")[1:] if line.strip() and not line.strip().startswith("For further"))[:600])
        changed = p.get("contract") != contract.model_dump()
        p["contract"] = contract.model_dump()
        projects.save(p)
        if changed:
            projects.clear_stages(project_id)
            projects.log(project_id, "contract_saved", {"target": contract.target, "task": contract.task, "forbidden": [f.column for f in contract.forbidden]})
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/stages/{stage}/run")
    async def run_stage(project_id: str, stage: str, wait: bool = False):
        p = project(project_id)
        if stage not in studio_engine.STAGE_BY_KEY: raise HTTPException(404, "Unknown stage")
        if not p.get("contract"): raise HTTPException(409, "Save the prediction contract first")
        job = start_job(project_id, [stage], wait)
        if wait: await job
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/run")
    async def run_all(project_id: str, start: str = "data", wait: bool = False):
        p = project(project_id)
        if not p.get("contract"): raise HTTPException(409, "Save the prediction contract first")
        if start not in studio_engine.STAGE_BY_KEY: raise HTTPException(404, "Unknown stage")
        keys = list(studio_engine.STAGE_BY_KEY)
        job = start_job(project_id, keys[keys.index(start):], wait)
        if wait: await job
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/stages/{stage}/approve")
    async def approve_stage(project_id: str, stage: str, request: Request):
        project(project_id)
        body = await request.json() if int(request.headers.get("content-length", "0") or 0) else {}
        try: studio_engine.approve(projects, project_id, stage, (body or {}).get("choice"))
        except (ValueError, KeyError) as exc: raise HTTPException(409, str(exc))
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/ask")
    async def ask_agent(project_id: str, request: Request):
        p = project(project_id)
        question = str((await request.json()).get("question", "")).strip()
        if not question: raise HTTPException(422, "Ask something")
        records = projects.records(project_id)
        task_type = next((r["task_type"] for r in records.values()), None)
        answer = await asyncio.to_thread(studio_agent.answer, question[:1000], p, records, task_type)
        projects.log(project_id, "question", {"question": question[:300], "proof": answer["proof"]})
        return answer
    @app.get("/api/projects/{project_id}/export/notebook")
    async def export_notebook(project_id: str):
        p = project(project_id)
        if not p.get("contract"): raise HTTPException(409, "Nothing to export yet")
        text = studio_export.dumps_notebook(studio_export.notebook(p, projects.records(project_id)))
        return Response(text, media_type="application/x-ipynb+json", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.ipynb"'})
    @app.get("/api/projects/{project_id}/export/report")
    async def export_report(project_id: str):
        p = project(project_id)
        return Response(studio_export.report(p, projects.records(project_id)), media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.md"'})
    @app.get("/api/evidence/{record_id}")
    async def evidence_record(record_id: str):
        from ..tools import get_record
        record = get_record(record_id)
        if "error" in record: raise HTTPException(404, record["error"])
        return record
    @app.get("/api/knowledge")
    async def knowledge(): return store.knowledge()
    @app.get("/guide")
    async def guide():
        path = ROOT / "evidence/knowledge" / "MODEL_BUILDING_FIELD_GUIDE.html"
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
