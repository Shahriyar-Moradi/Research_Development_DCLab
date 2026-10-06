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
from ..storage import open_stores
from .. import models
from .. import lessons as workspace_lessons
from ..studio import ProjectStore, agent as studio_agent, solution as studio_solution, data as studio_data, engine as studio_engine, export as studio_export, graph as studio_graph, memory as studio_memory, sft as studio_sft
from ..intern import Intern, SessionStore
from ..models import settings as model_settings
from ..intern.sessions import EXAMPLE_TASKS
from ..intern.tools import Toolbox
from ..agents.traces import open_traces
from .. import mcp_server
from . import pages
from ..draft import api as draft_api
from ..draft.store import DraftStore

load_dotenv(ROOT / ".env", override=False)
STATIC = Path(__file__).with_name("static")
# No inline scripts or styles anywhere, and nothing is loaded from outside this app: the fonts are served from /static/app/fonts.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; "
       "font-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'")

def version(package):
    try: return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: return "not installed"

def create_app(home=None):
    store = Store(Path(home or os.environ.get("DCLAB_AGENT_HOME", ROOT / "agent_runs")))
    projects, drafts, intern_sessions = open_stores(store.home)  # files, or PostgreSQL when DCLAB_DATABASE_URL is set
    def project_cap(project_id: str) -> float | None:  # the wizard's euro budget is the project's monthly cap
        try:
            value = (projects.get(project_id).get("budget") or {}).get("eur")
        except KeyError:
            return None
        return float(value) if value not in (None, "") else None
    gateway = models.Gateway(models.open_usage(store.home), project_cap=project_cap)  # every model request goes through it
    models.install(gateway)  # the engine's stage notes run outside a request
    tasks = {}
    jobs = {}
    intern_jobs = {}
    draft_jobs = {}
    traces = open_traces(store.home)  # a row per agent step (package A2.3)
    lesson_store = workspace_lessons.open_lessons(store.home)  # the workspace's lessons table (A5.3)
    workspace_lessons.install(lesson_store)  # accepted lessons join every evidence search
    def intern():
        return Intern(intern_sessions, Toolbox(projects), gateway.client("intern"), traces=traces)
    csrf = secrets.token_urlsafe(32)
    mcp = mcp_server.session_manager(Toolbox(projects)) if mcp_server.available() else None
    @asynccontextmanager
    async def lifespan(app):
        for run in store.list():
            if run["status"] in ("queued", "running", "pausing"):
                store.update(run["id"], status="interrupted", phase="Server restarted; resume explicitly")
        if mcp is None:
            yield
        else:
            async with mcp.run():  # the MCP transport lives as long as the server
                yield
        active = list(tasks.values()) + list(jobs.values()) + list(intern_jobs.values()) + list(draft_jobs.values())
        for task in active: task.cancel()
        if active: await asyncio.gather(*active, return_exceptions=True)
    app = FastAPI(title="DCLab notebook", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
    app.state.store, app.state.tasks, app.state.projects, app.state.drafts, app.state.models = store, tasks, projects, drafts, gateway
    @app.middleware("http")
    async def protect(request: Request, call_next):
        # /mcp is JSON-RPC for local MCP clients (Chat UI, Claude Desktop…); the host check still applies to it.
        if request.method not in ("GET", "HEAD", "OPTIONS") and not request.url.path.startswith("/mcp"):
            token = request.headers.get("x-dclab-token", "")
            if not secrets.compare_digest(token, csrf):
                return JSONResponse({"detail": "Missing local UI request token"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Content-Security-Policy"] = CSP
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
        return {"csrf": csrf, "api_key_configured": bool(os.environ.get("OPENAI_API_KEY")), "ml_python_available": Path(os.environ.get("DCLAB_ML_PYTHON", ROOT / ".venv/bin/python")).exists(), "default_goal": DEFAULT_GOAL, "default_model": os.environ.get("OPENAI_MODEL", "gpt-5.6-terra"), "default_project": "general", "projects": project_catalog(), "datasets": catalog(), "frameworks": [f"NOOA {version('nooa')} · typed Predict specialists", f"LangGraph {version('langgraph')} · durable research loop", "OpenAI · Responses API · store=false"], "commands": {"serve": ".venv-agent/bin/python -m dclab_rnd.agentic serve", "hyperack": ".venv-agent/bin/python -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4", "churn_campaign": ".venv/bin/python -m dclab_rnd.churn_suite run", "churn_agent": ".venv-agent/bin/python -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4"}, "privacy": "When a model is configured, aggregate data profiles (with up to three example values per column) and scientific evidence are sent to it; the Home agent sends summaries only. Full rows and API keys are never sent. store=false; provider policies still apply."}
    @app.get("/api/models")
    async def model_overview():
        """Which model serves which purpose, what each purpose may be shown, and the usage so far (no keys)."""
        return await asyncio.to_thread(gateway.summary)
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
        return {**p, "records": projects.records(p["id"]), "activity": projects.activity(p["id"]), "stage_meta": studio_engine.STAGES,
                "graph": studio_graph.describe(p), "transitions": projects.transitions(p["id"], 40),
                "memory_read": [n["id"] for n in studio_memory.active(p)]}  # A5.2: the notes the agents read now
    def validate(p, move, **args):
        """The workflow graph checks a person's move before anything starts; a refused move is logged and returned as 409."""
        verdict = studio_graph.check(p, move, "human", **args)
        if not verdict.allowed:
            studio_graph.log(projects, p["id"], verdict, p)
            raise HTTPException(409, verdict.message)
        return verdict
    def load_frame(p):
        if not p.get("data"): raise HTTPException(409, "Upload data first")
        return studio_data.load_table(projects.data_dir(p["id"]) / p["data"]["filename"])
    def attach_data(p, filename):
        try: return studio_data.attach_data(projects, p["id"], filename)
        except studio_data.DataError as exc: raise HTTPException(400, str(exc))
    async def run_job(project_id, stages, reuse_reason=None):
        def work():
            for stage in stages:
                studio_engine.execute(projects, project_id, stage, reuse_reason=reuse_reason if stage == "final" else None)
        try: await asyncio.to_thread(work)
        except Exception: pass  # the failure is recorded on the project by the engine
        finally:
            jobs.pop(project_id, None)
            p = projects.get(project_id)
            for stage in stages:
                if p["stages"][stage].get("status") == "queued": p["stages"][stage] = {"status": "pending"}
            p["running"] = None
            projects.save(p)
    def start_job(project_id, stages, wait, reuse_reason=None):
        if project_id in jobs and not jobs[project_id].done(): raise HTTPException(409, "This project is already running a stage")
        p = projects.get(project_id)
        for stage in stages: p["stages"][stage] = {"status": "queued"}
        p["running"] = stages[0]
        projects.save(p)
        job = asyncio.create_task(run_job(project_id, stages, reuse_reason))
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
            if "folds" in settings: p["settings"]["folds"] = max(studio_engine.FOLD_CAPS[0], min(int(settings["folds"]), studio_engine.FOLD_CAPS[1]))
        changed = {}
        if isinstance(body.get("policy"), dict):  # which gates wait for a person; the invariants hold either way
            named = {studio_graph.LEGACY_POLICY.get(k, k): v for k, v in body["policy"].items()}
            before = {**studio_graph.DEFAULT_POLICY, **(p.get("policy") or {})}
            p["policy"] = {**(p.get("policy") or {}), **{k: bool(v) for k, v in named.items() if k in studio_graph.DEFAULT_POLICY}}
            changed = {k: {"from": before.get(k), "to": v} for k, v in p["policy"].items() if before.get(k) != v}
        projects.save(p)
        if changed: projects.log(project_id, "policy_changed", {"changes": changed, "actor": "human"})  # the platform audit reads it
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
        name = studio_data.safe_name(filename)  # never a path: "../x" or "a/b" cannot leave the data folder
        if not name: raise HTTPException(422, "Give the file a name")
        (projects.data_dir(project_id) / name).write_bytes(body)
        return with_records(attach_data(p, name))
    @app.post("/api/projects/{project_id}/data/sample")
    async def use_sample(project_id: str, request: Request):
        p = project(project_id)
        key = (await request.json()).get("key")
        try: await asyncio.to_thread(studio_data.use_sample, projects, project_id, key)
        except KeyError: raise HTTPException(404, "Unknown sample dataset")
        except studio_data.DataError as exc: raise HTTPException(400, str(exc))
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/solution/proposal")
    @app.post("/api/projects/{project_id}/contract/proposal", include_in_schema=False)  # old name, kept for one release
    async def solution_proposal(project_id: str, request: Request):
        p = project(project_id)
        body = await request.json()
        frame = load_frame(p)
        try: proposal = await asyncio.to_thread(studio_solution.propose, frame, p["data"]["profile"], str(body.get("target", "")), body.get("task"))
        except ValueError as exc: raise HTTPException(422, str(exc))
        if p.get("suggestion"):  # a studied sample carries the R&D's whole solution; a draft-built project only what the chat established
            known = {f["column"] for f in proposal["forbidden"]}
            proposal["forbidden"] = [{**f, "proof": ["DCLAB-R01"]} for f in p["suggestion"].get("forbidden") or [] if f["column"] not in known] + proposal["forbidden"]
            proposal["identifiers"] = sorted(set(proposal["identifiers"]) | set(p["suggestion"].get("identifiers") or []))
            if p["suggestion"].get("prediction_moment"): proposal["prediction_moment_hint"] = p["suggestion"]["prediction_moment"]
            if p["suggestion"].get("time_column"): proposal["time_candidates"] = [p["suggestion"]["time_column"]] + [c for c in proposal["time_candidates"] if c != p["suggestion"]["time_column"]]
        projects.update(project_id, proposal=proposal)  # only the proposal: the project read above is seconds old (notes, approvals)
        return proposal
    @app.put("/api/projects/{project_id}/solution")
    @app.put("/api/projects/{project_id}/contract", include_in_schema=False)  # old name, kept for one release
    async def save_solution(project_id: str, request: Request):
        p = project(project_id)
        body = await request.json()
        try:
            solution = studio_solution.Solution(**body)
            solution.check_columns(p["data"]["columns"] if p.get("data") else [])
        except Exception as exc: raise HTTPException(422, str(exc).split("\n")[0][:400] if "validation error" not in str(exc) else "; ".join(line.strip() for line in str(exc).split("\n")[1:] if line.strip() and not line.strip().startswith("For further"))[:600])
        changed = p.get("solution") != solution.model_dump()
        before = p.get("solution")
        if changed:
            verdict = validate(p, "set_solution", target=solution.target)
            studio_graph.log(projects, project_id, verdict, p, outcome="done: solution saved")
        p["solution"] = solution.model_dump()
        if changed:
            studio_memory.solution_saved(p, before, p["solution"], "human")  # A5.2: a correction when it changes the agent's
        projects.save(p)
        if changed:
            projects.clear_stages(project_id)
            projects.log(project_id, "solution_saved", {"target": solution.target, "task": solution.task, "forbidden": [f.column for f in solution.forbidden]})
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/stages/{stage}/run")
    async def run_stage(project_id: str, stage: str, wait: bool = False, reuse_reason: str = ""):
        p = project(project_id)
        if stage not in studio_engine.STAGE_BY_KEY: raise HTTPException(404, "Unknown stage")
        if not p.get("solution"): raise HTTPException(409, "Save the solution first")
        validate(p, "run_stage", stage=stage, reuse_reason=reuse_reason)
        job = start_job(project_id, [stage], wait, reuse_reason or None)
        if wait: await job
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/run")
    async def run_all(project_id: str, start: str = "data", wait: bool = False):
        p = project(project_id)
        if not p.get("solution"): raise HTTPException(409, "Save the solution first")
        if start not in studio_engine.STAGE_BY_KEY: raise HTTPException(404, "Unknown stage")
        keys = list(studio_engine.STAGE_BY_KEY)
        validate(p, "run_stage", stage=start)
        job = start_job(project_id, keys[keys.index(start):], wait)
        if wait: await job
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/stages/{stage}/approve")
    async def approve_stage(project_id: str, stage: str, request: Request):
        project(project_id)
        body = await request.json() if int(request.headers.get("content-length", "0") or 0) else {}
        try: studio_engine.approve(projects, project_id, stage, (body or {}).get("choice"))
        except (ValueError, KeyError) as exc: raise HTTPException(409, str(exc))
        if stage == "final":  # A5.3: a finished project proposes what it taught, for a reviewer
            try: await asyncio.to_thread(propose_lessons, project_id)
            except Exception as exc:  # noqa: BLE001 — the approval is saved; POST /lessons proposes them again later
                projects.log(project_id, "lessons_not_proposed", {"error": type(exc).__name__})
        return with_records(projects.get(project_id))
    def propose_lessons(project_id):
        p = projects.get(project_id)
        record = projects.read_stage(project_id, "final")
        return workspace_lessons.propose_for(lesson_store, p, record, gateway.client("lesson_proposal", project_id=project_id))
    @app.post("/api/projects/{project_id}/lessons")
    async def project_lessons(project_id: str):
        """Propose the lessons of a finished project (once per final stage record; a second call returns the same ones)."""
        p = project(project_id)
        if (p["stages"].get("final") or {}).get("status") != "approved":
            raise HTTPException(409, "Lessons are proposed after a person approves the final stage")
        return {"lessons": await asyncio.to_thread(propose_lessons, project_id), "label": workspace_lessons.LABEL}
    @app.get("/api/lessons")
    async def list_lessons(project_id: str | None = None, status: str | None = None):
        """The workspace's lessons, for the Evidence library's "From projects" panel and a project's page."""
        return {"lessons": lesson_store.list(project_id=project_id, status=status), "label": workspace_lessons.LABEL,
                "reviewers": list(workspace_lessons.REVIEWERS)}
    @app.post("/api/lessons/{lesson_id}/review")
    async def review_lesson(lesson_id: str, request: Request):
        """A reviewer accepts, edits or rejects a lesson. Until accounts exist the role is the one the request declares
        (X-DCLab-Role), so this keeps an honest record of who reviewed in which role, not a verified login."""
        try: body = await request.json()
        except ValueError: raise HTTPException(422, "The body must be JSON") from None
        body = body if isinstance(body, dict) else {}
        role = (request.headers.get("x-dclab-role") or "").strip().lower() or None
        try:
            lesson = workspace_lessons.review(lesson_store, lesson_id, str(body.get("action", "")), role, str(body.get("reason") or ""),
                                              {k: body.get(k) for k in ("claim", "against", "next_test")})
        except PermissionError as exc: raise HTTPException(403, str(exc)) from None
        except KeyError: raise HTTPException(404, "No such lesson") from None
        except workspace_lessons.ReviewError as exc: raise HTTPException(422, str(exc)) from None
        try: projects.log(lesson["project_id"], "lesson_reviewed", {"lesson": lesson_id, "action": lesson["review"]["action"], "by": role,
                                                                   "status": lesson["status"], "synthetic": lesson.get("synthetic", False)})
        except (KeyError, OSError): pass  # the project was deleted (KeyError on PostgreSQL, a missing folder on files); the lesson stands on its own
        return lesson
    @app.get("/api/projects/{project_id}/graph")
    async def project_graph(project_id: str):
        p = project(project_id)
        return {**studio_graph.describe(p), "transitions": projects.transitions(project_id)}
    @app.post("/api/projects/{project_id}/graph/check")
    async def check_move(project_id: str, request: Request):
        p = project(project_id)
        body = await request.json()
        move = str(body.pop("move", ""))
        actor = "agent" if body.pop("actor", "human") == "agent" else "human"
        args = {k: body[k] for k in ("stage", "choice", "gate", "reuse_reason") if k in body}
        return studio_graph.check(p, move, actor, **args).to_dict()
    @app.delete("/api/projects/{project_id}/memory/{note_id}")
    async def remove_memory_note(project_id: str, note_id: str, request: Request):
        """A person removes a note from the project's memory (A5.2); it stays in the project, marked removed, and the
        activity log keeps who removed it and why."""
        project(project_id)
        try: body = await request.json() if int(request.headers.get("content-length", "0") or 0) else {}
        except ValueError: raise HTTPException(422, "The body must be JSON: {\"reason\": \"...\"}") from None
        reason = body.get("reason", "") if isinstance(body, dict) else ""
        try: studio_memory.remove(projects, project_id, note_id, "human", str(reason or ""))
        except KeyError: raise HTTPException(404, "No such note") from None
        return with_records(projects.get(project_id))
    @app.post("/api/projects/{project_id}/approvals")
    async def approve_gate(project_id: str, request: Request):
        project(project_id)
        body = await request.json()
        try: studio_graph.approve_gate(projects, project_id, str(body.get("gate", "")), "owner", str(body.get("reason", "")))
        except studio_graph.GraphBlocked as exc: raise HTTPException(409, str(exc))
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
        if not p.get("solution"): raise HTTPException(409, "Nothing to export yet")
        studio_graph.capture(projects, project_id, "human")
        text = studio_export.dumps_notebook(studio_export.notebook(p, projects.records(project_id)))
        return Response(text, media_type="application/x-ipynb+json", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.ipynb"'})
    fixing: set[str] = set()  # projects whose notebook fixes are being written (one model request at a time each)
    def project_review(project_id):
        """The copilot's full review of the project's exported notebook (proof with its text), read-only: unlike the
        export route it logs no capture move and saves nothing."""
        p = project(project_id)
        if not p.get("solution"): raise HTTPException(409, "Nothing to review yet: save the solution first")
        text = studio_export.dumps_notebook(studio_export.notebook(p, projects.records(project_id)))
        import tempfile
        from ..copilot import review_notebook
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notebook.ipynb"
            path.write_text(text, encoding="utf-8")
            return review_notebook(path)
    @app.get("/api/projects/{project_id}/review")
    async def review_project_notebook(project_id: str):
        """The notebook copilot's review of the project's exported notebook, in the demo's review shape."""
        project(project_id)
        def review():
            report = project_review(project_id)
            keys = ("detector", "severity", "cell", "line", "title", "message", "suggestion", "rules")
            return {"summary": report["summary"], "cells": [{"type": c["type"], "source": c["source"]} for c in report["cells"]],
                    "findings": [{**{k: f.get(k) for k in keys}, "proof": [r["record_id"] for r in f.get("proof", [])]} for f in report["findings"]],
                    "limitations": report.get("limitations", []), "fixes_available": gateway.available("notebook_review")}
        return await asyncio.to_thread(review)
    @app.post("/api/projects/{project_id}/review/fixes")
    async def review_project_fixes(project_id: str):
        """A model's two-sentence fix per finding (package A3.4), by finding number. The findings are the deterministic
        review's, unchanged; a fix that does not cite its rule and pitfall record, or states a number its records do not,
        is dropped. Asked for by the person, since each call is a model request."""
        client = gateway.client("notebook_review", project_id=project_id)
        if client is None: raise HTTPException(409, "No model is configured for notebook fixes")
        if project_id in fixing: raise HTTPException(409, "Fixes for this notebook are already being written")
        from ..copilot.fixes import key_of, write_fixes
        def run():
            findings = project_review(project_id)["findings"]
            # keyed by the finding (detector:cell:line), not its position: the page matches them to the findings it shows
            return {"fixes": {key_of(findings[n]): fix for n, fix in write_fixes(findings, client).items()}, "findings": len(findings)}
        fixing.add(project_id)
        try: return await asyncio.to_thread(run)
        finally: fixing.discard(project_id)
    @app.get("/api/projects/{project_id}/export/report")
    async def export_report(project_id: str):
        p = project(project_id)
        return Response(studio_export.report(p, projects.records(project_id)), media_type="text/markdown; charset=utf-8", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.md"'})
    @app.get("/api/projects/{project_id}/export/sft")
    async def export_sft(project_id: str):
        p = project(project_id)
        rows = studio_sft.examples_from_project(p, projects.records(project_id))
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        return Response(text, media_type="application/x-ndjson", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.sft.jsonl"'})
    # ------------------------------------------------------------------ the intern
    def session(session_id):
        try: return intern_sessions.get(session_id)
        except KeyError: raise HTTPException(404, "Session not found")
    def public(s): return {k: v for k, v in s.items() if k != "messages"}
    async def intern_job(session_id, fn):
        try: await asyncio.to_thread(fn)
        except Exception: pass  # recorded on the session
        finally: intern_jobs.pop(session_id, None)
    def start_intern_job(session_id, fn, wait):
        if session_id in intern_jobs and not intern_jobs[session_id].done(): raise HTTPException(409, "This session is still working")
        job = asyncio.create_task(intern_job(session_id, fn))
        intern_jobs[session_id] = job
        return job
    @app.get("/api/intern")
    async def intern_status(request: Request):
        cfg = model_settings.public(model_settings.purpose("intern").tier)  # the tier the intern's requests go to (no key)
        return {**cfg, "mode": "llm" if gateway.available("intern") else "standard", "examples": EXAMPLE_TASKS,
                "mcp_url": (str(request.base_url).rstrip("/") + "/mcp") if mcp is not None else None,
                "chat_ui": {"command": "make chat-ui", "intern_command": "make chat-ui-intern", "url": "http://localhost:5173/", "intern_url": "http://localhost:5173/?mode=ml-intern"},
                "tools": Toolbox(projects).names(), "default_budget": {"max_steps": 24, "max_minutes": 20},
                "note": ("The model plans and calls the tools; deterministic code runs every stage." if cfg["available"] else
                         "No model is configured, so the intern follows the standard DCLab plan. Set OPENAI_API_KEY (and OPENAI_BASE_URL for the Hugging Face router or Ollama) to let a model plan.")}
    @app.get("/api/intern/sessions")
    async def intern_list(): return intern_sessions.list()
    @app.post("/api/intern/sessions", status_code=201)
    async def intern_start(request: Request, wait: bool = False):
        body = await request.json()
        task = str(body.get("task", "")).strip()
        if len(task) < 8: raise HTTPException(422, "Describe the task in at least a sentence")
        project_id = body.get("project_id") or None
        if project_id: project(project_id)
        agent = intern()
        try: s = agent.start(task, body.get("budget") if isinstance(body.get("budget"), dict) else None, project_id)
        except (TypeError, ValueError) as exc: raise HTTPException(422, f"budget: {exc}") from None
        job = start_intern_job(s["id"], lambda: agent.run(s["id"]), wait)
        if wait: await job
        return public(session(s["id"]))
    def trace_of(session_id):
        try: return traces.steps(session_id)
        except Exception: return []  # a damaged trace never breaks the session page
    @app.get("/api/intern/sessions/{session_id}")
    async def intern_get(session_id: str): return {**public(session(session_id)), "trace": trace_of(session_id)}
    @app.post("/api/intern/sessions/{session_id}/message")
    async def intern_message(session_id: str, request: Request, wait: bool = False):
        s = session(session_id)
        if s["status"] in ("queued", "running"): raise HTTPException(409, "Wait for the current turn to finish")
        text = str((await request.json()).get("text", "")).strip()
        if not text: raise HTTPException(422, "Say something")
        agent = intern()
        job = start_intern_job(session_id, lambda: agent.message(session_id, text), wait)
        if wait: await job
        return public(session(session_id))
    @app.delete("/api/intern/sessions/{session_id}", status_code=204)
    async def intern_delete(session_id: str):
        session(session_id)
        if session_id in intern_jobs and not intern_jobs[session_id].done(): raise HTTPException(409, "The session is still working")
        intern_sessions.delete(session_id)
        try: traces.delete(session_id)
        except Exception: pass  # the session is gone either way; a stray trace row is harmless
        return Response(status_code=204)
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
    async def index():
        # The product frontend (built from dclab_rnd/agentic/web); the earlier UI stays at /classic until the new one covers it.
        page = STATIC / "app" / "index.html"
        if not page.is_file(): raise HTTPException(404, "Build the frontend with: python -m dclab_rnd.agentic.web.build")
        return FileResponse(page)
    @app.get("/classic")
    async def classic(): return FileResponse(STATIC / "index.html")
    draft_api.register(app, drafts, projects, gateway, draft_jobs, traces)
    pages.register_all(app, pages.Context(store=store, projects=projects, drafts=drafts, intern_sessions=intern_sessions, models=gateway,
                                          jobs=jobs, intern_jobs=intern_jobs, draft_jobs=draft_jobs))
    @app.get("/api/workspace")
    async def workspace():
        """What Home shows: real counts, the projects with their progress, and what waits for a person."""
        def summary():
            items, forbidden, opened, holdout_projects, blocked = [], 0, 0, 0, 0
            for p in projects.list():
                solution = p.get("solution") or {}
                forbidden += len(solution.get("forbidden") or [])
                uses = int(p.get("holdout_uses") or 0)
                if uses:
                    holdout_projects += 1
                    opened += 1 if uses == 1 else 0
                moves = projects.transitions(p["id"], 500)
                blocked += sum(1 for m in moves if m.get("status") == "blocked")
                items.append({"id": p["id"], "name": p["name"], "updated": p.get("updated"), "goal": p.get("goal"),
                              "data": p["data"] and {k: p["data"].get(k) for k in ("filename", "rows", "synthetic")},
                              "pack": ((p.get("draft") or {}).get("pack") or {}).get("key"), "has_solution": bool(p.get("solution")),
                              "stages": {k: (v or {}).get("status") for k, v in (p.get("stages") or {}).items()},
                              "state": studio_graph.state_string(p), "current": studio_graph.current_node(p),
                              "holdout_uses": uses, "running": p.get("running")})
            needs = []
            for d in drafts.list(20):
                if d.get("status") != "open":
                    continue
                q = next((q for q in d.get("questions", []) if not q.get("answered")), None)
                if q:
                    needs.append({"kind": "question", "draft": d["id"], "title": q["text"], "sub": d["problem"][:140], "tag": "WF-01"})
            for it in items:
                if not it["has_solution"]:
                    needs.append({"kind": "solution", "project": it["id"], "title": f"{it['name']}: write the solution draft", "sub": "Nothing is trained until the solution is saved.", "tag": "WF-01"})
            return {"projects": items, "needs": needs[:8],
                    "stats": {"forbidden": forbidden, "holdouts_once": opened, "holdout_projects": holdout_projects, "blocked_moves": blocked,
                              "projects": len(items), "drafts": sum(1 for d in drafts.list(200) if d.get("status") == "open")}}
        return await asyncio.to_thread(summary)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon(): return Response(status_code=204)
    if mcp is not None:
        app.mount("/mcp", app=mcp.handle_request, name="mcp")
    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app

app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("DCLAB_PORT", "8765")))
