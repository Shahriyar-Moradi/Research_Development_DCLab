"""What the routers share (package 10.1): the stores, the model gateway, the running jobs and the helpers every route
of an area used to reach as closures inside ``create_app``. One instance per app, on ``app.state.services``; a router
receives it through ``Depends(services)``. The helpers are the same code they were, with the same status codes."""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from .. import lessons as workspace_lessons, models
from ..agents.traces import open_traces
from ..intern import Intern
from ..intern.tools import Toolbox
from ..settings import Settings
from ..storage import open_stores
from ..studio import data as studio_data, engine as studio_engine, export as studio_export, graph as studio_graph, memory as studio_memory
from .store import Store


class Services:
    def __init__(self, settings: Settings):
        from ..models.routing import open_routing
        from ..models.shadow import open_shadows

        self.settings = settings
        self.store = Store(Path(settings.agent_home))
        self.projects, self.drafts, self.intern_sessions = open_stores(self.store.home)  # files, or PostgreSQL when DCLAB_DATABASE_URL is set
        # every model request goes through it; the routing moves a purpose to another model or names a shadow (A6.4)
        self.gateway = models.Gateway(models.open_usage(self.store.home), project_cap=self.project_cap,
                                      routing=open_routing(self.store.home), shadows=open_shadows(self.store.home))
        models.install(self.gateway)  # the engine's stage notes run outside a request
        self.tasks: dict[str, asyncio.Task] = {}  # research runs
        self.jobs: dict[str, asyncio.Task] = {}  # project stage jobs
        self.intern_jobs: dict[str, asyncio.Task] = {}
        self.draft_jobs: dict[str, asyncio.Task] = {}
        self.traces = open_traces(self.store.home)  # a row per agent step (package A2.3)
        self.lesson_store = workspace_lessons.open_lessons(self.store.home)  # the workspace's lessons table (A5.3)
        workspace_lessons.install(self.lesson_store)  # accepted lessons join every evidence search
        self.csrf = secrets.token_urlsafe(32)
        self.fixing: set[str] = set()  # projects whose notebook fixes are being written (one model request at a time each)
        from ..settings import UPLOAD_MAX_BYTES
        self.upload_max_bytes = UPLOAD_MAX_BYTES  # a constant, not an environment variable
        self.mcp = None

    # ------------------------------------------------------------------ the model budget
    def project_cap(self, project_id: str) -> float | None:  # the wizard's euro budget is the project's monthly cap
        try:
            value = (self.projects.get(project_id).get("budget") or {}).get("eur")
        except KeyError:
            return None
        return float(value) if value not in (None, "") else None

    # ------------------------------------------------------------------ research runs
    def run(self, run_id: str) -> dict[str, Any]:
        try:
            return self.store.get(run_id)
        except KeyError:
            raise HTTPException(404, "Research run not found")

    def launch(self, run_id: str, resume: bool = False) -> None:
        if any(not task.done() for task in self.tasks.values()):
            raise HTTPException(409, "One active research run at a time; pause it before starting another")
        from .engine import run_research  # imported on first run: the map and history work without the agent stack

        task = asyncio.create_task(run_research(self.store, run_id, resume))
        self.tasks[run_id] = task
        task.add_done_callback(lambda _: self.tasks.pop(run_id, None))

    # ------------------------------------------------------------------ projects
    def project(self, project_id: str) -> dict[str, Any]:
        try:
            return self.projects.get(project_id)
        except KeyError:
            raise HTTPException(404, "Project not found")

    def with_records(self, p: dict[str, Any]) -> dict[str, Any]:
        projects = self.projects
        return {**p, "records": projects.records(p["id"]), "activity": projects.activity(p["id"]), "stage_meta": studio_engine.STAGES,
                "graph": studio_graph.describe(p), "transitions": projects.transitions(p["id"], 40),
                "memory_read": [n["id"] for n in studio_memory.active(p)]}  # A5.2: the notes the agents read now

    def validate(self, p: dict[str, Any], move: str, **args: Any):
        """The workflow graph checks a person's move before anything starts; a refused move is logged and returned as 409."""
        verdict = studio_graph.check(p, move, "human", **args)
        if not verdict.allowed:
            studio_graph.log(self.projects, p["id"], verdict, p)
            raise HTTPException(409, verdict.message)
        return verdict

    def load_frame(self, p: dict[str, Any]):
        if not p.get("data"):
            raise HTTPException(409, "Upload data first")
        try:
            return studio_data.load_table(studio_data.data_path(self.projects, p))  # checked against the recorded SHA-256
        except studio_data.DataError as exc:
            raise HTTPException(409, str(exc)) from None

    def attach_data(self, p: dict[str, Any], filename: str) -> dict[str, Any]:
        try:
            return studio_data.attach_data(self.projects, p["id"], filename)
        except studio_data.DataError as exc:
            raise HTTPException(400, str(exc))

    async def run_job(self, project_id: str, stages: list[str], reuse_reason: str | None = None) -> None:
        def work():
            for stage in stages:
                studio_engine.execute(self.projects, project_id, stage, reuse_reason=reuse_reason if stage == "final" else None)
        try:
            await asyncio.to_thread(work)
        except Exception:  # noqa: BLE001 — the failure is recorded on the project by the engine
            pass
        finally:
            self.jobs.pop(project_id, None)
            p = self.projects.get(project_id)
            for stage in stages:
                if p["stages"][stage].get("status") == "queued":
                    p["stages"][stage] = {"status": "pending"}
            p["running"] = None
            self.projects.save(p)

    def start_job(self, project_id: str, stages: list[str], wait: bool, reuse_reason: str | None = None) -> asyncio.Task:
        if project_id in self.jobs and not self.jobs[project_id].done():
            raise HTTPException(409, "This project is already running a stage")
        p = self.projects.get(project_id)
        for stage in stages:
            p["stages"][stage] = {"status": "queued"}
        p["running"] = stages[0]
        self.projects.save(p)
        job = asyncio.create_task(self.run_job(project_id, stages, reuse_reason))
        self.jobs[project_id] = job
        return job

    def propose_lessons(self, project_id: str) -> list[dict[str, Any]]:
        p = self.projects.get(project_id)
        record = self.projects.read_stage(project_id, "final")
        return workspace_lessons.propose_for(self.lesson_store, p, record, self.gateway.client("lesson_proposal", project_id=project_id))

    def project_review(self, project_id: str) -> dict[str, Any]:
        """The copilot's full review of the project's exported notebook (proof with its text), read-only: unlike the
        export route it logs no capture move and saves nothing."""
        import tempfile

        from ..copilot import review_notebook

        p = self.project(project_id)
        if not p.get("solution"):
            raise HTTPException(409, "Nothing to review yet: save the solution first")
        text = studio_export.dumps_notebook(studio_export.notebook(p, self.projects.records(project_id)))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "notebook.ipynb"
            path.write_text(text, encoding="utf-8")
            return review_notebook(path)

    # ------------------------------------------------------------------ the intern
    def intern(self) -> Intern:
        return Intern(self.intern_sessions, Toolbox(self.projects), self.gateway.client("intern"), traces=self.traces)

    def session(self, session_id: str) -> dict[str, Any]:
        try:
            return self.intern_sessions.get(session_id)
        except KeyError:
            raise HTTPException(404, "Session not found")

    @staticmethod
    def public(s: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in s.items() if k != "messages"}

    async def intern_job(self, session_id: str, fn) -> None:
        try:
            await asyncio.to_thread(fn)
        except Exception:  # noqa: BLE001 — recorded on the session
            pass
        finally:
            self.intern_jobs.pop(session_id, None)

    def start_intern_job(self, session_id: str, fn, wait: bool) -> asyncio.Task:
        if session_id in self.intern_jobs and not self.intern_jobs[session_id].done():
            raise HTTPException(409, "This session is still working")
        job = asyncio.create_task(self.intern_job(session_id, fn))
        self.intern_jobs[session_id] = job
        return job

    def trace_of(self, session_id: str) -> list[dict[str, Any]]:
        try:
            return self.traces.steps(session_id)
        except Exception:  # noqa: BLE001 — a damaged trace never breaks the session page
            return []


def services(request: Request) -> Services:
    """The dependency every router takes."""
    return request.app.state.services
