"""What the routers share (package 10.1): the stores, the model gateway, the running jobs and the helpers every route
of an area used to reach as closures inside ``create_app``. One instance per app, on ``app.state.services``; a router
receives it through ``Depends(services)``. The helpers are the same code they were, with the same status codes."""

from __future__ import annotations

import asyncio
import secrets
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from .. import audit, lessons as workspace_lessons, models
from ..agents.traces import open_traces
from ..intern import Intern
from ..intern.tools import Toolbox
from ..settings import Settings
from ..storage import open_stores
from ..studio import data as studio_data, engine as studio_engine, export as studio_export, graph as studio_graph, memory as studio_memory
from ..draft.work import DraftWork
from ..jobs import ActiveJob, Worker, open_jobs
from ..jobs import handlers as _job_handlers  # noqa: F401 — registers the four kinds of job
from ..jobs.live import Live
from .store import Store

BUSY = {"stage": "This project is already running a stage", "intern": "This session is still working",
        "pipeline": "This file is already being processed", "synthetic": "Synthetic data is already being generated for this draft"}


class Services:
    def __init__(self, settings: Settings, primary: bool = True):
        from ..models.routing import open_routing
        from ..models.shadow import open_shadows

        self.settings = settings
        self.store = Store(Path(settings.agent_home))
        from ..storage import db

        # the workspace this Services serves (its folder's marker); None on files, where a folder is the only workspace
        self.workspace_id = db.workspace_for(self.store.home) if db.database_url(required=False) else None
        self.projects, self.drafts, self.intern_sessions = open_stores(self.store.home)  # files, or PostgreSQL when DCLAB_DATABASE_URL is set
        # every model request goes through it; the routing moves a purpose to another model or names a shadow (A6.4)
        self.gateway = models.Gateway(models.open_usage(self.store.home), project_cap=self.project_cap,
                                      routing=open_routing(self.store.home), shadows=open_shadows(self.store.home))
        if primary:  # the process's fallback is the server's own workspace; a request or job names its own (dclab_rnd.context)
            models.install(self.gateway)  # the engine's stage notes run outside a request
        self.tasks: dict[str, asyncio.Task] = {}  # research runs (the legacy campaign; not in the job table)
        self.traces = open_traces(self.store.home)  # a row per agent step (package A2.3)
        # Package 10.3: stage runs, data pipelines, synthetic data and intern turns are rows in the job table, run by a
        # worker (here, unless DCLAB_WORKER=external, or `python -m dclab_rnd.worker`); the pages read who runs what from it.
        self.job_store = open_jobs(self.store.home)
        self.worker = Worker(self.job_store, self, threads=0 if settings.worker == "external" else settings.worker_threads)
        self.draft_tasks: dict[str, asyncio.Task] = {}  # the Home agent's turns: asyncio tasks of this process
        self.jobs = Live(self.job_store, "stage")  # by project id
        self.intern_jobs = Live(self.job_store, "intern")  # by session id
        self.draft_jobs = Live(self.job_store, ("pipeline", "synthetic"), tasks=self.draft_tasks)  # by "draft:asset" or "draft:synthetic"
        self.draft_work = DraftWork(self.drafts, self.gateway, self.traces, self)
        # Package 10.4: one append-only audit for the workspace; the history the per-project logs hold is imported once
        self.audit = audit.for_store(self.projects) or audit.open_audit(self.store.home)  # the same log the stores write
        try:
            audit.backfill(self.audit, self.projects, self.gateway.routing)
        except Exception as exc:  # noqa: BLE001 — the history stays in the project logs; the server starts either way
            print(f"audit: the earlier history was not imported ({type(exc).__name__})", flush=True)
        self.lesson_store = workspace_lessons.open_lessons(self.store.home)  # the workspace's lessons table (A5.3)
        if primary:
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

    # ------------------------------------------------------------------ jobs (package 10.3)
    def queue_job(self, kind: str, key: str, payload: dict[str, Any], by: str = "human", here: bool = False) -> dict[str, Any]:
        """A row in the job table. ``here``: claimed at once by this process's worker, for a request that runs it now
        (``?wait=true``, or the Home agent's synthetic request inside its turn). ActiveJob when the key is held."""
        self.worker.start()  # once: the worker threads (or only the heartbeat, with DCLAB_WORKER=external)
        job = self.job_store.enqueue(kind, key, payload, by=by, worker=self.worker.id if here else None)
        if not here:
            self.worker.notify()
        return job

    def submit(self, kind: str, key: str, payload: dict[str, Any], by: str = "human", here: bool = False, busy: str | None = None) -> dict[str, Any]:
        try:
            return self.queue_job(kind, key, payload, by=by, here=here)
        except ActiveJob:
            raise HTTPException(409, busy or BUSY[kind]) from None

    async def run_here(self, job: dict[str, Any]) -> dict[str, Any]:
        return await asyncio.to_thread(self.worker.run_here, job)

    def mark_queued(self, project_id: str, stages: list[str]) -> dict[str, Any]:
        """Stages "queued" before their job is inserted; returns what they were, for ``unmark_queued``."""
        p = self.projects.get(project_id)
        before = {"stages": {stage: p["stages"].get(stage) for stage in stages}, "running": p.get("running")}
        for stage in stages:
            p["stages"][stage] = {"status": "queued"}
        p["running"] = stages[0]
        self.projects.save(p)
        return before

    def unmark_queued(self, project_id: str, before: dict[str, Any]) -> None:
        """Another app instance's job took the project between the check and the insert: put back what this request changed."""
        p = self.projects.get(project_id)
        for stage, was in before["stages"].items():
            if (p["stages"].get(stage) or {}).get("status") == "queued" and was is not None:
                p["stages"][stage] = was
        p["running"] = before["running"]
        self.projects.save(p)

    def start_job(self, project_id: str, stages: list[str], wait: bool, reuse_reason: str | None = None) -> dict[str, Any]:
        if self.job_store.active("stage", project_id) is not None:
            raise HTTPException(409, BUSY["stage"])
        before = self.mark_queued(project_id, stages)
        try:
            return self.queue_job("stage", project_id, {"project_id": project_id, "stages": stages, "reuse_reason": reuse_reason}, here=wait)
        except ActiveJob:
            self.unmark_queued(project_id, before)
            raise HTTPException(409, BUSY["stage"]) from None

    def prepare_retry(self, job: dict[str, Any]) -> Any:
        """Before a job is queued again: the same checks a new request passes, and its domain put back to "queued".
        Returns what ``undo_retry`` needs when the retry then loses the key to another job."""
        payload = job.get("payload") or {}
        if job["kind"] == "stage":
            p = self.project(payload["project_id"])
            if self.job_store.active("stage", p["id"]) is not None:
                raise HTTPException(409, BUSY["stage"])
            self.validate(p, "run_stage", stage=payload["stages"][0], reuse_reason=payload.get("reuse_reason") or "")
            return self.mark_queued(p["id"], payload["stages"])
        elif job["kind"] == "intern":
            if self.session(payload["session_id"])["status"] in ("queued", "running"):
                raise HTTPException(409, BUSY["intern"])
        elif job["kind"] == "pipeline":
            from ..draft import pipeline

            try:
                self.drafts.get(payload["draft_id"])
            except KeyError:
                raise HTTPException(404, "Draft not found") from None
            if self.job_store.active("pipeline", f"{payload['draft_id']}:{payload['asset_id']}") is not None:
                raise HTTPException(409, BUSY["pipeline"])
            pipeline.set_asset(self.drafts, payload["draft_id"], payload["asset_id"], status="queued", error=None)
            self.drafts.emit(payload["draft_id"], "pipeline", {"asset": payload["asset_id"], "step": "queued", "text": "Trying again"})
        return None

    def undo_retry(self, job: dict[str, Any], before: Any) -> None:
        if job["kind"] == "stage" and before is not None:
            self.unmark_queued(job["payload"]["project_id"], before)

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

    def start_intern_job(self, session_id: str, payload: dict[str, Any], wait: bool) -> dict[str, Any]:
        """An intern turn as a job: ``{"action": "run"}`` for the first one, ``{"action": "message", "text": …}`` after."""
        return self.submit("intern", session_id, {"session_id": session_id, **payload}, here=wait)

    def trace_of(self, session_id: str) -> list[dict[str, Any]]:
        try:
            return self.traces.steps(session_id)
        except Exception:  # noqa: BLE001 — a damaged trace never breaks the session page
            return []


def services(request: Request) -> Services:
    """The dependency every router takes: the Services of the workspace this request acts in (package 10.2)."""
    from .. import context

    return context.services() or request.app.state.services
