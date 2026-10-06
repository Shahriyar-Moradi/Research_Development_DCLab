"""The job table over HTTP (package 10.3): list and read jobs, stop one, retry one.

    GET  /api/jobs?kind=&active=   the newest jobs first
    GET  /api/jobs/{id}
    POST /api/jobs/{id}/stop       a queued job is cancelled now; a running one stops at its next checkpoint
    POST /api/jobs/{id}/retry      a failed, stopped or interrupted job is queued again, after the checks a new request passes
"""

from __future__ import annotations

import asyncio

from fastapi import Depends, HTTPException

from ...jobs import ActiveJob
from ...jobs.worker import _settle
from ..api_models import Doc, Router
from ..services import BUSY, Services, services

router = Router()
KINDS = ("stage", "pipeline", "synthetic", "intern")


FREE_TEXT = ("text", "prompt")  # a person's message, a description sent to a model: kept out of responses, as a session's messages are


def public(job: dict) -> dict:
    """A job without the free text its payload carries; ids, stages and sizes stay."""
    return {**job, "payload": {k: v for k, v in (job.get("payload") or {}).items() if k not in FREE_TEXT}}


def _job(s: Services, job_id: str) -> dict:
    try:
        return s.job_store.get(job_id)
    except KeyError:
        raise HTTPException(404, "Job not found") from None


@router.get("/api/jobs", response_model=list[Doc])
async def list_jobs(kind: str | None = None, active: bool | None = None, limit: int = 100, s: Services = Depends(services)):
    if kind is not None and kind not in KINDS:
        raise HTTPException(422, f"kind is one of {', '.join(KINDS)}")
    return [public(j) for j in s.job_store.list(kind=kind, active=active, limit=max(1, min(limit, 500)))]


@router.get("/api/jobs/{job_id}", response_model=Doc)
async def read_job(job_id: str, s: Services = Depends(services)):
    return public(_job(s, job_id))


@router.post("/api/jobs/{job_id}/stop", response_model=Doc)
async def stop_job(job_id: str, s: Services = Depends(services)):
    before = _job(s, job_id)
    if before["status"] not in ("queued", "running"):
        raise HTTPException(409, f"This job has already ended ({before['status']})")
    job = s.job_store.cancel(job_id)
    if job["status"] == "cancelled":  # it never started: its domain goes back to how it was (stages "pending", the asset failed)
        _settle(s, job, "stopped")
    return public(job)


@router.post("/api/jobs/{job_id}/retry", response_model=Doc)
async def retry_job(job_id: str, s: Services = Depends(services)):
    job = _job(s, job_id)
    if job["status"] not in ("failed", "cancelled", "interrupted"):
        raise HTTPException(409, f"Only a failed, stopped or interrupted job can be retried; this one is {job['status']}")
    before = s.prepare_retry(job)

    def requeue():
        with s.job_slot():  # counted and queued under the workspace's lock, like a new job (package 10.6)
            s.job_room()
            return s.job_store.retry(job_id)
    try:
        queued = await asyncio.to_thread(requeue)
    except (ActiveJob, ValueError):  # another instance took the key, or retried this job, since the checks above
        s.undo_retry(job, before)
        raise HTTPException(409, BUSY.get(job["kind"], "Another job holds this already")) from None
    except HTTPException:  # the job limit
        s.undo_retry(job, before)
        raise
    s.worker.start()
    s.worker.notify()
    return public(queued)
