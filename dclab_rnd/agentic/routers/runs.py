"""The research campaign's runs (the Jobs page's research rows): start, pause, resume, export and artifacts."""

from __future__ import annotations

import json
import re
import uuid

from fastapi import Depends, HTTPException
from fastapi.responses import FileResponse, Response

from ..api_models import Router, Run, Status, download
from ..schemas import RunRequest
from ..services import Services, services

router = Router()


@router.get("/api/runs", response_model=list[Run])
async def runs(s: Services = Depends(services)):
    return s.store.list()


@router.post("/api/runs", status_code=201, response_model=Run)
async def start(request: RunRequest, s: Services = Depends(services)):
    if not s.settings.openai_key_set:
        raise HTTPException(409, "Set OPENAI_API_KEY in the server environment or local .env, then restart. Do not paste it in a research goal.")
    if any(not task.done() for task in s.tasks.values()):
        raise HTTPException(409, "A research run is already active")
    run_id = uuid.uuid4().hex
    run = s.store.create(run_id, request.model_dump())
    s.launch(run_id)
    return run


@router.get("/api/runs/{run_id}", response_model=Run)
async def detail(run_id: str, s: Services = Depends(services)):
    run = s.run(run_id)
    events = [e for e in s.store.events(run_id) if e["kind"] != "llm_request"]
    return {**run, "events": events}


@router.post("/api/runs/{run_id}/pause", response_model=Status)
async def pause(run_id: str, s: Services = Depends(services)):
    s.run(run_id)
    task = s.tasks.get(run_id)
    if not task or task.done():
        raise HTTPException(409, "Run is not active")
    s.store.update(run_id, status="pausing")
    task.cancel()
    return {"status": "pausing"}


@router.post("/api/runs/{run_id}/resume", response_model=Status)
async def resume(run_id: str, s: Services = Depends(services)):
    run = s.run(run_id)
    if run["status"] not in ("paused", "interrupted", "failed"):
        raise HTTPException(409, "This run cannot be resumed")
    if not s.settings.openai_key_set:
        raise HTTPException(409, "OPENAI_API_KEY is missing")
    s.launch(run_id, True)
    return {"status": "running"}


@router.get("/api/runs/{run_id}/export", response_class=Response, responses=download("application/json"))
async def export(run_id: str, s: Services = Depends(services)):
    s.run(run_id)
    return Response(json.dumps(s.store.export(run_id), indent=2), media_type="application/json",
                    headers={"Content-Disposition": f'attachment; filename="dclab-{run_id}-trajectory.json"'})


@router.get("/api/runs/{run_id}/trials/{trial_id}/{filename}", response_class=FileResponse, responses=download("application/octet-stream"))
async def artifact(run_id: str, trial_id: str, filename: str, s: Services = Depends(services)):
    s.run(run_id)
    if not re.fullmatch(r"trial-\d{3}", trial_id) or filename not in ("result.json", "recipe.json", "oof_predictions.jsonl"):
        raise HTTPException(404, "Unknown artifact")
    path = s.store.home / run_id / trial_id / filename
    if not path.is_file():
        raise HTTPException(404, "Artifact is not available")
    return FileResponse(path, filename=filename)
