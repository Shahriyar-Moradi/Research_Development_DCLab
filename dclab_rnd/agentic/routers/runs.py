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
    from .. import campaign_model

    picked = campaign_model.choice()  # the product's Claude when it runs on Claude, else an OpenAI-compatible endpoint
    if picked is None:
        raise HTTPException(409, campaign_model.why_not())
    model = campaign_model.model_for(request.model, picked)
    if picked["provider"] == "anthropic" and not campaign_model.priced(model):
        raise HTTPException(409, f"{model} has no price, so the euro caps could not count it (DCLAB_CAMPAIGN_MODEL)")
    if any(not task.done() for task in s.tasks.values()):
        raise HTTPException(409, "A research run is already active")
    run_id = uuid.uuid4().hex
    config = {**request.model_dump(), "model": model, "provider": picked["provider"]}
    run = s.store.create(run_id, config)
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
    from .. import campaign_model

    picked = campaign_model.choice()
    if picked is None:
        raise HTTPException(409, campaign_model.why_not())
    if (run["config"].get("provider") or "openai") != picked["provider"]:  # a run resumes on the provider it began with
        raise HTTPException(409, f"This run began on {run['config'].get('provider') or 'openai'}; the server now runs the campaign on {picked['provider']}")
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
