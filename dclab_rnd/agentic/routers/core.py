"""The app's own routes: configuration, models and their routing, the research map, evidence records, the workspace
summary Home reads, and the pages themselves."""

from __future__ import annotations

import asyncio
import importlib.metadata
from pathlib import Path
from typing import Any

from fastapi import Depends, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response

from ... import research_map
from ...studio import data as studio_data, engine as studio_engine, graph as studio_graph
from ..api_models import Config, Doc, ModelsSummary, Page, Router, RoutingChange, Workspace
from ..catalog import ROOT, catalog
from ..projects import project_catalog
from ..schemas import DEFAULT_GOAL
from ..services import Services, services


def declared_role(request: Request) -> str | None:
    """The reviewer's role: the signed-in person's (package 10.2); without accounts, the one the request declares."""
    from ...accounts.principal import current

    who = current()
    if who is not None and who.user_id:
        return who.role
    return (request.headers.get("x-dclab-role") or "").strip().lower() or None


router = Router()
STATIC = Path(__file__).resolve().parents[1] / "static"


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


@router.get("/healthz", response_model=Doc)
async def healthz():
    """The process is up (package 12.4): nothing else is checked, so a busy database never restarts a healthy server."""
    return {"status": "ok"}


_PROBE = asyncio.Lock()


@router.get("/readyz", response_model=Doc)
async def readyz(s: Services = Depends(services)):
    """The server can do its work (package 12.4): the database answers and the file storage can be written. 503 names
    what is not ready, never a URL or a host."""
    if _PROBE.locked():  # a probe still waiting on a silent database: answer from it, not with one more blocked thread
        return JSONResponse({"status": "not ready", "checks": {"probe": "busy"}}, status_code=503)
    async with _PROBE:
        try:
            checks = await asyncio.wait_for(asyncio.to_thread(readiness, s), timeout=5)
        except asyncio.TimeoutError:
            checks = {"answer within 5 s": False}
    ready = all(checks.values())
    body = {"status": "ready" if ready else "not ready", "checks": {k: "ok" if v else "failed" for k, v in checks.items()}}
    return body if ready else JSONResponse(body, status_code=503)


def readiness(s: Services) -> dict[str, bool]:
    import tempfile

    from ...storage import db

    checks: dict[str, bool] = {}
    if s.settings.database_url:
        checks["database"] = db.reachable(s.settings.database_url) is None
    try:
        if s.settings.files_url:
            from ...storage.files import files_for

            store = files_for(s.projects)
            store._s3().head_bucket(Bucket=store.bucket)
        else:
            with tempfile.NamedTemporaryFile(dir=s.store.home, prefix=".ready-"):
                pass
        checks["files"] = True
    except Exception:  # noqa: BLE001 — not writable, not reachable, no credentials: not ready
        checks["files"] = False
    return checks


@router.get("/api/config", response_model=Config)
async def configuration(s: Services = Depends(services)):
    st = s.settings
    from ...accounts.principal import current

    who = current()
    csrf = who.csrf if who is not None and who.via == "session" and who.csrf else s.csrf  # a signed-in browser's writes carry its session's
    return {"csrf": csrf, "auth": {"mode": st.auth, **(who.public() if who is not None else {"signed_in": False})}, "api_key_configured": st.openai_key_set, "ml_python_available": Path(st.ml_python).exists(), "default_goal": DEFAULT_GOAL,
            "default_model": st.openai_model, "default_project": "general", "projects": project_catalog(), "datasets": catalog(),
            "frameworks": [f"NOOA {version('nooa')} · typed Predict specialists", f"LangGraph {version('langgraph')} · durable research loop", "OpenAI · Responses API · store=false"],
            "commands": {"serve": ".venv-agent/bin/python -m dclab_rnd.agentic serve", "hyperack": ".venv-agent/bin/python -m dclab_rnd.agentic run --project hyperack --datasets hyperack --experiments 4",
                         "churn_campaign": ".venv/bin/python -m dclab_rnd.churn_suite run", "churn_agent": ".venv-agent/bin/python -m dclab_rnd.agentic run --project telco_churn --datasets telco_churn --experiments 4"},
            "privacy": "When a model is configured, aggregate data profiles (with up to three example values per column) and scientific evidence are sent to it; the Home agent sends summaries only. Full rows and API keys are never sent. store=false; provider policies still apply."}


@router.get("/api/models", response_model=ModelsSummary)
async def model_overview(s: Services = Depends(services)):
    """Which model serves which purpose, what each purpose may be shown, and the usage so far (no keys)."""
    return await asyncio.to_thread(s.gateway.summary)


@router.post("/api/models/routing", response_model=ModelsSummary)
async def change_routing(body: RoutingChange, request: Request, s: Services = Depends(services)):
    """Set a purpose's shadow (a local model that answers beside it, never used) or move the purpose to another
    tier. Moving it needs a reviewer (X-DCLab-Role, declared until accounts exist) and a reason; moving it back to
    its own tier (tier null) is one setting. Every change is kept in the routing history and the platform audit."""
    from ...models import routing as model_routing

    role = declared_role(request)
    tier = body.tier
    try:
        kept = len(s.gateway.routing.get().get("history") or [])
    except Exception:  # noqa: BLE001 — the change below reports what is wrong with the routing
        kept = None
    try:
        doc = model_routing.change(s.gateway.routing, str(body.purpose or ""), str(body.setting or ""), tier if isinstance(tier, str) and tier else None,
                             role, str(body.reason or ""))
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from None
    except KeyError as exc:
        raise HTTPException(422, str(exc).strip("'\"")) from None
    except model_routing.RoutingError as exc:
        raise HTTPException(422, str(exc)) from None
    from ... import audit

    history = (doc or {}).get("history") or []
    if kept is not None and len(history) > kept:  # a change was made (the same setting again changes nothing and is not audited)
        change = history[-1]
        audit.record(s.audit, "routing", "human", (change.get("by") or role or "person").capitalize(), move="set_routing", status="allowed",
                     args={k: change.get(k) for k in ("purpose", "setting", "from", "to")}, message=change.get("reason") or "")
    return await asyncio.to_thread(s.gateway.summary)


@router.get("/api/research", response_model=Page)
async def research():
    """Every research idea with its champion, experiments, notebooks, evaluation, reports and relations."""
    return research_map.research_map()


@router.get("/api/research/file", response_model=Page)
async def research_file(path: str):
    try:
        return research_map.preview(path)
    except ValueError as error:
        raise HTTPException(404, str(error))


@router.get("/api/studio", response_model=Page)
async def studio_status():
    return {**studio_engine.capabilities(), "industries": list(studio_data.INDUSTRIES) if hasattr(studio_data, "INDUSTRIES") else [], "stages": studio_engine.STAGES}


@router.get("/api/samples", response_model=list[Doc])
async def samples():
    return studio_data.sample_catalog()


@router.get("/api/evidence/{record_id}", response_model=Doc)
async def evidence_record(record_id: str):
    from ...tools import get_record

    record = get_record(record_id)
    if "error" in record:
        raise HTTPException(404, record["error"])
    return record


@router.get("/api/knowledge", response_model=list[Doc])
async def knowledge(s: Services = Depends(services)):
    return s.store.knowledge()


@router.get("/api/workspace", response_model=Workspace)
async def workspace(s: Services = Depends(services)):
    """What Home shows: real counts, the projects with their progress, and what waits for a person."""
    projects, drafts = s.projects, s.drafts

    def summary() -> dict[str, Any]:
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


@router.get("/guide", response_class=FileResponse, responses={200: {"content": {"text/html": {}}}})
async def guide():
    path = ROOT / "evidence/knowledge" / "MODEL_BUILDING_FIELD_GUIDE.html"
    if not path.is_file():
        raise HTTPException(404, "Build the field guide with: .venv/bin/python -m dclab_rnd.master_review build")
    return FileResponse(path, media_type="text/html")


@router.get("/", response_class=FileResponse, responses={200: {"content": {"text/html": {}}}})
async def index():
    # The product frontend (built from dclab_rnd/agentic/web); the earlier UI stays at /classic until the new one covers it.
    page = STATIC / "app" / "index.html"
    if not page.is_file():
        raise HTTPException(404, "Build the frontend with: python -m dclab_rnd.agentic.web.build")
    return FileResponse(page)


@router.get("/classic", response_class=FileResponse, responses={200: {"content": {"text/html": {}}}})
async def classic():
    return FileResponse(STATIC / "index.html")


@router.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)
