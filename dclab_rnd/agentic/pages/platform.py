"""Integrations and Admin: what connects to DCLab and which rules it enforces, read from the code that does it.

    GET /api/platform/integrations   MCP endpoint and tools, connectors, REST routes, make targets, Chat UI commands
    GET /api/platform/policies       invariants the validator enforces (probed now), per-project gate switches, actors
    GET /api/platform/limits         budget defaults and caps, upload limit, what leaves this machine
    GET /api/platform/audit          newest validated moves and gate approvals across every project (paged)

Nothing here is configuration: the numbers come from the modules that apply them, and every invariant
is checked against ``studio.graph.check`` on each request, so the page cannot claim a rule the code
no longer enforces. There are no user accounts; "who" is the validator's actor (a person or the intern).
"""

from __future__ import annotations

import asyncio
import functools
import re
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException, Request
from fastapi.routing import APIRoute
from starlette.routing import Mount

from ... import connectors
from ...models import settings as model_settings
from ...intern.sessions import DEFAULT_BUDGET
from ...intern.tools import EVIDENCE_TOOLS, Toolbox
from ...studio import graph as studio_graph
from ...studio.engine import QUICK_ROWS
from ...studio.store import STAGE_KEYS
from ..catalog import ROOT
from ..api_models import Page, Question, Router, api_routes  # package 10.1

# The same Chat UI commands GET /api/intern returns (tests keep them equal).
CHAT_UI = {"command": "make chat-ui", "intern_command": "make chat-ui-intern",
           "url": "http://localhost:5173/", "intern_url": "http://localhost:5173/?mode=ml-intern"}
STANDALONE_MCP = {"command": "python -m dclab_rnd.mcp_server --port 8777", "make": "make mcp-serve", "url": "http://127.0.0.1:8777/mcp"}
GRAPH_TOOLS = ("get_graph", "check_move")

# Clamps applied by the routes that take these values (tests check them against the routes themselves).
INTERN_CAPS = {"max_steps": (3, 80), "max_minutes": (1, 240)}                        # intern/sessions.py SessionStore.create
DRAFT_DEFAULTS = {"max_rows": 20000, "folds": 3, "calls": 24, "minutes": 20, "eur": 5.0}  # draft/api.py PUT …/settings
DRAFT_CAPS = {"max_rows": (200, 200000), "folds": (2, 10), "calls": (1, 80), "minutes": (1, 240), "eur": (0.0, 1000.0)}
from ...settings import UPLOAD_MAX_BYTES  # noqa: E402 — the limit every upload route enforces (dclab_rnd/settings.py)


# ---------------------------------------------------------------------------- integrations
def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


@functools.lru_cache(maxsize=4)
def _make_targets(path: str, mtime: float) -> tuple[dict[str, str], ...]:
    """``target:  ## help`` lines of the Makefile, with the ``# --- section ---`` they sit under (what ``make help`` prints)."""
    out, section = [], ""
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError:
        return ()
    for line in lines:
        head = re.match(r"^# -+ (.+?) -*$", line)
        if head:
            section = head.group(1).strip()
            continue
        m = re.match(r"^([A-Za-z0-9_.-]+):[^#\n]*?##\s*(.+)$", line)
        if m:
            out.append({"target": m.group(1), "help": m.group(2).strip(), "section": section})
    return tuple(out)


VERIFIED_RE = re.compile(r"in plain mode the model was offered the (\d+) DCLab tools; in ML Intern mode (\d+) tools(?: \(([^)]*)\))?")


@functools.lru_cache(maxsize=4)
def _ml_intern_record(path: str, mtime: float) -> dict[str, Any] | None:
    """The Chat UI verification the README records. ML Intern's own tools live in Chat UI, not in this repository."""
    try:
        m = VERIFIED_RE.search(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None
    if not m:
        return None
    return {"verified_dclab": int(m.group(1)), "verified_total": int(m.group(2)), "breakdown": m.group(3) or ""}


def ml_intern_tools(dclab_now: int) -> dict[str, Any]:
    readme = ROOT / "README.md"
    rec = _ml_intern_record(str(readme), _mtime(readme))
    base = {"source": "README.md, section “Use Hugging Face Chat UI and ML Intern with DCLab”",
            "where": "ML Intern's own tools are compiled into Hugging Face Chat UI (cloned by `make chat-ui-intern`), so this repository cannot count them from source."}
    if not rec:
        return {**base, "verified_total": None, "own": None, "expected_now": None, "current": False,
                "note": "No recorded verification was found in the README."}
    own = rec["verified_total"] - rec["verified_dclab"]
    current = rec["verified_dclab"] == dclab_now
    return {**base, **rec, "own": own, "dclab_now": dclab_now, "expected_now": dclab_now + own, "current": current,
            "note": (f"Verified once: {rec['verified_total']} = {rec['verified_dclab']} DCLab tools + {own} of ML Intern's own." +
                     ("" if current else f" DCLab now serves {dclab_now}, so ML Intern mode should offer {dclab_now + own}; not re-verified."))}


def mcp_tools(projects) -> list[dict[str, Any]]:
    out = []
    for spec in Toolbox(projects).schemas():
        fn = spec["function"]
        name = fn["name"]
        group = "evidence" if name in EVIDENCE_TOOLS else "graph" if name in GRAPH_TOOLS else "project"
        out.append({"name": name, "description": fn["description"], "required": list((fn.get("parameters") or {}).get("required") or []),
                    "arguments": sorted((fn.get("parameters") or {}).get("properties") or {}), "group": group})
    return out


METHOD_ORDER = {"GET": 0, "POST": 1, "PUT": 2, "PATCH": 3, "DELETE": 4}


def rest_routes(app) -> list[dict[str, Any]]:
    """Every public ``/api/`` route on this server (hidden aliases excluded), one row per method."""
    rows = []
    for route in api_routes(app):  # included routers too (package 10.1)
        if not route.include_in_schema or not route.path.startswith("/api/"):
            continue
        doc = (route.endpoint.__doc__ or "").strip().split("\n")[0].strip()
        for method in sorted(route.methods - {"HEAD", "OPTIONS"}, key=lambda m: METHOD_ORDER.get(m, 9)):
            rows.append({"method": method, "path": route.path, "group": route.path.split("/")[2], "summary": doc})
    rows.sort(key=lambda r: (r["group"], r["path"], METHOD_ORDER.get(r["method"], 9)))
    return rows


def mcp_mounted(app) -> bool:
    return any(isinstance(r, Mount) and r.path == "/mcp" for r in app.routes)


def integrations(app, projects, base_url: str) -> dict[str, Any]:
    tools = mcp_tools(projects)
    mounted = mcp_mounted(app)
    rest = rest_routes(app)
    makefile = ROOT / "Makefile"
    status = connectors.status()
    ready = {"kaggle": bool(status["kaggle"]["configured"]), "hf": bool(status["hf"]["configured"]),
             "database": bool(status["database"]["configured"]), "cloud": bool(status["cloud"]["s3"] or status["cloud"]["gcs"])}
    return {
        "mcp": {"mounted": mounted, "url": (base_url.rstrip("/") + "/mcp") if mounted else None,
                "note": None if mounted else "The `mcp` package is not installed in this server's environment: `pip install mcp`, then restart.",
                "standalone": STANDALONE_MCP, "transport": "Streamable HTTP, stateless, JSON responses",
                "tools": tools, "count": len(tools), "ml_intern_tools": ml_intern_tools(len(tools))},
        "connectors": {**status, "ready": ready},
        "rest": rest, "rest_groups": sorted({r["group"] for r in rest}),
        "cli": list(_make_targets(str(makefile), _mtime(makefile))),
        "chat_ui": CHAT_UI,
        "vscode": {"status": "not packaged yet", "engine": "dclab_rnd/notebook_assist.py (cell-level companion) and `python -m dclab_rnd.copilot review`"},
    }


# ---------------------------------------------------------------------------- policies
def _probe(done: tuple[str, ...] = (), **extra: Any) -> dict[str, Any]:
    """A minimal in-memory project for asking the validator; never saved."""
    p = {"id": "probe", "solution": {"target": "y"}, "data": {"sha256": "probe", "filename": "probe.csv"},
         "stages": {s: {"status": "completed" if s in done else "pending"} for s in STAGE_KEYS},
         "holdout_uses": 0, "running": None, "approvals": [], "policy": {}}
    p.update(extra)
    return p


ALL = tuple(STAGE_KEYS)
BEFORE_FINAL = tuple(s for s in STAGE_KEYS if s != "final")


def _signed() -> dict[str, Any]:
    p = _probe()
    p["signoff"] = {"gate": "solution", "solution_hash": studio_graph.solution_hash(p["solution"])}
    return p


def _status(project: dict[str, Any], move: str, actor: str, **args: Any) -> str:
    return studio_graph.check(project, move, actor, **args).status


INVARIANTS: list[dict[str, Any]] = [
    {"id": "holdout-once", "title": "The holdout opens once, after every choice is locked",
     "detail": "Once the final stage has used the holdout, the intern may not run it again: choosing by holdout score makes it optimistic.",
     "rules": ["DCLAB-R17"], "evidence": ["PIT-006"], "where": "dclab_rnd/studio/graph.py · _run_stage, check “Holdout unused”",
     "probe": lambda: _status(_probe(ALL, holdout_uses=1), "run_stage", "agent", stage="final") == "blocked"},
    {"id": "holdout-reason", "title": "A person reruns the final stage only with a written reason",
     "detail": "Without a reason the rerun waits; with one, the reason is recorded next to the use number.",
     "rules": ["DCLAB-R17"], "evidence": ["PIT-006"], "where": "dclab_rnd/studio/graph.py · _run_stage, check “Holdout reuse confirmed”",
     "probe": lambda: (_status(_probe(ALL, holdout_uses=1), "run_stage", "human", stage="final") == "needs_approval"
                       and _status(_probe(ALL, holdout_uses=1), "run_stage", "human", stage="final", reuse_reason="new snapshot") == "allowed")},
    {"id": "person-approves-gates", "title": "Only a person approves a gate",
     "detail": "The intern cannot sign the solution or approve opening the holdout; it can only ask.",
     "rules": ["DCLAB-R01", "DCLAB-R17"], "evidence": [], "where": "dclab_rnd/studio/graph.py · _approve_gate, check “A person approves”",
     "probe": lambda: (all(_status(_probe(), "approve_gate", "agent", gate=g) == "blocked" for g in studio_graph.GATES)
                       and all(_status(_probe(), "approve_gate", "human", gate=g) == "allowed" for g in studio_graph.GATES))},
    {"id": "stages-in-order", "title": "Stages run in order",
     "detail": "Each stage needs the one before it: " + " → ".join(STAGE_KEYS) + ". Skipping ahead is refused for a person and the intern alike.",
     "rules": ["DCLAB-R15"], "evidence": [], "where": "dclab_rnd/studio/graph.py · _run_stage, check “Edge exists”",
     "probe": lambda: all(_status(_probe(("data",)), "run_stage", actor, stage="features") == "blocked" for actor in ("human", "agent"))},
    {"id": "signed-solution", "title": "A signed solution is changed only by a person",
     "detail": "When the owner has signed the current solution, the intern's change waits for a person.",
     "rules": ["DCLAB-R01"], "evidence": [], "where": "dclab_rnd/studio/graph.py · _set_solution, check “Signed solution unchanged”",
     "probe": lambda: _status(_signed(), "set_solution", "agent") != "allowed" and _status(_signed(), "set_solution", "human") == "allowed"},
    {"id": "solution-first", "title": "Nothing trains before a solution and data exist",
     "detail": "Running any stage needs a saved solution (WF-01) and an attached table (WF-02).",
     "rules": ["DCLAB-R01"], "evidence": [], "where": "dclab_rnd/studio/graph.py · _run_stage, checks “Solution” and “Data”",
     "probe": lambda: (_status(_probe(solution=None), "run_stage", "human", stage="data") == "blocked"
                       and _status(_probe(data=None), "run_stage", "human", stage="data") == "blocked")},
    {"id": "no-choice-after-holdout", "title": "The intern cannot change an earlier choice after the holdout is used",
     "detail": "Swapping the recipe or model once the holdout score is known would be choosing by holdout score.",
     "rules": ["DCLAB-R16", "DCLAB-R17"], "evidence": ["PIT-006"], "where": "dclab_rnd/studio/graph.py · _approve_stage, check “Holdout unused”",
     "probe": lambda: _status(_probe(ALL, holdout_uses=1), "approve_stage", "agent", stage="models", choice="other") == "blocked"},
    {"id": "forbidden-columns", "title": "Forbidden columns never reach a model",
     "detail": "Columns the solution forbids are left out of every recipe the stages fit; the leakage stage fits them once, apart, only to measure how much they would inflate the score.",
     "rules": ["DCLAB-R04", "DCLAB-R06"], "evidence": [], "where": "dclab_rnd/expansion/runner.py · recipes exclude spec.blocked_features",
     "probe": None},
]

SWITCH_APPLIES = {
    "require_solution_signoff": "Every run from WF-04 on waits until the current solution is signed, whoever runs it.",
    "require_holdout_approval": "The intern's final-stage run waits for your approval; a person running the stage is not asked.",
}
SWITCH_RULES = {"require_solution_signoff": ["DCLAB-R01"], "require_holdout_approval": ["DCLAB-R17"]}


def _cell(project: dict[str, Any], move: str, actor: str, **args: Any) -> dict[str, Any]:
    v = studio_graph.check(project, move, actor, **args)
    return {"status": v.status, "message": v.message, "failed": [c["name"] for c in v.checks if not c["ok"]]}


ACTOR_ROWS: list[tuple[str, Callable[[], dict[str, Any]], str, dict[str, Any]]] = [
    ("Run the next stage", lambda: _probe(("data",)), "run_stage", {"stage": "leakage"}),
    ("Save or change an unsigned solution", _probe, "set_solution", {}),
    ("Change a signed solution", _signed, "set_solution", {}),
    ("Sign the solution (gate)", _probe, "approve_gate", {"gate": "solution"}),
    ("Approve opening the holdout (gate)", _probe, "approve_gate", {"gate": "holdout"}),
    ("Run the final stage when the project requires holdout approval",
     lambda: _probe(BEFORE_FINAL, policy={"require_holdout_approval": True}), "run_stage", {"stage": "final"}),
    ("Rerun the final stage after the holdout is used", lambda: _probe(ALL, holdout_uses=1), "run_stage", {"stage": "final"}),
    ("Change an earlier choice after the holdout is used", lambda: _probe(ALL, holdout_uses=1), "approve_stage", {"stage": "models", "choice": "other"}),
]


def policies(projects) -> dict[str, Any]:
    invariants = []
    for inv in INVARIANTS:
        holds = None
        if inv["probe"] is not None:
            try:
                holds = bool(inv["probe"]())
            except Exception:  # a probe that crashes is reported as not holding, never hidden
                holds = False
        invariants.append({k: v for k, v in inv.items() if k != "probe"} | {"locked": True, "checked": inv["probe"] is not None, "holds": holds})
    plist = projects.list()
    switches = []
    for key, default in studio_graph.DEFAULT_POLICY.items():
        gate = "solution" if key == "require_solution_signoff" else "holdout"
        rows = [{"id": p["id"], "name": p.get("name") or p["id"], "on": bool(studio_graph.policy(p).get(key))} for p in plist]
        switches.append({"key": key, "gate": gate, "title": studio_graph.GATES[gate].rstrip("."), "applies": SWITCH_APPLIES.get(key, ""),
                         "rules": SWITCH_RULES.get(key, []), "default": bool(default), "on": sum(r["on"] for r in rows), "total": len(rows), "projects": rows})
    actors = [{"action": label, "person": _cell(make(), move, "human", **args), "intern": _cell(make(), move, "agent", **args)}
              for label, make, move, args in ACTOR_ROWS]
    return {"invariants": invariants, "switches": switches, "actors": actors, "gates": studio_graph.GATES,
            "default_policy": studio_graph.DEFAULT_POLICY, "projects": len(plist),
            "accounts": {"users": False, "roles": False, "sign_in": False, "csrf": True,
                         "view_as": "presentation only: it changes which panels and details show and grants or removes no right"}}


# ---------------------------------------------------------------------------- limits and privacy
def _endpoints() -> str:
    """Where each tier sends requests, host only (GET /api/models lists every purpose with its tier)."""
    tiers = [model_settings.public(t) for t in model_settings.TIERS]
    hosts: dict[str, list[str]] = {}
    for t in tiers:
        if t["available"]:
            hosts.setdefault(t["endpoint"], []).append(t["name"])
    return "; ".join(f"{host} ({', '.join(names)} tier{'s' if len(names) > 1 else ''})" for host, names in hosts.items()) or "no endpoint"


def _spend() -> dict[str, Any]:
    from ...models import installed
    from ...models.gateway import workspace_cap
    from ...models.prices import load
    from ...models.usage import month_start

    gateway = installed()
    t = gateway.usage.totals(since=month_start()) if gateway.usage else None
    priced = sorted(load())
    return {"tracked": gateway.usage is not None, "eur_this_month": t["eur"] if t else None, "unpriced_this_month": t["unpriced"] if t else None,
            "workspace_cap": workspace_cap(), "priced_models": priced,
            "note": ("Model requests are metered through the gateway: each project's wizard budget is its monthly cap, an intern session can carry "
                     "its own cap, and DCLAB_WORKSPACE_MONTHLY_EUR caps the workspace. A request that would pass a cap is refused before it is sent. "
                     + ("Prices are configured for: " + ", ".join(priced) + "." if priced else
                        "No model has a configured price yet, so spend is counted in tokens and euro caps cannot stop remote requests; local models cost nothing.")
                     + " GPU jobs are not switched on.")}


def limits(projects) -> dict[str, Any]:
    cfg = model_settings.public("standard")  # the gateway's default tier; GET /api/models lists every tier and purpose
    return {
        "intern_session": {"defaults": dict(DEFAULT_BUDGET), "caps": {k: list(v) for k, v in INTERN_CAPS.items()}, "enforced": True,
                           "note": "The intern stops when either the tool calls or the minutes run out."},
        "draft_settings": {"defaults": DRAFT_DEFAULTS, "caps": {k: list(v) for k, v in DRAFT_CAPS.items()},
                           "note": "Set in the new-project wizard and saved with the project. Rows, quick mode and folds reach the stages; the split follows the solution's time or group column. The euro budget is the project's monthly cap on priced model requests."},
        "rows": {"quick": QUICK_ROWS, "caps": list(DRAFT_CAPS["max_rows"])},
        "spend": _spend(),
        "upload_max_bytes": UPLOAD_MAX_BYTES, "connector_max_bytes": connectors.MAX_BYTES,
        "storage": projects.location(),
        "model": {"key_configured": bool(cfg["key_configured"]), "available": bool(cfg["available"]), "endpoint": cfg["endpoint"], "model": cfg["model"]},
        "privacy": [
            {"title": "Data stays on this machine", "on": True,
             "text": "The server answers only on 127.0.0.1 and localhost. Uploads, connector imports, projects and logs are files in the workspace folder."},
            {"title": "What a model sees", "on": bool(cfg["available"]),
             "text": ("A model is configured, so these go to " + _endpoints() + ": " if cfg["available"] else "No model is configured, so nothing is sent. With one: ")
             + "the Home agent gets the problem, your answers, column summaries (name, kind, missing rate, unique count; 40 columns at a time) "
               "and the descriptive findings (counts and shares), never rows or cell values; "
               "a file that no built-in reader can parse is the one exception: up to 20 of its lines (400 characters each) go to the "
               "model to work out the format, the data step says when that happened, and DCLAB_MODEL_READS_SAMPLE_LINES=0 turns it off; "
               "the intern gets its tool results, and describe_data includes up to three example values per column."},
            {"title": "Scan uploads for personal data", "on": False, "text": "Not built yet: uploads are not scanned for personal data."},
            {"title": "Retention", "on": False, "text": "No timed retention: raw uploads stay until their project or draft is deleted."},
            {"title": "Training the policy model on your projects", "on": False,
             "text": "Off for every project until its owner opts it in (the Policy model page; settings.share_for_training). Nothing is collected "
                     "automatically: trajectory records leave only opted-in projects, and only when exported (python -m dclab_rnd.studio.sft --trajectories), "
                     "without cell values or free text. Stage examples leave a project only when you export them (GET …/export/sft)."},
        ],
    }


# ---------------------------------------------------------------------------- audit log
STATUSES = ("allowed", "blocked", "needs_approval")


def audit_log(ctx) -> Any:
    """The workspace's audit (package 10.4): the server's, or the one the project store belongs to."""
    from ... import audit

    return getattr(ctx, "audit", None) or audit.for_store(ctx.projects)


# ---------------------------------------------------------------------------- routes
def register(app, ctx) -> None:
    router = Router()  # package 10.1: this page's routes are one router
    @router.get("/api/platform/integrations", response_model=Page)
    async def platform_integrations(request: Request):
        """MCP endpoint and tools, data connectors, REST routes, make targets and Chat UI commands."""
        return await asyncio.to_thread(integrations, request.app, ctx.projects, str(request.base_url))

    @router.get("/api/platform/policies", response_model=Page)
    async def platform_policies():
        """Invariants the workflow validator enforces (checked now), per-project gate switches, and what each actor may do."""
        return await asyncio.to_thread(policies, ctx.projects)

    @router.get("/api/platform/limits", response_model=Page)
    async def platform_limits():
        """Budget defaults and caps, upload limits, and what leaves this machine."""
        return await asyncio.to_thread(limits, ctx.projects)

    @router.get("/api/platform/audit", response_model=Page)
    async def platform_audit(limit: int = 50, offset: int = 0, kind: str | None = None, status: str | None = None,
                             actor: str | None = None, project: str | None = None):
        """The workspace's audit trail, newest first: one query on the append-only table (package 10.4), with filters."""
        from ... import audit

        if limit < 1 or limit > 500 or offset < 0:
            raise HTTPException(422, "limit is 1 to 500 and offset is 0 or more")
        for name, value, allowed in (("kind", kind, audit.KINDS), ("status", status, STATUSES), ("actor", actor, ("human", "agent"))):
            if value not in (None, "") and value not in allowed:
                raise HTTPException(422, f"{name} is one of {', '.join(allowed)}")
        if project not in (None, "") and not re.fullmatch(r"[0-9a-zA-Z_-]{1,64}", project):
            raise HTTPException(422, "project is a project id")
        return await asyncio.to_thread(audit.page, audit_log(ctx), ctx.projects, limit, offset,
                                       kind=kind, status=status, actor=actor, project_id=project)

    app.include_router(router)
