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
from ...intern import llm as intern_llm
from ...intern.sessions import DEFAULT_BUDGET
from ...intern.tools import EVIDENCE_TOOLS, Toolbox
from ...studio import graph as studio_graph
from ...studio.engine import QUICK_ROWS
from ...studio.store import STAGE_KEYS
from ..catalog import ROOT

# The same Chat UI commands GET /api/intern returns (tests keep them equal).
CHAT_UI = {"command": "make chat-ui", "intern_command": "make chat-ui-intern",
           "url": "http://localhost:5173/", "intern_url": "http://localhost:5173/?mode=ml-intern"}
STANDALONE_MCP = {"command": "python -m dclab_rnd.mcp_server --port 8777", "make": "make mcp-serve", "url": "http://127.0.0.1:8777/mcp"}
GRAPH_TOOLS = ("get_graph", "check_move")

# Clamps applied by the routes that take these values (tests check them against the routes themselves).
INTERN_CAPS = {"max_steps": (3, 80), "max_minutes": (1, 240)}                        # intern/sessions.py SessionStore.create
DRAFT_DEFAULTS = {"max_rows": 20000, "folds": 3, "calls": 24, "minutes": 20, "eur": 5.0}  # draft/api.py PUT …/settings
DRAFT_CAPS = {"max_rows": (200, 200000), "folds": (2, 10), "calls": (1, 80), "minutes": (1, 240), "eur": (0.0, 1000.0)}
UPLOAD_MAX_BYTES = 200 * 1024 * 1024                                                    # server.py PUT /api/projects/{id}/data


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
    for route in app.routes:
        if not isinstance(route, APIRoute) or not route.include_in_schema or not route.path.startswith("/api/"):
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
def limits(projects) -> dict[str, Any]:
    cfg = intern_llm.settings()
    return {
        "intern_session": {"defaults": dict(DEFAULT_BUDGET), "caps": {k: list(v) for k, v in INTERN_CAPS.items()}, "enforced": True,
                           "note": "The intern stops when either the tool calls or the minutes run out."},
        "draft_settings": {"defaults": DRAFT_DEFAULTS, "caps": {k: list(v) for k, v in DRAFT_CAPS.items()},
                           "note": "Set in the new-project wizard and saved with the project. Rows, quick mode and folds reach the stages; the split follows the solution's time or group column. The call, minute and euro budgets are stored with the project but nothing meters spend yet."},
        "rows": {"quick": QUICK_ROWS, "caps": list(DRAFT_CAPS["max_rows"])},
        "spend": {"tracked": False, "note": "No GPU job or model spend is metered yet, so per-job, per-project and workspace caps are not enforced."},
        "upload_max_bytes": UPLOAD_MAX_BYTES, "connector_max_bytes": connectors.MAX_BYTES,
        "storage": projects.location(),
        "model": {"key_configured": bool(cfg["key_configured"]), "available": bool(cfg["available"]), "endpoint": cfg["endpoint"], "model": cfg["model"]},
        "privacy": [
            {"title": "Data stays on this machine", "on": True,
             "text": "The server answers only on 127.0.0.1 and localhost. Uploads, connector imports, projects and logs are files in the workspace folder."},
            {"title": "What a model sees", "on": bool(cfg["available"]),
             "text": ("A model is configured, so these go to " + cfg["endpoint"] + ": " if cfg["available"] else "No model is configured, so nothing is sent. With one: ")
             + "the Home agent gets the problem, your answers, column summaries (name, kind, missing rate, unique count; 40 columns at a time) "
               "and the descriptive findings (counts and shares), never rows or cell values; "
               "a file that no built-in reader can parse is the one exception: up to 20 of its lines (400 characters each) go to the "
               "model to work out the format, the data step says when that happened, and DCLAB_MODEL_READS_SAMPLE_LINES=0 turns it off; "
               "the intern gets its tool results, and describe_data includes up to three example values per column."},
            {"title": "Scan uploads for personal data", "on": False, "text": "Not built yet: uploads are not scanned for personal data."},
            {"title": "Retention", "on": False, "text": "No timed retention: raw uploads stay until their project or draft is deleted."},
            {"title": "Training the policy model on your projects", "on": False,
             "text": "Nothing is collected automatically. Training examples leave a project only when you export them (GET …/export/sft or python -m dclab_rnd.studio.sft)."},
        ],
    }


# ---------------------------------------------------------------------------- audit log
def _entries(project: dict[str, Any], moves: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ref = {"id": project["id"], "name": project.get("name") or project["id"]}
    out = []
    for m in moves:
        if m.get("move") == "approve_gate" and m.get("status") == "allowed":
            continue  # the approval below carries who, when and the reason
        out.append({"kind": "move", "at": m.get("at"), "actor": m.get("actor"), "who": "Intern" if m.get("actor") == "agent" else "Person",
                    "move": m.get("move"), "args": m.get("args") or {}, "status": m.get("status"), "message": m.get("message"),
                    "outcome": m.get("outcome"), "rules": m.get("rules") or [], "from": m.get("from"), "to": m.get("to"), "project": ref})
    for a in project.get("approvals") or []:
        gate = studio_graph.LEGACY_GATES.get(a.get("gate"), a.get("gate"))
        out.append({"kind": "approval", "at": a.get("at"), "actor": "human", "who": (a.get("by") or "owner").capitalize(),
                    "move": "approve_gate", "args": {"gate": gate}, "status": "allowed", "message": a.get("reason") or "",
                    "outcome": None, "rules": ["DCLAB-R01"] if gate == "solution" else ["DCLAB-R17"], "project": ref})
        if a.get("used"):
            out.append({"kind": "approval_used", "at": a.get("used"), "actor": "agent", "who": "Intern",
                        "move": "use_approval", "args": {"gate": gate}, "status": "allowed", "message": "The final stage used this approval.",
                        "outcome": None, "rules": ["DCLAB-R17"], "project": ref})
    return out


POLICY_NAMES = {"require_solution_signoff": "solution sign-off", "require_holdout_approval": "holdout approval"}


def _policy_entries(project: dict[str, Any], activity: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Gate switch changes from the project's activity log (PATCH /api/projects/{id} writes them)."""
    ref = {"id": project["id"], "name": project.get("name") or project["id"]}
    out = []
    for a in activity:
        if a.get("kind") != "policy_changed":
            continue
        for key, change in ((a.get("payload") or {}).get("changes") or {}).items():
            out.append({"kind": "policy", "at": a.get("at"), "actor": "human", "who": "Person", "move": "set_policy",
                        "args": {"policy": key, "label": POLICY_NAMES.get(key, key), "on": bool((change or {}).get("to"))},
                        "status": "allowed", "message": "", "outcome": None, "rules": [], "project": ref})
    return out


def audit(projects, limit: int, offset: int) -> dict[str, Any]:
    keyed = []
    plist = projects.list()
    for p in plist:
        # (time, position in the project's own log) so moves logged in the same second keep their order, newest first
        entries = _entries(p, projects.transitions(p["id"], 1_000_000)) + _policy_entries(p, projects.activity(p["id"], 1_000_000))
        keyed += [((e.get("at") or "", i), e) for i, e in enumerate(entries)]
    keyed.sort(key=lambda pair: pair[0], reverse=True)
    items = [e for _, e in keyed]
    counts = {"moves": sum(1 for e in items if e["kind"] == "move"), "approvals": sum(1 for e in items if e["kind"] == "approval"),
              "blocked": sum(1 for e in items if e.get("status") == "blocked"), "waiting": sum(1 for e in items if e.get("status") == "needs_approval"),
              "policy_changes": sum(1 for e in items if e["kind"] == "policy")}
    return {"total": len(items), "limit": limit, "offset": offset, "items": items[offset:offset + limit], "projects": len(plist), "counts": counts}


# ---------------------------------------------------------------------------- routes
def register(app, ctx) -> None:
    @app.get("/api/platform/integrations")
    async def platform_integrations(request: Request):
        """MCP endpoint and tools, data connectors, REST routes, make targets and Chat UI commands."""
        return await asyncio.to_thread(integrations, request.app, ctx.projects, str(request.base_url))

    @app.get("/api/platform/policies")
    async def platform_policies():
        """Invariants the workflow validator enforces (checked now), per-project gate switches, and what each actor may do."""
        return await asyncio.to_thread(policies, ctx.projects)

    @app.get("/api/platform/limits")
    async def platform_limits():
        """Budget defaults and caps, upload limits, and what leaves this machine."""
        return await asyncio.to_thread(limits, ctx.projects)

    @app.get("/api/platform/audit")
    async def platform_audit(limit: int = 50, offset: int = 0):
        """Newest validated moves, gate approvals and gate switch changes across every project, newest first."""
        if limit < 1 or limit > 500 or offset < 0:
            raise HTTPException(422, "limit is 1 to 500 and offset is 0 or more")
        return await asyncio.to_thread(audit, ctx.projects, limit, offset)
