"""DCLab's agents behind an external chat (phase A7, docs/guides/AGENTIC_FOUNDATION_PLAN.md): the registry's ``chat`` scope.

An external chat (DCLab Code, Hugging Face Chat UI) reaches DCLab over ``/mcp``. The project tools there let its model
drive the notebook itself; these let it hand a whole job to one of DCLab's own agents, as Claude Code hands one to a
subagent. Each tool runs the agent exactly as the product does (the same objects the routes call, in the workspace's
services), waits for its turn, and returns what a person would see on the page: the agent's words, its open questions,
the pack, the table, the workflow, and the project once built. The agent keeps its own guards, gates, ledger and
traces; the MCP call itself is kept whole (``mcp/calls.jsonl``).

A7.2, the Home agent: ``home_start``, ``home_message``, ``home_add_data``, ``home_status``, ``home_build`` and
``home_list``. Each goes through the page's own route inside this app (its checks, its background turn and the twin
beside it, its refusals as statuses), then waits, without holding a thread, until the agent's turn is over (a turn
longer than ``TURN_WAIT_S`` is said, to read later with ``home_status``). Before a build the Home agent answers; after
it, the Home orchestrator and its specialists (as on the project's chat). ``home_add_data`` gives a draft its table as
the page does (a sample, or a file on the server's machine when it runs without sign-in; a sealed or holdout set, a
hidden or secret file and DCLab's own folders are refused). ``home_build`` sends the person's words asking to build:
the build ends with the holdout read once, so it runs only on words the Home agent reads as an order to build.

A7.3, every capability of the product: ``dclab_capabilities`` lists the product server's operations (its own OpenAPI
description: method, path, what it does, its parameters and body, and what a chat may do with it), and ``dclab_call``
runs one inside the same app, on the server's own loop, with every check a page's request passes (the request token, the
role of the caller's API token when accounts are on). Each call is matched to its route as the app matches it (a path
with %-escapes, # or ? is refused), then judged by the route's template: reads run, except those returning a table's
rows, a file holding data or a stream; writes run only from ``AGENT_WRITES`` (agent work and proposals). Every other
write is the person's (routes record their writes as the person's and their stage runs as a human's move): approvals and
gates, choices and labels, consent and policy, deletes, uploads, builds. Whatever the routes record during an MCP call is
the agent's (``audit.record``) and lies in the chat's visit (surface ``mcp``).

The text an external chat sends is its model's, written for the person: everything an MCP call records, and the turns
it starts, carries the surface ``mcp`` (``experience.via``), and the conversations set leaves such a draft out: a
person's words are kept apart from another model's.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import re
import time
from pathlib import Path
from typing import Any

from .agents.registry import Registry, Tool

SCOPE = "chat"
CHAT_MOVES = ("agent.call",)  # every write here hands a job to one of DCLab's agents, whose own guard checks what it writes
MAX_TEXT = 4000
SHOWN = 12  # the last messages a status shows
DATA_FILE = (".csv", ".tsv", ".txt", ".parquet", ".json", ".jsonl", ".xlsx", ".xls")
PIPELINE_WAIT_S = 180  # a table's pipeline and the agent's reading of it
TURN_WAIT_S = 120  # an agent's turn a tool waits for; longer, the view says it is still working (home_status reads it later)
POLL_S = 0.4
FAILED_GRACE_S = 10  # a failed table the agent has not spoken about yet
METHODS = ("GET", "POST", "PUT", "PATCH", "DELETE")
# A7.3: the writes a chat may run, by route template: agent work and proposals, each with the route's own checks; the
# body keys after each are the person's part of it (the team walk's approval of its holdout gate). Every other write is
# the person's: an approval or a gate, a choice or a label, consent and policy, a delete, an upload, a build (home_build),
# a stage run the route records as the person's (the project tools run them as the agent). Reads run, but not those
# that return a table's rows, a file holding data, or a stream.
AGENT_WRITES: dict[tuple[str, str], tuple[str, ...]] = {
    ("POST", "/api/connectors/kaggle/search"): (),
    ("POST", "/api/decide"): (),
    ("POST", "/api/drafts"): ("intent",),  # H2 (the review): the type of help is the person's choice, never a chat's
    ("POST", "/api/drafts/{draft_id}/data/sample"): (),
    ("POST", "/api/drafts/{draft_id}/data/synthetic/design"): (),
    ("POST", "/api/drafts/{draft_id}/map"): ("redraw",),  # a redraw drops the person's edits and confirmation
    ("POST", "/api/drafts/{draft_id}/team"): ("approve", "answer"),  # the understanding walk: its answers, map confirmation and build approval are the person's
    ("POST", "/api/drafts/{draft_id}/solution/proposal"): (),
    ("POST", "/api/evidence/ask"): (),
    ("POST", "/api/intern/sessions"): (),
    ("POST", "/api/intern/sessions/{session_id}/message"): (),
    ("POST", "/api/jobs/{job_id}/stop"): (),
    ("POST", "/api/projects/{project_id}/ask"): (),
    ("POST", "/api/projects/{project_id}/data-value"): (),
    ("POST", "/api/projects/{project_id}/decision/what-if"): (),
    ("POST", "/api/projects/{project_id}/graph/check"): (),
    ("POST", "/api/projects/{project_id}/lessons"): (),
    ("POST", "/api/projects/{project_id}/solution/proposal"): (),
    ("POST", "/api/projects/{project_id}/team"): ("approve", "answer"),  # its holdout gate and its drop question are the person's
}
READ_REFUSED = re.compile(r"/(rows|download|events)$|/export/(code|notebook|sft)$|/trials/\{trial_id\}/\{filename\}$|^/api/config$|^/api/auth/")
# Never in a chat's answer whatever the route: the page's request token and any key (a chat with them could act as the person)
SECRET_KEYS = re.compile(r"^(csrf|token|api_?key|secret|password)$", re.I)


def _scrub(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: ("(hidden)" if SECRET_KEYS.match(str(k)) and isinstance(v, str) else _scrub(v)) for k, v in x.items()}
    if isinstance(x, list):
        return [_scrub(v) for v in x]
    return x
READ_REFUSED_WHY = "This operation returns a table's rows, a file holding data, or a stream: a chat never reads it."
PERSONS_WHY = ("This operation is the person's (an approval or a gate, a choice or a label, consent or policy, a delete, an upload, "
               "a build, or a run the route records as theirs): they do it on its DCLab page. Stages run through the project tools "
               "(run_stage, run_all), a build through home_build.")
RESULT_MAX = 60_000  # characters of a result returned whole; a longer one is kept in the workspace and its start returned
CALL_WAIT_S = 100  # a write still working after this (a team walk running its stages) goes on; the chat is told how to follow it
_GOING_ON: set[asyncio.Task] = set()  # those writes, held until they end (a task nobody holds may be collected)
AUTHORIZATION: contextvars.ContextVar[str | None] = contextvars.ContextVar("dclab_chat_authorization", default=None)  # the MCP caller's token


def register(registry: Registry) -> None:
    for tool in TOOLS:
        registry.register(tool)
    registry.guards[SCOPE] = ChatGuard()


class ChatGuard:
    """The chat scope's writes are calls of DCLab's own agents: each agent checks its own writes (the draft guard, the
    Home orchestrator's guard, the graph's validator). This guard only refuses a call without the workspace's services."""

    def __call__(self, tool: Tool, context: Any, arguments: dict[str, Any], proceed):
        if context is None or getattr(context, "drafts", None) is None:
            return {"error": "DCLab's agents need the workspace's services: call them through the product server's or the standalone /mcp."}
        return proceed()


def _draft(services: Any, draft_id: str) -> dict[str, Any] | None:
    try:
        return services.drafts.get(draft_id)
    except (KeyError, FileNotFoundError):
        return None


def _busy(services: Any, draft_id: str) -> bool:
    """Whether an agent turn holds this draft now: one queued by a route in this process, or the turn lock on any
    instance (the routes' own check, ``draft.api.turn_held``: on PostgreSQL an advisory lock that is only probed)."""
    from .draft.api import turn_held

    task = (getattr(services, "draft_tasks", None) or {}).get(draft_id + ":agent")
    return (task is not None and not task.done()) or turn_held(services.drafts, draft_id)


def view(draft: dict[str, Any], since: int | None = None) -> dict[str, Any]:
    """What the Home page shows of a draft, for a chat: the messages after ``since`` (else the last few), its open
    questions with their options (as the page's buttons: the Home agent's own words, which never hold a cell value),
    pack, table, understanding, workflow and project, and what to do next."""
    messages = draft.get("messages") or []
    shown = messages[since:] if since is not None else messages[-SHOWN:]
    asset = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset")), None)
    open_questions = [{"field": q.get("field"), "question": str(q.get("text") or q.get("question") or "")[:600],
                       "options": [str(o)[:120] for o in q.get("options") or []][:12]}
                      for q in draft.get("questions") or [] if not q.get("answered")]
    nodes = ((draft.get("workflow") or {}).get("nodes") or [])[:12]
    built = draft.get("project_id")
    if built:
        nxt = (f"Built as project {built}: talk to its orchestrator with home_message, or use the project tools "
               "(get_results, run_stage, ask_project) with this project_id.")
    elif open_questions:
        nxt = "Put the open questions to the person and send their answers with home_message."
    elif asset is None:
        nxt = "There is no table yet: give one with home_add_data, or the person asks the agent to simulate one (home_message)."
    else:
        nxt = "When the person asks to build, call home_build with their words (it reads the holdout once, at the end)."
    return {
        "draft_id": draft.get("id"), "status": draft.get("status"), "project_id": built,
        "problem": str(draft.get("problem") or "")[:600], "pack": (draft.get("pack") or {}).get("key"),
        "messages": [{"role": m.get("role"), "kind": m.get("kind"), "text": str(m.get("text") or "")[:MAX_TEXT],
                      **({"card": _card(m["card"])} if isinstance(m.get("card"), dict) else {})} for m in shown],
        "message_count": len(messages),
        "open_questions": open_questions,
        "table": {k: asset.get(k) for k in ("name", "rows", "status") if k in asset} if asset else None,
        "understanding": {k: v for k, v in (draft.get("understanding") or {}).items() if isinstance(v, (str, int, float, bool)) and len(str(v)) < 600},
        "workflow": [{"label": n.get("label"), "detail": str(n.get("detail") or "")[:300]} for n in nodes if n.get("label")],
        "next": nxt,
    }


def _card(card: dict[str, Any]) -> dict[str, Any]:
    """A specialist's result card as the page shows it (live, 2026-10-25: the orchestrator answered with cards and no text,
    and the chat saw nothing): its title, facts and table, cut short. Cards never hold a table's row (the orchestrator's rule)."""
    def short(v: Any) -> Any:
        return v if isinstance(v, (int, float, bool)) or v is None else str(v)[:200]

    labels = {c.get("key"): c.get("label") or c.get("key") for c in card.get("columns") or [] if isinstance(c, dict)}
    rows = [{labels.get(k, k): short(v) for k, v in r.items()} for r in (card.get("rows") or [])[:12] if isinstance(r, dict)]
    return {"agent": short(card.get("agent")), "title": short(card.get("title")), "subtitle": short(card.get("subtitle")) if card.get("subtitle") else None,
            "facts": [[short(f[0]), short(f[1])] for f in (card.get("facts") or [])[:16] if isinstance(f, (list, tuple)) and len(f) == 2],
            **({"rows": rows, "rows_total": len(card.get("rows") or [])} if rows else {})}


async def _settle(services: Any, draft_id: str, since: int | None, until=None, wait_s: float | None = None) -> dict[str, Any]:
    """Wait (without holding a thread) until no agent turn holds the draft and ``until(draft)`` is true, then return the
    page's view; past ``wait_s`` (TURN_WAIT_S) the view says the agent is still working, to read later with home_status."""
    deadline = time.monotonic() + (TURN_WAIT_S if wait_s is None else wait_s)
    while True:
        draft = await asyncio.to_thread(_draft, services, draft_id)
        if draft is None:
            return {"error": f"The draft {draft_id!r} is gone (deleted while the agent worked)."}
        busy = await asyncio.to_thread(_busy, services, draft_id)
        if not busy and (until is None or until(draft)):
            return {**view(draft, since), "answering": False}
        if time.monotonic() >= deadline:
            return {**view(draft, since), "answering": True,
                    "note": "The agent is still working on this draft: read it with home_status in a moment."}
        await asyncio.sleep(POLL_S)


def _refusal(status: int, data: Any) -> dict[str, Any]:
    detail = data.get("detail") if isinstance(data, dict) else data
    if isinstance(detail, dict):
        detail = detail.get("message") or detail
    return {"error": str(detail)[:2000], "status": status}


async def home_start(services: Any, problem: str, pack: str | None = None) -> dict[str, Any]:
    """POST /api/drafts, the page's own route (its checks, its background turn, the twin beside it), then the agent's first turn."""
    status, data = await _route(services, "POST", "/api/drafts", body={"problem": str(problem).strip(), "pack": pack})
    if status != 201:
        return _refusal(status, data)
    # the agent's opening message, not only a free turn: a loaded machine may read the draft before its first turn begins
    return await _settle(services, data["id"], 0, until=lambda d: any(m.get("role") == "agent" for m in d.get("messages") or []))


async def home_message(services: Any, draft_id: str, text: str) -> dict[str, Any]:
    """POST /api/drafts/{id}/messages: the agent (or, on a built draft, the Home orchestrator) answers as on the page.
    Words the Home agent would read as an order to build (or a yes to its offer to build) are refused here: a build
    goes through home_build, the one call a chat's host asks the person about before it reads the holdout."""
    from .draft.chat import asks_to_build

    draft = await asyncio.to_thread(_draft, services, draft_id)
    if draft is None:
        return {"error": f"No draft {draft_id!r}. Start one with home_start, or list them with home_list."}
    if not draft.get("project_id") and asks_to_build([*(draft.get("messages") or []), {"role": "user", "text": str(text)}]):
        return {"error": "These words ask to build: call home_build with them. A build ends by reading the holdout once, so it is "
                         "its own call, which the person approves."}
    return await _send(services, draft, str(text))


async def _send(services: Any, draft: dict[str, Any], text: str) -> dict[str, Any]:
    draft_id = draft["id"]
    since = len(draft.get("messages") or [])
    status, data = await _route(services, "POST", f"/api/drafts/{draft_id}/messages", body={"text": str(text)[:MAX_TEXT]})
    if status != 202:
        return _refusal(status, data)  # 409: a turn is under way; 422: nothing written
    return await _settle(services, draft_id, since, until=lambda d: len(d.get("messages") or []) > since)


SEALED_PART = re.compile(r"(^|[^a-z])(sealed|hold[-_ ]?out|held[-_ ]?out)([^a-z]|$)")  # a whole word: "unsealed" is not one
# A key or credential file by its name (never a table named kaggle_house_prices.csv or tokens.parquet)
SECRET_FILE = re.compile(r"^(kaggle\.json|credentials(\.json)?|client_secrets?\.json|secrets?\.(json|ya?ml|toml)|token\.json|"
                         r"service[-_]account.*\.json|\.env(\..*)?|.*\.(pem|key|p12|pfx)|id_(rsa|dsa|ecdsa|ed25519)(\.pub)?)$", re.I)


def _inside(file: Path, root: Path) -> bool:
    """Whether the file lies in the folder, compared by the file system's own identity (macOS ignores letter case;
    /tmp is /private/tmp), on the path as given and as resolved."""
    try:
        target = root.stat()
    except OSError:
        return False
    for parent in {*file.absolute().parents, *file.resolve().parents}:
        try:
            st = parent.stat()
        except OSError:
            continue
        if (st.st_dev, st.st_ino) == (target.st_dev, target.st_ino):
            return True
    return False


def _path_refusal(services: Any, file: Path) -> str | None:
    """Why a file path may not be read for a chat, or None. The path is read on the server's machine, as the server's
    user: only on a machine without sign-in (one owner), never inside DCLab's own folders, never a hidden or secret file,
    never a sealed or holdout set (read once, on purpose)."""
    settings = getattr(services, "settings", None)
    if str(getattr(settings, "auth", "none") or "none") != "none":
        return "With sign-in on, a path would be read on the server's machine: give the table on Home (upload) instead."
    absolute, resolved = file.absolute(), file.resolve()
    for part in {*absolute.parts, *resolved.parts}:
        if SEALED_PART.search(part.lower()):
            return "That file is named as a sealed or holdout set: it is read once, on purpose, never sent through a chat."
    if any(part.startswith(".") and part not in (".", "..") for part in (*absolute.parts[1:], *resolved.parts[1:])):
        return "That file is in a hidden folder (where keys and settings live): copy the table somewhere visible first."
    if SECRET_FILE.match(resolved.name) or SECRET_FILE.match(absolute.name):
        return "That file's name reads as a key or a credential: it is never taken as a table."
    roots = {Path(str(getattr(settings, "agent_home", "") or "")).expanduser(), Path(str(getattr(services.store, "home", "") or ""))}
    if any(_inside(file, root) for root in roots if str(root) not in ("", ".")):
        return "That file is inside DCLab's own workspace folders: their tables are reached by their drafts and projects, never by path."
    if file.suffix.lower() not in DATA_FILE:
        return f"Not a data file ({', '.join(DATA_FILE)})."
    return None


async def home_add_data(services: Any, draft_id: str, sample: str | None = None, path: str | None = None) -> dict[str, Any]:
    """The draft's table through the page's own routes (a sample: POST .../data/sample; a file: PUT .../data with its
    bytes), so their checks hold (the job room, the day's allowance, the size); then waits until the Home agent has
    read the result (its message on that table) and its turn is over."""
    draft = await asyncio.to_thread(_draft, services, draft_id)
    if draft is None:
        return {"error": f"No draft {draft_id!r}."}
    if bool(sample) == bool(path):
        return {"error": "Give one of sample (a sample key, see list_samples) or path (a data file on this machine)."}
    since = len(draft.get("messages") or [])
    if sample:
        status, data = await _route(services, "POST", f"/api/drafts/{draft_id}/data/sample", body={"key": str(sample)})
    else:
        file = Path(str(path)).expanduser()
        if not file.is_file():
            return {"error": f"No file at {path!r}."}
        refused = _path_refusal(services, file)
        if refused:
            return {"error": refused}
        from .draft.api import MAX_UPLOAD

        if file.stat().st_size > MAX_UPLOAD:
            return {"error": "Files up to 200 MB; for larger tables connect the source on Home instead."}
        content = await asyncio.to_thread(file.read_bytes)
        status, data = await _route(services, "PUT", f"/api/drafts/{draft_id}/data", query={"filename": file.name}, content=content)
    if status != 200 or not isinstance(data, dict) or not data.get("id"):
        return _refusal(status, data)
    asset_id, started = data["id"], time.monotonic()

    def read(d: dict[str, Any]) -> bool:  # the agent's message on this table (ready or failed), or a failure it never said
        if any(m.get("asset") == asset_id for m in (d.get("messages") or [])[since:]):
            return True
        now = next((a for a in d.get("assets") or [] if a.get("id") == asset_id), {})
        return now.get("status") in ("failed", "error") and time.monotonic() - started > FAILED_GRACE_S

    out = await _settle(services, draft_id, since, until=read, wait_s=PIPELINE_WAIT_S)
    if "error" in out:
        return out
    now = next((a for a in (await asyncio.to_thread(_draft, services, draft_id) or {}).get("assets") or [] if a.get("id") == asset_id), {})
    return {**out, "table_status": now.get("status")}


async def home_status(services: Any, draft_id: str) -> dict[str, Any]:
    draft = await asyncio.to_thread(_draft, services, draft_id)
    if draft is None:
        return {"error": f"No draft {draft_id!r}. List them with home_list."}
    return {**view(draft), "answering": await asyncio.to_thread(_busy, services, draft_id)}


SHOWN_KEYS = ("problem", "target", "prediction_moment", "entity", "constraints", "costs", "budget", "active_asset")


def _changed_since(draft: dict[str, Any], shown: Any) -> list[str]:
    """The fields a host showed the person (``shown``) that the draft no longer holds: a yes to an older card builds nothing."""
    if not isinstance(shown, dict):
        return []
    u = draft.get("understanding") or {}
    now = {"problem": draft.get("problem"), "active_asset": draft.get("active_asset"), **{k: u.get(k) for k in SHOWN_KEYS if k not in ("problem", "active_asset")}}
    norm = lambda v: None if v in (None, "") else str(v)  # noqa: E731
    return [k for k in SHOWN_KEYS if k in shown and norm(shown[k]) != norm(now.get(k))]


async def home_build(services: Any, draft_id: str, person_words: str, shown: Any = None) -> dict[str, Any]:
    """The person's own "build it", sent as their message: the Home agent reads it as it reads Home's (``asks_to_build``)
    and builds with every check of the step-by-step path, or says what is missing. The build ends with the holdout read
    once, so it runs only on words that ask for it, kept in the draft's chat. ``shown``: what the host's card showed the
    person (the draft's problem, understanding and table); a draft changed since then is refused."""
    from .draft.chat import asks_to_build

    from . import experience

    draft = await asyncio.to_thread(_draft, services, draft_id)
    if draft is None:
        return {"error": f"No draft {draft_id!r}."}
    changed = _changed_since(draft, shown)
    if changed:
        return {"error": f"The draft changed since the person's card ({', '.join(changed)}): show them the draft as it is now and ask again."}
    words = str(person_words or "").strip()
    before = draft.get("project_id")
    # A built draft: the person's yes to the orchestrator's planned new version (it checks its own offer and the yes).
    if not before and not asks_to_build([*(draft.get("messages") or []), {"role": "user", "text": words}]):
        return {"error": "These words do not ask to build (for example \"build it\", or \"yes\" to the agent's offer to build). "
                         "Ask the person; send anything else with home_message."}
    with experience.chat_build():  # the one build a chat may start: its host asked the person (the turn it starts inherits it)
        out = await _send(services, draft, words)
    if "error" in out:
        return out
    after = (await asyncio.to_thread(_draft, services, draft_id) or {}).get("project_id")
    if before:  # the orchestrator builds a new version as a new draft and project; this one stays as it was
        return {**out, "project_id": before, "note": "A new version, once built, is a new draft: see home_list."}
    return {**out, "project_id": after, "built": bool(after)}


async def home_list(services: Any, limit: int = 20) -> dict[str, Any]:
    rows = await asyncio.to_thread(services.drafts.list, limit=max(1, min(int(limit or 20), 50)))
    return {"drafts": [{"draft_id": d.get("id"), "problem": str(d.get("problem") or "")[:200], "status": d.get("status"),
                        "project_id": d.get("project_id"), "updated": d.get("updated")} for d in rows]}


def _operations(services: Any) -> list[dict[str, Any]]:
    spec = services.app.openapi()  # FastAPI keeps it once built
    schemas = (spec.get("components") or {}).get("schemas") or {}
    out = []
    for path, item in (spec.get("paths") or {}).items():
        if not path.startswith("/api/"):
            continue
        for method, op in item.items():
            if method.upper() not in METHODS or not isinstance(op, dict):
                continue
            doc = " ".join(str(op.get("description") or "").split())
            params = [{"name": prm.get("name"), "in": prm.get("in"), "required": bool(prm.get("required")),
                       "type": (prm.get("schema") or {}).get("type")} for prm in op.get("parameters") or []]
            body = None
            content = ((op.get("requestBody") or {}).get("content") or {})
            schema = (content.get("application/json") or {}).get("schema") if content else None
            if schema is not None:
                ref = schema.get("$ref") or next((x.get("$ref") for x in schema.get("anyOf") or [] if isinstance(x, dict) and x.get("$ref")), None)
                target = schemas.get(ref.rsplit("/", 1)[-1], {}) if ref else schema
                props = target.get("properties") or {}
                body = {"fields": {k: (v.get("type") or ("one of" if v.get("anyOf") else "object")) for k, v in props.items()},
                        "required": target.get("required") or []} if props else "any JSON object (the description says what it reads)"
            elif content:
                body = f"raw ({', '.join(content)})"
            out.append({"method": method.upper(), "path": path, "what": (op.get("summary") or "").strip(),
                        "description": doc[:600], "parameters": params, "body": body,
                        "chat": "never: rows, files or a stream" if _refusal_of(method.upper(), path, None) == READ_REFUSED_WHY
                        else "the person's, on its page" if _refusal_of(method.upper(), path, None) else "runs"})
    return out


def dclab_capabilities(services: Any, query: str | None = None, method: str | None = None, limit: int = 40) -> dict[str, Any]:
    ops = _operations(services)
    words = [w for w in re.split(r"[^a-z0-9]+", str(query or "").lower()) if len(w) > 1]
    if method:
        ops = [o for o in ops if o["method"] == str(method).upper()]
    if words:
        def score(o):
            hay = " ".join([o["path"], o["what"], o["description"]]).lower()
            return sum(hay.count(w) for w in words) + 3 * sum(w in o["path"].lower() for w in words)
        ops = sorted((o for o in ops if score(o)), key=score, reverse=True)
    total = len(ops)
    shown = ops[: max(1, min(int(limit or 40), 100))]
    return {"operations": shown, "shown": len(shown), "matching": total,
            "how": "Run one whose chat field is \"runs\" with dclab_call(method, path, query, body), every {name} filled. "
                   "\"the person's\" operations (approvals, gates, choices and labels, consent and policy, deletes, uploads, builds, "
                   "stage runs as the person) are theirs, on their DCLab page; stages run through the project tools (run_stage, "
                   "run_all) and a build through home_build."}


async def _request(services: Any, method: str, path: str, query: dict[str, Any] | None = None, body: Any = None,
                   content: bytes | None = None) -> Any:
    """One request to the product's own route, inside this app and on its loop, with the checks a page's request passes:
    the caller's API token when accounts are on (its role checked by the route), else the server's request token."""
    import httpx

    from . import observe

    headers = {"accept": "application/json"}
    token = AUTHORIZATION.get()
    if token:
        headers["authorization"] = token  # accounts on: the caller's own API token, its role checked by the route as any request's
    else:
        headers["x-dclab-token"] = services.csrf  # the local owner, without sign-in: the server's own request token, as its page sends
    rid = observe.REQUEST_ID.get()
    if rid:
        headers["x-request-id"] = rid  # the route's events group under the MCP call that asked for them
    from . import experience

    visit = experience.bound().get("visit")
    if visit:
        headers["cookie"] = f"dclab_visit={visit}"  # the MCP call's own visit: the routes it runs start no visit of their own
    kwargs: dict[str, Any] = {"params": {k: v for k, v in (query or {}).items() if v is not None}, "headers": headers}
    if content is not None:
        kwargs["content"] = content
        headers["content-type"] = "application/octet-stream"
    elif body is not None and method != "GET":
        kwargs["json"] = body
    transport = httpx.ASGITransport(app=services.app)
    async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1", timeout=600) as client:
        return await client.request(method, path, **kwargs)


async def _route(services: Any, method: str, path: str, query: dict[str, Any] | None = None, body: Any = None,
                 content: bytes | None = None) -> tuple[int, Any]:
    """A route's status and JSON body (or its text)."""
    response = await _request(services, method, path, query, body, content)
    try:
        return response.status_code, (response.json() if response.content else None)
    except ValueError:
        return response.status_code, response.text


def _template(services: Any, method: str, path: str) -> str | None:
    """The operation a request reaches, from the app's own API description: the most specific path template
    (``/api/projects/{project_id}/team`` before ``/api/projects/{project_id}/{name}``) whose pattern the whole path
    fills, for this method. A route left out of the description is no operation for a chat."""
    best: tuple[int, str] | None = None
    for template, item in (services.app.openapi().get("paths") or {}).items():
        if method.lower() not in item:
            continue
        pattern = "".join(("[^/]+" if not part.endswith(":path}") else ".+") if part.startswith("{") else re.escape(part)
                          for part in re.split(r"(\{[^}]+\})", template))
        if re.fullmatch(pattern, path):
            literal = len(re.sub(r"\{[^}]+\}", "", template))
            if best is None or literal > best[0]:
                best = (literal, template)
    return best[1] if best else None


def _refusal_of(method: str, template: str, body: Any) -> str | None:
    """Why a chat may not run this operation, or None: by the route's template, never by the raw path."""
    if method == "GET":
        return READ_REFUSED_WHY if READ_REFUSED.search(template) else None
    keys = AGENT_WRITES.get((method, template))
    if keys is None:
        return PERSONS_WHY
    if isinstance(body, dict) and any(k in body for k in keys):
        return f"{', '.join(k for k in keys if k in body)}: that part of this operation is the person's, on its DCLab page."
    return None


async def dclab_call(services: Any, method: str, path: str, query: dict[str, Any] | None = None, body: Any = None) -> dict[str, Any]:
    method = str(method).upper()
    if method not in METHODS:
        return {"error": f"method must be one of {', '.join(METHODS)}"}
    if (not path.startswith("/api/") or ".." in path or "//" in path or any(c in path for c in "%#?\\")
            or any(seg == "." for seg in path.split("/"))):  # a dot segment is dropped by the client: judged and sent would differ
        return {"error": "path must be one of the product's /api/ operations as written (no %-escapes, #, ? or ..: put the query "
                         "in query); dclab_capabilities lists them"}
    if "{" in path or "}" in path:
        return {"error": f"fill every {{name}} in the path first: {path}"}
    template = _template(services, method, path)
    if template is None:
        return {"error": f"No {method} operation at {path}: dclab_capabilities lists them."}
    refused = _refusal_of(method, template, body)
    if refused:
        return {"error": refused, "operation": f"{method} {template}"}
    # The request runs as its own task (with this call's context: its channel, visit and request id), so a long write is
    # never cut halfway when the chat's own call times out or is cancelled; a read is waited for.
    task = asyncio.ensure_future(_request(services, method, path, query, body))
    _GOING_ON.add(task)
    task.add_done_callback(_ended)
    done, _ = await asyncio.wait({task}, timeout=None if method == "GET" else CALL_WAIT_S)
    if not done:
        return {"status": 202, "ok": True, "running": True, "operation": f"{method} {template}",
                "says": f"Still working after {CALL_WAIT_S:g} s: DCLab goes on with it, so never send it again. {_follow(template, path)}"}
    response = task.result()
    out: dict[str, Any] = {"status": response.status_code, "ok": 200 <= response.status_code < 300}
    kind = response.headers.get("content-type", "").split(";")[0].strip()
    if kind == "application/json" or not response.content:
        try:
            data = _scrub(response.json()) if response.content else None
        except ValueError:
            data = response.text
        text = json.dumps(data, ensure_ascii=False, default=str)
        if len(text) > RESULT_MAX:
            kept = _keep(services, f"{method.lower()}{re.sub(r'[^A-Za-z0-9]+', '_', path)}.json", text.encode("utf-8"))
            out.update({"truncated": True, "size": len(text), "saved": kept, "start": text[:RESULT_MAX // 4]})
        else:
            out["result" if out["ok"] else "error"] = data
        return out
    disposition = response.headers.get("content-disposition", "")
    named = re.search(r'filename="?([^";]+)"?', disposition)
    name = named.group(1) if named else re.sub(r"[^A-Za-z0-9._-]+", "_", path.rsplit("/", 1)[-1]) or "download"
    out.update({"type": kind, "size": len(response.content), "saved": _keep(services, name, response.content)})
    return out


def _ended(task: asyncio.Task) -> None:
    _GOING_ON.discard(task)
    if not task.cancelled() and task.exception() is not None:  # a write that ended after the chat was answered: said in the log
        import logging

        logging.getLogger(__name__).warning("a chat's write failed after its answer: %r", task.exception())


def _follow(template: str, path: str) -> str:
    """Where a chat follows a write that goes on."""
    if template.endswith("/team"):
        return f"Follow it with GET {path}: the walk's board, and the pause waiting on the person when it stops."
    if template.startswith("/api/intern/sessions/"):
        return f"Follow it with GET {path.removesuffix('/message')}."
    return "Follow it with GET /api/jobs (DCLab's running work) and the project's or draft's own reads."


def _keep(services: Any, name: str, data: bytes) -> str:
    """A result too long for a chat, or a file, kept in the workspace (``mcp/results/``) for the person and the chat to open."""
    folder = Path(services.store.home) / "mcp" / "results"
    folder.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).lstrip("._")[:120] or "result"
    file = folder / f"{time.strftime('%Y%m%dT%H%M%S')}-{safe}"
    file.write_bytes(data)
    return str(file)


def _tool(name: str, description: str, properties: dict[str, Any], required: list[str], handler, effect: str = "read") -> Tool:
    return Tool(name=name, description=description, handler=handler, effect=effect, scope=SCOPE, takes_context=True,
                move=CHAT_MOVES[0] if effect == "write" else None,
                parameters={"type": "object", "properties": properties, "required": required, "additionalProperties": False})


DRAFT_ID = {"type": "string", "description": "The draft's id (from home_start or home_list)."}
TOOLS = [
    _tool("home_start", "Start DCLab's Home agent on a new problem, as a person does on Home: it restates the problem, picks the "
          "industry pack, draws the solution workflow and asks its first questions. Use it when the person describes an ML problem "
          "to solve end to end. Returns the agent's words, its open questions and what to do next.",
          {"problem": {"type": "string", "minLength": 8, "maxLength": 2000, "description": "The problem in the person's words, one or two sentences."},
           "pack": {"type": "string", "description": "An industry pack key, only when the person named one."}},
          ["problem"], home_start, "write"),
    _tool("home_message", "Send the person's next message to DCLab's Home agent on a draft (their answer to its question, a "
          "correction, 'simulate data', 'build it'). On a built draft the Home orchestrator and its specialists answer instead. "
          "Waits for the agent's turn and returns its new messages, open questions and what to do next.",
          {"draft_id": DRAFT_ID, "text": {"type": "string", "minLength": 1, "maxLength": MAX_TEXT, "description": "The person's message, in their words."}},
          ["draft_id", "text"], home_message, "write"),
    _tool("home_add_data", "Give a Home draft its table, as the page does: a sample by its key (list_samples names them) or a "
          "data file (CSV, Parquet, JSON, Excel; up to 200 MB) on the DCLab server's machine, only when it runs without sign-in. "
          "Its pipeline cleans and analyses it, then the Home agent reads it; waits for both. Refused: a sealed or holdout set, a "
          "hidden folder, a file named as a key or credential, anything inside DCLab's own workspace folders.",
          {"draft_id": DRAFT_ID, "sample": {"type": "string", "description": "A sample key."},
           "path": {"type": "string", "description": "An absolute path to a data file on the server's machine."}},
          ["draft_id"], home_add_data, "write"),
    _tool("home_status", "Read a Home draft as the page shows it: the last messages, open questions, pack, table, understanding, "
          "workflow, the project once built, and whether the agent is still answering.",
          {"draft_id": DRAFT_ID}, ["draft_id"], home_status),
    _tool("home_build", "Build the draft into a project and start its five stages, when the person asks for it: their words "
          "(\"build it\", or \"yes\" to the agent's offer to build) are sent as their message and the Home agent builds as on Home. "
          "The stages end by reading the holdout once, so ask the person first. Every check of the step-by-step path still runs "
          "(the outcome column, the prediction moment, the leakage review, the build's blockers); a refusal says what is missing.",
          {"draft_id": DRAFT_ID, "person_words": {"type": "string", "minLength": 1, "maxLength": 400,
                                                  "description": "The person's request to build as an order the Home agent reads: "
                                                                 "\"build it\", \"build the model\", \"go ahead and build it\", or "
                                                                 "\"yes\" right after the agent offered to build. When the person "
                                                                 "said it another way (\"I want it built\"), pass \"build it\"."},
           "shown": {"type": "object", "description": "Filled by the host that asked the person: the draft's fields its card showed "
                                                      "(a draft changed since is refused). Leave it out."}},
          ["draft_id", "person_words"], home_build, "write"),
    _tool("home_list", "List the workspace's Home drafts (newest first): their problem, status and project.",
          {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}, [], home_list),
    _tool("dclab_capabilities", "Search everything the DCLab product can do: every operation of its API (projects, stages, Home "
          "drafts, data and connectors, decisions and the ledger, students, evidence and lessons, reports, lineage and replay, "
          "programmes, twin runs, kinds, models, the store, the workspace), with what each does, its parameters and its body. "
          "Search by words (for example 'lineage', 'lessons review', 'synthetic data'); then run one with dclab_call.",
          {"query": {"type": "string", "description": "Words to search for."}, "method": {"type": "string", "enum": list(METHODS)},
           "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, [], dclab_capabilities),
    _tool("dclab_call", "Run one operation of the DCLab product whose chat field is \"runs\" (dclab_capabilities lists them), inside "
          "the workspace, with every check the product's own page has: reads, and agent work such as asking, starting or messaging "
          "the intern, drafting a map or a proposal, a what-if, stopping a job. The person's operations are refused with the reason. "
          "A long result or a file is kept in the workspace and its path returned.",
          {"method": {"type": "string", "enum": list(METHODS)},
           "path": {"type": "string", "description": "The operation's path with every {name} filled, e.g. /api/projects/abc/lineage."},
           "query": {"type": "object", "description": "Query parameters."},
           "body": {"description": "The JSON body, for POST, PUT and PATCH."}},
          ["method", "path"], dclab_call, "write"),
]

__all__ = ["AGENT_WRITES", "AUTHORIZATION", "CHAT_MOVES", "SCOPE", "TOOLS", "dclab_call", "dclab_capabilities", "home_add_data", "home_build", "home_list", "home_message", "home_start", "home_status", "register", "view"]
