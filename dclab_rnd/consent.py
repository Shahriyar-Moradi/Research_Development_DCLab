"""Consent for what leaves a workspace, and the export that obeys it (package 19.10).

Two uses, each a switch the owner decided (2026-10-13): **improve** (use what happened here to improve DCLab: read
by the team, turned into benchmark cases) and **train** (use it to train DCLab's models). Each is off by default, per
workspace and per project; a use is allowed for a project only when both its workspace and the project allow it.
Customer data is included when the switches are on (the owner's answer), but never as a table's values: those
(``cell_values``) and model prompts never leave, whatever the consent.

The workspace's switches live in ``consent.json`` in the workspace folder; a project's in its settings (``consent``;
the earlier training opt-in, ``share_for_training``, counts as its ``train``). Only an owner changes them (the
``admin`` permission), and every change is audited and recorded.

:func:`export_events` is the one way events leave: refused for a use the workspace does not allow; per event, kept only
for a project (or a project-less session) that allows the use; only ``meta`` and ``count`` fields, and person and
agent text redacted (``dclab_core.redact``). ``train`` adds the decision records (their states hold bands, never
values).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from dclab_core import consent as core

USES = ("improve", "train")
FILE = "consent.json"
NOTICE_VERSION = core.NOTICE_VERSION
PROJECT_TTL = 10.0  # seconds a project's switches are kept in memory by a writer (a change through the API invalidates at once)
_WORKSPACE: dict[str, tuple[tuple[int, int], dict[str, bool]]] = {}  # home -> ((consent.json's mtime_ns, size), switches)
_STORES: dict[str, Any] = {}  # home -> the project store a writer with no services opens for the stamp (4)
_PROJECTS: dict[str, tuple[float, dict[str, bool] | None]] = {}  # project id -> (read at, switches; None: no such project)
SHAREABLE = ("meta", "count", "person_text", "agent_text")  # what may leave with consent; text redacted on the way out


def workspace_consent(home: Path) -> dict[str, bool]:
    """The workspace's switches (off unless the owner turned them on)."""
    try:
        data = json.loads((Path(home) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    return {use: data.get(use) is True for use in USES}


def set_workspace_consent(home: Path, use: str, on: bool, who: str = "owner") -> dict[str, bool]:
    if use not in USES:
        raise ValueError(f"use is one of {', '.join(USES)}")
    current = workspace_consent(home)
    current[use] = bool(on)
    path = Path(home) / FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(current, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)
    return current


def project_consent(project: dict[str, Any] | None) -> dict[str, bool]:
    """A project's switches. Its own ``consent.train``, once set either way, wins over the earlier opt-in."""
    settings = (project or {}).get("settings") or {}
    own = settings.get("consent") or {}
    train = own["train"] if isinstance(own.get("train"), bool) else settings.get("share_for_training") is True
    return {"improve": own.get("improve") is True, "train": train}


def allowed(workspace: dict[str, bool], project: dict[str, Any] | None, use: str) -> bool:
    """A use is allowed for a project when its workspace and the project both allow it; a session with no project,
    when the workspace does."""
    if not workspace.get(use):
        return False
    return project is None or project_consent(project).get(use, False)


def _workspace_now(home: Path) -> dict[str, bool]:
    """The workspace's switches, read again only when consent.json changed (its mtime)."""
    path = Path(home) / FILE
    try:
        st = path.stat()
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = (0, 0)
    key = str(path)
    cached = _WORKSPACE.get(key)
    if cached is None or cached[0] != stamp:
        cached = (stamp, workspace_consent(home))
        _WORKSPACE[key] = cached
    return cached[1]


def invalidate(project_id: str | None = None) -> None:
    """Forget a project's cached switches (its settings changed), or every project's."""
    if project_id is None:
        _PROJECTS.clear()
        _WORKSPACE.clear()
    else:
        _PROJECTS.pop(project_id, None)


NO_STORE = object()  # no project store is reachable from this writer: the workspace's level stands (the export checks the project again)


def _project_now(project_id: str, projects: Any = None, home: Path | None = None) -> Any:
    """A project's switches, kept ``PROJECT_TTL`` seconds; None for a project the store does not have; ``NO_STORE`` when
    no store is reachable here (a log written to directly, outside the server and the worker)."""
    import time

    if projects is None:
        from . import context

        projects = getattr(context.services(), "projects", None)
    if projects is None and home is not None:  # a writer outside a request (a script, a test): the stores of that folder
        key = str(Path(home))
        if key not in _STORES:
            try:
                from .storage import open_stores

                _STORES[key] = open_stores(Path(home))[0]
            except Exception:  # noqa: BLE001 - no store to open here
                _STORES[key] = None
        projects = _STORES[key]
    if projects is None:
        return NO_STORE
    cached = _PROJECTS.get(project_id)
    if cached is not None and PROJECT_TTL > 0 and time.monotonic() - cached[0] < PROJECT_TTL:
        return cached[1]
    switches: dict[str, bool] | None
    try:
        switches = project_consent(projects.get(project_id))
    except Exception:  # noqa: BLE001 - no such project: nothing is allowed for it
        switches = None
    _PROJECTS[project_id] = (time.monotonic(), switches)
    return switches


def current_level(*, home: Path | None = None, project: str | None = None, projects: Any = None) -> str:
    """The level a record written now is stamped with (23.1): the workspace's switches, and the project's when the record
    belongs to one. A record of a project the store does not know, or written where no workspace folder is known, is
    ``operate``: it can never leave."""
    if home is None:
        from . import context

        home = getattr(getattr(context.services(), "store", None), "home", None)
    if home is None:
        return "operate"
    workspace = _workspace_now(home)
    if project:
        switches = _project_now(project, projects, home)
        if switches is NO_STORE:
            return "operate"  # a project's record with no way to read its switch never leaves
        return "operate" if switches is None else core.level_of(workspace, switches)
    return core.level_of(workspace, None)


def stamped_counts(log: Any, limit: int = 10**9) -> dict[str, int]:
    """How many events of the record carry each level (the consent page's counts)."""
    counts = {level: 0 for level in core.LEVELS}
    if log is None:
        return counts
    engine = getattr(log, "engine", None)
    if engine is not None:  # PostgreSQL: one grouped query over the workspace's rows
        import sqlalchemy as sa

        from .storage import models

        t = models.experience_event
        q = sa.select(t.c.doc["consent"].astext.label("level"), sa.func.count()).where(t.c.workspace_id == log.workspace).group_by("level")
        with engine.connect() as c:
            for level, n in c.execute(q):
                counts[level or "operate"] = counts.get(level or "operate", 0) + int(n)
        return counts
    for event in log.read(limit=limit):  # files: the whole record
        counts[event.consent] = counts.get(event.consent, 0) + 1
    return counts


class ConsentRefused(Exception):
    pass


def _scrub(value: Any, removed: dict[str, int]) -> Any:
    """Every string in ``value`` (a scalar, a list, a dict's values) through the redaction, its counts added up."""
    from dclab_core.redact import redact_text

    if isinstance(value, str):
        text, found = redact_text(value)
        for k, v in found.items():
            removed[k] = removed.get(k, 0) + v
        return text
    if isinstance(value, list):
        return [_scrub(v, removed) for v in value]
    if isinstance(value, dict):
        return {k: _scrub(v, removed) for k, v in value.items()}
    return value


def _project_of(project: str | None, conversation: str | None, drafts: dict[str, str]) -> str | None:
    """The project an event or decision belongs to: its own, or the one its draft was built into."""
    return project if project is not None else drafts.get(conversation or "")


def _allowed_for(project: str | None, use: str, workspace: dict[str, bool], projects: dict[str, dict[str, Any]]) -> bool:
    return project is None or (project in projects and allowed(workspace, projects[project], use))


def _by(value: str | None) -> str | None:
    """Who labelled, without the person: ``person:u3f…`` leaves as ``person``; a replica or a measurement keeps its name."""
    if not value:
        return value
    kind, _, name = str(value).partition(":")
    return value if kind in ("replica", "measured", "benchmark", "rule", "model") and name else kind  # a model's name is not a person


def draft_projects(drafts: Any) -> dict[str, str]:
    """Each built draft's project: its Home conversation follows that project's consent."""
    out = {}
    try:
        for d in drafts.list(limit=100_000):
            if d.get("id") and d.get("project_id"):
                out[str(d["id"])] = str(d["project_id"])
    except Exception:  # noqa: BLE001 - no drafts store: a conversation then follows only the workspace
        pass
    return out


def export_events(events: Iterable[Any], use: str, workspace: dict[str, bool], projects: dict[str, dict[str, Any]],
                  drafts: dict[str, str] | None = None) -> list[dict[str, Any]]:
    """The events that may leave for ``use``: refused outright when the workspace does not allow it; each kept only
    when its project allows it (a draft's events follow the project it became, ``drafts``; an event of an unknown or
    deleted project never leaves); only shareable fields, every string redacted (a file's name too: only its
    extension leaves), and what was removed named."""
    from dclab_core import experience as contract

    if use not in USES:
        raise ValueError(f"use is one of {', '.join(USES)}")
    if not workspace.get(use):
        raise ConsentRefused(f"the workspace does not allow use for {use}")
    out = []
    drafts = drafts or {}
    for event in events:
        if not _allowed_for(_project_of(event.ids.project, event.ids.conversation, drafts), use, workspace, projects):
            continue
        if not core.exportable(event.consent, use):  # 23.1: the level it was written under, whatever the switches say now
            continue
        kept = contract.redact(event, SHAREABLE)
        payload, removed = {}, {}
        for name, value in kept.payload.items():
            if event.kind == "data.uploaded" and name == "name" and isinstance(value, str):
                value = ("file" + Path(value).suffix[:10]) if value else value  # a file's name can be a person's
            payload[name] = _scrub(value, removed)
        doc = json.loads(kept.model_dump_json())
        doc["payload"] = payload
        doc["redacted_text"] = removed
        doc["ids"] = {k: v for k, v in doc["ids"].items() if k not in ("user", "request")}  # who and which request stay home
        out.append(doc)
    return out


def export_decisions(records: Iterable[Any], labels: Iterable[Any], workspace: dict[str, bool],
                     projects: dict[str, dict[str, Any]], drafts: dict[str, str] | None = None, outcomes: Iterable[Any] = ()) -> list[dict[str, Any]]:
    """Decision records with their labels, for ``train`` only: states are bands (never values), reasons redacted, who
    answered or labelled named by kind, never by person; a draft's decisions follow the project it became."""

    if not workspace.get("train"):
        raise ConsentRefused("the workspace does not allow use for train")
    drafts = drafts or {}
    by_record: dict[str, list] = {}
    for label in labels:
        doc = json.loads(label.model_dump_json())
        doc["by"] = _by(doc.get("by"))
        by_record.setdefault(label.record, []).append(doc)
    outcomes_of: dict[str, list] = {}
    for outcome in outcomes:  # 23.5: what followed, with its own stamp; a person's note redacted
        if not core.exportable(outcome.consent, "train"):
            continue
        doc = json.loads(outcome.model_dump_json())
        doc["by"] = _by(doc.get("by"))
        doc.pop("note", None)  # a person's words stay home
        outcomes_of.setdefault(outcome.record, []).append(doc)
    out = []
    for record in records:
        if not core.exportable(record.consent, "train"):  # 23.1: stamped at write time
            continue
        if not _allowed_for(_project_of(record.project, record.draft, drafts), "train", workspace, projects):
            continue
        doc = json.loads(record.model_dump_json())
        removed: dict[str, int] = {}
        for a in doc.get("answers") or []:
            if a.get("source") != "shadow":  # a shadow stays marked (22.4): it was never an answer, and must not read as one
                a.pop("source", None)  # a prompt's hash or a request's id: it stays home
            if a.get("reason"):
                a["reason"] = _scrub(a["reason"], removed)
        doc["labels"] = by_record.get(record.id, [])
        doc["outcomes"] = outcomes_of.get(record.id, [])
        out.append(doc)
    return out


def export_main_product(records: Iterable[Any], labels: Iterable[Any], workspace: dict[str, bool],
                        projects: dict[str, dict[str, Any]], drafts: dict[str, str] | None = None) -> dict[str, Any]:
    """22.7: the decisions of the main product's Jev purposes in its label format (``semantic_decision_answers``'
    ``ground_truth``, ``label_source`` and ``labels_version``), through the one way records leave (``export_decisions``:
    refused without ``train``, bands only, nobody named). A label source it has no word for (``measured``) stays here
    and is counted; nothing is sent by this function: what the main product receives, and under which of its own
    consents (its ADR 0008), is its change."""
    from dclab_core.adapters.main_product_r3 import label_rows, releases

    points = set(releases())
    out, left = [], {}
    for doc in export_decisions(records, labels, workspace, projects, drafts):
        if doc["point"] not in points:
            continue
        rows, skipped = label_rows(doc["labels"], answer_id=doc["id"])  # the main product maps this record to its answer row
        for k, v in skipped.items():
            left[k] = left.get(k, 0) + v
        out.append({"decision": {k: doc.get(k) for k in ("id", "point", "subject", "state", "options", "answers", "value_used", "used", "level")},
                    "labels": rows})
    return {"decisions": out, "left_out": left, "points": sorted(points)}


def tombstone(kind: str, ident: str) -> dict[str, Any]:
    """What a deletion leaves behind: what was deleted, never what it held."""
    from datetime import datetime, timezone

    return {"deleted": kind, "id": ident, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def consented_projects(projects: Any, workspace: dict[str, bool], use: str) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in projects.list() if allowed(workspace, p, use)}


__all__ = ["USES", "workspace_consent", "set_workspace_consent", "project_consent", "allowed", "export_events", "export_decisions", "export_main_product",
           "ConsentRefused", "tombstone", "consented_projects", "draft_projects"]
