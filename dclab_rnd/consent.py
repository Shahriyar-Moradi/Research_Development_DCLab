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

USES = ("improve", "train")
FILE = "consent.json"
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
    return value if kind in ("replica", "measured", "benchmark", "rule") and name else kind


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
                     projects: dict[str, dict[str, Any]], drafts: dict[str, str] | None = None) -> list[dict[str, Any]]:
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
    out = []
    for record in records:
        if not _allowed_for(_project_of(record.project, record.draft, drafts), "train", workspace, projects):
            continue
        doc = json.loads(record.model_dump_json())
        removed: dict[str, int] = {}
        for a in doc.get("answers") or []:
            a.pop("source", None)  # a prompt's hash or a request's id: it stays home
            if a.get("reason"):
                a["reason"] = _scrub(a["reason"], removed)
        doc["labels"] = by_record.get(record.id, [])
        out.append(doc)
    return out


def tombstone(kind: str, ident: str) -> dict[str, Any]:
    """What a deletion leaves behind: what was deleted, never what it held."""
    from datetime import datetime, timezone

    return {"deleted": kind, "id": ident, "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def consented_projects(projects: Any, workspace: dict[str, bool], use: str) -> dict[str, dict[str, Any]]:
    return {p["id"]: p for p in projects.list() if allowed(workspace, p, use)}


__all__ = ["USES", "workspace_consent", "set_workspace_consent", "project_consent", "allowed", "export_events", "export_decisions",
           "ConsentRefused", "tombstone", "consented_projects", "draft_projects"]
