"""Which model serves a purpose, and which one shadows it (package A6.4).

By default a purpose is served by its tier (``settings.PURPOSES``). A workspace can change two things per purpose:

- ``shadow``: a second model receives a copy of every request after the served one answered. Its reply is compared
  and logged (``models/shadow.py``), never returned, never shown to the validator, never counted against a budget.
  Only a local tier can shadow (a model on this machine costs nothing; a paid shadow would spend money no cap sees).
- ``serve``: the tier that answers instead of the purpose's own. Moving a purpose to another tier needs a reviewer
  (the role ``reviewer`` or ``owner``) and a reason; moving it back to its own tier is one setting, for anyone.

Every change is kept in the routing history, which the platform audit shows. Until accounts exist (package 10.2) the
role is the one the request declares (``X-DCLab-Role``), not a verified login.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Protocol

from . import settings

REVIEWERS = ("reviewer", "owner")
# purposes whose requests go through another client (the research campaign's NOOA calls): the gateway cannot route them
NOT_ROUTED = frozenset({"campaign", "campaign_review"})
SETTINGS = ("shadow", "serve")
TEXT = 300


class RoutingError(ValueError):
    pass


def _now() -> str:
    from .usage import now

    return now()


class RoutingStore(Protocol):
    def get(self) -> dict[str, Any]: ...
    def save(self, doc: dict[str, Any]) -> dict[str, Any]: ...


def empty() -> dict[str, Any]:
    return {"purposes": {}, "history": []}


class FileRouting:
    """``model_routing.json`` in the workspace folder, replaced whole on each write."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def get(self) -> dict[str, Any]:
        try:
            return {**empty(), **json.loads(self.path.read_text(encoding="utf-8"))}
        except (OSError, json.JSONDecodeError):
            return empty()

    def save(self, doc: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
            os.replace(temporary, self.path)
        return doc


class PgRouting:
    def __init__(self, workspace_id: str, url: str | None = None):
        from ..storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def get(self) -> dict[str, Any]:
        import sqlalchemy as sa

        from ..storage.models import model_routing as t

        with self.engine.connect() as c:
            doc = c.execute(sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id)).scalar()
        return {**empty(), **(doc or {})}

    def save(self, doc: dict[str, Any]) -> dict[str, Any]:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from ..storage.models import model_routing as t

        with self.engine.begin() as c:
            c.execute(insert(t).values(workspace_id=self.workspace_id, doc=doc).on_conflict_do_update(
                index_elements=[t.c.workspace_id], set_={"doc": doc, "updated": sa.func.now()}))
        return doc


def open_routing(home: Path) -> RoutingStore:
    from ..storage.url import database_url

    if database_url(required=False):
        from ..storage import db

        return PgRouting(db.workspace_for(Path(home)))
    return FileRouting(Path(home) / "model_routing.json")


def route(doc: dict[str, Any], purpose: str) -> dict[str, Any]:
    entry = (doc.get("purposes") or {}).get(purpose) or {}
    return {"shadow": entry.get("shadow"), "serve": entry.get("serve"), "serve_model": entry.get("serve_model"),
            "serve_endpoint": entry.get("serve_endpoint")}


def approved(entry: dict[str, Any]) -> bool:
    """The served tier still points at the model the reviewer approved: a new model under the same tier name is not
    approved, and the purpose falls back to its own tier until someone approves it."""
    tier = entry.get("serve")
    if not tier:
        return False
    current = settings.public(tier)
    return current["model"] == entry.get("serve_model") and current["endpoint"] == entry.get("serve_endpoint")


def change(store: RoutingStore, purpose: str, setting: str, tier: str | None, role: str | None, reason: str = "") -> dict[str, Any]:
    """Set a purpose's shadow or serving tier (None: no shadow, or back to the purpose's own tier)."""
    settings.purpose(purpose)  # an unknown purpose is an error
    if purpose in NOT_ROUTED:
        raise RoutingError(f"{purpose} requests go through the research campaign's own client: the gateway cannot route or shadow them")
    if setting not in SETTINGS:
        raise RoutingError(f"setting must be one of {', '.join(SETTINGS)}")
    own = settings.purpose(purpose).tier
    if tier is not None:
        if tier not in settings.ROUTABLE:
            raise RoutingError(f"tier must be one of {', '.join(settings.ROUTABLE)}")
        if tier == own:
            tier = None  # the purpose's own tier is the default
    if setting == "shadow" and tier is not None:
        public = settings.public(tier)
        if not public["local"]:
            raise RoutingError("only a local tier can shadow: a shadow's requests are not counted against any budget")
    reason = " ".join(str(reason or "").split())[:TEXT]
    model = endpoint = None
    if setting == "serve" and tier is not None:
        if role not in REVIEWERS:
            raise PermissionError("Only a reviewer or the owner moves a purpose to another model")
        if not reason:
            raise RoutingError("Say why: the reason is kept in the audit trail")
        public = settings.public(tier)
        if not public["model"] or not public["key_configured"]:
            raise RoutingError(f"the {tier} tier has no model configured: there is nothing to approve")
        model, endpoint = public["model"], public["endpoint"]  # the approval is for this model at this endpoint
    doc = store.get()
    current = route(doc, purpose)
    if current[setting] == tier and (setting == "shadow" or current.get("serve_model") == model):
        return doc
    purposes = dict(doc.get("purposes") or {})
    purposes[purpose] = {**current, setting: tier, **({"serve_model": model, "serve_endpoint": endpoint} if setting == "serve" else {})}
    entry = {"at": _now(), "purpose": purpose, "setting": setting, "from": current[setting], "to": tier, "by": role or "human", "reason": reason or None,
             **({"model": model, "endpoint": endpoint} if setting == "serve" and tier else {})}
    return store.save({**doc, "purposes": purposes, "history": [*(doc.get("history") or []), entry]})
