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
import math
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
# 21.4: the roles that read what the person means; a stronger model serves one only when it measured better on the
# conversations (or the owner says so, audited, with the reason)
UNDERSTANDING = frozenset({"problem_analyst", "conversation_reader", "brief_checker", "data_investigator", "decomposer", "map_reviewer", "data_need"})
RANK = {"cheap": 0, "standard": 1, "strong": 2}  # how strong a tier is; tuned and local tiers are not "stronger"


def prompt_of(purpose: str) -> str | None:
    """The version of the prompt a purpose is sent with: its content hash (dclab_core's roles, or the studio's prompts)."""
    try:
        from dclab_core import prompts as core_prompts

        found = core_prompts.fingerprints().get(purpose)
        if found:
            return found
    except Exception:  # noqa: BLE001 - a purpose without a role prompt
        pass
    try:
        from .. import prompts as studio_prompts

        return studio_prompts.fingerprints().get(f"prompt:{purpose}")
    except Exception:  # noqa: BLE001
        return None


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


def change(store: RoutingStore, purpose: str, setting: str, tier: str | None, role: str | None, reason: str = "",
           measured: dict[str, float] | None = None, override: bool = False) -> dict[str, Any]:
    """Set a purpose's shadow or serving tier (None: no shadow, or back to the purpose's own tier). Moving an
    understanding role to a stronger tier needs ``measured`` (each model's score on the conversations, at this prompt)
    to show the new model better than the one serving it now, or the owner's ``override`` with a reason."""
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
        return doc  # the same setting again changes nothing
    evidence = None
    if setting == "serve" and tier is not None and purpose in UNDERSTANDING and not (tier in RANK and RANK[tier] < RANK.get(own, -1)):
        # every move but one to a weaker tier than its own (a tuned or unranked tier included) needs the new model to have
        # measured better than its own tier's model and than the model a reviewer approved for it, on the same conversations
        bars = {str(settings.public(own)["model"])}
        if current["serve"] and approved(current):
            bars.add(str(current.get("serve_model")))
        bars.discard(str(model))
        scores = {str(k): float(v) for k, v in (measured or {}).items() if v is not None}
        better = model in scores and all(b in scores and scores[model] > scores[b] for b in bars)
        if not better and not (override and role == "owner"):
            raise RoutingError(f"{purpose} reads what the person means: a model serves it only when it measured better on the same "
                               f"conversations than {', '.join(sorted(bars)) or 'its own tier'} ({model}: {scores.get(model, 'not measured')}; "
                               + "; ".join(f"{b}: {scores.get(b, 'not measured')}" for b in sorted(bars)) + "), or the owner overrides with a reason")
        evidence = {"measured": {k: scores[k] for k in sorted({str(model), *bars}) if k in scores}, "override": not better}
    if setting == "serve" and tier is not None:  # 21.5: no model serves a purpose before the harness suite passed with it here
        from ..extensions import optional

        regression = optional("agent_eval.regression")  # the open edition: no suite, so only the owner's override with a reason
        suite = regression.passed_with(purpose, str(model)) if regression is not None else None
        if suite is None and not (override and role == "owner"):
            raise RoutingError(f"{model} has not passed the harness suite serving {purpose} on these prompts and gates: run "
                               "make harness-eval ARGS='--live --cap-eur N' with it, or the owner overrides with a reason")
        evidence = {**(evidence or {}), "suite": suite["id"] if suite else None, **({"override": True} if suite is None else {})}
    purposes = dict(doc.get("purposes") or {})
    purposes[purpose] = {**current, setting: tier, **({"serve_model": model, "serve_endpoint": endpoint} if setting == "serve" else {})}
    entry = {"at": _now(), "purpose": purpose, "setting": setting, "from": current[setting], "to": tier, "by": role or "human", "reason": reason or None,
             "prompt": prompt_of(purpose), **({"model": model, "endpoint": endpoint} if setting == "serve" and tier else {}),
             **({"evidence": evidence} if evidence else {})}
    return store.save({**doc, "purposes": purposes, "history": [*(doc.get("history") or []), entry]})


def set_cap(store: RoutingStore, purpose: str, eur_month: float | None, role: str | None, reason: str = "") -> dict[str, Any]:
    """A purpose's monthly spend cap in euros (21.4; None: no cap of its own). A reviewer or the owner sets it, with a
    reason; the gateway refuses a request that could pass it."""
    settings.purpose(purpose)
    if role not in REVIEWERS:
        raise PermissionError("Only a reviewer or the owner sets a role's spend cap")
    reason = " ".join(str(reason or "").split())[:TEXT]
    if not reason:
        raise RoutingError("Say why: the reason is kept in the audit trail")
    if eur_month is not None and (not isinstance(eur_month, (int, float)) or not math.isfinite(eur_month) or eur_month < 0):
        raise RoutingError("a cap is a number of euros, zero or more")
    doc = store.get()
    entry = dict((doc.get("purposes") or {}).get(purpose) or {})
    before = entry.get("cap_eur_month")
    if before == eur_month:
        return doc
    entry["cap_eur_month"] = None if eur_month is None else round(float(eur_month), 4)
    history = {"at": _now(), "purpose": purpose, "setting": "cap", "from": before, "to": entry["cap_eur_month"], "by": role, "reason": reason}
    return store.save({**doc, "purposes": {**(doc.get("purposes") or {}), purpose: entry}, "history": [*(doc.get("history") or []), history]})


def cap(doc: dict[str, Any], purpose: str) -> float | None:
    value = ((doc.get("purposes") or {}).get(purpose) or {}).get("cap_eur_month")
    return float(value) if value is not None else None


def switch(store: RoutingStore, audit_log: Any, purpose: str, setting: str, value: Any, role: str | None, reason: str = "",
           measured: dict[str, float] | None = None, override: bool = False) -> dict[str, Any]:
    """One audited change of a role's model (21.4): the routing changed and its audit row written in the same call, the
    row naming the model, its endpoint, the prompt version and the measurement it was decided on. ``setting``: serve,
    shadow (``value`` a tier or None) or cap (``value`` euros a month or None)."""
    kept = len(store.get().get("history") or [])
    doc = (set_cap(store, purpose, value, role, reason) if setting == "cap"
           else change(store, purpose, setting, value, role, reason, measured=measured, override=override))
    history = doc.get("history") or []
    if len(history) > kept and audit_log is not None:  # the same setting again changes nothing and is not audited
        from .. import audit

        made = history[-1]
        audit.record(audit_log, "routing", "human", str(made.get("by") or role or "person").capitalize(), move="set_routing", status="allowed",
                     args={**{k: made.get(k) for k in ("purpose", "setting", "from", "to")},  # "to" None: back to its own tier
                           **{k: made[k] for k in ("model", "endpoint", "prompt", "evidence") if made.get(k) is not None}},
                     message=made.get("reason") or "")
    return doc
