"""Sessions: the record of every visit, on the Admin page (package 19.4).

    GET /api/experience/sessions            the latest visits as cards: who, when, the problem and its family, how it ended, spend
    GET /api/experience/sessions/{visit}    one visit's timeline, top to bottom, with its drafts and projects named
    GET /api/experience/search              events by person, project, draft, kind, words, code and dates, with their visits

Only the owner reads them (``accounts.guard``: the ``timelines`` permission). A timeline shows what people and agents
wrote and every meta field; a tool's arguments and results (cell values) never, and prompts and replies only with
``model_text=true`` (``experience.finder.view``).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import HTTPException

from ..api_models import Page, Router

ID = re.compile(r"[0-9a-zA-Z_-]{1,80}")
TIMELINE = 20_000  # the events one timeline shows at most (the newest)


def _log(ctx) -> Any:
    return getattr(ctx, "experience", None)


def _names(ctx, cards: list[Any]) -> dict[str, Any]:
    """What the cards' ids stand for: project names, each draft's problem and family, people's names (never e-mails)."""
    from ...draft.card import card_of

    projects: dict[str, str] = {}
    try:
        projects = {p["id"]: p.get("name") or p["id"] for p in ctx.projects.list()}
    except Exception:  # noqa: BLE001 - a page of sessions still reads without names
        pass
    drafts: dict[str, dict[str, Any]] = {}
    for conversation in sorted({c for card in cards for c in card.conversations}):
        try:
            d = ctx.drafts.get(conversation)
        except Exception:  # noqa: BLE001 - an intern session, or a draft deleted since
            continue
        entry: dict[str, Any] = {"problem": (d.get("problem") or "")[:300] or None}
        try:
            card = card_of(d)
            entry.update(family=card.family.value, ready=card.ready)
        except Exception:  # noqa: BLE001 - an old draft the card cannot read yet
            pass
        drafts[conversation] = entry
    people: dict[str, str] = {}
    users = sorted({card.user for card in cards if card.user})
    if users:
        try:
            from ...accounts.store import Accounts

            accounts = Accounts()
            for u in users:
                try:
                    people[u] = accounts.user(u).get("name") or "A member"
                except KeyError:
                    people[u] = "A former member"
        except Exception:  # noqa: BLE001 - accounts off: there are no ids to name
            pass
    return {"projects": projects, "drafts": drafts, "people": people}


def _card(card: Any, names: dict[str, Any]) -> dict[str, Any]:
    out = card.model_dump()
    drafts = [names["drafts"].get(c) for c in card.conversations]
    first = next((d for d in drafts if d), None)
    out["who"] = names["people"].get(card.user) if card.user else "Owner (this machine)"
    out["problem"] = card.problem or (first or {}).get("problem")
    out["family"] = (first or {}).get("family")
    out["project_names"] = {p: names["projects"].get(p) for p in card.projects}
    out["draft_problems"] = {c: (names["drafts"].get(c) or {}).get("problem") for c in card.conversations}
    return out


def _when(name: str, value: str | None) -> str | None:
    if value in (None, ""):
        return None
    from ... import experience

    try:
        experience._utc(value)
    except ValueError:
        raise HTTPException(422, f"{name} is a date or a time with its offset (2026-10-13, or 2026-10-13T09:30+03:30)") from None
    return value


def _id(name: str, value: str | None) -> str | None:
    if value in (None, ""):
        return None
    if not ID.fullmatch(value):
        raise HTTPException(422, f"{name} is an id")
    return value


def latest(ctx, limit: int) -> dict[str, Any]:
    from ... import experience

    log = _log(ctx)
    cards = experience.sessions(log, limit) if log is not None else []
    names = _names(ctx, cards)
    items = [_card(c, names) for c in cards]
    totals = {"sessions": len(items), "refused": sum(c.refused for c in cards), "errors": sum(c.errors for c in cards),
              "built": sum(c.outcome != "open" for c in cards), "cost": {}}
    for c in cards:
        for currency, value in c.cost.items():
            totals["cost"][currency] = round(totals["cost"].get(currency, 0.0) + value, 6)
    return {"items": items, "totals": totals, "kinds": sorted(experience.contract.KINDS), "families": _families(),
            "groups": ["refused", "errors"], "recording": log is not None}


def _families() -> list[str]:
    from ...experience import finder

    return finder.families()


def one(ctx, visit: str, model_text: bool) -> dict[str, Any]:
    from ...experience import contract, finder

    log = _log(ctx)
    events = log.read(visit=visit, limit=TIMELINE + 1) if log is not None else []
    if not events:
        raise HTTPException(404, "No session with this id in this workspace")
    cut = len(events) > TIMELINE
    events = events[-TIMELINE:]  # the newest: a longer visit says it was cut, and from where
    card = contract.session_card(events)
    names = _names(ctx, [card])
    return {"session": _card(card, names), "events": [finder.view(e, model_text) for e in events], "cut": cut,
            "model_text": model_text, "projects": names["projects"], "drafts": names["drafts"]}


def find(ctx, **query: Any) -> dict[str, Any]:
    from ...experience import finder

    log = _log(ctx)
    if log is None:
        return {"total": 0, "events": [], "sessions": [], "visits": 0, "scanned": 0, "capped": False}
    try:
        found = finder.search(log, **query)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    names = _names(ctx, found["sessions"])
    return {**found, "events": [finder.view(e) for e in found["events"]], "sessions": [_card(c, names) for c in found["sessions"]]}


def register(app, ctx) -> None:
    router = Router()

    @router.get("/api/experience/sessions", response_model=Page)
    async def experience_sessions(limit: int = 50):
        """The latest visits, newest first: who, when, the problem and its family, how it ended, refusals, errors, spend."""
        if limit < 1 or limit > 200:
            raise HTTPException(422, "limit is 1 to 200")
        return await asyncio.to_thread(latest, ctx, limit)

    @router.get("/api/experience/sessions/{visit}", response_model=Page)
    async def experience_session(visit: str, model_text: bool = False):
        """One visit's timeline as it happened; prompts and replies only with model_text=true, a tool's values never."""
        return await asyncio.to_thread(one, ctx, _id("visit", visit), model_text)

    @router.get("/api/experience/search", response_model=Page)
    async def experience_search(person: str | None = None, project: str | None = None, draft: str | None = None,
                                kind: str | None = None, text: str | None = None, code: str | None = None,
                                since: str | None = None, until: str | None = None, limit: int = 50):
        """Events matching every condition given, newest first, and the visits they belong to."""
        if limit < 1 or limit > 500:
            raise HTTPException(422, "limit is 1 to 500")
        if text is not None and len(text) > 200 or code is not None and len(code) > 80:
            raise HTTPException(422, "text is at most 200 characters and code at most 80")
        return await asyncio.to_thread(find, ctx, person=_id("person", person), project=_id("project", project),
                                       draft=_id("draft", draft), kind=kind or None, text=text or None, code=code or None,
                                       since=_when("since", since), until=_when("until", until), limit=limit)

    app.include_router(router)
