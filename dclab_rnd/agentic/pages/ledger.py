"""Decisions: the ledger on the Admin page (packages 19.7, 19.8).

    GET  /api/ledger/summary              how many decisions by point, kind and who decided; labels by source
    GET  /api/ledger/blind/next           the next decision drawn for a blind label: its question and state only
    POST /api/ledger/blind/{record_id}    a blind reviewer's answer, kept with its sampling weight

Only the owner and reviewers read or label them (``accounts.guard``: ``review``). A blind reviewer never sees an
answer, the value used or who decided (``dclab_core.ledger.blind_view``).
"""

from __future__ import annotations

import asyncio
import re
from typing import Any

from fastapi import Body, HTTPException

from ..api_models import Page, Router

ID = re.compile(r"[0-9a-zA-Z_.:-]{6,80}")


def register(app, ctx) -> None:
    router = Router()

    def store() -> Any:
        return getattr(ctx, "ledger", None)

    @router.get("/api/ledger/summary", response_model=Page)
    async def ledger_summary():
        """Decisions by point, kind and who decided; labels by source; how many are drawn for blind labels."""
        from ...ledger import labels

        return await asyncio.to_thread(labels.summary, store())

    @router.get("/api/ledger/blind/next", response_model=Page)
    async def ledger_blind_next(point: str | None = None):
        """The next decision drawn for a blind label (its question, subject, state and options), or none left."""
        from ...ledger import labels

        if point is not None and not re.fullmatch(r"[a-z_.]{3,60}", point):
            raise HTTPException(422, "point is a decision point")
        item = await asyncio.to_thread(labels.next_blind, store(), "blind-v1", point)
        return {"item": item}

    @router.post("/api/ledger/blind/{record_id}", response_model=Page)
    async def ledger_blind_label(record_id: str, body: Any = Body(None)):
        """A blind reviewer's answer: {"value": ...}; kept with the weight of how the decision was drawn."""
        from ...accounts.principal import current
        from ...ledger import labels

        if not ID.fullmatch(record_id):
            raise HTTPException(422, "record_id is a decision's id")
        value = (body or {}).get("value") if isinstance(body, dict) else None
        if value is None or not isinstance(value, (bool, int, float, str)):
            raise HTTPException(422, "value is a flag, a number or a short text")
        who = current()
        by = f"reviewer:{who.user_id}" if who is not None and who.user_id else "reviewer"
        try:
            out = await asyncio.to_thread(labels.blind_label, record_id, value, by, "blind-v1", store())
        except KeyError:
            raise HTTPException(404, "No such decision") from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"label": out.model_dump(mode="json")}

    @router.post("/api/feedback", response_model=Page)
    async def give_feedback(body: Any = Body(None)):
        """A person's feedback (19.9): {"on": answer | proposal | decision | report, "ref": its id, "rating": helpful |
        unhelpful | wrong, "reason": optional, "draft" or "project": where}. It joins the session record; "wrong" on a
        yes/no decision from someone who may change the work or review it is also that person's label saying the
        opposite (a correction)."""
        return await asyncio.to_thread(feedback, body if isinstance(body, dict) else {}, store())

    app.include_router(router)


ON = ("answer", "proposal", "decision", "report")
RATINGS = ("helpful", "unhelpful", "wrong")


def feedback(body: dict[str, Any], book: Any) -> dict[str, Any]:
    from dclab_core.ledger import POINTS

    from ... import experience, ledger
    from ...accounts.principal import current
    from ...accounts.roles import allows

    on, rating = body.get("on"), body.get("rating")
    ref, reason = str(body.get("ref") or "")[:80], str(body.get("reason") or "").strip()[:300]
    if on not in ON or rating not in RATINGS:
        raise HTTPException(422, f"on is one of {', '.join(ON)} and rating one of {', '.join(RATINGS)}")
    ref = ref if ID.fullmatch(ref) else ""  # "on" is meta and leaves with an export: only an id's shape, never free text
    ids = {k: str(body[k])[:64] for k in ("draft", "project") if isinstance(body.get(k), str) and re.fullmatch(r"[0-9a-zA-Z_-]{1,64}", body[k])}
    record = None
    if on == "decision" and rating == "wrong" and book is not None and ref:
        record = next((r for r in book.records() if r.id == ref), None)
        if record is None:
            raise HTTPException(404, "No such decision")
    event = experience.record("feedback.given", "person", {"on": f"{on}:{ref}" if ref else on, "rating": rating, "reason": reason or None},
                              conversation=ids.get("draft"), project=ids.get("project"))
    labelled = None
    who = current()
    role = who.role if who is not None else "owner"  # no accounts: the local owner
    point = POINTS.get(record.point) if record is not None else None
    if point is not None and point.answer == "noul" and isinstance(record.value_used, bool) and (allows(role, "write") or allows(role, "review")):
        by = f"person:{who.user_id}" if who is not None and who.user_id else "person"
        if not any(x.source == "user_decision_exposed" and x.by == by for x in book.labels(record=record.id)):  # once per person
            out = ledger.label(record, not record.value_used, "user_decision_exposed", by, ledger=book)
            labelled = out.id if out is not None else None
    return {"recorded": event is not None, "label": labelled}
