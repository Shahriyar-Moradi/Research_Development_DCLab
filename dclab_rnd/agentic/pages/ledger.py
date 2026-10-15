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

from fastapi import Body, HTTPException, Request

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
    async def ledger_blind_next(point: str | None = None, skip: str | None = None):
        """The next decision drawn for a blind label (its question, subject, state and options), or none left.
        ``skip``: decision ids, comma-separated, that this reviewer could not tell (they stay unlabelled)."""
        from ...ledger import labels

        if point is not None and not re.fullmatch(r"[a-z_.]{3,60}", point):
            raise HTTPException(422, "point is a decision point")
        skipped = [x for x in (skip or "").split(",") if x][:500]
        if not all(ID.fullmatch(x) for x in skipped):
            raise HTTPException(422, "skip lists decision ids")
        item = await asyncio.to_thread(labels.next_blind, store(), "blind-v1", point, skipped)
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
        demoted: list = []
        if store() is not None and getattr(ctx, "models", None) is not None:  # 22.6: a new label may send a routed model back to shadow
            from ...ledger import routing

            def check() -> list:
                routed = [p for p, e in (ctx.models.routing.get().get("decisions") or {}).items() if e.get("level")]
                if out.point not in routed:  # only a label at a routed point can change a routed model's figures
                    return []
                return routing.demote(ctx.models.routing, ctx.audit, store().records(point=out.point), store().labels(point=out.point))

            try:
                demoted = await asyncio.to_thread(check)
            except Exception:  # noqa: BLE001 - the label is kept; a failed check runs again with the next label
                demoted = []
        return {"label": out.model_dump(mode="json"), "demoted": demoted}

    @router.get("/api/ledger/routing", response_model=Page)
    async def ledger_routing():
        """Which answerer serves each decision point and at which level (22.6), the answerers that may, best first, and
        the changes made."""
        from dclab_core.ledger import POINTS

        from ...ledger import routing, shadow

        doc = ctx.models.routing.get()
        measured = await asyncio.to_thread(shadow.measure, store().records() if store() is not None else [], store().labels() if store() is not None else [])
        points = sorted({p for p in POINTS if routing.entry(doc, p) or routing.candidates(p)})
        return {"points": {p: {"routed": routing.entry(doc, p) or None, "candidates": routing.candidates(p, measured)} for p in points},
                "history": [h for h in doc.get("history") or [] if h.get("decision_point")][-50:]}

    @router.post("/api/ledger/routing", response_model=Page)
    async def ledger_route(request: Request, body: Any = Body(None)):
        """One audited change of a point's answerer: {"point", "answerer" (None: the rule alone), "level" (0 shadow, 1 flag,
        2 act with a check, 3 act), "reason", "bar"}; a reviewer or the owner, and above shadow only an answerer that
        passed its sealed gate."""
        from ...ledger import routing
        from ..routers.core import declared_role

        from ...ledger import shadow

        b = body if isinstance(body, dict) else {}
        point = str(b.get("point") or "")
        book = store()

        def measured() -> dict[str, Any]:  # the shadow figures a promotion to act rests on
            return shadow.measure(book.records(point=point), book.labels(point=point)) if book is not None else {}

        try:
            level = int(b.get("level") or 0)
            doc = await asyncio.to_thread(lambda: routing.switch(ctx.models.routing, ctx.audit, point, b.get("answerer"), level, declared_role(request),
                                                                 str(b.get("reason") or ""), b.get("bar", routing.BAR),
                                                                 measured=measured() if level >= 2 else None))
        except PermissionError as exc:
            raise HTTPException(403, str(exc)) from None
        except (routing.DecisionRoutingError, ValueError, TypeError) as exc:
            raise HTTPException(422, str(exc)) from None
        return {"routed": routing.entry(doc, str(b.get("point")))}

    @router.post("/api/decide", response_model=Page)
    async def decide_route(body: Any = Body(None)):
        """One technical decision about a column, asked from its state (name, name tokens and bands, never a value): what
        DCLab would do, the rule's and the classifiers' answers with their confidence, any disagreement, the evidence
        (22.8; the same as the MCP tool and the notebook companion)."""
        from ... import decide

        b = body if isinstance(body, dict) else {}
        try:
            return await asyncio.to_thread(decide.ask, str(b.get("point") or ""), b.get("state") if isinstance(b.get("state"), dict) else {},
                                           b.get("options"), ctx.models.routing.get() if getattr(ctx, "models", None) is not None else None)
        except decide.DecideError as exc:
            raise HTTPException(422, str(exc)) from None

    @router.post("/api/decide/choose", response_model=Page)
    async def decide_choose(body: Any = Body(None)):
        """A person's own answer to a technical decision: {"point", "state", "value"}; recorded as a decision with the
        rule's and the models' answers beside it, and as a label when it differs from what DCLab said."""
        from ... import decide
        from ...accounts.principal import current

        b = body if isinstance(body, dict) else {}
        who = current()
        by = f"person:{who.user_id}" if who is not None and who.user_id else "person"
        try:
            return await asyncio.to_thread(decide.choose, str(b.get("point") or ""), b.get("state") if isinstance(b.get("state"), dict) else {},
                                           b.get("value"), by, None, store())
        except decide.DecideError as exc:
            raise HTTPException(422, str(exc)) from None

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
