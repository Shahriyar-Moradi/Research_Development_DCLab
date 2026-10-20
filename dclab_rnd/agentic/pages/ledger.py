"""Decisions: the ledger on the Admin page (packages 19.7, 19.8).

    GET  /api/ledger/summary              how many decisions by point, kind and who decided; labels by source
    GET  /api/ledger/blind/next           the next decision drawn for a blind label: its question and state only
    GET  /api/ledger/queue                29.8: the drawn disagreements (a point's student against the rule), shown blind
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

    def home() -> Any:  # 29.8: the workspace's folder, where the testers' collected records are
        from ... import context

        found = getattr(context.services() or ctx, "store", None)
        return getattr(found, "home", None)

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

    @router.get("/api/ledger/queue", response_model=Page)
    async def ledger_queue(point: str | None = None, limit: int = 50, skip: str | None = None, order: str = "fair"):
        """29.8: the decisions where a point's student disagrees with the rule, drawn for blind labels and not yet labelled,
        in an order drawn from the seed (any part labelled is a fair sample): each shown blind (its question, subject, state
        and options), with the weight its label will carry; ``points``: how many each point has waiting. Label one with
        POST /api/ledger/blind/{record_id}. ``order=teach`` (29.11): by what a label would teach, on any decision a
        student answered; label one with {"value", "taught": true}: it trains, never estimates."""
        from ...ledger import labels

        if order not in ("fair", "teach"):
            raise HTTPException(422, "order is fair or teach")
        if point is not None and not re.fullmatch(r"[a-z_.]{3,60}", point):
            raise HTTPException(422, "point is a decision point")
        skipped = [x for x in (skip or "").split(",") if x][:500]
        if not all(ID.fullmatch(x) for x in skipped):
            raise HTTPException(422, "skip lists decision ids")
        return await asyncio.to_thread(lambda: labels.queue(store(), point, max(1, min(int(limit), 200)), "blind-v1", skipped, order=order))

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
        taught = isinstance(body, dict) and body.get("taught") is True  # 29.11: asked for by the teaching order (exactly true)
        who = current()
        by = f"reviewer:{who.user_id}" if who is not None and who.user_id else "reviewer"
        try:
            out = await asyncio.to_thread(lambda: labels.blind_label(record_id, value, by, "blind-v1", store(), taught=taught))
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
                from ...ledger import shadow

                return routing.demote(ctx.models.routing, ctx.audit, *shadow.labelled(store(), home(), point=out.point))  # 29.8: testers' labels too

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
        measured = await asyncio.to_thread(lambda: shadow.measure(*shadow.labelled(store(), home())))  # 29.8: the testers' collected labels too
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
            return shadow.measure(*shadow.labelled(book, home(), point=point)) if book is not None else {}  # 29.8: and the testers' labels

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
        """A person's feedback (19.9, 23.6): {"on": answer | message | step | proposal | decision | note | report, "ref": its
        id, "kind": thumbs | rating | correction | edited_output | override (from the rating when not given), "rating":
        helpful | unhelpful | wrong, "score": 1 to 5, "edited": the reply as the person would have it, "reason": optional,
        "draft", "project" or "conversation": where}. It joins the session record with the version of the message
        judged; "wrong" on a yes/no decision from someone who may change the work or review it is also that person's
        label saying the opposite (a correction)."""
        return await asyncio.to_thread(feedback, body if isinstance(body, dict) else {}, store())

    app.include_router(router)


def message_version(on: str, ref: str, conversation: str | None) -> str | None:
    """23.6: the release that wrote the message judged (the chat.agent event carrying its id); None when that cannot be
    found (a message from before the stamp, a note, a log that cannot be read): the release now would claim today's
    prompts wrote what they did not."""
    from ... import experience

    log = experience.current_log()
    if log is not None and ref and on in ("answer", "message"):
        try:
            for e in reversed(log.read(conversation=conversation, kinds=["chat.agent"], limit=2000) if conversation else
                              log.read(kinds=["chat.agent"], limit=2000)):
                if e.payload.get("message") == ref and e.payload.get("version"):
                    return str(e.payload["version"])
        except Exception:  # noqa: BLE001 - a log that cannot be read: no version
            pass
    return None


def feedback(body: dict[str, Any], book: Any) -> dict[str, Any]:
    from dclab_core import feedback as core
    from dclab_core.ledger import POINTS

    from ... import experience, ledger
    from ...accounts.principal import current
    from ...accounts.roles import allows

    who = current()
    role = who.role if who is not None else "owner"  # no accounts: the local owner
    try:
        payload = core.check({**body, "role": role})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    if payload["kind"] == "edited_output" and role in core.READ_ONLY:
        raise HTTPException(403, "Your role can react to a reply but not rewrite it: what DCLab learns from is for the people who change the work")
    on, _, ref = payload["on"].partition(":")
    rating = payload["rating"]
    ids = {k: str(body[k])[:64] for k in ("draft", "project", "conversation") if isinstance(body.get(k), str) and re.fullmatch(r"[0-9a-zA-Z_-]{1,64}", body[k])}
    conversation = ids.get("draft") or ids.get("conversation")
    record = None
    if on == "decision" and rating in ("wrong", "overridden") and book is not None and ref:
        record = next((r for r in book.records() if r.id == ref), None)
        if record is None:
            raise HTTPException(404, "No such decision")
    event = experience.record("feedback.given", "person", {**payload, "version": message_version(on, ref, conversation)},
                              conversation=conversation, project=ids.get("project"))
    from ... import agent_metrics

    if rating in ("helpful", "unhelpful", "wrong"):  # the agents' ledger: a person's rating of a reply, by the role that wrote it
        agent_name = "home_agent" if on in ("message", "reply", "") else (record.point if record is not None else on)  # a chat reply is the Home agent's
        agent_metrics.emit(agent_name, "feedback_helpful", 1.0 if rating == "helpful" else 0.0, ref=ref or None)
    labelled = None
    point = POINTS.get(record.point) if record is not None and rating == "wrong" else None
    if point is not None and point.answer == "noul" and isinstance(record.value_used, bool) and (allows(role, "write") or allows(role, "review")):
        by = f"person:{who.user_id}" if who is not None and who.user_id else "person"
        if not any(x.source == "user_decision_exposed" and x.by == by for x in book.labels(record=record.id)):  # once per person
            out = ledger.label(record, not record.value_used, "user_decision_exposed", by, ledger=book)
            labelled = out.id if out is not None else None
    return {"recorded": event is not None, "label": labelled}
