"""Memory through the API (35.3): what a person and their agents remember, a correction, a review, a stated preference."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request

from ...memory import Refused, latest, open_memory, readers, view, writing
from ..api_models import MemoryCorrection, MemoryItemDoc, MemoryProposal, MemoryRetirement, MemoryReview, MemoryView, Router
from ..services import Services, services
from .intern import declared_role

router = Router()


def _person() -> str:
    from ...accounts.principal import current

    who = current()
    return who.user_id if who is not None and who.user_id else "owner"  # without sign-in, the local owner


def _audit(s: Services, request: Request, move: str, args: dict[str, Any], message: str) -> None:
    """Who reviewed or corrected what, in which role, in the workspace's audit trail (as a lesson's review is)."""
    try:
        from ... import audit

        role = declared_role(request) or "person"
        audit.record(s.audit, "memory", "human", role.capitalize(), move=move, status="allowed", args={**args, "by": _person()}, message=message[:300])
    except Exception:  # noqa: BLE001 - the change stands; its audit line is best effort, as for lessons
        pass


def _workspace(s: Services) -> str:
    return s.workspace_id or "local"


def _read(s: Services, project_id: str | None) -> list:
    """The items read where they live: the project's notes, the workspace's accepted lessons."""
    out = list(readers.from_lessons(s.lesson_store.list(status="accepted"), _workspace(s)))
    if project_id:
        try:
            out += readers.from_project(s.projects.get(project_id))
        except KeyError:
            raise HTTPException(404, "No such project") from None
    return out


def _chain(s: Services, project_id: str | None) -> list[tuple[str, str]]:
    chain = [("global", "dclab"), ("workspace", _workspace(s)), ("person", _person())]
    return chain + ([("project", project_id)] if project_id else [])


@router.get("/api/memory", response_model=MemoryView)
async def memory_view(request: Request, project_id: str | None = None, s: Services = Depends(services)):
    """What this person's agents remember (their scopes), and the drafts they may review."""
    store = open_memory(s.store.home)
    chain = _chain(s, project_id)
    items = view(store, chain, _read(s, project_id))
    reviewer = declared_role(request) in writing.REVIEWERS
    drafts = [i for i in latest(store.log()) if i.status == "draft" and (i.scope, i.scope_id) in set(chain)
              and (reviewer or (i.scope == "person" and i.scope_id == _person()))]
    return {"items": [i.model_dump(exclude={"vector"}) for i in items], "drafts": [i.model_dump(exclude={"vector"}) for i in drafts],
            "chain": [list(c) for c in chain]}


@router.post("/api/memory/{item_id}/correct", response_model=MemoryItemDoc)
async def memory_correct(item_id: str, body: MemoryCorrection, request: Request, s: Services = Depends(services)):
    """A person's correction: a new item superseding this one (stored, or read where it lives)."""
    store = open_memory(s.store.home)
    project_id = str(body.project_id) if body.project_id else None
    seen = {i.id: i for i in view(store, _chain(s, project_id), _read(s, project_id))}  # only what this person's agents see
    old = seen.get(item_id)
    if old is None:
        raise HTTPException(404, "No such item in what you see (a draft is reviewed, not corrected)")
    try:
        fixed = writing.correct(store, old, body.text, by=_person(), role=declared_role(request), reason=str(body.reason or ""), log=s.experience)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from None
    except (Refused, writing.ReviewError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_correct", {"item": item_id, "by_item": fixed.id, "scope": fixed.scope}, str(body.reason or ""))
    return fixed.model_dump(exclude={"vector"})


@router.post("/api/memory/{item_id}/retire", response_model=MemoryItemDoc)
async def memory_retire(item_id: str, body: MemoryRetirement, request: Request, s: Services = Depends(services)):
    """35.7: a person retires a stored line their agents see, with no replacement (kept in the log, never shown again)."""
    store = open_memory(s.store.home)
    project_id = str(body.project_id) if body.project_id else None
    seen = {i.id: i for i in view(store, _chain(s, project_id), _read(s, project_id))}  # only what this person's agents see
    old = seen.get(item_id)
    if old is None:
        raise HTTPException(404, "No such item in what you see (a draft is reviewed, not retired)")
    try:
        gone = writing.forget(store, old, by=_person(), role=declared_role(request), reason=str(body.reason or ""), log=s.experience)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from None
    except (Refused, writing.ReviewError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_retire", {"item": item_id, "scope": gone.scope}, str(body.reason or ""))
    return gone.model_dump(exclude={"vector"})


@router.post("/api/memory/{item_id}/review", response_model=MemoryItemDoc)
async def memory_review(item_id: str, body: MemoryReview, request: Request, s: Services = Depends(services)):
    """A draft accepted, edited or rejected: know-how by a reviewer, a preference by its own person."""
    try:
        item = writing.review(open_memory(s.store.home), item_id, str(body.action or ""), by=_person(), role=declared_role(request),
                              text=body.text, log=s.experience)
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from None
    except KeyError:
        raise HTTPException(404, "No such draft") from None
    except (Refused, writing.ReviewError, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_review", {"item": item_id, "action": str(body.action or ""), "status": item.status, "scope": item.scope}, "")
    return item.model_dump(exclude={"vector"})


@router.post("/api/memory/proposals", response_model=MemoryItemDoc | None)
async def memory_propose(body: MemoryProposal, s: Services = Depends(services)) -> Any:
    """A preference in the person's own words, proposed as a draft for them to confirm; null when the words state none."""
    item = writing.propose_preference(open_memory(s.store.home), person=_person(), message=body.message,
                                      source=str(body.source or "api:proposals")[:200], log=s.experience)
    return None if item is None else item.model_dump(exclude={"vector"})
