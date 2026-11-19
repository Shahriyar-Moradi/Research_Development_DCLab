"""Memory through the API (35.3): what a person and their agents remember, a correction, a review, a stated preference."""

from __future__ import annotations

from typing import Any

from fastapi import Depends, HTTPException, Request

from ...memory import Refused, bases, latest, open_memory, readers, scope as mem_scope, summary as mem_summary, view, writing
from ..api_models import (
    BaseDraftConfirm, BaseFactCorrect, BaseFactRetire, BaseFactUpsert, BaseMutationResult, BaseView,
    MemoryCorrection, MemoryItemDoc, MemoryProposal, MemoryRetirement, MemoryReview, MemoryView,
    ProblemCreate, ProblemDoc, ProblemsView, ProblemSummaryView, Router,
)
from ..services import Services, services
from .intern import declared_role

router = Router()


def _person() -> str:
    from ...accounts.principal import current

    who = current()
    if mem_scope.auth_enabled():
        if who is None or not who.user_id or not who.role or who.via == "anonymous":
            raise HTTPException(403, "Authenticated user is required")
        return str(who.user_id)
    return who.user_id if who is not None and who.user_id else "owner"  # without sign-in, the local owner


def _resolve(s: Services, *, project_id: str | None = None, problem_id: str | None = None,
             require_write: bool = False, require_problem: bool = False, auto_create_legacy: bool = True) -> mem_scope.AuthorizedScope:
    try:
        return mem_scope.resolve_scope(
            workspace_id=_workspace(s),
            projects=s.projects if project_id else None,
            drafts=s.drafts,
            project_id=project_id,
            problem_id=problem_id,
            require_write=require_write,
            require_problem=require_problem,
            auto_create_legacy_problem=auto_create_legacy,
        )
    except mem_scope.AmbiguousProblemError as exc:
        raise HTTPException(409, {"error": "ambiguous_problem", "message": str(exc), "candidates": exc.candidates}) from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    except KeyError:
        raise HTTPException(404, "No such project") from None


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


def _chain(s: Services, project_id: str | None, problem_id: str | None = None) -> list[tuple[str, str]]:
    auth = _resolve(s, project_id=project_id, problem_id=problem_id, require_problem=False, auto_create_legacy=False)
    return auth.chain(include_personal=True)


@router.get("/api/memory", response_model=MemoryView)
async def memory_view(request: Request, project_id: str | None = None, problem_id: str | None = None, s: Services = Depends(services)):
    """What this person's agents remember (their authorized scopes), and the drafts they may review."""
    store = open_memory(s.store.home)
    chain = _chain(s, project_id, problem_id)
    items = view(store, chain, _read(s, project_id))
    reviewer = declared_role(request) in writing.REVIEWERS
    person_key = mem_scope.person_scope_id(_workspace(s), _person())
    drafts = [i for i in latest(store.log()) if i.status == "draft" and (i.scope, i.scope_id) in set(chain)
              and (reviewer or (i.scope == "person" and i.scope_id in (_person(), person_key)))]
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
    person_id = mem_scope.person_scope_id(_workspace(s), _person())
    item = writing.propose_preference(open_memory(s.store.home), person=person_id, message=body.message,
                                      source=str(body.source or "api:proposals")[:200], log=s.experience)
    return None if item is None else item.model_dump(exclude={"vector"})


# ------------------------------------------------------------------ Project & Problem Bases and Resumable Summary (A3-A6)

@router.get("/api/projects/{project_id}/problems", response_model=ProblemsView)
async def project_problems_list(project_id: str, s: Services = Depends(services)):
    """List durable active problems bound to a project (creates a single default problem lazily for legacy projects)."""
    auth = _resolve(s, project_id=project_id, require_problem=False, auto_create_legacy=False)
    try:
        probs = mem_scope.list_problems(s.projects, project_id, workspace_id=auth.workspace_id)
        if not probs:
            probs = [mem_scope.ensure_default_problem(s.projects, project_id, workspace_id=auth.workspace_id, by=auth.principal_id)]
        proj = s.projects.get(project_id)
    except KeyError:
        raise HTTPException(404, "No such project") from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    return {"project_id": project_id, "default_problem_id": proj.get("default_problem_id"), "problems": probs}


@router.post("/api/projects/{project_id}/problems", response_model=ProblemDoc)
async def project_problems_create(project_id: str, body: ProblemCreate, request: Request, s: Services = Depends(services)):
    """Create a sibling problem inside a project with its own opaque problem_id and independent Problem Base."""
    auth = _resolve(s, project_id=project_id, require_write=True, require_problem=False, auto_create_legacy=False)
    try:
        prob = mem_scope.create_problem(
            s.projects, project_id, str(body.name or "Untitled problem"),
            workspace_id=auth.workspace_id, by=auth.principal_id,
            draft_id=str(body.draft_id) if body.draft_id else None,
            description=str(body.description or ""),
        )
    except KeyError:
        raise HTTPException(404, "No such project") from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    _audit(s, request, "problem_create", {"project_id": project_id, "problem_id": prob["problem_id"]}, prob["name"])
    return prob


@router.get("/api/projects/{project_id}/memory/base", response_model=BaseView)
async def project_memory_base_view(project_id: str, target: str = "problem", problem_id: str | None = None,
                                   s: Services = Depends(services)):
    """View the editable Project Base or Problem Base with source, scope, active revision, and pending drafts."""
    if target not in ("project", "problem"):
        raise HTTPException(422, "target must be 'project' or 'problem'")
    auth = _resolve(s, project_id=project_id, problem_id=problem_id, require_problem=(target == "problem"))
    return bases.read_base(s.store.home, auth, target=target)


@router.post("/api/projects/{project_id}/memory/base", response_model=BaseMutationResult)
async def project_memory_base_upsert(project_id: str, body: BaseFactUpsert, request: Request, s: Services = Depends(services)):
    """Add or update a fact on the Project Base or Problem Base with optimistic revision conflict checking."""
    target = str(body.target or "problem")
    if target not in ("project", "problem"):
        raise HTTPException(422, "target must be 'project' or 'problem'")
    auth = _resolve(s, project_id=project_id, problem_id=(str(body.problem_id) if body.problem_id else None),
                    require_write=True, require_problem=(target == "problem"))
    try:
        res = bases.upsert_base_fact(
            s.store.home, auth, target=target,
            subject=str(body.subject or target), field=str(body.field or "fact"),
            text=str(body.text or ""), source=str(body.source or f"ui:{target}_base"),
            expected_revision=(int(body.expected_revision) if body.expected_revision is not None else None),
            reason=str(body.reason or ""), log=s.experience,
        )
    except bases.RevisionConflictError as exc:
        raise HTTPException(409, exc.to_detail()) from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    except (Refused, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_base_upsert",
           {"project_id": project_id, "problem_id": auth.problem_id, "target": target, "revision": res["revision"]},
           str(body.field or ""))
    return res


@router.post("/api/projects/{project_id}/memory/base/{entry_id}/correct", response_model=BaseMutationResult)
async def project_memory_base_correct(project_id: str, entry_id: str, body: BaseFactCorrect, request: Request,
                                      s: Services = Depends(services)):
    """Correct an active Project Base or Problem Base fact, preserving history and marking dependent advice stale."""
    target = str(body.target or "problem")
    if target not in ("project", "problem"):
        raise HTTPException(422, "target must be 'project' or 'problem'")
    auth = _resolve(s, project_id=project_id, problem_id=(str(body.problem_id) if body.problem_id else None),
                    require_write=True, require_problem=(target == "problem"))
    try:
        res = bases.correct_base_fact(
            s.store.home, auth, entry_id, str(body.text or ""), target=target,
            expected_revision=(int(body.expected_revision) if body.expected_revision is not None else None),
            reason=str(body.reason or ""), log=s.experience,
        )
    except KeyError:
        raise HTTPException(404, "No such base entry") from None
    except bases.RevisionConflictError as exc:
        raise HTTPException(409, exc.to_detail()) from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    except (Refused, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_base_correct",
           {"project_id": project_id, "problem_id": auth.problem_id, "entry_id": entry_id, "revision": res["revision"]},
           str(body.reason or ""))
    return res


@router.post("/api/projects/{project_id}/memory/base/{entry_id}/retire", response_model=BaseMutationResult)
async def project_memory_base_retire(project_id: str, entry_id: str, body: BaseFactRetire, request: Request,
                                     s: Services = Depends(services)):
    """Retire an active Project Base or Problem Base fact, invalidating dependent summaries and advice."""
    target = str(body.target or "problem")
    if target not in ("project", "problem"):
        raise HTTPException(422, "target must be 'project' or 'problem'")
    auth = _resolve(s, project_id=project_id, problem_id=(str(body.problem_id) if body.problem_id else None),
                    require_write=True, require_problem=(target == "problem"))
    try:
        res = bases.retire_base_fact(
            s.store.home, auth, entry_id, target=target,
            expected_revision=(int(body.expected_revision) if body.expected_revision is not None else None),
            reason=str(body.reason or ""), log=s.experience,
        )
    except KeyError:
        raise HTTPException(404, "No such base entry") from None
    except bases.RevisionConflictError as exc:
        raise HTTPException(409, exc.to_detail()) from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    except (Refused, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_base_retire",
           {"project_id": project_id, "problem_id": auth.problem_id, "entry_id": entry_id, "revision": res["revision"]},
           str(body.reason or ""))
    return res


@router.post("/api/projects/{project_id}/memory/base/{entry_id}/confirm", response_model=BaseMutationResult)
async def project_memory_base_confirm(project_id: str, entry_id: str, body: BaseDraftConfirm, request: Request,
                                      s: Services = Depends(services)):
    """Confirm or reject a provisional AI draft on the Project or Problem Base; blocks stale drafts until rechecked."""
    target = str(body.target or "problem")
    if target not in ("project", "problem"):
        raise HTTPException(422, "target must be 'project' or 'problem'")
    auth = _resolve(s, project_id=project_id, problem_id=(str(body.problem_id) if body.problem_id else None),
                    require_write=True, require_problem=(target == "problem"))
    try:
        res = bases.confirm_draft(
            s.store.home, auth, entry_id, action=str(body.action or "accept"), target=target,
            expected_revision=(int(body.expected_revision) if body.expected_revision is not None else None),
            allow_stale=bool(body.allow_stale), log=s.experience,
        )
    except KeyError:
        raise HTTPException(404, "No such draft entry") from None
    except bases.RevisionConflictError as exc:
        raise HTTPException(409, exc.to_detail()) from None
    except bases.StaleDependencyError as exc:
        raise HTTPException(409, {"error": "stale_dependency", "message": str(exc), "changed_sources": exc.changed_sources}) from None
    except mem_scope.AccessDenied as exc:
        raise HTTPException(403, str(exc)) from None
    except (Refused, ValueError) as exc:
        raise HTTPException(422, str(exc)) from None
    _audit(s, request, "memory_base_confirm",
           {"project_id": project_id, "problem_id": auth.problem_id, "entry_id": entry_id, "action": str(body.action or "accept")}, "")
    return res


@router.get("/api/projects/{project_id}/memory/summary", response_model=ProblemSummaryView)
async def project_memory_summary(project_id: str, problem_id: str | None = None, s: Services = Depends(services)):
    """Read the active versioned resumable summary for a problem, with resume URL and viewer-only personal overlay."""
    auth = _resolve(s, project_id=project_id, problem_id=problem_id, require_problem=True)
    doc = mem_summary.read_summary(s.store.home, auth, auto_rebuild=True, include_personal_overlay=True)
    doc["resume_url"] = f"/projects/{project_id}?problem={auth.problem_id}"
    return doc
