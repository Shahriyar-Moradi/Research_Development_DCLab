"""The intern (chat mode over the notebook) and the workspace's lessons."""

from __future__ import annotations

from fastapi import Depends, HTTPException, Request
from fastapi.responses import Response

from ... import lessons as workspace_lessons
from ...intern.sessions import EXAMPLE_TASKS
from ...intern.tools import Toolbox
from ...models import settings as model_settings
from ..api_models import InternMessage, InternStart, InternStatus, Lesson, LessonReview, Lessons, Router, Session
from ..services import Services, services


def declared_role(request: Request) -> str | None:
    """The reviewer's role: the signed-in person's (package 10.2); without accounts, the one the request declares."""
    from ...accounts.principal import current

    who = current()
    if who is not None and who.user_id:
        return who.role
    return (request.headers.get("x-dclab-role") or "").strip().lower() or None


router = Router()


@router.get("/api/intern", response_model=InternStatus)
async def intern_status(request: Request, s: Services = Depends(services)):
    cfg = model_settings.public(s.gateway.route("intern")[0])  # the tier the intern's requests go to now, routing included (no key)
    return {**cfg, "mode": "llm" if s.gateway.available("intern") else "standard", "examples": EXAMPLE_TASKS,
            "mcp_url": (str(request.base_url).rstrip("/") + "/mcp") if s.mcp is not None else None,
            "chat_ui": {"command": "make chat-ui", "intern_command": "make chat-ui-intern", "url": "http://localhost:5173/", "intern_url": "http://localhost:5173/?mode=ml-intern"},
            "tools": Toolbox(s.projects).names(), "default_budget": {"max_steps": 24, "max_minutes": 20},
            "note": ("The model plans and calls the tools; deterministic code runs every stage." if cfg["available"] else
                     "No model is configured, so the intern follows the standard DCLab plan. Set OPENAI_API_KEY (and OPENAI_BASE_URL for the Hugging Face router or Ollama) to let a model plan.")}


@router.get("/api/intern/sessions", response_model=list[Session])
async def intern_list(s: Services = Depends(services)):
    return s.intern_sessions.list()


@router.post("/api/intern/sessions", status_code=201, response_model=Session)
async def intern_start(body: InternStart, wait: bool = False, s: Services = Depends(services)):
    task = str(body.task if body.task is not None else "").strip()
    if len(task) < 8:
        raise HTTPException(422, "Describe the task in at least a sentence")
    project_id = body.project_id or None
    if project_id:
        s.project(project_id)
    agent = s.intern()
    try:
        session = agent.start(task, body.budget if isinstance(body.budget, dict) else None, project_id)
    except (TypeError, ValueError) as exc:
        raise HTTPException(422, f"budget: {exc}") from None
    job = s.start_intern_job(session["id"], {"action": "run"}, wait)
    if wait:
        await s.run_here(job)
    return s.public(s.session(session["id"]))


@router.get("/api/intern/sessions/{session_id}", response_model=Session)
async def intern_get(session_id: str, s: Services = Depends(services)):
    return {**s.public(s.session(session_id)), "trace": s.trace_of(session_id)}


@router.post("/api/intern/sessions/{session_id}/message", response_model=Session)
async def intern_message(session_id: str, body: InternMessage, wait: bool = False, s: Services = Depends(services)):
    session = s.session(session_id)
    if session["status"] in ("queued", "running"):
        raise HTTPException(409, "Wait for the current turn to finish")
    text = str(body.text if body.text is not None else "").strip()
    if not text:
        raise HTTPException(422, "Say something")
    job = s.start_intern_job(session_id, {"action": "message", "text": text}, wait)
    if wait:
        await s.run_here(job)
    return s.public(s.session(session_id))


@router.delete("/api/intern/sessions/{session_id}", status_code=204, response_class=Response)
async def intern_delete(session_id: str, s: Services = Depends(services)):
    s.session(session_id)
    if session_id in s.intern_jobs and not s.intern_jobs[session_id].done():
        raise HTTPException(409, "The session is still working")
    s.intern_sessions.delete(session_id)
    try:
        s.traces.delete(session_id)
    except Exception:  # noqa: BLE001 — the session is gone either way; a stray trace row is harmless
        pass
    return Response(status_code=204)


# ---------------------------------------------------------------------- the workspace's lessons (A5.3)


@router.get("/api/lessons", response_model=Lessons)
async def list_lessons(project_id: str | None = None, status: str | None = None, s: Services = Depends(services)):
    """The workspace's lessons, for the Evidence library's "From projects" panel and a project's page."""
    return {"lessons": s.lesson_store.list(project_id=project_id, status=status), "label": workspace_lessons.LABEL,
            "reviewers": list(workspace_lessons.REVIEWERS)}


@router.post("/api/lessons/{lesson_id}/review", response_model=Lesson)
async def review_lesson(lesson_id: str, body: LessonReview, request: Request, s: Services = Depends(services)):
    """A reviewer accepts, edits or rejects a lesson. Until accounts exist the role is the one the request declares
    (X-DCLab-Role), so this keeps an honest record of who reviewed in which role, not a verified login."""
    role = declared_role(request)
    try:
        lesson = workspace_lessons.review(s.lesson_store, lesson_id, str(body.action if body.action is not None else ""), role, str(body.reason or ""),
                                          {k: getattr(body, k) for k in ("claim", "against", "next_test")})
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from None
    except KeyError:
        raise HTTPException(404, "No such lesson") from None
    except workspace_lessons.ReviewError as exc:
        raise HTTPException(422, str(exc)) from None
    try:
        from ... import audit

        audit.record(s.audit, "lesson_review", "human", (role or "reviewer").capitalize(), project_id=lesson["project_id"],
                     move="review_lesson", status="allowed", args={"lesson": lesson_id, "action": lesson["review"]["action"], "status": lesson["status"]},
                     message=str(body.reason or "")[:300])
        s.projects.log(lesson["project_id"], "lesson_reviewed", {"lesson": lesson_id, "action": lesson["review"]["action"], "by": role,
                                                                "status": lesson["status"], "synthetic": lesson.get("synthetic", False)})
    except (KeyError, OSError):
        pass  # the project was deleted (KeyError on PostgreSQL, a missing folder on files); the lesson stands on its own
    return lesson
