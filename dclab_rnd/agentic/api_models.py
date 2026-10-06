"""The request and response models of the API (package 10.1).

Each model names the fields a page relies on, so the OpenAPI schema describes every route, and lets the rest through
(``extra="allow"``): a model documents a payload, it never trims one. Routes are built with ``Router``, which serialises
with ``exclude_unset`` so a response carries exactly the keys it carried before, no new nulls. A request model is
permissive on purpose: the route keeps the checks and the messages it had (a missing name is "A project needs a name",
not a list of pydantic errors); what the model adds is that a body that is not JSON, or not an object, is a 422
everywhere instead of a 500 in some places.
"""

from __future__ import annotations

from typing import Any, Callable

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict


class Router(APIRouter):
    """An APIRouter whose responses keep their own keys: a response model documents, ``exclude_unset`` adds nothing.
    Every route it builds passes the one role check (package 10.2, ``accounts.guard``): what it needs follows from its
    method and path, so no route is left unchecked."""

    def add_api_route(self, path: str, endpoint: Callable[..., Any], **kwargs: Any) -> None:
        from fastapi import Depends

        from ..accounts.guard import permission, require

        kwargs["response_model_exclude_unset"] = True  # always: FastAPI's decorators pass False explicitly, so a default would never hold
        methods = sorted(kwargs.get("methods") or ["GET"])
        needed = sorted({permission(m, self.prefix + path) for m in methods})  # every method's: a route with several needs them all
        kwargs["dependencies"] = [*(kwargs.get("dependencies") or []), Depends(require(*needed))]
        super().add_api_route(path, endpoint, **kwargs)


def api_routes(app: Any) -> list[Any]:
    """Every route of the app, those of included routers too (this FastAPI keeps an included router as one entry)."""
    from fastapi.routing import APIRoute

    def walk(routes):
        for route in routes:
            inner = getattr(route, "original_router", None)
            if inner is not None:
                yield from walk(inner.routes)
            elif isinstance(route, APIRoute):
                yield route
    return list(walk(app.routes))


class Doc(BaseModel):
    """A JSON object whose named fields are documented and whose other fields pass through unchanged."""

    model_config = ConfigDict(extra="allow")


OCTET = {"requestBody": {"required": True, "content": {"application/octet-stream": {"schema": {"type": "string", "format": "binary"}}}}}


def download(media_type: str) -> dict[int | str, dict[str, Any]]:
    return {200: {"content": {media_type: {}}, "description": "A file, sent as an attachment"}}


# ---------------------------------------------------------------------- requests


class ProjectCreate(Doc):
    name: Any = ""
    industry: Any = "general"
    goal: Any = ""


class ProjectPatch(Doc):
    name: Any = None
    goal: Any = None
    industry: Any = None
    settings: Any = None  # max_rows, quick, folds, share_for_training
    policy: Any = None  # require_solution_signoff, require_holdout_approval


class SampleChoice(Doc):
    key: Any = None


class ProposalRequest(Doc):
    target: Any = ""
    task: Any = None


class SolutionBody(Doc):
    """The solution as the Solution & data page saves it; ``studio.solution.Solution`` checks it and words its errors."""

    target: Any = None
    task: Any = None
    positive_label: Any = None
    prediction_moment: Any = None
    forbidden: Any = None
    identifiers: Any = None
    time_column: Any = None
    group_column: Any = None
    text_columns: Any = None
    metric: Any = None
    notes: Any = None


class StageApproval(Doc):
    choice: Any = None


class GateApproval(Doc):
    gate: Any = ""
    reason: Any = ""


class MoveCheck(Doc):
    move: Any = ""
    actor: Any = "human"
    stage: Any = None
    choice: Any = None
    gate: Any = None
    reuse_reason: Any = None


class Question(Doc):
    question: Any = ""


class LessonReview(Doc):
    action: Any = ""
    reason: Any = None
    claim: Any = None
    against: Any = None
    next_test: Any = None


class RoutingChange(Doc):
    purpose: Any = ""
    setting: Any = ""
    tier: Any = None
    reason: Any = None


class InternStart(Doc):
    task: Any = ""
    project_id: Any = None
    budget: Any = None


class InternMessage(Doc):
    text: Any = ""


class DraftCreate(Doc):
    problem: Any = ""
    pack: Any = None


class AnyObject(Doc):
    """A JSON object the route reads itself (its fields are listed in the route's description)."""


# ---------------------------------------------------------------------- responses


class Config(Doc):
    csrf: str | None = None
    api_key_configured: bool | None = None
    default_model: str | None = None
    projects: list[Any] | None = None
    datasets: Any = None


class Status(Doc):
    status: str | None = None


class Run(Doc):
    id: str | None = None
    status: Any = None
    events: list[Any] | None = None


class Project(Doc):
    id: str | None = None
    name: Any = None
    goal: Any = None
    industry: Any = None
    data: Any = None
    solution: Any = None
    settings: Any = None
    stages: Any = None
    records: Any = None
    activity: Any = None
    transitions: Any = None
    graph: Any = None
    memory: Any = None


class Proposal(Doc):
    target: Any = None
    task: Any = None
    forbidden: Any = None
    identifiers: Any = None


class Graph(Doc):
    nodes: Any = None
    current: Any = None
    state: Any = None
    moves: Any = None
    transitions: Any = None


class Verdict(Doc):
    move: Any = None
    status: Any = None
    message: Any = None


class Answer(Doc):
    answer: Any = None
    proof: Any = None


class Review(Doc):
    summary: Any = None
    cells: Any = None
    findings: Any = None
    fixes_available: Any = None


class Fixes(Doc):
    fixes: Any = None
    findings: Any = None


class Lesson(Doc):
    id: Any = None
    status: Any = None
    claim: Any = None
    scope: Any = None


class Lessons(Doc):
    lessons: list[Any] | None = None
    label: Any = None


class ModelsSummary(Doc):
    tiers: Any = None
    purposes: Any = None
    usage: Any = None
    agreement: Any = None


class InternStatus(Doc):
    mode: Any = None
    model: Any = None
    tools: Any = None


class Session(Doc):
    id: Any = None
    status: Any = None
    task: Any = None
    project_id: Any = None
    steps: Any = None
    trace: Any = None


class Workspace(Doc):
    projects: Any = None
    needs: Any = None
    stats: Any = None


class Draft(Doc):
    id: Any = None
    problem: Any = None
    status: Any = None
    assets: Any = None
    solution: Any = None


class Page(Doc):
    """A page's data (the Lab, Learn, Platform, Evidence and Jobs pages): see the route's description for its parts."""
