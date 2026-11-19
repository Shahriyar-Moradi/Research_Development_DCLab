"""Host-resolved identity, durable problem bindings, and authorized scope checks (Package A3).

Every memory read, write, summary build, and background job resolves an :class:`AuthorizedScope` from validated server
state. Local single-user fallback (``owner``) applies only in explicit no-auth mode (when ``DCLAB_AUTH`` is unset; the
Admin sign-in there is still the machine's owner). When authentication is enabled or a worker runs on behalf of a user
(``require_principal``), missing identity, revoked membership, cross-workspace/project/problem references, or guessed
IDs fail closed. A read never creates a problem (:class:`NoProblemError`); write routes and the build do, under the
project's lock, and each creation is recorded.
"""

from __future__ import annotations

import contextlib
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Optional

from dclab_core.memory import problem_scope_id

from ..accounts.roles import PERMISSIONS, ROLES

# The Router's own role sets (accounts/roles.py): who may write a workspace, and who may read it. A reviewer reads here;
# a base edit is a write, as every other project write is.
WRITE_ROLES = PERMISSIONS["write"]
READ_ROLES = frozenset(ROLES)


class AccessDenied(PermissionError):
    """Raised when a principal lacks verified access to a workspace, project, problem, draft, or write action."""


class NoProblemError(AccessDenied):
    """Raised when a read finds a project with no bound problem: a read never creates one (a write route or the build does)."""


class AmbiguousProblemError(ValueError):
    """Raised when a project has multiple active problems and no explicit problem_id was selected."""

    def __init__(self, project_id: str, candidates: list[dict[str, Any]]):
        self.project_id = project_id
        self.candidates = list(candidates)
        names = ", ".join(f"{c['problem_id']} ({c.get('name') or 'Untitled'})" for c in self.candidates)
        super().__init__(f"Project {project_id} has multiple active problems; select one explicitly: {names}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def auth_enabled() -> bool:
    """True when account sign-in is configured for this deployment."""
    return bool(os.environ.get("DCLAB_AUTH", "").strip())


def workspace_of(store: Any) -> str:
    """The one workspace id every memory caller uses: the store's own (PostgreSQL's ``w…``), else ``local`` (a folder
    workspace). The router, Home, the build, the intern's brief and the tools all ask this, so they agree."""
    return str(getattr(store, "workspace_id", None) or "local")


def _same_workspace(recorded: Any, workspace_id: str) -> bool:
    """A document stamped with no workspace, or with ``local`` (a folder workspace, or one migrated from it), belongs to the
    store it was read from: that store already holds one workspace's documents. Only another named workspace refuses."""
    stamp = str(recorded or "").strip()
    return not stamp or stamp == "local" or stamp == str(workspace_id)


# Person, project and session scope ids stay the raw ids every writer has always used (a project's notes, the build's
# hand-off, a person's preferences): the store is already one workspace's (its folder, or its rows), so a workspace
# prefix would only hide what was written before. Only the problem scope carries its parent chain (dclab_core's key).
def person_scope_id(workspace_id: str, principal_id: str) -> str:
    uid = str(principal_id or "").strip()
    if not uid:
        raise AccessDenied("principal_id is required")
    return uid


def project_scope_id(workspace_id: str, project_id: str) -> str:
    pid = str(project_id or "").strip()
    if not pid:
        raise AccessDenied("project_id is required")
    return pid


def session_scope_id(workspace_id: str, session_id: str) -> str:
    sid = str(session_id or "").strip()
    if not sid:
        raise AccessDenied("session_id is required")
    return sid


@dataclass(frozen=True)
class AuthorizedScope:
    workspace_id: str
    principal_id: str
    role: str
    project_id: Optional[str] = None
    problem_id: Optional[str] = None
    draft_id: Optional[str] = None
    session_id: Optional[str] = None
    auth_mode: str = "local"  # "local" (explicit no-auth mode) or "authenticated"
    problem_workspace: Optional[str] = None  # the workspace the problem was bound in (its key survives a move to PostgreSQL)

    @property
    def can_write(self) -> bool:
        return self.role in WRITE_ROLES

    @property
    def person_key(self) -> str:
        return person_scope_id(self.workspace_id, self.principal_id)

    @property
    def project_key(self) -> Optional[str]:
        return project_scope_id(self.workspace_id, self.project_id) if self.project_id else None

    @property
    def problem_key(self) -> Optional[str]:
        if not (self.project_id and self.problem_id):
            return None
        return problem_scope_id(self.problem_workspace or self.workspace_id, self.project_id, self.problem_id)

    @property
    def session_key(self) -> Optional[str]:
        return session_scope_id(self.workspace_id, self.session_id) if self.session_id else None

    def chain(self, *, include_personal: bool = True, include_session: bool = False) -> list[tuple[str, str]]:
        """Authorized (scope, scope_id) pairs wide to narrow for dclab_core.memory.resolve."""
        out: list[tuple[str, str]] = [("global", "dclab"), ("workspace", self.workspace_id)]
        if include_personal:
            out.append(("person", self.person_key))
        if self.project_id:
            out.append(("project", self.project_key))  # type: ignore[arg-type]
        if self.project_id and self.problem_id:
            out.append(("problem", self.problem_key))  # type: ignore[arg-type]
        if include_session and self.session_id:
            out.append(("session", self.session_key))  # type: ignore[arg-type]
        return out

    def shared_chain(self) -> list[tuple[str, str]]:
        """Authorized scopes containing ONLY shareable (non-personal, non-session) items."""
        return self.chain(include_personal=False, include_session=False)

    def personal_chain(self) -> list[tuple[str, str]]:
        """Viewer-specific private personal preference scope only."""
        return [("person", self.person_key)]

    def to_record(self) -> dict[str, Any]:
        return {
            "workspace_id": self.workspace_id,
            "principal_id": self.principal_id,
            "role": self.role,
            "project_id": self.project_id,
            "problem_id": self.problem_id,
            "draft_id": self.draft_id,
            "session_id": self.session_id,
            "auth_mode": self.auth_mode,
            "problem_workspace": self.problem_workspace,
        }


# ------------------------------------------------------------------ Durable Problem Identity on Projects

def _project_workspace(project: Mapping[str, Any], default_workspace: str) -> str:
    return str(default_workspace or project.get("workspace_id") or "local")


def _active_problems(project: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [dict(p) for p in (project.get("problems") or []) if isinstance(p, Mapping) and p.get("status", "active") == "active"]


def _check_project_workspace(project: Mapping[str, Any], project_id: str, workspace_id: str) -> None:
    if not _same_workspace(project.get("workspace_id"), workspace_id):
        raise AccessDenied(f"project {project_id!r} does not belong to workspace {workspace_id!r}")


def _locked(projects: Any, project_id: str):
    """The project's own write lock (both stores' ``turn``, re-entrant with their ``update``), so a read-modify-write of
    its problem list never loses a concurrent one; a store without it (a test double) runs unlocked."""
    turn = getattr(projects, "turn", None)
    return turn(project_id) if callable(turn) else contextlib.nullcontext()


def _record_problem(log: Any, action: str, prob: Mapping[str, Any], by: str) -> None:
    """A problem created or a draft bound to one, in the workspace's record (ids only, never the person's words)."""
    try:
        from .. import experience

        actor = "person" if by and by not in ("system", "agent") else "system"
        experience.record("decision.accepted", actor, {"point": f"problem.{action}", "choice": str(prob.get("problem_id") or ""),
                                                      "by": f"person:{by}" if actor == "person" else str(by or "system")},
                          log=log, project=str(prob.get("project_id") or "") or None,
                          **({"conversation": str(prob["draft_id"])} if prob.get("draft_id") else {}))
    except Exception:  # noqa: BLE001 - the binding stands; its record is best effort
        pass


def _new_problem(project: Mapping[str, Any], project_id: str, ws: str, *, name: str, by: str, draft_id: Optional[str],
                 description: Optional[str] = None) -> dict[str, Any]:
    prob = {
        "problem_id": "prob_" + secrets.token_hex(6),
        "name": str(name or "Untitled problem").strip()[:120],
        "workspace_id": ws,  # the problem's own identity: its memory key keeps this even after a move to PostgreSQL
        "project_id": project_id,
        "draft_id": draft_id,
        "status": "active",
        "revision": 1,
        "created_at": _now(),
        "updated_at": _now(),
        "created_by": by,
    }
    if description is not None:
        prob["description"] = str(description or "").strip()[:400]
    return prob


def list_problems(projects: Any, project_id: str, *, workspace_id: str = "local") -> list[dict[str, Any]]:
    """Return active problems bound to this project in this workspace."""
    project = projects.get(project_id)
    _check_project_workspace(project, project_id, workspace_id)
    return _active_problems(project)


def ensure_default_problem(
    projects: Any,
    project_id: str,
    *,
    workspace_id: str = "local",
    by: str = "owner",
    name: Optional[str] = None,
    draft_id: Optional[str] = None,
    log: Any = None,
) -> dict[str, Any]:
    """Create/bind the single default problem for a legacy or newly built project, idempotently by draft_id, under the
    project's write lock. Called by write routes and the build only (a read never creates one); each change is recorded."""
    with _locked(projects, project_id):
        project = projects.get(project_id)
        _check_project_workspace(project, project_id, workspace_id)
        ws = _project_workspace(project, workspace_id)
        existing = list(project.get("problems") or [])
        if draft_id:
            for entry in existing:
                if isinstance(entry, Mapping) and entry.get("draft_id") == draft_id:
                    return dict(entry)
        active = [e for e in existing if isinstance(e, Mapping) and e.get("status", "active") == "active"]
        if active:
            chosen = dict(active[0])
            if draft_id and not chosen.get("draft_id"):
                chosen["draft_id"] = draft_id
                chosen["updated_at"] = _now()
                updated = [chosen if isinstance(e, Mapping) and e.get("problem_id") == chosen["problem_id"] else e for e in existing]
                projects.update(project_id, problems=updated, default_problem_id=project.get("default_problem_id") or chosen["problem_id"])
                _record_problem(log, "bound", chosen, by)
            return chosen

        prob = _new_problem(project, project_id, ws, name=str(name or project.get("name") or "Primary problem"), by=by, draft_id=draft_id)
        existing.append(prob)
        projects.update(project_id, problems=existing, default_problem_id=prob["problem_id"])
    _record_problem(log, "created", prob, by)
    return dict(prob)


def create_problem(
    projects: Any,
    project_id: str,
    name: str,
    *,
    workspace_id: str = "local",
    by: str = "owner",
    draft_id: Optional[str] = None,
    description: str = "",
    log: Any = None,
) -> dict[str, Any]:
    """Create a new sibling problem inside a project (idempotent when the same draft_id is already bound), under the
    project's write lock; recorded."""
    with _locked(projects, project_id):
        project = projects.get(project_id)
        _check_project_workspace(project, project_id, workspace_id)
        ws = _project_workspace(project, workspace_id)
        existing = list(project.get("problems") or [])
        if draft_id:
            for entry in existing:
                if isinstance(entry, Mapping) and entry.get("draft_id") == draft_id:
                    return dict(entry)
        prob = _new_problem(project, project_id, ws, name=name, by=by, draft_id=draft_id, description=description)
        existing.append(prob)
        projects.update(project_id, problems=existing, default_problem_id=project.get("default_problem_id") or prob["problem_id"])
    _record_problem(log, "created", prob, by)
    return dict(prob)


def resolve_problem(
    projects: Any,
    project_id: str,
    problem_id: Optional[str] = None,
    *,
    draft_id: Optional[str] = None,
    workspace_id: str = "local",
    by: str = "owner",
    auto_create_legacy: bool = False,
) -> dict[str, Any]:
    """Resolve the active problem from validated server state; raises AmbiguousProblemError when >1 active problems exist."""
    project = projects.get(project_id)
    _check_project_workspace(project, project_id, workspace_id)

    active = _active_problems(project)
    if problem_id:
        found = next((p for p in active if p.get("problem_id") == problem_id), None)
        if found is None:
            raise AccessDenied(f"problem {problem_id!r} is not bound to project {project_id!r}")
        if str(found.get("project_id") or "") != str(project_id) or not _same_workspace(found.get("workspace_id"), workspace_id):
            raise AccessDenied(f"problem {problem_id!r} parentage check failed")
        return dict(found)

    if draft_id:
        bound = next((p for p in active if p.get("draft_id") == draft_id), None)
        if bound is not None:
            return dict(bound)

    if not active:
        if not auto_create_legacy:
            raise NoProblemError(f"project {project_id!r} has no bound problem yet")
        return ensure_default_problem(projects, project_id, workspace_id=workspace_id, by=by, draft_id=draft_id)

    if len(active) == 1:
        return dict(active[0])

    raise AmbiguousProblemError(project_id, active)


# ------------------------------------------------------------------ Host-Resolved Authorization

def _check_workspace_membership(principal: Any, workspace_id: str, membership_checker: Optional[Callable[[str, str], Optional[str]]] = None) -> str:
    """Verify principal membership in workspace_id; returns verified role or raises AccessDenied."""
    uid = getattr(principal, "user_id", None)
    role = getattr(principal, "role", None)
    if not uid or not role or getattr(principal, "via", None) == "anonymous":
        raise AccessDenied("authenticated principal is required")
    if membership_checker is not None:
        live_role = membership_checker(str(uid), str(workspace_id))
        if not live_role:
            raise AccessDenied(f"principal {uid!r} is not an active member of workspace {workspace_id!r}")
        return str(live_role)
    p_ws = getattr(principal, "workspace_id", None)
    workspaces = getattr(principal, "workspaces", ()) or ()
    for wid, _name, wrole in workspaces:
        if str(wid) == str(workspace_id):
            return str(wrole)
    if p_ws is not None and str(p_ws) == str(workspace_id):
        return str(role)
    if p_ws is None and not workspaces and getattr(principal, "via", None) == "local":
        return str(role)
    raise AccessDenied(f"principal {uid!r} does not belong to workspace {workspace_id!r}")


def _check_project_access(project: Mapping[str, Any], workspace_id: str, principal_id: str, role: str) -> None:
    """Verify project workspace binding and optional per-project member ACL."""
    if not _same_workspace(project.get("workspace_id"), workspace_id):
        raise AccessDenied(f"project {project.get('id')!r} belongs to another workspace")
    # Optional per-project member ACL: if configured, non-members (even within workspace) are denied.
    members = project.get("members")
    if isinstance(members, (list, tuple, set)):
        allowed_ids = {str(m.get("user_id") if isinstance(m, Mapping) else m) for m in members}
        if str(principal_id) not in allowed_ids:
            raise AccessDenied(f"principal {principal_id!r} is not authorized for project {project.get('id')!r}")


def resolve_scope(
    *,
    workspace_id: str = "local",
    principal: Any = None,
    projects: Any = None,
    drafts: Any = None,
    project_id: Optional[str] = None,
    problem_id: Optional[str] = None,
    draft_id: Optional[str] = None,
    session_id: Optional[str] = None,
    require_write: bool = False,
    require_principal: bool = False,
    require_problem: bool = False,
    auto_create_legacy_problem: bool = False,
    membership_checker: Optional[Callable[[str, str], Optional[str]]] = None,
) -> AuthorizedScope:
    """Resolve and verify an AuthorizedScope from validated server state. Fails closed on every violation."""
    from ..accounts.principal import current

    who = principal if principal is not None else current()
    # Sign-in decides the mode, never how the principal arrived: with DCLAB_AUTH off the Admin sign-in (via "admin") is
    # still the machine's owner, and a worker that carries a person asks for the check with require_principal.
    is_auth = auth_enabled() or require_principal

    if not is_auth:
        # Explicit no-auth local single-user mode only.
        if who is not None and getattr(who, "via", None) == "anonymous":
            raise AccessDenied("anonymous access is denied")
        uid = str(getattr(who, "user_id", None) or "owner")
        role = str(getattr(who, "role", None) or "owner")
        ws = str(workspace_id or "local")
        auth_mode = "local"
    else:
        if who is None:
            raise AccessDenied("missing principal in authenticated or worker context")
        ws = str(workspace_id or getattr(who, "workspace_id", None) or "")
        if not ws:
            raise AccessDenied("workspace_id is required")
        role = _check_workspace_membership(who, ws, membership_checker=membership_checker)
        uid = str(who.user_id)
        auth_mode = "authenticated"

    if role not in READ_ROLES:
        raise AccessDenied(f"role {role!r} is not permitted")
    if require_write and role not in WRITE_ROLES:
        raise AccessDenied(f"role {role!r} does not have write permission")

    # Validate draft binding if draft_id is supplied.
    resolved_project_id = project_id
    resolved_problem_id = problem_id
    problem_ws: Optional[str] = None
    if draft_id and drafts is not None:
        try:
            draft = drafts.get(draft_id)
        except KeyError as exc:
            raise AccessDenied(f"draft {draft_id!r} not found") from exc
        if not _same_workspace(draft.get("workspace_id"), ws):
            raise AccessDenied(f"draft {draft_id!r} does not belong to workspace {ws!r}")
        bound_proj = draft.get("project_id")
        bound_prob = draft.get("problem_id")
        if bound_proj is None:
            # Unbound draft: never attach project or problem memory unless caller explicitly passes a verified project_id?
            # Wait: if caller passed a project_id or problem_id with an unbound draft, refuse mismatch!
            if resolved_project_id is not None or resolved_problem_id is not None:
                raise AccessDenied(f"draft {draft_id!r} is unbound and cannot access project or problem memory")
        else:
            if resolved_project_id is not None and str(resolved_project_id) != str(bound_proj):
                raise AccessDenied(f"draft {draft_id!r} is bound to another project")
            resolved_project_id = str(bound_proj)
            if bound_prob is not None:
                if resolved_problem_id is not None and str(resolved_problem_id) != str(bound_prob):
                    raise AccessDenied(f"draft {draft_id!r} is bound to another problem")
                resolved_problem_id = str(bound_prob)

    if resolved_problem_id and not resolved_project_id:
        raise AccessDenied("problem_id requires a verified project_id")

    if resolved_project_id:
        if projects is None:
            raise AccessDenied("project store is required to verify project access")
        try:
            project = projects.get(resolved_project_id)
        except KeyError as exc:
            raise AccessDenied(f"project {resolved_project_id!r} not found") from exc
        _check_project_access(project, ws, uid, role)

        if resolved_problem_id or require_problem:
            prob = resolve_problem(
                projects, resolved_project_id, resolved_problem_id,
                draft_id=draft_id, workspace_id=ws, by=uid,
                # only a writer's write creates a legacy project's first problem; a read (and a viewer) never writes the project
                auto_create_legacy=auto_create_legacy_problem and require_write and role in WRITE_ROLES,
            )
            resolved_problem_id = prob["problem_id"]
            problem_ws = str(prob.get("workspace_id") or "") or None

    return AuthorizedScope(
        workspace_id=ws,
        principal_id=uid,
        role=role,
        project_id=resolved_project_id,
        problem_id=resolved_problem_id,
        draft_id=draft_id,
        session_id=session_id,
        auth_mode=auth_mode,
        problem_workspace=problem_ws,
    )


def live_membership(url: Optional[str] = None) -> Optional[Callable[[str, str], Optional[str]]]:
    """The person's role in a workspace as the accounts database says it now (``Accounts.role_of``), or None when there is
    no database to ask. A failed lookup answers None, so a revoked or unreachable membership fails closed."""
    from ..storage import db

    url = url or db.database_url(required=False)
    if not url:
        return None

    def _role(uid: str, wid: str) -> Optional[str]:
        try:
            from ..accounts.store import Accounts

            role = Accounts(url).role_of(str(uid), str(wid))
        except Exception:  # noqa: BLE001 - the database cannot say: no verified role
            return None
        return str(role) if role else None

    return _role


def revalidate_job_scope(
    env: Any,
    job: Mapping[str, Any],
    *,
    require_write: bool = False,
    require_problem: bool = False,
    membership_checker: Optional[Callable[[str, str], Optional[str]]] = None,
) -> AuthorizedScope:
    """Revalidate a background job's initiating subject and resource IDs at execution time; fails closed if revoked."""
    from ..accounts.principal import Principal

    user = job.get("user")
    payload = job.get("payload") or {}
    # The job acts in its worker's workspace (the env's store): the person's role is checked there, not where they were.
    workspace_id = str(
        getattr(env, "workspace_id", None)
        or (user.get("workspace_id") if isinstance(user, Mapping) else None)
        or "local"
    )
    principal = None
    if isinstance(user, Mapping) and user.get("role") and user.get("user_id"):
        principal = Principal(
            role=str(user["role"]),
            workspace_id=str(user.get("workspace_id") or workspace_id),
            user_id=str(user["user_id"]),
            email=str(user.get("email") or ""),
            name=str(user.get("name") or ""),
            via="job",
        )
    elif auth_enabled():
        raise AccessDenied("background job is missing an authenticated initiating principal")

    if membership_checker is None and auth_enabled():
        membership_checker = live_membership()

    return resolve_scope(
        workspace_id=workspace_id,
        principal=principal,
        projects=getattr(env, "projects", None),
        drafts=getattr(env, "drafts", None),
        project_id=payload.get("project_id"),
        problem_id=payload.get("problem_id"),
        draft_id=payload.get("draft_id"),
        session_id=payload.get("session_id"),
        require_write=require_write,
        require_principal=auth_enabled() or principal is not None,
        require_problem=require_problem,
        auto_create_legacy_problem=False,  # a check never writes the project
        membership_checker=membership_checker,
    )


__all__ = [
    "AccessDenied", "AmbiguousProblemError", "AuthorizedScope", "NoProblemError", "READ_ROLES", "WRITE_ROLES",
    "auth_enabled", "create_problem", "ensure_default_problem", "list_problems", "live_membership",
    "person_scope_id", "problem_scope_id", "project_scope_id", "resolve_problem",
    "resolve_scope", "revalidate_job_scope", "session_scope_id", "workspace_of",
]
