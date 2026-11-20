"""What the rest of the code may ask of storage.

Projects, drafts, intern sessions, problem memory's bases and summaries, and the work passed between team agents are files in a workspace folder, or PostgreSQL rows. These Protocols are the whole surface
callers use, so a PostgreSQL implementation can replace the file stores without touching a route, an agent or the
engine (docs/guides/PRODUCT_BUILD_PLAYBOOK.md, packages 9.1 and 9.2).

Rules for callers:
- Reach storage only through these methods. No caller builds a path inside a store's folder or reads its files.
- ``data_dir`` and ``export_dir`` return a local folder for tables and exports. They are the one file-shaped part
  of the interface and move behind file storage in package 9.3.
- Logs are append-only: ``log``/``activity``, ``transition``/``transitions``, ``emit``/``events`` never rewrite.

``tests/test_storage_interface.py`` fails when a file store stops satisfying its Protocol or when code outside the
stores reaches into a store's folder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, ContextManager, Protocol, runtime_checkable

Document = dict[str, Any]


@runtime_checkable
class Projects(Protocol):
    def turn(self, project_id: str, purpose: str = "project", *, require_process: bool = False) -> ContextManager[Any]: ...
    def create(self, name: str, industry: str = "general", goal: str = "") -> Document: ...
    def get(self, project_id: str) -> Document: ...                      # KeyError when unknown
    def save(self, project: Document) -> Document: ...
    def update(self, project_id: str, **values: Any) -> Document: ...
    def list(self, limit: int | None = None, offset: int = 0) -> list[Document]: ...   # newest first
    def delete(self, project_id: str) -> None: ...
    # stage records: written once per run, cleared together from a stage onward
    def write_stage(self, project_id: str, stage: str, record: Document) -> Any: ...
    def read_stage(self, project_id: str, stage: str) -> Document | None: ...
    def has_stage(self, project_id: str, stage: str) -> bool: ...
    def records(self, project_id: str) -> dict[str, Document]: ...
    def clear_stages(self, project_id: str, from_stage: str = "data") -> None: ...
    # append-only logs
    def log(self, project_id: str, kind: str, payload: Any, at: str | None = None) -> None: ...   # ``at`` is for importing history
    def activity(self, project_id: str, limit: int = 60) -> list[Document]: ...
    def transition(self, project_id: str, entry: Document) -> None: ...
    def transitions(self, project_id: str, limit: int = 200) -> list[Document]: ...
    # files (package 9.3 replaces these with file storage)
    def data_dir(self, project_id: str) -> Path: ...
    def export_dir(self, project_id: str) -> Path: ...
    def location(self) -> str: ...                                       # where this workspace lives, for the Admin page


@runtime_checkable
class Drafts(Protocol):
    def create(self, problem: str = "", pack: str | None = None) -> Document: ...
    def get(self, draft_id: str) -> Document: ...                        # KeyError when unknown
    def save(self, draft: Document) -> Document: ...
    def update(self, draft_id: str, fn: Callable[[Document], Any]) -> Document: ...   # read, change, save as one step
    def list(self, limit: int = 50) -> list[Document]: ...
    def delete(self, draft_id: str) -> None: ...
    # the event log the page streams: sequence numbers start at 1 and never repeat
    def emit(self, draft_id: str, kind: str, data: Document | None = None) -> Document: ...
    def events(self, draft_id: str, after: int = 0) -> list[Document]: ...
    def events_version(self, draft_id: str) -> Any: ...                  # changes whenever an event is added (a cache key)
    def turn(self, draft_id: str) -> ContextManager[Any]: ...            # one agent turn at a time per draft; re-entrant
    def data_dir(self, draft_id: str) -> Path: ...


@runtime_checkable
class Sessions(Protocol):
    def create(self, task: str, *, mode: str, model: str | None, budget: Document | None = None, project_id: str | None = None) -> Document: ...
    def get(self, session_id: str) -> Document: ...                      # KeyError when unknown
    def save(self, session: Document) -> Document: ...
    def list(self, limit: int | None = None, offset: int = 0) -> list[Document]: ...   # newest first, without messages and steps
    def delete(self, session_id: str) -> None: ...


@runtime_checkable
class Teams(Protocol):
    """Teams and who is in them (W2, storage/teams.py): a team says whom an agent represents, never what a person may do
    (that is the role's). Every read sees this workspace's teams only."""
    def create_team(self, name: str, kind: str) -> Document: ...          # TeamError: a bad name or kind, a name taken
    def teams(self) -> list[Document]: ...                                # each with its members
    def team(self, team_id: str) -> Document | None: ...
    def add_member(self, team_id: str, user_id: str) -> tuple[Document, bool]: ...   # NoSuchTeam; (membership, whether new)
    def remove_member(self, team_id: str, user_id: str) -> bool: ...
    def remove_user(self, user_id: str) -> int: ...                       # the person left the workspace: every team
    def teams_of(self, user_id: str) -> list[Document]: ...
    def is_member(self, team_id: str, user_id: str) -> bool: ...
    def counts(self) -> dict[str, int]: ...
    def location(self) -> str: ...


@runtime_checkable
class MemoryBases(Protocol):
    """Problem memory's editable bases and their derived summaries (dclab_rnd/memory/bases.py and summary.py)."""
    # bases: one live document per (scope, scope_id); a change is read, step, write as one step across hosts
    def read(self, scope: str, scope_id: str) -> Document | None: ...
    def change(self, scope: str, scope_id: str, blank: Document, fn: Callable[[Document], tuple[bool, Any]]) -> Any: ...
    def purge(self, belongs: Callable[[Document], bool], reason: str) -> list[Document]: ...   # a tombstone, never a deletion
    def all_bases(self) -> list[tuple[str, Document]]: ...                                     # live and purged, for a move
    # summaries: a publish never replaces a newer cursor; derived, so invalidating deletes them
    def summary(self, key: str) -> Document | None: ...
    def publish(self, key: str, doc: Document, before: Callable[[], Document | None] | None = None) -> Document: ...
    def invalidate(self, project_id: str, problem_id: str | None = None) -> int: ...
    def summaries(self) -> list[tuple[str, Document]]: ...
    def counts(self) -> dict[str, int]: ...
    def location(self) -> str: ...


@runtime_checkable
class TeamWork(Protocol):
    """The work passed between team agents (W4, dclab_rnd/teamspace/store.py): tasks, actions, reports and the delivery
    outbox. Every change goes through dclab_core.team_work's rules; a task and its delivery are saved together."""
    def create_task(self, task: Any, *, initiator: Any = None) -> Any: ...
    def delegate(self, project_id: str, parent_id: str, **child: Any) -> Any: ...                 # team_work.delegate, siblings from the store
    def move(self, project_id: str, task_id: str, new_status: str, by: Any, **kw: Any) -> Any: ...   # validate_transition, children from the store
    def record_action(self, project_id: str, action: Any) -> Any: ...
    def finish_action(self, project_id: str, task_id: str, action_id: str, status: str, **kw: Any) -> Any: ...  # one terminal record
    def store_report(self, project_id: str, task_id: str, **kw: Any) -> Any: ...                  # build_report + accept_reply; one per (task, revision)
    def supersede_report(self, project_id: str, task_id: str, **kw: Any) -> Any: ...
    def task(self, project_id: str, task_id: str) -> Any: ...
    def task_view(self, project_id: str, task_id: str) -> Document | None: ...
    def open_tasks(self, team: str, project_id: str | None = None) -> list[Any]: ...
    def handoff_history(self, project_id: str, problem_id: str) -> list[Document]: ...
    # the outbox (dclab_rnd/teamspace/delivery.py): a task's rows are delivered in order
    def delivery(self, delivery_id: str, project_id: str | None = None) -> Document | None: ...
    def deliveries(self, status: str | None = None, project_id: str | None = None, task_id: str | None = None) -> list[Document]: ...
    def pending_heads(self) -> list[Document]: ...                                                 # each task's oldest pending row
    def delivery_done(self, delivery_id: str, receipt: Any = None, project_id: str | None = None) -> tuple[Document, bool]: ...
    def delivery_failed(self, delivery_id: str, error: Any, *, max_attempts: int, backoff_seconds: float, cap_seconds: float = 60.0,
                        project_id: str | None = None) -> Document: ...
    def delivery_stop(self, delivery_id: str, status: str, reason: str, *, error: Any = None, project_id: str | None = None) -> Document: ...
    def counts(self) -> dict[str, int]: ...
    def location(self) -> str: ...


def open_team_work(home: Path, projects: Projects | None = None) -> TeamWork:
    """The team work of the workspace in ``home``: PostgreSQL with DCLAB_DATABASE_URL set, else its folder's files.
    ``projects`` is the workspace's projects store, which a new task's problem is checked against (opened when needed)."""
    from ..teamspace.store import open_team_work as choose

    return choose(home, projects=projects)


def open_memory_bases(home: Path) -> MemoryBases:
    """The bases and summaries of the workspace in ``home``: PostgreSQL with DCLAB_DATABASE_URL set, else its files."""
    from .memory_bases import open_memory_bases as choose

    return choose(home)


def open_teams(home: Path) -> Teams:
    """The teams of the workspace in ``home``: PostgreSQL with DCLAB_DATABASE_URL set, else its folder's file."""
    from .teams import open_teams as choose

    return choose(home)


def open_stores(home: Path, projects_home: Path | None = None) -> tuple[Projects, Drafts, Sessions]:
    """The projects, drafts and intern sessions for the workspace in ``home``.

    With DCLAB_DATABASE_URL set they are PostgreSQL stores (the home folder keeps the files and names the workspace);
    without it they are the file stores. The three environment variables below move one store's folder, as before.
    """
    import os

    from . import db

    if db.database_url(required=False):
        from .postgres import open_stores as postgres

        return postgres(home)
    from ..draft.store import DraftStore
    from ..intern.sessions import SessionStore
    from ..studio.store import ProjectStore

    projects = ProjectStore(Path(projects_home or os.environ.get("DCLAB_STUDIO_HOME") or (home / "projects")))
    drafts = DraftStore(Path(os.environ.get("DCLAB_DRAFT_HOME") or (projects.home.parent / "drafts")))
    projects.audit_path = drafts.audit_path = Path(home) / "audit.jsonl"  # one audit per workspace, wherever its folders are (10.4)
    return projects, drafts, SessionStore(Path(os.environ.get("DCLAB_INTERN_HOME") or (projects.home.parent / "intern")))


__all__ = ["Document", "MemoryBases", "Projects", "Drafts", "Sessions", "TeamWork", "Teams", "open_memory_bases", "open_stores", "open_team_work", "open_teams"]
