"""What the rest of the code may ask of storage.

Projects, drafts and intern sessions are files in a workspace folder today. These Protocols are the whole surface
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


__all__ = ["Document", "Projects", "Drafts", "Sessions", "open_stores"]
