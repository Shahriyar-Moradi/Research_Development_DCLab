"""One ``Services`` per workspace a server serves (package 10.2).

The server's own folder (``DCLAB_AGENT_HOME``) is its first workspace, as before. Another workspace lives in
``<home>/workspaces/<id>/`` with its own files; its rows in PostgreSQL carry its id, so every query of its stores is
filtered to it. A workspace's Services (stores, gateway, lessons, jobs, audit) is opened the first time a request
or a job needs it and kept for the life of the process.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from ..settings import Settings
from .services import Services


class Pool:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.default = Services(settings)
        self._by_id: dict[str | None, Services] = {self.default.workspace_id: self.default}
        self._guard = threading.Lock()

    def folder(self, workspace_id: str) -> Path:
        return Path(self.settings.agent_home) / "workspaces" / workspace_id

    def get(self, workspace_id: str | None) -> Services:
        if workspace_id is None or workspace_id == self.default.workspace_id:
            return self.default
        with self._guard:
            found = self._by_id.get(workspace_id)
            if found is None:
                found = self._by_id[workspace_id] = self._open(workspace_id)
            return found

    def _open(self, workspace_id: str) -> Services:
        from ..storage import db
        from ..accounts.store import Accounts

        key = Accounts().workspace(workspace_id)["key"]  # KeyError for a workspace that does not exist
        folder = self.folder(workspace_id)
        folder.mkdir(parents=True, exist_ok=True)
        marker = folder / db.MARKER
        if not marker.exists():
            marker.write_text(key + "\n", encoding="utf-8")  # this folder is that workspace, wherever it moves
        services = Services(self.settings.model_copy(update={"agent_home": folder}), primary=False)
        if services.workspace_id != workspace_id:
            raise RuntimeError(f"the folder of workspace {workspace_id} names {services.workspace_id}")
        services.csrf = self.default.csrf
        services.worker.start()  # recovers what a dead worker left in this workspace, then runs its queue
        return services

    def workspace_ids(self) -> list[str]:
        """Every workspace this server serves: its own, and each made for accounts (``dclab-workspace:`` keys)."""
        import sqlalchemy as sa

        from ..storage import db
        from ..storage.models import workspace as t

        with db.engine().connect() as c:
            ids = [r[0] for r in c.execute(sa.select(t.c.id).where(t.c.key.like("dclab-workspace:%")).order_by(t.c.id))]
        return [self.default.workspace_id, *[i for i in ids if i != self.default.workspace_id]]

    def start_all(self) -> None:
        """A worker for every workspace (accounts on): a job queued by any instance, in any workspace, is run and recovered.
        A database briefly away, or one workspace that cannot open, is reported; the others start, and the next call tries again."""
        import sys

        self.default.worker.start()
        if self.settings.auth == "none":
            return
        try:
            ids = self.workspace_ids()[1:]
        except Exception as exc:  # noqa: BLE001
            print(f"workspaces not listed ({type(exc).__name__}); trying again later", file=sys.stderr, flush=True)
            return
        for workspace_id in ids:
            try:
                self.get(workspace_id)
            except Exception as exc:  # noqa: BLE001
                print(f"workspace {workspace_id} not opened ({type(exc).__name__}); trying again later", file=sys.stderr, flush=True)

    def all(self) -> list[Services]:
        with self._guard:
            return list(self._by_id.values())

    def __getattr__(self, name: str) -> Any:  # code that held one Services keeps working: it reaches the default
        return getattr(self.default, name)
