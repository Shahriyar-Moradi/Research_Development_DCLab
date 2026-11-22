"""Teams and who is in them (W2 of the collaborative workspace): a JSON file in the workspace folder, or PostgreSQL rows.

A team is not a role. A role (``accounts/roles.py``) says what a person may do; a team says whom an agent represents:
the Business team talks to its own agent, and that agent hands tasks to the Data Science team's (``dclab_rnd/teamspace``).
A person may be in several teams, or in none. ``kind`` is configuration: ``KINDS`` are the five the plan names, and any
other short lowercase word is allowed too (a workspace may have a Finance team).

- Files (``FileTeams``): ``teams/teams.json`` behind its folder's lock (one host). The folder is the workspace, so
  another folder sees nothing.
- PostgreSQL (``PgTeams``, migration 0027): ``team`` rows of one workspace and ``team_membership`` rows of its teams;
  every read joins through ``team.workspace_id``, so another workspace sees nothing. A member is an account
  (``app_user``): an unknown id is refused, and leaving the workspace removes the person from its teams
  (``accounts/store.py``).

Who may change teams is the routes' (``agentic/routers/teams.py``: the owner's); this module keeps the rows only.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

Document = dict[str, Any]

KINDS = ("business", "data_science", "ml_engineering", "sales", "marketing")
LABELS = {"business": "Business", "data_science": "Data Science", "ml_engineering": "ML Engineering", "sales": "Sales", "marketing": "Marketing"}
KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,39}$")
NAME_MAX = 80
TEAMS_FILE = Path("teams") / "teams.json"


class TeamError(ValueError):
    """A team or a membership the store refuses: a bad name or kind, a name taken, an unknown person."""


class BrokenTeamsFile(TeamError):
    """The workspace's teams file is not one (hand-edited or cut short): read and written by nobody until it is mended."""


class NoSuchTeam(KeyError):
    """No team with this id in this workspace."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _new_id() -> str:
    return "team_" + secrets.token_hex(8)


def normal_name(name: Any) -> str:
    text = " ".join(str(name or "").split())
    if not text:
        raise TeamError("a team needs a name")
    if len(text) > NAME_MAX:
        raise TeamError(f"a team's name is at most {NAME_MAX} characters")
    return text


def normal_kind(kind: Any) -> str:
    text = str(kind or "").strip().lower().replace("-", "_").replace(" ", "_")
    if not KIND_PATTERN.match(text):
        raise TeamError(f"a team's kind is one of {', '.join(KINDS)}, or another lowercase word (letters, digits, _)")
    return text


def label(kind: str) -> str:
    return LABELS.get(kind) or kind.replace("_", " ").title()


def _user(user_id: Any) -> str:
    text = str(user_id or "").strip()
    if not text or len(text) > 64:
        raise TeamError("a member is a person's id")
    return text


def _shown(team: Document, members: list[Document]) -> Document:
    return {**team, "kind_label": label(team["kind"]), "members": sorted(members, key=lambda m: (m["since"], m["user_id"]))}


# ---------------------------------------------------------------------------------------------------------- files
class FileTeams:
    """One JSON document of the folder's teams and memberships, behind the folder's lock."""

    def __init__(self, home: Path):
        self.home = Path(home)
        self.path = self.home / TEAMS_FILE

    def location(self) -> str:
        return f"files in {self.path.parent}"

    def _load(self) -> Document:
        try:
            doc = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return {"teams": [], "members": []}
        except (OSError, ValueError):
            doc = None
        if not isinstance(doc, dict) or not isinstance(doc.get("teams"), list) or not isinstance(doc.get("members"), list):
            # never overwritten while unreadable; the message names no path (it reaches an HTTP answer)
            raise BrokenTeamsFile("the workspace's teams file cannot be read: nothing was changed")
        return doc

    def _save(self, doc: Document) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(f".{self.path.name}.{secrets.token_hex(4)}.part")
        tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, self.path)

    def _held(self):
        from ..filelock import held

        self.path.parent.mkdir(parents=True, exist_ok=True)
        return held(self.path.parent)

    def create_team(self, name: str, kind: str) -> Document:
        name, kind = normal_name(name), normal_kind(kind)
        with self._held():
            doc = self._load()
            if any(t["name"].casefold() == name.casefold() for t in doc["teams"]):
                raise TeamError(f"this workspace already has a team named {name!r}")
            team = {"id": _new_id(), "name": name, "kind": kind, "created": _now()}
            doc["teams"].append(team)
            self._save(doc)
        return _shown(team, [])

    def teams(self) -> list[Document]:
        doc = self._load()
        return [_shown(t, [{"user_id": m["user_id"], "since": m["since"]} for m in doc["members"] if m["team_id"] == t["id"]])
                for t in sorted(doc["teams"], key=lambda t: (t["created"], t["name"]))]

    def team(self, team_id: str) -> Optional[Document]:
        return next((t for t in self.teams() if t["id"] == team_id), None)

    def add_member(self, team_id: str, user_id: str) -> tuple[Document, bool]:
        """The membership, and whether it is new (adding a member twice changes nothing)."""
        user_id = _user(user_id)
        with self._held():
            doc = self._load()
            if not any(t["id"] == team_id for t in doc["teams"]):
                raise NoSuchTeam(team_id)
            found = next((m for m in doc["members"] if m["team_id"] == team_id and m["user_id"] == user_id), None)
            if found is not None:
                return dict(found), False
            row = {"team_id": team_id, "user_id": user_id, "since": _now()}
            doc["members"].append(row)
            self._save(doc)
        return dict(row), True

    def remove_member(self, team_id: str, user_id: str) -> bool:
        with self._held():
            doc = self._load()
            if not any(t["id"] == team_id for t in doc["teams"]):
                raise NoSuchTeam(team_id)
            kept = [m for m in doc["members"] if not (m["team_id"] == team_id and m["user_id"] == user_id)]
            if len(kept) == len(doc["members"]):
                return False
            doc["members"] = kept
            self._save(doc)
        return True

    def rename_team(self, team_id: str, name: str) -> Document:
        """W7: a new name (its kind and members unchanged); refused when another team of the workspace has it."""
        name = normal_name(name)
        with self._held():
            doc = self._load()
            team = next((t for t in doc["teams"] if t["id"] == team_id), None)
            if team is None:
                raise NoSuchTeam(team_id)
            if any(t["id"] != team_id and t["name"].casefold() == name.casefold() for t in doc["teams"]):
                raise TeamError(f"this workspace already has a team named {name!r}")
            team["name"] = name
            self._save(doc)
        return self.team(team_id)

    def delete_team(self, team_id: str) -> int:
        """W7: the team removed with its memberships; how many members it had. Its threads and its memory stay in the
        workspace's records (the store keeps what was written); the team memory routes answer 404 for it from now on."""
        with self._held():
            doc = self._load()
            if not any(t["id"] == team_id for t in doc["teams"]):
                raise NoSuchTeam(team_id)
            members = sum(1 for m in doc["members"] if m["team_id"] == team_id)
            doc["teams"] = [t for t in doc["teams"] if t["id"] != team_id]
            doc["members"] = [m for m in doc["members"] if m["team_id"] != team_id]
            self._save(doc)
        return members

    def remove_user(self, user_id: str) -> int:
        """The person leaves every team of this workspace (they left the workspace); how many."""
        with self._held():
            doc = self._load()
            kept = [m for m in doc["members"] if m["user_id"] != user_id]
            gone = len(doc["members"]) - len(kept)
            if gone:
                doc["members"] = kept
                self._save(doc)
        return gone

    def teams_of(self, user_id: str) -> list[Document]:
        return [t for t in self.teams() if any(m["user_id"] == user_id for m in t["members"])]

    def is_member(self, team_id: str, user_id: str) -> bool:
        doc = self._load()
        return any(m["team_id"] == team_id and m["user_id"] == user_id for m in doc["members"])

    def counts(self) -> dict[str, int]:
        doc = self._load()
        return {"teams": len(doc["teams"]), "memberships": len(doc["members"])}


# ---------------------------------------------------------------------------------------------------------- PostgreSQL
class PgTeams:
    """The ``team`` rows of one workspace and the ``team_membership`` rows of its teams."""

    def __init__(self, workspace_id: str, url: str | None = None):
        from . import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def location(self) -> str:
        return f"PostgreSQL workspace {self.workspace_id}"

    @staticmethod
    def _iso(value: Any) -> str:
        return value.astimezone(timezone.utc).isoformat(timespec="seconds") if hasattr(value, "astimezone") else str(value)

    def _lock(self, c: Any) -> None:
        """One team change of this workspace at a time, so a name differing only in case is refused, never doubled."""
        import hashlib

        import sqlalchemy as sa

        digest = hashlib.sha256(f"team:{self.workspace_id}".encode("utf-8")).digest()
        c.execute(sa.text("select pg_advisory_xact_lock(:key)"), {"key": int.from_bytes(digest[:8], "big", signed=True)})

    def _owns(self, c: Any, team_id: str) -> bool:
        import sqlalchemy as sa

        from .models import team as t

        return c.execute(sa.select(t.c.id).where(t.c.id == team_id, t.c.workspace_id == self.workspace_id)).first() is not None

    def create_team(self, name: str, kind: str) -> Document:
        import sqlalchemy as sa

        from .models import team as t

        name, kind = normal_name(name), normal_kind(kind)
        with self.engine.begin() as c:
            self._lock(c)
            names = c.execute(sa.select(t.c.name).where(t.c.workspace_id == self.workspace_id)).scalars()
            if any(n.casefold() == name.casefold() for n in names):  # as the files compare (a workspace has few teams)
                raise TeamError(f"this workspace already has a team named {name!r}")
            row = c.execute(sa.insert(t).values(id=_new_id(), workspace_id=self.workspace_id, name=name, kind=kind)
                            .returning(t.c.id, t.c.name, t.c.kind, t.c.created)).one()
        return _shown({"id": row.id, "name": row.name, "kind": row.kind, "created": self._iso(row.created)}, [])

    def _read(self, c: Any, *, team_id: str | None = None, user_id: str | None = None) -> list[Document]:
        import sqlalchemy as sa

        from .models import team as t, team_membership as m

        query = sa.select(t.c.id, t.c.name, t.c.kind, t.c.created).where(t.c.workspace_id == self.workspace_id)
        if team_id is not None:
            query = query.where(t.c.id == team_id)
        if user_id is not None:
            query = query.where(t.c.id.in_(sa.select(m.c.team_id).where(m.c.user_id == user_id)))
        teams = c.execute(query.order_by(t.c.created, t.c.name)).all()
        ids = [r.id for r in teams]
        members: dict[str, list[Document]] = {i: [] for i in ids}
        if ids:
            for r in c.execute(sa.select(m.c.team_id, m.c.user_id, m.c.since).where(m.c.team_id.in_(ids))):
                members[r.team_id].append({"user_id": r.user_id, "since": self._iso(r.since)})
        return [_shown({"id": r.id, "name": r.name, "kind": r.kind, "created": self._iso(r.created)}, members[r.id]) for r in teams]

    def teams(self) -> list[Document]:
        with self.engine.connect() as c:
            return self._read(c)

    def team(self, team_id: str) -> Optional[Document]:
        with self.engine.connect() as c:
            found = self._read(c, team_id=team_id)
        return found[0] if found else None

    def add_member(self, team_id: str, user_id: str) -> tuple[Document, bool]:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from .models import app_user as u, membership as w, team_membership as m

        user_id = _user(user_id)
        with self.engine.begin() as c:
            if not self._owns(c, team_id):
                raise NoSuchTeam(team_id)
            if c.execute(sa.select(u.c.id).where(u.c.id == user_id)).first() is None:
                raise TeamError("no account has this id")
            # a member of this workspace, held until the commit: a removal from the workspace racing this insert waits,
            # and its leave_teams then sees the new row (accounts/store.py), so no team keeps someone who left
            if c.execute(sa.select(w.c.role).where(w.c.workspace_id == self.workspace_id, w.c.user_id == user_id).with_for_update(key_share=True)).first() is None:
                raise TeamError("this person is not a member of this workspace")
            made = c.execute(insert(m).values(team_id=team_id, user_id=user_id).on_conflict_do_nothing(index_elements=[m.c.team_id, m.c.user_id])
                             .returning(m.c.since)).first()
            since = made.since if made is not None else c.execute(sa.select(m.c.since).where(m.c.team_id == team_id, m.c.user_id == user_id)).scalar()
        return {"team_id": team_id, "user_id": user_id, "since": self._iso(since)}, made is not None

    def import_team(self, team: Document) -> bool:
        """A file workspace's team, moved by ``storage migrate`` with its id and time; False when its id or its name is
        here already (a second run copies nothing)."""
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from .models import team as t

        name, kind = normal_name(team.get("name")), normal_kind(team.get("kind"))
        with self.engine.begin() as c:
            self._lock(c)
            names = c.execute(sa.select(t.c.name).where(t.c.workspace_id == self.workspace_id)).scalars()
            if any(n.casefold() == name.casefold() for n in names):
                return False
            values = {"id": str(team["id"]), "workspace_id": self.workspace_id, "name": name, "kind": kind}
            if team.get("created"):
                values["created"] = datetime.fromisoformat(str(team["created"]))
            return c.execute(insert(t).values(**values).on_conflict_do_nothing(index_elements=[t.c.id]).returning(t.c.id)).first() is not None

    def remove_member(self, team_id: str, user_id: str) -> bool:
        import sqlalchemy as sa

        from .models import team_membership as m

        with self.engine.begin() as c:
            if not self._owns(c, team_id):
                raise NoSuchTeam(team_id)
            return bool(c.execute(sa.delete(m).where(m.c.team_id == team_id, m.c.user_id == user_id)).rowcount)

    def rename_team(self, team_id: str, name: str) -> Document:
        import sqlalchemy as sa

        from .models import team as t

        name = normal_name(name)
        with self.engine.begin() as c:
            self._lock(c)
            if not self._owns(c, team_id):
                raise NoSuchTeam(team_id)
            names = c.execute(sa.select(t.c.name).where(t.c.workspace_id == self.workspace_id, t.c.id != team_id)).scalars()
            if any(n.casefold() == name.casefold() for n in names):
                raise TeamError(f"this workspace already has a team named {name!r}")
            c.execute(sa.update(t).where(t.c.id == team_id, t.c.workspace_id == self.workspace_id).values(name=name))
        return self.team(team_id)

    def delete_team(self, team_id: str) -> int:
        import sqlalchemy as sa

        from .models import team as t, team_membership as m

        with self.engine.begin() as c:
            self._lock(c)
            if not self._owns(c, team_id):
                raise NoSuchTeam(team_id)
            members = c.execute(sa.delete(m).where(m.c.team_id == team_id)).rowcount
            c.execute(sa.delete(t).where(t.c.id == team_id, t.c.workspace_id == self.workspace_id))
        return int(members or 0)

    def remove_user(self, user_id: str) -> int:
        with self.engine.begin() as c:
            return leave_teams(c, self.workspace_id, user_id)

    def teams_of(self, user_id: str) -> list[Document]:
        with self.engine.connect() as c:
            return self._read(c, user_id=user_id)

    def is_member(self, team_id: str, user_id: str) -> bool:
        import sqlalchemy as sa

        from .models import team as t, team_membership as m

        with self.engine.connect() as c:
            return c.execute(sa.select(m.c.team_id).join(t, t.c.id == m.c.team_id)
                             .where(t.c.workspace_id == self.workspace_id, m.c.team_id == team_id, m.c.user_id == user_id)).first() is not None

    def counts(self) -> dict[str, int]:
        import sqlalchemy as sa

        from .models import team as t, team_membership as m

        with self.engine.connect() as c:
            teams = int(c.execute(sa.select(sa.func.count()).select_from(t).where(t.c.workspace_id == self.workspace_id)).scalar())
            members = int(c.execute(sa.select(sa.func.count()).select_from(m.join(t, t.c.id == m.c.team_id))
                                    .where(t.c.workspace_id == self.workspace_id)).scalar())
        return {"teams": teams, "memberships": members}


def leave_teams(c: Any, workspace_id: str, user_id: str) -> int:
    """In the caller's transaction: ``user_id`` leaves every team of ``workspace_id`` (they left that workspace)."""
    import sqlalchemy as sa

    from .models import team as t, team_membership as m

    return int(c.execute(sa.delete(m).where(m.c.user_id == user_id,
                                            m.c.team_id.in_(sa.select(t.c.id).where(t.c.workspace_id == workspace_id)))).rowcount or 0)


def open_teams(home: Path, url: str | None = None) -> "FileTeams | PgTeams":
    """The teams of the workspace in ``home``: PostgreSQL when DCLAB_DATABASE_URL is set, as the other stores choose
    (``storage.open_stores``), else the folder's file."""
    from . import db

    if url or db.database_url(required=False):
        return PgTeams(db.workspace_for(Path(home), url), url)
    return FileTeams(Path(home))


__all__ = ["KINDS", "LABELS", "BrokenTeamsFile", "FileTeams", "NoSuchTeam", "PgTeams", "TeamError", "label", "leave_teams", "normal_kind", "normal_name", "open_teams"]
