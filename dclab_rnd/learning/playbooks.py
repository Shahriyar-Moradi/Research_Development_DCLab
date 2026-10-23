"""Where playbooks live (H2): an append-only log of deltas per family and lineage, the playbook rebuilt from it.

``<workspace>/playbook/<family>/<lineage>.deltas.jsonl``: one :class:`dclab_core.playbook.Delta` a line, only ever
appended (the store's keepers keep it with the rest of the workspace). :meth:`FilePlaybooks.book` applies them in order;
a delta the curation rules refuse is kept in the log and changes nothing (:func:`dclab_core.playbook.build` says why).
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dclab_core.playbook import Bullet, Delta, Playbook, build

_LOCK = threading.Lock()
LINEAGE = "main"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class FilePlaybooks:
    def __init__(self, home: Path):
        self.root = Path(home) / "playbook"

    def _path(self, family: str, lineage: str) -> Path:
        if not family.replace("_", "").isalnum() or not lineage.replace("_", "").replace("-", "").isalnum():
            raise ValueError("a family and a lineage are plain names")
        return self.root / family / f"{lineage}.deltas.jsonl"

    def deltas(self, family: str, lineage: str = LINEAGE) -> list[Delta]:
        path = self._path(family, lineage)
        if not path.is_file():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(Delta.read(line))
                except ValueError:
                    continue  # a torn line never stops the playbook
        return out

    def book(self, family: str, lineage: str = LINEAGE) -> tuple[Playbook, list[tuple[Delta, str]]]:
        return build(family, lineage, self.deltas(family, lineage))

    def append(self, delta: Delta) -> Playbook:
        """Append one delta and return the playbook after it; a delta the rules refuse raises and is not written."""
        with _LOCK:
            book, _ = self.book(delta.family, delta.lineage)
            from dclab_core.playbook import apply

            if delta.op == "add" and delta.bullet is not None:  # the screen at the store's own door, however the bullet was made
                from dclab_core.playbook import Refused

                from .filter import door

                why = door(delta.bullet, delta.family)
                if why:
                    raise Refused(why)
            after = apply(book, delta)  # raises Refused before anything is written
            path = self._path(delta.family, delta.lineage)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(delta.model_dump_json() + "\n")
            return after

    def add(self, family: str, bullet: Bullet, reason: str, lineage: str = LINEAGE) -> Playbook:
        return self.append(Delta(family=family, lineage=lineage, op="add", bullet=bullet, by=bullet.by, reason=reason, at=now()))

    def count(self, family: str, target: str, counts: dict[str, int], by: str, reason: str, lineage: str = LINEAGE) -> Playbook:
        return self.append(Delta(family=family, lineage=lineage, op="count", target=target, counts=counts, by=by, reason=reason, at=now()))

    def status(self, family: str, target: str, status: str, by: str, reason: str, lineage: str = LINEAGE) -> Playbook:
        return self.append(Delta(family=family, lineage=lineage, op="status", target=target, status=status, by=by, reason=reason, at=now()))  # type: ignore[arg-type]

    def lineages(self, family: str) -> list[str]:
        folder = self.root / family
        return sorted(p.name.removesuffix(".deltas.jsonl") for p in folder.glob("*.deltas.jsonl")) if folder.is_dir() else []


def summary(book: Playbook) -> dict[str, Any]:
    by_status: dict[str, int] = {}
    for b in book.bullets:
        by_status[b.status] = by_status.get(b.status, 0) + 1
    return {"family": book.family, "lineage": book.lineage, "version": book.version, "bullets": len(book.bullets), "by_status": by_status}


__all__ = ["LINEAGE", "FilePlaybooks", "summary", "now"]


if __name__ == "__main__":  # pragma: no cover - a look at a playbook
    import sys

    from ..settings import Settings

    b, refused = FilePlaybooks(Settings.load().agent_home).book(sys.argv[1] if len(sys.argv) > 1 else "groups")
    print(json.dumps({**summary(b), "refused": len(refused)}, indent=1))
