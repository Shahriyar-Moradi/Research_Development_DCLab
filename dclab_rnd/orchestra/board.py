"""The board in the studio (28.2): a team's shared state as ``dclab_core.orchestra.Board``, every hand-off and every pause
an experience event (``team.handoff``, ``team.interrupt``) with the board's version before and after, every walk a
``TeamRecord`` kept in the workspace (``orchestra/teams.jsonl``), and a board that would hold a table's value refused.

The capture rule (the owner's): what a node read and left is kept whole; the board itself is small by contract (names,
codes and counts), so the whole board is kept at every hand-off in ``orchestra/boards.jsonl``.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_LOCK = threading.Lock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new(thread: str, state: str = "problem") -> Any:
    from dclab_core.orchestra import Board

    return Board(thread=thread, state=state)


def _conversation(thread: str) -> dict[str, str]:
    """The draft a walk belongs to, as the experience log's conversation (the review of 28.2: deleting a draft removes its
    team's events with the rest of its record)."""
    kind, _, ident = str(thread).partition(":")
    return {"conversation": ident} if kind == "draft" and ident else {"project": ident} if kind == "project" and ident else {}


def forget(home: Path, thread: str) -> int:
    """Remove a walk's boards, pauses and record from the workspace's orchestra files, and its checkpoints (they hold the
    person's answers: the review of 28.3), for a deleted draft or project."""
    removed = _forget_checkpoints(home, thread)
    for name in ("boards.jsonl", "interrupts.jsonl", "teams.jsonl", "walks.jsonl"):
        path = Path(home) / "orchestra" / name
        if not path.is_file():
            continue
        keep = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                doc = json.loads(line)
            except ValueError:
                keep.append(line)
                continue
            if doc.get("thread") == thread or (doc.get("board") or {}).get("thread") == thread:
                removed += 1
            else:
                keep.append(line)
        with _LOCK:
            path.write_text("".join(x + "\n" for x in keep), encoding="utf-8")
    return removed


def _checkpoint_tables(url: str) -> bool:
    from sqlalchemy import inspect

    from ..storage import db

    return inspect(db.engine(url)).has_table("checkpoints")


def _forget_checkpoints(home: Path, thread: str) -> int:
    """Delete the thread from every store that may hold it: the workspace's SQLite file when one exists (a walk run on files,
    or before the workspace moved to PostgreSQL) and PostgreSQL when the workspace runs on it."""
    import os

    from .checkpoints import open_checkpointer, where

    places = []
    if Path(where(home, database_url="")["path"]).is_file():
        places.append("")  # the SQLite file
    if os.environ.get("DCLAB_DATABASE_URL") and _checkpoint_tables(os.environ["DCLAB_DATABASE_URL"]):
        places.append(None)  # the database the environment names, when a walk ever wrote there (a deletion never makes the tables)
    had = 0
    for url in places:
        with open_checkpointer(home, database_url=url) as saver:
            had += saver.get_tuple({"configurable": {"thread_id": thread}}) is not None
            saver.delete_thread(thread)
    return had


def _keep(home: Optional[Path], name: str, doc: dict[str, Any]) -> None:
    if home is None:
        return
    path = Path(home) / "orchestra" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, path.open("a", encoding="utf-8") as out:
        out.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")


def _level(home: Optional[Path], thread: str) -> str:
    """The consent level a record written now carries (23.1): it leaves only at the level it was written under."""
    from .. import consent

    kind, _, ident = str(thread).partition(":")
    try:
        return consent.current_level(home=Path(home), project=ident if kind == "project" else None) if home is not None else "operate"
    except Exception:  # noqa: BLE001 - a level that cannot be read is the one that never leaves
        return "operate"


def advance(board: Any, node: str, what: str, team: str, *, home: Optional[Path] = None, frame: Any = None, walk: Optional[str] = None,
            **changes: Any) -> Any:
    """The next board: ``changes`` applied by ``node`` (refused when it would hold one of ``frame``'s values), its hand-off
    an event, the whole board kept with its walk's id (``walk``) and the consent level it was written under."""
    from dclab_core import orchestra as core

    from .. import experience

    after = core.advance(board, node, what, now(), **changes)
    if frame is not None:
        found = core.values_in(after, distinct_values(frame))
        if found:
            raise ValueError(f"the board would hold a table's value at {', '.join(found)}: a board holds names, codes and counts")
    h = after.handoffs[-1]
    experience.record("team.handoff", "agent", {"thread": after.thread, "team": team, "from_node": h.from_node, "to_node": h.to_node,
                                               "what": h.what, "board_before": h.board_before, "board_after": h.board_after, "state": after.state},
                      **_conversation(after.thread))
    _keep(home, "boards.jsonl", {"at": h.at, "team": team, "node": node, "what": what, "walk": walk, "consent": _level(home, after.thread) if home is not None else None,
                                  "board": json.loads(after.model_dump_json())})
    return after


def interrupt(board: Any, kind: str, node: str, team: str, *, field: Optional[str] = None, gate: Optional[str] = None, text: str = "",
              status: str = "open", home: Optional[Path] = None) -> Any:
    """A pause a person answers, on the board and as an event."""
    from dclab_core.orchestra import Board, Interrupt

    from .. import experience

    item = Interrupt(kind=kind, node=node, field=field, gate=gate, text=text[:600], status=status, at=now())
    raised = status != "open"  # an answered or declined pause changes the board; an open one is the walk waiting (an event only)
    after = Board.model_validate({**board.model_dump(), "version": board.version + (1 if raised else 0),
                                  "interrupts": [*(i.model_dump() for i in board.interrupts), *([item.model_dump()] if raised else [])]})
    experience.record("team.interrupt", "agent", {"thread": board.thread, "team": team, "node": node, "interrupt": kind, "field": field,
                                                 "gate": gate, "status": status, "text": text[:600], "board_before": board.version,
                                                 "board_after": after.version}, **_conversation(board.thread))
    _keep(home, "interrupts.jsonl", {"thread": board.thread, "team": team, **item.model_dump()})
    return after


def distinct_values(frame: Any, per_column: int = 200) -> set[str]:
    """A table's values as text (at most ``per_column`` distinct per column): what a board must never hold."""
    out: set[str] = set()
    for column in getattr(frame, "columns", []):
        for v in frame[column].dropna().astype(str).unique()[:per_column]:
            out.add(v)
    return out - {str(c) for c in getattr(frame, "columns", [])}


def record(thread: str, team: str, nodes: list[dict[str, Any]], board: Any, status: str, started: str, home: Optional[Path] = None,
           spent_eur: Optional[float] = None, walk: Optional[str] = None, models: Optional[list[str]] = None) -> Any:
    """The walk's record, node by node, kept in the workspace and as a ``team.recorded`` event (and returned). ``spent_eur``
    what its model requests cost (the board's own spend when not given); ``walk`` its id and ``models`` the models its
    requests were answered by, kept beside it (``walks.jsonl``): what pairs a walk's boards with its record, and what says
    whose answers its hand-offs are (28.9's review)."""
    from dclab_core.orchestra import TeamRecord

    from .. import experience

    spent = round(float(board.spent_eur if spent_eur is None else spent_eur), 6)
    rec = TeamRecord(thread=thread, team=team, nodes=nodes, handoffs=len(board.handoffs), interrupts=len(board.interrupts),
                     spent_eur=spent, board_version=board.version, status=status, started=started, finished=now())
    _keep(home, "teams.jsonl", json.loads(rec.model_dump_json()))
    _keep(home, "walks.jsonl", {"thread": thread, "walk": walk, "team": team, "status": status, "spent_eur": spent, "models": None if models is None else sorted(set(models)),
                                "started": started, "finished": rec.finished, "consent": _level(home, thread) if home is not None else None})
    experience.record("team.recorded", "agent", {"thread": thread, "team": team, "status": status, "nodes": len(rec.nodes), "handoffs": rec.handoffs,
                                                "interrupts": rec.interrupts, "board_version": rec.board_version, "cost": spent,
                                                "seconds": round(sum(n.seconds for n in rec.nodes), 3), "started": started, "finished": rec.finished},
                      **_conversation(thread))
    return rec


def merge_parts(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The reducer for parts solved in parallel: by part id, the later state of a part wins, the order of first sight kept."""
    out: dict[str, dict[str, Any]] = {}
    for p in [*(left or []), *(right or [])]:
        out[str(p.get("part"))] = {**out.get(str(p.get("part")), {}), **p}
    return list(out.values())


__all__ = ["new", "advance", "interrupt", "record", "merge_parts", "distinct_values", "forget"]
