"""Drafts on disk: one folder per draft, a JSON document and an append-only event log.

    <home>/<draft_id>/draft.json      the draft (problem, pack, messages, assets, analysis, workflow…)
    <home>/<draft_id>/events.jsonl    every event the Home page streams (pipeline steps, chat, workflow)
    <home>/<draft_id>/data/           raw files as received, and the cleaned table (clean.parquet)

The event log is the source of the live stream: the server tails it from a sequence number, so a page
that reconnects (or opens later) replays what happened. Writes are atomic (temp file, then replace).
"""

from __future__ import annotations

import json
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ASSET_KINDS = ("upload", "sample", "kaggle", "hf", "database", "cloud", "synthetic")
ASSET_STATES = ("queued", "structuring", "cleaning", "analysing", "ready", "failed")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_draft(draft_id: str, problem: str = "", pack: str | None = None) -> dict[str, Any]:
    """A fresh draft document. Shared by every store."""
    return {
        "id": draft_id, "created": now(), "updated": now(), "status": "open", "project_id": None,
        "problem": problem.strip(),
        "pack": {"key": pack, "source": "user", "why": "Chosen on Home"} if pack else None,
        "messages": [], "questions": [], "assets": [], "active_asset": None,
        "cleaning_log": [], "structure": None, "analysis": None, "workflow": None,
        "understanding": {}, "agent": {"mode": None, "turns": 0},
    }


class DraftStore:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._turns: dict[str, threading.RLock] = {}
        self._guard = threading.Lock()

    # ------------------------------------------------------------------ documents
    def directory(self, draft_id: str) -> Path:
        if not draft_id or not all(c in "0123456789abcdef" for c in draft_id):
            raise KeyError(draft_id)
        return self.home / draft_id

    def data_dir(self, draft_id: str) -> Path:
        path = self.directory(draft_id) / "data"
        path.mkdir(exist_ok=True)  # not parents: a deleted draft's folder is never made again by a late writer
        return path

    def lock(self, draft_id: str) -> threading.Lock:
        return self._locks.setdefault(draft_id, threading.Lock())

    def turn(self, draft_id: str) -> threading.RLock:
        """Held for one agent turn on this draft (``with store.turn(id):``). Re-entrant: a reply that starts a
        simulation runs the pipeline and its data-ready turn in the same thread."""
        with self._guard:
            return self._turns.setdefault(draft_id, threading.RLock())

    def events_version(self, draft_id: str) -> tuple[int, int]:
        """Changes whenever an event is added: a cache key for readers of the log."""
        path = self.directory(draft_id) / "events.jsonl"
        if not path.is_file():
            return (0, 0)
        stat = path.stat()
        return (stat.st_mtime_ns, stat.st_size)

    def create(self, problem: str = "", pack: str | None = None) -> dict[str, Any]:
        draft_id = secrets.token_hex(6)
        self.directory(draft_id).mkdir(parents=True)
        return self.save(new_draft(draft_id, problem, pack))

    def get(self, draft_id: str) -> dict[str, Any]:
        path = self.directory(draft_id) / "draft.json"
        if not path.exists():
            raise KeyError(draft_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, draft: dict[str, Any]) -> dict[str, Any]:
        draft["updated"] = now()
        path = self.directory(draft["id"]) / "draft.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(draft, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        temp.replace(path)
        return draft

    def update(self, draft_id: str, fn) -> dict[str, Any]:
        """Read, change with fn(draft) and save under the draft's lock (the pipeline and the chat both write)."""
        with self.lock(draft_id):
            draft = self.get(draft_id)
            fn(draft)
            return self.save(draft)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        drafts = []
        for path in self.home.glob("*/draft.json"):
            try:
                drafts.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
        return sorted(drafts, key=lambda d: d.get("updated", ""), reverse=True)[:limit]

    def delete(self, draft_id: str) -> None:
        import shutil

        # under the draft's locks: an update already reading it (its agent starting) cannot write it back after this
        with self.lock(draft_id), self.lock(draft_id + ":events"):
            shutil.rmtree(self.directory(draft_id), ignore_errors=True)

    # ------------------------------------------------------------------ events
    def emit(self, draft_id: str, kind: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Append one event; returns it with its sequence number."""
        path = self.directory(draft_id) / "events.jsonl"
        with self.lock(draft_id + ":events"):
            seq = 0
            if path.exists():
                with path.open("rb") as handle:
                    seq = sum(1 for _ in handle)
            event = {"seq": seq + 1, "at": now(), "kind": kind, "data": data or {}}
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        return event

    def events(self, draft_id: str, after: int = 0) -> list[dict[str, Any]]:
        path = self.directory(draft_id) / "events.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            event = json.loads(line)
            if event["seq"] > after:
                out.append(event)
        return out
