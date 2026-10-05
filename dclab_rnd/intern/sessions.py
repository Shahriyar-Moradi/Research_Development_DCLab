"""Intern conversations on disk: ``<home>/<session id>.json``."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_BUDGET = {"max_steps": 24, "max_minutes": 20}
EXAMPLE_TASKS = [
    "Build a leakage-safe churn model on the Telco sample and tell me what the honest score is.",
    "Audit the credit-card fraud data for leakage, then screen algorithms with average precision as the metric.",
    "Forecast daily bike rentals: time-ordered split, calendar features, and say whether tuning helped.",
    "Classify the letter-recognition data (26 classes) and report macro-F1 with its interval.",
    "Use the clothing reviews: does the text beat the structured columns? Keep the star rating out.",
]


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def new_session(session_id: str, task: str, *, mode: str, model: str | None, budget: dict[str, Any] | None = None, project_id: str | None = None) -> dict[str, Any]:
    """A fresh intern session document, with its budget clamped. Shared by every store."""
    budget = {**DEFAULT_BUDGET, **{k: int(v) for k, v in (budget or {}).items() if k in DEFAULT_BUDGET}}
    budget["max_steps"] = max(3, min(budget["max_steps"], 80))
    budget["max_minutes"] = max(1, min(budget["max_minutes"], 240))
    return {
        "id": session_id, "task": task.strip()[:4000], "status": "queued", "mode": mode, "model": model,
        "budget": budget, "used": {"steps": 0, "minutes": 0.0, "input_tokens": 0, "output_tokens": 0},
        "plan": None, "steps": [], "messages": [], "final": None, "project_id": project_id, "error": None,
        "created": now(), "updated": now(),
    }


class SessionStore:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)

    def path(self, session_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", session_id or ""):
            raise KeyError(session_id)
        return self.home / f"{session_id}.json"

    def create(self, task: str, *, mode: str, model: str | None, budget: dict[str, Any] | None = None, project_id: str | None = None) -> dict[str, Any]:
        return self.save(new_session(uuid.uuid4().hex[:12], task, mode=mode, model=model, budget=budget, project_id=project_id))

    def get(self, session_id: str) -> dict[str, Any]:
        path = self.path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        return json.loads(path.read_text(encoding="utf-8"))

    def save(self, session: dict[str, Any]) -> dict[str, Any]:
        session["updated"] = now()
        path = self.path(session["id"])
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(session, indent=1, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        return session

    def list(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        out = []
        for path in self.home.glob("*.json"):
            try:
                s = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out.append({k: v for k, v in s.items() if k not in ("messages", "steps")} | {"steps": len(s.get("steps", []))})
        out.sort(key=lambda s: s.get("updated", ""), reverse=True)
        return out[offset:offset + limit] if limit is not None else out

    def delete(self, session_id: str) -> None:
        path = self.path(session_id)
        if path.exists():
            path.unlink()
