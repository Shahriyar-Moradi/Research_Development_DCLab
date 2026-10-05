"""Projects on disk: ``<home>/<project id>/`` holds project.json, data/, stages/, activity.jsonl and transitions.jsonl."""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

STAGE_KEYS = ("data", "leakage", "features", "models", "final")
INDUSTRIES = (
    "general", "fintech and banking", "insurance", "retail and e-commerce", "telecom", "logistics and delivery",
    "health", "manufacturing", "energy", "marketing", "human resources", "public sector",
)


def now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def safe_name(name: str, default: str = "data.csv") -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name or default).name).strip("._") or default
    return name[:120]


def migrate(project: dict[str, Any]) -> bool:
    """Bring a project.json written before the rename of the prediction contract to the solution up to date.

    Old files kept the solution under "contract", the owner's sign-off as "contract_hash", the gate as
    "contract" and the policy switch as "require_contract_signoff". Returns True when anything changed.
    """
    changed = False
    if "contract" in project:
        old = project.pop("contract")
        project.setdefault("solution", old)
        changed = True
    pol = project.get("policy")
    if isinstance(pol, dict) and "require_contract_signoff" in pol:
        pol.setdefault("require_solution_signoff", pol.pop("require_contract_signoff"))
        changed = True
    for entry in [project.get("signoff") or {}, *(project.get("approvals") or [])]:
        if isinstance(entry, dict):
            if entry.get("gate") == "contract":
                entry["gate"] = "solution"; changed = True
            if "contract_hash" in entry:
                entry.setdefault("solution_hash", entry.pop("contract_hash")); changed = True
    return changed


def new_project(project_id: str, name: str, industry: str = "general", goal: str = "") -> dict[str, Any]:
    """A fresh project document. Shared by every store, so a project looks the same wherever it is kept."""
    return {
        "id": project_id,
        "name": (name or "Untitled project").strip()[:120],
        "industry": industry if industry in INDUSTRIES else "general",
        "goal": (goal or "").strip()[:4000],
        "created": now(),
        "updated": now(),
        "data": None,
        "solution": None,
        "proposal": None,
        "settings": {"max_rows": 20000, "quick": False},
        "stages": {key: {"status": "pending"} for key in STAGE_KEYS},
        "decisions": {},
        "holdout_uses": 0,
        "running": None,
    }


class ProjectStore:
    def __init__(self, home: Path):
        self.home = Path(home)
        self.home.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ paths
    def directory(self, project_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{12}", project_id or ""):
            raise KeyError(project_id)
        return self.home / project_id

    def data_dir(self, project_id: str) -> Path:
        return self.directory(project_id) / "data"

    def export_dir(self, project_id: str) -> Path:
        """The folder a project's exported notebook and report are written to."""
        path = self.directory(project_id) / "exports"
        path.mkdir(exist_ok=True)
        return path

    def location(self) -> str:
        """Where this workspace lives, in words for the Admin page."""
        return str(self.home.parent)

    def has_stage(self, project_id: str, stage: str) -> bool:
        return self.stage_path(project_id, stage).exists()

    def stage_path(self, project_id: str, stage: str) -> Path:
        if stage not in STAGE_KEYS:
            raise KeyError(stage)
        return self.directory(project_id) / "stages" / f"{stage}.json"

    # ------------------------------------------------------------------ CRUD
    def create(self, name: str, industry: str = "general", goal: str = "") -> dict[str, Any]:
        project = new_project(uuid.uuid4().hex[:12], name, industry, goal)
        directory = self.directory(project["id"])
        (directory / "data").mkdir(parents=True)
        (directory / "stages").mkdir()
        self.save(project)
        self.log(project["id"], "created", {"name": project["name"]})
        return project

    def get(self, project_id: str) -> dict[str, Any]:
        path = self.directory(project_id) / "project.json"
        if not path.exists():
            raise KeyError(project_id)
        project = json.loads(path.read_text(encoding="utf-8"))
        if migrate(project):
            self.save(project)
        return project

    def save(self, project: dict[str, Any]) -> dict[str, Any]:
        project["updated"] = now()
        path = self.directory(project["id"]) / "project.json"
        temp = path.with_suffix(".json.tmp")
        temp.write_text(json.dumps(project, indent=2, ensure_ascii=False), encoding="utf-8")
        temp.replace(path)
        return project

    def update(self, project_id: str, **values: Any) -> dict[str, Any]:
        project = self.get(project_id)
        project.update(values)
        return self.save(project)

    def list(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        projects = []
        for path in self.home.glob("*/project.json"):
            try:
                project = json.loads(path.read_text(encoding="utf-8"))
                migrate(project)
                projects.append(project)
            except (OSError, json.JSONDecodeError):
                continue
        projects.sort(key=lambda p: p.get("updated", ""), reverse=True)
        return projects[offset:offset + limit] if limit is not None else projects

    def delete(self, project_id: str) -> None:
        import shutil

        directory = self.directory(project_id)
        if directory.exists():
            shutil.rmtree(directory)

    # ------------------------------------------------------------------ stages and activity
    def write_stage(self, project_id: str, stage: str, record: dict[str, Any]) -> Path:
        path = self.stage_path(project_id, stage)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        return path

    def read_stage(self, project_id: str, stage: str) -> dict[str, Any] | None:
        path = self.stage_path(project_id, stage)
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

    def records(self, project_id: str) -> dict[str, dict[str, Any]]:
        out = {}
        for stage in STAGE_KEYS:
            record = self.read_stage(project_id, stage)
            if record:
                out[stage] = record
        return out

    def clear_stages(self, project_id: str, from_stage: str = "data") -> None:
        """Downstream results are invalid once the data or the solution changes."""
        start = STAGE_KEYS.index(from_stage)
        project = self.get(project_id)
        for stage in STAGE_KEYS[start:]:
            path = self.stage_path(project_id, stage)
            if path.exists():
                path.unlink()
            project["stages"][stage] = {"status": "pending"}
        project["decisions"] = {k: v for k, v in project["decisions"].items() if STAGE_KEYS.index(k) < start} if project["decisions"] else {}
        self.save(project)

    def log(self, project_id: str, kind: str, payload: Any, at: str | None = None) -> None:
        line = json.dumps({"at": at or now(), "kind": kind, "payload": payload}, ensure_ascii=False)
        with (self.directory(project_id) / "activity.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def activity(self, project_id: str, limit: int = 60) -> list[dict[str, Any]]:
        path = self.directory(project_id) / "activity.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]

    # ------------------------------------------------------------------ the workflow graph's transition log
    def transition(self, project_id: str, entry: dict[str, Any]) -> None:
        """Append one validated move (allowed or not). Written by ``graph.log``; never edited."""
        with (self.directory(project_id) / "transitions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def transitions(self, project_id: str, limit: int = 200) -> list[dict[str, Any]]:
        path = self.directory(project_id) / "transitions.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:] if line.strip()]
