"""DCLab's arms: the twin's workflow lane (the rules) and its ai_first lane (the model as the agent), on a project seeded
from the case's table; each finished project read into the same outcome shape as the plain arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

FEATURE = "feature"  # the role a column the model learns from has in the data stage's column_roles


def seed_source(case: Any, folder: Path) -> str:
    """A workspace folder holding one project with the case's table and the request as its goal; returns the project id."""
    from dclab_rnd.storage import open_stores
    from dclab_rnd.studio import data as sd

    projects = open_stores(Path(folder))[0]
    project = projects.create(case.id, "general", case.request()[:300])
    case.table.to_csv(projects.data_dir(project["id"]) / case.filename, index=False)
    sd.attach_data(projects, project["id"], case.filename, source={"kind": "e2e", "case": case.id}, actor="human")
    return project["id"]


def outcome(arm: str, project: Optional[dict[str, Any]], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """The same shape as the plain arm's: what the project finished with, read from its own records."""
    final = (records.get("final") or {}).get("evidence") or {}
    roles = ((records.get("data") or {}).get("evidence") or {}).get("column_roles") or {}
    metric = (records.get("final") or {}).get("primary_metric")
    value = (final.get("holdout_metrics") or {}).get(metric) if metric else None
    ci = final.get("holdout_primary_metric_ci") or {}
    if project is None or value is None:
        return {"arm": arm, "finished": None, "holdout_uses": (project or {}).get("holdout_uses")}
    solution = project.get("solution") or {}
    return {"arm": arm, "holdout_uses": project.get("holdout_uses"),
            "finished": {"target": solution.get("target"), "task": solution.get("task"), "metric": metric, "value": float(value),
                         "low": ci.get("low"), "high": ci.get("high"), "features": sorted(c for c, r in roles.items() if r == FEATURE),
                         "report": ""}}
