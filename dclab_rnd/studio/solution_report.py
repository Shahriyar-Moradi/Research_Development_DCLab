"""The solution report of a studio project (package 17.7): ``dclab_core.report`` over what the project recorded.

The inputs are the project's stage records (``store.records``), its brief (``core_adapter.brief_from_project``), its
adopted plan (``project["plan"]``, kept by the intern's adopt_plan) and, unless asked not to, the data findings the plan
is read from (``solution.core_plan``: the provisional training rows only, the label file joined when there is one).
The report's prose is written by code from those records; no model writes any of it. Reading the report changes
nothing in the project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from dclab_core import report as core_report
from dclab_core.contracts import DataFindings, SolutionReport

from ..core_adapter import brief_from_project

FORMATS = ("md", "html", "pdf")


def findings_for(store: Any, project: dict[str, Any]) -> Optional[DataFindings]:
    """dclab_core's findings on the project's table, as the plan reads them; None, logged, when they cannot be read."""
    solution = project.get("solution") or {}
    if not project.get("data") or not solution.get("target"):
        return None
    try:
        from . import labels as label_file
        from .data import data_path, load_table
        from .solution import core_plan

        frame = load_table(data_path(store, project))
        if (project.get("labels") or {}).get("target") == solution["target"]:
            frame = label_file.apply(store, project, frame)
        target = solution["target"]
        frame = frame[frame[target].notna()]
        return core_plan(frame.drop(columns=[target]), frame[target], target, solution.get("task") or "unknown")[0]
    except Exception as exc:  # noqa: BLE001 - the report then says the findings are missing
        from ..observe import log

        log.warning("findings unavailable for the report: %s", type(exc).__name__)
        return None


def inputs(store: Any, project_id: str, rubric: Optional[list[dict[str, Any]]] = None, analyse: bool = True) -> dict[str, Any]:
    """What ``dclab_core.report.build`` (and ``context``, to check it) takes for this project."""
    project = store.get(project_id)
    return {"brief": brief_from_project(project), "findings": findings_for(store, project) if analyse else None,
            "plan": project.get("plan"), "records": store.records(project_id), "rubric": rubric,
            "title": f"Solution report: {project.get('name') or 'project'}"}


def build(store: Any, project_id: str, rubric: Optional[list[dict[str, Any]]] = None, analyse: bool = True) -> SolutionReport:
    return core_report.build(**inputs(store, project_id, rubric, analyse))


def write(store: Any, project_id: str, directory: Path, rubric: Optional[list[dict[str, Any]]] = None,
          formats: tuple[str, ...] = FORMATS, analyse: bool = True) -> dict[str, Path]:
    """The report written as files in ``directory`` (``report.md``, ``report.html``, ``report.pdf``). The PDF needs
    reportlab and raises ``dclab_core.report.PdfUnavailable`` without it."""
    unknown = set(formats) - set(FORMATS)
    if unknown:
        raise ValueError(f"unknown report formats: {sorted(unknown)}")
    report = build(store, project_id, rubric, analyse)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    out: dict[str, Path] = {}
    if "md" in formats:
        out["md"] = directory / "report.md"
        out["md"].write_text(core_report.to_markdown(report), encoding="utf-8")
    if "html" in formats:
        out["html"] = directory / "report.html"
        out["html"].write_text(core_report.to_html(report), encoding="utf-8")
    if "pdf" in formats:
        out["pdf"] = core_report.to_pdf(report, directory / "report.pdf")
    return out
