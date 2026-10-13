"""The solution report of a studio project (package 17.7): ``dclab_core.report`` over what the project recorded.

The inputs are the project's stage records (``store.records``), its brief (``core_adapter.brief_from_project``), its
adopted plan (``project["plan"]``, kept by the intern's adopt_plan) and, unless asked not to, dclab_core's data findings
read on the run's own training rows (``engine.prepare``: the table, sample and locked holdout the stages used, the
label file joined when there is one; the holdout is left out; at most 20,000 rows, as ``solution.propose`` reads).
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
MAX_ROWS = 20_000  # what solution.propose reads at most


def findings_for(store: Any, project: dict[str, Any]) -> tuple[Optional[DataFindings], Optional[str]]:
    """dclab_core's findings on the run's training rows, and how those rows were chosen; (None, None), logged, when
    they cannot be read. When the run's split cannot be rebuilt, the plan's own reading (a seeded 75 % of the table,
    read before the holdout was locked) is used, and the note says plainly that it includes most of the held-out rows."""
    solution = project.get("solution") or {}
    if not project.get("data") or not solution.get("target"):
        return None, None
    target, task = solution["target"], solution.get("task") or "unknown"
    try:
        from dclab_core.analyse import analyse

        from . import engine

        prepared = engine.prepare(store, project)  # the same table, sample and locked holdout the stages used
        if hasattr(prepared, "machine"):  # a kind's own machine (16.15) keeps its training rows itself
            frame = prepared.machine.training_rows()
            frame = frame.sample(MAX_ROWS, random_state=42) if len(frame) > MAX_ROWS else frame
            return analyse({"table": frame}, target=target, task=task, train_index=frame.index), "the run's training rows; the holdout is left out"
        y = prepared.bundle.y_train
        if prepared.class_labels:  # multiclass codes back to the class names
            labels = list(prepared.class_labels)
            y = y.map(lambda code: labels[int(code)])
        frame = prepared.bundle.X_train.assign(**{target: y.to_numpy()})
        note = "the run's training rows; the holdout is left out"
        if len(frame) > MAX_ROWS:
            frame = frame.sample(MAX_ROWS, random_state=42)
            note = f"a seeded sample of {MAX_ROWS} of the run's training rows; the holdout is left out"
        return analyse({"table": frame}, target=target, task=task, train_index=frame.index), note
    except Exception as exc:  # noqa: BLE001 - fall back to the plan's own reading, said plainly
        from ..observe import log

        log.warning("training-row findings unavailable for the report: %s", type(exc).__name__)
    try:
        from . import labels as label_file
        from .data import data_path, load_table
        from .solution import core_plan

        frame = load_table(data_path(store, project))
        if (project.get("labels") or {}).get("target") == target:
            frame = label_file.apply(store, project, frame)
        frame = frame[frame[target].notna()]
        return (core_plan(frame.drop(columns=[target]), frame[target], target, task)[0],
                "a seeded 75% of the table, read before the run locked its holdout, so it includes most of the held-out rows")
    except Exception as exc:  # noqa: BLE001 - the report then says the findings are missing
        from ..observe import log

        log.warning("findings unavailable for the report: %s", type(exc).__name__)
        return None, None


def inputs(store: Any, project_id: str, rubric: Optional[list[dict[str, Any]]] = None, analyse: bool = True) -> dict[str, Any]:
    """What ``dclab_core.report.build`` (and ``context``, to check it) takes for this project."""
    project = store.get(project_id)
    findings, note = findings_for(store, project) if analyse else (None, None)
    return {"brief": brief_from_project(project), "findings": findings, "rows_note": note,
            "plan": project.get("plan"), "records": store.records(project_id), "rubric": rubric,
            "title": f"Solution report: {project.get('name') or 'project'}"}


def build(store: Any, project_id: str, rubric: Optional[list[dict[str, Any]]] = None, analyse: bool = True,
          advisor: Any = None) -> SolutionReport:
    """The report from the records; with ``advisor`` (a dclab_core Advisor, such as ``core_advisor.GatewayAdvisor``), the
    reporter's summaries added above each section's lines, every sentence kept only when ``cited.check`` passes it
    (package 18.7). A private project never gets prose: no model sees its report."""
    report = core_report.build(**inputs(store, project_id, rubric, analyse))
    if advisor is None or (store.get(project_id).get("settings") or {}).get("private"):
        return report
    from .. import cited

    report, info = core_report.write_prose(advisor, report, cited.check)
    store.log(project_id, "report_prose", {k: info[k] for k in ("source", "reason", "kept", "dropped")})
    from ..ledger import capture  # 19.7: each summary sentence kept or dropped by the citation check

    capture.report_claims(project_id, info)
    return report


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
