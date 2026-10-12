"""Challenge mode (package 16.13): from a brief and its files to every deliverable, with a coverage table.

``solve(brief_text, files, out)``:

1. reads the brief (``dclab_core.challenge.read_brief``): its parts, kinds, requirements with points, confidentiality;
2. for each part, builds a project from its files (a table, and a label file when given): the part's text is read into
   a brief, the training rows are analysed, the plan is drawn and adopted as the solution, every stage runs and the
   final stage is approved;
3. exports per part: the notebook (or why it cannot be written), the report (Markdown, and PDF when reportlab is
   there), and the scoring service with its Dockerfile and tests (15.10);
4. writes ``COVERAGE.md``: every requirement, its points, and the file that meets it, or "not met" and why;
5. puts the deliverables in a git repository (one commit).

Confidentiality: when the brief forbids sharing the data, no data file is written into the deliverables (they never
are), and no model is asked anything (challenge mode runs on the rules; ``advisor`` must be None for a private brief).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np

COVERED_BY = {  # a requirement's kind -> where the deliverables meet it (a path per part, or the repository)
    "model": "partN/REPORT.md (results, with their intervals) and partN/notebook.ipynb",
    "features": "partN/REPORT.md (decisions and reasons: the feature recipe chosen and why)",
    "analysis": "partN/REPORT.md (what the data shows)",
    "report": "partN/REPORT.md (and REPORT.pdf)",
    "api": "partN/service/app.py (Flask, served by Gunicorn)",
    "docker": "partN/service/Dockerfile",
    "tests": "partN/service/tests/test_service.py",
    "source_control": "the git repository of the deliverables (.git)",
}
NOT_BUILT = {
    "decision": "no decision was stated for this part: give the action's value and cost (a review budget) or the price column "
                "(a price), and DCLab chooses it by expected value (16.9)",
    "other": "DCLab did not recognise what this line asks for: a person should check it",
}


def _moment(brief: Any, spec: dict[str, Any], part: dict[str, Any]) -> str:
    if spec.get("moment"):
        return str(spec["moment"])
    if brief.prediction_moment.known:
        return str(brief.prediction_moment.value)
    return f"When the prediction is made for {part['title']} (not stated in the brief: to confirm)."


def _report(project: dict[str, Any], records: dict[str, Any], part: dict[str, Any], plan: Any, quick: bool = False) -> str:
    """The part's report in Markdown, from the records (every number read from a record)."""
    final = (records.get("final") or {}).get("evidence") or {}
    features = (records.get("features") or {}).get("decision") or {}
    models = (records.get("models") or {}).get("evidence") or {}
    metric = (records.get("final") or {}).get("primary_metric")
    ci = final.get("holdout_primary_metric_ci") or {}
    lines = [f"# {part['title']}", "", "## Problem", "", part["text"][:1500], "", "## Data findings", ""]
    data = (records.get("data") or {}).get("claims") or []
    lines += [f"- {c['statement']}" for c in data[:6]] or ["- (no data record)"]
    lines += ["", "## Decisions", ""]
    solution = project.get("solution") or {}
    lines += [f"- **{d.key}**: {json.dumps(d.choice, default=str)[:200]} — {d.reason}" for d in plan.decisions
              if d.key != "labels.coverage"][:16]
    lines.append(f"- **labels**: {solution.get('labels', 'complete')} — {solution.get('label_provenance') or 'the target column of the table'}")
    lines += [f"- **feature recipe**: `{features.get('chosen')}` — {features.get('rule', '')}",
              f"- **model**: `{final.get('model')}` — {models.get('selection_rule', '')}", "", "## Results", ""]
    if metric and final.get("holdout_metrics"):
        lines.append(f"- Holdout {metric}: {final['holdout_metrics'].get(metric)} (95% {ci.get('method', 'interval')}: "
                     f"{ci.get('low')} to {ci.get('high')}); the holdout was used once.")
    for key in ("recall_at_budget", "precision_at_budget", "ap_lift", "mae", "macro_f1", "roc_auc"):
        if key in (final.get("holdout_metrics") or {}) and key != metric:
            lines.append(f"- {key}: {final['holdout_metrics'][key]}")
    lines += ["", "## Limits", ""]
    lines += [f"- {x}" for x in final.get("limits", [])] + ([f"- {final['precision_note']}"] if final.get("precision_note") else [])
    if quick:
        lines.append("- Run in quick mode: at most 3,000 rows were used, so the numbers are a first look, not the full table's.")
    lines += ["- These are scores on held-out data from the same source: evidence about the method, not a claim that the model is "
              "ready to be used on its own."]
    return "\n".join(lines) + "\n"


def _pdf(markdown: str, path: Path) -> bool:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    except ImportError:
        return False
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    story = []
    for line in markdown.splitlines():
        if line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["Title"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["Heading2"]))
        elif line.strip():
            story.append(Paragraph(escape(line.lstrip("- ")), styles["BodyText"]))
        else:
            story.append(Spacer(1, 6))
    SimpleDocTemplate(str(path), pagesize=A4).build(story)
    return True


def solve(brief_text: str, files: dict[str, dict[str, Any]], out: Path, *, advisor: Any = None,
          store_root: Optional[Path] = None, quick: bool = True, approve: bool = False) -> dict[str, Any]:
    """Solve the challenge into ``out``; ``files`` maps a part id ("part1") to {"table": path, "target": column,
    "labels": path (optional), "label_target": name, "moment": text (optional), "group_column", "time_column"}.

    ``approve``: the person running challenge mode approves each part's final stage (recorded as their approval); without
    it nothing is approved, and no service is exported (a service is built only from an approved result). Without
    ``store_root`` the projects live in a temporary folder that is removed at the end (a confidential file is not left
    behind)."""
    cleanup = store_root is None
    store_root = Path(store_root or tempfile.mkdtemp())
    try:
        return _solve(brief_text, files, Path(out), advisor=advisor, store_root=store_root, quick=quick, approve=approve)
    finally:
        if cleanup:
            shutil.rmtree(store_root, ignore_errors=True)


def _solve(brief_text: str, files: dict[str, dict[str, Any]], out: Path, *, advisor: Any, store_root: Path, quick: bool,
           approve: bool) -> dict[str, Any]:
    from dclab_core.analyse import analyse
    from dclab_core.challenge import read_brief
    from dclab_core.plan import plan as draw_plan
    from dclab_core.understand import understand

    from ..core_adapter import solution_fields_from_plan
    from ..studio import ProjectStore, data as sd, engine, export, service
    from ..studio.solution import Solution

    reading = read_brief(brief_text)
    if reading["private"] and advisor is not None:
        raise ValueError(f"The brief says the data is private (\"{reading['private_words']}\"): challenge mode runs it without a model")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    store = ProjectStore(store_root)
    done: dict[str, dict[str, Any]] = {}
    for part in reading["parts"]:
        spec = files.get(part["id"])
        if not spec:
            done[part["id"]] = {"error": "no files were given for this part"}
            continue
        try:  # one part that fails is said in the coverage; the others still deliver
            done[part["id"]] = _part(part, spec, store, out, reading, quick, approve)
        except Exception as exc:  # noqa: BLE001 - the coverage table says why
            done[part["id"]] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
    return _deliver(reading, done, out)


def _part(part: dict[str, Any], spec: dict[str, Any], store: Any, out: Path, reading: dict[str, Any], quick: bool,
          approve: bool) -> dict[str, Any]:
    from dclab_core.analyse import analyse
    from dclab_core.plan import plan as draw_plan
    from dclab_core.understand import understand

    from ..core_adapter import solution_fields_from_plan
    from ..studio import data as sd, engine, export, graph, service
    from ..studio.solution import Solution

    project = store.create(part["title"][:60], "general", part["text"][:600])
    pid = project["id"]
    table_path = Path(spec["table"])
    shutil.copyfile(table_path, store.data_dir(pid) / table_path.name)
    sd.attach_data(store, pid, table_path.name, source={"kind": "challenge"})
    target = spec.get("target")
    if spec.get("labels"):
        labels_path = Path(spec["labels"])
        shutil.copyfile(labels_path, store.data_dir(pid) / labels_path.name)
        sd.attach_labels(store, pid, labels_path.name, target=spec.get("label_target", "is_case"))
        target = spec.get("label_target", "is_case")
    project = store.get(pid)
    frame = sd.load_table(sd.data_path(store, project))
    if project.get("labels"):
        from ..studio import labels as label_file

        frame = label_file.apply(store, project, frame)
    brief = understand(part["text"], {table_path.name: list(frame.columns)})
    task = (project.get("suggestion") or {}).get("task") or part["task"] or (brief.task.value if brief.task.known else None)
    train = np.sort(np.random.default_rng(0).choice(len(frame), int(0.8 * len(frame)), replace=False))
    findings = analyse({"table": frame}, target=target, task=task, brief=brief, train_index=frame.index[train])
    plan = draw_plan(brief, findings)
    moment = _moment(brief, spec, part)
    fields = solution_fields_from_plan(plan, target, moment)
    suggestion = project.get("suggestion") or {}
    merged = {**fields, **{k: v for k, v in suggestion.items() if k in ("labels", "label_provenance", "positive_label")},
              **{k: spec[k] for k in ("group_column", "time_column", "alert_budget", "decision") if spec.get(k)}}
    project["solution"] = Solution(**merged).model_dump()
    verdict = graph.check(project, "set_solution", "agent", target=target)  # the same move, checked and logged
    graph.log(store, pid, verdict, project, outcome="done: solution from the challenge's plan" if verdict.allowed else None)
    if not verdict.allowed:
        raise ValueError(verdict.message)
    project["plan"] = plan.model_dump(mode="json")
    project["settings"]["quick"] = quick
    project["settings"]["private"] = bool(reading["private"])  # no stage asks a model about private data
    store.save(project)
    engine.run_all(store, pid)
    records = store.records(pid)
    folder = out / part["id"]
    folder.mkdir(parents=True, exist_ok=True)
    approved = False
    if approve:
        engine.approve(store, pid, "final", actor="human")  # the person who ran challenge mode with approve=True
        approved = True
        records = store.records(pid)
    rubric = [r for r in reading["requirements"] if r["part"] in (None, part["id"])]
    try:  # 17.7: the solution report from the records, every number traced to one
        from dclab_core import report as core_report

        from ..studio import solution_report

        built = solution_report.build(store, pid, rubric=rubric)
        report = f"> Part of the challenge: {part['title']}\n\n" + core_report.to_markdown(built)
        if quick:
            report += "\n\nRun in quick mode: at most 3,000 rows were used, so the numbers are a first look, not the full table's.\n"
        (folder / "REPORT.html").write_text(core_report.to_html(built), encoding="utf-8")
        try:
            core_report.to_pdf(built, folder / "REPORT.pdf")
            pdf = True
        except core_report.PdfUnavailable:
            pdf = False
    except Exception:  # noqa: BLE001 - the report from the records failed: the short one says what the records hold
        report = _report(store.get(pid), records, part, plan, quick)
        pdf = _pdf(report, folder / "REPORT.pdf")
    (folder / "REPORT.md").write_text(report, encoding="utf-8")
    try:
        (folder / "notebook.ipynb").write_text(export.dumps_notebook(export.notebook(store.get(pid), records)), encoding="utf-8")
        notebook = "notebook.ipynb"
    except export.ExportUnsupported as exc:
        (folder / "NOTEBOOK.md").write_text(f"# Notebook not written\n\n{exc}\n", encoding="utf-8")
        notebook = None
    service_out = None
    if approved:
        try:
            key = service.export(store, pid, folder / "service")["key"]
            service_out = {"folder": "service", "key_given": bool(key)}  # the key goes to the caller, never into the folder
        except service.ServiceRefused as exc:
            (folder / "SERVICE.md").write_text(f"# Service not exported\n\n{exc}\n", encoding="utf-8")
    final = records["final"]["evidence"]
    decision = final.get("decision")
    return {"decision": None if decision is None else {"action": decision.get("decision"), "made": bool(decision.get("learnable", True)
                                                       and not decision.get("warning")), "warning": decision.get("warning")},
            "project": pid, "task": project["solution"]["task"], "recipe": final["feature_recipe"], "model": final["model"],
            "holdout": final["holdout_metrics"].get(records["final"]["primary_metric"]), "notebook": notebook, "pdf": pdf,
            "approved": approved, "service": service_out, "service_key": key if service_out else None}


def _deliver(reading: dict[str, Any], done: dict[str, dict[str, Any]], out: Path) -> dict[str, Any]:
    coverage = []
    for r in reading["requirements"]:
        if r["part"] and "error" in done.get(r["part"], {"error": "no files were given for this part"}):
            coverage.append({**r, "met": False, "why": f"{r['part']} was not solved: {done.get(r['part'], {}).get('error', 'no files')}"})
            continue
        parts = [r["part"]] if r["part"] else [p for p in done if "error" not in done[p]]
        if r["kind"] in ("api", "docker", "tests"):  # served only from an approved result
            unapproved = [p for p in parts if not done[p].get("service")]
            if unapproved:
                coverage.append({**r, "met": False, "why": f"no service for {', '.join(unapproved)}: run challenge mode with approve=True "
                                                            "(a service is built only from an approved final stage), or see SERVICE.md"})
                continue
        if r["kind"] == "decision" and parts and any(done[p].get("decision") for p in parts):
            made = [p for p in parts if (done[p].get("decision") or {}).get("made")]
            refused = [p for p in parts if done[p].get("decision") and not done[p]["decision"]["made"]]
            if made and not refused:
                coverage.append({**r, "met": True, "where": "; ".join(f"{p}/REPORT.md (the decision, its interval and its assumption)" for p in made)})
            else:
                coverage.append({**r, "met": False, "why": "; ".join(f"{p}: {done[p]['decision']['warning']}" for p in refused)})
            continue
        if r["kind"] in COVERED_BY and parts:
            if r["kind"] == "source_control":
                where = COVERED_BY["source_control"]  # one repository for every part
            else:
                where = "; ".join(COVERED_BY[r["kind"]].replace("partN", p) for p in parts)
            if r["kind"] == "report":
                where = "; ".join(f"{p}/REPORT.md" + (" and REPORT.pdf" if done[p].get("pdf") else "") for p in parts)
            if r["kind"] == "model":
                for p in parts:
                    if not done[p].get("notebook"):  # the note that says why, never a file that is not there
                        where = where.replace(f"{p}/notebook.ipynb", f"{p}/NOTEBOOK.md (why the notebook is not written yet)")
            coverage.append({**r, "met": True, "where": where})
        else:
            coverage.append({**r, "met": False, "why": NOT_BUILT.get(r["kind"], NOT_BUILT["other"])})
    lines = ["# Coverage of the brief", "", "| Requirement | Points | Met | Where, or why not |", "|---|---|---|---|"]
    for c in coverage:
        lines.append(f"| {c['text']} | {c['points'] if c['points'] is not None else ''} | {'yes' if c['met'] else 'no'} | "
                     f"{c.get('where') or c.get('why')} |")
    met = sum(c["points"] or 0 for c in coverage if c["met"])
    lines += ["", f"Points covered: {met} of {reading['total_points'] or 0}." if reading["total_points"] else "",
              "Private data: " + ("the brief forbids sharing it; no data file is in these deliverables and no model was asked anything."
                                  if reading["private"] else "the brief puts no restriction on the data; still, no data file is in these deliverables.")]
    (out / "COVERAGE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "README.md").write_text(f"# Deliverables\n\nSolved by DCLab's challenge mode: {len(done)} part(s). See COVERAGE.md.\n", encoding="utf-8")
    (out / ".gitignore").write_text("*.csv\n*.parquet\n*.xlsx\n__pycache__/\n", encoding="utf-8")
    git = ["git", "-c", "user.name=DCLab", "-c", "user.email=dclab@localhost"]
    subprocess.run(["git", "init", "-q"], cwd=out, check=True)
    subprocess.run(git + ["add", "-A"], cwd=out, check=True)
    subprocess.run(git + ["commit", "-q", "-m", "Deliverables of the challenge, as DCLab solved it"], cwd=out, check=True)
    return {"reading": reading, "parts": done, "coverage": coverage, "out": str(out)}
