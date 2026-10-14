"""A project's pipeline of workflows (package 20.8), from what the project and its stage records hold.

The steps: the problem card the draft built (or none, for a project made in the wizard), the table and its card, the
saved solution, the drawn workflow and plan, the analysis of the training rows (its findings), each engine stage with
its run record (19.5), and the solution report. A stage is stale when what its run read differs from the project now
(the table, the solution, the settings, the related tables and the other inputs the run recorded, or an earlier
stage's run and choice), or when its record no longer says what its run made (edited after the run); every later step
is stale with it. The code a run ran on is not compared here (``lineage.replay`` checks it).

``rerun`` reruns one step from its record: a stage is replayed in a scratch copy (``lineage.replay``) and compared with
its record; the report is rebuilt and each number compared with the value its stage's run recorded when it ran. It runs
as a job (``jobs``: the limits on jobs at once, and Stop); the project is never changed by it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Optional

from dclab_core import lineage as contract
from dclab_core import pipeline as core

from .graph import solution_hash
from .store import STAGE_KEYS

DONE = ("completed", "approved")
NODES = {"data": ["WF-03", "WF-04"], "leakage": ["WF-05"], "features": ["WF-06"], "models": ["WF-07"], "final": ["WF-08", "WF-09"]}
FINDINGS = ("findings", "findings_meta")


def _hash(value: Any) -> Optional[str]:
    if value in (None, {}, []):
        return None
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _edited(record: dict[str, Any], run: contract.RunRecord) -> list[str]:
    """What the record says now that its run did not make: its chosen value, or a number the run recorded."""
    from .lineage import _chosen_value

    out = []
    value = _chosen_value(record)
    decision = record.get("decision") or {}
    approved = decision.get("chosen") is not None and str(decision.get("chosen"))[:300] != run.outputs.chosen  # a person chose another option
    if not approved and run.outputs.value is not None and (value is None or abs(value - run.outputs.value) > 1e-9 * max(1.0, abs(run.outputs.value))):
        out.append(f"{run.outputs.metric or 'its value'}")
    now = contract.numbers(record.get("evidence") or {}, "evidence", limit=10**6)
    out += [k for k, v in run.outputs.numbers.items() if k in now and abs(now[k] - v) > 1e-9 * max(1.0, abs(v))][:5]
    return out


def _stage_step(store: Any, project: dict[str, Any], stage: str, earlier: dict[str, Optional[dict[str, Any]]]) -> dict[str, Any]:
    from .lineage import inputs_of

    record = store.read_stage(project["id"], stage)
    status = ((project.get("stages") or {}).get(stage) or {}).get("status")
    depends = [f"stage:{STAGE_KEYS[STAGE_KEYS.index(stage) - 1]}"] if stage != STAGE_KEYS[0] else ["workflow"]
    step = {"id": f"stage:{stage}", "kind": "stage", "name": stage, "depends_on": depends, "nodes": NODES[stage], "roots": [stage],
            "record": f"stages/{stage}.json"}
    if not record or status not in DONE:
        why = {"failed": "The stage failed.", "running": "The stage is running."}.get(status or "", "The stage has not run since its inputs changed.")
        return {**step, "status": "failed" if status == "failed" else "not_run", "why": why, "rerun": "rerun"}
    decision = record.get("decision") or {}
    chosen = decision.get("chosen") if decision.get("chosen") is not None else decision.get("selected")
    run = record.get("run")
    if not run:
        return {**step, "status": "not_recorded", "why": "It ran before runs were recorded (19.5): what it read cannot be checked.",
                "rerun": "rerun", "outputs": {"metric": record.get("primary_metric"), "chosen": None if chosen is None else str(chosen)[:300]}}
    run = contract.RunRecord.model_validate(run)
    then = run.inputs.model_dump()
    now = inputs_of(project, earlier).model_dump()
    upstream_then, upstream_now = then.pop("upstream"), now.pop("upstream")
    then.pop("sampling"), now.pop("sampling")  # derived by the run from the table, settings, solution and labels compared here
    changed = sorted(k for k in set(then) | set(now) if then.get(k) != now.get(k) and k != "seed")
    if sorted(upstream_then) != sorted(upstream_now):
        changed.append("an earlier stage's run")
    edited = _edited(record, run)
    why = " ".join(([f"Changed since it ran: {', '.join(changed)}."] if changed else [])
                   + ([f"Its record no longer says what its run made: {', '.join(edited)}."] if edited else [])) or None
    overridden = chosen is not None and run.outputs.chosen is not None and str(chosen)[:300] != run.outputs.chosen
    return {**step, "status": "stale" if why else "done", "why": why,
            "inputs": {"data_sha256": run.inputs.data_sha256, "solution_hash": run.inputs.solution_hash, **{f"ref.{k}": v for k, v in run.inputs.refs.items()},
                       **{f"chose.{k}": v for k, v in run.inputs.decisions.items()}},
            "outputs": {"metric": run.outputs.metric, "value": run.outputs.value, "low": run.outputs.low, "high": run.outputs.high,
                        "chosen": None if chosen is None else str(chosen)[:300], "rule_chose": run.outputs.chosen, "overridden": overridden},
            "run": run.run, "fingerprint": run.fingerprint, "rerun": "replay",
            # every number the stage's record holds, by its record path (the run keeps only the first ones)
            "numbers": list(contract.numbers(record.get("evidence") or {}, f"{stage}.evidence", limit=core.MAX_NUMBERS))}


def _report(store: Any, project_id: str) -> Any:
    """The report as the product builds it: the findings of the training rows with it."""
    from . import solution_report

    return solution_report.build(store, project_id, analyse=True)


def of(store: Any, project_id: str) -> core.Pipeline:
    """The project's pipeline now: every step, its lineage and its status."""
    project = store.get(project_id)
    draft = project.get("draft") or {}
    data = project.get("data") or {}
    card = draft.get("card")
    steps: list[dict[str, Any]] = [
        {"id": "problem", "kind": "problem", "name": "the problem card", "record": "project.draft.card",
         **({"status": "done", "outputs": {"card": _hash(card), "family": (card or {}).get("family")}} if card else
            {"status": "not_recorded", "why": "The project was made without a problem card (the wizard): its solution states the problem."})},
        {"id": "data", "kind": "data", "name": "the table and its card", "nodes": ["WF-02"], "record": "project.data",
         **({"status": "done", "inputs": {"cleaning_steps": len(draft.get("cleaning_log") or []) or None},
             "outputs": {"sha256": data.get("sha256"), "rows": data.get("rows"),
                         "columns": len(data["columns"]) if isinstance(data.get("columns"), list) else None, "card": _hash(data.get("card")),
                         "tables": _hash({n: t.get("sha256") for n, t in sorted((project.get("tables") or {}).items())})}} if data else
            {"status": "not_run", "why": "No table is attached."})},
        {"id": "solution", "kind": "solution", "name": "the saved solution", "depends_on": ["problem", "data"], "nodes": ["WF-01"],
         "roots": ["brief"], "record": "project.solution",
         **({"status": "done", "outputs": {"solution_hash": solution_hash(project.get("solution")), "kind": (project.get("solution") or {}).get("kind")}}
            if project.get("solution") else {"status": "not_run", "why": "No solution is saved."})},
        {"id": "workflow", "kind": "workflow", "name": "the solution workflow and plan", "depends_on": ["solution"],
         "roots": ["plan", "plan_risks", "plan_would_change"], "record": "project.draft.workflow and project.plan",
         **({"status": "done", "outputs": {"workflow": _hash(draft.get("workflow")), "plan": _hash(project.get("plan"))}}
            if draft.get("workflow") or project.get("plan") else
            {"status": "not_recorded", "why": "No workflow was drawn and no plan adopted: the engine's five stages are its workflow."})},
    ]
    analysis = {"id": "analysis", "kind": "analysis", "name": "the analysis of the training rows", "depends_on": ["data", "solution"],
                "roots": list(FINDINGS), "record": "the findings, built from the training rows", "rerun": "rerun"}
    earlier: dict[str, Optional[dict[str, Any]]] = {}
    stages = []
    for stage in STAGE_KEYS:
        stages.append(_stage_step(store, project, stage, dict(earlier)))
        earlier[stage] = store.read_stage(project_id, stage)
    report = {"id": "report", "kind": "report", "name": "the solution report", "depends_on": ["stage:final", "analysis"],
              "roots": ["title", "rubric"], "record": "the report built from the records", "rerun": "rerun"}
    if stages[-1]["status"] in ("done", "stale"):  # a stale final still has its records: the report is built, then marked stale
        built = _report(store, project_id)
        found = sorted(p for p in built.numbers if p.split(".", 1)[0].split("@")[0] in FINDINGS)
        analysis.update({"status": "done", "numbers": found, "outputs": {"numbers": len(found)}} if found else
                        {"status": "not_recorded", "why": "The analysis found nothing the report cites (or the training rows could not be read)."})
        report.update({"status": "done", "numbers": sorted(built.numbers)[:core.MAX_NUMBERS], "outputs": {"numbers": len(built.numbers)}})
    else:
        analysis.update({"status": "not_run", "why": "The report, which reads the analysis, is built after the final stage."})
        report.update({"status": "not_run", "why": "The final stage has not run."})
    return core.from_steps(project_id, steps + stages + [analysis] + [report])  # the analysis is built with the report, after the stages


def _same(a: Any, b: Any) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-9 * max(1.0, abs(float(a)))
    except (TypeError, ValueError):
        return False


def _ran(store: Any, project_id: str, path: str) -> tuple[bool, Any]:
    """The value at a report's record path as the stage's run recorded it when it ran: (known, value)."""
    root, _, rest = path.partition(".")
    if root not in STAGE_KEYS or "@" in path or "#" in path:
        return False, None
    record = store.read_stage(project_id, root) or {}
    if not record.get("run"):
        return False, None
    run = contract.RunRecord.model_validate(record["run"])
    if rest in run.outputs.numbers:
        return True, run.outputs.numbers[rest]
    metric = run.outputs.metric
    if root == "final" and metric and rest == f"evidence.holdout_metrics.{metric}":
        return True, run.outputs.value
    return False, None


def rerun(store: Any, project_id: str, step_id: str) -> dict[str, Any]:
    """Rerun one step from its record: a stage is replayed in a scratch copy and compared with its record; the report is
    rebuilt and each number compared with what its stage's run recorded when it ran. The project is never changed."""
    pipe = of(store, project_id)
    step = pipe.step(step_id)
    if step.kind == "stage":
        from .lineage import replay

        return {"step": step_id, **replay(store, project_id, step.name)}
    if step.kind == "report":
        if step.status not in ("done", "stale"):
            return {"step": step_id, "replayed": False, "why": [step.why or "The report cannot be built yet."]}
        again = _report(store, project_id).numbers
        compared, differ = 0, []
        for path, value in sorted(again.items()):
            known, ran = _ran(store, project_id, path)
            if known:
                compared += 1
                if not _same(ran, value):
                    differ.append(path)
        return {"step": step_id, "replayed": True, "numbers": len(again), "compared_with_runs": compared, "same": not differ,
                "differ": differ[:20], "why": [f"{len(differ)} report numbers differ from what their runs made."] if differ else []}
    return {"step": step_id, "replayed": False, "why": [f"The {step.name} is not rerun from a record: it is what the person stated, attached or "
                                                          "what the report reads afresh."]}
