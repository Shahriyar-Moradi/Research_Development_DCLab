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


def _programme_steps(project: dict[str, Any]) -> list[dict[str, Any]]:
    """The steps of a problem with several parts (24.5): a step per leaf, the composition chosen on validation and the one read of the holdout, each with the programme's
    fingerprint and the record paths of its numbers. Stale when the data, the tables, the labels or the solution map changed since the programme ran."""
    from . import programme

    plan = programme.plan_of(project)
    p = programme.stored(project)
    run = project.get("programme") or {}
    stale = programme.stale_reasons(project)
    why_stale = (" ".join(stale) + ".") if stale else None
    sm = programme.maps(project)[1]
    block_of = {b.id: b for b in sm.blocks}
    leaf_by = {l.block: l for l in (p.leaves if p else [])}
    steps: list[dict[str, Any]] = []
    numbers = list(contract.numbers(run.get("document") or {}, "programme.document", limit=core.MAX_NUMBERS)) if p else []
    ids = {}
    for b in sm.blocks:
        if b.approach == "review_budgeted":  # the budget step is the composition's, not a model
            continue
        leaf = leaf_by.get(b.id)
        step = {"id": f"programme:{b.id}", "kind": "stage", "name": f"{b.title} ({b.approach or 'no approach'})", "depends_on": ["data"], "nodes": ["WF-04", "WF-05", "WF-06", "WF-07"],
                "roots": ["programme"], "record": "project.programme.document.leaves", "rerun": "replay" if p else "rerun"}
        if leaf is None:
            step.update(status="not_run", why="The programme has not run.")
        elif leaf.status == "built":
            step.update(status="stale" if why_stale else "done", why=why_stale, fingerprint=p.fingerprint,
                        inputs={"reads": ",".join(leaf.reads) or None, "columns": len(leaf.columns)},
                        outputs={"chosen": leaf.chosen, **{f"validation.{k}": v for k, v in leaf.validation.items()}},
                        numbers=[n for n in numbers if n.startswith(f"programme.document.leaves[{[l.block for l in p.leaves].index(b.id)}]")])
        elif leaf.status == "dropped":
            step.update(status="not_recorded", why=leaf.why)
        else:
            step.update(status="failed", why=leaf.why or "The leaf did not run.")
        ids[b.id] = step["id"]
        steps.append(step)
    leaf_steps = [x["id"] for x in steps if x["status"] != "not_recorded"]  # a leaf the person dropped is listed but is no input of the composition: it does not make it stale
    chosen = p.composition if p else None
    comp = {"id": "programme:composition", "kind": "stage", "name": "the composition, chosen on validation", "depends_on": leaf_steps or ["data"], "nodes": ["WF-08"],
            "roots": ["programme"], "record": "project.programme.document.composition", "rerun": "replay" if p else "rerun"}
    if chosen is None:
        comp.update(status="not_run", why="Not chosen yet" + (": " + "; ".join(p.blockers[:2]) if p and p.blockers else "."))
    else:
        comp.update(status="stale" if why_stale else "done", why=why_stale, fingerprint=p.fingerprint,
                    outputs={"chosen": chosen.chosen, "best_leaf": chosen.best_leaf, "best_leaf_validation": chosen.best_leaf_score,
                             "chosen_validation": chosen.candidates.get(chosen.chosen)},
                    numbers=[n for n in numbers if n.startswith("programme.document.composition")])
    hold = {"id": "programme:holdout", "kind": "stage", "name": "the holdout, read once", "depends_on": ["programme:composition"], "nodes": ["WF-09"], "roots": ["programme"],
            "record": "project.programme.document.holdout", "rerun": "replay" if p and p.holdout else "rerun"}
    if p is None or p.holdout is None:
        hold.update(status="not_run", why="The holdout stays closed until every leaf has run and the composition is chosen.")
    else:
        h = p.holdout
        hold.update(status="stale" if why_stale else "done", why=why_stale, fingerprint=p.fingerprint,
                    outputs={"metric": h.metric, "value": h.composed.value, "low": h.composed.low, "high": h.composed.high, "best_leaf_value": h.best_leaf_estimate.value,
                             "difference": h.difference.value, "difference_low": h.difference.low, "beats_best_leaf": h.beats_best_leaf, "reads": h.reads},
                    numbers=[n for n in numbers if n.startswith("programme.document.holdout")])
    return steps + [comp, hold]


def _report(store: Any, project_id: str) -> Any:
    """The report as the product builds it: the findings of the training rows with it."""
    from . import solution_report

    return solution_report.build(store, project_id, analyse=True)


def _family(card: dict[str, Any]) -> Optional[str]:
    """The card's family as a word: a card keeps each field with its status and source (``{"value": ...}``)."""
    family = (card or {}).get("family")
    family = family.get("value") if isinstance(family, dict) else family
    return str(family) if family is not None else None


def _stage_status_of(project: dict[str, Any], stage: str) -> str:
    return ((project.get("stages") or {}).get(stage) or {}).get("status", "pending")


def of(store: Any, project_id: str) -> core.Pipeline:
    """The project's pipeline now: every step, its lineage and its status."""
    project = store.get(project_id)
    draft = project.get("draft") or {}
    data = project.get("data") or {}
    card = draft.get("card")
    steps: list[dict[str, Any]] = [
        {"id": "problem", "kind": "problem", "name": "the problem card", "record": "project.draft.card",
         **({"status": "done", "outputs": {"card": _hash(card), "family": _family(card)}} if card else
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
    from . import programme as studio_programme

    programme_steps = _programme_steps(project) if project.get("programme_plan") else []
    if programme_steps and studio_programme.stored(project) is None and any(_stage_status_of(project, st) in DONE for st in STAGE_KEYS):
        programme_steps = []  # a problem of several parts that was solved by the engine's stages and never run as a programme: its stages are its pipeline
    if programme_steps and not project.get("solution"):  # the plan the person confirmed stands in for the saved solution and the workflow: it is what the programme runs
        fingerprint = project["programme_plan"]["solution_map"].get("fingerprint")
        for step in steps:
            if step["id"] == "solution":
                step.update(name="the plan of the programme", status="done", why=None, outputs={"solution_hash": fingerprint, "kind": "programme"}, record="project.programme_plan")
            elif step["id"] == "workflow":
                step.update(name="the solution map's workflows", status="done", why=None, outputs={"solution_map": fingerprint}, record="project.programme_plan.solution_map")
    if programme_steps and not any(_stage_status_of(project, st) in DONE for st in STAGE_KEYS):
        return core.from_steps(project_id, steps + programme_steps)  # a problem of several parts runs as a programme, not as the five stages
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
    return core.from_steps(project_id, steps + stages + [analysis] + [report] + programme_steps)  # the analysis is built with the report, after the stages


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
    if step_id.startswith("programme:"):  # 24.5: the programme again in a scratch copy, compared with its record; the holdout is not consumed
        from . import programme

        if step.status == "not_run":
            return {"step": step_id, "replayed": False, "why": [step.why or "The programme has not run."]}
        try:
            got = programme.replay(store, project_id)
        except programme.ProgrammeError as exc:  # the inputs changed, or the holdout was never read: a replay would be a second look at it, so it is refused with the reason
            return {"step": step_id, "replayed": False, "why": [str(exc)]}
        from .. import audit

        audit.record_for(store, "replay", "human", project={"id": project_id}, move="replay_programme", status="allowed", run=got.get("fingerprint"))
        return {"step": step_id, **got, "why": [] if got.get("same") else ["The rerun did not make the same document: " + ", ".join(got.get("differ") or ["see the fingerprints"])]}
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
