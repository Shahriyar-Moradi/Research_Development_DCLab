"""The DCLab notebook's projects: create, change, delete; data and the solution; stages, gates and the workflow graph;
memory; questions; exports; the copilot's review of the exported notebook."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import Body, Depends, HTTPException, Request
from fastapi.responses import Response

from ... import audit, lessons as workspace_lessons
from ...studio import agent as studio_agent, data as studio_data, engine as studio_engine, export as studio_export
from ...studio import graph as studio_graph, lineage as studio_lineage, memory as studio_memory, sft as studio_sft, solution as studio_solution
from ..api_models import (OCTET, AnyObject, Answer, Fixes, GateApproval, Graph, Lessons, MoveCheck, Project, ProjectCreate, ProjectPatch, Proposal,
                          ProposalRequest, Page, Question, Review, Router, SampleChoice, SolutionBody, StageApproval, Verdict, download)
from ..services import charge_upload, Services, services

router = Router()


@router.get("/api/projects", response_model=list[Project])
async def list_projects(s: Services = Depends(services)):
    return [{k: v for k, v in p.items() if k not in ("proposal",)} | {"data": p["data"] and {k: v for k, v in p["data"].items() if k != "profile"}}
            for p in s.projects.list()]


@router.post("/api/projects", status_code=201, response_model=Project)
async def create_project(body: ProjectCreate, s: Services = Depends(services)):
    if not str(body.name or "").strip():
        raise HTTPException(422, "A project needs a name")
    p = s.projects.create(str(body.name or ""), str(body.industry if body.industry is not None else "general"), str(body.goal or ""))
    return s.with_records(p)


@router.get("/api/projects/{project_id}", response_model=Project)
async def get_project(project_id: str, s: Services = Depends(services)):
    return s.with_records(s.project(project_id))


def approver() -> str:
    """Who approves: the signed-in person's name (package 10.2), or "owner" on a machine without accounts."""
    from ...accounts.principal import current

    who = current()
    return who.who if who is not None and who.user_id else "owner"


@router.patch("/api/projects/{project_id}", response_model=Project)
async def patch_project(project_id: str, payload: ProjectPatch, s: Services = Depends(services)):
    projects = s.projects
    p = s.project(project_id)
    body = payload.model_dump(exclude_unset=True)
    if isinstance(body.get("policy"), dict) or {"share_for_training", "consent"} & set(body.get("settings") or {}):
        from ...accounts.guard import demand

        demand("admin", "change a gate switch, the training opt-in or consent")  # the owner's (package 10.2, 19.10)
    for key in ("name", "goal"):
        if key in body:
            p[key] = str(body[key]).strip()[:4000 if key == "goal" else 120]
    if body.get("industry") in studio_data.INDUSTRIES if hasattr(studio_data, "INDUSTRIES") else False:
        p["industry"] = body["industry"]
    opted = None
    if isinstance(body.get("settings"), dict):
        settings = body["settings"]
        before = dict(p["settings"])
        if "max_rows" in settings:
            p["settings"]["max_rows"] = max(200, min(int(settings["max_rows"]), 200000))
        if "quick" in settings:
            p["settings"]["quick"] = bool(settings["quick"])
        if "folds" in settings:
            p["settings"]["folds"] = max(studio_engine.FOLD_CAPS[0], min(int(settings["folds"]), studio_engine.FOLD_CAPS[1]))
        from ... import consent as consents

        if isinstance(settings.get("share_for_training"), bool) and settings["share_for_training"] != bool(p["settings"].get("share_for_training")):
            p["settings"]["share_for_training"] = settings["share_for_training"]  # A6.1: only the owner's choice puts a project's runs in the export
            opted = {"share_for_training": settings["share_for_training"], "actor": "human"}
            if isinstance((p["settings"].get("consent") or {}).get("train"), bool):  # one switch, two names: both move
                p["settings"]["consent"] = {**p["settings"]["consent"], "train": settings["share_for_training"]}
        consented = {}
        if isinstance(settings.get("consent"), dict):  # 19.10: use to improve DCLab, use to train its models; each off by default
            now = consents.project_consent(p)
            own = dict(p["settings"].get("consent") or {})
            for use in ("improve", "train"):
                value = settings["consent"].get(use)
                if isinstance(value, bool) and (value != now[use] or not isinstance(own.get(use), bool)):
                    own[use] = value
                    if value != now[use]:
                        consented[use] = value
            if "train" in own:  # the earlier opt-in says the same as the train switch
                p["settings"]["share_for_training"] = own["train"]
            p["settings"]["consent"] = own
        from ...experience import hooks  # 19.3: how it will run, as the person set it (the opt-in is a policy, recorded as one)

        hooks.settings_changed({k: p["settings"].get(k) for k in ("max_rows", "quick", "folds") if before.get(k) != p["settings"].get(k)},
                               project=project_id)
    changed = {}
    if isinstance(body.get("policy"), dict):  # which gates wait for a person; the invariants hold either way
        named = {studio_graph.LEGACY_POLICY.get(k, k): v for k, v in body["policy"].items()}
        before = {**studio_graph.DEFAULT_POLICY, **(p.get("policy") or {})}
        p["policy"] = {**(p.get("policy") or {}), **{k: bool(v) for k, v in named.items() if k in studio_graph.DEFAULT_POLICY}}
        changed = {k: {"from": before.get(k), "to": v} for k, v in p["policy"].items() if before.get(k) != v}
    projects.save(p)
    if changed:
        projects.log(project_id, "policy_changed", {"changes": changed, "actor": "human"})
        for key, change in changed.items():
            audit.record(s.audit, "policy", "human", project=p, move="set_policy", status="allowed",
                         args={"policy": key, "label": audit.POLICY_NAMES.get(key, key), "on": bool(change["to"]), "was": change["from"]})
    for use, on in (consented if isinstance(body.get("settings"), dict) else {}).items():
        projects.log(project_id, "consent_changed", {"use": use, "on": on, "actor": "human"})
        audit.record(s.audit, "policy", "human", project=p, move="set_consent", status="allowed",
                     args={"policy": f"consent_{use}", "label": audit.POLICY_NAMES[f"consent_{use}"], "on": on})
    if opted:
        projects.log(project_id, "training_opt_in_changed", opted)
        audit.record(s.audit, "policy", "human", project=p, move="set_training_opt_in", status="allowed",
                     args={"policy": "share_for_training", "label": audit.POLICY_NAMES["share_for_training"], "on": opted["share_for_training"]})
    return s.with_records(projects.get(project_id))


@router.delete("/api/projects/{project_id}", status_code=204, response_class=Response)
async def delete_project(project_id: str, s: Services = Depends(services)):
    projects = s.projects
    project = s.project(project_id)
    if project_id in s.jobs and not s.jobs[project_id].done():
        raise HTTPException(409, "Stop the running stage first")
    from ...storage.files import files_for

    try:
        files = files_for(projects)
        prefix = files.key_of(projects.data_dir(project_id).parent)
    except Exception:  # noqa: BLE001 — a project is always deletable; its file records then stay (they point at nothing)
        files = prefix = None
    data = project.get("data") or {}
    projects.delete(project_id)
    forgotten: list[str] = []
    if files is not None:
        try:
            forgotten = files.forget(prefix)  # the records of its files go with it
        except Exception:  # noqa: BLE001
            pass
    from ...experience import retention  # 19.10: and everything it left in the session record and the decision ledger

    gone = await asyncio.to_thread(retention.forget, project=project_id, log=s.experience, ledger=s.ledger, usage=getattr(s.gateway, "usage", None))
    # the project and its files are gone; who deleted it, and which table it held, stay (package 12.5): its tombstone
    audit.record(s.audit, "deletion", "human", project={"id": project_id, "name": project.get("name")}, move="delete_project", status="allowed",
                 filename=data.get("filename"), sha256=data.get("sha256"), files=len(forgotten), **gone)
    return Response(status_code=204)


@router.put("/api/projects/{project_id}/data", response_model=Project, openapi_extra=OCTET)
async def upload_data(project_id: str, request: Request, filename: str = "data.csv", s: Services = Depends(services)):
    """The table as the request's raw bytes (CSV, TSV, Excel, Parquet, JSON); ``filename`` names it."""
    p = s.project(project_id)
    body = await request.body()
    if len(body) > s.upload_max_bytes:
        raise HTTPException(413, "Files above 200 MB are not supported in the local notebook")
    if not body:
        raise HTTPException(400, "Empty file")
    await asyncio.to_thread(charge_upload, len(body))  # the day's allowance (package 10.6)
    name = studio_data.safe_name(filename)  # never a path: "../x" or "a/b" cannot leave the data folder
    if not name:
        raise HTTPException(422, "Give the file a name")
    (s.projects.data_dir(project_id) / name).write_bytes(body)
    return s.with_records(await asyncio.to_thread(s.attach_data, p, name))  # the profile and the card: never on the event loop


@router.put("/api/projects/{project_id}/tables/{name}", response_model=Project, openapi_extra=OCTET)
async def upload_table(project_id: str, name: str, request: Request, filename: str = "", s: Services = Depends(services)):
    """One more table beside the main one (16.15 C: the orders and tickets of a table of customers), as raw bytes, under
    ``name``; the solution names it there. The stages are cleared: a machine may read it."""
    p = s.project(project_id)
    body = await request.body()
    if len(body) > s.upload_max_bytes:
        raise HTTPException(413, "Files above 200 MB are not supported in the local notebook")
    if not body:
        raise HTTPException(400, "Empty file")
    file = studio_data.safe_name(filename or f"{name}.csv")
    if not file:
        raise HTTPException(422, "Give the file a name")
    if file in studio_data.files_in_use(p, but=name):  # written before it is checked: never over the main table, the labels or another table
        raise HTTPException(409, f"{file} is already one of the project's files: give the table another file name")
    await asyncio.to_thread(charge_upload, len(body))
    staged = f"{file}.part"  # checked before it replaces anything: a refused or broken upload never touches an attached table
    (s.projects.data_dir(project_id) / staged).write_bytes(body)
    try:
        await asyncio.to_thread(studio_data.attach_table, s.projects, project_id, file, name, "human", staged)
    except studio_data.DataError as exc:
        raise HTTPException(400, str(exc))
    return s.with_records(s.projects.get(project_id))


@router.post("/api/projects/{project_id}/data/sample", response_model=Project)
async def use_sample(project_id: str, body: SampleChoice, s: Services = Depends(services)):
    s.project(project_id)
    try:
        await asyncio.to_thread(studio_data.use_sample, s.projects, project_id, body.key)
    except KeyError:
        raise HTTPException(404, "Unknown sample dataset")
    except studio_data.DataError as exc:
        raise HTTPException(400, str(exc))
    return s.with_records(s.projects.get(project_id))


async def _proposal(project_id: str, body: ProposalRequest, s: Services) -> dict[str, Any]:
    p = s.project(project_id)
    frame = s.load_frame(p)
    try:
        proposal = await asyncio.to_thread(studio_solution.propose, frame, p["data"]["profile"], str(body.target if body.target is not None else ""), body.task,
                                               studio_engine.pack_of(p))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    proposal["rule"] = {k: proposal.get(k) for k in ("identifiers", "forbidden", "time_candidates", "group_candidates")}  # 19.7: the rules' own
    if p.get("suggestion"):  # a studied sample carries the R&D's whole solution; a draft-built project only what the chat established
        known = {f["column"] for f in proposal["forbidden"]}
        proposal["forbidden"] = [{**f, "proof": ["DCLAB-R01"]} for f in p["suggestion"].get("forbidden") or [] if f["column"] not in known] + proposal["forbidden"]
        proposal["identifiers"] = sorted(set(proposal["identifiers"]) | set(p["suggestion"].get("identifiers") or []))
        if p["suggestion"].get("prediction_moment"):
            proposal["prediction_moment_hint"] = p["suggestion"]["prediction_moment"]
        if p["suggestion"].get("time_column"):
            proposal["time_candidates"] = [p["suggestion"]["time_column"]] + [c for c in proposal["time_candidates"] if c != p["suggestion"]["time_column"]]
    s.projects.update(project_id, proposal=proposal)  # only the proposal: the project read above is seconds old (notes, approvals)
    return proposal


@router.post("/api/projects/{project_id}/solution/proposal", response_model=Proposal)
async def solution_proposal(project_id: str, body: ProposalRequest, s: Services = Depends(services)):
    return await _proposal(project_id, body, s)


@router.post("/api/projects/{project_id}/contract/proposal", response_model=Proposal, include_in_schema=False)  # old name, kept for one release
async def contract_proposal(project_id: str, body: ProposalRequest, s: Services = Depends(services)):
    return await _proposal(project_id, body, s)


async def _save_solution(project_id: str, payload: SolutionBody, s: Services) -> dict[str, Any]:
    projects = s.projects
    p = s.project(project_id)
    body = payload.model_dump(exclude_unset=True)
    try:
        solution = studio_solution.Solution(**body)
        solution.check_columns(p["data"]["columns"] if p.get("data") else [])
    except Exception as exc:  # noqa: BLE001 — the solution's own wording, as one line
        raise HTTPException(422, str(exc).split("\n")[0][:400] if "validation error" not in str(exc) else "; ".join(
            line.strip() for line in str(exc).split("\n")[1:] if line.strip() and not line.strip().startswith("For further"))[:600])
    changed = p.get("solution") != solution.model_dump()
    before = p.get("solution")
    if changed:
        verdict = s.validate(p, "set_solution", target=solution.target)
        studio_graph.log(projects, project_id, verdict, p, outcome="done: solution saved")
    p["solution"] = solution.model_dump()
    if changed:
        studio_memory.solution_saved(p, before, p["solution"], "human")  # A5.2: a correction when it changes the agent's
    projects.save(p)
    if changed:
        audit.solution(projects, p, p["solution"], "human")
        from ...ledger import capture  # 19.7: the person's choices against the proposal, as decisions; a studied sample's truth as labels

        await asyncio.to_thread(capture.solution_saved, p.get("proposal"), (p.get("data") or {}).get("profile"), p["solution"], "human",
                                p.get("suggestion"), project=project_id)
        projects.clear_stages(project_id)
        projects.log(project_id, "solution_saved", {"target": solution.target, "task": solution.task, "forbidden": [f.column for f in solution.forbidden]})
    return s.with_records(projects.get(project_id))


@router.put("/api/projects/{project_id}/solution", response_model=Project)
async def save_solution(project_id: str, body: SolutionBody, s: Services = Depends(services)):
    return await _save_solution(project_id, body, s)


@router.put("/api/projects/{project_id}/contract", response_model=Project, include_in_schema=False)  # old name, kept for one release
async def save_contract(project_id: str, body: SolutionBody, s: Services = Depends(services)):
    return await _save_solution(project_id, body, s)


@router.post("/api/projects/{project_id}/stages/{stage}/run", response_model=Project)
async def run_stage(project_id: str, stage: str, wait: bool = False, reuse_reason: str = "", s: Services = Depends(services)):
    p = s.project(project_id)
    if stage not in studio_engine.STAGE_BY_KEY:
        raise HTTPException(404, "Unknown stage")
    if not p.get("solution"):
        raise HTTPException(409, "Save the solution first")
    s.validate(p, "run_stage", stage=stage, reuse_reason=reuse_reason)
    if (p["stages"].get(stage) or {}).get("status") in ("completed", "approved"):  # 19.7: before the job marks it queued
        from ...ledger import capture

        capture.revisit(stage, "human", project_id, int(p.get("holdout_uses") or 0))
    job = s.start_job(project_id, [stage], wait, reuse_reason or None)
    if wait:
        await s.run_here(job)
    return s.with_records(s.projects.get(project_id))


@router.post("/api/projects/{project_id}/run", response_model=Project)
async def run_all(project_id: str, start: str = "data", wait: bool = False, s: Services = Depends(services)):
    p = s.project(project_id)
    if not p.get("solution"):
        raise HTTPException(409, "Save the solution first")
    if start not in studio_engine.STAGE_BY_KEY:
        raise HTTPException(404, "Unknown stage")
    keys = list(studio_engine.STAGE_BY_KEY)
    s.validate(p, "run_stage", stage=start)
    job = s.start_job(project_id, keys[keys.index(start):], wait)
    if wait:
        await s.run_here(job)
    return s.with_records(s.projects.get(project_id))


@router.post("/api/projects/{project_id}/stages/{stage}/approve", response_model=Project)
async def approve_stage(project_id: str, stage: str, body: StageApproval | None = Body(None), s: Services = Depends(services)):
    """Approve a completed stage; ``choice`` overrides the rule's option (a body is optional)."""
    projects = s.projects
    s.project(project_id)
    try:
        studio_engine.approve(projects, project_id, stage, body.choice if body is not None else None)
    except (ValueError, KeyError) as exc:
        raise HTTPException(409, str(exc))
    if stage == "final":  # A5.3: a finished project proposes what it taught, for a reviewer
        try:
            await asyncio.to_thread(s.propose_lessons, project_id)
        except Exception as exc:  # noqa: BLE001 — the approval is saved; POST /lessons proposes them again later
            projects.log(project_id, "lessons_not_proposed", {"error": type(exc).__name__})
    return s.with_records(projects.get(project_id))


@router.post("/api/projects/{project_id}/lessons", response_model=Lessons)
async def project_lessons(project_id: str, s: Services = Depends(services)):
    """Propose the lessons of a finished project (once per final stage record; a second call returns the same ones)."""
    p = s.project(project_id)
    if (p["stages"].get("final") or {}).get("status") != "approved":
        raise HTTPException(409, "Lessons are proposed after a person approves the final stage")
    return {"lessons": await asyncio.to_thread(s.propose_lessons, project_id), "label": workspace_lessons.LABEL}


@router.get("/api/projects/{project_id}/graph", response_model=Graph)
async def project_graph(project_id: str, s: Services = Depends(services)):
    p = s.project(project_id)
    return {**studio_graph.describe(p), "transitions": s.projects.transitions(project_id)}


@router.get("/api/projects/{project_id}/lineage", response_model=Page)
async def project_lineage(project_id: str, path: str | None = None, s: Services = Depends(services)):
    """Where the project's numbers come from (19.5): each stage's run (its inputs, parameters, outputs and code), or,
    with ``path`` (a report's record path such as ``final.evidence.holdout_metrics.roc_auc``), the value there and
    the run that wrote it."""
    s.project(project_id)
    if path is None:
        return {"runs": await asyncio.to_thread(studio_lineage.runs, s.projects, project_id)}
    if len(path) > 300:
        raise HTTPException(422, "path is a record path of at most 300 characters")
    try:
        return await asyncio.to_thread(studio_lineage.trace, s.projects, project_id, path)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0]) if exc.args else "No such path") from None


@router.post("/api/projects/{project_id}/graph/check", response_model=Verdict)
async def check_move(project_id: str, payload: MoveCheck, s: Services = Depends(services)):
    p = s.project(project_id)
    body = payload.model_dump(exclude_unset=True)
    move = str(body.pop("move", ""))
    actor = "agent" if body.pop("actor", "human") == "agent" else "human"
    args = {k: body[k] for k in ("stage", "choice", "gate", "reuse_reason") if k in body}
    return studio_graph.check(p, move, actor, **args).to_dict()


@router.delete("/api/projects/{project_id}/memory/{note_id}", response_model=Project)
async def remove_memory_note(project_id: str, note_id: str, body: Any = Body(None), s: Services = Depends(services)):
    """A person removes a note from the project's memory (A5.2): ``{"reason": "..."}``. It stays in the project,
    marked removed, and the activity log keeps who removed it and why."""
    s.project(project_id)
    if isinstance(body, (bytes, str)) and body:
        try:
            body = json.loads(body)
        except ValueError:
            raise HTTPException(422, "The body must be JSON: {\"reason\": \"...\"}") from None
    reason = body.get("reason", "") if isinstance(body, dict) else ""
    try:
        studio_memory.remove(s.projects, project_id, note_id, "human", str(reason or ""))
    except KeyError:
        raise HTTPException(404, "No such note") from None
    audit.record(s.audit, "memory_removed", "human", project=s.projects.get(project_id), move="remove_memory_note", status="allowed",
                 args={"note": note_id}, message=str(reason or "")[:300])
    return s.with_records(s.projects.get(project_id))


@router.post("/api/projects/{project_id}/review", response_model=Project)
async def confirm_review(project_id: str, payload: AnyObject, s: Services = Depends(services)):
    """{"confirmed": [keys], "rejected": [keys]}: alerts of an anomaly project's review list a person reviewed (20.7). The
    confirmed join the project's known cases, kept across rounds, and become its labels (the problem is then a rare-event
    one); the rejected are kept as reviewed, never as labels. A round the label check refuses changes nothing."""
    from ...kinds.anomaly import confirm

    project = s.project(project_id)
    body = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})
    keys = [str(k)[:200] for k in (body.get("confirmed") or [])][:5000]
    rejected = [str(k)[:200] for k in (body.get("rejected") or [])][:5000]
    solution = project.get("solution") or {}
    key = ((solution.get("kind_fields") or {}).get("key_column") if solution.get("kind") == "anomaly" else None) \
        or (project.get("review") or {}).get("key_column")  # a later round: the first one made the cases its labels
    if not key:
        raise HTTPException(409, "Only an anomaly project with a key column takes reviewed alerts")
    if not keys:
        raise HTTPException(422, "Name the confirmed alerts by their key")
    try:
        await asyncio.to_thread(confirm, s.projects, project_id, keys, key, rejected)
    except (studio_data.DataError, ValueError) as exc:
        raise HTTPException(409, str(exc)) from None
    audit.record(s.audit, "move", "human", None, project_id=project_id, move="confirm_review", status="allowed",
                 args={"confirmed": len(keys)})
    return s.with_records(s.projects.get(project_id))


@router.post("/api/projects/{project_id}/approvals", response_model=Project)
async def approve_gate(project_id: str, body: GateApproval, s: Services = Depends(services)):
    s.project(project_id)
    try:
        studio_graph.approve_gate(s.projects, project_id, str(body.gate if body.gate is not None else ""), approver(),
                                  str(body.reason if body.reason is not None else ""))
    except studio_graph.GraphBlocked as exc:
        raise HTTPException(409, str(exc))
    return s.with_records(s.projects.get(project_id))


@router.post("/api/projects/{project_id}/ask", response_model=Answer)
async def ask_agent(project_id: str, body: Question, s: Services = Depends(services)):
    p = s.project(project_id)
    question = str(body.question if body.question is not None else "").strip()
    if not question:
        raise HTTPException(422, "Ask something")
    records = s.projects.records(project_id)
    task_type = next((r["task_type"] for r in records.values()), None)
    answer = await asyncio.to_thread(studio_agent.answer, question[:1000], p, records, task_type)
    s.projects.log(project_id, "question", {"question": question[:300], "proof": answer["proof"]})
    s.projects.log(project_id, "answer", {"answer": str(answer.get("answer") or "")[:4000], "proof": answer["proof"]})  # 19.3: kept, not dropped
    return answer


@router.get("/api/projects/{project_id}/export/notebook", response_class=Response, responses=download("application/x-ipynb+json"))
async def export_notebook(project_id: str, s: Services = Depends(services)):
    p = s.project(project_id)
    if not p.get("solution"):
        raise HTTPException(409, "Nothing to export yet")
    try:
        nb = studio_export.notebook(p, s.projects.records(project_id))
    except studio_export.ExportUnsupported as exc:
        raise HTTPException(409, str(exc)) from None
    studio_graph.capture(s.projects, project_id, "human")
    text = studio_export.dumps_notebook(nb)
    return Response(text, media_type="application/x-ipynb+json", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.ipynb"'})


@router.get("/api/projects/{project_id}/review", response_model=Review)
async def review_project_notebook(project_id: str, s: Services = Depends(services)):
    """The notebook copilot's review of the project's exported notebook, in the demo's review shape."""
    s.project(project_id)

    def review():
        report = s.project_review(project_id)
        keys = ("detector", "severity", "cell", "line", "title", "message", "suggestion", "rules")
        return {"summary": report["summary"], "cells": [{"type": c["type"], "source": c["source"]} for c in report["cells"]],
                "findings": [{**{k: f.get(k) for k in keys}, "proof": [r["record_id"] for r in f.get("proof", [])]} for f in report["findings"]],
                "limitations": report.get("limitations", []), "fixes_available": s.gateway.available("notebook_review")}
    return await asyncio.to_thread(review)


@router.post("/api/projects/{project_id}/review/fixes", response_model=Fixes)
async def review_project_fixes(project_id: str, s: Services = Depends(services)):
    """A model's two-sentence fix per finding (package A3.4), by finding number. The findings are the deterministic
    review's, unchanged; a fix that does not cite its rule and pitfall record, or states a number its records do not,
    is dropped. Asked for by the person, since each call is a model request."""
    client = s.gateway.client("notebook_review", project_id=project_id)
    if client is None:
        raise HTTPException(409, "No model is configured for notebook fixes")
    if project_id in s.fixing:
        raise HTTPException(409, "Fixes for this notebook are already being written")
    from ...copilot.fixes import key_of, write_fixes

    def run():
        findings = s.project_review(project_id)["findings"]
        # keyed by the finding (detector:cell:line), not its position: the page matches them to the findings it shows
        return {"fixes": {key_of(findings[n]): fix for n, fix in write_fixes(findings, client).items()}, "findings": len(findings)}
    s.fixing.add(project_id)
    try:
        return await asyncio.to_thread(run)
    finally:
        s.fixing.discard(project_id)


@router.get("/api/projects/{project_id}/export/report", response_class=Response, responses=download("text/markdown"))
async def export_report(project_id: str, s: Services = Depends(services)):
    p = s.project(project_id)
    return Response(studio_export.report(p, s.projects.records(project_id)), media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.md"'})


@router.get("/api/projects/{project_id}/export/solution-report", response_class=Response, responses=download("text/markdown"))
async def export_solution_report(project_id: str, s: Services = Depends(services)):
    """The solution report of 17.7 (every number traced to a record), with the reporter's summaries (18.7) when that role
    is on (18.9) and a model serves it; never for a private project."""
    from dclab_core import report as core_report

    from ... import roles
    from ...studio import solution_report

    import hashlib

    p = s.project(project_id)
    if not s.projects.has_stage(project_id, "final"):
        raise HTTPException(409, "Run the final stage first: the report reads the finished records")
    digest = hashlib.sha256(json.dumps(s.projects.records(project_id), sort_keys=True, default=str).encode()).hexdigest()[:16]
    cached = p.get("solution_report") or {}  # the same records: the same report, no new request
    if cached.get("digest") == digest and cached.get("markdown"):
        text = cached["markdown"]
    else:
        advisor = roles.advisor("reporter", project=p, cap_eur=roles.CALL_CAP_EUR)
        built = await asyncio.to_thread(solution_report.build, s.projects, project_id, None, True, advisor)
        text = core_report.to_markdown(built)
        s.projects.update(project_id, solution_report={"digest": digest, "markdown": text})
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="dclab-solution-report-{p["name"][:40].replace(" ", "_")}.md"'})


@router.get("/api/projects/{project_id}/export/sft", response_class=Response, responses=download("application/x-ndjson"))
async def export_sft(project_id: str, s: Services = Depends(services)):
    p = s.project(project_id)
    rows = studio_sft.examples_from_project(p, s.projects.records(project_id))
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return Response(text, media_type="application/x-ndjson", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.sft.jsonl"'})
