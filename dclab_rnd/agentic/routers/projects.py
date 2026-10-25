"""The DCLab notebook's projects: create, change, delete; data and the solution; stages, gates and the workflow graph;
memory; questions; exports; the copilot's review of the exported notebook."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import Body, Depends, HTTPException, Request
from fastapi.responses import Response

from ... import audit, lessons as workspace_lessons
from ...studio import agent as studio_agent, data as studio_data, engine as studio_engine, export as studio_export
from ...studio import graph as studio_graph, lineage as studio_lineage, memory as studio_memory, solution as studio_solution
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
    from ...consent import inherit

    p = inherit(p, s.projects, s.store.home)  # decision 6: a workspace whose projects inherit its consent
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
    legacy_moved = None  # 23.1: the earlier opt-in moved the train switch (its own audit row; the record and the writers learn it below)
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

        was_train = consents.project_consent(p)["train"]  # 23.1: the legacy opt-in may move the train switch without naming it
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
        if consents.project_consent(p)["train"] != was_train and "train" not in consented:
            legacy_moved = consents.project_consent(p)["train"]
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
        from dclab_core import consent as core

        from ... import consent as consents, experience

        consents.invalidate(project_id)  # 23.1: the next record of this project is stamped with the new level
        if not on:
            from ...extensions import optional

            datasets = optional("datasets")  # the training sets are the full edition's
            if datasets is not None:
                datasets.mark_stale(s.store.home)  # 23.10: what was built from this project under the use now off is removed by the next sweep
        experience.record("consent.changed", "person", {"scope": "project", "use": use, "on": on, "was": not on,
                                                        "level": core.level_of(consents.workspace_consent(s.store.home), consents.project_consent(p)),
                                                        "notice_version": consents.NOTICE_VERSION}, project=project_id)
    if legacy_moved is not None:
        from dclab_core import consent as core

        from ... import consent as consents, experience

        consents.invalidate(project_id)
        if not legacy_moved:
            from ... import datasets

            datasets.mark_stale(s.store.home)
        experience.record("consent.changed", "person", {"scope": "project", "use": "train", "on": legacy_moved, "was": not legacy_moved,
                                                        "level": core.level_of(consents.workspace_consent(s.store.home), consents.project_consent(p)),
                                                        "notice_version": consents.NOTICE_VERSION}, project=project_id)
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
    s.kernels.restart(project_id)  # its kernel stops; its folder, inside the project's, goes with it
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
                                               studio_engine.pack_of(p), None, **studio_solution.card_context(p))  # S9: the kind its card understood
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    proposal["rule"] = {k: proposal.get(k) for k in ("identifiers", "forbidden", "time_candidates", "group_candidates", "modifiers", "modifier_facts")}  # 19.7: the rules' own
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
        from ...kinds.binary import as_table_label, profile_booleans

        if body.get("positive_label") is not None and p.get("data"):  # 20.10: "yes" on a column cleaned into booleans is True
            body["positive_label"] = as_table_label(body["positive_label"], profile_booleans(p["data"].get("profile"), body.get("target")))
        solution = studio_solution.Solution(**body)
        label_target = (p.get("labels") or {}).get("target")  # 20.10: a label file's target is a column of the joined table (as the intern's)
        solution.check_columns((list(p["data"]["columns"]) + ([label_target] if label_target else [])) if p.get("data") else [])
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


@router.get("/api/projects/{project_id}/steps", response_model=Page)
async def project_steps(project_id: str, stage: str | None = None, decision: str | None = None, s: Services = Depends(services)):
    """23.8: the steps inside a stage's latest run (``stage``), or the steps that carry the numbers a decision was made on
    (``decision``, a decision record's id): each with its parameters, shapes, fold, metric and value, time and the digest of what
    it produced. A replay (``POST .../pipeline/rerun`` or ``make replay``) reports the first step whose digest differs."""
    s.project(project_id)
    if (stage is None) == (decision is None):
        raise HTTPException(422, "name a stage or a decision")
    try:
        if decision is not None:
            return await asyncio.to_thread(studio_lineage.decision_steps, s.projects, getattr(s, "ledger", None), project_id, decision)
        return await asyncio.to_thread(studio_lineage.stage_steps, s.projects, project_id, stage)
    except KeyError as exc:
        raise HTTPException(404, str(exc.args[0]) if exc.args else "Not found") from None


@router.get("/api/projects/{project_id}/decisions", response_model=Page)
async def project_decisions(project_id: str, s: Services = Depends(services)):
    """22.8: for each column of the project's table (the first 40), what DCLab would decide on its identifier, time,
    role and leak questions, from its bands, with the rule's and the classifiers' answers and any disagreement."""
    from ... import decide
    from ...ledger import column_state

    p = s.project(project_id)
    data = p.get("data") or {}
    profile = (data.get("profile") or {}).get("columns") or []
    rows = data.get("rows")
    doc = s.gateway.routing.get() if getattr(s.gateway, "routing", None) is not None else {}
    target = (p.get("solution") or {}).get("target")

    def answers() -> list[dict[str, Any]]:
        latest: dict[tuple[str, str], Any] = {}  # the project's own recorded decision per point and column: the rule beside the models
        for r in (s.ledger.records(project=project_id) if getattr(s, "ledger", None) is not None else []):
            key = (r.point, r.subject.key)
            if r.point in decide.TECHNICAL and (key not in latest or r.at >= latest[key].at):
                latest[key] = r
        out = []
        for column in profile[:40]:
            if column.get("name") == target:
                continue
            state = {**column_state(column, rows), **({"target": str(target)[:200]} if target else {})}
            out.append({"column": column.get("name"), "decisions": {point: decide.ask(point, state, routing_doc=doc, recorded=latest.get((point, str(column.get("name")))))
                                                                    for point in ("column.is_identifier", "column.time", "column.semantic_role", "feature.leakage_suspect")}})
        return out

    return {"columns": await asyncio.to_thread(answers)}


def _person() -> str:
    from ...accounts.principal import current

    who = current()
    return f"person:{who.user_id}" if who is not None and who.user_id else "person"


@router.get("/api/projects/{project_id}/disagreements", response_model=Page)
async def project_disagreements(project_id: str, s: Services = Depends(services)):
    """29.10: the project's decisions where a point's student disagrees with the rule and this person has not answered:
    the question, the rule's answer and the student's, the options. Answer one with POST .../disagreements/{record_id}."""
    from ...ledger import notes

    s.project(project_id)  # 404 when the project is not there
    return await asyncio.to_thread(notes.pending, getattr(s, "ledger", None), project_id, _person())


@router.post("/api/projects/{project_id}/disagreements/{record_id}", response_model=Page)
async def project_disagreement_answer(project_id: str, record_id: str, body: Any = Body(None), s: Services = Depends(services)):
    """29.10: which is right, {"value": an option}: a label given having seen both answers, never a change of what was
    decided. It may send a routed student back to shadow (where it says the student was wrong), never promote one."""
    from ...ledger import notes, routing, shadow

    s.project(project_id)
    value = (body or {}).get("value") if isinstance(body, dict) else None
    if value is None or not isinstance(value, (bool, str)):
        raise HTTPException(422, "value is one of the decision's options")
    book = getattr(s, "ledger", None)
    try:
        out = await asyncio.to_thread(notes.answer, book, project_id, record_id, value, _person())
    except KeyError:
        raise HTTPException(404, "No such decision in this project") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    demoted: list = []
    try:  # as a blind label does (22.6): a routed student this label says was wrong may go back to shadow
        routed = [pt for pt, e in (s.gateway.routing.get().get("decisions") or {}).items() if e.get("level")]
        if out.point in routed:
            demoted = await asyncio.to_thread(lambda: routing.demote(s.gateway.routing, s.audit, *shadow.labelled(book, s.store.home, point=out.point)))
    except Exception:  # noqa: BLE001 - the label is kept; a failed check runs again with the next label
        demoted = []
    return {"label": out.model_dump(mode="json"), "demoted": demoted}


@router.get("/api/projects/{project_id}/pipeline", response_model=Page)
async def project_pipeline(project_id: str, path: str | None = None, s: Services = Depends(services)):
    """The project's pipeline of workflows (20.8): every step behind its numbers, from the problem card to the report,
    with what each read and made, its run, its status (stale when what it read has changed) and where to resume; with
    ``path`` (a report's record path), the steps that produced that number."""
    from dclab_core import pipeline as core_pipeline

    from ...studio import pipeline as studio_pipeline

    s.project(project_id)
    pipe = await asyncio.to_thread(studio_pipeline.of, s.projects, project_id)
    out = {**pipe.model_dump(mode="json"), **core_pipeline.summary(pipe), "nodes": core_pipeline.by_node(pipe)}
    if path is not None:
        if len(path) > 300:
            raise HTTPException(422, "path is a record path of at most 300 characters")
        out["produced"] = core_pipeline.produced(pipe, path)
    return out


@router.post("/api/projects/{project_id}/pipeline/rerun", response_model=Page)
async def rerun_pipeline_step(project_id: str, step: str, wait: bool = True, s: Services = Depends(services)):
    """Rerun one step from its record (20.8), as a job (the limits on jobs at once apply, and Stop): a stage is replayed
    in a scratch copy and compared with its record, the report rebuilt and compared with what its runs made; the
    project is never changed. ``wait``: run it now and answer with its result; else the job, whose result lands in
    the project's activity (``pipeline_rerun``)."""
    from ...studio import pipeline as studio_pipeline

    s.project(project_id)
    pipe = await asyncio.to_thread(studio_pipeline.of, s.projects, project_id)
    try:
        pipe.step(step)
    except KeyError:
        raise HTTPException(404, f"No step {step[:60]!r} in this project's pipeline") from None
    job = s.submit("rerun", f"{project_id}:{step}", {"project_id": project_id, "step": step}, here=wait)
    if not wait:
        return {"job": job}
    done = await s.run_here(job)
    result = next((e["payload"] for e in reversed(s.projects.activity(project_id, limit=200))
                   if e.get("kind") == "pipeline_rerun" and (e.get("payload") or {}).get("job") == job["id"]), None)
    if done.get("status") == "failed" or result is None:
        raise HTTPException(500, done.get("error") or "The rerun ended without a result")
    return {**result, "job": {"id": job["id"], "status": done.get("status")}}


@router.post("/api/projects/{project_id}/improve", response_model=Page)
async def improve_project(project_id: str, s: Services = Depends(services)):
    """Try ways to improve the model, as a job (Stop, the limits on jobs at once): other settings, other families, other features and a
    blend, each cross-validated on the training folds only, the same folds as the current model. The held-back cases are never touched.
    The report is the project's ``improvements`` record and a message in its Home chat."""
    project = s.project(project_id)
    if not all(s.projects.read_stage(project_id, k) for k in ("features", "models", "final")):
        raise HTTPException(409, "Run the project's stages up to the final one first: there is no model to improve yet.")
    if project.get("running"):
        raise HTTPException(409, "Something is running in this project: wait for it, or stop it")
    job = s.submit("improve", f"{project_id}:improve", {"project_id": project_id}, here=False)
    return {"job": job}


@router.get("/api/projects/{project_id}/improvements", response_model=Page)
async def get_improvements(project_id: str, s: Services = Depends(services)):
    """The last improvement run's record (the ways, their paired gains on the folds, what cleared), or ``{}``."""
    return s.project(project_id).get("improvements") or {}


@router.post("/api/projects/{project_id}/data-value", response_model=Page)
async def measure_data_value(project_id: str, wait: bool = True, s: Services = Depends(services)):
    """Would more rows help (34.13)? The learning curve of the chosen model on the training folds only (25 to 100 % of each fold's fit
    rows, cut the way the split is made), the last step's gain with its 90 % interval, the answer (``likely_help``, ``unlikely_help`` or
    ``cannot_tell``) with its assumption, and the same per segment where errors concentrate. A job (Stop, the limits on jobs at once);
    the held-back cases are never read and no stage record changes. ``wait``: answer with the record; else the job."""
    from ...studio import data_value as studio_data_value

    project = s.project(project_id)
    if not all(s.projects.read_stage(project_id, k) for k in ("features", "models")):
        raise HTTPException(409, "Run the project's stages up to the models stage first: the curve is drawn for the model it chose.")
    if project.get("running"):
        raise HTTPException(409, "Something is running in this project: wait for it, or stop it")
    job = s.submit("data_value", f"{project_id}:data_value", {"project_id": project_id}, here=wait)
    if not wait:
        return {"job": job}
    done = await s.run_here(job)
    if done.get("status") != "done":
        raise HTTPException(409, done.get("error") or "The measurement ended without a result")
    return {**studio_data_value.view(s.project(project_id)), "job": {"id": job["id"], "status": done.get("status")}}


@router.get("/api/projects/{project_id}/data-value", response_model=Page)
async def get_data_value(project_id: str, s: Services = Depends(services)):
    """The last measurement of whether more rows would help (34.13), or ``{}``."""
    from ...studio import data_value as studio_data_value

    return studio_data_value.view(s.project(project_id))


@router.get("/api/projects/{project_id}/decision", response_model=Page)
async def get_decision(project_id: str, s: Services = Depends(services)):
    """The project's decision on top of its model (34.4): ``status`` (none, pending, chosen, refused, missing), ``stated`` (the solution's
    decision), ``chosen`` (the final stage's ``evidence.decision``: the operating point with its interval, curve, holdout read and
    sensitivity table, or a review budget, a price, a cost matrix), ``sentence`` (its claim) and ``what_if`` (whether one can be asked,
    the numbers it may change, or why not). Numbers from the stage record only."""
    from ...studio import decision as studio_decision

    s.project(project_id)
    return studio_decision.view(s.projects, project_id)


@router.post("/api/projects/{project_id}/decision/what-if", response_model=Page)
async def decision_what_if(project_id: str, payload: AnyObject, s: Services = Depends(services)):
    """The chosen act decision re-scored under ``changes`` (an object of the decision's numbers: ``value_per_case``,
    ``cost_per_action``, ``cost_false_positive``, ``cost_false_negative``, ``capacity_share``, or ``base_rate``), on the out-of-fold
    rows it was chosen on: ``before``, ``after``, ``keep_the_old_threshold``, ``regret_if_unchanged`` and what was not refitted. The
    model, the stage records and the holdout are untouched. 409 when no what-if can be asked (and why), 422 for a change it does not state."""
    from ...experience import hooks
    from ...studio import decision as studio_decision

    s.project(project_id)
    body = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})
    changes = body.get("changes") if isinstance(body.get("changes"), dict) else {}
    try:
        out = await asyncio.to_thread(studio_decision.what_if, s.projects, project_id, changes)
    except LookupError as exc:
        raise HTTPException(409, str(exc.args[0]) if exc.args else "No what-if can be asked") from None
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    hooks.decision_what_if(project_id, changes, out, "human")
    return out


@router.post("/api/projects/{project_id}/retrain", response_model=Page)
async def retrain_project(project_id: str, payload: AnyObject, wait: bool = True, s: Services = Depends(services)):
    """Champion and challenger (34.11): ``newer`` names newer labelled rows kept beside the project (``PUT .../newer-rows/{name}``:
    the project's columns and its outcome). The challenger is the champion's recipe, family and settings fitted on the training rows and the newer rows before
    the latest ``policy.window_share`` of them; both are compared there, on rows neither trained on, with a paired bootstrap;
    promoted only when better beyond the 90% interval and ``policy.margin``. ``live_report``: the service's ``GET /monitor`` answer,
    whose alerts say why (and become a lesson, 34.12). ``policy``: ``window_share``, ``min_new_labelled``, ``margin``. A job
    (Stop, the limits on jobs at once); the project's holdout is never read. ``wait``: answer with the comparison; else the job."""
    from dclab_core.retrain import RetrainPolicy

    from ...studio import retrain as studio_retrain

    project = s.project(project_id)
    body = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})
    newer = str(body.get("newer") or "")
    if newer not in (project.get("newer_rows") or {}):
        raise HTTPException(409, "Add the newer labelled rows first (PUT /api/projects/{id}/newer-rows/{name}), then name them")
    if (project.get("stages", {}).get("final") or {}).get("status") not in ("completed", "approved"):
        raise HTTPException(409, "Run the final stage first: there is no champion to compare with")
    if project.get("running"):
        raise HTTPException(409, "Something is running in this project: wait for it, or stop it")
    try:
        policy = RetrainPolicy(**{k: v for k, v in (body.get("policy") or {}).items() if k in ("window_share", "min_new_labelled", "margin")})
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from None
    live_report = body.get("live_report") if isinstance(body.get("live_report"), dict) else None
    if live_report is not None:
        from dclab_core.live import LiveReport

        try:
            LiveReport.model_validate(live_report)
        except ValueError as exc:
            raise HTTPException(422, f"live_report is not a monitor's answer (GET /monitor): {str(exc)[:300]}") from None
    job = s.submit("retrain", f"{project_id}:retrain", {"project_id": project_id, "newer": newer, "policy": policy.model_dump(mode="json"),
                                                        "live_report": live_report}, here=wait)
    if not wait:
        return {"job": job}
    done = await s.run_here(job)
    if done.get("status") != "done":
        raise HTTPException(409, done.get("error") or "The comparison ended without a result")
    return {**studio_retrain.view(s.projects, project_id), "job": {"id": job["id"], "status": done.get("status")}}


@router.put("/api/projects/{project_id}/newer-rows/{name}", response_model=Page, openapi_extra=OCTET)
async def upload_newer_rows(project_id: str, name: str, request: Request, filename: str = "", s: Services = Depends(services)):
    """Newer labelled rows (34.11), as raw bytes, kept beside the project under ``name`` for a champion/challenger comparison:
    the project's columns and its outcome. Never read by the engine: the stages and the solution stay, and the holdout is not
    read again."""
    from ...studio import retrain as studio_retrain

    s.project(project_id)
    body = await request.body()
    if len(body) > s.upload_max_bytes:
        raise HTTPException(413, "Files above 200 MB are not supported in the local notebook")
    if not body:
        raise HTTPException(400, "Empty file")
    await asyncio.to_thread(charge_upload, len(body))
    try:
        added = await asyncio.to_thread(studio_retrain.add_newer_rows, s.projects, project_id, body, filename, name, "human")
    except studio_retrain.RetrainRefused as exc:
        raise HTTPException(400, str(exc)) from None
    return {"newer_rows": added}


@router.get("/api/projects/{project_id}/retrain", response_model=Page)
async def get_retrains(project_id: str, s: Services = Depends(services)):
    """The champion/challenger comparisons kept (34.11), newest last, and the fit the last promotion prefers (never a production
    approval: serving it is the host's step)."""
    from ...studio import retrain as studio_retrain

    s.project(project_id)
    return studio_retrain.view(s.projects, project_id)


@router.post("/api/projects/{project_id}/collect-next", response_model=Page)
async def collect_next(project_id: str, wait: bool = True, s: Services = Depends(services)):
    """What to collect next (34.14): which column groups carry the score (each made constant in turn on the training folds, with the
    model the models stage chose; the holdout is never read), the domain's usual kinds of data the table lacks (not measured: worth
    asking for), where more labels would help (from the last learning curve, 34.13) and the open data questions. A job (Stop, the
    limits on jobs at once). ``wait``: answer with the record; else the job."""
    from ...studio import collect as studio_collect

    project = s.project(project_id)
    if not all(s.projects.read_stage(project_id, k) for k in ("features", "models")):
        raise HTTPException(409, "Run the project's stages up to the models stage first: the groups are weighed with the model it chose.")
    if project.get("running"):
        raise HTTPException(409, "Something is running in this project: wait for it, or stop it")
    job = s.submit("collect_next", f"{project_id}:collect_next", {"project_id": project_id}, here=wait)
    if not wait:
        return {"job": job}
    done = await s.run_here(job)
    if done.get("status") != "done":
        raise HTTPException(409, done.get("error") or "The weighing ended without a result")
    return {**studio_collect.view(s.project(project_id)), "job": {"id": job["id"], "status": done.get("status")}}


@router.get("/api/projects/{project_id}/collect-next", response_model=Page)
async def get_collect_next(project_id: str, s: Services = Depends(services)):
    """The last advice on what to collect next (34.14), or ``{}``."""
    from ...studio import collect as studio_collect

    return studio_collect.view(s.project(project_id))


@router.post("/api/projects/{project_id}/improve/adopt", response_model=Page)
async def adopt_improvement(project_id: str, payload: AnyObject, s: Services = Depends(services)):
    """Adopt a way that cleared: its parameters are kept for the next final run. That run reads the held-back cases a second time and needs
    its written reason (the final stage's usual rule); nothing is re-run here."""
    from ...studio import improve as studio_improve

    s.project(project_id)
    candidate = str((payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})).get("candidate") or "")
    try:
        adopted = await asyncio.to_thread(studio_improve.adopt_candidate, s.projects, project_id, candidate)
    except KeyError as exc:
        raise HTTPException(404, exc.args[0]) from None
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from None
    return {**adopted, "next": "Run the final stage again with a written reason: it scores the held-back cases a second time, so that estimate is no longer a fresh one."}


@router.get("/api/projects/{project_id}/programme", response_model=Page)
async def get_programme(project_id: str, s: Services = Depends(services)):
    """The programme of a problem with several parts (24.5): its state, what blocks it, a section per leaf, the composition and the holdout read once. ``available``
    is false for a project built from one problem."""
    from ...studio import programme as studio_programme

    return await asyncio.to_thread(lambda: studio_programme.view(s.project(project_id)))


@router.post("/api/projects/{project_id}/programme/run", response_model=Page)
async def run_programme(project_id: str, body: Any = Body(None), wait: bool = True, s: Services = Depends(services)):
    """Run the programme as a job (the limits on jobs at once apply, and Stop): the leaves, the composition chosen on validation and, when no leaf is blocked, the one
    read of the holdout. The graph's move ``run_programme`` is checked first: a second read of the holdout needs ``{"reuse_reason": "..."}``. ``wait``: answer with the
    programme; else the job."""
    from ...studio import programme as studio_programme

    project = s.project(project_id)
    reason = str(body.get("reuse_reason") or "").strip() if isinstance(body, dict) else ""
    verdict = studio_graph.check(project, "run_programme", "human", reuse_reason=reason)
    if not verdict.allowed:
        studio_graph.log(s.projects, project_id, verdict, project)
        raise HTTPException(409, verdict.message)
    job = s.submit("programme", project_id, {"project_id": project_id, "actor": "human", "reuse_reason": reason or None}, here=wait)
    if not wait:
        return {"job": job}
    done = await s.run_here(job)
    if done.get("status") == "failed":
        raise HTTPException(409, done.get("error") or "The programme ended without a result")
    return {**(await asyncio.to_thread(lambda: studio_programme.view(s.project(project_id)))), "job": {"id": job["id"], "status": done.get("status")}}


@router.post("/api/projects/{project_id}/twin", response_model=Page)
async def run_twin(project_id: str, body: Any = Body(None), s: Services = Depends(services)):
    """25.6: a twin run of this project as a job: the current system and Claude, as the AI-first expert, on the same table in
    two lanes (each a folder of its own; nothing of the project changes), compared step by step. ``{"blind": true}`` (the
    default) keeps the known solution from both lanes. Needs the owner's switch on and Claude reachable; capped at the
    workspace's session budget."""
    import secrets
    from datetime import datetime, timezone

    from ...extensions import optional

    twin_home = optional("twin.home")
    if twin_home is None:
        raise HTTPException(404, "The twin lane is not part of this edition")
    project = s.project(project_id)
    if not twin_home.settings(s.store.home)["on"]:
        raise HTTPException(409, "Claude beside the system is off in this workspace (the owner's switch, Admin)")
    if not project.get("data"):
        raise HTTPException(409, "Bring the data first: a twin run compares the two on the project's table")
    body = body if isinstance(body, dict) else {}
    from ... import limits
    from ...accounts.principal import current

    cap = float(twin_home.settings(s.store.home)["session_eur"])
    who = current()
    if limits.on() and who is not None and who.user_id:  # 25.10: never past what is left of the person's own month
        from ...models.usage import month_start

        mine = limits.user_monthly(who.user_id, s.store.home)
        if mine is not None:
            left = max(0.0, mine - float(s.gateway.usage.spent(month_start(), user_id=who.user_id)))
            if left <= 0:
                raise HTTPException(409, "Your monthly model budget is used up: the owner can raise your limit (Admin, budgets)")
            cap = min(cap, left)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
    job = s.submit("twin", f"{project_id}:twin", {"project_id": project_id, "run_id": run_id, "blind": body.get("blind", True) is not False,
                                                  "cap_eur": round(cap, 4)})
    return {"job": job, "run_id": run_id}


@router.get("/api/projects/{project_id}/twin", response_model=Page)
async def project_twin_runs(project_id: str, s: Services = Depends(services)):
    """25.6: this project's twin runs, newest first: each lane's status, the differences, the steps judged, and the comparison and the
    expert's story as text (COMPARE.md, DESIGN.md)."""
    import json as _json

    from ...extensions import optional

    twin = optional("twin")
    s.project(project_id)
    if twin is None:  # the open edition has no twin runs
        return {"runs": []}

    def read() -> list[dict[str, Any]]:
        out = []
        for r in reversed(twin.list_runs(s.store.home)):
            if r.get("seed") != f"project:{project_id}":
                continue
            folder = Path(r["path"])
            compare = {}
            try:
                compare = _json.loads((folder / "compare.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                pass
            text = lambda p: p.read_text(encoding="utf-8")[:60_000] if p.is_file() else None  # noqa: E731
            out.append({**{k: r.get(k) for k in ("run_id", "model", "started", "finished", "status", "differences")},
                        "holdout": compare.get("holdout"), "judged": compare.get("judged"), "expert_judged": compare.get("expert_judged"),
                        "compare_md": text(folder / "COMPARE.md"), "design_md": text(folder / "ai_first" / "design" / "DESIGN.md")})
        return out[:20]

    return {"runs": await asyncio.to_thread(read)}


@router.post("/api/projects/{project_id}/programme/drop", response_model=Page)
async def drop_programme_leaf(project_id: str, body: Any = Body(None), s: Services = Depends(services)):
    """A person drops a leaf that cannot run, with the reason: ``{"block": "blk3", "reason": "..."}``. The composition goes on without it, and the programme says so."""
    from ...studio import programme as studio_programme

    s.project(project_id)
    if not isinstance(body, dict):
        raise HTTPException(422, 'Send {"block": ..., "reason": ...}')
    try:
        return await asyncio.to_thread(studio_programme.drop, s.projects, project_id, str(body.get("block") or ""), str(body.get("reason") or ""), "human")
    except studio_programme.ProgrammeError as exc:
        raise HTTPException(409, str(exc)) from None


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


@router.post("/api/projects/{project_id}/outcome", response_model=Page)
async def record_outcome(project_id: str, payload: AnyObject, request: Request, s: Services = Depends(services)):
    """23.5: a business number observed after the model ran ({"metric", "value", "reference"?, "higher_is_better"?, "note"?}),
    written as the outcome of every decision of the project; judged only when both a reference and a direction (true or
    false) are given, otherwise recorded with the verdict unknown and the holdout's verdict stays current. Owner and ML
    engineer (the default write permission)."""
    from ...accounts.principal import current
    from ...ledger import outcomes

    import math
    import re

    project = s.project(project_id)
    if ((project.get("stages") or {}).get("final") or {}).get("status") not in ("completed", "approved"):
        raise HTTPException(409, "A business number follows the model: run the final stage first")
    body = payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})
    metric = str(body.get("metric") or "").strip()
    if not re.fullmatch(r"[A-Za-z0-9_.:%/-]{1,80}", metric):
        raise HTTPException(422, "metric is the number's name as one word, like retained_customers")

    def number(name: str, required: bool) -> float | None:
        raw = body.get(name)
        if raw in (None, "") and not required:
            return None
        try:
            found = float(raw)
        except (TypeError, ValueError):
            raise HTTPException(422, f"{name} is a number") from None
        if not math.isfinite(found):
            raise HTTPException(422, f"{name} is a finite number")
        return found

    value, reference = number("value", True), number("reference", False)
    higher = body.get("higher_is_better")
    if higher is not None and not isinstance(higher, bool):
        raise HTTPException(422, "higher_is_better is true or false")
    who = current()
    out = await asyncio.to_thread(outcomes.observe, project_id, metric=metric, value=value, observed="business",
                                  by=getattr(who, "role", None) or "person", reference=reference,
                                  higher_is_better=higher, note=str(body.get("note") or "")[:300] or None)
    return {"project_id": project_id, **out}


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


@router.get("/api/projects/{project_id}/brief", response_model=Page)
async def project_brief(project_id: str, s: Services = Depends(services)):
    """30.14: the project's Solution Brief (the stored one, or a draft by rules), what G3 still asks before it can be
    signed (a success bar, a baseline, an alternative) and whether it is signed; signing is the solution gate."""
    from ...studio import brief as studio_brief

    p = s.project(project_id)
    try:
        return await asyncio.to_thread(studio_brief.view, s.projects, p)
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.put("/api/projects/{project_id}/brief", response_model=Page)
async def save_project_brief(project_id: str, body: AnyObject, s: Services = Depends(services)):
    """30.14: a person's Solution Brief saved (the graph's set_brief move: never while a stage runs, never after the holdout
    read); an edit after signing unsigns it, and the solution gate signs it."""
    from pydantic import ValidationError

    from ...studio import brief as studio_brief

    s.project(project_id)
    doc = body.model_dump() if hasattr(body, "model_dump") else dict(body or {})
    try:
        return await asyncio.to_thread(studio_brief.save, s.projects, project_id, doc, approver())
    except studio_graph.GraphBlocked as exc:
        raise HTTPException(409, str(exc))
    except ValidationError as exc:
        raise HTTPException(422, "The brief is not well formed: " + "; ".join(e.get("msg", "") for e in exc.errors()[:3]))
    except ValueError as exc:
        raise HTTPException(409, str(exc))


@router.get("/api/projects/{project_id}/workflow-run", response_model=Page)
async def project_workflow_run(project_id: str, s: Services = Depends(services)):
    """30.15: the framework's twelve steps read from the project's records (each with its check, its status and where its
    evidence is; a step the records do not show is missing) and G4, honest and usable, with the phase a failure sends the
    work back to."""
    from ...studio import workflow_run as studio_workflow_run

    p = s.project(project_id)
    return await asyncio.to_thread(studio_workflow_run.view, s.projects, p)


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


@router.get("/api/projects/{project_id}/export/code", response_class=Response, responses=download("application/zip"))
async def export_code(project_id: str, check: bool = False, s: Services = Depends(services)):
    """The project's pipeline as code (20.9): a zip with ``pipeline.py`` (one block per step), ``pipeline.ipynb``, the
    data it reads, dclab_core, the pinned requirements and ``expected.json``; ``python pipeline.py`` computes the
    holdout number again and exits 1 when it differs from the record. ``check`` (30.15): run it first, in an isolated
    Python, and record on the project whether it reproduced the number (the Workflow Run's package step)."""
    import io
    import tempfile
    import zipfile
    from pathlib import Path

    from ...studio import code as studio_code

    p = s.project(project_id)

    def build() -> bytes:
        with tempfile.TemporaryDirectory(prefix="dclab-code-") as tmp:
            folder = Path(tmp) / "code"
            studio_code.export(s.projects, project_id, folder, check=check)  # 30.15: ?check=true reruns it first and records whether it matched
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(folder).as_posix())
            return buffer.getvalue()

    try:
        data = await asyncio.to_thread(build)
    except (studio_code.CodeRefused, studio_data.DataError, ValueError) as exc:  # refused, or the data changed since the run
        raise HTTPException(409, str(exc)) from None
    from urllib.parse import quote

    name = p["name"][:40].replace(" ", "_")
    ascii_name = "".join(ch for ch in name if ch.isascii() and (ch.isalnum() or ch in "-_")) or "project"
    return Response(data, media_type="application/zip", headers={  # an ASCII name, and the real one for browsers that read it
        "Content-Disposition": f'attachment; filename="dclab-{ascii_name}-code.zip"; filename*=UTF-8\'\'{quote(f"dclab-{name}-code.zip")}'})


@router.get("/api/projects/{project_id}/review", response_model=Review)
async def review_project_notebook(project_id: str, s: Services = Depends(services)):
    """The notebook copilot's review of the project's exported notebook, in the demo's review shape."""
    s.project(project_id)

    def review():
        report = s.project_review(project_id)
        keys = ("detector", "severity", "cell", "line", "title", "message", "suggestion", "rules")
        exported = studio_export.notebook(s.project(project_id), s.projects.records(project_id))["cells"]
        roles = [((c.get("metadata") or {}).get("dclab") or {}).get("role") for c in exported] if len(exported) == len(report["cells"]) else []
        return {"summary": report["summary"], "cells": [{"type": c["type"], "source": c["source"], **({"role": roles[i]} if roles and roles[i] else {})}
                                                        for i, c in enumerate(report["cells"])],
                "agent_cells": [c for c in s.project(project_id).get("notebook_agent_cells") or [] if c.get("status") in ("proposed", "accepted")],
                "writes_cells": s.gateway.available("notebook_cell"),
                "findings": [{**{k: f.get(k) for k in keys}, "proof": [r["record_id"] for r in f.get("proof", [])]} for f in report["findings"]],
                "limitations": report.get("limitations", []), "fixes_available": s.gateway.available("notebook_review")}
    return await asyncio.to_thread(review)


@router.post("/api/projects/{project_id}/notebook/run", response_model=Page)
async def run_notebook_cell(project_id: str, payload: AnyObject, s: Services = Depends(services)):
    """Run one code cell of the project's notebook in its kernel (the owner, 2026-10-20): the cell the server generates for
    that index, never code sent by the page; the kernel has the training rows only, and the holdout cell is refused (409).
    The outputs come back as text, pictures (PNG) and errors; every run is kept in the workspace's code runs."""
    from ...accounts.principal import current
    from ...studio.kernels import CellRefused

    project = s.project(project_id)
    if not project.get("solution"):
        raise HTTPException(409, "Save the solution first: the notebook is generated from it")
    raw = (payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})).get("cell")
    if isinstance(raw, str) and raw.startswith("a"):
        cell: Any = raw  # an agent's cell the person accepted
    else:
        try:
            cell = int(raw)
        except (TypeError, ValueError):
            raise HTTPException(422, "cell is the index of a code cell, or the id of a cell you accepted") from None
    who = current()

    def run():
        return s.kernels.run(s.projects, project, s.projects.records(project_id), cell, by=getattr(who, "user_id", None))
    try:
        return await asyncio.to_thread(run)
    except (CellRefused, studio_export.ExportUnsupported) as exc:
        raise HTTPException(409, str(exc)) from None
    except studio_data.DataError as exc:
        raise HTTPException(409, f"The project's table cannot be read: {exc}") from None


@router.get("/api/projects/{project_id}/evaluation", response_model=Page)
async def project_evaluation(project_id: str, s: Services = Depends(services)):
    """The Evaluation page: the holdout's measures, confusion counts and curves, and the validation folds' (a project whose
    final stage ran before they were kept gets what its numbers give back, said so). Read from the final stage's record: the
    holdout is never read again, and no row of a table is in it."""
    from dclab_core import model_evaluation as me

    s.project(project_id)
    final = s.projects.read_stage(project_id, "final")
    if not final:
        raise HTTPException(409, "The final stage has not run yet: the evaluation is read from its record, and the holdout is read once, there.")
    evidence = final.get("evidence") or {}
    holdout = me.from_record(evidence)
    kept = evidence.get("evaluation") or {}
    return {"holdout": holdout.model_dump(mode="json") if holdout else None, "validation": kept.get("validation"),
            "model": evidence.get("model"), "primary_metric": final.get("primary_metric"), "primary_metric_ci": evidence.get("holdout_primary_metric_ci"),
            "cv_selected_metric_mean": evidence.get("cv_selected_metric_mean"), "holdout_split": evidence.get("holdout_split"),
            "holdout_rows": evidence.get("holdout_rows"), "train_rows": evidence.get("train_rows"), "baselines": evidence.get("baselines"),
            "success_bar": evidence.get("success_bar"), "production_approved": False}


@router.post("/api/projects/{project_id}/notebook/write", response_model=Page)
async def write_notebook_cell(project_id: str, payload: AnyObject, s: Services = Depends(services)):
    """The next cell for the person's request, written by a model (``notebook_cell``) from the notebook's code and the
    columns, never a row; checked (it parses, it never touches X_test or y_test) and kept as a proposal to accept or reject."""
    from ...studio import notebook_agent

    project = s.project(project_id)
    if not project.get("solution"):
        raise HTTPException(409, "Save the solution first: the notebook is generated from it")
    request = str((payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})).get("request") or "").strip()
    if not request:
        raise HTTPException(422, "Say what the cell should do")
    client = s.gateway.client("notebook_cell", project_id=project_id)
    if client is None:
        raise HTTPException(409, "No model is configured to write cells")
    try:
        cell = await asyncio.to_thread(notebook_agent.write, project, s.projects.records(project_id), request, client)
    except studio_export.ExportUnsupported as exc:
        raise HTTPException(409, str(exc)) from None
    except Exception as exc:  # noqa: BLE001 - a model that failed or was refused (a cap): said, nothing kept
        raise HTTPException(502, f"The model could not write the cell ({type(exc).__name__})") from None
    def add() -> None:  # the list as it is now (another request may have changed it while the model wrote), under the project's lock
        with notebook_agent.lock(project_id):
            now = s.project(project_id).get("notebook_agent_cells") or []
            s.projects.update(project_id, notebook_agent_cells=notebook_agent.keep(list(now) + [cell]))

    await asyncio.to_thread(add)
    return cell


@router.post("/api/projects/{project_id}/notebook/cells", response_model=Page)
async def add_own_cell(project_id: str, payload: AnyObject, s: Services = Depends(services)):
    """A cell the person wrote themselves (the owner, 2026-10-20: "an option of adding a cell to continue"): checked as an agent's
    cell is (it parses, it never touches X_test or y_test), kept on the project as an accepted cell, and run by its id like the
    others. The person's own words need no accepting: they wrote it."""
    from ...studio import notebook_agent

    project = s.project(project_id)
    if not project.get("solution"):
        raise HTTPException(409, "Save the solution first: the notebook is generated from it")
    code = str((payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})).get("code") or "").strip()[:notebook_agent.MAX_OWN_CODE]
    if not code:
        raise HTTPException(422, "Write some code first")
    found = notebook_agent.problems(code)
    if found:
        raise HTTPException(422, "; ".join(found))
    cell = notebook_agent.own_cell(code)

    def add() -> None:
        with notebook_agent.lock(project_id):
            now = s.project(project_id).get("notebook_agent_cells") or []
            s.projects.update(project_id, notebook_agent_cells=notebook_agent.keep(list(now) + [cell]))

    await asyncio.to_thread(add)
    return cell


@router.delete("/api/projects/{project_id}/notebook/cells/{cell_id}", status_code=204, response_class=Response)
async def delete_notebook_cell(project_id: str, cell_id: str, s: Services = Depends(services)):
    """Take a cell the agent wrote or the person added out of the notebook. Its runs stay recorded (the workspace's code runs)."""
    from ...studio import notebook_agent

    s.project(project_id)

    def drop() -> bool:
        with notebook_agent.lock(project_id):
            cells = list(s.project(project_id).get("notebook_agent_cells") or [])
            kept = [c for c in cells if c.get("id") != cell_id]
            if len(kept) == len(cells):
                return False
            s.projects.update(project_id, notebook_agent_cells=kept)
            return True

    if not await asyncio.to_thread(drop):
        raise HTTPException(404, "No such cell")
    return Response(status_code=204)


@router.post("/api/projects/{project_id}/notebook/cells/{cell_id}/answer", response_model=Page)
async def answer_notebook_cell(project_id: str, cell_id: str, payload: AnyObject, s: Services = Depends(services)):
    """The person accepts or rejects a proposed cell; only an accepted one runs. The answer is kept beside the proposal."""
    from ...accounts.principal import current
    from ...studio import notebook_agent

    accepted = bool((payload.model_dump() if hasattr(payload, "model_dump") else dict(payload or {})).get("accepted"))
    who = getattr(current(), "user_id", None)

    def settle() -> dict[str, Any]:
        with notebook_agent.lock(project_id):  # read, check and write under the lock: a write running meanwhile keeps this answer
            cells = [dict(c) for c in s.project(project_id).get("notebook_agent_cells") or []]
            found = next((c for c in cells if c.get("id") == cell_id), None)
            if found is None:
                raise HTTPException(404, "No such proposed cell")
            if found.get("status") != "proposed":
                raise HTTPException(409, f"This cell was already {found.get('status')}")
            answered = notebook_agent.answer(found, accepted, who, project_id)
            s.projects.update(project_id, notebook_agent_cells=[answered if c.get("id") == cell_id else c for c in cells])
            return answered

    return await asyncio.to_thread(settle)


@router.post("/api/projects/{project_id}/notebook/restart", response_model=Page)
async def restart_notebook_kernel(project_id: str, s: Services = Depends(services)):
    """Stop the project's kernel: the next cell starts a fresh one, with the training rows of the current split."""
    s.project(project_id)
    return await asyncio.to_thread(s.kernels.restart, project_id)


@router.get("/api/projects/{project_id}/notebook/kernel", response_model=Page)
async def notebook_kernel(project_id: str, s: Services = Depends(services)):
    """Whether the project's kernel runs, how many cells it ran, and how long it has been idle."""
    s.project(project_id)
    return s.kernels.status(project_id)


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

    from ...studio import solution_report

    p = s.project(project_id)
    from ...studio import programme as studio_programme

    programme = studio_programme.stored(p)
    if programme is not None and programme.holdout is not None and not s.projects.has_stage(project_id, "final"):  # 24.5: a problem of several parts reports its programme
        from dclab_core import programme as core_programme

        changed = studio_programme.stale_reasons(p)
        if changed:
            raise HTTPException(409, "The programme is stale (" + "; ".join(changed) + "): run it again before reporting it")
        report = core_programme.report(programme, f"{p['name']}: the programme")
        problems = core_report.check(report, core_programme.programme_context(programme))
        if problems:
            raise HTTPException(500, "The programme report has a number it cannot trace: " + "; ".join(problems[:2]))
        return Response(core_report.to_markdown(report), media_type="text/markdown; charset=utf-8",
                        headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}-programme.md"'})
    if not s.projects.has_stage(project_id, "final"):
        raise HTTPException(409, "Run the final stage first: the report reads the finished records")
    text = (await asyncio.to_thread(solution_report.latest, s.projects, project_id))["markdown"]  # the same records: the same report, no new request
    return Response(text, media_type="text/markdown; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="dclab-solution-report-{p["name"][:40].replace(" ", "_")}.md"'})


@router.get("/api/projects/{project_id}/export/sft", response_class=Response, responses=download("application/x-ndjson"))
async def export_sft(project_id: str, s: Services = Depends(services)):
    from ...extensions import optional

    studio_sft = optional("studio.sft")
    if studio_sft is None:
        raise HTTPException(404, "Training examples are not part of this edition")
    p = s.project(project_id)
    rows = studio_sft.examples_from_project(p, s.projects.records(project_id))
    text = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    return Response(text, media_type="application/x-ndjson", headers={"Content-Disposition": f'attachment; filename="dclab-{p["name"][:40].replace(" ", "_")}.sft.jsonl"'})
