"""HTTP routes for drafts, mounted by ``dclab_rnd.agentic.server.create_app``.

    GET    /api/packs                         the domain packs (cards on Home and in the solution draft)
    GET    /api/drafts                        recent drafts
    POST   /api/drafts {problem, pack?}       start a draft; the agent answers on the event stream
    GET    /api/drafts/{id}                   the draft
    DELETE /api/drafts/{id}
    POST   /api/drafts/{id}/messages {text}   a user turn
    POST   /api/drafts/{id}/pack {key}        the user picks a pack (wins over detection)
    PUT    /api/drafts/{id}/data?filename=    upload a file (raw body), processed in the background
    POST   /api/drafts/{id}/data/sample {key} a dataset the R&D already studied
    POST   /api/drafts/{id}/data/synthetic {prompt?, rows?, template?}  simulated rows, labelled synthetic
    GET    /api/synthetic/templates           the built-in templates, and whether a model designs the table instead
    GET    /api/connectors                    what each data connector can do on this server (no secrets)
    POST   /api/connectors/kaggle/search {query, page?}       Kaggle datasets
    POST   /api/drafts/{id}/data/kaggle {ref, file?}           a Kaggle dataset (largest table, or one file)
    POST   /api/drafts/{id}/data/hf {dataset, revision?, split?, config?}  a Hugging Face dataset split
    POST   /api/drafts/{id}/data/database {connection, table? | query?, limit?}  a read-only query on a server-defined connection
    POST   /api/drafts/{id}/data/cloud {uri}                  one s3:// or gs:// object
    GET    /api/drafts/{id}/events            server-sent events (resume with Last-Event-ID or ?after=)
    PATCH  /api/drafts/{id} {problem}         edit the problem sentence (wizard step 1)
    POST   /api/drafts/{id}/solution/proposal {target, task?}  the column audit on the cleaned table (step 3)
    PUT    /api/drafts/{id}/solution          accept the solution draft (validated like a project's solution)
    PUT    /api/drafts/{id}/settings          rows, folds, where it runs, budget (step 4); the split follows the solution
    POST   /api/drafts/{id}/build             turn the draft into a project; returns the project
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..agentic.api_models import OCTET, AnyObject, Doc, Draft, DraftCreate, Page, Project, Router  # module level: annotations resolve here

from .. import connectors
from . import pack as packs
from . import pipeline
from .chat import HomeAgent
from .store import DraftStore

MAX_UPLOAD = 200 * 1024 * 1024
STREAM_SECONDS = 600  # the browser's EventSource reconnects and resumes from Last-Event-ID


def register(app: FastAPI, drafts: DraftStore, projects, models, jobs: dict[str, asyncio.Task], traces: Any = None, services: Any = None) -> None:
    """``models`` is the model gateway (dclab_rnd.models.Gateway): each use names its purpose, so it is routed, timed and counted.
    ``traces`` (``agents.traces``) keeps a row per step of the Home agent's model turns, under the draft's id. ``jobs``
    holds the Home agent's turns (asyncio tasks in this process); the data pipelines and synthetic data are jobs in the
    job table (package 10.3), queued through ``services`` and run by a worker."""
    router = Router()  # package 10.1: the draft routes are one router; a body is a typed JSON object (a 422 otherwise)
    from ..studio import data as studio_data
    from .work import DraftWork

    work = services.draft_work if services is not None else DraftWork(drafts, models, traces)

    def dump(payload: Any) -> dict[str, Any]:
        return payload.model_dump(exclude_unset=True) if payload is not None else {}

    def get(draft_id: str) -> dict[str, Any]:
        try:
            return drafts.get(draft_id)
        except KeyError:
            raise HTTPException(404, "Draft not found") from None

    def agent(draft_id: str | None = None) -> HomeAgent:
        return work.agent(draft_id)

    def background(key: str, fn, *args) -> None:
        async def go():
            try:
                await asyncio.to_thread(fn, *args)
            finally:
                jobs.pop(key, None)
        task = asyncio.get_running_loop().create_task(go())
        jobs[key] = task

    def process(draft_id: str, asset_id: str) -> None:
        """The pipeline of one asset, as a job (in a task of this process when no job table is wired)."""
        if services is None:
            background(f"{draft_id}:{asset_id}", work.process, draft_id, asset_id)
            return
        services.submit("pipeline", f"{draft_id}:{asset_id}", {"draft_id": draft_id, "asset_id": asset_id})

    @router.get("/api/packs", response_model=list[Doc])
    async def list_packs():
        return packs.PACKS

    @router.get("/api/drafts", response_model=list[Doc])
    async def list_drafts():
        return [{k: d.get(k) for k in ("id", "problem", "status", "updated", "project_id", "pack")} | {"assets": len(d.get("assets", []))}
                for d in drafts.list()]

    @router.post("/api/drafts", status_code=201, response_model=Draft)
    async def create_draft(payload: DraftCreate):
        body = dump(payload)
        problem = str(body.get("problem", "")).strip()
        if len(problem) < 8:
            raise HTTPException(422, "Describe the problem in one sentence first")
        key = body.get("pack") if body.get("pack") in packs.KEYS else None
        draft = drafts.create(problem[:2000], key)
        background(draft["id"] + ":agent", lambda: agent(draft["id"]).start(draft["id"]))
        return draft

    def trace_of(draft_id: str) -> list[dict[str, Any]]:
        try:
            return traces.steps(draft_id) if traces is not None else []
        except Exception:  # noqa: BLE001 — a damaged trace never breaks the draft page
            return []

    @router.get("/api/drafts/{draft_id}", response_model=Draft)
    async def read_draft(draft_id: str):
        return {**get(draft_id), "trace": trace_of(draft_id)}

    @router.delete("/api/drafts/{draft_id}", status_code=204, response_class=Response)
    async def delete_draft(draft_id: str):
        get(draft_id)
        from ..storage.files import files_for
        try:
            files = files_for(drafts)
            prefix = files.key_of(drafts.data_dir(draft_id).parent)
        except Exception:  # noqa: BLE001 — a draft is always deletable
            files = prefix = None
        drafts.delete(draft_id)
        if files is not None:
            try:
                files.forget(prefix)  # the records of its files go with it (package 9.3)
            except Exception:  # noqa: BLE001
                pass
        try:
            if traces is not None:
                traces.delete(draft_id)
        except Exception:  # noqa: BLE001 — the draft is gone either way; a stray trace row is harmless
            pass

    @router.patch("/api/drafts/{draft_id}", response_model=Draft)
    async def edit_draft(draft_id: str, payload: AnyObject):
        get(draft_id)
        body = dump(payload)
        problem = str(body.get("problem", "")).strip()
        if problem:
            if len(problem) < 8:
                raise HTTPException(422, "Describe the problem in one sentence first")
            before = drafts.get(draft_id)["problem"]
            drafts.update(draft_id, lambda d: d.update(problem=problem[:2000]))
            if problem[:2000] != before:  # what was read from the old sentence is read again from the new one
                await asyncio.to_thread(HomeAgent(drafts, None).replan, draft_id)
        return drafts.get(draft_id)

    def clean_frame(draft: dict[str, Any]):
        import pandas as pd

        asset = next((a for a in draft["assets"] if a["id"] == draft.get("active_asset") and a.get("status") == "ready"), None)
        if asset is None:
            raise HTTPException(409, "Bring the data first: the solution draft is checked against the table")
        return pd.read_parquet(drafts.data_dir(draft["id"]) / asset.get("clean_file", "clean.parquet")), asset

    @router.post("/api/drafts/{draft_id}/solution/proposal", response_model=Doc)
    async def draft_proposal(draft_id: str, payload: AnyObject | None = Body(None)):
        from ..studio import solution as studio_solution

        draft = get(draft_id)
        body = dump(payload)
        frame, asset = clean_frame(draft)
        profile = (draft.get("analysis") or {}).get("profile") or studio_data.profile_table(frame)
        # the target the caller names, else what the chat established, the sample's own, or the profile's first candidate
        hinted = [(draft.get("understanding") or {}).get("target"), (asset.get("suggestion") or {}).get("target"), *(profile.get("target_candidates") or [])[:1]]
        target = str(body["target"]) if body.get("target") else next((str(t) for t in hinted if t and str(t) in frame.columns), "")
        if not target:
            raise HTTPException(422, "Name the target column")
        try:
            proposal = await asyncio.to_thread(studio_solution.propose, frame, profile, target, body.get("task"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        hint = asset.get("suggestion") or {}
        if hint:  # a studied sample: merge the solution the R&D wrote for it (same rule as a project's proposal)
            known = {f["column"] for f in proposal["forbidden"]}
            proposal["forbidden"] = [{**f, "proof": ["DCLAB-R01"]} for f in hint.get("forbidden") or [] if f["column"] not in known] + proposal["forbidden"]
            proposal["identifiers"] = sorted(set(proposal["identifiers"]) | set(hint.get("identifiers") or []))
            if hint.get("prediction_moment"):
                proposal["prediction_moment"] = hint["prediction_moment"]
            if hint.get("time_column"):
                proposal["time_candidates"] = [hint["time_column"]] + [c for c in proposal["time_candidates"] if c != hint["time_column"]]
        # the moment the wizard's sheet shows (the person may have edited it), else the chat's answer
        moment = str(body.get("prediction_moment") or "").strip() or (draft.get("understanding") or {}).get("prediction_moment")
        if moment:
            proposal["prediction_moment"] = moment
        # "prediction_moment" is a real moment (the chat's answer, or the studied sample's own); when it is missing the
        # page shows "prediction_moment_hint" as guidance only, so an instruction is never saved as the moment (DCLAB-R01)
        proposal.setdefault("prediction_moment", None)
        # The leakage reviewer (A3.2): the audit's flags stay ticked; what the moment or a model adds is shown, never applied.
        from ..studio import leakage_review
        proposal["review"] = await asyncio.to_thread(
            leakage_review.review, profile, proposal["prediction_moment"] or hint.get("prediction_moment"), proposal["forbidden"], target,
            proposal["identifiers"], models.client("leakage_review", draft_id=draft_id))
        drafts.update(draft_id, lambda d: d.update(proposal=proposal))
        return proposal

    @router.put("/api/drafts/{draft_id}/solution", response_model=Draft)
    async def draft_solution(draft_id: str, payload: AnyObject):
        from pydantic import ValidationError
        from ..studio import solution as studio_solution

        draft = get(draft_id)
        frame, _ = clean_frame(draft)
        try:
            solution = studio_solution.Solution(**dump(payload))
            solution.check_columns([str(c) for c in frame.columns])
        except ValidationError as exc:
            raise HTTPException(422, "; ".join(e["msg"].removeprefix("Value error, ") for e in exc.errors())) from None
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        value = solution.model_dump()

        def put(d):
            d["solution"] = value
            d.setdefault("understanding", {}).update(target=value["target"], prediction_moment=value["prediction_moment"])
            if d.get("settings"):
                d["settings"]["split"] = split_of(d)
        drafts.update(draft_id, put)
        HomeAgent(drafts, None).refresh_workflow(draft_id)
        return drafts.get(draft_id)

    @router.put("/api/drafts/{draft_id}/settings", response_model=Draft)
    async def draft_settings(draft_id: str, payload: AnyObject):
        draft = get(draft_id)
        body = dump(payload)
        budget = body.get("budget") or {}
        settings = {
            # the solution decides the split (engine.split_for), so the stored value is what will run, whatever was sent
            "split": split_of(draft),
            "quick": bool(body.get("quick", True)),
            "max_rows": max(200, min(int(body.get("max_rows") or 20000), 200000)),
            "folds": max(2, min(int(body.get("folds") or 3), 10)),
            "where": body.get("where") if body.get("where") in ("local", "sandbox", "gpu") else "local",
            "budget": {"max_steps": max(1, min(int(budget.get("calls") or 24), 80)), "max_minutes": max(1, min(int(budget.get("minutes") or 20), 240)),
                       "eur": max(0.0, min(float(budget.get("eur") or 5), 1000.0))},
            "ask_over_eur": bool(body.get("ask_over_eur", True)),
        }
        drafts.update(draft_id, lambda d: d.update(settings=settings))
        return drafts.get(draft_id)

    @router.post("/api/drafts/{draft_id}/messages", status_code=202, response_model=Doc)
    async def message(draft_id: str, payload: AnyObject):
        get(draft_id)
        text = str(dump(payload).get("text", "")).strip()
        if not text:
            raise HTTPException(422, "Write a message first")
        if (draft_id + ":agent") in jobs and not jobs[draft_id + ":agent"].done():
            raise HTTPException(409, "The agent is still answering")
        background(draft_id + ":agent", lambda: agent(draft_id).reply(draft_id, text))
        return {"accepted": True}

    @router.post("/api/drafts/{draft_id}/pack", response_model=Draft)
    async def choose_pack(draft_id: str, payload: AnyObject):
        get(draft_id)
        key = dump(payload).get("key")
        a = agent(draft_id)
        if key in (None, "", "auto"):
            det = packs.detect(drafts.get(draft_id)["problem"], drafts.get(draft_id).get("analysis"))
            a.set_pack(draft_id, det["key"], det["source"], det["why"])
        elif key in packs.KEYS:
            a.set_pack(draft_id, key, "user", "You chose this pack.")
        else:
            raise HTTPException(422, "Unknown pack")
        a.refresh_workflow(draft_id)
        return drafts.get(draft_id)

    @router.put("/api/drafts/{draft_id}/data", response_model=Doc, openapi_extra=OCTET)
    async def upload(draft_id: str, request: Request, filename: str):
        get(draft_id)
        name = studio_data.safe_name(filename)
        if not name:
            raise HTTPException(422, "Give the file a name")
        body = await request.body()
        if not body:
            raise HTTPException(422, "The file is empty")
        if len(body) > MAX_UPLOAD:
            raise HTTPException(413, "Files up to 200 MB; for larger tables connect the source instead")
        (drafts.data_dir(draft_id) / name).write_bytes(body)
        asset = pipeline.new_asset(drafts, draft_id, "upload", name, name, bytes=len(body))
        process(draft_id, asset["id"])
        return asset

    @router.post("/api/drafts/{draft_id}/data/sample", response_model=Doc)
    async def sample(draft_id: str, payload: AnyObject):
        get(draft_id)
        key = str(dump(payload).get("key", ""))
        if key not in {s["key"] for s in studio_data.sample_catalog()}:
            raise HTTPException(404, "Unknown sample dataset")
        frame, policy = await asyncio.to_thread(studio_data.load_sample, key)
        name = f"{key}.parquet"
        await asyncio.to_thread(pipeline._to_parquet, frame, drafts.data_dir(draft_id) / name)
        asset = pipeline.new_asset(drafts, draft_id, "sample", key, name, suggestion=_suggestion(policy))
        process(draft_id, asset["id"])
        return asset

    @router.post("/api/drafts/{draft_id}/data/synthetic", status_code=202, response_model=Doc)
    async def synthetic_data(draft_id: str, payload: AnyObject):
        get(draft_id)
        body = dump(payload)
        from . import synthetic

        rows = max(100, min(int(body.get("rows") or 5000), 200_000))
        template = body.get("template") or None
        if template is not None and template not in synthetic.TEMPLATES:
            raise HTTPException(422, "Unknown template")
        payload = {"draft_id": draft_id, "prompt": str(body.get("prompt", "")), "rows": rows, "template": template}
        if services is None:
            background(f"{draft_id}:synthetic", work.simulate, draft_id, payload["prompt"], rows, template)
        else:
            services.submit("synthetic", f"{draft_id}:synthetic", payload, busy="Synthetic data is already being generated for this draft")
        return {"accepted": True, "rows": rows, "template": template}

    @router.get("/api/synthetic/templates", response_model=Page)
    async def synthetic_templates():
        """The built-in synthetic tables, and whether a model is configured to design one from a description instead."""
        from . import synthetic

        return {"model": models.available("synthetic_schema"), "templates": synthetic.template_catalog()}

    # ------------------------------------------------------------------ data connectors (dclab_rnd/connectors)
    # Each import downloads into the draft's data folder in a worker thread, then registers the file as an
    # asset and processes it in the background exactly like an upload. Credentials live on the server only.
    def text_field(body: dict[str, Any], key: str, required: bool = False, limit: int = 300) -> str | None:
        value = body.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            if required:
                raise HTTPException(422, f"{key} is required")
            return None
        if not isinstance(value, str) or len(value) > limit:
            raise HTTPException(422, f"{key} must be text of at most {limit:,} characters")
        return value.strip()

    async def connect(draft_id: str, kind: str, name: str, fetch: Callable[[Path], dict[str, Any]]) -> dict[str, Any]:
        get(draft_id)
        directory = drafts.data_dir(draft_id)
        try:
            meta = await asyncio.to_thread(fetch, directory)
        except connectors.BadInput as exc:
            raise HTTPException(422, str(exc)) from None
        except connectors.ConnectorError as exc:
            raise HTTPException(400, str(exc)) from None
        size = (directory / meta["filename"]).stat().st_size
        asset = pipeline.new_asset(drafts, draft_id, kind, name, meta["filename"], source=meta["source"], bytes=size)
        process(draft_id, asset["id"])
        return asset

    @router.get("/api/connectors", response_model=Page)
    async def connectors_status():
        return await asyncio.to_thread(connectors.status)

    @router.post("/api/connectors/kaggle/search", response_model=list[Doc])
    async def kaggle_search(payload: AnyObject):
        from ..connectors import kaggle

        body = dump(payload)
        query = text_field(body, "query", required=True, limit=200)
        try:
            page = int(body.get("page") or 1)
        except (TypeError, ValueError):
            raise HTTPException(422, "page must be a number") from None
        try:
            return await asyncio.to_thread(kaggle.search, query, page)
        except connectors.BadInput as exc:
            raise HTTPException(422, str(exc)) from None
        except connectors.ConnectorError as exc:
            raise HTTPException(400, str(exc)) from None

    @router.post("/api/drafts/{draft_id}/data/kaggle", response_model=Doc)
    async def data_kaggle(draft_id: str, payload: AnyObject):
        from ..connectors import kaggle

        get(draft_id)
        body = dump(payload)
        ref = text_field(body, "ref", required=True, limit=200)
        file = text_field(body, "file", limit=255)
        if not kaggle.valid_ref(ref):
            raise HTTPException(422, "A Kaggle dataset is written owner/dataset-name")
        name = ref + (f" · {file}" if file else "")
        return await connect(draft_id, "kaggle", name, lambda directory: kaggle.download(ref, directory, file))

    @router.post("/api/drafts/{draft_id}/data/hf", response_model=Doc)
    async def data_hf(draft_id: str, payload: AnyObject):
        from ..connectors import hf

        get(draft_id)
        body = dump(payload)
        dataset = text_field(body, "dataset", required=True, limit=200)
        revision, config = text_field(body, "revision", limit=200), text_field(body, "config", limit=100)
        split = text_field(body, "split", limit=100) or "train"
        name = f"{dataset} · {config + ' · ' if config else ''}{split}"
        return await connect(draft_id, "hf", name, lambda directory: hf.fetch(dataset, directory, revision, split, config))

    @router.post("/api/drafts/{draft_id}/data/database", response_model=Doc)
    async def data_database(draft_id: str, payload: AnyObject):
        from ..connectors import db

        get(draft_id)
        body = dump(payload)
        connection = text_field(body, "connection", required=True, limit=60)
        table, query = text_field(body, "table", limit=200), text_field(body, "query", limit=db.MAX_QUERY)
        if (table is None) == (query is None):
            raise HTTPException(422, "Give either a table name or a query")
        try:
            limit = int(body.get("limit") or 200_000)
        except (TypeError, ValueError):
            raise HTTPException(422, "limit must be a number") from None
        name = f"{connection.lower()} · {table or 'query'}"
        return await connect(draft_id, "database", name, lambda directory: db.fetch(connection, directory, table, query, limit))

    @router.post("/api/drafts/{draft_id}/data/cloud", response_model=Doc)
    async def data_cloud(draft_id: str, payload: AnyObject):
        from ..connectors import cloud

        get(draft_id)
        body = dump(payload)
        uri = text_field(body, "uri", required=True, limit=1100)
        return await connect(draft_id, "cloud", uri, lambda directory: cloud.fetch(uri, directory))

    @router.get("/api/drafts/{draft_id}/events", response_class=StreamingResponse, responses={200: {"content": {"text/event-stream": {}}, "description": "The draft's events, as server-sent events"}})
    async def events(draft_id: str, request: Request, after: int = 0, wait: float = STREAM_SECONDS):
        get(draft_id)
        start = int(request.headers.get("last-event-id") or after or 0)
        limit = max(0.0, min(float(wait), STREAM_SECONDS))  # how long to keep the stream open (tests use a short one)

        async def stream():
            seq, idle, waited = start, 0.0, 0.0
            yield "retry: 1500\n\n"
            first = True
            while first or waited < limit:
                first = False
                if await request.is_disconnected():
                    break
                new = await asyncio.to_thread(drafts.events, draft_id, seq)
                for event in new:
                    seq = event["seq"]
                    yield f"id: {seq}\nevent: {event['kind']}\ndata: {json.dumps(event['data'], ensure_ascii=False, default=str)}\n\n"
                if new:
                    idle = 0.0
                else:
                    idle += 0.4
                    if idle >= 15:
                        yield ": keep-alive\n\n"
                        idle = 0.0
                await asyncio.sleep(0.4)
                waited += 0.4
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @router.post("/api/drafts/{draft_id}/build", status_code=201, response_model=Project)
    async def build(draft_id: str):
        draft = get(draft_id)
        if draft.get("project_id"):
            try:
                return projects.get(draft["project_id"])
            except KeyError:
                pass
        return await asyncio.to_thread(build_project, drafts, projects, draft_id)

    app.include_router(router)


def split_of(draft: dict[str, Any]) -> str:
    """The split the engine will use for this draft's solution (stratified until a solution names a time or group column)."""
    from ..studio import engine as studio_engine

    return studio_engine.split_for(draft.get("solution") or {})


def _suggestion(policy: dict[str, Any] | None) -> dict[str, Any] | None:
    if not policy:
        return None
    return {k: policy.get(k) for k in ("target", "task", "positive_label", "forbidden", "identifiers", "time_column", "prediction_moment") if k in policy}


def build_project(drafts: DraftStore, projects, draft_id: str) -> dict[str, Any]:
    """Create the project from the draft: its name and goal, the cleaned table, and what the chat established."""
    from ..studio import data as studio_data

    draft = drafts.get(draft_id)
    u = draft.get("understanding") or {}
    name = (u.get("target") or draft["problem"])[:60].strip().rstrip(".") or "New project"
    project = projects.create(name[:1].upper() + name[1:], "general", draft["problem"])
    pid = project["id"]
    asset = next((a for a in draft["assets"] if a["id"] == draft.get("active_asset") and a.get("status") == "ready"), None)
    if asset is not None:
        source = drafts.data_dir(draft_id) / asset.get("clean_file", "clean.parquet")
        filename = Path(asset["name"]).stem[:40] or "data"
        filename = studio_data.safe_name(filename + ".parquet")
        shutil.copyfile(source, projects.data_dir(pid) / filename)
        studio_data.attach_data(projects, pid, filename)
    project = projects.get(pid)
    if asset is not None and project.get("data"):
        project["data"]["synthetic"] = bool(asset.get("synthetic"))
        suggestion = dict(asset.get("suggestion") or {})
        target = u.get("target")
        if target and target in project["data"]["columns"]:
            suggestion["target"] = target
        if u.get("prediction_moment"):
            suggestion.setdefault("prediction_moment", u["prediction_moment"])
        project["suggestion"] = suggestion or None
    if draft.get("settings"):
        st = draft["settings"] = {**draft["settings"], "split": split_of(draft)}  # the copy kept with the project says what runs
        project["settings"] = {**project.get("settings", {}), "quick": st["quick"], "max_rows": st["max_rows"], "folds": st.get("folds", 3)}
        project["budget"] = st["budget"]
    project["draft"] = {"id": draft_id, "problem": draft["problem"], "pack": draft.get("pack"), "understanding": u, "settings": draft.get("settings"),
                        "workflow": draft.get("workflow"), "cleaning_log": draft.get("cleaning_log"),
                        "synthetic": bool(asset and asset.get("synthetic"))}
    projects.save(project)
    projects.log(pid, "built_from_draft", {"draft": draft_id, "rows": (project.get("data") or {}).get("rows")})
    if draft.get("solution") and project.get("data"):
        from ..studio import graph as studio_graph

        project = projects.get(pid)
        verdict = studio_graph.check(project, "set_solution", "human", target=draft["solution"]["target"])  # same check as the project page
        studio_graph.log(projects, pid, verdict, project, outcome="done: solution from the draft" if verdict.allowed else None)
        if verdict.allowed:
            from ..studio import memory as studio_memory

            project = projects.get(pid)
            project["solution"] = draft["solution"]
            studio_memory.solution_saved(project, None, draft["solution"], "human")  # A5.2: the person saved it in the wizard
            projects.save(project)

    def done(d):
        d["status"], d["project_id"] = "built", pid
    drafts.update(draft_id, done)
    drafts.emit(draft_id, "status", {"built": True, "project_id": pid})
    return projects.get(pid)
