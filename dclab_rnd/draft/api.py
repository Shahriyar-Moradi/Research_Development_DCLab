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
import re
import shutil
from pathlib import Path
from typing import Any, Callable

from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import Response, StreamingResponse

from ..agentic.api_models import OCTET, AnyObject, Doc, Draft, DraftCreate, Page, Project, Router  # module level: annotations resolve here

from .. import connectors
from . import pack as packs
from . import plan as plans
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

    fallback = None if services is not None else DraftWork(drafts, models, traces)

    def work_now() -> DraftWork:
        """The request's workspace's draft work (package 10.2: read per call, never kept from registration)."""
        return services.draft_work if services is not None else fallback

    def dump(payload: Any) -> dict[str, Any]:
        return payload.model_dump(exclude_unset=True) if payload is not None else {}

    def get(draft_id: str) -> dict[str, Any]:
        try:
            return drafts.get(draft_id)
        except KeyError:
            raise HTTPException(404, "Draft not found") from None

    def agent(draft_id: str | None = None) -> HomeAgent:
        return work_now().agent(draft_id)

    def drafts_exists(draft_id: str) -> bool:
        try:
            drafts.get(draft_id)
            return True
        except Exception:  # noqa: BLE001
            return False

    def background(key: str, fn, *args) -> None:
        async def go():
            try:
                await asyncio.to_thread(fn, *args)
            except (KeyError, FileNotFoundError):  # the draft was deleted while this ran: nothing left to do
                if drafts_exists(key.split(":")[0]):
                    raise
            finally:
                jobs.pop(key, None)
        task = asyncio.get_running_loop().create_task(go())
        jobs[key] = task

    def answering(draft_id: str) -> bool:
        """Whether an agent turn holds this draft now, on any app instance (the turn lock: an advisory lock on
        PostgreSQL, package 10.5; a thread lock on files). This instance's own queued turn is checked by the caller."""
        lock = drafts.turn(draft_id)
        if hasattr(lock, "release"):  # files: a re-entrant thread lock; this thread never holds it
            if lock.acquire(blocking=False):
                lock.release()
                return False
            return True
        return not lock.acquire(0)

    def process(draft_id: str, asset_id: str) -> None:
        """The pipeline of one asset, as a job (in a task of this process when no job table is wired)."""
        if services is None:
            background(f"{draft_id}:{asset_id}", work_now().process, draft_id, asset_id)
            return
        services.submit("pipeline", f"{draft_id}:{asset_id}", {"draft_id": draft_id, "asset_id": asset_id})

    @router.get("/api/packs", response_model=list[Doc])
    async def list_packs():
        return packs.cards()  # labelled from the capability matrix (16.2)

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
        from .. import experience  # 19.3: the problem as the person first wrote it

        experience.record("draft.created", "person", {"problem": problem[:2000], "pack": key}, conversation=draft["id"])
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
        draft = get(draft_id)
        from ..storage.files import files_for
        try:
            files = files_for(drafts)
            prefix = files.key_of(drafts.data_dir(draft_id).parent)
        except Exception:  # noqa: BLE001 — a draft is always deletable
            files = prefix = None
        await asyncio.to_thread(drafts.delete, draft_id)  # it waits for a write in progress: never on the event loop
        forgotten: list[str] = []
        if files is not None:
            try:
                forgotten = files.forget(prefix)  # the records of its files go with it (package 9.3)
            except Exception:  # noqa: BLE001
                pass
        from .. import audit  # the draft and its files are gone; who deleted it stays (package 12.5)
        from ..experience import retention  # 19.10: and what it left in the session record and the decision ledger

        gone = await asyncio.to_thread(retention.forget, draft=draft_id, usage=getattr(models, "usage", None))
        audit.record_for(drafts, "deletion", "human", draft_id=draft_id, move="delete_draft", status="allowed",
                         assets=len(draft.get("assets") or []), project_id=draft.get("project_id"), files=len(forgotten), **gone)
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
                from .. import experience  # 19.3: a rewritten problem is a signal of what was not understood

                experience.record("draft.edited", "person", {"field": "problem", "before": before, "after": problem[:2000]}, conversation=draft_id)
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
        # the target the caller names, else what the chat established or the sample's own; never a guess from the profile's
        # ranking: a column chosen because it has few values became the outcome once (customer_id, 2026-10-13)
        hinted = [(draft.get("understanding") or {}).get("target"), (asset.get("suggestion") or {}).get("target"),
                  studio_data.named_outcome(profile, " ".join(m.get("text", "") for m in draft.get("messages") or [] if m.get("role") == "user")
                                            + " " + str(draft.get("problem") or ""))]
        target = str(body["target"]) if body.get("target") else next((str(t) for t in hinted if t and str(t) in frame.columns), "")
        if not target:
            raise HTTPException(422, "Choose the outcome column: DCLab does not pick it for you")
        refused = studio_data.target_refusal(next((c for c in profile.get("columns") or [] if c.get("name") == target), None))
        if refused:
            from .. import experience

            experience.record("solution.refused", "system", {"target": target[:200], "reason": refused, "rule": "Target can be an outcome"},
                              conversation=draft_id)
            raise HTTPException(422, refused)
        hint = asset.get("suggestion") or {}
        try:  # a studied sample's time column orders the rows the proposal reads, as it will order the run's split
            proposal = await asyncio.to_thread(studio_solution.propose, frame, profile, target, body.get("task"),
                                               (draft.get("pack") or {}).get("key"), hint.get("time_column"))
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        proposal["rule"] = {k: proposal.get(k) for k in ("identifiers", "forbidden", "time_candidates", "group_candidates")}  # 19.7: the rules' own
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
        from .. import experience  # 19.3: what was proposed, to compare with what the person saved

        experience.record("solution.proposed", "system", {"by": "audit", "target": target[:200], "task": proposal.get("task"), "kind": proposal.get("kind"),
                                                          "forbidden": [str(f["column"])[:120] for f in proposal["forbidden"]][:200] or None},
                          conversation=draft_id)
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
            profile = (draft.get("analysis") or {}).get("profile") or studio_data.profile_table(frame)
            refused = studio_data.target_refusal(next((c for c in profile.get("columns") or [] if c.get("name") == solution.target), None))
            if refused:  # the same rule as the project's set_solution check
                from .. import experience

                experience.record("solution.refused", "system", {"target": solution.target[:200], "reason": refused, "rule": "Target can be an outcome"},
                                  conversation=draft_id)
                raise ValueError(refused)
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
        from .. import audit

        if draft.get("solution") != value:
            audit.solution(drafts, draft, value, "human", draft=True)
            from ..ledger import capture  # 19.7: the person's choices against the proposal, as decisions

            active = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset")), None) or {}
            await asyncio.to_thread(capture.solution_saved, draft.get("proposal"), profile, value, "human", active.get("suggestion"),
                                    draft=draft_id, conversation=draft_id)
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
        before = draft.get("settings") or {}
        drafts.update(draft_id, lambda d: d.update(settings=settings))
        from ..experience import hooks  # 19.3: how it will run, as the person set it

        hooks.settings_changed({k: v for k, v in settings.items() if k != "budget" and before.get(k) != v}, conversation=draft_id)
        return drafts.get(draft_id)

    @router.post("/api/drafts/{draft_id}/messages", status_code=202, response_model=Doc)
    async def message(draft_id: str, payload: AnyObject):
        get(draft_id)
        text = str(dump(payload).get("text", "")).strip()
        if not text:
            raise HTTPException(422, "Write a message first")
        busy = await asyncio.to_thread(answering, draft_id)  # first: nothing awaits between the check below and registering the turn
        if busy or ((draft_id + ":agent") in jobs and not jobs[draft_id + ":agent"].done()):
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
        from ..agentic.services import charge_upload

        if services is not None:
            await asyncio.to_thread(services.job_room)  # its pipeline is a job: refused before anything is stored or charged (package 10.6)
        await asyncio.to_thread(charge_upload, len(body))  # the day's allowance
        (drafts.data_dir(draft_id) / name).write_bytes(body)
        asset = await asyncio.to_thread(pipeline.new_asset, drafts, draft_id, "upload", name, name, bytes=len(body))
        process(draft_id, asset["id"])
        return asset

    @router.post("/api/drafts/{draft_id}/data/from-draft", response_model=Doc)
    async def bring_table(draft_id: str, payload: AnyObject):
        """{"draft": another draft's id}: that draft's table, copied into this one (21.3), so a file of cases and the
        table it labels are read together; the other draft is left as it was."""
        get(draft_id)
        if services is not None:
            await asyncio.to_thread(services.job_room)  # its pipeline is a job (package 10.6)
        try:
            brought = await asyncio.to_thread(bring_from_draft, drafts, draft_id, str(dump(payload).get("draft") or ""))
        except BringRefused as refused:
            raise HTTPException(refused.status, str(refused)) from None
        process(draft_id, brought["id"])
        return brought

    @router.post("/api/drafts/{draft_id}/data/sample", response_model=Doc)
    async def sample(draft_id: str, payload: AnyObject):
        get(draft_id)
        if services is not None:
            await asyncio.to_thread(services.job_room)  # its pipeline is a job (package 10.6): refused before a copy is made
        key = str(dump(payload).get("key", ""))
        if key not in {s["key"] for s in studio_data.sample_catalog()}:
            raise HTTPException(404, "Unknown sample dataset")
        frame, policy = await asyncio.to_thread(studio_data.load_sample, key)
        name = f"{key}.parquet"
        await asyncio.to_thread(pipeline._to_parquet, frame, drafts.data_dir(draft_id) / name)
        asset = await asyncio.to_thread(pipeline.new_asset, drafts, draft_id, "sample", key, name, suggestion=_suggestion(policy))
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
            background(f"{draft_id}:synthetic", work_now().simulate, draft_id, payload["prompt"], rows, template)
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
        if services is not None:
            await asyncio.to_thread(services.job_room)  # its pipeline is a job (package 10.6): refused before anything is downloaded
        directory = drafts.data_dir(draft_id)
        try:
            meta = await asyncio.to_thread(fetch, directory)
        except connectors.BadInput as exc:
            raise HTTPException(422, str(exc)) from None
        except connectors.ConnectorError as exc:
            raise HTTPException(400, str(exc)) from None
        size = (directory / meta["filename"]).stat().st_size
        from ..agentic.services import charge_upload

        try:
            await asyncio.to_thread(charge_upload, size)  # a connector's import counts against the day's allowance too (package 10.6)
        except HTTPException:
            (directory / meta["filename"]).unlink(missing_ok=True)
            raise
        asset = await asyncio.to_thread(pipeline.new_asset, drafts, draft_id, kind, name, meta["filename"], source=meta["source"], bytes=size)
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

        # On PostgreSQL the stream sleeps until an instance commits an event for this draft (LISTEN/NOTIFY, package 10.5),
        # with a slow poll as the fallback; on files it polls, as one process writes them.
        listening = drafts.wait_for_events() if hasattr(drafts, "wait_for_events") else None

        async def stream():
            loop = asyncio.get_running_loop()
            seq, began, quiet = start, loop.time(), loop.time()
            yield "retry: 1500\n\n"
            # subscribed before the first read: an event committed between a read and the wait still wakes the stream
            woken = listening.subscribe(draft_id) if listening is not None else None
            try:
                first = True
                while first or loop.time() - began < limit:
                    first = False
                    if await request.is_disconnected():
                        break
                    if woken is not None:
                        woken.clear()
                    new = await asyncio.to_thread(drafts.events, draft_id, seq)
                    for event in new:
                        seq = event["seq"]
                        yield f"id: {seq}\nevent: {event['kind']}\ndata: {json.dumps(event['data'], ensure_ascii=False, default=str)}\n\n"
                    if new:
                        quiet = loop.time()
                        continue
                    left = max(0.05, limit - (loop.time() - began))
                    if woken is not None and listening.connected.is_set():
                        try:
                            await asyncio.wait_for(woken.wait(), min(2.0, left))
                        except asyncio.TimeoutError:
                            pass
                    else:
                        await asyncio.sleep(min(0.4, left))
                    if loop.time() - quiet >= 15:
                        yield ": keep-alive\n\n"
                        quiet = loop.time()
            finally:
                if woken is not None:
                    listening.unsubscribe(draft_id, woken)
        return StreamingResponse(stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"})

    @router.get("/api/drafts/{draft_id}/card", response_model=Doc)
    async def draft_card(draft_id: str):
        """The problem card (20.1): the family, the label situation, the questions still open and whether a project
        can be built yet, with why not."""
        from .card import card_of, meeting

        draft = get(draft_id)
        card = await asyncio.to_thread(card_of, draft)
        met = await asyncio.to_thread(meeting, draft, drafts)  # 20.4: what the problem and the data disagree on
        return {**card.model_dump(mode="json"), "meets_data": met.model_dump(mode="json") if met is not None else None}

    @router.post("/api/drafts/{draft_id}/build", status_code=201, response_model=Project)
    async def build(draft_id: str):
        draft = get(draft_id)
        if draft.get("project_id"):
            try:
                return projects.get(draft["project_id"])
            except KeyError:
                pass
        try:
            return await asyncio.to_thread(build_project, drafts, projects, draft_id)
        except BuildRefused as exc:
            from .. import experience

            experience.record("draft.refused", "system", {"reason": str(exc)[:2000], "rule": exc.rule}, conversation=draft_id)
            raise HTTPException(409, str(exc)) from None

    app.include_router(router)


def split_of(draft: dict[str, Any]) -> str:
    """The split the engine will use for this draft's solution (stratified until a solution names a time or group column)."""
    from ..studio import engine as studio_engine

    return studio_engine.split_for(draft.get("solution") or {})


def _suggestion(policy: dict[str, Any] | None) -> dict[str, Any] | None:
    if not policy:
        return None
    return {k: policy.get(k) for k in ("target", "task", "positive_label", "forbidden", "identifiers", "time_column", "prediction_moment") if k in policy}


def _coverage(answer: Any) -> str:
    """"complete" only when the answer says every case is labelled and nothing says otherwise; partial by default (a
    wrong "complete" would present a precision floor as exact)."""
    text = str(answer or "").lower()
    if re.search(r"\b(not|n't|never|some|sample|partial|only|few|haven't|didn't)\b|n't\b", text):
        return "partial"
    return "complete" if re.search(r"\b(every|all) (the )?(cases?|frauds?|rows?)\b.*\b(labell?ed|known|found)\b|\bcomplete\b", text) else "partial"


def _entity_column(answer: Any, columns: list[str]) -> str | None:
    """The column an answer names as the entity: the column itself, or the one whose name shares a word with it
    ("the biker" finds biker_id); none for "No single entity"."""
    text = str(answer or "").strip()
    if not text or re.search(r"\bno (single )?entity\b", text, re.I):
        return None
    if text in columns:
        return text
    words = {w[:5] for w in re.findall(r"[a-z]{4,}", text.lower())}
    named = [c for c in columns if {w[:5] for w in re.findall(r"[a-z]{4,}", c.lower())} & words]
    return sorted(named, key=lambda c: (not re.search(r"(_id|_ref|_no)$", c.lower()), len(c)))[0] if named else None


def _budget_share(answer: Any, projects, project: dict[str, Any]) -> float | None:
    """The review budget as a share of rows: "0.5%", or "50 a day" over the days the table spans (its time column)."""
    if not answer or not project.get("data"):
        return None
    from dclab_core.assess import capacity_share

    from ..studio import data as studio_data

    import pandas as pd  # lazily: the draft store and the chat run without pandas

    span = None
    times = (project["data"].get("profile") or {}).get("time_candidates") or []
    if times:
        frame = studio_data.load_table(studio_data.data_path(projects, project))
        stamps = pd.to_datetime(frame[times[0]], errors="coerce").dropna()
        span = (stamps.max() - stamps.min()).total_seconds() / 86400 if len(stamps) > 1 else None
    share = capacity_share(str(answer), int(project["data"]["rows"]), span)
    return round(share, 6) if share else None


class BuildRefused(ValueError):
    """The draft cannot become a project yet; the message says what is missing, ``rule`` which check said so."""

    def __init__(self, message: str, rule: str):
        super().__init__(message)
        self.rule = rule


def build_blockers(drafts: DraftStore, draft_id: str, capture_card: bool = False) -> list[BuildRefused]:
    """Every check that refuses a build now, in order (21.2: the coordinator reads them; ``build_project`` raises the
    first): only known cases, a problem card not ready, a problem and data that disagree, a family not built yet."""
    draft = drafts.get(draft_id)
    out: list[BuildRefused] = []
    cases = (draft.get("analysis") or {}).get("cases_table") or {}
    columns = {c.get("name") for c in ((draft.get("analysis") or {}).get("profile") or {}).get("columns") or []}
    named = (draft.get("solution") or {}).get("target") or ((draft.get("understanding") or {}).get("target") if
                                                              (draft.get("understanding") or {}).get("target") in columns else None)
    active = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset")), None) or {}
    if active.get("kind") == "labels":  # the cases moved aside for a table that never became ready: they are not the table
        out.append(BuildRefused("The known cases are here but the table they label is not ready: add it again, and its cases "
                                "become the labels", "list of known cases"))
    elif cases and plans.label_asset(draft) is None and not named:  # the morning of 2026-10-13: the cases' file alone became a project
        out.append(BuildRefused("This draft holds only a list of known cases: add the table they belong to (with "
                                f"{', '.join(cases['keys'][:2])}), and its cases become the labels", "list of known cases"))
    from .card import card_of

    card = card_of(draft)
    if capture_card:
        from ..ledger import capture  # 19.7: the card's family and readiness, at the moment a build is asked for

        capture.problem_card(draft, card)
    has_data = any(a.get("status") == "ready" for a in draft.get("assets") or [])
    if has_data and not card.ready:  # 20.1: a project is built from a ready problem card
        out.append(BuildRefused("Not yet: " + " ".join(card.blocking[:3]), "problem card not ready"))
    if not has_data and card.in_play:  # a plan-only draft carries its questions, but not one about what the problem is
        out.append(BuildRefused("Not yet: " + " ".join(b for b in card.blocking if "confirmed" in b), "problem card not ready"))
    if has_data:  # 20.4: the problem meets the data; a mismatch that would answer another problem waits for the person
        from .card import meeting

        met = meeting(draft, drafts)
        blocking = [m.question for m in (met.mismatches if met is not None else []) if m.blocking]
        if blocking:
            out.append(BuildRefused("Not yet: " + " ".join(blocking[:2]), "problem and data disagree"))
    return out  # 20.7: anomaly detection without labels is built as its own kind (never as a model of some column)


class BringRefused(ValueError):
    """Another draft's table cannot be brought here; ``status`` is the HTTP code the route answers with."""

    def __init__(self, message: str, status: int):
        super().__init__(message)
        self.status = status


def bring_from_draft(drafts: DraftStore, draft_id: str, other_id: str) -> dict[str, Any]:
    """Copy another draft's ready table into this one (21.3), as a new asset to read; the other draft is left as it was.
    Refused (``BringRefused``) for this draft itself, an unknown draft, one with no table ready, a table already here,
    or one the pipeline would not read with these cases. The caller runs the new asset's pipeline."""
    if not other_id or other_id == draft_id:
        raise BringRefused("Name another draft of this workspace", 422)
    try:
        other = drafts.get(other_id)
    except KeyError:
        raise BringRefused("No such draft in this workspace", 404) from None
    asset = next((a for a in other.get("assets") or [] if a.get("id") == other.get("active_asset") and a.get("status") == "ready"
                  and a.get("kind") != "labels" and a.get("clean_file")), None)
    if asset is None:
        raise BringRefused("That draft has no table ready to bring", 409)
    here = drafts.get(draft_id)
    if any((a.get("source") or {}).get("draft_id") == other_id for a in here.get("assets") or []):
        raise BringRefused("That table is already here", 409)
    source = drafts.data_dir(other_id) / asset["clean_file"]
    mine = next((a for a in here.get("assets") or [] if a.get("id") == here.get("active_asset") and a.get("status") == "ready"
                 and a.get("clean_file")), None)
    if mine is not None:  # the pipeline's own rules: what it would not read as labels is not brought over the cases
        import pandas as pd

        from ..studio.labels import joinable

        cases = pd.read_parquet(drafts.data_dir(draft_id) / mine["clean_file"])
        if not joinable(cases, source):
            raise BringRefused("That table would not be read with these cases: they are more than half its rows, or have no column of their own", 409)
    name = f"from-{other_id}-{asset['id']}.parquet"  # never a name of this draft's own files (clean.parquet, labels_clean.parquet)
    shutil.copyfile(source, drafts.data_dir(draft_id) / name)
    return pipeline.new_asset(drafts, draft_id, "draft", asset.get("name") or name, name,
                              source={"kind": "draft", "draft_id": other_id, "asset": asset.get("id")})


def _anomaly_solution(projects, project: dict[str, Any], u: dict[str, Any]) -> dict[str, Any]:
    """The solution of an anomaly project (20.7): no target; the entity the person named, the table's day column, the
    key of a row (a column unique on every row, named as an id) and the review budget they stated."""
    import pandas as pd

    from dclab_core import anomaly

    from ..studio import data as studio_data
    from ..studio.solution import Solution

    frame = studio_data.load_table(studio_data.data_path(projects, project))
    columns = list(frame.columns)
    entity = _entity_column(u.get("entity"), columns)
    time = None
    for c in columns:
        days = anomaly._days(frame, c)
        if days is not None and days.nunique() >= 3:
            time = c
            break
    # the key a person names a row by: an id column nearly unique (an order with two drop-offs repeats), before a reference
    named = [c for c in columns if c != entity and re.search(r"(^id$|_id$|_ref$)", str(c), re.I)]
    share = {c: frame[c].nunique() / max(len(frame), 1) for c in named}
    ids = sorted((c for c in named if share[c] >= 0.95 and re.search(r"(^id$|_id$)", str(c), re.I)), key=lambda c: -share[c])
    refs = sorted((c for c in named if share[c] >= 0.95), key=lambda c: -share[c])
    key = (ids or refs or [None])[0]
    budget = project.get("suggestion", {}).get("alert_budget") if isinstance(project.get("suggestion"), dict) else None
    moment = str(u.get("prediction_moment") or "When the rows are reviewed after they happen; nothing is labelled yet.")
    return Solution(target=None, task="binary", prediction_moment=moment if len(moment) >= 12 else moment + " (nothing is labelled yet)",
                    labels="none", kind="anomaly", kind_fields={"entity_column": entity, "time_column": time, "key_column": key,
                                                                "budget": budget}).model_dump()


def build_project(drafts: DraftStore, projects, draft_id: str) -> dict[str, Any]:
    """Create the project from the draft: its name and goal, the cleaned table, and what the chat established. Refused
    while the draft holds only a list of known cases (every row a case: nothing to tell them from)."""
    from ..studio import data as studio_data

    refused = build_blockers(drafts, draft_id, capture_card=True)
    if refused:  # the first check that refuses says what is missing
        raise refused[0]
    from .card import card_of

    draft = drafts.get(draft_id)
    card = card_of(draft)
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
        studio_data.attach_data(projects, pid, filename, source={"kind": "draft", "draft_id": draft_id, "asset": asset["id"],
                                                                 "asset_kind": asset.get("kind"), "synthetic": bool(asset.get("synthetic"))})
    labels = plans.label_asset(draft) if asset is not None else None
    if labels is not None:  # 15.6: the draft's label file comes with the table, its key checked again by attach_labels
        coverage = _coverage(u.get("labels"))
        shutil.copyfile(drafts.data_dir(draft_id) / labels["clean_file"], projects.data_dir(pid) / "labels.parquet")
        try:
            studio_data.attach_labels(projects, pid, "labels.parquet", target="is_case", labels=coverage)
        except studio_data.DataError as exc:
            projects.log(pid, "labels_refused", {"reason": str(exc)[:300]})
    project = projects.get(pid)
    if asset is not None and project.get("data"):
        project["data"]["synthetic"] = bool(asset.get("synthetic"))
        suggestion = {**dict(asset.get("suggestion") or {}), **dict(project.get("suggestion") or {})}
        entity = _entity_column(u.get("entity"), project["data"]["columns"])
        if entity:
            suggestion["group_column"] = entity
        budget = _budget_share(u.get("budget"), projects, project)
        if budget:
            suggestion["alert_budget"] = budget
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
    project["pack"] = (draft.get("pack") or {}).get("key")  # the engine reads the pack's defaults from here (15.6)
    project["draft"] = {"id": draft_id, "problem": draft["problem"], "pack": draft.get("pack"), "understanding": u, "settings": draft.get("settings"),
                        "card": card.model_dump(mode="json"),  # 20.1: the problem as understood, its open questions with it
                        "workflow": draft.get("workflow"), "cleaning_log": draft.get("cleaning_log"),
                        "synthetic": bool(asset and asset.get("synthetic"))}
    built_solution = draft.get("solution")
    if not built_solution and card.family.value == "anomaly" and labels is None and project.get("data"):  # 20.7: no outcome
        built_solution = _anomaly_solution(projects, project, u)  # saved below like the wizard's, through the same check
    projects.save(project)
    projects.log(pid, "built_from_draft", {"draft": draft_id, "rows": (project.get("data") or {}).get("rows")})
    if built_solution and project.get("data"):
        from ..studio import graph as studio_graph

        project = projects.get(pid)
        verdict = studio_graph.check(project, "set_solution", "human", target=built_solution["target"])  # same check as the project page
        studio_graph.log(projects, pid, verdict, project, outcome="done: solution from the draft" if verdict.allowed else None)
        if verdict.allowed:
            from ..studio import memory as studio_memory

            project = projects.get(pid)
            project["solution"] = built_solution
            studio_memory.solution_saved(project, None, built_solution, "human")  # A5.2: the person saved it in the wizard
            projects.save(project)
            from .. import audit

            audit.solution(projects, project, built_solution, "human", built=True)

    def done(d):
        d["status"], d["project_id"] = "built", pid
    drafts.update(draft_id, done)
    drafts.emit(draft_id, "status", {"built": True, "project_id": pid})
    return projects.get(pid)
