"""The kinds of job (package 10.3): stage runs, draft data pipelines, synthetic data, intern turns, and (20.8) a
pipeline step rerun from its record.

``env`` is the app's ``Services`` (``agentic/services.py``): the stores, the model gateway and the draft work, the same
in the server and in ``python -m dclab_rnd.worker``. A payload holds ids and plain values only, so any worker can run it.
"""

from __future__ import annotations

from typing import Any

from .store import now
from .worker import TEXT, checkpoint, current, handler, progress


# ---------------------------------------------------------------------- stage runs: one project, one or more stages in order

def _stage_run(s: Any, payload: dict[str, Any]) -> None:
    from ..studio import engine as studio_engine

    pid, stages, reuse = payload["project_id"], list(payload["stages"]), payload.get("reuse_reason")
    try:
        for n, stage in enumerate(stages, 1):
            checkpoint()  # between stages only: a stage that started finishes, so its record is the deterministic one
            progress(stage=stage, step=n, of=len(stages))
            studio_engine.execute(s.projects, pid, stage, reuse_reason=reuse if stage == "final" else None)
    finally:
        control = current()
        if control is None or control.owned():  # a job taken over (even during its last stage) belongs to the worker that took it
            _stage_tidy(s, payload, None)


def _stage_tidy(s: Any, payload: dict[str, Any], reason: str | None) -> None:
    """Stages still "queued" go back to "pending"; a stage left "running" (its worker died) is failed; nothing runs."""
    p = s.projects.get(payload["project_id"])
    changed = False
    for stage in payload["stages"]:
        st = p["stages"].get(stage) or {}
        if st.get("status") == "queued":
            p["stages"][stage] = {"status": "pending"}
            changed = True
        elif st.get("status") == "running" and reason is not None:
            p["stages"][stage] = {**st, "status": "failed", "error": TEXT.get(reason, "The run ended before this stage finished"), "finished": now()}
            changed = True
    if p.get("running") is not None:
        p["running"] = None
        changed = True
    if changed:
        s.projects.save(p)


def _stage_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    _stage_tidy(s, payload, reason)


# ---------------------------------------------------------------------- the draft pipeline: one asset

def _pipeline_run(s: Any, payload: dict[str, Any]) -> None:
    s.draft_work.process(payload["draft_id"], payload["asset_id"])


def _pipeline_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    from ..draft import pipeline

    did, aid = payload["draft_id"], payload["asset_id"]
    asset = next((a for a in s.drafts.get(did).get("assets") or [] if a["id"] == aid), None)
    if asset is None or asset.get("status") in ("ready", "failed"):
        return
    text = TEXT.get(reason, "The data pipeline ended before the table was ready")
    pipeline.set_asset(s.drafts, did, aid, status="failed", error=text)
    s.drafts.emit(did, "pipeline", {"asset": aid, "step": "failed", "text": text})


# ---------------------------------------------------------------------- synthetic data: design, generate, then the pipeline

def _synthetic_run(s: Any, payload: dict[str, Any]) -> None:
    s.draft_work.simulate(payload["draft_id"], payload.get("prompt") or "", int(payload.get("rows") or 5000), payload.get("template"),
                          actor="agent" if payload.get("by") == "agent" else "human")


def _synthetic_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    text = TEXT.get(reason, f"The synthetic data could not be generated: {job.get('error') or 'the job failed'}")
    s.drafts.emit(payload["draft_id"], "pipeline", {"step": "failed", "text": text, "synthetic": True})


# ---------------------------------------------------------------------- intern turns: the first run, or a follow-up message

def _intern_run(s: Any, payload: dict[str, Any]) -> None:
    agent = s.intern()
    if payload.get("action") == "message":
        agent.message(payload["session_id"], payload.get("text") or "", again=True)
    else:
        agent.run(payload["session_id"])


def _intern_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    session = s.intern_sessions.get(payload["session_id"])
    if session.get("status") not in ("queued", "running"):
        return
    if reason == "stopped":
        session.update(status="stopped", final=session.get("final") or TEXT["stopped"] + ".")
    else:
        session.update(status="failed", error=TEXT.get(reason) or job.get("error") or "The turn ended before it finished")
    s.intern_sessions.save(session)


# ---------------------------------------------------------------------- a pipeline step rerun from its record (20.8)

def _rerun_run(s: Any, payload: dict[str, Any]) -> None:
    from .. import audit
    from ..studio import pipeline

    pid, step = payload["project_id"], payload["step"]
    checkpoint()
    result = pipeline.rerun(s.projects, pid, step)
    control = current()
    keep = {k: result.get(k) for k in ("replayed", "same_numbers", "same_value", "same_choice", "same", "numbers_compared", "compared_with_runs",
                                        "run", "original_run", "metric", "original", "rerun", "why", "differ", "changed_numbers")}
    s.projects.log(pid, "pipeline_rerun", {"job": control.job["id"] if control else None, "step": step, **{k: v for k, v in keep.items() if v is not None}})
    if step == "stage:final" and result.get("replayed"):  # the holdout read again in a scratch copy: said in the audit trail
        audit.record_for(s.projects, "replay", "human", project={"id": pid}, move="replay_final", status="allowed", run=result.get("run"),
                         original_run=result.get("original_run"))


def _rerun_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    s.projects.log(payload["project_id"], "pipeline_rerun", {"job": job["id"], "step": payload["step"], "replayed": False,
                                                             "why": [TEXT.get(reason, "The rerun ended before it finished")]})


def register() -> None:
    handler("stage", _stage_run, _stage_settle)
    handler("pipeline", _pipeline_run, _pipeline_settle)
    handler("synthetic", _synthetic_run, _synthetic_settle)
    handler("intern", _intern_run, _intern_settle)
    handler("rerun", _rerun_run, _rerun_settle)


register()
