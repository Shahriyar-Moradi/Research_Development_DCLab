"""The kinds of job (package 10.3): stage runs, draft data pipelines, synthetic data, intern turns, and (20.8) a
pipeline step rerun from its record, (24.5) a programme, (25.6) a twin run of a project: the current system and Claude
on the same problem in two lanes, and (29.7) the students' loop of a workspace.

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
                          actor="agent" if payload.get("by") == "agent" else "human", proposal=payload.get("proposal") or None)


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
                                        "run", "original_run", "metric", "original", "rerun", "why", "differ", "changed_numbers", "same_steps",
                                        "first_step_difference", "steps_compared")}
    s.projects.log(pid, "pipeline_rerun", {"job": control.job["id"] if control else None, "step": step, **{k: v for k, v in keep.items() if v is not None}})
    if step == "stage:final" and result.get("replayed"):  # the holdout read again in a scratch copy: said in the audit trail
        audit.record_for(s.projects, "replay", "human", project={"id": pid}, move="replay_final", status="allowed", run=result.get("run"),
                         original_run=result.get("original_run"))


def _rerun_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    s.projects.log(payload["project_id"], "pipeline_rerun", {"job": job["id"], "step": payload["step"], "replayed": False,
                                                             "why": [TEXT.get(reason, "The rerun ended before it finished")]})


# ---------------------------------------------------------------------- a programme: the leaves, the composition, the holdout once (24.5)

def _programme_run(s: Any, payload: dict[str, Any]) -> None:
    from ..studio import programme

    from ..draft import map as draft_map

    checkpoint()
    advisor = draft_map.advisor_for(s.gateway, draft_id=((s.projects.get(payload["project_id"]).get("draft") or {}).get("id")))  # None unless the roles passed their measurement and are on
    programme.run(s.projects, payload["project_id"], payload.get("actor") or "human", payload.get("reuse_reason"), advisor=advisor)


def _programme_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    p = s.projects.get(payload["project_id"])
    if p.get("running") == "programme":  # the worker died or was stopped mid-run: the project is free again
        p["running"] = None
        s.projects.save(p)
    s.projects.log(payload["project_id"], "programme_run", {"job": job["id"], "state": "not_finished", "why": TEXT.get(reason, "The programme ended before it finished")})


# ---------------------------------------------------------------------- a twin run of a project: the system and Claude side by side (25.6)

def _twin_run(s: Any, payload: dict[str, Any]) -> None:
    from ..extensions import optional

    twin, twin_home = optional("twin"), optional("twin.home")  # registered only when the edition has the twin lane

    settings = twin_home.settings(s.store.home)
    if not settings["on"]:
        raise RuntimeError("Claude beside the system is off in this workspace (the owner's switch)")
    checkpoint()
    progress(stage="twin", step=1, of=1)
    manifest = twin.run(f"project:{payload['project_id']}", home=s.store.home, source_home=s.store.home, blind=bool(payload.get("blind", True)),
                        cap_eur=float(payload.get("cap_eur") or settings["session_eur"]), max_requests=int(payload.get("max_requests") or 80),
                        run_id=payload["run_id"], log=lambda *a: None)
    s.projects.log(payload["project_id"], "twin_run", {"run": payload["run_id"], "lanes": {k: v.get("status") for k, v in (manifest.get("lanes") or {}).items()},
                                                       "differences": manifest.get("differences"), "spent_eur": (manifest.get("model_usage") or {}).get("eur")})


def _twin_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    s.projects.log(payload["project_id"], "twin_run", {"run": payload.get("run_id"), "state": "not_finished",
                                                       "why": TEXT.get(reason) or job.get("error") or "The twin run ended before it finished"})


def _students_run(s: Any, payload: dict[str, Any]) -> None:
    """29.7: the students' loop on this workspace: sets, the students whose homework grew, demotion, the register."""
    from pathlib import Path

    from ..extensions import optional

    optional("students_loop").run(Path(s.store.home), s.ledger, s.gateway.routing, s.audit)  # registered only in the full edition


def _students_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    from pathlib import Path

    from ..extensions import optional

    optional("students_loop").failed(Path(s.store.home))  # the next run waits an hour


STORE_RUN_SECONDS = 3 * 3600  # a whole run of a large workspace takes about twenty minutes; three hours means it is stuck


def _store_run(s: Any, payload: dict[str, Any]) -> None:
    """32.2: one run of the store's conductor (the keepers, then the later tiers), under the hood. It runs in a process of
    its own at a lower priority: in the worker's own interpreter its twenty minutes of Python would slow every stage run
    and intern turn the worker runs beside it (2026-10-20: a models stage crawled while a store run held the worker)."""
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    from ..settings import ROOT

    command = [sys.executable, "-m", "dclab_rnd.store", "run", "--agent-runs", str(Path(s.store.home))]
    done = subprocess.run(command, cwd=ROOT, env=dict(os.environ), capture_output=True, text=True, timeout=STORE_RUN_SECONDS,
                          preexec_fn=(lambda: os.nice(10)) if hasattr(os, "nice") else None)
    try:
        json.loads(done.stdout[done.stdout.index("{"):])  # the run's record: a run whose keepers partly failed still ran
    except ValueError:
        raise RuntimeError(f"the store's run ended without its record (exit {done.returncode}): {done.stderr.strip()[-500:]}") from None


def _store_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    """Nothing to put back (the store is append-only); the failure recorded, so the next run waits its hour."""
    from ..extensions import optional

    optional("store.conductor").failed(why=reason)  # registered only in the full edition


# ---------------------------------------------------------------------- improving a finished model, on the training folds only

def _improve_run(s: Any, payload: dict[str, Any]) -> None:
    from ..studio import improve

    pid = payload["project_id"]
    checkpoint()
    p = s.projects.get(pid)
    p["running"] = "improve"
    s.projects.save(p)
    draft_id = (p.get("draft") or {}).get("id")
    try:
        record = improve.run(s.projects, pid)
    except ValueError as exc:  # a project that cannot be improved here: said in its chat, not a failed job
        if draft_id:
            s.draft_work.agent(draft_id).say(draft_id, str(exc))
        return
    finally:
        done = s.projects.get(pid)
        if done.get("running") == "improve":
            done["running"] = None
            s.projects.save(done)
    if draft_id:
        tried, clear = len(record["candidates"]), sum(1 for c in record["candidates"] if c["clears"])
        best = next((c for c in record["candidates"] if c["id"] == record["best"]), None)
        text = (f"I tried {tried} ways to improve the model, on the training folds only. "
                + (f"{clear} cleared the margin; the best is {best['label']}, {record['metric']} {best['mean']:.4f} against {record['current']['mean']:.4f} "
                   f"({best['folds_better']} of {record['folds']} folds better). Adopt it to use it in the next final run." if best else
                   (f"{clear} cleared the margin, but none can be adopted here (another family, other features or a blend: choose it on the Models page)." if clear else
                    "None beat the current model by more than the noise of the folds, so the honest answer is that this model is about as good as these ways can make it: "
                    "more cases, a new column known at the prediction moment, or a cleaner label would be the next step.")))
        s.draft_work.agent(draft_id).say(draft_id, text, kind="improvement_report", report={"project": pid, **{k: record[k] for k in ("metric", "higher_is_better", "folds", "current", "aspects", "best", "skipped", "note", "adopted")},
                                                                                             "candidates": [{k: c[k] for k in ("id", "aspect", "label", "mean", "gain", "folds_better", "clears", "adoptable", "why")} for c in record["candidates"]]})


def _improve_settle(s: Any, payload: dict[str, Any], reason: str, job: dict[str, Any]) -> None:
    p = s.projects.get(payload["project_id"])
    if p.get("running") == "improve":  # the worker died or was stopped mid-run: the project is free again
        p["running"] = None
        s.projects.save(p)
    s.projects.log(payload["project_id"], "improve_run", {"job": job["id"], "state": "not_finished", "why": TEXT.get(reason, "The improvement ended before it finished")})


def register() -> None:
    handler("stage", _stage_run, _stage_settle)
    handler("pipeline", _pipeline_run, _pipeline_settle)
    handler("synthetic", _synthetic_run, _synthetic_settle)
    handler("intern", _intern_run, _intern_settle)
    handler("rerun", _rerun_run, _rerun_settle)
    handler("programme", _programme_run, _programme_settle)
    handler("improve", _improve_run, _improve_settle)
    from ..extensions import available

    if available("twin"):  # the closed parts' jobs exist only in the full edition
        handler("twin", _twin_run, _twin_settle)
    if available("students_loop"):
        handler("students", _students_run, _students_settle)
    if available("store"):
        handler("store", _store_run, _store_settle)


register()
