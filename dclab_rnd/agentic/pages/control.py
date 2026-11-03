"""The Admin page's Control tab (A3 of the 2026-11-03 plan): the owner's switches and the machine's state, in one place.

    GET  /api/admin/control                    the switches (control.json, and the environment settings that win over them),
                                               this month's spend against the cap, paused models, running jobs, disk, swap,
                                               the nightly store job and the last bench run
    POST /api/admin/control {live_models?, workspace_monthly_eur?, password}   change a switch (the Admin password again)
    POST /api/admin/control/resume-models      clear every model pause (after a refusal that is now fixed)
    POST /api/admin/control/stop-jobs {password}   ask every running job to stop at its next checkpoint, cancel queued ones

All owner-only (``accounts.guard``: ``admin``); every change is in the audit trail (kind ``admin``), never a password.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from ..api_models import AnyObject, Page, Router


def _swap_gb() -> float | None:
    try:
        out = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, timeout=5).stdout
        used = out.split("used = ")[1].split()[0]
        return round(float(used.rstrip("M")) / 1024, 2) if used.endswith("M") else None
    except Exception:  # noqa: BLE001 - not macOS, or no swap: unknown
        return None


def _store_job() -> dict[str, Any]:
    try:
        loaded = subprocess.run(["launchctl", "list", "ai.dclab.store"], capture_output=True, timeout=5).returncode == 0
    except Exception:  # noqa: BLE001
        loaded = None
    log = Path.home() / "Library" / "Logs" / "dclab" / "store-scheduled.log"
    last = [line for line in log.read_text(encoding="utf-8", errors="ignore").splitlines() if line[:4].isdigit()][-1:] if log.is_file() else []
    return {"loaded": loaded, "last": last[0][:200] if last else None}


def register(app, ctx) -> None:
    router = Router()

    def audit(move: str, status: str = "allowed", **values: Any) -> None:
        from ... import audit as audit_mod

        audit_mod.record(ctx.audit, "admin", "human", "Owner", move=move, status=status, **values)

    def confirmed(body: dict[str, Any], move: str) -> None:
        from ...accounts import admin_login

        if not admin_login.confirm(ctx.store.home, str(body.get("password") or "")):
            audit(move, "refused", reason="password")
            raise HTTPException(403, "The Admin password is needed again for this (or the sign-in is locked)")

    def state() -> dict[str, Any]:
        from ... import bench, control
        from ...models import pause
        from ...models.gateway import workspace_cap
        from ...models.usage import month_start

        home = ctx.store.home
        usage = getattr(ctx.models, "usage", None)
        jobs = ctx.job_store.list(active=True, limit=200) if ctx.job_store is not None else []
        runs = bench.run_rows()
        return {"switches": control.get(home),
                "environment_wins": {"DCLAB_NO_LIVE_MODELS": os.environ.get("DCLAB_NO_LIVE_MODELS") == "1",
                                     "DCLAB_WORKSPACE_MONTHLY_EUR": bool(os.environ.get("DCLAB_WORKSPACE_MONTHLY_EUR", "").strip())},
                "spend": {"month_eur": round(usage.spent(month_start()), 2) if usage is not None else None, "cap_eur": workspace_cap(home)},
                "paused_models": len(getattr(pause, "_LOCAL", {})),
                "jobs": [{"id": j.get("id"), "kind": j.get("kind"), "status": j.get("status"), "since": j.get("started") or j.get("created")} for j in jobs],
                "disk_free_gb": round(shutil.disk_usage(home).free / 1024 ** 3, 1), "swap_used_gb": _swap_gb(), "store_job": _store_job(),
                "last_bench_run": {k: runs[-1].get(k) for k in ("at", "commit", "states")} if runs else None}

    @router.get("/api/admin/control", response_model=Page)
    async def admin_control():
        return await asyncio.to_thread(state)

    @router.post("/api/admin/control", response_model=Page)
    async def admin_control_set(payload: AnyObject):
        from ... import control

        body = payload.model_dump(exclude_unset=True) if payload is not None else {}
        change = {k: body[k] for k in ("live_models", "workspace_monthly_eur") if k in body}
        if not change:
            raise HTTPException(422, "Say what to change: live_models, workspace_monthly_eur")
        confirmed(body, "control_set")
        try:
            doc = await asyncio.to_thread(control.set_values, ctx.store.home, **change)
        except ValueError as exc:
            raise HTTPException(422, str(exc))
        audit("control_set", **{k: v for k, v in change.items()})
        return {"switches": doc}

    @router.post("/api/admin/control/resume-models", response_model=Page)
    async def admin_resume_models():
        from ...models import pause

        await asyncio.to_thread(pause.clear)
        audit("resume_models")
        return {"resumed": True}

    @router.post("/api/admin/control/stop-jobs", response_model=Page)
    async def admin_stop_jobs(payload: AnyObject):
        body = payload.model_dump(exclude_unset=True) if payload is not None else {}
        confirmed(body, "stop_jobs")
        if ctx.job_store is None:
            return {"stopped": 0}
        jobs = await asyncio.to_thread(ctx.job_store.list, None, None, True, 200)
        stopped = sum(1 for j in jobs if ctx.job_store.cancel(j["id"]))
        audit("stop_jobs", stopped=stopped)
        return {"stopped": stopped}

    app.include_router(router)
