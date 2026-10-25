"""The conductor (32.2): runs the keepers, then the shapers, the curators and the builders (as each package adds them), under
the hood, and records every run.

``conduct(env)`` is one run: a harness session of the agent ``conductor`` (each keeper a step), and one record in
``runs/<yyyy-mm-dd>/<run>.json`` saying, for each step, what it read, kept, skipped and why, and any error. A keeper that
fails is recorded and the others still run; a source a keeper offers that it does not own is refused (it belongs to
another keeper's claim). ``sweep_due`` queues the job ``store`` from the worker's loop, at most once an hour, never in a
request's path; ``DCLAB_STORE_LOOP=0`` turns it off (the tests do).
"""

from __future__ import annotations

import json
import os
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Optional

from .keepers import Keeper
from .layout import Layout
from .store import Store, now, write_json

EVERY_SECONDS = 3600


def context() -> dict[str, Any]:
    """32.22: what every record of this run was kept with: the code's commit (and whether the tree was clean), Python and the
    libraries that make the numbers, the platform, and the machine (a hash of its name, never the name)."""
    import hashlib
    import platform
    import subprocess

    from .layout import ROOT

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""

    versions = {}
    for name in ("numpy", "pandas", "sklearn", "pydantic", "dclab_core"):
        try:
            versions[name] = getattr(__import__(name), "__version__", None)
        except ImportError:
            versions[name] = None
    return {"commit": git("rev-parse", "HEAD") or None, "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
            "python": platform.python_version(), "libraries": versions, "platform": platform.platform(terse=True),
            "machine": hashlib.sha256(platform.node().encode()).hexdigest()[:12]}


BINARY = (".pkl", ".pickle", ".joblib", ".sqlite3", ".npy", ".npz", ".parquet", ".onnx", ".pt", ".bin")


def _cleaned(src: Any) -> Any:
    """Every source's bytes with credentials removed after its own transform, whichever keeper offers it (a fix's failing output, a
    walk, a database row); a binary file left as it is (a regex on its bytes could break it), a gzip opened and closed again."""
    from .keepers.agents import redact
    from .keepers.standard import _gz_secrets

    if src.source.endswith(BINARY):
        return src.transform
    own = src.transform
    clean = _gz_secrets if src.source.endswith(".gz") else redact

    def both(raw: bytes) -> bytes:
        return clean(own(raw) if own is not None else raw)
    return both


def _step(keeper: Keeper, env: Any, store: Store, run: str) -> dict[str, Any]:
    """One keeper's sources kept, each on its own: a source that fails (unreadable, gone) is recorded with its error and the rest
    still run (nothing is refused at capture: an unreadable file is kept as its error)."""
    started = time.monotonic()
    out: dict[str, Any] = {"name": keeper.name, "stage": "keep", "kinds": sorted({c.kind for c in keeper.claims}), "feeds": list(keeper.feeds),
                           "read": 0, "kept": 0, "unchanged": 0, "refused": [], "errors": [], "error": None}
    try:
        for src in keeper.sources(env):
            out["read"] += 1
            kind = next((c.kind for c in keeper.claims if src.source.startswith(c.prefix)), None)
            if kind is None:
                out["refused"].append({"source": src.source, "why": f"not one of {keeper.name}'s claims"})
                continue
            args = dict(consent=src.consent, run=run, note=src.note, transform=_cleaned(src))
            try:
                if src.log:
                    entry = store.keep_log(kind, src.workspace, src.source, src.data, **args)
                else:
                    entry = store.keep(kind, src.workspace, src.source, src.data, **args)
            except Exception as exc:  # noqa: BLE001 - this source's error is kept; the others go on
                error = f"{type(exc).__name__}: {exc}"
                out["errors"].append({"source": src.source, "workspace": src.workspace, "error": error})
                try:
                    store.keep(kind, src.workspace, f"{src.source}~error.json", json.dumps({"source": src.source, "error": error}).encode(),
                               consent=src.consent, run=run, note="the source could not be read: its error is kept instead")
                except Exception:  # noqa: BLE001
                    pass
                continue
            out["kept" if entry is not None else "unchanged"] += 1
    except Exception as exc:  # noqa: BLE001 - the keeper itself failed (listing its sources); recorded, the others still run
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["trace"] = traceback.format_exc()[-4000:]
    out["seconds"] = round(time.monotonic() - started, 3)
    return out


def conduct(env: Any = None, layout: Optional[Layout] = None, keepers: Optional[dict[str, Keeper]] = None) -> dict[str, Any]:
    """One run of the store's agents over ``env`` (a mapping: ``agent_runs`` the workspaces' folder, ``services`` the running
    services, ``test_runs``, and ``only`` for one workspace before its removal). Returns the run's record."""
    from ..agents import harness
    from .keepers import standard

    layout = layout or Layout()
    store = Store(layout)
    keepers = standard.install() if keepers is None else keepers
    env = dict(env) if isinstance(env, dict) else {"services": env} if env is not None else {}
    env.setdefault("store_root", str(layout.root))  # the keepers that keep local state beside the store (the people keeper's salt) read it
    scope = "one" if env.get("only") else "all"
    run = f"store-{scope}-{uuid.uuid4().hex[:12]}"
    record: dict[str, Any] = {"id": run, "scope": scope, "started": now(), "context": context(), "steps": []}
    with harness.session("conductor", run_id=run) as session:
        for keeper in keepers.values():
            step = _step(keeper, env, store, run)
            record["steps"].append(step)
            session.step(f"keep:{keeper.name}", {"kinds": step["kinds"]},
                         {k: step[k] for k in ("read", "kept", "unchanged", "error")} | {"refused": len(step["refused"])}, step["seconds"],
                         verdict="failed" if step["error"] else "ok")
        if scope == "all":  # tier 2 after tier 1 (a removal's run keeps only; the next whole run shapes it)
            started = time.monotonic()
            shaped: dict[str, Any] = {"name": "shapers", "stage": "shape", "error": None, "errors": [], "read": 0, "kept": 0, "unchanged": 0}
            try:
                from . import shapers

                report = shapers.shape(layout)
                shaped.update(records=report["records"], unshaped=report["unshaped"], unchanged_kinds=report["unchanged"])
            except Exception as exc:  # noqa: BLE001 - the raw tier stands; the next run shapes again
                shaped["error"] = f"{type(exc).__name__}: {exc}"
                shaped["trace"] = traceback.format_exc()[-4000:]
            shaped["seconds"] = round(time.monotonic() - started, 3)
            record["steps"].append(shaped)
            session.step("shape", {}, {k: shaped.get(k) for k in ("records", "unshaped", "error")}, shaped["seconds"],
                         verdict="failed" if shaped["error"] else "ok")
            started = time.monotonic()  # the curators after the shapers: each record placed for the models it can teach
            curated: dict[str, Any] = {"name": "curators", "stage": "curate", "error": None, "errors": [], "read": 0, "kept": 0, "unchanged": 0}
            try:
                from . import curators

                shadow = None
                if os.environ.get("DCLAB_STORE_CURATOR_MODEL") == "shadow":  # Claude beside the rules, never used (costs: off unless asked)
                    from .curators import model as curator_model

                    shadow = curator_model.ask
                report = curators.curate(layout, run=f"{run}-curation", model=shadow)
                curated.update(decisions=report["decisions"], shadow=report["shadow"], by_model=report["by_model"], quarantined=report["quarantined"])
            except Exception as exc:  # noqa: BLE001
                curated["error"] = f"{type(exc).__name__}: {exc}"
                curated["trace"] = traceback.format_exc()[-4000:]
            curated["seconds"] = round(time.monotonic() - started, 3)
            record["steps"].append(curated)
            session.step("curate", {}, {k: curated.get(k) for k in ("decisions", "shadow", "error")}, curated["seconds"],
                         verdict="failed" if curated["error"] else "ok")
            started = time.monotonic()  # the sets after the curators: every model type's training sets rebuilt from the records
            built: dict[str, Any] = {"name": "builders", "stage": "build", "error": None, "errors": [], "read": 0, "kept": 0, "unchanged": 0, "sets": {}}
            from .sets import dclab_code as dclab_code_sets, decisions as decision_sets, experiments as experiment_sets, llm as llm_sets, others as other_sets

            for name, builder in (("decisions", decision_sets.build), ("llm", llm_sets.build), ("others", other_sets.build),
                                  ("experiments", experiment_sets.build), ("dclab_code", dclab_code_sets.build)):
                try:
                    built["sets"].update(builder(layout)["sets"])
                except Exception as exc:  # noqa: BLE001 - one builder's failure leaves the others' sets standing
                    built["errors"].append({"source": name, "error": f"{type(exc).__name__}: {exc}"})
            try:  # the tutor after the sets: every point's student refitted and measured, never promoted
                from . import tutor

                built["students"] = tutor.teach(layout, run=f"{run}-tutor")["students"]
            except Exception as exc:  # noqa: BLE001
                built["errors"].append({"source": "tutor", "error": f"{type(exc).__name__}: {exc}"})
            try:  # 32.24 to 32.31: the builds and their verdicts, the curves, the catalog, the engagements and the playbook
                from . import service

                built["service"] = service.write(layout)
            except Exception as exc:  # noqa: BLE001
                built["errors"].append({"source": "service", "error": f"{type(exc).__name__}: {exc}"})
            built["seconds"] = round(time.monotonic() - started, 3)
            record["steps"].append(built)
            session.step("build", {}, {"sets": len(built["sets"]), "errors": len(built["errors"])}, built["seconds"],
                         verdict="failed" if built["errors"] else "ok")
    record["ended"] = now()
    record["failed"] = [s["name"] for s in record["steps"] if s["error"]]
    record["source_errors"] = sum(len(s["errors"]) for s in record["steps"])
    write_json(layout.run_record(record["started"][:10], run), record)
    if scope == "all":
        try:  # the catalog, the README and the schemas follow every whole run (the repository's copy only for the owner's store)
            from . import catalog

            catalog.write(layout)
        except Exception:  # noqa: BLE001 - the catalog is rebuilt by the next run
            traceback.print_exc()
    return record


def failed(layout: Optional[Layout] = None, why: str = "") -> None:
    """A whole run that failed before it could record itself: recorded, so the next one waits its hour (no retry every minute)."""
    layout = layout or Layout()
    stamp = now()
    write_json(layout.run_record(stamp[:10], f"store-all-failed-{uuid.uuid4().hex[:8]}"), {"scope": "all", "started": stamp, "failed": True, "why": why})


def _last(layout: Layout) -> Optional[float]:
    """When the last whole run (or a whole run's failure) was recorded: a removal's one-workspace run does not count."""
    base = layout.runs()
    files = sorted(base.rglob("store-all-*.json"), key=lambda p: p.stat().st_mtime) if base.exists() else []
    return files[-1].stat().st_mtime if files else None


def due(layout: Optional[Layout] = None, clock: Optional[float] = None) -> bool:
    if os.environ.get("DCLAB_STORE_LOOP", "1") == "0":
        return False
    last = _last(layout or Layout())
    return last is None or (clock or time.time()) - last >= EVERY_SECONDS


def sweep_due(services: Any) -> Optional[dict[str, Any]]:
    """The job ``store``, queued when a run is due (one at a time: a queued or running one holds its key)."""
    from ..jobs import ActiveJob

    if getattr(services, "primary", True) is False:  # one run covers every workspace: only the server's own queues it, never each account's
        return None
    if not due():
        return None
    try:
        return services.queue_job("store", "conduct", {}, by="agent")
    except ActiveJob:
        return None


__all__ = ["conduct", "due", "sweep_due", "EVERY_SECONDS"]
