"""Workflows as records (32.5): each walk from a draft to a report as one :class:`dclab_core.store.WorkflowRecord`.

``walks(services)`` reads, for every draft (and the project it became): the draft's events, the project's graph moves (a
refused move kept as refused), its stage runs and their status, G4, the Workflow Run's twelve steps, the ledger's decisions
about the draft and the project, and the agent runs recorded for them, all in time order. The keeper ``walks`` keeps each
record under ``raw/workflows/`` as ``walks/<draft or project>.json``, a new version whenever the walk moved on. Each step
holds names, codes, statuses and counts: never a row of the table or the person's words (those stay in their own records).
"""

from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from dclab_core.store import WalkStep, WorkflowRecord


def _safe(make: Any, default: Any = None) -> Any:
    try:
        return make()
    except Exception:  # noqa: BLE001 - a part that does not read leaves that part out, never the walk
        return default


def _ledger_records(ledger: Any, project: Optional[str], draft: Optional[str]) -> list[Any]:
    if ledger is None:
        return []
    found = {}
    for kw in ({"project": project} if project else None, {"draft": draft} if draft else None):
        for r in _safe(lambda: ledger.records(**kw), []) if kw else []:
            found[r.id] = r
    return sorted(found.values(), key=lambda r: r.at)


def _runs(log: Any, project: Optional[str], draft: Optional[str]) -> list[tuple[str, str]]:
    """(time, run id) of every agent run recorded for the project or the draft's conversation."""
    if log is None:
        return []
    events = (_safe(lambda: log.read(project=project, kinds=["run.recorded"]), []) if project else []) + \
             (_safe(lambda: log.read(conversation=draft, kinds=["run.recorded"]), []) if draft else [])
    return sorted({(str(e.at), e.ids.run) for e in events if getattr(e.ids, "run", None)})


def walk(services: Any, draft: Optional[dict[str, Any]], project: Optional[dict[str, Any]]) -> WorkflowRecord:
    store, drafts = services.projects, services.drafts
    draft_id, project_id = (draft or {}).get("id"), (project or {}).get("id")
    steps: list[WalkStep] = []
    for e in _safe(lambda: drafts.events(draft_id), []) if draft_id else []:
        data = e.get("data") or {}
        steps.append(WalkStep(at=str(e.get("at")), kind="draft_event", ref=f"{draft_id}:{e.get('seq')}",
                              detail={"kind": e.get("kind"), **{k: data[k] for k in ("stage", "status", "step", "asset_id") if k in data}}))
    run = None
    if project_id:
        for t in reversed(_safe(lambda: store.transitions(project_id, limit=100_000), [])):
            steps.append(WalkStep(at=str(t.get("at")), kind="move" if t.get("status") == "allowed" else "refused", ref=str(t.get("move")),
                                  detail={k: t.get(k) for k in ("from", "to", "status", "actor", "rules", "failed_checks")}))
        for stage, rec in (_safe(lambda: store.records(project_id), {}) or {}).items():
            at = rec.get("finished") or rec.get("at") or rec.get("started")
            if at:
                steps.append(WalkStep(at=str(at), kind="stage", ref=stage, detail={"status": rec.get("status")}))
        from ..studio import workflow_run

        run = _safe(lambda: workflow_run.build(store, project).model_dump(mode="json"))
    records = _ledger_records(getattr(services, "ledger", None), project_id, draft_id)
    steps += [WalkStep(at=str(r.at), kind="decision", ref=r.id, detail={"point": r.point, "used": r.used}) for r in records]
    runs = _runs(getattr(services, "experience", None), project_id, draft_id)
    steps += [WalkStep(at=at, kind="agent_run", ref=rid) for at, rid in runs]
    steps.sort(key=lambda s: s.at)
    final = ((_safe(lambda: store.records(project_id), {}) or {}).get("final") or {}) if project_id else {}
    g4 = (final.get("evidence") or {}).get("gate_g4")
    outcome = "reported" if final.get("status") == "done" else "built" if project_id else "stopped_in_draft"
    return WorkflowRecord(id=f"walk:{draft_id or project_id}", draft=draft_id, project=project_id, started=steps[0].at if steps else None,
                          ended=steps[-1].at if steps else None, steps=steps, workflow_run=run, gates={"g4": g4} if g4 else {},
                          decisions=[r.id for r in records], agent_runs=[rid for _, rid in runs], outcome=outcome)


def walks(services: Any) -> Iterator[WorkflowRecord]:
    """Every walk of the workspace: each draft with the project it became, then each project that came from no draft."""
    projects = {p["id"]: p for p in _safe(lambda: services.projects.list(), []) or []}
    seen = set()
    for draft in _safe(lambda: services.drafts.list(limit=100_000), []) or []:
        pid = draft.get("project_id")
        full = _safe(lambda: services.drafts.get(draft["id"]), draft)
        project = _safe(lambda: services.projects.get(pid)) if pid in projects else None
        seen.add(pid)
        yield walk(services, full, project)
    for pid in projects:
        if pid not in seen:
            yield walk(services, None, _safe(lambda: services.projects.get(pid)))


class _Nothing:
    """A store a folder does not have: it lists nothing and holds nothing."""

    def list(self, *a: Any, **k: Any) -> list[Any]:
        return []

    def get(self, *a: Any, **k: Any) -> None:
        return None

    events = transitions = list

    def records(self, *a: Any, **k: Any) -> dict[str, Any]:
        return {}


class _Folder:
    """A file workspace's stores read from its folder alone, opening only the folders it has (so reading never adds one): what
    ``walks`` needs when no server runs (a command-line or backfill run of the conductor)."""

    def __init__(self, home: Any):
        from pathlib import Path

        from ..draft.store import DraftStore
        from ..experience import open_log
        from ..ledger import open_ledger
        from ..studio.store import ProjectStore

        home = Path(home)
        self.store = type("Home", (), {"home": home})()
        self.projects = ProjectStore(home / "projects") if (home / "projects").is_dir() else _Nothing()
        self.drafts = DraftStore(home / "drafts") if (home / "drafts").is_dir() else _Nothing()
        self.ledger = open_ledger(home) if (home / "ledger").is_dir() else None
        self.experience = open_log(home) if (home / "experience").is_dir() else None


def _folder_sources(env: dict[str, Any]) -> Iterator[Any]:
    """Without a running server: every file workspace under ``agent_runs`` whose consent keeps content, its walks read from its
    folder (a PostgreSQL workspace's walks are kept by its server's own run)."""
    from ..storage import db
    from .keepers import Source
    from .keepers.workspaces import AGENT_RUNS, discover

    if db.database_url(required=False):
        return
    for ws in discover(env.get("agent_runs") or AGENT_RUNS):
        folder = ws.root  # a workspace's own folder, never a store path
        if ws.consent == "operate" or not ((folder / "projects").is_dir() or (folder / "drafts").is_dir()):
            continue
        services = _safe(lambda: _Folder(folder))
        if services is None:
            continue
        for record in _safe(lambda: list(walks(services)), []) or []:
            yield Source(ws.name, f"walks/{(record.draft or record.project)}.json", json.dumps(record.model_dump(mode="json"), indent=1).encode(),
                         consent=ws.consent)


def keeper_sources(env: Any) -> Iterator[Any]:
    """The keeper's sources: each walk of the services' workspace as ``walks/<id>.json`` (a new version when the walk moved on);
    without services, each file workspace's walks read from its folder."""
    from .keepers import Source

    services = env.get("services") if isinstance(env, dict) else env
    if services is None or not hasattr(services, "drafts"):
        if isinstance(env, dict):
            yield from _folder_sources(env)
        return
    from .keepers.workspaces import of

    home = getattr(getattr(services, "store", None), "home", None)
    ws = of(home) if home else None
    if ws is None or ws.consent == "operate":
        return  # no folder to name it by, or a consent that keeps no content: its walks are not kept
    for record in walks(services):
        yield Source(ws.name, f"walks/{(record.draft or record.project)}.json", json.dumps(record.model_dump(mode="json"), indent=1).encode(),
                     consent=ws.consent)


__all__ = ["walk", "walks", "keeper_sources"]
