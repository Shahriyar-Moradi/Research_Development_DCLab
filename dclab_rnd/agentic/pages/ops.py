"""Compute & jobs, and Home's Activity tab: what ran on this machine, what runs now, what is new in the evidence.

There is no sandbox or GPU job system yet. What really runs, and is listed here as a job:

- ``stage``     a project stage run (data, leakage, features, models, final), from the project's activity log
                (``stage_started`` / ``stage_completed`` / ``stage_failed``); the latest completed run of a stage
                owns the stage record on disk (notes, claims, provenance)
- ``intern``    an intern session (budget of tool calls and minutes, steps, tokens)
- ``data``      one asset of a draft's data pipeline on Home (structure → clean → analyse)
- ``research``  a legacy LLM research run (the SQLite store behind /classic)

Every job runs on this machine and nothing is metered, so ``cost`` is always null. Routes are read-only.
"""

from __future__ import annotations

import asyncio
import functools
import json
import os
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException

from ...evidence_index import INDEX_PATH, ROOT
from ...studio import engine as studio_engine
from ...studio import graph as studio_graph
from ...studio.store import STAGE_KEYS

WHERE = "this machine"
RECORDS = ROOT / INDEX_PATH  # the evidence index /api/evidence/{id} serves (via tools.get_record)
DATED_SOURCE = re.compile(r"^evidence/campaigns/[^/]+/results/[^/]+\.json$")
RECORD_ID = re.compile(r"\b(DCLAB-R\d{2}|WF-\d{2}|EXP-\d{3}|PIT-\d{3}|CAT-\d{3}|LEAK-[a-z_]+|FINDING-[a-z-]+|DATASET-[a-z_]+)\b")
DATA_RUNNING = ("structuring", "cleaning", "analysing")
# Raw status → (bucket, label). Buckets are what the page counts: running | queued | done | failed.
INTERN_STATUS = {"queued": ("queued", "queued"), "running": ("running", "running"), "completed": ("done", "done"),
                 "failed": ("failed", "failed"), "budget_exhausted": ("done", "budget used up"), "stopped": ("done", "stopped")}
RESEARCH_STATUS = {"queued": ("queued", "queued"), "running": ("running", "running"), "pausing": ("running", "pausing"),
                   "completed": ("done", "done"), "paused": ("done", "paused"), "budget_exhausted": ("done", "budget used up"),
                   "interrupted": ("failed", "interrupted"), "failed": ("failed", "failed")}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat()


def _parse(value: Any) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _span(start: Any, end: Any) -> float | None:
    a, b = _parse(start), _parse(end)
    return round(max((b - a).total_seconds(), 0.0), 2) if a and b else None


def _short(text: Any, n: int = 160) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def _active(tasks: dict, key: str) -> bool:
    task = tasks.get(key)
    return task is not None and not task.done()


def _job(**values: Any) -> dict[str, Any]:
    base = {"id": None, "kind": None, "title": "", "subtitle": "", "project_id": None, "project": None, "draft_id": None,
            "session_id": None, "run_id": None, "stage": None, "status": "done", "label": "done", "started": None,
            "finished": None, "seconds": None, "where": WHERE, "cost": None, "by": None, "detail": ""}
    return {**base, **values}


# ---------------------------------------------------------------------- evidence: newest records


@functools.lru_cache(maxsize=4)
def _evidence_sorted(path: str, mtime_ns: int, size: int) -> tuple[dict[str, Any], ...]:
    """All records, newest first. Cached per file version (mtime and size): the index only changes on a rebuild.

    records.jsonl carries no date. A record that cites a campaign result (evidence/campaigns/*/results/*.json, which
    are immutable) takes that result's completed_at as its date. Records without such a source (rules, workflow
    blocks, dataset cards, cross-dataset findings) have no date at all: they follow the dated ones in reverse file
    order, which is file order and nothing more; ``ordered_by`` says which applies to each record.
    """
    rows = []
    for position, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        r = json.loads(line)
        meta = r.get("metadata") or {}
        date = None
        for cite in r.get("citations") or []:
            if isinstance(cite, str) and DATED_SOURCE.match(cite):
                try:
                    result = json.loads((ROOT / cite).read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                date = result.get("completed_at") or result.get("started_at")
                if date:
                    break
        related = []
        if isinstance(meta.get("rule"), str):
            related.append(meta["rule"])
        datasets = [meta["dataset"]] if isinstance(meta.get("dataset"), str) else list(meta.get("datasets") or [])
        related += [f"DATASET-{d}" for d in datasets[:2]]
        related += [m for m in RECORD_ID.findall(r.get("text") or "")]
        related = list(dict.fromkeys(x for x in related if x != r["record_id"]))
        rows.append({"id": r["record_id"], "type": r.get("type"), "title": r.get("title") or r["record_id"],
                     "text": _short(r.get("text"), 220), "date": date, "campaign": meta.get("campaign"),
                     "stage": meta.get("stage"), "related": related, "ordered_by": "date" if date else "file order",
                     "_pos": position})
    known = {r["id"] for r in rows}
    for r in rows:  # a related ID is only kept when the index can open it
        r["related"] = [x for x in r["related"] if x in known][:4]
    dated = sorted((r for r in rows if r["date"]), key=lambda r: (r["date"], r["_pos"]), reverse=True)
    undated = sorted((r for r in rows if not r["date"]), key=lambda r: r["_pos"], reverse=True)
    return tuple({k: v for k, v in r.items() if k != "_pos"} for r in dated + undated)


def recent_evidence(limit: int = 3, path: Path | None = None) -> dict[str, Any]:
    path = Path(path or RECORDS)
    if not path.is_file():
        return {"records": [], "total": 0, "source": str(INDEX_PATH), "note": "The evidence index is not built yet: make knowledge."}
    st = path.stat()
    rows = _evidence_sorted(str(path), st.st_mtime_ns, st.st_size)
    return {"records": list(rows[: max(1, min(int(limit), 20))]), "total": len(rows), "source": str(INDEX_PATH),
            "note": "Newest first by the completed_at of the campaign result a record cites; records with no dated "
                    "source keep their order in records.jsonl."}


# ---------------------------------------------------------------------- jobs


class Jobs:
    """Builds the unified job list from the server's stores. One instance per server; ``started`` is the
    server's start time: a job that says it is running but started before this server cannot be running now."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.started = _now()

    # ------------------------------------------------------------------ stage runs
    def _project_runner(self, project: dict[str, Any]) -> str | None:
        """Who is running this project's stages now: "human" (a job the server started), "agent" (an intern session
        working on it) or None."""
        pid = project["id"]
        if _active(self.ctx.jobs, pid):
            return "human"
        for sid, task in list(self.ctx.intern_jobs.items()):
            if not task.done():
                try:
                    if self.ctx.intern_sessions.get(sid).get("project_id") == pid:
                        return "agent"
                except KeyError:
                    continue
        return None

    def stage_runs(self, project: dict[str, Any]) -> list[dict[str, Any]]:
        """Every stage run of one project, oldest first, from its activity log (and its records when the log is empty)."""
        projects, pid = self.ctx.projects, project["id"]
        stages = project.get("stages") or {}
        runs, open_ = [], {}
        for entry in projects.activity(pid, 100_000):
            kind, stage = entry.get("kind"), (entry.get("payload") or {}).get("stage")
            if stage not in STAGE_KEYS:
                continue
            if kind == "stage_started":
                if stage in open_:
                    runs.append((stage, open_.pop(stage), None))
                open_[stage] = entry
            elif kind in ("stage_completed", "stage_failed"):
                runs.append((stage, open_.pop(stage, None), entry))
        runs += [(stage, entry, None) for stage, entry in open_.items()]
        runs.sort(key=lambda r: (r[1] or r[2] or {}).get("at") or "")
        moves = [m for m in projects.transitions(pid, 100_000) if m.get("move") == "run_stage" and m.get("status") == "allowed"]
        used: set[int] = set()
        has_record = {s: projects.stage_path(pid, s).exists() for s in STAGE_KEYS}
        last_completed = {}
        for i, (stage, _start, end) in enumerate(runs):
            if end and end.get("kind") == "stage_completed":
                last_completed[stage] = i
        counts: dict[str, int] = {}
        runner = self._project_runner(project)
        out = []
        for i, (stage, start, end) in enumerate(runs):
            counts[stage] = counts.get(stage, 0) + 1
            meta = studio_engine.STAGE_BY_KEY[stage]
            started = (start or {}).get("at")
            finished = (end or {}).get("at")
            payload = (end or {}).get("payload") or {}
            current = has_record[stage] and last_completed.get(stage) == i
            if end and end.get("kind") == "stage_completed":
                status, label = "done", "done"
                seconds = (stages.get(stage) or {}).get("elapsed_seconds") if current else payload.get("elapsed_seconds")
                detail = _short(payload.get("summary"), 200)
            elif end:
                status, label, seconds = "failed", "failed", _span(started, finished)
                detail = _short(payload.get("error") or "The stage failed.", 200)
            else:
                live = (stages.get(stage) or {}).get("status") == "running" and (runner or ((_parse(started) or self.started) >= self.started))
                is_last = all(r[0] != stage for r in runs[i + 1:])
                if live and is_last:
                    status, label, seconds = "running", "running", _span(started, _iso(_now()))
                    detail = "Running now on this machine."
                else:
                    status, label, seconds = "failed", "interrupted", None
                    detail = "Started but no end was recorded: the server stopped or the run was superseded."
            by = None  # who moved: the graph logs each run_stage move once the run ends (an unfinished run has none yet)
            for j, m in enumerate(moves if end else []):
                if j not in used and (m.get("args") or {}).get("stage") == stage and (m.get("at") or "") >= (started or ""):
                    used.add(j)
                    by = "agent" if m.get("actor") == "agent" else "human"
                    break
            if status == "running":
                by = runner
            if status != "done":
                current = False
            out.append(_job(id=f"stage-{pid}-{stage}-{counts[stage]}", kind="stage", title=meta["title"], subtitle=f"{meta['workflow']} · run {counts[stage]}",
                            project_id=pid, project=project.get("name"), stage=stage, status=status, label=label,
                            started=started, finished=finished if status != "running" else None,
                            seconds=round(float(seconds), 2) if seconds is not None else None, by=by, detail=detail,
                            current=current, superseded=status == "done" and not current))
        # A run the graph allowed but that failed while preparing the data leaves no stage_started entry, only its move.
        for j, m in enumerate(moves):
            stage = (m.get("args") or {}).get("stage")
            if j in used or stage not in STAGE_KEYS or not str(m.get("outcome") or "").startswith("failed"):
                continue
            meta = studio_engine.STAGE_BY_KEY[stage]
            out.append(_job(id=f"stage-{pid}-{stage}-m{j}", kind="stage", title=meta["title"], subtitle=f"{meta['workflow']} · failed before it started",
                            project_id=pid, project=project.get("name"), stage=stage, status="failed", label="failed",
                            started=m.get("at"), finished=m.get("at"), by="agent" if m.get("actor") == "agent" else "human",
                            detail=_short(str(m["outcome"]).removeprefix("failed: "), 200), current=False, superseded=False))
        # A stage with a record but no activity (older projects): one run from the record itself.
        logged = {r[0] for r in runs}
        for stage in STAGE_KEYS:
            if stage in logged or not has_record[stage]:
                continue
            record = projects.read_stage(pid, stage) or {}
            meta = studio_engine.STAGE_BY_KEY[stage]
            out.append(_job(id=f"stage-{pid}-{stage}-1", kind="stage", title=meta["title"], subtitle=f"{meta['workflow']} · run 1",
                            project_id=pid, project=project.get("name"), stage=stage, started=record.get("started_at"),
                            finished=record.get("completed_at"), seconds=record.get("elapsed_seconds"),
                            detail=_short(record.get("setup_summary"), 200), current=True, superseded=False))
        # Stages waiting in a job that is still alive (a run of several stages runs them one by one).
        if _active(self.ctx.jobs, pid):
            for stage in STAGE_KEYS:
                if (stages.get(stage) or {}).get("status") == "queued":
                    meta = studio_engine.STAGE_BY_KEY[stage]
                    out.append(_job(id=f"stage-{pid}-{stage}-q", kind="stage", title=meta["title"], subtitle=f"{meta['workflow']} · waiting",
                                    project_id=pid, project=project.get("name"), stage=stage, status="queued", label="queued",
                                    by="human", detail="Waits for the stage before it in the same run.", queued_at=project.get("updated")))
        return out

    # ------------------------------------------------------------------ intern sessions
    def intern_job(self, s: dict[str, Any]) -> dict[str, Any]:
        status, label = INTERN_STATUS.get(s.get("status"), ("done", str(s.get("status") or "unknown")))
        live = status in ("running", "queued")
        if live and not _active(self.ctx.intern_jobs, s["id"]) and (_parse(s.get("updated")) or self.started) < self.started:
            status, label = "failed", "interrupted"
        used, budget = s.get("used") or {}, s.get("budget") or {}
        steps = s["steps"] if isinstance(s.get("steps"), int) else len(s.get("steps") or [])
        minutes = used.get("minutes")  # measured by the loop, but only written at each step: a live session counts from its start
        if status in ("running", "queued"):
            seconds = _span(s.get("created"), _iso(_now()))
        else:
            seconds = round(float(minutes) * 60, 1) if minutes else _span(s.get("created"), s.get("updated"))
        mode = "standard plan" if s.get("mode") != "llm" else f"model · {s.get('model') or 'configured model'}"
        detail = f"{steps} of {budget.get('max_steps', '?')} tool calls · {mode}"
        if s.get("error"):
            detail += " · " + _short(s["error"], 120)
        return _job(id=f"intern-{s['id']}", kind="intern", title=_short(s.get("task"), 90), subtitle=f"Intern session · {mode}",
                    project_id=s.get("project_id"), session_id=s["id"], status=status, label=label, started=s.get("created"),
                    finished=None if status in ("running", "queued") else s.get("updated"), seconds=seconds, by="agent", detail=detail,
                    tokens={"input": int(used.get("input_tokens") or 0), "output": int(used.get("output_tokens") or 0)})

    # ------------------------------------------------------------------ draft data pipelines
    @staticmethod
    @functools.lru_cache(maxsize=256)
    def _pipeline_events(path: str, mtime_ns: int, size: int) -> tuple[dict[str, Any], ...]:
        """The draft's ``pipeline`` events (the other kinds are skipped unread). Cached per file version."""
        out = []
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if '"pipeline"' not in line[:200]:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("kind") == "pipeline":
                    out.append(event)
        return tuple(out)

    def pipeline_events(self, draft_id: str) -> tuple[dict[str, Any], ...]:
        path = self.ctx.drafts.directory(draft_id) / "events.jsonl"
        if not path.is_file():
            return ()
        st = path.stat()
        return self._pipeline_events(str(path), st.st_mtime_ns, st.st_size)

    def data_jobs(self, draft: dict[str, Any]) -> list[dict[str, Any]]:
        did = draft["id"]
        events = self.pipeline_events(did)
        synth_live = _active(self.ctx.draft_jobs, f"{did}:synthetic")
        out = []
        for a in draft.get("assets") or []:
            raw = a.get("status") or "queued"
            status = "running" if raw in DATA_RUNNING else {"ready": "done", "failed": "failed", "queued": "queued"}.get(raw, "done")
            label = {"ready": "done"}.get(raw, raw)
            if status in ("running", "queued") and not (_active(self.ctx.draft_jobs, f"{did}:{a['id']}") or synth_live) \
                    and (_parse(a.get("added")) or self.started) < self.started:
                status, label = "failed", "interrupted"
            ends = [e for e in events if _asset_of(e) == a["id"] and (e.get("data") or {}).get("step") in ("ready", "failed")]
            finished = ends[-1]["at"] if ends and status in ("done", "failed") else None
            seconds = _span(a.get("added"), finished or (_iso(_now()) if status == "running" else None))
            if a.get("status") == "failed":
                detail = _short(a.get("error") or "The file could not be processed.", 200)
            elif a.get("rows") is not None:
                detail = f"{a['rows']:,} rows · {a.get('columns', '?')} columns" + (" · synthetic" if a.get("synthetic") else "")
            else:
                detail = {"queued": "Waiting to start.", "structuring": "Reading the file", "cleaning": "Cleaning", "analysing": "Describing the table"}.get(raw, raw)
            out.append(_job(id=f"data-{did}-{a['id']}", kind="data", title=a.get("name") or a.get("filename") or "data",
                            subtitle=f"Data pipeline · {a.get('kind', 'upload')}", project_id=draft.get("project_id"), draft_id=did,
                            project=_short(draft.get("problem"), 90), status=status, label=label, started=a.get("added"), finished=finished,
                            seconds=seconds, by="human", detail=detail, rows=a.get("rows"), draft_open=draft.get("status") == "open"))
        if synth_live and not any(a.get("synthetic") and a.get("status") not in ("ready", "failed") for a in draft.get("assets") or []):
            design = [e for e in events if (e.get("data") or {}).get("step") == "designing"]
            started = design[-1]["at"] if design else None
            out.append(_job(id=f"data-{did}-synthetic", kind="data", title="Synthetic data · designing", subtitle="Data pipeline · synthetic",
                            project_id=draft.get("project_id"), draft_id=did, project=_short(draft.get("problem"), 90), status="running",
                            label="designing", started=started, seconds=_span(started, _iso(_now())), by="human",
                            detail="Designing a synthetic dataset from the conversation.", draft_open=draft.get("status") == "open"))
        return out

    # ------------------------------------------------------------------ legacy research runs
    def research_job(self, run: dict[str, Any]) -> dict[str, Any]:
        status, label = RESEARCH_STATUS.get(run.get("status"), ("done", str(run.get("status") or "unknown")))
        cfg = run.get("config") or {}
        usage = run.get("usage") or {}
        detail = _short(run.get("error") or run.get("phase") or "", 200)
        return _job(id=f"research-{run['id']}", kind="research", title=_short(cfg.get("goal") or "Research run", 90),
                    subtitle="Research run · " + ", ".join(cfg.get("datasets") or []), project=cfg.get("project"), run_id=run["id"],
                    status=status, label=label, started=run.get("created"), finished=None if status in ("running", "queued") else run.get("updated"),
                    seconds=round(float(run.get("active_seconds") or 0), 1) or None, by="agent", detail=detail,
                    tokens={"input": int(usage.get("input_tokens") or 0), "output": int(usage.get("output_tokens") or 0)})

    # ------------------------------------------------------------------ approvals
    def approvals(self, projects: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Gates a project requires and nobody has approved yet. The holdout gate stops counting once the final stage
        is done (its approval is spent on that run); the solution gate needs a saved solution before it can be signed."""
        out = []
        for p in projects:
            stages = p.get("stages") or {}
            for g in studio_graph.describe(p)["gates"]:
                if not g["required"] or g["approved"]:
                    continue
                if g["gate"] == "holdout" and (stages.get("final") or {}).get("status") in ("completed", "approved"):
                    continue
                if g["gate"] == "solution" and not p.get("solution"):
                    continue
                out.append({"project_id": p["id"], "project": p.get("name"), "gate": g["gate"], "what": g["what"],
                            "page": "solution" if g["gate"] == "solution" else "project", "since": p.get("updated")})
        return out

    # ------------------------------------------------------------------ everything
    def collect(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        projects = self.ctx.projects.list()
        jobs: list[dict[str, Any]] = []
        for p in projects:
            jobs += self.stage_runs(p)
        names = {p["id"]: p.get("name") for p in projects}
        for s in self.ctx.intern_sessions.list():
            job = self.intern_job(s)
            job["project"] = names.get(job["project_id"]) if job["project_id"] else None
            jobs.append(job)
        for d in self.ctx.drafts.list(500):
            jobs += self.data_jobs(d)
        for run in self.ctx.store.list():
            jobs.append(self.research_job(run))
        jobs.sort(key=_sort_key, reverse=True)
        return jobs, projects

    def listing(self, limit: int = 200) -> dict[str, Any]:
        jobs, projects = self.collect()
        now = _now()
        month = now.strftime("%Y-%m")
        by_kind = {k: 0.0 for k in ("stage", "intern", "data", "research")}
        tokens = {"input": 0, "output": 0}
        for j in jobs:
            if (j.get("started") or "")[:7] == month:
                by_kind[j["kind"]] += float(j.get("seconds") or 0)
                for k in tokens:
                    tokens[k] += int((j.get("tokens") or {}).get(k) or 0)
        def count(status: str) -> int:
            return sum(1 for j in jobs if j["status"] == status)
        approvals = self.approvals(projects)
        return {
            "jobs": jobs[: max(1, min(int(limit), 1000))],
            "totals": {"total": len(jobs), "running": count("running"), "queued": count("queued"), "failed": count("failed"),
                       "done": count("done"), "approvals": len(approvals), "month": month,
                       "seconds_this_month": {k: round(v, 1) for k, v in by_kind.items()}, "tokens_this_month": tokens,
                       "by_kind": {k: sum(1 for j in jobs if j["kind"] == k) for k in by_kind}},
            "approvals": approvals,
            "machine": {"name": WHERE, "cpus": os.cpu_count(), "system": platform.system(), "arch": platform.machine(),
                        "python": platform.python_version()},
            "spend": {"metered": False, "note": "Nothing is metered yet: every job runs on this machine and model calls are counted in tokens, not priced."},
            "targets": [{"key": "local", "name": "This machine", "state": "ready"},
                        {"key": "sandbox", "name": "Sandboxes", "state": "not switched on"},
                        {"key": "gpu", "name": "Hugging Face Jobs", "state": "not switched on"},
                        {"key": "cluster", "name": "Your cluster", "state": "not switched on"}],
            "generated": _iso(now),
        }

    # ------------------------------------------------------------------ one job in full
    def detail(self, job_id: str) -> dict[str, Any]:
        kind, _, rest = job_id.partition("-")
        if kind == "stage":
            return self._stage_detail(job_id, rest)
        if kind == "intern":
            return self._intern_detail(rest)
        if kind == "data":
            return self._data_detail(job_id, rest)
        if kind == "research":
            return self._research_detail(rest)
        raise KeyError(job_id)

    def _stage_detail(self, job_id: str, rest: str) -> dict[str, Any]:
        pid = rest.split("-", 1)[0]
        project = self.ctx.projects.get(pid)
        job = next((j for j in self.stage_runs(project) if j["id"] == job_id), None)
        if job is None:
            raise KeyError(job_id)
        stage = job["stage"]
        meta = studio_engine.STAGE_BY_KEY[stage]
        record = self.ctx.projects.read_stage(pid, stage) if job.get("current") else None
        logs = []
        if job["started"]:
            logs.append(_line(job["started"], "dim", f"{meta['workflow']} {meta['title']} started · {WHERE}"))
        if record:
            logs.append(_line(None, "dim", _short(record.get("setup_summary"), 400)))
            for n in record.get("notes") or []:
                level = {"warning": "warn", "danger": "bad", "error": "bad", "success": "ok"}.get(n.get("severity"), "acc")
                logs.append(_line(None, level, f"note · {n.get('title', '')}: {_short(n.get('text'), 300)}", n.get("proof")))
            for c in record.get("claims") or []:
                logs.append(_line(None, "dim", f"claim {c.get('claim_id', '')} ({c.get('kind', 'fact')}) · {_short(c.get('statement'), 400)}"))
                for lim in c.get("limitations") or []:
                    logs.append(_line(None, "warn", f"  limit · {_short(lim, 300)}"))
        if job["status"] == "done":
            logs.append(_line(job["finished"], "ok", f"completed in {_secs(job['seconds'])} · {job['detail']}"))
        elif job["status"] == "failed":
            logs.append(_line(job["finished"], "bad", f"{job['label']} · {job['detail']}"))
        elif job["status"] == "running":
            logs.append(_line(None, "acc", "running…"))
        else:
            logs.append(_line(None, "dim", "queued: waits for the stage before it"))
        note = None
        if job.get("superseded"):
            note = "A later run of this stage replaced its record (a project keeps the latest record of each stage), so only the activity log remains."
        metrics = [{"label": "Elapsed", "value": _secs(job["seconds"])}]
        artifacts, repro, repro_note = [], [], None
        if record:
            ev, prov = record.get("evidence") or {}, record.get("provenance") or {}
            sampling = prov.get("sampling") or {}
            if sampling.get("rows_used"):
                metrics.append({"label": "Rows used", "value": f"{sampling['rows_used']:,} of {sampling.get('source_rows', sampling['rows_used']):,}"})
            if ev.get("train_rows") is not None:
                metrics.append({"label": "Train · holdout rows", "value": f"{ev['train_rows']:,} · {ev.get('holdout_rows', 0):,}"})
            metric = record.get("primary_metric")
            if metric:
                metrics.append({"label": "Primary metric", "value": metric})
            if ev.get("selected_metric_mean") is not None:
                metrics.append({"label": f"Training-CV {metric or 'score'} of the choice", "value": f"{ev['selected_metric_mean']:.4f}"})
            ci = ev.get("holdout_primary_metric_ci") or {}
            if ci.get("point") is not None:
                metrics.append({"label": f"Holdout {metric or 'score'} (95% interval)", "value": f"{ci['point']:.4f} [{ci.get('low', float('nan')):.4f}, {ci.get('high', float('nan')):.4f}]"})
            decision = record.get("decision") or {}
            if decision.get("selected"):
                metrics.append({"label": "Choice", "value": str(decision["selected"])})
            metrics.append({"label": "Notes · claims", "value": f"{len(record.get('notes') or [])} · {len(record.get('claims') or [])}"})
            if project.get("solution"):
                base = f"/api/projects/{pid}/export"
                artifacts = [{"label": "Notebook (.ipynb)", "href": f"{base}/notebook", "note": "every stage, runnable; exporting logs a WF-10 capture move"},
                             {"label": "Report (.md)", "href": f"{base}/report", "note": "the project's results with their limits"},
                             {"label": "SFT examples (.jsonl)", "href": f"{base}/sft", "note": "training examples drawn from this project"}]
            repro = [{"label": "Record", "value": record.get("experiment_id"), "mono": True},
                     {"label": "Seed", "value": prov.get("random_state"), "mono": True},
                     {"label": "Data", "value": f"{prov.get('filename', '?')} · sha256 {prov.get('data_sha256', '?')}", "mono": True},
                     {"label": "Sampling", "value": sampling.get("rule")},
                     {"label": "Cross-validation", "value": prov.get("cv")},
                     {"label": "Holdout", "value": prov.get("holdout")},
                     {"label": "Prediction moment", "value": record.get("decision_time_rule")},
                     {"label": "Code version", "value": prov.get("code_version") or prov.get("git") or "not recorded in the stage record"}]
            repro = [r for r in repro if r["value"] not in (None, "")]
        else:
            repro_note = ("The record is written when the stage finishes." if job["status"] in ("running", "queued") else
                          "This run's record was replaced by a later run of the same stage." if job.get("superseded") else
                          "This run left no record: it failed; the error is in the logs." if job["label"] == "failed" else
                          "This run left no record: it did not finish.")
        return {"job": job, "logs": logs, "log_note": note, "metrics": metrics, "artifacts": artifacts,
                "artifact_note": None if artifacts else "No files: only a finished run that still owns its stage record has exports.",
                "repro": repro, "repro_note": repro_note, "stop": None}

    def _intern_detail(self, sid: str) -> dict[str, Any]:
        s = self.ctx.intern_sessions.get(sid)
        job = self.intern_job({k: v for k, v in s.items() if k != "messages"})
        if job["project_id"]:
            try:
                job["project"] = self.ctx.projects.get(job["project_id"]).get("name")
            except KeyError:
                pass
        logs = [_line(s.get("created"), "dim", f"task · {_short(s.get('task'), 300)}")]
        if s.get("plan"):
            logs.append(_line(None, "dim", "plan · " + _short(s["plan"], 400)))
        for st in s.get("steps") or []:
            ok = st.get("ok") is not False
            args = st.get("arguments") if isinstance(st.get("arguments"), dict) else {}
            shown = ", ".join(f"{k}={_short(json.dumps(v, ensure_ascii=False) if not isinstance(v, str) else v, 40)}" for k, v in args.items() if k != "project_id")
            logs.append(_line(st.get("at"), "ok" if ok else "bad", f"{st.get('tool')}({_short(shown, 120)}) → {_short(st.get('summary'), 150)} · {_secs(st.get('elapsed_seconds'))}"))
        if s.get("final"):
            logs.append(_line(s.get("updated"), "acc", "final · " + _short(s["final"] if isinstance(s["final"], str) else json.dumps(s["final"]), 400)))
        if s.get("error"):
            logs.append(_line(s.get("updated"), "bad", "error · " + _short(s["error"], 300)))
        used, budget = s.get("used") or {}, s.get("budget") or {}
        metrics = [{"label": "Tool calls", "value": f"{len(s.get('steps') or [])} of {budget.get('max_steps', '?')}"},
                   {"label": "Minutes", "value": f"{float(used.get('minutes') or 0):.2f} of {budget.get('max_minutes', '?')}"},
                   {"label": "Tokens in · out", "value": f"{int(used.get('input_tokens') or 0):,} · {int(used.get('output_tokens') or 0):,}"},
                   {"label": "Mode", "value": "standard plan (no model)" if s.get("mode") != "llm" else f"model · {s.get('model')}"}]
        artifacts = []
        if job["project_id"]:
            try:
                p = self.ctx.projects.get(job["project_id"])
                if p.get("solution") and self.ctx.projects.records(p["id"]):
                    base = f"/api/projects/{p['id']}/export"
                    artifacts = [{"label": "Notebook (.ipynb)", "href": f"{base}/notebook", "note": f"project {p.get('name')}; exporting logs a WF-10 capture move"},
                                 {"label": "Report (.md)", "href": f"{base}/report", "note": f"project {p.get('name')}"}]
            except KeyError:
                pass
        repro = [{"label": "Task", "value": s.get("task")},
                 {"label": "Planner", "value": "standard DCLab plan (deterministic)" if s.get("mode") != "llm" else f"model {s.get('model')}"},
                 {"label": "Budget", "value": f"{budget.get('max_steps')} tool calls · {budget.get('max_minutes')} minutes"},
                 {"label": "Project", "value": job["project_id"], "mono": True}]
        return {"job": job, "logs": logs, "log_note": None, "metrics": metrics, "artifacts": artifacts,
                "artifact_note": None if artifacts else "No files: this session has no project with results yet.",
                "repro": [r for r in repro if r["value"]],
                "repro_note": "The same task and budget replay the standard plan step for step on the same data; a model's plan can differ between runs.",
                "stop": None}

    def _data_detail(self, job_id: str, rest: str) -> dict[str, Any]:
        did = rest.split("-", 1)[0]
        draft = self.ctx.drafts.get(did)
        job = next((j for j in self.data_jobs(draft) if j["id"] == job_id), None)
        if job is None:
            raise KeyError(job_id)
        aid = job_id.split("-", 2)[2]
        asset = next((a for a in draft.get("assets") or [] if a["id"] == aid), {})
        logs = []
        for e in self.pipeline_events(did):
            data = e.get("data") or {}
            if _asset_of(e) != aid and not (aid == "synthetic" and data.get("step") == "designing"):
                continue
            step = data.get("step")
            level = {"ready": "ok", "failed": "bad", "queued": "dim"}.get(step, "acc")
            logs.append(_line(e.get("at"), level, f"{step} · {_short(data.get('text'), 300)}"))
            for entry in data.get("log") or []:
                logs.append(_line(None, "dim", f"  {entry.get('title', '')} · {_short(entry.get('detail'), 240)}"))
        structure = asset.get("structure") or {}
        metrics = [{"label": "Elapsed", "value": _secs(job["seconds"])}]
        for label, value in (("Rows", asset.get("rows")), ("Columns", asset.get("columns")), ("Bytes received", asset.get("bytes"))):
            if value is not None:
                metrics.append({"label": label, "value": f"{int(value):,}"})
        if structure.get("format"):
            metrics.append({"label": "Read as", "value": str(structure["format"]).replace("_", " ")})
        if structure.get("parse_rate") is not None:
            metrics.append({"label": "Lines parsed", "value": f"{float(structure['parse_rate']):.0%}"})
        if asset.get("category_codes") is not None:
            metrics.append({"label": "Category-code columns", "value": str(len(asset.get("category_codes") or []))})
        repro = [{"label": "Source", "value": asset.get("kind")}, {"label": "File", "value": asset.get("filename"), "mono": True}]
        for k, v in (asset.get("source") or {}).items():
            if isinstance(v, (str, int, float)) and v != "":
                repro.append({"label": k.replace("_", " "), "value": v, "mono": k in ("sha256", "ref", "version", "revision", "uri")})
        if asset.get("synthetic"):
            repro.append({"label": "Synthetic spec", "value": f"{asset.get('spec_file', '?')} · designed by {asset.get('spec_source') or '?'}", "mono": True})
        repro.append({"label": "Code version", "value": "not recorded for draft pipelines"})
        artifacts = []
        if draft.get("project_id"):
            artifacts.append({"label": "Project built from this draft", "href": None, "project_id": draft["project_id"],
                              "note": "the cleaned table became the project's data"})
        return {"job": job, "logs": logs, "log_note": None, "metrics": metrics, "artifacts": artifacts,
                "artifact_note": None if artifacts else "No files to download: the cleaned table stays with the draft and becomes the project's data when the draft is built.",
                "repro": [r for r in repro if r["value"] not in (None, "")], "repro_note": None, "stop": None}

    def _research_detail(self, run_id: str) -> dict[str, Any]:
        run = self.ctx.store.get(run_id)  # KeyError → 404
        job = self.research_job(run)
        logs, fingerprints = [], 0
        for e in self.ctx.store.events(run_id):
            k, p = e["kind"], e.get("payload") or {}
            if k in ("llm_request", "llm_usage"):
                continue
            text = _research_text(k, p)
            if k == "cycle_verification" and isinstance(p, dict):
                fingerprints = max(fingerprints, len(p.get("code_sha256") or {}))
            level = "bad" if k == "failure" or (k == "trial" and p.get("status") != "completed") else "ok" if k in ("trial", "synthesis") else "dim"
            logs.append(_line(e.get("created"), level, f"{k} · {_short(text, 260)}" if text else k))
        cfg, usage = run.get("config") or {}, run.get("usage") or {}
        metrics = [{"label": "Active time", "value": _secs(run.get("active_seconds"))},
                   {"label": "Model calls", "value": str(run.get("llm_calls", 0))},
                   {"label": "Tokens in · out", "value": f"{int(usage.get('input_tokens') or 0):,} · {int(usage.get('output_tokens') or 0):,}"}]
        repro = [{"label": "Goal", "value": cfg.get("goal")}, {"label": "Datasets", "value": ", ".join(cfg.get("datasets") or [])},
                 {"label": "Model", "value": cfg.get("model"), "mono": True},
                 {"label": "Limits", "value": f"{cfg.get('max_experiments')} experiments · {cfg.get('max_rows')} rows · {cfg.get('repeats')} repeats · {cfg.get('max_minutes')} minutes"},
                 {"label": "Code version", "value": f"{fingerprints} source-file fingerprints recorded" if fingerprints else "not recorded"}]
        return {"job": job, "logs": logs, "log_note": None, "metrics": metrics,
                "artifacts": [{"label": "Trajectory (.json)", "href": f"/api/runs/{run_id}/export", "note": "decisions and tool evidence, for review"}],
                "artifact_note": None, "repro": [r for r in repro if r["value"]], "repro_note": "LLM research runs are not deterministic; the trajectory records what was decided and why.",
                "stop": None}


def _research_text(kind: str, p: Any) -> str:
    """One readable line for a legacy research event (model requests and usage are skipped by the caller)."""
    if kind == "profiles":
        return ", ".join(x.get("dataset", "") for x in p if isinstance(x, dict)) if isinstance(p, list) else ""
    if not isinstance(p, dict):
        return ""
    if kind == "trial":
        return f"{p.get('status')} · {(p.get('plan') or {}).get('title', '')}"
    key = {"phase": "name", "proposal": "title", "critique": "observation", "synthesis": "summary", "failure": "type", "agenda": "goal_interpretation"}.get(kind)
    return str(p.get(key) or "") if key else ""


def _asset_of(event: dict[str, Any]) -> str | None:
    a = (event.get("data") or {}).get("asset")
    return a.get("id") if isinstance(a, dict) else a


def _sort_key(job: dict[str, Any]) -> tuple[str, int, int]:
    # Waiting jobs first (they have no start yet), then newest start; within one second, live jobs and later stages first.
    at = "9999" if job["status"] == "queued" and not job.get("started") else job.get("started") or job.get("finished") or ""
    stage = STAGE_KEYS.index(job["stage"]) if job.get("stage") in STAGE_KEYS else -1
    return at, int(job["status"] in ("running", "queued")), stage


def _line(at: str | None, level: str, text: str, proof: list[str] | None = None) -> dict[str, Any]:
    return {"at": at, "level": level, "text": text, **({"proof": list(proof)} if proof else {})}


def _secs(value: Any) -> str:
    if value is None:
        return "—"
    v = float(value)
    return f"{v:.1f} s" if v < 90 else f"{v / 60:.1f} min" if v < 5400 else f"{v / 3600:.1f} h"


# ---------------------------------------------------------------------- routes


def register(app: FastAPI, ctx) -> None:
    jobs = Jobs(ctx)

    @app.get("/api/ops/jobs")
    async def ops_jobs(limit: int = 200):
        """Every job this machine ran or runs now, newest first, with totals and the approvals waiting for a person."""
        return await asyncio.to_thread(jobs.listing, limit)

    @app.get("/api/ops/jobs/{job_id}")
    async def ops_job(job_id: str):
        """One job in full: logs, metrics, artifacts (existing export routes) and what is needed to reproduce it."""
        if not re.fullmatch(r"[a-z]+-[0-9a-z_-]{1,80}", job_id or ""):
            raise HTTPException(404, "Job not found")
        try:
            return await asyncio.to_thread(jobs.detail, job_id)
        except KeyError:
            raise HTTPException(404, "Job not found") from None

    @app.get("/api/ops/evidence-recent")
    async def ops_evidence_recent(limit: int = 3):
        """The newest records of the evidence index (the file /api/evidence/{id} reads)."""
        return await asyncio.to_thread(recent_evidence, limit)
