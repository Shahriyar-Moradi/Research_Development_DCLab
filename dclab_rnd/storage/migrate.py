"""Move a file workspace into the database (package 9.4): ``python -m dclab_rnd.storage migrate SOURCE --to TARGET``.

One-way and repeatable. SOURCE is a workspace folder the file stores wrote (``projects/``, ``drafts/``, ``intern/`` and
the logs beside them); TARGET is the folder of the database workspace it becomes (its files live there, or in the
bucket DCLAB_FILES_URL names; its rows in the database DCLAB_DATABASE_URL names). The two folders may not contain
each other. Copied, with their own ids and timestamps (so every list keeps its order):

- projects: the document (solution, approvals, settings, memory…), the stage records, the transition log and the
  activity log, and the files of ``data/`` and ``exports/`` through file storage (``storage/files.py``);
- drafts: the document, its events and its ``data/`` files;
- intern sessions; the agent traces, the model usage and output checks, the lessons, the model routing and the shadow log.

The source folder is only read (a project document in an old format is upgraded in memory, never written back).
Nothing the target already holds is copied again: a project, draft, session or lesson already there is left as it
is, a file already recorded is not copied over (the database workspace may have changed it since), and an
append-only log is copied only into a workspace that has none of it. An id another workspace owns is reported,
never overwritten. Then every copy is checked: stage records, transitions, activity entries, draft events, sessions,
every log and every file. Fewer rows in the database than in the source is a difference; more is a note (the
database workspace was used after the move). The exit code is 1 when anything differs or failed. The research
campaign runs of the legacy store (``research.sqlite3``, the Jobs page) are not moved: they are not workspace data.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ..studio.store import STAGE_KEYS, migrate as upgrade_document


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a line torn by a crash
    return rows


def _files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and not p.name.endswith((".part", ".tmp"))) if folder.is_dir() else []


def _times(doc: dict[str, Any]) -> dict[str, datetime]:
    """A document's own created and updated times, for the row's columns: the lists sort on them."""
    out = {}
    for column in ("created", "updated"):
        try:
            if doc.get(column):
                out[column] = datetime.fromisoformat(str(doc[column]))
        except ValueError:
            continue
    return out


class Source:
    """The file workspace, read without writing (the file stores' get can upgrade and save an old document)."""

    LOG_FILES = {"agent_steps": "agent_steps.jsonl", "model_requests": "model_requests.jsonl",
                 "model_output_checks": "model_requests_checks.jsonl", "model_shadow": "model_shadow.jsonl"}

    def __init__(self, home: Path):
        self.home = Path(home)
        self.projects_home, self.drafts_home, self.intern_home = self.home / "projects", self.home / "drafts", self.home / "intern"

    def projects(self) -> list[dict[str, Any]]:
        out = []
        for path in sorted(self.projects_home.glob("*/project.json")):
            doc = _json(path)
            upgrade_document(doc)  # in memory only
            out.append(doc)
        return out

    def stages(self, pid: str) -> dict[str, dict[str, Any]]:
        return {s: _json(p) for s in STAGE_KEYS if (p := self.projects_home / pid / "stages" / f"{s}.json").is_file()}

    def transitions(self, pid: str) -> list[dict[str, Any]]:
        return _jsonl(self.projects_home / pid / "transitions.jsonl")

    def activity(self, pid: str) -> list[dict[str, Any]]:
        return _jsonl(self.projects_home / pid / "activity.jsonl")

    def drafts(self) -> list[dict[str, Any]]:
        return [_json(p) for p in sorted(self.drafts_home.glob("*/draft.json"))]

    def events(self, did: str) -> list[dict[str, Any]]:
        return _jsonl(self.drafts_home / did / "events.jsonl")

    def sessions(self) -> list[dict[str, Any]]:
        return [_json(p) for p in sorted(self.intern_home.glob("*.json"))]

    def log(self, name: str) -> list[dict[str, Any]]:
        return _jsonl(self.home / self.LOG_FILES[name])

    def lessons(self) -> list[dict[str, Any]]:
        path = self.home / "lessons.json"
        return list(_json(path).values()) if path.is_file() else []


def _contains(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a == b or a in b.parents


def _storage(target: Path, wid: str, url: str | None):
    """The target's file storage, chosen as the server chooses it: the bucket DCLAB_FILES_URL names, or the folder."""
    from .files import LocalFiles, PgRecords, S3Files

    records = PgRecords(wid, url)
    bucket = os.environ.get("DCLAB_FILES_URL", "").strip()
    return S3Files(target, bucket, records) if bucket.startswith("s3://") else LocalFiles(target, records)


def _log_tables():
    from .models import agent_step, model_output_check, model_request, model_shadow
    from ..agents.traces import FIELDS as TRACE_FIELDS
    from ..models.shadow import FIELDS as SHADOW_FIELDS
    from ..models.usage import CHECK_FIELDS, FIELDS as USAGE_FIELDS

    return {"agent_steps": (agent_step, TRACE_FIELDS), "model_requests": (model_request, USAGE_FIELDS),
            "model_output_checks": (model_output_check, CHECK_FIELDS), "model_shadow": (model_shadow, SHADOW_FIELDS)}


def _count(table, wid: str, url: str | None) -> int:
    import sqlalchemy as sa

    from . import db

    with db.engine(url).connect() as c:
        return int(c.execute(sa.select(sa.func.count()).select_from(table).where(table.c.workspace_id == wid)).scalar())


def migrate(source: Path, target: Path, url: str | None = None) -> dict[str, Any]:
    import sqlalchemy as sa
    from sqlalchemy.dialects.postgresql import insert

    from . import db
    from .files import sha256
    from .models import activity, draft, draft_event, intern_session, project, stage_record, transition

    source, target = Path(source), Path(target)
    if _contains(source, target) or _contains(target, source):
        raise SystemExit("SOURCE and TARGET may not be the same folder or inside each other: the source is only read, never changed")
    if not any((source / d).is_dir() for d in ("projects", "drafts", "intern")):
        raise SystemExit(f"{source} is not a file workspace (no projects/, drafts/ or intern/ folder)")
    src = Source(source)
    wid = db.workspace_for(target, url)
    engine = db.engine(url)
    files = _storage(target, wid, url)
    report: dict[str, Any] = {"source": str(source), "target": str(target), "workspace": wid,
                              "projects": {"copied": [], "already": [], "conflict": []}, "drafts": {"copied": [], "already": [], "conflict": []},
                              "sessions": {"copied": 0, "already": 0, "conflict": []}, "lessons": {"copied": 0, "already": 0, "conflict": [], "skipped_invalid": 0},
                              "files": {"copied": 0, "already": 0}, "logs": {}, "failed": [], "differences": [], "notes": []}

    def owner(table, key: str) -> str | None:
        with engine.connect() as c:
            return c.execute(sa.select(table.c.workspace_id).where(table.c.id == key)).scalar()

    def copy_files(folder: Path, prefix: str, expected: dict[str, str] | None = None) -> None:
        for path in _files(folder):
            key = f"{prefix}/{path.relative_to(folder).as_posix()}"
            original = sha256(path)
            if (expected or {}).get(path.name) not in (None, original):
                report["differences"].append(f"{key}: the source file is not the one its project recorded (its SHA-256 differs)")
            if files.records.get(key) is not None and files.exists(key):
                report["files"]["already"] += 1  # the database workspace's own copy, which may be newer: never copied over
                continue
            record = files.put(key, path)
            report["files"]["copied"] += 1
            if record["sha256"] != original:
                report["differences"].append(f"{key}: the copy's SHA-256 differs from the source's")

    def attempt(kind: str, key: str, fn) -> None:
        """One item; a failure is reported by its type and id (never the row, which can hold a conversation) and the run goes on."""
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            report["failed"].append(f"{kind} {key}: {type(exc).__name__}")

    # ---------------------------------------------------------------- projects
    for doc in src.projects():
        def one_project(doc=doc, pid=doc["id"]):
            home = owner(project, pid)
            if home not in (None, wid):
                report["projects"]["conflict"].append(pid)
                return
            if home is None:
                with engine.begin() as c:  # one project, one transaction: never half a project
                    c.execute(insert(project).values(id=pid, workspace_id=wid, name=str(doc.get("name") or ""), industry=str(doc.get("industry") or "general"),
                                                     doc=json.loads(db.dumps(doc)), **_times(doc)))
                    for stage, record in src.stages(pid).items():
                        c.execute(insert(stage_record).values(project_id=pid, stage=stage, record=json.loads(json.dumps(record, default=str))))
                    for e in src.transitions(pid):
                        c.execute(sa.insert(transition).values(project_id=pid, at=e.get("at"), move=e.get("move"), status=e.get("status"), actor=e.get("actor"),
                                                               entry=json.loads(db.dumps(e))))
                    for a in src.activity(pid):
                        c.execute(sa.insert(activity).values(project_id=pid, at=a.get("at") or "", kind=a.get("kind") or "", payload=json.loads(db.dumps(a.get("payload")))))
                report["projects"]["copied"].append(pid)
            else:
                report["projects"]["already"].append(pid)
            data = doc.get("data") or {}
            copy_files(source / "projects" / pid / "data", f"projects/{pid}/data", {data.get("filename"): data.get("sha256")} if data.get("sha256") else None)
            copy_files(source / "projects" / pid / "exports", f"projects/{pid}/exports")
        attempt("project", doc["id"], one_project)

    # ---------------------------------------------------------------- drafts
    for doc in src.drafts():
        def one_draft(doc=doc, did=doc["id"]):
            home = owner(draft, did)
            if home not in (None, wid):
                report["drafts"]["conflict"].append(did)
                return
            if home is None:
                events = src.events(did)
                with engine.begin() as c:
                    c.execute(insert(draft).values(id=did, workspace_id=wid, status=str(doc.get("status") or "open"), doc=json.loads(db.dumps(doc)),
                                                   event_seq=max([int(e.get("seq") or 0) for e in events] or [0]), **_times(doc)))
                    for e in events:
                        c.execute(sa.insert(draft_event).values(draft_id=did, seq=int(e["seq"]), at=e.get("at") or "", kind=e.get("kind") or "",
                                                                data=json.loads(db.dumps(e.get("data") or {}))))
                report["drafts"]["copied"].append(did)
            else:
                report["drafts"]["already"].append(did)
            copy_files(source / "drafts" / did / "data", f"drafts/{did}/data")
        attempt("draft", doc["id"], one_draft)

    # ---------------------------------------------------------------- intern sessions
    for doc in src.sessions():
        def one_session(doc=doc):
            home = owner(intern_session, doc["id"])
            if home not in (None, wid):
                report["sessions"]["conflict"].append(doc["id"])
                return
            if home is None:
                with engine.begin() as c:
                    c.execute(insert(intern_session).values(id=doc["id"], workspace_id=wid, status=str(doc.get("status") or "queued"),
                                                            doc=json.loads(db.dumps(doc)), **_times(doc)))
                report["sessions"]["copied"] += 1
            else:
                report["sessions"]["already"] += 1
        attempt("session", str(doc.get("id")), one_session)

    attempt("logs", "of the workspace", lambda: _logs(src, wid, url, report))
    report["differences"] += verify(src, target, wid, url, report)
    report["ok"] = not report["differences"] and not report["failed"] and not any(report[k]["conflict"] for k in ("projects", "drafts", "sessions", "lessons"))
    return report


def _logs(src: Source, wid: str, url: str | None, report: dict[str, Any]) -> None:
    """The workspace's logs. An append-only log is copied, in one transaction, only into a workspace that has none of it,
    so a second run adds nothing; the lessons and the routing are documents, copied when absent."""
    import sqlalchemy as sa

    from . import db
    from .models import workspace_lesson
    from ..lessons import PgLessons
    from ..models.routing import PgRouting

    engine = db.engine(url)
    for name, (table, fields) in _log_tables().items():
        rows = src.log(name)
        present = _count(table, wid, url)
        if present:
            report["logs"][name] = {"source": len(rows), "copied": 0, "already": present}
            continue
        with engine.begin() as c:
            for r in rows:
                c.execute(sa.insert(table).values(workspace_id=wid, **{k: r.get(k) for k in fields if k in table.c}))
        report["logs"][name] = {"source": len(rows), "copied": len(rows), "already": 0}
    lessons, out = PgLessons(wid, url), report["lessons"]
    for lesson in src.lessons():
        lid = lesson.get("id")
        if not lid or not lesson.get("project_id") or lesson.get("status") not in ("proposed", "accepted", "rejected", "superseded"):
            out["skipped_invalid"] += 1  # a record the database would refuse: reported, not half-copied
            continue
        with engine.connect() as c:
            home = c.execute(sa.select(workspace_lesson.c.workspace_id).where(workspace_lesson.c.id == lid)).scalar()
        if home not in (None, wid):
            out["conflict"].append(lid)
        elif home == wid:
            out["already"] += 1
        else:
            lessons.save(lesson)
            out["copied"] += 1
    routing = PgRouting(wid, url)
    path = src.home / "model_routing.json"
    if path.is_file() and not routing.get().get("history"):
        routing.save(_json(path))
        report["logs"]["model_routing"] = {"copied": 1}


def verify(src: Source, target: Path, wid: str, url: str | None, report: dict[str, Any] | None = None) -> list[str]:
    """What the database workspace lacks against the source: rows, logs and files. More in the database than in the
    source is a note, not a difference (the workspace was used after the move)."""
    from .files import sha256
    from .postgres import open_stores
    from ..lessons import PgLessons

    projects, drafts, sessions = open_stores(target, url)
    files = _storage(target, wid, url)
    notes = report["notes"] if report is not None else []
    out = []

    def compare(what: str, source_n: int, db_n: int) -> None:
        if db_n < source_n:
            out.append(f"{what}: {source_n} in the source, {db_n} in the database")
        elif db_n > source_n:
            notes.append(f"{what}: {db_n - source_n} more in the database than in the source (used since the move)")

    def check_file(key: str, path: Path) -> None:
        record = files.records.get(key)
        if record is None or not files.exists(key):
            out.append(f"{key}: not in the database workspace")
        elif record["sha256"] != sha256(path):
            notes.append(f"{key}: the database workspace holds other bytes than the source (changed there since the move)")

    for doc in src.projects():
        pid = doc["id"]
        try:
            projects.get(pid)
        except KeyError:
            continue  # a conflict or a failure, already reported
        compare(f"project {pid}: stage records", len(src.stages(pid)), len(projects.records(pid)))
        compare(f"project {pid}: transitions", len(src.transitions(pid)), len(projects.transitions(pid, 10**9)))
        compare(f"project {pid}: activity entries", len(src.activity(pid)), len(projects.activity(pid, 10**9)))
        for folder, prefix in ((src.projects_home / pid / "data", f"projects/{pid}/data"), (src.projects_home / pid / "exports", f"projects/{pid}/exports")):
            for path in _files(folder):
                check_file(f"{prefix}/{path.relative_to(folder).as_posix()}", path)
    for doc in src.drafts():
        try:
            drafts.get(doc["id"])
        except KeyError:
            continue
        compare(f"draft {doc['id']}: events", len(src.events(doc["id"])), len(drafts.events(doc["id"])))
        folder = src.drafts_home / doc["id"] / "data"
        for path in _files(folder):
            check_file(f"drafts/{doc['id']}/data/{path.relative_to(folder).as_posix()}", path)
    ids = {s["id"] for s in sessions.list(limit=None)}
    missing = [s["id"] for s in src.sessions() if s["id"] not in ids]
    if missing:
        out.append(f"{len(missing)} intern session(s) not in the database")
    for name, (table, _) in _log_tables().items():
        compare(f"log {name}", len(src.log(name)), _count(table, wid, url))
    have = {l["id"] for l in PgLessons(wid, url).list()}
    lost = [l.get("id") for l in src.lessons() if l.get("id") and l.get("project_id") and l.get("id") not in have]
    if lost:
        out.append(f"{len(lost)} lesson(s) not in the database")
    return out


def print_report(report: dict[str, Any]) -> None:
    p, d, s, l = report["projects"], report["drafts"], report["sessions"], report["lessons"]
    conflict = lambda x: f", {len(x['conflict'])} owned by another workspace" if x["conflict"] else ""  # noqa: E731
    print(f"Workspace {report['workspace']} (files in {report['target']}), from {report['source']} (not changed)")
    print(f"  projects: {len(p['copied'])} copied, {len(p['already'])} already there{conflict(p)}")
    print(f"  drafts:   {len(d['copied'])} copied, {len(d['already'])} already there{conflict(d)}")
    print(f"  sessions: {s['copied']} copied, {s['already']} already there{conflict(s)}")
    print(f"  lessons:  {l['copied']} copied, {l['already']} already there{conflict(l)}" + (f", {l['skipped_invalid']} invalid, skipped" if l["skipped_invalid"] else ""))
    print(f"  files:    {report['files']['copied']} copied and recorded, {report['files']['already']} already there")
    for name, n in report["logs"].items():
        print(f"  {name}: " + ", ".join(f"{k} {v}" for k, v in n.items()))
    for title, lines in (("Failed", report["failed"]), ("Differences", report["differences"]), ("Notes", report["notes"])):
        if lines:
            print(f"{title}:")
            for line in lines:
                print("  - " + line)
    print("Verified: nothing the source holds is missing from the database workspace." if report["ok"] else "NOT verified: see above.")
