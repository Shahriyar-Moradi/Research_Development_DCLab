"""Workspace lessons (package A5.3): what a finished project taught, reviewed by a person, searchable as evidence.

After a person approves a project's final stage, up to three lessons are proposed from that stage's claims. Each has
a claim (citing the claims it rests on), a scope that code writes (this project, this dataset, this task, this many
rows, this split), what argues against it and the next test. A model writes them when one is configured, and code
keeps a claim only when ``dclab_rnd.cited`` passes it (it cites a given claim, its numbers are in it, it claims no
production readiness, cause or proof); otherwise code writes them from the claims and their stated limits.

A reviewer (the role ``reviewer`` or ``owner``) accepts, edits or rejects each one. Accepted lessons live in the
workspace's lessons table (``lessons.json`` beside the projects, or the ``workspace_lesson`` table on PostgreSQL)
and join every evidence search as ``workspace_lesson`` records labelled with their project and scope: the evidence
index under ``evidence/knowledge`` is never written. A lesson from synthetic data is labelled synthetic and never
offered as evidence.

Until accounts exist (package 10.2) a role is what the request declares (``X-DCLab-Role``), not a verified login.
The server installs its workspace's lessons for the process (as it installs the model gateway): one workspace per
process, which is how the product runs; two apps in one process would search the last one's lessons.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Protocol

from . import cited, prompts

MOST = 3  # lessons per finished project
TEXT = 600  # characters of a claim, a counter-argument or a next test
REVIEWERS = ("reviewer", "owner")
ACTIONS = ("accept", "edit", "reject")
TYPE = "workspace_lesson"
LABEL = "Workspace lesson: from a finished project of this workspace, accepted by a reviewer; holds for its scope only."
PROMPT = prompts.text("lesson_proposal", most=MOST)  # dclab_rnd/prompts/lesson_proposal.md

NEXT_TEST = {
    "C1": "Score the same recipe and model on data collected after this holdout, or on a second sample of the same population, before relying on it beyond this dataset.",
    "C2": "Search a wider set of parameters with the same training-CV rule and see whether the tuning decision changes.",
    "C3": "Compare with the simplest rule the team uses today, on the same holdout.",
}
AGAINST_C3 = "The baselines are trivial (the majority class or the training mean); a rule a person would write may be harder to beat."


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _line(text: Any, limit: int = TEXT) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def record_id(lesson: dict[str, Any]) -> str:
    return f"LESSON-{lesson['id']}"


# ---------------------------------------------------------------------- proposing


def sources_of(record: dict[str, Any]) -> dict[str, str]:
    """What a lesson may cite: the final stage's claims with their stated limits (aggregates, never rows)."""
    return {c["claim_id"]: _line(f"{c['statement']} {' '.join(c.get('limitations') or [])}", 1200) for c in record.get("claims") or []}


def scope_of(project: dict[str, Any], record: dict[str, Any]) -> dict[str, Any]:
    """Where a lesson holds, written by code from the record: never by a model."""
    evidence = record.get("evidence") or {}
    rows = (evidence.get("train_rows") or 0) + (evidence.get("holdout_rows") or 0)
    return {"project": project.get("name"), "project_id": project.get("id"), "dataset": (record.get("provenance") or {}).get("filename"),
            "task": record.get("task_type") or record.get("task"), "metric": record.get("primary_metric"), "rows": rows or None,
            "split": evidence.get("holdout_split") or (record.get("provenance") or {}).get("holdout")}


def scope_line(scope: dict[str, Any]) -> str:
    parts = [f"project {scope.get('project')}", f"dataset {scope.get('dataset')}", f"task {scope.get('task')}"]
    if scope.get("rows"):
        parts.append(f"{scope['rows']} rows")
    if scope.get("metric"):
        parts.append(f"metric {scope['metric']}")
    if scope.get("split"):
        parts.append(f"split: {scope['split']}")
    return "; ".join(parts)


def _suffix(claim_id: str) -> str:
    return claim_id.rsplit("-", 1)[-1]


def drafts(record: dict[str, Any]) -> list[dict[str, Any]]:
    """The lessons code writes from the final stage's claims, each with the claim's own limit as its counter-argument."""
    out = []
    for claim in record.get("claims") or []:
        key = _suffix(claim["claim_id"])
        if key not in NEXT_TEST:
            continue
        limits = " ".join(claim.get("limitations") or []) or (AGAINST_C3 if key == "C3" else "")
        if key == "C1" and (record.get("evidence") or {}).get("cv_to_holdout_gap") is not None:
            limits = (limits + f" The holdout differed from the training-CV mean by {(record['evidence']['cv_to_holdout_gap']):+.4f}.").strip()
        out.append({"claim": _line(claim["statement"]), "against": _line(limits or "One project, one split: the claim may not hold elsewhere."),
                    "next_test": NEXT_TEST[key], "cites": [claim["claim_id"]], "written_by": "code"})
    return out[:MOST]


def _parse(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    found = re.search(r"\[.*\]", text, re.S)
    try:
        items = json.loads(found.group(0) if found else text)
    except (json.JSONDecodeError, AttributeError):
        return []
    return [i for i in items if isinstance(i, dict)][:MOST] if isinstance(items, list) else []


def _from_model(record: dict[str, Any], client: Any) -> list[dict[str, Any]]:
    """The model's lessons that pass the check; the reason any was dropped is reported to the gateway, never logged."""
    sources = sources_of(record)
    facts = "\n".join(f"[{i}] {body}" for i, body in sources.items())
    try:
        reply = client.complete([{"role": "user", "content": f"{PROMPT}\n\nStage: {record.get('title')}\nFacts:\n{facts}"}], max_tokens=700)
        items = _parse(reply.get("content") or "")
    except Exception:  # noqa: BLE001 — the model is optional; code writes the lessons without it
        return []
    known_numbers = set().union(*(cited.numbers(b, source=True) for b in sources.values())) if sources else set()
    kept, problems = [], []
    for item in items:
        claim = cited.check(str(item.get("claim") or ""), sources)
        if not claim["kept"] or claim["dropped"]:
            problems += claim["dropped"] or ["no claim"]
            continue
        cites = list(dict.fromkeys(c for s in claim["kept"] for c in s["cites"]))
        against = cited.check(str(item.get("against") or ""), sources)
        fallback = " ".join(" ".join(next((c.get("limitations") or [] for c in record["claims"] if c["claim_id"] == i), [])) for i in cites)
        next_test, removed = cited.strip_overclaims(_line(item.get("next_test")))
        if removed or not next_test or not cited.numbers(next_test) <= known_numbers:
            next_test = NEXT_TEST.get(_suffix(cites[0]), NEXT_TEST["C1"])
        kept.append({"claim": _line(" ".join(s["text"] for s in claim["kept"])), "cites": cites,
                     "against": _line(" ".join(s["text"] for s in against["kept"]) or fallback or "One project, one split: the claim may not hold elsewhere."),
                     "next_test": next_test, "written_by": "model"})
    report = getattr(client, "output", None)
    if report:
        report(bool(kept) and not problems, "; ".join(sorted(set(problems)))[:200] if problems else ("" if kept else "no lesson passed"))
    return kept


def basis_of(record: dict[str, Any]) -> str:
    """Which final stage run a lesson comes from: the experiment, the run's number (each run uses the holdout once
    more) and when it completed. Claim ids stay the same when the stage runs again; this does not."""
    run = (record.get("evidence") or {}).get("holdout_uses_in_this_project") or 1
    return f"{record.get('experiment_id')}:run{run}:{record.get('completed_at')}"


def propose(project: dict[str, Any], record: dict[str, Any], client: Any = None) -> list[dict[str, Any]]:
    """Up to three proposed lessons for a project's final stage record (a model's when one passes the check, else code's)."""
    if record.get("stage") != "final" or not record.get("claims"):
        return []
    items = (_from_model(record, client) if client is not None else []) or drafts(record)
    sources = sources_of(record)
    scope, synthetic = scope_of(project, record), bool((project.get("data") or {}).get("synthetic") or (project.get("draft") or {}).get("synthetic"))
    basis = basis_of(record)
    return [{"id": secrets.token_hex(5), "project_id": project["id"], "project_name": project.get("name"), "basis": basis,
             "experiment_id": record.get("experiment_id"), "status": "proposed", "claim": item["claim"], "against": item["against"],
             "next_test": item["next_test"], "cites": item["cites"], "sources": {c: sources[c] for c in item["cites"] if c in sources},
             "scope": scope, "synthetic": synthetic, "written_by": item["written_by"], "proposed_at": now(), "review": None, "history": []}
            for item in items[:MOST]]


def propose_for(store: "LessonStore", project: dict[str, Any], record: dict[str, Any] | None, client: Any = None) -> list[dict[str, Any]]:
    """The lessons of this final stage record: proposed once (a second call returns the same ones)."""
    if not record or record.get("stage") != "final":
        return []
    basis = basis_of(record)
    mine = store.list(project_id=project["id"])
    existing = [l for l in mine if l.get("basis") == basis]
    if existing:
        return existing
    for old in mine:  # the final stage ran again: the earlier run's unreviewed lessons describe numbers that changed
        if old.get("status") == "proposed":
            entry = {"action": "supersede", "by": "system", "at": now(), "reason": "the final stage ran again", "changed": None}
            store.save({**old, "status": "superseded", "review": entry, "history": [*(old.get("history") or []), entry]})
    made = propose(project, record, client)
    for lesson in made:
        store.save(lesson)
    return made


# ---------------------------------------------------------------------- reviewing


class ReviewError(ValueError):
    pass


def review(store: "LessonStore", lesson_id: str, action: str, role: str | None, reason: str = "", edits: dict[str, Any] | None = None) -> dict[str, Any]:
    """A reviewer accepts, edits (and so accepts) or rejects a lesson. The scope, the citations and the synthetic label
    stay what code wrote; an edit that claims production readiness, a cause or a proof is refused."""
    if role not in REVIEWERS:
        raise PermissionError("Only a reviewer or the owner reviews a lesson")
    if action not in ACTIONS:
        raise ReviewError(f"action must be one of {', '.join(ACTIONS)}")
    lesson = store.get(lesson_id)
    if lesson.get("status") == "superseded":
        raise ReviewError("This lesson came from an earlier run of the final stage; review the lessons of the latest run")
    changed: dict[str, Any] = {}
    if action == "edit":
        for key in ("claim", "against", "next_test"):
            if (edits or {}).get(key) not in (None, ""):
                text = _line(edits[key])
                if cited.overclaims(text):
                    raise ReviewError(f"The {key.replace('_', ' ')} claims production readiness, a cause or a proof; a lesson holds for its scope only")
                if text != lesson[key]:
                    changed[key] = {"from": lesson[key], "to": text}
                    lesson[key] = text
        if not changed:
            raise ReviewError("An edit changes the claim, the counter-argument or the next test")
    entry = {"action": action, "by": role, "at": now(), "reason": _line(reason, 300), "changed": changed or None}
    lesson["status"] = "rejected" if action == "reject" else "accepted"
    lesson["review"] = entry
    lesson["history"] = [*(lesson.get("history") or []), entry]
    return store.save(lesson)


# ---------------------------------------------------------------------- the workspace's lessons table


class LessonStore(Protocol):
    def save(self, lesson: dict[str, Any]) -> dict[str, Any]: ...
    def get(self, lesson_id: str) -> dict[str, Any]: ...  # KeyError when unknown
    def list(self, project_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]: ...


_LOCKS: dict[Path, threading.Lock] = {}
_GUARD = threading.Lock()


class FileLessons:
    """``lessons.json`` in the workspace folder, replaced whole on each write (never a half-written file)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _GUARD:
            self._lock = _LOCKS.setdefault(self.path.resolve(), threading.Lock())

    def _all(self) -> dict[str, dict[str, Any]]:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, lesson: dict[str, Any]) -> dict[str, Any]:
        lesson = {**lesson, "updated": now()}
        with self._lock:
            rows = self._all()
            rows[lesson["id"]] = lesson
            temporary = self.path.with_suffix(".tmp")
            temporary.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(temporary, self.path)
        return lesson

    def get(self, lesson_id: str) -> dict[str, Any]:
        with self._lock:
            return self._all()[lesson_id]

    def list(self, project_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            rows = list(self._all().values())
        return [r for r in rows if (project_id is None or r.get("project_id") == project_id) and (status is None or r.get("status") == status)]  # insertion order


class PgLessons:
    def __init__(self, workspace_id: str, url: str | None = None):
        from .storage import db

        self.workspace_id, self.engine = workspace_id, db.engine(url)

    def save(self, lesson: dict[str, Any]) -> dict[str, Any]:
        import sqlalchemy as sa
        from sqlalchemy.dialects.postgresql import insert

        from .storage.models import workspace_lesson as t

        lesson = {**lesson, "updated": now()}
        values = {"id": lesson["id"], "workspace_id": self.workspace_id, "project_id": lesson["project_id"], "status": lesson["status"], "doc": lesson}
        with self.engine.begin() as c:
            written = c.execute(insert(t).values(**values).on_conflict_do_update(
                index_elements=[t.c.id], set_={"status": values["status"], "doc": values["doc"], "updated": sa.func.now()},
                where=t.c.workspace_id == self.workspace_id).returning(t.c.id)).scalar()
        if written is None:  # the id belongs to another workspace: nothing was written
            raise KeyError(lesson["id"])
        return lesson

    def get(self, lesson_id: str) -> dict[str, Any]:
        import sqlalchemy as sa

        from .storage.models import workspace_lesson as t

        with self.engine.connect() as c:
            doc = c.execute(sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id, t.c.id == lesson_id)).scalar()
        if doc is None:
            raise KeyError(lesson_id)
        return doc

    def list(self, project_id: str | None = None, status: str | None = None) -> list[dict[str, Any]]:
        import sqlalchemy as sa

        from .storage.models import workspace_lesson as t

        q = sa.select(t.c.doc).where(t.c.workspace_id == self.workspace_id)
        if project_id is not None:
            q = q.where(t.c.project_id == project_id)
        if status is not None:
            q = q.where(t.c.status == status)
        with self.engine.connect() as c:
            rows = [r[0] for r in c.execute(q.order_by(t.c.created, t.c.id))]
        return rows


def open_lessons(home: Path) -> LessonStore:
    from .storage import db

    if db.database_url(required=False):
        return PgLessons(db.workspace_for(Path(home)))
    return FileLessons(Path(home) / "lessons.json")


_INSTALLED: list[LessonStore | None] = [None]


def install(store: LessonStore | None) -> None:
    """The workspace's lessons, for the evidence search (the server installs them, as it installs the model gateway)."""
    _INSTALLED[0] = store


def installed() -> LessonStore | None:
    """The lessons of the workspace this request or job acts in (``dclab_rnd.context``), else the process's."""
    from . import context

    current = getattr(context.services(), "lesson_store", None)
    return current if current is not None else _INSTALLED[0]


# ---------------------------------------------------------------------- searched with the repository's records


def _completed(lesson: dict[str, Any]) -> str:
    """When the final stage run the lesson came from was completed (claim ids stay the same when the stage runs again)."""
    parts = (lesson.get("basis") or "").split(":", 2)
    return f"{parts[2]} ({parts[1].replace('run', 'run ')})" if len(parts) == 3 else "at an unknown time"


def as_record(lesson: dict[str, Any]) -> dict[str, Any]:
    """An accepted lesson as an evidence record: its claim, scope, counter-argument and next test, and where it came from."""
    scope = lesson.get("scope") or {}
    text = (f"{lesson['claim']} Scope: {scope_line(scope)}. Against it: {lesson['against']} Next test: {lesson['next_test']} "
            f"From project {lesson.get('project_name')}, citing {', '.join(lesson.get('cites') or [])} of the final stage run named in this record's metadata; "
            f"accepted by a {((lesson.get('review') or {}).get('by') or 'reviewer')}.")  # no date here: every number in the text is one a sentence citing it may state
    return {"record_id": record_id(lesson), "type": TYPE, "title": f"Workspace lesson: {_line(lesson['claim'], 110)}", "text": text,
            "metadata": {"workspace_lesson": True, "label": LABEL, "project_id": lesson.get("project_id"), "project": lesson.get("project_name"),
                         "scope": scope, "task_type": scope.get("task"), "dataset": scope.get("dataset"), "status": lesson.get("status"),
                         "written_by": lesson.get("written_by"), "reviewed_by": (lesson.get("review") or {}).get("by"),
                         "experiment_id": lesson.get("experiment_id"), "final_run": _completed(lesson),
                         "edited": bool((lesson.get("review") or {}).get("changed"))},
            "citations": list(lesson.get("cites") or [])}


def offered(store: LessonStore | None = None) -> list[dict[str, Any]]:
    """The lessons that are evidence: accepted, and not from synthetic data."""
    store = store or installed()
    if store is None:
        return []
    try:
        return [l for l in store.list(status="accepted") if not l.get("synthetic")]
    except Exception:  # noqa: BLE001 — a store that cannot be read leaves the repository's records as they are
        return []


_MERGED: dict[tuple[int, str], tuple[Any, Any]] = {}  # (id of the base, the lessons' content) -> (the base, the merged index)
_MERGED_LOCK = threading.Lock()


def merged(base: Any, store: LessonStore | None = None) -> Any:
    """``base`` (an EvidenceIndex) with the accepted lessons added, or ``base`` itself when there are none. Cached on
    the lessons' content (an edit within the same second is a new index) and on the base object itself."""
    from .evidence_index import EvidenceIndex

    lessons = offered(store)
    if not lessons:
        return base
    content = json.dumps([as_record(l) for l in lessons], sort_keys=True, ensure_ascii=False, default=str)
    key = (id(base), content)
    with _MERGED_LOCK:
        hit = _MERGED.get(key)
        if hit is not None and hit[0] is base:
            return hit[1]
    index = EvidenceIndex([*base.records, *json.loads(content)])
    with _MERGED_LOCK:
        if len(_MERGED) >= 8:
            _MERGED.clear()
        _MERGED[key] = (base, index)
    return index


def records(lessons: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [as_record(l) for l in lessons]
