"""Tier 2, the shapers (32.8): every raw entry into normalised records (:class:`dclab_core.store.StoreRecord`), each with its
provenance, its capture rule's findings and its split; or a recorded reason why not.

Raw entries never change, so each is shaped once (``records/<kind>/<workspace>/<entry>.<n>.jsonl.gz``, streamed line by line
into chunks of at most :data:`CHUNK` bytes, so a 280 MB transcript neither fills the memory nor makes a file too large to
push), and its result remembered in ``records/<kind>/.shaped.json``. Two kinds are rebuilt whole on every run, because a
record's split depends on others: ``decisions`` (a label is sealed with its decision, a draft's decisions with the project it
became) and ``teachers`` (an answer is sealed with its decision). Every kind is rebuilt whole when the shapers' version
changes (:data:`VERSION`: this module's, the redaction's and the capture rules'), into a folder beside it, then swapped in,
so no stale file survives. One run at a time (a lock on ``records/``).

What a record never holds:

- a credential (anything shaped like one, ``keepers.agents.SECRET``, after a gzip source is opened);
- a person's contact or a card number in its words: ``dclab_core.redact`` on every field that holds words, and on every
  string of an agent's transcript; ids and hashes, and anything under ``ids``, are left as they are;
- a table's rows: the keys that quote a table's cells (``testers.QUOTES_CELLS``) are removed, and an output that prints a
  table (three or more lines with the same separators) is replaced by its size;
- a line that does not parse: kept as its length and hash.
"""

from __future__ import annotations

import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import shutil
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from dclab_core.store import KINDS, Provenance, StoreRecord

from .layout import Layout
from .store import Store

CHUNK = 20_000_000
RETRY = ("OSError", "MemoryError", "PermissionError", "TimeoutError")  # failures that may pass next time
ALWAYS = ("decisions", "teachers")
TEXT_KEYS = {"text", "content", "reason", "question", "answer", "prompt", "reply", "thinking", "note", "notes", "problem", "goal", "said", "words",
             "message", "summary", "why", "query", "detail", "error", "title", "description", "stdout", "stderr", "command", "output", "result"}
LIFT = ("project", "draft", "run", "conversation", "visit", "job", "request", "session", "user")  # 32.32: whose it is
RENAME = {"project_id": "project", "draft_id": "draft", "record_id": "record", "sessionId": "session", "timestamp": "at", "user_id": "user"}
BINARY = (".joblib", ".pkl", ".pickle", ".npy", ".npz", ".sqlite3", ".parquet", ".bin", ".onnx", ".pt")


def _version() -> str:
    from dclab_core import redact, store_rules

    return f"shapers-4:{redact.VERSION}:{store_rules.VERSION}"  # 3: one group for every kind; tables printed with spaces; 4: whose it is (body.user, provenance.by)


def _id_key(key: Optional[str]) -> bool:
    k = str(key or "")
    return k in ("ids", "id", "uuid", "signature", "sha256", "hash", "whole_sha256", "at", "timestamp", "time", "created", "updated", "started",
                 "finished", "captured_at", "state_at", "version", "schema_version") or \
        k.endswith(("_id", "Id", "_ids", "uuid", "Uuid", "_sha256", "_hash", "_at"))  # ids, hashes and times: how records link and order


def _columns(line: str) -> Optional[tuple[str, int]]:
    """How a line splits into a table's cells: by a separator (, tab ; |) at least twice, or by runs of two or more spaces into
    three or more fields (a printed frame); None when it does not read as a table's line."""
    for sep in (",", "\t", ";", "|"):
        n = line.count(sep)
        tight = n >= 1 and not re.search(rf"{re.escape(sep)} | {re.escape(sep)}", line) and " " not in line.strip()  # a,b with no space: a CSV row
        if n >= 2 or tight:
            return sep, n
    fields = [f for f in re.split(r"\s{2,}", line.strip()) if f]
    return ("  ", len(fields)) if len(fields) >= 3 else None


def _tables_out(text: str) -> str:
    """A printed table replaced by its size: two or more consecutive lines that split the same way (a header and a row is
    already a row of someone's table), by separators or by aligned spaces."""
    if "\n" not in text:
        return text
    lines = text.split("\n")
    out: list[str] = []
    run: list[str] = []
    shape: Optional[tuple[str, int]] = None

    def flush() -> None:
        if len(run) >= 2:
            out.append(f"[a table of {len(run)} lines removed]")
        else:
            out.extend(run)
        run.clear()

    for line in lines:
        cols = _columns(line)
        same = cols is not None and shape is not None and cols[0] == shape[0] and (cols[1] == shape[1] or (cols[0] == "  " and abs(cols[1] - shape[1]) <= 1))
        if cols is not None and (not run or same):  # a printed frame's header has one field less than its rows (the index)
            run.append(line)
            shape = cols
            continue
        flush()
        if cols is not None:
            run.append(line)
            shape = cols
        else:
            out.append(line)
            shape = None
    flush()
    return "\n".join(out)


def _redact(value: Any, key: Optional[str] = None, every: bool = False) -> Any:
    from dclab_core import redact

    if _id_key(key):
        return value
    if isinstance(value, dict):
        return {k: _redact(v, k, every) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v, key, every) for v in value]
    if isinstance(value, str) and (every or key in TEXT_KEYS):
        return redact.redact_text(_tables_out(value))[0]
    return value


def _clean(doc: Any, every: bool = False) -> Any:
    from ..testers import _scrub

    return _redact(_scrub(doc), every=every)


def _body(doc: Any, kind: str, source: str) -> dict[str, Any]:
    body = dict(doc) if isinstance(doc, dict) else {"value": doc}
    if isinstance(body.get("doc"), dict):  # a database row: its document's fields lifted beside its columns
        body = {**body["doc"], **{k: v for k, v in body.items() if k != "doc"}}
    if source.startswith("db/stage_record") and isinstance(body.get("record"), dict):  # a stage row: its record lifted, never read as an id
        body = {**body.pop("record"), **body}
    for old, new in RENAME.items():
        if old in body and body.get(new) is None:
            body[new] = body[old]
    ids = body.get("ids") if isinstance(body.get("ids"), dict) else {}
    for k in LIFT:
        if body.get(k) is None and ids.get(k) is not None:
            body[k] = ids[k]
    for k in ("project", "draft", "session", "record", "conversation"):
        if body.get(k) is not None and not isinstance(body[k], str):
            body[k] = str(body[k].get("id")) if isinstance(body[k], dict) and body[k].get("id") else None
    if kind == "agents":  # a Claude Code transcript line: which session, where, by whom
        body.setdefault("agent", "claude_code")
        parts = source.split("/")
        body.setdefault("where", parts[1] if len(parts) > 2 else "main")
    return body


def _type(kind: str, source: str, doc: dict[str, Any]) -> str:
    name = source.rsplit("/", 1)[-1]
    version = str(doc.get("schema_version") or "")
    if kind == "decisions":
        if "decision_label" in version or "/labels/" in f"/{source}" or "decision_label" in name:
            return "label"
        if "outcome" in version or "/outcomes/" in f"/{source}" or "decision_outcome" in name:
            return "outcome"
        return "correction" if doc.get("before") is not None and doc.get("after") is not None else "decision"
    if kind == "workflows":
        return "graph_move" if name in ("transitions.jsonl", "transition.jsonl") else "walk" if source.startswith("walks/") else \
            "stage_run" if "/stages/" in source or name.startswith("stage_record") else "board" if "board" in name else \
            "interrupt" if "interrupt" in name else "walk" if "walk" in name else "snapshot"
    if kind == "conversations":
        return "intern_session" if "/versions/" in source or name.startswith("intern_session") else \
            "turn" if source.startswith(("workspace_chat/", "home_orchestrator/", "drafts/", "db/draft_event")) and name.endswith(".jsonl") else \
            "mcp_call" if source.startswith("mcp/") else "snapshot"
    if kind == "actions" and source.startswith("metrics/"):  # the agents' metrics ledger: a measurement of an agent
        return "metric"
    if kind == "sessions" and source.startswith("people/"):  # 32.32: a person's record, or a session of an account
        return "session" if name.startswith("session") else "person"
    return {"sessions": "event", "actions": "step", "audit": "row", "model_requests": "request", "tasks": "job_state", "teachers": "answer",
            "students": "version", "deletions": "deletion", "fixes": "fix", "agents": "transcript_line", "tests": "test_run"}.get(kind, "snapshot")


def _lines(fh: Any, source: str) -> Iterator[Any]:
    """The source's documents, one by one: a JSON Lines source line by line (credentials removed from each line first)."""
    from .keepers.agents import redact as secrets

    if source.endswith((".jsonl", ".jsonl.gz")):
        for raw in fh:
            raw = secrets(raw)
            if not raw.strip():
                continue
            try:
                yield json.loads(raw)
            except ValueError:
                yield {"unparsed_chars": len(raw), "unparsed_sha256": hashlib.sha256(raw).hexdigest()}
    else:
        yield json.loads(secrets(fh.read()))


@contextmanager
def _opened(store: Store, entry: Any) -> Iterator[Any]:
    """The kept bytes as a stream (a source that is itself gzip, opened too), its hash checked when the stream ends."""
    with gzip.open(store.layout.entry(entry.path), "rb") as raw:
        digest = hashlib.sha256()

        class Checked(io.RawIOBase):
            def readable(self) -> bool:
                return True

            def readinto(self, b: Any) -> int:
                data = raw.read(len(b))
                digest.update(data)
                b[: len(data)] = data
                return len(data)

        stream = io.BufferedReader(Checked())
        yield gzip.GzipFile(fileobj=stream) if entry.source.endswith(".gz") else stream
        while stream.read(1 << 20):
            pass
        if digest.hexdigest() != entry.sha256:
            raise ValueError(f"{entry.id}: the kept bytes do not match their hash")


class Context:
    """What one record's split and flags need from the others: each decision's group, each draft's project, what was deleted,
    and each keeping run's context (its commit: 32.22)."""

    def __init__(self) -> None:
        self.groups: dict[str, str] = {}
        self.drafts: dict[str, str] = {}
        self.deleted: set[str] = set()
        self.runs: dict[str, dict[str, Any]] = {}

    def group(self, body: dict[str, Any]) -> Optional[str]:
        from dclab_core.datasets import group_of

        return group_of(project=body.get("project"), draft=body.get("draft"), dataset=body.get("dataset"), drafts=self.drafts) or None


def _context(store: Store) -> Context:
    ctx = Context()
    base = store.layout.runs()
    for path in base.rglob("store-*.json") if base.exists() else []:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("id") and doc.get("context"):
                ctx.runs[str(doc["id"])] = doc["context"]
        except (OSError, ValueError):
            continue
    for e in store.entries("deletions"):
        try:
            doc = json.loads(store.read(e))
            if doc.get("id"):
                ctx.deleted.add(str(doc["id"]))
        except Exception:  # noqa: BLE001
            continue
    latest: dict[tuple[str, str], Any] = {}
    for e in store.entries("conversations"):  # each draft's newest version: the project it became
        if e.source.endswith("/draft.json") or e.source == "db/draft.jsonl":
            latest[(e.workspace, e.source)] = e
    for e in latest.values():
        try:
            with _opened(store, e) as fh:
                for doc in _lines(fh, e.source):
                    body = _body(doc, "conversations", e.source)
                    if body.get("id") and (body.get("project_id") or body.get("project")):
                        ctx.drafts[str(body.get("id"))] = str(body.get("project_id") or body.get("project"))
        except Exception:  # noqa: BLE001
            continue
    for e in sorted(store.entries("decisions"), key=lambda e: (e.workspace, e.source, e.version)):  # every decision's group, before any label
        try:
            with _opened(store, e) as fh:
                for doc in _lines(fh, e.source):
                    body = _body(doc, "decisions", e.source)
                    if _type("decisions", e.source, body) == "decision" and body.get("draft") and body.get("project"):
                        ctx.drafts.setdefault(str(body["draft"]), str(body["project"]))
        except Exception:  # noqa: BLE001
            continue
    for e in sorted(store.entries("decisions"), key=lambda e: (e.workspace, e.source, e.version)):
        try:
            with _opened(store, e) as fh:
                for doc in _lines(fh, e.source):
                    body = _body(doc, "decisions", e.source)
                    if _type("decisions", e.source, body) == "decision" and body.get("id"):
                        group = ctx.group(body)
                        if group:
                            ctx.groups[str(body["id"])] = group
        except Exception:  # noqa: BLE001
            continue
    return ctx


def _record(entry: Any, n: int, doc: Any, ctx: Context) -> StoreRecord:
    from dclab_core import store_rules
    from dclab_core.datasets import split_of

    kind = entry.kind
    body = _clean(_body(doc, kind, entry.source), every=kind == "agents")
    type_ = _type(kind, entry.source, body)
    if kind == "decisions":
        own = ctx.group(body) if type_ == "decision" else None
        body["group"] = body.get("group") or own or ctx.groups.get(str(body.get("record") or ""))
        body.setdefault("state_at", body.get("at"))  # a ledger decision's state is captured where and when it is made
    if kind == "teachers" and not body.get("group"):
        body["group"] = ctx.groups.get(str(body.get("record") or body.get("decision") or "")) or ctx.group(body)
    if kind not in ("decisions", "teachers") and not body.get("group"):  # one group for a project everywhere: its draft's, its own
        body["group"] = ctx.group(body)
    at = str(body.get("at") or body.get("captured_at") or entry.captured_at)
    rule = store_rules.rule_for(kind, type_)
    flags = [f"{f.what}:{f.field or '-'}" for f in store_rules.check({"kind": kind, "type": type_, "at": at, "body": body}, rule)]
    key = body.get("group") or (body.get(rule.sealed_key) if rule and rule.sealed_key else None)  # the group first: one split per project
    ids = {str(body.get(k)) for k in ("project", "draft", "session", "conversation", "id") if isinstance(body.get(k), str)}
    run_context = ctx.runs.get(str(entry.run or "")) or {}
    if run_context:
        body.setdefault("_context", run_context)  # 32.22: the code, libraries and machine it was kept with
    prov = Provenance(entry=entry.id, sha256=entry.sha256, source=entry.source, workspace=entry.workspace, consent=entry.consent, at=entry.captured_at,
                      commit=(run_context.get("commit") or None) and str(run_context["commit"])[:64],
                      project=body.get("project") if isinstance(body.get("project"), str) and len(body["project"]) <= 120 else None,
                      draft=body.get("draft") if isinstance(body.get("draft"), str) and len(body["draft"]) <= 120 else None,
                      deleted=bool(ids & ctx.deleted), rule=f"{store_rules.VERSION}:{rule.task}" if rule else None,
                      by=body.get("user") if isinstance(body.get("user"), str) and len(body["user"]) <= 120 else None)  # whose it is: the pseudonymous user id
    return StoreRecord(id=f"{entry.id}#{n}", kind=kind, type=type_, at=at, body=body, provenance=prov,
                       split=split_of(str(key)) if key not in (None, "") else None, flags=flags)


class _Writer:
    """An entry's records, streamed into chunks of at most CHUNK bytes each (gzip with no time inside: the same records, the same bytes)."""

    def __init__(self, base: Path, stem: str):
        self.base, self.stem, self.n, self.size, self.fh, self.files = base, stem, 0, 0, None, []

    def write(self, line: str) -> None:
        data = line.encode("utf-8")
        if self.fh is None or self.size + len(data) > CHUNK:
            self.close()
            self.base.mkdir(parents=True, exist_ok=True)
            path = self.base / f"{self.stem}.{len(self.files)}.jsonl.gz"
            self.files.append(path)
            self.fh, self.size = gzip.GzipFile(filename="", fileobj=open(path, "wb"), mode="wb", mtime=0), 0
        self.fh.write(data)
        self.size += len(data)
        self.n += 1

    def close(self) -> None:
        if self.fh is not None:
            raw = self.fh.fileobj
            self.fh.close()
            raw.close()
            self.fh = None


def _shape_entry(store: Store, entry: Any, base: Path, ctx: Context) -> dict[str, Any]:
    """One entry's records written; what it made and what it could not, each line on its own."""
    stem = re.sub(r"[^a-z0-9_.\-]+", "-", entry.id.lower()).strip("-")[:150]
    if entry.source.endswith("~counts.json") or entry.source.endswith(BINARY):
        return {"records": 0, "files": [], "unshaped": "counts only" if "~counts" in entry.source else "a binary file: kept raw, not shaped"}
    writer, failed = _Writer(base / entry.workspace, stem), []
    try:
        with _opened(store, entry) as fh:
            for n, doc in enumerate(_lines(fh, entry.source)):
                try:
                    writer.write(_record(entry, n, doc, ctx).model_dump_json() + "\n")
                except Exception as exc:  # noqa: BLE001 - this line only, with its reason; the others go on
                    failed.append(f"line {n}: {type(exc).__name__}: {str(exc)[:200]}")
    except Exception as exc:  # noqa: BLE001 - the entry could not be read: none of it is shaped, and why is kept
        writer.close()
        for f in writer.files:
            f.unlink(missing_ok=True)
        return {"records": 0, "files": [], "unshaped": f"{type(exc).__name__}: {str(exc)[:400]}"}
    writer.close()
    return {"records": writer.n - 0, "files": [store.layout.relative(f) for f in writer.files], "unshaped": None, "failed_lines": failed[:50],
            "failed": len(failed)}


@contextmanager
def _locked(layout: Layout) -> Iterator[None]:
    lock = layout.records_lock()
    lock.parent.mkdir(parents=True, exist_ok=True)
    with open(lock, "a") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def shape(layout: Optional[Layout] = None, force: bool = False) -> dict[str, Any]:
    """``records/`` brought up to ``raw/``: each new entry shaped once; decisions and teachers, and every kind when the shapers'
    version changed or ``force``, rebuilt whole beside the old and swapped in."""
    layout = layout or Layout()
    store = Store(layout)
    version = _version()
    report: dict[str, Any] = {"version": version, "records": 0, "unshaped": 0, "by_kind": {}, "unchanged": []}
    with _locked(layout):
        ctx = _context(store)
        for kind in KINDS:
            state_path = layout.shaped_state(kind)
            state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
            entries = sorted(store.entries(kind), key=lambda e: (e.workspace, e.source, e.version))
            whole = force or kind in ALWAYS or state.get("version") != version
            done = {} if whole else {k: v for k, v in (state.get("entries") or {}).items()
                                     if not str(v.get("unshaped") or "").startswith(RETRY)}  # a passing failure is tried again
            todo = [e for e in entries if e.id not in done]
            if not todo and not whole:
                report["unchanged"].append(kind)
                continue
            target = layout.records_of(kind)
            base = target.with_name(f"{kind}.building-{os.getpid()}") if whole else target
            if whole:
                shutil.rmtree(base, ignore_errors=True)
            for entry in todo:
                done[entry.id] = _shape_entry(store, entry, base, ctx)
            if whole:
                old = target.with_name(f"{kind}.old-{os.getpid()}")
                if target.exists():
                    target.rename(old)
                base.mkdir(parents=True, exist_ok=True)
                base.rename(target)
                shutil.rmtree(old, ignore_errors=True)
                done = {k: {**v, "files": [f.replace(f"records/{kind}.building-{os.getpid()}/", f"records/{kind}/") for f in v["files"]]} for k, v in done.items()}
            unshaped: dict[str, list[str]] = defaultdict(list)
            for e in entries:
                why = (done.get(e.id) or {}).get("unshaped")
                if why:
                    unshaped[e.workspace].append(json.dumps({"entry": e.id, "why": why}) + "\n")
            for ws, lines in unshaped.items():
                path = layout.unshaped(kind, ws)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(sorted(lines)), encoding="utf-8")
            state_path.parent.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({"version": version, "entries": done}, sort_keys=True), encoding="utf-8")
            n = sum(v.get("records") or 0 for v in done.values())
            report["by_kind"][kind] = {"records": n, "unshaped": sum(len(v) for v in unshaped.values()), "shaped_now": len(todo), "whole": whole}
            report["records"] += n
            report["unshaped"] += report["by_kind"][kind]["unshaped"]
    return report


def records(layout: Layout, kind: str) -> Iterator[StoreRecord]:
    """Every shaped record of a kind (for the curators and the builders), file by file."""
    base = layout.records_of(kind)
    for path in sorted(base.rglob("*.jsonl.gz")) if base.exists() else []:
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for line in fh:
                yield StoreRecord.read(line)


__all__ = ["shape", "records", "TEXT_KEYS", "CHUNK"]
