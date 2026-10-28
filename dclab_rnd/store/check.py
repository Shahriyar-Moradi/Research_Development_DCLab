"""The storing phase's check (32.18): ``make store-check`` runs one scripted day in a scratch workspace and captures it end to
end, through all three tiers, the curators, the sets and the tutor, then checks what the phase promises.

The day:

- a draft and a project, with its graph moves;
- a decision with a person's label;
- an intern session saved three times;
- a job through three states;
- a team's walk;
- a model exchange with its thinking;
- an MCP call;
- a test failing, then passing;
- a project deleted;
- a planted key in a log and a planted contact in a person's words.

The checks:

- every kind the day produced is in ``raw/`` and in ``records/``;
- no file is kept twice;
- the deleted project is still whole, its deletion recorded;
- no planted key or contact reaches ``raw/`` or ``records/``;
- every set's sealed groups are apart from its train and dev groups;
- the run is a session of the conductor.

Each check passes or says why.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional
from unittest import mock

KEY = "sk-ant-api03-" + "Z" * 32
CONTACT = "jane.check@example.org"
AT = "2026-10-20T10:00:00+00:00"


def _day(home: Path, runs: Path) -> dict[str, Any]:
    """The scripted day, in ``home`` (a workspace under an ``agent_runs`` folder) and ``runs`` (the test runs)."""
    from fastapi.testclient import TestClient

    from ..agentic.server import create_app
    from ..intern.sessions import SessionStore
    from ..jobs.store import open_jobs

    (home / "consent.json").write_text(json.dumps({"train": True, "improve": True}), encoding="utf-8")
    client = TestClient(create_app(home))
    services = client.app.state.services
    h = {"X-DCLab-Token": client.get("/api/config").json()["csrf"]}
    draft = services.drafts.create(f"Which orders arrive late? Mail me at {CONTACT}")
    pid = client.post("/api/projects", json={"name": "Late orders", "goal": "late orders"}, headers=h).json()["id"]
    gone = client.post("/api/projects", json={"name": "To delete", "goal": "g"}, headers=h).json()["id"]
    services.drafts.update(draft["id"], lambda d: d.update(project_id=pid))
    for i, (move, a, b, status) in enumerate([("set_solution", "WF-01", "WF-02", "allowed"), ("read_holdout", "WF-02", "WF-02", "blocked")]):
        services.projects.transition(pid, {"at": f"2026-10-20T10:00:0{i}+00:00", "actor": "human", "move": move, "from": a, "to": b, "status": status,
                                           "rules": [], "failed_checks": []})
    ledger_dir = home / "ledger" / "decisions" / "2026" / "10"
    ledger_dir.mkdir(parents=True, exist_ok=True)
    (ledger_dir / "20.jsonl").write_text(json.dumps({"schema_version": "dclab_core.decision_record.v2", "id": "dgcheck0000000000000001",
                                                     "point": "column.is_identifier", "at": AT, "project": pid, "state": {"rows": "medium"},
                                                     "answers": [{"by": "rule", "value": True}], "used": "rule", "value_used": True}) + "\n", encoding="utf-8")
    labels = home / "ledger" / "labels" / "2026" / "10"
    labels.mkdir(parents=True, exist_ok=True)
    (labels / "20.jsonl").write_text(json.dumps({"schema_version": "dclab_core.decision_label.v2", "id": "dlcheck1", "record": "dgcheck0000000000000001",
                                                 "point": "column.is_identifier", "value": True, "source": "user_decision_exposed", "at": AT}) + "\n", encoding="utf-8")
    sessions = SessionStore(home / "intern")
    s = sessions.create("Look around", mode="standard", model=None)
    for n in range(2):
        s["steps"] = [{"n": i} for i in range(n + 1)]
        sessions.save(s)
    jobs = open_jobs(home)
    job = jobs.enqueue("stage", f"{pid}:data", {})
    jobs.claim("w1")
    jobs.finish(job["id"], "done", worker="w1")
    (home / "orchestra").mkdir(exist_ok=True)
    (home / "orchestra" / "walks.jsonl").write_text(json.dumps({"walk": "w1", "node": "understand", "board": {}}) + "\n", encoding="utf-8")
    exchanges = home / "model_exchanges" / "2026" / "10"
    exchanges.mkdir(parents=True, exist_ok=True)
    (exchanges / "20.jsonl").write_text(json.dumps({"at": AT, "model_request": "mrc1", "purpose": "home_agent", "whole": True,
                                                    "messages": [{"role": "user", "content": "Predict late orders."}], "reply": "When?",
                                                    "thinking": [{"text": "The moment decides.", "signature": "s"}]}) + "\n", encoding="utf-8")
    (home / "mcp").mkdir(exist_ok=True)
    (home / "mcp" / "calls.jsonl").write_text(json.dumps({"tool": "search_evidence", "arguments": {"query": "leakage"}, "result": {}}) + "\n", encoding="utf-8")
    (home / "audit_extra.log").write_text(f"a pasted key {KEY}\n", encoding="utf-8")
    for ident, at, ok in (("r1", "2026-10-20T09:00:00+00:00", False), ("r2", "2026-10-20T11:00:00+00:00", True)):
        day = runs / "2026-10-20"
        day.mkdir(parents=True, exist_ok=True)
        with gzip.open(day / f"{ident}.jsonl.gz", "wt", encoding="utf-8") as fh:
            fh.write(json.dumps({"kind": "unit", "run": ident, "file": "test_a.py", "ok": ok,
                                 "output": f"FAIL: test_x (t.T) env {KEY}" if not ok else "OK"}) + "\n")  # a key inside a gzip too
        with open(runs / "runs.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"run": ident, "at": at, "commit": "c", "pg": False, "ok": ok, "failing_files": [] if ok else ["test_a.py"],
                                 "path": f"2026-10-20/{ident}.jsonl.gz"}) + "\n")
    client.delete(f"/api/projects/{gone}", headers=h)
    return {"project": pid, "deleted": gone, "draft": draft["id"], "session": s["id"]}


def _dclab_code(home: Path) -> Path:
    """32.33: DCLab Code's records: a session on this repository (with a pasted key) and one on another folder."""
    from .layout import ROOT

    (home / "records").mkdir(parents=True)
    (home / "records" / "s_here01.jsonl").write_text(json.dumps({"kind": "sdk", "message": {"type": "user", "content": f"cat .env: {KEY}"}}) + "\n", encoding="utf-8")
    (home / "records" / "s_away01.jsonl").write_text(json.dumps({"kind": "sdk", "message": {"content": "another folder"}}) + "\n", encoding="utf-8")
    (home / "sessions.json").write_text(json.dumps({"sessions": [{"id": "s_here01", "projectPath": str(ROOT)},
                                                                 {"id": "s_away01", "projectPath": "/elsewhere/product"}]}), encoding="utf-8")
    return home


def run(log: Any = print) -> dict[str, Any]:
    """The day, captured, and every check; ``ok`` only when all pass."""
    from .. import experience
    from . import Layout, Store, conductor, shapers
    from .keepers import standard

    scratch = Path(tempfile.mkdtemp(prefix="dclab-store-check-"))
    base, runs = scratch / "agent_runs", scratch / "test_runs"
    home = base / "day"
    home.mkdir(parents=True)
    layout = Layout(scratch / "store")
    env = {"DCLAB_STORE": str(layout.root), "DCLAB_STORE_CAPTURE": "1", "DCLAB_STORE_LOOP": "0", "DCLAB_DATABASE_URL": "",
           "DCLAB_STORE_CLAUDE_PROJECTS": "off", "DCLAB_LOG_PROMPTS": "1"}
    checks: list[dict[str, Any]] = []
    with mock.patch.dict(os.environ, env), mock.patch("dclab_rnd.store.keepers.workspaces.AGENT_RUNS", base):
        log_ = experience.FileLog(scratch / "conductor_experience", "check")
        experience.install(log_)
        try:
            ids = _day(home, runs)
            code_home = _dclab_code(scratch / "dclab-code")
            record = conductor.conduct({"agent_runs": base, "test_runs": runs, "claude_projects": None, "dclab_code_home": code_home},
                                       layout, standard.install({}))
        finally:
            experience.install(None)
        store = Store(layout)

        def check(name: str, ok: bool, detail: str = "") -> None:
            checks.append({"check": name, "ok": bool(ok), "detail": detail})

        produced = ("sessions", "decisions", "conversations", "workflows", "tasks", "tests", "model_requests", "deletions", "fixes", "curation", "other")
        for kind in produced:
            check(f"{kind} kept in raw", bool(store.entries(kind)))
        for kind in ("sessions", "decisions", "conversations", "workflows", "tasks", "model_requests", "fixes"):
            check(f"{kind} shaped into records", any(True for _ in shapers.records(layout, kind)))
        keys = [(e.kind, e.workspace, e.source, e.version) for k in produced for e in store.entries(k)]
        check("nothing kept twice", len(keys) == len(set(keys)))
        workflows = {e.source for e in store.entries("workflows")}
        check("the deleted project is still whole", f"projects/{ids['deleted']}/project.json" in workflows)
        check("its deletion is recorded", any(ids["deleted"] in e.source for e in store.entries("deletions")))
        def opened(path: Path) -> bytes:
            data = gzip.decompress(path.read_bytes())
            return gzip.decompress(data) if data[:2] == b"\x1f\x8b" else data  # a source that was itself a gzip, opened too

        raw_text = b"".join(opened(p) for p in layout.tier("raw").rglob("*.gz"))
        records_text = b"".join(gzip.decompress(p.read_bytes()) for p in layout.tier("records").rglob("*.gz"))
        check("no planted key in raw or records", KEY.encode() not in raw_text and KEY.encode() not in records_text)
        check("no planted contact in records", CONTACT.encode() not in records_text)
        code = {e.source for e in store.entries("agents")}  # 32.33
        check("a DCLab Code session on this repository kept", "dclab_code/records/s_here01.jsonl" in code)
        check("a DCLab Code session on another folder only counted", "dclab_code/records/s_away01.jsonl" not in code
              and "dclab_code/~other-projects~counts.json" in code)
        leaks = []
        for manifest in layout.tier("sets").rglob("MANIFEST.json"):
            groups = {}
            for split in ("train", "dev", "sealed"):
                path = manifest.parent / f"{split}.jsonl"
                rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []
                groups[split] = {(r.get("provenance") or {}).get("project") for r in rows if (r.get("provenance") or {}).get("project")}
            if groups["sealed"] & (groups["train"] | groups["dev"]):
                leaks.append(manifest.parent.relative_to(layout.tier("sets")).as_posix())
        check("sealed groups apart in every set", not leaks, ", ".join(leaks))
        # the owner (2026-10-28): every training example carries reasoning with its source, or says why not
        known = {"recorded", "from_record", "teacher", "visible", "replayed", "lost", "none", "missing", "no_field"}
        newest: dict[Path, Path] = {}  # the newest version of each set (an older version is never rewritten)
        for m in layout.tier("sets").rglob("MANIFEST.json"):
            folder, version = m.parent.parent, m.parent.name
            if folder not in newest or int(version.lstrip("v") or 0) > int(newest[folder].parent.name.lstrip("v") or 0):
                newest[folder] = m
        manifests = list(newest.values())
        missing = [m.parent.relative_to(layout.tier("sets")).as_posix() for m in manifests if not (m.parent / "REASONING.json").exists()]
        check("every set states its reasoning by source", bool(manifests) and not missing, ", ".join(missing[:5]))
        odd = [m.parent.relative_to(layout.tier("sets")).as_posix() for m in manifests if (m.parent / "REASONING.json").exists()
               and set(json.loads((m.parent / "REASONING.json").read_text(encoding="utf-8"))) - known]
        check("every reasoning has a known source", not odd, ", ".join(odd[:5]))
        check("the run recorded every step", {s["name"] for s in record["steps"]} >= {"sessions", "decisions", "shapers", "curators", "builders"})
        check("no keeper failed", not record["failed"], ", ".join(record["failed"]))
    ok = all(c["ok"] for c in checks)
    for c in checks:
        log(f"{'ok  ' if c['ok'] else 'FAIL'} {c['check']}" + (f": {c['detail']}" if c["detail"] else ""))
    log(f"store-check: {'passed' if ok else 'FAILED'} ({sum(c['ok'] for c in checks)} of {len(checks)})")
    return {"ok": ok, "checks": checks, "scratch": str(scratch)}


def main(argv: Optional[list[str]] = None) -> int:
    return 0 if run()["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["run", "main"]
