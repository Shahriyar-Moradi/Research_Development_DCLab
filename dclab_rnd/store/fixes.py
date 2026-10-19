"""Fixes, linked (32.6): a failing test joined to the run where it passed next and the commits between, proposed as a fix
for a person to accept.

``link(runs_dir)`` reads the kept test runs (``training_records/test_runs``: ``runs.jsonl`` and each run's units, whole) in
time order. For every test file that failed in a run, the next run in which that file ran and passed closes it; the fix
carries the failing unit's whole output, both runs, both commits, and the commits between with their subjects and the
files they changed (``git log``; names only, never a diff's content). A file fixed in the working tree before any commit
says so. ``propose`` keeps each fix under ``raw/fixes/`` (the keeper ``fixes``); ``accept`` is a person's yes, written to
``research/llm-fine-tuning/experiments/dev_trajectories/fixes/<id>.json`` beside the development records (never into them
by itself: a record is written by its package).
"""

from __future__ import annotations

import gzip
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .layout import ROOT

RUNS = ROOT / "training_records" / "test_runs"
ACCEPTED = ROOT / "research" / "llm-fine-tuning" / "experiments" / "dev_trajectories" / "fixes"


def _units(runs_dir: Path, path: str) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    try:
        with gzip.open(Path(runs_dir) / path, "rt", encoding="utf-8") as fh:
            for line in fh:
                doc = json.loads(line)
                if doc.get("kind") == "unit":
                    name = str(doc.get("file") or "").split("::")[0]
                    prev = out.get(name)
                    out[name] = doc if prev is None else {**prev, "ok": prev.get("ok") and doc.get("ok"),
                                                          "output": (prev.get("output") or "") + (doc.get("output") or "")}
    except (OSError, ValueError):
        pass
    return out


def _commits(older: Optional[str], newer: Optional[str], repo: Path = ROOT) -> list[dict[str, Any]]:
    if not older or not newer or older == newer:
        return []
    try:
        text = subprocess.run(["git", "log", "--format=%x00%H%x09%s", "--name-only", f"{older}..{newer}"], cwd=repo, capture_output=True,
                              text=True, timeout=30, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return [{"sha": None, "subject": f"the commits between {older[:10]} and {newer[:10]} could not be read here", "files": []}]
    out = []
    for block in text.split("\x00")[1:]:
        head, *files = [line for line in block.splitlines() if line.strip()]
        sha, _, subject = head.partition("\t")
        out.append({"sha": sha, "subject": subject, "files": files})
    return out


def _failed_tests(output: str) -> list[str]:
    return sorted({line.split(":", 1)[1].strip().split(" ")[0] for line in (output or "").splitlines() if line.startswith(("FAIL:", "ERROR:"))})


def link(runs_dir: Path = RUNS, repo: Path = ROOT) -> Iterator[dict[str, Any]]:
    """Every failure with the run that fixed it (or still open, when no later run passed it)."""
    index = Path(runs_dir) / "runs.jsonl"
    if not index.exists():
        return
    runs = []
    for line in index.read_text(encoding="utf-8").splitlines():
        try:
            runs.append(json.loads(line))
        except ValueError:
            continue
    runs.sort(key=lambda r: (r.get("at") or "", r.get("run") or ""))
    cache: dict[str, dict[str, dict[str, Any]]] = {}

    def units(run: dict[str, Any]) -> dict[str, dict[str, Any]]:
        if run["run"] not in cache:
            cache[run["run"]] = _units(runs_dir, run.get("path") or "")
        return cache[run["run"]]

    for i, run in enumerate(runs):
        for name in run.get("failing_files") or []:
            failing = units(run).get(name, {})
            fixed = None
            for later in runs[i + 1:]:
                if later.get("pg") != run.get("pg"):
                    continue  # a PostgreSQL failure is fixed by a PostgreSQL pass, a file one by a file pass
                unit = units(later).get(name)
                if unit is None:
                    continue
                if unit.get("ok"):
                    fixed = later
                break  # the next run of that file decides: passed (fixed) or failed again (a later failure links on its own)
            ident = f"fix:{run['run']}:{name}"
            yield {"id": ident, "test": name, "tests": _failed_tests(failing.get("output") or ""), "pg": bool(run.get("pg")),
                   "failing_run": run["run"], "failing_at": run.get("at"), "failing_commit": run.get("commit"), "failing_dirty": run.get("dirty"),
                   "failing_output": failing.get("output"),  # whole: the owner's rule
                   "passing_run": fixed and fixed["run"], "passing_at": fixed and fixed.get("at"), "passing_commit": fixed and fixed.get("commit"),
                   "commits": _commits(run.get("commit"), fixed.get("commit"), repo) if fixed else [],
                   "in_working_tree": bool(fixed and fixed.get("commit") == run.get("commit")),
                   "status": "proposed" if fixed else "open", "source": None}


def keeper_sources(env: Any) -> Iterator[Any]:
    """The keeper's sources: each fix as ``fixes/<id>.json``, a new version when it was closed or accepted."""
    from .keepers import Source

    runs_dir = Path(env["test_runs"]) if isinstance(env, dict) and env.get("test_runs") else RUNS
    if isinstance(env, dict) and env.get("only"):
        return
    accepted = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in ACCEPTED.glob("*.json")} if ACCEPTED.exists() else {}
    for fix in link(runs_dir):
        key = fix["id"].replace(":", "_")
        if key in accepted:
            fix = {**fix, **{k: accepted[key][k] for k in ("status", "source", "accepted_by", "accepted_at", "note") if k in accepted[key]}}
        yield Source("rnd", f"fixes/{key}.json", json.dumps(fix, ensure_ascii=False, indent=1).encode(), consent="train")


def accept(fix_id: str, by: str, note: str = "", runs_dir: Path = RUNS, out: Path = ACCEPTED, repo: Path = ROOT) -> Path:
    """A person's yes to one proposed fix: kept beside the development records with who accepted it."""
    fix = next((f for f in link(runs_dir, repo) if f["id"] == fix_id), None)
    if fix is None:
        raise KeyError(f"no fix {fix_id!r}")
    if fix["status"] != "proposed":
        raise ValueError(f"{fix_id} is {fix['status']}: only a proposed fix (a later run passed it) can be accepted")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{fix_id.replace(':', '_')}.json"
    doc = {**fix, "status": "accepted", "source": "person", "accepted_by": by, "note": note,
           "accepted_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    return path


__all__ = ["link", "accept", "keeper_sources", "RUNS", "ACCEPTED"]
