"""Trusted testers running the studio on their own machines (package 25.11): everything they do comes back to the owner.

A tester clones the repository, runs it locally with their own Anthropic key, and works as anyone would. Their
workspace folder already records everything (the session record, the decision ledger, the audit, the twin folder with
Claude beside the system, the intern's sessions, the projects' stage records). This package adds the way back:

* :func:`setup` (``python -m dclab_rnd.testers setup`` / ``make tester-setup``): shows the notice and needs the
  tester's agreement, writes ``tester.json`` (their id, name, e-mail, the records repository, when they agreed), turns
  the workspace's consent (improve and train) and Claude beside the system on, caps the month at 15
  (``DCLAB_WORKSPACE_MONTHLY_EUR`` in their ``.env``; their key stays theirs), and clones the owner's private records
  repository.
* :func:`share` (``make share``): every record file that changed since the last share is copied, whole, into
  ``<records repo>/<tester id>/<when>/files/…`` with a ``manifest.json`` (the files with their SHA-256 and sizes, the code's
  commit, the tables included), then committed and pushed. A table (a project's or a draft's data file) goes only when
  the tester said yes to it, once per table (remembered in ``tester.json``); a file over 45 MB is left out and said.
  Never ``.env``, a key, or anything outside the workspace's record files.
* :func:`collect` (``make collect``, the owner's): pulls the records repository and rebuilds each tester's workspace
  under ``<owner workspace>/testers/<tester id>/`` (the bundles applied in order, a later copy of a file replacing the
  earlier one), so the owner's Evaluation page and the command-line readers (with ``DCLAB_AGENT_HOME`` set to a
  tester's folder) see them;
  :func:`summary` says per tester what came back.

Nothing here runs a model. Git runs as the person who runs the command, with their own access to the repository.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from ..agents.traces import CELL_KEYS

FILE = "tester.json"
NOTICE_VERSION = "1"
MONTHLY_EUR = 15.0
MAX_FILE_BYTES = 45 * 1024 * 1024  # GitHub refuses files over 100 MB and warns over 50
NOTICE = """DCLab R&D: testing on your machine

You are testing the DCLab R&D studio for its owner. While you use it, everything you do in it is recorded in your
workspace folder: the problems you describe and your messages, every answer and decision DCLab and Claude make, the
projects you build and their results, and the time and cost of each model request. `make share` sends these records,
as they are, to the owner's private repository, where they are used to improve DCLab and to train its models.
A table you upload is sent only if you say yes to that table when you share. Your Anthropic key never leaves your
machine; your model spend is capped at 15 a month. You can stop sharing at any time; ask the owner to delete what you
sent."""

# the workspace's record files (relative globs); tables are separate and need a yes each
RECORDS = ("experience/**/*.jsonl", "ledger/**/*.jsonl", "audit.jsonl", "agent_steps.jsonl", "model_requests.jsonl", "model_shadow.jsonl",
           "consent.json", "twin.json", "budgets.json", ".dclab_workspace", "twin/**/*", "intern/**/*.json",
           "projects/*/project.json", "projects/*/stages/*.json", "projects/*/transitions.jsonl", "projects/*/activity.jsonl",
           "projects/*/memory.jsonl", "drafts/*/draft.json", "drafts/*/events.jsonl", "datasets/**/*", "lessons/**/*")
TABLES = ("projects/*/data/*", "drafts/*/data/*")
NEVER = re.compile(r"(^|/)\.env|api[_-]?key|secret|token|\.pem$|\.key$", re.I)
QUOTES_CELLS = frozenset(CELL_KEYS | {"categories", "numeric_levels", "prefix", "shape", "digits", "top"})  # what quotes a table's cells
SCRUBBED = re.compile(r"^(projects/[^/]+/(project\.json|stages/.*\.json|activity\.jsonl)|drafts/[^/]+/(draft\.json|events\.jsonl)|intern/.*\.json)$")


class TesterError(RuntimeError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=check)


def load(home: Path) -> Optional[dict[str, Any]]:
    path = Path(home) / FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _save(home: Path, doc: dict[str, Any]) -> None:
    path = Path(home) / FILE
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def slug(name: str) -> str:
    return (re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:24] or "tester") + "-" + secrets.token_hex(3)


# ---------------------------------------------------------------------------------------------------- setup


def setup(home: Path, *, name: str, email: str, repo: str, clone: Optional[Path] = None, agreed: bool = False,
          env_file: Optional[Path] = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The tester's agreement, their id, the switches, the cap, and a clone of the records repository."""
    if not agreed:
        raise TesterError("the tester must agree to the notice first (--agree after reading it)")
    if not name.strip() or "@" not in email:
        raise TesterError("a name and an e-mail address are needed")
    _files_only()
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    doc = load(home) or {"id": slug(name), "tables": {}, "shared": {}}
    doc.update(name=name.strip()[:80], email=email.strip()[:120], repo=repo, agreed_at=now(), notice_version=NOTICE_VERSION)
    from .. import consent
    from ..twin import home as twin_home

    for use in ("improve", "train"):
        consent.set_workspace_consent(home, use, True)
    twin_home.set_settings(home, {"on": True})
    if env_file is not None:
        _cap(env_file)
    clone = Path(clone) if clone is not None else home / "tester-records"
    if not (clone / ".git").is_dir():
        result = subprocess.run(["git", "clone", repo, str(clone)], capture_output=True, text=True)
        if result.returncode != 0:
            raise TesterError(f"could not clone the records repository: {result.stderr.strip()[:300]} (ask the owner for access)")
    doc["clone"] = str(clone)
    _save(home, doc)
    log(f"tester {doc['id']} set up: consent on, Claude beside the system on, the month capped at {MONTHLY_EUR:g}; records go to {repo}")
    return doc


def _files_only() -> None:
    if os.environ.get("DCLAB_DATABASE_URL", "").strip():
        raise TesterError("a tester's workspace keeps its records in files: leave DCLAB_DATABASE_URL empty in .env and start DCLab with make notebook")


def _cap(env_file: Path) -> None:
    """The month capped at 15 in the tester's .env: a missing, empty or higher cap is set to 15; a lower one stays."""
    text = env_file.read_text(encoding="utf-8") if env_file.is_file() else ""
    line = f"DCLAB_WORKSPACE_MONTHLY_EUR={MONTHLY_EUR:g}   # the tester's monthly model cap (25.11)"
    found = re.search(r"^\s*DCLAB_WORKSPACE_MONTHLY_EUR\s*=\s*([^\s#]*).*$", text, re.M)
    if found:
        try:
            keep = 0 < float(found.group(1)) <= MONTHLY_EUR
        except ValueError:
            keep = False
        if keep:
            return
        text = text[:found.start()] + line + text[found.end():]
    else:
        text = text + ("\n" if text and not text.endswith("\n") else "") + line + "\n"
    env_file.write_text(text, encoding="utf-8")


def _scrub(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items() if k not in QUOTES_CELLS}
    if isinstance(value, list):
        return [_scrub(v) for v in value]
    return value


def _copy(path: Path, target: Path, scrub: bool) -> None:
    """The file into the bundle; with ``scrub``, a JSON or JSON-lines record without the keys that quote a table's cells."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if not scrub:
        shutil.copy2(path, target)
        return
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        lines = []
        for line in text.splitlines():
            try:
                lines.append(json.dumps(_scrub(json.loads(line)), ensure_ascii=False))
            except ValueError:
                continue
        target.write_text("".join(x + "\n" for x in lines), encoding="utf-8")
    else:
        try:
            target.write_text(json.dumps(_scrub(json.loads(text)), ensure_ascii=False, indent=1), encoding="utf-8")
        except ValueError:
            target.write_text("{}", encoding="utf-8")


def _pending(clone: Path) -> bool:
    """Commits made here and not yet pushed."""
    out = _git(clone, "rev-list", "--count", "@{u}..HEAD", check=False)
    return out.returncode == 0 and out.stdout.strip() not in ("", "0")


def _push(clone: Path) -> None:
    out = _git(clone, "push", "--quiet", check=False)
    if out.returncode != 0:  # another tester pushed first: take theirs (their folders only) and try once more
        _git(clone, "pull", "--rebase", "--quiet", check=False)
        out = _git(clone, "push", "--quiet", check=False)
    if out.returncode != 0:
        raise TesterError(f"the bundle is committed but not pushed: {out.stderr.strip()[:300]} (run make share again when online)")


# ---------------------------------------------------------------------------------------------------- share


def _files(home: Path, globs: tuple[str, ...]) -> list[Path]:
    out: set[Path] = set()
    for pattern in globs:
        for path in home.glob(pattern):
            if path.is_file() and not NEVER.search(path.relative_to(home).as_posix()) and "tester-records" not in path.relative_to(home).parts:
                out.add(path)
    return sorted(out)


def share(home: Path, *, ask: Optional[Callable[[str, int], bool]] = None, push: bool = True, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The records that changed since the last share, and the tables the tester agreed to, committed and pushed."""
    _files_only()
    home = Path(home)
    doc = load(home)
    if doc is None:
        raise TesterError("not set up as a tester: run make tester-setup first")
    from .. import consent

    switches = consent.workspace_consent(home)
    if not (switches.get("improve") and switches.get("train")):
        raise TesterError("this workspace's consent is off: nothing is shared (turn it on again with make tester-setup, or stop here)")
    clone = Path(doc["clone"])
    if not (clone / ".git").is_dir():
        raise TesterError(f"the records repository is not at {clone}: run make tester-setup again")
    tables = doc.setdefault("tables", {})
    chosen: list[Path] = []
    for path in _files(home, TABLES):
        rel = path.relative_to(home).as_posix()
        digest = _sha(path)
        known = tables.get(rel)
        if not isinstance(known, dict) or known.get("sha256") != digest:  # a new table, or new content under the same name: asked again
            tables[rel] = {"sha256": digest, "yes": bool(ask(rel, path.stat().st_size)) if ask is not None else False}
        if tables[rel]["yes"]:
            chosen.append(path)
    agreed_owners = {"/".join(p.relative_to(home).parts[:2]) for p in chosen}  # projects/<id> or drafts/<id> whose table goes too
    shared = doc.setdefault("shared", {})
    current = {p.relative_to(home).as_posix(): p for p in _files(home, RECORDS) + chosen}
    removed = sorted(rel for rel in shared if rel not in current)
    when = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + secrets.token_hex(3)  # in order of time, and two shares never meet
    bundle = clone / doc["id"] / when
    files, skipped = [], []
    staged: dict[str, str] = {}
    for rel, path in sorted(current.items()):
        size = path.stat().st_size
        if size > MAX_FILE_BYTES:
            skipped.append({"path": rel, "bytes": size, "why": "over 45 MB"})
            continue
        scrub = bool(SCRUBBED.match(rel)) and "/".join(Path(rel).parts[:2]) not in agreed_owners
        target = bundle / "files" / rel
        _copy(path, target, scrub)  # copied first, then hashed: what the manifest says is what was sent
        digest = _sha(target)
        if shared.get(rel) == digest:
            target.unlink()
            continue
        staged[rel] = digest
        files.append({"path": rel, "sha256": digest, "bytes": target.stat().st_size, "table": path in chosen, "cells_removed": scrub})
    if push and _pending(clone):  # an earlier bundle committed offline goes first
        _push(clone)
    if not files and not removed:
        shutil.rmtree(bundle, ignore_errors=True)
        _save(home, doc)
        log("nothing new to share")
        return {"bundle": None, "files": 0, "skipped": skipped}
    if push:
        _git(clone, "pull", "--rebase", "--quiet", check=False)  # other testers' folders since; never this one
    commit = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).resolve().parents[2]).stdout.strip() or None
    except OSError:
        pass
    manifest = {"tester": doc["id"], "name": doc.get("name"), "at": now(), "notice_version": doc.get("notice_version"), "code_commit": commit,
                "files": files, "removed": removed, "skipped": skipped, "tables": [f["path"] for f in files if f["table"]]}
    bundle.mkdir(parents=True, exist_ok=True)
    (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    profile = clone / doc["id"] / "tester.json"
    profile.write_text(json.dumps({k: doc.get(k) for k in ("id", "name", "email", "agreed_at", "notice_version")}, indent=2), encoding="utf-8")
    _git(clone, "add", str(bundle.relative_to(clone)), str(profile.relative_to(clone)))
    out = _git(clone, "-c", "user.name=" + (doc.get("name") or doc["id"]), "-c", "user.email=" + (doc.get("email") or "tester@dclab"),
               "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", f"{doc['id']}: {len(files)} files ({when})", check=False)
    if out.returncode != 0:
        raise TesterError(f"the bundle could not be committed: {out.stderr.strip()[:300]}")
    if push:
        _push(clone)
    shared.update(staged)  # only now: a bundle that never reached the owner is sent again
    for rel in removed:
        shared.pop(rel, None)
    doc["last_share"] = now()
    _save(home, doc)
    log(f"shared {len(files)} files ({sum(1 for f in files if f['table'])} tables, {len(removed)} removed) as {doc['id']}/{when}")
    return {"bundle": str(bundle), "files": len(files), "tables": manifest["tables"], "removed": removed, "skipped": skipped}


# ---------------------------------------------------------------------------------------------------- collect (the owner's)


def collect(home: Path, records: Path, *, pull: bool = True, log: Callable[[str], None] = print) -> dict[str, Any]:
    """The owner's side: pull the records repository and rebuild every tester's workspace under ``<home>/testers/``."""
    home, records = Path(home), Path(records)
    if not (records / ".git").is_dir():
        raise TesterError(f"no records repository at {records}: clone the private repository there first")
    if pull:
        out = _git(records, "pull", "--quiet", check=False)
        if out.returncode != 0:
            log(f"could not pull ({out.stderr.strip()[:200]}): rebuilding from what is here")
    out: dict[str, Any] = {}
    for tester in sorted(p for p in records.iterdir() if p.is_dir() and not p.is_symlink() and not p.name.startswith(".")):
        target = (home / "testers" / tester.name).resolve()
        applied = skipped = 0
        for bundle in sorted(b for b in tester.iterdir() if b.is_dir() and not b.is_symlink() and (b / "manifest.json").is_file()):
            try:
                manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
                if not isinstance(manifest, dict):
                    raise ValueError("not a record")
            except ValueError as exc:
                log(f"{tester.name}/{bundle.name}: the manifest cannot be read ({exc}); skipped")
                continue
            root = (bundle / "files").resolve()
            for f in manifest.get("files") or []:
                rel = str(f.get("path") or "") if isinstance(f, dict) else ""
                source = bundle / "files" / rel
                dest = (target / rel).resolve()
                inside = rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts and dest.is_relative_to(target)
                if not inside or not source.is_file() or source.is_symlink() or not source.resolve().is_relative_to(root) or _sha(source) != f.get("sha256"):
                    skipped += 1  # a path outside the tester's folder, a link, or a file that does not match its manifest is never applied
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, dest)
            for rel in manifest.get("removed") or []:  # what the tester deleted is deleted here too
                dest = (target / str(rel)).resolve()
                if isinstance(rel, str) and ".." not in Path(rel).parts and dest.is_relative_to(target) and dest.is_file():
                    dest.unlink()
            applied += 1
        if (tester / "tester.json").is_file() and not (tester / "tester.json").is_symlink():
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(tester / "tester.json", target / "tester.json")
        out[tester.name] = {"bundles": applied, "skipped_files": skipped, "folder": str(target)}
        log(f"{tester.name}: {applied} bundles → {target}" + (f" ({skipped} files skipped)" if skipped else ""))
    return out


def summary(home: Path) -> list[dict[str, Any]]:
    """Per tester, what came back: their sessions, drafts, projects, Claude's turns beside them, decisions, spend."""
    rows = []
    for folder in sorted((Path(home) / "testers").glob("*")) if (Path(home) / "testers").is_dir() else []:
        profile = json.loads((folder / "tester.json").read_text(encoding="utf-8")) if (folder / "tester.json").is_file() else {}
        spend = 0.0
        for line in (folder / "model_requests.jsonl").read_text(encoding="utf-8").splitlines() if (folder / "model_requests.jsonl").is_file() else []:
            try:
                spend += float(json.loads(line).get("cost_eur") or 0.0)
            except ValueError:
                pass
        rows.append({"tester": folder.name, "name": profile.get("name"), "agreed_at": profile.get("agreed_at"),
                     "visits": len(list((folder / "experience").rglob("*.jsonl"))) if (folder / "experience").is_dir() else 0,
                     "drafts": len(list(folder.glob("drafts/*/draft.json"))), "projects": len(list(folder.glob("projects/*/project.json"))),
                     "twin_turns": sum(1 for p in folder.glob("twin/sessions/**/ai_first.jsonl") for _ in p.read_text(encoding="utf-8").splitlines()),
                     "decisions": sum(1 for p in folder.glob("ledger/decisions/**/*.jsonl") for _ in p.read_text(encoding="utf-8").splitlines()),
                     "model_eur": round(spend, 4)})
    return rows
