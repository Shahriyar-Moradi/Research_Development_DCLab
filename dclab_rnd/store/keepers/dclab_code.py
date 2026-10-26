"""DCLab Code's own records (32.33): the owner's developer app (the separate ``dclab-code`` repository) records every
session in ``~/.dclab-code/records/<session>.jsonl``: each Agent SDK message whole (thinking, tool inputs, full tool
outputs) and, beside them, what only the app knows: the person's messages, every permission asked and the answer, the
sealed and holdout guard's questions, the answers to the agent's questions, plan approvals, notes. ``sessions.json`` holds
each session's title, project, model, cost and its own copy (worktree), if any.

Only sessions on this repository are kept (their project this checkout or a folder in it, the app's own copies under
``.claude/worktrees/`` included), as the agents keeper keeps only this repository's Claude Code sessions (32.7): a session
on another folder (the main product, a client's data) is counted (name, size, sha256), never kept, and left out of the
kept ``sessions.json``. The SDK's own transcripts of these sessions are Claude Code's, kept by the agents keeper; a record
links to its transcript by ``sdkSessionId``.

Scripted live tests (the owner, 2026-10-26): a DCLab Code home under ``agent_runs/live_tests/`` whose ``config.json`` says
``decidedBy: "script"`` is read too, kept apart under ``dclab_code/tests/<its folder>/``: its sessions on this repository or on
a throwaway scratch folder (the system's temp folders) are kept, another repository's only counted; the sets read every act in
them as the script's, never a person's label (even a record older than ``by``), so they are agent episodes only.

Kept under ``raw/agents/`` (workspace ``rnd``, source ``dclab_code/...``), each record by its new lines, anything shaped
like a credential replaced by a marker. Read only on the owner's machine: never on a tester's (a ``tester.json`` in the
workspace) unless ``DCLAB_STORE_DCLAB_CODE`` names the folder; ``DCLAB_STORE_CLAUDE_PROJECTS=off`` (the tests' switch for
other agents' sessions) or ``DCLAB_STORE_DCLAB_CODE=off`` reads nothing, and a check or test that names the agents'
folders reads only the folder it hands over.

Their own builder (``sets/dclab_code.py``, 32.34) turns them into M6 episodes, M7 judgements and M2/M1 decisions, each
session's split keyed by its ``sdkSessionId``, the same session as its Claude Code transcript, so the two never land in
different splits. The transcript builders (M5) skip these lines: they are app records and SDK messages wrapped in
``message``, not a transcript's ``user``/``assistant`` lines.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional

from ..layout import ROOT
from . import Claim, Keeper, Source
from .agents import IMAGES, _counted, redact

HOME = Path.home() / ".dclab-code"


def home_folder(env: Any = None) -> Optional[Path]:
    """The app's folder to read, or None. An env that names the agents' folders (a test, a check) decides alone."""
    if isinstance(env, dict) and ("dclab_code_home" in env or "claude_projects" in env):
        value = env.get("dclab_code_home")
        return Path(value) if value else None
    if os.environ.get("DCLAB_STORE_CLAUDE_PROJECTS") == "off":  # the tests' switch wins over any named folder
        return None
    value = os.environ.get("DCLAB_STORE_DCLAB_CODE", "")
    if value == "off":
        return None
    if value:
        return Path(value)
    from .workspaces import AGENT_RUNS

    return None if (AGENT_RUNS / "tester.json").exists() else HOME


def _inside(path: Optional[str], root: Path) -> bool:
    if not path:
        return False
    try:
        p = Path(path).resolve()
    except OSError:
        return False
    r = root.resolve()
    return p == r or r in p.parents


SCRATCH = tuple(Path(p) for p in {tempfile.gettempdir(), "/tmp", "/private/tmp", "/private/var/folders"})  # a live test's throwaway project


def test_homes(env: Any = None) -> list[Path]:
    """DCLab Code homes of scripted live tests under ``agent_runs/live_tests/`` (the owner, 2026-10-26: keep them, as a script's
    acts): a folder with ``sessions.json``, ``records/`` and a ``config.json`` saying ``decidedBy: "script"``. A test or check
    that names its folders decides alone (``dclab_test_homes``); never on a tester's machine; the off switches read nothing."""
    if isinstance(env, dict) and "dclab_test_homes" in env:
        return [Path(p) for p in env["dclab_test_homes"] or []]
    if isinstance(env, dict) and ("dclab_code_home" in env or "claude_projects" in env):
        return []  # a test or check that names the agents' folders reads only those
    if os.environ.get("DCLAB_STORE_CLAUDE_PROJECTS") == "off" or os.environ.get("DCLAB_STORE_DCLAB_CODE") == "off":
        return []
    from .workspaces import AGENT_RUNS

    base = Path(env["agent_runs"]) if isinstance(env, dict) and env.get("agent_runs") else AGENT_RUNS
    if (base / "tester.json").exists():
        return []
    out = []
    for index in sorted((base / "live_tests").glob("*/**/sessions.json")) if (base / "live_tests").is_dir() else ():
        home = index.parent
        if index.is_symlink() or not (home / "records").is_dir() or len(home.relative_to(base).parts) > 5:
            continue
        try:
            if json.loads((home / "config.json").read_text(encoding="utf-8")).get("decidedBy") == "script":
                out.append(home)
        except (OSError, ValueError, AttributeError):
            continue
    return out


def _home_sources(home: Path, root: Path, prefix: str, scripted: bool) -> Iterator[Source]:
    """One DCLab Code home's records and index under ``prefix``: the sessions on this repository (and, in a scripted live
    test, on a throwaway scratch folder) kept, any other folder's counted only."""
    index = home / "sessions.json"
    try:
        sessions = json.loads(index.read_text(encoding="utf-8")).get("sessions", []) if index.is_file() and not index.is_symlink() else []
    except (OSError, ValueError):
        sessions = []
    def here(s: dict[str, Any]) -> bool:
        p = s.get("projectPath")
        return _inside(p, root) or (scripted and any(_inside(p, t) for t in SCRATCH))
    ours = {s.get("id") for s in sessions if isinstance(s, dict) and here(s)}
    records = home / "records"
    others = []
    what = ("a scripted live test's DCLab Code session (every act its driver's, by: script): agent episodes only, never a person's label"
            if scripted else "a DCLab Code session on this repository: SDK messages whole and the person's decisions")
    for path in sorted(records.glob("*.jsonl")) if records.is_dir() and not records.is_symlink() else ():
        if path.is_symlink() or not path.is_file():
            continue
        if path.stem not in ours:
            others.append(path)
            continue
        yield Source("rnd", f"{prefix}records/{path.name}", path, consent="train", log=True, transform=redact,
                     note=f"{what}; anything shaped like a credential replaced")
    if others:
        yield Source("rnd", f"{prefix}~other-projects~counts.json", lambda paths=others: _counted(paths, "a session on another folder: counted only"),
                     consent="train", note="counts only: DCLab Code sessions on other folders are never kept")
    images = []
    for sid in sorted(i for i in ours if isinstance(i, str) and re.fullmatch(r"[A-Za-z0-9_-]+", i)):
        folder = home / "attachments" / sid
        if folder.is_dir() and not folder.is_symlink():
            images += [p for p in sorted(folder.iterdir()) if p.is_file() and not p.is_symlink() and p.suffix.lower() in IMAGES]
    if images:
        yield Source("rnd", f"{prefix}~images~counts.json", lambda paths=images: _counted(paths, "an image the person attached: counted only"),
                     consent="train", note="counts only: images attached in DCLab Code are never kept, no redaction can read them")
    kept = [s for s in sessions if isinstance(s, dict) and s.get("id") in ours]
    if kept:
        yield Source("rnd", f"{prefix}sessions.json", lambda rows=kept: json.dumps({"sessions": rows}, indent=1).encode(), consent="train",
                     transform=redact, note=("a scripted live test's DCLab Code sessions" if scripted else "DCLab Code's sessions on this repository")
                     + ": title, project, model, cost, own copy, SDK session id")


def sources(env: Any) -> Iterator[Source]:
    if isinstance(env, dict) and env.get("only"):
        return  # a workspace's removal is not the development sessions'
    root = Path(env["repo"]) if isinstance(env, dict) and env.get("repo") else ROOT
    from .workspaces import AGENT_RUNS

    base = Path(env["agent_runs"]) if isinstance(env, dict) and env.get("agent_runs") else AGENT_RUNS
    for home in test_homes(env):  # 2026-10-26: the scripted live tests' sessions, kept apart under tests/<their folder>/
        try:
            slug = "-".join(home.relative_to(base).parts)
        except ValueError:
            slug = home.name
        yield from _home_sources(home, root, f"dclab_code/tests/{re.sub(r'[^A-Za-z0-9._-]+', '_', slug)}/", True)
    home = home_folder(env)
    if home is None:
        return
    yield from _home_sources(home, root, "dclab_code/", False)


KEEPER = Keeper("dclab_code", (Claim("agents", "dclab_code/"),), sources, ("m6_stepwise_decisions", "m7_judges", "m2_workflow_decisions", "m1_students"),
                "DCLab Code's sessions on this repository: every SDK message whole, with the person's permissions, answers and plan approvals")

__all__ = ["KEEPER", "HOME", "SCRATCH", "home_folder", "sources", "test_homes"]
