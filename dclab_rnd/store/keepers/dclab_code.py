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
from pathlib import Path
from typing import Any, Iterator, Optional

from ..layout import ROOT
from . import Claim, Keeper, Source
from .agents import _counted, redact

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


def sources(env: Any) -> Iterator[Source]:
    if isinstance(env, dict) and env.get("only"):
        return  # a workspace's removal is not the development sessions'
    home = home_folder(env)
    if home is None:
        return
    root = Path(env["repo"]) if isinstance(env, dict) and env.get("repo") else ROOT
    index = home / "sessions.json"
    try:
        sessions = json.loads(index.read_text(encoding="utf-8")).get("sessions", []) if index.is_file() and not index.is_symlink() else []
    except (OSError, ValueError):
        sessions = []
    ours = {s.get("id") for s in sessions if isinstance(s, dict) and _inside(s.get("projectPath"), root)}
    records = home / "records"
    others = []
    for path in sorted(records.glob("*.jsonl")) if records.is_dir() else ():
        if path.is_symlink() or not path.is_file():
            continue
        if path.stem not in ours:
            others.append(path)  # another folder's session, or one the index does not know: counted only
            continue
        yield Source("rnd", f"dclab_code/records/{path.name}", path, consent="train", log=True, transform=redact,
                     note="a DCLab Code session on this repository: SDK messages whole and the person's decisions; anything shaped like a credential replaced")
    if others:
        yield Source("rnd", "dclab_code/~other-projects~counts.json", lambda paths=others: _counted(paths, "a session on another folder: counted only"),
                     consent="train", note="counts only: DCLab Code sessions on other folders are never kept")
    kept = [s for s in sessions if isinstance(s, dict) and s.get("id") in ours]
    if kept:
        yield Source("rnd", "dclab_code/sessions.json", lambda rows=kept: json.dumps({"sessions": rows}, indent=1).encode(), consent="train",
                     transform=redact, note="DCLab Code's sessions on this repository: title, project, model, cost, own copy, SDK session id")


KEEPER = Keeper("dclab_code", (Claim("agents", "dclab_code/"),), sources, ("m6_stepwise_decisions", "m7_judges", "m2_workflow_decisions", "m1_students"),
                "DCLab Code's sessions on this repository: every SDK message whole, with the person's permissions, answers and plan approvals")

__all__ = ["KEEPER", "HOME", "home_folder", "sources"]
