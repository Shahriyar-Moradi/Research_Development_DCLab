"""Other agents' sessions (32.7): every Claude Code session on this repository, the main checkout's and every worktree's,
kept under ``raw/agents/`` (workspace ``rnd``, source ``claude_code/<checkout>/<file>``): the transcripts (their reasoning,
tool calls and results, task lists included), their subagents' transcripts, and the sessions' memory.

Only on the owner's machine: where the studio runs as a tester (a ``tester.json`` in its workspace), this machine's Claude
Code sessions are counted, never kept, unless ``DCLAB_STORE_CLAUDE_PROJECTS`` names their folder (the owner's explicit
choice); a tester's own sessions reach the store only through their share, with their yes (32.16).

A transcript grows as its session goes on, so it is kept by its new lines. Before keeping, anything shaped like a credential
is replaced by a marker, line by line (:data:`SECRET`): API keys and tokens (Anthropic, OpenAI, GitHub, Google, Slack,
Hugging Face, AWS), private keys, JSON web tokens, a password in a connection string, a key or a secret named in an
assignment. Images (screenshots of a desktop) are counted only: no redaction can see text in a picture.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Iterator, Optional

from ..layout import ROOT
from . import Claim, Keeper, Source

PROJECTS = Path.home() / ".claude" / "projects"
IMAGES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".tiff", ".heic")
SECRET = re.compile(
    rb"(sk-ant-[A-Za-z0-9_\-]{20,}|sk-[A-Za-z0-9_\-]{20,}|github_pat_[A-Za-z0-9_]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}"
    rb"|AIza[0-9A-Za-z_\-]{30,}|xox[abpors]-[A-Za-z0-9\-]{10,}|AKIA[0-9A-Z]{16}|ASIA[0-9A-Z]{16}"
    rb"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}"
    rb"|-----BEGIN [A-Z ]*PRIVATE KEY-----.*?(-----END [A-Z ]*PRIVATE KEY-----|$)"
    rb"|(?i:bearer)\s+[A-Za-z0-9._\-]{20,}"
    rb"|[a-z][a-z0-9+\-.]*://[^:\s/@\"']+:[^@\s\"']+@"
    rb"|(?i:(api[_-]?key|secret[_-]?key|secret|token|password|passwd|aws_secret_access_key|access[_-]?key)[\"'\\\s]*[:=][\\\s\"']*[^\s\"',\\]{8,})"
    rb"|(?i:\"key\"\s*:\s*\"[0-9a-f]{32}\"))", re.S)


def encoded(path: Path) -> str:
    """Claude Code's folder name for a checkout: every character that is not a letter or a digit as a dash."""
    return re.sub(r"[^A-Za-z0-9]", "-", str(path))


def redact(raw: bytes) -> bytes:
    return SECRET.sub(b"[SECRET REMOVED]", raw)


def projects_folder() -> Optional[Path]:
    """Where to read this machine's Claude Code sessions: ``DCLAB_STORE_CLAUDE_PROJECTS`` (``off``: none, as in the tests), else the
    default, but never on a tester's machine unless set explicitly."""
    value = os.environ.get("DCLAB_STORE_CLAUDE_PROJECTS", "")
    if value == "off":
        return None
    if value:
        return Path(value)
    from .workspaces import AGENT_RUNS

    return None if (AGENT_RUNS / "tester.json").exists() else PROJECTS


def folders(projects: Path = PROJECTS, root: Path = ROOT) -> Iterator[tuple[str, Path]]:
    """(the checkout's name, its sessions' folder): the main checkout (``main``), each worktree (its name), each subfolder session."""
    head = encoded(root)
    if not projects.exists():
        return
    for folder in sorted(projects.iterdir()):
        if not folder.is_dir():
            continue
        if folder.name == head:
            yield "main", folder
        elif folder.name.startswith(head + "--claude-worktrees-"):
            yield folder.name[len(head + "--claude-worktrees-"):], folder
        elif folder.name.startswith(head + "-"):
            yield "sub-" + folder.name[len(head) + 1:], folder


def _counted(paths: list[Path], why: str) -> bytes:
    return json.dumps([{"source": p.name, "bytes": p.stat().st_size, "sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "why": why}
                       for p in paths], indent=0).encode()


def sources(env: Any) -> Iterator[Source]:
    if isinstance(env, dict) and env.get("only"):
        return  # a workspace's removal is not the development sessions'
    from .standard import _for_kind

    for src in _for_kind("agents")(env):  # testers' own sessions, brought by their share into their folder
        if src.source.lower().endswith(IMAGES):
            continue
        src.transform, src.note = redact, (src.note or "a tester's own Claude Code session") + "; anything shaped like a credential replaced"
        yield src
    explicit = isinstance(env, dict) and "claude_projects" in env
    projects = (Path(env["claude_projects"]) if env["claude_projects"] else None) if explicit else projects_folder()
    root = Path(env["repo"]) if isinstance(env, dict) and env.get("repo") else ROOT
    for checkout, folder in folders(projects, root) if projects is not None else ():
        images = []
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.is_symlink() or path.name.endswith((".lock", ".part", ".tmp")):
                continue
            rel = path.relative_to(folder).as_posix()
            if path.suffix.lower() in IMAGES:
                images.append(path)
                continue
            yield Source("rnd", f"claude_code/{checkout}/{rel}", path, consent="train", log=path.suffix == ".jsonl", transform=redact,
                         note="Claude Code's session on this repository; anything shaped like a credential replaced by a marker")
        if images:
            yield Source("rnd", f"claude_code/{checkout}/~images~counts.json", lambda imgs=images: _counted(imgs, "an image: counted only"),
                         consent="train", note="counts only: images (screenshots) are never kept, no redaction can read them")


KEEPER = Keeper("agents", (Claim("agents", "claude_code/"),), sources, ("m4_workflow_llm", "m5_solver_llm", "m7_judges", "m2_workflow_decisions"),
                "Claude Code's sessions on this repository (the main checkout and every worktree), their reasoning and tool calls whole")

__all__ = ["KEEPER", "folders", "encoded", "redact", "sources", "SECRET", "projects_folder"]
