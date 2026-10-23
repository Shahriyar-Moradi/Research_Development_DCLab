"""The read-only layer (H1 of the harness-learning plan): what a learned change may write, and nothing else.

A learned change (a skill's files in H4, an evolved prompt in H5, a candidate patch from any later search) is a set of
paths it would write. It may write only inside :data:`ALLOWED` (an allow list: a list of what to forbid always misses
something, as the review of this package showed with the guards that call the validator). :data:`PROTECTED` is a
second check inside the allowed places and the list a reader can trust: the validator and every guard that calls it,
the engine, the gateway and its caps, the registry and the loop, the grader and the generator, the gates a promotion
is judged by, every prompt, the tests, the Makefile, the repository's own settings and this layer. Paths are compared
without case (the Mac's disk opens ``GRAPH.py`` as ``graph.py``); a path outside the repository is refused; a rename
counts as both of its paths. A prompt that says "do not weaken the rules" is not a guard; this is.
"""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]

ALLOWED: tuple[str, ...] = ("skills/",)  # H4's skills; H5 adds its candidate prompts when it is built

PROTECTED: tuple[str, ...] = (
    # the validator and every guard that calls it
    "dclab_rnd/studio/graph.py", "dclab_rnd/intern/tools.py", "dclab_rnd/intern/loop.py", "dclab_rnd/draft/chat.py",
    "dclab_rnd/draft/orchestrator.py", "dclab_rnd/agentic/campaign_tools.py", "dclab_rnd/agents/",
    # splits, metrics, the one holdout read; what a solution may say
    "dclab_rnd/studio/engine.py", "dclab_rnd/studio/solution.py",
    # every model request: its caps, prices, routes, shadows, tuned models, purposes
    "dclab_rnd/models/",
    # the grader, the generator, each kind's acceptance, the planted traps
    "dclab_core/challenge_pack.py", "dclab_core/variants.py", "dclab_rnd/kinds/", "dclab_core/replicas/",
    # the gates a promotion is judged by
    "dclab_rnd/agent_eval/", "dclab_core/decision_gate.py", "dclab_rnd/ledger/", "dclab_rnd/critic_gate.py", "dclab_rnd/cited.py",
    # what never leaves a workspace
    "dclab_core/redact.py", "dclab_rnd/consent.py",
    # every prompt a model is sent (hash-gated)
    "dclab_rnd/prompts/", "dclab_core/prompts/",
    # the learning harness itself and the measurements it is judged by
    "dclab_rnd/learning/", "evidence/campaigns/learning_v1/",
    # the tests, the gates' commands and the repository's own settings and hooks
    "tests/", "Makefile", "CLAUDE.md", ".claude/", ".git/", ".github/",
)


class Protected(PermissionError):
    """A learned change outside the allowed places, or inside the read-only layer."""


def _relative(path: str | Path) -> str | None:
    """The path relative to the repository, folded to lower case, or None when it lies outside it (refused)."""
    p = Path(path)
    absolute = (ROOT / p if not p.is_absolute() else p).resolve()
    try:
        return PurePosixPath(absolute.relative_to(ROOT.resolve())).as_posix().casefold()
    except ValueError:
        return None


def _within(rel: str, entries: tuple[str, ...]) -> bool:
    for entry in entries:
        e = entry.casefold()
        if e.endswith("/") and (rel == e.rstrip("/") or rel.startswith(e)):
            return True
        if not e.endswith("/") and rel == e:
            return True
    return False


def refused(paths: Iterable[str | Path]) -> list[str]:
    """Every path a learned change may not write: outside the repository, outside the allowed places, or protected."""
    return [str(p) for p in paths if (rel := _relative(p)) is None or not _within(rel, ALLOWED) or _within(rel, PROTECTED)]


def check_diff(paths: Iterable[str | Path]) -> None:
    """Raise :class:`Protected` when a learned change would write any path it may not."""
    bad = refused(paths)
    if bad:
        raise Protected(f"a learned change may write only {', '.join(ALLOWED)}, never the read-only layer: {', '.join(sorted(bad))}")


def check_git(base: str, head: str = "HEAD") -> list[str]:
    """The paths a commit range changed that a learned change may not: renames listed as both of their paths
    (``--no-renames``), and any symbolic link refused (a link can point into the layer)."""
    out = subprocess.run(["git", "diff", "--no-renames", "--raw", f"{base}..{head}"], capture_output=True, text=True, cwd=ROOT, timeout=30)
    if out.returncode != 0:
        raise RuntimeError(f"git diff failed: {out.stderr.strip()[:200]}")
    bad = []
    for line in out.stdout.splitlines():
        if not line.startswith(":") or "\t" not in line:
            continue
        modes, path = line.split("\t", 1)
        parts = modes.split()
        if "120000" in parts[:2]:  # a symbolic link, old or new
            bad.append(path)
        elif refused([path]):
            bad.append(path)
    return bad


__all__ = ["ALLOWED", "PROTECTED", "Protected", "refused", "check_diff", "check_git"]
