"""The development diffs (32.21): every commit of this repository kept as its patch (``git/<sha>.patch``, under the kind
``fixes``), its message and the files it changed, generated files left out (they are rebuilt from the others), so how DCLab
is built and fixed is kept beside the failures it fixed. A commit already kept is not read again.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Iterator

from ..layout import ROOT
from . import Claim, Keeper, Source

GENERATED = (":(exclude)research/llm-fine-tuning/experiments/sft", ":(exclude)evidence/knowledge", ":(exclude)dclab_rnd/agentic/static",
             ":(exclude)research/*/INDEX.md", ":(exclude)docs/product-demo/index.html", ":(exclude)docs/recap", ":(exclude)*.pdf")


def sources(env: Any) -> Iterator[Source]:
    if isinstance(env, dict) and env.get("only"):
        return
    if isinstance(env, dict) and "repo" in env and not env["repo"]:
        return  # told: no repository here
    if os.environ.get("DCLAB_STORE_COMMITS") == "off" and not (isinstance(env, dict) and env.get("repo")):
        return  # the tests: never this repository's whole history, unless a test gives its own
    repo = Path(env["repo"]) if isinstance(env, dict) and env.get("repo") else ROOT
    from ..layout import Layout
    from ..store import Store

    store = Store(Layout())
    done = {e.source[len("git/"):-len(".patch")] for e in store.entries("fixes") if e.source.startswith("git/")}  # a commit kept once
    try:
        shas = subprocess.run(["git", "-C", str(repo), "rev-list", "--reverse", "HEAD"], capture_output=True, text=True, timeout=60, check=True).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return
    for sha in shas:
        if sha in done:
            continue

        def patch(sha: str = sha) -> bytes:
            return subprocess.run(["git", "-C", str(repo), "show", "--format=commit %H%nAuthor date: %aI%n%n%B", "--stat", "--patch", sha, "--", ".", *GENERATED],
                                  capture_output=True, timeout=120, check=True).stdout
        yield Source("rnd", f"git/{sha}.patch", patch, consent="train", note="a commit of this repository: its message, files and patch")


KEEPER = Keeper("commits", (Claim("fixes", "git/"),), sources, ("m4_workflow_llm", "m5_solver_llm", "m7_judges"),
                "every commit of this repository as its patch, generated files left out")

__all__ = ["KEEPER", "sources"]
