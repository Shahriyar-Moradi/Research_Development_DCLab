"""What every measurement campaign's run keeps, and what pins its code (the owner's rule: capture everything, for the
six models; the review of 34.7 and 34.1: a campaign's code is pinned whole).

``recorded(home)``: while a campaign runs the engine in a workspace folder under ``agent_runs/``, that folder gets its
own decision ledger and session log (the store collects both, beside the stages and the activity), and the decisions
are made by the rules alone: a model routed to a decision point in the owner's workspace never answers inside a
measurement.

``code_sha256(paths)``: one hash of every Python file under the folders (and the files) that make a campaign's numbers,
so a resume on changed code is refused.
"""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence

ROOT = Path(__file__).resolve().parents[1]
ENGINE_CODE = ("dclab_rnd/studio", "dclab_rnd/kinds", "dclab_rnd/expansion", "dclab_core", "general_pipeline")


@contextmanager
def recorded(home: Path) -> Iterator[None]:
    from . import experience, ledger
    from .ledger import use

    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    log, book = experience.open_log(home), ledger.open_ledger(home)
    before_log, before_book = experience._INSTALLED.get("log"), ledger._INSTALLED.get("ledger")
    experience.install(log)
    ledger.install(book)
    try:
        with use.scope({"decisions": {}}):  # the rules decide: no routed model inside a measurement
            yield
    finally:
        experience.install(before_log)
        ledger.install(before_book)


def code_sha256(paths: Sequence[str], root: Path = ROOT) -> str:
    """One hash over every ``.py`` file under ``paths`` (folders or files, relative to ``root``), in a fixed order."""
    files: list[Path] = []
    for name in paths:
        path = root / name
        files.extend(sorted(path.rglob("*.py")) if path.is_dir() else [path])
    digest = hashlib.sha256()
    for path in sorted(set(files)):
        digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()
