"""Search and reading (32.23): every search an agent or a role makes (the evidence index, the business examples), what it was
offered, and later what its answer cited and whether the answer passed (``cited.check``), kept in
``<workspace>/searches/YYYY/MM/DD.jsonl``: M8's data, and where the knowledge base is thin (a search that offered nothing
used). Each line carries the run, the conversation and the project it was made in. Never a reason for a search to fail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Iterable, Optional


def _write(doc: dict[str, Any]) -> None:
    try:
        from . import experience
        from .corrections import _home

        home = _home()
        if home is None:
            return
        ids = experience.bound()
        now = datetime.now(timezone.utc)
        doc = {"at": now.isoformat(timespec="microseconds"), "run": ids.get("run"), "conversation": ids.get("conversation"),
               "project": ids.get("project"), **doc}
        folder = home / "searches" / f"{now:%Y}" / f"{now:%m}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{now:%d}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
    except Exception:  # noqa: BLE001
        pass


def offered(corpus: str, query: str, ids: Iterable[str], method: Optional[str] = None) -> None:
    _write({"what": "search", "corpus": corpus, "query": query, "offered": [str(i) for i in ids], "method": method})


def cited(ids: Iterable[str], passed: bool, dropped: int = 0) -> None:
    """What an answer cited of what it was offered, and whether its sentences passed their check."""
    ids = [str(i) for i in ids]
    if ids or dropped:
        _write({"what": "answer", "cited": sorted(set(ids)), "passed": passed, "dropped": dropped})


__all__ = ["offered", "cited"]
