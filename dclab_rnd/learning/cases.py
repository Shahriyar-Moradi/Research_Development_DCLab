"""Case memory (H2; Phase 9's 31.6): what earlier graded attempts of a family went through, retrieved by family and code.

A case keeps codes, lesson ids and verdicts, never a column or a value: the variants' skins reuse names (a household
key appears in many), so a remembered name would let a solver pass by recall. ``<workspace>/learning/cases.jsonl``,
one case a line, only ever appended. :func:`similar` returns the newest cases of the family that share a code with
the attempt at hand, as short lines a solver can read beside the playbook.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dclab_core.challenge_pack import CODES

_LOCK = threading.Lock()


class FileCases:
    def __init__(self, home: Path):
        self.path = Path(home) / "learning" / "cases.jsonl"

    def add(self, family: str, pack: str, codes: list[str], verdict: str, shown: list[str] | None = None,
            lessons: list[str] | None = None) -> dict[str, Any]:
        """One case: the family, the opaque pack, the codes it failed on (known codes only), the verdict, the lesson ids
        it was shown and the ones drawn from it."""
        case = {"family": family, "pack": pack, "codes": [c for c in codes if c in CODES], "verdict": verdict,
                "shown": list(shown or []), "lessons": list(lessons or []), "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with _LOCK:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(case, ensure_ascii=False) + "\n")
        return case

    def all(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return out


def similar(cases: list[dict[str, Any]], family: str, codes: list[str] | None = None, k: int = 3) -> list[dict[str, Any]]:
    """The newest ``k`` cases of the family; with ``codes``, only those sharing one (structured fields first, Phase 9)."""
    wanted = set(codes or [])
    chosen = [c for c in reversed(cases) if c.get("family") == family and (not wanted or wanted & set(c.get("codes") or []))]
    return chosen[:k]


def lines(cases: list[dict[str, Any]]) -> list[str]:
    """Cases as lines a solver reads: what a past problem of this kind failed on, the lessons it led to, how it ended."""
    out = []
    for c in cases:
        codes = ", ".join(c.get("codes") or []) or "nothing"
        lessons = f"; lessons drawn: {', '.join(c['lessons'])}" if c.get("lessons") else ""
        out.append(f"A past problem of this kind ({c.get('verdict')}) failed on: {codes}{lessons}.")
    return out


__all__ = ["FileCases", "similar", "lines"]
