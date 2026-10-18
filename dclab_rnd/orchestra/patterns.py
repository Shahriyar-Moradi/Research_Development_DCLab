"""The team's patterns as experiments (package 28.6): each measured on the journeys against the single agent, its result in
``evidence/campaigns/orchestra_v1/``, and its switch (``orchestra/settings.py``) on only where that result passed."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from . import ROOT

PATTERNS = ("second_opinion", "debate")
RESULTS = ROOT / "evidence" / "campaigns" / "orchestra_v1" / "results"


def result(name: str, results: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """The newest stored result of a pattern (completed runs only), or None."""
    found = []
    for path in sorted(Path(results or RESULTS).glob("ORC-*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if doc.get("pattern") == name and doc.get("status") == "completed":
            found.append(doc)
    return found[-1] if found else None


def passed(name: str, results: Optional[Path] = None) -> bool:
    """True only when the pattern's newest result passed its gate."""
    doc = result(name, results)
    return bool(doc and doc.get("verdict") == "pass")


def store(name: str, material: list[str], by_arm: dict[str, list[dict[str, Any]]], against: dict[str, Any], extra: Optional[dict[str, Any]] = None,
          results: Optional[Path] = None) -> Path:
    """A pattern's run as one result file (``ORC-NNN_<pattern>.json``): both arms' rows, the verdict against its gate."""
    from datetime import datetime, timezone

    out = Path(results or RESULTS)
    out.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in out.glob("ORC-*.json")] or [0])
    ident = f"ORC-{n:03d}"
    doc = {"id": ident, "campaign_id": "orchestra_v1", "kind": "team_pattern", "pattern": name, "material": material,
           "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "gate": against["gate"], "arms": by_arm,
           "summary": {k: against[k] for k in ("parts", "questions", "asked_twice", "broke", "refused", "failures", "spent_eur", "priced", "cost_ok",
                                               "gained", "gained_by_the_pattern", "lost")},
           "verdict": "pass" if against["passed"] else "fail", "status": "completed", "experiment_id": ident, **(extra or {}),
           "rule": "Dev evidence on written people over seeded replicas (the fifth sealed set was spent in 28.7 and read while the pattern "
                   "was designed): it decides the pattern's switch, never a sealed verdict, and says nothing about production readiness."}
    path = out / f"{ident}_{name}.json"
    path.write_text(json.dumps(doc, indent=2, default=str) + "\n", encoding="utf-8")
    return path


__all__ = ["PATTERNS", "RESULTS", "result", "passed", "store"]
