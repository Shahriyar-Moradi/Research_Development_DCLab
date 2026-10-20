"""Corrections (32.19): every time a person changes what DCLab proposed, kept whole as before and after, with the reason
when one is given: the strongest signal there is of where DCLab was wrong.

``record(what, before, after, ...)`` appends one line to ``<workspace>/ledger/corrections/YYYY/MM/DD.jsonl`` (the store's
decisions keeper keeps it; its shaper types it a correction, sealed by its project or draft). Called where the change is
made: a person saving a solution over the one proposed or the agent's, a stage approved with another choice than the rule's,
a problem map edited. Never a reason for the change itself to fail.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


def _home() -> Optional[Path]:
    from . import experience
    from .ledger import current_ledger

    for owner in (current_ledger(), experience.current_log()):
        home = getattr(owner, "home", None)
        if home:
            return Path(home)
    return None


def record(what: str, before: Any, after: Any, *, reason: Optional[str] = None, by: str = "person", project: Optional[str] = None,
           draft: Optional[str] = None, home: Optional[Path] = None) -> Optional[dict[str, Any]]:
    """One correction kept; None when nothing changed or no workspace is known."""
    if before == after:
        return None
    try:
        home = Path(home) if home else _home()
        if home is None:
            return None
        now = datetime.now(timezone.utc)
        doc = {"at": now.isoformat(timespec="microseconds"), "what": what, "before": before, "after": after, "reason": reason, "by": by,
               "source": "person" if by in ("person", "human", "owner") else by, "project": project, "draft": draft}
        folder = home / "ledger" / "corrections" / f"{now:%Y}" / f"{now:%m}"
        folder.mkdir(parents=True, exist_ok=True)
        with open(folder / f"{now:%d}.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(doc, ensure_ascii=False, default=str) + "\n")
        from . import agent_metrics  # the agents' ledger: what a person did with what was proposed

        answer = after.get("answer") if isinstance(after, dict) else None
        if what == "notebook_cell" and answer in ("accepted", "rejected"):
            agent_metrics.emit("notebook_cell", "proposal_accepted", 1.0 if answer == "accepted" else 0.0, ref=project, home=home)
        else:
            agent_metrics.emit(what, "correction", 1.0, ref=project or draft, home=home)
        return doc
    except Exception:  # noqa: BLE001 - a record of the change, never a reason for it to fail
        return None


__all__ = ["record"]
