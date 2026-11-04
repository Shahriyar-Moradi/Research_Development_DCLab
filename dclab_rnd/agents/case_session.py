"""The lead agent's long sessions (package 37.8, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): a checkpoint after every turn,
resume after a stop or a crash with nothing lost, and a soft budget per case that stops and asks.

* :func:`save` writes a case's state after each turn (the case file, the turns, what code computed, the table's view and
  its name, the choices and flags), atomically, so a kill mid-write leaves the last good checkpoint; :func:`load` brings
  it back, the table reread by its name, never stored in the checkpoint.
* :func:`over_budget` and :func:`ask_to_continue`: a case that has spent its soft cap (``DCLAB_CASE_CAP_EUR``, default 5,
  the owner's) stops before its next model call and asks the person; their yes in their own words adds another cap.
* Compaction is the context builder's (``agents.context``): old turns leave the context, the case file never does.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

from dclab_core import case as C

CAP_EUR = 5.0  # the owner's soft cap per case (2026-10-30)
YES = re.compile(r"^\s*(?:yes|yeah|yep|ok|okay|go on|continue|carry on|sure|please do|go ahead)\b", re.I)
ASK = ("This case has used its budget of €{cap:g} of model work. Shall I go on? Say yes to continue, or tell me what to "
       "focus on.")


def cap_eur() -> float:
    try:
        return float(os.environ.get("DCLAB_CASE_CAP_EUR") or CAP_EUR)
    except ValueError:
        return CAP_EUR


def snapshot(state: Any) -> dict[str, Any]:
    """What a checkpoint holds: everything a turn needs but the table itself and the model clients."""
    return {"version": 1, "case": json.loads(state.case.model_dump_json()), "table": state.table, "computed": list(state.computed),
            "turns": list(state.turns), "asking": state.asking, "build": state.build, "spent_eur": float(getattr(state, "spent_eur", 0.0)),
            "cap_eur": float(getattr(state, "cap_eur", cap_eur())), "waiting_on_budget": bool(getattr(state, "waiting_on_budget", False)),
            "consulted": list(getattr(state, "consulted", []) or []), "run_id": getattr(getattr(state, "trace", None), "run_id", None),
            "group_names": sorted(getattr(state, "group_names", set()) or set()), "exposure": list(getattr(state, "exposure", []) or []),
            "route_told": list(getattr(state, "route_told", []) or []), "chip": getattr(state, "chip", None), "chip_intent": getattr(state, "chip_intent", None),
            "chip_checked": bool(getattr(state, "chip_checked", False)), "trace_n": getattr(getattr(state, "trace", None), "n", 0), "trace_reply": getattr(getattr(state, "trace", None), "last_reply", -1)}


def save(state: Any, path: Path) -> Path:
    """The checkpoint, written whole or not at all (a temporary file renamed over the old one)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".case-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(snapshot(state), f, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def load(path: Path, frame_of: Optional[Callable[[str], Any]] = None) -> Any:
    """A case's state from its checkpoint; ``frame_of(table_name)`` rereads the shared table (never kept in the checkpoint)."""
    from .case_agent import CaseState

    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    table = doc.get("table")
    state = CaseState(case=C.CaseFile.read(doc["case"]), table=table, computed=list(doc.get("computed") or []),
                      turns=list(doc.get("turns") or []), asking=bool(doc.get("asking")), build=doc.get("build"))
    state.frame = frame_of(table["table"]) if (frame_of and table) else None
    state.spent_eur = float(doc.get("spent_eur") or 0.0)
    state.cap_eur = float(doc.get("cap_eur") or cap_eur())
    state.waiting_on_budget = bool(doc.get("waiting_on_budget"))
    state.consulted = list(doc.get("consulted") or [])
    restore(state, doc)
    state.checkpoint = Path(path)  # turns after a resume are saved too (the review: a second crash lost them)
    return state


def restore(state: Any, doc: dict[str, Any]) -> None:
    """What a resumed case keeps beyond its case file: the names it was shown (the cap of 40 holds across turns), what its
    answers looked at, and its tracer's place, so the same run's steps keep counting on (the review of 37.8)."""
    from .harness import tracer

    state.group_names = set(doc.get("group_names") or [])
    state.exposure = list(doc.get("exposure") or [])
    state.route_told = list(doc.get("route_told") or [])
    state.chip, state.chip_checked, state.chip_intent = doc.get("chip"), bool(doc.get("chip_checked")), doc.get("chip_intent")  # 2026-11-04: the person's chip on Home  # the plan's step 1: a route doubt is told once a case, across turns
    if doc.get("run_id"):
        trace = tracer("case", run_id=doc["run_id"])
        trace.n = int(doc.get("trace_n") or 0)
        trace.last_reply = int(doc.get("trace_reply") if doc.get("trace_reply") is not None else -1)
        trace.reply_base = trace.last_reply + 1
        state.trace = trace


def over_budget(state: Any) -> bool:
    return float(getattr(state, "spent_eur", 0.0)) >= float(getattr(state, "cap_eur", cap_eur()))


def ask_to_continue(state: Any, words: str) -> Optional[str]:
    """Before a turn: the message that stops a case at its cap (None when the turn may run). A yes after the stop adds a cap."""
    if getattr(state, "waiting_on_budget", False) and YES.search(str(words or "")):
        state.waiting_on_budget = False
        state.cap_eur = max(float(getattr(state, "cap_eur", cap_eur())), float(getattr(state, "spent_eur", 0.0))) + cap_eur()
        return None
    if over_budget(state):
        state.waiting_on_budget = True
        return ASK.format(cap=float(getattr(state, "cap_eur", cap_eur())))
    return None


__all__ = ["ASK", "CAP_EUR", "ask_to_continue", "cap_eur", "load", "over_budget", "save", "snapshot"]
