"""A project's decision on top of its model (34.4): what the final stage chose, and a what-if on it.

The final stage chooses the decision a project states (``Solution.decision``: a review budget, a price, an ``act``
decision through dclab_core's decision core, or a cost matrix over the classes) and writes it to ``evidence.decision``.
An ``act`` decision also keeps the out-of-fold rows it chose on beside the project (``engine.save_decision_rows``), so a
what-if re-scores it under other numbers without refitting: the API (``/api/projects/{id}/decision``), the intern's
``decision_what_if`` tool and the web page read this module, one answer everywhere.
"""

from __future__ import annotations

from typing import Any

from dclab_core import decide

from . import engine

WHAT_IF_TERMS = (*decide.WHAT_IF_TERMS, "base_rate", "value_scale")


def _terms(rows: dict[str, Any] | None) -> list[str]:
    """The numbers a what-if may change: with a value per row, every value scales together (``value_scale``) and a single
    value per case is not read."""
    if rows is None:
        return []
    per_row = rows.get("value") is not None
    return [t for t in WHAT_IF_TERMS if not (per_row and t == "value_per_case") and not (not per_row and t == "value_scale")]


def _final(store: Any, project_id: str) -> dict[str, Any] | None:
    return store.read_stage(project_id, "final")


def view(store: Any, project_id: str) -> dict[str, Any]:
    """The decision the project states, what the final stage chose and whether a what-if can be asked. Numbers only from
    the stage record; never a row."""
    project = store.get(project_id)
    stated = (project.get("solution") or {}).get("decision")
    final = _final(store, project_id)
    chosen = ((final or {}).get("evidence") or {}).get("decision")
    run = ((final or {}).get("run") or {}).get("run")
    rows = engine.load_decision_rows(store, project_id, run) if (run and chosen and chosen.get("decision") == "operating_point") else None
    claim = next((c for c in (final or {}).get("claims", []) if str(c.get("claim_id", "")).endswith("-D1")), None)
    status = ("none" if stated is None else "pending" if final is None else "refused" if chosen and chosen.get("learnable") is False
              else "chosen" if chosen else "missing")
    heard = None if stated else (project.get("suggestion") or {}).get("decision")  # amounts Home heard but did not add (34.4)
    return {"status": status, "stated": stated, "heard": heard, "chosen": chosen, "sentence": (claim or {}).get("statement"),
            "what_if": {"available": rows is not None, "terms": _terms(rows),
                        "why_not": None if rows is not None else _why_not(stated, final, chosen)},
            "run": run}


def _why_not(stated: dict[str, Any] | None, final: dict[str, Any] | None, chosen: dict[str, Any] | None) -> str:
    if stated is None:
        return "The project states no decision: say what is done to a row, what a case is worth and what acting costs."
    if final is None:
        return "Run the final stage first: the decision is chosen on the selected model's out-of-fold scores."
    if (stated or {}).get("action") != "act":
        return "A what-if is asked of an act decision (a value and costs per row); a review budget, a price or a cost matrix is not re-scored here."
    if chosen and chosen.get("learnable") is False:
        return str(chosen.get("warning") or "The decision could not be made from this data.")
    return "The rows the decision was chosen on are not kept for this run: run the final stage again."


def what_if(store: Any, project_id: str, changes: dict[str, Any]) -> dict[str, Any]:
    """The chosen act decision re-scored under ``changes`` (the spec's values, costs or capacity share, or ``base_rate``),
    nothing refitted; plus the sensitivity table of the stated numbers. Raises ``LookupError`` when it cannot be asked,
    ``ValueError`` for a change the decision does not state."""
    if not isinstance(changes, dict) or not changes:
        raise ValueError("Say what changes: one or more of " + ", ".join(WHAT_IF_TERMS))
    bad = [k for k, v in changes.items() if k not in WHAT_IF_TERMS or isinstance(v, bool) or not isinstance(v, (int, float))]
    if bad:
        raise ValueError("A what-if changes numbers the decision states: " + ", ".join(WHAT_IF_TERMS) + "; not " + ", ".join(map(str, bad)))
    seen = view(store, project_id)
    if not seen["what_if"]["available"]:
        raise LookupError(seen["what_if"]["why_not"])
    chosen = seen["chosen"]
    rows = engine.load_decision_rows(store, project_id, seen["run"])
    unread = [k for k in changes if k not in _terms(rows)]
    if unread:
        raise ValueError("This decision's what-if changes " + ", ".join(_terms(rows)) + "; not " + ", ".join(unread))
    spec = decide.DecisionSpec(**chosen["spec"])
    out = decide.what_if(rows["y"], rows["score"], spec, {k: float(v) for k, v in changes.items()}, value_per_case=rows["value"],
                         rows_per_day=rows["rows_per_day"])
    out["what"] = chosen.get("what") or spec.action
    out["run"] = seen["run"]
    return out
