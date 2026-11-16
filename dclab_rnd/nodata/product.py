"""The rebuilt no-data design as the product uses it (stage N2 of the no-data plan): Home's default since 2026-11-16.

Unless ``DCLAB_NO_DATA_V2=0``, Home's three entries (the lead agent's tool, the rules-first Home, the API route) answer a
person with no table through ``dclab_rnd.nodata.solve``. ``for_product`` gives the solution the fields those entries and
the Home card already read (a family, a summary line, the Markdown, the questions before made-up data), so nothing else
changes shape. With no model, a prediction is answered by the first design's rules, as before; a need that is not a
prediction gets a plain note and never a model blueprint. Build project is not touched by anything here."""

from __future__ import annotations

import os
from typing import Any, Mapping, Optional

from . import checks, render, simulate, solve

SWITCH = "DCLAB_NO_DATA_V2"
NO_MODEL_NOTE = ("I can design this with you without your data, but no model is available to me right now, and DCLab's rules can only "
                 "draw a plan for something to predict. Tell me the decision you face and what you already know, and I will lay out "
                 "the options and a plan as soon as a model is available.")


def enabled() -> bool:
    """On unless ``DCLAB_NO_DATA_V2`` is set to 0 (the owner's decision of 2026-11-16).

    The decision was made with the gate not passed: on the eight held-out people (HAB-051) the design matched the same
    model alone on understanding (3.88) and the solution (4.00), was 0.38 behind on the plan (half a point allowed),
    stated no number nobody gave and did nothing forbidden, and failed ``nodata_gate_v1`` on one check: 374 seconds a
    person where the gate allows 300. It replaces a design that took about 21 minutes a person and scored 1.88 on the
    plan on the practice people. 0 puts the first design back."""
    return os.environ.get(SWITCH, "").strip().lower() not in ("0", "false", "no", "off")


def is_note(solution: Mapping[str, Any]) -> bool:
    """No solution was written: only the plain note that no model is available (never shown as a solution card)."""
    return solution.get("mode") == "rules" and solution.get("version") == "nodata_v2" and not (solution.get("card") or {}).get("parts")


def summary(solution: Mapping[str, Any]) -> str:
    """One short message for the chat, above the card: what was written, and that nothing needed their data."""
    names = [p["title"] for p in render.card(solution)["parts"]]
    return ("Here is a solution designed without your data: " + ", ".join(names) + ". Every number in it is yours or a sum of yours that "
            "code checked. No data was made: if you want a demonstration on made-up data, I will first ask how it should look.")


def for_product(solution: Mapping[str, Any]) -> dict[str, Any]:
    """The solution with the fields Home's entries and card read."""
    out = dict(solution)
    out.update({
        "family": solution.get("kind") or solution.get("need") or "no_data", "family_source": solution.get("mode"),
        "mode_label": "No-Data Solution Design", "fit": {"fits": True, "why": ""},
        "summary": summary(solution), "markdown": render.markdown(solution), "card": render.card(solution),
        "simulation": {"questions": simulate.questions(solution), "seed_description": simulate.description(solution, [])},
        "as_said": list((solution.get("economics") or {}).get("as_said") or []),
    })
    return out


def solve_for_product(words: str, *, gateway: Any = None, draft_id: Optional[str] = None, project_id: Optional[str] = None,
                      cap_eur: float = 0.12) -> dict[str, Any]:
    """The answer for a person with no table. A model writes it when one serves; otherwise the first design's rules
    answer a prediction (its own card, unchanged), and any other need gets a plain note."""
    solution = solve(words, gateway=gateway, draft_id=draft_id, project_id=project_id, cap_eur=cap_eur)
    if solution.get("mode") == "architect":
        return for_product(solution)
    from dclab_rnd.draft import no_data

    if no_data.blueprint_fit(str(words or ""))["fits"]:
        old = no_data.solve_without_data(str(words or ""), gateway=None, draft_id=draft_id, cap_eur=0.0)
        # a request that was paid for and came back empty is still spent: the caller's budget counts it
        return {**old, "rules_note": solution.get("why"), "cost_eur": float(solution.get("cost_eur") or 0.0), "written_by": "rules"}
    note = {"version": solution.get("version"), "mode": "rules", "why": solution.get("why"), "words": str(words or ""), "need": None,
            "text": NO_MODEL_NOTE, "parts": {}, "columns": [], "plan": [], "kind": None, "questions": [], "checks": [], "removed": [], "critic": [],
            "planner": {}, "sums": {"code": [], "verified": [], "wrong": []}, "economics": {"as_said": [], "assumed": []}, "calls": [],
            "cost_eur": float(solution.get("cost_eur") or 0.0), "harness_report": {"mode": "rules", "roles_called": [], "cost_eur": 0.0, "gates": {}}}
    return {**for_product(note), "summary": NO_MODEL_NOTE, "markdown": NO_MODEL_NOTE, "simulation": {"questions": [], "seed_description": ""}}
