"""Made-up data, only after the person has been asked how it should look and has said yes.

Nothing in this package creates data. ``questions`` writes what to ask; ``description`` turns the answers into the words
the existing design step reads (``dclab_rnd.draft.synthetic_design.propose``), which shows a design for the person to
approve; the table itself is created by the existing route on their yes (``POST .../data/synthetic`` with the proposal)."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def questions(solution: Mapping[str, Any]) -> list[str]:
    """What the person is asked before any design is drawn: the questions the architect wrote for them, then the ones a
    design cannot do without."""
    asked = [q for q in solution.get("questions") or [] if "?" in q]
    names = [c["column"] for c in solution.get("columns") or []]
    base = [
        "Which of these columns should the made-up table have, and is one missing: " + ", ".join(names) + "?" if names
        else "Which columns should the made-up table have?",
        "How many rows, and what period should they cover?",
        "What should drive the outcome in the made-up data, and how often should it happen? Use your own numbers where you have them.",
    ]
    return [*asked, *[q for q in base if q not in asked]][:6]


def description(solution: Mapping[str, Any], answers: Sequence[str]) -> str:
    """The design request in words, from the solution's own table and the person's answers; never a value of their data."""
    rows = "; ".join(f"{c['column']} ({c['what']}; {c['role']})" for c in solution.get("columns") or [])
    return (f"A made-up table for a demonstration. The problem, in the person's words: {str(solution.get('words') or '').strip()} "
            f"Columns: {rows}. The person's wishes for the made-up data: " + " ".join(str(a).strip() for a in answers if str(a).strip()))
