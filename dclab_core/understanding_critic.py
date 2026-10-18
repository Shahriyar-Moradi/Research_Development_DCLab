"""The understanding critic (package 28.4): reads the problem card against the data card before a solution is drawn and
names what is still wrong or unsure, each as a question for the person; it never edits the card.

The role's rules (its fallback) are the checks a careful reviewer runs on the card and the tables: the meet check's
blocking mismatches (``dclab_core.meet``), an outcome that is a column the data card reads as an identifier, a time or a
column written after the outcome, and a prediction moment left unstated while the table holds a time. A model may add
concerns of its own; each is kept only when it names a card field and only columns the tables hold (the validator), and
the concerns are merged with the rules' (the rules first, one per field). The host turns each concern into a question the
interviewer asks next; a field is never sent back twice, and the walk sends back at most ``ROUNDS`` times.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from dclab_core.agents import Role

FIELDS = ("target", "prediction_moment", "entity", "action", "labels", "family")  # the card fields a concern may name
ROUNDS = 2  # send-backs per walk
NOT_AN_OUTCOME = ("identifier", "entity", "time", "leak_suspect", "row_number", "constant")
# a meet mismatch's code, by what it names, to the card field its question settles (the review of 28.4)
MISMATCH_FIELD = (("label", "labels"), ("target", "target"), ("outcome", "target"), ("time", "prediction_moment"),
                  ("moment", "prediction_moment"), ("unit", "entity"), ("grain", "entity"), ("entity", "entity"))
IDENTIFIER = re.compile(r"`([^`]{1,80})`|\b([A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)+)\b")  # a backticked name, or a snake_case one


def concerns(card: Mapping[str, Any], columns: list[Mapping[str, Any]], mismatches: list[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
    """The rules' concerns: ``card`` the summary (family, outcome, moment, labels, ready), ``columns`` the data card's
    columns (name, purpose, role), ``mismatches`` the meet check's blocking ones (code, question)."""
    out: list[dict[str, Any]] = []
    by_name = {str(c.get("name")): c for c in columns}
    outcome = card.get("outcome")
    for m in mismatches or ():
        code = str(m.get("code") or "")
        field = next((f for word, f in MISMATCH_FIELD if word in code), "family")
        out.append({"field": field, "question": str(m.get("question") or "")[:300], "why": f"the meet check: {m.get('code')}", "columns": []})
    if outcome and outcome in by_name and by_name[outcome].get("purpose") in NOT_AN_OUTCOME:
        purpose = by_name[outcome].get("purpose")
        out.append({"field": "target", "question": f"{outcome} reads as {purpose.replace('_', ' ')}, not an outcome: which column holds what you want to predict?",
                    "why": f"the data card reads {outcome} as {purpose}", "columns": [outcome]})
    times = [n for n, c in by_name.items() if c.get("purpose") == "time"]
    if times and not card.get("moment"):
        out.append({"field": "prediction_moment", "question": f"The table has a time ({times[0]}): at what moment is the prediction made, and what is known then?",
                    "why": "a time in the table and no prediction moment: the split and the leaks depend on it", "columns": times[:1]})
    return _one_per_field(out)


def _one_per_field(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen, out = set(), []
    for c in items:
        if c["field"] in seen or c["field"] not in FIELDS or not c.get("question"):
            continue
        seen.add(c["field"])
        out.append(c)
    return out


def _evidence(inputs: Mapping[str, Any]) -> Mapping[str, Any]:
    """A summary, never a table: one line per column ("name | purpose | role | type | missing | distinct"), in lists of 50
    (as the data investigator's), the card's fields, and the checks' mismatches as lines."""
    card = inputs.get("card") or {}
    lines = [" | ".join(str(c.get(k) or "-")[:200] for k in ("name", "purpose", "role", "dtype", "nulls", "uniqueness"))
             for c in (inputs.get("columns") or [])][:150]
    return {"card": {k: (None if card.get(k) is None else str(card.get(k))[:300]) for k in ("family", "labels", "outcome", "moment", "unit", "action", "ready")},
            "columns": {f"{i + 1}-{i + len(chunk)}": chunk for i, chunk in ((i, lines[i:i + 50]) for i in range(0, len(lines), 50))},
            "mismatches": [f"{m.get('code')}: {str(m.get('question') or '')[:300]}" for m in inputs.get("mismatches") or []][:20],
            "fields": list(FIELDS)}


def _parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    names = {str(c.get("name")) for c in inputs.get("columns") or []}
    out = []
    for c in (answer or {}).get("concerns") or []:
        if not isinstance(c, Mapping):
            raise ValueError("a concern is an object")
        field, cols = str(c.get("field") or ""), [str(x) for x in c.get("columns") or []]
        if field not in FIELDS:
            raise ValueError(f"{field!r} is not a card field")
        if any(x not in names for x in cols):  # the gate: a column the tables do not hold refuses the answer
            raise ValueError(f"a concern names columns the tables do not hold: {[x for x in cols if x not in names]}")
        question, why = str(c.get("question") or "").strip(), str(c.get("why") or "").strip()
        if not question:
            raise ValueError("a concern without a question")
        named = {a or b for a, b in IDENTIFIER.findall(f"{question} {why}")} - names - set(FIELDS)
        if named:  # a name in its words the tables do not hold refuses the answer as a listed one does
            raise ValueError(f"a concern's words name columns the tables do not hold: {sorted(named)}")
        out.append({"field": field, "question": question[:300], "why": why[:300], "columns": cols[:10]})
    return out


def _merge(rules: list[dict[str, Any]], model: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _one_per_field([*rules, *model])  # the rules first: a model adds, never overrides


UNDERSTANDING_CRITIC = Role(
    name="understanding_critic",
    description="Reads the problem card against the data card before a solution is drawn; each concern a question for the "
                "person, never an edit of the card.",
    evidence=_evidence, parse=_parse, fallback=lambda inputs: concerns(inputs.get("card") or {}, list(inputs.get("columns") or []),
                                                                       list(inputs.get("mismatches") or [])),
    merge=_merge)


def critique(advisor: Optional[Any], card: Mapping[str, Any], columns: list[Mapping[str, Any]], mismatches: list[Mapping[str, Any]] = ()) -> Any:
    from dclab_core.agents import run_role

    return run_role(UNDERSTANDING_CRITIC, advisor, {"card": dict(card), "columns": list(columns), "mismatches": list(mismatches)})


__all__ = ["FIELDS", "ROUNDS", "UNDERSTANDING_CRITIC", "concerns", "critique"]
