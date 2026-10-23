"""What a solver is shown (H2): the family's accepted lessons and its similar cases, within one token budget.

:func:`for_solver` builds the payload an agent's session carries (``session["playbook"]``): the text, the bullet ids
shown, the playbook's lineage and version. The intern puts the text in its **first user message**, never in the system
prompt, so the prompt hashes that gate every change stay meaningful. :func:`cited_in` reads which shown ids the agent
named in its replies and thoughts afterwards (a bullet it used).
"""

from __future__ import annotations

from typing import Any

from dclab_core.playbook import BUDGET, Playbook, cited, render, tokens

from .cases import lines


def for_solver(book: Playbook, cases: list[dict[str, Any]] | None = None, budget: int = BUDGET,
               statuses: tuple[str, ...] = ("accepted",)) -> dict[str, Any] | None:
    """The playbook a solver reads, or None when there is nothing to show. Cases take what the bullets leave of the
    budget, never more."""
    text, ids = render(book, budget=budget, statuses=statuses)
    header = "Similar past problems:"
    left = budget - tokens(text) - tokens(header) - 2  # the header and the line breaks count too
    case_lines = []
    for line in lines(cases or []):
        if tokens(line) + 1 > left:
            break
        case_lines.append(line)
        left -= tokens(line) + 1
    if not text and not case_lines:
        return None
    body = "\n".join([text, *([header] if case_lines else []), *case_lines]).strip()
    return {"family": book.family, "lineage": book.lineage, "version": book.version, "shown": ids, "text": body}


def cited_in(session: dict[str, Any]) -> list[str]:
    """The shown bullets a session's steps or replies name by id."""
    shown = list(((session.get("playbook") or {}).get("shown")) or [])
    if not shown:
        return []
    words = " ".join([str(s.get("thought") or "") for s in session.get("steps") or []]
                     + [str(m.get("content") or "") for m in session.get("messages") or [] if m.get("role") == "assistant"])
    return cited(words, shown)


__all__ = ["for_solver", "cited_in"]
