"""The free layer: one model pass writes the whole solution for a person with no table to share.

The model reads the person's words and writes under fixed headings (``prompts/no_data_architect.md``): the problem, what
needs care, the business solution, the data to record, the model when one is asked for, the plan. It chooses what the
person needs before it writes, and names the kind of problem last. Code writes nothing of the solution; it checks it
(``checks.run``) and may send the failed checks back once.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from dclab_rnd import prompts

PURPOSE = "no_data_architect"
PROMPT = "no_data_architect"
MAX_TOKENS = 16000  # the room the model alone had in the measurements of 2026-11-15: a cut answer scores as a bad one


def opening(words: str, sums: Sequence[Mapping[str, str]]) -> str:
    """What the architect is sent: the person's words and, apart from them, the sums code already checked."""
    note = ""
    if sums:
        note = ("\n\n[DCLab, not the person] Sums already checked by code from the numbers above. Use one only where it fits what "
                "the person meant:\n" + "\n".join(f"- `{s['sum']}`: {s['means']}" for s in sums))
    return str(words).strip() + note


def _said(out: Any) -> str:
    return str((out or {}).get("content") or "").strip()


def write(client: Any, words: str, sums: Sequence[Mapping[str, str]] = ()) -> tuple[str, list[dict[str, str]]]:
    """The architect's answer and the messages that produced it (kept for one repair)."""
    messages = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": opening(words, sums)}]
    return _said(client.complete(messages, None, max_tokens=MAX_TOKENS)), messages


def repair(client: Any, messages: Sequence[Mapping[str, str]], draft: str, failed: Sequence[Mapping[str, Any]]) -> str:
    """The answer written again after code's checks: once, never more. An empty rewrite keeps the draft."""
    note = ("[DCLab's checks, not the person] The answer above does not pass these checks:\n"
            + "\n".join(f"- {c['detail']}" for c in failed if c.get("detail"))
            + "\nWrite the whole answer again, with the same headings, so that every check passes. Change nothing else.")
    again = _said(client.complete([*messages, {"role": "assistant", "content": draft}, {"role": "user", "content": note}], None, max_tokens=MAX_TOKENS))
    return again or draft
