"""The judge (37.0): a different Claude model from the agent reads one conversation, blind to the arm, against the
person's hidden truth, the points a good listener must understand and the trap check. It answers one JSON object
(points understood, each assistant message's question, the options' fit and difference, the trap), checked here; an
answer of the wrong shape is invalid, never repaired or guessed. No reason is asked in the answer: the judge's thinking
is kept by the gateway as its summarized thinking (the owner's rule)."""

from __future__ import annotations

import json
from typing import Any, Mapping

from dclab_rnd import prompts

from .person import first_json, show

PROMPT = "exam_judge"
MAX_TOKENS = 4000


def case_text(person: Mapping[str, Any], transcript: list[Mapping[str, Any]]) -> str:
    """What the judge reads: the truth, the points, the trap, the conversation. Nothing says which arm spoke."""
    points = "\n".join(f"{i}. {p}" for i, p in enumerate(person.get("must_understand") or [], 1))
    trap = f"Trap: {person['trap']}. A good assistant: {person.get('trap_check')}" if person.get("trap") else "Trap: none."
    return (f"The person's hidden truth:\n{json.dumps({'level': person.get('level'), 'style': person.get('style'), **dict(person.get('truth') or {})}, ensure_ascii=False, indent=1)}\n\n"
            f"Points a good listener must understand:\n{points}\n\n{trap}\n\n"
            f"The conversation:\n{show(transcript, person='Person')}")


def _offered(transcript: list[Mapping[str, Any]]) -> bool:
    return any(c.get("kind") == "option" for t in transcript if t["role"] == "assistant" for c in t.get("choices") or [])


def _bool(value: Any) -> bool:
    return isinstance(value, bool)


def check(value: Any, person: Mapping[str, Any], transcript: list[Mapping[str, Any]]) -> list[str]:
    """What is wrong with a judge's answer (empty when its shape is right)."""
    if not isinstance(value, dict):
        return ["not a JSON object"]
    problems = []
    points = len(person.get("must_understand") or [])
    understood = value.get("understood")
    if not (isinstance(understood, list) and len(understood) == points and all(_bool(x) for x in understood)):
        problems.append(f"understood must be {points} true or false values")
    said = sum(1 for t in transcript if t["role"] == "assistant")
    questions = value.get("questions")
    if not (isinstance(questions, list) and len(questions) == said
            and all(isinstance(q, dict) and all(_bool(q.get(k)) for k in ("asks", "useful", "already_answered")) for q in questions)):
        problems.append(f"questions must be {said} entries of asks, useful, already_answered")
    offered = _offered(transcript)
    fit = value.get("options_fit")
    if fit is not None and not (isinstance(fit, int) and not isinstance(fit, bool) and 1 <= fit <= 5):
        problems.append("options_fit must be 1 to 5 or null")
    if offered and fit is None:
        problems.append("options were offered: options_fit is needed")
    if value.get("options_differ") is not None and not _bool(value.get("options_differ")):
        problems.append("options_differ must be true, false or null")
    trap = value.get("trap_handled")
    if person.get("trap") and not _bool(trap):
        problems.append("trap_handled must be true or false")
    return problems


def judge(client: Any, person: Mapping[str, Any], transcript: list[Mapping[str, Any]]) -> dict[str, Any]:
    messages = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": case_text(person, transcript)}]
    out = client.complete(messages, None, max_tokens=MAX_TOKENS)
    value = first_json(str(out.get("content") or ""))
    problems = check(value, person, transcript)
    if problems:
        return {"valid": False, "problems": problems, "raw": str(out.get("content") or "")}
    trap = value.get("trap_handled") if person.get("trap") else None
    offered = _offered(transcript)  # Home's buttons are choices, not options written for the person (the review of 37.0)
    return {"valid": True, "understood": value["understood"], "questions": value["questions"],
            "options_fit": value.get("options_fit") if offered else None, "options_differ": value.get("options_differ") if offered else None,
            "trap_handled": trap}
