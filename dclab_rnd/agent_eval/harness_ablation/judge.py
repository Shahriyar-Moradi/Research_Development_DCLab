"""The judge of the harness ablation: a model that is not the arms' model reads one conversation against the case's answer
key, blind to the arm. It answers one JSON object, checked here; an answer of the wrong shape is invalid, never repaired.
No reason is asked for in the answer: the judge's thinking is kept by the gateway as its API thinking summary.

What the judge cannot be blind to: an arm that shows option cards or draws a chart looks different from one that writes
a plan. The arm's name is never given; its manner is visible. The owner's blind labels (``labels``) check the judge."""

from __future__ import annotations

from typing import Any, Mapping

from dclab_rnd import prompts

from ..conversation_exam.person import first_json, show

PROMPT = "ablation_judge"
MAX_TOKENS = 8000  # room for a thinking model: its thinking counts against the limit (a cut answer is an invalid one)
DIMENSIONS = ("understanding", "solution_design", "planning", "evidence_discipline", "adaptation")


def case_text(case: Mapping[str, Any], key: Mapping[str, Any], transcript: list[Mapping[str, Any]]) -> str:
    """What the judge reads: the person's words are in the conversation; then the key. Nothing says which arm spoke."""
    def numbered(items: Any) -> str:
        return "\n".join(f"{i}. {x}" for i, x in enumerate(items or [], 1)) or "(none)"

    check = f"Adaptation check: {key['adaptation_check']}" if key.get("adaptation_check") else "Adaptation check: none."
    table = f"The person shared a table: {case['upload']}.csv." if case.get("upload") else "The person shared no table."
    return (f"{table}\n\nPoints a good expert must understand:\n{numbered(key.get('must_understand'))}\n\n"
            f"Approaches an expert would accept:\n{numbered(key.get('acceptable_approaches'))}\n\n"
            f"Things an expert must not do:\n{numbered(key.get('forbidden'))}\n\n{check}\n\n"
            f"The conversation:\n{show(transcript, person='Person')}")


def _score(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 4


def check(value: Any, key: Mapping[str, Any]) -> list[str]:
    """What is wrong with a judge's answer (empty when its shape is right)."""
    if not isinstance(value, dict):
        return ["not a JSON object"]
    problems = []
    for name, wanted in (("understood", len(key.get("must_understand") or [])), ("forbidden_done", len(key.get("forbidden") or []))):
        got = value.get(name)
        if not (isinstance(got, list) and len(got) == wanted and all(isinstance(x, bool) for x in got)):
            problems.append(f"{name} must be {wanted} true or false values")
    for name in DIMENSIONS[:-1]:
        if not _score(value.get(name)):
            problems.append(f"{name} must be a whole number from 0 to 4")
    adaptation = value.get("adaptation")
    if key.get("adaptation_check"):
        if not _score(adaptation):
            problems.append("adaptation must be a whole number from 0 to 4: the case has an adaptation check")
    elif adaptation is not None:
        problems.append("adaptation must be null: the case has no adaptation check")
    if not isinstance(value.get("accepted"), bool):
        problems.append("accepted must be true or false")
    return problems


def judge(client: Any, case: Mapping[str, Any], key: Mapping[str, Any], transcript: list[Mapping[str, Any]]) -> dict[str, Any]:
    messages = [{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": case_text(case, key, transcript)}]
    out = client.complete(messages, None, max_tokens=MAX_TOKENS)
    value = first_json(str(out.get("content") or ""))
    problems = check(value, key)
    if problems:
        return {"valid": False, "problems": problems, "raw": str(out.get("content") or "")}
    return {"valid": True, **{k: value[k] for k in ("understood", "forbidden_done", *DIMENSIONS, "accepted")}}


def judged(client: Any, case: Mapping[str, Any], key: Mapping[str, Any], transcript: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Asked once more when the first answer has the wrong shape (a judge's slip must not count against an arm)."""
    first = judge(client, case, key, transcript)
    if first.get("valid"):
        return first
    again = judge(client, case, key, transcript)
    return again if again.get("valid") else {**again, "first_problems": first.get("problems")}
