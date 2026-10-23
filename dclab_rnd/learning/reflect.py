"""The reflector: lessons from a graded attempt (H2 of the harness-learning plan; Phase 9's 31.5 with the reports' editor
discipline).

It reads what the grader gives and nothing else: the codes, the steps of the solve with their verdicts, the family's
current lessons. Never the table, a column's name, a value or the key. It finds the first step that went wrong, then
drafts at most a few lessons, each citing that step and answering a code. Two drafters, one rule:

* the **rules** (no model, the default): one general lesson per code, written once here, never from the attempt;
* a **model** (the ``reflector`` purpose, capped per challenge): the same input, its drafts screened like the rules'.

Editor discipline, as code: an attempt that passed teaches nothing; an attempt that delivered nothing (no holdout)
teaches nothing; a code an existing lesson already answers is not answered again; every draft passes the screen
(:mod:`.filter`) before it reaches a playbook.
"""

from __future__ import annotations

import json
from typing import Any

from dclab_core.challenge_pack import ChallengeKey
from dclab_core.playbook import Playbook, render

from .filter import MOST, screen_all

# Where in a solve each kind of mistake first shows: the solution saved (split, inputs), the leakage review, the models
# chosen on validation, the final read of the holdout. Their order is the order of the solve.
STEPS = ("solution", "leakage", "models", "final")
CODE_STEP = {"SPLIT_WRONG:repeated_entity": "solution", "IDENTIFIER_USED:row_key": "solution", "LEAK_KEPT:after_outcome": "leakage",
             "LEAK_LEANED:after_outcome": "leakage", "FALSE_ALARM:known_before": "leakage", "GAP_OPTIMISTIC:validation_over_holdout": "models",
             "NO_SIGNAL:not_above_baseline": "final"}

# The rules' lessons: one a code, general, written here once (never from an attempt), so they can name nothing of a table.
RULE_LESSONS: dict[str, tuple[str, str]] = {
    "SPLIT_WRONG:repeated_entity": ("domain", "When one column's values repeat across many rows and each value stands for one thing with "
                                    "lasting traits, split and fold by it so no thing sits on both sides, even when its name does not look "
                                    "like a key."),
    "FALSE_ALARM:known_before": ("invariant", "Before keeping a column out as written after the outcome, check when its value is known: a "
                                 "count or total of earlier events is known at the prediction moment, whatever words its name uses."),
    "LEAK_KEPT:after_outcome": ("domain", "A column that is filled or changes only once the outcome is known must stay out of the inputs; "
                                "check how well each column alone predicts the outcome on training rows."),
    "LEAK_LEANED:after_outcome": ("domain", "If the model leans hardest on one column that alone almost gives the outcome away, ask when "
                                  "that column is written before trusting the score."),
    "IDENTIFIER_USED:row_key": ("invariant", "A column that is different on every row identifies the row, not the case: keep it out of the inputs."),
    "GAP_OPTIMISTIC:validation_over_holdout": ("invariant", "When validation beats the holdout by far more than their noise allows, look for a "
                                               "leak or a split that let validation see what it scores before trusting either number."),
    "NO_SIGNAL:not_above_baseline": ("invariant", "Check the model beats the simplest baseline on the holdout before reporting it; if it "
                                     "does not, say so and look at what was kept out."),
}


def steps_of(arm: dict[str, Any]) -> list[dict[str, Any]]:
    """The solve's steps as the reflector sees them: each step and whether a check failed there (no names, no numbers)."""
    failing = {CODE_STEP.get(c["code"]) for c in arm.get("checks") or [] if not c.get("ok") and c.get("code")}
    return [{"step": s, "verdict": "wrong" if s in failing else "ok"} for s in STEPS]


def first_wrong_step(arm: dict[str, Any]) -> str | None:
    wrong = {CODE_STEP.get(c) for c in arm.get("codes") or []}
    return next((s for s in STEPS if s in wrong), None)


def _covered(book: Playbook | None) -> set[str]:
    return {c for b in (book.bullets if book else []) if b.status != "retired" for c in b.codes}


def rule_drafts(arm: dict[str, Any], book: Playbook | None) -> list[dict[str, Any]]:
    covered = _covered(book)
    out = []
    for code in arm.get("codes") or []:
        if code in covered or code not in RULE_LESSONS:
            continue
        scope, text = RULE_LESSONS[code]
        out.append({"text": text, "scope": scope, "codes": [code], "step": CODE_STEP[code]})
    return out


class ReflectorFailed(RuntimeError):
    """The reflector's model gave no answer (its cap reached, or the provider failed)."""


def editor(drafts: list[dict[str, Any]], arm: dict[str, Any], book: Playbook | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """The editor's rules for a model's drafts, as code: a draft answers only codes this attempt failed, none already
    answered, and cites the step where its code's mistake first shows."""
    failed, covered = set(arm.get("codes") or []), _covered(book)
    kept, refused = [], []
    for d in drafts:
        codes = [c for c in d.get("codes") or [] if isinstance(c, str)]
        if not codes or not set(codes) <= failed:
            refused.append({"draft": d, "why": "answers a code this attempt did not fail"})
        elif set(codes) & covered:
            refused.append({"draft": d, "why": "answers a code a lesson already answers"})
        else:
            kept.append(d)
    return kept, refused


def model_drafts(arm: dict[str, Any], book: Playbook | None, client: Any, family: str, cap_eur: float) -> list[dict[str, Any]]:
    """The model's drafts, from the codes and the steps alone; its reply parsed, never trusted (the screen decides)."""
    from .. import prompts

    current, _ = render(book, statuses=("draft", "probed", "accepted")) if book else ("", [])
    evidence = {"family": family, "codes": arm.get("codes") or [], "steps": steps_of(arm), "first_wrong_step": first_wrong_step(arm),
                "lessons_already_held": current}
    if hasattr(client, "run_limit_eur"):  # capped per challenge: what this challenge spent starts at zero
        client.run_limit_eur = cap_eur
        client.spent_eur = client.reserved_eur = 0.0
    import time

    from ..agents import harness

    started = time.monotonic()
    try:
        out = client.complete([{"role": "system", "content": prompts.text("reflector", most=MOST)},
                               {"role": "user", "content": json.dumps(evidence, ensure_ascii=False)}], max_tokens=800)
    except RuntimeError as exc:  # the cap reached, or the provider failed: no lesson, the reason kept
        harness.role_step("reflector", time.monotonic() - started, "error")
        raise ReflectorFailed(str(exc).split(": ", 1)[-1][:300]) from None
    harness.role_step("reflector", time.monotonic() - started, "ok")
    text = str(out.get("content") or "")
    try:
        start, end = text.index("{"), text.rindex("}") + 1
        drafts = json.loads(text[start:end]).get("lessons") or []
    except (ValueError, AttributeError):
        return []
    return [d for d in drafts if isinstance(d, dict)]


def reflect(row: dict[str, Any], key: ChallengeKey, columns: list[str], book: Playbook | None = None, client: Any = None,
            cap_eur: float = 0.10, run: str | None = None) -> dict[str, Any]:
    """Lessons from one graded attempt's rules arm: the screened bullets, the refused drafts with why, and the step."""
    arm = row["arms"]["rules"]
    if arm.get("verdict") == "pass":
        return {"bullets": [], "refused": [], "step": None, "why": "the attempt passed: nothing to learn"}
    if arm.get("holdout") is None:
        return {"bullets": [], "refused": [], "step": None, "why": "the attempt delivered nothing: no lesson from it"}
    by = "rule:reflect_v1" if client is None else f"model:{getattr(client, 'model', 'unknown')}"
    edited: list[dict[str, Any]] = []
    if client is None:
        drafts = rule_drafts(arm, book)
    else:
        try:
            drafts = model_drafts(arm, book, client, key.family, cap_eur)
        except ReflectorFailed as exc:
            return {"bullets": [], "refused": [], "drafts": [], "step": first_wrong_step(arm), "why": f"the reflector gave no answer: {exc}"}
        drafts, edited = editor(drafts, arm, book)
    bullets, refused = screen_all(drafts, key, columns, list(STEPS), row["pack"], run, by, CODE_STEP)
    return {"bullets": bullets, "refused": edited + refused, "drafts": drafts, "step": first_wrong_step(arm),
            "why": "" if bullets else "no draft passed the screen"}


__all__ = ["STEPS", "CODE_STEP", "RULE_LESSONS", "ReflectorFailed", "steps_of", "first_wrong_step", "editor", "rule_drafts", "model_drafts", "reflect"]
