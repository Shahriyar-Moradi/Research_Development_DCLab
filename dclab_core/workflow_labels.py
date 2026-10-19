"""Labels for the workflow's own decision points (package 26.3), from a written journey's truth.

A journey (a written person with their problem, the answers they can give, and the kind of problem their files are) is
played through the Home conversation; every workflow decision it leaves in the ledger is labelled here from what the
journey says is true: the answers it accepts, never DCLab's answer, and an answer of DCLab's or of a model is scored
right when it is one of them (``canonical``). These labels train and judge generation-1 classifiers for the
workflow points (``decision_models.train`` and ``decision_gate.verdict``, the same as the column points).

Learned points (``POINTS``) and what each journey accepts (the label a classifier learns is the first of them):

- ``problem.family``: the families the journey accepts; a family the person stated is not the rules' decision and is
  left out by the runner;
- ``problem.ready``: ready only with a family the journey accepts; a card the rules held back with the person's family
  is not labelled (the journey cannot tell whether it was ready);
- ``workflow.next_question``: while one of the card's fields (outcome, moment, action) is open and the person can
  answer it, any question the person can answer (the state does not say which further fields are open); once they are
  settled, stopping (learned) or any further field the person can answer; not labelled at the conversation's cap
  (``QUESTION_LIMIT``) nor when the outcome is asked again with the table's columns, both the product's own rules;
- ``solution.route``: the options a programme journey expects for a sub-problem, and for a one-part map the journey's
  own kind; a route the rules refused (every approach blocked by an unmet need) is not labelled.

Every other workflow point is named in ``NOT_LEARNED`` with the reason it gets no classifier from journeys.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple, Optional, Sequence

VERSION = "workflow_labels_v2"  # v2 (the review of 26.3): accepted sets, never DCLab's answer as the label
POINTS = ("problem.family", "problem.ready", "workflow.next_question", "solution.route")
NOT_LEARNED = {
    "map.size": "its state holds the rules' own drawing (the parts and the template): a classifier on it can only copy the rules",
    "map.edit": "the person's own change to the map: an action recorded, not a decision DCLab makes",
    "problem.pack": "only two problem kinds declare a pack, too few to learn from",
    "workflow.template": "drawn from the pack by a fixed table: there is nothing to learn",
    "workflow.move": "the workflow graph's validator owns it: no model decides a move (principle 1 of the plan)",
    "workflow.revisit": "a person's or an agent's request to run a stage again: no journey asks for one",
    "workflow.stage_approval": "the written person approves every gate: the labels would carry no information",
    "workflow.stop": "no agent with a model runs in a journey; its labels come from recorded sessions' outcomes (27.2)",
    "workflow.g4": "G4 is a rule on the run's facts; its truth is where people take the work next, from recorded projects (30.15)",
    "agent.tool_choice": "no agent with a model runs in a journey; its labels come from recorded sessions' outcomes (27.2)",
    "experiment.next": "the engine's own improvement loop: labelled by its measured results, not by a journey",
    "solution.compose": "labelled where it is decided, by the holdout's measured comparison (24.7)",
    "team.next_node": "the walk's own routing rule (situate) chooses the node from the coordinator's answer: a classifier on it can only copy the rule (29.6)",
    "critic.send_back": "labelled where it is decided, by whether the person's answer changed the card's field (29.6)",
    "command.intent_route": "not decided by the studio yet",
    "proposal.completeness": "not decided by the studio yet",
    "department.proposal": "labelled where it is judged: a person's judgment of each proposal, not a journey (30.22)",
}
SAME = {"outcome": "target", "target_check": "target", "labels_check": "labels"}  # one question asked under two names
ORDER = ("target", "prediction_moment", "action")  # the fields the card needs, in the order it asks for them
# every field the Home conversation can ask (its plan's, a pack's, and the card's family), and none: the options of the next question
QUESTIONS = ("target", "prediction_moment", "action", "costs", "data_plan", "entity", "labels", "budget", "family", "none")
QUESTION_LIMIT = 5  # the Home conversation's own cap (draft.chat.MAX_QUESTIONS): stopping there is its rule, not a decision


@dataclass(frozen=True)
class Truth:
    """What a journey says is true: the families it accepts, its kind, the fields the person can answer, and, for a
    programme journey, the approaches its sub-problems expect."""

    families: tuple[str, ...]
    kind: Optional[str] = None
    answers: frozenset[str] = field(default_factory=frozenset)
    approaches: tuple[str, ...] = ()
    built: bool = True


def truth_of(journey: Mapping[str, Any]) -> Truth:
    family = journey.get("family")
    families = tuple(str(f) for f in (family if isinstance(family, list) else [family]) if f)
    expect = journey.get("expect") or {}
    if not families and expect.get("families"):
        families = tuple(str(f) for f in expect["families"] if f)
    answers = frozenset(k for k, v in (journey.get("answers") or {}).items() if v)
    return Truth(families=families, kind=journey.get("kind"), answers=answers,
                 approaches=tuple(str(a) for a in expect.get("approaches") or [] if a), built=bool(journey.get("built", True)))


def _can_answer(truth: Truth, name: str) -> bool:
    return name in truth.answers or SAME.get(name, name) in truth.answers


class Label(NamedTuple):
    """A recorded decision's label: every answer the journey accepts (``accepted``), the one a classifier learns
    (``value``: the first accepted in a fixed order, never chosen by what DCLab answered), and why."""

    value: Any
    accepted: tuple[Any, ...]
    why: str


def canonical(answer: Any, label: Label) -> Any:
    """An answer as it is scored: one the journey accepts counts as the label's value (the rules and a model alike)."""
    a = SAME.get(answer, answer) if isinstance(answer, str) else answer
    return label.value if a in label.accepted else answer


def label(point: str, value: Any, state: Mapping[str, Any], options: Sequence[Any], truth: Truth) -> Optional[Label]:
    """The label of one recorded decision, or None when the journey cannot tell. What is accepted never depends on what
    DCLab answered; whether a decision is labelled does in two places, said where: a route the rules refused (every
    approach blocked by an unmet need) and a card the rules held back with the person's family."""
    if point == "problem.family":
        if not truth.families:
            return None
        return Label(truth.families[0], truth.families, "the families the journey accepts")
    if point == "problem.ready":
        family = state.get("family")
        if not truth.families or not family:
            return None
        right = family in truth.families
        if value is not True and right:
            return None  # held back with the person's family: whether it was ready the journey cannot tell
        return Label(right, (right,), "ready only with a family the journey accepts")
    if point == "workflow.next_question":
        steps_left = state.get("steps_left")
        if (steps_left is not None and int(steps_left) <= 0) or int(state.get("step") or 0) >= QUESTION_LIMIT:
            return None  # the conversation's cap was reached: stopping is the product's rule, which the journey cannot judge
        if value == "target" and state.get("outcome_known"):
            return None  # the outcome asked again with the table's columns to choose from: the product's rule, not judged here
        # a file of known cases defines the outcome: its question is withdrawn (the product's rule), so it is not open
        known = {"target": bool(state.get("outcome_known")) or state.get("labels") == "file", "prediction_moment": bool(state.get("moment_known"))}
        open_count = int(state.get("open_questions") or 0)
        known["action"] = open_count <= sum(1 for k in ("target", "prediction_moment") if not known[k])
        core = tuple(n for n in ORDER if not known[n] and _can_answer(truth, n))
        more = tuple(q for q in QUESTIONS if q not in ORDER and q != "none" and _can_answer(truth, q))  # the state cannot say if they are open
        if core:  # any question the person can answer while one of the card's own fields is open; stopping is not
            return Label(core[0], (*core, *more), "a question the person can answer while a field of the card is open")
        accepted = ("none", *more)  # stopping first: the state cannot say a further field was asked already
        return Label("none", accepted, "the card's fields settled: stop, or a further field the person can answer")
    if point == "solution.route":
        if value is None:
            return None  # every approach blocked by an unmet need: the rules refused, which the journey cannot judge
        allowed = [str(o) for o in options or []]
        if truth.approaches:
            expected = tuple(o for o in allowed if o in truth.approaches)
            return Label(expected[0], expected, "the approaches the programme expects") if expected else None
        if int(state.get("parts") or 0) == 1 and truth.kind and truth.kind in allowed:
            return Label(truth.kind, (truth.kind,), "the journey's kind, for its one-part map")
        return None
    return None


__all__ = ["VERSION", "POINTS", "NOT_LEARNED", "QUESTIONS", "QUESTION_LIMIT", "SAME", "Truth", "Label", "truth_of", "label", "canonical"]
