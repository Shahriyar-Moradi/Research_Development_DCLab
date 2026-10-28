"""A lesson drawn from a graded attempt, with the kind of gap it shows (Phase 9, package 31.5).

:class:`Lesson` v1 is the reflector's structured record: the family, the **gap type** (what kind of gap the mistake
shows), the decision point it concerns, what was observed and what was expected (in the grader's words, never a column's
name), the general lesson, how far it reaches, the confidence, and the evidence (pack, trace, grader, code). The gap type
decides where the lesson goes (:data:`ROUTE`): a playbook bullet, a labelled sample for a decision point's student, a new
check, a new recipe part, critiques for training, or nothing when the data cannot answer the question. Every lesson also
goes to case memory. A rule or capability gap leads to code, so it is confirmed by a person before it is acted on.

:func:`propose` is the rules' proposal from a grader code (``rule:gaps_v1``): plain mapping, written once here, never from
an attempt; :data:`RULES` holds the gap for an attempt a model made; an attempt DCLab's rules made shows rule gaps (no
check stopped the trap), and a variant the reference cannot solve shows the data's. A model or a person may propose
otherwise, and the record says who did.

Pure: codes in, records out. Nothing here reads a file or calls a model.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Gap = Literal["knowledge", "reasoning", "decision", "rule", "capability", "data"]
GAPS: tuple[str, ...] = ("knowledge", "reasoning", "decision", "rule", "capability", "data")
CONFIRMED_BY_A_PERSON = ("rule", "capability")  # they lead to code

# Where a lesson of each gap goes, and its level in the loop (Part F of the plan); every lesson also goes to case memory.
ROUTE: dict[str, tuple[str, Optional[int]]] = {
    "knowledge": ("playbook", 2),
    "reasoning": ("critique_samples", 4),
    "decision": ("student_samples", 4),
    "rule": ("check", 3),
    "capability": ("recipe_part", 3),
    "data": ("none", None),
}

# The rules' proposal for each grader code: the gap, the decision point, what was expected, how far it reaches, and how
# sure the rule is. Written here once; a code it does not know gets no proposal.
RULES: dict[str, tuple[str, Optional[str], str, str, float]] = {
    "SPLIT_WRONG:repeated_entity": ("decision", "split.strategy", "the folds and the holdout grouped by the entity whose rows repeat",
                                    "any table where one entity has several rows with lasting traits", 0.8),
    "IDENTIFIER_USED:row_key": ("decision", "column.is_identifier", "a row's own key kept out of the inputs",
                                "any table with a column that is different on every row", 0.8),
    "LEAK_KEPT:after_outcome": ("decision", "feature.leakage_suspect", "a column written after the outcome kept out of the inputs",
                                "any table with columns filled once the outcome is known", 0.7),
    "LEAK_LEANED:after_outcome": ("decision", "feature.leakage_suspect", "the model never leaning on a column written after the outcome",
                                  "any table with columns filled once the outcome is known", 0.7),
    "FALSE_ALARM:known_before": ("decision", "feature.leakage_suspect", "a column known before the prediction moment kept in, whatever its name",
                                 "any table whose honest columns have names that sound like an outcome", 0.7),
    "GAP_OPTIMISTIC:validation_over_holdout": ("reasoning", None, "a validation score no better than its noise allows over the holdout, or the gap explained",
                                               "any solve whose validation and holdout disagree", 0.5),
    "NO_SIGNAL:not_above_baseline": ("capability", None, "a holdout score shown better than the simplest baseline on a variant a reference solves",
                                     "variants of this family the pipeline does not solve", 0.3),
}


class Evidence(Contract):
    pack: str = Field(pattern=r"^pk-[0-9a-f]{6,40}$")
    trace: Optional[str] = Field(default=None, max_length=300, description="the attempt's run or kept folder")
    grader: str = Field(max_length=40)
    code: str = Field(max_length=80)


class Lesson(Versioned):
    schema_version: Literal["dclab_core.lesson.v1"] = _version("lesson")
    id: str = Field(pattern=r"^L-[0-9a-f]{10}$")
    family: str = Field(max_length=40)
    gap_type: Gap
    proposed_by: str = Field(max_length=120, description="rule:gaps_v1, model:<name>, person")
    decision_point: Optional[str] = Field(default=None, max_length=60)
    observed: str = Field(min_length=3, max_length=600, description="in the grader's words: never a column's name")
    expected: str = Field(min_length=3, max_length=600)
    lesson: str = Field(min_length=3, max_length=600)
    generalises_to: str = Field(default="", max_length=300)
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: list[Evidence] = Field(min_length=1)
    bullet: Optional[str] = Field(default=None, pattern=r"^B-[0-9a-f]{10}$", description="the playbook bullet it became, if any")
    status: Literal["proposed", "confirmed", "refused"] = "proposed"
    confirmed_by: Optional[str] = Field(default=None, max_length=120)
    at: str

    @model_validator(mode="after")
    def _confirmed(self) -> "Lesson":
        if self.status == "confirmed" and self.gap_type in CONFIRMED_BY_A_PERSON and not str(self.confirmed_by or "").startswith("person"):
            raise ValueError(f"a {self.gap_type} gap leads to code: a person confirms it")
        if self.status != "proposed" and not self.confirmed_by:
            raise ValueError("a confirmed or refused lesson says who decided")
        return self

    @property
    def route(self) -> tuple[str, Optional[int]]:
        return ROUTE[self.gap_type]

    @property
    def actionable(self) -> bool:
        """Whether its route may act now: a rule or capability gap only once a person confirmed it, a data gap never."""
        if self.gap_type == "data" or self.status == "refused":
            return False
        return self.status == "confirmed" or self.gap_type not in CONFIRMED_BY_A_PERSON


def propose(family: str, code: str, meaning: str, lesson: str, evidence: Evidence, at: str, lesson_id: str,
            bullet: Optional[str] = None, producer: str = "rule:dclab", solvable: bool = True) -> Optional[Lesson]:
    """The rules' lesson for one grader code (``meaning`` is the grader's own words for it), or None for a code the rules
    do not know. ``producer`` is who made the attempt: when DCLab's rules made it, nothing chose or reasoned, so a miss is
    a check that did not stop the trap (a rule gap; a pipeline that found no signal stays a capability gap) and a person
    confirms it. On a variant the reference does not solve either, the gap is the data's."""
    if code not in RULES:
        return None
    gap, point, expected, reach, confidence = RULES[code]
    if not solvable:
        gap = "data"
    elif str(producer).startswith("rule:") and gap in ("decision", "reasoning"):
        gap = "rule"
    return Lesson(id=lesson_id, family=family, gap_type=gap, proposed_by="rule:gaps_v1", decision_point=point, observed=meaning,
                  expected=expected, lesson=lesson, generalises_to=reach, confidence=confidence, evidence=[evidence], bullet=bullet, at=at)


CONTRACTS["lesson"] = Lesson


__all__ = ["Gap", "GAPS", "ROUTE", "RULES", "CONFIRMED_BY_A_PERSON", "Evidence", "Lesson", "propose"]
