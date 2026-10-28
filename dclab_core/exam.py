"""A frozen exam per model type, and the no-forgetting gate (package 36.4).

A model trained again on new data can lose what it knew (it "forgets"). To see that, every version of a model type sits
the same exam, and a new version is accepted only if it kept every old skill.

* :class:`Exam` (v1): a model type's exam, drawn once and never changed. Its sections are skills: one per set of the type
  (a decision point, a stage, a kind of trajectory), plus, for the language models, general skills (tool calling, Persian,
  code, refusals). Each :class:`ExamItem` names one row by its key (its raw entry and its place there, the same in every
  model type's set), its hash, its group (a table or a project: rows of one group are not independent), how an answer
  is checked and the expected answer. No row's content is kept here: the host reads the row from where it was sealed.
  ``hash`` pins the exam: a sha256 of everything in it but the time it was drawn.
* :class:`ExamResult` (v1): one model version's score on one exam: every item's score (0 to 1), each section's score with
  its 95% interval (a bootstrap that draws whole groups, as ``decision_gate`` does).
* :func:`no_forgetting`: the gate, fixed before any run (``no_forgetting_v1``). For every section, the paired difference
  (new minus previous, item by item) with its 95% interval by whole groups; a section *forgets* when even the interval's
  upper end is below ``-drop_floor``, that is, it dropped by more than its interval. One forgotten section refuses the
  version. A section with few items cannot show a small drop; the verdict names it as weakly protected, and an empty
  section as unprotected.

A verdict is evidence about one version on one exam, never a statement that a model is ready for production.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

GATE = "no_forgetting_v1"
THRESHOLDS = {
    "draws": 2000,                 # bootstrap draws of whole groups, seed 0
    "seed": 0,
    "drop_floor": 0.01,            # a section forgets when the 95% upper bound of (new - previous) is below -0.01
    "min_items_protected": 20,     # below this a section is named weakly protected (a drop of 0.1 may not show)
    "min_groups": 10,              # below this the bootstrap draws items, not groups (two groups cannot bound a drop), and the section is weakly protected
}
Check = Literal["label", "tool_name", "code", "text_f1", "choice", "score_close", "set_f1", "persian_reply", "refusal"]
GENERAL = ("tool_calling", "persian", "code", "refusals")  # the general skills of a language model's exam
_SLUG = r"^[a-z0-9][a-z0-9_.\-]{0,119}$"  # a store folder name: no colon


class ExamItem(Contract):
    """One question: a sealed row, how its answer is checked and what is expected (never the row's content)."""

    id: str = Field(max_length=300)              # the row's key: its raw entry and its place in it
    example: str = Field(max_length=300)         # the example's id in its set
    set: str = Field(max_length=300)             # <model type>/<folder>/<version>: where it is sealed
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")   # of the row as it is written in its sealed file
    content: str = Field(pattern=r"^[0-9a-f]{64}$")  # of the row without its id, split and provenance: a copy kept again has the same
    group: str = Field(max_length=300)           # its table, project or entry (drawn together by the bootstrap)
    check: Check
    expect: Any = None
    expect_source: str = Field(max_length=80)    # label:<who>, rule, recorded (what the agent did)
    step: Optional[int] = Field(default=None, ge=0)  # a step of an episode or a trajectory: the question is the step
    swap: bool = False                           # a pair shown in the other order (the expected answer says which)


class ExamSection(Contract):
    name: str = Field(max_length=200)
    general: bool = False                        # a general skill (tool calling, Persian, code, refusals)
    items: list[ExamItem] = Field(default_factory=list)
    note: Optional[str] = Field(default=None, max_length=600)  # where its rows come from, or why it is empty


class Exam(Versioned):
    """A model type's frozen exam (v1): drawn once, pinned by its hash, never in a training set."""

    schema_version: Literal["dclab_core.exam.v1"] = _version("exam")
    model_type: str = Field(pattern=_SLUG)
    version: str = Field(pattern=_SLUG)          # exam_v1: a new exam is a new version, never a change to this one
    drawn_at: str
    rule: str = Field(max_length=1200)           # how it was drawn
    sections: list[ExamSection] = Field(min_length=1)
    hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _sound(self):
        names = [s.name for s in self.sections]
        if len(names) != len(set(names)):
            raise ValueError("each section once")
        ids = [i.id for s in self.sections for i in s.items]
        if len(ids) != len(set(ids)):
            raise ValueError("each row once in an exam (one section per item)")
        if self.hash != exam_hash(self.model_type, self.version, self.rule, self.sections):
            raise ValueError("the exam's hash does not match its content: it was changed after it was drawn")
        return self

    def items(self) -> list[ExamItem]:
        return [i for s in self.sections for i in s.items]

    def section_of(self) -> dict[str, str]:
        return {i.id: s.name for s in self.sections for i in s.items}


def exam_hash(model_type: str, version: str, rule: str, sections: list[Any]) -> str:
    """The pin: everything in the exam but when it was drawn."""
    body = {"model_type": model_type, "version": version, "rule": rule,
            "sections": [s.model_dump(mode="json") if hasattr(s, "model_dump") else s for s in sections]}
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")).hexdigest()


def build(model_type: str, version: str, rule: str, sections: list[ExamSection], drawn_at: str) -> Exam:
    return Exam(model_type=model_type, version=version, drawn_at=drawn_at, rule=rule, sections=sections,
                hash=exam_hash(model_type, version, rule, sections))


class SectionScore(Contract):
    name: str = Field(max_length=200)
    items: int = Field(ge=0)
    answered: int = Field(ge=0)
    score: Optional[float] = None                # the mean item score (an unanswered item scores 0); None: an empty section
    lower: Optional[float] = None                # 95% interval, whole groups drawn
    upper: Optional[float] = None


class ExamResult(Versioned):
    """One model version's score on one exam (v1), kept with the version."""

    schema_version: Literal["dclab_core.exam_result.v1"] = _version("exam_result")
    model_type: str = Field(pattern=_SLUG)
    exam_version: str = Field(pattern=_SLUG)
    exam_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    model_version: str = Field(max_length=200)
    model: Optional[str] = Field(default=None, max_length=200)  # the name it is served by (the tuned tier's model)
    scored_at: str
    scores: dict[str, float] = Field(default_factory=dict)      # item id -> score in [0, 1] (unanswered: 0)
    unanswered: list[str] = Field(default_factory=list)
    sections: list[SectionScore] = Field(default_factory=list)

    @model_validator(mode="after")
    def _in_range(self):
        bad = [k for k, v in self.scores.items() if not 0.0 <= v <= 1.0]
        if bad:
            raise ValueError(f"an item's score is between 0 and 1: {bad[:3]}")
        return self


def _weights(groups: list[str], k: int, rng: Any) -> Any:
    import numpy as np

    ids = {g: i for i, g in enumerate(dict.fromkeys(groups))}
    cl = np.array([ids[g] for g in groups])
    return np.bincount(rng.integers(0, k, k), minlength=k)[cl].astype(float)


def interval(values: list[float], groups: list[str], draws: int = THRESHOLDS["draws"], seed: int = THRESHOLDS["seed"]) -> tuple[Optional[float], Optional[float], Optional[float]]:
    """The mean of ``values`` and its 95% interval from a bootstrap that draws whole groups (``decision_gate``'s method)."""
    import numpy as np

    if not values:
        return None, None, None
    v = np.array(values, dtype=float)
    k = len(set(groups))
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        w = _weights(groups, k, rng)
        if w.sum():
            drawn.append(float((v * w).sum() / w.sum()))
    return round(float(v.mean()), 4), round(float(np.percentile(drawn, 2.5)), 4), round(float(np.percentile(drawn, 97.5)), 4)


def score(exam: Exam, item_scores: Mapping[str, Optional[float]], model_version: str, scored_at: str, model: Optional[str] = None) -> ExamResult:
    """An :class:`ExamResult` from each item's score (None or absent: unanswered, scored 0). An id that is not an item of
    the exam is refused: a result answers this exam and no other."""
    known = {i.id for i in exam.items()}
    stray = sorted(set(item_scores) - known)
    if stray:
        raise ValueError(f"{len(stray)} scored ids are not items of this exam (first: {stray[0]})")
    scores, unanswered = {}, []
    for i in exam.items():
        value = item_scores.get(i.id)
        if value is None:
            unanswered.append(i.id)
        scores[i.id] = float(min(1.0, max(0.0, value or 0.0)))
    sections = []
    for s in exam.sections:
        vals = [scores[i.id] for i in s.items]
        mean, lo, hi = interval(vals, [i.group for i in s.items])
        sections.append(SectionScore(name=s.name, items=len(s.items), answered=sum(i.id not in unanswered for i in s.items),
                                     score=mean, lower=lo, upper=hi))
    return ExamResult(model_type=exam.model_type, exam_version=exam.version, exam_hash=exam.hash, model_version=model_version,
                      model=model, scored_at=scored_at, scores=scores, unanswered=unanswered, sections=sections)


def no_forgetting(exam: Exam, previous: Optional[ExamResult], new: ExamResult) -> dict[str, Any]:
    """The no-forgetting gate: may ``new`` follow ``previous`` (the version accepted before it) on this exam?"""
    import numpy as np

    t = THRESHOLDS
    base = {"gate": GATE, "thresholds": dict(t), "exam": exam.version, "exam_hash": exam.hash, "model_type": exam.model_type,
            "new": new.model_version, "previous": previous.model_version if previous else None,
            "scope": "Evidence about this version on this exam, not a claim that it is ready for production."}
    wrong = [r.model_version for r in (previous, new) if r is not None and (r.exam_hash != exam.hash or r.model_type != exam.model_type)]
    if wrong:
        return {**base, "pass": False, "sections": [], "forgotten": [], "refused": f"not a result of this exam: {', '.join(wrong)}"}
    rows = []
    for s in exam.sections:
        n = len(s.items)
        if not n:
            rows.append({"section": s.name, "items": 0, "status": "unprotected", "note": s.note or "no item"})
            continue
        now = [new.scores.get(i.id, 0.0) for i in s.items]
        if previous is None:
            rows.append({"section": s.name, "items": n, "new": round(float(np.mean(now)), 4), "status": "first"})
            continue
        before = [previous.scores.get(i.id, 0.0) for i in s.items]
        groups = [i.group for i in s.items]
        few = len(set(groups)) < t["min_groups"]  # with few groups the bootstrap over groups cannot bound a drop: draw items
        d, lo, hi = interval([a - b for a, b in zip(now, before)], [i.id for i in s.items] if few else groups, t["draws"], t["seed"])
        forgot = hi is not None and hi < -t["drop_floor"]
        status = "forgot" if forgot else ("weakly_protected" if n < t["min_items_protected"] or few else "kept")
        rows.append({"section": s.name, "items": n, "previous": round(float(np.mean(before)), 4), "new": round(float(np.mean(now)), 4),
                     "difference": d, "lower": lo, "upper": hi, "status": status})
    forgotten = [r["section"] for r in rows if r["status"] == "forgot"]
    return {**base, "pass": not forgotten, "sections": rows, "forgotten": forgotten,
            "weakly_protected": [r["section"] for r in rows if r["status"] == "weakly_protected"],
            "unprotected": [r["section"] for r in rows if r["status"] == "unprotected"],
            "note": None if previous else "the first version: nothing earlier to forget"}


CONTRACTS["exam"] = Exam
CONTRACTS["exam_result"] = ExamResult

__all__ = ["GATE", "THRESHOLDS", "GENERAL", "Check", "ExamItem", "ExamSection", "Exam", "SectionScore", "ExamResult", "exam_hash",
           "build", "interval", "score", "no_forgetting"]
