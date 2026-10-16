"""The twin (package 25.1): the system's answer and an AI-first agent's answer to the same step, their differences, and
where each approach is best.

DCLab solves a problem step by step: it reads the person's words into a problem card, asks the next question, chooses a
pack, draws the map, routes each part, draws the workflow, decides whether a build is ready, decides each column's role,
and makes each stage's choices. At any of these steps a second approach can answer beside the system: an agent that
solves the step the way a strong model would, with its own reasoning and its own read-only tools. A :class:`TwinStep`
holds both answers to **one input** (its fingerprint is on the step and on both answers; two inputs are refused, so a
difference is a difference of approach), the :class:`Difference` of every field, the truth when it is known and the
:class:`Verdict` against it.

The step catalogue :data:`STEPS` says, for each step, its answer's fields and how each field is compared: ``equal``,
``set`` (what each arm alone gave), ``ranked`` (an order), ``numeric`` (within a tolerance), ``text`` (by key terms) and
``mapping`` (key by key). Every decision point of the ledger is a step too, ``decision:<point>``, with one field
``value``.

:func:`effectiveness` reads judged steps and gives, per step, each arm's accuracy with its interval, the agreement, the
cost and time, what one arm got right that the other missed, and the **mode** the evidence recommends, by
:data:`RULE` (``twin_rule_v1``), fixed before any run:

* ``undecided``: fewer than :data:`MIN_JUDGED` judged steps;
* ``software``: no approach using the AI is better than the system by :data:`MARGIN`, and the AI does not catch enough of
  the system's errors to review it;
* ``software_reviewed`` (software proposes, the AI reviews): the system is the better answerer, but where it is wrong the
  AI differs and is right at least :data:`REVIEW_RECALL` of the time, with at least :data:`REVIEW_PRECISION` of its
  disagreements being the system's errors (a person reads each disagreement);
* ``ai_checked`` (the AI proposes, the software checks): the AI's answer where the step's validator accepts it and the
  system's where it refuses is better than the system by the margin, and better than the AI alone;
* ``ai``: the AI alone is better than the system by the margin, and the validator adds nothing.

"The AI alone" shows the system's answer where the AI gave none (it failed, hit its cap or abstained), as a product
would; the margin is counted in whole judged steps. A disagreement counts only on a field the truth judges.

Ties go to the software: it is deterministic, free and explainable. Deterministic; no model, no store.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import median
from typing import Any, Literal, Optional

from pydantic import Field, JsonValue, field_validator, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Arm = Literal["system", "ai_first"]
ARMS: tuple[str, ...] = ("system", "ai_first")
Compare = Literal["equal", "set", "ranked", "numeric", "text", "mapping"]
Status = Literal["answered", "abstained", "failed", "capped", "skipped"]
TruthSource = Literal["planted", "person_pick", "measured", "label"]
Mode = Literal["software", "software_reviewed", "ai_checked", "ai", "undecided"]
Winner = Literal["system", "ai_first", "both", "neither", "unjudged"]
RULE = "twin_rule_v1"
MIN_JUDGED = 10
MARGIN = 0.10
REVIEW_RECALL = 0.5
REVIEW_PRECISION = 0.3
TEXT_MATCH = 0.5  # key-term overlap (Jaccard) at which two texts say the same
MAX_TEXT = 2000
DECISION = "decision:"
MAX_ITEMS = 60  # a list inside an answer, a truth or an input summary
MAX_STRING = 400  # a string inside one (a name, a code, a short phrase; never a row)
MAX_DEPTH = 3
FINGERPRINT = r"^[0-9a-f]{64}$"


@dataclass(frozen=True)
class StepSpec:
    title: str
    fields: Mapping[str, str]  # field -> comparison
    tolerance: float = 0.05  # for numeric fields


STEPS: dict[str, StepSpec] = {
    # Home: understanding the problem
    "home.understanding": StepSpec("What the problem is", {"outcome": "text", "moment": "text", "action": "text", "unit": "text",
                                                           "family": "equal", "labels": "equal", "target_column": "equal"}),
    "home.next_question": StepSpec("What to ask next", {"field": "equal", "options": "set"}),
    "home.pack": StepSpec("Which domain pack", {"pack": "equal"}),
    "home.map": StepSpec("The parts of the problem", {"size": "equal", "families": "set", "parts": "set"}),
    "home.route": StepSpec("The approach per part", {"routes": "mapping"}),
    "home.workflow": StepSpec("The solution workflow", {"template": "equal", "steps": "ranked"}),
    "home.ready": StepSpec("Is it ready to build", {"ready": "equal", "blockers": "set"}),
    # the solution's decisions about the table
    "solution.columns": StepSpec("The table's roles", {"target": "equal", "task": "equal", "identifiers": "set", "leakage": "set",
                                                       "time": "set", "group": "set"}),
    # the project's stages
    "stage.data": StepSpec("Split and metric", {"split": "equal", "metric": "equal"}),
    "stage.features": StepSpec("Feature recipe", {"recipe": "equal"}),
    "stage.models": StepSpec("Model family", {"family": "equal"}),
    "stage.next_experiment": StepSpec("Next experiment", {"experiment": "equal"}),
    "stage.threshold": StepSpec("Operating point", {"threshold": "numeric"}, tolerance=0.05),
    "report.summary": StepSpec("What the result says", {"summary": "text", "limits": "set"}),
}


def spec(step: str) -> StepSpec:
    """A step's spec; ``decision:<point>`` for any ledger point."""
    if step in STEPS:
        return STEPS[step]
    if step.startswith(DECISION):
        from dclab_core.ledger import POINTS
        point = step[len(DECISION):]
        if point in POINTS:
            return StepSpec(POINTS[point].about, {"value": "equal"})
    raise ValueError(f"unknown twin step {step!r}: one of {sorted(STEPS)} or 'decision:<ledger point>'")


def fingerprint(value: Any) -> str:
    """The input's fingerprint: SHA-256 of its canonical JSON (keys sorted)."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode()).hexdigest()


# ------------------------------------------------------------------------------------------------- the contract


def bounded(value: Any, depth: int = 0) -> list[str]:
    """Why a value cannot be kept in a twin record (empty when it can): nested at most MAX_DEPTH deep, lists of at most
    MAX_ITEMS, strings of at most MAX_STRING characters. A row of a table does not fit; names, codes and counts do."""
    if depth > MAX_DEPTH:
        return [f"nested more than {MAX_DEPTH} deep"]
    if isinstance(value, str):
        return [f"a string of {len(value)} characters (at most {MAX_STRING})"] if len(value) > MAX_STRING else []
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_ITEMS:
            return [f"a list of {len(value)} items (at most {MAX_ITEMS})"]
        return [p for v in value for p in bounded(v, depth + 1)]
    if isinstance(value, Mapping):
        if len(value) > MAX_ITEMS:
            return [f"a record of {len(value)} keys (at most {MAX_ITEMS})"]
        return [p for k, v in value.items() for p in bounded(str(k), depth + 1) + bounded(v, depth + 1)]
    return []


def _summary_problems(summary: Mapping[str, Any]) -> list[str]:
    """An input summary holds names and counts: scalars, or lists of short scalars; never a record (a row)."""
    out = []
    for key, value in summary.items():
        items = value if isinstance(value, (list, tuple)) else [value]
        if len(items) > MAX_ITEMS or any(isinstance(v, (Mapping, list, tuple)) for v in items):
            out.append(f"input_summary.{key}: names and counts only (scalars, or a list of at most {MAX_ITEMS} of them)")
        elif any(isinstance(v, str) and len(v) > 120 for v in items):
            out.append(f"input_summary.{key}: a string over 120 characters")
    return out


class ToolUse(Contract):
    name: str = Field(max_length=80)
    arguments: str = Field(default="", max_length=400, description="The call's arguments, as short text (names and codes; never a row).")
    seconds: float = Field(default=0.0, ge=0)
    ok: bool = True


class ArmAnswer(Contract):
    arm: Arm
    answerer: str = Field(max_length=120, description="Who answered: 'rules', a role with its prompt version, 'ai_first@v1', a model id.")
    status: Status = "answered"
    input_fingerprint: str = Field(default="", max_length=64)
    value: dict[str, JsonValue] = Field(default_factory=dict, description="The answer in the step's shape (STEPS).")
    said: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="What the arm said to the person, if anything.")
    reasons: list[str] = Field(default_factory=list, max_length=12)
    tools: list[ToolUse] = Field(default_factory=list, max_length=60)
    valid: Optional[bool] = Field(default=None, description="Whether the step's validator accepted this answer (None: not checked, or no validator).")
    refusals: list[str] = Field(default_factory=list, max_length=12)
    model: Optional[str] = Field(default=None, max_length=120)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cost_eur: float = Field(default=0.0, ge=0)
    seconds: float = Field(default=0.0, ge=0)
    note: Optional[str] = Field(default=None, max_length=400)
    dropped: dict[str, int] = Field(default_factory=dict, description="What was cut to fit: characters of 'said', reasons, tools, refusals.")

    @model_validator(mode="before")
    @classmethod
    def _fit(cls, data: Any) -> Any:
        """A long reply is cut, never refused: the record keeps what fits and says how much was dropped."""
        if not isinstance(data, dict):
            return data
        data, dropped = dict(data), dict(data.get("dropped") or {})
        for name, limit in (("said", MAX_TEXT), ("note", 400)):
            text = data.get(name)
            if isinstance(text, str) and len(text) > limit:
                dropped[name] = dropped.get(name, 0) + len(text) - limit
                data[name] = text[:limit]
        for name, limit in (("reasons", 12), ("refusals", 12), ("tools", 60)):
            items = data.get(name)
            if isinstance(items, list):
                if len(items) > limit:
                    dropped[name] = dropped.get(name, 0) + len(items) - limit
                    items = items[:limit]
                data[name] = [i[:400] if isinstance(i, str) else i for i in items]
        if dropped:
            data["dropped"] = dropped
        return data

    @field_validator("value")
    @classmethod
    def _bounded(cls, v: dict[str, Any]) -> dict[str, Any]:
        problems = bounded(v)
        if problems:
            raise ValueError("an answer holds names, codes and counts, never a row: " + "; ".join(problems[:3]))
        return v


class Difference(Contract):
    field: str = Field(max_length=80)
    compare: Compare
    kind: Literal["same", "differs", "only_system", "only_ai", "neither"]
    system: JsonValue = None
    ai_first: JsonValue = None
    only_system: list[JsonValue] = Field(default_factory=list, max_length=60)
    only_ai: list[JsonValue] = Field(default_factory=list, max_length=60)


class Truth(Contract):
    value: dict[str, JsonValue] = Field(default_factory=dict, description="The right answer for the fields it covers (a subset of the step's fields).")
    source: TruthSource
    note: Optional[str] = Field(default=None, max_length=400)

    @field_validator("value")
    @classmethod
    def _bounded(cls, v: dict[str, Any]) -> dict[str, Any]:
        problems = bounded(v)
        if problems:
            raise ValueError("a truth holds names, codes and counts, never a row: " + "; ".join(problems[:3]))
        return v


class Verdict(Contract):
    fields: list[str] = Field(default_factory=list, max_length=20, description="The fields judged: the truth's.")
    system: dict[str, bool] = Field(default_factory=dict)
    ai_first: dict[str, bool] = Field(default_factory=dict)
    winner: Winner = "unjudged"


class TwinStep(Versioned):
    """One step answered by both approaches on one input."""

    schema_version: Literal["dclab_core.twin_step.v1"] = _version("twin_step")
    id: str = Field(max_length=80)
    session: str = Field(max_length=80, description="The twin session (one per draft or project).")
    step: str = Field(max_length=80)
    subject: Optional[str] = Field(default=None, max_length=200, description="What the step is about where there are several (a part, a column, a stage).")
    at: str = Field(max_length=40)
    input_fingerprint: str = Field(pattern=FINGERPRINT, description="SHA-256 of the input both arms answered (fingerprint()).")
    input_summary: dict[str, JsonValue] = Field(default_factory=dict, description="What the input held, as names and counts (the host decides; never rows).")
    system: ArmAnswer
    ai_first: ArmAnswer
    differences: list[Difference] = Field(default_factory=list, max_length=40, description="Computed from the two answers; a stored one that differs is refused.")
    truth: Optional[Truth] = None
    verdict: Verdict = Field(default_factory=Verdict, description="Computed from the answers and the truth; a stored one that differs is refused.")

    @model_validator(mode="after")
    def _one_input(self) -> "TwinStep":
        fields = spec(self.step).fields
        problems = _summary_problems(self.input_summary)
        if problems:
            raise ValueError("; ".join(problems))
        for name in ARMS:
            arm: ArmAnswer = getattr(self, name)
            if arm.arm != name:
                raise ValueError(f"the {name} answer says it is the {arm.arm} arm")
            if arm.status in ("answered", "abstained") and arm.input_fingerprint != self.input_fingerprint:
                raise ValueError(f"the {name} arm answered another input: a twin step compares two approaches on one input")
            extra = sorted(set(arm.value) - set(fields))
            if extra:
                raise ValueError(f"the {name} arm's answer has fields the step {self.step!r} does not have: {extra}")
        if self.truth is not None:
            extra = sorted(set(self.truth.value) - set(fields))
            if extra:
                raise ValueError(f"the truth has fields the step {self.step!r} does not have: {extra}")
        if self.step.startswith(DECISION):  # a ledger point's answer has its point's shape (a flag, a choice, a short name)
            from dclab_core.ledger import POINTS, answer_problem

            point = POINTS[self.step[len(DECISION):]]
            for name, value in (("system", self.system.value.get("value")), ("ai_first", self.ai_first.value.get("value")),
                                ("truth", (self.truth.value if self.truth else {}).get("value"))):
                problem = answer_problem(point, value)
                if problem:
                    raise ValueError(f"the {name} answer at {point.key}: {problem}")
        expected_d = compare(self.step, self.system.value, self.ai_first.value)
        expected_v = judge(self.step, self.system, self.ai_first, self.truth)
        for name, expected in (("differences", expected_d), ("verdict", expected_v)):
            if name in self.model_fields_set and getattr(self, name) != expected:
                raise ValueError(f"the stored {name} do not follow from the answers and the truth: a twin step's {name} are computed, never written")
            object.__setattr__(self, name, expected)
            self.__pydantic_fields_set__.add(name)
        return self


# ------------------------------------------------------------------------------------------------- comparing

_STOP = frozenset("a an the of to in on for by at is are be it its this that and or with from as when what which who will would "
                  "should can each per their our we they them i you your my de".split())


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") and not word.endswith("ss") else word


def terms(text: Any) -> frozenset[str]:
    """Key terms of a text in any script (casefolded words, stop words and one-letter words out, a plural's s dropped)."""
    words = re.findall(r"\w+", str(text or "").casefold())
    return frozenset(_stem(w) for w in words if w not in _STOP and len(w) > 1)


def canon(value: Any) -> Any:
    """A value in its comparable form: strings stripped and casefolded, records with sorted keys, all the way down."""
    if isinstance(value, str):
        return value.strip().casefold()
    if isinstance(value, Mapping):
        return {str(k): canon(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [canon(v) for v in value]
    return value


def _key(value: Any) -> str:
    return json.dumps(canon(value), sort_keys=True, ensure_ascii=False, default=str)


def _items(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_key(v) for v in value]
    return [_key(value)]


def _equal(a: Any, b: Any) -> bool:
    if isinstance(a, bool) != isinstance(b, bool):  # True is not 1
        return False
    return _key(a) == _key(b)


def same(compare: str, a: Any, b: Any, tolerance: float = 0.05) -> bool:
    """Whether two values say the same under a comparison."""
    if a is None or b is None:
        return a is None and b is None
    if compare == "equal":
        return _equal(a, b)
    if compare == "set":
        return set(_items(a)) == set(_items(b))
    if compare == "ranked":
        return _items(a) == _items(b)
    if compare == "numeric":
        if isinstance(a, bool) or isinstance(b, bool):
            return a is b
        try:
            return abs(float(a) - float(b)) <= tolerance + 1e-12
        except (TypeError, ValueError):
            return False
    if compare == "text":
        ta, tb = terms(a), terms(b)
        if not ta or not tb:  # nothing to weigh: the texts must be the same
            return canon(str(a)) == canon(str(b))
        return len(ta & tb) / len(ta | tb) >= TEXT_MATCH
    if compare == "mapping":
        if not isinstance(a, Mapping) or not isinstance(b, Mapping):
            return False
        return _key(dict(a)) == _key(dict(b))
    raise ValueError(f"unknown comparison {compare!r}")


def right(compare: str, answer: Any, truth: Any, tolerance: float = 0.05) -> bool:
    """Whether an answer is right against the truth: as ``same``, except a text, which is right when it contains enough of
    the truth's key terms (a longer, correct answer is not wrong for saying more)."""
    if compare == "text" and answer is not None and truth is not None:
        tt, ta = terms(truth), terms(answer)
        if tt and ta:
            return len(tt & ta) / len(tt) >= TEXT_MATCH
    return same(compare, answer, truth, tolerance)


def difference(field: str, compare: str, system: Any, ai: Any, tolerance: float = 0.05) -> Difference:
    if system is None and ai is None:
        return Difference(field=field, compare=compare, kind="neither")
    if system is None or ai is None:
        return Difference(field=field, compare=compare, kind="only_ai" if system is None else "only_system", system=system, ai_first=ai)
    if same(compare, system, ai, tolerance):
        return Difference(field=field, compare=compare, kind="same", system=system, ai_first=ai)
    only_s: list[Any] = []
    only_a: list[Any] = []
    if compare in ("set", "ranked"):
        rs, ra = list(system) if isinstance(system, (list, tuple)) else [system], list(ai) if isinstance(ai, (list, tuple)) else [ai]
        ks, ka = set(_items(ra)), set(_items(rs))
        only_s, only_a = [x for x in rs if _key(x) not in ks], [x for x in ra if _key(x) not in ka]
    elif compare == "mapping" and isinstance(system, Mapping) and isinstance(ai, Mapping):
        cs, ca = {str(k): _key(v) for k, v in system.items()}, {str(k): _key(v) for k, v in ai.items()}
        keys = sorted(set(cs) | set(ca))
        only_s = [k for k in keys if k in cs and cs[k] != ca.get(k)]
        only_a = [k for k in keys if k in ca and ca[k] != cs.get(k)]
    elif compare == "text":
        ts, ta = terms(system), terms(ai)
        only_s, only_a = sorted(ts - ta)[:30], sorted(ta - ts)[:30]
    return Difference(field=field, compare=compare, kind="differs", system=system, ai_first=ai, only_system=only_s[:60], only_ai=only_a[:60])


def compare(step: str, system: Mapping[str, Any], ai: Mapping[str, Any]) -> list[Difference]:
    """Every field of the step, compared."""
    s = spec(step)
    return [difference(f, c, system.get(f), ai.get(f), s.tolerance) for f, c in s.fields.items()]


def judge(step: str, system: ArmAnswer, ai: ArmAnswer, truth: Optional[Truth]) -> Verdict:
    """Each arm against the truth, field by field; an arm that did not answer is wrong on every field."""
    if truth is None or not truth.value:
        return Verdict()
    s = spec(step)
    fields = [f for f in s.fields if f in truth.value]
    out: dict[str, dict[str, bool]] = {}
    for arm in (system, ai):
        answered = arm.status == "answered"
        out[arm.arm] = {f: answered and right(s.fields[f], arm.value.get(f), truth.value[f], s.tolerance) for f in fields}
    rs, ra = all(out["system"].values()), all(out["ai_first"].values())
    winner: Winner = "both" if rs and ra else "system" if rs else "ai_first" if ra else "neither"
    return Verdict(fields=fields, system=out["system"], ai_first=out["ai_first"], winner=winner)


def step(id: str, session: str, step_name: str, at: str, input: Any, system: ArmAnswer, ai: ArmAnswer, *,
         subject: Optional[str] = None, input_summary: Optional[Mapping[str, Any]] = None, truth: Optional[Truth] = None) -> TwinStep:
    """A twin step from both answers: the fingerprint of the input stamped on the step and on both answers (an answer that
    carries none is taken to have answered this input: the host vouches for it), the differences and the verdict computed."""
    fp = fingerprint(input)
    if system.status in ("answered", "abstained") and system.input_fingerprint and system.input_fingerprint != fp:
        raise ValueError("the system arm answered another input")
    if ai.status in ("answered", "abstained") and ai.input_fingerprint and ai.input_fingerprint != fp:
        raise ValueError("the ai_first arm answered another input")
    system = system.model_copy(update={"input_fingerprint": fp})
    ai = ai.model_copy(update={"input_fingerprint": fp})
    return TwinStep(id=id, session=session, step=step_name, subject=subject, at=at, input_fingerprint=fp,
                    input_summary=dict(input_summary or {}), system=system, ai_first=ai, truth=truth)


def with_truth(record: TwinStep, truth: Truth) -> TwinStep:
    """The step with a truth and its verdict (a new step; the old one is not edited)."""
    data = record.model_dump()
    data.pop("verdict")
    data.update(truth=truth.model_dump())
    return TwinStep.model_validate(data)


# ------------------------------------------------------------------------------------------------- where each is best


def wilson(right: int, n: int, z: float = 1.96) -> list[Optional[float]]:
    if not n:
        return [None, None]
    p = right / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


class StepEffect(Contract):
    step: str = Field(max_length=80)
    steps: int = Field(ge=0, description="Twin steps recorded.")
    judged: int = Field(ge=0, description="Steps with a truth.")
    agreement: Optional[float] = Field(default=None, description="Share of steps where both arms answered and every field was the same.")
    system_right: int = 0
    ai_right: int = Field(default=0, description="Right with the AI's answer, and the system's where the AI gave none (failed, capped, abstained).")
    checked_right: int = Field(default=0, description="Right when the AI's answer is used where the validator accepts it and the system's elsewhere.")
    system_accuracy: Optional[float] = None
    ai_accuracy: Optional[float] = None
    checked_accuracy: Optional[float] = None
    system_interval: list[Optional[float]] = Field(default_factory=lambda: [None, None])
    ai_interval: list[Optional[float]] = Field(default_factory=lambda: [None, None])
    ai_only_right: int = Field(default=0, description="The system wrong, the AI right.")
    system_only_right: int = Field(default=0, description="The AI wrong, the system right.")
    review_recall: Optional[float] = Field(default=None, description="Of the system's errors, the share where the AI differed and was right.")
    review_precision: Optional[float] = Field(default=None, description="Of the disagreements, the share where the system was wrong.")
    ai_failed: int = Field(default=0, description="Steps where the AI arm failed, abstained or hit its cap.")
    cost_eur: dict[str, Optional[float]] = Field(default_factory=dict, description="Median cost per step, per arm.")
    seconds: dict[str, Optional[float]] = Field(default_factory=dict, description="Median time per step, per arm.")
    fields: dict[str, dict[str, int]] = Field(default_factory=dict, description="Per field: judged, the system right, the AI right.")
    mode: Mode = "undecided"
    reasons: list[str] = Field(default_factory=list, max_length=8)


class TwinMatrix(Versioned):
    """Where the AI, the software, or the two together are best, step by step."""

    schema_version: Literal["dclab_core.twin_matrix.v1"] = _version("twin_matrix")
    rule: Literal["twin_rule_v1"] = RULE
    source: str = Field(max_length=200, description="What the steps were: a measurement on planted truth, live sessions, both.")
    steps: list[StepEffect] = Field(default_factory=list, max_length=200)
    totals: dict[str, float] = Field(default_factory=dict)


def _all_same(record: TwinStep) -> bool:
    return all(d.kind in ("same", "neither") for d in record.differences)


def _med(values: Sequence[float]) -> Optional[float]:
    return round(median(values), 6) if values else None


def mode_of(effect: Mapping[str, Any]) -> tuple[str, list[str]]:
    """``twin_rule_v1``: the recommended mode from a step's counts. ``ai_right`` counts the AI with the system's answer
    where the AI gave none (a failure, a cap, an abstention: what a product would show), ``checked_right`` also where
    the step's validator refused the AI's answer. The margin is compared in whole steps, so the rule does not move
    with the base rate."""
    n = int(effect["judged"])
    if n < MIN_JUDGED:
        return "undecided", [f"{n} judged steps; the rule needs {MIN_JUDGED}"]
    sr, ar, cr = int(effect["system_right"]), int(effect["ai_right"]), int(effect["checked_right"])
    need = MARGIN * n - 1e-9
    s, a, c = sr / n, ar / n, cr / n
    if max(ar, cr) - sr >= need:
        if cr > ar:
            return "ai_checked", [f"the AI where the validator accepts it, the system elsewhere, is right {c:.0%} against the system's {s:.0%} and the AI's {a:.0%}"]
        return "ai", [f"the AI is right {a:.0%} against the system's {s:.0%}; the validator adds nothing"]
    errors = n - sr
    recall, precision = effect.get("review_recall"), effect.get("review_precision")
    if errors >= 3 and recall is not None and precision is not None and recall >= REVIEW_RECALL and precision >= REVIEW_PRECISION:
        return "software_reviewed", [f"the system is right {s:.0%}; where it is wrong the AI catches {recall:.0%}, and {precision:.0%} of its disagreements are real"]
    return "software", [f"the system is right {s:.0%}, the AI {a:.0%}, the AI checked {c:.0%}: no approach with the AI is better by {MARGIN:.0%}"]


def _judged_differ(record: TwinStep) -> bool:
    """Whether the two arms differ on a field the truth judges (a difference elsewhere is not a disagreement about it)."""
    return any(d.kind not in ("same", "neither") for d in record.differences if d.field in record.verdict.fields)


def effect(step_name: str, records: Sequence[TwinStep]) -> StepEffect:
    judged = [r for r in records if r.verdict.winner != "unjudged"]
    both = [r for r in records if r.system.status == "answered" and r.ai_first.status == "answered"]
    sys_right = sum(r.verdict.winner in ("system", "both") for r in judged)
    ai_right = checked = 0
    for r in judged:
        s_ok, a_ok = r.verdict.winner in ("system", "both"), r.verdict.winner in ("ai_first", "both")
        answered = r.ai_first.status == "answered"
        ai_right += a_ok if answered else s_ok  # no answer from the AI: the system's is shown
        checked += a_ok if answered and r.ai_first.valid is not False else s_ok
    sys_wrong = [r for r in judged if r.verdict.winner in ("ai_first", "neither")]
    disagree = [r for r in judged if r.ai_first.status == "answered" and _judged_differ(r)]
    caught = sum(r.verdict.winner == "ai_first" for r in sys_wrong)
    real = sum(r.verdict.winner in ("ai_first", "neither") for r in disagree)
    fields: dict[str, dict[str, int]] = {}
    for r in judged:
        for f in r.verdict.fields:
            row = fields.setdefault(f, {"judged": 0, "system": 0, "ai_first": 0})
            row["judged"] += 1
            row["system"] += int(r.verdict.system.get(f, False))
            row["ai_first"] += int(r.verdict.ai_first.get(f, False))
    n = len(judged)
    counts = dict(judged=n, system_right=sys_right, ai_right=ai_right, checked_right=checked,
                  review_recall=round(caught / len(sys_wrong), 4) if sys_wrong else None,
                  review_precision=round(real / len(disagree), 4) if disagree else None)
    mode, reasons = mode_of(counts)
    spent = {a: [getattr(r, a).cost_eur for r in records if getattr(r, a).status != "skipped"] for a in ARMS}  # a failed arm's spend counts
    return StepEffect(step=step_name, steps=len(records), agreement=round(sum(_all_same(r) for r in both) / len(both), 4) if both else None,
                      system_accuracy=round(sys_right / n, 4) if n else None, ai_accuracy=round(ai_right / n, 4) if n else None,
                      checked_accuracy=round(checked / n, 4) if n else None, system_interval=wilson(sys_right, n), ai_interval=wilson(ai_right, n),
                      ai_only_right=sum(r.verdict.winner == "ai_first" for r in judged), system_only_right=sum(r.verdict.winner == "system" for r in judged),
                      ai_failed=sum(r.ai_first.status in ("failed", "abstained", "capped") for r in records),
                      cost_eur={a: _med(spent[a]) for a in ARMS},
                      seconds={a: _med([getattr(r, a).seconds for r in records if getattr(r, a).status == "answered"]) for a in ARMS},
                      fields=fields, mode=mode, reasons=reasons,
                      **{k: v for k, v in counts.items() if k != "judged"}, judged=n)


def effectiveness(records: Iterable[TwinStep], source: str) -> TwinMatrix:
    """The matrix: one row per step (``decision:<point>`` steps each their own), sorted by the step catalogue's order."""
    by: dict[str, list[TwinStep]] = {}
    for r in records:
        by.setdefault(r.step, []).append(r)
    order = {k: i for i, k in enumerate(STEPS)}
    rows = [effect(k, v) for k, v in sorted(by.items(), key=lambda kv: (order.get(kv[0], len(order)), kv[0]))]
    all_records = [r for v in by.values() for r in v]
    totals = {"steps": float(len(all_records)), "judged": float(sum(r.judged for r in rows)),
              **{f"cost_eur_{a}": round(sum(getattr(r, a).cost_eur for r in all_records), 6) for a in ARMS}}
    return TwinMatrix(source=source[:200], steps=rows, totals=totals)


CONTRACTS["twin_step"] = TwinStep
CONTRACTS["twin_matrix"] = TwinMatrix

__all__ = ["ARMS", "STEPS", "StepSpec", "spec", "fingerprint", "ToolUse", "ArmAnswer", "Difference", "Truth", "Verdict", "TwinStep",
           "terms", "same", "difference", "compare", "judge", "step", "with_truth", "StepEffect", "TwinMatrix", "mode_of", "effect",
           "effectiveness", "RULE", "MIN_JUDGED", "MARGIN", "REVIEW_RECALL", "REVIEW_PRECISION"]
