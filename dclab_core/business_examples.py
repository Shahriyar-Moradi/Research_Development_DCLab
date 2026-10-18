"""Business examples (package 30.5): real problems written as problem → understanding → data → solution → workflow, each
with its pitfalls and its evidence, for the understanding team (what to ask, what is often missed) and the solution team
(which approach, which split, which trap).

``BusinessExample`` v1 is one example; ``search(examples, words, top)`` finds the ones closest to a person's words
(token overlap weighted by rarity: no model, the same on every host). An example states where it comes from
(``source.kind``): a research track's contract and champion, a campaign's measured experiments, a replica's planted traps,
an owner's own solution, or ``written`` (composed from public knowledge, marked so, never a record of a run). Its
results cite their source; a written example has none.

v2 (30.7): ``owner_document`` (a pattern the owner wrote from a real conversation), the pattern itself (the general
problem, why it is hard, how to solve and measure it, the common mistakes, the same problem in other industries), the
practice questions with what a strong answer names, the recipe it maps to, and its ``visibility``: ``model`` (a model may
read its general pattern) or ``rules_only`` (read by the search and shown to the owner, never put in a model's context:
the owner's private solutions). ``for_model`` is the one filter every model reader goes through.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, ClassVar, Iterable, Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

TEXT = 600


class Understanding(Contract):
    unit: Optional[str] = Field(default=None, max_length=TEXT, description="One row: what is predicted for.")
    target: Optional[str] = Field(default=None, max_length=TEXT)
    moment: Optional[str] = Field(default=None, max_length=TEXT, description="When the prediction is made, and what is known then.")
    labels: Optional[str] = Field(default=None, max_length=TEXT, description="How the outcome is known: every row, a sample of cases, none.")
    action: Optional[str] = Field(default=None, max_length=TEXT, description="What is done with a prediction, and what a mistake costs.")
    family: Optional[str] = Field(default=None, max_length=80)


class ExampleSource(Contract):
    kind: Literal["research_track", "campaign", "replica", "owner_solution", "owner_document", "written"]
    ref: str = Field(min_length=1, max_length=TEXT, description="Where it is: a path, a record id, or the public knowledge it rests on.")


class Result(Contract):
    metric: str = Field(max_length=80)
    value: float
    note: str = Field(default="", max_length=TEXT, description="The protocol it was measured on, and its limits.")
    source: str = Field(max_length=TEXT, description="The record the number comes from.")


class Elsewhere(Contract):
    industry: str = Field(min_length=1, max_length=80)
    case: str = Field(min_length=1, max_length=TEXT)


class Pattern(Contract):
    """v2: the problem as a general pattern (the owner's problem cards: what was said, the general problem, why it is hard,
    how to solve it, how to measure it, the common mistakes, the same problem elsewhere)."""

    said: Optional[str] = Field(default=None, max_length=2000, description="What was said at the source, without anyone's name.")
    general_problem: Optional[str] = Field(default=None, max_length=1000, description="The same challenge with the industry's details removed.")
    why_hard: list[str] = Field(default_factory=list, max_length=12)
    how_to_solve: list[str] = Field(default_factory=list, max_length=12)
    how_to_measure: list[str] = Field(default_factory=list, max_length=12)
    mistakes: list[str] = Field(default_factory=list, max_length=12)
    elsewhere: list[Elsewhere] = Field(default_factory=list, max_length=12)


class Practice(Contract):
    """v2: a practice question and what a strong answer names (the metric, the split, the first baseline...)."""

    question: str = Field(min_length=1, max_length=1000)
    names: list[str] = Field(default_factory=list, max_length=12)


class RecipeLink(Contract):
    """v2: the task kind and modifiers the problem maps to, and whether DCLab can solve it today."""

    kind: Optional[str] = Field(default=None, max_length=40)
    modifiers: list[str] = Field(default_factory=list, max_length=12)
    capability: Literal["built", "planned", "knowledge"] = "knowledge"
    package: Optional[str] = Field(default=None, max_length=20, description="The plan's package that builds it, when planned.")


class BusinessExample(Versioned):
    """One real problem, as a careful data scientist would write it up for the next one."""

    schema_version: Literal["dclab_core.business_example.v2"] = _version("business_example", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.business_example.v1"})  # v2 only added fields
    id: str = Field(min_length=1, max_length=80)
    title: str = Field(min_length=1, max_length=200)
    industry: str = Field(default="", max_length=80)
    problem: str = Field(min_length=1, max_length=2000, description="The problem in a person's words.")
    understanding: Understanding = Field(default_factory=Understanding)
    data: list[str] = Field(default_factory=list, max_length=40, description="The tables and their important columns, in words (never a value).")
    pitfalls: list[str] = Field(default_factory=list, max_length=20, description="What goes wrong here: leaks, wrong splits, wrong labels, the trap.")
    solution: list[str] = Field(default_factory=list, max_length=20, description="The approach, the split, the metric, the model family, why.")
    workflow: list[str] = Field(default_factory=list, max_length=20, description="The steps, in order.")
    results: list[Result] = Field(default_factory=list, max_length=12)
    source: ExampleSource
    tags: list[str] = Field(default_factory=list, max_length=20)
    visibility: Literal["model", "rules_only"] = Field(default="model", description="v2: rules_only is never put in a model's context.")
    pattern: Optional[Pattern] = None
    practice: list[Practice] = Field(default_factory=list, max_length=20)
    recipe: Optional[RecipeLink] = None
    replicas: list[str] = Field(default_factory=list, max_length=12, description="The replicas that plant its traps.")

    @model_validator(mode="before")
    @classmethod
    def _read_v1(cls, data: Any) -> Any:
        """An example as stored; a v1 example is read as v2 (v2 only added fields): visible to a model, except an owner's
        solution, which v1 stored before visibility existed and which is read by the rules only (the review of 30.7)."""
        if isinstance(data, dict) and data.get("schema_version") == "dclab_core.business_example.v1":
            owner = (data.get("source") or {}).get("kind") == "owner_solution"
            data = {**data, "schema_version": "dclab_core.business_example.v2", **({"visibility": "rules_only"} if owner else {})}
        return data

    @model_validator(mode="after")
    def _owner_is_private(self) -> "BusinessExample":
        if self.source.kind == "owner_solution" and self.visibility == "model":
            raise ValueError("an owner's solution is read by the rules only: visibility must be rules_only")
        return self


CONTRACTS["business_example"] = BusinessExample

_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and the of to in on for with by is are be we our it its at as or from that this which what when who how do does one each".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text).lower()) if w not in STOP and len(w) > 1]


def _doc(e: BusinessExample) -> list[str]:
    u, p = e.understanding, e.pattern
    pattern = [p.said or "", p.general_problem or "", *p.why_hard, *p.how_to_solve, *p.how_to_measure, *p.mistakes,
               *(f"{x.industry} {x.case}" for x in p.elsewhere)] if p else []
    # the practice questions are not indexed: they are the evaluation set of the search (30.8), and a question in its own
    # example's index would find that example by construction (the review of 30.7)
    return tokens(" ".join([e.title, e.industry, e.problem, *(x or "" for x in (u.unit, u.target, u.moment, u.labels, u.action, u.family)),
                            *e.data, *e.pitfalls, *e.solution, *e.tags, *pattern]))


def search(examples: Iterable[BusinessExample], words: str, top: int = 3) -> list[tuple[BusinessExample, float]]:
    """The examples closest to ``words``: each shared word weighs by its rarity among the examples (BM25's idea, without
    a model); the best ``top`` with their scores, best first; nothing when no word is shared."""
    examples = list(examples)
    docs = [Counter(_doc(e)) for e in examples]
    n = len(docs) or 1
    df: Counter = Counter()
    for d in docs:
        df.update(set(d))
    avg = sum(sum(d.values()) for d in docs) / n if docs else 1.0
    query = set(tokens(words))
    scored = []
    for e, d in zip(examples, docs):
        length = sum(d.values()) or 1
        score = 0.0
        for w in query:
            if w in d:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                tf = d[w]
                score += idf * tf * 2.2 / (tf + 1.2 * (0.25 + 0.75 * length / avg))
        if score > 0:
            scored.append((e, round(score, 4)))
    return sorted(scored, key=lambda x: -x[1])[:top]


def brief(e: BusinessExample) -> dict[str, Any]:
    """An example as a role reads it: its understanding, its pitfalls, its solution and workflow, the general pattern and
    the recipe it maps to, and what kind of source it is (never where it is kept)."""
    out = {"id": e.id, "title": e.title, "understanding": e.understanding.model_dump(exclude_none=True), "pitfalls": list(e.pitfalls),
           "solution": list(e.solution), "workflow": list(e.workflow), "source": e.source.kind}
    if e.pattern is not None:
        out["pattern"] = e.pattern.model_dump(exclude={"said"}, exclude_none=True)  # the general problem, not the conversation
    if e.recipe is not None:
        out["recipe"] = e.recipe.model_dump(exclude_none=True)
    return out


def practice_recall(examples: Iterable[BusinessExample], top: int = 3) -> dict[str, Any]:
    """30.8: for every practice question, whether the search (which never indexes the questions) ranks its own example in the
    first ``top`` among all the examples: the knowledge base's evaluation."""
    examples = list(examples)
    rows = []
    for e in examples:
        for q in e.practice:
            ranked = [x.id for x, _ in search(examples, q.question, top)]
            rows.append({"pattern": e.id, "question": q.question, "found": e.id in ranked,
                         "rank": ranked.index(e.id) + 1 if e.id in ranked else None, "first": ranked[0] if ranked else None})
    return {"questions": len(rows), "found": sum(r["found"] for r in rows), "top": top, "rows": rows}


def for_model(examples: Iterable[BusinessExample]) -> list[BusinessExample]:
    """The examples a model may read: never a ``rules_only`` one (30.7). Every model reader goes through this."""
    return [e for e in examples if e.visibility == "model"]


__all__ = ["Understanding", "ExampleSource", "Result", "Elsewhere", "Pattern", "Practice", "RecipeLink", "BusinessExample",
           "tokens", "search", "brief", "for_model", "practice_recall"]
