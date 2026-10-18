"""Business examples (package 30.5): real problems written as problem → understanding → data → solution → workflow, each
with its pitfalls and its evidence, for the understanding team (what to ask, what is often missed) and the solution team
(which approach, which split, which trap).

``BusinessExample`` v1 is one example; ``search(examples, words, top)`` finds the ones closest to a person's words
(token overlap weighted by rarity: no model, the same on every host). An example states where it comes from
(``source.kind``): a research track's contract and champion, a campaign's measured experiments, a replica's planted traps,
an owner's own solution, or ``written`` (composed from public knowledge, marked so, never a record of a run). Its
results cite their source; a written example has none.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Iterable, Literal, Optional

from pydantic import Field

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
    kind: Literal["research_track", "campaign", "replica", "owner_solution", "written"]
    ref: str = Field(min_length=1, max_length=TEXT, description="Where it is: a path, a record id, or the public knowledge it rests on.")


class Result(Contract):
    metric: str = Field(max_length=80)
    value: float
    note: str = Field(default="", max_length=TEXT, description="The protocol it was measured on, and its limits.")
    source: str = Field(max_length=TEXT, description="The record the number comes from.")


class BusinessExample(Versioned):
    """One real problem, as a careful data scientist would write it up for the next one."""

    schema_version: Literal["dclab_core.business_example.v1"] = _version("business_example")
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


CONTRACTS["business_example"] = BusinessExample

_WORD = re.compile(r"[a-z0-9]+")
STOP = frozenset("a an and the of to in on for with by is are be we our it its at as or from that this which what when who how do does one each".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(str(text).lower()) if w not in STOP and len(w) > 1]


def _doc(e: BusinessExample) -> list[str]:
    u = e.understanding
    return tokens(" ".join([e.title, e.industry, e.problem, *(x or "" for x in (u.unit, u.target, u.moment, u.labels, u.action, u.family)),
                            *e.data, *e.pitfalls, *e.solution, *e.tags]))


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
    """An example as a role reads it: its understanding, its pitfalls, its solution and workflow, where it comes from."""
    return {"id": e.id, "title": e.title, "understanding": e.understanding.model_dump(exclude_none=True), "pitfalls": list(e.pitfalls),
            "solution": list(e.solution), "workflow": list(e.workflow), "source": e.source.kind}


__all__ = ["Understanding", "ExampleSource", "Result", "BusinessExample", "tokens", "search", "brief"]
