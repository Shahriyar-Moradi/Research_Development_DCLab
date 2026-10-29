"""The routing data of the 7th model (P8 of the Home plan, 2026-10-29): one example per routing decision Home makes, as a
routing model learns it (a fine-tuned GLiNER, or a small classifier first).

A :class:`RoutingExample` holds what was known when the route was chosen and what came of it:

- the person's words (``text``), only where the workspace allows training on them; a row without them says why
  (``no_text``: no consent, an exam's request kept as its shape only, the draft not in the store);
- the decision's state as the ledger keeps it (bands and codes, never a table's value);
- every answerer's reading (:class:`RoutingReading`): the rules', the model's with its API thinking summary (the thinking
  blocks the API returned with display "summarized", never reasoning asked for in the reply's text), and our own model's
  answer in shadow (never used);
- the route taken (``value_used``, ``used_by``) and the label with its class (``label_source``), kept apart:

  * ``real``: a person on a real request (their confirm or change on the card, a result kept or changed, a measured label);
  * ``written``: a truth assigned by code to a written request (the written requests, an eval's gold, a scripted person);
  * ``silver``: a model's reading (Claude's) taken as the label of a real request that has no confirmed one, weighted lower.

``model_type`` is ``m7_routing``. It is not one of the eight model types of the other example contracts
(:data:`dclab_core.store.ModelType`): naming it there would change every one of their shapes; this contract is its own.
Pure pydantic: the host joins the records.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.store import Provenance, Split

MODEL_TYPE = "m7_routing"
RoutingLabelSource = Literal["real", "written", "silver"]
LABEL_SOURCES: tuple[str, ...] = ("real", "written", "silver")
WEIGHTS: dict[str, float] = {"real": 1.0, "written": 0.5, "silver": 0.5}  # written and silver: apart and lower (the owner's rule)
Origin = Literal["real", "scripted", "written", "eval"]  # where the request came from: a person, a script, a writer, an eval run
ReadingBy = Literal["rule", "model", "own_model", "person"]
TRACK_MEANINGS: dict[str, str] = {  # the kinds of help Home reads (draft/intent.TRACKS), in the words of its prompt
    "prediction": "a model that predicts something not yet known, for each new case",
    "analysis": "numbers about what already happened in their data, including finding or describing groups",
    "preparation": "cleaning, combining or reshaping data",
    "reporting": "a report or dashboard kept up to date",
    "business": "a business question or decision: why something changed, what to do",
    "advice": "an explanation, no data work",
}
_LANG = re.compile(r"[؀-ۿ]")


class RoutingReading(Contract):
    """One answerer's reading of the request."""

    by: ReadingBy
    answerer: str = Field(max_length=120)         # rule:<version>, a model purpose (home_intent), classifier:<name>, the person's role
    value: Any = None                              # its route (None: it abstained or failed)
    confidence: Optional[float] = Field(default=None, ge=0, le=1)
    used: bool = False                             # this answerer's route was the one taken
    shadow: bool = False                           # answered beside the others, never used (our own model until its gate)
    model: Optional[str] = Field(default=None, max_length=120)
    request: Optional[str] = Field(default=None, max_length=120)  # the model request whose exchange it is
    thinking: Optional[str] = Field(default=None, max_length=20000)  # the API thinking summary, as returned
    thinking_source: Optional[Literal["recorded"]] = None
    no_thinking: Optional[str] = Field(default=None, max_length=300)  # a model's reading without its summary: why

    @model_validator(mode="after")
    def _thinking_labelled(self):
        if (self.thinking is None) != (self.thinking_source is None):
            raise ValueError("a thinking summary and its source come together")
        if self.thinking is not None and self.by != "model":
            raise ValueError("only a model's reading carries a thinking summary")
        if self.by == "model" and self.thinking is None and not self.no_thinking:
            raise ValueError("a model's reading has its thinking summary, or says why not (no_thinking)")
        return self


class RoutingExample(Versioned):
    """One routing decision (v1), as the 7th model learns it."""

    schema_version: Literal["dclab_core.routing_example.v1"] = _version("routing_example")
    id: str = Field(max_length=200)
    model_type: Literal["m7_routing"] = MODEL_TYPE
    split: Split
    provenance: Provenance
    more: list[Provenance] = Field(default_factory=list)  # the draft's and the exchange's entries it was joined from
    weight: float = Field(default=1.0, ge=0, le=1)
    point: str = Field(max_length=120)             # the routing point (request.intent, request.exit, workflow.next_question ...)
    origin: Origin
    split_group: str = Field(max_length=200)       # one request's words in one split everywhere (request:<hash>), else its draft
    text: Optional[str] = Field(default=None, max_length=8000)  # the person's words that were read
    no_text: Optional[str] = Field(default=None, max_length=300)
    language: Optional[Literal["en", "fa", "mixed"]] = None
    state: dict[str, Any] = Field(default_factory=dict)  # the ledger's state: bands, codes, counts, flags; never a value
    options: list[str] = Field(default_factory=list)
    readings: list[RoutingReading] = Field(default_factory=list)
    value_used: Any = None                         # the route taken
    used_by: Optional[str] = Field(default=None, max_length=40)
    outcome: Optional[str] = Field(default=None, max_length=40)  # what the person did with the result (kept, changed, refused ...)
    label: Any = None
    label_source: Optional[RoutingLabelSource] = None
    label_by: Optional[str] = Field(default=None, max_length=160)  # who gave it: person:card, person:outcome, written:<set>, model:<purpose>
    exam: Optional[str] = Field(default=None, max_length=120)      # the exam whose request this is (its split decided by it)

    @field_validator("state")
    @classmethod
    def _bands(cls, value: dict[str, Any]) -> dict[str, Any]:
        from dclab_core.ledger import check_state

        problems = check_state(value)
        if problems:
            raise ValueError("; ".join(problems))
        return value

    @model_validator(mode="after")
    def _said(self):
        if (self.text is None) == (self.no_text is None):
            raise ValueError("a row has the person's words, or says why not (no_text), never both")
        if self.label is not None and self.label_source is None:
            raise ValueError("a label says its source (real, written or silver)")
        if self.label_source is not None and self.label is None:
            raise ValueError("a label source without a label")
        if self.label_source in ("written", "silver") and self.weight > WEIGHTS[self.label_source]:
            raise ValueError(f"a {self.label_source} label weighs at most {WEIGHTS[self.label_source]}")
        return self


def language(text: Optional[str]) -> Optional[str]:
    """``fa`` (mostly Persian script), ``mixed`` (Persian and Latin words), ``en`` (Latin only), None (no text)."""
    if not text:
        return None
    persian = len(_LANG.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if persian and latin >= 6 and latin > persian * 0.25:
        return "mixed"
    return "fa" if persian else "en" if latin else None


def label_class(source: Optional[str], *, scripted: bool) -> Optional[str]:
    """A ledger label's class: a person's, a measurement's or a blind draw's is ``real`` (``written`` where a script plays the
    person: S8, a script's click is no human's); a benchmark's truth is ``written``; a teacher's is ``silver``; a rule's is
    no label (None)."""
    if source in ("user_decision_exposed", "person", "measured", "blind"):
        return "written" if scripted else "real"
    if source == "benchmark":
        return "written"
    if source in ("model", "teacher"):
        return "silver"
    return None


CONTRACTS["routing_example"] = RoutingExample

__all__ = ["RoutingExample", "RoutingReading", "MODEL_TYPE", "LABEL_SOURCES", "WEIGHTS", "TRACK_MEANINGS", "language", "label_class"]
