"""The target a person confirmed, and what a table's label column records (T1 of the team workspace, 2026-11-23).

"Can the data answer this?" is only a true yes when the outcome the person means is the outcome the table records. A
renewal table records whether an account renewed by its contractual date. When Business confirms that only an explicit
cancellation counts, that table cannot answer: it has no cancellation record. This module holds the two sides and the
rule that compares them. Pure pydantic and Python: the host stores the records and asks the person.

- :class:`TargetContract`: what was confirmed: the event (a code) with the person's own words, the entity, the
  prediction time, the horizon, where the label comes from, when it is mature, on which brief revision, and by whom.
- :class:`LabelDeclaration`: which event a table's column records, who declared it, and whether its outcomes are known.
- :func:`read_event`: the person's words as an event code, by a keyword rule that reads negation ("explicit
  cancellation only, not non-renewal"). Words that name two events, none, or one both ways return ``None``: the question
  stays open. It never guesses. (That ``None`` is the seat where a host may let a model read the words; whatever the
  model answers is used only as an event code that :func:`agreement` then checks.)
- :func:`agreement`: whether a confirmed target is recorded by a mature label, or exactly what is missing.

Contracts ``dclab_core.target_contract.v1`` and ``dclab_core.label_declaration.v1``. Codes, column names and the
person's short words: never a table's value.
"""

from __future__ import annotations

import re
from typing import Literal, Optional, Sequence, Union

from pydantic import Field, field_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

CONTRACT = "dclab_core.target_contract.v1"
CODE = r"^[a-z][a-z0-9_]{0,59}$"
REF = r"^[A-Za-z0-9][A-Za-z0-9_.:@/-]{0,119}$"
TEXT = 2000
SHORT = 300


class Event(Contract):
    """One event an outcome can mean: its code, how it is said to people, and the words that name it."""

    code: str = Field(pattern=CODE)
    label: str = Field(min_length=1, max_length=SHORT)
    names: str = Field(min_length=1, max_length=TEXT, description="A regular expression of the phrases that name the event.")


# The events this release reads. A phrase that names an event carries its own "not" where the event is a thing not
# happening ("do not renew" names non-renewal), so a negation is read only from the words around a phrase.
EVENTS: dict[str, Event] = {e.code: e for e in (
    Event(code="explicit_cancellation", label="an explicit cancellation",
          names=r"cancell?ations?|cancell?(?:s|ed|ing)?|terminat(?:e|es|ed|ing|ions?)\s+(?:the\s+|their\s+)?(?:contract|subscription|account)"),
    Event(code="non_renewal_by_date", label="failure to renew by the contractual renewal date",
          names=r"non[- ]?renew(?:al|als|ing)?|fail(?:s|ed|ing|ure|ures)?\s+to\s+renew|without\s+renew(?:ing|al)|"
                r"(?:do|does|did|will)\s+not\s+renew|(?:don't|doesn't|didn't|won't)\s+renew|not\s+renew(?:ing|ed)|"
                r"laps(?:e|es|ed|ing)|(?:contract|subscription)\s+(?:expires|expired|runs\s+out)|let(?:s|ting)?\s+(?:it|the\s+contract)\s+expire"),
)}


def event_label(code: str) -> str:
    """An event as it is said to people: its label, else its code in words."""
    found = EVENTS.get(str(code))
    return found.label if found is not None else str(code).replace("_", " ")


_CLAUSE = re.compile(r"[.;:!?,\n]+|\s+(?:but|however|though|whereas|while)\s+", re.I)
# a hedge, a doubt or a both-ways answer: the words do not settle it
_UNSURE = re.compile(r"\bnot\s+(?:only|just|merely|simply)\b|\b(?:both|either|maybe|perhaps|probably|possibly|whichever|any\s+of|"
                     r"not\s+sure|unsure|don't\s+know|do\s+not\s+know|no\s+idea|i\s+guess|i\s+think|depends|too|also|as\s+well)\b|\?|"
                     # a cancellation of the renewal itself (turning off auto-renew) is arguably a non-renewal: ask
                     r"\bauto[- ]?renew\w*|\bcancell?\w*\s+(?:the\s+|their\s+|its\s+|a\s+)?renew\w*|\brenew\w*\s+(?:is\s+|was\s+|were\s+|got\s+)?cancell?\w*",
                     re.I)
# a negation word: "n't" in any spelling, with or without its apostrophe, and the words that set an event aside
_NEGATION = (r"not|no|never|without|excluding|exclude[sd]?|except|ignor(?:e|es|ed|ing)|rather\s+than|instead\s+of|other\s+than|"
             r"neither|nor|cannot|\w+n't|(?:is|are|was|were|do|does|did|wo|ca|could|should|would|have|has|had)nt|dont|wont|cant|"
             r"forget|irrelevant|leave\s+out|leaving\s+out|apart\s+from|aside|excluded|disregard\w*")
_BEFORE = re.compile(rf"\b(?:{_NEGATION})\b(?:\s+[\w'-]+){{0,4}}\s*$", re.I)
_AFTER = re.compile(rf"^(?:\s*[\w'-]+){{0,3}}?\s*\b(?:{_NEGATION})\b", re.I)
_NEGATED = re.compile(rf"\b(?:{_NEGATION})\b", re.I)
# between two phrases, only a verb's negation sets the first aside ("cancellation does not count, lapse does"); a bare
# "rather than" or "not" there points at the second ("cancellation rather than non-renewal")
_AFTER_VERB = re.compile(r"^(?:\s+[\w'-]+){0,2}?\s+(?:(?:is|are|was|were|does|do|did|should|would|will|can|could)\s+(?:not|never)|"
                         r"\w+n't|(?:is|are|was|were|do|does|did|wo|ca|should|would)nt|cannot|excluded|aside)\b", re.I)
_ONLY_NO = re.compile(r"^\s*(?:no|not|never|nope|none|not\s+that|not\s+really)\s*$", re.I)  # "cancellation: no"
_MARK = "\x00{}\x00"
_MARKED = re.compile(r"\x00(\d+)\x00")


def read_event(words: Optional[str], offered: Sequence[str]) -> Optional[str]:
    """The one event of ``offered`` (codes of :data:`EVENTS`) that the words clearly mean, or ``None``.

    How it reads, in order:

    1. A hedge, a question, "both", "either", "not only" or "not just" anywhere: ``None`` (the words do not settle it).
    2. Every phrase naming an offered event is found, clause by clause (a clause ends at punctuation or "but").
    3. A phrase is *negated* when a negation stands just before it in its clause ("not", "rather than", "excluding",
       within four words) or just after it ("… does not count", "… is not required"). A "not" that belongs to another
       event's own phrase ("do not renew") is never read as a negation of its neighbour.
    4. The answer is the event affirmed, when exactly one is and it is not also negated elsewhere. Two affirmed events,
       none, an event said both ways, or only a negation ("not cancellation") return ``None``.

    Why ``None`` instead of the likeliest reading: a wrong event silently changes what "feasible" means. An open question
    costs one more message; a wrong guess costs a model built on another outcome."""
    said = str(words or "").strip().lower().replace("’", "'")
    codes = [c for c in dict.fromkeys(offered) if c in EVENTS]
    if not said or not codes or _UNSURE.search(said):
        return None
    affirmed: set[str] = set()
    negated: set[str] = set()
    clauses = _CLAUSE.split(said)
    if any(_ONLY_NO.match(c) for c in clauses):  # a clause that only says no: which event it refuses is not clear
        return None
    for clause in clauses:
        found = sorted((m.start(), m.end(), n) for n, code in enumerate(codes)
                       for m in re.finditer(rf"\b(?:{EVENTS[code].names})\b", clause))
        kept: list[tuple[int, int, int]] = []
        for start, end, n in found:  # the earliest phrase wins where two overlap
            if not kept or start >= kept[-1][1]:
                kept.append((start, end, n))
        if not kept:
            continue
        masked, at = "", 0
        for start, end, n in kept:  # each phrase becomes a mark, so its own "not" is never a negation of another
            masked += clause[at:start] + _MARK.format(n)
            at = end
        masked += clause[at:]
        if len(kept) == 1:  # one event in the clause: any negation in it sets that event aside ("we don't mean cancellation")
            code = codes[kept[0][2]]
            (negated if _NEGATED.search(_MARKED.sub(" ", masked)) else affirmed).add(code)
            continue
        for m in _MARKED.finditer(masked):
            code = codes[int(m.group(1))]
            before = _MARKED.split(masked[:m.start()])[-1]  # the words since the phrase before it
            after = masked[m.end():]
            nxt = _MARKED.search(after)
            after_negates = _AFTER_VERB.match(after[:nxt.start()]) if nxt else _AFTER.match(after)
            (negated if _BEFORE.search(before) or after_negates else affirmed).add(code)
    clear = affirmed - negated
    if len(affirmed) != 1 or len(clear) != 1:
        return None
    return next(iter(clear))


class TargetContract(Versioned):
    """What the outcome means, as a person confirmed it."""

    schema_version: Literal["dclab_core.target_contract.v1"] = _version("target_contract")
    event: str = Field(pattern=CODE, description="The event the outcome means, as a code (non_renewal_by_date).")
    words: str = Field(min_length=1, max_length=TEXT, description="The person's own words, whole.")
    entity: Optional[str] = Field(default=None, max_length=SHORT, description="What one prediction is for (an account).")
    prediction_time: Optional[str] = Field(default=None, max_length=SHORT, description="When each prediction is made.")
    horizon: Optional[str] = Field(default=None, max_length=SHORT, description="By when the event must happen to count.")
    label_source: Optional[str] = Field(default=None, pattern=REF, description="The table the label comes from, by reference.")
    maturity_rule: Optional[str] = Field(default=None, max_length=SHORT, description="When a row's outcome is known.")
    brief_revision: int = Field(ge=1, description="The brief revision this answer made or was given on.")
    confirmed_by: str = Field(min_length=1, max_length=120, description="The person who confirmed it, by their id.")
    response: Optional[str] = Field(default=None, pattern=REF, description="The recorded response that carried the answer.")

    @field_validator("words")
    @classmethod
    def _said(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("a confirmed target keeps the person's words")
        return value


class LabelDeclaration(Versioned):
    """Which event a table's column records. Declared, never inferred from the column's name."""

    schema_version: Literal["dclab_core.label_declaration.v1"] = _version("label_declaration")
    column: str = Field(min_length=1, max_length=120)
    event: str = Field(pattern=CODE)
    declared_by: str = Field(min_length=1, max_length=120, description="Who says so: a person's id, a team, or 'fixture'.")
    source: Optional[str] = Field(default=None, pattern=REF, description="The table, by reference.")
    mature: bool = Field(default=True, description="Whether the outcomes are known for the rows it would be judged on.")


class TargetVerdict(Contract):
    """Whether the confirmed target is recorded, and what is missing when it is not. Codes and column names only."""

    ok: bool
    kind: Literal["agrees", "no_target", "no_label", "event_mismatch", "immature"]
    event: Optional[str] = Field(default=None, pattern=CODE, description="The event confirmed.")
    column: Optional[str] = Field(default=None, max_length=120, description="The column that records it (or would, once mature).")
    missing: Optional[str] = Field(default=None, pattern=CODE, description="The event whose record is missing or not mature.")
    recorded: list[str] = Field(default_factory=list, max_length=20, description="The events the table does record.")
    columns: list[str] = Field(default_factory=list, max_length=20, description="The declared label columns.")
    reason: str = Field(min_length=1, max_length=SHORT)


def agreement(contract: Optional[TargetContract],
              declaration: Union[LabelDeclaration, Sequence[LabelDeclaration], None]) -> TargetVerdict:
    """Whether a table records the confirmed target.

    - ``no_target``: nothing is confirmed yet, so there is nothing to agree with.
    - ``no_label``: the table declares no outcome column at all.
    - ``event_mismatch``: it records other events only. ``missing`` is the event that needs a record.
    - ``immature``: a column records the event, but its outcomes are not known yet.
    - ``agrees``: a mature column records the event; ``column`` names it.

    The reason names codes and columns, never the person's words: it travels in events and in the ledger."""
    labels = [declaration] if isinstance(declaration, LabelDeclaration) else list(declaration or [])
    recorded = list(dict.fromkeys(d.event for d in labels))[:20]
    columns = list(dict.fromkeys(d.column for d in labels))[:20]
    if contract is None:
        return TargetVerdict(ok=False, kind="no_target", recorded=recorded, columns=columns, reason="no outcome is confirmed yet")
    event, said = contract.event, event_label(contract.event)
    base = {"event": event, "recorded": recorded, "columns": columns}
    if not labels:
        return TargetVerdict(ok=False, kind="no_label", missing=event, **base,
                             reason=f"the table declares no outcome column, so nothing records {said}"[:SHORT])
    same = [d for d in labels if d.event == event]
    if not same:
        held = "; ".join(f"{d.column} records {event_label(d.event)}" for d in labels[:3])
        return TargetVerdict(ok=False, kind="event_mismatch", missing=event, **base,
                             reason=f"the outcome is {said}, but {held}"[:SHORT])
    mature = [d for d in same if d.mature]
    if not mature:
        return TargetVerdict(ok=False, kind="immature", missing=event, column=same[0].column, **base,
                             reason=f"{same[0].column} records {said}, but its outcomes are not known yet"[:SHORT])
    return TargetVerdict(ok=True, kind="agrees", column=mature[0].column, **base,
                         reason=f"{mature[0].column} records {said}"[:SHORT])


def changed(before: Optional[TargetContract], after: Optional[TargetContract]) -> bool:
    """Whether the outcome itself changed: another event, or a target where there was none. The same event in other
    words is the same target (work that rested on it still stands)."""
    return (before.event if before is not None else None) != (after.event if after is not None else None)


for _model in (TargetContract, LabelDeclaration):
    CONTRACTS[_model.model_fields["schema_version"].default.split(".")[1]] = _model

__all__ = ["CODE", "CONTRACT", "EVENTS", "Event", "LabelDeclaration", "TargetContract", "TargetVerdict", "agreement", "changed",
           "event_label", "read_event"]
