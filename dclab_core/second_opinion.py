"""The second opinion (package 28.6, D5 of the phase 6 plan): two readers of the problem card's family, and a judge that keeps
what they agree on and asks the person about the rest.

The readers are the conversation reader (a model, reading the person's words) and the card's rules (the words and the data).
Where both name a family and the two differ, neither is taken silently: the person is asked to choose, the two readings as
the choices, once. Where they agree, or only one of them names a family, the card stands and nothing is asked: the person's
time is what a second opinion costs. The person's choice is a family they stated (a choice between two families is never a
shape for the rules to refine). Framework-free: the host supplies the two readings, asks the question, and records the
verdict with the person's choice.

Kept only where it measured better than one reader on the journeys (its gate, ``second_opinion_gate_v1``, fixed before the
run in the host's measurement); off elsewhere.
"""

from __future__ import annotations

import re
from typing import Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Versioned, _version

Verdict = Literal["agree", "one_reader", "ask"]


class Opinion(Versioned):
    """What the judge made of the two readings of one card field."""

    schema_version: Literal["dclab_core.second_opinion.v1"] = _version("second_opinion")
    field: Literal["family"] = "family"
    reader: Optional[str] = Field(default=None, max_length=40, description="The conversation reader's family (a model's), or None.")
    rules: Optional[str] = Field(default=None, max_length=40, description="The card rules' family from the words and the data, or None.")
    verdict: Verdict
    choices: list[str] = Field(default_factory=list, max_length=2, description="The families the person chooses between (ask only).")
    question: str = Field(default="", max_length=600, description="What the person is asked (ask only), in DCLab's words.")
    why: str = Field(max_length=400)


CONTRACTS["second_opinion"] = Opinion


def _families() -> dict[str, str]:
    """The card's families and what each means, read when used: ``contracts`` imports this module, and ``card`` imports
    ``contracts`` (the review of 28.6: a module-level import broke ``import dclab_core.card`` in a fresh interpreter)."""
    from dclab_core.card import FAMILIES

    return FAMILIES


def judge_family(reader: Optional[str], rules: Optional[str]) -> Opinion:
    """The judge: agree when the two name one family, keep the card when only one names a family, ask when they differ."""
    FAMILIES = _families()
    reader = reader if reader in FAMILIES else None
    rules = rules if rules in FAMILIES else None
    if reader is None or rules is None:
        return Opinion(reader=reader, rules=rules, verdict="one_reader",
                       why="only one reader named a family: the card stands" if (reader or rules) else "no reader named a family")
    if reader == rules:
        return Opinion(reader=reader, rules=rules, verdict="agree", why=f"both readers read {_name(reader)}")
    choices = [reader, rules]
    question = (f"I can read your problem two ways. {_name(reader).capitalize()}: {FAMILIES[reader]}. Or {_name(rules)}: "
                f"{FAMILIES[rules]}. Which is it?")
    return Opinion(reader=reader, rules=rules, verdict="ask", choices=choices, question=question,
                   why=f"the reader read {_name(reader)} and the rules {_name(rules)}: the person chooses")


def chosen(opinion: Opinion, family: Optional[str]) -> Optional[str]:
    """The family the person chose: one of the choices, read from their answer (the host's reading), or None."""
    return family if opinion.verdict == "ask" and family in opinion.choices else None


def option_label(family: str) -> str:
    """A choice as the person sees it: the family's name and what it means."""
    return f"{_name(family)}: {_families()[family]}"


def from_label(text: str, choices: list[str]) -> Optional[str]:
    """The choice an answer names by its label or its family's name (a click on an option, or the name typed)."""
    said = " ".join(str(text or "").lower().replace("_", " ").split())
    named = [c for c in choices if re.search(rf"(?<![\w]){re.escape(_name(c))}(?![\w])", said)]
    return named[0] if len(named) == 1 else None


# What a person says when they mean one family, written by hand (the review of 28.6: words taken from each family's meaning
# chose on "not", "review" or "predict"); phrases, matched whole. A family with no cue is chosen only by its name.
CUES: dict[str, tuple[str, ...]] = {
    "classification": ("which class", "which category", "which type", "a category", "a label for each"),
    "regression": ("a number", "the number", "how much", "an amount", "the amount", "a value", "the minutes", "in minutes", "the price"),
    "rare_event": ("rare", "fraud", "fake", "abuse", "defaults", "the few cases"),
    "anomaly": ("unusual", "anomal", "outlier", "odd ones", "strange", "nothing is labelled", "no labels"),
    "forecasting": ("forecast", "ahead in time", "next week", "next month", "the coming", "future values"),
    "time_to_event": ("how long until", "time until", "time to", "when will", "still waiting", "has not happened", "hasn't happened", "survival"),
    "ranking": ("rank", "ranking", "order the", "top items", "recommend"),
    "segmentation": ("groups of", "segments", "clusters", "no outcome"),
    "uplift": ("because of the", "the difference the", "treatment", "uplift", "who changes"),
    "entity_tables": ("related tables", "several tables", "per customer from"),
    "streams": ("as they arrive", "sensor", "readings", "live events"),
    "text_or_images": ("the text", "long text", "images", "photos", "documents"),
}
NEGATION = re.compile(r"\b(not|no|never|isn'?t|aren'?t|doesn'?t|don'?t|without)\s+(\w+\s+){0,2}$", re.I)


def _cued(text: str, family: str) -> bool:
    """A cue of ``family`` said, and not negated just before it ("not a number")."""
    said = " ".join(str(text or "").lower().split())
    for cue in CUES.get(family, ()):
        for m in re.finditer(rf"(?<![\w]){re.escape(cue)}", said):
            if not NEGATION.search(said[:m.start()]):
                return True
    return False


def read_choice(text: str, choices: list[str], stated: Optional[str] = None) -> Optional[str]:
    """The choice an answer makes between the two families, or None (the card stands). (1) A choice named by its name or
    its label (a click). (2) Otherwise a cue phrase of exactly one choice ("a number" is regression's, "how long until"
    time to event's), not negated. (3) ``stated``: a family the host's rules read in the answer as stated (never a shape:
    "a yes or no" fits a rare event and a class alike). An answer that says nothing ("not sure"), cues of both, or two
    readings that differ choose nothing (the review of 28.6)."""
    from dclab_core.conversation import NOT_SAID

    if NOT_SAID.search(str(text or "")):
        return None
    named = from_label(text, choices)
    if named is not None:
        return named
    hits = [c for c in choices if _cued(text, c)]
    if len(hits) > 1:
        return None  # cues of both: the answer does not choose, whatever else reads it
    picks = set(hits) | ({stated} if stated in choices else set())
    return next(iter(picks)) if len(picks) == 1 else None


def choices_of(options: list[str]) -> list[str]:
    """The families of the options the person was shown (``option_label``'s labels), in order."""
    return [str(o).split(":")[0].strip().replace(" ", "_") for o in options or []]


def _name(family: str) -> str:
    return family.replace("_", " ")


__all__ = ["Opinion", "Verdict", "judge_family", "chosen", "option_label", "from_label", "read_choice", "choices_of"]
