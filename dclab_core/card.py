"""The problem card (package 20.1): what problem this is, in a form a project can be built from, and whether it can be yet.

A :class:`ProblemCard` holds the brief (``dclab_core.understand``) and adds three things the brief cannot say alone:

* the **family** of the problem (classification, regression, rare events with labels, anomalies without labels,
  forecasting, time to event, ranking, segmentation, uplift), with the families still in play when the words do not
  settle it, and how it was decided;
* the **label situation**: a column marks the cases, a file lists the known ones, or nothing is labelled;
* the **questions** still open, each marked blocking or not, and a **readiness** verdict with its reasons: a project
  is built only when the card is ready.

The rules are code, not a model's judgement, and each says why. Words like "anomaly" or "fraud" with no label situation
known leave two families in play (rare events with labels, anomalies without them) and raise a blocking question: are
some cases confirmed? The owner's test of 2026-10-13 ("I want build anomaly detection") was built while exactly that
was open, into a model of customer ids.

The main product maps the card's brief onto its ProblemSpec as before (``adapters.main_product.problem_spec``); the
family and the readiness are proposals at its ``spec.objective`` and ``target.column`` decision points.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, MAX_TEXT, Contract, Fact, ProblemBrief, Versioned, _version

Family = Literal["classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking",
                 "segmentation", "uplift", "entity_tables", "streams", "text_or_images"]
FAMILIES: dict[str, str] = {
    "classification": "a class for each row, learned from labelled examples",
    "regression": "a number for each row, learned from examples",
    "rare_event": "the rare cases (fraud, abuse, defaults), learned from cases that are labelled, all or some",
    "anomaly": "the unusual rows, when no case is labelled: scores to review, judged by what reviewers confirm",
    "forecasting": "the values of series ahead in time",
    "time_to_event": "how long until something happens, with rows still waiting (censored)",
    "ranking": "an order of items for each person or query",
    "segmentation": "groups of similar rows, with no outcome to predict",
    "uplift": "who changes because of an action (a treatment), not who would act anyway",
    "entity_tables": "an outcome per entity from several related tables (customers and their orders, accounts and events)",
    "streams": "events or readings over time, scored as they arrive (sensor incidents, sessions)",
    "text_or_images": "an outcome from long text or images, read through an encoder",
}
# rare events cover partial labels too (a sample of the cases found); the last three have no word rules: a host names them
# a family whose model learns from an outcome: the outcome must be named before a build
NEEDS_OUTCOME = frozenset({"classification", "regression", "rare_event", "forecasting", "time_to_event", "ranking", "uplift",
                           "entity_tables", "streams", "text_or_images"})
# a family whose cases are scored at a moment: what is known then decides every column (leakage)
NEEDS_MOMENT = frozenset({"classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking", "uplift",
                          "entity_tables", "streams", "text_or_images"})

WORDS = {  # what the person's words say, family by family (stems: "anomal" catches anomaly and anomalous)
    "anomaly": re.compile(r"\b(anomal\w*|unusual|outlier\w*|abnormal\w*|odd (rows|behaviou?r|orders|transactions))", re.I),
    "rare_event": re.compile(r"\b(fraud\w*|fake (deliver\w*|orders?|accounts?|rides?|reviews?|transactions?)|abus(e|es|ive)\b|scam\w*|"
                             r"chargeback\w*|(loan|credit|payment) defaults?|default(s|ed)? on (a |their |the )?(loan|payment|credit|debt)s?|"
                             r"intrusion\w*|money laundering)", re.I),
    "forecasting": re.compile(r"\b(forecast\w*|demand (per|for|each)|how many .{0,30}\b(tomorrow|next (day|week|month)))", re.I),
    "time_to_event": re.compile(r"\b(how long (until|before)|time (to|until) (churn|failure|event|cancel\w*)|survival analysis|time[- ]to[- ]event)", re.I),
    "ranking": re.compile(r"\b(rank\w* (items|products|results|documents|offers|listings)|recommend\w* (items|products|content|movies|songs|offers)|"
                          r"recommendations? (system|engine)|which items? to show)", re.I),
    "segmentation": re.compile(r"\b(segment (our|the) (customers|users|stores)|customer segmentation|cluster\w* (our|the) (customers|users|stores)|"
                               r"group (our|the|similar) (customers|users|stores))", re.I),
    "uplift": re.compile(r"\b(uplift|treatment effect|incremental (effect|impact|lift|sales|revenue)|persuad\w*|who (to|should we) target)", re.I),
}
LABEL_WORDS = (  # what a person says about labels, in the order tried (a "no" first: "no file of known cases" is none)
    ("none", re.compile(r"\b(nothing is labell?ed|no labell?s?\b|not labell?ed|none (are|is) (labell?ed|confirmed)|no (known|confirmed) cases|"
                        r"unlabell?ed|don'?t have (any )?(labell?s?|confirmed|known)|do not have (any )?(labell?s?|confirmed|known))|^\s*(no|none|nope)\b", re.I)),
    ("file", re.compile(r"\b(a (file|list|sample) of (the )?(known|confirmed|found)|known (frauds?|cases)|confirmed (frauds?|cases)|"
                        r"a file lists (the )?(known|confirmed|cases|frauds?)|some are confirmed)", re.I)),
    ("column", re.compile(r"\b(a column (marks|says|holds)|label column|is[_ ]?(fraud|case)\b column|the outcome is (the )?\w+ column)", re.I)),
)
LABELS_QUESTION = ("Are some cases already confirmed — a column that marks them, or a file of known cases — or is nothing "
                   "labelled at all?")
LABELS_OPTIONS = ["A column marks them", "A file lists the known ones", "Nothing is labelled"]


class FamilyFact(Fact):
    value: Optional[Family] = None


class LabelsFact(Fact):
    value: Optional[Literal["column", "file", "none"]] = None


class CardQuestion(Contract):
    """A question still open on the card: the field it settles, why it matters, and whether a build waits for it."""

    field: str = Field(max_length=60)
    text: str = Field(max_length=1000)
    why: str = Field(max_length=600)
    blocks: bool
    options: list[str] = Field(default_factory=list, max_length=8)


class ProblemCard(Versioned):
    """What problem this is, and whether a project can be built from it yet."""

    schema_version: Literal["dclab_core.problem_card.v1"] = _version("problem_card")
    brief: ProblemBrief
    family: FamilyFact
    in_play: list[Family] = Field(default_factory=list, description="The families the words still leave open.")
    labels: LabelsFact
    outcome: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The outcome: a column, or how the cases are known.")
    questions: list[CardQuestion] = Field(default_factory=list, max_length=20)
    ready: bool
    blocking: list[str] = Field(default_factory=list, max_length=20, description="Why a project cannot be built yet, in plain words.")


def _said(brief: ProblemBrief, words: str) -> str:
    parts = [words, brief.goal.value or "", brief.action.value or "", brief.target_definition.value or ""]
    return " ".join(p for p in parts if p)


def _labels(brief: ProblemBrief, data: Mapping[str, Any], words: str = "") -> LabelsFact:
    if data.get("label_file"):
        return LabelsFact(value="file", status="inferred", source="data", reason="a file of known cases joins the table")
    if brief.label_source.value is not None:
        return LabelsFact(value=brief.label_source.value, status=brief.label_source.status, source=brief.label_source.source,
                          quote=brief.label_source.quote, reason=brief.label_source.reason)
    if brief.target_column.value:
        return LabelsFact(value="column", status="inferred", source="rule", reason=f"the outcome is the column {brief.target_column.value}")
    for value, pattern in LABEL_WORDS:
        found = pattern.search(words or "")
        if found:
            return LabelsFact(value=value, status="stated", source="text", quote=found.group(0))
    return LabelsFact()


def _family(brief: ProblemBrief, words: str, labels: LabelsFact, stated: Optional[str]) -> tuple[FamilyFact, list[str]]:
    """The family: one the person stated first; then the label situation for rare-event and anomaly words; then what a
    stated task or a named outcome says, before any other word ("market segment" in a churn problem is not
    segmentation); the other families' words only when nothing the person said settles it."""
    if stated:
        return FamilyFact(value=stated, status="stated", source="person", quote=stated), []
    said = _said(brief, words)
    hits = [f for f, pattern in WORDS.items() if pattern.search(said)]
    task = brief.task.value
    settled = (brief.task.status == "stated" and task in ("binary", "multiclass", "regression"))
    if "anomaly" in hits or "rare_event" in hits:  # the same words fit both: the label situation decides
        if labels.value in ("column", "file") or brief.label_coverage.value in ("partial", "complete"):
            return FamilyFact(value="rare_event", status="inferred", source="rule",
                              reason=f"rare cases ({', '.join(hits)} in the words) with labels: {labels.value or brief.label_coverage.value}"), []
        if labels.value == "none":
            return FamilyFact(value="anomaly", status="inferred", source="rule",
                              reason=f"unusual rows ({', '.join(hits)} in the words) and nothing labelled"), []
        if not settled:
            return FamilyFact(), ["rare_event", "anomaly"]
    if not settled:
        for family in ("forecasting", "time_to_event", "ranking", "uplift"):
            if family in hits:
                return FamilyFact(value=family, status="inferred", source="rule", reason=f"the words say {family.replace('_', ' ')}"), []
        if "segmentation" in hits and not brief.target_column.value and task is None:  # no outcome at all: groups
            return FamilyFact(value="segmentation", status="inferred", source="rule", reason="groups asked for, and no outcome named"), []
    if task in ("binary", "multiclass"):
        return FamilyFact(value="classification", status="inferred", source="rule", reason=f"a {task} outcome"), []
    if task == "regression":
        return FamilyFact(value="regression", status="inferred", source="rule", reason="a numeric outcome"), []
    return FamilyFact(), []


def problem_card(brief: ProblemBrief, *, words: str = "", data: Optional[Mapping[str, Any]] = None,
                 family: Optional[str] = None) -> ProblemCard:
    """The card of a brief. ``words``: what the person wrote (the problem and their answers), read for the family.
    ``data``: what the files say, when there are files: ``label_file`` (a file of known cases joins the table),
    ``cases_only`` (the only table is a list of known cases), ``tables`` (how many tables). ``family``: one the person
    stated (a host's chosen problem kind), which no word overrides."""
    data = dict(data or {})
    labels = _labels(brief, data, words)
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}: one of {', '.join(FAMILIES)}")
    stated_family = family
    family, in_play = _family(brief, words, labels, stated_family)
    outcome = brief.target_column.value or brief.target_definition.value
    if not outcome and labels.value == "file" and labels.source == "data":  # a file of cases joined, never one only spoken of
        outcome = "the cases listed in the file of known cases"
    questions: list[CardQuestion] = []

    def ask(field: str, text: str, why: str, blocks: bool, options: Optional[list[str]] = None) -> None:
        questions.append(CardQuestion(field=field, text=text, why=why, blocks=blocks, options=options or []))

    if labels.value == "file" and labels.source != "data" and data:  # the words speak of a file the draft does not have
        ask("data", "Add the file of known cases to this draft, with the table it belongs to.",
            "The known cases become the labels only once their file is joined to the table.", True)
    if data.get("cases_only"):
        ask("data", "Add the table these known cases belong to (the full table of orders, accounts or events).",
            "A file of known cases alone has nothing to tell the cases from.", True)
    if family.value is None:
        if in_play:
            ask("labels", LABELS_QUESTION, "Confirmed cases mean learning from them (rare events); none means finding the "
                "unusual rows and judging them by review (anomalies).", True, LABELS_OPTIONS)
        else:
            ask("family", "What should DCLab give for each row: a yes or no, a class, a number, a forecast, or the unusual rows?",
                "The kind of answer decides the method, the split and the metric.", True)
    fam = family.value or (in_play[0] if len(in_play) == 1 else None)
    if fam in NEEDS_OUTCOME and not outcome:
        ask("outcome", "Which column holds the outcome, or which file lists the known cases?",
            "The outcome decides the task, the metric and which columns could leak it.", True)
    if (fam in NEEDS_MOMENT or fam is None) and not brief.prediction_moment.value:
        ask("prediction_moment", "When is each prediction made, and what is known at that moment?",
            "Anything written after that moment must stay out of the model: it is the main source of leakage.", True)
    if not brief.unit.value:
        ask("unit", "What does one row stand for: an order, a customer, a day, an event?",
            "The unit decides what is scored and which rows must stay together in a split.", False)
    if not brief.action.value:
        ask("action", "What happens with each prediction: reviewed, blocked, contacted, or only reported?",
            "The action sets the cost of each mistake and the operating point.", False)
    blocking = [f"{q.text} ({q.why})" for q in questions if q.blocks]
    return ProblemCard(brief=brief, family=family, in_play=in_play, labels=labels, outcome=outcome, questions=questions,
                       ready=not blocking, blocking=blocking)


CONTRACTS["problem_card"] = ProblemCard  # its schema is published with the others
