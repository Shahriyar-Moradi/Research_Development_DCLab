"""The case file (package 37.1, docs/guides/PHASE13_CLAUDE_FIRST_PLAN.md): the one living record of a case, from the
person's first sentence to the build, that every agent reads and writes.

It holds what the person wants and why: the goal, the decision it serves, who uses the result, the data and what it
cannot answer, their limits, the assumptions with their status, the unknowns ranked by what they block, the options
offered and the one chosen, and the route with its reasons.

Three rules make it safe for a model to write it:

* **Every fact says where it came from** (:class:`CaseFact`): ``said`` quotes the person's own words (at least two
  whole words of a message, sharing a word with the value); ``computed`` is written by code only, and only about the data;
  ``assumed`` is tentative, shown to the person and never counted as known until they confirm it in a later message of
  their own (``confirmed``) or reject it. A model cannot record an understanding the person never gave, nor rewrite
  their words: the words grow only by :func:`hear`.
* **Every change is a version** (:class:`CaseChange`): who made it (the person, the agent, code, a specialist), why, and
  the value before. A case file is frozen; :func:`apply` returns the next version.
* **Readiness and the options' checks are code** (:func:`readiness`, :func:`check_options`): the agent leads the
  conversation; code says when it may propose and which options may be shown.

The problem card (20.1) is a view of the case file: :func:`from_card` keeps the card's rule-made parts whole (the family,
the labels, its questions and readiness) and :func:`to_card` rebuilds it, changed only where the case's facts changed
after it was read. Pure pydantic: the host stores it and joins it to its own records.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union

from pydantic import Field

from typing import ClassVar

from dclab_core.contracts import CONTRACTS, MAX_TEXT, Contract, TextFact, Versioned, _version

FactStatus = Literal["said", "computed", "assumed", "confirmed", "rejected"]
KNOWN = frozenset({"said", "computed", "confirmed"})  # what readiness counts; an assumption is shown, never counted
SINGLE = ("goal", "decision", "user", "data")  # one fact each
LISTS = ("cannot_answer", "constraints", "assumptions")  # facts added one by one
TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")
TOOLS = ("ml_project", "segmentation_project", "query", "recipe", "dashboard", "business_brief", "cited_answer")  # what DCLab runs
TRACK_OF_TOOL = {"ml_project": "prediction", "segmentation_project": "prediction", "query": "analysis", "recipe": "preparation",
                 "dashboard": "reporting", "business_brief": "business", "cited_answer": "advice"}  # each tool serves one kind of help
COMPUTED_FIELDS = ("data", "cannot_answer")  # the review of 37.1: what code may compute; the goal, decision and user are the person's
# the owner (2026-10-30): no fixed count of questions, but a ceiling: after about six, or two the person declined, the agent
# proposes with what is still open stated as assumptions
CEILING = {"asked": 6, "declined": 2}
READY_NEEDS = ("goal", "decision", "user", "data")
# The kind of help a person's own words plainly name (0.152.0; the plan of 2026-11-01, step 1: the lead agent fell back on an
# analysis when unsure). A reading of the person's words only, never of the agent's: code makes sure such a kind of help is
# offered, and a clear person (the goal said and the kind of help named) may be proposed to at once. Kept general on purpose:
# a phrase here must name the deliverable in any business, not fit one written person.
OUTCOME = r"(?:leave|churn|cancel\w*|quit|buy|default|return|stop|lapse|renew|pay|come back|show up)"  # what a prediction is about
NAMED = {
    # a request about cases not yet known; a bare "prediction" or "forecast" is not one ("a churn prediction for the board" wanted
    # past rates, "is an AI forecast worth it" advice: the 20 route practice people, 2026-11-01)
    "prediction": re.compile(r"\b(?:predict (?:which|who|whether|if|how many|when)|forecast (?:the|our|next|how|which)|(?:who|which \w+(?: \w+)?) "
                             r"(?:will|is going to|are going to|is likely to|are likely to|might) (?:get |be )?" + OUTCOME + r"|(?:likely|about|going) to "
                             r"(?:get |be )?" + OUTCOME + r"|before (?:they|it) (?:churn|cancel|stop buying|quit|lapse))\b", re.I),
    # a business decision about the business, never about a method: "should I drop the duplicate rows", "decide whether to use
    # SMOTE" and "why is the accuracy so high" are method questions (the review of step 1)
    "business": re.compile(r"\bshould (?:we|i) (?:drop|cut|keep|close|open|raise|lower|increase|reduce|invest|hire|stop|launch|expand|"
                           r"move|cancel|discontinue|switch|sell|buy|outsource)\b(?!\s+(?:the\s+|our\s+|my\s+|these\s+|this\s+)?(?:duplicate|rows?|"
                           r"thresholds?|outliers?|models?|features?|columns?|nulls?|missing|learning rate|epochs?|samples?|weights?|classifier|split))|"
                           r"\bwhat (?:should|do) (?:we|i) do\b(?! (?:about|with) (?:the |my |our )?(?:leak\w*|missing|nulls?|outliers?|duplicates?|imbalance|overfit\w*|"
                           r"the model|my model))|\b(?:drop|cut|close|discontinue|stop selling|stop offering|get rid of)\b[^.?!]{0,30}\b(?:worst|weakest|"
                           r"least|bottom|lowest)\b", re.I),
    "analysis": re.compile(r"\b(?:paste|list|send|give|show|pull|export|get)(?: me| us)? (?:all |the |a list of |every )?(?:\w+ ){0,3}"
                           r"(?:customers|members|users|clients|orders|bookings|rows|records|accounts|ids)\b[^.?!]{0,60}\b(?:who|that|which|with)\b|"
                           r"\bwith their (?:ids|emails|names|numbers)\b", re.I),
    "reporting": re.compile(r"\b(?:every|each) (?:week|month|monday|morning|quarter|day)\b[^.?!]{0,40}\b(?:report|dashboard|list|view|update|email|"
                            r"summary|numbers|figures)\b|\b(?:report|dashboard|list|view|update|email|summary)\b[^.?!]{0,30}\b(?:every|each) (?:week|month|"
                            r"monday|morning|quarter|day)\b|\b(?:weekly|monthly|daily|quarterly) (?:report|dashboard|view|update|email|summary)\b|"
                            r"\bdashboard\b", re.I),
    "preparation": re.compile(r"\bclean (?:up )?(?:my|the|our|this|these)\b|\bdedup\w*|\bremove (?:the )?duplicates\b|\b(?:merge|combine|join) "
                              r"(?:the|our|these|my|two|both)\b|\breshape\b|\bfix the (?:dates|formats?|typos|duplicates)\b", re.I),
    "advice": re.compile(r"\breal or hype\b|\bis (?:it|that|this) (?:real|true|possible|realistic|a good idea)\b|\bwhat(?:'s| is) the difference\b|"
                         r"\b(?:in general|in theory|explain (?:to me )?(?:what|how|why)|no data work|just (?:the )?answer)\b", re.I),
}


# 2026-11-02 (the owner's Telco session): a person who asks for a number, a ranking, a report or a chart of the table they shared
# is asking for something the table can answer now. They are not asked what they will do with it first: the figures come first.
FIGURES = re.compile(
    r"\bwhich (?:\w+ ){0,2}(?:segments?|groups?|types?|kinds?|categor(?:y|ies)|regions?|plans?|products?|channels?|contracts?|cohorts?)\b|"
    r"\bhow many\b|\bwhat (?:share|percent\w*|proportion|number|rate|fraction)\b|\bbreak ?down\b|\bcompare\b|\brank\w*\b|"
    r"\b(?:report|dashboard|plot|chart|graph|ranking)\b|\b(?:show|give|send|make|draw)(?: me| us)? (?:my |the |a |an )?(?:\w+ )?"
    r"(?:table|numbers|figures|summary|totals?|counts?)\b", re.I)


def asks_for_figures(words: Iterable[str] | str) -> bool:
    """The person's own words ask for figures, a ranking, a report or a chart (a reading of their words only)."""
    if isinstance(words, str):
        words = [words]
    return any(FIGURES.search(str(w)) for w in words)


MODEL_WORDS = re.compile(r"\b(?:predict\w*|classifier|model(?:s|ling|ing)?|forecast\w*|train(?:ed|ing)?|machine learning|baseline|hold-?out|"
                         r"logistic|boosting|likely to|will (?:leave|churn|cancel)|score each|risk|scores?|likelihood|probabilit\w*|random forest|svm|xgboost|"
                         r"lightgbm|regression|neural|cross.?validat\w*|folds?|imbalanc\w*|overfit\w*|should (?:i|we))\b", re.I)


def wants_figures(words: Iterable[str] | str) -> bool:
    """Figures, a ranking, a report or a chart of the table, and no model asked for: what the table can answer now."""
    if isinstance(words, str):
        words = [words]
    words = list(words)
    return asks_for_figures(words) and not any(MODEL_WORDS.search(str(w)) for w in words)


OUTCOME_LANG = re.compile(r"\b(?:outcome|target|label|predict\w*|whether|marks?|means?|says?|shows?|tells?|flags?|left|leave\w*|leaving|churn\w*|cancel\w*|lost|los\w+|"
                          r"default\w*|fraud\w*|stopped|quit)\b", re.I)


def grounded_columns(case: "CaseFile", columns: Iterable[str]) -> list[str]:
    """The table's columns that the person themselves tied to what happened: a column named, in one sentence of their own words or of
    an assumption they confirmed, together with outcome language ("Churn means the customer left", "the churn_flag column says who
    left"). A column only mentioned ("is it different by gender?") is not one, and the agent's own paraphrases ground nothing. The
    outcome of a build is taken from here and never guessed."""
    texts = [*case.words, *(f.value for f in case.assumptions if f.status == "confirmed")]
    sentences = [s for t in texts for s in re.split(r"(?<=[.?!;])\s+|\n+", str(t)) if s.strip()]
    found = []
    for column in columns:
        pattern = re.compile(r"(?<![\w])" + re.escape(str(column)) + r"(?![\w])", re.I)
        if any(pattern.search(s) and OUTCOME_LANG.search(pattern.sub(" ", s)) for s in sentences):
            found.append(str(column))
    return found


def named_tracks(words: Iterable[str]) -> dict[str, str]:
    """The kinds of help the person's own words name, each with the phrase that names it (the first one found)."""
    found: dict[str, str] = {}
    for w in words:
        for track, pattern in NAMED.items():
            m = pattern.search(str(w))
            if m and track not in found:
                found[track] = m.group(0)
    return found


class CaseRefused(ValueError):
    """A change the rules refuse: the reasons say what to fix."""


class CaseFact(Contract):
    value: str = Field(min_length=1, max_length=MAX_TEXT)
    status: FactStatus
    quote: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The person's exact words (said), or their yes (confirmed).")
    computed: Optional[str] = Field(default=None, max_length=300, description="What code computed it from (a record path, a profile).")
    after_words: Optional[int] = Field(default=None, ge=0, description="An assumption: how many of the person's messages there were when it was made.")


class Unknown(Contract):
    field: str = Field(max_length=60)
    question: str = Field(max_length=1000)
    blocks: Literal["all", "some", "none"] = Field(description="Which options wait for it: every one, some, or none.")
    rank: int = Field(default=0, ge=0, le=100, description="Lower first: what unblocks the most.")


class CaseOption(Contract):
    """One way DCLab could help, written for this person: what it does, what they get, the trade-off and its effort."""

    title: str = Field(min_length=1, max_length=200)
    what: str = Field(max_length=1000)
    gets: str = Field(default="", max_length=1000)
    tradeoff: str = Field(default="", max_length=1000)
    effort: str = Field(default="", max_length=200)
    track: str = Field(max_length=40)
    tool: str = Field(max_length=60)
    columns: list[str] = Field(default_factory=list, max_length=30)


class CaseRoute(Contract):
    track: str = Field(max_length=40)
    tool: str = Field(max_length=60)
    reasons: str = Field(default="", max_length=MAX_TEXT)
    by: Literal["person", "agent", "code"] = "agent"


class CaseFinding(Contract):
    """37.7: what a specialist found, written into the case for the lead agent: never shown to the person directly."""

    by: str = Field(max_length=60, description="The specialist: data_investigator, leakage_auditor, critic, domain_expert, solution_designer.")
    kind: Literal["fact", "risk", "question", "assumption"]
    text: str = Field(min_length=1, max_length=600)
    field: str = Field(default="", max_length=60)


class CaseBuild(Contract):
    """37.6: the build the person said yes to, with their words; recorded, not run, until the product's build path takes it."""

    tool: str = Field(max_length=60)
    quote: str = Field(min_length=1, max_length=MAX_TEXT)


class CaseChange(Contract):
    version: int = Field(ge=1)
    field: str = Field(max_length=60)
    op: Literal["set", "add", "remove", "counts"] = Field(default="set", description="v2: how the field changed, so a case rebuilds from its changes")
    by: str = Field(max_length=120, description="person, agent, code, card, or specialist:<name>.")
    why: str = Field(max_length=1000)
    before: Any = None
    after: Any = None


class CaseFile(Versioned):
    """The living record of one case. v2 (37.6, 37.7) adds the specialists' findings and the build the person said yes to;
    v1 reads as v2 with neither."""

    schema_version: Literal["dclab_core.case_file.v2"] = _version("case_file", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.case_file.v1"})
    case_id: str = Field(min_length=1, max_length=120)
    version: int = Field(default=0, ge=0)
    words: list[str] = Field(default_factory=list, max_length=200, description="The person's messages, whole, in order.")
    goal: Optional[CaseFact] = None
    decision: Optional[CaseFact] = None
    user: Optional[CaseFact] = Field(default=None, description="Who uses the result.")
    data: Optional[CaseFact] = Field(default=None, description="The data: computed from an upload, or the person's words (including 'none').")
    cannot_answer: list[CaseFact] = Field(default_factory=list, max_length=30, description="What the data cannot answer, computed.")
    constraints: list[CaseFact] = Field(default_factory=list, max_length=30)
    assumptions: list[CaseFact] = Field(default_factory=list, max_length=30)
    unknowns: list[Unknown] = Field(default_factory=list, max_length=30)
    asked: int = Field(default=0, ge=0)
    declined: int = Field(default=0, ge=0)
    options: list[CaseOption] = Field(default_factory=list, max_length=4)
    chosen: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The option's title, or the person's own words.")
    route: Optional[CaseRoute] = None
    card: Optional[dict[str, Any]] = Field(default=None, description="The problem card as read (from_card): its rule-made parts, kept whole.")
    findings: list[CaseFinding] = Field(default_factory=list, max_length=30, description="v2: what the specialists found.")
    build: Optional[CaseBuild] = Field(default=None, description="v2: the build the person said yes to.")
    changes: list[CaseChange] = Field(default_factory=list)

    @classmethod
    def read(cls, document: Any) -> "CaseFile":
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        if data.get("schema_version") == "dclab_core.case_file.v1":
            data = {**data, "schema_version": cls.model_fields["schema_version"].default}
        return super().read(data)


class Readiness(Contract):
    ready: bool
    missing: list[str] = Field(default_factory=list)
    blocking: list[str] = Field(default_factory=list, description="Unknowns that block every option.")
    may_propose: bool = Field(description="Ready, or the ceiling reached: propose with what is open stated as assumptions.")
    reasons: list[str] = Field(default_factory=list)


def new(case_id: str, words: Iterable[str] = ()) -> CaseFile:
    return CaseFile(case_id=case_id, words=[str(w) for w in words if str(w).strip()])


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


_WORD = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)?", re.UNICODE)
_STOP = frozenset("the a an and or of to in on for with my our we i you it is are be that this so can about some just".split())
MIN_QUOTE_WORDS = 2
YES_NO = frozenset({"yes", "yeah", "yep", "right", "correct", "exactly", "no", "nope", "not", "wrong"})  # a short answer to an assumption


def _content(text: str) -> set[str]:
    return {w[:4] for w in (x.lower() for x in _WORD.findall(str(text or ""))) if len(w) >= 3 and w not in _STOP}


def quote_ok(quote: Optional[str], value: str, words: Iterable[str]) -> bool:
    """A quote is the person's: at least two whole words, found on word boundaries in one message, sharing a content word
    with the value it supports (the review of 37.1: "a" and "ales te" passed, and nothing tied the quote to the value)."""
    q = _norm(quote or "")
    if len(_WORD.findall(q)) < MIN_QUOTE_WORDS:
        return False
    pattern = re.compile(r"(?<!\w)" + re.escape(q) + r"(?!\w)")
    return any(pattern.search(_norm(w)) for w in words) and bool(_content(value) & _content(q))


def whole_short_message(quote: Optional[str], words: Iterable[str]) -> bool:
    """The quote is the whole of a short message (up to four words): a bare "yes" or "go" is the person's yes and cannot be quoted in two
    words (the owner's second Telco session: five messages to start one build)."""
    plain = lambda t: re.sub(r"[\W_]+", " ", str(t or "")).strip().lower()  # noqa: E731
    q = plain(quote)
    return bool(q) and len(q.split()) <= 4 and any(plain(w) == q for w in words)


_NUMBER = re.compile(r"(?<![\w.])(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s?(?:%|percent\b|per cent\b)|[kKmMbB]\b|[xX]\b)?(?:\s+(?:in|out of)\s+\d+)?")


def _canon(token: str) -> str:
    t = re.sub(r"\s+", "", token.lower()).replace(",", "").replace("percent", "%")
    num = re.match(r"^(\d*\.?\d+)", t)
    if num:
        try:
            value = float(num.group(1))
            t = (str(int(value)) if value == int(value) else str(value)) + t[num.end():]
        except ValueError:
            pass
    return t


def numbers_unsaid(text: str, sources: Iterable[str]) -> list[str]:
    """Numbers a text states in digits that no source holds. A bare whole number up to ten is a count (two options, step
    one), never a claim; with a unit or a ratio ("5x", "9 in 10", "8%") it is one."""
    have = {_canon(n) for s in sources for n in _NUMBER.findall(re.sub(r"per cent", "percent", str(s or ""), flags=re.I))}
    out = []
    for n in _NUMBER.findall(re.sub(r"\b[A-Z]{1,4}-\d{3,}\b", " ", re.sub(r"per cent", "percent", str(text or ""), flags=re.I))):
        c = _canon(n)
        if c.isdigit() and int(c) <= 10:
            continue
        if c not in have:
            out.append(n.strip())
    return out


def check_fact(fact: CaseFact, case: CaseFile, field: str = "", by: str = "") -> list[str]:
    """What is wrong with a fact before it enters the case (empty when it may)."""
    problems = []
    if fact.status == "said" and not quote_ok(fact.quote, fact.value, case.words):
        problems.append(f"the quote {fact.quote!r} is not the person's words for {fact.value!r}")
    if fact.status == "confirmed":
        problems.append("an assumption is confirmed only through confirm()")
    if fact.status == "computed":
        if not (fact.computed or "").strip():
            problems.append("a computed fact names what computed it")
        if field and field not in COMPUTED_FIELDS:
            problems.append(f"{field} is the person's to say: code computes only {', '.join(COMPUTED_FIELDS)}")
        if by and by != "code":
            problems.append("only code writes a computed fact")
    return problems


def _dump(value: Any) -> Any:
    return value.model_dump(mode="json") if hasattr(value, "model_dump") else value


def _next(case: CaseFile, field: str, value: Any, by: str, why: str, op: str = "set", **update: Any) -> CaseFile:
    """The next version, validated whole and sharing nothing with the one before (the review of 37.1: a copy skipped the
    limits and shared its lists, so an append rewrote history)."""
    version = case.version + 1
    before = getattr(case, field, None)
    change = CaseChange(version=version, field=field, op=op, by=by, why=why, before=json.loads(json.dumps(_dump(before), default=_dump)),
                        after=json.loads(json.dumps(_dump(value), default=_dump)))
    doc = json.loads(case.model_dump_json())
    doc.update(json.loads(json.dumps({k: _dump(v) if not isinstance(v, list) else [_dump(x) for x in v] for k, v in update.items()}, default=_dump)))
    doc.update({"version": version, "changes": [*doc["changes"], json.loads(change.model_dump_json())]})
    try:
        return CaseFile.model_validate(doc)
    except ValueError as error:
        raise CaseRefused(f"the case file would not be valid: {error}") from None


def apply(case: CaseFile, field: str, value: Union[CaseFact, Unknown, CaseOption, CaseRoute, list, str, None], *, by: str, why: str) -> CaseFile:
    """The next version with ``field`` changed: a single fact replaced, a fact or unknown added to a list, the options
    replaced, the choice or the route set. Refused (:class:`CaseRefused`) when a fact breaks the source rules. The person's
    words are never changed here (:func:`hear`)."""
    if field in SINGLE or field in LISTS:
        facts = value if isinstance(value, list) else [value]
        problems = [p for f in facts if f is not None for p in check_fact(f, case, field, by)]
        if problems:
            raise CaseRefused("; ".join(problems))
        facts = [f.model_copy(update={"after_words": len(case.words)}) if f is not None and f.status == "assumed" else f for f in facts]
        if field in SINGLE:
            return _next(case, field, facts[0], by, why, **{field: facts[0]})
        return _next(case, field, facts, by, why, op="add", **{field: [*getattr(case, field), *facts]})
    if field == "unknowns":
        items = value if isinstance(value, list) else [value]
        return _next(case, field, items, by, why, op="add", unknowns=sorted([*case.unknowns, *items], key=lambda u: u.rank))
    if field == "resolved":  # an unknown answered: removed by its field
        return _next(case, "unknowns", value, by, why, op="remove", unknowns=[u for u in case.unknowns if u.field != value])
    if field == "options":
        options = list(value or [])
        wrong = [f"{o.title}: DCLab cannot run {o.track}/{o.tool}" for o in options if TRACK_OF_TOOL.get(o.tool) != o.track]
        if wrong:
            raise CaseRefused("; ".join(wrong) + " (show only options check_options kept)")
        return _next(case, field, options, by, why, options=options)
    if field in ("chosen", "route"):
        return _next(case, field, value, by, why, **{field: value})
    if field == "findings":  # 37.7: a specialist's findings, appended
        items = value if isinstance(value, list) else [value]
        return _next(case, field, items, by, why, op="add", findings=[*case.findings, *items])
    if field == "build":  # 37.6: the person's yes to a build, with their words
        latest = case.words[-1:] if case.words else []
        if value is None or not (quote_ok(value.quote, value.quote, latest) or whole_short_message(value.quote, latest)):
            raise CaseRefused("a build needs the person's own words from their latest message")
        return _next(case, field, value, by, why, build=value)
    raise CaseRefused(f"no field {field!r} the agent may change")


def hear(case: CaseFile, words: str) -> CaseFile:
    """The person's next message, kept whole: a version by the person (the words are the record every fact quotes)."""
    if not str(words).strip():
        return case
    return _next(case, "words", [str(words)], "person", "the person wrote", op="add", words=[*case.words, str(words)])


def confirm(case: CaseFile, index: int, *, quote: str, by: str = "person", reject: bool = False) -> CaseFile:
    """An assumption the person confirmed (or rejected) in their own words, in a message after it was made."""
    if by != "person":
        raise CaseRefused("only the person confirms or rejects an assumption")
    old = case.assumptions[index]
    later = case.words[old.after_words or 0:]  # only a message written after the assumption was made can answer it
    q = _norm(quote)
    in_later = bool(q) and any(re.search(r"(?<!\w)" + re.escape(q) + r"(?!\w)", _norm(w)) for w in later)
    answers = in_later and (quote_ok(quote, old.value, later) or (_WORD.findall(q) or [""])[0] in YES_NO)
    if not answers:
        raise CaseRefused(f"the quote {quote!r} is not the person's answer after the assumption was shown")
    fact = old.model_copy(update={"status": "rejected" if reject else "confirmed", "quote": quote})
    assumptions = [*case.assumptions[:index], fact, *case.assumptions[index + 1:]]
    return _next(case, "assumptions", assumptions, by, "rejected by the person" if reject else "confirmed by the person", assumptions=assumptions)


def asked(case: CaseFile, declined: bool = False, by: str = "agent") -> CaseFile:
    """A question was put to the person (and declined, when they would not or could not answer): a version, since the
    ceiling it counts toward decides when options may be shown."""
    counts = {"asked": case.asked + 1, "declined": case.declined + int(declined)}
    return _next(case, "asked", counts, by, "a question was declined" if declined else "a question was asked", op="counts", **counts)


def declined(case: CaseFile, by: str = "code") -> CaseFile:
    """The person passed over the last question (they did not know or would not say): a version, counted toward the ceiling."""
    counts = {"asked": case.asked, "declined": case.declined + 1}
    return _next(case, "declined", counts, by, "the person passed over a question", op="counts", **counts)


MODELS = {"goal": CaseFact, "decision": CaseFact, "user": CaseFact, "data": CaseFact, "cannot_answer": CaseFact, "constraints": CaseFact,
          "assumptions": CaseFact, "unknowns": Unknown, "options": CaseOption, "route": CaseRoute, "findings": CaseFinding, "build": CaseBuild}


def rebuild(case_id: str, changes: Iterable[Mapping[str, Any]]) -> CaseFile:
    """37.12: a case file from its changes alone, in version order (the recorder's events): the same case, version for version."""
    doc: dict[str, Any] = json.loads(new(case_id).model_dump_json())
    kept: list[dict[str, Any]] = []
    for ch in sorted((dict(c) for c in changes), key=lambda c: int(c["version"])):
        field, op, after = ch["field"], ch.get("op", "set"), ch.get("after")
        if op == "counts":
            doc.update({k: int(v) for k, v in (after or {}).items()})
        elif op == "add":
            doc[field] = [*doc.get(field, []), *(after or [])]
            if field == "unknowns":
                doc[field] = sorted(doc[field], key=lambda u: int(u.get("rank", 0)))
        elif op == "remove":
            doc[field] = [u for u in doc.get(field, []) if u.get("field") != after]
        else:
            doc[field] = after
        doc["version"] = int(ch["version"])
        kept.append(ch)
    doc["changes"] = kept
    return CaseFile.model_validate(doc)


def known(fact: Optional[CaseFact]) -> bool:
    return fact is not None and fact.status in KNOWN


def readiness(case: CaseFile) -> Readiness:
    """Ready to propose: the goal and the decision it serves are known, who uses the result is known, the data is known or
    its absence is, and no unknown blocks every option. Known means said, computed or confirmed: never assumed. The
    ceiling (about six questions, or two declined) lets the agent propose before, with what is open stated."""
    named = named_tracks(case.words)
    # a figures request on a shared table (not a prediction, a business decision or advice) needs only the goal and the data:
    # the decision and the user are asked after the answer, and stay open as assumptions meanwhile
    descriptive = known(case.data) and wants_figures(case.words) and not ({"prediction", "business"} & set(named))
    needs = ("goal", "data") if descriptive else READY_NEEDS
    missing = [f for f in needs if not known(getattr(case, f))]
    blocking = [u.field for u in case.unknowns if u.blocks == "all" and not (descriptive and u.field in ("decision", "user"))]
    ready = not missing and not blocking
    ceiling = case.asked >= CEILING["asked"] or case.declined >= CEILING["declined"]
    reasons = [f"{f} is not known yet" for f in missing] + [f"{f} blocks every option" for f in blocking]
    if ceiling and not ready:
        reasons.append("the question ceiling is reached: propose, stating what is still open as assumptions")
    named = {t: p for t, p in named.items() if t != "advice"}  # advice words are too weak a sign to skip questions on
    clear = known(case.goal) and bool(named) and not blocking  # 0.152.0: a clear person is not made to answer what they said
    if clear and not ready and not ceiling:
        reasons.append("the person said their goal and named the help they want (" + ", ".join(f"{t}: \"{p}\"" for t, p in named.items())
                       + "): propose now, stating what is open as assumptions")
    return Readiness(ready=ready, missing=missing, blocking=blocking, may_propose=ready or ceiling or clear, reasons=reasons)


def check_options(options: Iterable[CaseOption], case: CaseFile, *, columns: Iterable[str], computed: Iterable[str] = (),
                  runnable: Iterable[str] = TOOLS, most: int = 3) -> tuple[list[CaseOption], dict[str, str]]:
    """The options code lets the agent show: a route DCLab can run, columns the table has, no number nobody gave or code
    computed, and each of a different kind (track and tool) from the ones kept before it; at most ``most``."""
    columns, runnable = set(columns), set(runnable)
    sources = [*case.words, *computed]
    kept: list[CaseOption] = []
    refused: dict[str, str] = {}
    for o in options:
        why = []
        if o.tool not in runnable or TRACK_OF_TOOL.get(o.tool) != o.track:
            why.append(f"DCLab cannot run {o.track}/{o.tool}")
        missing = [c for c in o.columns if c not in columns]
        if missing:
            why.append(f"the table has no column {', '.join(missing)}")
        if not (o.gets.strip() and o.tradeoff.strip() and o.effort.strip()):  # 37.5: written for the person, in full
            why.append("an option says what they get, its trade-off and its effort")
        unsaid = numbers_unsaid(" ".join((o.title, o.what, o.gets, o.tradeoff, o.effort)), sources)
        if unsaid:
            why.append(f"numbers nobody gave: {', '.join(unsaid)}")
        if any(k.tool == o.tool for k in kept):  # the review: three tracks on one tool were one plan in three wordings
            why.append("the same kind as an option already kept")
        if not why and len(kept) >= most:
            why.append(f"at most {most} options")
        if why:
            refused[o.title] = "; ".join(why)
        else:
            kept.append(o)
    return kept, refused


# ---------------------------------------------------------------------------------------------------- the problem card view

CARD_FIELDS = {"goal": ("brief", "goal"), "decision": ("decision",), "user": ("decision_owner",)}


def _case_fact(fact: TextFact) -> Optional[CaseFact]:
    if fact.value is None:
        return None
    if fact.status == "stated" and fact.quote:
        return CaseFact(value=str(fact.value), status="said", quote=fact.quote)
    if fact.source == "data":
        return CaseFact(value=str(fact.value), status="computed", computed=f"problem card: {fact.reason or 'data'}"[:300])
    return CaseFact(value=str(fact.value), status="assumed")  # a rule's or an advisor's reading is tentative here


def _text_fact(fact: Optional[CaseFact]) -> TextFact:
    if fact is None or fact.status in ("assumed", "rejected"):
        return TextFact()
    if fact.status == "computed":
        return TextFact(value=fact.value, status="inferred", source="data", reason=fact.computed or "")
    return TextFact(value=fact.value, status="stated", source="person", quote=fact.quote)


def from_card(case_id: str, card: Any, words: Iterable[str] = ()) -> CaseFile:
    """A case file from a problem card: its goal, decision and decision owner as facts, the card kept whole as the view's base."""
    from dclab_core.card import ProblemCard

    card = card if isinstance(card, ProblemCard) else ProblemCard.read(card)
    case = new(case_id, words)
    facts = {"goal": _case_fact(card.brief.goal), "decision": _case_fact(card.decision), "user": _case_fact(card.decision_owner)}
    for name, fact in facts.items():  # the review: a card's quote that is not in these words is no said fact here
        if fact is not None and check_fact(fact, case):
            facts[name] = CaseFact(value=fact.value, status="assumed", after_words=len(case.words))
    return CaseFile.model_validate({**json.loads(case.model_dump_json()), **{k: json.loads(v.model_dump_json()) for k, v in facts.items() if v is not None},
                                    "card": json.loads(card.model_dump_json())})


def to_card(case: CaseFile) -> Any:
    """The problem card as the case file now stands: the card it was read from, with the goal, the decision and its owner
    replaced only where a change after the read set them."""
    from dclab_core.card import ProblemCard

    if case.card is None:
        raise CaseRefused("this case file was not read from a problem card")
    card = ProblemCard.read(case.card)
    changed = {c.field for c in case.changes if c.by != "card"}
    update: dict[str, Any] = {}
    brief = card.brief
    filled: set[str] = set()
    for name, path in CARD_FIELDS.items():
        if name not in changed or not known(getattr(case, name)):  # an assumed or rejected change keeps the card's own value
            continue
        fact = _text_fact(getattr(case, name))
        filled.add(path[-1])
        if path[0] == "brief":
            brief = brief.model_copy(update={path[1]: fact})
        else:
            update[path[0]] = fact
    if brief is not card.brief:
        update["brief"] = brief
    if filled:  # the review: a field the person now filled is no longer open
        update["open_fields"] = [f for f in card.open_fields if f.field not in filled]
    return ProblemCard.model_validate({**json.loads(card.model_dump_json()), **json.loads(json.dumps({k: _dump(v) if not isinstance(v, list) else [_dump(x) for x in v] for k, v in update.items()}))}) if update else card


def views(case: CaseFile) -> Mapping[str, Any]:
    """What a page or another agent reads at a glance: the facts by status, the readiness, the open unknowns."""
    r = readiness(case)
    facts = {f: _dump(getattr(case, f)) for f in SINGLE}
    return {"facts": facts, "assumptions": [_dump(a) for a in case.assumptions if a.status == "assumed"],
            "unknowns": [_dump(u) for u in case.unknowns], "readiness": _dump(r)}


CONTRACTS["case_file"] = CaseFile  # its schema is published with the others

__all__ = ["CaseFile", "CaseFact", "CaseChange", "CaseFinding", "CaseBuild", "CaseOption", "CaseRoute", "Unknown", "Readiness", "CaseRefused", "CEILING", "TOOLS",
           "TRACKS", "TRACK_OF_TOOL", "new", "apply", "hear", "confirm", "asked", "declined", "known", "readiness", "check_fact", "check_options",
           "numbers_unsaid", "quote_ok", "rebuild",
           "from_card", "to_card", "views"]
