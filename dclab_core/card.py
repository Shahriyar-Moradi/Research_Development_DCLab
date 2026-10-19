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

import json
import re
from collections.abc import Mapping
from typing import Any, ClassVar, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, MAX_TEXT, Contract, Fact, ProblemBrief, TextFact, Versioned, _version

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
# a family the words may say beside a named outcome column: then asked, never inferred (20.10)
ASKED_AGAINST_A_COLUMN = frozenset({"ranking", "uplift", "streams", "entity_tables", "text_or_images", "time_to_event"})
# a family whose outcome is computed from related tables, not read from a column of the main one (20.10)
FROM_TABLES = frozenset({"entity_tables", "streams"})
# a family whose outcome is an event in the table itself (a purchase in the window after the list), said in words (16.15 B)
OUTCOME_AS_EVENT = frozenset({"ranking"})
# a family whose cases are scored at a moment: what is known then decides every column (leakage)
NEEDS_MOMENT = frozenset({"classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking", "uplift",
                          "entity_tables", "streams", "text_or_images"})

WORDS = {  # what the person's words say, family by family (stems: "anomal" catches anomaly and anomalous)
    "anomaly": re.compile(r"\b(anomal\w*|unusual|outlier\w*|abnormal\w*|odd (rows|behaviou?r|orders|transactions))", re.I),
    "rare_event": re.compile(r"\b(fraud\w*|fake (deliver\w*|orders?|accounts?|rides?|reviews?|transactions?)|abus(e|es|ive)\b|scam\w*|"
                             r"chargeback\w*|(loan|credit|payment) defaults?|default(s|ed)? on (a |their |the )?(loan|payment|credit|debt)s?|"
                             r"intrusion\w*|money laundering)", re.I),
    "forecasting": re.compile(r"\b(forecast\w*|demand (per|for|each)|how (many|much) [^.;]{0,40}\b(tomorrow|(next|coming|following) (day|week|month)))", re.I),
    "time_to_event": re.compile(r"\b(how long (until|before)|(lifetime(?! value)|tenure|lifespan|how long (\w+ ){0,4}stick around|survival|how (long|many (days|weeks|months|years)) (\w+ ){0,6}(keep|stay|remain|last|stick))\b.{0,100}\b((do not|don'?t) know (their|the|its) end|(has|have)(n'?t| not) (yet )?(cancell?ed|churned|left|failed|ended)|(still|not yet) (active|with us|subscribed|running|cancell?ed|churned|left)|censor\w*)|((has|have)(n'?t| not) (yet )?(cancell?ed|churned|left|failed|ended)|censor\w*)\b.{0,100}\b(lifetime(?! value)|tenure|lifespan|how long|survival)|how long (\w+ ){0,4}stick around|how long (will|does|do|would|did) (\w+ ){1,4}(stay|last|survive|remain|keep \w+)\b|time (to|until|before) (\w+ ){0,3}(churn|fail|leave|cancel|quit|die|default)"
                                r"(s|ed|ing)?\b|time (to|until|before) (\w+ ){0,3}breaks? down|time to failure|survival analysis|time[- ]to[- ]event)", re.I),
    "ranking": re.compile(r"\b(rank\w* (items|products|results|documents|offers|listings)|(products|items|articles|songs|movies|offers) (they|each \w+|he|she)('re| are| is)? (most )?likely to (buy|click|watch|read|want)|personali[sz]\w* (the |our )?(carousel|feed|shelf|homepage|recommendations?|picks)|picked for you|recommender|recommend\w* (items|products|content|movies|songs|offers)|"
                          r"recommendations? (system|engine)|which items? to show)", re.I),
    "segmentation": re.compile(r"\b(segment (our|the) (customers|users|stores)|customer segmentation|cluster\w* (our|the) (customers|users|stores)|"
                               r"group (our|the|similar) (customers|users|stores))", re.I),
    "uplift": re.compile(r"\b(uplift|treatment effect|incremental (effect|impact|lift|sales|revenue)|persuad\w*|who (to|should we) target|"
                        r"(would|will|it) make (them |each |the customer )?(buy|stay|convert|respond|renew)|not (those|the ones) who (buy|would buy|renew) anyway|(offer|discount|coupon|voucher|campaign|promotion|email|call)s?\b.{0,80}\bwould have (bought|renewed|stayed|converted|signed up|booked) anyway|would have (bought|renewed|stayed|converted|signed up|booked) anyway\b.{0,80}\b(offer|discount|coupon|voucher|campaign|promotion|email|call)s?|(not|rather than) (just |only )?(those|the ones|people|customers|users) who('d| would| will)? (buy|renew|stay|convert)\w* (anyway|regardless)|(driven|caused|changed) by the (offer|discount|campaign|coupon|treatment)|(offer|discount|coupon|voucher|campaign|promotion|email)s?\b.{0,80}\b(changes?|sways?) (their|his|her) mind|difference the (offer|discount|campaign|coupon) makes)", re.I),
    # 20.10: three families the words alone named nowhere
    "streams": re.compile(r"\b(sensor\w*|telemetry|readings? (from|of|over)|event streams?|predictive maintenance|((early[- ]warning|(log|record|measure|track)\w* (the )?(temperature|vibration|pressure|voltage|rpm))\b.{0,100}\b(machines?|units?|pumps?|compressors?|engines?|turbines?|devices?|equipment|fleet|plant|line)|(machines?|units?|pumps?|compressors?|engines?|turbines?|devices?|equipment|fleet|plant|line)\b.{0,100}\b(early[- ]warning|(log|record|measure|track)\w* (the )?(temperature|vibration|pressure|voltage|rpm)))|"
                          r"(alert|warn\w*) (before|ahead of) (a |the |each )?\w+ (fails?|breaks?( down)?|goes down))", re.I),
    "entity_tables": re.compile(r"\b((several|multiple|related|linked|separate) (tables|files)|(order|purchase|transaction|visit) history,? (and |plus |with )?(their |the )?(support )?(tickets|returns|addresses|address data|payments|visits|events)|(goes?|going|went) (the )?(next |following )?\d+ days without (an? )?(order|purchas|visit|log)\w*|plus the (orders|events|transactions|tickets)|(no|zero|not \w+ an?|without an?) (new )?(orders?|purchases?|visits?|logins?|payments?) (in|within|during|over) the (next|following|coming) \d+ (days|weeks)|(for|per) each (customer|user|account|member) (on|at|per) (each|every) (date|cutoff|month[- ]end)|at (several|each|every) (month[- ]end|cutoff) dates?|from (their|its|the) \w+, \w+,? and \w+|"
                                r"from (their|its) (orders|events|transactions|visits|purchases|tickets|payments) and (their |its )?\w+)\b", re.I),
    "text_or_images": re.compile(r"\b(from (its|their|the) (text|wording|words|images?|photos?|pictures?|scans?)|reads? (the )?(note|ticket|message|document|report|email)s? itself|from what (the|each|a) (note|ticket|message|document|report) says|(long|free)[- ]text (notes?|tickets?|messages?|reports?|descriptions?)|read(ing)? (the|its|their) "
                                 r"(text|notes?|images?|photos?)|(classify|label|tag|sort|recogni[sz]e) (the |each |our )?(images?|photos?|pictures?|scans?))\b", re.I),
}
FORECAST_SAID = re.compile(r"\b(forecast\w*|demand (per|for|each)|time series|per (day|week|month) for each)\b", re.I)  # beyond a "how much ... next"
LABEL_WORDS = (  # what a person says about labels, in the order tried (a "no" first: "no file of known cases" is none)
    ("none", re.compile(r"\b(nothing is labell?ed|no labell?s?\b|not labell?ed|none (are|is) (labell?ed|confirmed)|no (known|confirmed) cases|"
                        r"no (history|record|list|file|log) of (any )?(confirmed|known|labell?ed|past|caught) |without (any )?(confirmed|known|labell?ed) (cases|frauds?)|"
                        r"(have|has) (not|never) (yet )?(confirmed|labell?ed|caught|marked|flagged|checked|investigated|reviewed|verified) (any|a single)|"
                        r"unlabell?ed|don'?t have (any )?(labell?s?|confirmed|known)|do not have (any )?(labell?s?|confirmed|known)|"
                        # 30.6: the ways the sealed people said it that no rule read ("nobody ever confirmed anything", "we never ran an investigation")
                        r"nobody (has |had )?(ever )?(confirmed|checked|investigated|labell?ed|marked|verified)|"
                        r"nothing (has been |was |is |has )?(ever )?(confirmed|checked|investigated|labell?ed|marked|verified)|no(thing| one)? (is |was )?confirmed|"
                        r"(never|not) (yet )?(run|ran|done|did|had|made|carried out) (an? |any )?(investigations?|reviews?|checks?)|"
                        r"(only|just) (have )?suspicions|no (list|file|record) of (bad|fraud\w*|confirmed|known|real)|there is no (list|file))|^\s*(no|none|nope)\b", re.I)),
    ("file", re.compile(r"\b(a (file|list|sample) of (the )?(known|confirmed|found)|known (frauds?|cases)|confirmed (frauds?|cases)|"
                        r"a file lists (the )?(known|confirmed|cases|frauds?)|some are confirmed|"
                        r"(have|has|had) (marked|flagged|confirmed|labell?ed) (some|a few|several|many|a number of))", re.I)),  # 30.6: "investigators have marked some deliveries as fraud"
    ("column", re.compile(r"\b(a column (marks|says|holds)|label column|is[_ ]?(fraud|case)\b column|the outcome is (the )?\w+ column)", re.I)),
    ("file", re.compile(r"\b(the )?(cases|frauds?|ones) (we|they|the team) (found|caught|confirmed|flagged)\b", re.I)),  # 21.5, after a column
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


Period = Literal["day", "week", "month"]
WORSE = ("false_alarm", "miss", "equal")


class ErrorCosts(Contract):
    """30.12-30.13: what each mistake costs, in the person's words, and which is worse when they say so."""

    false_alarm: TextFact = Field(default_factory=TextFact, description="What a false alarm costs: a wasted review, an annoyed customer.")
    miss: TextFact = Field(default_factory=TextFact, description="What a missed case costs.")
    worse: Optional[Literal["false_alarm", "miss", "equal"]] = None


class Capacity(Contract):
    """How many cases a team can act on a period, as a number when the words give one."""

    k: Optional[int] = Field(default=None, ge=1, le=10_000_000)
    period: Optional[Period] = None
    said: TextFact = Field(default_factory=TextFact, description="The person's words, kept whole.")


class Success(Contract):
    business: TextFact = Field(default_factory=TextFact, description="The business number that says the work helped.")
    ml_metric: TextFact = Field(default_factory=TextFact, description="The model's metric that stands in for it.")


class OpenField(Contract):
    """A field of the card nobody has filled yet, and who can answer it: never blocking (only what changes the solution
    blocks the build)."""

    field: str = Field(max_length=60)
    text: str = Field(max_length=1000)
    why: str = Field(max_length=600)
    owner: str = Field(max_length=120, description="Who can answer it: the business owner, the team doing the work today, the data owner.")


class ProblemCard(Versioned):
    """What problem this is, and whether a project can be built from it yet.

    v2 (30.13) adds the fields the owner's framework asks of a problem (the decision and who owns it, what each mistake
    costs, how many cases a team can act on a period, the success numbers, how the work is done today, the horizon), each
    filled from the person's answers or left an open field with the person who can answer it. Only the questions that
    change the solution block (``blocking``, as in v1): ``complete`` says every field is filled or owned (G1)."""

    schema_version: Literal["dclab_core.problem_card.v2"] = _version("problem_card", 2)
    READS: ClassVar[frozenset[str]] = frozenset({"dclab_core.problem_card.v1"})  # v1 only lacked the v2 fields: read as unfilled
    brief: ProblemBrief
    family: FamilyFact
    in_play: list[Family] = Field(default_factory=list, description="The families the words still leave open.")
    labels: LabelsFact
    outcome: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The outcome: a column, or how the cases are known.")
    questions: list[CardQuestion] = Field(default_factory=list, max_length=20)
    ready: bool
    blocking: list[str] = Field(default_factory=list, max_length=20, description="Why a project cannot be built yet, in plain words.")
    decision: TextFact = Field(default_factory=TextFact, description="The decision each prediction feeds.")
    decision_owner: TextFact = Field(default_factory=TextFact, description="Who owns that decision.")
    error_costs: ErrorCosts = Field(default_factory=ErrorCosts)
    capacity: Capacity = Field(default_factory=Capacity)
    success: Success = Field(default_factory=Success)
    baseline_today: TextFact = Field(default_factory=TextFact, description="How the work is done today, and how well.")
    horizon: TextFact = Field(default_factory=TextFact, description="How far ahead each prediction looks (the brief's).")
    open_fields: list[OpenField] = Field(default_factory=list, max_length=20)
    complete: bool = Field(default=False, description="G1: nothing blocks, and every field is filled or an open field with its owner.")

    @classmethod
    def read(cls, document):
        data = json.loads(document) if isinstance(document, (str, bytes)) else dict(document)
        if data.get("schema_version") == "dclab_core.problem_card.v1":
            data = {**data, "schema_version": cls.model_fields["schema_version"].default}
        return super().read(data)


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
    said = label_said(words or "")
    if said:
        found = next(p.search(words) for v, p in LABEL_WORDS if v == said and p.search(words))
        return LabelsFact(value=said, status="stated", source="text", quote=found.group(0))
    return LabelsFact()


def _family(brief: ProblemBrief, words: str, labels: LabelsFact, stated: Optional[str], text_only: bool = False) -> tuple[FamilyFact, list[str]]:
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
        # a label situation said ("nothing is labelled yet") outweighs a coverage the words only suggest ("no history of
        # confirmed fraud" reads as some coverage): 20.10 v2
        if labels.value in ("column", "file") or (labels.value != "none" and brief.label_coverage.value in ("partial", "complete")):
            return FamilyFact(value="rare_event", status="inferred", source="rule",
                              reason=f"rare cases ({', '.join(hits)} in the words) with labels: {labels.value or brief.label_coverage.value}"), []
        if labels.value == "none":
            return FamilyFact(value="anomaly", status="inferred", source="rule",
                              reason=f"unusual rows ({', '.join(hits)} in the words) and nothing labelled"), []
        if not settled:
            return FamilyFact(), ["rare_event", "anomaly"]
    if not settled:
        # time to event before forecasting: rows still waiting (censored) say lifetimes, whatever else is "planned" (20.10 v2)
        for family in ("time_to_event", "forecasting", "ranking", "uplift", "streams", "entity_tables", "text_or_images"):
            if family in hits:
                duration = family == "time_to_event" and task == "regression"  # a lifetime with a duration column: what survival reads
                # a count "next month" said beside a yes or no column ("how many customers will churn next month") is asked too
                counted = family == "forecasting" and (task in ("binary", "multiclass") or not FORECAST_SAID.search(said))  # "how much ... next month" beside a number column: asked
                if (family in ASKED_AGAINST_A_COLUMN or counted) and brief.target_column.value and task in ("binary", "multiclass", "regression") and not duration:
                    # 20.10: an outcome column is named and the words also say another family ("who to target", "from its
                    # sensor readings", "from their photos"): the person says which, never a word in passing
                    return FamilyFact(), [family, "regression" if task == "regression" else "classification"]
                return FamilyFact(value=family, status="inferred", source="rule", reason=f"the words say {family.replace('_', ' ')}"), []
        if "segmentation" in hits and not brief.target_column.value and task is None:  # no outcome at all: groups
            return FamilyFact(value="segmentation", status="inferred", source="rule", reason="groups asked for, and no outcome named"), []
    if text_only and task in ("binary", "multiclass", "regression", None) and not hits:  # 20.10 v2: the table's only input is a long text
        return FamilyFact(value="text_or_images", status="inferred", source="data",
                          reason="the table's only input is a long text: its outcome is read from the words, through an encoder"), []
    if task in ("binary", "multiclass"):
        return FamilyFact(value="classification", status="inferred", source="rule", reason=f"a {task} outcome"), []
    if task == "regression":
        return FamilyFact(value="regression", status="inferred", source="rule", reason="a numeric outcome"), []
    return FamilyFact(), []


# a family answer that only says the output's shape ("a yes or no per note", "a number of days") keeps a family the
# person's other words name, whose output has that shape (20.10 v2): the answer says what comes out, the words what it is
SHAPED_AS = {"classification": ("text_or_images", "entity_tables", "streams"), "regression": ("time_to_event", "text_or_images", "entity_tables")}


def refine_family(family: Optional[str], words: str, shape_only: bool = True) -> Optional[str]:
    """The family a person's answer states, or the one family of that output shape their words name instead. Only an
    answer that gave a shape alone is refined ("a yes or no per note"); one that said "classification" or "regression"
    is kept as said."""
    if not shape_only:
        return family
    named = [f for f in SHAPED_AS.get(family or "", ()) if WORDS[f].search(words or "")]
    return named[0] if len(named) == 1 else family


def label_said(text: str) -> Optional[str]:
    """The label situation a text says: the first of ``LABEL_WORDS`` that matches, except that a file of known cases said
    apart from a "no ... of" ("no record of past chargebacks, but a file of known frauds exists") is a file."""
    found = [(value, m) for value, pattern in LABEL_WORDS if (m := pattern.search(text or ""))]
    if not found:
        return None
    value, first = found[0]
    if value == "none":
        def negated(m: re.Match) -> bool:  # "there is no file of known cases": the file is denied in its own clause
            start = max(text.rfind(c, 0, m.start()) for c in ",.;:") + 1
            return bool(re.search(r"\b(no|not|never|without|nothing|none)\b", text[start:m.start()], re.I))

        later = next((m for v, m in found[1:] if v == "file" and (m.start() >= first.end() or m.end() <= first.start()) and not negated(m)), None)
        if later is not None:
            return "file"
    return value


NUMBER = r"(\d[\d,]*(?:\.\d+)?)\s*(k\b|thousand\b)?"
CAPACITY_WORDS = re.compile(NUMBER + r"\s*(?:[a-z]+\s+){0,3}?(?:a|an|per|each|every)\s+(day|week|month)\b", re.I)
DAILY = {"daily": "day", "weekly": "week", "monthly": "month"}
ACTING = re.compile(r"\b(review|check|call|investigat|handle|look at|act on|examine|audit|inspect|contact)\w*", re.I)
MAX_K = 10_000_000
MISS = re.compile(r"\b(miss(ed|es|ing)?|let through|not caught|goes? unnoticed|lost)\b", re.I)
FALSE_ALARM = re.compile(r"\b(false alarms?|wrong (alerts?|ones?|flags?)|wasted|annoy\w*|bother\w*|for nothing|needless)\b", re.I)
MORE = re.compile(r"\b(more|worse|higher|bigger|greater)\s+than\b", re.I)
LESS = re.compile(r"\b(less|lower|smaller|cheaper|fewer)\s+than\b", re.I)
QUEUE_FAMILIES = frozenset({"classification", "rare_event", "anomaly", "ranking", "uplift"})  # where a team acts on a list


def _amount(number: str, suffix: Optional[str]) -> Optional[int]:
    try:
        value = float(number.replace(",", "")) * (1000 if suffix else 1)
    except ValueError:
        return None
    k = int(value)
    return k if 1 <= k <= MAX_K else None


def capacity_said(text: Optional[str]) -> Capacity:
    """How many cases a period the words give: "about 80 a day" is 80 a day; among several numbers, the one beside a word
    of acting (review, check, call) first ("40,000 orders a day; two analysts check 150 a day" is 150); a year or a
    number out of range is never a capacity; words without a usable number are kept as said."""
    if not text or not str(text).strip():
        return Capacity()
    text = str(text)
    said = TextFact(value=text[:MAX_TEXT], status="stated", quote=text[:MAX_TEXT], source="person")
    found: list[tuple[bool, int, str]] = []
    for m in CAPACITY_WORDS.finditer(text):
        k = _amount(m.group(1), m.group(2))
        if k is not None:
            found.append((bool(ACTING.search(text[max(0, m.start() - 60):m.end()])), k, m.group(3).lower()))
    for word, period in DAILY.items():
        for m in re.finditer(NUMBER + rf"(?:\s+\w+){{0,4}}?\s+{word}\b", text, re.I):
            k = _amount(m.group(1), m.group(2))
            if k is not None and not (1900 <= k <= 2100 and not m.group(2)):  # "since 2019 we check some daily" is a year
                found.append((bool(ACTING.search(text[max(0, m.start() - 60):m.end()])), k, period))
    if found:
        acting, k, period = next((f for f in found if f[0]), found[0])
        return Capacity(k=k, period=period, said=said)
    return Capacity(said=said)


def costs_said(text: Optional[str]) -> ErrorCosts:
    """What each mistake costs, from the person's words: the sentence that speaks of a miss, the one that speaks of a false
    alarm, and which is worse only when one sentence compares them ("a missed fraud costs more than a wasted review":
    the miss; "a miss costs less than a false alarm": the false alarm)."""
    if not text or not str(text).strip():
        return ErrorCosts()
    parts = [p.strip() for p in re.split(r"[.;\n]", str(text)) if p.strip()]
    fact = lambda p: TextFact(value=p[:MAX_TEXT], status="stated", quote=p[:MAX_TEXT], source="person")  # noqa: E731
    miss = next((p for p in parts if MISS.search(p)), None)
    alarm = next((p for p in parts if FALSE_ALARM.search(p)), None)
    worse = None
    for p in parts:
        m, a = MISS.search(p), FALSE_ALARM.search(p)
        if not (m and a):
            continue
        more, less = MORE.search(p), LESS.search(p)
        pivot = more or less
        if pivot is not None:
            first = "miss" if m.start() < a.start() else "false_alarm"
            second = "false_alarm" if first == "miss" else "miss"
            if min(m.start(), a.start()) < pivot.start() < max(m.start(), a.start()):
                worse = first if more is not None and pivot is more else second
        elif re.search(r"\b(the same|equal|as much as)\b", p, re.I):
            worse = "equal"
    return ErrorCosts(false_alarm=fact(alarm) if alarm else TextFact(), miss=fact(miss) if miss else TextFact(), worse=worse)


def _filled(fact: Any) -> bool:
    return bool(getattr(fact, "known", False))


def problem_card(brief: ProblemBrief, *, words: str = "", data: Optional[Mapping[str, Any]] = None,
                 family: Optional[str] = None, answers: Optional[Mapping[str, Any]] = None) -> ProblemCard:
    """The card of a brief. ``words``: what the person wrote (the problem and their answers), read for the family.
    ``data``: what the files say, when there are files: ``label_file`` (a file of known cases joins the table),
    ``cases_only`` (the only table is a list of known cases), ``tables`` (how many tables), ``columns`` (the table's
    columns are known: an outcome is then a column, or for entity tables and streams the words), ``text_only`` (the table's
    only input is a long text: read through an encoder). ``family``: one the person
    stated (a host's chosen problem kind), which no word overrides. ``answers`` (30.13): what the person said of the v2
    fields (``decision``, ``decision_owner``, ``costs``, ``capacity``, ``baseline``, ``success``), each kept as stated."""
    data = dict(data or {})
    labels = _labels(brief, data, words)
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}: one of {', '.join(FAMILIES)}")
    stated_family = family
    family, in_play = _family(brief, words, labels, stated_family, bool(data.get("text_only")))
    if stated_family is None and family.value is None and not in_play and labels.value == "none":
        # 30.6: nothing was ever confirmed, so no outcome can be learned: the rows that stand out are all that can be found (the
        # sealed people who said "nobody ever confirmed anything" and could not name a kind of answer were stopped here)
        grouped = bool(re.search(r"\b(segment\w*|cluster\w*|group (them|our|the))\b", words or "", re.I))
        family = FamilyFact(value="segmentation" if grouped else "anomaly", status="inferred", source="rule",
                            reason="nothing is labelled, so no outcome can be learned: the rows that stand out (or the groups) are what can be found")
    if (stated_family is None and family.value in (None, "classification") and not in_play and (data.get("tables") or 0) >= 2 and not brief.target_column.value
            and brief.target_definition.value and not WORDS["streams"].search(words or "")):
        # 20.10 v2: several tables, and an outcome said in words, not a column ("no order in the next 30 days"): a yes or no
        # computed per entity from its related tables, whatever shape the answer gave it
        family = FamilyFact(value="entity_tables", status="inferred", source="data",
                            reason=f"{data.get('tables')} tables and an outcome said in words: computed per entity from the related tables")
    fam_now = family.value or (in_play[0] if len(in_play) == 1 else None)
    # with a table, an outcome is one of its columns or a file of cases; for a family whose outcome is computed from the
    # related tables (no order in 30 days, a failure in the incidents), the outcome said in words is one (20.10)
    defined = brief.target_definition.value if not data.get("columns") or fam_now in FROM_TABLES | OUTCOME_AS_EVENT else None
    outcome = brief.target_column.value or defined
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
        if in_play and set(in_play) != {"rare_event", "anomaly"}:  # 20.10: two families the words and the column each say
            ask("family", f"Is this {' or '.join(f.replace('_', ' ') for f in in_play)}? Say which answer DCLab should give for each row.",
                "The words say one kind of problem and the outcome column another; the kind decides the method, the split and the "
                "metric.", True, list(in_play))
        elif in_play:
            ask("labels", LABELS_QUESTION, "Confirmed cases mean learning from them (rare events); none means finding the "
                "unusual rows and judging them by review (anomalies).", True, LABELS_OPTIONS)
        else:
            ask("family", "What should DCLab give for each row: a yes or no, a class, a number, a forecast, or the unusual rows?",
                "The kind of answer decides the method, the split and the metric.", True)
    fam = family.value or (in_play[0] if len(in_play) == 1 else None)
    if fam in FROM_TABLES and defined and not brief.target_column.value and data.get("columns") and (data.get("tables") or 0) <= 1:
        ask("data", "Attach the table whose events decide the outcome (the incidents, the orders) to the project, by name.",
            "The outcome is computed from that table: nothing is modelled until it is attached.", False)
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
    v2 = _v2_fields(brief, dict(answers or {}), fam)
    return ProblemCard(brief=brief, family=family, in_play=in_play, labels=labels, outcome=outcome, questions=questions,
                       ready=not blocking, blocking=blocking, **v2, complete=not blocking)


def _v2_fields(brief: ProblemBrief, answers: dict[str, Any], fam: Optional[str]) -> dict[str, Any]:
    """30.13: the framework's fields, each filled from the person's answers (stated) or the brief (inferred), or left an
    open field with the person who can answer it. None of them blocks."""
    def stated(key: str) -> TextFact:
        value = str(answers.get(key) or "").strip()
        return TextFact(value=value[:MAX_TEXT], status="stated", quote=value[:MAX_TEXT], source="person") if value else TextFact()

    decision = stated("decision")
    if not _filled(decision) and brief.action.known:
        decision = TextFact(value=str(brief.action.value)[:MAX_TEXT], status="inferred", source="rule",
                            reason="the action each prediction feeds, as the person described it")
    try:  # the person's words never stop a card: a reading that fails leaves its field open
        costs = costs_said(answers.get("costs"))
    except Exception:  # noqa: BLE001
        costs = ErrorCosts()
    try:
        capacity = capacity_said(answers.get("capacity"))
        if capacity.k is None and not _filled(capacity.said):
            said = next((c.text for c in brief.constraints if c.kind == "review_capacity"), None)
            capacity = capacity_said(said) if said else capacity
    except Exception:  # noqa: BLE001
        capacity = Capacity()
    business = stated("success")
    success = Success(business=business if _filled(business) else (brief.success_criteria[0] if brief.success_criteria else TextFact()),
                      ml_metric=brief.primary_metric if brief.primary_metric.known else TextFact())
    fields = {"decision": decision, "decision_owner": stated("decision_owner"), "error_costs": costs, "capacity": capacity,
              "success": success, "baseline_today": stated("baseline"), "horizon": brief.horizon}
    open_fields: list[OpenField] = []

    def own(field: str, text: str, why: str, owner: str) -> None:
        open_fields.append(OpenField(field=field, text=text, why=why, owner=owner))

    if not _filled(decision):
        own("decision", "Which decision does each prediction feed?", "The decision sets what a good prediction is worth.", "the business owner")
    if not _filled(fields["decision_owner"]):
        own("decision_owner", "Who owns that decision?", "The owner says what each mistake costs and signs off the success bar.", "the person")
    if not (_filled(costs.miss) or _filled(costs.false_alarm)):
        own("error_costs", "What does a false alarm cost, what does a miss cost, and which is worse?",
            "The costs choose the operating point.", "the business owner")
    if fam in QUEUE_FAMILIES and capacity.k is None:
        own("capacity", "How many cases can the team act on a day, a week or a month?",
            "A team of K a period is judged by precision at K, and its threshold is the K-th score.", "the team that acts on the cases")
    if not (_filled(success.business) or _filled(success.ml_metric)):
        own("success", "Which business number says the work helped, and which model metric stands in for it?",
            "The success bar is agreed before building.", "the business owner")
    if not _filled(fields["baseline_today"]):
        own("baseline_today", "How is it done today, and how well does that work?", "A model is worth building only if it beats today's way.",
            "the team doing the work today")
    if not _filled(brief.horizon):
        own("horizon", "How far ahead does each prediction look?", "The horizon decides which rows are history and which are future.", "the person")
    return {**fields, "open_fields": open_fields[:20]}


CONTRACTS["problem_card"] = ProblemCard  # its schema is published with the others
