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
    "forecasting": re.compile(r"\b(forecast\w*|demand (per|for|each)|how (many|much) .{0,40}\b(tomorrow|(next|coming|following) (day|week|month)))", re.I),
    "time_to_event": re.compile(r"\b(how long (until|before)|(lifetime(?! value)|tenure|lifespan|how long (\w+ ){0,4}stick around|survival|how (long|many (days|weeks|months|years)) (\w+ ){0,6}(keep|stay|remain|last|stick))\b.{0,100}\b((do not|don'?t) know (their|the|its) end|(has|have)(n'?t| not) (yet )?(cancell?ed|churned|left|failed|ended)|(still|not yet) (active|with us|subscribed|running|cancell?ed|churned|left)|censor\w*)|((has|have)(n'?t| not) (yet )?(cancell?ed|churned|left|failed|ended)|censor\w*)\b.{0,100}\b(lifetime(?! value)|tenure|lifespan|how long|survival)|how long (\w+ ){0,4}stick around|how long (will|does|do|would|did) (\w+ ){1,4}(stay|last|survive|remain|keep \w+)\b|time (to|until|before) (\w+ ){0,3}(churn|fail|leave|cancel|quit|die|default)"
                                r"(s|ed|ing)?\b|time (to|until|before) (\w+ ){0,3}breaks? down|time to failure|survival analysis|time[- ]to[- ]event)", re.I),
    "ranking": re.compile(r"\b(rank\w* (items|products|results|documents|offers|listings)|(products|items|articles|songs|movies|offers) (they|each \w+|he|she)('re| are| is)? (most )?likely to (buy|click|watch|read|want)|personali[sz]\w* (the |our )?(carousel|feed|shelf|homepage|recommendations?|picks)|picked for you|recommender|recommend\w* (items|products|content|movies|songs|offers)|"
                          r"recommendations? (system|engine)|which items? to show)", re.I),
    "segmentation": re.compile(r"\b(segment (our|the) (customers|users|stores)|customer segmentation|cluster\w* (our|the) (customers|users|stores)|"
                               r"group (our|the|similar) (customers|users|stores))", re.I),
    "uplift": re.compile(r"\b(uplift|treatment effect|incremental (effect|impact|lift|sales|revenue)|persuad\w*|who (to|should we) target|"
                        r"(would|will|it) make (them |each |the customer )?(buy|stay|convert|respond|renew)|not (those|the ones) who (buy|would buy|renew) anyway|would have (bought|renewed|stayed|converted|signed up|booked) anyway|(not|rather than) (just |only )?(those|the ones|people|customers|users) who('d| would| will)? (buy|renew|stay|convert)\w* (anyway|regardless)|(driven|caused|changed) by the (offer|discount|campaign|coupon|treatment)|(offer|discount|coupon|voucher|campaign|promotion|email)s?\b.{0,80}\b(changes?|sways?) (their|his|her) mind|difference the (offer|discount|campaign|coupon) makes)", re.I),
    # 20.10: three families the words alone named nowhere
    "streams": re.compile(r"\b(sensor\w*|telemetry|readings? (from|of|over)|event streams?|predictive maintenance|((early[- ]warning|(log|record|measure|track)\w* (the )?(temperature|vibration|pressure|voltage|rpm))\b.{0,100}\b(machines?|units?|pumps?|compressors?|engines?|turbines?|devices?|equipment|fleet|plant|line)|(machines?|units?|pumps?|compressors?|engines?|turbines?|devices?|equipment|fleet|plant|line)\b.{0,100}\b(early[- ]warning|(log|record|measure|track)\w* (the )?(temperature|vibration|pressure|voltage|rpm)))|"
                          r"(alert|warn\w*) (before|ahead of) (a |the |each )?\w+ (fails?|breaks?( down)?|goes down))", re.I),
    "entity_tables": re.compile(r"\b((several|multiple|related|linked|separate) (tables|files)|(order|purchase|transaction|visit) history,? (and |plus |with )?(their |the )?(support )?(tickets|returns|addresses|address data|payments|visits|events)|(goes?|going|went) (the )?(next |following )?\d+ days without (an? )?(order|purchas|visit|log)\w*|plus the (orders|events|transactions|tickets)|(no|zero|not \w+ an?|without an?) (new )?(orders?|purchases?|visits?|logins?|payments?) (in|within|during|over) the (next|following|coming) \d+ (days|weeks)|(for|per) each (customer|user|account|member) (on|at|per) (each|every) (date|cutoff|month[- ]end)|at (several|each|every) (month[- ]end|cutoff) dates?|from (their|its|the) \w+, \w+,? and \w+|"
                                r"from (their|its) (orders|events|transactions|visits|purchases|tickets|payments) and (their |its )?\w+)\b", re.I),
    "text_or_images": re.compile(r"\b(from (its|their|the) (text|wording|words|images?|photos?|pictures?|scans?)|reads? (the )?(note|ticket|message|document|report|email)s? itself|from what (the|each|a) (note|ticket|message|document|report) says|(long|free)[- ]text (notes?|tickets?|messages?|reports?|descriptions?)|read(ing)? (the|its|their) "
                                 r"(text|notes?|images?|photos?)|(classify|label|tag|sort|recogni[sz]e) (the |each |our )?(images?|photos?|pictures?|scans?))\b", re.I),
}
LABEL_WORDS = (  # what a person says about labels, in the order tried (a "no" first: "no file of known cases" is none)
    ("none", re.compile(r"\b(nothing is labell?ed|no labell?s?\b|not labell?ed|none (are|is) (labell?ed|confirmed)|no (known|confirmed) cases|"
                        r"no (history|record|list|file|log) of (any )?(confirmed|known|labell?ed|past|caught) |without (any )?(confirmed|known|labell?ed) (cases|frauds?)|"
                        r"(have|has) (not|never) (yet )?(confirmed|labell?ed|caught) (any|a single)|"
                        r"unlabell?ed|don'?t have (any )?(labell?s?|confirmed|known)|do not have (any )?(labell?s?|confirmed|known))|^\s*(no|none|nope)\b", re.I)),
    ("file", re.compile(r"\b(a (file|list|sample) of (the )?(known|confirmed|found)|known (frauds?|cases)|confirmed (frauds?|cases)|"
                        r"a file lists (the )?(known|confirmed|cases|frauds?)|some are confirmed)", re.I)),
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
                counted = family == "forecasting" and task in ("binary", "multiclass")
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


def problem_card(brief: ProblemBrief, *, words: str = "", data: Optional[Mapping[str, Any]] = None,
                 family: Optional[str] = None) -> ProblemCard:
    """The card of a brief. ``words``: what the person wrote (the problem and their answers), read for the family.
    ``data``: what the files say, when there are files: ``label_file`` (a file of known cases joins the table),
    ``cases_only`` (the only table is a list of known cases), ``tables`` (how many tables), ``columns`` (the table's
    columns are known: an outcome is then a column, or for entity tables and streams the words), ``text_only`` (the table's
    only input is a long text: read through an encoder). ``family``: one the person
    stated (a host's chosen problem kind), which no word overrides."""
    data = dict(data or {})
    labels = _labels(brief, data, words)
    if family is not None and family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}: one of {', '.join(FAMILIES)}")
    stated_family = family
    family, in_play = _family(brief, words, labels, stated_family, bool(data.get("text_only")))
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
    return ProblemCard(brief=brief, family=family, in_play=in_play, labels=labels, outcome=outcome, questions=questions,
                       ready=not blocking, blocking=blocking)


CONTRACTS["problem_card"] = ProblemCard  # its schema is published with the others
