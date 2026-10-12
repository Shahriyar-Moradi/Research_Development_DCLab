"""Understand a problem from its description, before any data is analysed (package 17.2).

Input: the description (a sentence, a page, a challenge's text) and, when files exist, their names and column names
(never values). Output: a :class:`~dclab_core.contracts.ProblemBrief`. Every field says how it is known: *stated* facts
quote the description, *inferred* ones give their reason, the rest stay *unknown* and become open questions, ranked by
how much the answer would change the plan.

A wrong value is worse than an unknown one: a prediction moment read from the outcome ("whether a purchase is made at
checkout") would let everything up to the outcome into the model. So every rule here is narrow, works sentence by
sentence, and gives up rather than guess:

* a moment inside the outcome's clause (after "whether", "which", "if") is never a moment, nor is "at the moment"
  (currently), nor a "when ..." clause unless its verb starts the process (placed, created, submitted, ...);
* a column is the target only when the sentence binds it ("X is the label", "the target is X", "... in X");
* a review capacity needs a review verb ("call the top 500"), an error cost needs an error ("a wrong route costs").

The rules work without a model. A host's :class:`~dclab_core.advisor.Advisor` may propose fields the rules left
unknown; a proposal is kept only when its quote is the person's whole words (two words or eight characters at least),
the quote supports the value, and the value is valid for the field. It is then the advisor's inference, never the
person's statement. Long input (a pasted PDF) is clipped, never an error.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Optional

from dclab_core.advisor import Advisor
from dclab_core.agents import Role, run_role
from dclab_core.contracts import (MAX_TEXT, Constraint, CoverageFact, Deliverable, Fact, LabelSourceFact, NamesFact,
                                  ProblemBrief, Question, TaskFact, TextFact)

Files = Mapping[str, Sequence[str]]

QUOTE_CHARS = 300          # a quote is a phrase or a sentence, at most this long (the Home agent's limit)
MAX_CONSTRAINTS, MAX_DELIVERABLES, MAX_SUCCESS = 30, 50, 20
UNIT_NOUNS = ("ride", "order", "ticket", "transaction", "payment", "customer", "subscriber", "claim", "loan",
              "account", "user", "session", "machine", "patient", "delivery", "trip", "visit", "application",
              "invoice", "product", "store", "shipment", "call", "message", "email", "review", "listing", "booking")
ENTITY_WORDS = {
    "courier": ("biker", "courier", "rider", "driver"), "biker": ("biker", "courier", "rider"),
    "rider": ("rider", "biker", "courier"), "driver": ("driver",),
    "customer": ("customer", "client"), "client": ("client", "customer"), "subscriber": ("subscriber",),
    "user": ("user",), "merchant": ("merchant", "vendor", "store"), "vendor": ("vendor", "merchant"),
    "restaurant": ("restaurant",), "store": ("store", "shop"), "patient": ("patient",), "device": ("device",),
    "machine": ("machine",), "account": ("account",), "seller": ("seller",), "doctor": ("doctor",),
}
TIME_COLUMN = re.compile(r"(^|_)(date|time|timestamp|month|day|week|year|created|updated)($|_)|_at$", re.I)
ID_COLUMN = re.compile(r"(^id$|_id$|_ref$|^id_|_key$)", re.I)

# Tried in this order, on the goal sentence first: specific cues before generic ones, so "detect fraudulent rides; one
# of 3 analysts reviews each" stays binary and "which customers will churn ... how much history matters" too.
TASK_RULES: tuple[tuple[str, str, str], ...] = (
    ("binary", r"\b(detect|detects|flag|flags|identify|identifies|spot|spots|find|finds)\b[^.;]*\b(fraud|fraudulent|forged|fake|anomal\w*|abuse|default\w*)\b",
     "asks whether each case is one of the rare cases or not"),
    ("binary", r"\bwhich [a-z ]+ (will|would) (cancel|churn|leave|default|buy|convert|respond|return|click|pay|upgrade)\b", "asks yes or no for each case"),
    ("binary", r"\bwhether\b[^.;]*", "asks yes or no for each case"),
    ("multiclass", r"\b(one|each) of (\d+|several|many|the) [a-z]+ (teams|classes|categories|types|groups|labels|departments|queues)\b",
     "chooses one of several classes"),
    ("multiclass", r"\b(route|classify|categori[sz]e|assign) (each|every|incoming)\b[^.;]*", "chooses one of several classes"),
    ("regression", r"\bhow (many|much|long)\b[^.;]*", "asks for a quantity"),
    ("regression", r"\b(estimate|predict|forecast) (the )?(amount|price|value|number|duration|demand|revenue|sales)\b[^.;]*", "asks for a quantity"),
    ("binary", r"\b(fraud|churn|default)\b[^.;]*", "names a yes-or-no outcome"),
)
OUTCOME_CLAUSE = re.compile(r"\b(whether|which|if|who)\b[^,;:]*", re.I)
START_VERBS = (r"(?:place|places|placed|create|creates|created|submit|submits|submitted|open|opens|opened|book|books|booked|"
               r"sign(?:s|ed)? up|log(?:s|ged)? in|appl(?:y|ies|ied)|request|requests|requested|start|starts|started|"
               r"check(?:s|ed)? in)")
MOMENTS = [re.compile(p, re.I) for p in (
    r"\bat the (?:moment|time|point) (?:it|they|a|an|the|each|of)\b[^,.;]+",
    r"\bwhen (?:they|the customer|the user|a customer|a user|it|the order|the ticket|an order|a ticket|the patient|a patient)\s+"
    r"(?:is |are )?" + START_VERBS + r"\b[^,.;]*",
    r"\bon the first day of [^,.;]+",
    r"\bat the (?:start|beginning|end) of (?:each|every) [^,.;]+",
    r"\bat (?:checkout|authori[sz]ation(?: time)?|booking|sign[- ]?up|admission|application time)\b",
    r"\bin real time\b",
)]
# set apart from the outcome by a comma or a scoring verb: "..., scored before the renewal call", "..., run every night"
MOMENT_SET_APART = re.compile(r"(?:,\s*(?:and\s+)?|\b(?:scored|predicted|computed|checked|evaluated|refreshed|run|estimated)\s+)"
                              r"((?:\d+|one|two|three|a)\s+(?:hours?|days?|weeks?|months?)\s+before\s+[^,.;]+|before (?:the|a|an|it|each) [^,.;]+"
                              r"|every (?:night|morning|evening|day|week|month|monday|tuesday|wednesday|thursday|friday)\b[^,.;]*"
                              r"|once a (?:day|week|month)\b[^,.;]*)", re.I)
HORIZON = re.compile(r"\b(within the next \d+ (hours?|days?|weeks?|months?|years?)|in the (next|coming) \d+ (hours?|days?|weeks?|months?)"
                     r"|next (day|week|month|quarter|year)|\d+ (days?|weeks?|months?) (ahead|later))\b", re.I)
GOAL_VERBS = re.compile(r"\b(build|develop|detect|predict|estimate|route|find|flag|classify|forecast|want to know|identify|rank|"
                        r"recommend|segment|decide|choose|price|score)\b", re.I)
ACTION = re.compile(r"\b(so (that )?(the|our|a|an) [a-z ]+ (can|could|will|may) [^.;]+|to (show|alert|warn|offer|send|notify|block|call) [^,.;]+)",
                    re.I)
CONFIDENTIAL = re.compile(r"\b(must not be (disclosed|distributed|shared)[^.]*|confidential[^.]*|do not share[^.]*|under (an )?nda[^.]*"
                          r"|personal data[^.]*)", re.I)
LATENCY = re.compile(r"\b(answer|answers|respond|responds|reply|replies|return|returns|run|runs|score|scores|serve|serves)\b[^.;]{0,40}?"
                     r"\b((?:under|below|within|in less than|less than|in) \d+ ?(?:ms|milliseconds?|seconds?))\b", re.I)
REVIEW_VERB = re.compile(r"\b(call|calls|review|reviews|investigate|investigates|check|checks|contact|contacts|handle|handles|inspect|inspects|"
                         r"audit|audits|examine|examines)\b", re.I)
CAPACITY = re.compile(r"\b(top \d[\d,]*|\d[\d,]* (?:alerts?|cases?|rides?|calls?|tickets?|reviews?|orders?|transactions?|claims?)? ?(?:a|per) "
                      r"(?:day|week|month))\b", re.I)
COST = re.compile(r"\bcosts?\b", re.I)
ERROR_WORDS = re.compile(r"\b(wrong|error|errors|mistake|missed|miss|lost|false|each call|a call|per call|handoff|penalty|fine|refund)\b", re.I)
SUCCESS = re.compile(r"\b(within \d+ [a-z]+ for (at least )?\d+ ?%[^.;,]*|at least \d+ ?%[^.;,]*|(precision|recall|accuracy|auc|f1|error) "
                     r"(of|above|below|at least|under) [\d.]+ ?%?[^.;,]*)", re.I)
METRIC = re.compile(r"\b(judged|evaluated|measured|scored|ranked) (on|by) (the )?(macro[- ]f1|weighted f1|f1|roc[- ]auc|pr[- ]auc|auc|"
                    r"average precision|precision|recall|accuracy|mae|rmse|mape|r2|log[- ]loss|ndcg@?\d*)\b", re.I)
RUBRIC_ITEM = re.compile(r"([A-Za-z][A-Za-z ,/+-]*?)\s*\((\d+(?:\.\d+)?)\s*(?:pts?|points?)\.?\)", re.I)
DELIVERABLE_NAMES = (
    ("feature extraction", r"feature"), ("api", r"\bapi\b|flask|fastapi|server|endpoint"), ("docker", r"docker"),
    ("source control", r"source control|\bgit\b|version control"), ("report", r"\breport\b"), ("tests", r"\btests?\b"),
    ("notebook", r"notebook"), ("eda", r"\beda\b|exploratory"), ("modelling", r"model(l)?ing|models?\b"),
)
LABEL_FILE_WORDS = re.compile(r"\b(sample|known|labell?ed|labels?|forged|examples?|confirmed|ground truth)\b", re.I)
PARTIAL_WORDS = re.compile(r"\b(a sample of [^.,;]+|a subset of [^.,;]+|some of the [^.,;]+|partial[^.,;]*|known [^.,;]+)", re.I)

FIELD_ORDER = ("goal", "target", "prediction_moment", "task", "unit", "label_coverage", "action", "entities", "horizon")
QUESTIONS = {
    "goal": (1, "What decision or action should the model support, and for whom?"),
    "target": (1, "What exactly is predicted for each row: which column, or how is the label defined?"),
    "prediction_moment": (1, "When is the prediction made, and what is already known at that moment? (Everything known "
                             "only later must stay out of the model.)"),
    "task": (2, "Is the answer a yes or no, one of several classes, or a quantity?"),
    "unit": (2, "What is one scored row: a transaction, a customer, an order, a ticket?"),
    "label_coverage": (2, "Do the labels cover every case, or only the cases someone found? (Unlabelled is not the same as negative.)"),
    "action": (2, "What happens with each prediction, and roughly what does a wrong one cost?"),
    "entities": (3, "Do rows belong to repeating entities (customers, couriers, machines) that a split must keep together?"),
    "horizon": (3, "How far ahead does the prediction look?"),
}
FACT_TYPES = {"goal": TextFact, "action": TextFact, "unit": TextFact, "task": TaskFact, "target_column": TextFact,
              "target_definition": TextFact, "label_source": LabelSourceFact, "label_coverage": CoverageFact,
              "prediction_moment": TextFact, "horizon": TextFact, "entities": NamesFact, "time_column": TextFact,
              "primary_metric": TextFact}


# ------------------------------------------------------------------------------------------------ text helpers


def _norm(text: str) -> str:
    return " ".join((text or "").split())


BULLET = re.compile(r"^\s*([-*•▪◦·]|\d+[.)]|[a-z][.)])\s+")


def _blocks(description: str) -> list[str]:
    """Lines joined back into their paragraphs or items: a line break ends a block only after a blank line, before a
    bullet, or after a line ending in punctuation (a soft wrap in the middle of a sentence is not a break)."""
    blocks: list[str] = []
    current = ""
    for line in (description or "").splitlines():
        if not line.strip():
            if current:
                blocks.append(current)
            current = ""
            continue
        if current and (BULLET.match(line) or re.search(r"[.!?:;]\s*$", current)):
            blocks.append(current)
            current = line
        else:
            current = f"{current} {line}" if current else line
    if current:
        blocks.append(current)
    return blocks


def _sentences(description: str) -> list[str]:
    """Sentences of the description (a pasted PDF's bullets, which have no full stops, are sentences too); each is a
    contiguous piece of the whitespace-normalised description, so a quote taken inside one is the person's words."""
    out = []
    for block in _blocks(description):
        for sentence in re.split(r"(?<=[.!?])\s+", _norm(block)):
            if sentence:
                out.append(sentence)
    return out


def _clip(text: str, n: int = QUOTE_CHARS) -> str:
    return text.strip()[:n].strip()


def _stated(value: Any, quote: str, source: str = "text") -> Fact:
    return Fact(value=value if not isinstance(value, str) else _clip(value, MAX_TEXT), status="stated",
                quote=_clip(quote), source=source)


def _inferred(value: Any, reason: str, source: str = "rule") -> Fact:
    return Fact(value=value if not isinstance(value, str) else _clip(value, MAX_TEXT), status="inferred",
                reason=_clip(reason, MAX_TEXT), source=source)


def _singular(word: str) -> str:
    word = word.lower()
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith("ses") or word.endswith("xes"):
        return word[:-2]
    return word[:-1] if word.endswith("s") and not word.endswith("ss") else word


def _columns(files: Files) -> list[str]:
    seen: list[str] = []
    for cols in files.values():
        for c in cols:
            if str(c) not in seen:
                seen.append(str(c))
    return seen


def _first_in(sentences: Sequence[str], pattern: re.Pattern) -> Optional[str]:
    for sentence in sentences:
        m = pattern.search(sentence)
        if m:
            return m.group(0).strip()
    return None


def quoted(quote: Any, text: str) -> bool:
    """The person's own words: whole words (at least two, or eight characters), at most QUOTE_CHARS, found in ``text``
    (the same rule as the R&D Home agent's)."""
    q = _norm(str(quote or "")).strip(" \"'“”‘’.,;:").lower()
    if not (len(q.split()) >= 2 or len(q) >= 8) or len(q) > QUOTE_CHARS:
        return False
    return re.search(r"(?<![\w])" + re.escape(q) + r"(?![\w])", _norm(text).lower()) is not None


def supports(quote: Any, value: Any) -> bool:
    """The value says something the quote says: they share a content word (compared as 5-letter stems)."""
    def stems(text: Any) -> set[str]:
        return {w[:5] for w in re.findall(r"[a-z0-9]{3,}", _norm(str(text)).lower())} - {
            "the", "and", "each", "every", "with", "that", "this", "which", "wheth"}
    return bool(stems(quote) & stems(value))


# ------------------------------------------------------------------------------------------------ the fields


def _label_files(sentences: Sequence[str], files: Files) -> list[tuple[str, str]]:
    out = []
    for name in files:
        for sentence in sentences:
            if name.lower() in sentence.lower() and LABEL_FILE_WORDS.search(sentence):
                out.append((name, sentence))
                break
    return out


def _main_file(files: Files, label_files: list[str]) -> Optional[str]:
    rest = [n for n in files if n not in label_files]
    return max(rest, key=lambda n: len(files[n])) if rest else None


def _goal_sentence(sentences: Sequence[str]) -> Optional[str]:
    return next((s for s in sentences if GOAL_VERBS.search(s)), None)


def _task(sentences: Sequence[str]) -> Fact:
    goal = _goal_sentence(sentences)
    pools: list[Sequence[str]] = ([[goal]] if goal else []) + [sentences]
    for pool in pools:
        for task, pattern, why in TASK_RULES:
            for sentence in pool:
                m = re.search(pattern, sentence, re.I)
                if m:
                    return _inferred(task, f'"{_clip(m.group(0), 160)}" {why}', source="text")
    return Fact.unknown()


def _goal(sentences: Sequence[str]) -> Fact:
    goal = _goal_sentence(sentences)
    return _stated(_clip(goal), goal) if goal else Fact.unknown()


def _unit(sentences: Sequence[str], files: Files, main: Optional[str]) -> Fact:
    for sentence in sentences:
        m = re.search(r"\bone row per ([a-z]+)(?: (?:per|and) ([a-z]+))?", sentence, re.I)
        if m:  # "one row per subscriber per month": the unit is a subscriber-month, and a subscriber repeats
            unit = _singular(m.group(1)) + (f"-{_singular(m.group(2))}" if m.group(2) else "")
            return _stated(unit, m.group(0))
    if main:
        stem = _singular(re.split(r"[._-]", main.rsplit("/", 1)[-1])[0])
        if stem in UNIT_NOUNS:
            return _inferred(stem, f"the main table is {main}", source="data")
    pattern = re.compile(r"\b(each|every|per) (incoming |new |single )?(?:[a-z]+ )?(" + "|".join(UNIT_NOUNS) + r")s?\b", re.I)
    for sentence in sentences:
        m = pattern.search(sentence)
        if m:
            return _stated(_singular(m.group(3)), m.group(0))
    return Fact.unknown()


def _bound_target(sentences: Sequence[str], columns: Sequence[str]) -> Optional[tuple[str, str]]:
    """(column, the words that bind it as the label), only when the sentence says so of that very column."""
    for sentence in sentences:
        for column in columns:
            c = re.escape(column)
            for pattern in (rf"(?<![\w]){c}(?![\w])\s+(?:column\s+)?(?:is|holds|contains|gives)\s+the\s+(?:label|target|outcome)\b",
                            rf"\bthe\s+(?:label|target|outcome)\s+(?:column\s+)?(?:is|=|:)\s+[`'\"]?{c}(?![\w])",
                            rf"\b(?:records?|stores?|holds?|keeps?|logs?)\b[^.;]*?\bin\s+[`'\"]?{c}(?![\w])"):
                m = re.search(pattern, sentence, re.I)
                if m and not ID_COLUMN.search(column):
                    return column, m.group(0)
    return None


def _target(sentences: Sequence[str], files: Files, task: Fact,
            label_files: list[tuple[str, str]]) -> tuple[Fact, Fact, Fact]:
    """(target column, target definition, label source)."""
    columns = [c for c in _columns(files) if not ID_COLUMN.search(c)]
    bound = _bound_target(sentences, columns)
    if bound:
        return _stated(bound[0], bound[1]), Fact.unknown(), _stated("column", bound[1])
    if task.value == "regression":  # "how many minutes" names its quantity; one named column carrying it is the target
        named = " ".join(sentences)
        for sentence in sentences:
            quantity = re.search(r"\bhow (?:many|much|long) ([a-z]+)", sentence, re.I)
            if not quantity:
                continue
            noun = _singular(quantity.group(1))
            carriers = [c for c in columns if noun in _singular(c.lower()) and re.search(rf"(?<![\w]){re.escape(c)}(?![\w])", named)]
            if len(carriers) == 1:
                why = f'the description asks "{_clip(quantity.group(0), 80)}" and names {carriers[0]}, the only column carrying {noun}'
                return (_inferred(carriers[0], why, source="text"), Fact.unknown(),
                        _inferred("column", f"{carriers[0]} holds the quantity asked for", source="text"))
    if label_files:
        name, sentence = label_files[0]
        return (Fact.unknown(),
                _inferred(f"listed in {name}", f"the labels are the cases listed in {name}; the main table has no label column"),
                _stated("file", sentence))
    return Fact.unknown(), Fact.unknown(), Fact.unknown()


UNMARKED_UNKNOWN = re.compile(
    r"\b(never (?:been )?(?:checked|reviewed|labell?ed|investigated|found|caught|detected|reported)|nobody (?:found|checked|reviewed)|"
    r"(?:un(?:marked|labell?ed|flagged|reported|confirmed|checked)) [a-z ]{0,30}?(?:may|might|could) (?:still |also )?be|"
    r"only (?:the )?(?:cases|frauds?|ones|positives|abuses?) (?:that )?(?:someone|we|the team|the review team|investigators?|reviewers?|analysts?) "
    r"(?:found|confirmed|caught|reviewed)|labels? (?:are|is) (?:partial|incomplete|a sample))", re.I)
# a sentence about the labels: what is marked, flagged, confirmed, found or a case (not "readings never checked by hand")
ABOUT_LABELS = re.compile(r"\b(labels?|labell?ed|marked|unmarked|flag(?:ged|s)?|unflagged|cases?|confirmed|frauds?|positives?|abuses?|"
                          r"found by|caught)\b", re.I)


def label_coverage(text: str) -> Optional[tuple[str, str]]:
    """("partial", the words) when a sentence about the labels says that unmarked rows are unknown rather than negative
    ("most rows were never checked, so an unmarked row may be a case", "unflagged claims may still be fraud"); None
    otherwise. Only sentences that speak of labels, cases or flags are read: "readings never checked by hand" says
    nothing about labels. Words, never a guess from the data."""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text or ""):
        m = UNMARKED_UNKNOWN.search(sentence)
        if m and ABOUT_LABELS.search(sentence):
            return "partial", m.group(0)
    return None


def _coverage(label_source: Fact) -> Fact:
    if label_source.value == "file" and label_source.quote:
        m = PARTIAL_WORDS.search(label_source.quote)
        if m:
            return _stated("partial", m.group(0))
    return Fact.unknown()


def _entities(text: str, files: Files, unit: Fact) -> Fact:
    lowered = text.lower()
    ids = [c for c in _columns(files) if ID_COLUMN.search(c)]
    unit_word = str(unit.value or "")
    found: list[str] = []
    words: list[str] = []
    for word, stems in ENTITY_WORDS.items():
        if not re.search(rf"\b{word}s?\b", lowered):
            continue
        for column in ids:
            low = column.lower()
            # the unit itself is not an entity when each row is one of it (a ticket_id per ticket); a compound unit
            # ("subscriber-month") repeats its first part, which is then an entity
            if any(stem in low for stem in stems) and (unit_word != low.removesuffix("_id")) and unit_word not in low.split("_") \
                    and column not in found:
                found.append(column)
                words.append(word)
    if found:
        return _inferred(found[:20], "the description names " + ", ".join(f"{w}s" for w in words) + " and the data has "
                         + ", ".join(found) + ": rows of one entity must stay on one side of a split", source="data")
    return Fact.unknown()


def _time_column(text: str, files: Files, main: Optional[str]) -> Fact:
    columns = list(files.get(main, [])) if main else _columns(files)
    timeish = [c for c in columns if TIME_COLUMN.search(str(c))]
    if not timeish:
        return Fact.unknown()
    mentioned = [c for c in timeish if re.search(rf"\b{re.escape(c.replace('_', ' '))}\b|\b{re.escape(c)}\b", text, re.I)]
    pick = (mentioned or timeish)[0]
    why = "the description mentions it" if mentioned else ("the only date or time column" if len(timeish) == 1
                                                            else f"the first of the date or time columns ({', '.join(timeish[:4])})")
    return _inferred(pick, why, source="data")


def _outside_outcome(sentence: str, start: int) -> bool:
    return not any(m.start() <= start < m.end() for m in OUTCOME_CLAUSE.finditer(sentence))


def _moment(sentences: Sequence[str]) -> Fact:
    for sentence in sentences:
        for pattern in MOMENTS:
            for m in pattern.finditer(sentence):
                if _outside_outcome(sentence, m.start()):
                    return _stated(m.group(0).strip(), m.group(0))
        m = MOMENT_SET_APART.search(sentence)
        if m and _outside_outcome(sentence, m.start(1)):
            return _stated(m.group(1).strip(), m.group(1))
    return Fact.unknown()


def _action(sentences: Sequence[str]) -> Fact:
    for sentence in sentences:
        m = ACTION.search(sentence)
        if m and _outside_outcome(sentence, m.start()):
            return _stated(m.group(0).strip(), m.group(0))
    return Fact.unknown()


def _constraints(sentences: Sequence[str]) -> list[Constraint]:
    out: list[Constraint] = []

    def add(kind: str, text: str) -> None:
        if len(out) < MAX_CONSTRAINTS and not any(c.kind == kind and c.text == _clip(text, MAX_TEXT) for c in out):
            out.append(Constraint(kind=kind, text=_clip(text, MAX_TEXT), quote=_clip(text)))

    for sentence in sentences:
        m = CONFIDENTIAL.search(sentence)
        if m:
            add("confidential", m.group(0))
        m = LATENCY.search(sentence)
        if m:
            add("latency", m.group(2))
        m = CAPACITY.search(sentence)
        if m and REVIEW_VERB.search(sentence):
            add("review_capacity", m.group(0))
        if COST.search(sentence) and ERROR_WORDS.search(sentence):
            add("budget", sentence)
    return out


def _deliverables(sentences: Sequence[str]) -> list[Deliverable]:
    out: dict[str, Deliverable] = {}

    def name_of(phrase: str) -> Optional[str]:
        for name, pattern in DELIVERABLE_NAMES:
            if re.search(pattern, phrase, re.I):
                return name
        return None

    for sentence in sentences:
        for m in RUBRIC_ITEM.finditer(sentence):
            phrase = m.group(1).split(",")[-1].split(":")[-1].strip()
            name = name_of(phrase)
            if name and name not in out:
                out[name] = Deliverable(name=name, points=float(m.group(2)), quote=_clip(m.group(0)))
    for sentence in sentences:
        if re.search(r"\b(deliver|submit|provide|write|include)\b", sentence, re.I):
            for name, pattern in DELIVERABLE_NAMES:
                if name not in out and re.search(pattern, sentence, re.I):
                    out[name] = Deliverable(name=name, points=None, quote=_clip(sentence))
    return list(out.values())[:MAX_DELIVERABLES]


def _success(sentences: Sequence[str]) -> list[Fact]:
    out = []
    for sentence in sentences:
        for m in SUCCESS.finditer(sentence):
            out.append(_stated(m.group(0).strip(), m.group(0)))
    return out[:MAX_SUCCESS]


def _questions(fields: dict[str, Fact], constraints: list[Constraint]) -> list[Question]:
    questions = []
    for field in FIELD_ORDER:
        if field == "target":
            if fields["target_column"].known or fields["target_definition"].known:
                continue
        elif fields[field].known:
            continue
        if field == "label_coverage" and fields["label_source"].value != "file":
            continue
        if field == "action" and any(c.kind in ("budget", "review_capacity") for c in constraints):
            continue
        if field == "horizon":
            continue  # asked by the kinds that need one (forecasting, churn windows), not by default
        impact, text = QUESTIONS[field]
        questions.append(Question(field=field, text=text, impact=impact))
    return sorted(questions, key=lambda q: (q.impact, FIELD_ORDER.index(q.field)))


# ------------------------------------------------------------------------------------------------ the advisor

ADVISABLE = ("goal", "action", "unit", "task", "target_column", "prediction_moment", "horizon", "primary_metric")


def _analyst_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    description, text, files = inputs["description"], inputs["text"], inputs["files"]
    paragraphs = [_clip(_norm(p), 2000) for p in re.split(r"\n\s*\n", description) if p.strip()][:20]
    return {"description_paragraphs": paragraphs or [_clip(text, 2000)],
            "files": {_clip(n, 100): [_clip(str(c), 200) for c in cols][:50] for n, cols in list(files.items())[:10]},
            "fields_to_propose": list(inputs["unknown"])}


def _analyst_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Fact]:
    """Keep a proposal only for an unknown field, with the person's whole words that support a valid value."""
    text, columns = inputs["text"], set(_columns(inputs["files"]))
    kept = {}
    for field in inputs["unknown"]:
        item = answer.get(field)
        if not isinstance(item, Mapping):
            continue
        value, quote = item.get("value"), item.get("quote")
        if not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
            continue
        if not quoted(quote, text):
            continue  # not the person's whole words: dropped
        if field == "task":
            if value not in ("binary", "multiclass", "regression"):
                continue
        elif field == "target_column":
            if value not in columns:
                continue  # a target must be a column the files have
        elif not supports(quote, value):
            continue  # the quote must say what the value says
        fact = FACT_TYPES[field](value=value, status="inferred", source="advisor",
                                 reason=f'proposed by the advisor from "{_clip(_norm(quote), 200)}"')
        kept[field] = Fact(**fact.model_dump())
    return kept


PROBLEM_ANALYST = Role(
    name="problem_analyst",
    description="Reads the description for the brief fields the rules left unknown; every value with the person's words.",
    evidence=_analyst_evidence, parse=_analyst_parse, fallback=lambda inputs: {}, merge=lambda base, new: {**base, **new})


def _advice(advisor: Optional[Advisor], description: str, text: str, files: Files, fields: dict[str, Fact]) -> dict[str, Fact]:
    unknown = [f for f in ADVISABLE if not fields[f].known]
    if not advisor or not unknown or not text:
        return {}
    inputs = {"description": description, "text": text, "files": files, "unknown": unknown}
    return run_role(PROBLEM_ANALYST, advisor, inputs).output


# ------------------------------------------------------------------------------------------------ the brief checker


def _checker_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    base = _analyst_evidence({**inputs, "unknown": []})
    base.pop("fields_to_propose", None)
    base["rules_read"] = {field: {"value": _clip(str(fact.value), 300),
                                  "basis": _clip(_norm(fact.quote or fact.reason or ""), 300)}
                          for field, fact in inputs["known"].items()}
    return base


def _checker_parse(answer: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[Question]:
    """A disagreement is kept only for a field the rules read, with a different value the person's own words support;
    it becomes an open question with both readings as options, and the rules' value stays until the person answers."""
    items = answer.get("disagree") or []
    if not isinstance(items, list):
        raise ValueError("disagree must be a list")
    text, columns, known = inputs["text"], set(_columns(inputs["files"])), inputs["known"]
    out, asked = [], set()
    for item in items[:10]:
        if not isinstance(item, Mapping):
            continue
        field, value, quote = item.get("field"), item.get("value"), item.get("quote")
        if field in asked:
            continue  # one question per field: the first valid reading
        if field not in known or not isinstance(value, str) or not value.strip() or len(value) > MAX_TEXT:
            continue
        ruled = str(known[field].value)
        if _norm(value).lower() == _norm(ruled).lower() or not quoted(quote, text):
            continue
        if field == "task" and value not in ("binary", "multiclass", "regression"):
            continue
        if field == "target_column" and value not in columns:
            continue
        if field not in ("task", "target_column") and not supports(quote, value):
            continue
        asked.add(field)
        name = {"target_column": "target"}.get(field, field)
        impact = QUESTIONS.get(name, (2, ""))[0]
        out.append(Question(field=name, impact=impact, options=[_clip(ruled, 200), _clip(value, 200)],
                            text=_clip(f'The rules read {field} as "{_clip(ruled, 120)}", but the description may say "{_clip(value, 120)}" '
                                       f'("{_clip(_norm(quote), 200)}"). Which is right?', 1000)))
    return out


BRIEF_CHECKER = Role(
    name="brief_checker",
    description="Reads the description against what the rules read; a different reading the person's words support becomes "
                "an open question with both options. It never changes a field.",
    evidence=_checker_evidence, parse=_checker_parse, fallback=lambda inputs: [])


def _disagreements(advisor: Optional[Advisor], description: str, text: str, files: Files, fields: dict[str, Fact]) -> list[Question]:
    known = {f: fields[f] for f in ADVISABLE if fields[f].known and fields[f].source != "advisor"}
    if not advisor or not known or not text:
        return []
    return run_role(BRIEF_CHECKER, advisor, {"description": description, "text": text, "files": files, "known": known}).output


# ------------------------------------------------------------------------------------------------ entry point


def understand(description: str, files: Optional[Files] = None, advisor: Optional[Advisor] = None,
               checker: Optional[Advisor] = None) -> ProblemBrief:
    """Read a problem's description (and its files' column names) into a :class:`ProblemBrief`. ``advisor``: the problem
    analyst, for fields the rules left unknown; ``checker``: the brief checker, whose disagreements with the rules become
    open questions (never a changed field)."""
    files = {str(k): [str(c) for c in v] for k, v in (files or {}).items()}
    sentences = _sentences(description)
    text = " ".join(sentences)
    labels = _label_files(sentences, files)
    main = _main_file(files, [n for n, _ in labels])
    task = _task(sentences)
    target_column, target_definition, label_source = _target(sentences, files, task, labels)
    unit = _unit(sentences, files, main)
    metric_match = next((m for m in (METRIC.search(x) for x in sentences) if m), None)
    horizon = _first_in(sentences, HORIZON)
    fields: dict[str, Fact] = {
        "goal": _goal(sentences),
        "action": _action(sentences),
        "unit": unit,
        "task": task,
        "target_column": target_column,
        "target_definition": target_definition,
        "label_source": label_source,
        "label_coverage": _coverage(label_source),
        "prediction_moment": _moment(sentences),
        "horizon": _stated(horizon, horizon) if horizon else Fact.unknown(),
        "entities": _entities(text, files, unit),
        "time_column": _time_column(text, files, main),
        "primary_metric": (_stated(re.sub(r"[\s-]+", "_", metric_match.group(4).lower()), metric_match.group(0))
                           if metric_match else Fact.unknown()),
    }
    disputes = _disagreements(checker, description or "", text, files, fields)  # before the analyst: only the rules' reading
    fields.update(_advice(advisor, description or "", text, files, fields))
    constraints = _constraints(sentences)
    digest = hashlib.sha256(json.dumps({"text": text, "files": files}, sort_keys=True).encode()).hexdigest()
    return ProblemBrief(source_digest=digest, constraints=constraints, deliverables=_deliverables(sentences),
                        success_criteria=[f.model_dump() for f in _success(sentences)],
                        open_questions=sorted(disputes + _questions(fields, constraints), key=lambda q: q.impact)[:20],
                        **{name: fact.model_dump() for name, fact in fields.items()})
