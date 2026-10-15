"""The understanding conversation (package 20.2): the next question chosen by what it decides, answers read into the
problem card, contradictions shown, the card shown to confirm.

A conversation is the problem's words, the tables' column names (when there are tables) and the person's answers so
far, keyed by the card field each answered. :func:`card_for` builds the :class:`~dclab_core.card.ProblemCard` from
them; :func:`next_question` picks the question that unblocks the most (the files first, then whether cases are
labelled, the kind of answer wanted, the outcome, the moment), with a contradiction between the words and the data
asked before anything else; :func:`read_answer` reads an answer into card fields by rules: "it is anomaly detection"
states the family whatever was asked; "nothing is labelled" states the labels. A model role (:data:`READER`) may read
an answer the rules could not, behind a validator (its value in the field's vocabulary, quoted from the answer); it
is off until it has been measured against the rules (``dclab_rnd.agent_eval.conversation``).

:func:`converse` plays a whole written conversation (a harness, or a host replaying one) in at most ``limit``
questions, and :func:`summary` is what the person confirms.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, Callable, Optional

from dclab_core.agents import Role
from dclab_core.card import FAMILIES, LABEL_WORDS, ProblemCard, problem_card
from dclab_core.contracts import ProblemBrief, TextFact

ORDER = ("data", "labels", "family", "outcome", "prediction_moment")  # the question that unblocks the most comes first
LIMIT = 5
# the kind of answer wanted, read only from an answer to that question; tried in this order (a yes or no before
# "how many": "how many are late, a yes or no for each" is a classification)
FAMILY_ANSWERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("classification", re.compile(r"\b(a yes or (a )?no|yes or no|yes/no|whether|a class\b|which (class|category|team|type)|classification)\b", re.I)),
    ("forecasting", re.compile(r"\b(forecast\w*|(next|coming) (days?|weeks?|months?)|how many .{0,30}\b(tomorrow|next)\b)", re.I)),
    ("time_to_event", re.compile(r"\b(how long (until|before)|time (to|until|before) \w+|when (it|they|each) will (fail|leave|cancel|churn)|"
                                 r"(days|weeks|months|time) (until|before) (\w+ ){0,3}(fails?|leaves?|cancels?|churns?|quits?|ends?))\b", re.I)),
    ("streams", re.compile(r"\b((an )?alerts? (before|ahead of) (a |the |each )?\w+ (fails?|breaks?|goes down)|from (the |its )?(sensor\w*|readings)|"
                           r"as (the )?(readings|events) arrive)\b", re.I)),
    ("anomaly", re.compile(r"\b(anomal\w*|outlier\w*|unusual (rows|ones|orders|cases|behaviou?r|transactions)|the unusual ones|odd ones)\b", re.I)),
    ("rare_event", re.compile(r"\b(rare (events?|cases)( we (learn|know))?|learn from the confirmed)\b", re.I)),
    ("ranking", re.compile(r"\b(rank\w*|recommend\w*|an order of (items|products|results))\b", re.I)),
    ("segmentation", re.compile(r"\b(segment\w*|cluster\w*|groups?\b)", re.I)),
    ("uplift", re.compile(r"\b(uplift|who (to|should we) (target|contact|treat)|effect of (the )?(offer|discount|campaign|treatment)|"
                          r"(would|will) make (them |each )?(buy|stay|convert|respond|renew))\b", re.I)),
    ("regression", re.compile(r"\b(a number|how (much|many|long)|the (amount|price|minutes|duration|value)|regression)\b", re.I)),
)
# a family said outright, whatever was asked: "it is anomaly detection", the morning's answer to the outcome question
EXPLICIT = re.compile(r"\b(?:it'?s|it is|this is|that'?s|we want|i want|we need)\s+(?:an?\s+|to\s+)?(?:build\s+|do\s+)?(?:an?\s+)?"
                      r"(anomaly|outlier|forecast(?:ing)?|classification|regression|segmentation|clustering|ranking|recommendation|uplift|"
                      r"survival|time[- ]to[- ]event)\b", re.I)
EXPLICIT_FAMILY = {"anomaly": "anomaly", "outlier": "anomaly", "forecast": "forecasting", "forecasting": "forecasting",
                   "classification": "classification", "regression": "regression", "segmentation": "segmentation",
                   "clustering": "segmentation", "ranking": "ranking", "recommendation": "ranking", "uplift": "uplift",
                   "survival": "time_to_event", "time to event": "time_to_event", "time-to-event": "time_to_event"}
NOT_SAID = re.compile(r"^\s*(i don'?t know|not sure|no idea|skip|later)\b", re.I)


def _said(value: str) -> TextFact:
    return TextFact(value=value[:400], status="stated", quote=value[:400], source="person")


def _named(text: str, columns: Optional[list[str]]) -> list[str]:
    """The columns an answer names, in the order it names them ("days_subscribed, with cancelled saying..." names the
    outcome first)."""
    found = {c: m.start() for c in columns or [] if (m := re.search(rf"(?<![\w]){re.escape(c)}(?![\w])", text))}
    return sorted(found, key=found.get)


def read_answer(field: str, text: str, columns: Optional[list[str]] = None) -> dict[str, str]:
    """What an answer says, by rules, as card fields (``family``, ``labels``, ``target_column``, ``target_definition``,
    ``prediction_moment``, ``unit``, ``action``). Empty when it says nothing the rules can read. The kind of answer is
    read from an answer to that question, or when said outright ("it is anomaly detection") to any other: a word in
    passing ("the customer segment", "the top ranked ones") never states it."""
    text = (text or "").strip()
    if not text or NOT_SAID.search(text):
        return {}
    out: dict[str, str] = {}
    explicit = EXPLICIT.search(text)
    if explicit:
        out["family"] = EXPLICIT_FAMILY[explicit.group(1).lower().replace("  ", " ")]
    elif field == "family":
        out.update(next(({"family": f} for f, pattern in FAMILY_ANSWERS if pattern.search(text)), {}))
    named = _named(text, columns)
    if field == "labels":
        if named or re.search(r"\bcolumn\b", text, re.I):  # "No file, but the is_fraud column marks them" is a column
            out["labels"] = "column"
        else:
            said = next((value for value, pattern in LABEL_WORDS if pattern.search(text)), None)
            out.update({"labels": said} if said else {"labels": "file"} if re.search(r"\b(file|list)\b", text, re.I) else {})
    elif field in ("family", "outcome"):
        said = next((value for value, pattern in LABEL_WORDS if pattern.search(text)), None)
        if said:
            out["labels"] = said
    if field == "outcome":
        if named:
            out["target_column"] = named[0]  # a column named is the outcome, whatever else the answer says
        elif not columns and "family" not in out:
            out["target_definition"] = text
    if field in ("prediction_moment", "unit", "action"):
        out[field] = text
    if out.get("family") == "anomaly" and "labels" not in out and re.search(r"\b(no|nothing|none)\b", text, re.I):
        out["labels"] = "none"
    if out.get("family") == "rare_event" and out.get("labels") == "none":
        out.pop("family")  # rare events are learned from confirmed cases: with none, the labels decide (anomaly)
    return out


def card_for(problem: str, said: Mapping[str, str], columns: Optional[list[str]] = None,
             data: Optional[Mapping[str, Any]] = None, advisor: Any = None, kinds: Optional[Mapping[str, str]] = None) -> ProblemCard:
    """The problem card of a conversation: the words, what the person stated (``read_answer``'s fields), the tables'
    column names and what the files say (``problem_card``'s ``data``)."""
    from dclab_core.understand import understand

    brief = understand(problem, {"table": list(columns)} if columns else None, advisor=advisor)
    update: dict[str, Any] = {}
    for name in ("target_column", "target_definition", "prediction_moment", "unit", "action"):
        if said.get(name):
            update[name] = _said(str(said[name]))
    outcome = said.get("target_column")
    if outcome and (kinds or {}).get(outcome) in ("binary", "multiclass", "regression"):  # what the table's values say of it
        from dclab_core.contracts import TaskFact

        update["task"] = TaskFact(value=kinds[outcome], status="inferred", source="data", reason=f"the outcome column {outcome}'s values")
    if said.get("labels") in ("column", "file", "none"):
        from dclab_core.contracts import LabelSourceFact

        update["label_source"] = LabelSourceFact(value=said["labels"], status="stated", quote=said["labels"], source="person")
    brief = ProblemBrief.model_validate({**brief.model_dump(), **{k: v.model_dump() for k, v in update.items()}})
    words = " ".join([problem] + [str(v) for k, v in said.items() if k not in ("labels", "family")])
    family = said.get("family") if said.get("family") in FAMILIES else None
    return problem_card(brief, words=words, data=data, family=family)


def next_question(card: ProblemCard, contradictions: tuple[Any, ...] = (), asked: tuple[str, ...] = ()) -> Optional[Any]:
    """The question to ask now, or None when nothing blocks: a contradiction first (the person must see it), then the
    blocking question that unblocks the most decisions, never one already asked."""
    for c in contradictions:
        if getattr(c, "code", None) not in asked:
            return c
    blocking = [q for q in card.questions if q.blocks and q.field not in asked]
    if any(q.field == "family" for q in blocking) and not card.outcome and "outcome" not in asked \
            and not any(q.field in ("data", "labels") for q in blocking):  # the files and the labels first
        from dclab_core.card import CardQuestion  # the outcome's values settle a yes or no, a class or a number: ask it first

        return CardQuestion(field="outcome", text="Which column holds what you want to predict, if any?",
                            why="Its values say whether DCLab gives a yes or no, a class or a number.", blocks=True)
    return min(blocking, key=lambda q: ORDER.index(q.field) if q.field in ORDER else len(ORDER), default=None)


def summary(card: ProblemCard) -> dict[str, Any]:
    """What the person confirms before a build: the family, the labels, the outcome, the moment, the unit, the action,
    and what still blocks."""
    b = card.brief
    return {"family": card.family.value, "family_how": card.family.status, "in_play": list(card.in_play), "labels": card.labels.value,
            "outcome": card.outcome, "moment": b.prediction_moment.value, "unit": b.unit.value, "action": b.action.value,
            "ready": card.ready, "blocking": list(card.blocking)}


def converse(problem: str, answers: Mapping[str, str], columns: Optional[list[str]] = None, data: Optional[Mapping[str, Any]] = None,
             reader: Optional[Callable[[str, str, Optional[list[str]]], dict[str, str]]] = None, limit: int = LIMIT,
             kinds: Optional[Mapping[str, str]] = None) -> dict[str, Any]:
    """A written conversation played to its end: each question the card asks is answered from ``answers`` (keyed by
    the question's field; a missing answer is "I don't know"), read by ``reader`` (a model role behind its validator)
    or the rules, for at most ``limit`` questions. Returns the card, the questions asked and the card's summary."""
    said: dict[str, str] = {}
    asked: list[str] = []
    card = card_for(problem, said, columns, data, kinds=kinds)
    while len(asked) < limit:
        question = next_question(card, asked=tuple(asked))
        if question is None:
            break
        field = getattr(question, "field", None) or getattr(question, "code", "")
        asked.append(field)
        answer = answers.get(field, "I don't know")
        read = (reader(field, answer, columns) if reader is not None else None) or read_answer(field, answer, columns)
        said.update(read)
        card = card_for(problem, said, columns, data, kinds=kinds)
    return {"card": card, "asked": asked, "summary": summary(card)}


# ------------------------------------------------------------------------------------------------ the model role

READ_FIELDS = ("family", "labels", "target_column", "prediction_moment", "unit", "action")


def _evidence(inputs: dict[str, Any]) -> dict[str, Any]:
    return {"question_field": inputs["field"], "answer": inputs["answer"][:400], "columns": list(inputs.get("columns") or [])[:200],
            "families": list(FAMILIES), "labels": ["column", "file", "none"]}


def _quoted(quote: str, answer: str) -> bool:
    """The quote is words of the answer, whole: at least two words or four characters ("no" or "a" quotes nothing)."""
    quote = " ".join(quote.lower().split())
    if len(quote.split()) < 2 and len(quote) < 4:
        return False
    return re.search(rf"(?<![\w]){re.escape(quote)}(?![\w])", " ".join(answer.lower().split())) is not None


def _parse(reply: Any, inputs: dict[str, Any]) -> Optional[dict[str, str]]:
    """Keep only what the answer says: a family or a label situation from their vocabularies, a column of the table
    named in its quote, and free text (a moment, a unit, an action) only as words of the answer itself."""
    if not isinstance(reply, dict):
        return None
    answer = inputs["answer"]
    out: dict[str, str] = {}
    for name in READ_FIELDS:
        item = reply.get(name)
        if not isinstance(item, dict) or not isinstance(item.get("value"), str):
            continue
        quote, value = str(item.get("quote") or ""), item["value"]
        if not _quoted(quote, answer):
            continue  # nothing the person did not say
        if name == "family" and value not in FAMILIES:
            continue
        if name == "labels" and value not in ("column", "file", "none"):
            continue
        if name == "target_column" and (value not in (inputs.get("columns") or []) or value.lower() not in quote.lower()):
            continue
        if name in ("prediction_moment", "unit", "action") and not _quoted(value, answer):
            continue  # a moment the model wrote is not the person's moment (leakage starts there)
        out[name] = value[:400]
    return out or None


READER = Role(name="conversation_reader", description="reads a person's answer into problem-card fields the rules could not read",
              evidence=_evidence, parse=_parse,
              fallback=lambda inputs: read_answer(inputs["field"], inputs["answer"], inputs.get("columns")),
              merge=lambda rules, model: {**rules, **(model or {})})


__all__ = ["card_for", "next_question", "read_answer", "converse", "summary", "READER", "ORDER", "LIMIT", "FAMILY_ANSWERS"]
