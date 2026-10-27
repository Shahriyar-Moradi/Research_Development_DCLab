"""What kind of help a person needs, read before anything else on Home (package H2).

Six tracks: ``prediction`` (a model that predicts what is not yet known), ``analysis`` (numbers about what already
happened), ``preparation`` (cleaning, combining or reshaping data), ``reporting`` (a report or dashboard that is kept up to
date), ``business`` (a business question or decision) and ``advice`` (an explanation; no data work).

Code decides, a model advises (CLAUDE.md rule 5):

1. **Rules first.** Phrases of each track score the request (``SIGNALS``: a strong phrase 3, a medium one 2, a weak one 1).
   The track with the highest score is the main one; every track that scored is kept, in score order. The quote is the
   person's own words that matched, checked with ``plan.quoted``.
2. **A model second**, only when the rules found nothing or the two leading tracks are the classic confusion (a prediction
   against an analysis or a report of what already happened, with equal scores): its reading is kept only with a known
   track and a quote that is the person's words.
3. **Ask when it is still unsure**: that confusion, with no model to settle it, becomes one question with the tracks as buttons.
4. **Nothing found, nothing asked**: the request is read as a prediction, as Home always did (``source: default``), and the
   person can change it on the card. Persian and other languages are read by the rules in package H3.

The person's choice on the card (``source: user``) always wins; a model or a later reading never changes it.
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from dclab_rnd import prompts

from . import lang

TRACKS = ("prediction", "analysis", "preparation", "reporting", "business", "advice")
LABELS = {
    "prediction": "A prediction (a model)",
    "analysis": "Numbers from my data",
    "preparation": "Cleaning or combining data",
    "reporting": "A report or dashboard",
    "business": "A business question",
    "advice": "Advice or an explanation",
}
# When two tracks score the same, the main one is the first in this order (a request that combines files and then
# predicts is a prediction that needs preparation first).
PRIORITY = ("prediction", "business", "reporting", "analysis", "preparation", "advice")
# The confusions worth a question: "show me which customers will buy" (a prediction) against "show me who bought" (an
# analysis), and a model asked for where a report of the past answers it.
ASK_PAIRS = ({"prediction", "reporting"},)  # a model or a report: both are real asks, so Home asks
# A prediction signal makes a prediction the main track over analysis or preparation: "predict how many units per store"
# and "merge the tables, then predict" need the counts or the join as a step of the prediction, not as the answer.
LEADS = {"prediction": ("analysis", "preparation")}
LEAD_MIN = 2
# Leaving the prediction path takes one strong phrase (weight 3 or more): a request that describes a model and happens to
# say "report the F1", "average precision" or "decide on" is still a prediction. Weaker matches leave the request
# unclear (the model reads it, or the prediction default stands), and only two strong readings can make Home ask.
STRONG = 3
PROMPT = "home_intent"
RULES_VERSION = "intent_rules_v1"

_N = r"(?:\w+(?:\s+\w+)?)"  # one or two words ("customers", "the bookings")
SIGNALS: list[tuple[str, str, int, str]] = [
    # (track, id, weight, pattern) — written from the H1 dev requests and ordinary English, never from the sealed set
    ("prediction", "predict", 3, r"\b(?:predict\w*|forecast\w*)\b"),
    ("prediction", "which_will", 3, rf"\b(?:which|what)\s+{_N}\s+(?:will|are going to|is going to|would)\b"),
    ("prediction", "who_will", 3, r"\bwho\s+(?:will|is going to|are going to|is likely|are likely)\b"),
    ("prediction", "will_act", 2, r"\bwill\s+(?:churn|cancel|leave|buy|convert|default|upgrade|book|renew|purchase|return|spend|pay|order)\b"),
    ("prediction", "going_to", 2, r"\b(?:are|is)\s+going\s+to\s+(?:churn|cancel|leave|buy|convert|default|upgrade|book|renew|purchase)\b"),
    ("prediction", "future_amount", 3, r"\bhow\s+(?:much|many)\b[^.?!]*\bwill\b|\bwill\s+be\s+\w+ed\b"),
    ("prediction", "next_period", 2, r"\b(?:tomorrow|next\s+(?:day|week|month|quarter|year))\b"),
    ("prediction", "detect_risk", 3, r"\b(?:detect|flag|catch)\s+(?:\w+\s+){0,2}(?:fraud\w*|anomal\w*|default\w*|risky|suspicious)\b"),
    ("prediction", "which_get", 2, rf"\bwhich\s+{_N}\s+(?:get|gets)\s+\w+ed\b"),
    ("prediction", "rank", 2, r"\b(?:rank|score|prioriti[sz]e)\s+(?:the\s+|our\s+)?(?:clients|customers|leads|accounts|users|members|tickets)\b"),
    ("prediction", "a_model", 3, r"\b(?:build|train|fit|make|need)\s+(?:an?\s+)?(?:[\w-]+\s+){0,3}(?:model|classifier)\b(?!\s+of\s+(?:last|past|the\s+past|previous)\b)"),  # "a model of last month's sales" is a report
    ("prediction", "classify", 2, r"\bclassif(?:y|ies)\b"),
    ("prediction", "estimate", 3, r"\bestimat\w*\b"),
    ("prediction", "will_happen", 2, r"\bwill\s+(?:take|arrive|need|cost|sell|happen|fail|break|reach|last)\b"),
    ("prediction", "likely_to", 2, r"\b(?:likely|probability|chance)\s+(?:to|of)\b"),
    ("analysis", "how_much", 3, r"\bhow\s+(?:much|many)\b(?!.*\bdo\s+(?:i|we)\s+need\b)"),  # "how many rows do I need" is advice
    ("analysis", "what_was", 3, r"\bwhat\s+(?:was|were)\s+(?:our|the|my)\b"),  # a question about what already happened
    ("analysis", "list_who", 3, r"\blist\s+(?:all\s+)?(?:the\s+)?\w+\s+(?:who|that|which)\b"),
    ("analysis", "the_most", 3, r"\b(?:sold|had|made|spent|bought|earned)\s+the\s+most\b"),
    ("analysis", "share_of", 3, r"\bwhat\s+(?:share|percentage|proportion|fraction|rate)\s+of\b"),
    ("analysis", "compare", 2, r"\bcompare\b"),
    ("analysis", "aggregate", 2, r"\b(?:average|median|total|sum|count|number)\s+of\b|\b(?:average|median)\b"),
    ("analysis", "show_measure", 2, r"\bshow\s+(?:me\s+)?(?:the\s+)?(?:average|median|total|number|count|revenue|refunds?|sales|share|orders)\b"),
    ("analysis", "break_down", 2, r"\bbreak\s*down\b|\bbreakdown\b"),
    ("analysis", "per_group", 1, r"\b(?:per|by|for each|in each)\s+(?:region|channel|city|segment|plan|destination|product|category|month|week|day|year|country|store|customer type|type)\b"),
    ("analysis", "trend", 1, r"\btrend\w*\b|\bdistribution\b"),
    ("preparation", "duplicates", 3, r"\b(?:remove|drop|delete)\s+(?:the\s+)?(?:duplicate\w*|duplicated)\b|\bde-?duplicat\w*\b"),
    ("preparation", "join", 3, r"\b(?:join|merge|combine)\b"),
    ("preparation", "reshape", 3, r"\bpivot\b|\bunpivot\b|\breshape\b"),
    ("preparation", "derive", 3, r"\bturn\s+\w+\s+into\b|\bnew\s+column\b|\bconvert\s+(?:the\s+)?\w+\s+(?:to|into)\b"),
    ("preparation", "keep_only", 3, r"\bkeep\s+only\b"),
    ("preparation", "clean_up", 3, r"\bclean\s+up\b|\bclean\s+(?:the|my|our|this)\s+(?:data|table|file|sheet|dataset|columns?)\b|\b(?:fill|impute)\s+(?:in\s+)?(?:the\s+)?missing\b"),
    ("preparation", "edit_table", 3, r"\brename\s+(?:the\s+)?columns?\b|\bdrop\s+(?:the\s+)?rows\b|\bappend\s+the\b|\bsplit\s+(?:the\s+)?\w+\s+column\b"),
    ("preparation", "clean", 2, r"\bclean\w*\b"),
    ("preparation", "export", 2, r"\bexport\b|\bas\s+(?:an?\s+)?(?:csv|excel|xlsx|parquet)(?:\s+file)?\b"),
    ("reporting", "dashboard", 3, r"\bdashboards?\b"),
    ("reporting", "periodic_report", 3, r"\b(?:daily|weekly|monthly|quarterly)\s+(?:kpi\s+)?(?:report|dashboard|summary)\b"),
    ("reporting", "make_report", 3, r"\b(?:make|build|create|write|send|prepare|give\s+me|want|need)\s+(?:me\s+)?(?:an?\s+|the\s+)?(?:\w+\s+){0,2}reports?\b"),
    ("reporting", "past_model", 3, r"\bmodel\s+of\s+(?:last|past|the\s+past|previous)\b"),  # "a model of last month's sales": a report of what happened
    ("reporting", "report", 2, r"\breports?\b"),
    ("reporting", "kpi", 2, r"\bkpis?\b"),
    ("reporting", "track", 2, r"\b(?:track|monitor)\b"),
    ("reporting", "every_period", 2, r"\bevery\s+(?:day|week|month|quarter)\b"),
    ("reporting", "refresh", 2, r"\brefresh\w*\b"),
    ("business", "why_did", 3, r"\bwhy\s+(?:did|do|does|is|are|was|were|has|have)\b"),
    ("business", "should_we", 3, r"\b(?:should\s+we|what\s+should\s+we|whether\s+we\s+should|do\s+we\s+need\s+to)\b"),
    ("business", "which_should", 3, rf"\bwhich\s+{_N}\s+should\s+we\b"),
    ("business", "what_would", 2, r"\bwhat\s+would\s+(?:it|that|this)\s+do\b|\bwhat\s+would\s+happen\b"),
    ("business", "worth_it", 3, r"\bis\s+it\s+worth\b"),
    ("business", "decide", 1, r"\bdecide\b"),
    ("advice", "explain", 3, r"\bexplain\b|\bwhat\s+does\s+\w+(?:\s+\w+)?\s+mean\b"),
    ("advice", "difference", 3, r"\bdifference\s+between\b|\bwhat(?:'s|\s+is)\s+the\s+difference\b"),
    ("advice", "is_it_ok", 3, r"\bis\s+it\s+(?:ok|okay|fine|right|safe|a good idea|wrong)\s+to\b"),
    ("advice", "should_i", 3, r"\bshould\s+i\s+(?:use|choose|pick|trust)\b|\bwhich\s+(?:(?:metric|model|one)\s+)?should\s+i\b"),
    ("advice", "how_much_data", 4, r"\bhow\s+(?:many\s+rows|much\s+data|many\s+examples)\s+do\s+i\s+need\b"),
    ("advice", "not_sure_need", 3, r"\b(?:not\s+sure|unsure)\s+(?:if|whether)\s+(?:i|we)\s+need\b"),
]
# English beside Persian (H3, lang.py): both match the normalized text
_COMPILED = [(t, i, w, re.compile(p, re.I)) for t, i, w, p in SIGNALS + lang.SIGNALS]

# Home's next step on each track until the packages that serve it are built (the card says what DCLab can do now)
NEXT = {
    "analysis": "Answering questions from your data in this chat is not built yet. Upload or connect the table and I'll prepare it "
                "and show what it holds; if you need a prediction instead, change the type of help above.",
    "preparation": "Preparing data on request is not built yet. When you upload the table, DCLab cleans it (names, numbers, dates, "
                   "empty rows, exact duplicates) and lists every step it took.",
    "reporting": "Reports and dashboards are not built yet. Upload or connect the data and I'll prepare it and show what it holds.",
    "business": "Answering business questions in this chat is not built yet. Upload the data and I'll prepare it and show what it holds; "
                "if part of the question is a prediction, change the type of help above.",
    "advice": "Ask it in the Evidence library: its answers cite DCLab's recorded experiments.",
}


_ENDS = " ,.;:?!\"'“”،؛؟"


def _quote(text: str, start: int, end: int) -> str:
    """The person's words around a match, grown to whole words until they make a quote (two words or eight characters)."""
    while start > 0 and text[start - 1].isalnum():
        start -= 1
    while end < len(text) and text[end].isalnum():
        end += 1
    quote = text[start:end]
    while (len(quote.split()) < 2 and len(quote) < 8) and end < len(text):
        nxt = re.match(r"\s*\S+", text[end:])
        if not nxt:
            break
        end += nxt.end()
        quote = text[start:end]
    while (len(quote.strip(_ENDS).split()) < 2 and len(quote.strip(_ENDS)) < 8) and start > 0:  # at the end of the text: grow back
        before = re.search(r"\S+\s*$", text[:start])
        if not before:
            break
        start = before.start()
        quote = text[start:end]
    return quote.strip(_ENDS)


def rules(text: str) -> dict[str, Any]:
    """Each track's score, the signals that matched (their ids, never the words) and the best quote per track."""
    scores = {t: 0 for t in TRACKS}
    signals: list[str] = []
    quotes: dict[str, tuple[int, str]] = {}
    text = str(text or "")
    norm, at = lang.normalize_map(text)  # matched on the normalized text, quoted from the person's own words
    for track, sid, weight, pattern in _COMPILED:
        match = pattern.search(norm)
        if not match:
            continue
        scores[track] += weight
        signals.append(sid)
        if weight > quotes.get(track, (0, ""))[0]:
            quotes[track] = (weight, _quote(text, *lang.span(at, match.start(), match.end())))
    return {"scores": scores, "signals": signals, "quotes": {t: q for t, (_, q) in quotes.items()}}


def _ranked(scores: dict[str, int]) -> list[str]:
    return sorted((t for t in TRACKS if scores[t] > 0), key=lambda t: (-scores[t], PRIORITY.index(t)))


def _first_json(text: str) -> Optional[dict[str, Any]]:
    decoder = json.JSONDecoder()
    for m in re.finditer(r"\{", text or ""):
        try:
            value, _ = decoder.raw_decode(text[m.start():])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _model_reading(text: str, client: Any, texts: list[str]) -> Optional[dict[str, Any]]:
    """The model's reading, or None when it fails or proposes what code cannot accept."""
    from .plan import quoted

    try:
        reply = client.complete([{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": text}], None)
    except Exception:  # noqa: BLE001 — no model, no problem: the rules' reading stands
        return None
    answer = _first_json(reply.get("content") if isinstance(reply, dict) else str(reply)) or {}
    tracks = [t for t in answer.get("tracks") or [answer.get("track")] if t in TRACKS]
    quote = str(answer.get("quote") or "")
    ok = bool(tracks) and quoted(quote, texts)
    report = getattr(client, "output", None)
    if report is not None:
        report(ok, "" if ok else "the reading named no known track or its quote is not the person's words")
    if not ok:
        return None
    try:
        confidence = max(0.0, min(1.0, float(answer.get("confidence", 0.6))))
    except (TypeError, ValueError):
        confidence = 0.6
    return {"track": tracks[0], "tracks": tracks, "quote": quote, "confidence": round(confidence, 2)}


def read(text: str, client: Any = None, texts: Optional[list[str]] = None) -> dict[str, Any]:
    """The intent of a request: track, tracks, status (stated, inferred, asked), source (rules, model, default), quote,
    confidence and the alternatives, with the rules' scores and signals (for the decision sample)."""
    from .plan import quoted

    found = rules(text)
    ranked = _ranked(found["scores"])
    for lead, over in LEADS.items():
        if ranked and ranked[0] in over and found["scores"][lead] >= LEAD_MIN:
            ranked = [lead] + [t for t in ranked if t != lead]
    if ranked and not quoted(found["quotes"].get(ranked[0]) or "", [text]):
        ranked = []  # a request too short to quote ("Predict.") is read as if no rule matched: never a reading without words
    base = {"scores": found["scores"], "signals": found["signals"], "rules_version": RULES_VERSION}
    if ranked and ranked[0] != "prediction" and found["scores"][ranked[0]] < STRONG:
        ranked = [t for t in ranked if t == "prediction"]  # only weak signals away from a prediction: not a reading
    tie = (len(ranked) >= 2 and found["scores"][ranked[0]] == found["scores"][ranked[1]] >= STRONG
           and set(ranked[:2]) in ASK_PAIRS)
    if ranked and ranked[0] == "reporting" and found["scores"]["prediction"] >= LEAD_MIN:
        tie = True  # "build a churn model and send me a weekly report": a report of a model's output, or a report alone? ask
    if ranked and not tie:
        main = ranked[0]
        total = sum(found["scores"].values())
        return {**base, "track": main, "tracks": ranked, "status": "stated", "source": "rules", "quote": found["quotes"].get(main),
                "confidence": round(found["scores"][main] / total, 2), "alternatives": ranked[1:]}
    if client is not None:
        reading = _model_reading(text, client, texts or [text])
        if reading is not None:
            rest = [t for t in ranked if t not in reading["tracks"]]
            # the rules' own answer kept beside the model's (the review of the H2 port: the ledger recorded the model's track as the
            # rules' too, so every comparison of the two agreed); None when the rules read nothing
            return {**base, **reading, "tracks": reading["tracks"] + rest, "status": "inferred", "source": "model",
                    "alternatives": [t for t in reading["tracks"][1:] + rest], "rule_track": ranked[0] if ranked else None}
    if tie:
        return {**base, "track": ranked[0], "tracks": ranked, "status": "asked", "source": "rules", "quote": found["quotes"].get(ranked[0]),
                "confidence": 0.5, "alternatives": ranked[1:]}
    return {**base, "track": "prediction", "tracks": ["prediction"], "status": "inferred", "source": "default", "quote": None,
            "confidence": 0.0, "alternatives": []}


def from_answer(answer: str, options: dict[str, str]) -> Optional[str]:
    """The track a person's answer to the intent question names: a button's label, else the rules on their words; None
    when it names none ("Something else", "I'm not sure"), which is not a choice."""
    for track, label in options.items():
        if answer.strip().casefold() == label.casefold():
            return track
    ranked = _ranked(rules(answer)["scores"])
    return ranked[0] if ranked else None


def chosen(track: str, previous: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """The intent the person chose on the card or by a button: it carries no quote and nothing overrides it."""
    previous = previous or {}
    others = [t for t in previous.get("tracks") or [] if t != track]
    return {"track": track, "tracks": [track] + others, "status": "confirmed", "source": "user", "quote": None, "confidence": 1.0,
            "alternatives": others, "scores": previous.get("scores"), "signals": previous.get("signals"),
            "rules_version": previous.get("rules_version", RULES_VERSION), "rule_track": previous.get("rule_track", previous.get("track"))}
