"""What kind of help a person needs, read before anything else on Home (package H2).

Six tracks: ``prediction`` (a model that predicts what is not yet known), ``analysis`` (numbers about what already
happened), ``preparation`` (cleaning, combining or reshaping data), ``reporting`` (a report or dashboard that is kept up to
date), ``business`` (a business question or decision) and ``advice`` (an explanation; no data work).

A model reads first, code checks (the owner, 2026-10-29: "AI first, rules as the fallback"):

1. **The model reads first** when Home has one (the ``home_intent`` purpose): the tracks, the need in one line, and the
   question that would make the request clear. Code keeps a track only when it is one of ``TRACKS`` and its quote is the
   person's own words (``plan.quoted``); a sentence the person will read keeps only numbers the person gave.
2. **The rules check it.** Phrases of each track score the request (``SIGNALS``: strong 3, medium 2, weak 1). When the
   rules read another track strongly and the model did not name it, Home asks which one, in two buttons.
3. **Unclear is asked, never guessed.** A model that says the request does not tell, or reads it with low confidence, and
   a request nobody could read, become one open question ("Tell me a bit more ...") with the kinds of help as buttons
   (``status: asked``, ``ask: open``). A failed or refused model call is said on the reading (``model_failed``) and the
   rules read alone; nothing read is never a prediction by default.
4. **No model**: the rules read alone, as before, and a tie between two strong readings is asked.

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
SEGMENT_SIGNALS = ("segments", "fa_segments")  # a request about groups: Home offers the two ways to find them
RULES_VERSION = "intent_rules_v3"  # v3 (2026-10-29): segments (English, Persian) and the no-model kind words; v2 (H4): plain
# asks for an analysis, stats, insights; Persian "analyse"

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
    # H4 (2026-10-28): a plain request for numbers ("I want an analysis of my tickets", "summarize my data", "some stats") fell
    # to the prediction default; asked outright, an analysis is a strong signal (a prediction named beside it still leads)
    # strong only when the analysis itself is asked for; "explore", "profile", "summarize" and "statistics" are said in passing
    # by prediction requests too ("explore the data and find who churns": the review of H4), so they weigh 2 and leave it to
    # the model, or to the prediction default
    ("analysis", "analyse_this", 3, r"\b(?:analy[sz]e|an\s+analysis\s+of)\s+(?:my|the|our|this|these|some)\b"),
    # 2026-10-29 (the owner's test: "segment of the my customer"): groups of customers, items or places, a need of its own
    ("analysis", "segments", 3, r"\b(?:segmentation|clustering|personas?|(?:customer|user|client|shopper)\s+segments?|"
                                r"(?:segment(?:s|ing)?|cluster(?:s|ing)?)\s+(?:of\s+)?(?:the\s+|my\s+|our\s+|these\s+)*"
                                r"(?:customers?|users?|clients?|shoppers?|members?|stores?|products?|buyers?)|"
                                r"groups?\s+of\s+(?:the\s+|my\s+|our\s+)*(?:customers|users|clients|shoppers|members|stores|buyers)|"
                                r"behave\s+alike)\b"),
    ("analysis", "explore_this", 2, r"\b(?:explore|summari[sz]e|profile)\s+(?:my|the|our|this|these|some)\b"),
    ("analysis", "stats", 2, r"\b(?:stats|statistics|insights?\s+(?:from|about|into|on))\b"),
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
    "analysis": "Upload or connect the table and I'll answer from it: numbers per group, never single rows. If you need a "
                "prediction instead, change the type of help above.",
    "preparation": "Upload or connect the table and I'll prepare it as you asked, step by step, into a new table you can download; "
                   "a step that removes many rows waits for your yes, and the original file is never changed.",
    "reporting": "Upload or connect the table and I'll build the dashboard you described: one tile per number, per group, refreshed "
                 "whenever the table changes; a list of single rows is never a tile.",
    "business": "Upload or connect the table and I'll show where a number changed between the periods you name, and what moved it "
                "most; the data shows association, not cause. If part of the question is a prediction, change the type of help above.",
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


UNSURE = 0.6  # a model reading below this confidence is asked as an open question, its tracks first among the buttons
OPEN_AGAIN = ("Thanks. What would you like to end up with? For example: numbers or charts about what happened, a cleaned or "
              "combined table, a dashboard, an answer to why something changed, a model that predicts something, or advice.")
OPEN_QUESTION = ("Tell me a bit more so I can help the right way: what is the situation, what data do you have, and what would "
                 "you like to end up with: a prediction, numbers or charts from your data, a cleaned or combined table, a "
                 "dashboard, an answer to a business question, or advice?")
_TEXT_FIELDS = ("need", "goal", "data_said", "best_question")


def _own_numbers(sentence: str, texts: list[str]) -> bool:
    """A sentence Home will say keeps only numbers the person wrote (cited.py, the one number check)."""
    from dclab_rnd import cited

    allowed = set().union(*(cited.numbers(t, source=True) for t in texts)) if texts else set()
    return not (cited.numbers(sentence) - allowed)


def _model_reading(text: str, client: Any, texts: list[str]) -> dict[str, Any]:
    """The model's reading: ``{"ok": True, ...}`` with a kept track and quote, ``{"unclear": True, ...}`` when it says the
    request does not tell (its question kept), or ``{"failed": reason}`` when the call fails or proposes what code
    cannot accept. Its sentences are kept only with the person's own numbers."""
    from .plan import quoted

    try:
        reply = client.complete([{"role": "system", "content": prompts.text(PROMPT)}, {"role": "user", "content": text}], None)
    except Exception as error:  # noqa: BLE001 - said on the reading; the rules read alone
        return {"failed": (str(error) or type(error).__name__)[:200], "request": getattr(client, "last_request", None)}
    request = getattr(client, "last_request", None)  # P8: the exchange (and its thinking summary) this reading came from
    answer = _first_json(reply.get("content") if isinstance(reply, dict) else str(reply)) or {}
    said = {k: str(answer.get(k) or "").strip()[:400] for k in _TEXT_FIELDS}
    said = {k: v for k, v in said.items() if v and _own_numbers(v, texts)}
    unknowns = [str(u)[:120] for u in answer.get("unknowns") or [] if isinstance(u, str)][:5]
    fields = _fields(answer.get("fields"), texts)
    tracks = [t for t in answer.get("tracks") or [answer.get("track")] if t in TRACKS]
    quote = str(answer.get("quote") or "")
    report = getattr(client, "output", None)
    if not tracks:
        named = [t for t in answer.get("tracks") or [] if t]
        if report is not None:  # an empty list is the model's honest "it does not say"; tracks it invented are not
            report(bool(answer) and not named, "" if answer and not named else ("the reading named no known track" if named else "the reading was not JSON"))
        return ({"unclear": True, **said, "unknowns": unknowns, "request": request} if answer
                else {"failed": "the reading was not JSON", "request": request})
    ok = quoted(quote, texts)
    if report is not None:
        report(ok, "" if ok else "the reading's quote is not the person's words")
    if not ok:
        return {"failed": "the reading's quote is not the person's words", "request": request}
    try:
        confidence = max(0.0, min(1.0, float(answer.get("confidence", 0.7))))
    except (TypeError, ValueError):
        confidence = 0.7
    return {"ok": True, "track": tracks[0], "tracks": tracks, "quote": quote, "confidence": round(confidence, 2), **said,
            "unknowns": unknowns, "request": request, **({"fields": fields} if fields else {})}


def _fields(raw: Any, texts: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """2026-10-29 (HOME-017: the rules read the fields of each kind of help, 17 right against Claude alone's 64): the model's
    fields per kind, each kept only when the kind's brief has it, its quote is the person's own words, and its value names
    no number the person did not write. {kind: {field: {value, status, source, quote}}}."""
    from .brief import TRACK_FIELDS
    from .plan import quoted

    out: dict[str, dict[str, dict[str, Any]]] = {}
    for kind, entries in (raw if isinstance(raw, dict) else {}).items():
        names = {f.name for f in TRACK_FIELDS.get(kind, ())}
        for name, item in (entries if isinstance(entries, dict) else {}).items():
            if name not in names or not isinstance(item, dict):
                continue
            value, quote = str(item.get("value") or "").strip()[:200], str(item.get("quote") or "").strip()[:300]
            if value and quoted(quote, texts) and _own_numbers(value, texts):
                if not quoted(value, texts):  # HOME-018: a value the model translated or reworded is said in the person's words
                    value = quote[:200]
                out.setdefault(kind, {})[name] = {"value": value, "status": "stated", "source": "model", "quote": quote}
    return out


def _open(base: dict[str, Any], source: str, tracks: list[str], said: dict[str, Any], quote: Optional[str] = None,
          confidence: float = 0.0) -> dict[str, Any]:
    """An open question: the request is not clear enough to route. Its tracks (a model's low-confidence reading, or
    none) come first among the buttons; the question is the model's own when it kept the person's numbers."""
    return {**base, "track": tracks[0] if tracks else None, "tracks": tracks, "status": "asked", "ask": "open", "source": source,
            "quote": quote, "confidence": confidence, "alternatives": tracks[1:],
            "question": said.get("best_question") or OPEN_QUESTION, **{k: v for k, v in said.items() if k != "best_question"}}


def _rules_reading(text: str) -> tuple[dict[str, Any], list[str], bool, dict[str, Any]]:
    """The rules' reading: (what they found, their ranked tracks, whether the two leaders are a tie to ask, the base record)."""
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
    return found, ranked, tie, base


def read(text: str, client: Any = None, texts: Optional[list[str]] = None) -> dict[str, Any]:
    """The intent of a request: track, tracks, status (stated, inferred, asked), source (model, rules), quote, confidence,
    the alternatives and, when asked, ``ask`` ("open" or "either") and the question; with the rules' scores and signals
    (for the decision sample) and their own track (``rule_track``) beside a model's."""
    texts = texts or [text]
    found, ranked, tie, base = _rules_reading(text)
    rule_track = ranked[0] if ranked else None
    failed = None
    if client is not None:
        reading = _model_reading(text, client, texts)
        if reading.get("ok"):
            said = {k: reading[k] for k in (*_TEXT_FIELDS, "unknowns", "fields") if reading.get(k)}
            model = {**base, **said, "rule_track": rule_track, "model_track": reading["track"], "model_request": reading.get("request")}
            # the rules check the model: a strong reading of another track that the model did not name is asked
            strong = rule_track is not None and found["scores"][rule_track] >= STRONG and not tie
            if strong and rule_track not in reading["tracks"]:
                pair = [reading["track"], rule_track]
                return {**model, "track": pair[0], "tracks": pair, "status": "asked", "ask": "either", "source": "model",
                        "quote": reading["quote"], "confidence": reading["confidence"], "alternatives": pair[1:]}
            rest = [t for t in ranked if t not in reading["tracks"]]
            if reading["confidence"] < UNSURE:
                return _open(model, "model", reading["tracks"] + rest, said, reading["quote"], reading["confidence"])
            return {**model, "track": reading["track"], "tracks": reading["tracks"] + rest, "status": "inferred", "source": "model",
                    "quote": reading["quote"], "confidence": reading["confidence"], "alternatives": reading["tracks"][1:] + rest}
        if reading.get("unclear"):
            said = {k: reading[k] for k in (*_TEXT_FIELDS, "unknowns") if reading.get(k)}
            return _open({**base, "rule_track": rule_track, "model_track": None, "model_request": reading.get("request")}, "model", ranked, said)
        failed = reading.get("failed")
    note = {"model_failed": failed, "model_track": None, "model_request": reading.get("request")} if failed else {}
    if ranked and not tie:
        main = ranked[0]
        total = sum(found["scores"].values())
        return {**base, **note, "track": main, "tracks": ranked, "status": "stated", "source": "rules", "quote": found["quotes"].get(main),
                "confidence": round(found["scores"][main] / total, 2), "alternatives": ranked[1:], "rule_track": main}
    if tie:
        return {**base, **note, "track": ranked[0], "tracks": ranked, "status": "asked", "ask": "either", "source": "rules",
                "quote": found["quotes"].get(ranked[0]), "confidence": 0.5, "alternatives": ranked[1:], "rule_track": ranked[0]}
    kind = _prediction_kind(text)
    if kind:  # the fallback with no model: the words name a kind of prediction (fraud, churn, a forecast), read as one
        return {**base, **note, "signals": [*base["signals"], "kind_words"], "track": "prediction", "tracks": ["prediction"],
                "status": "inferred", "source": "rules", "fallback": True, "quote": kind, "confidence": 0.3, "alternatives": [],
                "rule_track": "prediction"}
    return _open({**base, **note, "rule_track": None}, "rules", [], {})


def _prediction_kind(text: str) -> Optional[str]:
    """The person's words that name a kind of prediction (a domain pack's own words: fraud, a forecast, reviews or tickets
    to classify; or which ones will churn, cancel, default), quoted; None when they name none. Only the no-model fallback
    reads it, and only when no rule read the request: "I have data analysis problems" names none and is asked."""
    from . import pack
    from .plan import quoted

    low = str(text or "").lower()
    # a pack's own words, an outcome that happens to someone (churn, cancel, default), or "will" with an outcome ("if it will
    # be late", "which ones will buy"); never "which" alone ("which products sell best" asks about the past, live 2026-10-29)
    # nor a bare future ("I'm going to upload my table", the review)
    outcome = r"(?:late|delayed|fail\w*|default\w*|churn\w*|cancel\w*|leave|quit|buy|purchase|pay|return\w*|convert\w*|renew\w*|respond\w*|click\w*)"
    for pattern in [p for _, p in pack.KEYWORDS] + [r"\b(?:churn|cancel|leav(?:e|es|ing)|convert|respond|defaults?|renew)\w*",
                                                 rf"\bwill\s+(?:be\s+)?(?:\w+\s+)?{outcome}\b"]:
        m = re.search(pattern, low)
        if m:
            q = _quote(str(text), m.start(), m.end())
            if quoted(q, [str(text)]):
                return q
    return None


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
