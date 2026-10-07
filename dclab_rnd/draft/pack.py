"""Which domain pack fits the problem. The user's choice wins; otherwise the problem text and the data decide.

A pack maps the ten DCLab workflow steps to one kind of problem: its own rules, metrics and split. The
list matches the cards the product shows (the Domain packs page and the solution draft). Detection is
deterministic: keywords in the problem sentence, then signals from the analysed table (a time column, long
text columns, a rare positive class). The result always carries a plain reason the user can check.
"""

from __future__ import annotations

import re
from typing import Any

PACKS: list[dict[str, Any]] = [
    {"key": "tabular", "name": "Tabular", "desc": "Binary, multiclass and regression on tables.", "maturity": "GA", "cls": "ok", "icon": "models"},
    {"key": "imbalanced", "name": "Imbalanced and fraud", "desc": "Rare positives: PR-AUC, cost curves, time splits.", "maturity": "GA", "cls": "ok", "icon": "alert"},
    {"key": "timeseries", "name": "Time series", "desc": "Forecasts with horizons and backtests against naive baselines.", "maturity": "Beta", "cls": "info", "icon": "trend"},
    {"key": "text", "name": "Text + tabular", "desc": "Reviews, tickets and notes next to structured fields.", "maturity": "Beta", "cls": "info", "icon": "text"},
    {"key": "vision", "name": "Computer vision", "desc": "Detection and classification, split by scene or drive.", "maturity": "Preview", "cls": "warn", "icon": "image"},
    {"key": "driving", "name": "Driving perception", "desc": "Operating domain (ODD) as the solution; split by trip and route.", "maturity": "Research", "cls": "", "icon": "car"},
    {"key": "maps", "name": "Maps and road flow", "desc": "Road graphs, speed and flow forecasts.", "maturity": "Research", "cls": "", "icon": "route"},
    {"key": "scenegraph", "name": "Scene graphs", "desc": "Objects, relations and events in video.", "maturity": "Research", "cls": "", "icon": "nodes"},
    {"key": "llm", "name": "LLM and SLM fine-tuning", "desc": "Training data, LoRA runs and evaluation.", "maturity": "Beta", "cls": "info", "icon": "brain"},
]
KEYS = [p["key"] for p in PACKS]

# Ordered: the first pack whose words appear wins (more specific packs first).
KEYWORDS: list[tuple[str, str]] = [
    ("driving", r"\b(driving|autonomous|self[- ]driving|odd|lidar|adas|pedestrian)"),
    ("scenegraph", r"\b(scene graph|video (event|relation)|who (hands|gives)|relations? between objects)"),
    ("vision", r"\b(image|images|photo|camera|vision|detect(ion)? (in|on) (frames|images)|frames|x-ray|scan)"),
    ("maps", r"\b(road|traffic|route|map|segment speed|gps|trajector)"),
    ("llm", r"\b(fine[- ]tun|lora|llm|language model|slm|chatbot|instruction data)"),
    ("timeseries", r"\b(forecast|demand|time series|per (day|week|hour))"),
    ("imbalanced", r"\b(fraud|rare|anomal|default(s|ed)? on|chargeback|intrusion|0\.\d+ ?% positive)"),
    ("text", r"\b(review|ticket|text|comment|email|note|description|sentiment|complaint)s?\b"),
]


# A horizon ("next month", "daily") suggests forecasting only when the sentence asks how much, not which ones:
# "which subscribers will cancel next month" is a classification with a horizon, not a time series.
HORIZON = r"\b(next (day|week|month|quarter|year)|tomorrow|daily|weekly|hourly)"
WHICH_ONES = r"\b(which|who|whether)\b|\b(churn|cancel|leav(e|es|ing)|convert|respond|defaults?|renew)"


# The first question Home asks, per pack: what the outcome is, in the words of that kind of problem. The examples are
# of the pack's own kind, so a fraud problem is never asked about a customer who leaves.
OPENING: dict[str, str] = {
    "tabular": "What exactly should the model predict, and for which rows (customers, orders, machines)? Name the outcome and its time window if it has one, "
               "for example: which loans default within 12 months?",
    "imbalanced": "What is the rare event the model should flag (for example a fraudulent payment, a chargeback or a default), what is scored (a transaction, "
                  "an account, a claim), and how long after it happens is the true label known?",
    "timeseries": "What quantity should be forecast, for which unit (a store, a product, a region), and how far ahead, for example daily demand per store, "
                  "7 days ahead?",
    "text": "What should the model predict from the text (for example a ticket's topic, or whether a review is negative), and which structured fields "
            "come with each text?",
    "vision": "What should the model find or classify in the images (for example defects or pedestrians), and what is one case: a whole image, a frame or a region?",
    "driving": "What should the model perceive or decide (for example detect pedestrians or predict a lane change), in which conditions (road type, weather, speed), "
               "and is one case a frame or a whole trip?",
    "maps": "What should be predicted on the road network (for example the speed or flow of each segment), and how far ahead?",
    "scenegraph": "Which objects, relations or events should the model recognise in the video, and over what stretch of time (a frame, a clip)?",
    "llm": "What should the fine-tuned model do (the task and the shape of its answer), and what examples do you already have for it?",
}


def opening_question(key: str | None) -> str:
    return OPENING.get(key or "", OPENING["tabular"])


def by_key(key: str | None) -> dict[str, Any] | None:
    return next((p for p in PACKS if p["key"] == key), None)


def _one_edit(a: str, b: str) -> bool:
    """Words a typist could have mixed up: one letter wrong, missing or extra, or two neighbours swapped."""
    if a == b or abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        wrong = [i for i in range(len(a)) if a[i] != b[i]]
        return len(wrong) == 1 or (len(wrong) == 2 and wrong[1] == wrong[0] + 1 and a[wrong[0]] == b[wrong[1]] and a[wrong[1]] == b[wrong[0]])
    short, long = sorted((a, b), key=len)
    return any(long[:i] + long[i + 1:] == short for i in range(len(long)))


_VOCABULARY = sorted({w for _, pattern in KEYWORDS for w in re.findall(r"[a-z]{5,}", pattern)})


def _typos(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Read a word that is one slip away from a keyword (fruad, forcast) as the keyword. Only words of five letters or
    more with the same first letter, and only when the sentence matched no keyword as typed."""
    fixed: list[tuple[str, str]] = []

    def swap(found: re.Match) -> str:
        word = found.group(0)
        if len(word) >= 5:
            for known in _VOCABULARY:
                if word[0] == known[0] and _one_edit(word, known):
                    fixed.append((word, known))
                    return known
        return word
    return re.sub(r"[a-z]+", swap, text), fixed


def detect(problem: str, analysis: dict[str, Any] | None = None, chosen: str | None = None) -> dict[str, Any]:
    """Return {"key", "source", "why", "signals"}; source is "user", "problem", "data" or "default"."""
    if chosen in KEYS:
        return {"key": chosen, "source": "user", "why": "You chose this pack.", "signals": []}
    text = (problem or "").lower()
    signals: list[str] = []
    from_text = next((key for key, pattern in KEYWORDS if re.search(pattern, text)), None)
    if not from_text:  # a typo in a keyword ("fruad") is read as the keyword, and the reason says so
        corrected, fixed = _typos(text)
        from_text = next((key for key, pattern in KEYWORDS if re.search(pattern, corrected)), None)
        if from_text:
            text = corrected
            typed = dict((known, wrong) for wrong, known in fixed)
    else:
        typed = {}
    if from_text:
        word = re.search(dict(KEYWORDS)[from_text], text).group(0).strip()
        signals.append(f'the problem mentions "{word}"' + (f' (typed "{typed[word]}")' if word in typed else ""))
    elif re.search(HORIZON, text) and not re.search(WHICH_ONES, text):
        from_text = "timeseries"
        signals.append(f'the problem mentions "{re.search(HORIZON, text).group(0).strip()}"')
    data_key = None
    if analysis:
        profile = analysis.get("profile") or {}
        kinds = (analysis.get("summary") or {}).get("kinds") or {}
        text_cols = profile.get("text_candidates") or []
        time_cols = profile.get("time_candidates") or []
        if text_cols:
            signals.append(f"long text in {', '.join(text_cols[:3])}")
            data_key = data_key or "text"
        if time_cols and kinds.get("datetime"):
            signals.append(f"a time column ({time_cols[0]})")
    if from_text:
        key, source = from_text, "problem"
    elif data_key:
        key, source = data_key, "data"
    else:
        key, source = "tabular", "default"
    why = ("Detected from " + " and ".join(signals) + ".") if signals else "Nothing points elsewhere, so DCLab starts from the tabular pack."
    return {"key": key, "source": source, "why": why, "signals": signals}
