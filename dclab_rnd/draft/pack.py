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


def by_key(key: str | None) -> dict[str, Any] | None:
    return next((p for p in PACKS if p["key"] == key), None)


def detect(problem: str, analysis: dict[str, Any] | None = None, chosen: str | None = None) -> dict[str, Any]:
    """Return {"key", "source", "why", "signals"}; source is "user", "problem", "data" or "default"."""
    if chosen in KEYS:
        return {"key": chosen, "source": "user", "why": "You chose this pack.", "signals": []}
    text = (problem or "").lower()
    signals: list[str] = []
    from_text = next((key for key, pattern in KEYWORDS if re.search(pattern, text)), None)
    if from_text:
        word = re.search(dict(KEYWORDS)[from_text], text).group(0).strip()
        signals.append(f'the problem mentions "{word}"')
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
