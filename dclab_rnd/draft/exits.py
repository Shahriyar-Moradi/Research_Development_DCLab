"""Where each kind of help leads, and the other route a request's words may mean (package H8 of the Home plan).

- ``EXITS``: one exit per track. A prediction leaves as an ML project, an analysis as answers in the chat, a preparation as a
  prepared table, a report as a dashboard, a business question as a decision brief, advice as a cited answer. The draft's
  ``exit`` says which, and whether it is reached (an answer given, a project built), open (the way there is built) or not
  built yet (said, never promised).
- ``SUGGESTIONS``: three rules on the person's own words, each pointing to another route with its reason and the words that
  show it. A suggestion is offered only when it names another track than the one the draft is on; it never moves the draft
  (the person moves it on the card).

Rules only: the same words always give the same suggestion, in English and Persian, with no model.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from . import lang

EXITS = {"prediction": "ml_project", "analysis": "analysis", "preparation": "dataset", "reporting": "dashboard",
         "business": "decision_brief", "advice": "answer"}
BUILT = {"ml_project", "analysis", "dataset", "decision_brief", "dashboard"}  # the exits DCLab can take a person to today; the others are said as not built yet

_FA = r"[ء-غف-يٱ-ۓۺ-ۿ]"
_NEXT = r"(?:next|coming|upcoming|following)\s+(?:day|week|month|quarter|year|season)s?|tomorrow|soon|in\s+the\s+future"

# (rule, the route it points to, the pattern on the normalized text, why)
SUGGESTIONS: list[tuple[str, str, re.Pattern[str], str]] = [
    ("future_of_each", "prediction",
     re.compile(rf"\b(?:which|who|whom|what)\b[^.?!]{{0,80}}\b(?:will|going\s+to|gonna|likely\s+to|about\s+to|expected\s+to)\b[^.?!]{{0,60}}"
                rf"\b(?:{_NEXT})\b|(?:کدام|چه\s+کسانی|چه\s+مشتری)(?!{_FA})[^.؟!]{{0,80}}(?:روز|هفته|ماه|سال|فصل)\s+(?:آینده|بعدی?)(?!{_FA})", re.I),
     "You asked what will happen next for each one: that needs a model trained on the past; a report of the past cannot "
     "say it."),
    ("model_of_the_past", "reporting",
     # a named past period only: "a model of churn by plan" is a real prediction
     re.compile(r"\bmodel\s+of\s+(?:(?:the|our|my)\s+)?(?:last|this|previous|past)\s+(?:day|week|month|quarter|year)(?:'s|s)?\s+\w+", re.I),
     "A summary of what already happened, split by group, is a report: nothing has to be predicted."),
    ("rows_list", "preparation",
     re.compile(rf"^(?![^.?!]*\b(?:total|average|avg|mean|sum|count|per|median)\b)[^.?!]*?\b(?:list(?:ing)?|every|each|all)\b[^.?!]{{0,30}}\b(?:customers?|clients?|users?|orders?|accounts?|employees?|members?|people|persons?|patients?|subscribers?|students?|buyers?|guests?|bookings?|transactions?|leads?|contacts?|suppliers?|drivers?|riders?)\b[^.?!]{{0,60}}\bwith\s+(?:their|its|his|her)\s+(?:\w+[_ ])?(?:ids?|emails?|names?|phones?|addresses?)\b|"
                rf"(?:فهرست|لیست)(?!{_FA})[^.؟!]{{0,60}}(?:شناسه|ایمیل|شماره\s+(?:تماس|تلفن|موبایل))(?!{_FA})", re.I),
     "Home never lists single rows in the chat: a prepared table you download holds them, and stays with you."),
]


def exit_for(track: Optional[str]) -> Optional[dict[str, Any]]:
    """The exit a track leads to, with whether DCLab can take the person there today."""
    kind = EXITS.get(str(track or ""))
    if kind is None:
        return None
    return {"kind": kind, "track": track, "status": "open" if kind in BUILT else "not_built"}


# where a rule stays silent (the review of H8): a scored list with IDs is what a prediction delivers; a model of a past
# period is training data when the same request asks for the future; advice is about a method, not a table
_PREDICTS = re.compile(r"\b(?:predict\w*|forecast\w*|flag\w*|score\w*|risk|next|will|future|upcoming|coming)\b|پیش\s*بینی|آینده", re.I)
SILENT = {"rows_list": ("prediction", "advice"), "model_of_the_past": ("advice",), "future_of_each": ("advice",)}


def suggest(text: str, track: Optional[str]) -> Optional[dict[str, Any]]:
    """The first rule whose words are in the request and which points to another track: {track, rule, why, quote}."""
    norm, at = lang.normalize_map(text)
    for rule, route, pattern, why in SUGGESTIONS:
        if route == track or track in SILENT.get(rule, ()):
            continue
        if rule == "model_of_the_past" and _PREDICTS.search(norm):
            continue
        hit = pattern.search(norm)
        if hit:
            a, b = lang.span(at, hit.start(), hit.end())
            return {"track": route, "rule": rule, "why": why, "quote": str(text)[a:b].strip()}
    return None


def conversion(draft: dict[str, Any], target: Optional[str]) -> Optional[dict[str, Any]]:
    """H8: what an analysis already showed, carried into the prediction made from it: the columns the chat's answers
    related to the outcome (never an identifier), and the dates they covered, so the person can keep the holdout after them.
    Said as it is: DCLab's split does not choose that date by itself yet. None when nothing was explored (a draft with no
    exposure converts as before)."""
    from dclab_core import query as Q

    entries = [e for e in draft.get("exposure") or [] if isinstance(e, dict)]
    built = next((a for a in draft.get("assets") or [] if a.get("id") == draft.get("active_asset")), None) or {}
    tables = [a for a in draft.get("assets") or [] if a.get("status") == "ready" and a.get("kind") != "labels"]
    # only what was explored on the table the prediction is built on (the review: answers on table A said about table B); an
    # entry from before tables were named counts only when the draft never had another table
    entries = [e for e in entries if (e.get("table_sha256") == built.get("clean_sha256") if e.get("table_sha256")
                                      else len(tables) <= 1)]
    if not entries:
        return None
    ids = {c.get("name") for c in (draft.get("analysis") or {}).get("columns") or [] if c.get("id_like")}
    related = sorted({b if a == target else a for e in entries for a, b in e.get("columns_related") or [] if target and target in (a, b)}
                     - {target} - ids - {c for e in entries for pair in e.get("columns_related") or [] for c in pair if Q.ID_NAME.search(str(c))})
    periods = [e.get("period") or {} for e in entries]
    columns = {p.get("column") for p in periods}
    out: dict[str, Any] = {"answers": len(entries), "target": target, "related_to_target": related}
    if all(p.get("end") for p in periods) and len(columns) == 1:  # one date column, every answer bounded: compared as dates
        import pandas as pd

        end = max(pd.Timestamp(p["end"]) for p in periods).date().isoformat()
        out["holdout_after"] = {"column": next(iter(columns)), "date": end}
        out["note"] = (f"The answers in the chat covered {out['holdout_after']['column']} up to {end}. Rows after that date were never "
                       "looked at: split by time on that column and check the holdout starts after it (DCLab's split does not "
                       "choose this date by itself yet).")
    else:
        out["holdout_seen"] = True
        out["note"] = ("Answers in the chat covered rows the holdout may hold, so its number may not be a first look; a time split "
                       "on a column the answers did not reach keeps it clean (the final stage says so beside the holdout number).")
    if related:
        out["note"] += f" Already compared with {target} on the rows the answers covered: {', '.join(related)}; worth checking for leakage."
    return out


__all__ = ["EXITS", "BUILT", "SUGGESTIONS", "exit_for", "suggest", "conversion"]
