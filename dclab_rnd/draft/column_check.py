"""Columns kept out on statistics alone are asked about before a build (package H3.1, from a live run).

The owner's HyperAck table went through DCLab Code's chat with a person who answered every question "Use your
recommendation" (2026-10-28): the column audit kept `deliverey_category_id` out because it alone separated the classes
(AUC 0.816), DCLab's own leakage reviewer read it as available when the order is placed, and Home never asked, so the
holdout was 0.664 against the research's honest 0.945. A strong column is not a leak by its strength alone: whether it
is known at the prediction moment is a fact about the business, which the person knows.

So before a build, every forbidden column that is

- flagged by statistics only (it separates the classes, or tracks the target; no name signal, no leakage precedent), or
- read as "available" by the leakage reviewer,

is a question, one at a time, with its evidence. The answer decides: known at the moment keeps it, written later keeps it
out. A person who does not know, or says "use your recommendation", gets DCLab's recommendation, which keeps the column
only when it was flagged by statistics alone and the reviewer read it as available and said when it is set (the owner's
choice, 2026-10-28). Anything else stays out, and the solution's notes say why for every column.
"""

from __future__ import annotations

import re
from typing import Any, Optional

FIELD = "column_check"
STATISTIC = re.compile(r"^\s*single column (?:separates the classes|tracks the target)\b", re.I)
# the reviewer said when the column is set ("chosen when the order is placed", "known at order creation")
NAMES_MOMENT = re.compile(r"\b(?:chosen|set|known|recorded|entered|selected|filled|assigned|available|fixed|decided|written)\b"
                          r"[^.;]*?\b(?:when|at|before|on|during)\b", re.I)
RECOMMEND = re.compile(r"\b(?:use|go\s+with|follow|take|accept)\s+(?:your|the|dclab'?s)\s+(?:recommend\w*|suggest\w*|advice|call)\b"
                       r"|\bwhat(?:ever)?\s+you\s+(?:think|recommend|suggest)\b|\byou\s+(?:decide|choose|pick)\b|\byour\s+call\b"
                       r"|\bas\s+you\s+(?:recommend|suggest)\b|\bnot\s+sure\b|\bdon'?t\s+know\b|پیشنهاد|توصیه|نمی\s*دانم|مطمئن\s+نیستم", re.I)
KNOWN = re.compile(r"^\W*(?:yes|yeah|yep|correct|right|known|it\s+is\s+known|available|it'?s\s+(?:known|available|set\s+before)|"
                   r"known\s+(?:then|at|before)|set\s+(?:before|at|when)|بله|آره|معلوم\s+است)\b", re.I)
LATER = re.compile(r"^\W*(?:no|nope|not\s+known|later|after|written\s+(?:later|after)|it'?s\s+(?:written|set|filled)\s+(?:later|after)|"
                   r"filled\s+(?:in\s+)?later|نه|بعدا)\b", re.I)
OPTIONS = ("Yes, it is known then", "No, it is written later", "Not sure: use your recommendation")
# Words that make an answer or a reviewer's reason about time late or unsure (review of H3.1: "yes it is written later" and
# "known after the delivery" read as known; "probably set at creation, though it could be updated after" counted as naming the moment).
LATE = re.compile(r"\b(?:later|after|afterwards?|not|no|never|n't|written|updated|changed|filled\s+in|until|once|"
                  r"complet\w*|finish\w*|closed|resolved|settled|end(?:ed|s)?)\b|بعد|نه", re.I)
HEDGE = re.compile(r"\b(?:might|may|maybe|possibly|probably|perhaps|likely|unlikely|unclear|uncertain|could|unless|though|although|"
                   r"however|depend\w*|assum\w*|presumably|seems?|appears?|guess\w*|think|suppose\w*|believe\w*)\b", re.I)
# The proof a forbidden column carries from the audit's statistics alone (tools.leakage_scan's rules R04 and R05); any other
# proof (a precedent, the research's own forbidden list R01, a label source, a decision model) is never asked about.
STAT_PROOF = {"DCLAB-R04", "DCLAB-R05"}


def _statistics_only(reason: str) -> bool:
    parts = [p for p in re.split(r";\s*", str(reason or "")) if p.strip()]
    return bool(parts) and all(STATISTIC.match(p) for p in parts)


def names_moment(reason: str) -> bool:
    """The reviewer said when the column is set, plainly: a clause that says it is known or set at a time, with no hedge
    and nothing late or negated in that clause (the reviewer's reasons always quote the moment, so a quote alone says nothing)."""
    for clause in re.split(r"[.;]\s*", str(reason or "")):
        if NAMES_MOMENT.search(clause) and not HEDGE.search(clause) and not LATE.search(clause):
            return True
    return False


def disputed(proposal: dict[str, Any], time_column: Optional[str] = None) -> list[dict[str, Any]]:
    """The forbidden columns worth a question, each with its evidence: only a column the audit kept out on statistics
    alone (no name signal, no precedent, no other proof); with the reviewer's verdict and words, and whether the reviewer
    said plainly when it is set. A column read "available" by the reviewer but kept out for any other reason is never asked."""
    review = proposal.get("review") or {}
    verdicts = {v.get("column"): v.get("verdict") for v in review.get("verdicts") or [] if isinstance(v, dict)}
    says = {d.get("column"): str(d.get("reason") or "") for d in review.get("disagreements") or [] if isinstance(d, dict)}
    out = []
    for f in proposal.get("forbidden") or []:
        column = f.get("column")
        if not column or column == time_column:
            continue
        stats = _statistics_only(f.get("reason") or "") and set(f.get("proof") or STAT_PROOF) <= STAT_PROOF
        if not stats:
            continue
        verdict = verdicts.get(column)
        reviewer_says = says.get(column, "")
        out.append({"column": column, "evidence": str(f.get("reason") or "")[:300], "statistics_only": True,
                    "reviewer": verdict, "reviewer_says": reviewer_says[:400],
                    "names_moment": verdict == "available" and names_moment(reviewer_says)})
    return out


def recommended(item: dict[str, Any]) -> bool:
    """DCLab's recommendation: keep the column only when statistics alone flagged it and the reviewer read it as
    available and said when it is set. A name signal or a leakage precedent is never overruled by the recommendation."""
    return bool(item.get("statistics_only") and item.get("reviewer") == "available" and item.get("names_moment"))


def question(item: dict[str, Any], moment: str) -> str:
    """The question, with its evidence, in plain words."""
    column = item["column"]
    text = f"`{column}` was kept out only because it is strong: {item['evidence']}." if item.get("statistics_only") else \
        f"`{column}` was kept out: {item['evidence']}."
    if item.get("reviewer") == "available":
        text += f" DCLab's leakage reviewer reads it as known in time: “{item['reviewer_says'] or 'available at the prediction moment'}”."
    text += f" Is `{column}` known at the prediction moment ({moment.strip().rstrip('.')})?"
    rec = "keep it" if recommended(item) else "keep it out"
    return f"{text} If you are not sure, I recommend to {rec}."


def read_answer(said: str) -> str:
    """known, later or recommendation (a person who does not know, or who leaves it to DCLab)."""
    text = str(said or "").strip()
    if text in OPTIONS:
        return ("known", "later", "recommendation")[OPTIONS.index(text)]
    if RECOMMEND.search(text):
        return "recommendation"
    if LATE.search(text) or LATER.search(text):  # "yes, but only after dispatch" is late, whatever word starts it
        return "later"
    if KNOWN.search(text) and not HEDGE.search(text):
        return "known"
    return "recommendation"


def current(answer: Any, target: Optional[str], moment: Optional[str]) -> bool:
    """An answer counts only for the outcome and the moment it was given about (review of H3.1: a new version of the
    project copies the understanding with a new target)."""
    return isinstance(answer, dict) and answer.get("target") == target and str(answer.get("moment") or "").strip() == str(moment or "").strip()


def decide(item: dict[str, Any], answer: dict[str, Any] | str) -> tuple[bool, str]:
    """(keep the column as a feature, why), from the person's answer."""
    kind = answer.get("answer") if isinstance(answer, dict) else str(answer)
    column = item["column"]
    if kind == "known":
        return True, f"`{column}`: the person said it is known at the prediction moment, so it is a feature."
    if kind == "later":
        return False, f"`{column}`: the person said it is written later, so it stays out."
    if recommended(item):
        return True, (f"`{column}`: flagged by statistics alone ({item['evidence']}) and read by the leakage reviewer as known "
                      f"in time (“{item['reviewer_says'][:200]}”); kept on DCLab's recommendation, the person left it to DCLab.")
    return False, f"`{column}`: kept out on DCLab's recommendation ({item['evidence']}); the person left it to DCLab."


__all__ = ["FIELD", "OPTIONS", "current", "decide", "disputed", "names_moment", "question", "read_answer", "recommended"]
