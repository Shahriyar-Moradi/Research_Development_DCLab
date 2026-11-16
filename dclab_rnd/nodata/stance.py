"""Whether a person has no table to share, read from natural sentences (English and Persian), with no set phrase.

The lead model reads this from the conversation itself; these patterns are the reading when no model serves, and a check
on it. ``read(words)`` says why there is no table (none yet, private, not ready) and quotes the words that say so. A
person who shared a table is never read as having none."""

from __future__ import annotations

import re
from typing import Any, Optional

_DATA = r"(?:data|dataset|table|history|records?|export|logs?)"
WHY = {
    "private": re.compile(
        r"\b(?:can(?:not|'t| not)(?!\s+wait)|won't|will not|do(?:es)? not want to|don't want to|not (?:allowed|able|permitted) to|rather not|prefer not to)\s+"
        r"(?:\w+\s+){0,3}?(?:upload|share|send|hand over|give|expos|provid|export)\w*"
        r"|\bwithout (?:upload|shar|send|export)\w*"
        r"|\b(?:for|because of|due to|under)\s+(?:\w+\s+){0,2}?(?:privacy|confidentiality|compliance|hipaa|gdpr)\b"
        rf"|\b{_DATA}\b[^.]{{0,20}}\b(?:must|has to|have to|needs? to|should) stay\b"
        r"|\b(?:cannot|can't|may not|must not) leave (?:our|the) (?:servers?|premises|company|building|network)\b"
        r"|محرمان|نمی‌?توان\w*\s+(?:\S+\s+){0,3}?(?:آپلود|ارسال|به اشتراک)|بدون آپلود|اجازه\s+(?:\S+\s+){0,2}?ندار", re.I),
    "none": re.compile(
        rf"\b(?:no(?!\s+(?:problem|issue|trouble|missing|doubt|bad|wrong))|don't have (?:any|a|the)?|do not have (?:any|a|the)?|have no(?!\s+(?:problem|issue|trouble|doubt))|"
        rf"without (?:any)?(?!\s+(?:any\s+)?(?:data|table)\s+(?:issues?|problems?))|not (?:yet )?(?:logging|recording|collecting|tracking))\s*(?:\w+\s+){{0,3}}?{_DATA}\b(?!\s+(?:issues?|problems?|errors?|gaps?))"
        rf"|\bthere (?:is|'s) no (?!missing)(?:\w+\s+){{0,2}}?{_DATA}\b|\bthere (?:isn't|is not|aren't|are not) any (?:\w+\s+){{0,2}}?{_DATA}\b"
        r"|\b(?:haven't|have not|hasn't|has not|never) (?:yet )?(?:collected|recorded|logged|gathered|tracked|started (?:logging|recording|collecting))\b"
        r"|\bnothing (?:is )?(?:in a system|clean|to (?:share|export|upload|send))\b|\bbefore we have\b"
        r"|\b(?:new|launching|opening)\b[^.]{0,40}\b(?:no|without)\b[^.]{0,20}\b(?:data|history)\b"
        r"|دیتا ندار|داده\s*(?:ای)?\s*ندار|هنوز\s+(?:دیتا|داده)|(?:دیتایی|داده‌?ای) در کار نیست", re.I),
    "not_ready": re.compile(
        r"\b(?:data|dataset|export|table)\b[^.]{0,30}\b(?:not|isn't|is not|aren't) (?:yet )?(?:ready|clean|available)\b"
        r"|\b(?:no|not a|don't have a) clean (?:data|dataset|table|export)\b|\bdon't know how to (?:get|export|pull)\b"
        r"|\bdo not know how to (?:get|export|pull)\b|آماده نیست", re.I),
}
ORDER = ("private", "none", "not_ready")


def read(words: str, has_table: bool = False) -> dict[str, Any]:
    """``{"no_table", "why", "quote"}``. With a shared table: never a no-table stance, whatever the words say."""
    text = str(words or "")
    if has_table:
        return {"no_table": False, "why": None, "quote": None}
    for why in ORDER:
        found = WHY[why].search(text)
        if found:
            return {"no_table": True, "why": why, "quote": _sentence(text, found.start(), found.end())}
    return {"no_table": False, "why": None, "quote": None}


def _sentence(text: str, start: int, end: int) -> Optional[str]:
    left = max(text.rfind(c, 0, start) for c in ".!?\n") + 1
    rights = [i for i in (text.find(c, end) for c in ".!?\n") if i != -1]
    return text[left:(min(rights) + 1 if rights else len(text))].strip() or None
