"""Redaction of free text before anything leaves a workspace (package 19.10).

:func:`redact_text` replaces what could identify a person or a business in what people and agents wrote: e-mail
addresses, phone numbers, card and account numbers (IBAN), web and IP addresses, long ids, amounts of money, street
addresses, and probable names (two to four capitalised words together once the product's own words around them are
set aside: "the Owner Jane Roe" keeps "the Owner"; a single capitalised word passes). Each becomes a bracketed placeholder (``[EMAIL]``, ``[NAME]``…), and the
counts say what was removed. It errs on removing: a training example loses a word, never leaks one.

It is a rule, not a guarantee: a name written in lower case, or a value with no shape, passes. That is why a table's
values (``cell_values``) and model prompts never leave at all, redacted or not (``dclab_core.experience`` classes).
"""

from __future__ import annotations

import re
from collections import Counter

DATE = r"\d{4}-\d{2}-\d{2}"
PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")),
    ("URL", re.compile(r"\b(?:https?://|www\.)\S+", re.I)),
    ("IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?:\s?[A-Z0-9]{4}){3,7}(?:\s?[A-Z0-9]{1,3})?\b")),
    ("CARD", re.compile(r"\b(?:\d[ -]?){12,18}\d\b")),  # kept only when it passes the Luhn check (below)
    ("IP", re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
    ("PHONE", re.compile(r"(?<![\w.])(?:(?:\+|00)\d[\d\s().-]{7,}\d|\(?0\d{1,4}\)?[\s.-]?\d{3,4}[\s.-]?\d{3,4}|\d{3}[\s.-]\d{3}[\s.-]\d{4})\b")),  # a date is none of these shapes
    ("AMOUNT", re.compile(r"(?:[$€£¥]\s?\d[\d,.\s]*\d|\b\d[\d,.]*\s?(?:EUR|USD|GBP|AED|IRR|euros?|dollars?|dirhams?|rials?)\b)", re.I)),
    ("ADDRESS", re.compile(r"\b\d{1,5}\s+(?:[A-Z][a-z]+\s){0,3}(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Way|Square|Sq)\b\.?")),
    # ids: a hyphenated or a mixed run of letters and digits with at least three digits (ORD12345, u3fa9c1d2e4b5a6), or six
    # digits and more; a date, "5-fold" and the product's own codes (WF-01) pass
    ("ID", re.compile(r"(?<![\w-])(?!" + DATE + r"\b)(?!\d+-fold\b)(?![A-Z]{2,5}-\d{1,3}\b)(?=(?:[A-Za-z-]*\d){3})[A-Za-z0-9]+(?:-[A-Za-z0-9]+)+\b"
                      r"|\b(?=[A-Za-z]*\d)(?=(?:[A-Za-z]*\d){3})(?=[0-9]*[A-Za-z])[A-Za-z0-9]{6,}\b|\b\d{6,}\b")),
)
# capitalised words the product writes itself: a run of them is a title, not a person
KNOWN_WORDS = frozenset({
    "DCLab", "Home", "Admin", "Sessions", "Decisions", "Compute", "Jobs", "Evidence", "Library", "Research", "Lab", "Domain",
    "Packs", "Policy", "Model", "Models", "Notebook", "Intern", "Solution", "Brief", "Project", "Projects", "Data", "Leakage",
    "Features", "Final", "Stage", "Stages", "Workflow", "Report", "Owner", "Reviewer", "Business", "Viewer", "ML", "Engineer",
    "Python", "PostgreSQL", "Kaggle", "Hugging", "Face", "Google", "Cloud", "Amazon", "OpenAI", "Jev", "The", "This", "That",
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February", "March", "April",
    "May", "June", "July", "August", "September", "October", "November", "December", "ROC", "AUC", "MAE", "RMSE", "CV",
    "Ask", "Call", "Send", "Write", "Find", "Predict", "Which", "What", "When", "Where", "Who", "How", "Why", "Is", "Are",
    "Can", "Please", "Our", "We", "My", "I", "It", "A", "An", "If", "For", "From", "To", "In", "On", "At", "And", "Or",
})
NAME = re.compile(r"\b[A-Z][a-z]{1,20}(?:\s+[A-Z][a-z]{1,20}){1,3}\b")


def _luhn(digits: str) -> bool:
    total = 0
    for i, d in enumerate(reversed(digits)):
        n = int(d) * (2 if i % 2 else 1)
        total += n - 9 if n > 9 else n
    return total % 10 == 0


def redact_text(text: str | None) -> tuple[str | None, dict[str, int]]:
    """``text`` with what could identify someone replaced by placeholders, and how many of each were replaced."""
    if not text:
        return text, {}
    counts: Counter[str] = Counter()
    out = str(text)
    for label, pattern in PATTERNS:
        if label == "CARD":
            def card(match: re.Match[str]) -> str:
                digits = re.sub(r"\D", "", match.group(0))
                if 13 <= len(digits) <= 19 and _luhn(digits) and digits.count("0") * 2 < len(digits):  # 1000 2000 3000 4000 is a count
                    counts["CARD"] += 1
                    return "[CARD]"
                return match.group(0)

            out = pattern.sub(card, out)
            continue
        out, n = pattern.subn(f"[{label}]", out)
        counts[label] += n

    def name(match: re.Match[str]) -> str:
        words = match.group(0).split()
        lead = 0
        while lead < len(words) and words[lead] in KNOWN_WORDS:
            lead += 1
        tail = len(words)
        while tail > lead and words[tail - 1] in KNOWN_WORDS:
            tail -= 1
        middle = words[lead:tail]
        if len(middle) < 2 or any(w in KNOWN_WORDS for w in middle):  # one word alone passes; a title of the product's words too
            return match.group(0)
        counts["NAME"] += 1
        return " ".join(words[:lead] + ["[NAME]"] + words[tail:])

    out = NAME.sub(name, out)
    return out, {k: v for k, v in counts.items() if v}
