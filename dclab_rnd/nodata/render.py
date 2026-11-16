"""The solution as the person reads it: the architect's answer, then what code did to it and found in it."""

from __future__ import annotations

import re
from typing import Any, Mapping

from . import checks

TITLES = {"business": "A business answer, without data", "data_design": "What to record", "full": "A solution designed without data",
          "simulate": "A demonstration on made-up data: the design comes first"}
# A Persian speaker reads Persian headings (2026-11-16: the body was Persian under English headings). The architect keeps
# the English ones, which code reads; they are replaced only in what the person is shown.
PERSIAN = re.compile(r"[\u0600-\u06FF]")
FA = {"A business answer, without data": "پاسخ کسب‌وکاری، بدون داده", "What to record": "چه چیزهایی را ثبت کنید",
      "A solution designed without data": "راه‌حل طراحی‌شده بدون داده", "A demonstration on made-up data: the design comes first": "نمایش روی دادهٔ ساختگی: اول طراحی",
      "The problem": "مسئله", "What needs care": "آنچه باید مراقبش بود", "Options": "گزینه‌ها", "Business solution": "راه‌حل کسب‌وکاری",
      "Data to record": "داده‌هایی که باید ثبت شود", "Model": "مدل", "Plan": "برنامه", "What only real data can settle": "آنچه فقط دادهٔ واقعی روشن می‌کند",
      "Questions for you": "پرسش‌هایی از شما", "Checked by code": "بررسی‌شده با کد",
      "Checks DCLab will run when real data arrives": "بررسی‌هایی که DCLab با رسیدن دادهٔ واقعی اجرا می‌کند", "A second reader's findings": "یافته‌های خوانندهٔ دوم",
      "Sums checked by code: ": "جمع‌هایی که کد بررسی کرد: ",
      "Code took out what stated a number you did not give (a sentence, a list item, or the number alone in a table).":
          "کد هر جا عددی آمده بود که شما نگفته بودید، آن را برداشت (یک جمله، یک مورد فهرست، یا خود عدد در جدول).",
      "column": "ستون", "what it is": "چیست", "known when": "کی معلوم است", "role": "نقش"}


def persian(words: Any) -> bool:
    text = str(words or "")
    return len(PERSIAN.findall(text)) > max(8, len(text) // 4)


def _say(label: str, fa: bool) -> str:
    return FA.get(label, label) if fa else label


def localised(body: str, fa: bool) -> str:
    """The architect's answer with its headings and the table's header in the person's language."""
    if not fa:
        return body
    out = []
    for line in body.split("\n"):
        found = checks.HEADING.match(line)
        name = next((h for h in checks.HEADINGS if found and h.lower() == found.group(1).strip().strip("*:").strip().lower()), None)
        if name:
            out.append(f"## {FA[name]}")
        elif line.strip().startswith("|") and "known when" in line.lower() and "role" in line.lower():
            out.append("| " + " | ".join(FA[c] for c in ("column", "what it is", "known when", "role")) + " |")
        else:
            out.append(line)
    return "\n".join(out)


def note_sections(solution: Mapping[str, Any]) -> list[dict[str, Any]]:
    """What code adds under the architect's answer, as titled lists: the sums it checked, what it took out, what still
    fails, the checks it will run on real data, and the second reader's findings. Empty ones are left out."""
    out: list[dict[str, Any]] = []
    fa = persian(solution.get("words"))
    sums = solution.get("sums") or {}
    checked = [*[s["sum"] for s in sums.get("code") or []], *(sums.get("verified") or [])]
    notes: list[str] = []
    if checked:
        notes.append(_say("Sums checked by code: ", fa) + "; ".join(f"`{s}`" for s in dict.fromkeys(checked)) + ".")
    if solution.get("removed"):  # said, never quoted: quoting would show the person the very number that was taken out
        notes.append(_say("Code took out what stated a number you did not give (a sentence, a list item, or the number alone in a table).", fa))
    for c in checks.failed(solution.get("checks") or []):
        notes.append(f"Not passed ({c['check']}): " + ("a number you did not give is still stated." if c["check"] == "numbers" else c["detail"]))
    if notes:
        out.append({"title": _say("Checked by code", fa), "items": notes})
    tasks = (solution.get("planner") or {}).get("calls") or []
    if tasks:
        out.append({"title": _say("Checks DCLab will run when real data arrives", fa), "items": [
            f"`{t.get('task_type')}`" + (f" on `{next(iter((t.get('params') or {}).values()))}`" if t.get("params") else "") + f": {t.get('reason')}" for t in tasks]})
    if solution.get("critic"):
        out.append({"title": _say("A second reader's findings", fa), "items": list(solution["critic"])})
    return out


def code_notes(solution: Mapping[str, Any]) -> str:
    return "\n\n".join(f"## {s['title']}\n" + "\n".join(f"- {i}" for i in s["items"]) for s in note_sections(solution))


def markdown(solution: Mapping[str, Any]) -> str:
    """The whole answer in Markdown. The architect's first line (``Need: ...``) is for code and is not shown."""
    if solution.get("mode") == "rules":
        return str(solution.get("text") or "")
    fa = persian(solution.get("words"))
    body = localised(checks.without_need_line(str(solution.get("text") or "")), fa)
    title = f"# {_say(TITLES.get(solution.get('need') or '', 'A solution designed without data'), fa)}"
    notes = code_notes(solution)
    return "\n\n".join(p for p in (title, body, notes) if p)


_ITEM = re.compile(r"^\s*(?:\d+[.)]|[-*+])\s+\S")


def tight(text: str) -> str:
    """A list whose items are set apart by blank lines, as one list. The Home page starts a new list after a blank line,
    so a plan written with a blank line between its steps showed every step as "1." (the browser walk of 2026-11-16)."""
    lines = str(text or "").split("\n")
    out: list[str] = []
    for i, line in enumerate(lines):
        if not line.strip():
            before = next((x for x in reversed(out) if x.strip()), "")
            after = next((x for x in lines[i + 1:] if x.strip()), "")
            if _ITEM.match(before) and _ITEM.match(after):
                continue  # a blank line between two items of a list
        out.append(line)
    return "\n".join(out)


def card(solution: Mapping[str, Any]) -> dict[str, Any]:
    """The Home card: only the parts that were written, each under the heading the person reads; the table of columns
    apart from its part's words, so the page can draw it as a table."""
    parts = solution.get("parts") or {}
    fa = persian(solution.get("words"))
    shown = []
    for h in checks.HEADINGS:
        if not parts.get(h):
            continue
        text = "\n".join(line for line in parts[h].split("\n") if not line.strip().startswith("|")).strip() if h == "Data to record" else parts[h]
        shown.append({"key": h, "title": _say(h, fa), "text": tight(text)})
    return {"need": solution.get("need"), "mode": solution.get("mode"), "title": _say(TITLES.get(solution.get("need") or "", TITLES["full"]), fa),
            "parts": shown, "columns": list(solution.get("columns") or []), "plan": list(solution.get("plan") or []), "kind": solution.get("kind"),
            "notes": note_sections(solution), "questions": list(solution.get("questions") or []), "rtl": fa,
            "column_titles": [_say(c, fa) for c in ("column", "what it is", "known when", "role")]}
