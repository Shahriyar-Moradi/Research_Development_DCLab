"""The solution as the person reads it: the architect's answer, then what code did to it and found in it."""

from __future__ import annotations

from typing import Any, Mapping

from . import checks

TITLES = {"business": "A business answer, without data", "data_design": "What to record", "full": "A solution designed without data",
          "simulate": "A demonstration on made-up data: the design comes first"}


def code_notes(solution: Mapping[str, Any]) -> str:
    """What code adds under the architect's answer: the sums it checked, what it removed, what still fails, the checks it
    will run on real data, and the second reader's findings. Empty parts are left out."""
    out: list[str] = []
    sums = solution.get("sums") or {}
    checked = [*[s["sum"] for s in sums.get("code") or []], *(sums.get("verified") or [])]
    notes: list[str] = []
    if checked:
        notes.append("Sums checked by code: " + "; ".join(f"`{s}`" for s in dict.fromkeys(checked)) + ".")
    if solution.get("removed"):  # said, never quoted: quoting would show the person the very number that was taken out
        notes.append("Code took out what stated a number you did not give (a sentence, a list item, or the number alone in a table).")
    for c in checks.failed(solution.get("checks") or []):
        notes.append(f"Not passed ({c['check']}): " + ("a number you did not give is still stated." if c["check"] == "numbers" else c["detail"]))
    if notes:
        out.append("## Checked by code\n" + "\n".join(f"- {n}" for n in notes))
    tasks = (solution.get("planner") or {}).get("calls") or []
    if tasks:
        out.append("## Checks DCLab will run when real data arrives\n" + "\n".join(
            f"- `{t.get('task_type')}`" + (f" on `{next(iter((t.get('params') or {}).values()))}`" if t.get("params") else "") + f": {t.get('reason')}" for t in tasks))
    if solution.get("critic"):
        out.append("## A second reader's findings\n" + "\n".join(f"- {f}" for f in solution["critic"]))
    return "\n\n".join(out)


def markdown(solution: Mapping[str, Any]) -> str:
    """The whole answer in Markdown. The architect's first line (``Need: ...``) is for code and is not shown."""
    if solution.get("mode") == "rules":
        return str(solution.get("text") or "")
    body = checks.without_need_line(str(solution.get("text") or ""))
    title = f"# {TITLES.get(solution.get('need') or '', 'A solution designed without data')}"
    notes = code_notes(solution)
    return "\n\n".join(p for p in (title, body, notes) if p)


def card(solution: Mapping[str, Any]) -> dict[str, Any]:
    """The Home card: only the parts that were written."""
    parts = solution.get("parts") or {}
    return {"need": solution.get("need"), "mode": solution.get("mode"), "title": TITLES.get(solution.get("need") or "", TITLES["full"]),
            "parts": [{"title": h, "text": parts[h]} for h in checks.HEADINGS if parts.get(h)],
            "columns": list(solution.get("columns") or []), "plan": list(solution.get("plan") or []), "kind": solution.get("kind"),
            "notes": code_notes(solution), "questions": list(solution.get("questions") or []), "markdown": markdown(solution)}
