"""The numbers of a solution written without data: code owns them.

Two rules (the no-data plan of 2026-11-16):

- a number is the person's own, or it is a sum of the person's numbers that code has checked;
- a sum is written out where it is used (``90 / 2,400 = 3.75%``), so a reader and code can check it.

``known(words)`` writes the standard sums of a decision (what one action costs against what it gains, how many cases a
period holds) from the numbers the person stated, never from a default. ``verify(text, sources)`` reads every sum a text
writes: one whose operands are sourced and whose arithmetic is right becomes a source itself; a wrong one is reported.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

from dclab_rnd import cited

_NUM = r"[$€£]?\s?\d[\d,]*(?:\.\d+)?\s?%?"
_UNIT = r"(?:\s(?!x\s)[^\W\d_]{2,}){0,2}"  # "2,400 members × €45": a word or two naming what a number counts
_OP = r"(?:\s*[/÷×*]\s*|\s+[x+\-−–]\s+)"
_EXPR = rf"{_NUM}(?:{_UNIT}{_OP}{_NUM})+"
SUM = re.compile(rf"(?<![\w.])({_EXPR}){_UNIT}\s*(?:=|≈)\s*(?:about\s+|roughly\s+|around\s+)?\**({_NUM})\**(?![\w])")
# the same sum written the other way round: "3.75% (90 / 2,400)"
AFTER = re.compile(rf"(?<![\w.])({_NUM})\**\s*\(\s*({_EXPR})\s*\)")
_TOKEN = re.compile(rf"({_NUM})|([/÷×*x+\-−–])")
_LIST_MARK = re.compile(r"^(\s*(?:[-*+]\s+)?(?:\*\*)?)\d+[.)](?=(?:\*\*)?\s)", re.M)  # "11. " at a line's start numbers a step: it states nothing


def unmarked(text: str) -> str:
    """The text without the numbers that only number its list items, and with Persian number marks read as marks."""
    from dclab_rnd import cited as _cited

    return _LIST_MARK.sub(r"\1", _cited.marks(str(text or "")))


def _value(token: str) -> tuple[float, bool]:
    """A written number as a value: 8% is 0.08; says whether it was a share."""
    share = token.strip().endswith("%")
    bare = re.sub(r"[$€£,%\s]", "", token)
    return (float(bare) / 100.0 if share else float(bare)), share


def _compute(expression: str) -> float:
    """The value of ``a op b op c``: multiplication and division before addition and subtraction."""
    parts: list[Any] = []
    for number, op in _TOKEN.findall(expression):
        parts.append(_value(number)[0] if number else {"÷": "/", "×": "*", "x": "*", "−": "-", "–": "-"}.get(op, op))
    terms: list[Any] = [parts[0]]
    for op, value in zip(parts[1::2], parts[2::2]):  # first pass: * and /
        if op == "*":
            terms[-1] = terms[-1] * value
        elif op == "/":
            terms[-1] = terms[-1] / value
        else:
            terms += [op, value]
    total = terms[0]
    for op, value in zip(terms[1::2], terms[2::2]):
        total = total + value if op == "+" else total - value
    return total


def _close(computed: float, shown: str) -> bool:
    """The written result is the computed one, to the digits it shows (and never further off than one part in a hundred)."""
    value, share = _value(shown)
    bare = re.sub(r"[$€£,%\s]", "", shown)
    decimals = len(bare.split(".")[1]) if "." in bare else 0
    step = 0.5 * 10 ** (-decimals) / (100.0 if share else 1.0)
    return abs(computed - value) <= step + 1e-12  # exactly the rounding its own digits allow: "= 108,900" is not 108,000


def _sourced(token: str, allowed: set[str]) -> bool:
    """An operand is the person's, or an earlier checked result. A small number is no exception: "9 / 10 = 90%" would turn
    any invented share into a checked one (the review of 2026-11-16)."""
    return bool(cited.numbers(token)) and not (cited.numbers(token) - allowed)


def verify(text: str, sources: Iterable[str]) -> dict[str, list[Any]]:
    """Every sum the text writes. ``verified``: its text, when every operand is sourced (the person's words, an earlier
    verified sum, a bare count up to ten) and the arithmetic is right: its result is then a source. ``wrong``: the
    arithmetic is not right. ``unsourced``: an operand nobody gave."""
    sources = list(sources)
    allowed: set[str] = set().union(*(cited.numbers(s, source=True) for s in sources)) if sources else set()
    out: dict[str, list[Any]] = {"verified": [], "wrong": [], "unsourced": []}
    text = unmarked(text)
    found = sorted([(m.start(), m.group(1), m.group(2), m.group(0).strip()) for m in SUM.finditer(text)]
                   + [(m.start(), m.group(2), m.group(1), m.group(0).strip()) for m in AFTER.finditer(text)])
    for _, expression, shown, whole in found:
        expression = re.sub(r"(?<=[\d%])\s[^\W\d_]{2,}(?:\s[^\W\d_]{2,})?(?=\s*[/÷×*+\-−–]|\s+x\s)", "", expression)  # the unit words are not the sum
        operands = [number for number, _ in _TOKEN.findall(expression) if number]
        try:
            computed = _compute(expression)
        except (ValueError, ZeroDivisionError, IndexError):
            out["wrong"].append({"sum": whole, "why": "it cannot be computed"})
            continue
        if not all(_sourced(o, allowed) for o in operands):
            out["unsourced"].append({"sum": whole, "operands": [o.strip() for o in operands if not _sourced(o, allowed)]})
            continue
        if not _close(computed, shown):
            out["wrong"].append({"sum": whole, "why": f"it comes to {_show(computed, shown)}"})
            continue
        out["verified"].append(whole)
        allowed |= cited.numbers(whole, source=True)
    return out


def _show(value: float, like: str = "") -> str:
    """A computed value written plainly: a share as a percent with at most two decimals, a count with separators."""
    if like.strip().endswith("%"):
        return f"{round(value * 100, 2):g}%"
    return f"{value:,.2f}".rstrip("0").rstrip(".") if abs(value) < 1000 and value != int(value) else f"{round(value):,}"


def stated(words: str) -> dict[str, float]:
    """The decision's inputs the person stated in digits (how many cases a period holds, how often the outcome happens,
    what one action costs, what one caught case is worth), read by the rules of the first no-data design; an input they
    did not state is left out, never filled with a default."""
    from dclab_rnd.draft import no_data

    try:
        inputs = no_data.compute_decision_economics(str(words or "")).get("inputs") or {}
    except Exception:  # noqa: BLE001 - a reading that fails is a person who stated nothing the rules can read
        return {}
    given = set(cited.numbers(str(words or ""), source=True))
    out: dict[str, float] = {}
    for name in ("monthly_volume", "base_rate", "action_cost", "value_per_positive"):
        value = inputs.get(name)
        if value is None or inputs.get(f"{name}_assumed", True):
            continue
        shown = f"{round(float(value) * 100, 4):g}%" if name == "base_rate" else f"{float(value):g}"
        if cited.numbers(shown) - given:  # read by the rules, yet not a number in the person's words: not theirs
            continue
        out[name] = float(value)
    return out


def known(words: str) -> list[dict[str, str]]:
    """The sums code can write from the person's stated inputs alone, each with what it means. Nothing is assumed: a sum
    whose inputs were not both stated is not written."""
    s = stated(words)
    sums: list[dict[str, str]] = []
    count = lambda v: f"{v:,.0f}" if v == int(v) else f"{v:,.2f}"  # noqa: E731
    share = lambda v: f"{round(v * 100, 2):g}%"  # noqa: E731
    cost, value, volume, rate = s.get("action_cost"), s.get("value_per_positive"), s.get("monthly_volume"), s.get("base_rate")
    if cost and value:
        sums.append({"sum": f"{count(cost)} / {count(value)} = {share(cost / value)}",
                     "means": "the cost of one action against the value of one case it catches: the share of actions that must "
                              "succeed for acting to pay, if a caught case always brings its whole value"})
    if volume and rate:
        sums.append({"sum": f"{count(volume)} × {share(rate)} = {count(volume * rate)}",
                     "means": "how many cases with the outcome the stated volume holds at the stated rate"})
    if volume and cost:
        sums.append({"sum": f"{count(volume)} × {count(cost)} = {count(volume * cost)}", "means": "what acting on every case would cost"})
    if volume and rate and value:
        sums.append({"sum": f"{count(volume)} × {share(rate)} × {count(value)} = {count(volume * rate * value)}",
                     "means": "the most that could be gained if every case with the outcome were caught and the action always worked"})
    return sums


def unsaid(text: str, sources: Iterable[str]) -> list[str]:
    """The numbers a text states that neither a source nor a verified sum of its own holds. The number of a list item
    ("11. Call the members") is not a statement."""
    sources = list(sources)
    return cited.unsaid(unmarked(text), [*sources, *verify(text, sources)["verified"]])


def as_sources(sums: Iterable[Mapping[str, str] | str]) -> list[str]:
    return [s if isinstance(s, str) else str(s.get("sum") or "") for s in sums]
