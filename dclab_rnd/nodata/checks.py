"""The strict layer of a solution written without data: code reads the architect's answer and checks it.

The architect writes freely under fixed headings; nothing here writes the solution. Each check is a rule a reader could
apply: a number is the person's or a checked sum; an input is known before the decision; the outcome is never an input;
there is a plan; a person who did not ask for a model gets none; nobody is asked to upload. A failed check is sent back
once (``architect.repair``); what still fails is shown to the person, never hidden.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Optional

from . import economics

NEEDS = ("business", "data_design", "full", "simulate")
HEADINGS = ("The problem", "What needs care", "Options", "Business solution", "Data to record", "Model", "Plan", "What only real data can settle",
            "Questions for you")
# "Options" (2026-11-16, after the first measurement): the answers had no place to weigh alternatives, which the model alone
# does unasked and an expert expects
ALWAYS = ("The problem", "What needs care", "Options", "Business solution", "Plan", "What only real data can settle")
WITH_DATA = ("data_design", "full", "simulate")
WITH_MODEL = ("full", "simulate")
MIN_COLUMNS = 4
MIN_STEPS = 3
NEED = re.compile(r"^[\s`*_>]*need[`*_]*\s*:[\s`*_]*([a-z_]+)", re.I)
HEADING = re.compile(r"^(?:#{1,4}\s*|\*\*(?=[^*\n]+\*\*\s*$))(?:\d+[.)]\s*)?(.+?)[\s:#*]*$", re.M)
STEP = re.compile(r"^\s*(?:\*\*)?\d+[.)](?:\*\*)?\s+\S", re.M)
KIND = re.compile(r"^[\s`*_]*kind[`*_]*\s*:[\s`*_]*(.+?)\s*$", re.I | re.M)
FIRST_LINES = 3  # the need is named at the top: a later "Need: trainers' time" in a plan step is not it
NO_LABEL_KINDS = re.compile(r"cluster|segment|unsupervised|anomal|group", re.I)  # kinds of problem with no outcome to record

# A request for the person's data, read sentence by sentence: the assistant asks to be sent, shown or given their file.
# A sentence that says it is not needed is never a request. The test's counter uses this same reading.
_THING = r"(?:csv|file|sample|export|table|data ?set|dataset|data|spreadsheet|rows|records|extract)"
_ASKS = re.compile(
    rf"\b(?:send|share|give|provide|show|paste|attach|export|upload)\s+(?:me|us)\b[^.?!\n]{{0,60}}\b{_THING}\b"
    rf"|\b(?:share|send|provide|attach|export|upload)\b[^.?!\n]{{0,50}}\b{_THING}\b[^.?!\n]{{0,30}}\bwith (?:me|us)\b"
    rf"|\bupload\b[^.?!\n]{{0,20}}\b(?:your|the|a|an|that|this)\b[^.?!\n]{{0,30}}\b{_THING}\b"
    rf"|\b(?:can|could|would|will) you\b[^.?!\n]{{0,20}}\b(?:share|send|upload|provide|paste|attach)\b[^.?!\n]{{0,60}}\b{_THING}\b"
    rf"|\bplease (?:upload|share|attach|send|provide|paste)\b(?! (?:the|your) (?:trainers?|team|staff|reps?|colleagues?|customers?|members?|manager|boss)\b)[^.?!\n]{{0,60}}\b{_THING}\b", re.I)
_NOT_NEEDED = re.compile(r"\b(?:no need|not need|needn't|never|without|don't have to|do not have to|won't need|will not need|not ask|nothing to (?:upload|share|send)|"
                         r"instead of|rather than|not required|no upload)\b", re.I)
_MADE = re.compile(
    r"\b(?:I|we)(?:'ve| have)? (?:generated|created|simulated|synthesi[sz]ed|made up|fabricated)\b[^.,;\n]{0,40}\b(?:data|table|dataset|rows|records|sample)\b"
    r"|\b(?:data|table|dataset|rows|records|sample)\b[^.,;\n]{0,30}\b(?:was|were|has been|have been) (?:generated|created|simulated|synthesi[sz]ed)\b"
    r"|\b(?:here is|here are|below is|below are)\b[^.,;\n]{0,30}\b(?:generated|synthetic|simulated|made-up)\b[^.,;\n]{0,20}\b(?:data|table|dataset|rows|records)\b", re.I)
_NOT_MADE = re.compile(r"\b(?:no|not|never|nothing|before any|until|once|after you|if you|when you|would|will|can|could)\b", re.I)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


def asks_upload(text: str) -> bool:
    """Whether the text asks the person to send, share or upload their data."""
    return any(_ASKS.search(s) and not _NOT_NEEDED.search(s) for s in _SENTENCE.split(str(text or "")))


def says_made(text: str) -> bool:
    """Whether the text says that data was generated or created (it never was: nothing here makes data)."""
    return any(_MADE.search(s) and not _NOT_MADE.search(s) for s in _SENTENCE.split(str(text or "")))


def need_of(text: str) -> Optional[str]:
    lines = [line for line in str(text or "").splitlines() if line.strip()][:FIRST_LINES]
    for line in lines:
        found = NEED.match(line)
        if found and found.group(1).lower() in NEEDS:
            return found.group(1).lower()
    return None


def without_need_line(text: str) -> str:
    """The answer without its first line when that line names the need (it is for code, not for the person)."""
    lines = str(text or "").splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        return "\n".join(lines[:i] + lines[i + 1:]).strip() if NEED.match(line) else str(text or "").strip()
    return ""


def sections(text: str) -> dict[str, str]:
    """The answer's parts by heading (the headings the prompt names; another heading stays inside the part before it)."""
    text = str(text or "")
    marks = [(m.start(), m.end(), next((h for h in HEADINGS if h.lower() == m.group(1).strip().strip("*:").strip().lower()), None)) for m in HEADING.finditer(text)]
    marks = [m for m in marks if m[2]]
    return {name: text[end:(marks[i + 1][0] if i + 1 < len(marks) else len(text))].strip() for i, (_, end, name) in enumerate(marks)}


def columns(part: str) -> list[dict[str, str]]:
    """The rows of the "Data to record" table: column, what it is, known when (before, at, after), role."""
    rows: list[dict[str, str]] = []
    for line in str(part or "").splitlines():
        cells = [c.strip().strip("`*").strip() for c in line.strip().strip("|").split("|")] if line.strip().startswith("|") else []
        if not cells or set("".join(cells)) <= set("-: ") or cells[0].lower() == "column":
            continue
        if len(cells) != 4:  # a row that is not four cells cannot be read: it is kept, unreadable, so the table check fails on it
            rows.append({"column": cells[0], "what": "", "known": "", "role": "", "unclear": "it does not have four cells"})
            continue
        when = [k for k in ("before", "at", "after") if re.search(rf"\b{k}\b", cells[2].lower())]
        roles = ["not used"] if "not used" in cells[3].lower() else [r for r in ("label", "input", "key", "time") if re.search(rf"\b{r}\b", cells[3].lower())]
        row = {"column": cells[0], "what": cells[1], "known": "after" if "after" in when else (when[0] if when else ""), "role": roles[0] if roles else ""}
        if len(when) != 1 or len(roles) != 1:  # "before or after", "label / input": the strict reading is taken, and the row is said to be unclear
            row["unclear"] = "known when is not one of before, at, after" if len(when) != 1 else "role is not one of key, time, input, label, not used"
        rows.append(row)
    return rows


def steps(part: str) -> list[str]:
    return [line.strip() for line in str(part or "").splitlines() if STEP.match(line)]


def kind_of(part: str) -> Optional[str]:
    found = KIND.findall(str(part or ""))
    return found[-1].strip().strip("`*.") if found else None


def run(text: str, sources: Iterable[str]) -> list[dict[str, Any]]:
    """Every check on one answer: ``{"check", "ok", "detail"}``. The detail of a failed check is written so that it can be
    sent back to the architect as it is."""
    sources = list(sources)
    out: list[dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        out.append({"check": name, "ok": bool(ok), "detail": "" if ok else detail})

    need = need_of(text)
    parts = sections(text)
    check("need", need is not None, f"The first line must be `Need: ` followed by one of {', '.join(NEEDS)}.")
    missing = [h for h in ALWAYS if not parts.get(h)]
    check("parts", not missing, f"These headings are missing or empty: {', '.join(missing)}.")
    sums = economics.verify(text, sources)
    asked = need in WITH_MODEL and not NO_LABEL_KINDS.search(kind_of(parts.get("Model", "")) or "")
    check("sums", not sums["wrong"], "These sums are not right: " + "; ".join(f"`{w['sum']}` ({w['why']})" for w in sums["wrong"]) + ".")
    unsaid = economics.unsaid(text, sources)
    check("numbers", not unsaid, "These numbers were not given by the person and are not a checked sum of their numbers: "
          + ", ".join(dict.fromkeys(unsaid)) + ". Say the thing in words, or ask for it under \"Questions for you\".")
    table = columns(parts.get("Data to record", ""))
    if need in WITH_DATA:
        unclear = [f"{r['column']} ({r['unclear']})" for r in table if r.get("unclear")]
        check("data table", len(table) >= MIN_COLUMNS and all(r["known"] and r["role"] for r in table) and not unclear,
              f"\"Data to record\" needs a table of at least {MIN_COLUMNS} rows with exactly the four columns column, what it is, known when "
              "(one of before, at, after) and role (one of key, time, input, label, not used), every cell filled."
              + (" Unclear rows: " + "; ".join(unclear) + "." if unclear else ""))
        if asked:  # a model needs an outcome to learn and something to learn it from (the review: a table whose label row was lost passed)
            check("an outcome and an input", any(r["role"] == "label" for r in table) and any(r["role"] == "input" for r in table),
                  "\"Data to record\" needs at least one row with role label (the outcome) and one with role input.")
    late = [r["column"] for r in table if r["role"] == "input" and r["known"] == "after"]
    check("inputs known in time", not late, "These columns are marked as inputs but are known only after the decision: "
          + ", ".join(late) + ". An input must be known before or at the decision.")
    early = [r["column"] for r in table if r["role"] == "label" and r["known"] != "after"]
    check("label after the decision", not early, "These columns are marked as the label but as known before or at the decision: "
          + ", ".join(early) + ". An outcome is known after the decision; if it is known before, it is not the outcome.")
    check("plan", len(steps(parts.get("Plan", ""))) >= MIN_STEPS, f"\"Plan\" needs at least {MIN_STEPS} numbered steps, each with what it needs and how to tell it is finished.")
    if need in ("business", "data_design"):
        check("no model unasked", not parts.get("Model"), "The person did not ask for a model: remove the \"Model\" section.")
    if need in WITH_MODEL:
        check("model", bool(parts.get("Model")) and kind_of(parts.get("Model", "")) is not None,
              "\"Model\" is needed, and its last line names the kind of problem as `Kind: ...`.")
    check("no upload asked", not asks_upload(text), "The person has no data to share: do not ask them to upload, share, send or export it.")
    if need == "simulate":
        check("questions before made-up data", bool(parts.get("Questions for you")) and "?" in parts.get("Questions for you", ""),
              "The person asked for made-up data: \"Questions for you\" must ask how it should look.")
    check("no data made", not says_made(text), "No data has been made: do not say that data was generated or created.")
    return out


def failed(checks: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return [c for c in checks if not c.get("ok")]


NUMBER_GONE = "[a number you did not give]"


def strip_unsaid(text: str, sources: Iterable[str]) -> tuple[str, list[str]]:
    """The text without what states a number nobody gave. A sentence of a paragraph goes; a list item goes whole; in a
    table row or a heading only the number goes (a marker stands in its place), so a table never loses the row that
    names its outcome. Returns the text and what was changed. A sum that is right stays, with its result. Repeated until
    nothing more goes: a result is sourced only while its sum is still there."""
    from dclab_rnd import cited

    sources = list(sources)
    removed: list[str] = []
    text = str(text or "")
    for _ in range(6):
        allowed = [*sources, *economics.verify(text, sources)["verified"]]
        kept: list[str] = []
        changed = False
        for line in text.split("\n"):
            bad = cited.unsaid(economics.unmarked(line), allowed)
            if not bad:
                kept.append(line)
                continue
            changed = True
            stripped = line.lstrip()
            if stripped.startswith(("|", "#")):
                for token in dict.fromkeys(bad):
                    line = re.sub(rf"(?<![\w.,]){re.escape(token.rstrip(','))}(?![\w]|[.,]\d)", NUMBER_GONE, line)
                kept.append(line)
                removed.append(stripped.strip())
            elif re.match(r"(?:[-*+]|(?:\*\*)?\d+[.)](?:\*\*)?)\s", stripped):
                removed.append(stripped.strip())  # a list item goes whole: half of one would mislead
            else:
                parts = re.split(r"(?<=[.!?])\s+", line)
                good = [p for p in parts if not cited.unsaid(economics.unmarked(p), allowed)]
                removed += [p.strip() for p in parts if cited.unsaid(economics.unmarked(p), allowed)]
                if good:
                    kept.append(" ".join(good))
        text = "\n".join(kept)
        if not changed:
            break
    return text, removed
