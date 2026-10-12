"""Challenge mode, the reading part (package 16.13): a brief into its parts, its requirements and its rubric.

A hiring challenge or a client brief says what to build, in parts ("Part 1", "Task 2"), with deliverables (a report,
an API with Flask and Gunicorn, Docker, source control, tests) and often a rubric with points. :func:`read_brief` turns
the text into plain data a host can work through and check off:

* ``parts``: each part's title and text, and the kind of problem it is (from :func:`dclab_core.understand.understand`
  and the words of the part: a rare event, a price or a number, classes);
* ``requirements``: every rubric line or deliverable, with its points when given, its kind, and the part it belongs to;
* ``private``: whether the brief forbids sharing the data ("must not be disclosed", "confidential"), with its words.
  A host treats private data as never leaving the workspace: not in an export, a prompt, or training data.

Pure: text in, data out. Nothing here reads a file or calls a model.
"""

from __future__ import annotations

import re
from typing import Any

PRIVATE = re.compile(r"\b(must not be (disclosed|distributed|shared|published)|do not (disclose|distribute|share|publish)|"
                     r"confidential|proprietary|not for (distribution|publication)|under (an )?nda)\b", re.I)
PART = re.compile(r"^\s*(?:#+\s*)?((?:part|task|question|section|challenge|problem)\s*(?:\d+|[ivx]+|[a-e])\b[^\n]*)$", re.I | re.M)
# a section that applies to the whole challenge, not to the part before it
GLOBAL = re.compile(r"^\s*(?:#+\s*)?((?:deliverables?|requirements?|submission|evaluation|rubric|grading|scoring|what to (submit|deliver))\b[^\n]*)$",
                    re.I | re.M)
POINTS = re.compile(r"\(\s*(\d+(?:\.\d+)?)\s*(?:pts?|points?|marks?)\.?\s*\)", re.I)
KINDS = [  # the deliverable a requirement asks for, by its words; the first that matches
    ("decision", r"\b(pric\w*|fee|threshold|budget|cost of|how much to charge|which offer|policy)\b"),
    ("report", r"\b(report\w*|document\w*|write[- ]?up|explain\w*|presentation|slides?)\b"),  # before tests: "report ... on a test set"
    ("api", r"\b(api|flask|fastapi|gunicorn|endpoints?|rest api|web service|scoring service|serving)\b"),
    ("docker", r"\b(docker\w*|container\w*)\b"),
    ("source_control", r"\b(source control|git(hub|lab)?|version control|repository)\b"),
    ("tests", r"\b(unit tests?|test suite|automated tests?|pytest|tests? (for|of) (the|your) (code|api|service))\b"),
    ("features", r"\b(feature (extraction|engineering)|features?)\b"),
    ("model", r"\b(model\w*|classif\w*|regress\w*|predict\w*|anomal\w*|detect\w*|algorithm\w*|cluster\w*)\b"),
    ("analysis", r"\b(analy[sz]\w*|eda|explor\w*|insight\w*|visuali[sz]\w*)\b"),
]
RARE_WORDS = re.compile(r"\b(fraud\w*|anomal\w*|rare|outlier\w*|abuse|chargeback\w*|intrusion)\b", re.I)
NUMBER_WORDS = re.compile(r"\b(price|pricing|fare|cost|how much|how many|amount|revenue|demand|duration|minutes|estimate)\b", re.I)


def _kind(text: str) -> str:
    return next((kind for kind, pattern in KINDS if re.search(pattern, text, re.I)), "other")


def _lines(text: str) -> list[str]:
    """Requirement-like lines: bullets, numbered items, or lines carrying points."""
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if re.match(r"^([-*•·]|\d+[.)]|[a-z][.)])\s+", line, re.I) or POINTS.search(line):
            out.append(re.sub(r"^([-*•·]|\d+[.)]|[a-z][.)])\s+", "", line, flags=re.I).strip())
    return out


def _problem_kind(text: str) -> dict[str, Any]:
    from dclab_core.understand import understand

    brief = understand(text)
    task = brief.task.value if brief.task.known else None
    rare = bool(RARE_WORDS.search(text))
    if task is None:
        task = "binary" if rare else ("regression" if NUMBER_WORDS.search(text) else None)
    return {"task": task, "rare_event": rare, "asks_for_a_number": bool(NUMBER_WORDS.search(text)),
            "goal": brief.goal.value if brief.goal.known else None}


def read_brief(text: str) -> dict[str, Any]:
    """The brief as parts, requirements (with points) and the confidentiality it asks for."""
    text = text or ""
    marks = sorted([(m, "part") for m in PART.finditer(text)] + [(m, "global") for m in GLOBAL.finditer(text)], key=lambda x: x[0].start())
    parts, shared = [], []
    for i, (m, what) in enumerate(marks):
        end = marks[i + 1][0].start() if i + 1 < len(marks) else len(text)
        body = text[m.end():end].strip()
        if what == "global":
            shared.append(body)
            continue
        parts.append({"id": f"part{len(parts) + 1}", "title": m.group(1).strip()[:200], "text": body[:6000],
                      **_problem_kind(m.group(1) + ". " + body)})
    heads = [m for m, _ in marks]
    if not parts:
        parts = [{"id": "part1", "title": "The challenge", "text": text[:6000], **_problem_kind(text)}]
    requirements = []
    for part in parts:
        for line in _lines(part["text"]):
            points = POINTS.search(line)
            requirements.append({"id": f"R{len(requirements) + 1}", "part": part["id"], "text": POINTS.sub("", line).strip()[:400],
                                 "points": float(points.group(1)) if points else None, "kind": _kind(line)})
    preamble = text[:heads[0].start()] if heads else ""
    for line in _lines("\n".join([preamble, *shared])):  # stated before the parts, or in a shared section: they apply to all
        points = POINTS.search(line)
        requirements.append({"id": f"R{len(requirements) + 1}", "part": None, "text": POINTS.sub("", line).strip()[:400],
                             "points": float(points.group(1)) if points else None, "kind": _kind(line)})
    private = PRIVATE.search(text)
    return {"parts": parts, "requirements": requirements, "private": bool(private),
            "private_words": private.group(0) if private else None,
            "total_points": sum(r["points"] or 0 for r in requirements) or None}
