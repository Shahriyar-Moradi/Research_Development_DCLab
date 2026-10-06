"""A model-written fix for each finding of the notebook review (package A3.4).

The copilot's findings are deterministic: the detectors, their severities, their messages and their proof are code and
never depend on a model. This module only adds, beside a finding, a fix in at most two sentences that a model wrote
from that finding and the records it cites, and that code checked:

- the sentences cite records the finding's own proof holds (``dclab_rnd.cited``: a cited source that was not given, a
  number that is not in a cited record, or a claim of production readiness removes the sentence);
- the fix cites at least one of the finding's rules and, when the finding has a pitfall or precedent record, that
  record too ("the rule and the matching pitfall"), or it is dropped.

The model sees each finding's title, message and suggestion and the text of its records. A finding quotes the code
in backticks (a column, a path, an example line): only a plain name such as `duration` or `imblearn.pipeline.Pipeline`
is kept, anything else (a path, a selector string, a line of code) is replaced by `<code>` before the request. It never
sees the notebook's source code or its data. Without a model, or when the
reply cannot be read, no fix is written and the review is exactly what it was.
"""

from __future__ import annotations

import copy
import json
import re
from typing import Any

from .. import cited

MAX_SENTENCES = 2
FIX_WORDS = 70
RECORD_CHARS = 500
PER_REQUEST = 8  # findings per model request: the reply is JSON, and a long one is cut off at the token limit
NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]{0,60}")
PRECEDENT_TYPES = ("pitfall", "leakage_precedent", "finding")  # the measured records beside a rule
LABEL = "Written by a model from the rule and the pitfall record this finding cites; the finding itself is deterministic."
PROMPT = f"""You are the DCLab notebook copilot's writer. For each finding below, write a fix in at most {MAX_SENTENCES} sentences:
what to change and why, using only the finding and its records. Put the id of the record in square brackets after the
sentence that uses it, for example [DCLAB-R04]; cite the rule, and the pitfall or precedent record when the finding lists one.
Copy any number exactly as the record writes it; never compute one. Never say the model is production-ready, safe or proven.
Answer with JSON only: {{"fixes": [{{"finding": 0, "fix": "..."}}]}} with the finding numbers given."""


def sources_of(finding: dict[str, Any]) -> dict[str, str]:
    """What a fix to this finding may cite: its proof records (a rule, and a pitfall or precedent), with their text."""
    return {p["record_id"]: f"{p['title']}. {p['text'][:RECORD_CHARS]}" for p in finding.get("proof") or []}


# Two detectors quote what they find in the code itself: a string that looks like a column name (it may be a comparison
# value, a file name or anything else the code subscripts) and an absolute path. What they quote never leaves.
FIXED = {"absolute_data_path": {"message": "A data file is loaded from a hard-coded absolute path that exists on one machine only, so nobody else can rerun the notebook "
                                           "and the data version is not recorded."}}
WHOLE = {"suspicious_column_name": "<column>"}  # every backticked name in the finding is replaced


def plain(text: str, replace_all: str | None = None) -> str:
    """The text of a finding without what it quotes from the notebook's code: a backticked plain name stays (a library
    class, a column the index knows), the rest (a path, a selector string, a line of code) becomes `<code>`; with
    ``replace_all`` every backticked span becomes that."""
    return re.sub(r"`([^`]*)`", lambda m: f"`{replace_all}`" if replace_all else m.group(0) if NAME.fullmatch(m.group(1)) else "`<code>`", str(text))


def prompt_for(findings: list[dict[str, Any]], start: int = 0) -> str:
    """The request for these findings, numbered from ``start`` (their place in the whole review)."""
    blocks = []
    for n, f in enumerate(findings, start):
        records = "\n".join(f"  [{rid}] {body}" for rid, body in sources_of(f).items())
        every = WHOLE.get(f.get("detector"))
        said = {k: FIXED.get(f.get("detector"), {}).get(k) or plain(f[k], every) for k in ("title", "message", "suggestion")}
        blocks.append(f"Finding {n} ({f['severity']}): {said['title']}\n  What was found: {said['message']}\n  Suggestion: {said['suggestion']}\n  Records:\n{records}")
    return PROMPT + "\n\n" + "\n\n".join(blocks)


def _needs(finding: dict[str, Any]) -> tuple[set[str], set[str]]:
    """The rules, and the pitfall or precedent records, of which a fix must cite one each (the second may be empty)."""
    rules = {r for r in finding.get("rules") or []}
    precedents = {p["record_id"] for p in finding.get("proof") or [] if p.get("type") in PRECEDENT_TYPES}
    return rules, precedents


def _parse(text: str) -> dict[int, str]:
    body = re.sub(r"^```(?:json)?|```$", "", str(text or "").strip(), flags=re.M).strip()
    try:
        value = json.loads(body)
    except json.JSONDecodeError:
        return {}
    out: dict[int, str] = {}
    for item in (value.get("fixes") if isinstance(value, dict) else None) or []:
        if isinstance(item, dict) and type(item.get("finding")) is int and isinstance(item.get("fix"), str):  # not a bool
            out.setdefault(item["finding"], item["fix"])
    return out


def write_fixes(findings: list[dict[str, Any]], client: Any) -> dict[int, dict[str, Any]]:
    """The checked fixes, by finding number. Empty without a model or when the reply is unusable; never raises."""
    if client is None or not findings:
        return {}
    said: dict[int, str] = {}
    for start in range(0, len(findings), PER_REQUEST):  # one request for a handful of findings, more for a long review
        try:
            reply = client.complete([{"role": "user", "content": prompt_for(findings[start:start + PER_REQUEST], start)}], max_tokens=1500)
            said |= {n: t for n, t in _parse(reply.get("content") or "").items() if start <= n < start + PER_REQUEST}
        except Exception:  # noqa: BLE001 — the model is optional; the findings stand alone
            continue
    fixes: dict[int, dict[str, Any]] = {}
    dropped = 0
    for n, finding in enumerate(findings):
        text = said.get(n)
        if not text:
            continue
        out = cited.check(text, sources_of(finding), FIX_WORDS)
        kept = out["kept"][:MAX_SENTENCES]
        cites = list(dict.fromkeys(c for k in kept for c in k["cites"]))
        rules, precedents = _needs(finding)
        if not kept or not (rules & set(cites)) or (precedents and not precedents & set(cites)):
            dropped += 1
            continue
        dropped += len(out["dropped"]) + (len(out["kept"]) - len(kept))
        fixes[n] = {"text": " ".join(k["text"] for k in kept), "cites": cites, "written_by": "model", "label": LABEL}
    report = getattr(client, "output", None)
    if report:
        report(len(fixes) == len(findings) and not dropped, "" if len(fixes) == len(findings) and not dropped else
               f"{len(findings) - len(fixes)} finding(s) without a fix that cites its rule and record; {dropped} sentence(s) removed")
    return fixes


def key_of(finding: dict[str, Any]) -> str:
    """Which finding a fix belongs to, in a way that survives the review being computed again: ``detector:cell:line``
    (the review never reports the same detector twice on one line)."""
    return f"{finding['detector']}:{finding['cell']}:{finding['line']}"


def with_fixes(report: dict[str, Any], client: Any) -> dict[str, Any]:
    """A copy of the review with ``fix`` beside each finding that got one. Every other key is untouched."""
    fixes = write_fixes(report.get("findings") or [], client)
    out = copy.deepcopy(report)
    for n, fix in fixes.items():
        out["findings"][n]["fix"] = fix
    return out
