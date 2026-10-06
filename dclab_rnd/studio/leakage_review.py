"""The leakage reviewer: which columns are not known at the prediction moment, each with a reason and a record (A3.2).

It reads the prediction moment, the column names with their summaries (kind, missing rate, unique count; never a
value) and the column audit's flags, and proposes columns that are written after the moment. Code decides:

- an item without a reason, or citing a record that is not in the evidence index, is dropped;
- a model's reason must quote the moment (its own words), so a reason cannot float free of the contract;
- a column the audit flagged stays flagged whatever a model says (the reviewer never un-forbids);
- only the audit's flags are ticked by default; every other proposal is shown to the user, never applied.

Without a model, the audit and one rule answer: a clause of the moment that excludes something ("Exclude …
durations, bounce/exit rates", "… are post-outcome and forbidden", "… are not available") names the columns whose
words it contains.
"""

from __future__ import annotations

import json
import re
from typing import Any

from dclab_rnd import prompts

VERDICTS = ("available", "after the moment")
SOURCES = ("audit", "moment", "model")
REASON_CHARS = 300
MOMENT_RECORD = "DCLAB-R01"  # write the prediction moment and judge every column against it
EXCLUDES = re.compile(r"\b(?:exclud\w*|forbidden|post-[a-z]+|not available|unavailable|not known|unknown)", re.I)
CLAUSES = re.compile(r"[.;]\s*|\s+\b(?:but|while|whereas|although)\b\s+", re.I)  # "region is known but the outcome is not": two clauses
STOP = {"the", "and", "are", "not", "for", "with", "from", "that", "this", "those", "these", "which", "only", "each", "every", "known",
        "avail", "forbi", "exclu", "post", "unkno", "unava", "itsel", "its", "own", "same", "use", "using"}

PROMPT = prompts.text("leakage_review")  # dclab_rnd/prompts/leakage_review.md (A4.3: prompts are versioned files)


def _stems(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", str(text)).lower())
    return {w[:5] for w in words if len(w) >= 3} - STOP


def _column_stems(name: str) -> set[str]:
    whole = re.sub(r"[^a-z0-9]", "", str(name).lower())
    return _stems(name) | ({whole[:5]} if len(whole) >= 3 else set())


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()


def quotes_moment(reason: str, moment: str | None) -> bool:
    """True when the reason repeats words of the moment: two consecutive words of it, one of them a content word of
    four letters or more ("before a marketing call" is quoted by "a marketing"; "at the" quotes nothing). With no
    moment written there is nothing to quote, so nothing passes."""
    words = re.findall(r"[a-z0-9][a-z0-9'/-]*", _norm(moment))
    if not words:
        return False
    text = " " + " ".join(re.findall(r"[a-z0-9][a-z0-9'/-]*", _norm(reason))) + " "
    content = lambda w: len(w) >= 4 and w[:5] not in STOP  # noqa: E731
    if len(words) == 1:
        return content(words[0]) and f" {words[0]} " in text
    return any(f" {words[i]} {words[i + 1]} " in text and (content(words[i]) or content(words[i + 1])) for i in range(len(words) - 1))


def known_records() -> set[str]:
    from dclab_rnd import tools
    return {r["record_id"] for r in tools._index().records}


def catalogue() -> list[dict[str, str]]:
    """The records a reviewer may cite: the rules and the leakage precedents (ids and titles only)."""
    from dclab_rnd import tools
    return [{"id": r["record_id"], "title": r.get("title", "")} for r in tools._index().records if r.get("type") in ("rule", "leakage_precedent")]


def columns_of(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """What the reviewer may know about a column: its name and summaries, never a value (no preview, no examples)."""
    return [{"name": c.get("name"), "kind": c.get("kind"), "missing_rate": round(float(c.get("missing_rate") or 0), 3), "unique": c.get("unique")}
            for c in profile.get("columns") or []]


def from_audit(flagged: list[dict[str, Any]], records: set[str]) -> list[dict[str, Any]]:
    """The audit's flags as reviewer items, ticked by default (as the solution sheet always did)."""
    items = []
    for f in flagged:
        cited = [r for r in f.get("proof") or [] if r in records]
        items.append({"column": f["column"], "verdict": "after the moment", "reason": str(f.get("reason") or "flagged by the column audit")[:REASON_CHARS],
                      "records": cited or ["DCLAB-R05"], "source": "audit", "apply": True})
    return items


def from_moment(columns: list[dict[str, Any]], moment: str | None, skip: set[str]) -> list[dict[str, Any]]:
    """The rule: a clause of the moment that excludes something names the columns whose words it contains."""
    items = []
    for clause in CLAUSES.split(str(moment or "")):
        if not EXCLUDES.search(clause):
            continue
        words = _stems(clause)
        for c in columns:
            name = str(c.get("name"))
            if name in skip or any(i["column"] == name for i in items):
                continue
            if _column_stems(name) & words:
                items.append({"column": name, "verdict": "after the moment", "reason": f"The moment says: “{clause.strip()[:240]}”.",
                              "records": [MOMENT_RECORD], "source": "moment", "apply": False})
    return items


def from_model(client: Any, columns: list[dict[str, Any]], moment: str | None, flagged: list[dict[str, Any]], target: str | None,
               records: set[str]) -> tuple[list[dict[str, Any]], int]:
    """A model's review, checked item by item. Returns the items kept and how many were dropped."""
    context = {"prediction_moment": moment or "(not written yet)", "target": target, "columns": columns,
               "audit_flags": [{"column": f["column"], "reason": f.get("reason")} for f in flagged], "records": catalogue()}
    reply = client.complete([{"role": "system", "content": PROMPT}, {"role": "user", "content": json.dumps(context, ensure_ascii=False, default=str)}], None)
    text = re.sub(r"^```(?:json)?|```$", "", str(reply.get("content") or "").strip(), flags=re.M).strip()
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        _verdict(client, False, "the reply was not JSON")
        return [], 0
    names = {c["name"] for c in columns}
    kept, dropped = [], 0
    for item in (value.get("columns") if isinstance(value, dict) else None) or []:
        if not isinstance(item, dict):
            dropped += 1
            continue
        column, verdict, reason = item.get("column"), item.get("verdict"), str(item.get("reason") or "").strip()
        cited = [str(r) for r in item.get("records") or []]
        if (column not in names or column == target or verdict not in VERDICTS or not reason or not cited
                or any(r not in records for r in cited) or not quotes_moment(reason, moment)):
            dropped += 1
            continue
        kept.append({"column": column, "verdict": verdict, "reason": reason[:REASON_CHARS], "records": cited, "source": "model", "apply": False})
    _verdict(client, dropped == 0, "" if dropped == 0 else f"{dropped} item(s) without a reason that quotes the moment or with a missing record")
    return kept, dropped


def _verdict(client: Any, ok: bool, reason: str) -> None:
    if getattr(client, "output", None):
        client.output(ok, reason[:200])


def review(profile: dict[str, Any], moment: str | None, flagged: list[dict[str, Any]], target: str | None = None,
           identifiers: list[str] | None = None, client: Any = None) -> dict[str, Any]:
    """The review: the audit's flags (ticked), and the columns the moment or a model says are written after it
    (shown, never ticked). A model's "available" for a flagged column is reported, and the flag stays."""
    records = known_records()
    columns = columns_of(profile)
    items = from_audit(flagged, records)
    flagged_names = {i["column"] for i in items}
    skip = flagged_names | {target or ""} | set(identifiers or [])
    items += from_moment(columns, moment, skip)
    mode, dropped, disagreements = "rules", 0, []
    if client is not None and str(moment or "").strip():  # without a written moment there is nothing to judge the columns against
        try:
            proposed, dropped = from_model(client, columns, moment, flagged, target, records)
            mode = "model"
        except Exception:  # noqa: BLE001 — no model, no problem: the audit and the moment rule still answer
            proposed = []
        for item in proposed:
            if item["column"] in flagged_names:
                if item["verdict"] == "available" and all(d["column"] != item["column"] for d in disagreements):  # never un-forbid: the flag stays, the disagreement is shown
                    disagreements.append({"column": item["column"], "reason": item["reason"], "records": item["records"]})
                continue
            if item["verdict"] != "after the moment" or item["column"] in skip:
                continue
            same = next((i for i in items if i["column"] == item["column"]), None)
            if same is None:
                items.append(item)
            else:
                same["records"] = sorted(set(same["records"]) | set(item["records"]))
    return {"items": items, "mode": mode, "dropped": dropped, "disagreements": disagreements,
            "note": "Ticked: the column audit's flags. Shown, not applied: what the moment" + (" or the reviewer model" if mode == "model" else "")
                    + " says is written after the prediction moment. You decide what to forbid."}


def measure(samples: list[str] | None = None, client_for: Any = None) -> dict[str, Any]:
    """Run the reviewer on the studied samples against the R&D's own forbidden columns: how many it finds, how many
    it adds that the R&D did not (shown to the user, never applied), per sample and in total."""
    from . import data as studio_data, solution as studio_solution

    rows = []
    for entry in studio_data.sample_catalog():
        if samples and entry["key"] not in samples:
            continue
        frame, suggestion = studio_data.load_sample(entry["key"])
        profile = studio_data.profile_table(frame)
        proposal = studio_solution.propose(frame, profile, suggestion["target"])
        out = review(profile, suggestion.get("prediction_moment"), proposal["forbidden"], suggestion["target"], proposal["identifiers"],
                     client_for(entry["key"]) if client_for else None)
        rd = {f["column"] for f in suggestion.get("forbidden") or []}
        proposed = {i["column"] for i in out["items"] if i["verdict"] == "after the moment"}
        rows.append({"sample": entry["key"], "rd_forbidden": sorted(rd), "found": sorted(rd & proposed), "missed": sorted(rd - proposed),
                     "added": sorted(proposed - rd), "mode": out["mode"]})
    total = sum(len(r["rd_forbidden"]) for r in rows)
    return {"samples": rows, "rd_forbidden": total, "found": sum(len(r["found"]) for r in rows), "added": sum(len(r["added"]) for r in rows)}
