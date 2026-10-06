"""A short explanation of a stage result, written by a model from the stage record and checked by code (A3.3).

A stage record holds what deterministic code measured: claims (each with an id such as ``PRJ-ab12cd-final-C1`` and
a statement with its numbers), notes (a title and text, citing rule and precedent records) and the limits of each
claim. A model may explain it in up to 120 words, citing those ids in square brackets. Code keeps a sentence only
when it cites an id it was given and every number in it is in a source it cites (``dclab_rnd.cited``); the rest is
removed, and a sentence that claims production readiness is removed whatever it cites.

The model sees claims, limits and notes (which hold aggregates and column names, never rows), through the gateway
purpose ``stage_notes``. The explanation is stored on the stage record beside the deterministic notes, which stay as
they are, and the pages label it "written by a model from these records". Without a model nothing is written.
"""

from __future__ import annotations

from typing import Any

from .. import cited

WORDS = 120
RECORD_CHARS = 400  # of each rule or precedent record the notes cite
RECORDS = 6  # how many of them (the first the notes cite)
LABEL = "Written by a model from the stage record; sentences with an uncited claim, a number not in a cited source, or a claim of production readiness were removed."
PROMPT = f"""You explain one stage result of a machine-learning project to a data scientist, in plain English, in at most {WORDS} words.
Use only the facts below. Put the id of the fact in square brackets after every sentence that uses it, for example [PRJ-ab12cd-final-C1].
Copy every number exactly as the fact writes it: do not round, convert or compute a number. End with the limit that matters most,
citing the fact that states it. Never say the model is production-ready, safe, proven or fair, and never claim a cause."""


def sources_of(record: dict[str, Any]) -> dict[str, str]:
    """What a sentence may cite: the record's claims (with their limits), its deterministic notes, and the rule and
    precedent records those notes cite. An id and the text behind it; the numbers of a sentence are checked against it."""
    from .. import tools

    eid = record.get("experiment_id") or record.get("stage") or "stage"
    sources: dict[str, str] = {}
    for claim in record.get("claims") or []:
        limits = " ".join(claim.get("limitations") or [])
        sources[claim["claim_id"]] = f"{claim['statement']} {limits}".strip()
    proof: list[str] = []
    for position, note in enumerate(record.get("notes") or [], 1):
        if note.get("source") == "llm":  # a model's text is not a source for a model
            continue
        sources[f"{eid}-N{position}"] = f"{note.get('title', '')}. {note.get('text', '')}".strip()
        proof += [r for r in note.get("proof") or [] if r not in proof]
    index = tools._index()
    for record_id in sorted(proof, key=lambda r: not r.startswith("DCLAB-R"))[:RECORDS]:  # the rules the notes cite first (a stable sort)
        hit = index.get(record_id)
        if hit is not None:
            sources[record_id] = f"{hit['title']}. {hit['text'][:RECORD_CHARS]}"
    return sources


def prompt_for(record: dict[str, Any], sources: dict[str, str]) -> str:
    kinds = {c["claim_id"]: c.get("kind", "fact") for c in record.get("claims") or []}
    facts = "\n".join(f"[{i}] ({kinds.get(i, 'note' if '-N' in i else 'rule')}) {body}" for i, body in sources.items())
    return f"{PROMPT}\n\nStage: {record.get('title')}\nQuestion: {record.get('question')}\n\nFacts:\n{facts}"


def explain(record: dict[str, Any], client: Any) -> dict[str, Any] | None:
    """The checked explanation of one stage record, or None (no model, a failing model, nothing that passed)."""
    if client is None or not record.get("claims"):
        return None
    sources = sources_of(record)
    try:
        reply = client.complete([{"role": "user", "content": prompt_for(record, sources)}], max_tokens=400)
        text = (reply.get("content") or "").strip()
    except Exception:  # noqa: BLE001 — the model is optional; the deterministic notes stand alone
        return None
    out = cited.check(text, sources, WORDS)
    report = getattr(client, "output", None)
    if report:
        report(bool(out["kept"]) and not out["dropped"], "" if not out["dropped"] and out["kept"] else
               "; ".join(sorted(set(out["dropped"])))[:200] or "no sentence was written")
    if not out["kept"]:
        return None
    return {"sentences": out["kept"], "words": out["words"], "dropped": len(out["dropped"]), "label": LABEL, "written_by": "model"}
