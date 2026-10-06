"""Evidence library: the live evidence index, and answers written only from it.

    GET  /api/evidence        every record of ``evidence/knowledge/rag/records.jsonl`` in the page's shape (id, type,
                              title, text, up to six citations, flat metadata), with counts by type
    POST /api/evidence/ask    {"question": "..."}: search the index, then answer from the records found

The page used to read a snapshot bundled at build time. The index is a generated file (``make knowledge``), so the
route reads it again whenever it changes (cached on the file's modification time).

How an answer is built (CLAUDE.md rule 5: the model advises, code checks):

1. BM25 search over the index (``EvidenceIndex.search``) returns up to ``K`` records. With none, the answer says the
   evidence does not cover the question.
2. With a model configured, it writes a short answer from those records only and cites their IDs in brackets.
3. Code then drops every sentence that cites an ID outside the retrieved records, cites nothing, or states a number
   that none of the retrieved records contains. If nothing survives, the answer falls back to step 4.
4. Without a model (or after a model error), the answer is the list of closest records: no generated text. When the
   best match scores below ``WEAK``, the answer says the matches are weak, because the records share words with the
   question but probably do not answer it.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request

from dclab_rnd import prompts
from ... import cited, retrieval
from ...evidence_index import EvidenceIndex

ROOT = Path(__file__).resolve().parents[3]
INDEX = ROOT / "evidence" / "knowledge" / "rag" / "records.jsonl"
GUIDE = ROOT / "evidence" / "knowledge" / "MODEL_BUILDING_FIELD_GUIDE.html"
K = 5
WEAK = 4.0  # best BM25 score below this: only loose word overlap (on today's index, real matches score about 8 to 10)
MAX_QUESTION = 500
NOT_COVERED = "The evidence does not cover this yet."
SYSTEM = prompts.text("evidence_answer", not_covered=NOT_COVERED)  # dclab_rnd/prompts/evidence_answer.md (A4.3: prompts are versioned files)



def _rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _page_record(r: dict[str, Any]) -> dict[str, Any]:
    """The shape the frontend's chips, drawer and library use (the same transform demo v1's build applied)."""
    meta = {k: v for k, v in (r.get("metadata") or {}).items() if isinstance(v, (str, int, float, bool, list))}
    return {"id": r["record_id"], "type": r["type"], "title": r["title"], "text": r["text"],
            "citations": (r.get("citations") or [])[:6], "meta": meta}


@lru_cache(maxsize=2)
def _load(path: str, _mtime: float) -> tuple[tuple[dict[str, Any], ...], EvidenceIndex]:
    raw = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    return tuple(raw), EvidenceIndex(raw)


def _index(path: Path = INDEX) -> tuple[tuple[dict[str, Any], ...], EvidenceIndex] | None:
    if not path.is_file():
        return None
    return _load(str(path), path.stat().st_mtime)


def library(path: Path = INDEX) -> dict[str, Any]:
    loaded = _index(path)
    raw = loaded[0] if loaded else ()
    records = [_page_record(r) for r in raw]
    updated = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(timespec="seconds") if loaded else None
    return {"records": records, "total": len(records), "counts": dict(Counter(r["type"] for r in records)),
            "datasets": len({r["meta"].get("dataset") for r in records if r["meta"].get("dataset")}),
            "source": _rel(path), "updated": updated, "guide": GUIDE.is_file(), "guide_path": _rel(GUIDE)}


def check(answer: str, hits: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    """Keep the sentences whose citations are retrieved records and whose numbers appear in the records they cite
    (the check is shared with the stage explanations: ``dclab_rnd.cited``)."""
    out = cited.check(answer, {h["record_id"]: f"{h['title']} {h['text']}" for h in hits})
    return out["kept"], len(out["dropped"])


def _brief(h: dict[str, Any]) -> dict[str, Any]:
    text = h["text"]
    return {"id": h["record_id"], "type": h["type"], "title": h["title"], "score": h.get("score"),
            "snippet": text[:260] + ("…" if len(text) > 260 else "")}


def ask(question: str, client=None, path: Path = INDEX) -> dict[str, Any]:
    loaded = _index(path)
    # the product's index searches by the measured method (A5.1: the hybrid only if it is not worse on any question group)
    hits = retrieval.search(question, k=K, index=loaded[1]) if loaded else []
    out: dict[str, Any] = {"question": question, "records": [_brief(h) for h in hits], "sentences": [], "dropped": 0}
    if not hits:
        return out | {"mode": "none", "covered": False, "note": NOT_COVERED}
    if client is not None:
        context = "\n\n".join(f"[{h['record_id']}] ({h['type']}) {h['title']}\n{h['text'][:1500]}" for h in hits)
        try:
            reply = client.complete([{"role": "system", "content": SYSTEM},
                                     {"role": "user", "content": f"Question: {question}\n\nRecords:\n{context}"}], max_tokens=500)
            text = (reply.get("content") or "").strip()
        except RuntimeError as error:  # the model is optional; the records stand alone
            return out | {"mode": "records", "covered": None, "note": f"The model request failed ({error}); these are the closest records."}
        report = getattr(client, "output", None) or (lambda passed, reason="": None)
        if text.startswith(NOT_COVERED):
            report(True)
            return out | {"mode": "model", "covered": False, "note": NOT_COVERED}
        kept, dropped = check(text, hits)
        report(bool(kept), "" if kept else "no sentence kept a citation and numbers found in the records")
        if kept:
            return out | {"mode": "model", "covered": True, "sentences": kept, "dropped": dropped,
                          "note": "Written by the model from these records only; sentences with an uncited claim or a number "
                                  "not in a cited record were removed." if dropped else "Written by the model from these records only."}
        return out | {"mode": "records", "covered": None, "dropped": dropped,
                      "note": "The model's answer did not pass the citation and number check; these are the closest records."}
    if max(h.get("score", 0) for h in hits) < WEAK:  # the strongest keyword match, wherever the fused ranking put it
        return out | {"mode": "records", "covered": False, "weak": True,
                      "note": f"{NOT_COVERED} Only weak matches came back (best score {max(h.get('score', 0) for h in hits):.1f}); they share words with the question but probably do not answer it."}
    return out | {"mode": "records", "covered": None, "note": "No model is configured, so nothing is written: these are the closest records."}


def _offline():
    from ...models import installed

    return installed()


def register(app, ctx) -> None:
    @app.get("/api/evidence")
    async def evidence_library():
        """Every record of the evidence index, in the shape the Evidence library and the record chips use."""
        return await asyncio.to_thread(library)

    @app.post("/api/evidence/ask")
    async def evidence_ask(request: Request):
        """Answer a question from the evidence index only (search, then a checked model answer or the closest records)."""
        body = await request.json()
        question = str((body or {}).get("question", "")).strip() if isinstance(body, dict) else ""
        if not question:
            raise HTTPException(422, "Ask a question")
        if len(question) > MAX_QUESTION:
            raise HTTPException(422, f"Keep the question under {MAX_QUESTION} characters")
        client = (ctx.models or _offline()).client("evidence_answer")
        return await asyncio.to_thread(ask, question, client)
