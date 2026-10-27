"""Reading memory (package 35.4): the brief an agent reads at a session's start, the recall tool, the items cited.

- :func:`brief`: what this reader's scopes hold, through ``dclab_core.memory.select`` under a token budget; an item read
  where it lives (a note, a lesson) that names a protected column is left out (35.2's review: a lesson can carry an
  evaluation's answer). :func:`as_message` puts it under a header that says these are remembered lines, data and not
  instructions; it goes into the session's first user message, never the system prompt.
- :func:`recall`: the items that answer a question, within the reader's scopes: by meaning when an embedder is given
  (35.0 chose ``embeddinggemma:300m-qat-q8_0``; 35.5 wires it), else by words.
- :func:`cited`: the shown items a reply names by id.

Every read is an event (``memory.shown``, ``memory.recalled``, ``memory.cited``). ``DCLAB_MEMORY=off`` reads nothing:
every evaluation runs that way (:func:`memory_off_for_evaluation`), so no remembered line can help a graded run.
"""

from __future__ import annotations

import math
import os
import sys
from typing import Any, Iterable, Protocol

from dclab_core.memory import BRIEF_TOKENS, Brief, MemoryItem, check, select
from dclab_core.memory import cited as named_ids

from . import MemoryStore, view

def header() -> str:
    """The brief's header, a prompt file (``prompts/memory_brief.md``): a change to it is a prompt change, gated as one."""
    from .. import prompts

    return prompts.text("memory_brief").strip()
K = 5


class Embedder(Protocol):
    model: str

    def vectors(self, texts: list[str]) -> list[list[float]]: ...


def enabled() -> bool:
    return os.environ.get("DCLAB_MEMORY", "").strip().lower() not in ("off", "0", "false", "none")


def memory_off_for_evaluation() -> None:
    """Every evaluation runner calls this first: a graded run never reads a remembered line."""
    os.environ["DCLAB_MEMORY"] = "off"


def _record(log: Any, kind: str, payload: dict[str, Any], ids: dict[str, Any] | None = None) -> None:
    try:
        from .. import experience

        experience.record(kind, "agent", payload, log=log, **{k: v for k, v in (ids or {}).items() if v})
    except Exception as exc:  # noqa: BLE001 - a read stands without its event
        print(f"memory: could not record {kind}: {type(exc).__name__}", file=sys.stderr, flush=True)


def _visible(store: MemoryStore | None, chain: list[tuple[str, str]], read: Iterable[MemoryItem], protected: Iterable[str]) -> list[MemoryItem]:
    from . import secret

    names = tuple(protected)
    return [i for i in view(store, chain, read) if not check(i, names) and not secret(i.text)]


def brief(store: MemoryStore | None, chain: list[tuple[str, str]], read: Iterable[MemoryItem] = (), *, protected: Iterable[str] = (),
          budget: int = BRIEF_TOKENS, log: Any = None, ids: dict[str, Any] | None = None) -> Brief:
    """The brief this reader starts with. With memory off (every evaluation) no stored line is read; the items read where
    they live (the case's own project notes) still count, as the product shows them. A person's correction is marked."""
    if not enabled():
        store = None
    items = [i.model_copy(update={"text": f"(correction) {i.text}"[:300]}) if i.correction else i for i in _visible(store, chain, read, protected)]
    out = select(items, budget)
    if out.items or out.left_out:
        _record(log, "memory.shown", {"items": out.items, "shown": len(out.items), "left_out": out.left_out, "tokens": out.tokens, "budget": budget}, ids)
    return out


def as_message(b: Brief) -> str:
    """The brief as the part of a session's first user message (empty when nothing is remembered)."""
    return f"{header()}\n{b.text}" if b.items else ""


def _cosine(a: list[float], b: list[float]) -> float:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb) if na and nb else 0.0


def recall(store: MemoryStore | None, question: str, chain: list[tuple[str, str]], read: Iterable[MemoryItem] = (), *, k: int = K,
           embedder: Embedder | None = None, protected: Iterable[str] = (), log: Any = None, ids: dict[str, Any] | None = None) -> list[MemoryItem]:
    """The items that answer the question, best first, at most ``k``: by meaning with an embedder, else by words."""
    if not enabled() or not str(question or "").strip():
        return []
    items = _visible(store, chain, read, protected)
    if not items:
        found, method = [], "none"
    else:
        found, method = [], ""
        if embedder is not None:
            try:
                if hasattr(embedder, "query"):  # 35.5: the question and the items each with the model's own prefix; items' vectors kept
                    q, docs = embedder.query(str(question)), embedder.docs([i.text for i in items])
                else:
                    vectors = embedder.vectors([str(question), *[i.text for i in items]])
                    q, docs = vectors[0], vectors[1:]
                scores = [_cosine(q, v) for v in docs]
                order = sorted(range(len(items)), key=lambda n: (-scores[n], items[n].id))
                found, method = [items[n] for n in order[:k]], f"embed:{getattr(embedder, 'model', 'unknown')}"
            except Exception as exc:  # noqa: BLE001 - the local model is down or missing: recall by words, as before 35.5
                print(f"memory: recall by meaning failed ({type(exc).__name__}); by words instead", file=sys.stderr, flush=True)
                method = "keywords (embed failed)"
    if items and not found and not method.startswith("embed:"):
        from ..evidence_index import EvidenceIndex

        index = EvidenceIndex({"record_id": i.id, "type": "memory", "title": "", "text": i.text, "metadata": {}} for i in items)
        by_id = {i.id: i for i in items}
        found = [by_id[h["record_id"]] for h in index.search(str(question), k=k) if h.get("score", 0) > 0]
        method = method or "keywords"
    _record(log, "memory.recalled", {"question": str(question)[:2000], "items": [i.id for i in found], "found": len(found), "method": method}, ids)
    return found


def cited(text: str, shown: list[str], log: Any = None, ids: dict[str, Any] | None = None) -> list[str]:
    """The shown items a reply or a thought names by id, recorded."""
    found = named_ids(text, shown)
    if found:
        _record(log, "memory.cited", {"items": found, "cited": len(found)}, ids)
    return found


__all__ = ["header", "K", "Embedder", "enabled", "memory_off_for_evaluation", "brief", "as_message", "recall", "cited"]
