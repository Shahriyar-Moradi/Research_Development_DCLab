"""One memory for DCLab's agents: the item, the order a reader sees, the brief (package 35.1, docs/guides/MEMORY_PLAN.md).

A :class:`MemoryItem` is one line an agent may remember: a **kind** (``fact``: what is true about a person, a workspace
or a project; ``episode``: what happened; ``know_how``: how to work; ``knowledge``: rules and evidence for everyone; the
working memory of a conversation is never stored), a **scope** (``global`` → ``workspace`` → ``person`` → ``project``
→ ``session``, wide to narrow) and its id, the **text** (one line, at most :data:`TEXT` characters, no control
character: a stored line can never pass for an instruction of its own), its **source** (no source, no item), who wrote
it, a **status** (``draft`` → ``active`` → ``retired``), the item it corrects, counters only code moves, and an
optional vector named by the model that made it.

An item's id is the hash of its kind, scope and normalised words, so the same words in the same scope are one item and a
forged id is refused. :func:`resolve` gives a reader the active items of its own scopes, a correction replacing what it
corrects, in a fixed order: corrections first, then the narrower scope, then the newest. :func:`select` turns them into
a :class:`Brief` under a token budget, naming how many it left out. :func:`check` refuses an item that names a protected
column (a benchmark's, a challenge's, a sealed set's), however it is written. Nothing here reads a table's value.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import datetime, timezone
from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from .contracts import CONTRACTS, Contract, Versioned, _version

TEXT = 300                 # characters of one item
BRIEF_TOKENS = 800         # tokens of the brief an agent reads at the start of a session
KINDS = ("fact", "episode", "know_how", "knowledge")
SCOPES = ("global", "workspace", "person", "project", "session")  # wide to narrow
STATUSES = ("draft", "active", "retired")
GLOBAL_KINDS = ("know_how", "knowledge")  # DCLab's own: never a fact about one person or project
Kind = Literal["fact", "episode", "know_how", "knowledge"]
Scope = Literal["global", "workspace", "person", "project", "session"]
Status = Literal["draft", "active", "retired"]
ID = r"^M-[0-9a-f]{12}$"


def tokens(text: str) -> int:
    """A generous token count (about four characters a token), as the gateway's estimate counts."""
    return (len(text) + 3) // 4


JOINERS = ("\u200c", "\u200d")  # Persian writes with the zero-width (non-)joiner: kept, every other invisible mark is refused
MEANING = re.compile(r"[<>=!≠≤≥%+]|(?<!\w)[-−](?=\d)|\w+")  # signs that change a fact's meaning are words too


def _norm(text: str) -> str:
    """The words of a text, in any script, compatibility-folded and lower-cased, with the signs that carry meaning
    (``<``, ``>``, ``=``, ``!``, ``%``, a minus before a number): spacing and punctuation never make two items, a sign does."""
    folded = unicodedata.normalize("NFKC", text).casefold().replace("_", " ")
    return " ".join(MEANING.findall(folded))


def _utc(value: str) -> str:
    """An ISO time with its zone, as UTC (so a later time always sorts later)."""
    moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if moment.tzinfo is None:
        raise ValueError("a time names its zone")
    return moment.astimezone(timezone.utc).isoformat(timespec="seconds")


def item_id(kind: str, scope: str, scope_id: str, text: str) -> str:
    return "M-" + hashlib.sha256(f"{kind}:{scope}:{scope_id}:{_norm(text)}".encode()).hexdigest()[:12]


class Counters(Contract):
    shown: int = Field(default=0, ge=0)
    cited: int = Field(default=0, ge=0)
    harmful: int = Field(default=0, ge=0)


class MemoryItem(Versioned):
    """One remembered line (v1)."""

    schema_version: Literal["dclab_core.memory_item.v1"] = _version("memory_item")
    id: str = Field(pattern=ID)
    kind: Kind
    scope: Scope
    scope_id: str = Field(min_length=1, max_length=120)
    text: str = Field(min_length=1, max_length=TEXT)
    source: str = Field(min_length=1, max_length=200)      # the event, run, note or lesson it came from
    who: str = Field(min_length=1, max_length=120)         # person:<id>, agent:<role>, rule:<name>
    status: Status = "active"
    correction: bool = False                               # written by a person correcting what was remembered
    supersedes: Optional[str] = Field(default=None, pattern=ID)
    retired_by: Optional[str] = Field(default=None, pattern=ID)
    counters: Counters = Field(default_factory=Counters)
    created_at: str = Field(min_length=10, max_length=40)
    last_used_at: Optional[str] = Field(default=None, max_length=40)
    vector: Optional[list[float]] = Field(default=None, max_length=8192)
    vector_model: Optional[str] = Field(default=None, max_length=160)

    @field_validator("text")
    @classmethod
    def _one_line(cls, value: str) -> str:
        """One visible line: no control character, no line or paragraph separator, no invisible format mark (a bidi
        override, a zero-width space) but Persian's joiners, no space at its ends, and at least one word."""
        bad = [c for c in value if unicodedata.category(c) in ("Cc", "Zl", "Zp") or (unicodedata.category(c) == "Cf" and c not in JOINERS)]
        if bad or value != value.strip() or not re.search(r"\w", value):
            raise ValueError("an item is one visible line of words: no line break, separator, tab, control or invisible mark")
        return value

    @field_validator("created_at", "last_used_at")
    @classmethod
    def _times(cls, value: Optional[str]) -> Optional[str]:
        return None if value is None else _utc(value)

    @model_validator(mode="after")
    def _rules(self):
        if self.scope == "global" and self.kind not in GLOBAL_KINDS:
            raise ValueError(f"a global item is {' or '.join(GLOBAL_KINDS)}: a fact or an episode belongs to its workspace, person or project")
        if self.vector is not None and not self.vector_model:
            raise ValueError("a vector names the model that made it")
        if self.id != item_id(self.kind, self.scope, self.scope_id, self.text):
            raise ValueError("an item's id is the hash of its kind, scope and normalised words")
        if self.supersedes == self.id:
            raise ValueError("an item cannot correct itself")
        if self.correction and not self.who.startswith("person:"):
            raise ValueError("a correction is a person's: an agent's change of mind is a new item, never a correction")
        if self.retired_by is not None and (self.status != "retired" or self.retired_by == self.id):
            raise ValueError("retired_by names another item, on a retired item only")
        return self


class Brief(Versioned):
    """What an agent reads at the start of a session (v1): the items' ids in order, their lines, the tokens used."""

    schema_version: Literal["dclab_core.memory_brief.v1"] = _version("memory_brief")
    items: list[str] = Field(default_factory=list)
    text: str = ""
    tokens: int = Field(default=0, ge=0)
    budget: int = Field(default=BRIEF_TOKENS, ge=0)
    left_out: int = Field(default=0, ge=0)


def new(kind: str, scope: str, scope_id: str, text: str, source: str, who: str, at: str, **fields) -> MemoryItem:
    """An item with its id; the contract's rules refuse what it may not hold."""
    return MemoryItem(id=item_id(kind, scope, scope_id, text), kind=kind, scope=scope, scope_id=scope_id, text=text,
                      source=source, who=who, created_at=at, **fields)


def _words(text: str) -> list[str]:
    """A text as lower-case words: compatibility-folded (full-width letters are letters), CamelCase split (a plural of
    capitals, ``IDs``, kept whole), every mark that is not a letter or a digit a space; letters of any script kept."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=[A-Z]{2})s\b", " s", text)
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return re.findall(r"[^\W_]+", split.casefold())


def named(text: str, names: list[str] | tuple[str, ...]) -> list[str]:
    """The names a text carries, however written (``order_id``, ``order id``, ``Order-ID``, ``OrderId``, ``orderid``,
    a plural ``order_ids`` or ``OrderIDs``, full-width letters, a name in any script): the challenge packs' matching, for
    any list of names, with the plurals it missed. Look-alike letters of another script are not caught."""
    words = _words(text)
    padded = f" {' '.join(words)} "
    singular = {w[:-1] if w.endswith("s") else w for w in words} | set(words)
    found = set()
    for name in {n for n in names if n}:
        parts = _words(name)
        if not parts:
            continue
        phrase = " ".join(parts)
        if f" {phrase} " in padded or f" {phrase}s " in padded or f" {phrase} s " in padded or "".join(parts) in singular:
            found.add(name)
    return sorted(found)


def check(item: MemoryItem, protected: list[str] | tuple[str, ...] = ()) -> str:
    """Why an item may not be stored ("" when it may): it names a protected column, which would carry an evaluation's
    answer into a later run."""
    found = named(item.text, protected)
    return f"names {', '.join(found)}: a protected column (a benchmark's, a challenge's or a sealed set's) never enters memory" if found else ""


def resolve(items: list[MemoryItem], chain: list[tuple[str, str]]) -> list[MemoryItem]:
    """The active items a reader with these (scope, id) pairs sees, a correction replacing what it corrects, in order:
    corrections first, then the narrower scope, then the newest, then by id (the input's order never matters)."""
    mine = set(chain)
    gone = {i.id for i in items if i.status == "retired"}  # retired is final: a copy of the same id still active never returns it
    by_id: dict[str, MemoryItem] = {}
    for i in items:
        if i.status == "active" and i.id not in gone and (i.scope, i.scope_id) in mine:
            if i.id not in by_id or _later(i, by_id[i.id]):
                by_id[i.id] = i
    replaced = _replaced(items)
    kept = sorted((i for i in by_id.values() if i.id not in replaced), key=lambda i: i.id)
    kept.sort(key=lambda i: i.created_at, reverse=True)
    kept.sort(key=lambda i: (not i.correction, -SCOPES.index(i.scope)))
    return kept


def _later(a: MemoryItem, b: MemoryItem) -> bool:
    """Of two copies of one id, the one a store wrote last: the newer time, then the higher counters, then its text, so
    the input's order never decides."""
    def key(i: MemoryItem):
        return (i.created_at, i.counters.shown + i.counters.cited + i.counters.harmful, i.model_dump_json())
    return key(a) > key(b)


def _replaced(items: list[MemoryItem]) -> set[str]:
    """The ids an honoured correction hides. An edge from an item to the one it ``supersedes`` counts when its author is
    active, or retired by a later item (replaced, not withdrawn: a chain of corrections keeps the first one hidden); it
    never hides a person's correction unless it is one itself; edges that form a cycle count for none of their items."""
    by_id = {i.id: i for i in items}
    honoured: dict[str, str] = {}
    for i in items:
        if not i.supersedes or not (i.status == "active" or (i.status == "retired" and i.retired_by)):
            continue
        target = by_id.get(i.supersedes)
        if target is not None and target.correction and not i.correction:
            continue
        honoured[i.id] = i.supersedes
    hidden = set()
    for source, target in honoured.items():
        walk, step = target, 0
        while walk in honoured and walk != source and step <= len(honoured):
            walk, step = honoured[walk], step + 1
        if walk != source:  # no way back to the source: not a cycle
            hidden.add(target)
    return hidden


def select(items: list[MemoryItem], budget: int = BRIEF_TOKENS) -> Brief:
    """The brief: the items in their order, one line each with its id, until the next would make the whole text pass
    the budget (the line breaks counted too)."""
    lines: list[str] = []
    ids: list[str] = []
    for i in items:
        line = f"- [{i.id}] {i.text}"
        if tokens("\n".join([*lines, line])) > budget:
            break
        lines.append(line)
        ids.append(i.id)
    text = "\n".join(lines)
    return Brief(items=ids, text=text, tokens=tokens(text), budget=budget, left_out=len(items) - len(ids))


def cited(text: str, ids: list[str]) -> list[str]:
    """The shown items a reply or a thought names by id."""
    return [i for i in ids if i in (text or "")]


def counted(item: MemoryItem, shown: int = 0, cited: int = 0, harmful: int = 0) -> MemoryItem:
    """The item with its counters moved up (only code moves them)."""
    if min(shown, cited, harmful) < 0:
        raise ValueError("counters only move up")
    c = item.counters
    return item.model_copy(update={"counters": Counters(shown=c.shown + shown, cited=c.cited + cited, harmful=c.harmful + harmful)})


def retire(item: MemoryItem, by: str | None = None) -> MemoryItem:
    """The item retired (kept, never deleted), naming the item that replaced it when there is one (a second retirement
    keeps the first one's name)."""
    return MemoryItem.model_validate({**item.model_dump(), "status": "retired", "retired_by": by or item.retired_by})


CONTRACTS["memory_item"] = MemoryItem
CONTRACTS["memory_brief"] = Brief

__all__ = ["TEXT", "BRIEF_TOKENS", "KINDS", "SCOPES", "STATUSES", "GLOBAL_KINDS", "tokens", "item_id", "Counters", "MemoryItem", "Brief",
           "new", "named", "check", "resolve", "select", "cited", "counted", "retire"]
