"""The playbook: lessons a solver reads, kept as small deltas merged by code (H2 of the harness-learning plan; ACE's
curation in plain code, Phase 9's 31.7, with the harness reports' additions).

A :class:`Bullet` is one lesson: a short text, its **scope** (``invariant``: how an agent works, whatever the problem;
``domain:<family>``: what a family of problems needs), the grader codes it answers, the moment of a graded run it was
drawn from, and counters (shown, helpful, harmful) that only code moves. A :class:`Playbook` is never edited: it is
rebuilt from its :class:`Delta` log (add, count, status, retire), so every change has its author, its reason and its
time, and "rewrite the whole text" is not an operation that exists.

ACE's rules, as code: an added bullet whose text matches one already there (by its normalised words) merges into it
instead of adding a second; a bullet whose harmful count exceeds its helpful count by :data:`QUARANTINE` is retired; a
bullet's text never changes (an edit is a retirement and an addition); :func:`render` gives the accepted bullets of a
family and the invariant ones, best first, within a token budget.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Optional

from pydantic import Field, field_validator, model_validator

from .contracts import CONTRACTS, Contract, Versioned, _version

TEXT = 300           # characters of one bullet
BUDGET = 2000        # tokens of rendered playbook a solver reads
QUARANTINE = 3       # harmful minus helpful at which a bullet is retired
STATUSES = ("draft", "probed", "accepted", "retired")
Status = Literal["draft", "probed", "accepted", "retired"]
SCOPE = re.compile(r"^(invariant|domain:[a-z0-9_]{1,40})$")


def tokens(text: str) -> int:
    """A generous token count (about four characters a token), as the gateway's estimate counts."""
    return (len(text) + 3) // 4


def _norm(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def bullet_id(text: str, scope: str) -> str:
    return "B-" + hashlib.sha256(f"{scope}:{_norm(text)}".encode()).hexdigest()[:10]


class Evidence(Contract):
    """The graded moment a lesson was drawn from: the pack, the run, the step or stage, the code, the grader's version."""

    pack: str = Field(max_length=80)
    run: Optional[str] = Field(default=None, max_length=120)
    step: str = Field(max_length=80)
    code: str = Field(max_length=80)
    grader: str = Field(max_length=40)


class Bullet(Contract):
    id: str = Field(pattern=r"^B-[0-9a-f]{10}$")
    text: str = Field(min_length=10, max_length=TEXT)
    scope: str
    codes: list[str] = Field(default_factory=list, max_length=6)
    status: Status = "draft"
    shown: int = Field(default=0, ge=0)
    helpful: int = Field(default=0, ge=0)
    harmful: int = Field(default=0, ge=0)
    evidence: list[Evidence] = Field(min_length=1)
    by: str = Field(max_length=120)            # rule:<drafter>, model:<name>, person

    @field_validator("scope")
    @classmethod
    def _scope(cls, value: str) -> str:
        if not SCOPE.match(value):
            raise ValueError("a scope is invariant or domain:<family>")
        return value

    @model_validator(mode="after")
    def _id_is_its_text(self):
        if self.id != bullet_id(self.text, self.scope):  # an id names one text: a forged id cannot replace another bullet
            raise ValueError("a bullet's id is the hash of its scope and its normalised text")
        return self


class Delta(Versioned):
    """One change to a playbook (v1): who made it, why, when. The playbook is these, applied in order."""

    schema_version: Literal["dclab_core.playbook_delta.v1"] = _version("playbook_delta")
    family: str = Field(max_length=40)
    lineage: str = Field(max_length=40)
    op: Literal["add", "count", "status", "retire"]
    bullet: Optional[Bullet] = None            # add
    target: Optional[str] = Field(default=None, pattern=r"^B-[0-9a-f]{10}$")  # count, status, retire
    counts: dict[str, int] = Field(default_factory=dict)  # count: {"shown": 1, "helpful": 1, "harmful": 0}
    status: Optional[Status] = None            # status
    by: str = Field(max_length=120)            # who made the change (rule:<name>, model:<name>, person, probe, curator)
    reason: str = Field(min_length=3, max_length=600)
    at: str


class Playbook(Versioned):
    """A family's playbook in one lineage (v1), as its deltas built it."""

    schema_version: Literal["dclab_core.playbook.v1"] = _version("playbook")
    family: str = Field(max_length=40)
    lineage: str = Field(max_length=40)
    version: int = Field(default=0, ge=0)      # the number of deltas applied
    bullets: list[Bullet] = Field(default_factory=list)


class Refused(ValueError):
    """A delta the curation rules do not take."""


def apply(book: Playbook, delta: Delta) -> Playbook:
    """The playbook after one delta; the curation rules decide, never the delta's author."""
    if (delta.family, delta.lineage) != (book.family, book.lineage):
        raise Refused(f"a delta of {delta.family}/{delta.lineage} on the playbook of {book.family}/{book.lineage}")
    bullets = {b.id: b for b in book.bullets}
    if delta.op == "add":
        if delta.bullet is None:
            raise Refused("an addition carries its bullet")
        new = delta.bullet
        if new.scope != "invariant" and new.scope != f"domain:{book.family}":
            raise Refused(f"a bullet of {new.scope} in the {book.family} playbook")
        same = next((b for b in bullets.values() if _norm(b.text) == _norm(new.text) and b.scope == new.scope), None)
        if same is not None:  # merged, not added twice: its evidence and codes join the one already there
            evidence = [*same.evidence, *[e for e in new.evidence if e not in same.evidence]][-20:]
            bullets[same.id] = same.model_copy(update={"evidence": evidence, "codes": sorted({*same.codes, *new.codes})[:6]})
        else:
            bullets[new.id] = new.model_copy(update={"status": "draft", "shown": 0, "helpful": 0, "harmful": 0})
    else:
        if delta.target not in bullets:
            raise Refused(f"no bullet {delta.target} in this playbook")
        b = bullets[delta.target]
        if delta.op == "count":
            if set(delta.counts) - {"shown", "helpful", "harmful"} or any(v < 0 for v in delta.counts.values()):
                raise Refused("a count moves shown, helpful or harmful, upwards")
            b = b.model_copy(update={k: getattr(b, k) + int(delta.counts.get(k, 0)) for k in ("shown", "helpful", "harmful")})
            if b.harmful - b.helpful >= QUARANTINE and b.status != "retired":
                b = b.model_copy(update={"status": "retired"})
        elif delta.op == "status":
            if delta.status is None or delta.status == "retired":
                raise Refused("a status change names draft, probed or accepted; a retirement is its own operation")
            if b.status == "retired":
                raise Refused("a retired bullet stays retired: add it again as a new lesson")
            if STATUSES.index(delta.status) != STATUSES.index(b.status) + 1:  # draft -> probed -> accepted: never a skip, never back
                raise Refused(f"a status moves one step forward ({b.status} -> {STATUSES[STATUSES.index(b.status) + 1]}), not to {delta.status}")
            b = b.model_copy(update={"status": delta.status})
        elif delta.op == "retire":
            b = b.model_copy(update={"status": "retired"})
        bullets[b.id] = b
    return book.model_copy(update={"bullets": list(bullets.values()), "version": book.version + 1})


def build(family: str, lineage: str, deltas: list[Delta]) -> tuple[Playbook, list[tuple[Delta, str]]]:
    """The playbook its deltas make, and the deltas refused on the way with why (a refused delta changes nothing)."""
    book = Playbook(family=family, lineage=lineage)
    refused: list[tuple[Delta, str]] = []
    for d in deltas:
        try:
            book = apply(book, d)
        except Refused as exc:
            refused.append((d, str(exc)))
    return book, refused


def render(book: Playbook, budget: int = BUDGET, statuses: tuple[str, ...] = ("accepted",)) -> tuple[str, list[str]]:
    """The text a solver reads and the ids it holds: the invariant and the family's bullets in ``statuses``, best first
    (helpful minus harmful, then most shown), each line with its id, until the token budget is reached."""
    chosen = [b for b in book.bullets if b.status in statuses]
    chosen.sort(key=lambda b: (b.scope != "invariant", -(b.helpful - b.harmful), -b.shown, b.id))
    lines, ids, used = [], [], 0
    for b in chosen:
        line = f"- [{b.id}] {b.text}"
        if used + tokens(line) > budget:
            break
        lines.append(line)
        ids.append(b.id)
        used += tokens(line)
    return "\n".join(lines), ids


def cited(text: str, ids: list[str]) -> list[str]:
    """The shown bullets a reply or a thought names by id."""
    return [i for i in ids if i in (text or "")]


CONTRACTS["playbook_delta"] = Delta
CONTRACTS["playbook"] = Playbook

__all__ = ["TEXT", "BUDGET", "QUARANTINE", "STATUSES", "tokens", "bullet_id", "Evidence", "Bullet", "Delta", "Playbook", "Refused",
           "apply", "build", "render", "cited"]
