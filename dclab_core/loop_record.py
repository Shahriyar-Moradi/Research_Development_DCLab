"""The learning loop's own record (Phase 9, package 31.17).

One :class:`LoopRecord` is one turn of the loop for one attempt: the pack it solved (its id, its table's hash and its
key's hash, never the key), the attempt (who solved it, its arm, where its whole trace is kept, what it read from the
playbook), the grade (the grader's version, the verdict and the codes), the reflection drawn from it (the bullets filed,
the drafts refused and why) and the playbook changes it caused (each delta's place in the change log, so the store's
version after it is known). Every record carries its producer and its consent, and whether it is an exam's: an attempt
on a sealed variant never enters a training set of its family.

:func:`replay` checks that a chain of records holds together from the records alone: a bullet an attempt read existed in
the playbook at the version it read; a bullet a reflection filed was added by a delta that names that pack; an exam's
record is never open to training; a probe's tally joins the count delta it wrote. :func:`row` recounts a scoreboard row (passed, of, the mean decision score) from the
records, so every number on the scoreboard traces back to its attempts.

Pure: records and a playbook's change log in, problems out. Nothing here reads a file or calls a model.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Split = Literal["learning", "check", "sealed"]
Kind = Literal["baseline", "draft", "pilot", "probe"]
HEX = r"^[0-9a-f]{8,64}$"


class PackRef(Contract):
    id: str = Field(pattern=r"^pk-[0-9a-f]{6,40}$")
    family: str = Field(max_length=40)
    split: Split
    table_sha256: Optional[str] = Field(default=None, pattern=HEX)
    key_sha256: Optional[str] = Field(default=None, pattern=HEX, description="the key's hash only: the key never travels")


class Reads(Contract):
    """What an attempt was shown from the playbook, and which of it the attempt named."""

    lineage: str = Field(max_length=40)
    version: int = Field(ge=0, description="the playbook's version shown: the number of deltas applied")
    shown: list[str] = Field(default_factory=list)
    used: list[str] = Field(default_factory=list)
    text_sha256: Optional[str] = Field(default=None, pattern=HEX)


class AttemptRef(Contract):
    id: str = Field(min_length=1, max_length=300, description="the attempt's own id: its kept folder, or its harness run")
    kind: Kind
    producer: str = Field(max_length=120, description="rule:<name>, model:<name>, person")
    arm: Optional[str] = Field(default=None, max_length=40, description="rules or reference (baseline); with or without (probe)")
    tag: Optional[str] = Field(default=None, max_length=80)
    run: Optional[str] = Field(default=None, max_length=80, description="the harness run id")
    trace: list[str] = Field(default_factory=list, description="where the whole trace is kept, relative to the workspace")
    valid: bool = True
    why_invalid: Optional[str] = Field(default=None, max_length=600)
    at: Optional[str] = None
    reads: Optional[Reads] = None
    counted: Optional[str] = Field(default=None, max_length=20, description="the bullet a probe's tally counted this attempt for")


class GradeRef(Contract):
    grader: str = Field(max_length=40)
    verdict: Literal["pass", "fail"]
    codes: list[str] = Field(default_factory=list)
    decision_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    sha256: Optional[str] = Field(default=None, pattern=HEX, description="the kept grade file's hash, when one is kept")


class Reflection(Contract):
    step: Optional[str] = Field(default=None, max_length=60, description="the first wrong step found")
    filed: list[str] = Field(default_factory=list, description="the bullets filed from this attempt")
    refused: list[Any] = Field(default_factory=list)
    why: str = Field(default="", max_length=600)
    producer: str = Field(default="rule:reflect_v1", max_length=120)
    at: Optional[str] = None


class DeltaRef(Contract):
    lineage: str = Field(max_length=40)
    index: int = Field(ge=0, description="its place in the change log: the playbook's version after it is index + 1")
    op: str = Field(max_length=20)
    bullet: Optional[str] = Field(default=None, max_length=20)
    by: str = Field(max_length=120)
    at: Optional[str] = None


class LoopRecord(Versioned):
    schema_version: Literal["dclab_core.loop_record.v1"] = _version("loop_record")
    id: str = Field(min_length=1, max_length=320)
    family: str = Field(max_length=40)
    pack: PackRef
    attempt: AttemptRef
    grade: Optional[GradeRef] = None
    reflection: Optional[Reflection] = None
    deltas: list[DeltaRef] = Field(default_factory=list)
    result: Optional[str] = Field(default=None, max_length=40, description="the stored result (LRN-NNN) its row is counted in")
    producer: str = Field(max_length=120)
    consent: str = Field(max_length=200)
    exam: bool = False
    training_allowed: bool = True

    @model_validator(mode="after")
    def _exam_closed(self) -> "LoopRecord":
        if self.pack.split == "sealed" and not self.exam:
            raise ValueError("an attempt on a sealed variant is an exam's")
        if self.exam and self.training_allowed:
            raise ValueError("an exam's attempt never enters a training set of its family")
        if self.training_allowed and self.pack.split != "learning":
            raise ValueError("only a learning variant's attempt is open to training: check variants measure the lessons")
        if self.pack.family != self.family:
            raise ValueError("the pack's family is the record's")
        return self


def _book(deltas: Sequence[dict[str, Any]], version: int) -> set[str]:
    """The bullets alive after the first ``version`` deltas of a change log (added, not retired)."""
    alive: set[str] = set()
    for d in deltas[:version]:
        if d.get("op") == "add" and d.get("bullet"):
            alive.add(str(d["bullet"]["id"] if isinstance(d["bullet"], dict) else d["bullet"]))
        elif d.get("op") == "retire" and d.get("target"):
            alive.discard(str(d["target"]))
    return alive


def replay(records: Iterable[LoopRecord], logs: dict[tuple[str, str], Sequence[dict[str, Any]]]) -> list[str]:
    """Problems that stop the records from replaying, or an empty list. ``logs`` maps (family, lineage) to its playbook's
    change log, as written (each delta a dict)."""
    problems: list[str] = []
    records = list(records)
    packs = {r.pack.id for r in records}
    for r in records:
        if r.attempt.reads is not None:
            log = logs.get((r.family, r.attempt.reads.lineage))
            if log is None:
                problems.append(f"{r.id}: read the {r.attempt.reads.lineage} playbook, whose change log is missing")
            elif r.attempt.reads.version > len(log):
                problems.append(f"{r.id}: read version {r.attempt.reads.version}, beyond the {len(log)} changes logged")
            else:
                missing = set(r.attempt.reads.shown) - _book(log, r.attempt.reads.version)
                if missing:
                    problems.append(f"{r.id}: shown {sorted(missing)}, not in the playbook at version {r.attempt.reads.version}")
            if set(r.attempt.reads.used) - set(r.attempt.reads.shown):
                problems.append(f"{r.id}: names a bullet it was not shown")
        for bullet in (r.reflection.filed if r.reflection else []):
            added = [d for d in r.deltas if d.op == "add" and d.bullet == bullet]
            if not added:
                problems.append(f"{r.id}: filed {bullet} but no delta of this record adds it")
        for d in r.deltas:
            log = logs.get((r.family, d.lineage))
            if log is None or d.index >= len(log):
                problems.append(f"{r.id}: delta {d.index} of {d.lineage} is not in its change log")
                continue
            written = log[d.index]
            bullet = written.get("bullet") or {}
            if d.op == "add" and bullet.get("id") != d.bullet:
                problems.append(f"{r.id}: delta {d.index} adds {bullet.get('id')}, not {d.bullet}")
            elif d.op == "add" and r.pack.id not in {e.get("pack") for e in bullet.get("evidence") or []}:
                problems.append(f"{r.id}: delta {d.index}'s evidence does not name the pack {r.pack.id}")
        if r.attempt.counted and not any(d.op == "count" and d.bullet == r.attempt.counted for d in r.deltas):
            problems.append(f"{r.id}: counted for {r.attempt.counted}, but no count delta of it joins this attempt")
        if r.exam and r.training_allowed:
            problems.append(f"{r.id}: an exam's attempt open to training")
    for (family, lineage), log in logs.items():
        for i, d in enumerate(log):
            for e in ((d.get("bullet") or {}).get("evidence") or []) if d.get("op") == "add" else []:
                if e.get("pack") and e["pack"] not in packs:
                    problems.append(f"{family}/{lineage} delta {i}: its evidence names the pack {e['pack']}, which no record holds")
    return problems


def row(records: Iterable[LoopRecord], result: str, arm: str = "rules") -> dict[str, Any]:
    """A scoreboard row recounted from the records counted in ``result``: the attempts of ``arm`` whose variant was
    solvable (the reference arm passed), the passes among them and the mean decision score."""
    mine = [r for r in records if r.result == result and r.attempt.valid and r.grade is not None]
    solvable = {r.pack.id for r in mine if r.attempt.arm == "reference" and r.grade.verdict == "pass"}
    counted = [r for r in mine if r.attempt.arm == arm and r.pack.id in solvable]
    scores = [r.grade.decision_score for r in counted if r.grade.decision_score is not None]
    return {"result": result, "arm": arm, "solvable": len(counted), "passed": sum(r.grade.verdict == "pass" for r in counted),
            "decision_score": round(sum(scores) / len(scores), 4) if scores else None,
            "records": sorted(r.id for r in counted)}


def training(records: Iterable[LoopRecord]) -> list[LoopRecord]:
    """The records a training set may take: valid attempts on learning variants, never a check's or an exam's."""
    return [r for r in records if r.training_allowed and not r.exam and r.pack.split == "learning" and r.attempt.valid]


CONTRACTS["loop_record"] = LoopRecord


__all__ = ["PackRef", "Reads", "AttemptRef", "GradeRef", "Reflection", "DeltaRef", "LoopRecord", "replay", "row", "training"]
