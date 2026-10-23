"""Challenge packs and their grader (H0 of the harness-learning plan; Phase 9's 31.1 and 31.2).

A **challenge pack** is an invented table with traps planted in it, in two parts that never travel together:

* :class:`ChallengePack`, what the solver may read: an opaque id, the person's request, the outcome column, the
  prediction moment, the task, the table's columns and the SHA-256 of the exact table file;
* :class:`ChallengeKey`, what only the grader reads: the seed and partition the variant was drawn from, the planted
  columns (a value written after the outcome, the entity whose rows repeat, the row's own key, a column that only looks
  suspicious), the honest columns known before the moment, and the knobs.

The pack never carries its seed: whoever could run the generator would rebuild the key from it.

:func:`grade` compares a key with what a solve did (:class:`Observed`: the columns it kept out, the group it split by,
the features its model leaned on, its validation and holdout scores, the baseline's) and returns a :class:`Grade`: a
check a trap, a verdict, and **codes** for whoever learns from it (``SPLIT_WRONG:repeated_entity``). A code names the
kind of mistake, never a column or a value: a lesson written from a code cannot carry a variant's answer to the next
variant (the leak the harness reports found in Phase 9's own example).

Packs fall into three partitions by a salted hash of their family and seed (about 5 : 3 : 2): ``learning`` (material a
learner may read), ``check`` (read by gates), ``sealed`` (read once per promotion). A seed is in exactly one.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal, Optional

from pydantic import Field

from .contracts import CONTRACTS, Contract, Versioned, _version

GRADER = "grader_v1"  # the grader's version: a change to the codes, the checks or a limit moves it
SALT = "dclab-challenges-v1"
PARTITIONS = ("learning", "check", "sealed")
BOUNDS = (("learning", 0.5), ("check", 0.8))  # the rest is sealed
GAP_LIMIT = 0.05  # without the holdout's interval: validation may beat the holdout by at most this share of its score
Z = 1.96  # the z of a 95% interval: turns the holdout's bootstrap interval into its standard error
GUARD_Z = 3.5  # validation is optimistic past this many noise units: fixed on 60 honest reference solves (learning and
# check variants, 2026-10-23), whose largest was 2.71 and 2 of 60 past 1.96 (the engine's intervals run narrow on
# small grouped holdouts); planted gaming traces sit at 5 to 8

Partition = Literal["learning", "check", "sealed"]
Task = Literal["binary", "regression"]

# What each code means. A code names a kind of mistake and the trap it fell into; never a column or a value.
CODES: dict[str, str] = {
    "LEAK_KEPT:after_outcome": "a column written after the outcome was among the model's inputs",
    "LEAK_LEANED:after_outcome": "a column written after the outcome was among the features the model leaned on most",
    "SPLIT_WRONG:repeated_entity": "rows of one entity sat on both sides of the split, so validation saw the entities it was scored on",
    "IDENTIFIER_USED:row_key": "a row's own key was among the model's inputs",
    "FALSE_ALARM:known_before": "a column known before the prediction moment was kept out as if it leaked",
    "GAP_OPTIMISTIC:validation_over_holdout": "the validation score beat the holdout by more than both estimates' noise allows, or there was none to compare",
    "NO_SIGNAL:not_above_baseline": "the holdout score was not shown better than the simplest baseline",
}
TRAPS = ("after_outcome", "repeated_entity", "row_key")  # the planted traps; the rest of the checks are guards
GUARDS = ("decoy", "known_before", "honest_estimate", "beats_baseline")


def partition_of(family: str, seed: int) -> Partition:
    """The partition a family's seed belongs to: the same answer every time, one partition a seed."""
    h = int(hashlib.sha256(f"{SALT}:{family}:{int(seed)}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return next((name for name, bound in BOUNDS if h < bound), "sealed")  # type: ignore[return-value]


class Brief(Contract):
    """What the person brings: the request in their words, the outcome, the moment a prediction is made, the task."""

    request: str = Field(max_length=600)
    target: str = Field(max_length=120)
    moment: str = Field(max_length=400)
    task: Task


class ChallengePack(Versioned):
    """The part of a challenge the solver may read (v1)."""

    schema_version: Literal["dclab_core.challenge_pack.v1"] = _version("challenge_pack")
    id: str = Field(max_length=80)            # opaque: never the seed, never the partition
    family: str = Field(max_length=60)
    brief: Brief
    columns: list[str] = Field(min_length=2)
    rows: int = Field(ge=1)
    table_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")  # of the table file's exact bytes (table.csv)


class ChallengeKey(Versioned):
    """The part only the grader reads (v1): the planted columns and the knobs the variant was drawn with. The knobs say
    how hard the variant is (``key_name: plain`` is a group key a name-based rule cannot spot), never a column."""

    schema_version: Literal["dclab_core.challenge_key.v1"] = _version("challenge_key")
    pack: str = Field(max_length=80)
    family: str = Field(max_length=60)
    seed: int = Field(ge=0)
    partition: Partition
    leaks: list[str] = Field(default_factory=list)          # written after the outcome
    group_column: Optional[str] = None                      # the entity whose rows repeat
    row_key: Optional[str] = None                           # each row's own key
    decoys: list[str] = Field(default_factory=list)         # known before the moment, named like a leak
    known_before: list[str] = Field(default_factory=list)   # the honest columns: known before the moment, none named like a leak
    time_column: Optional[str] = None
    duplicate_rows: int = Field(default=0, ge=0)            # exact copies of an entity's rows (the group split holds them)
    knobs: dict[str, Any] = Field(default_factory=dict)


class Observed(Contract):
    """What a solve did, as the grader reads it (filled from a run by the host: in this studio,
    ``dclab_rnd.learning.grading.observe``)."""

    kept_out: list[str] = Field(default_factory=list)       # columns that did not reach the model (blocked or identifiers)
    group_column: Optional[str] = None
    split: Optional[str] = None                             # group, time_group, time, stratified, ...
    leaned_on: list[str] = Field(default_factory=list)      # the final model's top features as the engine names them
    metric: Optional[str] = None
    validation: Optional[float] = None
    holdout: Optional[float] = None
    holdout_interval: Optional[list[float]] = Field(default=None, min_length=2, max_length=2)  # [low, high], the bootstrap's
    validation_std: Optional[float] = Field(default=None, ge=0)  # the spread of the validation score across folds
    folds: int = Field(default=5, ge=1)
    gap: Optional[float] = None                             # holdout's gain over validation, oriented (negative: validation won)
    baseline: Optional[float] = None                        # the best simple baseline's holdout score
    higher_is_better: bool = True


class GradeCheck(Contract):
    trap: str = Field(max_length=40)
    ok: bool
    code: Optional[str] = None                              # the mistake, when not ok (a key of CODES)


class Grade(Versioned):
    """A solve graded against its key (v1): a check a trap and a guard, the codes of what went wrong, the verdict."""

    schema_version: Literal["dclab_core.grade.v1"] = _version("grade")
    pack: str = Field(max_length=80)
    grader: str = GRADER
    checks: list[GradeCheck] = Field(default_factory=list)
    codes: list[str] = Field(default_factory=list)
    decision_score: float = Field(ge=0, le=1)               # the share of checks passed
    traps_caught: int = Field(ge=0)
    traps: int = Field(ge=0)
    metric: Optional[str] = None
    validation: Optional[float] = None
    holdout: Optional[float] = None
    verdict: Literal["pass", "fail"]


def _leaned(column: str, leaned_on: list[str]) -> bool:
    """A feature the model leaned on is the column itself or one of its encoded levels (``column=value``)."""
    return any(f == column or f.startswith(f"{column}=") or f.startswith(f"{column}_") for f in leaned_on)


def grade(key: ChallengeKey, observed: Observed) -> Grade:
    """The deterministic grade of one solve. Every planted trap and every guard is a check; a check that fails carries
    its code. The verdict passes only when every check passes."""
    kept = set(observed.kept_out)
    checks: list[GradeCheck] = []
    for leak in key.leaks:
        if leak in kept:
            checks.append(GradeCheck(trap="after_outcome", ok=True))
        else:
            code = "LEAK_LEANED:after_outcome" if _leaned(leak, observed.leaned_on) else "LEAK_KEPT:after_outcome"
            checks.append(GradeCheck(trap="after_outcome", ok=False, code=code))
    if key.group_column:
        grouped = observed.group_column == key.group_column and (observed.split or "").startswith(("group", "time_group"))
        checks.append(GradeCheck(trap="repeated_entity", ok=grouped, code=None if grouped else "SPLIT_WRONG:repeated_entity"))
    if key.row_key:
        ok = key.row_key in kept
        checks.append(GradeCheck(trap="row_key", ok=ok, code=None if ok else "IDENTIFIER_USED:row_key"))
    for decoy in key.decoys:
        ok = decoy not in kept
        checks.append(GradeCheck(trap="decoy", ok=ok, code=None if ok else "FALSE_ALARM:known_before"))
    if key.known_before:  # keeping every honest column out avoids every leak and solves nothing
        ok = not (set(key.known_before) & kept)
        checks.append(GradeCheck(trap="known_before", ok=ok, code=None if ok else "FALSE_ALARM:known_before"))
    # the guards hold only on numbers: a solve with no validation score, no holdout or no baseline to beat fails them
    ok = False
    if observed.validation is not None and observed.holdout is not None and observed.holdout_interval:
        # two noisy numbers: validation beats the holdout honestly by up to what both estimates' noise allows together
        low, high = observed.holdout_interval
        se_holdout = (high - low) / (2 * Z)
        se_validation = (observed.validation_std or 0.0) / observed.folds ** 0.5
        excess = observed.validation - observed.holdout if observed.higher_is_better else observed.holdout - observed.validation
        ok = excess <= GUARD_Z * (se_holdout ** 2 + se_validation ** 2) ** 0.5
    elif observed.validation is not None and observed.gap is not None:
        ok = observed.gap >= -GAP_LIMIT * max(abs(observed.validation), 1e-9)
    checks.append(GradeCheck(trap="honest_estimate", ok=ok, code=None if ok else "GAP_OPTIMISTIC:validation_over_holdout"))
    ok = False
    if observed.holdout is not None and observed.baseline is not None:
        ok = (observed.holdout - observed.baseline if observed.higher_is_better else observed.baseline - observed.holdout) > 0
    checks.append(GradeCheck(trap="beats_baseline", ok=ok, code=None if ok else "NO_SIGNAL:not_above_baseline"))
    planted = [c for c in checks if c.trap in TRAPS]
    codes = list(dict.fromkeys(c.code for c in checks if c.code))
    return Grade(pack=key.pack, checks=checks, codes=codes,
                 decision_score=round(sum(c.ok for c in checks) / len(checks), 4) if checks else 0.0,
                 traps_caught=sum(c.ok for c in planted), traps=len(planted), metric=observed.metric,
                 validation=observed.validation, holdout=observed.holdout,
                 verdict="pass" if checks and all(c.ok for c in checks) else "fail")


def _words(text: str) -> list[str]:
    """A text as lower-case words: CamelCase split, every mark that is not a letter or a digit a space."""
    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", text)
    return re.sub(r"[^a-z0-9]+", " ", split.lower()).split()


def names_in(text: str, key: ChallengeKey, columns: list[str]) -> list[str]:
    """The names a text carries (a code, a lesson), however they are written: ``refund_settled``, ``refund settled``,
    ``refund-settled``, ``RefundSettled``, ``refundsettled``. None may reach a learner."""
    words = _words(text)
    padded, joined_tokens = f" {' '.join(words)} ", set(words)
    names = {c for c in [*columns, *key.leaks, *key.decoys, *key.known_before, key.group_column or "", key.row_key or ""] if c}
    found = set()
    for name in names:
        parts = _words(name)
        if parts and (f" {' '.join(parts)} " in padded or "".join(parts) in joined_tokens):
            found.add(name)
    return sorted(found)


CONTRACTS["challenge_pack"] = ChallengePack
CONTRACTS["challenge_key"] = ChallengeKey
CONTRACTS["grade"] = Grade

__all__ = ["GRADER", "SALT", "PARTITIONS", "GAP_LIMIT", "Z", "GUARD_Z", "CODES", "TRAPS", "GUARDS", "partition_of", "Brief",
           "ChallengePack", "ChallengeKey", "Observed", "GradeCheck", "Grade", "grade", "names_in"]
