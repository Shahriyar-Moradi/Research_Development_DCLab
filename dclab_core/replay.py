"""Replay rounds (package 36.5 of the training-data plan): a model must not forget what it learned before.

A training round's set is its new rows (never in an earlier round) plus a fixed share of every earlier round's new rows,
drawn stratum by stratum (the set a row came from, its kind of help and its language), so a rare kind (a Persian
conversation, a track with few rows) comes back every round. The draw is a hash of the salt, the round and the row's id:
the same round always draws the same rows, and the next round draws another share. ``ReplayRound`` is the round's record:
what it read, how many rows came back from each earlier round in each stratum, and the hash of what it wrote, so a round
can be rebuilt byte for byte. Nothing is deleted: every round keeps its own new rows.
"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Iterable, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

SHARE = 0.3  # the plan's proposal: 30 % of every earlier round comes back
SALT = "dclab-replay-v1"
RULE = ("a row is its set and its id; each earlier row comes back only from the latest round holding it, only while it is "
        "still in the current train split and outside every exam, and never when it is new again (changed since); per earlier "
        "round and stratum (set, track, language, frozen when the row was first kept) the rows with the lowest "
        "sha256(salt:round:row), ceil(share x rows) of them, so a stratum of one row comes back too")


def stratum(folder: str, track: Optional[str], language: Optional[str]) -> str:
    """The key a row is drawn under: the set it came from, its kind of help, its language ("none" when not known)."""
    return f"{folder}|{track or 'none'}|{language or 'none'}"


def pick(rows: Iterable[tuple[str, str]], share: float, salt: str, round_: int) -> set[str]:
    """Of ``(id, stratum)`` rows, the ids that come back in round ``round_``: per stratum, ceil(share x n) of them, those
    with the lowest sha256(salt:round:id). Deterministic: the same inputs give the same ids."""
    if not 0.0 <= share <= 1.0:
        raise ValueError("the share is between 0 and 1")
    by: dict[str, list[str]] = defaultdict(list)
    for id_, key in rows:
        by[key].append(id_)
    out: set[str] = set()
    for key, ids in by.items():
        ranked = sorted(set(ids), key=lambda i: hashlib.sha256(f"{salt}:{round_}:{i}".encode()).hexdigest())
        out.update(ranked[: math.ceil(share * len(ranked))])
    return out


class ReplayDraw(Contract):
    """What came back from one earlier round (named by its number and the hash of its new rows): the rows it could give
    back (only those whose latest version is its own, still allowed in training, and not new again) and the drawn ones,
    per stratum ``[rows, drawn]``; every drawn row is in the train set."""

    round: int = Field(ge=1)
    new_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    available: int = Field(ge=0)
    drawn: int = Field(ge=0)
    by_stratum: dict[str, list[int]] = Field(default_factory=dict)


class ReplayRound(Versioned):
    schema_version: Literal["dclab_core.replay_round.v1"] = _version("replay_round")
    model_type: str = Field(max_length=40)
    round: int = Field(ge=1)
    share: float = Field(ge=0.0, le=1.0)
    salt: str = Field(max_length=80)
    rule: str = Field(max_length=600)
    sets: list[dict[str, str]] = Field(default_factory=list, description="the sets read: folder, version, the manifest's sha256")
    new: int = Field(ge=0, description="rows in no earlier round (or changed since)")
    new_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replayed: list[ReplayDraw] = Field(default_factory=list)
    withheld: int = Field(default=0, ge=0, description="earlier rows no longer allowed in training (consent, curation, an exam): never replayed")
    withheld_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    strata: dict[str, int] = Field(default_factory=dict, description="the round's train rows per stratum")
    train: int = Field(ge=0)
    train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    exam_version: Optional[str] = Field(default=None, max_length=40, description="the frozen exam the round's model will sit (36.4)")
    built_at: str


CONTRACTS["replay_round"] = ReplayRound


__all__ = ["SHARE", "SALT", "RULE", "stratum", "pick", "ReplayDraw", "ReplayRound"]
