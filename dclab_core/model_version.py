"""Trained versions and their roll-back (package 36.7 of the training-data plan).

Every version of DCLab's own models is kept with what it was trained on (the replay round of 36.5: its number, the hash
of its train set, its share and the sets it read) and the frozen exam it sits (36.4). Its history only grows: trained,
then accepted or refused by the no-forgetting gate, then retired when a later look shows it forgot. A retirement is one
change: the roles it serves go back to the model before it, its exam record says retired, and its own history says why.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Status = Literal["trained", "accepted", "refused", "retired"]


class VersionEvent(Contract):
    at: str
    status: Status
    by: str = Field(max_length=120)
    reason: str = Field(default="", max_length=600)
    roles: list[str] = Field(default_factory=list, description="the roles switched back by a retirement")
    forgotten: list[str] = Field(default_factory=list, description="the exam sections it fell on, from the exam's decision")


class TrainedVersion(Versioned):
    schema_version: Literal["dclab_core.trained_version.v1"] = _version("trained_version")
    model_type: str = Field(max_length=40)
    version: str = Field(min_length=1, max_length=120, description="the version's own name (the exam's model_version)")
    served_as: str = Field(min_length=1, max_length=120, description="the name the tuned tier serves it under (DCLAB_TIER_TUNED_MODEL)")
    base_model: str = Field(max_length=120)
    round: int = Field(ge=1, description="the replay round (36.5) it was trained on")
    round_train_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    replay_share: float = Field(ge=0.0, le=1.0)
    sets: list[dict[str, str]] = Field(default_factory=list, description="the sets the round read: folder, version, manifest hash")
    exam_version: Optional[str] = Field(default=None, max_length=40)
    exam_hash: Optional[str] = Field(default=None, max_length=64)
    history: list[VersionEvent] = Field(min_length=1)

    @property
    def status(self) -> Status:
        return self.history[-1].status


CONTRACTS["trained_version"] = TrainedVersion


__all__ = ["Status", "VersionEvent", "TrainedVersion"]
