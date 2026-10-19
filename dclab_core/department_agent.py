"""Department agents (package 30.22, the owner's pattern 8): an agent that acts on company data, safely.

A pricing agent suggests a 30% discount on 2,000 products; what stops it from being applied on its own? Here:

* **A policy in code** (:class:`Policy`): the actions the agent may take (typed, narrow), hard limits (the largest change of one value,
  a budget for all the changes of a run, an amount above which a person approves), and the level it acts at.
* **Every proposal checked** (:func:`check`) against the policy before anything happens: a proposal past a limit is refused, one above
  the approval amount waits for a person, whatever the level.
* **Levels earned, never assumed**: ``shadow`` (recorded only), ``flag`` (recorded and shown to a person), ``act_with_check`` (applied
  when allowed and small, else flagged), ``act`` (applied when allowed). An agent starts in shadow; a person's judgments of its proposals
  build its track record (:class:`TrackRecord`), and :func:`level_for` moves it up one level after ``promote_after`` right calls in a row
  and down one on a wrong one.
* **Every step logged with how to undo it** (:class:`Step`): the proposal, the verdict, whether it was applied, and the value before; an
  applied step is undone exactly (:func:`undo`).

Pure Python and pydantic; the host owns the data, the person and the log's storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Level = Literal["shadow", "flag", "act_with_check", "act"]
LEVELS: tuple[Level, ...] = ("shadow", "flag", "act_with_check", "act")


class Policy(Versioned):
    """What an agent may do, its hard limits, and the level it acts at (v1)."""

    schema_version: Literal["dclab_core.department_policy.v1"] = _version("department_policy")
    name: str = Field(max_length=120)
    actions: list[str] = Field(min_length=1, max_length=20)          # the typed actions it may propose, nothing else
    field: str = Field(max_length=80)                               # the value its actions change (a price, a limit)
    max_change: float = Field(gt=0, le=1)                            # the largest relative change of one value
    budget: Optional[float] = Field(default=None, ge=0)              # the total impact a run may apply
    approval_above: Optional[float] = Field(default=None, ge=0)      # an impact above it waits for a person, at every level
    level: Level = "shadow"
    promote_after: int = Field(default=50, ge=1)                     # right calls in a row before it moves up one level
    read_only: bool = False                                          # a read-only agent proposes and never applies


class Proposal(Contract):
    action: str = Field(max_length=80)
    target: str = Field(max_length=200)
    before: float
    after: float
    impact: float = 0.0          # what the change costs or moves (the host's unit: revenue per period, units)
    reason: str = Field(default="", max_length=600)


class Verdict(Contract):
    allowed: bool
    needs_approval: bool = False
    why: list[str] = Field(default_factory=list)


class Step(Contract):
    id: str = Field(max_length=40)
    at: str
    proposal: Proposal
    verdict: Verdict
    level: Level
    applied: bool
    flagged: bool
    undone: bool = False
    judged: Optional[bool] = None  # a person's judgment of the proposal (right or wrong), when given


class TrackRecord(Contract):
    right_in_a_row: int = 0   # counted since the current level was reached: a new level earns its own run
    right: int = 0
    wrong: int = 0


CONTRACTS["department_policy"] = Policy


def check(policy: Policy, proposal: Proposal, spent: float = 0.0) -> Verdict:
    """The proposal against the policy: an action it may take, a change within the limit, within the run's budget; above the approval
    amount it waits for a person."""
    why: list[str] = []
    if proposal.action not in policy.actions:
        why.append(f"{proposal.action} is not an action this agent may take ({', '.join(policy.actions)})")
    change = abs(proposal.after - proposal.before) / abs(proposal.before) if proposal.before else (0.0 if proposal.after == proposal.before else float("inf"))
    if change > policy.max_change + 1e-12:
        why.append(f"a change of {change:.1%} is past the limit of {policy.max_change:.1%}")
    if policy.budget is not None and spent + abs(proposal.impact) > policy.budget + 1e-9:
        why.append(f"it would take the run's impact to {spent + abs(proposal.impact):,.2f}, past the budget of {policy.budget:,.2f}")
    allowed = not why
    needs = bool(allowed and policy.approval_above is not None and abs(proposal.impact) > policy.approval_above)
    if needs:
        why.append(f"an impact of {abs(proposal.impact):,.2f} is above {policy.approval_above:,.2f}: a person approves it")
    return Verdict(allowed=allowed, needs_approval=needs, why=why)


class Agent:
    """One agent at work on the host's ``state`` (a mapping target -> value of ``policy.field``): proposals checked, applied at its level,
    every step logged with the value before."""

    def __init__(self, policy: Policy, state: dict[str, float], impact_of: Optional[Callable[[str, float, float], float]] = None):
        """``impact_of(target, before, after)``: the host's own reckoning of a change's impact. With a budget or an approval amount it
        is required: an agent's declared impact is never what the limits read."""
        if impact_of is None and (policy.budget is not None or policy.approval_above is not None):
            raise ValueError("a policy with a budget or an approval amount needs the host's impact_of: a declared impact is never trusted")
        self.policy, self.state, self.steps, self.spent, self.impact_of = policy, state, [], 0.0, impact_of

    def consider(self, proposal: Proposal) -> Step:
        if self.impact_of is not None:
            computed = float(self.impact_of(proposal.target, proposal.before, proposal.after))
            if abs(computed - proposal.impact) > 1e-6 * max(1.0, abs(computed)):
                proposal = proposal.model_copy(update={"impact": computed, "reason": (proposal.reason + f" [declared impact {proposal.impact:g}; "
                                                                                                         f"the host's {computed:g} is used]")[:600]})
        verdict = check(self.policy, proposal, self.spent)
        level = self.policy.level
        acts = not self.policy.read_only and verdict.allowed and not verdict.needs_approval and level in ("act_with_check", "act")
        if acts and level == "act_with_check" and self.policy.approval_above is not None and abs(proposal.impact) > self.policy.approval_above / 2:
            acts = False  # with a check: only the small ones go through on their own
        before = self.state.get(proposal.target)
        if acts and (before is None or abs(before - proposal.before) > 1e-9):
            acts = False  # the value moved since the proposal was made: never applied on a stale reading
            verdict = verdict.model_copy(update={"why": [*verdict.why, "the value changed since the proposal: not applied"]})
        if acts:
            self.state[proposal.target] = proposal.after
            self.spent += abs(proposal.impact)
        step = Step(id=f"s{len(self.steps) + 1:05d}", at=datetime.now(timezone.utc).isoformat(timespec="seconds"), proposal=proposal, verdict=verdict,
                    level=level, applied=acts, flagged=(not acts and level != "shadow" and verdict.allowed))
        self.steps.append(step)
        return step


def undo(agent: Agent, step_id: str) -> Step:
    """An applied step put back exactly: the value before it, the impact returned to the budget; the step marked undone (never deleted)."""
    i, step = next((i, s) for i, s in enumerate(agent.steps) if s.id == step_id)
    if not step.applied or step.undone:
        raise ValueError(f"{step_id} was not applied, or was undone already")
    if abs(agent.state.get(step.proposal.target, float("nan")) - step.proposal.after) > 1e-9:
        raise ValueError(f"{step_id}'s value was changed after it: undo the later steps on {step.proposal.target} first")
    agent.state[step.proposal.target] = step.proposal.before
    agent.spent -= abs(step.proposal.impact)
    agent.steps[i] = step.model_copy(update={"undone": True})
    return agent.steps[i]


def judge(record: TrackRecord, right: bool) -> TrackRecord:
    """A person's judgment of one proposal, added to the track record."""
    return TrackRecord(right_in_a_row=record.right_in_a_row + 1 if right else 0, right=record.right + int(right), wrong=record.wrong + int(not right))


def level_for(policy: Policy, record: TrackRecord, last_right: Optional[bool] = None) -> tuple[Level, TrackRecord]:
    """The level the record earns and the record to keep: one up after ``promote_after`` right calls in a row at this level, one down
    on a wrong call, never past ``act``; on a change of level the run of right calls starts again, so each level is earned on its own."""
    i = LEVELS.index(policy.level)
    if last_right is False:
        level = LEVELS[max(0, i - 1)]
    elif record.right_in_a_row >= policy.promote_after:
        level = LEVELS[min(len(LEVELS) - 1, i + 1)]
    else:
        level = policy.level
    if level != policy.level:
        record = record.model_copy(update={"right_in_a_row": 0})
    return level, record


__all__ = ["Policy", "Proposal", "Verdict", "Step", "TrackRecord", "Agent", "LEVELS", "check", "undo", "judge", "level_for"]
