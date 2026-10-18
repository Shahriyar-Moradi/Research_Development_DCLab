"""Roles: where a model may help, on dclab_core's terms (package 18.1, Part 7 of the playbook).

A :class:`Role` is one place in the solution workflow where a model can propose something: it names the advisor
``purpose`` (and the prompt file of that name in ``dclab_core/prompts``), builds compact ``evidence`` from its inputs
(aggregates and names, never rows), ``parse``-s an answer into a validated output (raising when the answer is not
acceptable), and has a deterministic ``fallback``. :func:`run_role` always returns a usable output and says whether it
came from the advisor or the fallback, and why; nothing a model says is used unchecked, and DCLab works with no model.

A host implements the advisor (``dclab_core.advisor.Advisor``): it sends ``prompts.text(purpose)`` as the system
prompt and the evidence as the user message through its own gateway, and returns the JSON answer. The R&D studio and
the main product therefore run the same roles with the same words.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Generic, Literal, Optional, TypeVar

from dclab_core import prompts
from dclab_core.advisor import Advisor, consult

I = TypeVar("I")
O = TypeVar("O")


@dataclass(frozen=True)
class Role(Generic[I, O]):
    name: str                                         # also the advisor purpose and the prompt file's name
    description: str
    evidence: Callable[[I], Mapping[str, Any]]        # inputs -> compact evidence (checked: aggregates only)
    parse: Callable[[Mapping[str, Any], I], O]        # an answer -> a validated output; raises when unacceptable
    fallback: Callable[[I], O]                        # the deterministic output
    merge: Optional[Callable[[O, O], O]] = None       # how an advisor's output combines with the fallback's

    @property
    def prompt(self) -> str:
        return prompts.text(self.name)


@dataclass(frozen=True)
class RoleResult(Generic[O]):
    output: O
    source: Literal["advisor", "fallback"]
    reason: str


def run_role(role: Role[I, O], advisor: Optional[Advisor], inputs: I) -> RoleResult[O]:
    """Run ``role``: the advisor's validated output (merged with the fallback's when the role says how), or the
    fallback with the reason (no advisor, it abstained, it failed, or its answer was refused)."""
    base = role.fallback(inputs)
    if advisor is None:
        return RoleResult(base, "fallback", "no advisor")
    errors: list[str] = []
    output = consult(advisor, role.name, role.evidence(inputs), parse=lambda answer: role.parse(answer, inputs),
                     on_error=lambda kind, exc: errors.append(kind))
    if output is None:
        return RoleResult(base, "fallback", errors[0] if errors else "the advisor abstained")
    return RoleResult(role.merge(base, output) if role.merge else output, "advisor", "accepted after validation")


class ScriptedAdvisor:
    """An advisor for tests: answers from a script per purpose (a mapping, a list taken in turn, or a function of the
    evidence), records every request, and never reaches a model."""

    def __init__(self, script: Mapping[str, Any]):
        self.script = dict(script)
        self.requests: list[tuple[str, Mapping[str, Any]]] = []

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
        self.requests.append((purpose, evidence))
        answer = self.script.get(purpose)
        if callable(answer):
            return answer(evidence)
        if isinstance(answer, list):
            return answer.pop(0) if answer else None
        return answer


def roles() -> dict[str, Role]:
    """Every role dclab_core defines, by name (the advisor purpose a host routes to a model)."""
    from dclab_core.analyse import DATA_INVESTIGATOR
    from dclab_core.debate import DEBATE_JUDGE, SOLUTION_CHALLENGER
    from dclab_core.features import FEATURE_ENGINEER
    from dclab_core.mapping import DATA_NEED, DECOMPOSER, MAP_REVIEWER
    from dclab_core.solutions import SOLUTION_ROUTER
    from dclab_core.plan import SOLUTION_PLANNER
    from dclab_core.report import REPORTER
    from dclab_core.review import CRITIC
    from dclab_core.understand import BRIEF_CHECKER, PROBLEM_ANALYST

    return {role.name: role for role in (PROBLEM_ANALYST, BRIEF_CHECKER, DATA_INVESTIGATOR, FEATURE_ENGINEER, SOLUTION_PLANNER, CRITIC, REPORTER,
                                         DECOMPOSER, MAP_REVIEWER, DATA_NEED, SOLUTION_ROUTER, SOLUTION_CHALLENGER, DEBATE_JUDGE)}
