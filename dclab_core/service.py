"""The service's knowledge (the storing phase, Parts G to I): how models are built for a business, whether each was built
correctly, what it takes, and how the service was given.

* :class:`ModelBuild` (32.24): one model built anywhere (a project's model for a business, a student of DCLab's own), its
  business context in bands, its data's shape, its recipe with each choice's reason, what it cost (people, labels, compute,
  euros), its result against its bar, what went wrong, what a person changed, and whether it was then used.
* :class:`BuildVerdict` and :func:`verdict` (32.26): the build judged against DCLab's rules by code, each check passed,
  failed or unknown with why; a check that cannot be read is never assumed passed.
* :class:`LearningCurve` (32.25): a point's or a kind's score at each count of labels, from its refits and its experiments.
* :class:`KnowledgeItem` (32.27): a practice, a rule or a pitfall with its evidence, its scope and its status.
* :class:`ServiceEngagement` (32.30) and :class:`PlaybookItem` (32.31): a piece of work done for someone, and what the
  engagements teach about serving well.

Pure pydantic; the host fills them from its records. Never a row, a name or a business's figures in a cross-business view.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

CheckStatus = Literal["passed", "failed", "unknown"]


class ModelBuild(Versioned):
    """One model built (v1)."""

    schema_version: Literal["dclab_core.model_build.v1"] = _version("model_build")
    id: str = Field(max_length=200)
    model_type: str = Field(max_length=60)          # business_model, or one of the store's eight (m1_students ...)
    workspace: str = Field(max_length=120)
    context: dict[str, Any] = Field(default_factory=dict)   # industry, size band, problem kind: bands only
    data: dict[str, Any] = Field(default_factory=dict)      # the table's shape: rows band, columns, label situation, problems found
    recipe: list[dict[str, Any]] = Field(default_factory=list)  # each choice with its reason
    spent: dict[str, Any] = Field(default_factory=dict)     # people's minutes, questions asked, labels asked, compute seconds, euros
    result: dict[str, Any] = Field(default_factory=dict)    # the metric, its value and interval, the bar, met or not
    went_wrong: list[str] = Field(default_factory=list)
    changed_by_person: list[str] = Field(default_factory=list)
    used: Optional[bool] = None                              # an outcome observed after it was built (None: not known)
    unknown: dict[str, str] = Field(default_factory=dict)    # a part not filled, and why
    at: str


class Check(Contract):
    name: str = Field(max_length=120)
    status: CheckStatus
    why: str = Field(default="", max_length=600)


class BuildVerdict(Versioned):
    """A build judged against DCLab's rules (v1): correct only when every check passed; incorrect when any failed (that check
    named); unknown otherwise."""

    schema_version: Literal["dclab_core.build_verdict.v1"] = _version("build_verdict")
    build: str = Field(max_length=200)
    verdict: Literal["correct", "incorrect", "unknown"]
    checks: list[Check] = Field(default_factory=list)
    rules: str = Field(default="DCLab rules: CLAUDE.md 1 to 6, the gates G1 to G4", max_length=200)


CHECKS = ("contract_before_scores", "no_leaking_column", "fitted_inside_folds", "holdout_read_once", "beats_baselines_with_interval",
          "export_reproduces", "uncertainty_said", "bar_met")


def verdict(build: ModelBuild, facts: dict[str, Any]) -> BuildVerdict:
    """The build judged on its own facts (each a bool, or None when its record cannot say):

    ``contract_before_scores`` the prediction contract written before any score; ``no_leaking_column`` no column unknown at
    the prediction moment used; ``fitted_inside_folds`` every fitting step inside the training folds; ``holdout_read_once``;
    ``beats_baselines_with_interval``; ``export_reproduces`` the code export gives the same number; ``uncertainty_said`` the
    interval and the limits reported; ``bar_met`` the bar agreed before building met (None: no bar was agreed)."""
    checks = []
    for name in CHECKS:
        value = facts.get(name)
        why = str(facts.get(f"{name}_why") or "")
        status: CheckStatus = "unknown" if value is None else "passed" if value else "failed"
        checks.append(Check(name=name, status=status, why=why or ("its record cannot say" if value is None else "")))
    failed = [c for c in checks if c.status == "failed"]
    unknown = [c for c in checks if c.status == "unknown"]
    return BuildVerdict(build=build.id, verdict="incorrect" if failed else "unknown" if unknown else "correct", checks=checks)


class CurvePoint(Contract):
    labels: int = Field(ge=0)
    score: Optional[float] = None
    low: Optional[float] = None
    high: Optional[float] = None
    metric: str = Field(default="accuracy", max_length=60)


class LearningCurve(Versioned):
    """A point's or a kind's score against how many labels it had (v1), from its refits and experiments."""

    schema_version: Literal["dclab_core.learning_curve.v1"] = _version("learning_curve")
    subject: str = Field(max_length=200)              # a decision point, or a problem kind
    kind: Literal["point", "problem_kind"]
    points: list[CurvePoint] = Field(default_factory=list)
    builds: int = 0
    enough: bool = False                              # at least the builds a curve needs to be said
    why: str = Field(default="", max_length=600)


class KnowledgeItem(Versioned):
    """A practice, a rule or a pitfall of building one model type correctly (v1)."""

    schema_version: Literal["dclab_core.knowledge_item.v1"] = _version("knowledge_item")
    id: str = Field(max_length=120)
    model_type: str = Field(max_length=60)
    stage: Literal["problem", "data", "labels", "split", "features", "model", "evaluation", "deployment", "monitoring", "service"]
    practice: str = Field(max_length=600)
    why: str = Field(default="", max_length=1200)
    scope: dict[str, Any] = Field(default_factory=dict)   # when it holds, in data-card bands
    evidence: list[dict[str, Any]] = Field(default_factory=list)  # campaign ids with numbers, or a citation (source, link, licence)
    counterexamples: list[str] = Field(default_factory=list)
    status: Literal["measured_here", "from_the_literature", "hypothesis"] = "hypothesis"
    accepted_by: Optional[str] = Field(default=None, max_length=120)  # a person's yes; none: a proposal
    flagged: Optional[str] = Field(default=None, max_length=600)      # a later campaign contradicting it: for review, never changed silently


class ServiceEngagement(Versioned):
    """A piece of work done for someone (v1): request, clarification, agreement, delivery, explanation, handover, feedback."""

    schema_version: Literal["dclab_core.service_engagement.v1"] = _version("service_engagement")
    id: str = Field(max_length=200)
    workspace: str = Field(max_length=120)
    context: dict[str, Any] = Field(default_factory=dict)   # bands only
    request: Optional[str] = None                           # the person's first words, redacted
    questions: list[dict[str, Any]] = Field(default_factory=list)  # each question, whether it changed the problem
    agreement: dict[str, Any] = Field(default_factory=dict)  # the bar agreed before building, the brief approved
    expected: dict[str, Any] = Field(default_factory=dict)
    delivered: dict[str, Any] = Field(default_factory=dict)  # time, cost, result
    explanation: dict[str, Any] = Field(default_factory=dict)  # what was said of the result and its limits, what was asked after
    handover: dict[str, Any] = Field(default_factory=dict)   # code, report, service; used or not
    feedback: list[dict[str, Any]] = Field(default_factory=list)
    stopped: Optional[str] = Field(default=None, max_length=300)  # where and why it stopped, when it did
    unknown: dict[str, str] = Field(default_factory=dict)
    at: str


class PlaybookItem(Versioned):
    """What the engagements teach about serving well (v1), resting on what it counts, accepted by the owner."""

    schema_version: Literal["dclab_core.playbook_item.v1"] = _version("playbook_item")
    id: str = Field(max_length=120)
    topic: Literal["questions", "expectations", "explanations", "failure_points", "time_and_cost", "after_the_first_model"]
    kind: Optional[str] = Field(default=None, max_length=60)   # the problem kind it is about
    finding: str = Field(max_length=800)
    evidence: list[str] = Field(default_factory=list)         # the engagements and builds it rests on (ids)
    count: int = 0
    enough: bool = False
    accepted_by: Optional[str] = Field(default=None, max_length=120)


for _name, _cls in (("model_build", ModelBuild), ("build_verdict", BuildVerdict), ("learning_curve", LearningCurve),
                    ("knowledge_item", KnowledgeItem), ("service_engagement", ServiceEngagement), ("playbook_item", PlaybookItem)):
    CONTRACTS[_name] = _cls

__all__ = ["ModelBuild", "BuildVerdict", "Check", "CHECKS", "verdict", "CurvePoint", "LearningCurve", "KnowledgeItem", "ServiceEngagement",
           "PlaybookItem"]
