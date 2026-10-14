"""The pipeline of workflows (package 20.8): every step behind a project's numbers as one object with its lineage.

A project's numbers come from a chain: the problem as understood (its card), the table and its card, the solution the
person saved, the solution workflow, the analysis of the training rows (its findings), each stage the engine ran, and
the report. A :class:`Pipeline` lists those steps
in order, each with what it read and made (hashes, its run and the run's fingerprint), the step it depends on, where
the host keeps its record, and how it is rerun from that record. A step is *stale* when something it read has changed
since it ran (its own record says what it read; the host compares it with now); every step after a stale one is stale
too (:func:`stale_from`).

- :func:`produced`: the steps that produced a number of the report: the step whose record holds it (by the path's
  root: a stage's name, ``findings``, ``brief``, ``plan``) and every step before it on the chain.
- :func:`resume`: the first step that must run again (stale, failed or never run) whose own inputs are all done: where
  a resumed pipeline starts.
- :func:`by_node`: each node of the host's workflow graph with the worst status of the steps on it.
- :func:`graph_definition` and :func:`content_digest`: the pipeline's definition as nodes and edges (what each step is
  and reads, never its state), and its digest computed the way
  DCLab's main product computes ``pipeline_versions.content_digest`` (SHA-256 of the sorted, compact JSON of
  ``{"config", "graph_definition"}``), so a host can store it there.

Pure: pydantic only.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional, Union

from pydantic import Field, model_validator

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Scalar = Union[str, int, float, bool, None]
StepKind = Literal["problem", "data", "solution", "workflow", "analysis", "stage", "report"]
StepStatus = Literal["done", "stale", "failed", "not_run", "not_recorded"]
Rerun = Literal["replay", "rerun", "none"]
DONE = ("done",)
WORST = ("failed", "stale", "not_run", "not_recorded", "done")  # a node shows the first of these its steps hold
_SUFFIX = re.compile(r"[@#]\d+$")
MAX_NUMBERS = 2000  # number paths kept per step


class PipelineStep(Contract):
    """One step: what it read and made, the run that made it, and how it is rerun."""

    id: str = Field(min_length=1, max_length=60, description="Unique in the pipeline: problem, data, solution, workflow, stage:<name>, report.")
    kind: StepKind
    name: str = Field(min_length=1, max_length=120)
    depends_on: list[str] = Field(default_factory=list, max_length=20)
    nodes: list[str] = Field(default_factory=list, max_length=10, description="The host's workflow-graph nodes the step carries out.")
    roots: list[str] = Field(default_factory=list, max_length=10, description="The record roots whose numbers it wrote "
                             "(a stage's name, findings, brief, plan): a number's path names its step by its root.")
    status: StepStatus
    why: Optional[str] = Field(default=None, max_length=600, description="Why it is stale, failed, not run or not recorded.")
    inputs: dict[str, Scalar] = Field(default_factory=dict, description="What it read, by hash or count: never a table's value.")
    outputs: dict[str, Scalar] = Field(default_factory=dict, description="What it made: hashes, its metric, value and choice.")
    run: Optional[str] = Field(default=None, max_length=80)
    fingerprint: Optional[str] = Field(default=None, max_length=64)
    record: Optional[str] = Field(default=None, max_length=200, description="Where the host keeps the step's record.")
    numbers: list[str] = Field(default_factory=list, max_length=MAX_NUMBERS, description="The record paths of the numbers it wrote.")
    rerun: Rerun = Field(default="none", description="replay: rerun from its record and compared; rerun: run again; none.")


class Pipeline(Versioned):
    """The steps behind a project's numbers, in the order they run."""

    schema_version: Literal["dclab_core.pipeline.v1"] = _version("pipeline")
    project: Optional[str] = Field(default=None, max_length=64)
    steps: list[PipelineStep] = Field(min_length=1, max_length=60)

    @model_validator(mode="after")
    def _ordered(self) -> "Pipeline":
        seen: set[str] = set()
        for step in self.steps:
            if step.id in seen:
                raise ValueError(f"step {step.id!r} appears twice")
            missing = [d for d in step.depends_on if d not in seen]
            if missing:
                raise ValueError(f"step {step.id!r} depends on {', '.join(missing)}, which do not come before it")
            seen.add(step.id)
        return self

    def step(self, step_id: str) -> PipelineStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)


CONTRACTS["pipeline"] = Pipeline


def downstream(pipeline: Pipeline, step_id: str) -> list[str]:
    """The steps that read, directly or not, what ``step_id`` made (in pipeline order, the step itself excluded)."""
    reached = {step_id}
    out = []
    for s in pipeline.steps:
        if s.id != step_id and reached & set(s.depends_on):
            reached.add(s.id)
            out.append(s.id)
    return out


def upstream(pipeline: Pipeline, step_id: str) -> list[str]:
    """The steps ``step_id`` read from, directly or not, in pipeline order (the step itself included, last)."""
    need = {step_id}
    for s in reversed(pipeline.steps):
        if s.id in need:
            need |= set(s.depends_on)
    return [s.id for s in pipeline.steps if s.id in need]


def stale_from(pipeline: Pipeline) -> Pipeline:
    """The pipeline with every step after a stale, failed or unrun step marked stale (a done step that read what is
    no longer done, or what changed since, cannot be trusted to hold)."""
    bad: set[str] = set()
    steps = []
    for s in pipeline.steps:
        hit = sorted(set(s.depends_on) & bad)
        if s.status == "done" and hit:
            s = s.model_copy(update={"status": "stale", "why": f"It read {', '.join(hit)}, which must run again."})
        if s.status in ("stale", "failed", "not_run") or (s.status == "not_recorded" and s.kind == "stage"):
            bad.add(s.id)  # a stage that cannot be checked cannot vouch for what read it
        steps.append(s)
    return pipeline.model_copy(update={"steps": steps})


def produced(pipeline: Pipeline, path: str) -> list[str]:
    """The steps that produced the number at record ``path`` (``final.evidence.holdout_metrics.roc_auc``,
    ``findings.F-time.statement@0``): the step whose numbers hold it, else the step whose roots hold the path's root
    (a ``@n``/``#n`` suffix read off), and every step it read from. A report step is the holder only for its own roots."""
    bare = _SUFFIX.sub("", path)
    holder = next((s.id for s in pipeline.steps if s.kind != "report" and (path in s.numbers or bare in s.numbers)), None)
    if holder is None:
        root = bare.split(".", 1)[0]
        holder = next((s.id for s in pipeline.steps if root in s.roots), None)
    return upstream(pipeline, holder) if holder else []


def resume(pipeline: Pipeline) -> Optional[str]:
    """The first step that must run (stale, failed or not run) whose inputs are all done: where to start again. None
    when every step is done (or only steps that cannot be recorded remain)."""
    status = {s.id: s.status for s in stale_from(pipeline).steps}
    for s in pipeline.steps:
        if status[s.id] in ("stale", "failed", "not_run") and all(status[d] in DONE + ("not_recorded",) for d in s.depends_on):
            return s.id
    return None


def by_node(pipeline: Pipeline) -> dict[str, str]:
    """Each workflow-graph node with the worst status of the steps that carry it out (stale propagated)."""
    out: dict[str, str] = {}
    for s in stale_from(pipeline).steps:
        for node in s.nodes:
            have = out.get(node)
            out[node] = s.status if have is None or WORST.index(s.status) < WORST.index(have) else have
    return out


def graph_definition(pipeline: Pipeline) -> dict[str, Any]:
    """The pipeline's definition as nodes and edges, for a host that stores a graph (the main product's
    ``pipeline_versions.graph_definition``): what each step is and reads, never its state or what it made, so the
    digest changes only when the pipeline itself does."""
    return {"schema_version": pipeline.schema_version,
            "nodes": [{"id": s.id, "kind": s.kind, "name": s.name, "nodes": list(s.nodes)} for s in pipeline.steps],
            "edges": [{"from": d, "to": s.id} for s in pipeline.steps for d in s.depends_on]}


def content_digest(graph: Mapping[str, Any], config: Optional[Mapping[str, Any]] = None) -> str:
    """The main product's digest of a pipeline version: SHA-256 of the sorted, compact JSON of its config and graph."""
    encoded = json.dumps({"config": dict(config or {}), "graph_definition": dict(graph)}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def summary(pipeline: Pipeline) -> dict[str, Any]:
    """Counts by status and the step to resume from."""
    steps = stale_from(pipeline).steps
    counts: dict[str, int] = {}
    for s in steps:
        counts[s.status] = counts.get(s.status, 0) + 1
    return {"count": len(steps), "by_status": counts, "resume_from": resume(pipeline)}


def from_steps(project: Optional[str], steps: Iterable[Mapping[str, Any] | PipelineStep]) -> Pipeline:
    """A pipeline from its steps (validated, staleness carried forward)."""
    return stale_from(Pipeline(project=project, steps=[s if isinstance(s, PipelineStep) else PipelineStep.model_validate(s) for s in steps]))
