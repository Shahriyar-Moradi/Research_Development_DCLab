"""The programme (package 24.5): a solution map run, leaf by leaf, then composed, with the holdout read once.

A solution map (24.4) says what to run and how to split; a programme is the record of running it. It has a leaf for every block of the map, in the order of
the work:

``planned`` -> ``leaves_built`` -> ``composed``

* **Leaves** (``leaves.run_leaf``) are fitted on the earlier rows, choose what they choose on the validation rows and hand their scores for the validation and
  holdout rows to the composition; none scores the holdout. A leaf that cannot run (no confirmed case to learn from, no related table, an approach with no runner
  yet) is *blocked*, with the reason, and it stays a blocker until the person drops it.
* **The composition** (``compose.choose``) is chosen on the validation rows among every leaf alone and the ways to combine them; the best leaf alone is the bar.
* **The holdout** is opened by ``read_holdout`` and only when every leaf that was to run has run, the composition is chosen and nothing has read it before. It is
  read once: the programme then holds the composed score, every leaf's, and the paired difference of the composition against the best leaf *chosen on
  validation*, each with an interval. A second read is refused.

``validate`` refuses a programme that has a holdout result with no composition, two reads, a composition that names a leaf that is not built, or a state its
parts do not support. A programme is never edited in place: every step returns a new one with a new fingerprint over everything it decides.
Deterministic (seeded); no model, no store; the rows stay in the host and only names, counts and scores are kept.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal, Optional

import pandas as pd
from pydantic import Field, model_validator

from dclab_core import compose, leaves
from dclab_core.compose import Metric
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.solution_map import SolutionMap, SplitDesign
from dclab_core.solution_map import validate as sm_validate

if TYPE_CHECKING:
    from dclab_core.problem_map import ProblemMap

State = Literal["planned", "leaves_built", "composed"]
Status = Literal["planned", "built", "blocked", "composition", "dropped"]
COMPOSITION_ONLY = ("review_budgeted",)
MAX_LEAVES = 40


class Estimate(Contract):
    value: float
    low: float
    high: float


class Choice(Contract):
    """The composition chosen on validation, and what each candidate did there."""

    chosen: str = Field(max_length=80)
    candidates: dict[str, float] = Field(description="Each candidate's score on the validation rows.")
    best_leaf: str = Field(max_length=60, description="The leaf that did best alone on validation: what the composition must beat.")
    best_leaf_score: float
    weights: dict[str, float] = Field(default_factory=dict)
    fused: list[str] = Field(default_factory=list, description="The leaves the fusions combine; the best leaf alone is taken among all of the leaves above.")
    metric: Metric
    budget_share: Optional[float] = None
    budget_rows: int = Field(ge=0)
    positives: int = Field(ge=0, description="Confirmed cases among the validation rows.")
    rule: str = Field(max_length=300)
    warnings: list[str] = Field(default_factory=list, max_length=6)


class HoldoutResult(Contract):
    """The one read of the holdout: the composed system and every leaf, with intervals, and the composed system against the best leaf."""

    reads: int = Field(ge=1, le=1, description="How many times the holdout was read for this programme: one.")
    metric: Metric
    chosen: str = Field(max_length=80)
    best_leaf: str = Field(max_length=60)
    budget_share: Optional[float] = None
    budget_rows: int = Field(ge=0)
    rows: int = Field(ge=0)
    positives: int = Field(ge=0)
    composed: Estimate
    leaves: dict[str, Estimate]
    best_leaf_estimate: Estimate
    difference: Estimate = Field(description="Composed minus the best leaf, paired on the same resamples.")
    beats_best_leaf: bool = Field(description="The difference's interval is above zero.")
    cluster_key: Optional[str] = Field(default=None, max_length=80)
    bootstraps: int = Field(ge=0)
    limits: list[str] = Field(default_factory=list, max_length=8)


class LeafRun(Contract):
    """One block of the solution map, run: what it chose, how it did alone on validation, what it read. No value of a row."""

    block: str = Field(pattern=r"^blk\d+$")
    sub_problem: str = Field(pattern=r"^sp\d+$")
    title: str = Field(max_length=200)
    approach: Optional[str] = Field(default=None, max_length=40)
    status: Status
    why: str = Field(default="", max_length=400, description="Why it is blocked, dropped, or only part of the composition.")
    chosen: Optional[str] = Field(default=None, max_length=120, description="The detector, model or rule the leaf chose on validation.")
    validation: dict[str, float] = Field(default_factory=dict, description="What each alternative scored on the validation rows, where the leaf chose among several.")
    reads: list[str] = Field(default_factory=list, max_length=MAX_LEAVES, description="The blocks whose scores this leaf reads.")
    columns: list[str] = Field(default_factory=list, max_length=60)
    rows: dict[str, int] = Field(default_factory=dict)
    left_out: Optional[str] = Field(default=None, max_length=400)
    notes: list[str] = Field(default_factory=list, max_length=6)
    limits: list[str] = Field(default_factory=list, max_length=6)
    holdout: Literal["none", "predicted"] = Field(default="none", description="predicted: it scored the holdout rows for the composition and never judged them.")


class Programme(Versioned):
    """A solution map run: the leaves, the composition chosen on validation, and the one read of the holdout."""

    schema_version: Literal["dclab_core.programme.v1"] = _version("programme")
    map: str = Field(max_length=64, description="The fingerprint of the problem map.")
    solution: str = Field(max_length=64, description="The fingerprint of the solution map it runs.")
    state: State = "planned"
    split: SplitDesign
    budget: Optional[float] = Field(default=None, gt=0)
    metric: Metric = "recall_at_budget"
    leaves: list[LeafRun] = Field(min_length=1, max_length=MAX_LEAVES)
    fused: list[str] = Field(default_factory=list, max_length=MAX_LEAVES, description="The blocks the composition combines.")
    gates: list[list[str]] = Field(default_factory=list, max_length=MAX_LEAVES, description="Pairs [by, leaf]: the leaf counts only for rows the first puts in its top share.")
    periods: dict[str, int] = Field(default_factory=dict, description="Rows in fit, validation and holdout, and the rows that belong to no period.")
    reads_before: int = Field(default=0, ge=0, description="How many times an earlier programme on the same split read the holdout: once any did, a leaf can no longer be dropped and a "
                                                           "run needs a person's reason.")
    composition: Optional[Choice] = None
    holdout: Optional[HoldoutResult] = None
    blockers: list[str] = Field(default_factory=list, max_length=MAX_LEAVES)
    notes: list[str] = Field(default_factory=list, max_length=10)
    fingerprint: str = Field(default="", max_length=64)

    @model_validator(mode="after")
    def _fingerprinted(self) -> "Programme":
        expected = fingerprint(self)
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("fingerprint does not match the programme: a programme is never edited in place; step it")
        object.__setattr__(self, "fingerprint", expected)
        self.__pydantic_fields_set__.add("fingerprint")
        return self


CONTRACTS["programme"] = Programme


def fingerprint(p: Programme) -> str:
    """The hash of everything the programme decides: the maps it runs, its state, split, budget, every leaf (its choice and its scores on validation), the
    composition and the holdout result."""
    body = {"map": p.map, "solution": p.solution, "state": p.state, "split": p.split.model_dump(), "budget": p.budget, "metric": p.metric,
            "leaves": [l.model_dump() for l in p.leaves], "fused": p.fused, "gates": p.gates, "periods": p.periods, "reads_before": p.reads_before,
            "composition": p.composition.model_dump() if p.composition else None, "holdout": p.holdout.model_dump() if p.holdout else None,
            "blockers": p.blockers, "notes": p.notes}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _step(p: Programme, **changes: Any) -> Programme:
    data = p.model_dump()
    data.update(changes)
    data["fingerprint"] = ""
    return Programme.model_validate(data)


# ------------------------------------------------------------------------------------------------------------------------ plan

def _blocker_texts(leaves_: Sequence[LeafRun]) -> list[str]:
    return [f"{l.block} ({l.title}): {l.why}" for l in leaves_ if l.status == "blocked"]


def plan(sm: SolutionMap, m: "ProblemMap", *, budget: Optional[float] = None) -> Programme:
    """A programme for a complex map: a leaf per block, the composition's inputs and gates read from the problem map. A single problem scores its own holdout and
    is not a programme."""
    if sm.holdout.read_by != "composition":
        raise ValueError("this solution map is one problem that scores its own holdout: there is nothing to compose, so it is not a programme")
    if sm.map != m.fingerprint:
        raise ValueError("this solution map answers another problem map: draw the solution again")
    unsound = sm_validate(sm, m)
    if unsound:
        raise ValueError("the solution map is not sound, so it cannot be run: " + "; ".join(unsound[:3]))
    by_sub = {b.sub_problem: b.id for b in sm.blocks}
    out: list[LeafRun] = []
    for b in sm.blocks:
        run = LeafRun(block=b.id, sub_problem=b.sub_problem, title=b.title, approach=b.approach, status="planned")
        if b.approach is None:
            run = run.model_copy(update={"status": "blocked", "why": "no approach was routed to it"})
        elif b.approach in COMPOSITION_ONLY:
            run = run.model_copy(update={"status": "composition", "why": "it is the composition's budget step, not a model"})
        elif b.approach not in leaves.RUNNERS:
            run = run.model_copy(update={"status": "blocked", "why": f"the programme has no runner for the approach {b.approach!r} yet"})
        out.append(run)
    fused: list[str] = []
    gates: list[list[str]] = []
    budget_step = False
    for st in m.composition.steps:
        if st.kind == "fusion":
            fused += [by_sub[i] for i in st.inputs if i in by_sub and by_sub[i] not in fused]
        elif st.kind == "gate" and len(st.inputs) >= 2 and st.inputs[0] in by_sub and st.inputs[1] in by_sub:
            gates.append([by_sub[st.inputs[0]], by_sub[st.inputs[1]]])
        elif st.kind == "budget":
            budget_step = True
    runnable = [r.block for r in out if r.status != "composition"]
    fused = fused or runnable
    metric: Metric = "recall_at_budget" if (budget_step or budget) else "average_precision"
    notes = list(sm.notes)
    if metric == "recall_at_budget" and not budget:
        notes.append("There is a budget step and no budget: the review list is 1% of the rows until the people who review say how many they can check.")
    return Programme(map=sm.map, solution=sm.fingerprint, split=sm.holdout.split, budget=budget, metric=metric, leaves=out, fused=fused, gates=gates,
                     blockers=_blocker_texts(out), notes=notes[:10])


def replan(sm: SolutionMap, m: "ProblemMap", prior: Optional[Programme] = None, *, budget: Optional[float] = None, replaying: bool = False) -> Programme:
    """A fresh plan that keeps what a person decided and what was already seen. The holdout is fixed by the split design, not by the routes: a programme on the same split
    design carries the count of times the holdout was read, even when the solution map changed (a new table, a new route). The drops of ``prior`` are carried only for the same
    solution map (a leaf is dropped by a decision about the data, not by a run), and a leaf can no longer be dropped once any programme read the holdout, so carrying a
    drop forward lets nobody drop one after seeing it. ``replaying``: the plan is to run that same programme again (a replay, an export), so the read it made is not counted
    twice."""
    fresh = plan(sm, m, budget=budget)
    if prior is None:
        return fresh
    if prior.solution == fresh.solution:
        for leaf in prior.leaves:
            if leaf.status == "dropped":
                fresh = drop_leaf(fresh, leaf.block, leaf.why.split(": ", 1)[-1] or "dropped earlier", by="person")
    if prior.split == fresh.split:
        fresh = _step(fresh, reads_before=prior.reads_before + (1 if prior.holdout is not None and not replaying else 0))
    return fresh


# ------------------------------------------------------------------------------------------------------------------------ the rules

def open_issues(p: Programme) -> list[str]:
    """Why the holdout cannot be read yet (empty: it can). The graph's rule: it opens when every leaf that was to run has run and the composition is chosen."""
    issues = []
    if p.holdout is not None:
        issues.append("the holdout was already read for this programme: it is read once")
    for l in p.leaves:
        if l.status in ("planned", "blocked"):
            issues.append(f"{l.block} ({l.title}) is {l.status}" + (f": {l.why}" if l.why else "") + (" (run it, or drop it with a reason)" if l.status == "blocked" else ""))
    if not any(l.status == "built" for l in p.leaves):
        issues.append("no leaf is built")
    if p.composition is None:
        issues.append("the composition is not chosen on the validation rows yet")
    elif p.composition.positives < compose.MIN_POSITIVES:
        issues.append(f"the composition would be chosen on {p.composition.positives} confirmed case(s) among the validation rows (at least {compose.MIN_POSITIVES} are needed): "
                      "it is a coin toss, and the holdout is too precious to spend on it")
    return issues


def validate(p: Programme, sm: Optional[SolutionMap] = None) -> list[str]:
    """What is wrong with a programme as a document (empty: it holds). With the solution map, also whether it runs that map."""
    issues = []
    ids = [l.block for l in p.leaves]
    if len(set(ids)) != len(ids):
        issues.append("two leaves have the same block")
    built = {l.block for l in p.leaves if l.status == "built"}
    if p.state in ("leaves_built", "composed") and any(l.status in ("planned",) for l in p.leaves):
        issues.append(f"the state is {p.state} but a leaf is still planned")
    if p.state == "composed" and (p.composition is None or p.holdout is None):
        issues.append("a composed programme has a chosen composition and a holdout result")
    if p.state != "composed" and p.holdout is not None:
        issues.append("a holdout result with a programme that is not composed")
    if p.holdout is not None and p.composition is None:
        issues.append("the holdout was read with no composition chosen on validation")
    if p.holdout is not None and p.holdout.reads != 1:
        issues.append("the holdout is read once")
    for name in (*p.fused, *(b for pair in p.gates for b in pair)):
        if name not in ids:
            issues.append(f"the composition names {name}, which is not a leaf of this programme")
    if p.composition is not None:
        c = p.composition
        names = set(c.candidates)
        if c.chosen not in names:
            issues.append("the chosen composition is not one of the candidates judged on validation")
        for l in p.leaves:
            if l.status != "built" and l.block in names:
                issues.append(f"the composition judged {l.block}, which is {l.status}")
        if c.best_leaf not in built:
            issues.append("the best leaf is not a built leaf")
        if set(c.weights) - built:
            issues.append("the composition was judged with a leaf that is not built")
        if p.metric != c.metric:
            issues.append("the composition was judged by another metric than the programme's")
    h = p.holdout
    if h is not None and p.composition is not None:
        c = p.composition
        if h.chosen != c.chosen or h.best_leaf != c.best_leaf:
            issues.append("the holdout was scored with another composition than the one chosen on validation")
        if h.metric != p.metric:
            issues.append("the holdout was scored by another metric than the programme's")
        if p.periods.get("holdout") not in (None, h.rows):
            issues.append("the holdout result counts other rows than the programme's holdout period")
        if h.metric == "recall_at_budget" and h.budget_rows != compose.budget_rows(h.rows, p.budget):
            issues.append("the holdout result used another review budget than the programme's")
        if set(h.leaves) != set(c.weights):
            issues.append("the holdout result does not score the leaves the composition was chosen among")
        if set(c.weights) != built:
            issues.append("a leaf was built or dropped after the composition was chosen")
        for l in p.leaves:
            if l.status in ("planned", "blocked"):
                issues.append(f"the holdout was read while {l.block} was {l.status}")
        if c.positives < compose.MIN_POSITIVES:
            issues.append("the holdout was read although the composition was chosen on too few confirmed cases")
    for l in p.leaves:
        if l.status == "built" and not l.chosen:
            issues.append(f"{l.block} is built but chose nothing")
        for r in l.reads:
            if r not in ids:
                issues.append(f"{l.block} reads {r}, which is not a leaf of this programme")
    if sm is not None:
        if sorted(ids) != sorted(b.id for b in sm.blocks):
            issues.append("the programme's leaves are not the blocks of the solution map")
        if p.solution != sm.fingerprint:
            issues.append("the solution map changed since this programme was planned: plan again")
        if p.map != sm.map:
            issues.append("the programme answers another problem map than the solution map does")
        if p.split != sm.holdout.split:
            issues.append("the programme splits differently from the solution map")
    return issues


def drop_leaf(p: Programme, block: str, reason: str, by: str = "person") -> Programme:
    """The person drops a blocked leaf: the composition goes on without it, and the programme says so. A leaf that ran is not dropped here (it is a result)."""
    leaf = next((l for l in p.leaves if l.block == block), None)
    if leaf is None:
        raise ValueError(f"no leaf {block}")
    if p.holdout is not None or p.reads_before > 0:
        raise ValueError("the holdout was already read on this split: a leaf cannot be dropped after it (that would be choosing the problem by its answer)")
    if leaf.status not in ("blocked", "planned"):
        raise ValueError(f"{block} is {leaf.status}: only a blocked or planned leaf is dropped")
    if not reason.strip():
        raise ValueError("a dropped leaf needs the reason")
    updated = [l.model_copy(update={"status": "dropped", "why": f"dropped by {by}: {reason.strip()[:300]}"}) if l.block == block else l for l in p.leaves]
    return _step(p, leaves=[l.model_dump() for l in updated], fused=[b for b in p.fused if b != block], blockers=_blocker_texts(updated),
                 notes=[*p.notes, f"{block} ({leaf.title}) was dropped by {by}: the composition goes on without it."][-10:])


# ------------------------------------------------------------------------------------------------------------------------ the run

@dataclass
class Execution:
    """What a run returns: the programme (the record) and, in memory only, the leaves' scores the host may keep for its report."""

    programme: Programme
    scores: dict[str, leaves.LeafScores] = field(default_factory=dict)


def _order(p: Programme, m: "ProblemMap", sm: SolutionMap) -> list[str]:
    """The leaves in the order they run: by the order their prediction is made in the problem, then by position, but a reader always after every leaf it reads (a
    topological order, so two sub-problems with the same rank cannot run the wrong way round)."""
    rank = {s.id: (s.predict_rank, i) for i, s in enumerate(m.sub_problems)}
    sub = {b.id: b.sub_problem for b in sm.blocks}
    reads = _reads(m, sm)
    waiting = sorted((l.block for l in p.leaves), key=lambda b: rank.get(sub[b], (99, 99)))
    placed: list[str] = []
    while waiting:
        ready = next((b for b in waiting if all(r in placed or r not in waiting for r in reads.get(b, []))), waiting[0])  # a cycle cannot happen in a sound map: run in rank order
        placed.append(ready)
        waiting.remove(ready)
    return placed


def _reads(m: "ProblemMap", sm: SolutionMap) -> dict[str, list[str]]:
    by_sub = {b.sub_problem: b.id for b in sm.blocks}
    out: dict[str, list[str]] = {}
    for s in m.sub_problems:
        out[by_sub[s.id]] = [by_sub[r.sub_problem] for r in s.reads if r.what == "score" and r.sub_problem in by_sub]
    return out


def build_leaves(p: Programme, sm: SolutionMap, m: "ProblemMap", ctx: leaves.Context) -> Execution:
    """Run every planned leaf in order. A leaf that raises is blocked with the reason; a leaf that reads a blocked leaf is blocked with that."""
    reads = _reads(m, sm)
    blocks = {b.id: b for b in sm.blocks}
    state = {l.block: l for l in p.leaves}
    scores: dict[str, leaves.LeafScores] = {}
    counts = {"fit": len(ctx.parts.fit), "validation": len(ctx.parts.validation), "holdout": len(ctx.parts.holdout)}
    periods = {**counts, "no_period": int(ctx.parts.info.get("no_period") or 0)}
    for bid in _order(p, m, sm):
        leaf = state[bid]
        if leaf.status != "planned":
            continue
        gone = [r for r in reads.get(bid, []) if r not in scores and r not in ctx.done]
        if gone:
            state[bid] = leaf.model_copy(update={"status": "blocked", "why": f"it reads {', '.join(gone)}, which did not run"})
            continue
        try:
            got = leaves.run_leaf(blocks[bid], ctx, reads=list(reads.get(bid, [])))
        except (ValueError, KeyError) as exc:
            state[bid] = leaf.model_copy(update={"status": "blocked", "why": str(exc)[:380]})
            continue
        scores[bid] = got
        ctx.done[bid] = got
        r = got.record
        state[bid] = leaf.model_copy(update={"status": "built", "chosen": r.get("chosen"), "validation": r.get("validation", {}), "reads": r.get("reads", []),
                                             "columns": [str(c) for c in r.get("columns", [])][:60], "rows": counts, "left_out": r.get("left_out"),
                                             "notes": r.get("notes", [])[:6], "limits": r.get("limits", [])[:6], "holdout": "predicted", "why": ""})
    updated = [state[l.block] for l in p.leaves]
    runnable = [l for l in updated if l.status != "composition"]
    ready = all(l.status in ("built", "dropped") for l in runnable)
    return Execution(_step(p, leaves=[l.model_dump() for l in updated], state="leaves_built" if ready else "planned", periods=periods,
                           blockers=_blocker_texts(updated)), scores)


def compose_leaves(p: Programme, scores: Mapping[str, leaves.LeafScores], labels: pd.Series) -> Programme:
    """Choose the composition on the validation rows (nothing of the holdout is read here). The best leaf alone, the bar to beat, is taken among every leaf that ran; the
    fusions combine the ones the problem map fuses."""
    built = [l.block for l in p.leaves if l.status == "built"]
    if not built:
        raise ValueError("no leaf is built: there is nothing to compose")
    fused = [b for b in p.fused if b in built] or built
    validation = {b: scores[b].scores["validation"] for b in built}
    gates = [(a, b) for a, b in p.gates if a in built and b in built]
    choice = compose.choose(validation, labels, metric=p.metric, share=p.budget if p.metric == "recall_at_budget" else None, gates=gates, leaf_order=built, fused=fused)
    return _step(p, composition=choice.model_dump(), notes=[*p.notes, *choice.warnings][-10:])


def cluster_series(frame: pd.DataFrame, entities: Sequence[str], rows: pd.Index, minimum: int = 30) -> Optional[pd.Series]:
    """The entity whose rows move together, for the bootstrap: of the entities with at least ``minimum`` distinct values among ``rows``, the one with the fewest
    (the coarsest dependence). None when there is none: rows are then resampled one by one."""
    best = None
    for e in entities:
        if e in frame.columns:
            n = int(frame.loc[rows, e].nunique())
            if n >= minimum and (best is None or n < best[0]):
                best = (n, e)
    return None if best is None else frame[best[1]].rename(best[1])


def read_holdout(p: Programme, scores: Mapping[str, leaves.LeafScores], labels: pd.Series, *, clusters: Optional[pd.Series] = None, bootstraps: int = compose.BOOTSTRAPS,
                 seed: int = 0) -> Programme:
    """Open the holdout, once: the issues of ``open_issues`` refuse it. Scores the chosen composition and every leaf, and the difference against the best leaf."""
    issues = open_issues(p)
    if issues:
        raise HoldoutClosed(issues)
    assert p.composition is not None
    use = [b for b in p.composition.weights]
    on_holdout = int(labels.reindex(scores[use[0]].scores["holdout"].index).fillna(0).astype(int).sum())
    if on_holdout < compose.MIN_POSITIVES:  # only a count is looked at: a holdout with this few cases cannot tell a gain from chance, so it is not spent
        raise HoldoutClosed([f"only {on_holdout} confirmed case(s) among the holdout rows (at least {compose.MIN_POSITIVES} are needed): a read of it would say nothing"])
    holdout = {b: scores[b].scores["holdout"] for b in use}
    result = compose.read_holdout(holdout, labels, p.composition, clusters=clusters, bootstraps=bootstraps, seed=seed)
    return _step(p, holdout=result.model_dump(), state="composed")


class HoldoutClosed(ValueError):
    """The holdout cannot be read yet; ``issues`` says why."""

    def __init__(self, issues: Sequence[str]):
        self.issues = list(issues)
        super().__init__("the holdout is closed: " + "; ".join(self.issues))


def execute(sm: SolutionMap, m: "ProblemMap", frame: pd.DataFrame, *, tables: Optional[Mapping[str, pd.DataFrame]] = None, labels: Optional[pd.Series] = None,
            known_at: Optional[pd.Series] = None, entities: Sequence[str] = (), time: Optional[str] = None, exclude: Sequence[str] = (), budget: Optional[float] = None,
            seed: int = 0, bootstraps: int = compose.BOOTSTRAPS) -> Execution:
    """Plan, split, run the leaves, choose the composition on validation and, when every leaf ran, read the holdout once. A leaf that is blocked leaves the programme
    at ``planned`` with its blockers: the person decides (drop it, or fix the data) and calls ``resume``."""
    p = plan(sm, m, budget=budget)
    parts = leaves.partition(frame, moment=sm.holdout.split.moment, keys=sm.holdout.split.keys, share=sm.holdout.split.share, seed=sm.holdout.split.seed)
    ctx = leaves.Context(frame=frame, parts=parts, time=time or sm.holdout.split.moment, entities=list(entities or sm.holdout.split.keys), labels=labels, known_at=known_at,
                         tables=dict(tables or {}), exclude=list(exclude), budget=budget, seed=seed)
    return resume(Execution(p), sm, m, ctx, labels, bootstraps=bootstraps)


def resume(run: Execution, sm: SolutionMap, m: "ProblemMap", ctx: leaves.Context, labels: Optional[pd.Series], *, bootstraps: int = compose.BOOTSTRAPS) -> Execution:
    """Carry a programme as far as it can go: build the leaves still planned, choose the composition on validation and, when nothing blocks, read the holdout once."""
    p = run.programme
    for l in p.leaves:
        if l.status == "built" and l.block in run.scores:
            ctx.done[l.block] = run.scores[l.block]
    built = build_leaves(p, sm, m, ctx)
    scores = {**run.scores, **built.scores}
    p = built.programme
    if labels is None:
        return Execution(_step(p, notes=[*p.notes, "No confirmed case was given: the composition cannot be judged, and the holdout stays closed."][-10:]), scores)
    if any(l.status in ("planned", "blocked") for l in p.leaves) or not any(l.status == "built" for l in p.leaves):
        return Execution(p, scores)
    p = compose_leaves(p, scores, labels)
    clusters = cluster_series(ctx.frame, ctx.entities, ctx.parts.holdout)
    try:
        p = read_holdout(p, scores, labels, clusters=clusters, bootstraps=bootstraps, seed=ctx.seed)
    except HoldoutClosed as closed:  # the holdout stays closed and the programme says why: nothing is lost, nothing is spent
        p = _step(p, state="leaves_built", notes=[*p.notes, "The holdout stays closed: " + "; ".join(closed.issues)][-10:])
    return Execution(p, scores)


# ------------------------------------------------------------------------------------------------------------------------ the report

def sections(p: Programme) -> list[dict[str, Any]]:
    """The programme as sections of a report: one per leaf, then the composition and the holdout. Each is ``{"title", "lines"}``; every number in a line is from the record."""
    out = []
    for l in p.leaves:
        lines = [f"Approach: {l.approach or 'none'}. Status: {l.status}" + (f" ({l.why})" if l.why else "") + "."]
        if l.status == "built":
            lines.append(f"Chosen on the validation rows: {l.chosen}" + (" among " + ", ".join(f"{k} {v:.2f}" for k, v in l.validation.items()) if l.validation else "") + ".")
            if l.reads:
                lines.append("Reads the score of " + ", ".join(l.reads) + ".")
            if l.left_out:
                lines.append(l.left_out + ".")
            lines += l.notes + l.limits
        out.append({"title": f"{l.block}: {l.title}", "lines": lines})
    c = p.composition
    if c is not None:
        lines = [f"Chosen on the validation rows: {c.chosen}. Rule: {c.rule}.",
                 "Judged on validation (" + ("recall at the budget" if c.metric == "recall_at_budget" else "average precision") + "): " + ", ".join(f"{k} {v:.2f}" for k, v in c.candidates.items()) + ".",
                 f"The best leaf alone on validation is {c.best_leaf} ({c.best_leaf_score:.2f}); the composition must beat it.", *c.warnings]
        out.append({"title": "The composition", "lines": lines})
    h = p.holdout
    if h is not None:
        d = h.difference
        lines = [f"The holdout was read once ({h.rows} rows, {h.positives} confirmed cases, a budget of {h.budget_rows} rows).",
                 f"Composed ({h.chosen}): {h.composed.value:.2f} (interval {h.composed.low:.2f} to {h.composed.high:.2f}).",
                 f"Best leaf ({h.best_leaf}, chosen on validation): {h.best_leaf_estimate.value:.2f} (interval {h.best_leaf_estimate.low:.2f} to {h.best_leaf_estimate.high:.2f}).",
                 f"Composed minus best leaf: {d.value:+.2f} (interval {d.low:+.2f} to {d.high:+.2f}): " + ("above zero." if h.beats_best_leaf else "not above zero: the gain is not shown."),
                 "Every leaf: " + ", ".join(f"{k} {v.value:.2f}" for k, v in h.leaves.items()) + ".", *h.limits]
        if p.reads_before:
            lines.insert(1, f"An earlier programme on this split already read the holdout {p.reads_before} time(s): this is not its first look, and the score is optimistic (PIT-006).")
        out.append({"title": "The holdout, read once", "lines": lines})
    if p.blockers:
        out.append({"title": "What blocks the programme", "lines": list(p.blockers)})
    return out


def report(p: Programme, title: Optional[str] = None) -> "Any":
    """The programme as a ``SolutionReport`` (17.7): a section per leaf, the composition and the holdout read once, with every number copied from the programme's record by
    code and traced to its path (``programme.holdout.difference.low``). A line whose numbers cannot all be traced, or that calls a model production-ready, is left out, as
    for the stages' report. ``report.check(report, context)`` re-reads every path: the context is ``{"programme": <the document>}``."""
    from dclab_core import report as R

    ctx = programme_context(p)
    w = R._Writer(ctx)
    n = lambda path, style="num": w.n(path, style)
    for i, leaf in enumerate(p.leaves):
        at = f"programme.leaves.{i}"
        w.section(f"{leaf.block}: {leaf.title}")
        w.line(f"Approach {w.q(at + '.approach', 40, code=True) or 'none'}; status {w.q(at + '.status', 20)}" + (f" ({w.q(at + '.why', 300)})." if leaf.why else "."))
        if leaf.status == "built":
            alternatives = ", ".join(f"{w.key(f'{at}.validation.{k}')} {n(f'{at}.validation.{k}')}" for k in leaf.validation)
            w.line(f"Chosen on the validation rows: {w.q(at + '.chosen', 120, code=True)}" + (f" among {alternatives}" if alternatives else "") + ".")
            if leaf.reads:
                w.line("Reads the score of " + ", ".join(w.q(f"{at}.reads.{j}", 20, code=True) for j in range(len(leaf.reads))) + ".")
            if leaf.columns:
                w.line("Columns read: " + ", ".join(w.q(f"{at}.columns.{j}", 60, code=True) for j in range(len(leaf.columns))) + ".")
            if leaf.left_out:
                w.line(w.q(at + ".left_out", 400) + ".")
            for j in range(len(leaf.notes)):
                w.line(w.q(f"{at}.notes.{j}", 400))
            for j in range(len(leaf.limits)):
                w.line(w.q(f"{at}.limits.{j}", 400), limit=True)
    c = p.composition
    if c is not None:
        w.section("The composition, chosen on validation")
        metric = "recall at the review budget" if c.metric == "recall_at_budget" else "average precision"
        w.line(f"Chosen: {w.q('programme.composition.chosen', 80, code=True)}. Rule: {w.q('programme.composition.rule', 300)}.")
        judged = ", ".join(f"{w.key(f'programme.composition.candidates.{k}')} {n(f'programme.composition.candidates.{k}')}" for k in c.candidates)
        w.line(f"Judged on the validation rows ({metric}, {n('programme.composition.positives', 'int')} confirmed cases): {judged}.")
        w.line(f"The best leaf alone is {w.q('programme.composition.best_leaf', 20, code=True)} ({n('programme.composition.best_leaf_score')}): the composition must beat it.")
        for j in range(len(c.warnings)):
            w.line(w.q(f"programme.composition.warnings.{j}", 300))
    h = p.holdout
    if h is not None:
        e = "programme.holdout"
        w.section("The holdout, read once")
        if p.reads_before:
            w.line(f"An earlier programme on this split already read the holdout {n('programme.reads_before', 'int')} time(s): this is not its first look, and the score is optimistic.")
        w.line(f"The holdout was read once: {n(f'{e}.rows', 'int')} rows, {n(f'{e}.positives', 'int')} confirmed cases, a budget of {n(f'{e}.budget_rows', 'int')} rows.")
        w.line(f"Composed ({w.q(e + '.chosen', 80, code=True)}): {n(e + '.composed.value')}, interval {n(e + '.composed.low')} to {n(e + '.composed.high')}.")
        w.line(f"Best leaf ({w.q(e + '.best_leaf', 20, code=True)}, chosen on validation): {n(e + '.best_leaf_estimate.value')}, interval "
               f"{n(e + '.best_leaf_estimate.low')} to {n(e + '.best_leaf_estimate.high')}.")
        gain = ("above zero." if h.beats_best_leaf else "not above zero: the gain is not shown.")
        w.line(f"Composed minus best leaf: {n(e + '.difference.value', 'signed')}, interval {n(e + '.difference.low', 'signed')} to {n(e + '.difference.high', 'signed')}: {gain}")
        w.line("Every leaf: " + ", ".join(f"{w.key(f'{e}.leaves.{k}')} {n(f'{e}.leaves.{k}.value')}" for k in h.leaves) + ".")
        for j in range(len(h.limits)):
            w.line(w.q(f"{e}.limits.{j}", 400), limit=True)
    if p.blockers:
        w.section("What blocks the programme")
        for j in range(len(p.blockers)):
            w.line(w.q(f"programme.blockers.{j}", 400))
    return w.finish(title or "The programme: parts, composition and the holdout read once")


def programme_context(p: Programme) -> dict[str, Any]:
    """What a programme report cites: ``{"programme": the document}``."""
    return {"programme": json.loads(p.model_dump_json())}


def markdown(p: Programme) -> str:
    return "\n\n".join(f"## {s['title']}\n\n" + "\n".join(f"- {line}" for line in s["lines"]) for s in sections(p))


__all__ = ["State", "Status", "LeafRun", "Programme", "Execution", "HoldoutClosed", "plan", "replan", "validate", "open_issues", "drop_leaf", "build_leaves", "compose_leaves",
           "read_holdout", "execute", "resume", "cluster_series", "sections", "markdown", "report", "programme_context", "fingerprint"]
