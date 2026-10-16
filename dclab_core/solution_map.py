"""The solution map (package 24.4): the problem map answered, block by block, with one holdout for the whole.

For each sub-problem of a problem map the solution map has a block: the approach the router chose, the solution workflow (20.6) of that leaf, and
the split the leaf uses. The composition's steps follow the problem map's (a sequence, a fusion of scores on a key, a gate, a budget) and end in
one step that scores the composed result on the holdout. The holdout is designed once for the whole map: every leaf splits on the same keys and the
same moment, no leaf reads the holdout on its own (each hands its predictions to the composition, ``final == "deferred"``), the composition's last
step reads it once, and nothing is chosen on it (the fusion's weights, a gate's threshold and the budget's cut are chosen on validation, before).

``validate`` refuses a map whose leaves split differently, whose leaf scores the holdout by itself, whose composition reads the holdout before its last
step or twice, or whose workflows do not hold. ``solution_map`` builds the map from a problem map, its routes and the data card, with the rules.
Deterministic and pure: no model, no store.
"""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal, Mapping, Optional

from pydantic import Field, model_validator

from dclab_core import solution_workflow as workflow
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.solution_workflow import SolutionWorkflow

if TYPE_CHECKING:  # the problem map is imported where it is used: either module may be imported first
    from dclab_core.problem_map import ProblemMap

Strategy = Literal["stratified", "time", "group", "time_group"]
StepKind = Literal["sequence", "fusion", "gate", "budget", "score"]
MAX_BLOCKS = 40
HOLDOUT_STEP = "holdout_score"


class SplitDesign(Contract):
    """How rows are split, pinned: by which keys (an entity never on both sides inside the training folds) and which moment (the latest period is the
    holdout), the share the holdout takes and the seed that draws what is random. One design is shared by every leaf."""

    strategy: Strategy
    keys: list[str] = Field(default_factory=list, max_length=4, description="The entity columns whose rows stay on one side.")
    moment: Optional[str] = Field(default=None, max_length=200, description="The time column the holdout follows, or None.")
    share: float = Field(default=0.2, gt=0, lt=0.5, description="The share of rows (the latest, or a stratified draw) that is the holdout.")
    seed: int = Field(default=0, ge=0, description="Draws what the split leaves to chance, the same for every leaf.")

    @model_validator(mode="after")
    def _coherent(self) -> "SplitDesign":
        wants_time, wants_keys = self.strategy in ("time", "time_group"), self.strategy in ("group", "time_group")
        if wants_time != bool(self.moment):
            raise ValueError(f"a {self.strategy} split {'needs' if wants_time else 'has no'} moment")
        if wants_keys != bool(self.keys):
            raise ValueError(f"a {self.strategy} split {'needs' if wants_keys else 'has no'} entity keys")
        return self


class Block(Contract):
    id: str = Field(pattern=r"^blk\d+$")
    part: str = Field(pattern=r"^part\d+$")
    sub_problem: str = Field(pattern=r"^sp\d+$")
    title: str = Field(min_length=1, max_length=200)
    family: str = Field(max_length=40)
    approach: Optional[str] = Field(default=None, max_length=40, description="The approach the router chose; None when none fits yet.")
    track: Optional[str] = Field(default=None, max_length=80)
    workflow: SolutionWorkflow
    split: SplitDesign
    final: Literal["own", "deferred"] = Field(description="own: the leaf scores the holdout itself (a single problem); deferred: it hands its predictions to the composition, which reads the holdout once.")
    defers: list[str] = Field(default_factory=list, max_length=4, description="The steps of the leaf's workflow that the composition runs in its place (the final score): the leaf never runs them.")

    @model_validator(mode="after")
    def _defers(self) -> "Block":
        if (self.final == "deferred") != (self.defers == ["final_once"]):
            raise ValueError("a deferred leaf defers its final score (final_once) and only that; an own leaf defers nothing")
        return self


class SolutionStep(Contract):
    kind: StepKind
    inputs: list[str] = Field(default_factory=list, max_length=MAX_BLOCKS)
    output: str = Field(min_length=1, max_length=60)
    joins: Optional[str] = Field(default=None, max_length=600)
    note: str = Field(default="", max_length=600)
    reads_holdout: bool = False


class HoldoutDesign(Contract):
    split: SplitDesign
    read_by: Literal["leaf", "composition"]
    step: Optional[str] = Field(default=None, max_length=60, description="The output name of the one step that reads it, for a composition.")
    chosen_on: Literal["validation"] = "validation"


class SolutionMap(Versioned):
    """A problem map answered: a block per sub-problem, the composition's steps, and the one holdout design."""

    schema_version: Literal["dclab_core.solution_map.v1"] = _version("solution_map")
    map: str = Field(max_length=64, description="The fingerprint of the problem map it answers.")
    blocks: list[Block] = Field(min_length=1, max_length=MAX_BLOCKS)
    steps: list[SolutionStep] = Field(default_factory=list, max_length=MAX_BLOCKS)
    holdout: HoldoutDesign
    blockers: list[str] = Field(default_factory=list, max_length=MAX_BLOCKS, description="The sub-problems with no approach yet, with why.")
    notes: list[str] = Field(default_factory=list, max_length=10, description="What the split or the build did that the person should know (a column left out, a combined split).")
    source: Literal["rules", "model", "person"] = "rules"
    fingerprint: str = Field(default="", max_length=64)

    @model_validator(mode="after")
    def _fingerprinted(self) -> "SolutionMap":
        expected = fingerprint(self)
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("fingerprint does not match the solution map: a map is never edited in place; build a new one")
        object.__setattr__(self, "fingerprint", expected)
        self.__pydantic_fields_set__.add("fingerprint")
        return self


CONTRACTS["solution_map"] = SolutionMap


def fingerprint(sm: SolutionMap) -> str:
    """The hash of what the map decides: the problem map it answers, each block's part, approach, split, deferral and every step of its workflow (content, not
    only its name), the steps in order with their notes, the holdout, what blocks it and what was noted."""
    body = {"map": sm.map,
            "blocks": [(b.id, b.part, b.sub_problem, b.title, b.family, b.approach, b.track, b.split.model_dump(), b.final, b.defers,
                        [s.model_dump() for s in b.workflow.steps]) for b in sm.blocks],
            "steps": [s.model_dump() for s in sm.steps], "holdout": sm.holdout.model_dump(), "blockers": sm.blockers, "notes": sm.notes, "source": sm.source}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- the rules' split

def _get(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def split_for(data: Any = None, notes: Optional[list[str]] = None) -> SplitDesign:
    """The one split of the whole map from what the data card names: its entity columns keep their rows on one side (the first four), its first time column is
    the moment (the latest period is the holdout). With neither, a stratified random split, said. What was left out is appended to ``notes``."""
    keys, moments = [], []
    for t in _get(data, "tables") or []:
        if _get(t, "role", "main") != "main":
            continue
        for c in _get(t, "columns") or []:
            if _get(c, "purpose") == "entity" and _get(c, "name") and str(_get(c, "name")) not in keys:
                keys.append(str(_get(c, "name"))[:200])
            if _get(c, "purpose") == "time" and _get(c, "name") and str(_get(c, "name")) not in moments:
                moments.append(str(_get(c, "name"))[:200])
    if notes is not None:
        if len(keys) > 4:
            notes.append(f"the data card names {len(keys)} entity columns: the first four keep their rows on one side ({', '.join(keys[4:])} left out)")
        if len(moments) > 1:
            notes.append(f"the data card names {len(moments)} time columns: the holdout follows {moments[0]} ({', '.join(moments[1:])} left out)")
    keys, moment = keys[:4], (moments[0] if moments else None)
    strategy: Strategy = "time_group" if keys and moment else "group" if keys else "time" if moment else "stratified"
    return SplitDesign(strategy=strategy, keys=keys, moment=moment)


# ---------------------------------------------------------------------------------------------------------------- building

def solution_map(m: ProblemMap, routes: Mapping[str, Any], data: Any = None, findings: Any = None) -> SolutionMap:
    """The solution map for a problem map: a block per sub-problem (its approach from ``routes``, its workflow from the problem and the data card),
    the problem map's composition steps and a last step that scores the composed result on the holdout, once. A single problem keeps its own final
    score. ``routes``: ``solutions.Route`` objects (or their JSON) by sub-problem id."""
    from dclab_core.solutions import BY_ID, Route

    notes: list[str] = []
    split = split_for(data, notes)
    if split.strategy == "time_group":
        notes.append("a split by time and by entity together: the holdout is the latest period, and inside the training folds an entity stays on one side; the main product has no "
                     "verified combined holdout (its planner answers 'unsupported'), so a person decides")
    complex_ = m.size == "complex"
    blocks, blockers = [], []
    for i, sub in enumerate(m.sub_problems, start=1):
        route = routes.get(sub.id)
        route = route if route is None or isinstance(route, Route) else Route.model_validate(route)
        approach = route.chosen if route else None
        if approach is None:
            blockers.append(f"{sub.id} ({sub.title}): " + (route.reason if route and route.reason else "no approach was routed"))
        card = {"family": sub.family, "labels": sub.labels}
        leaf = workflow.solution_workflow(card, data, findings)
        if any(st.id == "relock_holdout" for st in leaf.steps):
            blockers.append(f"{sub.id} ({sub.title}): the analysis found a split the data card did not name, so its workflow locks the holdout again; one holdout is locked once for the "
                            "whole map: name that split (an entity or a time column) and draw the map again")
        blocks.append(Block(id=f"blk{i}", part=sub.part, sub_problem=sub.id, title=sub.title, family=sub.family, approach=approach,
                            track=BY_ID[approach].evidence_track if approach in BY_ID else None, workflow=leaf, split=split,
                            final="deferred" if complex_ else "own", defers=["final_once"] if complex_ else []))
    steps: list[SolutionStep] = []
    if complex_:
        steps = [SolutionStep(kind=c.kind, inputs=list(c.inputs), output=c.output, joins=c.joins, note=c.note) for c in m.composition.steps]
        last = steps[-1].output if steps else blocks[-1].sub_problem
        steps.append(SolutionStep(kind="score", inputs=[last], output=HOLDOUT_STEP, reads_holdout=True,
                                  note="The composed result is scored once on the holdout the whole map shares; nothing is chosen on it."))
    holdout = HoldoutDesign(split=split, read_by="composition" if complex_ else "leaf", step=HOLDOUT_STEP if complex_ else None)
    return SolutionMap(map=m.fingerprint, blocks=blocks, steps=steps, holdout=holdout, blockers=blockers[:MAX_BLOCKS], notes=notes[:10])


# ---------------------------------------------------------------------------------------------------------------- validation

def _split_text(d: SplitDesign) -> str:
    return f"{d.strategy} on {', '.join(d.keys) or 'no key'}{' following ' + d.moment if d.moment else ''}"


def validate(sm: SolutionMap, m: Optional["ProblemMap"] = None) -> list[str]:
    """What is wrong with a solution map, in words (empty: sound). ``m``: the problem map it answers, to check the blocks and the steps against it."""
    out: list[str] = []
    if sm.fingerprint != fingerprint(sm):
        out.append("the fingerprint is stale: the solution map was changed after it was made; build a new one")
    complex_ = len(sm.blocks) > 1 or sm.holdout.read_by == "composition"
    ids = [b.id for b in sm.blocks]
    if len(set(ids)) != len(ids):
        out.append("two blocks have the same id")
    subs = [b.sub_problem for b in sm.blocks]
    if len(set(subs)) != len(subs):
        out.append("two blocks answer the same sub-problem")
    leaf_readers: list[str] = []
    for b in sm.blocks:
        if b.split != sm.holdout.split:
            out.append(f"{b.id} ({b.title}) splits {_split_text(b.split)}, but the map's holdout is {_split_text(sm.holdout.split)}: every leaf splits as the whole does")
        for problem in workflow.validate(b.workflow):
            out.append(f"{b.id}'s workflow: {problem}")
        if b.workflow.family != b.family:
            out.append(f"{b.id} is {b.family} but its workflow is for {b.workflow.family}")
        ids_of_steps = {st.id for st in b.workflow.steps}
        for key in sm.holdout.split.keys[:3]:  # the workflow groups the first three entities of the card
            if not any(st.id.startswith("group_folds_") and key in st.columns for st in b.workflow.steps):
                out.append(f"{b.id}'s workflow does not group the folds by {key}, but the map's split does")
        if sm.holdout.split.moment and "time_split" not in ids_of_steps:
            out.append(f"{b.id}'s workflow does not split by time, but the map's split follows {sm.holdout.split.moment}")
        if "relock_holdout" in ids_of_steps:
            out.append(f"{b.id}'s workflow locks the holdout again on its own: one holdout is locked once for the whole map")
        leaf_readers += [f"{b.id}:{st.id}" for st in b.workflow.steps if "holdout" in st.inputs and st.id not in b.defers]
    if complex_:
        for b in sm.blocks:
            if b.final != "deferred":
                out.append(f"{b.id} ({b.title}) scores the holdout on its own: in a map of several parts only the composition reads it, once")
        if sm.holdout.read_by != "composition":
            out.append("the holdout of a map of several parts is read by the composition")
        if leaf_readers:
            out.append(f"the holdout would be read by the leaves too ({', '.join(leaf_readers[:4])}): their final scores are the composition's to run")
        readers = [i for i, st in enumerate(sm.steps) if st.reads_holdout]
        if len(readers) != 1:
            out.append(f"the holdout must be read by exactly one composition step (read by {len(readers)})")
        elif readers[0] != len(sm.steps) - 1:
            out.append(f"the holdout is read at step {readers[0] + 1} of {len(sm.steps)}: a step after it chooses or combines on what the holdout showed")
        elif sm.steps[readers[0]].kind != "score" or sm.steps[readers[0]].output != sm.holdout.step:
            out.append("the step that reads the holdout is the map's score and nothing else")
        out += _steps_issues(sm, m)
    else:
        if sm.steps:
            out.append("a single problem has no composition steps")
        if any(b.final != "own" for b in sm.blocks) or sm.holdout.read_by != "leaf":
            out.append("a single problem scores its own holdout once")
        if len(leaf_readers) != 1 and any(b.final == "own" for b in sm.blocks):
            out.append(f"a single problem's holdout is read by its final score alone (read by {len(leaf_readers)} steps)")
    if m is not None:
        if sm.map != m.fingerprint:
            out.append("this solution map answers another problem map")
        wanted = {s.id for s in m.sub_problems}
        if set(subs) != wanted:
            out.append(f"the blocks answer {', '.join(sorted(subs))} but the problem map has {', '.join(sorted(wanted))}")
        for b in sm.blocks:
            sub = next((s for s in m.sub_problems if s.id == b.sub_problem), None)
            if sub is not None and (sub.family != b.family or sub.part != b.part):
                out.append(f"{b.id} does not match {sub.id}'s family or part")
    return list(dict.fromkeys(out))[:30]


def _steps_issues(sm: SolutionMap, m: Optional["ProblemMap"]) -> list[str]:
    """The composition's steps: built in order from blocks and earlier outputs, each output consumed, the score taking the last result and nothing else, and the
    steps those of the problem map."""
    out: list[str] = []
    subs = {b.sub_problem for b in sm.blocks}
    made: dict[str, int] = {}
    for i, st in enumerate(sm.steps, start=1):
        if not st.inputs:
            out.append(f"step {i} ({st.kind} → {st.output}) has no inputs")
        if len(st.inputs) != len(set(st.inputs)):
            out.append(f"step {i} lists an input twice")
        if st.output in subs:
            out.append(f"step {i} makes '{st.output}', which is a sub-problem's id")
        elif st.output in made:
            out.append(f"step {i} makes '{st.output}', which step {made[st.output]} already made")
        if st.kind in ("fusion", "gate") and len(st.inputs) < 2:
            out.append(f"a {st.kind} combines at least two inputs: step {i} has {len(st.inputs)}")
        for name in st.inputs:
            if name not in subs and name not in made:
                out.append(f"step {i} ({st.kind} → {st.output}) uses '{name}', which no block or earlier step makes")
        made.setdefault(st.output, i)
    composed = [st for st in sm.steps if st.kind != "score" or not st.reads_holdout]
    if sm.steps and sm.steps[-1].reads_holdout:
        score = sm.steps[-1]
        last = composed[-1].output if composed else None
        if last is not None and score.inputs != [last]:
            out.append(f"the score takes {', '.join(score.inputs) or 'nothing'}, but the composed result is '{last}': the system that is scored is the one the composition makes")
        for st in composed[:-1]:  # every result but the last is used by a later step
            if not any(st.output in later.inputs for later in sm.steps[sm.steps.index(st) + 1:]):
                out.append(f"'{st.output}' (step {sm.steps.index(st) + 1}) is made and never used: it is left out of what is scored")
    if m is not None:
        wanted = [(c.kind, list(c.inputs), c.output, c.joins) for c in m.composition.steps]
        have = [(st.kind, list(st.inputs), st.output, st.joins) for st in composed]
        if wanted != have:
            out.append("the composition's steps are not the problem map's: the solution map composes what the problem map says, in its order")
    return out


__all__ = ["SolutionMap", "Block", "SolutionStep", "HoldoutDesign", "SplitDesign", "solution_map", "validate", "split_for", "fingerprint", "HOLDOUT_STEP"]
import dclab_core.programme  # noqa: E402,F401  (the programme contract, 24.5: registered once the solution map is defined; either module may be imported first)
