"""The expert's record (package 25.3): how an AI-first agent, asked to act as a machine-learning expert of twenty years,
understands a problem and its data, explores it, prepares it, decides what kind of problem it is and which algorithms
fit, and draws the problem's workflow and the solution's.

An :class:`ExpertTrace` is one version of that record, in typed sections, so each can be set beside what the system
records for the same thing:

==================  =====================================================  ==========================================
section             what the expert says                                   the system's own record
==================  =====================================================  ==========================================
``problem``         the problem restated, the decision it serves, the      the problem card (20.1)
                    unit, the moment, the outcome and how it is known
``problem_type``    family, kind, task, labels, why, the alternatives      the card's family, the map's families
``data``            each table's grain, the columns' roles, identifiers,   the data card (19.11)
                    times, groups, the leak suspects with the test
``eda``             each question asked of the data, its aggregate         the data findings (dclab_core.analyse)
                    answer, the conclusion
``preprocessing``   each step, why, whether it is fitted inside the folds  the engine's recipe
``problem_workflow`` the parts, their families, how they compose            the problem map (24.1)
``algorithms``      the families ranked with why, the baseline, what       the router (24.3), the models stage
                    would change the choice
``evaluation``      split, metric, the holdout's one read, the threshold   the solution map (24.4), the data stage
``solution_workflow`` steps on WF-01…WF-10 (tool, stage, why, evidence,     the solution workflow (20.6), the graph
                    rule, dependencies) and the routes back
==================  =====================================================  ==========================================

:func:`read` builds a trace from a model's answer leniently (a missing or misshapen section is a problem kept on the
trace, never a refusal: a partial record is still compared); :func:`workflow_problems` checks the solution workflow
on its content with the principles of :mod:`dclab_core.solution_workflow` (the holdout read once, at the end, after
the leakage audit; nothing chosen before it), plus ids, dependencies and routes; :func:`intents` gives the trace's
decisions keyed by the ledger's points (``point`` → ``subject`` → value), so they sit beside the rule's on the same
records; :func:`twin_steps` gives the twin steps (25.1) of a trace against the system's answers. No model, no store.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any, Literal, Optional

from typing import Annotated

from pydantic import Field

from dclab_core.bands import FAMILIES, LABEL_SITUATIONS
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

WF_IDS = tuple(f"WF-{i:02d}" for i in range(1, 11))
TASK_WORDS = ("binary", "multiclass", "regression", "ranking", "forecast", "survival", "anomaly", "clustering", "other")
SPLITS = ("stratified", "time", "group", "random", "entity_time", "other")
STAGES = ("data", "leakage", "features", "models", "final")
SHORT, TEXT, LONG = 120, 400, 1200
MAX = 40
Name = Annotated[str, Field(max_length=SHORT)]
Line = Annotated[str, Field(max_length=TEXT)]
LIST_ITEMS = 60  # dclab_core.twin's bound on a list inside an answer
EXCLUDE = ("exclude", "drop", "remove", "forbid", "forbidden", "block", "blocked", "ban")
REVIEW = ("review", "check", "flag", "investigate", "test", "suspect")
KEEP = ("keep", "clear", "allow", "use")


class _C(Contract):
    pass


class Problem(_C):
    restatement: Optional[str] = Field(default=None, max_length=LONG)
    decision: Optional[str] = Field(default=None, max_length=TEXT, description="The decision the prediction serves.")
    unit: Optional[str] = Field(default=None, max_length=SHORT)
    moment: Optional[str] = Field(default=None, max_length=TEXT, description="When the prediction is made and what is known then.")
    outcome: Optional[str] = Field(default=None, max_length=TEXT)
    outcome_known: Optional[str] = Field(default=None, max_length=TEXT, description="How and when the outcome becomes known.")
    target: Optional[str] = Field(default=None, max_length=SHORT, description="The column holding the outcome, if one does.")
    success: Optional[str] = Field(default=None, max_length=TEXT)
    constraints: list[Line] = Field(default_factory=list, max_length=12)


class Alternative(_C):
    family: str = Field(max_length=SHORT)
    why_not: Optional[str] = Field(default=None, max_length=TEXT)


class ProblemType(_C):
    family: Optional[str] = Field(default=None, max_length=SHORT, description=f"One of {', '.join(FAMILIES)} where it fits.")
    kind: Optional[str] = Field(default=None, max_length=SHORT, description="The kind within the family (binary, multiclass, a horizon, …).")
    task: Optional[str] = Field(default=None, max_length=SHORT)
    labels: Optional[str] = Field(default=None, max_length=SHORT, description=f"One of {', '.join(LABEL_SITUATIONS)}.")
    why: Optional[str] = Field(default=None, max_length=LONG)
    alternatives: list[Alternative] = Field(default_factory=list, max_length=8)


class ColumnRead(_C):
    name: str = Field(max_length=SHORT)
    role: Optional[str] = Field(default=None, max_length=SHORT, description="feature, target, identifier, time, group, leak, text, constant, …")
    why: Optional[str] = Field(default=None, max_length=TEXT)


class LeakSuspect(_C):
    column: str = Field(max_length=SHORT)
    why: Optional[str] = Field(default=None, max_length=TEXT)
    test: Optional[str] = Field(default=None, max_length=TEXT)
    action: Optional[str] = Field(default=None, max_length=SHORT, description="exclude, review or keep")


class TableRead(_C):
    name: Optional[str] = Field(default=None, max_length=SHORT)
    grain: Optional[str] = Field(default=None, max_length=TEXT, description="What one row is.")
    columns: list[ColumnRead] = Field(default_factory=list, max_length=MAX * 3)
    identifiers: list[Name] = Field(default_factory=list, max_length=MAX)
    times: list[Name] = Field(default_factory=list, max_length=MAX)
    groups: list[Name] = Field(default_factory=list, max_length=MAX)
    leak_suspects: list[LeakSuspect] = Field(default_factory=list, max_length=MAX)
    quality: list[Line] = Field(default_factory=list, max_length=12)


class EdaItem(_C):
    question: str = Field(max_length=TEXT)
    tool: Optional[str] = Field(default=None, max_length=SHORT)
    answer: Optional[str] = Field(default=None, max_length=TEXT, description="The aggregate answer (counts, rates, ranges); never a row.")
    conclusion: Optional[str] = Field(default=None, max_length=TEXT)


class PrepStep(_C):
    step: str = Field(max_length=TEXT)
    columns: list[Name] = Field(default_factory=list, max_length=MAX)
    why: Optional[str] = Field(default=None, max_length=TEXT)
    inside_folds: Optional[bool] = Field(default=None, description="Fitted inside the training folds (it learns from data) or not.")


class Part(_C):
    title: str = Field(max_length=SHORT)
    family: Optional[str] = Field(default=None, max_length=SHORT)
    unit: Optional[str] = Field(default=None, max_length=SHORT)
    moment: Optional[str] = Field(default=None, max_length=TEXT)
    reads: list[Name] = Field(default_factory=list, max_length=8, description="The parts whose outputs it reads.")


class ProblemWorkflow(_C):
    size: Optional[Literal["single", "complex"]] = None
    parts: list[Part] = Field(default_factory=list, max_length=12)
    composition: Optional[str] = Field(default=None, max_length=TEXT)


class Algorithm(_C):
    family: str = Field(max_length=SHORT)
    rank: int = Field(default=0, ge=0, le=50)
    why: Optional[str] = Field(default=None, max_length=TEXT)


class Algorithms(_C):
    ranked: list[Algorithm] = Field(default_factory=list, max_length=12)
    baseline: Optional[str] = Field(default=None, max_length=TEXT)
    would_change: Optional[str] = Field(default=None, max_length=TEXT, description="What evidence would change the choice.")


class Evaluation(_C):
    split: Optional[str] = Field(default=None, max_length=SHORT)
    split_column: Optional[str] = Field(default=None, max_length=SHORT)
    split_why: Optional[str] = Field(default=None, max_length=TEXT)
    metric: Optional[str] = Field(default=None, max_length=SHORT)
    baseline: Optional[str] = Field(default=None, max_length=TEXT)
    holdout: Optional[str] = Field(default=None, max_length=TEXT)
    threshold: Optional[str] = Field(default=None, max_length=TEXT)


class WorkflowStep(_C):
    id: str = Field(max_length=12)
    wf: Optional[str] = Field(default=None, max_length=8)
    tool: Optional[str] = Field(default=None, max_length=SHORT)
    stage: Optional[str] = Field(default=None, max_length=SHORT)
    does: Optional[str] = Field(default=None, max_length=TEXT)
    why: Optional[str] = Field(default=None, max_length=TEXT)
    evidence: Optional[str] = Field(default=None, max_length=TEXT)
    decision_rule: Optional[str] = Field(default=None, max_length=TEXT)
    after: list[Annotated[str, Field(max_length=12)]] = Field(default_factory=list, max_length=12)


class Route(_C):
    from_step: str = Field(max_length=12)
    to_step: str = Field(max_length=12)
    when: Optional[str] = Field(default=None, max_length=TEXT)


class Change(_C):
    field: str = Field(max_length=SHORT)
    before: Optional[str] = Field(default=None, max_length=TEXT)
    after: Optional[str] = Field(default=None, max_length=TEXT)
    because: Optional[str] = Field(default=None, max_length=TEXT)
    evidence: Optional[str] = Field(default=None, max_length=TEXT, description="The number or finding that made the change.")


class Retrospective(_C):
    what_worked: list[Line] = Field(default_factory=list, max_length=12)
    what_changed: list[Line] = Field(default_factory=list, max_length=12)
    would_do_differently: list[Line] = Field(default_factory=list, max_length=12)


class ExpertTrace(Versioned):
    """One version of the expert's record: the first one before anything runs, then one per checkpoint."""

    schema_version: Literal["dclab_core.expert_trace.v1"] = _version("expert_trace")
    version: int = Field(default=1, ge=1, le=99)
    checkpoint: str = Field(default="initial", max_length=40)
    model: Optional[str] = Field(default=None, max_length=SHORT)
    problem: Problem = Field(default_factory=Problem)
    problem_type: ProblemType = Field(default_factory=ProblemType)
    data: list[TableRead] = Field(default_factory=list, max_length=12)
    eda: list[EdaItem] = Field(default_factory=list, max_length=MAX)
    preprocessing: list[PrepStep] = Field(default_factory=list, max_length=MAX)
    problem_workflow: ProblemWorkflow = Field(default_factory=ProblemWorkflow)
    algorithms: Algorithms = Field(default_factory=Algorithms)
    evaluation: Evaluation = Field(default_factory=Evaluation)
    solution_workflow: list[WorkflowStep] = Field(default_factory=list, max_length=MAX)
    routes: list[Route] = Field(default_factory=list, max_length=20)
    risks: list[Line] = Field(default_factory=list, max_length=12)
    questions: list[Line] = Field(default_factory=list, max_length=12, description="What the expert would ask the person.")
    confidence: Optional[Literal["low", "medium", "high"]] = None
    changes: list[Change] = Field(default_factory=list, max_length=MAX)
    retrospective: Retrospective = Field(default_factory=lambda: Retrospective(), description="Only at the end of a run.")
    problems: list[Line] = Field(default_factory=list, max_length=60, description="What was missing or misshapen in the answer it was read from.")


SECTIONS = ("problem", "problem_type", "data", "eda", "preprocessing", "problem_workflow", "algorithms", "evaluation", "solution_workflow",
            "routes", "risks", "questions", "confidence", "changes")


# ------------------------------------------------------------------------------------------------- reading an answer


def _s(value: Any, n: int) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (list, dict)):
        value = json.dumps(value, ensure_ascii=False)
    text = str(value).strip()
    return text[:n] if text else None


def _list(value: Any) -> list[Any]:
    if value is None:
        return []
    return list(value) if isinstance(value, list) else [value]


def _strs(value: Any, n: int = SHORT, limit: int = MAX) -> list[str]:
    return [t for t in (_s(v, n) for v in _list(value)) if t][:limit]


def _obj(value: Any, name: str, problems: list[str]) -> dict[str, Any]:
    if value is None:
        problems.append(f"missing: {name}")
        return {}
    if not isinstance(value, Mapping):
        problems.append(f"{name} is not an object")
        return {}
    return dict(value)


def _items(value: Any, name: str, problems: list[str], limit: int) -> list[dict[str, Any]]:
    if value is None:
        problems.append(f"missing: {name}")
        return []
    items = _list(value)
    out = [dict(v) for v in items if isinstance(v, Mapping)]
    if len(out) < len(items):
        problems.append(f"{name}: {len(items) - len(out)} item(s) are not objects")
    if len(out) > limit:
        problems.append(f"{name}: {len(out)} items, kept {limit}")
    return out[:limit]


def _rank(value: Any, default: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return default
    try:
        n = int(float(value))
    except (ValueError, OverflowError):  # "inf", Infinity, NaN
        return default
    return n if 0 <= n <= 50 else default


def extract(text: Optional[str]) -> tuple[Optional[dict[str, Any]], list[str]]:
    """The first JSON object in a model's answer (fenced or not)."""
    if not text or "{" not in text:
        return None, ["no JSON object in the answer"]
    body = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", body, re.S)
    body = fence.group(1) if fence else body[body.index("{"):body.rindex("}") + 1] if "}" in body else body
    try:
        out = json.loads(body)
    except json.JSONDecodeError as exc:
        return None, [f"not valid JSON: {exc.msg} at {exc.pos}"]
    return (out, []) if isinstance(out, dict) else (None, ["the JSON is not an object"])


def read(answer: Mapping[str, Any], *, version: int = 1, checkpoint: str = "initial", model: Optional[str] = None,
         tools: Optional[Sequence[str]] = None) -> ExpertTrace:
    """A trace from a model's parsed answer. Nothing is refused: what is missing or misshapen is in ``problems``."""
    problems: list[str] = []
    p = _obj(answer.get("problem"), "problem", problems)
    t = _obj(answer.get("problem_type"), "problem_type", problems)
    family = _s(t.get("family"), SHORT)
    if family and family.casefold() not in FAMILIES:
        problems.append(f"problem_type.family {family!r} is not one of the families")
    labels = _s(t.get("labels"), SHORT)
    if labels and labels.casefold() not in LABEL_SITUATIONS:
        problems.append(f"problem_type.labels {labels!r} is not one of {', '.join(LABEL_SITUATIONS)}")
    tables = []
    for d in _items(answer.get("data"), "data", problems, 12):
        columns = [ColumnRead(name=_s(c.get("name"), SHORT) or "?", role=(_s(c.get("role"), SHORT) or "").casefold() or None, why=_s(c.get("why"), TEXT))
                   for c in _items(d.get("columns", []), "data.columns", problems, MAX * 3)]
        lists = {"identifiers": _strs(d.get("identifiers")), "times": _strs(d.get("times")), "groups": _strs(d.get("groups"))}
        for c in columns:  # a column given a role in the column list belongs in that role's list too
            name = {"identifier": "identifiers", "id": "identifiers", "time": "times", "group": "groups"}.get(c.role or "")
            if name and c.name not in lists[name] and len(lists[name]) < MAX:
                lists[name].append(c.name)
        suspects = []
        for x in _items(d.get("leak_suspects", []), "data.leak_suspects", problems, MAX):
            column, action = _s(x.get("column"), SHORT) or "?", (_s(x.get("action"), SHORT) or "review").casefold()
            if action in EXCLUDE:
                action = "exclude"
            elif action in REVIEW:
                action = "review"
            elif action in KEEP:
                action = "keep"
            else:
                problems.append(f"data.leak_suspects {column}: action {action!r} is not exclude, review or keep (read as review)")
                action = "review"
            suspects.append(LeakSuspect(column=column, why=_s(x.get("why"), TEXT), test=_s(x.get("test"), TEXT), action=action))
        named = {x.column for x in suspects}
        for c in columns:  # a column marked a leak with no suspect entry is an excluded suspect
            if c.role == "leak" and c.name not in named and len(suspects) < MAX:
                suspects.append(LeakSuspect(column=c.name, why=c.why, action="exclude"))
        tables.append(TableRead(name=_s(d.get("name"), SHORT), grain=_s(d.get("grain"), TEXT), columns=columns, leak_suspects=suspects,
                                quality=_strs(d.get("quality"), TEXT, 12), **lists))
    pw = _obj(answer.get("problem_workflow"), "problem_workflow", problems)
    size = _s(pw.get("size"), 10)
    if size not in (None, "single", "complex"):
        problems.append(f"problem_workflow.size {size!r} is not single or complex")
        size = None
    al = _obj(answer.get("algorithms"), "algorithms", problems)
    ev = _obj(answer.get("evaluation"), "evaluation", problems)
    split = _s(ev.get("split"), SHORT)
    if split and split.casefold() not in SPLITS:
        problems.append(f"evaluation.split {split!r} is not one of {', '.join(SPLITS)}")
    steps = []
    for i, s in enumerate(_items(answer.get("solution_workflow"), "solution_workflow", problems, MAX)):
        steps.append(WorkflowStep(id=_s(s.get("id"), 12) or f"?{i + 1}", wf=_s(s.get("wf"), 8), tool=_s(s.get("tool"), SHORT), stage=_s(s.get("stage"), SHORT),
                                  does=_s(s.get("does"), TEXT), why=_s(s.get("why"), TEXT), evidence=_s(s.get("evidence"), TEXT),
                                  decision_rule=_s(s.get("decision_rule"), TEXT), after=_strs(s.get("after"), 12, 12)))
    routes = [Route(from_step=_s(r.get("from"), 12) or "?", to_step=_s(r.get("to"), 12) or "?", when=_s(r.get("when"), TEXT))
              for r in _items(answer.get("routes", []), "routes", problems, 20)]
    confidence = _s(answer.get("confidence"), 10)
    if confidence not in (None, "low", "medium", "high"):
        problems.append(f"confidence {confidence!r} is not low, medium or high")
        confidence = None
    retro = answer.get("retrospective")
    trace = ExpertTrace(
        version=version, checkpoint=checkpoint[:40], model=_s(model, SHORT),
        problem=Problem(restatement=_s(p.get("restatement"), LONG), decision=_s(p.get("decision"), TEXT), unit=_s(p.get("unit"), SHORT),
                        moment=_s(p.get("moment"), TEXT), outcome=_s(p.get("outcome"), TEXT), outcome_known=_s(p.get("outcome_known"), TEXT),
                        target=_s(p.get("target"), SHORT), success=_s(p.get("success"), TEXT), constraints=_strs(p.get("constraints"), TEXT, 12)),
        problem_type=ProblemType(family=family.casefold() if family else None, kind=_s(t.get("kind"), SHORT), task=_s(t.get("task"), SHORT),
                                 labels=labels.casefold() if labels else None, why=_s(t.get("why"), LONG),
                                 alternatives=[Alternative(family=_s(a.get("family"), SHORT) or "?", why_not=_s(a.get("why_not"), TEXT))
                                               for a in _items(t.get("alternatives", []), "problem_type.alternatives", problems, 8)]),
        data=tables,
        eda=[EdaItem(question=_s(e.get("question"), TEXT) or "?", tool=_s(e.get("tool"), SHORT), answer=_s(e.get("answer"), TEXT), conclusion=_s(e.get("conclusion"), TEXT))
             for e in _items(answer.get("eda"), "eda", problems, MAX)],
        preprocessing=[PrepStep(step=_s(x.get("step"), TEXT) or "?", columns=_strs(x.get("columns")), why=_s(x.get("why"), TEXT),
                                inside_folds=x.get("inside_folds") if isinstance(x.get("inside_folds"), bool) else None)
                       for x in _items(answer.get("preprocessing"), "preprocessing", problems, MAX)],
        problem_workflow=ProblemWorkflow(size=size, composition=_s(pw.get("composition"), TEXT),
                                         parts=[Part(title=_s(x.get("title"), SHORT) or "?", family=_s(x.get("family"), SHORT), unit=_s(x.get("unit"), SHORT),
                                                     moment=_s(x.get("moment"), TEXT), reads=_strs(x.get("reads"), SHORT, 8))
                                                for x in _items(pw.get("parts", []), "problem_workflow.parts", problems, 12)]),
        algorithms=Algorithms(ranked=[Algorithm(family=_s(a.get("family"), SHORT) or "?", rank=_rank(a.get("rank"), i + 1), why=_s(a.get("why"), TEXT))
                                      for i, a in enumerate(_items(al.get("ranked", []), "algorithms.ranked", problems, 12))],
                              baseline=_s(al.get("baseline"), TEXT), would_change=_s(al.get("would_change"), TEXT)),
        evaluation=Evaluation(split=split.casefold() if split else None, split_column=_s(ev.get("split_column"), SHORT), split_why=_s(ev.get("split_why"), TEXT),
                              metric=_s(ev.get("metric"), SHORT), baseline=_s(ev.get("baseline"), TEXT), holdout=_s(ev.get("holdout"), TEXT),
                              threshold=_s(ev.get("threshold"), TEXT)),
        solution_workflow=steps, routes=routes, risks=_strs(answer.get("risks"), TEXT, 12), questions=_strs(answer.get("questions"), TEXT, 12),
        confidence=confidence,
        changes=[Change(field=_s(c.get("field"), SHORT) or "?", before=_s(c.get("from", c.get("before")), TEXT), after=_s(c.get("to", c.get("after")), TEXT),
                        because=_s(c.get("because"), TEXT), evidence=_s(c.get("evidence"), TEXT))
                 for c in _items(answer.get("changes", []), "changes", problems, MAX)],
        retrospective=Retrospective(**{k: _strs(v, TEXT, 12) for k, v in retro.items() if k in ("what_worked", "what_changed", "would_do_differently")})
        if isinstance(retro, Mapping) else Retrospective(),
        problems=[])
    problems += workflow_problems(trace, tools)
    return trace.model_copy(update={"problems": problems[:60]})


def workflow_problems(trace: ExpertTrace, tools: Optional[Sequence[str]] = None) -> list[str]:
    """What is wrong with the solution workflow on its content: ids unique, steps on WF-01…WF-10, tools that exist,
    dependencies and routes on steps that exist, and the principles of :mod:`dclab_core.solution_workflow`: the
    holdout (the final stage) is read once, after the leakage audit, and no model is chosen before that audit."""
    out: list[str] = []
    ids: list[str] = []
    for s in trace.solution_workflow:
        if s.id in ids:
            out.append(f"workflow step {s.id} appears twice")
        ids.append(s.id)
        if s.wf and s.wf not in WF_IDS:
            out.append(f"workflow step {s.id}: {s.wf!r} is not WF-01…WF-10")
        if tools is not None and s.tool and s.tool not in tools:
            out.append(f"workflow step {s.id}: tool {s.tool!r} does not exist")
        if not s.tool:
            out.append(f"workflow step {s.id} names no tool")
    known = set(ids)
    for s in trace.solution_workflow:
        out += [f"workflow step {s.id}: needs {a}, which is not a step" for a in s.after if a not in known]
    out += [f"route {r.from_step}→{r.to_step} names a step that is not in the workflow" for r in trace.routes
            if r.from_step not in known or r.to_step not in known]
    stages = [(i, (s.stage or "").casefold(), (s.tool or "").casefold()) for i, s in enumerate(trace.solution_workflow)]
    finals = [i for i, st, tool in stages if st == "final" and tool.startswith("run")]
    audits = [i for i, st, tool in stages if (st == "leakage" and tool.startswith("run")) or tool == "review_leakage"]
    if len(finals) > 1:
        out.append("the holdout is read more than once (the final stage runs twice)")
    if finals and audits and finals[0] < audits[0]:
        out.append("the holdout is read before the leakage audit")
    if finals and not audits:
        out.append("the holdout is read and no leakage audit is in the workflow")
    for stage in ("features", "models"):
        chosen = [i for i, st, tool in stages if st == stage and tool.startswith("approve")]
        if chosen and (not audits or chosen[0] < audits[0]):
            out.append(f"{'a feature recipe' if stage == 'features' else 'a model'} is chosen before the leakage audit")
    return out


# ------------------------------------------------------------------------------------------------- against the system


def intents(trace: Optional[ExpertTrace]) -> dict[str, dict[str, Any]]:
    """The trace's decisions keyed by the ledger's points: point → subject (a column, or ``-`` for the problem) → value,
    each value in its point's answer shape."""
    if trace is None:
        return {}
    out: dict[str, dict[str, Any]] = {}

    def put(point: str, subject: str, value: Any) -> None:
        if value not in (None, "", []):
            out.setdefault(point, {})[subject] = value

    put("target.column", "-", trace.problem.target)
    task = (trace.problem_type.task or trace.problem_type.kind or "").casefold()
    put("task.framing", "-", next((t for t in ("binary", "multiclass", "regression") if t in task), None))
    put("problem.family", "-", trace.problem_type.family if trace.problem_type.family in FAMILIES else None)
    put("labels.situation", "-", trace.problem_type.labels if trace.problem_type.labels in LABEL_SITUATIONS else None)
    put("split.strategy", "-", trace.evaluation.split)
    put("metric.primary", "-", (trace.evaluation.metric or "").casefold() or None)
    put("map.size", "-", trace.problem_workflow.size)
    ranked = sorted(trace.algorithms.ranked, key=lambda a: a.rank or 99)
    put("model.family", "-", ranked[0].family.casefold() if ranked else None)
    for table in trace.data:
        for c in table.identifiers:
            put("column.is_identifier", c, True)
        for c in table.times:
            put("column.time", c, True)
        for c in table.groups:
            put("column.group", c, True)
        for x in table.leak_suspects:
            action = (x.action or "review").casefold()
            put("feature.leakage_suspect", x.column, action in ("exclude", "review"))
    return out


def step_values(trace: ExpertTrace) -> dict[str, dict[str, Any]]:
    """The trace's answers in the shapes of the twin steps (dclab_core.twin.STEPS)."""
    tables = trace.data
    leak = sorted({x.column for t in tables for x in t.leak_suspects if (x.action or "review").casefold() == "exclude"})
    ranked = sorted(trace.algorithms.ranked, key=lambda a: a.rank or 99)
    task = (trace.problem_type.task or trace.problem_type.kind or "").casefold()
    out = {
        "home.understanding": {"outcome": trace.problem.outcome, "moment": trace.problem.moment, "unit": trace.problem.unit, "family": trace.problem_type.family,
                               "labels": trace.problem_type.labels, "target_column": trace.problem.target},
        "solution.columns": {"target": trace.problem.target, "task": next((t for t in ("binary", "multiclass", "regression") if t in task), None),
                             # an empty list is an answer ("none") once the expert read the data; with no table read it says nothing
                             **({"identifiers": sorted({c for t in tables for c in t.identifiers}), "leakage": leak,
                                 "time": sorted({c for t in tables for c in t.times}), "group": sorted({c for t in tables for c in t.groups})} if tables else {})},
        "stage.data": {"split": trace.evaluation.split, "metric": (trace.evaluation.metric or "").casefold() or None},
        "stage.models": {"family": ranked[0].family.casefold() if ranked else None},
        "home.map": {"size": trace.problem_workflow.size, "families": sorted({p.family.casefold() for p in trace.problem_workflow.parts if p.family}) or None},
        "home.workflow": {"steps": [s.wf for s in trace.solution_workflow if s.wf] or None},
    }
    def cut(v: Any) -> Any:  # a wide record (several tables of many columns) still fits a twin answer
        return v[:LIST_ITEMS] if isinstance(v, list) else v

    return {k: {f: cut(v) for f, v in d.items() if v is not None} for k, d in out.items()}


def twin_steps(trace: ExpertTrace, system: Mapping[str, Mapping[str, Any]], *, session: str, at: str, input: Any,
               answerer: str = "ai_first@expert", system_answerer: str = "rules", truth: Optional[Mapping[str, Mapping[str, Any]]] = None) -> list[Any]:
    """The trace against the system's answers (``system``: step → value in the step's shape), one twin step per step
    both hold, judged where ``truth`` (step → value) is given."""
    from dclab_core import twin

    mine = step_values(trace)
    out = []
    for name, value in mine.items():
        theirs = {k: v for k, v in dict(system.get(name) or {}).items() if k in twin.spec(name).fields}
        if not value and not theirs:
            continue
        system_status = "answered" if theirs else "skipped" if name not in system else "abstained"  # a step the system never makes is skipped
        planted = dict((truth or {}).get(name) or {})
        out.append(twin.step(f"{session}:{name}:v{trace.version}", session, name, at, input,
                             twin.ArmAnswer(arm="system", answerer=system_answerer, value=theirs, status=system_status,
                                            note="the system makes no such step here" if system_status == "skipped" else None),
                             twin.ArmAnswer(arm="ai_first", answerer=answerer, model=trace.model, value=value, status="answered" if value else "abstained",
                                            reasons=[r for r in (trace.problem_type.why, trace.evaluation.split_why) if r][:2]),
                             subject=f"v{trace.version}:{trace.checkpoint}",
                             truth=twin.Truth(value=planted, source="planted") if planted else None))
    return out


CONTRACTS["expert_trace"] = ExpertTrace

__all__ = ["ExpertTrace", "SECTIONS", "extract", "read", "workflow_problems", "intents", "step_values", "twin_steps", "WF_IDS"]
