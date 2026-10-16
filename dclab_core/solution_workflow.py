"""The solution workflow from the two cards (package 20.6): the steps that solve this problem, each one checkable.

``solution_workflow(problem_card, data_card, findings)`` builds the graph of steps for a problem: the safety steps every
solution keeps (lock the holdout first, analyse the training rows only, audit leakage before any model, choose on
validation, a person's gate before the holdout, score the holdout once, report the limits), the family's own steps,
and the steps the cards and the findings call for (a split by the data card's entity or time, the labels joined, the
leaks kept out, per-entity features, censoring kept, and, when the analysis finds a split the cards did not name, the
holdout locked again and the analysis read again). Each step says its purpose, its inputs and outputs (from a closed
list of artefacts), the decision it makes (a decision key of the plan, ``analyse.AFFECTS``), the check that proves it,
the rule it rests on and the findings it answers.

``validate(workflow)`` is the code that decides, on the steps' content and not their names: the safety and family steps
as their template says, the workflow blocks never going back, the holdout read by one step only, after the person's
gate, nothing chosen before the leakage audit or after the holdout is read, every finding answered by a step. A model
may propose another workflow (``propose``); it is used only when it validates, otherwise the current one stays and the
problems are said. Deterministic: no model, no storage.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

BLOCKS = tuple(f"WF-{i:02d}" for i in range(1, 11))
Artefact = Literal["table", "label file", "training rows", "holdout", "findings", "columns kept", "chosen model", "approval",
                   "holdout score", "report"]
StepKind = Literal["safety", "family", "card", "finding"]
FAMILIES = ("classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking", "segmentation",
            "uplift", "entity_tables", "streams", "text_or_images")


class Step(Contract):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,60}$")
    block: str = Field(description="The workflow block it belongs to, WF-01 to WF-10.")
    name: str = Field(max_length=80)
    purpose: str = Field(max_length=400)
    inputs: list[Artefact] = Field(default_factory=list, max_length=10)
    outputs: list[Artefact] = Field(default_factory=list, max_length=10)
    decision: Optional[str] = Field(None, max_length=60, description="The decision it makes: a plan key (analyse.AFFECTS), or none.")
    check: str = Field(max_length=400, description="What proves the step was done right.")
    rule: str = Field(max_length=200, description="The principle it rests on.")
    rests_on: list[str] = Field(default_factory=list, max_length=200, description="The findings it answers.")
    columns: list[str] = Field(default_factory=list, max_length=200, description="The columns it acts on.")
    kind: StepKind = "family"
    gate: bool = Field(False, description="A person signs before the next step runs.")


class SolutionWorkflow(Versioned):
    """The steps of one problem's solution, in order, and what they rest on."""
    schema_version: Literal["dclab_core.solution_workflow.v1"] = _version("solution_workflow")
    family: str = Field(max_length=40)
    steps: list[Step] = Field(default_factory=list, max_length=60)
    findings_used: list[str] = Field(default_factory=list, max_length=200)
    source: Literal["template", "model"] = "template"
    problems: list[str] = Field(default_factory=list, max_length=20, description="Why a proposal was refused (empty when valid).")


CONTRACTS["solution_workflow"] = SolutionWorkflow


def _cut(text: Any, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _s(id: str, block: str, name: str, purpose: str, decision: Optional[str], check: str, rule: str, kind: StepKind = "family",
       inputs: Iterable[str] = (), outputs: Iterable[str] = (), gate: bool = False, rests_on: Iterable[str] = (),
       columns: Iterable[str] = ()) -> Step:
    return Step(id=id, block=block, name=_cut(name, 80), purpose=_cut(purpose, 400), inputs=list(inputs), outputs=list(outputs),
                decision=decision, check=_cut(check, 400), rule=_cut(rule, 200), kind=kind, gate=gate,
                rests_on=list(rests_on)[:200], columns=[_cut(c, 120) for c in list(columns)[:200]])


# the steps every solution keeps, whatever the family; their content and their order are the rule
SAFETY: tuple[Step, ...] = (
    _s("lock_holdout", "WF-03", "Lock the holdout", "Set aside the rows the final score is read on, before any choice looks at the data.",
       "split.strategy", "The holdout's rows are fixed and fingerprinted before any analysis or model", "the holdout is used once",
       "safety", ["table"], ["training rows", "holdout"]),
    _s("train_only_analysis", "WF-04", "Analyse the training rows", "Read the data for the decisions on the training rows only.",
       None, "Every finding about the target is computed on the training rows", "no choice reads the holdout", "safety",
       ["training rows"], ["findings"]),
    _s("leakage_audit", "WF-05", "Audit leakage", "Check every column against the prediction moment before any model sees it.",
       "features.forbidden", "Each column kept is known at the prediction moment, each one dropped has its reason",
       "a column written after the moment is never a feature", "safety", ["training rows", "findings"], ["columns kept"]),
    _s("choose_on_validation", "WF-07", "Choose on validation", "Compare model families on the training rows' folds and choose.",
       "model.families", "The choice is made on folds of the training rows, with each score's spread", "the holdout never chooses",
       "safety", ["training rows", "columns kept"], ["chosen model"]),
    _s("holdout_gate", "WF-09", "The holdout gate", "A person signs that the solution is final before the holdout is read.",
       None, "The approval is recorded before the final score", "a person decides when the holdout is spent", "safety",
       ["chosen model"], ["approval"], gate=True),
    _s("final_once", "WF-09", "Score the holdout once", "Read the holdout once, with an interval.", "metric.primary",
       "The holdout is read once; a second read needs a person's reason and is flagged", "the holdout is used once", "safety",
       ["chosen model", "holdout", "approval"], ["holdout score"]),
    _s("report_limits", "WF-10", "Report with limits", "Say the score with its uncertainty and what it does not show.",
       "risk.check", "Every number traces to its record; never called production-ready from benchmark evidence",
       "report results with uncertainty and limits", "safety", ["holdout score", "findings"], ["report"]),
)
SAFETY_ORDER = tuple(s.id for s in SAFETY)
TEMPLATE = {s.id: s for s in SAFETY}

# each family's own steps (on folds of the training rows, or after the one read of the holdout)
FAMILY_STEPS: dict[str, tuple[Step, ...]] = {
    "classification": (
        _s("class_balance", "WF-03", "Keep the class balance", "Stratify the folds by the outcome.", "split.strategy",
           "Each fold holds every class at its share", "a fold without a class cannot score it", inputs=["table"]),
        _s("threshold", "WF-08", "Choose the operating point", "Pick the threshold from what an alert costs, on validation folds.",
           "threshold.policy", "The threshold is chosen on the training rows' folds", "the holdout never chooses",
           inputs=["training rows", "chosen model"]),),
    "regression": (
        _s("target_shape", "WF-04", "Read the target's shape", "Decide a transform for a skewed or capped target.", "target.transform",
           "The transform is chosen on the training rows", "a skewed target misleads squared error", inputs=["training rows"]),),
    "rare_event": (
        _s("review_budget", "WF-01", "Set the review budget", "How many alerts a day the people who review them can take.",
           "threshold.policy", "The budget is a number the person stated", "a rare-event model is judged at its budget"),
        _s("rank_screen", "WF-07", "Screen rare-event rankers", "Supervised rankers beside detectors and rules, ranked by recall at the budget.",
           "model.families", "Every candidate is scored at the same budget on the same folds", "accuracy says nothing at a rare rate",
           inputs=["training rows", "columns kept"]),
        _s("precision_bound", "WF-10", "Precision as a lower bound", "With partial labels an unlabelled alert may be a case nobody found.",
           "metric.primary", "Precision is reported as a lower bound when labels are partial", "partial labels make precision a floor",
           inputs=["holdout score"]),),
    "anomaly": (
        _s("normal_shape", "WF-04", "Describe normal", "What normal rows and entities look like, on the training rows.", "features.direction",
           "The shape of normal is measured, not assumed", "a detector scores distance from normal", inputs=["training rows"]),
        _s("detectors", "WF-07", "Screen detectors", "Rules, per-entity scores and detectors, judged on planted anomalies of the found shapes.",
           "model.families", "Each detector's recall at the review budget on planted anomalies", "without labels, evaluate on what is planted",
           inputs=["training rows", "columns kept"]),
        _s("review_loop", "WF-10", "Close the review loop", "Reviewed alerts become labels; with enough, the problem becomes a rare-event one.",
           "labels.coverage", "Each reviewed alert is recorded with its answer", "confirmed cases are the first labels", inputs=["holdout score"]),),
    "forecasting": (
        _s("rolling_origins", "WF-03", "Backtest from rolling origins", "Origins before the holdout, each forecasting only from its past.",
           "split.strategy", "Nothing after an origin enters its fit", "a forecast never sees its future", inputs=["table"]),
        _s("seasonal_naive", "WF-07", "Beat the seasonal naive", "Every model is compared with the seasonal naive forecast.",
           "baseline", "Each model's score beside the naive one", "a forecast that does not beat naive adds nothing",
           inputs=["training rows", "columns kept"]),),
    "time_to_event": (
        _s("censoring", "WF-04", "Read the censoring", "Which rows are still going, and how far the follow-up reaches.", "task.framing",
           "The censored share and the follow-up per horizon are measured", "a duration of a row still going is a lower bound",
           inputs=["training rows"]),
        _s("survival_models", "WF-07", "Survival models", "Kaplan-Meier, Cox and discrete hazards, scored where the follow-up reaches.",
           "model.families", "Horizons beyond the follow-up are not scored", "censoring is modelled, never dropped",
           inputs=["training rows", "columns kept"]),),
    "uplift": (
        _s("arms_check", "WF-04", "Check the arms", "Whether the treatment was randomised, and the arms' overlap.", "task.framing",
           "The propensity of the arms is measured", "an uplift is measured between comparable arms", inputs=["training rows"]),
        _s("qini", "WF-10", "Read the uplift by the Qini curve", "The holdout's uplift read as a Qini curve, with its interval.",
           "metric.primary", "The Qini area with its interval, from the one holdout read", "uplift ranks who responds to the treatment",
           inputs=["holdout score"]),),
    "ranking": (
        _s("popularity_baseline", "WF-07", "Beat popularity", "Every ranker is compared with the most popular items.", "baseline",
           "Each ranker's score beside popularity's, with the long tail's coverage", "popularity is the baseline to beat",
           inputs=["training rows", "columns kept"]),),
    "segmentation": (
        _s("scale", "WF-06", "Scale the columns", "Put the columns on one scale before any distance.", "features.direction",
           "Each column scaled by its robust spread", "a distance is dominated by its widest column", inputs=["training rows"]),
        _s("stability", "WF-07", "Check the segments' stability", "Segments compared across resamples of the training rows.",
           "model.families", "Agreement of the segments across resamples", "a segmentation that changes with the sample is noise",
           inputs=["training rows", "columns kept"]),),
    "streams": (
        _s("unit_baseline", "WF-04", "Each unit's baseline", "Readings read against each unit's own past.", "features.entity_history",
           "Windows use only readings before their end", "a unit's normal is its own", inputs=["training rows"]),),
    "entity_tables": (
        _s("cutoff", "WF-02", "Cut every table at the moment", "Events after the prediction moment are dropped from every table.",
           "features.forbidden", "No row dated after the cutoff enters a feature", "a later event is the future", inputs=["table"]),),
    "text_or_images": (
        _s("near_copies", "WF-03", "Keep near copies together", "Near copies of an item stay on one side of every split.",
           "split.strategy", "Near copies found and grouped before the split", "a copy on both sides scores memory", inputs=["table"]),),
}
REQUIRED_FAMILY = {f: tuple(s.id for s in steps) for f, steps in FAMILY_STEPS.items()}
BEFORE_LOCK = {None, "split.strategy", "labels.join", "threshold.policy", "features.forbidden"}  # what may come before the lock
AFTER_FINAL = {None, "risk.check", "labels.coverage", "metric.primary"}  # what may come after the one read: reading it, never choosing


def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, Mapping) else getattr(obj, name, None)


def _family(card: Any) -> Optional[str]:
    fam = _get(card, "family")
    fam = _get(fam, "value") if fam is not None and not isinstance(fam, str) else fam
    return fam if fam in FAMILIES else None


def _labels(card: Any) -> Optional[str]:
    lab = _get(card, "labels")
    return _get(lab, "value") if lab is not None and not isinstance(lab, str) else lab


def _slug(text: str) -> str:
    """An id part: ASCII words, or a short hash for a name that has none (so two names never share an id)."""
    plain = re.sub(r"[^a-z0-9_]+", "_", str(text).lower()).strip("_")[:30]
    return plain if plain and plain != "x" else "c" + hashlib.sha1(str(text).encode()).hexdigest()[:8]


def _card_columns(data_card: Any, purpose: str) -> list[str]:
    out = []
    for t in _get(data_card, "tables") or []:
        if _get(t, "role") not in (None, "main"):
            continue
        for c in _get(t, "columns") or []:
            if _get(c, "purpose") == purpose and _get(c, "name"):
                out.append(str(_get(c, "name")))
    return out


def _from_cards(data_card: Any, labels: Optional[str]) -> list[Step]:
    """The steps the cards call for before any analysis: the split by their entity or time, the labels joined."""
    out: list[Step] = []
    for entity in _card_columns(data_card, "entity")[:3]:
        out.append(_s(f"group_folds_{_slug(entity)}", "WF-03", f"Group the folds by {entity}", f"Keep each {entity}'s rows on one side "
                      "of every split.", "split.strategy", f"No {entity} has rows on both sides of a fold or of the holdout",
                      "the data card names it an entity", "card", ["table"], columns=[entity]))
    times = _card_columns(data_card, "time")
    if times:
        out.append(_s("time_split", "WF-03", "Split by time", "The holdout is the latest period; folds respect time.", "split.strategy",
                      "Every training row is earlier than every holdout row", "the data card names a time column", "card", ["table"],
                      columns=times[:1]))
    has_labels = any(_get(t, "role") == "labels" for t in _get(data_card, "tables") or [])
    if has_labels or labels == "file":
        out.append(_s("join_labels", "WF-02", "Join the labels", "Mark the cases on the table by the label file's key.", "labels.join",
                      "The key marks no more rows than the label file has", "a file of known cases is the outcome", "card",
                      ["table", "label file"], ["table"]))
    return out


def _leak_columns(f: Any) -> list[str]:
    evidence = _get(f, "evidence") or {}
    for key in ("later", "columns"):
        value = evidence.get(key)
        if isinstance(value, list) and value:
            return [str(v) for v in value][:5]
        if isinstance(value, str):
            return [value]
    column = evidence.get("column")
    return [str(column)] if column else [str(_get(f, "id") or "").split("-", 2)[-1]]


def _from_findings(findings: Any, card_steps: list[Step]) -> tuple[list[Step], list[str]]:
    """The steps the findings call for, each naming the findings it answers."""
    items = list(_get(findings, "findings") or (findings if isinstance(findings, list) else []))
    split_by: dict[str, list[str]] = {}
    leaks: dict[str, list[str]] = {}
    out: list[Step] = []
    named = {c for s in card_steps for c in s.columns}
    used: list[str] = []
    for f in items:
        fid, topic, severity = str(_get(f, "id") or ""), _get(f, "topic"), _get(f, "severity") or "info"
        evidence = _get(f, "evidence") or {}
        if topic in ("entities", "behaviour") and severity in ("warning", "critical"):
            entity = str(evidence.get("column") or evidence.get("entity") or fid.split("-", 2)[-1])
            split_by.setdefault(entity, []).append(fid)
        elif fid in ("F-time", "F-places") and severity == "warning":
            split_by.setdefault("time" if fid == "F-time" else "area", []).append(fid)
        elif topic == "leakage" and severity in ("warning", "critical"):
            for c in _leak_columns(f):
                leaks.setdefault(c, []).append(fid)
        elif topic == "detectability" and severity == "warning":
            out.append(_s("entity_features", "WF-06", "Derive per-entity features", "Features that make the missed shapes visible "
                          "(a row against its entity's history, the relation between columns).", "features.entity_history",
                          "The planted shapes are found after the new features", "a detector sees only what its features show", "finding",
                          ["training rows", "columns kept"], rests_on=[fid]))
        elif fid == "F-censoring" and severity == "warning":
            out.append(_s("censoring_aware", "WF-07", "Keep censoring in the model", "Rows still going stay in, as censored.",
                          "task.framing", "No row is dropped for being censored", "censoring is modelled, never dropped", "finding",
                          ["training rows", "columns kept"], rests_on=[fid]))
        else:
            continue
        used.append(fid)
    if leaks:
        out.append(_s("drop_leaks", "WF-05", "Keep the leaks out", f"{len(leaks)} column(s) are not known at the prediction moment.",
                      "features.forbidden", "None of them is in any fold's features", "a column written after the moment is never a feature",
                      "finding", ["training rows", "findings"], ["columns kept"], rests_on=sorted({f for fs in leaks.values() for f in fs}),
                      columns=sorted(leaks)))
    new = {k: v for k, v in split_by.items() if k not in named}
    if new:  # the analysis found a split the cards did not name: the holdout is locked again, the analysis read again
        out.append(_s("relock_holdout", "WF-04", "Lock the holdout again by what the analysis found",
                      f"Split by {', '.join(sorted(new))}, then read the analysis again on the new training rows.", "split.strategy",
                      "The analysis' training rows equal the rows locked by the new split", "a split found by the analysis reshapes the holdout",
                      "finding", ["table", "findings"], ["training rows", "holdout", "findings"],
                      rests_on=sorted({f for fs in new.values() for f in fs}), columns=sorted(new)))
    known = {k: v for k, v in split_by.items() if k in named}
    for step in card_steps:
        answered = [f for c in step.columns for f in known.get(c, [])]
        if answered:
            out.append(step.model_copy(update={"rests_on": sorted(set(answered))}))
    return out, used


def _ordered(steps: list[Step]) -> list[Step]:
    """By block; within a block the steps the cards and the family call for before the safety step, the others after."""
    rank = {sid: i for i, sid in enumerate(SAFETY_ORDER)}
    place = {"card": 0, "family": 0, "safety": 1, "finding": 2}
    return sorted(steps, key=lambda s: (BLOCKS.index(s.block), place[s.kind], rank.get(s.id, 0)))


def solution_workflow(problem_card: Any, data_card: Any = None, findings: Any = None) -> SolutionWorkflow:
    """The workflow for this problem: the safety steps, the family's, the cards' and the findings'; validated before it
    is returned. A card whose family is not settled is refused: no family is guessed (the owner's morning)."""
    family = _family(problem_card)
    if family is None:
        raise ValueError("the problem card's family is not settled: settle it before drawing the solution workflow")
    cards = _from_cards(data_card, _labels(problem_card))
    extra, used = _from_findings(findings, cards)
    by_id: dict[str, Step] = {s.id: s for s in cards}
    for s in extra:
        by_id[s.id] = s  # a card's step that a finding confirms carries the finding
    steps = _ordered([*SAFETY, *FAMILY_STEPS[family], *by_id.values()])
    workflow = SolutionWorkflow(family=family, steps=steps[:60], findings_used=used[:200])
    problems = validate(workflow)
    if problems:  # the template itself must hold; a failure here is a bug, said, never hidden
        raise ValueError("the template workflow does not validate: " + "; ".join(problems))
    return workflow


def validate(workflow: SolutionWorkflow, current: Optional[SolutionWorkflow] = None) -> list[str]:
    """What is wrong with a workflow (empty: valid), read on the steps' content. ``current``: the workflow a proposal
    would replace; its card and finding steps must stay as they were."""
    from dclab_core.analyse import AFFECTS

    problems: list[str] = []
    steps = workflow.steps
    ids = [s.id for s in steps]
    where = {sid: i for i, sid in enumerate(ids)}
    if workflow.family not in FAMILIES:
        problems.append(f"'{workflow.family}' is not a problem family")
    if len(set(ids)) != len(ids):
        problems.append("a step appears twice")
    templates = {**TEMPLATE, **{s.id: s for s in FAMILY_STEPS.get(workflow.family, ())}}
    if current is not None:
        templates.update({s.id: s for s in current.steps if s.kind in ("card", "finding")})
    for sid, template in templates.items():
        if sid not in where:
            what = "safety" if sid in TEMPLATE else workflow.family if sid in REQUIRED_FAMILY.get(workflow.family, ()) else "required"
            problems.append(f"the {what} step '{sid}' is missing")
        elif steps[where[sid]].model_dump() != template.model_dump():
            problems.append(f"the step '{sid}' was changed: its content is the rule, not its name")
    present = [sid for sid in SAFETY_ORDER if sid in where]
    for a, b in zip(present, present[1:]):
        if where[a] > where[b]:
            problems.append(f"'{a}' must come before '{b}'")
    for a, b in zip(steps, steps[1:]):
        if a.block in BLOCKS and b.block in BLOCKS and BLOCKS.index(b.block) < BLOCKS.index(a.block):
            problems.append(f"'{b.id}' ({b.block}) comes after '{a.id}' ({a.block}): the workflow never goes back")
    readers = [s.id for s in steps if "holdout" in s.inputs]
    if readers != ["final_once"]:
        problems.append(f"the holdout must be read by the final score alone, after the gate (read by: {', '.join(readers) or 'none'})")
    makers = [s.id for s in steps if "chosen model" in s.outputs]
    if makers != ["choose_on_validation"]:
        problems.append(f"only the choice on validation chooses the model (chosen by: {', '.join(makers) or 'none'})")
    lock, audit, final = where.get("lock_holdout"), where.get("leakage_audit"), where.get("final_once")
    for i, s in enumerate(steps):
        if s.block not in BLOCKS:
            problems.append(f"step '{s.id}' is in no workflow block")
        if s.decision is not None and s.decision not in AFFECTS:
            problems.append(f"step '{s.id}' makes an unknown decision '{s.decision}'")
        if not s.check.strip():
            problems.append(f"step '{s.id}' has no check")
        if lock is not None and i < lock and (s.decision not in BEFORE_LOCK or set(s.inputs) - {"table", "label file"}):
            problems.append(f"'{s.id}' comes before the holdout is locked but reads or decides more than the split")
        if audit is not None and i < audit and s.decision in ("model.families", "threshold.policy") and (lock is None or i > lock):
            problems.append(f"'{s.id}' chooses before the leakage audit")
        if final is not None and i > final and (s.decision not in AFTER_FINAL or s.block != "WF-10" or s.outputs and s.outputs != ["report"]):
            problems.append(f"'{s.id}' comes after the holdout is read and may only report it")
    if "holdout_gate" in where and not steps[where["holdout_gate"]].gate:
        problems.append("the holdout gate is not a person's gate")
    answered = {f for s in steps for f in s.rests_on}
    missing = [f for f in workflow.findings_used if f not in answered]
    if missing:
        problems.append(f"findings no step answers: {', '.join(missing[:5])}")
    return list(dict.fromkeys(problems))[:20]


def propose(current: SolutionWorkflow, proposed: Any) -> tuple[SolutionWorkflow, list[str]]:
    """A model's proposed workflow (a list of steps, or a mapping with ``steps``): used only when it validates against
    the current one; else the current workflow is kept and the problems are returned."""
    raw = proposed.get("steps") if isinstance(proposed, Mapping) else proposed
    try:
        steps = [s if isinstance(s, Step) else Step.model_validate(s) for s in (raw or [])]
        candidate = SolutionWorkflow(family=current.family, steps=steps, findings_used=current.findings_used, source="model")
    except Exception as exc:  # noqa: BLE001 - a proposal not in the contract's shape is refused like an invalid one
        return current, [f"the proposal is not a workflow: {type(exc).__name__}: {str(exc)[:200]}"]
    problems = validate(candidate, current)
    if problems:
        return current.model_copy(update={"problems": problems}), problems
    return candidate, []


__all__ = ["SAFETY", "SAFETY_ORDER", "FAMILY_STEPS", "FAMILIES", "Step", "SolutionWorkflow", "solution_workflow", "validate", "propose"]
import dclab_core.solution_map  # noqa: E402,F401  (the solution map contract, 24.4: registered once the workflow is defined; either module may be imported first)
