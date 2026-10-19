"""The solution router and its catalogue (package 24.3): for each sub-problem of a map, the approaches that could solve it, ranked, with why.

The catalogue is the library's own list of approaches, each for the families it serves: what it needs (labels, a time column, several tables,
a column of place or text, a stated budget, other sub-problems' scores), what it is good at, what it risks, and the evidence track that
measured it (the problem kinds of the studio's engine are the catalogue's entries, by key; ``review_budgeted`` is the one that is a composition,
not a model). ``route`` reads one sub-problem against the data card and the map and returns a :class:`Route`: every candidate with each
precondition met, unmet or unknown (never guessed), the chosen one (the first with none unmet), and the alternatives with what blocks them. A
single problem routes to the kind the host's own detection found (``detected``), never replaced. The role ``solution_router`` may choose
among the candidates that are in the catalogue and whose preconditions the data card does not refute; it cannot add an approach.

Deterministic and pure: no model, no store; a model's choice is merged by the caller through ``dclab_core.agents``.
"""

from __future__ import annotations

import re
from typing import Any, Literal, Mapping, Optional

from pydantic import Field

from dclab_core.agents import Role
from dclab_core.behaviour import coordinate_pairs
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.problem_map import ProblemMap, SubProblem

#: what an approach may need of the problem and the data; each is judged met, unmet, or unknown (no data card, no outcome named)
PRECONDITIONS = ("labels", "binary_target", "many_classes", "number_target", "text_column", "place_columns", "time_column", "entity_column",
                 "several_tables", "treatment_column", "duration_column", "budget_step", "not_budget_step", "pairs_table")
Met = Optional[bool]
# the route contract's limits, frozen as v1 published them (30.16 added a precondition and an approach: a route lists one
# family's approaches, each with its few needs, far under either)
MAX_NEEDS, MAX_CANDIDATES = 13, 16
PAIR_WORDS = re.compile(r"match|pair|link|dup|cop(?:y|ies)|same|original|mapping")  # a table of known pairs names itself so


class Approach(Contract):
    """One way to solve a kind of sub-problem."""

    id: str = Field(pattern=r"^[a-z_]{3,40}$", description="The studio's problem-kind key where one runs it, else the approach's own name.")
    families: tuple[str, ...] = Field(min_length=1, max_length=4)
    title: str = Field(max_length=120)
    summary: str = Field(max_length=300)
    needs: tuple[str, ...] = ()
    specificity: int = Field(default=0, description="A more specific approach is asked first among those whose needs are met.")
    strengths: tuple[str, ...] = ()
    risks: tuple[str, ...] = ()
    evidence_track: Optional[str] = Field(default=None, description="research/<track>/ whose records measured it.")
    engine: Literal["engine", "composition"] = "engine"


def _a(id: str, families: Any, title: str, summary: str, needs: tuple[str, ...] = (), specificity: int = 0, strengths: tuple[str, ...] = (),
       risks: tuple[str, ...] = (), evidence_track: Optional[str] = None, engine: str = "engine") -> Approach:
    families = (families,) if isinstance(families, str) else tuple(families)
    return Approach(id=id, families=families, title=title, summary=summary, needs=needs, specificity=specificity, strengths=strengths, risks=risks,
                    evidence_track=evidence_track, engine=engine)


CATALOGUE: tuple[Approach, ...] = (
    _a("binary", "classification", "A yes or no, learned from known outcomes", "Gradient-boosted and linear models on the table's features for an outcome of two values.",
       ("labels", "binary_target"), 1, ("fast", "reads the table as it is"), ("a rare outcome needs the rare-event kind", "leaks when a column is known only after"), "tabular-classification"),
    _a("multiclass", "classification", "One of several classes, learned from known outcomes", "The same models with one outcome of three or more classes, judged per class.",
       ("labels", "many_classes"), 0, ("per-class scores",), ("small classes are scored with wide intervals",), "tabular-classification"),
    _a("text_tabular", "classification", "A class from long text beside the table", "The text is encoded and joined to the table's own columns for a yes or no.",
       ("labels", "text_column", "binary_target"), 10, ("reads the words",), ("a text column that names the outcome leaks it",), "nlp-and-text"),
    _a("regression", "regression", "A number, learned from known values", "Gradient-boosted and linear regressors for a numeric outcome, judged by error in its own unit.",
       ("labels", "number_target"), 0, ("error in the outcome's own unit",), ("a column written after the outcome leaks it",), "tabular-regression"),
    _a("places", "regression", "A number that depends on where", "A regressor whose folds are blocks of places, so a place is never on both sides of the split.",
       ("labels", "number_target", "place_columns"), 10, ("honest about places never seen",), ("a random split would flatter it",), "places-and-spatial"),
    _a("rare_event", ("rare_event",), "A rare event, learned from the cases found", "Models for the few confirmed cases, judged at a review budget, with the unmarked rows read as unknown.",
       ("labels",), 20, ("works with partial labels", "judged at a budget"), ("an unmarked row is unknown, not clean", "rows of one person on both sides of a split"), "anomaly-and-fraud-detection"),
    _a("anomaly", ("anomaly", "rare_event"), "The unusual rows, with no labels", "Detectors chosen on planted anomalies of a validation period, and a review list of the budget's size.",
       (), 5, ("needs no labels",), ("unusual is not wrong", "no label to prove a hit"), "anomaly-and-fraud-detection"),
    _a("forecasting", "forecasting", "Values ahead in time", "Rolling-origin models for a series, judged on the periods after the cut.",
       ("time_column", "entity_column", "number_target"), 0, ("judged ahead in time",), ("a feature known only on the day itself leaks",), "time-series-forecasting"),
    _a("time_to_event", "time_to_event", "How long until it happens", "Survival models that keep the cases not yet finished, judged by concordance.",
       ("duration_column",), 0, ("keeps unfinished cases",), ("dropping the unfinished biases it",), "churn-prediction"),
    _a("ranking", "ranking", "An order of items for each person", "Models for the items a person will take first, judged by the order of the list.",
       ("entity_column", "not_budget_step"), 10, ("judged by the order",), ("an item shown first is taken more",), "recommender-systems"),
    _a("review_budgeted", ("ranking",), "The cases a team can review, cut to its budget", "The scores of other parts are fused into one risk and the list is cut to the number a team can check.",
       ("budget_step",), 30, ("the budget, not the score, sets the size",), ("a score is only as good as its parts",), None, "composition"),
    _a("segmentation", "segmentation", "Groups of alike rows", "Clusters chosen for stability and for what separates them, with no outcome.",
       (), 0, ("needs no outcome",), ("groups are only useful if someone acts on them",), "data-science-foundations"),
    _a("uplift", "uplift", "Who an action changes", "Models of the difference an action makes, from a table that says who was treated.",
       ("treatment_column", "labels", "binary_target"), 0, ("separates who stays anyway",), ("without a control the effect is a guess",), "causal-inference-and-experimentation"),
    _a("entity_tables", "entity_tables", "An outcome per entity from its related tables", "Features per entity and cutoff built from event tables, judged on the latest cutoff the log reaches.",
       ("entity_column", "several_tables"), 0, ("reads the events themselves",), ("a cutoff that lets the future in",), "churn-prediction"),
    _a("streams", "streams", "Alerts from readings over time", "Detectors on a unit's readings in windows, judged by alerts that come early enough to act on.",
       ("time_column", "entity_column", "several_tables"), 0, ("judged by lead time",), ("an alert after the failure is no alert",), "anomaly-and-fraud-detection"),
    _a("matching", ("rare_event", "anomaly", "entity_tables"), "The same record twice, found pair by pair",
       "Pairs of records compared on the fields named, blocked so only plausible pairs form, judged by pair precision and recall with a record and its matches on one side.",
       ("pairs_table",), 40, ("finds copies no single column shows", "judged against the exact rule"), ("blocking that drops a true pair", "a record and its copy on two sides"),
       "record-linkage"),
    _a("embedded", "text_or_images", "An outcome read from long text or images", "A pretrained encoder turns each source into a vector, then the table's models read them.",
       (), 0, ("reads what a table cannot",), ("the encoder is a download",), "computer-vision"),
)
BY_ID: dict[str, Approach] = {a.id: a for a in CATALOGUE}


class Precondition(Contract):
    code: str = Field(max_length=40)
    met: Met = Field(default=None, description="True: the data card and the map meet it; False: they refute it; None: not known yet.")
    why: str = Field(default="", max_length=300)


class Candidate(Contract):
    approach: str = Field(max_length=40)
    rank: int = Field(ge=1)
    preconditions: list[Precondition] = Field(default_factory=list, max_length=MAX_NEEDS)
    blocking: list[str] = Field(default_factory=list, max_length=MAX_NEEDS, description="The needs the data card or the map refutes.")
    unknown: list[str] = Field(default_factory=list, max_length=MAX_NEEDS, description="The needs not known yet: a question for the person or the data.")
    reasons: list[str] = Field(default_factory=list, max_length=6)
    evidence_track: Optional[str] = Field(default=None, max_length=80)


class Route(Versioned):
    """The approaches for one sub-problem, ranked, with the one chosen and why."""

    schema_version: Literal["dclab_core.route.v1"] = _version("route")
    sub_problem: str = Field(max_length=40)
    family: str = Field(max_length=40)
    candidates: list[Candidate] = Field(default_factory=list, max_length=MAX_CANDIDATES)
    chosen: Optional[str] = Field(default=None, max_length=40, description="The approach routed to; None when every candidate is refuted.")
    source: Literal["rules", "advisor", "detected"] = "rules"
    reason: str = Field(default="", max_length=300)
    single: bool = False
    map: Optional[str] = Field(default=None, max_length=64, description="The fingerprint of the map the route was made for.")


CONTRACTS["route"] = Route


# ---------------------------------------------------------------------------------------------------------------- the data card, read

def _tables(data: Any) -> list[Any]:
    return list(getattr(data, "tables", None) or (data.get("tables") if isinstance(data, Mapping) else None) or [])


def _get(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _columns(data: Any) -> list[Any]:
    return [c for t in _tables(data) for c in (_get(t, "columns") or [])]


#: families that learn from one outcome column of the data card (a card names one outcome: the sub-problem it belongs to is judged on it)
OUTCOME_FAMILIES = frozenset({"classification", "regression"})


def _target_column(data: Any, sub: SubProblem, m: Optional[ProblemMap] = None) -> Optional[Any]:
    """The column of the data card that is THIS sub-problem's outcome: the card's named target when the sub-problem is the only one of the map that
    learns from an outcome, else a column named in the sub-problem's own outcome words. Never a guess from the table's candidates: an outcome nobody
    named is not known, and a need that rests on it is unknown."""
    columns = _columns(data)
    outcome = " ".join((sub.outcome or "").lower().split())
    if outcome:
        found = [c for c in columns if str(_get(c, "name")).lower() in outcome.replace("_", " ").split() or str(_get(c, "name")).lower() in outcome]
        if len(found) == 1:
            return found[0]
    wanted = _get(data, "target")
    if wanted:
        owners = [x for x in (m.sub_problems if m else [sub]) if x.family in OUTCOME_FAMILIES]
        if len(owners) <= 1 or m is None or m.size == "single":
            return next((c for c in columns if _get(c, "name") == wanted), None)
    return None


TREATMENT_TOKENS = frozenset({"treated", "treatment", "exposed", "exposure", "control", "campaign", "offer", "received", "targeted", "variant", "arm"})
DURATION_TOKENS = frozenset({"duration", "tenure", "lifetime", "survival", "days", "weeks", "months", "minutes"})


def _tokens(name: Any) -> set[str]:
    """The lower-case words of a column's name (got_offer, gotOffer and Got Offer are alike)."""
    spaced = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(name))
    return {t for t in re.split(r"[^a-z0-9]+", spaced.lower()) if t}


def _has_purposes(columns: list[Any]) -> bool:
    """Whether the data card says what its columns are for (a card from before purposes existed does not: then a need that rests on them is unknown)."""
    return any(_get(c, "purpose") for c in columns)


def _feeds_budget(sub: SubProblem, m: Optional[ProblemMap]) -> bool:
    return bool(m) and any(c.kind == "budget" and sub.id in c.inputs for c in m.composition.steps)


def check(code: str, sub: SubProblem, data: Any = None, m: Optional[ProblemMap] = None) -> tuple[Met, str]:
    """One need, judged: met, unmet or not known, with why in words (names and counts, never a value)."""
    if code not in PRECONDITIONS:
        raise ValueError(f"unknown precondition {code!r}")
    has = data is not None and bool(_tables(data))
    columns = _columns(data) if has else []
    purposes = [str(_get(c, "purpose") or "") for c in columns]
    if code == "labels":
        return ((True, "the outcome is in a column or a file of cases") if sub.labels in ("column", "file") else
                (False, "nothing is labelled") if sub.labels == "none" else (None, "the words do not say where the outcome is"))
    if code == "budget_step":
        yes = _feeds_budget(sub, m)
        return yes, "its result is cut to a budget" if yes else "no budget step takes it"
    if code == "not_budget_step":
        yes = _feeds_budget(sub, m)
        return (not yes), "not a review list" if not yes else "its result is cut to a budget: a review list, not an order for each person"
    if code in ("binary_target", "many_classes", "number_target"):
        if not has:
            return None, "no table yet"
        target = _target_column(data, sub, m)
        if target is None:
            return None, "no outcome column is named for this part yet"
        distinct = int(_get(target, "distinct") or 0)
        name, dtype = str(_get(target, "name")), str(_get(target, "dtype"))
        if code == "binary_target":
            return (distinct == 2), f"{name} has {distinct} values"
        if code == "many_classes":
            return (3 <= distinct <= 100), f"{name} has {distinct} values"
        if dtype == "float":
            return True, f"{name} is a number ({distinct} values)"
        if dtype == "integer":  # a count of a few values may be a class or a number: not refuted, not settled
            return (True, f"{name} is a whole number with {distinct} values") if distinct > 20 else (None, f"{name} is a whole number with {distinct} values: classes or a count")
        return False, f"{name} is {dtype}, not a number"
    if code == "pairs_table":  # 30.16: a table of known matches (a duplicate and its original, or two catalogues' ids), and nothing else
        if not has:  # no table: no table of pairs is shown, never an unknown that would rank matching first
            return False, "no table yet: no table of known pairs is shown"
        if not _has_purposes(columns):
            return False, "the data card does not say which columns are identifiers: a table of known pairs is not shown"
        found = []
        for t in _tables(data):
            cols = list(_get(t, "columns") or [])
            if _get(t, "role", "main") in ("main", "labels") or not 2 <= len(cols) <= 3:
                continue
            ids = [c for c in cols if str(_get(c, "purpose") or "") == "identifier"]
            timed = any(str(_get(c, "purpose") or "") == "time" for c in cols)
            words = " ".join([str(_get(t, "name") or "")] + [str(_get(c, "name") or "") for c in cols]).lower()
            # the review: an orders table, or a file of cases, has two id-like columns too; a table of pairs says what it is
            if len(ids) >= 2 and not timed and PAIR_WORDS.search(words):
                found.append(str(_get(t, "name")))
        return (True, f"a table of known pairs: {found[0]}") if found else (False, "no table of two identifiers named as matches, pairs or copies")
    if not has:
        return None, "no table yet"
    if code in ("time_column", "entity_column", "place_columns") and not _has_purposes(columns):
        return None, "the data card does not say what its columns are for"
    if code == "text_column":
        n = sum(1 for p, c in zip(purposes, columns) if p == "text" or str(_get(c, "role")) == "free_text")
        return n > 0, f"{n} long-text column(s)"
    if code == "place_columns":
        pairs = coordinate_pairs([str(_get(c, "name")) for c in columns])
        return bool(pairs), (f"a latitude and longitude pair: {pairs[0][1]}, {pairs[0][2]}" if pairs else "no latitude and longitude pair")
    if code == "time_column":
        n = purposes.count("time")
        return n > 0, f"{n} time column(s)"
    if code == "entity_column":
        n = purposes.count("entity")
        return n > 0, f"{n} entity column(s)"
    if code == "several_tables":
        n = sum(1 for t in _tables(data) if _get(t, "role", "main") != "labels")  # a file of known cases is not a table of events
        return n >= 2, f"{n} table(s) beside the labels"
    if code == "treatment_column":
        found = [str(_get(c, "name")) for c in columns if _tokens(_get(c, "name")) & TREATMENT_TOKENS]
        return (True, f"a column named like a treatment: {found[0]}") if found else (None, "no column is named like a treatment: which one says who was treated?")
    if code == "duration_column":
        found = [str(_get(c, "name")) for c in columns if _tokens(_get(c, "name")) & DURATION_TOKENS]
        return (True, f"a column named like a duration: {found[0]}") if found else (None, "no column is named like a duration: which one says how long?")
    raise AssertionError(code)  # every code of PRECONDITIONS is handled above


# ---------------------------------------------------------------------------------------------------------------- the rules' route

def _candidate(approach: Approach, sub: SubProblem, data: Any, m: Optional[ProblemMap], reasons: Optional[list[str]] = None) -> Candidate:
    judged = [(code, *check(code, sub, data, m)) for code in approach.needs]
    shown = [why for _, met, why in judged if met is True][:3]
    if not judged:
        shown.append("needs nothing of the data beyond the problem")
    return Candidate(approach=approach.id, rank=1, preconditions=[Precondition(code=c, met=met, why=why[:300]) for c, met, why in judged],
                     blocking=[c for c, met, _ in judged if met is False], unknown=[c for c, met, _ in judged if met is None],
                     reasons=(reasons or shown)[:6], evidence_track=approach.evidence_track)


def candidates_for(sub: SubProblem, data: Any = None, m: Optional[ProblemMap] = None) -> list[Candidate]:
    """Every approach of the catalogue for the sub-problem's family, each with its needs judged, ranked: those no need refutes first, then the one
    the map itself names (its kind), then the more specific, then the fewer unknown needs, then the catalogue's order."""
    found = [(a, position, _candidate(a, sub, data, m)) for position, a in enumerate(a for a in CATALOGUE if sub.family in a.families)]
    found.sort(key=lambda t: (bool(t[2].blocking), 0 if t[0].id == sub.kind else 1, -t[0].specificity, len(t[2].unknown), t[1]))
    return [c.model_copy(update={"rank": i}) for i, (_, _, c) in enumerate(found, start=1)]


def _refuted(c: Candidate) -> str:
    return f"the data or the map refutes it: it needs {', '.join(c.blocking)}"


def route(sub: SubProblem, data: Any = None, m: Optional[ProblemMap] = None, detected: Optional[str] = None) -> Route:
    """The route for one sub-problem. ``detected``: the kind the host's own detection found for a single problem the rules drew: it is chosen, never replaced (and
    said, when its family is not the map's or the data or the map refutes it: the candidates list what blocks it). A single map the person edited
    says its own family: the detection is not used."""
    ranked = candidates_for(sub, data, m)
    single = m is not None and m.size == "single"
    chosen, source, reason = None, "rules", ""
    usable = bool(single and detected and detected in BY_ID and m is not None and m.source == "rules")  # a person's own single map says its family itself
    if usable:
        chosen, source = detected, "detected"
        if all(c.approach != detected for c in ranked):
            ranked = [*ranked, _candidate(BY_ID[detected], sub, data, m, ["the kind the detection found"])]
            ranked = [c.model_copy(update={"rank": i}) for i, c in enumerate(ranked, start=1)]
        mine = next(c for c in ranked if c.approach == detected)
        other = f" (a {'/'.join(BY_ID[detected].families)} approach, where the map reads {sub.family})" if sub.family not in BY_ID[detected].families else ""
        reason = f"a single problem keeps the kind the detection found ({detected}){other}" + (f"; {_refuted(mine)}" if mine.blocking else "")
    elif single and detected and detected in BY_ID:
        reason = f"the detection found {detected}, but the map is the person's own reading ({sub.family}): routed by the rules. "
    if not usable:
        first = next((c for c in ranked if not c.blocking), None)
        if first is not None:
            chosen = first.approach
            reason += (f"the first approach for {sub.family} that no need of the data or the map refutes" + (f"; still to settle: {', '.join(first.unknown)}" if first.unknown else ""))
        else:
            reason += f"every approach for {sub.family} is refuted by the data or the map: " + "; ".join(f"{c.approach} needs {', '.join(c.blocking)}" for c in ranked[:3])
    return Route(sub_problem=sub.id, family=sub.family, candidates=ranked, chosen=chosen, source=source, reason=reason[:300], single=single, map=m.fingerprint if m else None)


def routes(m: ProblemMap, data: Any = None, detected: Optional[str] = None) -> dict[str, Route]:
    """The route of every sub-problem of a map (``detected``: the kind of a single problem)."""
    return {s.id: route(s, data, m, detected if m.size == "single" else None) for s in m.sub_problems}


# ---------------------------------------------------------------------------------------------------------------- the model role

def lines(r: Route) -> list[str]:
    return [f"{c.rank}. {c.approach}: " + ("BLOCKED by " + ", ".join(c.blocking) if c.blocking else "needs met" + (f"; unknown: {', '.join(c.unknown)}" if c.unknown else ""))
            for c in r.candidates][:20]


def _router_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    r: Route = inputs["route"]
    sub: SubProblem = inputs["sub"]
    return {"sub_problem": f"{sub.id}: {sub.family}, labels={sub.labels}, unit={sub.unit or '?'}", "question": (sub.question or sub.title)[:300], "candidates": lines(r),
            "chosen_by_rules": r.chosen or "", "approaches": {c.approach: BY_ID[c.approach].summary[:160] for c in r.candidates[:10]},
            **({"similar_problems": [str(x)[:480] for x in inputs["examples"][:3]]} if inputs.get("examples") else {})}


def _router_parse(reply: Mapping[str, Any], inputs: Mapping[str, Any]) -> tuple[str, str]:
    """The approach the model chooses and why: it must be a candidate of the catalogue that the data card and the map do not refute."""
    r: Route = inputs["route"]
    choice, why = reply.get("choice"), reply.get("reason")
    candidate = next((c for c in r.candidates if c.approach == choice), None) if isinstance(choice, str) else None
    if candidate is None:
        raise ValueError("the choice is not one of the candidates")
    if candidate.blocking:
        raise ValueError(f"{choice} is refuted by {', '.join(candidate.blocking)}")
    if candidate.unknown and choice != r.chosen:
        raise ValueError(f"{choice} has needs nothing has met yet ({', '.join(candidate.unknown)}): only the rules' own choice may stand on needs not yet known")
    if not isinstance(why, str) or not why.strip():
        raise ValueError("a choice comes with its reason")
    return str(choice), " ".join(why.split())[:300]


SOLUTION_ROUTER = Role(name="solution_router", description="Chooses among the catalogue's approaches for a sub-problem; one the data card or the map refutes is refused.",
                       evidence=_router_evidence, parse=_router_parse, fallback=lambda inputs: (inputs["route"].chosen or "", inputs["route"].reason), merge=lambda rules, model: model)


def route_with(sub: SubProblem, data: Any = None, m: Optional[ProblemMap] = None, detected: Optional[str] = None, advisor: Any = None,
               examples: Optional[list[str]] = None) -> tuple[Route, str]:
    """The route, with the model's choice among its candidates when the problem is not single and the advisor serves. Returns the route and why.
    ``examples`` (30.18): the closest business examples, read as advice; the choice is still one of the candidates nothing refutes."""
    from dclab_core.agents import run_role

    base = route(sub, data, m, detected)
    if base.single or not base.candidates or advisor is None:
        return base, "no advisor" if advisor is None else "a single problem keeps its detected kind"
    result = run_role(SOLUTION_ROUTER, advisor, {"route": base, "sub": sub, "examples": list(examples or [])})
    if result.source != "advisor":
        return base, result.reason
    choice, reason = result.output
    if choice == base.chosen:
        return base.model_copy(update={"source": "advisor", "reason": f"the advisor agrees: {reason}"[:300]}), result.reason
    return base.model_copy(update={"chosen": choice, "source": "advisor", "reason": reason[:300]}), result.reason


__all__ = ["Approach", "Candidate", "Precondition", "Route", "CATALOGUE", "BY_ID", "PRECONDITIONS", "check", "candidates_for", "route", "routes", "route_with", "SOLUTION_ROUTER"]
