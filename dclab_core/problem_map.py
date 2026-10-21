"""The problem map (package 24.1): a problem as parts and sub-problems, how their outputs combine, and the rules that draw and judge it.

A small problem is a map with one part and one sub-problem: nothing changes for it except that the map says so and why. A
complex one (for example fraud among couriers and customers, with a few confirmed cases and a review budget) is several
parts, each with its question, its unit and its moment; each part has a sub-problem shaped like a problem card (a family, labels,
an outcome, what it needs, what it reads of the others); and a composition says how their outputs combine (a sequence, a fusion
of scores, a gate, a budget), step by step, each step on what exists before it.

``decompose`` draws a map from the person's words by rules: an explicit brief with parts becomes the map's parts (challenge mode's
``read_brief``), a few well-known shapes have templates (a fraud problem; a routing problem asked in two steps), and anything else
is one part. A template needs the words to say its shape and the problem card not to say another family; a sentence that only
shares words with it stays one problem. ``validate`` judges a map, whoever drew it (the rules, a model, the person's edits): units
and moments agree between a part and its sub-problems, no sub-problem reads another's outcome before it is known, no cycle, a
composition built in order from sub-problems and earlier outputs that uses every sub-problem nobody else reads, a key wherever
two units are joined, and a size that matches the parts.

The constructor checks shape (types, limits, the fingerprint); ``validate`` checks meaning and returns every issue in words, so a
person's edit can be shown its reasons instead of being refused with the first one. A missing unit or moment is not an unsound
map (the words may not say): it is a question the map reviewer asks (24.2). Pure functions over contracts; no model, no store.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Annotated, Any, Literal, Optional

from pydantic import Field, StringConstraints, model_validator

from dclab_core.bands import LABEL_SITUATIONS
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Family = Literal["classification", "regression", "rare_event", "anomaly", "forecasting", "time_to_event", "ranking", "segmentation", "uplift",
                 "entity_tables", "streams", "text_or_images"]
Source = Literal["rules", "model", "person"]
Reads = Literal["score", "outcome"]
Method = Literal["none", "sequence", "fusion", "gate", "budget", "mixed"]
StepKind = Literal["sequence", "fusion", "gate", "budget"]
Engine = Literal["engine", "library"]
MAX_TEXT = 600
MAX_PARTS = 20
MAX_SUB_PROBLEMS = 40
#: families the engine does not run as a project yet (a leaf of one is a library kind, and the map says so); none today
LIBRARY_ONLY: frozenset[str] = frozenset()
Title = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]


class Need(Contract):
    """What a sub-problem needs of the data: a table, columns by name (never values), a label situation."""

    table: Optional[str] = Field(default=None, max_length=200)
    columns: list[str] = Field(default_factory=list, max_length=40)
    label: Optional[Literal["column", "file", "none"]] = None
    why: str = Field(default="", max_length=MAX_TEXT)


class Read(Contract):
    """Another sub-problem's output this one reads: its score (the model's prediction) or its outcome (what became known)."""

    sub_problem: str = Field(min_length=1, max_length=40)
    what: Reads
    via: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The key that joins this unit to the other's, when they differ (the account of a ride).")


class Part(Contract):
    id: str = Field(pattern=r"^part\d+$")
    title: Title
    question: str = Field(default="", max_length=MAX_TEXT)
    unit: Optional[str] = Field(default=None, max_length=120, description="What one row of this part is: a ride, an account, a ticket.")
    moment: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="When the answer is needed, and what is known then.")
    outcome: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    risks: list[str] = Field(default_factory=list, max_length=10)
    quote: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The person's words this part comes from.")
    why: str = Field(default="", max_length=MAX_TEXT, description="The rule that drew it, when it is not the person's words.")


class SubProblem(Contract):
    id: str = Field(pattern=r"^sp\d+$")
    part: str = Field(pattern=r"^part\d+$")
    title: Title
    family: Family
    kind: Optional[str] = Field(default=None, max_length=60, description="The problem kind the engine runs, when finer than the family.")
    question: str = Field(default="", max_length=MAX_TEXT)
    unit: Optional[str] = Field(default=None, max_length=120)
    moment: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    predict_rank: int = Field(default=0, ge=0, description="When its prediction is made, in the order of the whole problem (0 first).")
    known_rank: Optional[int] = Field(default=None, ge=0, description="When its outcome becomes known, after it was predicted (an investigation's verdict is late); None: never in time to use.")
    outcome: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    labels: Literal["column", "file", "none", "unknown"] = "unknown"
    needs: list[Need] = Field(default_factory=list, max_length=10)
    reads: list[Read] = Field(default_factory=list, max_length=10)
    engine: Engine = "engine"
    quote: Optional[str] = Field(default=None, max_length=MAX_TEXT)
    why: str = Field(default="", max_length=MAX_TEXT)


class CompositionStep(Contract):
    kind: StepKind
    inputs: list[str] = Field(default_factory=list, max_length=MAX_SUB_PROBLEMS, description="Sub-problem ids and the outputs of earlier steps, in order.")
    output: str = Field(min_length=1, max_length=60)
    joins: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="The key that joins the inputs' units, when they differ (the account of a ride).")
    note: str = Field(default="", max_length=MAX_TEXT)


class Composition(Contract):
    method: Method = "none"
    steps: list[CompositionStep] = Field(default_factory=list, max_length=MAX_SUB_PROBLEMS)
    budget: Optional[str] = Field(default=None, max_length=MAX_TEXT, description="A review budget or a limit on what the composed system may flag, in words.")


class ProblemMap(Versioned):
    """A problem as parts, sub-problems and how they combine."""

    schema_version: Literal["dclab_core.problem_map.v1"] = _version("problem_map")
    size: Literal["single", "complex"]
    parts: list[Part] = Field(min_length=1, max_length=MAX_PARTS)
    sub_problems: list[SubProblem] = Field(min_length=1, max_length=MAX_SUB_PROBLEMS)
    composition: Composition = Field(default_factory=Composition)
    source: Source = "rules"
    template: Optional[str] = Field(default=None, max_length=60, description="The rule template that drew it, or 'brief' (the person's parts) or 'single'.")
    notes: list[str] = Field(default_factory=list, max_length=20)
    fingerprint: str = Field(default="", max_length=64, description="The hash of the structure (units, moments, families, labels, ranks, needs, reads, composition): the same map, the same fingerprint.")

    @model_validator(mode="after")
    def _fingerprinted(self) -> "ProblemMap":
        expected = fingerprint(self)
        if self.fingerprint and self.fingerprint != expected:
            raise ValueError("fingerprint does not match the map: a map is never edited in place; draw a new one")
        object.__setattr__(self, "fingerprint", expected)
        self.__pydantic_fields_set__.add("fingerprint")  # a dump without defaults keeps it
        return self


CONTRACTS["problem_map"] = ProblemMap


def fingerprint(m: ProblemMap) -> str:
    """The hash of what the map says structurally: units, moments, families, labels, ranks, needs, reads (with their keys) and the
    composition (its steps in order, their keys and its budget). Not the wording of titles, questions, quotes and reasons."""
    body = {"parts": [(p.id, p.unit, p.moment) for p in m.parts],
            "subs": [(s.id, s.part, s.family, s.kind, s.labels, s.unit, s.moment, s.predict_rank, s.known_rank, s.engine,
                      sorted((r.sub_problem, r.what, r.via) for r in s.reads), [(n.table, sorted(n.columns), n.label) for n in s.needs]) for s in m.sub_problems],
            "composition": (m.composition.method, m.composition.budget, [(c.kind, list(c.inputs), c.output, c.joins) for c in m.composition.steps])}
    return hashlib.sha256(json.dumps(body, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------------------------- validation

def _unit_of(sub: SubProblem, parts: dict[str, Part]) -> Optional[str]:
    """The unit a sub-problem works on: its own, else its part's."""
    part = parts.get(sub.part)
    return sub.unit or (part.unit if part else None)


def validate(m: ProblemMap) -> list[str]:
    """What is wrong with a map, in words (empty: it is sound), every issue and not only the first. Whoever drew it, the same rules judge it."""
    out: list[str] = []
    if m.fingerprint != fingerprint(m):
        out.append("the fingerprint is stale: the map was changed after it was made; draw a new one from the change")
    parts = {p.id: p for p in m.parts}
    subs = {s.id: s for s in m.sub_problems}
    if len(parts) != len(m.parts):
        out.append("two parts have the same id")
    if len(subs) != len(m.sub_problems):
        out.append("two sub-problems have the same id")
    for p in m.parts:
        if not any(s.part == p.id for s in m.sub_problems):
            out.append(f"{p.id} ({p.title}) has no sub-problem")
    for s in m.sub_problems:
        part = parts.get(s.part)
        if part is None:
            out.append(f"{s.id} belongs to {s.part}, which is not a part of this map")
        else:
            if part.unit and s.unit and part.unit != s.unit:
                out.append(f"{s.id} works on {s.unit}, but its part {part.id} works on {part.unit}: one part, one unit")
            if part.moment and s.moment and part.moment != s.moment:
                out.append(f"{s.id}'s moment differs from its part's ({part.id}): '{s.moment[:60]}' against '{part.moment[:60]}'")
        if s.family in LIBRARY_ONLY and s.engine == "engine":
            out.append(f"{s.id} is {s.family}, which the engine does not run as a project: say it is a library kind")
        if s.known_rank is not None and s.known_rank <= s.predict_rank:
            out.append(f"{s.id}'s outcome would be known (rank {s.known_rank}) no later than its own prediction (rank {s.predict_rank}): then there is nothing to predict")
        if s.known_rank is not None and s.labels == "none":
            out.append(f"{s.id} says no label exists, but its outcome becomes known at rank {s.known_rank}: which is it")
        mine = _unit_of(s, parts)
        for r in s.reads:
            other = subs.get(r.sub_problem)
            if other is None:
                out.append(f"{s.id} reads {r.sub_problem}, which is not in this map")
                continue
            if other.id == s.id:
                out.append(f"{s.id} reads itself")
                continue
            theirs = _unit_of(other, parts)
            if mine and theirs and mine != theirs and not r.via:
                out.append(f"{s.id} works on {mine} and reads {other.id}, which works on {theirs}: say the key that joins them")
            if r.what == "outcome":
                if other.known_rank is None or other.known_rank >= s.predict_rank:
                    when = "never in time" if other.known_rank is None else f"only at rank {other.known_rank}"
                    out.append(f"{s.id} reads the outcome of {other.id}, which is known {when}, not before {s.id}'s own moment (rank {s.predict_rank}): that is a leak")
            elif other.predict_rank > s.predict_rank:
                out.append(f"{s.id} reads the score of {other.id}, which is only made at rank {other.predict_rank}, after {s.id}'s own moment (rank {s.predict_rank})")
    graph = {s.id: [r.sub_problem for r in s.reads if r.sub_problem in subs] for s in m.sub_problems}
    seen: set[str] = set()

    def cyclic(node: str, path: tuple[str, ...]) -> bool:
        if node in path:
            out.append("a cycle of reads: " + " → ".join(path[path.index(node):] + (node,)))
            return True
        if node in seen:
            return False
        seen.add(node)
        return any(cyclic(n, path + (node,)) for n in graph.get(node, []))

    for node in graph:
        cyclic(node, ())
    out += _composition_issues(m, parts, subs)
    single = len(m.sub_problems) == 1
    if single and m.size != "single":
        out.append("one sub-problem is a single problem, not a complex one")
    if not single and m.size != "complex":
        out.append("several sub-problems make a complex problem")
    if single and (m.composition.method != "none" or m.composition.steps):
        out.append("a single problem has nothing to compose")
    if not single and m.composition.method == "none":
        out.append("sub-problems need a composition to say how their outputs combine (even 'sequence')")
    return out


def _composition_issues(m: ProblemMap, parts: dict[str, Part], subs: dict[str, SubProblem]) -> list[str]:
    """The steps are built in order: a step uses sub-problems and the outputs of steps before it, makes a name nothing else has, and
    joins units with a key. In a complex map every sub-problem nobody reads feeds a step: nothing solved is left out of the result."""
    out: list[str] = []
    made: dict[str, set[str]] = {}  # output -> the units it is on
    used: set[str] = set()
    for index, step in enumerate(m.composition.steps, start=1):
        where = f"step {index} ({step.kind} → {step.output})"
        if not step.inputs:
            out.append(f"{where} has no inputs")
        if len(step.inputs) != len(set(step.inputs)):
            out.append(f"{where} lists an input twice")
        if step.output in subs:
            out.append(f"{where} makes '{step.output}', which is a sub-problem's id")
        elif step.output in made:
            out.append(f"{where} makes '{step.output}', which an earlier step already made")
        units: set[str] = set()
        for name in step.inputs:
            if name in subs:
                used.add(name)
                unit = _unit_of(subs[name], parts)
                if unit:
                    units.add(unit)
            elif name in made:
                units |= made[name]
            else:
                later = any(c.output == name for c in m.composition.steps[index:])
                out.append(f"{where} uses '{name}', " + ("which a later step makes: steps are in order" if later
                                                         else "which is neither a sub-problem nor an earlier step's output"))
        if step.kind in ("fusion", "gate") and len(step.inputs) < 2:
            out.append(f"a {step.kind} combines at least two inputs: '{step.output}' has {len(step.inputs)}")
        if len(units) > 1 and not step.joins:
            out.append(f"{where} combines inputs on different units ({', '.join(sorted(units))}): say the key that joins them")
        if step.joins and units and step.inputs:  # the joined result is on the first input's unit
            first = step.inputs[0]
            first_units = ({u for u in [_unit_of(subs[first], parts)] if u} if first in subs else made.get(first, set()))
            made[step.output] = set(first_units) or units
        else:
            made[step.output] = units
    if len(m.sub_problems) > 1:
        read_by_another = {r.sub_problem for s in m.sub_problems for r in s.reads if r.sub_problem != s.id}
        for s in m.sub_problems:
            if s.id not in read_by_another and s.id not in used:
                out.append(f"{s.id} ({s.title}) is read by no sub-problem and feeds no composition step: its result is left out")
    return out


# ---------------------------------------------------------------------------------------------------------------- the rule decomposer

#: a fraud problem: the words name fraud (not abuse of a rule, not an abusive comment) among things people do
FRAUD = re.compile(r"\b(fraud\w*|fake\w*|scam\w*|cheat\w*|suspicious|misuse\w*|collu\w*|ghost\w*|bogus|phony|sham|staged)\b", re.I)
#: 30.6: fraud said by what people do, not by its name ("closing jobs with no movement", "accounts set up together to feed fares")
FRAUD_DEEDS = re.compile(r"\b(never (drove|made|delivered|happened)|no movement|without moving|(do|did|does) not move|not (really |actually )?(done|real|made|delivered)|"
                         r"false (orders?|rides?|deliveries)|(feed|collect|take|farm)\w* (the )?fares?|(set|opened) up together|(opened|created) together|"
                         r"finish\w* (the )?deliver\w* (very )?(quickly|without)|abuse of the fare)\b", re.I)
#: the rider-level fraud: the rides or deliveries themselves are not what they say
RIDE_LEVEL = re.compile(r"\b(fake (rides?|deliver\w*|trips?)|(rides?|deliver\w*|jobs?|trips?) (they |he |she |that )?(never|not)|never (drove|made|delivered)|"
                        r"no movement|without moving|(do|did|does) not move|ghost deliver\w*|close\w* (rides?|deliver\w*|jobs?)|finish\w* (the )?deliver\w*|"
                        r"deliveries that are not real|not really done|(rides?|deliver\w*) (that )?(look|seem)\w* (wrong|odd|off|strange)|losing money on|"
                        r"(bikers?|riders?|couriers?|drivers?) (are )?(cheat\w*|scam\w*|fak\w*|steal\w*))\b", re.I)
#: linked accounts: customers or accounts acting together with riders
RINGS = re.compile(r"\b(accounts?|customers?)\b[^.;]{0,80}\b(together|team up|teaming|work\w* with|collu\w*|groups?|connected|ring|set up|opened|share|feed\w*|fake orders?|false orders?|same few)\b|"
                   r"\b(collu\w*|groups? of (customer )?accounts|linked accounts|fake orders?|false orders?|(weird|strange|odd|suspicious) (customer )?(orders?|accounts?))\b", re.I)
#: a limit on what a team can check, said without a number too ("only part of the rides every day", "one ride in fifty", "not many")
LIMITED = re.compile(r"\b(can (only )?(check|review|look at|investigate|read|handle) (only )?(a limited number|a few|part|some|only part|roughly|about|around|maybe)|"
                     r"(check|review|look at) only (a |part|some|a few)|limited number of|ordered shortlist|shortlist|one (ride|case|order|delivery) in \w+|"
                     r"not many|only look at a few|few each day|a few (a|per|each) (day|week)|a small (slice|share|fraction|number|part) of|"
                     r"(which|what) (orders|rides|deliveries|ones|cases) to (look at|check|review)|(one )?prioriti[sz]ed list|ranked list|"
                     r"\d[\d,]*\s+(a|per|each|every)\s+(day|week|shift))\b", re.I)
FRAUD_SETTING = re.compile(r"\b(ride\w*|trip\w*|order\w*|deliver\w*|biker\w*|courier\w*|driver\w*|payment\w*|transaction\w*|account\w*|claim\w*)\b", re.I)
BUDGET = re.compile(r"\b(review budget|budget(ed)? (of|for)|(reviewers?|analysts?|investigators?|team|we|they) (can|could|only can) (check|review|look at|investigate|handle) (about |around |only |just )?(\d+(?:[.,]\d+)?\s?%?|a few|\w+ hundred)|"
                    r"(only|just) (\d+|a few) (can|could) be (checked|reviewed)|top[- ]?\d+|(\d+|a few) (cases|rides|orders|alerts) (a|per|each) (day|week|shift)|per (day|week) (we )?can (check|review))(?!\w)", re.I)
CONFIRMED = re.compile(r"\b(confirmed|known|labell?ed|verified|a few (cases|frauds)|sample of)\b", re.I)
NEGATION = re.compile(r"\b(no|not|none|never|nothing|nobody|without|n't|neither|haven't|hasn't|unlabell?ed|unconfirmed)\b|n't\b", re.I)
#: a column the person says holds the label: the outcome is a column, not a hunt (one problem, however it is worded)
STATED_LABEL = re.compile(r"\b(column|field|flag)\b[^.;]{0,40}\b(label|target|outcome)\b|\b(label|target|outcome)\b[^.;]{0,20}\b(column|field)\b|\b(label|target)\s+is\s+(the\s+)?(column|field)?\s*\w*_\w+", re.I)
ROUTING = re.compile(r"\b(rout\w+|assign\w*|triage|hand(le)? ?over|escalat\w+)\b", re.I)
TICKETS = re.compile(r"\b(ticket\w*|request\w*|case\w*|issue\w*)\b", re.I)
TEAM = re.compile(r"\b(team|queue|department|desk)\b", re.I)
URGENCY = re.compile(r"\b(urgen\w+|priorit\w+|how (fast|soon)|severity)\b", re.I)
THEN = re.compile(r"\b(and then|then|after that|next)\b", re.I)
PROMO = re.compile(r"\b(promo\w*|campaign\w*|discount\w*|offer\w*|price (cut|change)s?)\b", re.I)
KNOWN_AHEAD = re.compile(r"\b(planned|scheduled|known|set) (weeks |days |months )?(ahead|in advance|beforehand)|\bahead of time\b", re.I)
UNITS = re.compile(r"\b(rides?|trips?|orders?|deliver(?:y|ies)|transactions?|payments?|claims?|accounts?)\b", re.I)
ACTORS = re.compile(r"\b(bikers?|couriers?|drivers?)\b", re.I)
MAX_BUDGET = 200


def _family_of(words: str) -> tuple[Optional[str], str]:
    """The family the words say (None when they do not decide), and the label situation: from the problem card's reading."""
    from dclab_core.card import problem_card
    from dclab_core.understand import understand

    card = problem_card(understand(words), words=words)
    return card.family.value, (card.labels.value or "unknown")


def _clip(text: Any, n: int = MAX_TEXT) -> str:
    return " ".join(str(text or "").split())[:n]


def _single(words: str, card_family: Optional[str] = None, *, template: str = "single", notes: Optional[list[str]] = None,
            needs: Optional[list[Need]] = None, unit: Optional[str] = None) -> ProblemMap:
    family, found = _family_of(words)
    family = card_family or family or "classification"
    quote = _clip(words) or None
    return ProblemMap(size="single", parts=[Part(id="part1", title="The problem", question=(quote or ""), quote=quote)],
                      sub_problems=[SubProblem(id="sp1", part="part1", title="The problem", family=family, question=(quote or ""), needs=needs or [], unit=unit,
                                               labels=found if found in LABEL_SITUATIONS else "unknown", quote=quote, why="One question, one family: a single problem.")],
                      template=template, notes=notes if notes is not None else ["The words name one problem; the router (24.3) records why it needs no more."])


def decompose(words: str, *, parts: Optional[list[Any]] = None) -> ProblemMap:
    """A problem map from the person's words by rules. ``parts``: the brief's parts, when it has them (challenge mode's ``read_brief``).
    Every word said counts (the owner's own morning named fraud and its budget only in its answers)."""
    words = (words or "").strip()
    brief = [p for p in (parts or []) if isinstance(p, dict)]
    if len(brief) >= 2:
        return _from_brief(brief)
    family, _ = _family_of(words) if words else (None, "unknown")
    if (FRAUD.search(words) or FRAUD_DEEDS.search(words)) and FRAUD_SETTING.search(words) and family in (None, "rare_event", "anomaly") and not STATED_LABEL.search(words):
        return _fraud(words)
    if _two_step_routing(words):
        return _routing(words)
    if family == "forecasting" and PROMO.search(words):
        ahead = bool(KNOWN_AHEAD.search(words))
        note = ("The planned promotions are known before the forecast is made: they are features of it, not a second problem."
                if ahead else "Promotions are in the words: whether they are known ahead of the forecast is the question to ask.")
        needs = [Need(why="The promotions planned for each future day (a flag or a discount per product and store), known before the forecast is made.")] if ahead else []
        return _single(words, "forecasting", template="demand_with_promotions", notes=[note], needs=needs)
    return _single(words)


def _two_step_routing(words: str) -> bool:
    """A routing sentence that asks for the team and then the urgency, in that order: a team, then a connector, then an urgency."""
    if not (ROUTING.search(words) and TICKETS.search(words)):
        return False
    team, urgency = TEAM.search(words), URGENCY.search(words)
    if not (team and urgency):
        return False
    return any(team.end() <= connector.start() and connector.end() <= urgency.start() for connector in THEN.finditer(words))


def _from_brief(brief: list[dict[str, Any]]) -> ProblemMap:
    kept, clipped = brief[:MAX_PARTS], len(brief) > MAX_PARTS
    made_parts, subs = [], []
    for i, raw in enumerate(kept, start=1):
        title, text = _clip(raw.get("title"), 200) or f"Part {i}", _clip(raw.get("text"))
        family, labels = _family_of(f"{title}. {text}")
        pid, sid = f"part{i}", f"sp{i}"
        made_parts.append(Part(id=pid, title=title, question=text, quote=title))
        subs.append(SubProblem(id=sid, part=pid, title=title, question=text, family=family or ("rare_event" if raw.get("rare_event") else "classification"),
                               labels=labels if labels in LABEL_SITUATIONS else "unknown", quote=title, why="The brief's own part."))
    notes = ["The parts are the brief's own; their units, moments and any reads between them are still to be said."]
    if clipped:
        notes.append(f"The brief has {len(brief)} parts: the first {MAX_PARTS} are read.")
    return ProblemMap(size="complex", parts=made_parts, sub_problems=subs, composition=Composition(method="sequence", steps=[
        CompositionStep(kind="sequence", inputs=[s.id for s in subs], output="parts", note="The brief's parts, in its order: how they combine is the question to ask.")]),
        template="brief", notes=notes)


def _labels_in(words: str) -> str:
    """What the words say of confirmed cases: ``file`` when a clause says some exist, ``none`` when every clause that mentions them
    denies them, ``unknown`` when they say nothing (unknown wins when in doubt)."""
    positive = negative = False
    for clause in re.split(r"[.;,!?]|\bbut\b|\band\b", words, flags=re.I):
        found = CONFIRMED.search(clause)
        if not found:
            continue
        before = " ".join(clause[:found.start()].split()[-4:])  # 30.6: the denial is said of the cases, near them ("no movement" later denies nothing)
        if NEGATION.search(before) or re.search(r"\b(no|not|none|never|nothing|nobody)\b", clause[found.end():found.end() + 12], re.I) and not re.search(r"\bcases\b", clause[found.end():found.end() + 8], re.I):
            negative = True
        else:
            positive = True
    return "file" if positive else "none" if negative else "unknown"


def _plural(noun: str) -> str:
    return noun[:-1] + "ies" if noun.endswith("y") else noun + "s"


def _unit_of_words(words: str) -> str:
    """What one row of a fraud problem is: the first thing it names (a ride, an order, a claim), else the actor, else a case."""
    found = UNITS.search(words)
    if found:
        word = found.group(0).lower()
        return "delivery" if word.startswith("deliver") else word.rstrip("s") if not word.endswith("ss") else word
    actor = ACTORS.search(words)
    return actor.group(0).lower().rstrip("s") if actor else "case"


def _assemble(pieces: list[dict[str, Any]], steps: list[dict[str, Any]], *, quote: Optional[str], why: str, method: Method, budget: Optional[str],
              template: str, notes: list[str]) -> ProblemMap:
    """Number the pieces in order, point their reads and the steps' inputs at the numbers, and build the map."""
    ids = {piece["key"]: f"sp{i}" for i, piece in enumerate(pieces, start=1)}
    parts, subs = [], []
    for i, piece in enumerate(pieces, start=1):
        pid, sid = f"part{i}", ids[piece["key"]]
        part = dict(piece["part"])
        sub = dict(piece["sub"])
        part.setdefault("why", why)
        parts.append(Part(id=pid, quote=quote, **part))
        reads = [Read(sub_problem=ids[key], what=what, via=via) for key, what, via in piece.get("reads", [])]
        subs.append(SubProblem(id=sid, part=pid, question=part.get("question", ""), outcome=part.get("outcome"), unit=part.get("unit"), moment=part.get("moment"),
                               reads=reads, quote=quote, why=part["why"], **sub))
    made = []
    for step in steps:
        made.append(CompositionStep(kind=step["kind"], inputs=[ids.get(x, x) for x in step["inputs"]], output=step["output"], joins=step.get("joins"), note=step.get("note", "")))
    return ProblemMap(size="complex", parts=parts, sub_problems=subs, composition=Composition(method=method, steps=made, budget=budget), template=template, notes=notes)


def _fraud(words: str) -> ProblemMap:
    found = BUDGET.search(words) or LIMITED.search(words)
    budget = _clip(found.group(0), MAX_BUDGET) if found else None
    labels = _labels_in(words)
    u = _unit_of_words(words)
    rides, rings_said, limited = bool(RIDE_LEVEL.search(words) or FRAUD.search(words) and not RINGS.search(words)), bool(RINGS.search(words)), bool(found)
    if not (rides or rings_said):
        rides = True  # fraud named with nothing more said: the rides themselves are what is screened
    # 30.6: the parts the person's words support, and only those (the independent sealed people and the owner's own morning agree; the
    # earlier rule drew the linked accounts and a review list for every fraud sentence, whatever it said)
    if rings_said and not rides and not limited:
        return _single(words, "entity_tables", template="fraud_rings", unit="account",
                       notes=["The words ask only for the linked accounts: one problem, not the whole fraud programme."])
    if rides and not rings_said and not limited:
        family = "anomaly" if labels == "none" else "rare_event"
        return _single(words, family, template="fraud_rides", unit=u, notes=["The words ask only about the rides themselves, with no linked accounts and no review limit: "
                                                                      "one problem" + (", learned from the confirmed cases." if family == "rare_event" else ", screened without labels.")
                                                                      + (" Whether any case was confirmed is the question to ask." if labels == "unknown" else "")])
    quote = _clip(words) or None
    u = _unit_of_words(words)
    us = _plural(u)
    moment = None  # the words do not say when the answers are needed: the reviewer asks once, for every part
    why = (f"Fraud among {us}: a screen that needs no labels, a classifier on the confirmed cases (when there are any), the linked accounts, "
           "and the cases a team can review.")
    via = None if u == "account" else f"the account of the {u}"
    screen = {"key": "screen",
              "part": dict(title="Screen for unusual behaviour", question=f"Which {us} look unlike the rest, without any label?", unit=u, moment=moment,
                           outcome=f"An anomaly score per {u}", risks=["Unusual is not fraud: busy places and new people look unusual."],
                           why="A screen needs no labels: it finds what looks unlike the rest, even when nothing else is marked."),
              "sub": dict(title=f"Unusual {us}", family="anomaly", kind="anomaly", predict_rank=1, labels="none",
                          needs=[Need(table=us, why=f"Times, places, amounts and the people of each {u}.", label="none")])}
    classifier = {"key": "classifier",
                  "part": dict(title="Classify the confirmed kinds of fraud", question=f"Which {us} are like the cases already confirmed?", unit=u, moment=moment,
                               outcome="Confirmed fraud (partly known)", risks=[f"Only some frauds were found: an unmarked {u} is unknown, not clean.",
                                                                                "Rows cluster on the same person: a random split lets the model recognise them."],
                               why="Some cases were confirmed: a classifier learns what they look like and reads the screen's score." if labels != "none"
                               else "The words say no case was confirmed."),
                  "sub": dict(title="Confirmed fraud", family="rare_event", kind="rare_event", predict_rank=1, known_rank=3, labels=labels,
                              needs=[Need(table=us, why=f"The same {us} as the screen."), Need(table="known cases", label="file", why="The confirmed cases, with the key that finds each one.")]),
                  "reads": [("screen", "score", None)]}
    rings = {"key": "rings",
             "part": dict(title="Find rings of linked accounts", question="Which phones, accounts or people work together?", unit="account", moment=moment,
                          outcome="Membership of a ring", risks=["A phone or an account may appear in two roles (sender and receiver) in different tables."],
                          why="Fraud often works through linked accounts: phones, accounts or people that appear together."),
             "sub": dict(title="Linked accounts", family="entity_tables", kind="entity_tables", predict_rank=1, labels="none",
                         needs=[Need(table=us, why=f"The phone, account and person columns of each {u}, to join {us} to the accounts they share.")])}
    review = {"key": "review",
              "part": dict(title="Choose what the team reviews", question="Which cases should the review budget go to?", unit=u, moment=moment,
                           outcome="A ranked list of the size of the budget", risks=["The budget, not the score, decides how many are checked."],
                           why="A team can only check so many: the list is cut to the budget in the words." if budget else "A team can only check so many: the size of the list is the question to ask."),
              "sub": dict(title="The review list", family="ranking", kind="ranking", predict_rank=2, known_rank=3, labels="unknown",
                          needs=[Need(why="How many cases the team can check in a day, and what makes a case worth checking first.")])}
    with_classifier = labels != "none" and rides
    pieces = ([screen] if rides else []) + ([classifier] if with_classifier else []) + ([rings] if rings_said else []) + ([review] if limited else [])
    scores = (["screen"] if rides else []) + (["classifier"] if with_classifier else []) + (["rings"] if rings_said else [])
    steps = ([dict(kind="fusion", inputs=scores, output="risk", joins=via if rings_said else None, note=f"One risk score per {u} from {', '.join(scores)}.")]
             if len(scores) > 1 else [])  # the screen and the classifier are on the same unit: a key joins them to the accounts only
    if limited:
        steps.append(dict(kind="budget", inputs=[("risk" if len(scores) > 1 else scores[0]), "review"], output="review_list",
                          note=(budget or "A review budget: how many a team can check") + "."))
    notes = ["The confirmed cases are partial: the classifier learns from them, the screen does not need them, and the holdout is read once, at the end of the composition."
             if with_classifier else "The words say no case was confirmed: there is no classifier to train, and the screen and the rings stand in its place.",
             ("Linked accounts are said: whether the data has what finds them is the question to ask." if rings_said
              else f"No linked accounts are said: the {us} are judged on their own.")]
    if labels == "unknown":
        notes.append("The words do not say whether any case was confirmed: the classifier stays, and whether its labels exist is the question to ask.")
    return _assemble(pieces, steps, quote=quote, why=why, method="mixed", budget=budget, template="fraud", notes=notes)


def _routing(words: str) -> ProblemMap:
    quote = _clip(words) or None
    why = "Routing asked in two steps: the team first, then how urgent it is given the team."
    pieces = [{"key": "team",
               "part": dict(title="Route to a team", question="Which team should handle the ticket?", unit="ticket", moment=None, outcome="The team that handles it",
                         why="The team comes first: the urgency is judged given the team."),
               "sub": dict(title="Team", family="classification", kind="multiclass", predict_rank=1, known_rank=3, labels="column")},
              {"key": "urgency",
               "part": dict(title="Set the urgency", question="How urgent is the ticket, once its team is known?", unit="ticket", moment=None, outcome="The urgency",
                         why="The urgency is asked once the team is known: it reads the first part's score, never its outcome."),
               "sub": dict(title="Urgency", family="classification", kind="multiclass", predict_rank=2, known_rank=3, labels="column"),
               "reads": [("team", "score", None)]}]
    steps = [dict(kind="sequence", inputs=["team", "urgency"], output="routed_ticket", note="The urgency model is given the team the first one chose.")]
    return _assemble(pieces, steps, quote=quote, why=why, method="sequence", budget=None, template="ticket_routing",
                     notes=["The team that handled the ticket is only known after it: the second step reads the first's score, never its outcome."])


__all__ = ["Family", "Need", "Read", "Part", "SubProblem", "CompositionStep", "Composition", "ProblemMap", "validate", "decompose", "fingerprint", "LIBRARY_ONLY", "MAX_TEXT"]

import dclab_core.mapping  # noqa: E402,F401  (the reviewer's contract, 24.2: registered once the map is defined; either module may be imported first)
import dclab_core.solutions  # noqa: E402,F401  (the route contract, 24.3: registered once the map is defined; either module may be imported first)
