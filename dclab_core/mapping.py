"""The understanding agents of phase 2 (package 24.2): the decomposer, the map reviewer and the data-need agent, and the person's edits.

Three roles stand beside the problem analyst and the conversation reader, each in the pattern every role here follows (evidence of
names and aggregates, never values; a typed output parsed behind a validator; the rules as the fallback; a prompt file):

* the **decomposer** draws a :class:`~dclab_core.problem_map.ProblemMap` from the person's words. Every part quotes the person's own
  words or names a rule template; a unit and a moment are kept only as the person's words (a moment the model wrote is where leakage
  starts); a family is one of the card's vocabulary; the map is refused whole unless ``problem_map.validate`` finds nothing;
* the **map reviewer** asks what the map misses (a unit nobody gave, labels nobody mentioned, an unmarked row taken as clean, a
  review budget nobody stated, data the parts need that the tables do not hold). Its output is questions for the person, never an
  edit; a model's questions are advisory (they never block), the rules' do;
* the **data-need agent** says, per sub-problem, which table and which columns it needs, by name. A column or a table the data card
  does not hold is dropped, and what the tables lack is found by :func:`coverage`, in code.

The rules run with no model (``review_rules``, ``coverage``, ``fill_needs``), and the person's edits (``add_part``, ``remove_part``,
``merge_parts``) are pure functions that return a new map or raise :class:`MapEditRefused` with the validator's reasons.
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Literal, Mapping, Optional, Sequence

from pydantic import Field

from dclab_core import problem_map as pm
from dclab_core.agents import Role
from dclab_core.bands import FAMILIES, LABEL_SITUATIONS
from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version
from dclab_core.problem_map import (Composition, CompositionStep, Need, Part, ProblemMap, Read, SubProblem, decompose, validate)

QuestionKind = Literal["leak", "risk", "missing_data", "ambiguity", "unit", "moment", "labels", "composition"]
MAX_QUESTIONS = 20
MAX_MODEL_QUESTIONS = 6
MAX_MODEL_PARTS = 8
MAX_MODEL_SUBS = 12
TEMPLATES = frozenset({"fraud", "ticket_routing", "demand_with_promotions", "brief"})  # the rule templates a part may name
NEEDS_LABELS = frozenset({"classification", "regression", "rare_event", "forecasting", "time_to_event", "uplift"})  # families that learn from an outcome


def quoted(quote: Any, text: str) -> bool:
    """The person's own words (``understand.quoted``, imported when used: ``contracts`` loads this module before ``understand``)."""
    from dclab_core.understand import quoted as words_of

    return words_of(quote, text)


class MapQuestion(Contract):
    """One thing to ask the person about the map, and what it rests on."""

    code: str = Field(pattern=r"^[a-z_]{3,40}$", description="A stable name for the kind of question, so a repeat is recognised.")
    kind: QuestionKind
    target: Optional[str] = Field(default=None, pattern=r"^(part|sp)\d+$", description="The part or sub-problem it is about, or the whole map.")
    question: str = Field(min_length=3, max_length=300)
    why: str = Field(default="", max_length=300, description="What it rests on: the person's words, the map, or the data card.")
    blocking: bool = Field(default=False, description="Only the rules block: the map is not confirmed while one is open.")
    source: Literal["rules", "model"] = "rules"


class MapReview(Versioned):
    """What the reviewer asks about one map (by its fingerprint)."""

    schema_version: Literal["dclab_core.map_review.v1"] = _version("map_review")
    map: str = Field(max_length=64)
    questions: list[MapQuestion] = Field(default_factory=list, max_length=MAX_QUESTIONS)
    ready: bool = Field(description="No blocking question is open.")
    source: Literal["rules", "advisor"] = "rules"


CONTRACTS["map_review"] = MapReview


class MapEditRefused(ValueError):
    """A person's edit would make an unsound map: the validator's reasons, in words."""

    def __init__(self, issues: list[str]):
        super().__init__("; ".join(issues))
        self.issues = issues


# ---------------------------------------------------------------------------------------------------------------- the data card

def _tokens(text: Any) -> set[str]:
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in re.findall(r"[a-z0-9]+", str(text or "").lower())} - {"table", "file", "csv", "parquet", "the", "of", "and"}


def _tables(data: Any) -> list[Any]:
    return list(getattr(data, "tables", None) or (data.get("tables") if isinstance(data, Mapping) else None) or [])


def _get(item: Any, name: str, default: Any = None) -> Any:
    return item.get(name, default) if isinstance(item, Mapping) else getattr(item, name, default)


def _columns(table: Any) -> list[Any]:
    return list(_get(table, "columns") or [])


def find_table(name: Optional[str], data: Any, assume_main: bool = False) -> Optional[Any]:
    """The table of the data card a need names: by name, else by the words they share, else the labels table for 'known cases'. With
    ``assume_main`` (the rules' own needs, named after the unit: 'rides' for a file called anything) the only main table when nothing else
    fits; a model's need never gets it (a table that is not like any is no need the data can meet). None when no table fits."""
    tables = _tables(data)
    if not tables or not name:
        return None
    for t in tables:
        if str(_get(t, "name")).lower() == name.lower():
            return t
    wanted = _tokens(name)
    if wanted & {"case", "label", "known", "confirmed"}:  # a file of known cases is a labels table: never the main table assumed in its place
        labels = [t for t in tables if _get(t, "role") == "labels"]
        return labels[0] if labels else None
    scored = sorted(((len(wanted & _tokens(_get(t, "name"))), t) for t in tables), key=lambda x: -x[0])
    if scored and scored[0][0] > 0:
        return scored[0][1]
    mains = [t for t in tables if _get(t, "role", "main") == "main"]
    return mains[0] if assume_main and len(mains) == 1 else None


def _suggest(table: Any, family: str, limit: int = 12) -> list[str]:
    """Columns of a table that fit a family, by the purposes the data card gave them (names only, never a value)."""
    by: dict[str, list[str]] = {}
    for c in _columns(table):
        by.setdefault(str(_get(c, "purpose") or "feature"), []).append(str(_get(c, "name")))
    order = {"entity_tables": ("identifier", "entity", "time"), "anomaly": ("time", "entity", "place", "feature"),
             "rare_event": ("entity", "time", "feature", "place"), "forecasting": ("time", "entity", "target_candidate", "feature"),
             "ranking": ("entity", "time", "feature")}.get(family, ("time", "entity", "feature", "place"))
    out: list[str] = []
    for purpose in order:
        out += [n for n in by.get(purpose, []) if n not in out]
    return out[:limit]


def coverage(m: ProblemMap, data: Any) -> list[dict[str, Any]]:
    """What the sub-problems need that the data card does not hold: a table that is not there, columns a need names that its table lacks,
    labels a sub-problem says exist (a file or a column) that no table or candidate holds. Names and counts only."""
    out: list[dict[str, Any]] = []
    tables = _tables(data)
    for s in m.sub_problems:
        for need in s.needs:
            table = find_table(need.table, data, assume_main=True) if need.table else None
            if need.table and table is None and need.label != "file":
                out.append({"sub_problem": s.id, "code": "table_absent", "table": need.table, "columns": [],
                            "question": f"'{s.title}' needs {need.table}, but no uploaded table is like it: upload it, or say this part cannot be solved."})
                continue
            if table is not None and need.columns:
                have = {str(_get(c, "name")) for c in _columns(table)}
                gone = [c for c in need.columns if c not in have]
                if gone:
                    out.append({"sub_problem": s.id, "code": "columns_absent", "table": str(_get(table, "name")), "columns": gone[:10],
                                "question": f"'{s.title}' needs the column(s) {', '.join(gone[:5])} of {_get(table, 'name')}, which it does not have: is there another name for them?"})
            if need.label == "file" and not any(_get(t, "role") == "labels" for t in tables):
                out.append({"sub_problem": s.id, "code": "labels_absent", "table": None, "columns": [],
                            "question": f"'{s.title}' learns from a file of confirmed cases, but none was uploaded: upload it, or say there is none."})
            if need.label == "column" and not any(_get(t, "target_candidates") for t in tables):
                out.append({"sub_problem": s.id, "code": "labels_absent", "table": None, "columns": [],
                            "question": f"'{s.title}' learns from a column that marks the outcome, but none of the tables has one that can be: which column is it?"})
    return out


def fill_needs(m: ProblemMap, data: Any) -> ProblemMap:
    """The same map with each need pointed at the table it names (its real name) and, where it names no column, the columns that fit its
    family by purpose. Nothing is invented: a need with no table in the data keeps its words and shows in ``coverage``."""
    changed = False
    subs = []
    for s in m.sub_problems:
        needs = []
        for need in s.needs:
            table = find_table(need.table, data, assume_main=True) if need.table else None
            if table is not None:
                columns = need.columns or (_suggest(table, s.family) if need.label != "file" else [])
                new = need.model_copy(update={"table": str(_get(table, "name"))[:200], "columns": columns})
                changed = changed or new != need
                needs.append(new)
            else:
                needs.append(need)
        subs.append(s.model_copy(update={"needs": needs}))
    return pm.ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": "", "sub_problems": [s.model_dump(mode="json") for s in subs]}) if changed else m


# ---------------------------------------------------------------------------------------------------------------- the reviewer's rules

def review_rules(m: ProblemMap, data: Any = None) -> list[MapQuestion]:
    """What the rules ask about a map. A small problem is the problem card's to question (its unit, moment and labels are asked there):
    for it only an unsound map is raised. A complex one is asked what the parts leave unsaid. Issues of ``validate`` block; the rest do not."""
    out: list[MapQuestion] = []
    for issue in validate(m):
        leak = "leak" in issue
        out.append(MapQuestion(code="map_unsound", kind="leak" if leak else "ambiguity", question=f"The map has a problem: {issue}. How should it change?"[:300],
                               why="the map's own rules (validate)", blocking=True))
    if m.size == "single":
        return out[:MAX_QUESTIONS]
    for p in m.parts:
        if not p.unit:
            out.append(MapQuestion(code="unit_missing", kind="unit", target=p.id, question=f"What is one row of '{p.title}'?"[:300], why="the part names no unit"))
    if not any(p.moment for p in m.parts):  # no part says it: one question for them all
        out.append(MapQuestion(code="moment_missing", kind="moment", question="When are the answers needed, and what is known by then?",
                               why="no part names a moment: a wrong one lets the future in"))
    else:
        for p in m.parts:
            if not p.moment:
                out.append(MapQuestion(code="moment_missing", kind="moment", target=p.id,
                                       question=f"When is the answer to '{p.title}' needed, and what is known then?"[:300], why="the part names no moment: a wrong one lets the future in"))
    for s in m.sub_problems:
        if s.labels == "unknown" and s.family in NEEDS_LABELS:
            out.append(MapQuestion(code="labels_unknown", kind="labels", target=s.id, question=f"Are the outcomes of '{s.title}' known: in a column, in a file of known cases, or not at all?"[:300],
                                   why=f"{s.family} learns from an outcome and the words do not say where it is"))
        if s.family == "rare_event" and s.labels in ("file", "unknown"):
            unit = s.unit or next((p.unit for p in m.parts if p.id == s.part and p.unit), None) or "case"
            out.append(MapQuestion(code="unmarked_is_unknown", kind="risk", target=s.id,
                                   question=f"Is a {unit} no one marked as fraud clean, or only not checked yet?"[:300],
                                   why="only some cases were found: treating the rest as clean teaches the model to miss what was never looked for"))
    if any(c.kind == "budget" for c in m.composition.steps) and not m.composition.budget:
        out.append(MapQuestion(code="budget_missing", kind="composition", question="How many cases can your team review in a day?", why="a budget step cuts the list to a size nobody stated"))
    if data is not None:
        for gap in coverage(m, data):
            out.append(MapQuestion(code=f"data_{gap['code']}"[:40], kind="missing_data", target=gap["sub_problem"], question=gap["question"][:300],
                                   why="the data card: " + (f"{gap['table']}" if gap.get("table") else "the uploaded tables")))
    return out[:MAX_QUESTIONS]


def _same(a: MapQuestion, b: MapQuestion) -> bool:
    """The same question: the same text, or the same rule's code about the same thing (a model's questions share one code and differ by text)."""
    text = lambda q: re.sub(r"\W+", " ", q.question.lower()).strip()  # noqa: E731
    return text(a) == text(b) or ((a.kind, a.target, a.code) == (b.kind, b.target, b.code) and a.code != "model_question")


def _merge_questions(rules: list[MapQuestion], model: list[MapQuestion]) -> list[MapQuestion]:
    out = list(rules)
    for q in model:
        if not any(_same(q, r) for r in out):
            out.append(q)
    return out[:MAX_QUESTIONS]


def review(m: ProblemMap, data: Any = None, extra: Optional[Iterable[MapQuestion]] = None, source: Literal["rules", "advisor"] = "rules") -> MapReview:
    """A map's review: the rules' questions and any extra (the model's), the map ready when none blocks."""
    questions = _merge_questions(review_rules(m, data), list(extra or []))
    return MapReview(map=m.fingerprint, questions=questions, ready=not any(q.blocking for q in questions), source=source)


# ---------------------------------------------------------------------------------------------------------------- the person's edits

def _renamed(m: ProblemMap, **changes: Any) -> ProblemMap:
    """A new map from this one with fields replaced, as a person's edit: its source is the person, and its fingerprint is made again."""
    from pydantic import ValidationError

    data = {**m.model_dump(mode="json"), "fingerprint": "", "source": "person", "template": "edited", **changes}
    try:
        return ProblemMap.model_validate(data)
    except ValidationError as error:  # a value over a limit is a refused edit with its reason, not a crash
        raise MapEditRefused([f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}"[:300] for e in error.errors()][:5]) from None


def _sound(new: ProblemMap) -> ProblemMap:
    issues = validate(new)
    if issues:
        raise MapEditRefused(issues)
    return new


def _next_number(prefix: str, ids: Iterable[str]) -> int:
    return max([int(i[len(prefix):]) for i in ids if i.startswith(prefix) and i[len(prefix):].isdigit()] + [0]) + 1


def _text(name: str, value: Any, limit: int) -> Optional[str]:
    """A person's text for a field: None when empty, refused when it is not text or over its limit (a long value is never cut into another meaning)."""
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise MapEditRefused([f"{name} is text"])
    value = " ".join(value.split())
    if len(value) > limit:
        raise MapEditRefused([f"{name} is at most {limit} characters"])
    return value or None


def _tidy(steps: list[dict[str, Any]], gone: set[str]) -> list[dict[str, Any]]:
    """The steps without the sub-problems that left. A step left with nothing is dropped. A fusion or a gate left with one input
    combines nothing: it is dropped, its output becomes that input in the steps after it, and the key it joined on goes to the step
    that now meets the units (a person removing parts one by one reaches a map that holds)."""
    out: list[dict[str, Any]] = []
    alias: dict[str, str] = {}
    carried: dict[str, str] = {}  # an input name -> the key the dropped step joined it on
    for step in steps:
        inputs = []
        for name in step["inputs"]:
            name = alias.get(name, name)
            if name not in gone and name not in inputs:
                inputs.append(name)
        if not inputs:
            alias[step["output"]] = ""  # nothing is left of it
            continue
        inherited = next((carried[i] for i in inputs if i in carried), None)
        if step["kind"] in ("fusion", "gate") and len(inputs) == 1:
            alias[step["output"]] = inputs[0]
            if step.get("joins") or inherited:
                carried[inputs[0]] = step.get("joins") or inherited
            continue
        joins = step.get("joins") or inherited
        out.append({**step, "inputs": inputs, "joins": joins})
    return [{**s, "inputs": [i for i in s["inputs"] if i]} for s in out if any(s["inputs"])]


def _as_map(m: ProblemMap, parts: list[dict[str, Any]], subs: list[dict[str, Any]], steps: list[dict[str, Any]], budget: Optional[str], note: str) -> ProblemMap:
    single = len(subs) == 1
    if single:
        steps, method = [], "none"
    else:
        method = m.composition.method if m.composition.method != "none" else "sequence"
    composition = {"method": method, "steps": steps, "budget": budget}
    return _sound(_renamed(m, size="single" if single else "complex", parts=parts, sub_problems=subs, composition=composition, notes=[*m.notes, note][-20:]))


def remove_part(m: ProblemMap, part_id: str) -> ProblemMap:
    """The map without a part and its sub-problems; what read them stops reading them and the steps lose them. Refused with the reasons
    when the rest is not sound (a fusion left with one input, a unit nothing joins now)."""
    if part_id not in {p.id for p in m.parts}:
        raise MapEditRefused([f"{part_id} is not a part of this map"])
    gone = {s.id for s in m.sub_problems if s.part == part_id}
    parts = [p.model_dump(mode="json") for p in m.parts if p.id != part_id]
    if not parts:
        raise MapEditRefused(["a map keeps at least one part"])
    subs = []
    for s in m.sub_problems:
        if s.id in gone:
            continue
        d = s.model_dump(mode="json")
        d["reads"] = [r for r in d["reads"] if r["sub_problem"] not in gone]
        subs.append(d)
    steps = _tidy([c.model_dump(mode="json") for c in m.composition.steps], gone)
    return _as_map(m, parts, subs, steps, m.composition.budget, f"The person removed {part_id}.")


def add_part(m: ProblemMap, title: str, family: str, *, question: str = "", unit: Optional[str] = None, moment: Optional[str] = None, labels: str = "unknown",
             quote: Optional[str] = None, joins: Optional[str] = None, into: Optional[str] = None, avoid: Iterable[str] = ()) -> ProblemMap:
    """The map with one more part and its sub-problem. It joins the composition at the step named by ``into`` (its output), else the first step
    that combines (a fusion, a gate or a sequence), else a sequence of the old problem and the new part is made; ``joins`` is the key when the
    units differ. ``avoid``: ids that were used and removed, never reused (a decision can still point at one). A value over its limit is refused."""
    if family not in FAMILIES:
        raise MapEditRefused([f"{family} is not a family of problem; one of: {', '.join(FAMILIES)}"])
    if labels not in LABEL_SITUATIONS:
        raise MapEditRefused([f"{labels} is not a label situation; one of: {', '.join(LABEL_SITUATIONS)}"])
    title = _text("the title", title, 200)
    if not title:
        raise MapEditRefused(["a part needs a title"])
    question, unit, moment = _text("the question", question, pm.MAX_TEXT) or "", _text("the unit", unit, 120), _text("the moment", moment, pm.MAX_TEXT)
    quote, joins = _text("the quote", quote, pm.MAX_TEXT), _text("the key", joins, pm.MAX_TEXT)
    used = [*(p.id for p in m.parts), *(s.id for s in m.sub_problems), *avoid]
    pid, sid = f"part{_next_number('part', used)}", f"sp{_next_number('sp', used)}"
    if len(m.parts) >= pm.MAX_PARTS:
        raise MapEditRefused([f"a map has at most {pm.MAX_PARTS} parts"])
    parts = [p.model_dump(mode="json") for p in m.parts] + [Part(id=pid, title=title, question=question, unit=unit, moment=moment, quote=quote,
                                                                  why="Added by the person.").model_dump(mode="json")]
    subs = [s.model_dump(mode="json") for s in m.sub_problems] + [SubProblem(id=sid, part=pid, title=title, family=family, question=question, unit=unit, moment=moment,
                                                                              predict_rank=max([s.predict_rank for s in m.sub_problems] + [0]), labels=labels, quote=quote,
                                                                              why="Added by the person.").model_dump(mode="json")]
    steps = [c.model_dump(mode="json") for c in m.composition.steps]
    if into is not None:
        target = next((c for c in steps if c["output"] == into), None)
        if target is None:
            raise MapEditRefused([f"no step makes '{into}'"])
    else:
        target = next((c for c in steps if c["kind"] in ("fusion", "gate", "sequence")), None)
    if target is not None:
        known = {u for name in target["inputs"] for u in [next((x.unit or next((p.unit for p in m.parts if p.id == x.part), None) for x in m.sub_problems if x.id == name), None)] if u}
        if unit and known and unit not in known and not joins:
            raise MapEditRefused([f"the new part works on {unit}, the step on {', '.join(sorted(known))}: say the key that joins them"])
        target["inputs"] = [*target["inputs"], sid]
        if joins:
            target["joins"] = f"{target['joins']}; {joins}"[:pm.MAX_TEXT] if target.get("joins") else joins
    else:
        first = m.sub_problems[0].id
        steps = [{"kind": "sequence", "inputs": [first, sid], "output": "parts", "joins": joins, "note": "The problem and the part the person added."}]
    return _as_map(m, parts, subs, steps, m.composition.budget, f"The person added {pid}.")


def merge_parts(m: ProblemMap, keep: str, other: str) -> ProblemMap:
    """Two parts as one: the sub-problems of ``other`` join ``keep``. Refused when the parts' units or moments differ (one part, one unit)."""
    ids = {p.id for p in m.parts}
    if keep not in ids or other not in ids or keep == other:
        raise MapEditRefused([f"{keep} and {other} are not two parts of this map"])
    first, second = (next(p for p in m.parts if p.id == i) for i in (keep, other))
    differ = [name for name, a, b in (("unit", first.unit, second.unit), ("moment", first.moment, second.moment)) if (a or None) != (b or None)]
    if differ:  # one part, one unit and one moment: neither is adopted from the other part, which would change what a sub-problem means
        raise MapEditRefused([f"{keep} and {other} differ in their {' and '.join(differ)}: one part, one {' and one '.join(differ)}"])
    parts = [p.model_dump(mode="json") for p in m.parts if p.id != other]
    for p in parts:
        if p["id"] == keep:
            p["title"] = f"{first.title} and {second.title}"[:200]
            p["question"] = " ".join(x for x in (first.question, second.question) if x)[:pm.MAX_TEXT]
            p["risks"] = [*first.risks, *second.risks][:10]
            p["why"] = f"{first.why} {second.why}".strip()[:pm.MAX_TEXT]
    subs = []
    for s in m.sub_problems:
        d = s.model_dump(mode="json")
        if d["part"] == other:
            d["part"] = keep
        subs.append(d)
    steps = [c.model_dump(mode="json") for c in m.composition.steps]
    return _as_map(m, parts, subs, steps, m.composition.budget, f"The person merged {other} into {keep}.")


# ---------------------------------------------------------------------------------------------------------------- summaries for evidence

def lines(m: ProblemMap) -> list[str]:
    """The map as short lines (one per sub-problem, one per step): names and structure, never a value."""
    parts = {p.id: p for p in m.parts}
    out = []
    for s in m.sub_problems:
        part = parts.get(s.part)
        reads = ", ".join(f"{r.sub_problem}:{r.what}" + (f" via {r.via[:60]}" if r.via else "") for r in s.reads) or "-"
        out.append(f"{s.id} [{s.part} '{(part.title if part else '')[:60]}'] {s.family} labels={s.labels} unit={s.unit or (part.unit if part else None) or '?'} "
                   f"predict={s.predict_rank} known={s.known_rank if s.known_rank is not None else 'never'} reads={reads}"[:400])
    out += [f"step {i}: {c.kind} {','.join(c.inputs)} -> {c.output}" + (f" joins {c.joins[:60]}" if c.joins else "") for i, c in enumerate(m.composition.steps, start=1)]
    if m.composition.budget:
        out.append(f"budget: {m.composition.budget[:200]}")
    return out[:60]


def table_lines(data: Any, limit: int = 40) -> dict[str, list[str]]:
    """Each table of a data card as 'column:purpose' strings (at most ``limit`` columns of at most four tables)."""
    return {str(_get(t, "name"))[:80]: [f"{_get(c, 'name')}:{_get(c, 'purpose') or 'feature'}"[:100] for c in _columns(t)[:limit]] for t in _tables(data)[:4]}


# ---------------------------------------------------------------------------------------------------------------- the decomposer

def _decomposer_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    card = inputs.get("card") or {}
    rules = inputs["rules"]
    return {"words": str(inputs["words"])[:1500],
            "card": {k: str(card.get(k))[:200] for k in ("family", "labels", "unit", "moment") if card.get(k)},
            "rules_read": lines(rules), "rules_template": rules.template or "",
            "tables": table_lines(inputs.get("data")) if inputs.get("data") is not None else {},
            "families": list(FAMILIES), "labels": list(LABEL_SITUATIONS), "templates": sorted(TEMPLATES),
            "step_kinds": list(pm.StepKind.__args__), "reads": list(pm.Reads.__args__),
            **({"similar_problems": [str(x)[:480] for x in inputs["examples"][:3]]} if inputs.get("examples") else {})}


def _grounded(item: Mapping[str, Any], words: str, template: Optional[str]) -> bool:
    """A part or sub-problem rests on the person's words (a quote) or on the template the rules found in them: a rule is never named that the
    rules did not find (a model cannot add a fraud part to a churn question by saying 'fraud')."""
    return quoted(item.get("quote"), words) or (template in TEMPLATES and item.get("rule") == template)


def _free(value: Any, words: str) -> Optional[str]:
    """A unit, a moment or an outcome is kept only as the person's own words."""
    return str(value).strip()[:pm.MAX_TEXT] if isinstance(value, str) and value.strip() and quoted(value, words) else None


def _decomposer_parse(reply: Mapping[str, Any], inputs: Mapping[str, Any]) -> ProblemMap:
    """The model's map, kept only as far as it rests on the person's words: its parts, its families, what each reads and how they combine. What
    decides a leak (the labels, the moment each prediction is made and when its outcome is known) is never the model's to write: the moment of a
    sub-problem follows from what it reads (after all of it), a sub-problem of a family the rules also drew takes the rules' labels and the fact
    that its outcome is known late, any other has unknown labels and no outcome known in time (so a read of its outcome is refused)."""
    words = str(inputs["words"])
    rules: ProblemMap = inputs["rules"]
    template = rules.template
    twins: dict[str, list[SubProblem]] = {}
    for rs in rules.sub_problems:
        twins.setdefault(rs.family, []).append(rs)
    raw_parts, raw_subs = reply.get("parts"), reply.get("sub_problems")
    if not isinstance(raw_parts, list) or not isinstance(raw_subs, list):
        raise ValueError("parts and sub_problems are lists")
    part_ids: dict[int, str] = {}
    parts: list[Part] = []
    for number, item in enumerate(raw_parts[:MAX_MODEL_PARTS], start=1):
        if not isinstance(item, Mapping) or not isinstance(item.get("title"), str) or not item["title"].strip() or not _grounded(item, words, template):
            continue  # nothing the person did not say and no rule found in their words
        pid = f"part{len(parts) + 1}"
        part_ids[number] = pid
        parts.append(Part(id=pid, title=item["title"].strip()[:200], question=str(item.get("question") or "")[:pm.MAX_TEXT] if isinstance(item.get("question"), str) else "",
                          unit=_free(item.get("unit"), words), moment=_free(item.get("moment"), words), outcome=_free(item.get("outcome"), words),
                          quote=str(item["quote"])[:pm.MAX_TEXT] if quoted(item.get("quote"), words) else None,
                          why=f"rule template: {template}" if item.get("rule") == template and template in TEMPLATES else ""))
    sub_ids: dict[int, str] = {}
    subs: list[SubProblem] = []
    pending: list[tuple[str, Any]] = []
    for number, item in enumerate(raw_subs[:MAX_MODEL_SUBS], start=1):
        number_of_part = item.get("part") if isinstance(item, Mapping) else None
        if (not isinstance(item, Mapping) or isinstance(number_of_part, bool) or part_ids.get(number_of_part) is None or item.get("family") not in FAMILIES
                or not _grounded(item, words, template)):
            continue
        sid = f"sp{len(subs) + 1}"
        sub_ids[number] = sid
        part = next(p for p in parts if p.id == part_ids[number_of_part])
        twin = twins[item["family"]].pop(0) if twins.get(item["family"]) else None  # the rules' reading of the same family, in order
        subs.append(SubProblem(id=sid, part=part.id, title=(str(item["title"]).strip()[:200] if isinstance(item.get("title"), str) and item["title"].strip() else part.title),
                               family=item["family"], kind=twin.kind if twin else None, question=part.question, unit=part.unit, moment=part.moment,
                               predict_rank=0, known_rank=twin.known_rank if twin else None, outcome=part.outcome,
                               labels=twin.labels if twin else "unknown", quote=part.quote, why=part.why or "proposed by the model"))
        pending.append((sid, item.get("reads")))
    if not subs:
        raise ValueError("no sub-problem rests on the person's words or a rule template")
    kept_parts = {s.part for s in subs}
    parts = [p for p in parts if p.id in kept_parts]
    final = []
    for s, (sid, raw_reads) in zip(subs, pending):
        reads = []
        for r in (raw_reads if isinstance(raw_reads, list) else [])[:10]:
            target = sub_ids.get(r.get("sub_problem")) if isinstance(r, Mapping) and not isinstance(r.get("sub_problem"), bool) else None
            if target and r.get("what") in pm.Reads.__args__:
                reads.append(Read(sub_problem=target, what=r["what"], via=str(r["via"])[:pm.MAX_TEXT] if isinstance(r.get("via"), str) and r["via"].strip() else None))
        final.append(s.model_copy(update={"reads": reads}))
    depth: dict[str, int] = {}

    def moment(sid: str, path: tuple[str, ...] = ()) -> int:
        """When a sub-problem predicts: after everything it reads (0 when it reads nothing). A cycle gets 0 here and is refused by the validator."""
        if sid in depth:
            return depth[sid]
        if sid in path:
            return 0
        sub = next(f for f in final if f.id == sid)
        depth[sid] = 1 + max([moment(r.sub_problem, path + (sid,)) for r in sub.reads] or [-1])
        return depth[sid]

    final = [f.model_copy(update={"predict_rank": moment(f.id),  # an outcome the rules know comes late: after its own prediction at least
                                  "known_rank": max(f.known_rank, moment(f.id) + 1) if f.known_rank is not None else None}) for f in final]
    comp = reply.get("composition") if isinstance(reply.get("composition"), Mapping) else {}
    renamed = {f"sp{n}": sid for n, sid in sub_ids.items()}  # the model's numbering of its own sub-problems -> the ids kept
    steps = []
    for item in (comp.get("steps") if isinstance(comp.get("steps"), list) else [])[:pm.MAX_SUB_PROBLEMS]:
        if not isinstance(item, Mapping) or item.get("kind") not in pm.StepKind.__args__ or not isinstance(item.get("output"), str) or not isinstance(item.get("inputs"), list):
            continue
        inputs_ = []
        for name in item["inputs"]:
            if not isinstance(name, str):
                continue
            if re.fullmatch(r"sp\d+", name):  # a sub-problem of the model's numbering: renamed, or dropped (never read as another's id)
                if name in renamed:
                    inputs_.append(renamed[name])
            else:
                inputs_.append(name)  # the output of an earlier step
        steps.append(CompositionStep(kind=item["kind"], inputs=inputs_, output=item["output"][:60], joins=str(item["joins"])[:pm.MAX_TEXT] if isinstance(item.get("joins"), str) and item["joins"].strip() else None))
    budget = comp.get("budget") if isinstance(comp.get("budget"), str) and quoted(comp.get("budget"), words) else None
    single = len(final) == 1
    method = "none" if single else (comp.get("method") if comp.get("method") in pm.Method.__args__ and comp.get("method") != "none" else "mixed" if len(steps) > 1 else "sequence")
    if single:
        steps = []
    made = ProblemMap(size="single" if single else "complex", parts=parts, sub_problems=final, composition=Composition(method=method, steps=steps, budget=budget[:pm.MAX_TEXT] if budget else None),
                      source="model", template="model", notes=["Drawn by the decomposer from the person's words; every part quotes them or names the rule template found in them."])
    issues = validate(made)
    if issues:
        raise ValueError("the map does not hold: " + "; ".join(issues[:3]))
    return made


def _decomposer_fallback(inputs: Mapping[str, Any]) -> ProblemMap:
    return inputs["rules"]


DECOMPOSER = Role(name="decomposer", description="Draws the problem map from the person's words: parts, sub-problems, how they combine; refused whole unless every part rests on the "
                                                 "person's words or a rule template and the map holds.",
                  evidence=_decomposer_evidence, parse=_decomposer_parse, fallback=_decomposer_fallback, merge=lambda rules, model: model)


def draw(words: str, *, advisor: Any = None, card: Optional[Mapping[str, Any]] = None, data: Any = None, parts: Optional[list[Any]] = None,
         examples: Optional[Sequence[str]] = None) -> tuple[ProblemMap, str, str]:
    """The map for the person's words: the model's when it is valid, else the rules'. Returns the map, where it came from ('advisor' or
    'fallback') and why. With no advisor it is the rules' map. ``examples`` (30.18): the closest business examples, one line each, read
    by the model as advice; a part still rests on the person's words or the rules' template."""
    from dclab_core.agents import run_role

    rules = decompose(words, parts=parts)
    result = run_role(DECOMPOSER, advisor, {"words": words, "card": dict(card or {}), "data": data, "rules": rules, "examples": list(examples or [])})
    return result.output, result.source, result.reason


# ---------------------------------------------------------------------------------------------------------------- the map reviewer

def _reviewer_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    m: ProblemMap = inputs["map"]
    return {"words": str(inputs.get("words") or "")[:1500], "map": lines(m), "size": m.size, "tables": table_lines(inputs.get("data")) if inputs.get("data") is not None else {},
            "rules_ask": [f"{q.code}:{q.target or '-'}" for q in review_rules(m, inputs.get("data"))][:20], "kinds": list(QuestionKind.__args__)}


def _reviewer_parse(reply: Mapping[str, Any], inputs: Mapping[str, Any]) -> list[MapQuestion]:
    m: ProblemMap = inputs["map"]
    words = str(inputs.get("words") or "")
    items = reply.get("questions")
    if not isinstance(items, list):
        raise ValueError("questions is a list")
    ids = {p.id for p in m.parts} | {s.id for s in m.sub_problems}
    names = {str(_get(t, "name")) for t in _tables(inputs.get("data"))}
    columns = {str(_get(c, "name")) for t in _tables(inputs.get("data")) for c in _columns(t)}
    out: list[MapQuestion] = []
    for item in items[:MAX_MODEL_QUESTIONS * 3]:
        if not isinstance(item, Mapping) or item.get("kind") not in QuestionKind.__args__ or not isinstance(item.get("question"), str):
            continue
        question = " ".join(item["question"].split())
        target = item.get("target")
        if target is not None and target not in ids:
            continue
        if not question.endswith("?") or not (3 <= len(question) <= 300):
            continue
        ref = item.get("ref")
        if not (quoted(item.get("quote"), words) or (isinstance(ref, str) and (ref in ids or ref in names or ref in columns))):
            continue  # it rests on nothing the person said, the map holds or the data has
        why = (f'the person said "{str(item["quote"])[:200]}"' if quoted(item.get("quote"), words) else f"the map or the data: {str(ref)[:100]}")
        out.append(MapQuestion(code="model_question", kind=item["kind"], target=target, question=question, why=why[:300], blocking=False, source="model"))
        if len(out) == MAX_MODEL_QUESTIONS:
            break
    return out


MAP_REVIEWER = Role(name="map_reviewer", description="Asks what the map misses: risks, leaks, missing data, ambiguity; questions for the person that rest on their words, the map or the "
                                                     "data card, never an edit and never blocking.",
                    evidence=_reviewer_evidence, parse=_reviewer_parse, fallback=lambda inputs: [], merge=lambda rules, model: model)


def review_with(m: ProblemMap, *, advisor: Any = None, words: str = "", data: Any = None) -> tuple[MapReview, str, str]:
    """The rules' questions and the reviewer's, merged. Returns the review, where the extra questions came from and why."""
    from dclab_core.agents import run_role

    result = run_role(MAP_REVIEWER, advisor, {"map": m, "words": words, "data": data})
    return review(m, data, extra=result.output, source="advisor" if result.source == "advisor" else "rules"), result.source, result.reason


# ---------------------------------------------------------------------------------------------------------------- the data-need agent

def _need_evidence(inputs: Mapping[str, Any]) -> dict[str, Any]:
    m: ProblemMap = inputs["map"]
    return {"map": lines(m), "tables": table_lines(inputs["data"]), "grains": {str(_get(t, "name"))[:80]: str(_get(t, "grain") or "")[:200] for t in _tables(inputs["data"])[:4]},
            "needs_now": [f"{s.id}: " + "; ".join(f"{n.table or '?'}[{','.join(n.columns[:6])}] label={n.label or '-'}" for n in s.needs) for s in m.sub_problems][:30],
            "labels": ["column", "file", "none"]}


def _need_parse(reply: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, list[Need]]:
    m: ProblemMap = inputs["map"]
    data = inputs["data"]
    items = reply.get("needs")
    if not isinstance(items, list):
        raise ValueError("needs is a list")
    known = {s.id for s in m.sub_problems}
    out: dict[str, list[Need]] = {}
    for item in items[:60]:
        if not isinstance(item, Mapping) or item.get("sub_problem") not in known:
            continue
        table = find_table(item.get("table"), data) if isinstance(item.get("table"), str) else None
        if table is None:
            continue  # a table the data card does not hold is no need the data can meet: coverage says what is missing, the model does not
        have = {str(_get(c, "name")) for c in _columns(table) if _get(c, "purpose") != "leak_suspect"}  # never a column the data card suspects of leaking
        columns = [c for c in (item.get("columns") if isinstance(item.get("columns"), list) else []) if isinstance(c, str) and c in have][:40]
        label = item.get("label") if item.get("label") in ("column", "file", "none") else None
        out.setdefault(item["sub_problem"], []).append(Need(table=str(_get(table, "name"))[:200], columns=columns, label=label, why=str(item.get("why") or "")[:pm.MAX_TEXT]))
    if not out:
        raise ValueError("no need names a table the data card holds")
    return out


def _need_fallback(inputs: Mapping[str, Any]) -> dict[str, list[Need]]:
    m = fill_needs(inputs["map"], inputs["data"])
    return {s.id: list(s.needs) for s in m.sub_problems if s.needs}


def _need_merge(rules: dict[str, list[Need]], model: dict[str, list[Need]]) -> dict[str, list[Need]]:
    out = {k: list(v) for k, v in rules.items()}
    for sid, needs in model.items():
        have = {(n.table, tuple(n.columns), n.label) for n in out.get(sid, [])}
        out.setdefault(sid, []).extend(n for n in needs if (n.table, tuple(n.columns), n.label) not in have)
    return out


DATA_NEED = Role(name="data_need", description="Says, per sub-problem, which table and which columns it needs, by name; a table or a column the data card does not hold is dropped.",
                 evidence=_need_evidence, parse=_need_parse, fallback=_need_fallback, merge=_need_merge)


def needs_for(m: ProblemMap, data: Any, advisor: Any = None) -> tuple[ProblemMap, str, str]:
    """The map with each sub-problem's needs set (the rules', plus the model's that the data card holds). Returns the map, the source and why."""
    from dclab_core.agents import run_role

    result = run_role(DATA_NEED, advisor, {"map": m, "data": data})
    subs = [s.model_dump(mode="json") for s in m.sub_problems]
    for s in subs:
        if s["id"] in result.output:
            s["needs"] = [n.model_dump(mode="json") for n in result.output[s["id"]]][:10]
    changed = ProblemMap.model_validate({**m.model_dump(mode="json"), "fingerprint": "", "sub_problems": subs})
    return (m if changed.fingerprint == m.fingerprint else changed), result.source, result.reason


__all__ = ["MapQuestion", "MapReview", "MapEditRefused", "review_rules", "review", "review_with", "coverage", "fill_needs", "find_table", "add_part", "remove_part", "merge_parts",
           "DECOMPOSER", "MAP_REVIEWER", "DATA_NEED", "draw", "needs_for", "lines", "table_lines", "QuestionKind", "TEMPLATES"]
