"""The problem meets the data (package 20.4): what the person said, checked against what the tables are.

:func:`check` reads a :class:`~dclab_core.card.ProblemCard` (what the problem is) and a
:class:`~dclab_core.datacard.DataCard` (what the tables are) and returns every mismatch, each a question to the person
with its evidence (column names and counts, never a value), blocking when building on it would answer another problem:

* ``target_absent``: the outcome named is not a column of the table;
* ``target_refused``: it is a column, but one that can never be an outcome (an identifier, an entity, a row number,
  a constant, text, a time, a coordinate, written after the outcome), with the data card's reason;
* ``target_family``: the column does not fit the family (a yes or no for a regression, a number with thousands of
  values for a classification); blocking when the person stated the family, a question when it was read from words;
* ``unit_grain``: one row of the table is not what is predicted (orders, when each prediction is for a customer):
  aggregate first, or predict per row;
* ``time_absent``: the moment or the family needs a time (when each prediction is made) and no column holds one, or
  the time column named is not in the table;
* ``labels_not_joined``: the cases are in a file that is not joined to the table yet;
* ``labels_contradict``: the words say nothing is labelled, while the table has a column that looks like the outcome; or
  the person said anomaly detection while known cases are here (a joined file, a named outcome column).

A pair that agrees raises nothing. The rules are code; a host may ask a model to phrase the questions, never to decide them.

:func:`fit` (30.13, the owner's framework's gate G2) turns the check into one of three verdicts: **fits** (go on),
**fits with changes** (each change a proposed edit of the problem card, with why, for a person to accept: the outcome is
another column, the unit is the table's grain, the family is the column's, the model is judged on confirmed labels
only), **does not fit** (stop, and name the data that is missing: the file of known cases, a column of dates, a column
that records the outcome).
"""

from __future__ import annotations

import sys
from typing import Any, Literal, Optional

from pydantic import Field

from dclab_core.contracts import CONTRACTS, Contract, Versioned, _version

Code = Literal["target_absent", "target_refused", "target_family", "unit_grain", "time_absent", "labels_not_joined", "labels_contradict",
               "not_checked"]
KEY_WORDS = frozenset({"id", "ids", "key", "ref", "nr", "number", "code", "uuid", "guid"})
UNIT_STOP = frozenset({"a", "an", "the", "one", "each", "per", "single", "row", "rows", "every", "is", "of"})
CLASSES = frozenset({"classification", "rare_event"})
NUMBERS = frozenset({"regression", "forecasting", "time_to_event"})
NEEDS_TIME = frozenset({"forecasting", "time_to_event"})
MANY_CLASSES = 100  # a classification outcome with more values than this is a number, or an identifier


class Mismatch(Contract):
    code: Code
    blocking: bool
    question: str = Field(max_length=600)
    evidence: dict[str, Any] = Field(default_factory=dict, max_length=12, description="Column names, purposes and counts: never a value.")


def _m(code: str, blocking: bool, question: str, evidence: dict[str, Any]) -> Mismatch:
    """A mismatch whose question and evidence always fit: a person's words inserted are clipped."""
    clip = lambda v: v[:120] if isinstance(v, str) else [x[:120] if isinstance(x, str) else x for x in v][:10] if isinstance(v, list) else v  # noqa: E731
    return Mismatch(code=code, blocking=blocking, question=question[:600], evidence={k: clip(v) for k, v in list(evidence.items())[:12]})


class ProblemDataCheck(Versioned):
    """What the problem and the data disagree on; ``ready`` when nothing blocks."""

    schema_version: Literal["dclab_core.problem_data_check.v1"] = _version("problem_data_check")
    mismatches: list[Mismatch] = Field(default_factory=list, max_length=20)
    ready: bool = True


CONTRACTS["problem_data_check"] = ProblemDataCheck


def _main(data: Any) -> Any:
    return next((t for t in data.tables if t.role == "main"), data.tables[0])


def _tokens(text: Optional[str]) -> set[str]:
    from dclab_core.bands import name_tokens

    words = set(name_tokens(text or ""))
    return words | {w[:-1] for w in words if w.endswith("s") and len(w) > 3}  # orders → order


def check(card: Any, data: Any, joined: Optional[bool] = None) -> ProblemDataCheck:
    """Every mismatch between the problem card and the data card of its tables. ``joined``: what the host's own label
    review says (a file of cases joined on a key), when it has one: it wins over the data card's guess."""
    table = _main(data)
    columns = {c.name: c for c in table.columns}
    family = card.family.value or (card.in_play[0] if len(card.in_play) == 1 else None)
    out: list[Mismatch] = []
    target = card.brief.target_column.value
    if target and target not in columns:
        near = table.target_candidates[:5]
        out.append(_m("target_absent", blocking=True,
                            question=f"The outcome named, {target}, is not a column of the table. Which column holds it"
                                     + (f": {', '.join(near)}?" if near else "?"),
                            evidence={"named": target, "candidates": near}))
    elif target:
        column = columns[target]
        stated = card.family.status == "stated"  # a family read from the words gives way to the outcome the person named
        if column.purpose not in (None, "target_candidate", "feature"):
            out.append(_m("target_refused", blocking=True,
                                question=f"{target} cannot be the outcome: it is {str(column.purpose).replace('_', ' ')} ({column.why}). "
                                         "Which column records what you want to predict?",
                                evidence={"column": target, "purpose": column.purpose, "candidates": table.target_candidates[:5]}))
        elif family in CLASSES and column.role == "numeric" and column.distinct > MANY_CLASSES:
            out.append(_m("target_family", blocking=stated,
                                question=f"The problem is a {family.replace('_', ' ')} problem, but {target} is a number with {column.distinct} values. "
                                         "Predict the number (a regression), or a yes or no made from it?",
                                evidence={"column": target, "family": family, "distinct": column.distinct}))
        elif family in NUMBERS and column.cardinality in ("constant", "binary"):
            out.append(_m("target_family", blocking=stated,
                                question=f"The problem asks for a number, but {target} is a yes or no. A classification, then?",
                                evidence={"column": target, "family": family, "cardinality": column.cardinality}))
    times = [c.name for c in table.columns if c.purpose == "time"]
    named_time = card.brief.time_column.value
    if named_time and named_time not in columns:
        out.append(_m("time_absent", blocking=True,
                            question=f"The time named, {named_time}, is not a column of the table. Which column says when each prediction is made"
                                     + (f": {', '.join(times[:5])}?" if times else "?"),
                            evidence={"named": named_time, "times": times[:5]}))
    elif not times and family in NEEDS_TIME:
        out.append(_m("time_absent", blocking=True,
                            question=f"A {family.replace('_', ' ')} needs a time for each row, and no column holds a date or a time. Which file has it?",
                            evidence={"family": family, "times": []}))
    elif not times and card.brief.prediction_moment.value and family not in (None, "segmentation", "anomaly"):
        out.append(_m("time_absent", blocking=False,
                            question="The moment says when each prediction is made, but no column holds a date or a time: the split cannot "
                                     "follow time. Is there a column for when each row happened?",
                            evidence={"moment": "stated", "times": []}))
    unit = card.brief.unit.value
    noun = _tokens(unit) - UNIT_STOP if unit else set()
    if noun and len(noun) <= 3:  # a unit is a noun or two ("a customer", "a store-day"), not a sentence
        for c in table.columns:
            core = _tokens(c.name) - KEY_WORDS  # customer_id → customer; delivery_zone_id → delivery, zone
            if c.purpose == "entity" and core and core <= noun and c.name not in table.key:
                said = card.brief.unit.status == "stated" and card.brief.unit.source in ("person", "text")
                grain = table.grain.replace("one row per ", "") if table.grain else "something finer"
                out.append(_m("unit_grain", said,  # a unit read from a file's name is a question, never a block
                              f"Each prediction is for {unit[:80]}, but one row of the table is {grain[:120]}: {c.name[:80]} repeats. "
                              f"Aggregate the rows per {c.name[:80]} first, or predict per row?",
                              {"unit": unit, "entity": c.name, "grain": table.grain, "key": table.key, "unit_how": card.brief.unit.status}))
                break
    labels = card.labels.value
    label_tables = [t for t in data.tables if t.role == "labels"]
    is_joined = joined if joined is not None else any(t.joins_on for t in label_tables)
    if labels == "file" and not is_joined and not (target and target in columns):
        if label_tables:  # a file of cases is here, its key not found: a question, never a dead end
            out.append(_m("labels_not_joined", False,
                          "The file of known cases is here, but no key joins it to the table yet. Which column do both share "
                          "(an order or a transaction id)?", {"labels": "file", "tables": len(data.tables)}))
        else:
            out.append(_m("labels_not_joined", True,
                          "The cases are in a file of known cases that is not here yet. Add that file to this draft, with the "
                          "table it belongs to.", {"labels": "file", "tables": len(data.tables)}))
    if labels == "none" and table.target_candidates:
        from dclab_core.datacard import outcome_named

        named = [c for c in table.target_candidates if outcome_named(c)]
        if named:
            out.append(_m("labels_contradict", blocking=False,
                                question=f"You said nothing is labelled, but {named[0]} looks like an outcome. Is it one?",
                                evidence={"labels": "none", "column": named[0]}))
    if family == "anomaly" and ((label_tables and is_joined) or (target and target in columns)):
        # 21.5: the morning's other half: "it is anomaly detection" said, while the known cases are here
        found = "a file of known cases joins the table" if label_tables and is_joined else f"{target} looks like the outcome"
        out.append(_m("labels_contradict", blocking=card.family.status == "stated" and labels != "none",  # "nothing is labelled": confirmed
                      question=f"You said anomaly detection, which looks for unusual rows when nothing is labelled, but {found}. "
                               "Learn from the known cases (a rare event), or say nothing is labelled to look for unusual rows anyway?",
                      evidence={"labels": "file" if label_tables and is_joined else "column", "family": "anomaly"}))
    return ProblemDataCheck(mismatches=out[:20], ready=not any(m.blocking for m in out))


Verdict = Literal["fits", "fits_with_changes", "does_not_fit"]


class CardEdit(Contract):
    """One change the data asks of the problem card: the field, its new value, why, and the check that found it."""

    field: str = Field(max_length=60)
    value: str = Field(max_length=600)
    why: str = Field(max_length=600)
    code: str = Field(max_length=40)


class FitVerdict(Versioned):
    """G2: whether the data fits the problem, as one of three verdicts, with the edits a person accepts or the data that
    is missing."""

    schema_version: Literal["dclab_core.fit_verdict.v1"] = _version("fit_verdict")
    verdict: Verdict
    edits: list[CardEdit] = Field(default_factory=list, max_length=12)
    missing: list[str] = Field(default_factory=list, max_length=12, description="The data that is missing, in plain words.")
    reasons: list[str] = Field(default_factory=list, max_length=20, description="Every mismatch and modifier behind the verdict.")
    back_to: Optional[Literal["problem", "data"]] = Field(None, description="Where the work goes back: the problem (edits) or the data (missing).")


CONTRACTS["fit_verdict"] = FitVerdict
FLIP = {"classification": "regression", "rare_event": "regression", "regression": "classification", "forecasting": "classification",
        "time_to_event": "classification"}


def _outcomes(table: Any, named: Any) -> list[str]:
    """The table's columns that could record an outcome, when the data card kept none as a candidate (a named outcome
    that was refused leaves the card without candidates): a yes or no, or a name that says an outcome."""
    from dclab_core.datacard import outcome_named

    return [c.name for c in table.columns if c.name != named and c.purpose in (None, "feature", "target_candidate")
            and (c.cardinality == "binary" or outcome_named(c.name))][:5]


def fit(card: Any, data: Any, checked: Optional[ProblemDataCheck] = None, joined: Optional[bool] = None) -> FitVerdict:
    """The three-way verdict of a problem card against the data card of its tables (G2): every blocking mismatch is an
    edit when the data says what the card should become, missing data when it cannot; a modifier the data card detected
    may ask its own edit (noisy labels: judge on the confirmed labels only)."""
    checked = checked or check(card, data, joined)
    edits: list[CardEdit] = []
    missing: list[str] = []
    reasons: list[str] = []

    def edit(field: str, value: str, why: str, code: str) -> None:
        edits.append(CardEdit(field=field, value=value[:600], why=why[:600], code=code))

    table = _main(data)
    for m in checked.mismatches:
        reasons.append(f"{m.code}: {m.question}"[:600])
        if not m.blocking:
            continue  # a question for the person, not a change the verdict waits for
        ev = m.evidence
        if m.code in ("target_absent", "target_refused"):
            near = list(ev.get("candidates") or []) or _outcomes(table, ev.get("named") or ev.get("column"))
            if near:
                edit("target_column", str(near[0]), m.question, m.code)
            else:
                missing.append("a column that records the outcome")
        elif m.code == "target_family" and ev.get("family") in FLIP:
            edit("family", FLIP[ev["family"]], m.question, m.code)
        elif m.code == "unit_grain":  # predict per row: the unit becomes what one row is, by its key when it has one
            key = [str(k) for k in ev.get("key") or []]
            edit("unit", ("one row per " + " and ".join(key)) if key else "one row of the table",
                 m.question + " (accepted: predict per row)", m.code)
        elif m.code == "time_absent":
            if ev.get("times"):
                edit("time_column", str(ev["times"][0]), m.question, m.code)
            else:
                missing.append("a column that says when each row happened")
        elif m.code == "labels_not_joined":
            missing.append("the file of known cases, with the table it belongs to")
        elif m.code == "labels_contradict" and ev.get("family") == "anomaly":  # known cases are here: learn from them
            edit("family", "rare_event", m.question, m.code)
    said = str(card.brief.target_definition.value or "").lower()
    for d in getattr(data, "detected_modifiers", None) or []:
        if d.name == "noisy_labels" and d.applies and "confirmed" not in said:  # the retail interview's example: the rule's labels are not the outcome
            edit("target_definition", "judged on the confirmed labels only (a person's review, the random audit)", d.reason, "noisy_labels")
            reasons.append(f"noisy_labels: {d.reason}"[:600])
    verdict = "does_not_fit" if missing else "fits_with_changes" if edits else "fits"
    return FitVerdict(verdict=verdict, edits=edits[:12], missing=list(dict.fromkeys(missing))[:12], reasons=reasons[:20],
                      back_to="data" if missing else "problem" if edits else None)


_cards = sys.modules.get("dclab_core.datacard")  # the data card names the verdict: complete its type if it loaded first (30.13)
if _cards is not None and hasattr(_cards, "_resolve_fit"):
    _cards._resolve_fit()

__all__ = ["check", "fit", "Mismatch", "ProblemDataCheck", "CardEdit", "FitVerdict"]
