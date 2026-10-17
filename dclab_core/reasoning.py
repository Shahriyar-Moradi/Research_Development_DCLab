"""Verified reasoning traces (package 27.9): the `<think>` a solver model learns, written from DCLab's own records.

A trace is the reasoning from what an example gives (the columns as bands, the prediction moment, the measured signals,
the split's protocol, the holdout's numbers) to its verified answer, step by step: what is observed, what is checked,
what it rests on, what follows. It is written by these rules from the example's own inputs and answer, never by a
model, and it is kept only when it is **grounded**: every number it states is one of the example's own numbers, every
column it names is one of the example's columns, and it reaches the answer the example verified. A trace that is not
grounded is dropped, never repaired.

- :func:`trace`: the steps for one example of a family (``problem_type``, ``leakage_judgment``, ``split_design``,
  ``algorithm_choice``, ``read_results``), or None when the family has no trace yet.
- :func:`grounded`: why a trace is not grounded (empty when it is).
- :func:`think`: the trace as the ``<think>`` block a served model writes before its answer.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Optional

VERSION = "reasoning_v1"
FAMILIES = ("problem_type", "leakage_judgment", "split_design", "algorithm_choice", "read_results", "solution_design")
_NUMBER = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?!\w)(?!\.\d)")  # a sentence's full stop after a number is not part of it
_NAME = re.compile(r"`([^`]{1,200})`")
UNIQUE = {"all", "over_99pct", "from_90_to_99pct"}


def _n(x: Any, digits: int = 3) -> str:
    """A number as a trace states it: at most three decimals, an integer as itself."""
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    return str(int(x)) if float(x).is_integer() and abs(x) >= 1 else f"{float(x):.{digits}f}".rstrip("0").rstrip(".")


def _numbers(obj: Any) -> set[str]:
    out: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, bool) or v is None:
            return
        if isinstance(v, (int, float)):
            out.update({_n(v), _n(v, 2), _n(v, 1), _n(round(float(v) * 100, 1), 1)})  # a share may be stated as a percentage
        elif isinstance(v, str):
            out.update(_NUMBER.findall(v))
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)

    walk(obj)
    return out


def _names(given: Mapping[str, Any], answer: Mapping[str, Any]) -> set[str]:
    names = {str(c.get("column")) for c in given.get("columns") or [] if isinstance(c, dict) and c.get("column")}
    for key in ("outcome", "time_column", "group_column"):
        for src in (given, answer):
            if src.get(key):
                names.add(str(src[key]))
    names |= {str(f.get("column")) for f in answer.get("forbidden") or [] if isinstance(f, dict)}
    names |= {str(c) for c in answer.get("identifiers") or []}
    names |= {str(s.get("column")) for s in given.get("train_only_signals") or [] if isinstance(s, dict)}
    names |= set(_NAME.findall(json.dumps([given, answer], ensure_ascii=False)))  # a name the example's own text writes (a derived column)
    return names


def _end(text: Any) -> str:
    """A sentence from a phrase: its own end kept (a question stays a question), else a full stop."""
    t = str(text or "").strip()
    return t if t.endswith((".", "?", "!")) else t + "."


def _cols(given: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(c["column"]): c for c in given.get("columns") or [] if isinstance(c, dict) and c.get("column")}


def _signal(s: Mapping[str, Any]) -> Optional[float]:
    for key in ("score", "auc", "value", "metric"):
        v = s.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
    return None


def trace(family: str, given: Mapping[str, Any], answer: Mapping[str, Any]) -> Optional[list[str]]:
    """The reasoning steps from ``given`` to ``answer`` for one family (None: no trace for this family)."""
    cols = _cols(given)
    outcome = given.get("outcome")
    moment = given.get("prediction_moment")
    steps: list[str] = []
    if family == "problem_type":
        o = cols.get(str(outcome)) or {}
        steps.append(f"The outcome is `{outcome}`." if outcome else "The outcome column is not named.")
        if o:
            steps.append(f"`{outcome}` is stored as {o.get('dtype')} with {o.get('cardinality')} cardinality: "
                         + {"binary": "two values, a yes or no.", "low": "a few distinct values.", "medium": "many values.",
                            "high": "many values, a number to predict.", "very_high": "nearly every row different, a number to predict."}.get(str(o.get("cardinality")), "its values decide the task."))
        if moment:
            steps.append(f"The prediction is made: {_end(moment)}")
        steps.append(f"So the base task is {answer.get('task')}, and the problem is of the kind {answer.get('kind')}: {str(answer.get('why') or '').rstrip('.')}.")
        return steps
    if family == "leakage_judgment":
        if moment:
            steps.append(f"The prediction is made: {_end(moment)} A column known only after that moment, or one that encodes the outcome, must stay out.")
        signals = [s for s in given.get("train_only_signals") or [] if isinstance(s, dict) and s.get("column") and _signal(s) is not None]
        if signals:
            top = sorted(signals, key=lambda s: -(_signal(s) or 0))[:3]
            steps.append("Each column alone against the outcome, measured on the training rows only: "
                         + "; ".join(f"`{s['column']}` {_n(_signal(s))}" for s in top) + ".")
        forbidden = [f for f in answer.get("forbidden") or [] if isinstance(f, dict) and f.get("column")]
        for f in forbidden:
            steps.append(f"`{f['column']}` stays out: {str(f.get('reason') or 'it is not known at the moment').rstrip('.')}.")
        if not forbidden:
            steps.append("No column is known only after the moment: none is forbidden for leakage.")
        for c in answer.get("identifiers") or []:
            band = cols.get(str(c)) or {}
            unique = band.get("uniqueness")
            steps.append(f"`{c}` names a row rather than describing it"
                         + (f" ({unique} of its values are distinct" + (f", name tokens {', '.join(band.get('name_tokens') or [])})" if band.get("name_tokens") else ")") if unique else "")
                         + ": an identifier is kept out, a model would learn the rows, not the problem.")
        steps.append("The other columns are known at the moment and kept.")
        return steps
    if family == "split_design":
        dated = [c for c, b in cols.items() if b.get("value_pattern") == "date_like" or b.get("dtype") == "datetime"]
        time_column, group_column = answer.get("time_column"), answer.get("group_column")
        if dated:
            steps.append("Columns that hold dates: " + ", ".join(f"`{c}`" for c in dated[:5]) + ".")
        if time_column:
            steps.append(f"The rows are ordered in time by `{time_column}`: a model is used on later rows, so the holdout must be later than training, never a random draw.")
        else:
            steps.append("No time order the prediction depends on: the holdout need not be the latest rows.")
        if group_column:
            steps.append(f"Rows of one `{group_column}` share their behaviour: whole `{group_column}` groups go to one side, or the score measures recognising them.")
        steps.append(f"So the holdout is {str(answer.get('holdout') or 'drawn once').rstrip('.')}, read once at the end.")
        if answer.get("cross_validation"):
            steps.append(f"The model is chosen on the training rows alone: {str(answer['cross_validation']).rstrip('.')}.")
        return steps
    if family == "algorithm_choice":
        t = given.get("target_summary") or {}
        size = [f"{_n(given['rows'])} rows" if isinstance(given.get("rows"), (int, float)) else None,
                f"{_n(given['usable_features'])} usable features" if isinstance(given.get("usable_features"), (int, float)) else None]
        steps.append(f"The task is {given.get('task')}" + (", on " + " with ".join(x for x in size if x) if any(size) else "") + ".")
        rate = t.get("positive_rate_train")
        if isinstance(rate, (int, float)):
            positives = t.get("positives_train")
            steps.append(f"The positive class is {_n(rate)} of the training rows" + (f" ({_n(positives)} positives)" if isinstance(positives, (int, float)) else "")
                         + (": rare, so accuracy says little and the ranking metric decides." if rate < 0.1 else "."))
        ranking = [str(r) for r in answer.get("ranking") or []]
        if ranking:
            steps.append("Every family was screened on the same training folds; best first: " + ", ".join(ranking) + ".")
        if answer.get("selection_rule"):
            steps.append(f"The rule: {str(answer['selection_rule']).rstrip('.')}.")
        steps.append(f"So the recipe is {answer.get('feature_recipe')} and the model {answer.get('model')}.")
        return steps
    if family == "solution_design":
        if not isinstance(answer.get("workflow"), list) or not all(isinstance(i, dict) for i in answer["workflow"]):
            return None  # a session's drawn workflow (its own shape): no trace from these rules
        if moment:
            steps.append(f"The prediction is made: {_end(moment)} Everything follows from that moment and from the outcome `{outcome}`."
                         if outcome else f"The prediction is made: {_end(moment)}")
        for item in answer.get("workflow") or []:
            d, step = item.get("decision") or {}, item.get("step")
            if step == "split":
                parts = [f"the holdout is {str(d.get('holdout')).rstrip('.')}" if d.get("holdout") else None,
                         f"ordered in time by `{d['time_column']}`" if d.get("time_column") else None,
                         f"whole `{d['group_column']}` groups on one side" if d.get("group_column") else None]
                steps.append("First the split, fixed before anything is learned: " + ", ".join(p for p in parts if p) + ".")
            elif step == "leakage":
                out = [f"`{c}`" for c in [*(d.get("forbidden") or []), *(d.get("identifiers") or [])]]
                steps.append("Then the leakage audit: " + (", ".join(out) + " stay out." if out else "no column is kept out."))
            elif step == "features" and d.get("recipe"):
                steps.append(f"Then the features, chosen inside the training folds: the {d['recipe']} recipe.")
            elif step == "models" and d.get("model"):
                steps.append(f"Then the model family, by the selection rule on the same folds: {d['model']}.")
            elif step == "final":
                steps.append(f"Last, the holdout is read once, scored by {d.get('metric')}." if d.get("holdout_read_once") else
                             f"Last, the holdout is scored by {d.get('metric')}.")
        steps.append(f"So the problem is a {answer.get('task')} one of the kind {answer.get('kind')}, solved in that order.")
        return steps
    if family == "read_results":
        primary, value, interval = answer.get("primary_metric"), answer.get("value"), answer.get("interval") or {}
        if primary and isinstance(value, (int, float)):
            steps.append(f"The primary metric is {primary}: {_n(value)} on the holdout.")
        if isinstance(interval, dict) and isinstance(interval.get("low"), (int, float)) and isinstance(interval.get("high"), (int, float)):
            steps.append(f"Its interval runs from {_n(interval['low'])} to {_n(interval['high'])} ({interval.get('method') or 'a bootstrap'}): "
                         "that width is how much this number could move on other rows like these.")
        if answer.get("beats_baseline") is True:
            steps.append("It is above what chance gives for this metric, so the model has learned something.")
        elif answer.get("beats_baseline") is False:
            steps.append("It is not above what chance gives for this metric: nothing may be claimed.")
        uses = given.get("holdout_uses")
        if isinstance(uses, int) and not isinstance(uses, bool):
            steps.append(f"The holdout was read {_n(uses)} time{'s' if uses != 1 else ''}: " + ("one look, an honest number." if uses == 1 else "a second look is optimistic."))
        steps.append(f"What may be claimed: {str(answer.get('claim') or '').rstrip('.')}.")
        return steps
    return None


def grounded(steps: list[str], given: Mapping[str, Any], answer: Mapping[str, Any]) -> list[str]:
    """Why a trace is not grounded: a number not among the example's own, a column not among its own."""
    numbers, names = _numbers(given) | _numbers(answer), _names(given, answer)
    text = "\n".join(steps)
    problems = [f"the number {x} is not one of the example's" for x in sorted(set(_NUMBER.findall(text)) - numbers)]
    problems += [f"the column `{x}` is not one of the example's" for x in sorted(set(_NAME.findall(text)) - names)]
    return problems


def think(steps: list[str]) -> str:
    return "<think>\n" + "\n".join(steps) + "\n</think>"


def of_messages(family: str, messages: list[Mapping[str, Any]]) -> tuple[Optional[list[str]], list[str]]:
    """The trace of a stored example (its question's JSON and its answer's), and why it is not grounded."""
    try:
        given, answer = json.loads(str(messages[-2]["content"])), json.loads(str(messages[-1]["content"]))
    except (ValueError, KeyError, IndexError, TypeError):
        return None, ["the example is not a question and an answer in JSON"]
    steps = trace(family, given, answer) if isinstance(given, dict) and isinstance(answer, dict) else None
    if steps is None:
        return None, []
    return steps, grounded(steps, given, answer)


__all__ = ["VERSION", "FAMILIES", "trace", "grounded", "think", "of_messages"]
