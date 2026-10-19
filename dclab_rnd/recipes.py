"""The recipe registry over the studio's problem kinds (package 30.10): each kind read as a ``dclab_core.recipes.RecipePart``
(no behaviour change), and the recipe one run resolves to, with the part and the reason behind every choice.

A kind's version is its own fingerprint (the attributes that decide its split, metrics, baselines, checks and export), so a
change to a kind changes its version. Its level stays ``experimental`` until a certification run (30.11) earns more: the
capability matrix (``make kinds-check``) shows a kind on its replicas only, while certification asks for two real datasets
and the same result with the model off. ``python -m dclab_rnd.recipes`` prints the registry.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import sys
from pathlib import Path
from typing import Any, Optional

from dclab_core.recipes import Choice, Detection, RecipePart, ResolvedRecipe, resolve
from dclab_core import modifiers as M

CAPABILITY = Path(__file__).resolve().parent / "kinds" / "capability.json"


def _value(solution: Any, key: str) -> Any:
    return solution.get(key) if isinstance(solution, dict) else getattr(solution, key, None)


def _fingerprint(part: RecipePart, kind: Any) -> str:
    """The part's own fields and the source of its kind's class and every kind class above it: any change to what the kind
    decides (a method, a threshold inside one, a field type) changes the version (the review of 30.10)."""
    from .kinds.base import ProblemKind

    sources = []
    for cls in type(kind).__mro__:
        if isinstance(cls, type) and issubclass(cls, ProblemKind):
            try:
                sources.append(inspect.getsource(cls))
            except (OSError, TypeError):
                sources.append(cls.__qualname__)
    shown = part.model_dump(mode="json", exclude={"version", "level", "certification"})
    return hashlib.sha256(json.dumps([shown, sources], sort_keys=True, default=str).encode()).hexdigest()[:10]


def _evidence() -> dict[str, list[str]]:
    if not CAPABILITY.is_file():
        return {}
    out: dict[str, list[str]] = {}
    for row in json.loads(CAPABILITY.read_text(encoding="utf-8")).get("results") or []:
        passed = sum(bool(c.get("ok")) for c in row.get("checks") or [])
        out[row["kind"]] = [f"replica: {passed} of {len(row.get('checks') or [])} decisions ({row.get('status')}, make kinds-check)"]
    return out


def _part_split(kind: Any) -> Optional[str]:
    """The split the kind itself takes when the solution declares no time or group column: None for a kind whose own machine
    splits (its record says how), spatial blocks for places, strata when the kind stratifies, else a plain random split."""
    if kind.own_machine() is not None:
        return None
    if kind.key == "places":
        return "spatial"
    return "stratified" if kind.stratify else "random"


def part_of(kind: Any, evidence: Optional[dict[str, list[str]]] = None) -> RecipePart:
    """A problem kind as a recipe part: its nine fields read from the kind's own attributes and defaults."""
    metrics = [m for m, _ in kind.metrics]
    primary = kind.default_metric({"task": kind.base_task}, None) if metrics else None
    detection = [Detection(signal="words", rule=str(kind.words)[:300])] if kind.words else []
    detection += [Detection(signal="domain pack", rule=", ".join(kind.packs)[:300])] if kind.packs else []
    inputs = (["target"] if getattr(kind, "needs_target", True) else []) + sorted(kind.Fields.model_fields)
    fields = dict(name=kind.key, part_type="kind", version="pending", level="experimental", detection=detection,
                  required_inputs=inputs[:12], split=_part_split(kind), primary_metric=primary,
                  guard_metrics=[m for m in metrics if m != primary][:12], baseline=kind.baselines[0] if kind.baselines else None,
                  leakage_checks=list(kind.checks)[:20], artifact_needs=[k for k, v in kind.export.items() if v is True][:20],
                  report_items=list(kind.reliability)[:20], certification=(evidence or {}).get(kind.key, [])[:20])
    draft = RecipePart(**fields)
    return RecipePart(**{**fields, "version": _fingerprint(draft, kind)})


def kind_parts() -> list[RecipePart]:
    from . import kinds

    evidence = _evidence()
    return [part_of(k, evidence) for k in kinds.all_kinds()]


def resolve_for(kind: Any, solution: Any, positive_rate: Optional[float] = None, machine: Any = None,
                detected: Optional[list[dict[str, Any]]] = None) -> ResolvedRecipe:
    """The recipe a run resolves to, before any modifier exists (30.11 adds them): the kind's part, the metric the person
    stated or the kind's default for the training rows' positive rate, and the split the solution's declared columns give.
    The values are exactly the run's (``kind.split``, ``kind.default_metric``, or a kind's own machine and its metric); what
    is new is the part and the reason behind each."""
    part = part_of(kind)
    stated_metric = _value(solution, "metric")
    found = {d.get("name"): d for d in detected or [] if d.get("applies") and d.get("name") in M.PARTS}  # 30.11: what the proposal applied
    run_metric = stated_metric or (getattr(machine, "metric", None) if machine is not None else kind.default_metric(solution, positive_rate))
    kept = {n: d for n, d in found.items() if (n == "rare_positives" and run_metric == M.PARTS["rare_positives"].primary_metric)
            or (n == "groups" and d.get("value") == _value(solution, "group_column"))
            or (n == "time_order" and bool(_value(solution, "time_column")))  # 30.12: the person may choose which column orders the rows, as the ledger counts it
            or (n == "noisy_labels" and _value(solution, "label_source_column"))
            or (n == "review_capacity" and _value(solution, "capacity_k"))}  # what the run applies (the review of 30.11; 30.12)
    chosen = _value(solution, "time_column")
    if "time_order" in kept and kept["time_order"].get("value") != chosen:  # the person chose another column: the reason names it
        kept["time_order"] = {**kept["time_order"], "value": chosen,
                              "reason": f"`{chosen}` orders the rows, as the person chose (the rows' first time column was "
                                        f"`{kept['time_order'].get('value')}`): the holdout is the latest period"}
    if machine is not None:  # 30.12: a kind's own machine reads neither a label source nor a capacity
        kept.pop("noisy_labels", None)
        kept.pop("review_capacity", None)
    elif _value(solution, "label_source_column") and "noisy_labels" not in kept:  # the person declared the column: the run applies it
        kept["noisy_labels"] = {"name": "noisy_labels", "reason": f"the solution declares `{_value(solution, 'label_source_column')}` as the labels' source"}
    if machine is None and _value(solution, "capacity_k") and "review_capacity" not in kept:
        kept["review_capacity"] = {"name": "review_capacity", "reason": f"the solution declares {_value(solution, 'capacity_k')} cases a "
                                                                       f"{_value(solution, 'capacity_period') or 'period'}"}
    machine_metric, machine_split = None, ""
    if machine is not None:  # a prepared machine (its metric, its holdout description) or the machine itself
        machine_metric = getattr(machine, "metric", None) or getattr(machine, "primary_metric", None)
        described = getattr(machine, "holdout_description", "")
        machine_split = str(described() if callable(described) else described or "")[:400]
    r = resolve(part, [], stated={"primary_metric": stated_metric} if stated_metric else None)
    choices = []
    for c in r.choices:
        if c.field == "split":
            if machine is not None:
                c = Choice(field="split", value=None, part="machine", reason=f"the kind's own machine splits: {machine_split}"[:600])
            else:
                split = kind.split(solution)
                declared = [f"{k} `{str(_value(solution, k))[:80]}`" for k in ("time_column", "group_column") if _value(solution, k)]
                if _value(solution, "spatial"):
                    declared.append("spatial blocks")
                by = [M.PARTS[n].ref for n in ("time_order", "groups") if n in kept]
                if by:  # the modifiers the rows showed and the person kept
                    why = "; ".join(str(kept[n].get("reason") or "") for n in ("time_order", "groups") if n in kept)
                    c = Choice(field="split", value=split, part=", ".join(by), reason=why[:600])
                elif declared:
                    c = Choice(field="split", value=split, part="solution", reason=f"the solution declares {' and '.join(declared)}"[:600])
                elif split != c.value:
                    c = Choice(field="split", value=split, part=part.ref, reason=f"the kind {kind.key}'s split for this solution")
        elif c.field == "primary_metric" and not stated_metric:
            metric = machine_metric if machine is not None else kind.default_metric(solution, positive_rate)
            rate = f" for a positive rate of {positive_rate:.3g} on the training rows" if positive_rate is not None else ""
            if machine is None and "rare_positives" in kept and metric == M.PARTS["rare_positives"].primary_metric:
                c = Choice(field="primary_metric", value=metric, part=M.PARTS["rare_positives"].ref,
                           reason=str(kept["rare_positives"].get("reason") or "")[:600])
            else:
                c = Choice(field="primary_metric", value=metric, part="machine" if machine is not None else part.ref,
                           reason="the kind's own machine judges by it" if machine is not None else f"the kind {kind.key}'s default{rate}")
        elif c.field == "guard_metrics":
            continue
        elif c.field == "model_families":
            c = Choice(field="model_families", value=None, part="machine" if machine is not None else part.ref,
                       reason="the kind's own machine chooses its models" if machine is not None else
                       "the engine's screen chooses them from the machine's available families (kind.model_families)")
        choices.append(c)
    primary = next(c.value for c in choices if c.field == "primary_metric")
    guards = [m for m, _ in kind.metrics if m != primary]
    choices.append(Choice(field="guard_metrics", value=guards, part=part.ref, reason="the kind's other metrics, beside the primary one"))
    if "noisy_labels" in kept:  # 30.12: what the run judges on, and why
        choices.append(Choice(field="evaluation", value="confirmed labels only", part=M.PARTS["noisy_labels"].ref,
                              reason=str(kept["noisy_labels"].get("reason") or "")[:600]))
    if "review_capacity" in kept:  # the decision measure and the operating point; the selection inside training stays the primary metric
        choices.append(Choice(field="decision_metric", value="precision_at_k", part=M.PARTS["review_capacity"].ref,
                              reason=str(kept["review_capacity"].get("reason") or "")[:600]))
        choices.append(Choice(field="threshold_rule", value=M.PARTS["review_capacity"].threshold_rule, part=M.PARTS["review_capacity"].ref,
                              reason="the K-th score of an average period, recorded with the model"))
    return ResolvedRecipe(kind=r.kind, modifiers=[M.PARTS[n].ref for n in ("rare_positives", "groups", "time_order", "noisy_labels", "review_capacity")
                                                  if n in kept], choices=choices, ask=r.ask)


def record_for(kind: Any, solution: Any, positive_rate: Optional[float] = None, machine: Any = None,
               detected: Optional[list[dict[str, Any]]] = None) -> dict[str, Any]:
    """The resolved recipe as a stage record holds it, with every modifier the rows showed (applied or not, and why); a
    failure to resolve is recorded, never raised: the recipe describes a run, it must not stop one (the review of 30.10)."""
    try:
        out = resolve_for(kind, solution, positive_rate, machine, detected).model_dump(mode="json")
        if detected:
            out["detected"] = [{k: d.get(k) for k in ("name", "applies", "reason", "strength", "value", "asks_user")} for d in detected][:12]
        return out
    except Exception as error:  # noqa: BLE001
        return {"error": f"{type(error).__name__}: {str(error)[:300]}"}


def main(argv: Optional[list[str]] = None) -> int:
    for p in kind_parts():
        print(f"{p.ref:28} {p.level:13} split {p.split or '-':11} metric {p.primary_metric or '-':22} {'; '.join(p.certification)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
