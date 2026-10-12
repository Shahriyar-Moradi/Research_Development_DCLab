"""The data investigator, measured (package 18.3): do a model's readings of column names find more planted traps
without accusing clean columns?

The cases are benchmark v2's seeded tables (``agent_eval/benchmark.py``): its dev split, with two replicas whose leaks
are known, for building; its sealed test split, whose fingerprints are committed, for the one measured run. Each case is
read three ways:

* **rules**: ``dclab_core.analyse`` alone (names on its list, out-of-fold power, missing-value power, sums, recomputed
  times; entities from id-like names);
* **investigator**: the same, with the data investigator's proposals (columns written after the moment, entities),
  each tested exactly as a telling name is;
* **plain**: the same model and prompt, its proposals taken as findings without any test.

A planted trap is *found* when its leak column gets a leakage finding of warning or worse, or, for a shared-label
entity, an entity finding of warning; a *false alarm* is a clean column (the case's innocent columns and honest inputs)
given a leakage finding of warning or worse, or an entity finding of warning (it would regroup the folds).

What this does not measure: the package text names "benchmark v3", which does not exist yet (15.9): this is v2. The two
replicas are in the dev split only, so the gated sealed run is v2's test split alone. The plain arm is the model alone,
its proposals untested and without the rules; it shows what the checks add, and is not gated. The gate (``GATE``, fixed before any run) asks, on the sealed split: more
traps found with the investigator than with the rules alone, and no more false alarms.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "investigator_v1" / "results"
ARMS = ("rules", "investigator", "plain")

GATE = {
    "version": "investigator_gate_v1",  # fixed on 2026-10-12, before any live run
    "split": "sealed",
    "more_found": "planted traps found with the investigator > found with the rules alone",
    "no_more_false_alarms": "clean columns accused with the investigator <= accused with the rules alone",
}


@dataclass(frozen=True)
class Item:
    id: str
    table: Callable[[], pd.DataFrame]
    target: str
    task: str
    moment: str
    leaks: tuple[str, ...] = ()
    entity: Optional[str] = None  # a column whose rows share the label: an entity finding must name it
    clean: tuple[str, ...] = field(default=())


def _benchmark(name: str) -> list[Item]:
    from dclab_rnd.agent_eval.benchmark import split

    out = []
    for case in split(name):
        task = "regression" if "regression" in case.tags else "binary"
        entity = case.group_column if case.trap == "group_label" else None
        out.append(Item(case.id, case.table, "target", task, case.moment, tuple(case.leaks), entity, tuple(case.clean_inputs)))
    return out


def _replicas() -> list[Item]:
    from dclab_core.replicas import delivery_time, ticket_routing

    def deliveries() -> pd.DataFrame:
        return delivery_time.generate(seed=0, n=3000).orders

    def tickets() -> pd.DataFrame:
        return ticket_routing.generate(seed=0, n=3000).tickets

    return [
        Item("REP-delivery_time", deliveries, "actual_minutes", "regression", "When the customer places the order.",
             ("delivered_at", "courier_rating"), clean=("distance_km", "items", "weather")),
        Item("REP-ticket_routing", tickets, "assigned_team", "multiclass", "When the ticket is created, from its subject, body and plan.",
             ("resolved_by", "first_response_minutes"), clean=("subject", "body", "plan")),
    ]


def load(split: str) -> list[Item]:
    if split == "dev":
        return _benchmark("dev") + _replicas()
    if split == "sealed":
        from dclab_rnd.agent_eval import benchmark

        recorded = benchmark.json.loads(benchmark.SEALED.read_text(encoding="utf-8"))["cases"]
        if recorded != benchmark.sealed_fingerprints():
            raise ValueError("the sealed benchmark changed: a sealed set is never edited")
        return _benchmark("test")
    raise ValueError(f"unknown split {split!r}")


# ------------------------------------------------------------------------------------------------ reading and scoring


def _flags(findings: list[Any]) -> tuple[set[str], set[str]]:
    """Columns given a leakage finding of warning or worse, and columns given an entity finding of warning."""
    leaks, entities = set(), set()
    for f in findings:
        if f.severity not in ("warning", "critical"):
            continue
        columns = f.evidence.get("columns") or [f.evidence.get("column")]
        if f.topic == "leakage":
            leaks |= {str(c) for c in columns if c}
        elif f.topic == "entities":
            entities |= {str(c) for c in columns if c}
    return leaks, entities


def read(item: Item, advisor: Optional[Any]) -> tuple[set[str], set[str]]:
    from dclab_core.analyse import analyse

    frame = item.table()
    found = analyse({"table": frame}, target=item.target, task=item.task, advisor=advisor, moment=item.moment)
    return _flags(found.findings)


def read_plain(item: Item, advisor: Any) -> tuple[set[str], set[str]]:
    """The same model and prompt; its proposals taken as findings, untested."""
    from dclab_core.advisor import consult
    from dclab_core.analyse import DATA_INVESTIGATOR

    frame = item.table()
    inputs = {"features": frame.drop(columns=[item.target]), "target": item.target, "task": item.task, "moment": item.moment}
    failures: list[BaseException] = []
    answer = consult(advisor, "data_investigator", DATA_INVESTIGATOR.evidence(inputs), parse=lambda a: DATA_INVESTIGATOR.parse(a, inputs),
                     on_error=lambda kind, exc: failures.append(exc or ValueError(kind)))
    if failures:
        raise failures[0]
    answer = answer or {"after": {}, "entities": {}}
    return set(answer["after"]), set(answer["entities"])


def score(item: Item, leaks: set[str], entities: set[str]) -> dict[str, Any]:
    traps = list(item.leaks) + ([item.entity] if item.entity else [])
    found = [c for c in item.leaks if c in leaks] + ([item.entity] if item.entity and item.entity in entities else [])
    # a clean column accused of a leak, or named an entity (that regroups the folds): both are false alarms
    accused = sorted(c for c in item.clean if c in leaks or (c in entities and c != item.entity))
    return {"traps": len(traps), "found": len(found), "missed": sorted(set(traps) - set(found)), "false_alarms": accused}


class Counting:
    """An advisor that counts its requests; a failed one stops the run (it is then stored unscored)."""

    def __init__(self, inner: Any):
        self.inner, self.requests, self.failures = inner, 0, []

    def ask(self, purpose: str, evidence: Mapping[str, Any]) -> Any:
        if self.failures:
            raise RuntimeError("stopped: an earlier request failed, so this run will not be scored")
        self.requests += 1
        try:
            return self.inner.ask(purpose, evidence)
        except Exception as error:  # noqa: BLE001 — recorded; the role then falls back as it would in the product
            self.failures.append(f"{type(error).__name__}: {str(error)[:160]}")
            raise


def measure(items: list[Item], advisor: Optional[Any]) -> dict[str, Any]:
    counting = Counting(advisor) if advisor is not None else None
    rows = []
    for item in items:
        scores = {"rules": score(item, *read(item, None))}
        if counting is not None:
            scores["investigator"] = score(item, *read(item, counting))
            try:
                scores["plain"] = score(item, *read_plain(item, counting))
            except Exception:  # noqa: BLE001 — already counted as a failure
                scores["plain"] = None
        rows.append({"case": item.id, "scores": scores})
    return {"rows": rows, "summary": summarize(rows), "requests": counting.requests if counting else 0,
            "failures": counting.failures if counting else []}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for arm in ARMS:
        cells = [r["scores"][arm] for r in rows if r["scores"].get(arm)]
        if not cells:
            continue
        out[arm] = {"traps": sum(c["traps"] for c in cells), "found": sum(c["found"] for c in cells),
                    "false_alarms": sum(len(c["false_alarms"]) for c in cells), "cases": len(cells)}
    return out


def verdict(result: Mapping[str, Any], split: str) -> dict[str, Any]:
    summary = result.get("summary") or {}
    if "investigator" not in summary:
        return {"gate": GATE["version"], "status": "not scored", "why": "no model: the rules alone were read"}
    if result.get("failures"):
        return {"gate": GATE["version"], "status": "not scored", "why": f"{len(result['failures'])} request(s) failed"}
    a, r = summary["investigator"], summary["rules"]
    checks = {"more_found": a["found"] > r["found"], "no_more_false_alarms": a["false_alarms"] <= r["false_alarms"]}
    status = "pass" if all(checks.values()) else "fail"
    if split != GATE["split"]:
        status = f"{status} (dev: the gate is read on the sealed split)"
    return {"gate": GATE["version"], "status": status, "checks": checks}


def plan(gateway: Any, items: list[Item], max_tokens: int) -> dict[str, Any]:
    import json

    from dclab_core import prompts
    from dclab_core.analyse import DATA_INVESTIGATOR
    from dclab_rnd.models import prices, settings

    tier = gateway.route("data_investigator")[0] if hasattr(gateway, "route") else settings.purpose("data_investigator").tier
    public = settings.public(tier)
    biggest = max(len(json.dumps(DATA_INVESTIGATOR.evidence({"features": i.table().drop(columns=[i.target]), "target": i.target,
                                                               "task": i.task, "moment": i.moment}))) for i in items)
    chars = len(prompts.text("data_investigator")) + biggest + 400
    requests = 2 * len(items)
    try:
        each = prices.estimate(public.get("model"), bool(public.get("local")), chars, max_tokens)
    except ValueError:
        each = None
    return {"purpose": "data_investigator", "tier": tier, "model": public.get("model"), "local": bool(public.get("local")),
            "available": gateway.available("data_investigator"), "cases": len(items), "max_requests": requests,
            "max_eur": round(each * requests, 3) if each is not None else None}


def sealed_runs() -> list[Path]:
    return sorted(RESULTS.glob("INV-*_sealed.json"))


def next_id() -> str:
    numbers = [int(m.group(1)) for p in RESULTS.glob("INV-*.json") if (m := re.match(r"INV-(\d+)_", p.name))]
    return f"INV-{max(numbers, default=0) + 1:03d}"
