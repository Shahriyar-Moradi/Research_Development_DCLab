"""The service's knowledge built from the store (Parts G to I of the storing phase): how models are built for a business and
whether correctly, what they take, how the service was given, and the knowledge of building correctly.

- ``builds(layout)`` (32.24): one :class:`dclab_core.service.ModelBuild` per project of every workspace (file or PostgreSQL,
  a deleted one too) and per student the tutor refitted: the business context in bands, the data's shape, the recipe with
  reasons, people's active time, the conversation, compute and euros, the result against its bar, what went wrong, what a
  person changed, whether it was used after. Every part filled or listed in ``unknown`` with why.
- ``verdicts`` (32.26): each build judged by code against DCLab's rules (``dclab_core.service.verdict``); a fact its records
  cannot say is None, never assumed passed.
- ``curves`` and ``service_catalog`` (32.25): learning curves (each student point by its training rows, each problem kind by
  its labelled rows' band) and, per problem kind, what was built, how often it met its bar (with an interval), its median cost and
  time, its common failures. Only aggregates across workspaces: a kind with fewer than :data:`MIN_BUILDS` builds or
  :data:`MIN_WORKSPACES` workspaces says so instead of a figure (one business's figures are never shown alone).
- ``knowledge`` (32.27): every rule, pitfall, finding and precedent of the evidence index, CLAUDE.md's rules and the outside
  findings proposed as :class:`dclab_core.service.KnowledgeItem`; ``accept`` is a person's yes, kept in
  ``docs/knowledge/building/``; an accepted item whose source changed since is flagged for review, never changed silently.
- ``external`` and ``tested`` (32.28): an outside finding kept as a summary in our own words with its citation, link and
  licence, refused when it copies its source; it stays ``from_the_literature`` until a DCLab campaign tests it.
- ``guides`` (32.29): a guide per problem kind, every sentence citing an accepted item or the catalog (``cited.check``).
- ``engagements`` and ``playbook`` (32.30, 32.31): each walk (its newest version) as a
  :class:`dclab_core.service.ServiceEngagement`, and the playbook's items from them, each resting on what it counts.

Consent: the service reads only records whose workspace allows improving DCLab (``improve`` or ``train``); a set built from
these for a model type must still require ``train`` (``sets.allowed``).
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from dclab_core.service import BuildVerdict, CurvePoint, KnowledgeItem, LearningCurve, ModelBuild, PlaybookItem, ServiceEngagement, verdict

from .layout import ROOT, Layout
from .store import Store, now, write_json

MIN_BUILDS = 3
MIN_WORKSPACES = 3
IDLE_MINUTES = 30.0  # a gap between two steps longer than this is a person away, not working
ACCEPTED = ROOT / "docs" / "knowledge" / "building"
DONE = ("completed", "approved")  # a stage record's finished statuses (studio/graph.py DONE)
READS = ("improve", "train")
USED_AFTER = ("business", "promotion")  # an outcome observed after the build (never the holdout's own read)


# ------------------------------------------------------------------------------------------- reading the store
def _current(layout: Layout, kind: str) -> Iterator[Any]:
    """Each source's current records only: its last whole version and the tails kept after it (a snapshot's newest version; a
    log's last whole copy and what was appended since), so a rewritten log or a walk kept again is never counted twice."""
    from . import shapers

    chains: dict[tuple[str, str], list[Any]] = defaultdict(list)
    for e in Store(layout).entries(kind):
        chains[(e.workspace, e.source)].append(e)
    keep: set[str] = set()
    for entries in chains.values():
        entries.sort(key=lambda e: e.version)
        start = max((i for i, e in enumerate(entries) if e.offset == 0), default=0)
        keep.update(e.id for e in entries[start:])
    for r in shapers.records(layout, kind):
        if r.provenance.entry in keep and r.provenance.consent in READS:
            yield r


def _band(rows: Any) -> str:
    try:
        n = int(rows)
    except (TypeError, ValueError):
        return "unknown"
    return "tiny" if n < 1_000 else "small" if n < 10_000 else "medium" if n < 100_000 else "large" if n < 1_000_000 else "very_large"


def _project(by: dict[tuple[str, str], dict[str, Any]], ws: str, pid: Any) -> dict[str, Any]:
    key = (ws, str(pid))
    if key not in by:
        by[key] = {"stages": {}, "moves": [], "corrections": [], "eur": 0.0, "used": 0, "workspace": ws}
    return by[key]


def _stage_body(r: Any) -> tuple[Optional[str], Optional[str], dict[str, Any]]:
    """(project, stage, record) of a stage record, from a file (``projects/<id>/stages/<stage>.json``) or a PostgreSQL row
    (``db/stage_record``: its ``project_id``, ``stage`` and its ``record`` lifted)."""
    parts = r.provenance.source.split("/")
    if parts[0] == "projects" and len(parts) >= 4:
        return parts[1], parts[-1].rsplit(".", 1)[0], r.body
    return r.body.get("project"), r.body.get("stage"), r.body  # a PostgreSQL row's record is lifted into its body by the shaper


def _latest(layout: Layout) -> dict[tuple[str, str], dict[str, Any]]:
    """Per (workspace, project): its project record, stage records, walk, graph moves, corrections, spend and uses after."""
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _current(layout, "workflows"):
        ws, src = r.provenance.workspace, r.provenance.source
        parts = src.split("/")
        if r.type == "walk":
            if r.body.get("project"):
                _project(by, ws, r.body["project"])["walk"] = r.body
            continue
        if src.startswith("db/project") or (parts[0] == "projects" and len(parts) >= 3 and parts[2] == "project.json"):
            pid = r.body.get("id") or (parts[1] if parts[0] == "projects" else None)
            if pid:
                p = _project(by, ws, pid)
                p["project"], p["deleted"], p["consent"] = r.body, p.get("deleted") or r.provenance.deleted, r.provenance.consent
        elif r.type == "stage_run":
            pid, stage, body = _stage_body(r)
            if pid and stage:
                p = _project(by, ws, pid)
                p["stages"][stage] = body
                p.setdefault("records", {})[stage] = r  # the record itself, for its provenance and split
        elif r.type == "graph_move":
            move = r.body.get("entry") if isinstance(r.body.get("entry"), dict) else r.body  # a PostgreSQL row holds it in ``entry``
            pid = r.body.get("project") or (parts[1] if parts[0] == "projects" else None)
            if pid:
                _project(by, ws, pid)["moves"].append(move)
    for r in _current(layout, "decisions"):
        if r.type == "correction" and r.body.get("project"):
            _project(by, r.provenance.workspace, r.body["project"])["corrections"].append(str(r.body.get("what")))
    for r in _current(layout, "model_requests"):
        if r.body.get("project") and isinstance(r.body.get("cost_eur"), (int, float)):
            _project(by, r.provenance.workspace, r.body["project"])["eur"] += float(r.body["cost_eur"])
    for r in _current(layout, "sessions"):
        payload = r.body.get("payload") if isinstance(r.body.get("payload"), dict) else {}
        if r.body.get("kind") == "outcome.observed" and r.body.get("project") and payload.get("observed") in USED_AFTER:
            _project(by, r.provenance.workspace, r.body["project"])["used"] += 1
    return {k: v for k, v in by.items() if isinstance(v.get("project"), dict)}


def _active_minutes(steps: list[dict[str, Any]]) -> Optional[float]:
    """People's active time: the gaps between a walk's steps, each capped at IDLE_MINUTES (a night away is not work)."""
    times = []
    for s in steps:
        try:
            times.append(datetime.fromisoformat(str(s.get("at"))))
        except ValueError:
            continue
    if len(times) < 2:
        return None
    times.sort()
    return round(sum(min((b - a).total_seconds() / 60, IDLE_MINUTES) for a, b in zip(times, times[1:])), 1)


# ------------------------------------------------------------------------------------------- 32.26 the verdict's facts
def _higher(metric: Optional[str]) -> Optional[bool]:
    from dclab_core.solution_brief import HIGHER

    return HIGHER.get(str(metric)) if metric else None


def _interval(ci: Any) -> tuple[Optional[float], Optional[float], str]:
    """(low, high, how) of a holdout interval as the engine writes it (``{low, high, method, draws}``, a bootstrap that drew
    nothing has 0 draws), or a pair; (None, None, why) when it is not an interval."""
    if isinstance(ci, dict):
        low, high = ci.get("low"), ci.get("high")
        if isinstance(low, (int, float)) and isinstance(high, (int, float)) and ci.get("draws") != 0:
            return float(low), float(high), str(ci.get("method") or "")
        return None, None, str(ci.get("method") or "no interval drawn")
    if isinstance(ci, (list, tuple)) and len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci):
        return float(min(ci)), float(max(ci)), ""
    return None, None, "no interval recorded"


def _facts(project: dict[str, Any], stages: dict[str, dict[str, Any]], moves: list[dict[str, Any]], metric: Optional[str]) -> dict[str, Any]:
    """The verdict's facts from the records alone; a fact its records cannot say is None, with why."""
    solution = project.get("solution") or {}
    final = stages.get("final") or {}
    evidence = final.get("evidence") or {}
    done = final.get("status") in DONE
    facts: dict[str, Any] = {}

    # the prediction contract (the solution and its moment) saved before the first stage ran
    saved = sorted(str(m.get("at")) for m in moves if m.get("move") == "set_solution" and m.get("status") == "allowed")
    ran = sorted(str(s.get("started_at") or s.get("started") or "") for s in stages.values() if s.get("status") in DONE)
    ran = [a for a in ran if a]
    if not solution.get("prediction_moment"):
        facts["contract_before_scores_why"] = "no prediction moment in its solution record"
    elif saved and ran:
        facts["contract_before_scores"] = saved[0] <= ran[0]
        facts["contract_before_scores_why"] = f"solution saved {saved[0]}, first stage started {ran[0]}"
    else:
        facts["contract_before_scores_why"] = "a prediction moment, but no recorded move or stage time to order it by"

    # no column unknown at the prediction moment used: the declared leakage columns kept out of the features, and every
    # heuristic candidate either declared or excluded
    data, leakage = stages.get("data") or {}, stages.get("leakage") or {}
    if data.get("status") in DONE and leakage.get("status") in DONE:
        roles = (data.get("evidence") or {}).get("column_roles") or {}
        lev = leakage.get("evidence") or {}
        declared = [str(c) for c in lev.get("declared_leakage_features") or []]
        candidates = [str(c.get("feature")) for c in lev.get("heuristic_review_candidates") or [] if isinstance(c, dict) and c.get("feature")]
        used = sorted(c for c in declared if roles.get(c) == "feature")
        open_ = sorted(c for c in candidates if c not in declared and roles.get(c) == "feature")
        if used:
            facts["no_leaking_column"] = False
            facts["no_leaking_column_why"] = f"declared leakage columns used as features: {', '.join(used)}"
        elif open_:
            facts["no_leaking_column_why"] = f"candidates kept as features without a decision recorded: {', '.join(open_[:10])}"
        else:
            facts["no_leaking_column"] = True
            facts["no_leaking_column_why"] = f"{len(declared)} declared leakage columns excluded; no open candidate kept as a feature"
    else:
        facts["no_leaking_column_why"] = "its data or leakage stage record is missing or not finished"

    models = stages.get("models") or {}
    if models.get("status") in DONE and (models.get("evidence") or {}).get("cv_protocol"):
        facts["fitted_inside_folds"] = True
        facts["fitted_inside_folds_why"] = "the engine fits every step inside its training folds (its CV protocol recorded)"
    else:
        facts["fitted_inside_folds_why"] = "no finished models stage with its CV protocol"

    if done:
        uses = evidence.get("holdout_uses_in_this_project", project.get("holdout_uses"))
        if isinstance(uses, (int, float)):
            facts["holdout_read_once"] = int(uses) <= 1
            facts["holdout_read_once_why"] = f"the holdout was read {int(uses)} time(s)" + (f": {evidence['holdout_reuse_reason']}" if evidence.get("holdout_reuse_reason") else "")
        else:
            facts["holdout_read_once_why"] = "no count of the holdout's reads in its records"
        holdout = (evidence.get("holdout_metrics") or {}).get(metric) if metric else None
        baselines = {k: v.get(metric) for k, v in (evidence.get("baselines") or {}).items() if isinstance(v, dict) and isinstance(v.get(metric), (int, float))}
        low, high, how = _interval(evidence.get("holdout_primary_metric_ci"))
        higher = _higher(metric)
        bounded = low is not None
        if isinstance(holdout, (int, float)) and baselines and bounded and higher is not None:
            bound = low if higher else high  # the interval's side that must clear every baseline
            facts["beats_baselines_with_interval"] = all((bound > b) if higher else (bound < b) for b in baselines.values())
            facts["beats_baselines_with_interval_why"] = f"{metric} interval [{low}, {high}] against {baselines}"
        else:
            facts["beats_baselines_with_interval_why"] = "its holdout metric, its direction, a baseline on the same metric, or the interval is missing"
        if bounded:
            facts["uncertainty_said"] = True
            facts["uncertainty_said_why"] = how or "an interval is recorded"
        elif "holdout_primary_metric_ci" in evidence or evidence.get("limits"):
            facts["uncertainty_said_why"] = f"no interval, but the record says why: {how}"  # said, not measured: neither passed nor failed
        else:
            facts["uncertainty_said"] = False
            facts["uncertainty_said_why"] = "neither an interval nor a reason for none"
        bar = evidence.get("success_bar") or {}
        if bar.get("verdict") in ("met", "not_met"):
            facts["bar_met"] = bar["verdict"] == "met"
            facts["bar_met_why"] = f"the signed brief's bar: {bar['verdict']}"
        else:
            facts["bar_met_why"] = bar.get("reason") or "no signed brief with a bar"
        export = (project.get("exports") or {}).get("code") or {}
        final_run = ((final.get("run") or {}).get("run")) if isinstance(final.get("run"), dict) else None
        if export.get("checked"):
            if final_run and export.get("final_run") and export["final_run"] != final_run:
                facts["export_reproduces_why"] = "the export was checked against an earlier final run"
            else:
                facts["export_reproduces"] = bool(export.get("same"))
                facts["export_reproduces_why"] = f"the exported pipeline gave {export.get('value')} against {export.get('expected')}"
        else:
            facts["export_reproduces_why"] = "the code export was never run and checked"
    else:
        for name in ("holdout_read_once", "beats_baselines_with_interval", "uncertainty_said", "bar_met", "export_reproduces"):
            facts[f"{name}_why"] = "its final stage has not finished"
    return facts


# ------------------------------------------------------------------------------------------- 32.24 the builds
def _project_build(key: tuple[str, str], p: dict[str, Any]) -> tuple[ModelBuild, BuildVerdict]:
    ws, pid = key
    project, stages = p["project"], p["stages"]
    data, solution = project.get("data") or {}, project.get("solution") or {}
    final = stages.get("final") or {}
    evidence = final.get("evidence") or {}
    metric = final.get("primary_metric") or (stages.get("models") or {}).get("primary_metric") or solution.get("metric")
    unknown: dict[str, str] = {}
    recipe = [c for c in (((stages.get("data") or {}).get("evidence") or {}).get("recipe") or {}).get("choices") or [] if isinstance(c, dict)]
    if not recipe:
        unknown["recipe"] = "no resolved recipe in its data stage"
    walk = p.get("walk") or {}
    steps = walk.get("steps") or []
    minutes = _active_minutes(steps)
    if minutes is None:
        unknown["people_minutes"] = "no walk with two timed steps"
    unknown["questions"] = "the questions asked are not told apart from other turns in the walk; chat_turns counts the turns"
    bar = evidence.get("success_bar") or {}
    result = {"metric": metric, "holdout": (evidence.get("holdout_metrics") or {}).get(metric) if metric else None,
              "interval": [x for x in _interval(evidence.get("holdout_primary_metric_ci"))[:2]] if _interval(evidence.get("holdout_primary_metric_ci"))[0] is not None else None, "bar": bar.get("bar"), "met": {"met": True, "not_met": False}.get(str(bar.get("verdict")))}
    if final.get("status") not in DONE:
        unknown["result"] = f"its final stage is {final.get('status') or 'not run'}"
    elif result["met"] is None:
        unknown["result.met"] = bar.get("reason") or "no signed brief with a bar"
    labelled = None  # the rows the engine kept with a label: its training and holdout rows (an unlabelled row is not one)
    if isinstance(evidence.get("train_rows"), int) and isinstance(evidence.get("holdout_rows"), int):
        labelled = evidence["train_rows"] + evidence["holdout_rows"]
    elif final.get("status") in DONE:
        unknown["data.labelled"] = "its final record does not count its training and holdout rows"
    columns = data.get("column_count") or (len(data["columns"]) if isinstance(data.get("columns"), list) else None)
    if columns is None:
        unknown["data.columns"] = "no column count in its data record"
    industry = project.get("industry")
    if not industry:
        unknown["context.industry"] = "no industry in its project record"
    if not p["used"]:
        unknown["used"] = "no outcome observed after the build (in the business or a promotion)"
    refused = [f"{m.get('move')}: {', '.join(map(str, m.get('failed_checks') or []))}" for m in p["moves"] if m.get("status") not in (None, "allowed")]
    failed = [f"stage {k}: {str(v.get('error') or 'failed')[:200]}" for k, v in stages.items() if v.get("status") == "failed"]
    kind = solution.get("kind") or ((stages.get("data") or {}).get("evidence") or {}).get("problem_kind") or solution.get("task")
    build = ModelBuild(id=f"build:{ws}:{pid}", model_type="business_model", workspace=ws,
                       context={"industry": industry, "size": _band(data.get("rows")), "problem_kind": kind, "task": solution.get("task"),
                                "deleted_in_product": bool(p.get("deleted"))},
                       data={"rows": _band(data.get("rows")), "columns": columns, "label": "column" if solution.get("target") else "none",
                             "labelled": _band(labelled) if labelled is not None else "unknown"},
                       recipe=recipe[:40],
                       spent={"people_minutes": minutes, "chat_turns": sum(1 for s in steps if s.get("kind") == "draft_event" and (s.get("detail") or {}).get("kind") == "chat"),
                              "compute_seconds": round(sum(float(v.get("elapsed_seconds") or 0) for v in stages.values()), 1), "eur": round(p["eur"], 6)},
                       result=result, went_wrong=(failed + refused)[:50], changed_by_person=p["corrections"][:50],
                       used=True if p["used"] else None, unknown=unknown, at=now())
    return build, verdict(build, _facts(project, stages, p["moves"], metric))


def _student_builds(layout: Layout) -> list[tuple[ModelBuild, BuildVerdict]]:
    """Each student the tutor refitted (its card and its measurement) as a build of the M1 type."""
    store = Store(layout)
    files: dict[str, dict[str, Any]] = defaultdict(dict)
    for e in store.entries("students"):
        if e.consent in READS and e.source.startswith("students/") and e.source.endswith(("/card.json", "/measurement.json")):
            prefix, name = e.source.rsplit("/", 1)
            try:
                files[prefix][name] = json.loads(store.read(e))
            except Exception:  # noqa: BLE001 - an unreadable card is not a build
                continue
    out = []
    for prefix, f in sorted(files.items()):
        card = f.get("card.json")
        if not card:
            continue
        dev, rule = card.get("dev") or {}, card.get("rule_dev") or {}
        ci = dev.get("accuracy_ci") or []
        build = ModelBuild(id=f"build:student:{prefix.split('/', 1)[1]}", model_type="m1_students", workspace="rnd",
                           context={"problem_kind": f"student:{card.get('point')}", "point": card.get("point")},
                           data={"rows": _band(card.get("trained_on")), "trained_on": card.get("trained_on"), "groups": card.get("groups"),
                                 "label_sources": card.get("label_sources")},
                           recipe=[{"field": "family", "value": card.get("family"), "why": card.get("chosen_by")},
                                   {"field": "calibration", "value": card.get("calibration")}],
                           spent={}, result={"metric": "accuracy", "holdout": dev.get("accuracy"), "interval": ci, "bar": {"rule_dev_accuracy": rule.get("accuracy")},
                                             "met": (f.get("measurement.json") or {}).get("above_rule")},
                           unknown={"spent": "the tutor's time and cost are in its run record, not the card", "used": "a student is never served by the tutor"},
                           at=now())
        bounded = len(ci) == 2 and all(isinstance(x, (int, float)) for x in ci)
        facts = {"fitted_inside_folds": "fold" in str(card.get("chosen_by") or "").lower() or None,
                 "fitted_inside_folds_why": card.get("chosen_by") or "its card does not say how it was chosen",
                 "holdout_read_once_why": "a student is measured on dev; its sealed set is read only by the gate",
                 "contract_before_scores_why": "a decision point's contract is its ledger point, not a project's solution",
                 "no_leaking_column_why": "the state's excluded fields are on the card; their leakage is not judged here",
                 "beats_baselines_with_interval": (min(ci) > rule["accuracy"]) if bounded and isinstance(rule.get("accuracy"), (int, float)) else None,
                 "beats_baselines_with_interval_why": f"dev accuracy interval {ci} against the rule's {rule.get('accuracy')}",
                 "uncertainty_said": bounded, "bar_met": build.result["met"], "bar_met_why": "above the rule on dev",
                 "export_reproduces_why": "a student is kept whole as its pickle; no export"}
        out.append((build, verdict(build, facts)))
    return out


def builds(layout: Optional[Layout] = None) -> list[tuple[ModelBuild, BuildVerdict]]:
    """Every project the store kept (a deleted one too, a tester's too) and every student refitted, as builds with verdicts."""
    layout = layout or Layout()
    return [_project_build(k, p) for k, p in sorted(_latest(layout).items())] + _student_builds(layout)


# ------------------------------------------------------------------------------------------- 32.25 curves and the catalog
def _wilson(k: int, n: int, z: float = 1.96) -> list[Optional[float]]:
    if not n:
        return [None, None]
    p = k / n
    centre, half = (p + z * z / (2 * n)) / (1 + z * z / n), z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def _enough(items: list[tuple[ModelBuild, Any]]) -> tuple[bool, str]:
    spaces = {b.workspace for b, _ in items}
    if len(items) < MIN_BUILDS:
        return False, f"{len(items)} builds: a figure needs {MIN_BUILDS}"
    if len(spaces) < MIN_WORKSPACES:
        return False, f"{len(spaces)} workspaces: a figure needs {MIN_WORKSPACES}, so no one business's figures show alone"
    return True, ""


def curves(layout: Optional[Layout] = None, judged: Optional[list[tuple[ModelBuild, BuildVerdict]]] = None) -> list[LearningCurve]:
    """Each student point's dev accuracy at each count of training labels, and each (problem kind, metric)'s median holdout by
    rows band across the businesses (a band from MIN_WORKSPACES workspaces only)."""
    judged = builds(layout) if judged is None else judged
    out = []
    by_point: dict[str, list[CurvePoint]] = defaultdict(list)
    for b, _ in judged:
        if b.model_type == "m1_students":
            ci = b.result.get("interval") or [None, None]
            by_point[str(b.context.get("point"))].append(CurvePoint(labels=int(b.data.get("trained_on") or 0), score=b.result.get("holdout"),
                                                                    low=ci[0] if ci else None, high=ci[1] if len(ci) > 1 else None))
    for point, pts in sorted(by_point.items()):
        ok = len(pts) >= MIN_BUILDS
        out.append(LearningCurve(subject=point, kind="point", points=sorted(pts, key=lambda c: c.labels), builds=len(pts), enough=ok,
                                 why="" if ok else f"{len(pts)} refits: a curve needs {MIN_BUILDS}"))
    order = ("tiny", "small", "medium", "large", "very_large")
    by_kind: dict[tuple[str, str], dict[str, list[ModelBuild]]] = defaultdict(lambda: defaultdict(list))
    for b, _ in judged:
        if b.model_type == "business_model" and isinstance(b.result.get("holdout"), (int, float)) and b.data.get("labelled") in order:
            by_kind[(str(b.context.get("problem_kind")), str(b.result.get("metric")))][b.data["labelled"]].append(b)
    for (kind, metric), bands in sorted(by_kind.items()):
        points = []
        for i, band in enumerate(order):
            bs = bands.get(band) or []
            if len({b.workspace for b in bs}) >= MIN_WORKSPACES:
                scores = [b.result["holdout"] for b in bs]
                points.append(CurvePoint(labels=i, score=round(statistics.median(scores), 4), low=min(scores), high=max(scores), metric=metric))
        n = sum(len(v) for v in bands.values())
        out.append(LearningCurve(subject=f"{kind}:{metric}", kind="problem_kind", points=points, builds=n, enough=len(points) >= 2,
                                 why="labels: the labelled rows' band's index (tiny 0 to very large 4)" + ("" if len(points) >= 2 else
                                     f"; {len(points)} bands with {MIN_WORKSPACES} workspaces: a curve needs 2")))
    return out


def service_catalog(layout: Optional[Layout] = None, judged: Optional[list[tuple[ModelBuild, BuildVerdict]]] = None) -> dict[str, Any]:
    """Per problem kind, across workspaces: builds, the share that met its bar (with a Wilson interval), median cost and active
    time, the share judged correct, common failures, the rows bands it was built on. A kind below MIN_BUILDS builds or
    MIN_WORKSPACES workspaces says so instead of a figure."""
    judged = builds(layout) if judged is None else judged
    kinds: dict[str, list[tuple[ModelBuild, BuildVerdict]]] = defaultdict(list)
    for b, v in judged:
        if b.model_type == "business_model":
            kinds[str(b.context.get("problem_kind") or "unknown")].append((b, v))
    out = {}
    for kind, items in sorted(kinds.items()):
        ok, why = _enough(items)
        if not ok:
            out[kind] = {"builds": len(items), "enough": False, "why": why}
            continue
        met = [b.result.get("met") for b, _ in items if b.result.get("met") is not None]
        eur = [b.spent.get("eur") for b, _ in items if isinstance(b.spent.get("eur"), (int, float))]
        minutes = [b.spent.get("people_minutes") for b, _ in items if isinstance(b.spent.get("people_minutes"), (int, float))]
        failures = Counter(w.split(":")[0] for b, _ in items for w in b.went_wrong)
        out[kind] = {"builds": len(items), "workspaces": len({b.workspace for b, _ in items}), "enough": True,
                     "met_share": round(sum(met) / len(met), 3) if met else None, "met_share_interval": _wilson(sum(met), len(met)), "met_judged": len(met),
                     "median_eur": statistics.median(eur) if eur else None, "median_people_minutes": statistics.median(minutes) if minutes else None,
                     "correct_share": round(sum(v.verdict == "correct" for _, v in items) / len(items), 3),
                     "common_failures": failures.most_common(5), "rows": Counter(b.data.get("rows") for b, _ in items).most_common(),
                     "labelled_rows": Counter(b.data.get("labelled") for b, _ in items).most_common(),
                     "labelled_rows_when_met": Counter(b.data.get("labelled") for b, _ in items if b.result.get("met")).most_common(),
                     "labels_needed": "labelled_rows_when_met: the labelled rows' bands of the builds that met their bar; the kind's curve (curves.json) by band"}
    return out


# ------------------------------------------------------------------------------------------- 32.27 the knowledge base
KNOWLEDGE_TYPES = ("rule", "pitfall", "finding", "leakage_precedent")
CATEGORY_STAGE = {"problem_contract": "problem", "eda": "data", "missingness": "data", "categoricals": "features", "feature_engineering": "features",
                  "feature_reliability": "features", "leakage": "data", "splitting": "split", "evaluation": "evaluation", "generalization": "evaluation",
                  "model_selection": "model", "optimization": "model", "promotion": "deployment", "knowledge": "service", "llm_governance": "service"}
STAGE_OF = {"leakage": "data", "split": "split", "feature": "features", "model": "model", "metric": "evaluation", "holdout": "evaluation",
            "label": "labels", "monitor": "monitoring", "deploy": "deployment"}


def _claude_rules(root: Path) -> list[KnowledgeItem]:
    """CLAUDE.md's numbered rules, each an item resting on the repository's own instructions."""
    path = root / "CLAUDE.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    block = text.split("## Rules", 1)[1].split("\n## ", 1)[0] if "## Rules" in text else ""
    out = []
    for n, rule in re.findall(r"^(\d+)\.\s+(.+?)(?=^\d+\.\s|\Z)", block, flags=re.M | re.S):
        rule = " ".join(rule.split())
        stage = next((v for k, v in STAGE_OF.items() if k in rule.lower()), "evaluation")
        out.append(KnowledgeItem(id=f"CLAUDE-R{n}", model_type="business_model", stage=stage, practice=rule[:600],
                                 evidence=[{"source": "CLAUDE.md", "rule": int(n)}], status="hypothesis"))
    return out


def _external_items(layout: Optional[Layout]) -> list[KnowledgeItem]:
    if layout is None:
        return []
    store, out = Store(layout), {}
    for e in sorted(store.entries("other"), key=lambda e: e.version):
        if e.source.startswith("knowledge/external/"):
            try:
                out[e.source] = KnowledgeItem.model_validate_json(store.read(e))
            except Exception:  # noqa: BLE001
                continue
    return list(out.values())


def knowledge(root: Path = ROOT, folder: Optional[Path] = None, layout: Optional[Layout] = None) -> list[KnowledgeItem]:
    """Every rule, pitfall, finding and leakage precedent of the evidence index, CLAUDE.md's rules and the outside findings,
    proposed as items. An item is accepted only where a person's yes is in ``folder`` (``docs/knowledge/building/``); one whose
    source changed since that yes is flagged, never changed silently."""
    from ..evidence_index import EvidenceIndex

    folder = folder or ACCEPTED
    accepted = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in folder.glob("*.json")} if folder.exists() else {}
    items = []
    for r in EvidenceIndex.load(root).records:
        if r.get("type") not in KNOWLEDGE_TYPES:
            continue
        rid, meta = str(r["record_id"]), r.get("metadata") or {}
        text = f"{r.get('title', '')} {r.get('text', '')}".lower()
        stage = CATEGORY_STAGE.get(str(meta.get("category"))) or ("data" if r["type"] == "leakage_precedent" else
                                                                  next((v for k, v in STAGE_OF.items() if k in text), "evaluation"))
        cites = [str(c) for c in r.get("citations") or [] if not str(c).startswith("evidence/knowledge/")]
        measured = any(c.startswith("evidence/campaigns/") for c in cites) or bool(meta.get("campaign_id"))
        items.append(KnowledgeItem(id=rid, model_type="business_model", stage=stage, practice=str(r.get("title") or rid)[:600], why=str(r.get("text") or "")[:1200],
                                   scope={k: meta[k] for k in ("category", "dataset", "task_type", "kind") if meta.get(k)},
                                   evidence=[{"record": rid, "type": r["type"]}] + [{"cites": c[:200]} for c in cites[:10]],
                                   status="measured_here" if measured else "hypothesis"))
    items += _claude_rules(root) + _external_items(layout)
    out = []
    for item in items:
        yes = accepted.get(item.id)
        if yes:
            changed = yes.get("practice") != item.practice or yes.get("evidence") != json.loads(json.dumps(item.evidence))
            item = item.model_copy(update={"accepted_by": yes.get("accepted_by"),
                                           "flagged": "its source changed since it was accepted: review it" if changed else None})
        out.append(item)
    return out


def accept(item_id: str, by: str, root: Path = ROOT, folder: Optional[Path] = None, layout: Optional[Layout] = None) -> Path:
    """A person's yes to one proposed item, kept whole in ``docs/knowledge/building/<id>.json``."""
    if not by.strip():
        raise ValueError("an item is accepted by a person: say who")
    item = next((i for i in knowledge(root, folder, layout) if i.id == item_id), None)
    if item is None:
        raise KeyError(f"no proposed item {item_id!r}")
    folder = folder or ACCEPTED
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{item_id}.json"
    write_json(path, json.loads(item.model_copy(update={"accepted_by": by.strip(), "flagged": None}).model_dump_json()))
    return path


# ------------------------------------------------------------------------------------------- 32.28 knowledge from outside
def copies(summary: str, source_text: str) -> bool:
    """Whether a summary copies its source: any 12 words in a row taken from it, or more than 30% of its 8-word runs found in
    it; a summary under 8 words copies when it is found in the source whole."""
    words = re.findall(r"\w+", summary.lower())
    src = " " + " ".join(re.findall(r"\w+", source_text.lower())) + " "
    if not words:
        return False
    if len(words) < 8:
        return f" {' '.join(words)} " in src and len(words) >= 5
    runs = [" ".join(words[i:i + 8]) for i in range(len(words) - 7)]
    if sum(f" {r} " in src for r in runs) / len(runs) > 0.3:
        return True
    return any(f" {' '.join(words[i:i + 12])} " in src for i in range(len(words) - 11))


def _keep_external(layout: Layout, item: KnowledgeItem) -> None:
    Store(layout).keep("other", "rnd", f"knowledge/external/{item.id}.json", item.model_dump_json(indent=1).encode(), consent="train",
                       note="an outside finding in our own words, with its citation; untrusted until a DCLab campaign tests it")


def external(layout: Layout, *, summary: str, citation: str, link: str, licence: str, source_text: str, model_type: str = "business_model",
             stage: str = "model", by: str = "research_agent") -> KnowledgeItem:
    """An outside finding kept as a proposal: refused without its citation, link and licence, without the source's text to check
    it against, or when it copies its source."""
    if not (citation.strip() and link.strip() and licence.strip()):
        raise ValueError("an outside finding needs its citation, its link and its licence")
    if not source_text.strip():
        raise ValueError("an outside finding needs its source's text, to check the summary is in our own words")
    if copies(summary, source_text):
        raise ValueError("the summary copies its source: write it in our own words")
    item = KnowledgeItem(id="EXT-" + hashlib.sha256(f"{citation}\n{summary}".encode()).hexdigest()[:12], model_type=model_type, stage=stage,
                         practice=summary[:600], evidence=[{"citation": citation[:300], "link": link[:300], "licence": licence[:100], "by": by[:120]}],
                         status="from_the_literature")
    _keep_external(layout, item)
    return item


def tested(layout: Layout, item_id: str, campaign: str, confirmed: bool, note: str = "") -> KnowledgeItem:
    """A DCLab campaign's test of an outside finding: confirmed, it is measured here and links to the campaign; contradicted, the
    campaign is its counterexample and the item is flagged for review."""
    item = next((i for i in _external_items(layout) if i.id == item_id), None)
    if item is None:
        raise KeyError(f"no outside finding {item_id!r}")
    if not campaign.strip():
        raise ValueError("a test names its campaign")
    link = {"campaign": campaign[:120], "confirmed": bool(confirmed), **({"note": note[:300]} if note else {})}
    item = item.model_copy(update={"evidence": [*item.evidence, link], **({"status": "measured_here"} if confirmed else
                                   {"counterexamples": [*item.counterexamples, campaign[:120]], "flagged": f"contradicted by {campaign[:120]}"})})
    _keep_external(layout, item)
    return item


# ------------------------------------------------------------------------------------------- 32.29 the guides
def guides(layout: Optional[Layout] = None, root: Path = ROOT, folder: Optional[Path] = None,
           judged: Optional[list[tuple[ModelBuild, BuildVerdict]]] = None) -> dict[str, str]:
    """A guide per problem kind from the accepted (and unflagged) items and the judged builds. Every sentence cites an accepted
    item or the catalog and is kept only when ``cited.check`` passes it; a kind with too few builds says so."""
    from .. import cited

    layout = layout or Layout()
    items = [i for i in knowledge(root, folder, layout) if i.accepted_by and not i.flagged]
    sources = {i.id: f"{i.practice} {i.why}" for i in items}
    out = {}
    for kind, facts in sorted(service_catalog(layout, judged).items()):
        claims = [f"{i.practice.rstrip('.')} [{i.id}]." for i in items]
        own = dict(sources)
        if facts.get("enough") and facts.get("met_share") is not None:
            own["CATALOG"] = json.dumps(facts, default=str)
            claims.insert(0, f"Across {facts['builds']} builds, a share of {facts['met_share']} met their bar [CATALOG].")
        kept = cited.check(" ".join(claims), own)["kept"] if claims else []
        lines = [f"# Building a {kind} model", ""]
        if not facts.get("enough"):
            lines.append(f"Too few builds to give typical numbers ({facts['why']}).")
        lines += [f"- {k['text'].rstrip()} " + "".join(f"[{c}]" for c in k["cites"]) for k in kept]  # the check strips the marks; they stay visible
        out[kind] = "\n".join(lines) + "\n"
    return out


# ------------------------------------------------------------------------------------------- 32.30, 32.31 engagements, playbook
def _conversation(layout: Layout) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[tuple[str, str], list[dict[str, Any]]]]:
    """Per (workspace, draft): the draft's own record (the person's first words, its questions, its brief); and per (workspace,
    draft or project): the people's feedback on what DCLab said (``feedback.given``: what it was on, the rating, the kind)."""
    drafts: dict[tuple[str, str], dict[str, Any]] = {}
    for r in _current(layout, "conversations"):
        if r.provenance.source.startswith("drafts/") and r.provenance.source.endswith("/draft.json") and r.body.get("id"):
            drafts[(r.provenance.workspace, str(r.body["id"]))] = r.body
    feedback: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in _current(layout, "sessions"):
        if r.body.get("kind") != "feedback.given":
            continue
        payload = r.body.get("payload") if isinstance(r.body.get("payload"), dict) else {}
        item = {k: payload.get(k) for k in ("on", "rating", "kind", "role", "score") if payload.get(k) is not None} | {"at": r.body.get("at")}
        for key in (r.body.get("conversation"), r.body.get("draft"), r.body.get("project")):
            if key:
                feedback[(r.provenance.workspace, str(key))].append(item)
    return drafts, feedback


def engagements(layout: Optional[Layout] = None, judged: Optional[list[tuple[ModelBuild, BuildVerdict]]] = None,
                latest: Optional[dict[tuple[str, str], dict[str, Any]]] = None) -> list[ServiceEngagement]:
    """Each walk (its newest version) as an engagement: the person's first words, the questions asked and answered, the brief
    agreed, what was delivered and how its result was explained (the final record's notes and limits), the feedback, whether
    it was used; where it stopped when it did. What its records cannot say is listed in ``unknown``."""
    layout = layout or Layout()
    judged = builds(layout) if judged is None else judged
    latest = _latest(layout) if latest is None else latest
    built = {b.id: b for b, _ in judged}
    drafts, feedback = _conversation(layout)
    out = []
    for r in _current(layout, "workflows"):
        if r.type != "walk":
            continue
        b = r.body
        pid, did = b.get("project"), b.get("draft")
        if not (pid or did):
            continue
        ws = r.provenance.workspace
        build = built.get(f"build:{ws}:{pid}") if pid else None
        draft = drafts.get((ws, str(did))) if did else None
        final = (((latest.get((ws, str(pid))) or {}).get("stages") or {}).get("final") or {}) if pid else {}
        steps = b.get("steps") or []
        unknown: dict[str, str] = {}
        request = str(draft.get("problem"))[:2000] if draft and draft.get("problem") else None
        if request is None:
            unknown["request"] = "no draft with the person's first words" if did else "a project started without a draft"
        questions = [{"field": q.get("field"), "answered": bool(q.get("answered")), **({"superseded": True} if q.get("superseded") else {})}
                     for q in (draft or {}).get("questions") or [] if isinstance(q, dict)][:100]
        if not questions:
            questions = [{"step": s.get("ref")} for s in steps if s.get("kind") == "draft_event" and (s.get("detail") or {}).get("kind") == "chat"][:100]
        brief = (draft or {}).get("brief") if isinstance((draft or {}).get("brief"), dict) else {}
        expected = {k: brief.get(k) for k in ("primary_metric", "success_criteria", "horizon", "deliverables") if brief.get(k) not in (None, "", [])}
        if not expected:
            unknown["expected"] = "no brief with a metric, success criteria or deliverables"
        evidence = final.get("evidence") or {}
        explanation = {"notes_for_person": len(evidence.get("notes_for_person") or []) if isinstance(evidence.get("notes_for_person"), list) else bool(evidence.get("notes_for_person")),
                       "limits": len(evidence.get("limits") or []) if isinstance(evidence.get("limits"), list) else bool(evidence.get("limits")),
                       "interval": evidence.get("holdout_primary_metric_ci") is not None} if evidence else {}
        if not explanation:
            unknown["explanation"] = "no final record, so nothing was explained of a result"
        said = feedback.get((ws, str(did)), []) + feedback.get((ws, str(pid)), [])
        if not said:
            unknown["feedback"] = "no feedback given on what DCLab said"
        stopped = None
        if not pid:
            last = steps[-1] if steps else {}
            stopped = f"stopped in the draft after {last.get('kind', 'no step')} {last.get('ref', '')}".strip()[:300]
            unknown["delivered"] = "it stopped in the draft"
        elif build is None:
            unknown["delivered"] = "its project's records are not in the store"
        out.append(ServiceEngagement(id=f"engagement:{ws}:{did or pid}", workspace=ws, context=build.context if build else {}, request=request,
                                     questions=questions, agreement={"bar": build.result.get("bar") if build else None, "brief": bool(brief)}, expected=expected,
                                     delivered={"result": build.result if build else None, "eur": build.spent.get("eur") if build else None,
                                                "minutes": build.spent.get("people_minutes") if build else None, "outcome": b.get("outcome")},
                                     explanation=explanation, handover={"used": build.used if build else None}, feedback=said[:100], stopped=stopped,
                                     unknown=unknown, at=now()))
    return out


def playbook(layout: Optional[Layout] = None, rows: Optional[list[ServiceEngagement]] = None) -> list[PlaybookItem]:
    """What the engagements teach (proposals: the owner accepts each): where they stall, what they cost, by kind; a figure only
    from MIN_BUILDS engagements in MIN_WORKSPACES workspaces."""
    rows = engagements(layout) if rows is None else rows

    def enough(es: list[ServiceEngagement]) -> bool:
        return len(es) >= MIN_BUILDS and len({e.workspace for e in es}) >= MIN_WORKSPACES

    stalled = [e for e in rows if e.stopped]
    where = Counter((e.stopped or "").split(" after ", 1)[-1].split(" ")[0] for e in stalled)
    out = [PlaybookItem(id="PB-failure-points", topic="failure_points", count=len(rows), enough=enough(rows), evidence=[e.id for e in stalled][:200],
                        finding=(f"{len(stalled)} of {len(rows)} engagements stopped before a project was built; most after: {where.most_common(3)}"
                                 if enough(rows) else f"{len(rows)} engagements: too few, or too few workspaces, to say"))]
    by_kind: dict[str, list[ServiceEngagement]] = defaultdict(list)
    for e in rows:
        by_kind[str(e.context.get("problem_kind") or "unknown")].append(e)
    for kind, es in sorted(by_kind.items()):
        eur = [e.delivered.get("eur") for e in es if isinstance(e.delivered.get("eur"), (int, float))]
        out.append(PlaybookItem(id=f"PB-cost-{kind}", topic="time_and_cost", kind=kind, count=len(es), enough=enough(es),
                                finding=(f"median model spend EUR {statistics.median(eur)} over {len(eur)} engagements" if enough(es) and eur
                                         else f"{len(es)} engagements: too few, or too few workspaces, to say"), evidence=[e.id for e in es][:200]))
    return out


PEOPLE_KINDS = ("sessions", "actions", "decisions", "conversations", "model_requests", "workflows", "audit")


def people(layout: Layout) -> dict[str, Any]:
    """Each person's account, sessions and work (32.32), from the records that say whose they are (``body.user``, the
    pseudonymous user id): their role and how they sign in, their sessions, actions, decisions and corrections, the projects
    they worked on, model spend, errors shown to them. A row per person per workspace, under the id alone (never an email, a
    name or a token: the keeper never kept one). Aggregates across people stand beside the rows, with whether there are enough
    of them to say anything."""
    rows: dict[tuple[str, str], dict[str, Any]] = {}

    def row(ws: str, user: str) -> dict[str, Any]:
        return rows.setdefault((ws, user), {"user": user, "workspace": ws, "role": None, "signs_in": None, "active": None, "removed": False, "joined": None,
                                            "sessions": set(), "first_seen": None, "last_seen": None, "records": Counter(), "projects": set(), "eur": 0.0,
                                            "errors": 0})

    for kind in PEOPLE_KINDS:
        for r in _current(layout, kind):
            b = r.body
            user = b.get("user")
            if not isinstance(user, str) or not user:
                continue
            p = row(r.provenance.workspace, user)
            if kind == "sessions" and r.type == "person":
                p.update(role=b.get("role"), signs_in=b.get("signs_in"), active=b.get("active"), joined=b.get("joined") or p["joined"],
                         removed=b.get("event") == "removed")
                continue
            if kind == "sessions" and r.type == "session":
                p["sessions"].add(str(b.get("session")))
                p["first_seen"] = min(filter(None, (p["first_seen"], b.get("started"))), default=p["first_seen"])
                p["last_seen"] = max(filter(None, (p["last_seen"], b.get("last_seen"))), default=p["last_seen"])
                continue
            p["records"][f"{kind}.{r.type}"] += 1
            if isinstance(b.get("project"), str):
                p["projects"].add(b["project"])
            if kind == "model_requests" and isinstance(b.get("cost_eur"), (int, float)):
                p["eur"] += float(b["cost_eur"])
            if kind == "sessions" and str(b.get("kind") or "") in ("ui.error_shown", "model.failed", "run.failed"):
                p["errors"] += 1
            at = r.at
            p["first_seen"] = min(filter(None, (p["first_seen"], at)), default=at)
            p["last_seen"] = max(filter(None, (p["last_seen"], at)), default=at)
    people_rows = [{**{k: v for k, v in p.items() if k not in ("sessions", "records", "projects", "eur")}, "sessions": len(p["sessions"]),
                    "records": dict(sorted(p["records"].items())), "projects": len(p["projects"]), "eur": round(p["eur"], 4)}
                   for _, p in sorted(rows.items())]
    total = Counter()
    for p in people_rows:
        for k, n in p["records"].items():
            total[k] += n
    workspaces = {p["workspace"] for p in people_rows}
    return {"people": people_rows, "count": len(people_rows), "workspaces": len(workspaces), "records": dict(sorted(total.items())),
            "sessions": sum(p["sessions"] for p in people_rows), "eur": round(sum(p["eur"] for p in people_rows), 4),
            "enough": len(people_rows) >= MIN_WORKSPACES and len(workspaces) >= 1,
            "note": "One row per person per workspace, under a pseudonymous id: no email, name, password or token is ever kept. "
                    "A removed person stays (removed true) until the owner's audited forget."}


def write(layout: Optional[Layout] = None) -> dict[str, Any]:
    """Every service record of the store into ``service/`` (``layout.service``), the store read once."""
    layout = layout or Layout()
    latest = _latest(layout)
    judged = [_project_build(k, p) for k, p in sorted(latest.items())] + _student_builds(layout)
    rows = engagements(layout, judged, latest)
    write_json(layout.service("builds"), [{"build": json.loads(b.model_dump_json()), "verdict": json.loads(v.model_dump_json())} for b, v in judged])
    write_json(layout.service("curves"), [json.loads(c.model_dump_json()) for c in curves(layout, judged)])
    write_json(layout.service("catalog"), service_catalog(layout, judged))
    write_json(layout.service("engagements"), [json.loads(e.model_dump_json()) for e in rows])
    write_json(layout.service("playbook"), [json.loads(p.model_dump_json()) for p in playbook(layout, rows)])
    kept_people = people(layout)
    write_json(layout.service("people"), kept_people)
    return {"people": kept_people["count"], "builds": len(judged), "correct": sum(v.verdict == "correct" for _, v in judged), "incorrect": sum(v.verdict == "incorrect" for _, v in judged),
            "engagements": len(rows)}


__all__ = ["builds", "curves", "service_catalog", "knowledge", "accept", "external", "tested", "copies", "guides", "engagements", "playbook", "people", "write",
           "MIN_BUILDS", "MIN_WORKSPACES"]
