"""Outcomes written where the number is observed (package 23.5): the final stage's holdout, the final stage's approval
(a promotion), and a business number a person types. Every decision of the project, and of the drafts that became it,
gets its row; one ``outcome.observed`` event says what was observed and how many decisions it reached. Never a value of
a table. Nothing is written inside a replay (``experience.quiet``), and a ledger that is away never fails a stage."""

from __future__ import annotations

import secrets
import sys
from typing import Any, Optional

from dclab_core import outcomes as contract

from . import _consent_now, _now, batch, current_ledger


def _interval(ci: Any) -> Optional[tuple[float, float]]:
    try:
        if isinstance(ci, dict):
            low, high = ci.get("low", ci.get("lower")), ci.get("high", ci.get("upper"))
        else:
            low, high = ci[0], ci[1]
        return (float(low), float(high)) if low is not None and high is not None else None
    except (TypeError, ValueError, IndexError, KeyError):
        return None


NONE = {"decisions": 0, "verdict": "unknown", "written": False}


def _records(target: Any, project_id: str) -> list[Any]:
    """The project's decisions and those its drafts made before it was built (Home's family, workflow and cleaning)."""
    records = list(target.records(project=project_id))
    try:
        from .. import context

        drafts = getattr(context.services(), "drafts", None)
        for d in (drafts.list(limit=100_000) if drafts is not None else []):
            if str(d.get("project_id") or "") == project_id and d.get("id"):
                records += target.records(draft=d["id"])
    except Exception:  # noqa: BLE001 - no draft store here: the project's own decisions stand
        pass
    return records


def observe(project_id: str, *, metric: str, value: Any, observed: str, by: str, reference: Any = None, higher_is_better: Any = None,
            interval: Any = None, note: str | None = None, run: str | None = None, verdict: str | None = None, ledger: Any = None) -> dict[str, Any]:
    """One observation, written as an outcome of every decision the project made so far; the count and the verdict."""
    from .. import experience

    if experience._QUIET.get():  # a replay on a scratch copy is not something that happened to the project
        return dict(NONE)
    target = ledger if ledger is not None else current_ledger()
    try:
        number = float(value)
        ref = float(reference) if reference is not None else None
    except (TypeError, ValueError):
        return dict(NONE)
    verdict = verdict or contract.judge(number, ref, higher_is_better)
    written = 0
    if target is not None:
        try:
            records = _records(target, project_id)
            level = _consent_now(target, project_id)
        except Exception as exc:  # noqa: BLE001 - never stop a stage because the ledger is briefly away
            print(f"ledger: could not read the decisions for outcomes: {type(exc).__name__}", file=sys.stderr, flush=True)
            return dict(NONE)
        docs = []
        for r in records:
            try:
                docs.append(contract.Outcome(id="do" + secrets.token_hex(10), record=r.id, point=r.point, metric=str(metric)[:80], value=number,
                                             interval=_interval(interval), reference=ref,
                                             higher_is_better=bool(higher_is_better) if higher_is_better is not None else None,
                                             observed=observed, verdict=verdict, by=str(by)[:120], note=(str(note)[:300] or None) if note else None,
                                             workspace=target.workspace, project=project_id, run=run, at=_now(), consent=level))
            except Exception as exc:  # noqa: BLE001 - the contract refused one: the rest stand
                print(f"ledger: outcome of {r.point} refused: {type(exc).__name__}", file=sys.stderr, flush=True)
        try:
            with batch():
                from . import _write

                for doc in docs:
                    _write(target, "outcome", doc)
            written = len(docs)
        except Exception as exc:  # noqa: BLE001 - never stop a stage because the ledger is briefly away
            print(f"ledger: could not record {len(docs)} outcomes: {type(exc).__name__}", file=sys.stderr, flush=True)
    experience.record("outcome.observed", "person" if observed == "business" else "system",
                      {"metric": str(metric)[:80], "value": number, "reference": ref, "observed": observed, "verdict": verdict, "decisions": written,
                       "note": (str(note)[:300] or None) if note else None},
                      project=project_id, **({"run": run} if run else {}))
    return {"decisions": written, "verdict": verdict, "written": bool(written)}


def from_final(project_id: str, record: dict[str, Any], run_id: str | None, higher_is_better: Any) -> dict[str, Any]:
    """The final stage's one holdout read, as the outcome of every decision of the project."""
    try:
        evidence = record.get("evidence") or {}
        metric = record.get("primary_metric")
        value = (evidence.get("holdout_metrics") or {}).get(metric) if metric else None
        if metric is None or value is None:
            return dict(NONE)
        return observe(project_id, metric=metric, value=value, observed="holdout", by="final_stage", reference=evidence.get("cv_selected_metric_mean"),
                       higher_is_better=higher_is_better, interval=evidence.get("holdout_primary_metric_ci"), run=run_id)
    except Exception as exc:  # noqa: BLE001 - the holdout is read: a failed outcome never fails the stage
        print(f"ledger: outcomes of the final stage not kept: {type(exc).__name__}", file=sys.stderr, flush=True)
        return dict(NONE)


def from_approval(project_id: str, record: dict[str, Any] | None, actor: str) -> dict[str, Any]:
    """The final stage approved: the model is promoted. Its holdout number is judged exactly as the final stage judged it
    (the direction is the holdout row's own), so a promotion never turns a bad holdout into a good outcome."""
    try:
        evidence = (record or {}).get("evidence") or {}
        metric = (record or {}).get("primary_metric")
        value = (evidence.get("holdout_metrics") or {}).get(metric) if metric else None
        if metric is None or value is None:
            return dict(NONE)
        target = current_ledger()
        direction = None
        if target is not None:
            held = [o for o in target.outcomes(project=project_id) if o.observed == "holdout" and o.higher_is_better is not None]
            if held:
                direction = max(held, key=lambda o: (o.at, o.id)).higher_is_better
        return observe(project_id, metric=metric, value=value, observed="promotion", by="intern" if actor == "agent" else "person",
                       reference=evidence.get("cv_selected_metric_mean"), higher_is_better=direction, interval=evidence.get("holdout_primary_metric_ci"),
                       run=(record or {}).get("run") if isinstance((record or {}).get("run"), str) else ((record or {}).get("run") or {}).get("run"))
    except Exception as exc:  # noqa: BLE001
        print(f"ledger: outcomes of the approval not kept: {type(exc).__name__}", file=sys.stderr, flush=True)
        return dict(NONE)


def by_point(ledger: Any = None, project: str | None = None) -> dict[str, dict[str, int]]:
    """Per point: well, badly, unknown and none (the Decisions tab)."""
    target = ledger if ledger is not None else current_ledger()
    if target is None:
        return {}
    return contract.by_point(target.records(project=project) if project else target.records(), target.outcomes(project=project) if project else target.outcomes())


__all__ = ["observe", "from_final", "from_approval", "by_point"]
