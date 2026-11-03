"""DCLab Watch's monitors and the bug loop (B8 of the 2026-11-03 plan): what the agents and their models do now, and where a bug becomes a test.

**Monitors**, per agent purpose over the last ``days`` (from the gateway's usage log, files or PostgreSQL): requests, failures
(any outcome but ``ok``) with the failure rate's 95% Wilson interval, the median and 95th-percentile seconds, tokens, euro in all
and per request. ``RULES`` (fixed, never tuned to a reading) mark a purpose: red when its failure rate's lower bound is above
5% on 20 requests or more; amber when its 95th-percentile request takes over a minute, or it spends over EUR 5 a day.

**The bug loop** already exists in pieces; this shows where each case stands: sessions the failure detectors flagged become
candidates (``datasets/failures/benchmark_candidates``), a reviewer accepts or rejects each (``reviews.json``), and an accepted
session is replayed by the harness suite as a regression case (``agent_eval.regression.session_cases``, ``make harness-eval``).

    python -m dclab_rnd.bench monitors [--days 7]
    GET /api/admin/bench/monitors
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from dclab_core import bench as core

RULES = {"failure_lower_bound": 0.05, "min_requests": 20, "p95_seconds": 60.0, "eur_per_day": 5.0}
LIMIT = 50_000  # the usage rows read at most


def _percentile(values: list[float], q: float) -> Optional[float]:
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return round(values[lo] + (values[hi] - values[lo]) * (k - lo), 2)


def agents(rows: Iterable[dict[str, Any]], days: int = 7, now: Optional[datetime] = None) -> list[dict[str, Any]]:
    """Per purpose: what its requests did in the last ``days``, and its mark under ``RULES`` with the reasons."""
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).isoformat()
    by: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if str(r.get("at") or "") >= since and r.get("purpose"):
            by[str(r["purpose"])].append(r)
    out = []
    for purpose, rs in sorted(by.items()):
        n = len(rs)
        failed = sum(1 for r in rs if (r.get("outcome") or "ok") != "ok")
        lo, hi = core.wilson(failed, n)
        seconds = [float(r["seconds"]) for r in rs if isinstance(r.get("seconds"), (int, float))]
        eur = round(sum(float(r.get("cost_eur") or 0) for r in rs), 4)
        p95 = _percentile(seconds, 0.95)
        reasons = []
        if n >= RULES["min_requests"] and lo is not None and lo > RULES["failure_lower_bound"]:
            reasons.append(f"failure rate at least {lo:.0%} ({failed} of {n})")
        mark = "red" if reasons else "ok"
        if p95 is not None and p95 > RULES["p95_seconds"]:
            reasons.append(f"95% of requests within {p95:.0f} s, over {RULES['p95_seconds']:.0f} s")
        if eur / max(days, 1) > RULES["eur_per_day"]:
            reasons.append(f"EUR {eur / max(days, 1):.2f} a day, over {RULES['eur_per_day']:.0f}")
        if reasons and mark == "ok":
            mark = "amber"
        out.append({"purpose": purpose, "requests": n, "failed": failed, "failure_rate": round(failed / n, 4) if n else None,
                    "failure_interval": [lo, hi], "p50_seconds": _percentile(seconds, 0.5), "p95_seconds": p95,
                    "tokens": sum(int(r.get("input_tokens") or 0) + int(r.get("output_tokens") or 0) for r in rs),
                    "eur": eur, "eur_per_request": round(eur / n, 5) if n else None,
                    "models": sorted({str(r.get("model")) for r in rs if r.get("model")}), "mark": mark, "reasons": reasons})
    return out


def bug_loop(home: Path) -> dict[str, Any]:
    """Where each flagged session stands: candidates, reviewed (accepted, rejected), and the regression cases built from them."""
    folder = Path(home) / "datasets" / "failures" / "benchmark_candidates"
    candidates = _jsonl(folder / "rows.jsonl")
    try:
        reviews = json.loads((Path(home) / "datasets" / "failures" / "reviews.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        reviews = {}
    accepted = sum(1 for v in reviews.values() if isinstance(v, dict) and v.get("verdict") == "accept")
    rejected = sum(1 for v in reviews.values() if isinstance(v, dict) and v.get("verdict") == "reject")
    from ..agent_eval import regression

    cases = regression.session_cases(Path(home))
    ids = {str(c.get("id")) for c in candidates}
    return {"candidates": len(candidates), "waiting_review": len(ids - set(reviews)), "accepted": accepted, "rejected": rejected,
            "regression_cases": len(cases), "how": "make datasets builds the candidates; accept one in datasets/failures/reviews.json; "
                                                   "make harness-eval replays the accepted ones"}


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def summary(home: Path, days: int = 7, usage: Any = None) -> dict[str, Any]:
    if usage is None:
        from ..models.usage import open_usage

        usage = open_usage(Path(home))
    since = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    window = getattr(usage, "window", None)
    rows = window(since, LIMIT) if callable(window) else [r for r in usage.recent(LIMIT) if str(r.get("at") or "") >= since]
    return {"days": days, "rules": RULES, "agents": agents(rows, days), "bug_loop": bug_loop(Path(home)),
            "truncated": len(rows) >= LIMIT}


__all__ = ["agents", "bug_loop", "summary", "RULES"]
