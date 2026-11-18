"""e2e_gate_v1 judged by code on the held-out run's scores. The words of gate_v1.json (pinned 2026-11-18, before any live
run) are read here as written; the one choice they leave open is fixed here, before the held-out run, and said:
a case whose plain arm has no real score (unfinished, or features that are not the table's columns) leaves its pair out
of the paired comparison, the count of pairs is reported, and a kind with no pair cannot be judged, so the gate cannot pass."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
GATE = HERE / "gate_v1.json"
PIN = HERE / "gate_v1.sha256"
NEED = {"finished": 8, "leaks_kept_out": 8, "inside": 6}
F1_MARGIN = 0.02  # ticket routing: the mean paired difference in macro F1 may be at most this much worse
MAE_SHARE = 0.05  # delivery time: the mean paired difference in MAE may be at most this share of plain's mean MAE worse


def pinned() -> bool:
    return hashlib.sha256(GATE.read_bytes()).hexdigest() == PIN.read_text(encoding="utf-8").strip()


def _paired(rows: list[dict[str, Any]], kind: str) -> dict[str, Any]:
    by = {(r["case"], r["arm"]): r for r in rows}
    pairs = []
    for case in sorted({r["case"] for r in rows if r["case"].startswith(kind + "-")}):
        d, p = by.get((case, "dclab_agent")) or {}, by.get((case, "plain")) or {}
        if d.get("real") is not None and p.get("real") is not None:
            pairs.append((float(d["real"]), float(p["real"])))
    out: dict[str, Any] = {"pairs": len(pairs)}
    if not pairs:
        return {**out, "met": None, "why": "no case where both arms have a real score"}
    diff = sum(a - b for a, b in pairs) / len(pairs)
    if kind == "ticket_routing":  # macro F1, higher is better
        return {**out, "mean_difference": round(diff, 4), "bound": -F1_MARGIN, "met": diff >= -F1_MARGIN}
    plain_mean = sum(b for _, b in pairs) / len(pairs)  # MAE, lower is better
    bound = MAE_SHARE * plain_mean
    return {**out, "mean_difference": round(diff, 4), "bound": round(bound, 4), "met": diff <= bound}


def judge(scores: list[dict[str, Any]], cases: int = 8) -> dict[str, Any]:
    """Every criterion with its count and whether it is met; ``passed`` only when all are met and none is unjudged."""
    agent = [r for r in scores if r["arm"] == "dclab_agent"]
    finished = [r for r in agent if r.get("finished")]
    checks = {
        "dclab_agent_finished": {"count": len(finished), "of": cases, "met": len(finished) >= NEED["finished"]},
        "dclab_agent_leaks_kept_out": {"count": sum(bool(r.get("leaks_kept_out")) for r in finished), "of": cases},
        "dclab_agent_holdout_reads": {"exactly_one": sum(r.get("holdout_uses") == 1 for r in finished), "of": len(finished)},
        "dclab_agent_real_inside_interval": {"count": sum(r.get("real_inside_interval") is True for r in finished), "of": cases},
        "dclab_agent_real_score_against_plain": {k: _paired(scores, k) for k in ("ticket_routing", "delivery_time")},
    }
    checks["dclab_agent_leaks_kept_out"]["met"] = checks["dclab_agent_leaks_kept_out"]["count"] >= NEED["leaks_kept_out"]
    reads = checks["dclab_agent_holdout_reads"]
    reads["met"] = bool(finished) and reads["exactly_one"] == len(finished)
    checks["dclab_agent_real_inside_interval"]["met"] = checks["dclab_agent_real_inside_interval"]["count"] >= NEED["inside"]
    paired = checks["dclab_agent_real_score_against_plain"]
    met = [paired[k]["met"] for k in paired]
    paired["met"] = None if None in met else all(met)
    verdicts = [c["met"] for c in checks.values()]
    return {"gate": "e2e_gate_v1", "pinned": pinned(), "checks": checks,
            "passed": pinned() and all(v is True for v in verdicts),
            "not_judged": [k for k, c in checks.items() if c["met"] is None]}
