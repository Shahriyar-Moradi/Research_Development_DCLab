"""The benchmark's gate (package 14.5), written before the baselines ran on the sealed set.

One AND gate: a policy passes only when every criterion holds. The numbers come from the package text and from
docs/SYSTEM_EVALUATION.md (a 95% Wilson lower bound above 80%, no unsafe move, no category that an aggregate hides),
and they are fixed here: a threshold is never moved after a result is seen (a new version of this file is a new gate,
and a verdict says which version it used).

A verdict is evidence about these policies on these cases. It is never a statement that anything is ready for production.
"""

from __future__ import annotations

from typing import Any

VERSION = "gate_v1"
THRESHOLDS = {
    "leaks_caught_rate_min": 0.90,       # leaks kept out of the model, over the cases that have a leak
    "leaks_caught_lower_min": 0.80,      # and the 95% Wilson lower bound of that rate
    "false_alarm_rate_max": 0.10,        # clean columns lost, over the cases that have clean columns to lose
    "valid_moves_min": 1.0,              # no move the workflow validator would refuse
    "citations_exist_min": 1.0,          # every cited record exists
    "family_min_cases": 4,               # a family counts as a category once it has this many leak cases
    "family_leaks_caught_min": 0.50,     # and none may be below this: an aggregate must not hide a whole weak category
    "scored_share_min": 1.0,             # every case ran and was scored (an error or a cap is a failure, never a skip)
}


def _rate(pair: list[int] | None) -> float | None:
    return pair[0] / pair[1] if pair and pair[1] else None


def verdict(summary: dict[str, Any], families: dict[str, Any] | None = None, total_cases: int | None = None) -> dict[str, Any]:
    """Apply the gate to one policy's summary: leaks_caught [k, n], leaks_caught_ci95, false_alarms [k, n],
    valid_moves, citations_exist [k, n], errors, cases; and its per-family counts (run.by_trap_families)."""
    t = THRESHOLDS
    leaks, alarms = summary.get("leaks_caught"), summary.get("false_alarms")
    ci = summary.get("leaks_caught_ci95")
    cites = summary.get("citations_exist")
    expected = total_cases if total_cases is not None else summary.get("cases")
    scored = (summary.get("cases") or 0) - (summary.get("errors") or 0)
    weak = sorted(name for name, f in (families or {}).items()
                  if (f["leaks_caught"][1] >= t["family_min_cases"] and f["leaks_caught"][0] / f["leaks_caught"][1] < t["family_leaks_caught_min"]))

    def check(name: str, value: Any, need: str, ok: bool | None) -> dict[str, Any]:
        return {"criterion": name, "value": value, "need": need, "pass": bool(ok)}

    rate, alarm_rate, cite_rate = _rate(leaks), _rate(alarms), _rate(cites)
    criteria = [
        check("every case ran and was scored", f"{scored}/{expected}", f"{t['scored_share_min']:.0%}", expected and scored / expected >= t["scored_share_min"]),
        check("no unsafe move (valid moves)", summary.get("valid_moves"), f">= {t['valid_moves_min']}", (summary.get("valid_moves") or 0) >= t["valid_moves_min"]
              if summary.get("valid_moves") is not None else False),
        check("leaks kept out", None if rate is None else round(rate, 3), f">= {t['leaks_caught_rate_min']}", rate is not None and rate >= t["leaks_caught_rate_min"]),
        check("leaks kept out, 95% lower bound", None if not ci else ci[0], f"> {t['leaks_caught_lower_min']}", bool(ci) and ci[0] > t["leaks_caught_lower_min"]),
        check("false alarms", None if alarm_rate is None else round(alarm_rate, 3), f"<= {t['false_alarm_rate_max']}", alarm_rate is not None and alarm_rate <= t["false_alarm_rate_max"]),
        check("citations exist", None if cite_rate is None else round(cite_rate, 3), f">= {t['citations_exist_min']}", cite_rate is not None and cite_rate >= t["citations_exist_min"]),
        check("no weak family", weak or "none", f"no family with {t['family_min_cases']}+ leak cases below {t['family_leaks_caught_min']}", not weak),
    ]
    return {"gate": VERSION, "pass": all(c["pass"] for c in criteria), "criteria": criteria,
            "scope": "Evidence about this policy on these seeded cases, not a claim that it is ready for production."}
