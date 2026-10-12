"""The agentic layer judged (package 18.9): every role, and the whole workflow, from the result its gate was judged on.

Each role was measured on its own before it may be switched on (18.2 to 18.8): rules alone, the role, and where it was
measured a plain prompted model, on replicas or a sealed set, with a gate written before the run. This module reads
those results (never re-runs them) into one table, with 95 % Wilson intervals where the result counts successes, the
euros spent and euros per solved problem, and the unsafe moves each result recorded. ``make roles-judged`` writes
``dclab_rnd/roles_judged.json``; ``dclab_rnd/roles.py`` reads it, so a role is on by default only where its gate
passed, and the others stay available and off. The file is generated, never edited: a role's verdict changes only with
a new result.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "dclab_rnd/roles_judged.json"
CAMPAIGNS = ROOT / "evidence/campaigns"
# the result each role is judged on: the one run against its gate (a sealed set where there is one), newest gate version
DECIDING = {
    "problem_analyst": "analyst_v1/results/ANA-004_sealed.json",
    "brief_checker": "brief_checker_v1/results/CHK-001_all.json",
    "data_investigator": "investigator_v1/results/INV-002_sealed.json",
    "solution_planner": "roles_v1/results/ROL-001_planner.json",
    "feature_engineer": "roles_v1/results/ROL-005_engineer_v2.json",
    "critic": "roles_v1/results/ROL-003_critic.json",
    "reporter": "roles_v1/results/ROL-006_reporter.json",
    "team": "team_v1/results/TEAM-001_three_replicas.json",
    "layer": "team_v1/results/LAYER-001_judged.json",  # the whole layer on the sealed benchmark (rules, plain, roles)
}
PIPELINE = ("problem_analyst", "data_investigator", "solution_planner")  # on only when the whole layer passed too


def wilson(hits: int, n: int) -> Optional[list[float]]:
    if not n:
        return None
    z, p = 1.959964, hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def _rate(hits: int, n: int) -> dict[str, Any]:
    return {"hits": hits, "of": n, "rate": round(hits / n, 3) if n else None, "ci95": wilson(hits, n)}


def _compare(role: str, d: dict[str, Any]) -> tuple[dict[str, Any], int, list[str]]:
    """(the comparison, problems solved, unsafe moves recorded) for one role's result."""
    s = d.get("summary") or (d.get("result") or {}).get("summary") or {}
    rows = (d.get("result") or {}).get("rows") or d.get("rows") or []
    if role == "problem_analyst":
        fields = s["rules"]["fields"]
        cmp = {name: {"right": _rate(s[name]["right"], fields), "wrong": _rate(s[name]["wrong"], fields)}
               for name in ("rules", "analyst", "plain") if name in s}
        return cmp, s.get("analyst", {}).get("right", 0), [f"{s.get('analyst', {}).get('wrong', 0)} fields read wrong"]
    if role == "data_investigator":
        cmp = {name: {"found": _rate(s[name]["found"], s[name]["traps"]), "false_alarms": s[name]["false_alarms"]}
               for name in ("rules", "investigator", "plain") if name in s}
        return cmp, s.get("investigator", {}).get("found", 0), [f"{s.get('investigator', {}).get('false_alarms', 0)} clean columns accused"]
    if role == "brief_checker":
        return ({"caught": _rate(s["caught"], s["wrong_fields"]), "noise": _rate(s["noise"], s["right_fields"])}, s["caught"],
                [f"{s['noise']} right fields questioned"])
    if role == "solution_planner":
        return ({"rules": _rate(s["rules"], s["decisions"]), "planner": _rate(s["planner"], s["decisions"])}, s["planner"],
                [f"{s.get('validator_problems', 0)} plans the validator refused"])
    if role == "feature_engineer":
        gains = [{"seed": r.get("seed"), "gain_over_raw": r.get("gain_over_raw"), "against_behaviour": r.get("against_behaviour")} for r in rows]
        return {"seeds": gains}, sum(1 for r in rows if (r.get("gain_over_raw") or 0) >= 0.05), [
            f"{sum(len(r.get('refused') or []) for r in rows)} programs refused by the language's checks"]
    if role == "critic":
        better = [r for r in rows if r.get("better")]
        return ({"replicas": [{"replica": r.get("replica"), "one_pass": r.get("one_pass"), "rules_loop": r.get("rules_loop"),
                               "with_critic": r.get("loop")} for r in rows]}, len(better),
                [f"holdout read {r.get('holdout_uses')} times on {r.get('replica')}" for r in rows if r.get("holdout_uses") != 1])
    if role == "reporter":
        kept = sum(r.get("kept", 0) for r in rows)
        failed = sum(len(r.get("failed_recheck") or []) for r in rows)
        return ({"kept": kept, "dropped": sum(r.get("dropped", 0) for r in rows), "failed_recheck": failed},
                sum(1 for r in rows if r.get("kept")), [f"{failed} kept sentences failing the re-check"])
    if role == "layer":
        v = d.get("verdict") or {}
        caught = v.get("caught") or {}
        return ({"caught": caught, "false_alarms": v.get("false_alarms"), "ci95": v.get("ci95"), "eur_per_caught_trap": v.get("eur_per_caught_trap")},
                (caught.get("roles") or [0])[0], [])
    if role == "team":
        ok = [r for r in rows if r.get("acceptance_ok")]
        return ({"replicas": [{"kind": r.get("kind"), "status": r.get("status"), "acceptance_ok": r.get("acceptance_ok"),
                               "roles_answered": r.get("roles_answered")} for r in rows]}, len(ok),
                [f"{r.get('kind')}: a move not validated" for r in rows if not r.get("moves_ok")])
    return {}, 0, []


def judge() -> dict[str, Any]:
    rows = []
    layer_path = CAMPAIGNS / DECIDING["layer"]
    layer_pass = layer_path.exists() and (json.loads(layer_path.read_text(encoding="utf-8")).get("verdict") or {}).get("status") == "pass"
    for role, rel in DECIDING.items():
        path = CAMPAIGNS / rel
        if not path.exists():
            rows.append({"role": role, "result": rel, "verdict": "not measured", "on": False})
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        verdict = (d.get("verdict") or {}).get("status", "not scored")
        comparison, solved, unsafe = _compare(role, d)
        spent = float(d.get("spent_eur") or 0.0)
        rows.append({"role": role, "result": rel, "experiment_id": d.get("experiment_id") or d.get("id"),
                     "gate": (d.get("gate") or {}).get("version") if isinstance(d.get("gate"), dict) else d.get("gate"),
                     "verdict": verdict, "on": verdict == "pass" and role not in ("team", "layer") and (role not in PIPELINE or layer_pass),
                     **({"waits_for": "the whole layer's gate (layer_gate_v1)"} if role in PIPELINE and not layer_pass else {}),
                     "comparison": comparison,
                     "spent_eur": round(spent, 4), "solved": solved,
                     "eur_per_solved": round(spent / solved, 5) if solved else None, "unsafe": unsafe})
    return {"generated_by": "make roles-judged", "package": "18.9",
            "rule": "a role is on by default only where the result its gate was judged on passed, and the analyst, the investigator "
                    "and the planner only when the whole layer's comparison on the sealed benchmark passed too; the others stay "
                    "available and off; the team is reported, and is switched by the roles it holds",
            "roles": rows}


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.judged")
    parser.add_argument("--check", action="store_true", help="fail when the committed file differs from a fresh judgement")
    args = parser.parse_args(argv)
    doc = judge()
    text = json.dumps(doc, indent=1, ensure_ascii=False) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("dclab_rnd/roles_judged.json is stale: run make roles-judged")
            return 1
        print("ok    roles_judged.json")
        return 0
    OUT.write_text(text, encoding="utf-8")
    for r in doc["roles"]:
        print(f"{r['role']:<18} {r['verdict']:<12} {'on ' if r['on'] else 'off'}  EUR {r.get('spent_eur')}  per solved {r.get('eur_per_solved')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
