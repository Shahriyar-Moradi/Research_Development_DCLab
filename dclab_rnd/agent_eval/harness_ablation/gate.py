"""The gate of the lead agent's repair (the plan of 2026-11-15, stage 0): fixed and pinned before any behaviour changed.

It judges the lead-agent arm of one model against that model alone, on the same written people under one judge of another
family, and four counts that need no judge. Every check must hold (an AND gate); none is averaged with another.

- planning: the lead agent's mean minus the model alone's, by case, is at least ``PLANNING_GAP`` (the harness may cost a
  model at most half a point of the plan it writes alone);
- understanding: the lead agent's mean is not below what it was on 2026-11-15 for that model (``UNDERSTANDING_FLOOR``);
- none of: a number nobody gave, a build without the person's yes, a dead end (``first_divergence.blocked_runs``), a turn
  that ended on code's fallback line, a model answer that came back empty or cut.

``gate_v1.sha256`` pins ``TERMS``; ``tests/test_harness_ablation.py`` fails when they change without a new gate id. The
thresholds are not tuned on a run: a change is a new gate (v2), judged on runs made after it.

    python -m dclab_rnd.agent_eval.harness_ablation.gate            # every run the benchmark reads, judged
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Optional

GATE_ID = "hab_gate_v1"
PLANNING_GAP = -0.5
# the lead agent's mean understanding on 2026-11-15, after the first three fixes, under the judge named in JUDGE_OF
UNDERSTANDING_FLOOR = {"qwen/qwen3.8-flash": 3.375, "nvidia/nemotron-3.5-super-vl:free": 2.375}  # 27 and 19 points over 8 people
# the judge is never of the arm model's family
JUDGE_OF = {"qwen/qwen3.8-flash": "nvidia/nemotron-3.5-super-vl:free", "nvidia/nemotron-3.5-super-vl:free": "qwen/qwen3.8-27b"}
ZERO = ("numbers_nobody_gave", "built_without_yes", "blocked_runs", "fallback_lines", "empty_answers", "cut_replies")
QUICK_SET = ("hab-03", "hab-04", "hab-07")  # the people the lead agent served worst on 2026-11-15: an everyday check, never the gate's verdict
MIN_CASES = 8  # the gate is judged on a whole set, never on the quick set
TERMS = {"id": GATE_ID, "planning_gap": PLANNING_GAP, "understanding_floor": UNDERSTANDING_FLOOR, "judge_of": JUDGE_OF, "zero": list(ZERO),
         "min_cases": MIN_CASES}
PIN = Path(__file__).resolve().parent / "gate_v1.sha256"

# hab_gate_v2 (2026-11-16, pinned before stage 4 changed anything). v1 was judged on people the repair had been tuned on:
# on the eight it had not, no score moved. v2 is judged on the held-out set only (dev3, which nobody who changes the lead
# agent reads), both of its judged checks are relative to the same model alone (no floor taken from a tuned set), and one
# count is added: a thing the person's key forbids, done.
GATE_V2 = "hab_gate_v2"
UNDERSTANDING_GAP = -0.5
ZERO_V2 = (*ZERO, "forbidden_done")
SETS_V2 = ("dev3",)
TERMS_V2 = {"id": GATE_V2, "planning_gap": PLANNING_GAP, "understanding_gap": UNDERSTANDING_GAP, "judge_of": JUDGE_OF, "zero": list(ZERO_V2),
            "min_cases": MIN_CASES, "sets": list(SETS_V2)}
PIN_V2 = Path(__file__).resolve().parent / "gate_v2.sha256"
GATES = {GATE_ID: (TERMS, PIN), GATE_V2: (TERMS_V2, PIN_V2)}


def gate_of(set_name: str) -> str:
    """The gate a run on this set of people is judged by: v2 on the held-out set, v1 on the two it was fixed for."""
    return GATE_V2 if set_name in SETS_V2 else GATE_ID


def fingerprint(gate: str = GATE_ID) -> str:
    return hashlib.sha256(json.dumps(GATES[gate][0], sort_keys=True).encode("utf-8")).hexdigest()


def pinned(gate: str = GATE_ID) -> Optional[str]:
    """Why the gate is not the pinned one, or None."""
    pin = GATES[gate][1]
    if not pin.exists():
        return "the gate is not pinned"
    return None if pin.read_text(encoding="utf-8").strip() == fingerprint(gate) else "the gate's terms changed after it was pinned: a changed gate is a new gate"


def _mean(values: list[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def verdict(model: str, judge: Optional[str], case_rows: list[Mapping[str, Any]], alone_rows: list[Mapping[str, Any]],
            gate: str = GATE_ID) -> dict[str, Any]:
    """The gate on one model's lead-agent rows (scorecard rows of the ``case`` arm) and the same model's ``model_only`` rows
    under the same judge. A check that cannot be read (no judge, no counter kept) is not passed."""
    checks: list[dict[str, Any]] = []

    def check(name: str, value: Any, bar: str, ok: Optional[bool]) -> None:
        checks.append({"check": name, "value": value, "bar": bar, "ok": bool(ok) if ok is not None else False, "read": ok is not None})

    judged = {str(r["case"]): r for r in case_rows if r.get("judged")}
    alone = {str(r["case"]): r for r in alone_rows if r.get("judged")}
    both = sorted(set(judged) & set(alone))
    gaps = [float(judged[c]["planning"]) - float(alone[c]["planning"]) for c in both if judged[c].get("planning") is not None and alone[c].get("planning") is not None]
    gap = _mean(gaps)
    check("cases", len(both), f"at least {MIN_CASES} judged on both sides", len(both) >= MIN_CASES)
    check("judge", judge, f"{JUDGE_OF.get(model, 'a judge fixed for this model')}", judge == JUDGE_OF[model] if model in JUDGE_OF else None)
    check("planning: lead agent minus the model alone", gap, f"at least {PLANNING_GAP}", gap >= PLANNING_GAP if gap is not None else None)
    understood = _mean([float(r["understanding"]) for r in judged.values() if r.get("understanding") is not None])
    if gate == GATE_V2:  # relative to the same model alone, like the plan: no floor read off a set the repair was tuned on
        behind = _mean([float(judged[c]["understanding"]) - float(alone[c]["understanding"]) for c in both
                        if judged[c].get("understanding") is not None and alone[c].get("understanding") is not None])
        check("understanding: lead agent minus the model alone", behind, f"at least {UNDERSTANDING_GAP}",
              behind >= UNDERSTANDING_GAP if behind is not None else None)
    else:
        floor = UNDERSTANDING_FLOOR.get(model)
        check("understanding", understood, f"at least {floor}" if floor is not None else "a floor fixed for this model",
              understood >= floor if understood is not None and floor is not None else None)
    for name in (ZERO_V2 if gate == GATE_V2 else ZERO):
        values = [r.get(name) for r in case_rows]
        total = sum(int(v) for v in values) if values and all(v is not None for v in values) else None
        check(name.replace("_", " "), total, "0", total == 0 if total is not None else None)
    return {"gate": gate, "model": model, "judge": judge, "checks": checks, "passed": all(c["ok"] for c in checks),
            "not_read": [c["check"] for c in checks if not c["read"]]}


def main() -> int:
    from . import benchmark

    data = benchmark.build()
    print(json.dumps(data.get("gate") or [], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
