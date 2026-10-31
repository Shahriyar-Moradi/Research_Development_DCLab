"""The lead agent against Claude alone, person by person (the owner, 2026-10-31: "a real benchmark that shows the
differences between the lead agent and Claude alone").

Only runs where both arms met the same people count: the sealed run (CEX-013) and the practice run CEX-009. For each
measure: how many people the lead agent did better, the same or worse on; each arm's mean; and the paired difference with
a 90% bootstrap interval (the gate's own resampling). Nothing is run and nothing is spent. Sealed people are named by
their ids only.

``python -m dclab_rnd.agent_eval.conversation_exam.headtohead`` writes ``HEAD_TO_HEAD.md`` and ``.json`` beside the results.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Callable, Mapping, Optional

from . import CAMPAIGN
from .gate import GATE_V1

RUNS = (("CEX-017_sealed_v2", "sealed v2, the short lead agent, 19 people, run once"), ("CEX-013_sealed_v1", "sealed, 40 people, run once"),
        ("CEX-009_dev", "practice, 41 people"))
JUDGED = {"understanding", "option fit", "trap handled", "useful questions (share)", "questions already answered"}  # by the judge
# name, how to read it from a person's score, higher is better
MEASURES: tuple[tuple[str, Callable[[Mapping[str, Any]], Optional[float]], bool], ...] = (
    ("understanding", lambda r: r.get("understanding"), True),
    ("option fit", lambda r: r.get("options_fit"), True),
    ("right route reached", lambda r: _b(r.get("route_reached", r.get("route_right"))), True),
    ("trap handled", lambda r: _b(r.get("trap_pass")), True),
    ("useful questions (share)", lambda r: (r["useful_questions"] / r["asked"]) if r.get("asked") else None, True),
    ("questions already answered", lambda r: r.get("already_answered"), False),
    ("invented numbers and broken rules", lambda r: r.get("safety_failures"), False),
    ("seconds per reply", lambda r: r.get("seconds_per_turn"), False),
    ("cost per person (EUR)", lambda r: r.get("spent_eur"), False),
)


def _b(v: Any) -> Optional[float]:
    return None if v is None else float(bool(v))


def _interval(gains: list[float], confidence: float = 0.90) -> tuple[Optional[float], Optional[float]]:
    if not gains:
        return None, None
    rng = random.Random(GATE_V1["bootstrap"]["seed"])
    means = sorted(sum(rng.choice(gains) for _ in gains) / len(gains) for _ in range(GATE_V1["bootstrap"]["resamples"]))
    cut = int((1 - confidence) / 2 * len(means))
    return round(means[cut], 4), round(means[-cut - 1], 4)


def compare(lead: list[Mapping[str, Any]], other: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Each measure on the people both arms were scored on: wins, ties, losses (for the lead), means and the difference."""
    a = {r["id"]: r for r in lead if not r.get("unjudged")}
    b = {r["id"]: r for r in other if not r.get("unjudged")}
    out = []
    for name, read, higher in MEASURES:
        pairs = [(read(a[i]), read(b[i])) for i in sorted(set(a) & set(b))]
        pairs = [(float(x), float(y)) for x, y in pairs if x is not None and y is not None]
        if not pairs:
            continue
        better = sum(1 for x, y in pairs if (x > y if higher else x < y))
        worse = sum(1 for x, y in pairs if (x < y if higher else x > y))
        gains = [x - y for x, y in pairs]
        low, high = _interval(gains)
        mean_gain = sum(gains) / len(gains)
        sure = low is not None and ((low > 0) if higher else (high < 0)) or (low is not None and ((high < 0) if higher else (low > 0)))
        out.append({"measure": name, "higher_is_better": higher, "people": len(pairs), "lead_better": better, "same": len(pairs) - better - worse,
                    "lead_worse": worse, "lead_mean": round(sum(x for x, _ in pairs) / len(pairs), 4),
                    "claude_mean": round(sum(y for _, y in pairs) / len(pairs), 4), "difference": round(mean_gain, 4),
                    "interval_90": [low, high], "clear": bool(sure),
                    "winner": ("lead agent" if (mean_gain > 0) == higher else "Claude alone") if sure else "no clear difference",
                    "by": "judge" if name in JUDGED else "code"})
    return out


def traps(lead: list[Mapping[str, Any]], other: list[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for arm, rows in (("lead", lead), ("claude", other)):
        for r in rows:
            if r.get("trap") and r.get("trap_pass") is not None:
                t = out.setdefault(r["trap"], {"lead": [0, 0], "claude": [0, 0]})
                t[arm][0] += int(bool(r["trap_pass"]))
                t[arm][1] += 1
    return out


def build(results: Optional[Path] = None) -> dict[str, Any]:
    results = Path(results or CAMPAIGN / "results")
    doc: dict[str, Any] = {"runs": []}
    for stem, about in RUNS:
        path = results / f"{stem}.json"
        if not path.exists():
            continue
        scores = json.loads(path.read_text(encoding="utf-8")).get("scores") or {}
        lead, other = list(scores.get("case") or []), list(scores.get("plain") or [])
        if not lead or not other:
            continue
        doc["runs"].append({"run": stem.split("_")[0], "about": about, "measures": compare(lead, other), "traps": traps(lead, other)})
    second = results / "CEX-013_sealed_v1_second_labeller.json"
    owner = results / "CEX-013_sealed_v1_owner_labels.json"
    if owner.exists():
        o = json.loads(owner.read_text(encoding="utf-8"))
        s = json.loads(second.read_text(encoding="utf-8")) if second.exists() else {}
        doc["people_check"] = {"papers": (o.get("owner_labels") or {}).get("n"), "judge_owner_agreement": o.get("judge_owner_agreement"),
                               "understood_by_arm": s.get("understanding_by_arm_in_the_ten") or s.get("understood_by_arm_in_the_ten") or o.get("understood_by_arm_in_the_ten")}
    return doc


def render(doc: Mapping[str, Any]) -> str:
    out = ["# The lead agent against Claude alone, person by person", "",
           "Rebuilt by `python -m dclab_rnd.agent_eval.conversation_exam.headtohead`; do not edit by hand. Both arms met the same",
           "people; 'better/same/worse' counts people for the lead agent; the interval is the paired difference's 90% bootstrap",
           "range; 'clear' means the whole range is on one side of zero.", ""]
    for run in doc["runs"]:
        out += [f"## {run['run']} ({run['about']})", "",
                "| measure | by | lead better | same | lead worse | lead agent | Claude alone | difference | 90% range | winner |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for m in run["measures"]:
            lo, hi = m["interval_90"]
            out.append(f"| {m['measure']} | {m['by']} | {m['lead_better']} | {m['same']} | {m['lead_worse']} | {m['lead_mean']:.2f} | {m['claude_mean']:.2f} | "
                       f"{m['difference']:+.2f} | {lo:+.2f} to {hi:+.2f} | {m['winner']} |")
        out += ["", "| trap | lead agent | Claude alone |", "|---|---|---|"]
        for t, v in sorted(run["traps"].items()):
            out.append(f"| {t} | {v['lead'][0]} of {v['lead'][1]} | {v['claude'][0]} of {v['claude'][1]} |")
        out.append("")
    pc = doc.get("people_check")
    if pc:
        out += ["## Checked by people", "", f"The judge agreed with the owner on {pc['judge_owner_agreement']:.2f} of {pc['papers']} sealed papers "
                "(the bar was 0.70), so the measures the judge marks (understanding, option fit, traps, useful and repeated",
                "questions) are not yet trusted; the measures counted by code (route reached, invented numbers, seconds, cost) do",
                "not depend on the judge.", ""]
    return "\n".join(out)


def main(argv: Optional[list[str]] = None) -> int:
    doc = build()
    (CAMPAIGN / "HEAD_TO_HEAD.json").write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    (CAMPAIGN / "HEAD_TO_HEAD.md").write_text(render(doc), encoding="utf-8")
    print(render(doc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
