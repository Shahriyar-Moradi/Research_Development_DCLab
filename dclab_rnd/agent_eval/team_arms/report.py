"""The comparison report of the team arms (B4, 2026-11-24): per role and arm, per family and per model, with bootstrap
intervals by case, the winner of each role by the pinned rule, the hard zeros by code, and what the runs cost.

Read from the plays' ``score.json`` (and ``judged.json`` when a judge ran) only. The JSON goes to
``evidence/campaigns/team_arms_v1/results/TAR-NNN_<set>.json`` with a short Markdown summary beside it; a quick (smoke)
set's report stays in its run folder: it is never evidence.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

from . import gate

RESULTS = gate.CAMPAIGN / "results"


def load_rows(runs: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Every scored play of the runs (``score.json`` with its judge's verdict), and each run's ``run.json``."""
    rows, manifests = [], []
    for run in runs:
        run = Path(run)
        manifest = json.loads((run / "run.json").read_text(encoding="utf-8")) if (run / "run.json").exists() else {"run": run.name}
        manifests.append(manifest)
        for path in sorted(run.glob("plays/*/*/score.json")):
            if ".cut-" in path.parent.name:  # a play cut by the cap or a crash, played again later: never a result
                continue
            row = json.loads(path.read_text(encoding="utf-8"))
            judged = path.parent / "judged.json"
            if judged.exists():
                row["judge"] = json.loads(judged.read_text(encoding="utf-8"))
            row["run"] = manifest.get("run") or run.name
            rows.append(row)
    return rows, manifests


def _mean(values: Sequence[float]) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


def group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Main score with its interval, hard zeros by code, the counts summed or averaged, and the judge's means."""
    done = [r for r in rows if r.get("complete")]
    mains = [float(r["main"]) for r in done]
    counts = [r.get("counts") or {} for r in done]
    fallbacks: Counter = Counter()
    for c in counts:
        fallbacks.update(c.get("fallbacks") or {})
    judged = [r["judge"] for r in done if (r.get("judge") or {}).get("valid")]
    return {"plays": len(rows), "complete": len(done), "main": _mean(mains), "main_interval": gate.interval(mains),
            "hard_zero_cases": sum(1 for r in done if r.get("hard_zeros")),
            "hard_zeros": dict(Counter(code for r in done for code in r.get("hard_zero_codes") or [])),
            "needless_questions": sum(int(c.get("needless_questions") or 0) for c in counts),
            "teams_missed": sum(int(c.get("teams_missed") or 0) for c in counts),
            "duplicate_replies": sum(int(c.get("duplicate_replies") or 0) for c in counts),
            "fallbacks": dict(fallbacks), "agent_failures": sum(int(c.get("agent_failures") or 0) for c in counts),
            "acted_without_writes": sum(int(c.get("acted_without_writes") or 0) for c in counts),
            "mean_seconds": _mean([float(c.get("seconds") or 0.0) for c in counts]),
            "eur": round(sum(float(c.get("euros") or 0.0) for c in counts), 6),
            "judge": {k: _mean([float(j[k]) for j in judged if isinstance(j.get(k), (int, float))]) for k in ("understanding", "plan", "report")}
            if judged else None}


def _models(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return sorted({str(r["model"]) for r in rows if r.get("model")})


def winners(rows: Sequence[Mapping[str, Any]], cases_of_role: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """Per role and model: each arm's summary and the winner, by the pinned rule. The rules arm's plays count for every model."""
    out = []
    for role, case_ids in sorted(cases_of_role.items()):
        mine = [r for r in rows if role in (r.get("roles") or []) and r["case"] in case_ids]
        for model in _models(mine) or [None]:
            arms = {}
            for arm in gate.SIMPLER:
                these = [r for r in mine if r.get("arm") == arm and (arm == "R" or r.get("model") == model)]
                if these or arm == "R":
                    arms[arm] = gate.arm_summary(these, list(case_ids))
            verdict = gate.winner(arms)
            out.append({"role": role, "model": model, "arms": {a: {k: v for k, v in s.items() if k != "by_case"} for a, s in arms.items()},
                        **verdict})
    return out


def build(rows: list[dict[str, Any]], manifests: list[dict[str, Any]], set_name: str) -> dict[str, Any]:
    by_role_arm_model, by_family, by_model = [], [], []
    roles = sorted({role for r in rows for role in r.get("roles") or []})
    for role in roles:
        for arm in gate.SIMPLER:
            for model in ([None] if arm == "R" else _models(rows)):
                these = [r for r in rows if role in r.get("roles", []) and r.get("arm") == arm and (arm == "R" or r.get("model") == model)]
                if these:
                    by_role_arm_model.append({"role": role, "arm": arm, "model": model, **group(these)})
    for family in sorted({r["family"] for r in rows}):
        for arm in gate.SIMPLER:
            for model in ([None] if arm == "R" else _models(rows)):
                these = [r for r in rows if r["family"] == family and r.get("arm") == arm and (arm == "R" or r.get("model") == model)]
                if these:
                    by_family.append({"family": family, "arm": arm, "model": model, **group(these)})
    for model in _models(rows):
        for arm in ("H", "A"):
            these = [r for r in rows if r.get("arm") == arm and r.get("model") == model]
            if these:
                by_model.append({"model": model, "arm": arm, **group(these)})
    cases_of_role: dict[str, list[str]] = defaultdict(list)
    for m in manifests:
        listed = m.get("cases")
        for c in listed if isinstance(listed, list) else []:  # a sealed run lists no case (counts only): its rows say them
            for role in c.get("roles") or []:
                if c["id"] not in cases_of_role[role]:
                    cases_of_role[role].append(c["id"])
    if not cases_of_role:  # runs without a manifest: the cases the rows hold
        for r in rows:
            for role in r.get("roles") or []:
                if r["case"] not in cases_of_role[role]:
                    cases_of_role[role].append(r["case"])
    chosen = winners(rows, cases_of_role)
    prints = sorted({str(m.get("fingerprint")) for m in manifests if m.get("fingerprint")})
    spent = round(sum(float(m.get("spent_eur") or 0.0) for m in manifests), 6)
    thinking = [m.get("thinking") or {} for m in manifests]
    stopped = [m.get("stopped") for m in manifests if m.get("stopped")]
    hard = Counter(code for r in rows if r.get("complete") for code in r.get("hard_zero_codes") or [])
    named = ", ".join(f"{w['role']}/{w['model'] or 'rules'}: {w['winner'] or 'none'}" for w in chosen)
    return {"gate": gate.GATE_ID, "fingerprint": prints[0] if len(prints) == 1 else prints, "pinned_now": gate.pinned() is None,
            "set": set_name, "runs": [m.get("run") for m in manifests], "models": _models(rows), "cases": len({r["case"] for r in rows}),
            "plays": len(rows), "complete": sum(1 for r in rows if r.get("complete")), "stopped": stopped, "spent_eur": spent,
            "thinking": {"ok": sum(int(t.get("ok") or 0) for t in thinking), "with_thinking": sum(int(t.get("with_thinking") or 0) for t in thinking),
                         "verified": all(t.get("verified", True) for t in thinking)},
            "hard_zeros": dict(hard), "winners": chosen, "by_role_arm_model": by_role_arm_model, "by_family": by_family, "by_model": by_model,
            "verdict": f"winners by {gate.GATE_ID}: {named}" if chosen else "no role compared", "evaluation_only": True,
            "limits": ["Code scores from records; the judge is advisory. A difference inside its interval is noise.",
                       "Never evidence that a team agent is production-ready."]}


def markdown(doc: Mapping[str, Any]) -> str:
    lines = [f"# Team arms: {doc.get('id', '')} ({doc['set']} set)", "",
             f"Gate {doc['gate']} (pinned now: {'yes' if doc['pinned_now'] else 'no'}). {doc['cases']} cases, {doc['plays']} plays "
             f"({doc['complete']} complete), models {', '.join(doc['models']) or 'none'}, spent EUR {doc['spent_eur']}.", "",
             "## Winners (code score only)", "", "| Role | Model | Winner | R | H | A | Within noise |", "|---|---|---|---|---|---|---|"]
    for w in doc["winners"]:
        cell = {a: (f"{s['main']} {s['main_interval']}" + ("" if s["eligible"] else " (not eligible)")) for a, s in w["arms"].items()}
        lines.append(f"| {w['role']} | {w['model'] or '-'} | {w['winner'] or 'none'} | {cell.get('R', '-')} | {cell.get('H', '-')} | "
                     f"{cell.get('A', '-')} | {w.get('within_noise', '-')} |")
    lines += ["", "## Hard zeros", "", ", ".join(f"{k}: {v}" for k, v in doc["hard_zeros"].items()) or "none", "",
              "## By family", "", "| Family | Arm | Model | Main | Interval | Hard-zero cases |", "|---|---|---|---|---|---|"]
    for f in doc["by_family"]:
        lines.append(f"| {f['family']} | {f['arm']} | {f['model'] or '-'} | {f['main']} | {f['main_interval']} | {f['hard_zero_cases']} |")
    lines += ["", "Limits: " + " ".join(doc["limits"]), ""]
    return "\n".join(lines)


def next_id(results: Path = RESULTS) -> str:
    used = [int(m.group(1)) for p in Path(results).glob("TAR-*.json") if (m := re.match(r"TAR-(\d+)", p.name))]
    return f"TAR-{1 + max(used or [0]):03d}"


def write(doc: dict[str, Any], results: Path = RESULTS) -> Path:
    """Into the campaign: ``TAR-NNN_<set>.json`` and its Markdown summary. A sealed set's result is written once."""
    results = Path(results)
    from dclab_rnd import bench

    if results.parent.name not in bench.LAYER_OF:  # the review of B4, F11: a result of a campaign without a layer breaks the gate
        raise ValueError(f"{results.parent.name} has no line in dclab_rnd.bench.LAYER_OF yet: add it (layer \"team\") before the first result")
    if doc["set"] == "quick":
        raise ValueError("the quick set is a smoke test: its report stays in its run folder, never in the campaign")
    if doc["set"] == "sealed" and list(results.glob("TAR-*_sealed.json")):
        raise ValueError("the sealed comparison is reported once")
    results.mkdir(parents=True, exist_ok=True)
    doc = {"id": next_id(results), "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **doc}
    path = results / f"{doc['id']}_{doc['set']}.json"
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    (results.parent / f"{doc['id']}_{doc['set']}.md").write_text(markdown(doc), encoding="utf-8")  # beside results/, as RESULT_*.md are
    return path


__all__ = ["RESULTS", "build", "group", "load_rows", "markdown", "next_id", "winners", "write"]
