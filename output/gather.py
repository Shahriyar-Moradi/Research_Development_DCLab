"""Gather every problem DCLab solves today into one JSON: the invented self-play kinds (story, table, traps, checks,
results) and the 16 studied samples (table, known solution, the rules' solver run, the Claude twin run).
Names and counts only: never a cell of a real table."""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/Users/shahriar/Desktop/Desktop/Work/MyStartUp/R&D")
sys.path.insert(0, str(ROOT))
OUT = Path(sys.argv[1])


def short(v, n=400):
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[: n - 1] + "…"


def selfplay_stats():
    rows = [json.loads(x) for x in (ROOT / "lake/bronze/selfplay/done.jsonl").read_text().splitlines() if x.strip()]
    by = defaultdict(lambda: {"tried": 0, "verified": 0, "seconds": [], "missed": Counter()})
    for r in rows:
        k = r["group"].split(":")[1]
        b = by[k]
        b["tried"] += 1
        b["verified"] += bool(r.get("verified"))
        if r.get("seconds"):
            b["seconds"].append(r["seconds"])
        for m in r.get("missed") or ([r["reason"]] if r.get("reason") else []):
            b["missed"][m] += 1
    return {k: {"tried": v["tried"], "verified": v["verified"], "mean_seconds": round(sum(v["seconds"]) / len(v["seconds"]), 1) if v["seconds"] else None,
                "missed": v["missed"].most_common(6)} for k, v in by.items()}


def kinds():
    from dclab_rnd.lake import selfplay

    stats = selfplay_stats()
    out = []
    for kind in selfplay._kinds():
        base = getattr(kind, "base", None)
        d = (kind.describe() if hasattr(kind, "describe") else dict(base.describe(), title=kind.title, description="The same kind's image case: each picture becomes a vector through an encoder, then a tabular model reads the vectors."))
        item = {"key": kind.key, "title": d.get("title"), "description": d.get("description"), "base_task": d.get("base_task"),
                "metrics": [m["label"] for m in d.get("metrics") or []], "baselines": d.get("baselines"), "checks": d.get("checks"),
                "reliability": d.get("reliability"), "packs": d.get("packs"), "runs_in_engine": getattr(kind, "runs_in_engine", True),
                "engine_key": getattr(kind, "engine_key", kind.key)}
        try:
            c = kind.case(0)
        except Exception as e:  # noqa: BLE001
            c, item["case_error"] = None, f"{type(e).__name__}: {str(e)[:120]}"
        if c is not None:
            t = getattr(c, "table", None)
            if t is None:
                t = getattr(c, "frame", None)
            item["case"] = {"moment": getattr(c, "moment", None), "target": getattr(c, "target", None), "task": getattr(c, "task", None),
                            "rows": len(t) if t is not None else None,
                            "columns": [{"name": str(col), "dtype": str(t[col].dtype)} for col in t.columns] if t is not None else [],
                            "facts": {k: short(v, 300) for k, v in (getattr(c, "facts", None) or {}).items()
                                      if k not in ("behaviour", "responsive", "seed")},
                            "solution": {k: short(v, 300) for k, v in (getattr(c, "solution", None) or {}).items()},
                            "other_tables": {name: {"rows": len(tab), "columns": [str(x) for x in tab.columns]}
                                             for name, tab in (getattr(c, "tables", None) or {}).items() if hasattr(tab, "columns")}}
        group_keys = [g for g in stats if g == kind.key]
        item["selfplay"] = {g: stats[g] for g in group_keys}
        out.append(item)
    return out, stats


def samples():
    from dclab_rnd.studio import data as sd

    runs = ROOT / "agent_runs/twin/runs"
    solver, twin = {}, {}
    for p in sorted(runs.glob("*/run.json")):
        m = json.loads(p.read_text())
        seed = str(m.get("seed") or "")
        if not seed.startswith("sample:") or not m.get("blind"):
            continue
        key = seed.split(":", 1)[1]
        steps = {}
        sf = p.parent / "steps.jsonl"
        if sf.exists():
            for line in sf.read_text().splitlines():
                s = json.loads(line)
                steps[s["step"]] = s
        col = steps.get("solution.columns") or {}
        rec = {"run": m["run_id"], "finished": bool(m.get("finished")),
               "rules_leaks": ((col.get("system") or {}).get("value") or {}).get("leakage"),
               "claude_leaks": ((col.get("ai_first") or {}).get("value") or {}).get("leakage"),
               "verdict": col.get("verdict"), "eur": round(float((m.get("model_usage") or {}).get("eur") or 0), 2)}
        summ = {}
        for lane in ("workflow", "ai_first"):
            f = p.parent / lane / "summary.json"
            if f.exists():
                s = json.loads(f.read_text())
                fin = (s.get("stages") or {}).get("final") or {}
                ci = fin.get("holdout_ci") or {}
                summ[lane] = {"status": s.get("status"), "metric": fin.get("primary_metric"), "score": fin.get("holdout"),
                              "low": ci.get("low"), "high": ci.get("high"), "model": fin.get("chosen")}
        rec["lanes"] = summ
        (solver if m.get("system_only") else twin).setdefault(key, []).append(rec)
    out = []
    for s in sd.sample_catalog():
        key = s["key"]
        try:
            frame, known = sd.load_sample(key)
            shape = {"rows": int(len(frame)), "columns": int(frame.shape[1]), "names": [str(c) for c in frame.columns][:60]}
        except Exception as e:  # noqa: BLE001
            known, shape = {}, {"error": type(e).__name__}
        out.append({"key": key, "name": s.get("name"), "industry": s.get("industry"), "table": shape,
                    "target": known.get("target"), "task": known.get("task"), "metric": known.get("metric"),
                    "known_leaks": [f.get("column") for f in known.get("forbidden") or []],
                    "known_leak_reasons": {f.get("column"): short(f.get("reason") or "", 200) for f in known.get("forbidden") or []},
                    "identifiers": known.get("identifiers"), "time_column": known.get("time_column"), "group_column": known.get("group_column"),
                    "prediction_moment": short(known.get("prediction_moment") or "", 300),
                    "solver_runs": solver.get(key, []), "twin_runs": twin.get(key, [])})
    return out


if __name__ == "__main__":
    k, stats = kinds()
    data = {"kinds": k, "selfplay_totals": stats, "samples": samples()}
    OUT.write_text(json.dumps(data, indent=1, default=str, ensure_ascii=False))
    print(len(data["kinds"]), "kinds,", len(data["samples"]), "samples ->", OUT)
