"""Results per problem kind, with intervals (package 16.14): ``python -m dclab_rnd.agent_eval.by_kind``.

One row per kind from what is stored: its acceptance on its replica (``kinds_v1``: the share of seeds on which every
decision held, with its 95% interval), the benchmark's scripted policies on its cases (``benchmark_v4``: leaks caught
and false alarms with their 95% Wilson intervals, the latest dev and sealed runs apart), and its pitfalls measured on
its replica (``pitfalls_by_kind_v1``: the mean cost over seeds with its interval). A kind the benchmark's policies cannot
judge says so, never an empty interval read as a pass.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGNS = ROOT / "evidence" / "campaigns"


def _latest(pattern: str, keep: Any = None) -> list[dict[str, Any]]:
    out = []
    for path in sorted((CAMPAIGNS).glob(pattern), key=lambda p: int(p.name.split("_")[0].split("-")[1])):
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc.get("status", "completed") == "completed" and (keep is None or keep(doc)):
            out.append({**doc, "_path": path.relative_to(ROOT).as_posix()})
    return out


def benchmark(split: str) -> Optional[dict[str, Any]]:
    runs = _latest("benchmark_v4/results/BEN4-*.json", lambda d: d.get("split") == split and d.get("by_kind"))
    return runs[-1] if runs else None


def row(kind: str) -> dict[str, Any]:
    from dclab_rnd.agent_eval import benchmark_v4

    acceptance = [d for d in _latest("kinds_v1/results/KND-*.json") if d.get("problem_kind") == kind]
    pitfalls = [d for d in _latest("pitfalls_by_kind_v1/results/PIK-*.json") if d.get("problem_kind") == kind]
    out: dict[str, Any] = {"kind": kind}
    if acceptance:
        rate = (acceptance[-1].get("evidence") or {}).get("rate") or {}
        out["acceptance"] = {"record": acceptance[-1]["experiment_id"], "seeds_all_held": rate.get("seeds_all_held"), "seeds": rate.get("seeds"),
                             "ci95": rate.get("ci95"), "path": acceptance[-1]["_path"]}
    if pitfalls:
        ev = pitfalls[-1]["evidence"]
        out["pitfalls"] = {"record": pitfalls[-1]["experiment_id"],
                           **{k: {"mean": v["mean"], "ci95": v["ci95"], "measured": bool(v.get("measured"))} for k, v in ev.items() if k != "runs"}}
    judged = kind in {c.tags[-1] for c in benchmark_v4.CASES}
    out["benchmark"] = {}
    for split in ("dev", "test"):
        run = benchmark(split)
        if run is not None and judged:
            out["benchmark"][split] = {"record": run.get("experiment_id"), **(run["by_kind"].get(kind) or {})}
    if not judged:
        out["benchmark"] = "not judged by the benchmark's scripted policies (one yes or no or numeric outcome column): see its acceptance"
    return out


def report() -> list[dict[str, Any]]:
    from dclab_rnd import kinds

    return [row(k.key) for k in kinds.all_kinds()]


def main() -> int:
    for r in report():
        acc = r.get("acceptance") or {}
        line = f"{r['kind']:14s} acceptance {acc.get('seeds_all_held')}/{acc.get('seeds')} {acc.get('ci95')}"
        if isinstance(r["benchmark"], dict):
            for split, cells in r["benchmark"].items():
                std = cells.get("standard") or {}
                line += f" | {split} standard leaks {std.get('leaks_caught')} {std.get('leaks_caught_ci95')}"
        else:
            line += " | benchmark: not judged"
        for name, p in (r.get("pitfalls") or {}).items():
            if name != "record":
                line += f" | {name} {p['mean']} {p['ci95']}"
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
