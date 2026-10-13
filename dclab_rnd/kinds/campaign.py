"""The kinds' first campaign (package 16.14): each problem kind's acceptance on several seeds of its replica, stored as
evidence agents can cite.

``python -m dclab_rnd.kinds.campaign`` runs every registered kind's acceptance (``kinds.acceptance.run``) on seeds 0, 1
and 2 and writes one result per kind, ``evidence/campaigns/kinds_v1/results/KND-<n>_<kind>.json``: the decisions a
careful data scientist reaches on that kind of problem, which held on which seed, and the share that held with a 95%
Wilson interval. The evidence index reads them as records of type ``kind`` (``KND-*``), tied to the kind's research
track, so a method question about a kind ("how should a forecasting backtest be split?") is answered from that kind's
own measured decisions. Replicas are invented data with planted traps; no kind has a sealed set, and nothing here is a
benchmark that may be tuned on.
"""

from __future__ import annotations

import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN = "kinds_v1"
RESULTS = ROOT / "evidence/campaigns" / CAMPAIGN / "results"
SEEDS = (0, 1, 2)


def wilson(hits: int, n: int, z: float = 1.959964) -> list[float] | None:
    if not n:
        return None
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def result_for(kind: Any, number: int, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """One kind's result: its acceptance on each seed, each decision with the seeds it held and failed on, the share
    that held (decisions the kind names as open, waiting on the owner, are kept apart and never counted), and claims."""
    from dclab_rnd.provenance import capture_provenance

    from . import acceptance

    started = time.perf_counter()
    begun = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    runs = [{"seed": s, **{k: v for k, v in acceptance.run(kind, seed=s).items() if k in ("status", "checks", "missing")}} for s in seeds]
    open_ones = set(getattr(kind, "open_decisions", ()) or ())
    decisions: dict[str, dict[int, bool]] = {}
    for r in runs:
        for c in r["checks"]:
            decisions.setdefault(c["decision"], {})[r["seed"]] = bool(c["ok"])
    counted = {d: v for d, v in decisions.items() if d not in open_ones}
    held = sum(sum(v.values()) for v in counted.values())
    total = sum(len(v) for v in counted.values())
    clean_seeds = sum(1 for s in seeds if all(v.get(s, True) for v in counted.values()))
    eid = f"KND-{number:03d}"
    always = [d for d, v in counted.items() if all(v.values())]
    failing = {d: [s for s, ok in v.items() if not ok] for d, v in counted.items() if not all(v.values())}
    claims = [{"claim_id": f"{eid}-C1", "kind": "fact",
               "statement": f"{kind.title}: on {clean_seeds} of {len(seeds)} seeds every decision held (95% interval over seeds "
                            f"{wilson(clean_seeds, len(seeds))}); {held} of {total} decision checks held"
                            + (f"; failed: " + "; ".join(f"{d} (seed {', '.join(map(str, ss))})" for d, ss in failing.items()) if failing else "")
                            + (f"; open, not measured: {'; '.join(sorted(open_ones & set(decisions)))}" if open_ones & set(decisions) else "") + ".",
               "evidence": ["evidence.decisions", "evidence.rate"],
               "limitations": [x for x in (
                   "Invented replicas with planted traps, three seeds: what the kind decides, not how well it does on real data.",
                   "Checks on one seed share its data, so they are not independent: the interval over seeds is the honest one.",
                   "" if kind.runs_in_engine else "The engine does not run this kind yet: it runs as a library (dclab_core).") if x]}]
    for n, decision in enumerate(always, 2):
        claims.append({"claim_id": f"{eid}-C{n}", "kind": "recommendation",
                       "statement": f"On a {kind.title.lower()} problem: {decision}.", "evidence": [f"evidence.decisions.{n - 2}"],
                       "limitations": ["Held on every seed of an invented replica."]})
    for decision, ss in failing.items():
        claims.append({"claim_id": f"{eid}-C{len(claims) + 1}", "kind": "risk",
                       "statement": f"On a {kind.title.lower()} problem this decision failed on seed {', '.join(map(str, ss))} of {len(seeds)}: {decision}.",
                       "evidence": ["evidence.decisions"], "limitations": ["Seen on an invented replica; the seed's details are in evidence.seeds."]})
    return {"schema_version": 1, "campaign_id": CAMPAIGN, "experiment_id": eid, "kind": "kind_acceptance", "problem_kind": kind.key,
            "title": kind.title, "description": kind.description, "metrics": [label for _, label in kind.metrics],
            "baselines": list(kind.baselines), "reliability": list(kind.reliability), "track": kind.evidence_track,
            "runs_in_engine": bool(kind.runs_in_engine),
            "question": f"Which decisions does DCLab reach on a {kind.title.lower()} problem ({kind.description.rstrip('.')}), and do they hold?",
            "hypothesis": "Each decision a careful data scientist reaches on this kind of problem holds on every seed of the kind's replica.",
            "setup_summary": f"The {kind.title.lower()} kind's replica on seeds {', '.join(map(str, seeds))}, brought as a person would; "
                             f"{'the five stages of the engine' if kind.runs_in_engine else 'its own machine in dclab_core'} solve it, and each decision "
                             "is checked on the finished run. No model is called.",
            "evidence": {"decisions": [{"decision": d, "held": sum(v.values()), "of": len(v), "failed_on_seeds": [s for s, ok in v.items() if not ok],
                                        "open": d in open_ones} for d, v in decisions.items()],
                         "rate": {"held": held, "of": total, "share": round(held / total, 3) if total else None, "ci95_checks": wilson(held, total),
                                  "seeds_all_held": clean_seeds, "seeds": len(seeds), "ci95": wilson(clean_seeds, len(seeds))},
                         "seeds": runs},
            "claims": claims, "status": "completed", "started_at": begun,
            "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "elapsed_seconds": round(time.perf_counter() - started, 1),
            "provenance": {**capture_provenance(ROOT), "code_committed": code_clean(), "command": "python -m dclab_rnd.kinds.campaign",
                           "seeds": list(seeds)}}


CODE = ("dclab_core", "dclab_rnd", "general_pipeline")  # what a kind's acceptance runs


def code_clean() -> bool | None:
    """Whether the code the campaign runs is all committed (new files included); the owner's other work in the tree does
    not count. None when git cannot say."""
    import subprocess

    try:
        out = subprocess.run(["git", "status", "--porcelain", "--", *CODE], cwd=ROOT, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return None if out.returncode else not out.stdout.strip()


def _numbers(every: list[Any]) -> dict[str, int]:
    """A new id for each kind's new result: results are never overwritten (rule 3); a rerun is a new experiment."""
    taken = [int(p.name[4:7]) for p in RESULTS.glob("KND-*.json") if p.name[4:7].isdigit()]
    nxt = max(taken, default=0) + 1
    out = {}
    for kind in sorted(every, key=lambda k: k.key):
        out[kind.key] = nxt
        nxt += 1
    return out


def latest(key: str) -> Path | None:
    """The newest result of a kind (the highest id)."""
    found = sorted(RESULTS.glob(f"KND-*_{key}.json"))
    return found[-1] if found else None


def main(argv: list[str] | None = None) -> int:
    import argparse

    from dclab_rnd import kinds

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.kinds.campaign")
    parser.add_argument("kinds", nargs="*", help="only these kinds (default: every registered kind)")
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not code_clean():
        raise SystemExit("commit the code first: a result must name the code that made it (dclab_core, dclab_rnd, general_pipeline)")
    every = [k for k in kinds.all_kinds() if not args.kinds or k.key in args.kinds]
    numbers = _numbers(every)
    for kind in every:
        doc = result_for(kind, numbers[kind.key])
        path = RESULTS / f"{doc['experiment_id']}_{kind.key}.json"
        if path.exists():
            raise SystemExit(f"{path.name} exists: results are never overwritten")
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp.replace(path)
        rate = doc["evidence"]["rate"]
        print(f"{doc['experiment_id']} {kind.key:<14} {rate['seeds_all_held']}/{rate['seeds']} seeds clean, {rate['held']}/{rate['of']} checks "
              f"({doc['elapsed_seconds']} s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
