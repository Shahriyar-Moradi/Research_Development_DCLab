"""``python -m dclab_rnd.agent_eval.journeys --split dev|sealed [--workers N] [--only ID|KIND ...] [--no-store]``."""

from __future__ import annotations

import argparse
import json
import os

from . import families, measure, sealed_runs, store, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev", "sealed"))
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))  # each one an engine: memory
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--seed", type=int, default=None, help="the replicas' seed (the sealed run uses the gate's)")
    args = parser.parse_args()
    if args.split == "sealed" and sealed_runs():
        print("the sealed journeys were run once already: a sealed set is never run again")
        return 2
    if args.split == "sealed" and (args.only or args.no_store):
        print("the sealed journeys are run whole and stored, once")
        return 2
    from . import GATE

    if args.split == "sealed" and args.seed not in (None, GATE["seed"]):
        print(f"the sealed journeys run on the gate's seed, {GATE['seed']}")
        return 2
    seed = GATE["seed"] if args.split == "sealed" else (0 if args.seed is None else args.seed)
    rows = measure(args.split, seed, args.workers, args.only)
    for r in rows:
        mark = "ok  " if r["right"] else "MISS"
        print(f"{mark} {r['id']:6} {r['kind']:14} {json.dumps(r['parts'])} {r.get('seconds', '')}s asked {r.get('asked')} card {r.get('got')}  {'; '.join(r['why'])[:300]}")
    for f in families(rows):
        print(f"  {f['family']:16} {f['right']}/{f['journeys']}  interval {f['interval']}")
    result = verdict(rows)
    print(result)
    if not args.no_store:
        print(store(rows, args.split, result, seed=seed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
