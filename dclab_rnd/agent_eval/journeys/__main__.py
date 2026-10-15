"""``python -m dclab_rnd.agent_eval.journeys --split dev|sealed|sealed_v2 [--workers N] [--only ID|KIND ...] [--no-store]``."""

from __future__ import annotations

import argparse
import json
import os

from . import GATES, SEALED, families, measure, sealed_runs, store, store_compare, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev", "sealed", "sealed_v2"))
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))  # each one an engine: memory
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--compare", action="store_true", help="22.9: every arm (the rules, each gated decision model, all) on the split")
    parser.add_argument("--seed", type=int, default=None, help="the replicas' seed (the sealed run uses the gate's)")
    args = parser.parse_args()
    sealed = args.split in SEALED
    if sealed and sealed_runs(args.split):
        print("the sealed journeys were run once already: a sealed set is never run again")
        return 2
    if sealed and (args.only or args.no_store):
        print("the sealed journeys are run whole and stored, once")
        return 2
    gate = GATES.get(args.split)
    if sealed and args.seed not in (None, gate["seed"]):
        print(f"the sealed journeys run on the gate's seed, {gate['seed']}")
        return 2
    seed = gate["seed"] if sealed else (0 if args.seed is None else args.seed)
    if args.compare:
        if sealed:
            print("the arms are compared on dev: a sealed set is run once, for its gate")
            return 2
        from . import arms, compare

        by_arm = {arm: measure(args.split, seed, args.workers, args.only, doc) for arm, doc in arms().items()}
        result = compare(by_arm)
        for arm, a in result["arms"].items():
            print(f"{'ok  ' if a['ok'] else 'WORSE'} {arm:30} {a['right']}/{a['journeys']} right  {json.dumps(a['parts'])}  changed by the model {a['decisions_by_model']} (agreed {a['agreed_with_the_rule']}), "
                  f"held for review {a['held_for_review']}, refused by the validator {a['refused_by_validator']}, moves refused {a['refused_moves']}, "
                  f"changed {a['changed']}")
        print(result["verdict"])
        if not args.no_store:
            print(store_compare(by_arm, result, args.split, seed))
        return 0
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
