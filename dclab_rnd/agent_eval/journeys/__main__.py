"""``python -m dclab_rnd.agent_eval.journeys --split dev|sealed|sealed_v2|sealed_v3 [--workers N] [--only ID|KIND ...] [--no-store]``."""

from __future__ import annotations

import argparse
import json
import os

from . import GATES, SEALED, families, measure, sealed_runs, store, store_compare, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="dev", choices=("dev", "sealed", "sealed_v2", "sealed_v3", "sealed_v4"))
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 2) - 1)))  # each one an engine: memory
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--compare", action="store_true", help="22.9: every arm (the rules, each gated decision model, all) on the split")
    parser.add_argument("--seed", type=int, default=None, help="the replicas' seed (the sealed run uses the gate's)")
    parser.add_argument("--reader", action="store_true", help="20.10 v4: the rules and the conversation reader (a model) on the same people")
    parser.add_argument("--yes", action="store_true", help="send the reader's requests (with --reader)")
    parser.add_argument("--cap-eur", type=float, default=0.5, help="the reader's spend cap for the whole run")
    parser.add_argument("--spent", action="store_true", help="read a sealed set already run as dev material: nothing is stored")
    args = parser.parse_args()
    sealed = args.split in SEALED
    if args.spent:
        if not (sealed and sealed_runs(args.split)):
            print("--spent reads a sealed set that was run already, as dev material")
            return 2
        args.no_store, sealed = True, False  # its people are dev material now: played, never stored
    if sealed and sealed_runs(args.split):
        print("the sealed journeys were run once already: a sealed set is never run again")
        return 2
    if args.split == "sealed_v4" and not args.spent and not args.reader:
        print("the fourth sealed set judges the reader against the rules: run it with --reader, both arms at once")
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
    if args.reader:
        return _reader_arm(args, seed)
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


def _reader_arm(args, seed: int) -> int:
    """The rules and the reader played on the same people, one after the other in this process (the reader's client lives
    here); a sealed set stores both arms, once."""
    from . import load, play, verdict

    people = [j for j in load(args.split) if not args.only or j["id"] in args.only or j["kind"] in args.only]
    if not args.yes:
        print(f"Plan: {len(people)} people, the conversation reader asked about each problem sentence and each answer that settles the "
              f"card (about 3 to 5 requests each), capped at {args.cap_eur} EUR for the run. Add --yes to send them.")
        return 0
    from dotenv import load_dotenv

    from dclab_rnd.draft.reader import reader
    from dclab_rnd.models.gateway import for_workspace
    from dclab_rnd.settings import ROOT

    load_dotenv(ROOT / ".env", override=False)  # the provider and its key, as the server reads them

    read = reader(for_workspace(), cap_eur=args.cap_eur, force=True, prebuild=True)
    if read is None:
        print("No model serves conversation_reader: set the provider's key (and leave DCLAB_NO_LIVE_MODELS unset).")
        return 2
    by_arm = {"rules": measure(args.split, seed, args.workers, args.only)}
    by_arm["reader"] = []
    for j in people:
        try:
            by_arm["reader"].append(play(j, seed, None, {"reader": read}))
        except Exception as error:  # noqa: BLE001 - a journey that breaks is wrong, with its reason (as in measure)
            by_arm["reader"].append({"id": j["id"], "kind": j["kind"], "family": j["family"], "parts": {}, "right": False,
                                     "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]})
    spent = round(read.advisor.spent_eur(), 4)
    right = {arm: sum(r["right"] for r in rows) for arm, rows in by_arm.items()}
    passed = verdict(by_arm["reader"]).startswith("pass") and right["reader"] >= right["rules"] and read.failures == 0
    against = {"reader_right": right["reader"], "rules_right": right["rules"], "reader_requests": read.asked,
               "reader_failures": read.failures, "passed": passed, "rule": GATES.get(args.split, {}).get("reader")}
    for arm, rows in by_arm.items():
        print(f"{arm:7} {sum(r['right'] for r in rows)} of {len(rows)} right, cards {sum(r['parts']['card'] for r in rows)}")
    rules = {r["id"]: r for r in by_arm["rules"]}
    for r in by_arm["reader"]:
        if r["right"] != rules[r["id"]]["right"]:
            print(f"  {'GAINED' if r['right'] else 'LOST':6} {r['id']} {r['kind']}: {'; '.join(r['why'] or rules[r['id']]['why'])[:200]}")
    print(f"spent EUR {spent} (cap {args.cap_eur}); {read.asked} reader requests, {read.failures} failed; reader against rules: {against}")
    if read.failures:
        print("a reader request failed (the provider, or the cap): the rules answered for it, so this is not a reader arm; nothing is stored")
        return 1
    if not args.no_store:
        for arm, rows in by_arm.items():
            print(store(rows, args.split, verdict(rows), arm=arm, seed=seed, spent_eur=spent if arm == "reader" else 0.0,
                        extra={"reader_against_rules": against} if arm == "reader" else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
