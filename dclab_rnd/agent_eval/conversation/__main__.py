"""python -m dclab_rnd.agent_eval.conversation --split dev|sealed [--arm rules|reader] [--yes --cap-eur N]

Plays the written conversations and scores them against conversation_gate_v1. The rules arm is free and runs at once;
the reader arm sends each answer the rules could not read to the conversation reader role (a model), so it prints its
plan and runs only with --yes and a cap. A sealed set is run once per arm: a second sealed run is refused.
"""

import argparse
import json
import sys


def main(argv=None) -> int:
    from . import GATES, measure, sealed_runs, store, verdict

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.conversation", description=__doc__.split("\n")[0])
    parser.add_argument("--split", choices=("dev", "sealed", "sealed_v2"), default="dev")
    parser.add_argument("--arm", choices=("rules", "reader"), default="rules")
    parser.add_argument("--yes", action="store_true", help="send the reader's requests (the reader arm only)")
    parser.add_argument("--cap-eur", type=float, default=0.5)
    parser.add_argument("--no-store", action="store_true", help="print the scores, store no result (dev only)")
    args = parser.parse_args(argv)
    if args.split.startswith("sealed") and sealed_runs("*", args.split):
        print(f"The {args.split} set was run already ({sealed_runs('*', args.split)[0].name}): a sealed set is spent for every arm.",
              file=sys.stderr)
        return 2
    reader = None
    spent = 0.0
    if args.arm == "reader":
        from dclab_core.agents import run_role
        from dclab_core.conversation import READER

        if not args.yes:
            print(f"Plan: the reader arm asks a model about each answer of the {args.split} conversations (at most 5 each), "
                  f"capped at {args.cap_eur} EUR. Add --yes to send them.")
            return 0
        from dclab_rnd.core_advisor import GatewayAdvisor

        advisor = GatewayAdvisor(cap_eur=args.cap_eur)
        reader = lambda field, answer, columns: run_role(READER, advisor, {"field": field, "answer": answer, "columns": columns}).output  # noqa: E731
    rows = measure(args.split, reader)
    if args.arm == "reader":
        spent = round(advisor.spent_eur(), 4)  # what the run cost, stored with it
    baseline = measure(args.split) if args.arm == "reader" else None
    result = verdict(rows, args.split, baseline)
    for r in rows:
        print(f"{r['id']}  {r['family']:<15} {'right' if r['right'] else 'WRONG':<6} {r['questions']} questions  {r['why'] or ''}")
    print(f"{sum(r['right'] for r in rows)} of {len(rows)} right; gate {GATES[args.split]['version']}: {result}")
    if not (args.no_store and args.split == "dev"):
        print(f"stored {store(rows, args.split, args.arm, result, spent).name}")
    return 0 if result.startswith("pass") else 1


if __name__ == "__main__":
    sys.exit(main())
