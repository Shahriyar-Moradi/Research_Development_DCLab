"""``python -m dclab_rnd.bench read|run``: DCLab Bench from the command line (see ``dclab_rnd.bench``)."""

from __future__ import annotations

import argparse
import json
import sys

from . import COMMANDS, run, summary

MARK = {"green": "OK  ", "amber": "??  ", "red": "RED ", "thin": "THIN", "empty": "--  "}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.bench")
    parser.add_argument("action", choices=("read", "run"))
    parser.add_argument("--layers", default="", help=f"comma-separated, of {', '.join(COMMANDS)} (run: all when empty)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "run":
        layers = [x.strip() for x in args.layers.split(",") if x.strip()] or None
        print(f"recorded {run(layers)}")
    s = summary()
    if args.json:
        print(json.dumps(s, indent=1, default=str))
        return 0
    for layer, state in s["layers"].items():
        print(f"{MARK[state['status']]} {layer:17} {state['status']:6} suites {state['suite_count']:3}  tracks {state.get('tracks', 0):4}  "
              f"cases {state['cases']:6}  failing {state.get('failed', 0)}")
        for reason in state["reasons"][:4]:
            print(f"       - {reason}")
    if s["unassigned"]:
        print("campaigns with no layer:", ", ".join(s["unassigned"]))
    print(f"{s['results']} results, {len(s['history'])} recorded runs, EUR {s['spent_eur']} spent by the results that say")
    return 0


if __name__ == "__main__":
    sys.exit(main())
