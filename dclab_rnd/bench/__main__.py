"""``python -m dclab_rnd.bench read|run``: DCLab Bench from the command line (see ``dclab_rnd.bench``)."""

from __future__ import annotations

import argparse
import json
import sys

from . import COMMANDS, run, summary

MARK = {"green": "OK  ", "amber": "??  ", "red": "RED ", "thin": "THIN", "empty": "--  "}


def _monitors(args: argparse.Namespace) -> int:
    from pathlib import Path

    from ..settings import Settings
    from .monitors import summary as monitor_summary

    s = monitor_summary(Path(Settings.load().agent_home), days=args.days)
    if args.json:
        print(json.dumps(s, indent=1, default=str))
        return 0
    for a in s["agents"]:
        print(f"{a['mark']:5} {a['purpose']:24} requests {a['requests']:5}  failed {a['failed']:4}  p95 {a['p95_seconds']} s  EUR {a['eur']:.2f}"
              + (f"  - {'; '.join(a['reasons'])}" if a["reasons"] else ""))
    print("bug loop:", s["bug_loop"])
    return 0


def _compare(args: argparse.Namespace) -> int:
    from .compare import run as compare_run

    if not (args.result and args.a and args.b):
        print("compare needs --result, --a and --b")
        return 2
    target = compare_run(args.result, args.a, args.b, args.gate)
    doc = json.loads(target.read_text())
    print(f"{doc['b']} against {doc['a']} on {doc['result']} ({doc['split']}), recorded in {target.name}")
    for metric, m in doc["metrics"].items():
        lo, hi = m["interval"]
        print(f"  {metric:18} n {m['n']:3}  {doc['a']} {m['mean_a']}  {doc['b']} {m['mean_b']}  difference {m['difference']} "
              f"[{lo}, {hi}]  wins {m['wins_b']} losses {m['losses_b']} ties {m['ties']}" + (f"  McNemar p {m['mcnemar_p']}" if m["mcnemar_p"] is not None else ""))
    if "verdict" in doc:
        print("  gate:", doc["verdict"])
    return 0


def _versions(args: argparse.Namespace) -> int:
    from .versions import timeline

    rows = timeline(args.agent)
    if args.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    for a in rows:
        if not a["versions"]:
            continue
        flag = "measured live" if a["measured_now"] else ("NOT measured live" if a["live_suites"] else "no live suite")
        print(f"{a['agent']:22} {a['changes']:2} versions  now {a['current']}  {'committed' if a['committed'] else 'UNCOMMITTED'}  {flag}")
        if args.agent:
            for v in a["versions"]:
                print(f"   {v['date'][:16]}  {v['commit']}  {v['hash']}  checks {v['checks']:3}  live {len(v['live']):3} "
                      f"(pass {v['passed']}, fail {v['failed']}, EUR {v['spent_eur']})  {v['subject'][:70]}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.bench")
    parser.add_argument("action", choices=("read", "run", "versions", "compare", "monitors", "external"))
    parser.add_argument("what", nargs="?", default="check", help="external: check or plan")
    parser.add_argument("--days", type=int, default=7, help="monitors: the last N days")
    parser.add_argument("--result", default=None, help="compare: SUITE/RESULT, e.g. conversation_exam_v1/CEX-032_sealed_v3")
    parser.add_argument("--a", default=None, help="compare: the arm compared against (e.g. plain: Claude alone)")
    parser.add_argument("--b", default=None, help="compare: the arm judged (e.g. case: the lead agent)")
    parser.add_argument("--gate", default=None, help="compare: a gate file written before the run")
    parser.add_argument("--agent", default=None, help="versions: one prompt's timeline (case_agent, home_agent ...)")
    parser.add_argument("--layers", default="", help=f"comma-separated, of {', '.join(COMMANDS)} (run: all when empty)")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.action == "versions":
        return _versions(args)
    if args.action == "compare":
        return _compare(args)
    if args.action == "monitors":
        return _monitors(args)
    if args.action == "external":
        from . import external

        print(json.dumps(external.plan() if args.what == "plan" else external.check(), indent=1))
        return 0
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
