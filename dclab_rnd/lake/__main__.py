"""``python -m dclab_rnd.lake build [--home DIR ...] [--lake DIR]``: the training lake (26.1); ``summary`` prints the counts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import LAKE, MANIFEST, build, workspaces


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.lake", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="bronze, silver and gold from every source; the committed manifest")
    b.add_argument("--home", type=Path, action="append", default=[], help="another workspace to read (repeatable)")
    b.add_argument("--lake", type=Path, default=LAKE)
    sub.add_parser("summary", help="the committed manifest's counts")
    am = sub.add_parser("automl", help="26.6a: a family picked from meta-features, judged by regret on held-out problem kinds")
    am.add_argument("--store", action="store_true", help="store the result in evidence/campaigns/automl_v1")
    bm = sub.add_parser("benchmark", help="27.6: an answerer on the verified invented problems, scored against their planted truth")
    bm.add_argument("--answerer", choices=("rules", "configured", "tuned"), required=True)
    bm.add_argument("--split", choices=("dev", "sealed"), default="dev")
    bm.add_argument("--limit", type=int, default=120)
    bm.add_argument("--cap-eur", type=float, default=None)
    bm.add_argument("--yes", action="store_true", help="send the questions to a model (without it, only the plan is printed)")
    e = sub.add_parser("export", help="26.4/27.7: training files from gold (never sealed nor evaluation-only rows)")
    e.add_argument("--target", choices=("decisions", "solver"), required=True)
    e.add_argument("--lake", type=Path, default=LAKE)
    e.add_argument("--out", type=Path, default=None)
    e.add_argument("--weights", default="", help="set=weight,...: repeat or thin a set in train (e.g. leakage_judgment=2)")
    e.add_argument("--no-think", action="store_true", help="27.9: write the solver's answers without their verified <think> block")
    o = sub.add_parser("openml", help="26.2: fetch and profile OpenML's public tables (resumable; the owner's yes, 2026-10-16)")
    o.add_argument("--limit", type=int, default=1200)
    a = sub.add_parser("answers", help="26.9: another answerer (GLiNER2.5-Decide) on every DCLab test, every detail kept (run in the ai12 env)")
    a.add_argument("--answerer", default="gliner2.5-decide")
    a.add_argument("--sources", default="lake,workflow,journeys,selfplay")
    a.add_argument("--points", default="")
    a.add_argument("--splits", default="train,dev,sealed")
    a.add_argument("--limit", type=int, default=None, help="at most this many new decisions per point and split")
    a.add_argument("--batch", type=int, default=32)
    sub.add_parser("answers-summary", help="26.9: what was captured, per answerer, source and point (dev agreement only); the committed manifest")
    sub.add_parser("harvest", help="26.9: every decision DCLab's rules, Claude, classifiers and people made, gathered whole from every workspace")
    sub.add_parser("rules", help="26.9: DCLab's rules as an answerer of every test, in the answers' layout")
    sub.add_parser("answers-export", help="26.9: every test's labelled decisions in GLiNER2's training format (training_records/datasets/gliner2/)")
    p = sub.add_parser("selfplay", help="27.3: solve the engine kinds' replicas by seed, keep only verified solves")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--kinds", default="", help="comma-separated kind keys (default: every engine kind)")
    args = parser.parse_args(argv)
    if args.command == "answers":
        from .answers import capture, summary

        got = capture(args.answerer, tuple(x for x in args.sources.split(",") if x), [x for x in args.points.split(",") if x] or None,
                      tuple(x for x in args.splits.split(",") if x), args.limit, args.batch)
        print(json.dumps({k: v for k, v in got["asked"].items() if v}, indent=1))
        summary()
        return 0
    if args.command == "harvest":
        from .harvest import harvest

        m = harvest()
        print(json.dumps({k: m[k] for k in ("ledgers", "decisions", "labels", "outcomes", "twin_runs", "model_requests", "answers_by_answerer", "database")}, indent=1))
        return 0
    if args.command == "rules":
        from .answers import record_rules, write_tests

        write_tests()
        print(json.dumps(record_rules(), indent=1)[:3000])
        return 0
    if args.command == "answers-export":
        from .answers import export_gliner2

        print(json.dumps(export_gliner2()["counts"], indent=1, default=str)[:4000])
        return 0
    if args.command == "answers-summary":
        from .answers import summary

        print(json.dumps(summary()["answerers"], indent=1, default=str)[:6000])
        return 0
    if args.command == "openml":
        from . import openml

        print(openml.fetch(args.limit))
        return 0
    if args.command == "selfplay":
        from . import selfplay

        print(selfplay.solve([k for k in args.kinds.split(",") if k] or None, args.seeds))
        return 0
    if args.command == "export":
        from .train import export

        weights = {k: float(v) for k, _, v in (w.partition("=") for w in args.weights.split(",") if w)}
        m = export(args.target, args.lake, args.out, weights, thinking=not args.no_think)
        print(json.dumps({k: m[k] for k in ("target", "train", "val", "by_origin", "with_reasoning", "refused")}, indent=1))
        return 0
    if args.command == "benchmark":
        from . import benchmark

        if args.answerer != "rules" and not args.yes:
            print(f"would ask the {args.answerer} model up to {args.limit} questions (cap €{args.cap_eur}); add --yes to run")
            return 0
        doc = benchmark.run(args.answerer, args.split, args.limit, args.cap_eur)
        print(json.dumps({k: doc.get(k) for k in ("id", "answerer", "split", "cases", "problems", "families", "spent_eur")}, indent=1, default=str))
        return 0
    if args.command == "automl":
        from . import automl

        doc = automl.run(args.store)
        print(json.dumps({k: doc.get(k) for k in ("id", "result")}, indent=1, default=str))
        return 0
    if args.command == "summary":
        print(MANIFEST.read_text(encoding="utf-8") if MANIFEST.exists() else "no lake built yet: make lake")
        return 0
    summary = build(args.lake, workspaces(args.home))
    for s in summary["sets"]:
        flag = "ok  " if s["written"] else "FAIL"
        print(f"{flag} {s['name']:<40} {s['rows']:>7} rows  {s['by_split']}  {s['by_origin']}" + ("" if s["written"] else f"  {s['problems'][:2]}"))
    print(f"{summary['inputs']} inputs; {summary['workspaces_read']} workspaces read, skipped: {summary['workspaces_skipped']}; "
          f"silver only: {summary['silver_only']} → {args.lake}")
    return 0 if all(s["written"] for s in summary["sets"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
