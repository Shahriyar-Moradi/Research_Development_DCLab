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
    o = sub.add_parser("openml", help="26.2: fetch and profile OpenML's public tables (resumable; the owner's yes, 2026-10-16)")
    o.add_argument("--limit", type=int, default=1200)
    p = sub.add_parser("selfplay", help="27.3: solve the engine kinds' replicas by seed, keep only verified solves")
    p.add_argument("--seeds", type=int, default=20)
    p.add_argument("--kinds", default="", help="comma-separated kind keys (default: every engine kind)")
    args = parser.parse_args(argv)
    if args.command == "openml":
        from . import openml

        print(openml.fetch(args.limit))
        return 0
    if args.command == "selfplay":
        from . import selfplay

        print(selfplay.solve([k for k in args.kinds.split(",") if k] or None, args.seeds))
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
