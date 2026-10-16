"""Trusted testers from the command line (package 25.11).

    python -m dclab_rnd.testers notice                                       # what a tester agrees to
    python -m dclab_rnd.testers setup --name "Ali" --email ali@example.com --repo git@github.com:OWNER/dclab-tester-records.git --agree
    python -m dclab_rnd.testers share [--tables ask|yes|no]                  # the tester's: send what changed (make share)
    python -m dclab_rnd.testers collect --records PATH                       # the owner's: pull and rebuild each tester (make collect)
    python -m dclab_rnd.testers list                                         # the owner's: what came back, per tester

The workspace is DCLAB_AGENT_HOME (agent_runs/ by default). Nothing is sent to a model.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import NOTICE, TesterError, collect, setup, share, summary


def _home() -> Path:
    from ..settings import Settings

    return Path(Settings().agent_home)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.testers", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("notice")
    s = sub.add_parser("setup")
    s.add_argument("--name", required=True)
    s.add_argument("--email", required=True)
    s.add_argument("--repo", required=True, help="the owner's private records repository (git URL)")
    s.add_argument("--clone", type=Path, default=None, help="where to keep its clone (default: <workspace>/tester-records)")
    s.add_argument("--agree", action="store_true", help="you read the notice (python -m dclab_rnd.testers notice) and agree")
    sh = sub.add_parser("share")
    sh.add_argument("--tables", choices=("ask", "yes", "no"), default="ask", help="for a table not decided before (or changed since): ask, send it, or keep it")
    sh.add_argument("--forget-tables", action="store_true", help="forget every earlier yes or no about tables, and decide again")
    sh.add_argument("--no-push", action="store_true", help="commit the bundle, push later")
    c = sub.add_parser("collect")
    c.add_argument("--records", type=Path, default=Path(os.environ.get("DCLAB_TESTER_RECORDS", "") or "tester-records"),
                   help="the owner's clone of the records repository (DCLAB_TESTER_RECORDS)")
    c.add_argument("--no-pull", action="store_true")
    sub.add_parser("list")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    from ..settings import ROOT

    load_dotenv(ROOT / ".env", override=False)  # DCLAB_AGENT_HOME, DCLAB_TESTER_RECORDS and the database setting, as the server reads them
    if args.command == "collect" and not os.environ.get("DCLAB_TESTER_RECORDS") and str(args.records) == "tester-records":
        args.records = ROOT / "tester-records"
    elif args.command == "collect" and os.environ.get("DCLAB_TESTER_RECORDS") and str(args.records) == "tester-records":
        args.records = Path(os.environ["DCLAB_TESTER_RECORDS"])
    home = _home()
    try:
        if args.command == "notice":
            print(NOTICE)
        elif args.command == "setup":
            if not args.agree:
                print(NOTICE + "\n\nRead it, then run again with --agree.")
                return 2
            setup(home, name=args.name, email=args.email, repo=args.repo, clone=args.clone, agreed=True, env_file=ROOT / ".env")
            print("Next: put your own ANTHROPIC_API_KEY and DCLAB_LLM_PROVIDER=anthropic in .env, start DCLab (make notebook), and run make share now and then.")
        elif args.command == "share":
            def ask(rel: str, size: int) -> bool:
                if args.tables != "ask":
                    return args.tables == "yes"
                if not sys.stdin.isatty():
                    return False
                return input(f"Send the table {rel} ({size / 1024 / 1024:.1f} MB) to the owner? [y/N] ").strip().lower() in ("y", "yes")

            if args.forget_tables:
                from . import _save, load as load_doc

                doc = load_doc(home)
                if doc:
                    doc["tables"] = {}
                    _save(home, doc)
            share(home, ask=ask, push=not args.no_push)
        elif args.command == "collect":
            collect(home, args.records, pull=not args.no_pull)
        elif args.command == "list":
            print(json.dumps(summary(home), indent=2, ensure_ascii=False))
    except TesterError as exc:
        print(f"testers: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
