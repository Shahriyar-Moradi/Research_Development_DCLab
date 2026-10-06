"""python -m dclab_rnd.storage upgrade | status | migrate SOURCE --to TARGET: the product's database schema, and moving a file workspace into it."""

import argparse
from pathlib import Path
import sys

from . import db
from .models import TABLES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.storage", description=__doc__.strip().splitlines()[0])
    parser.add_argument("command", choices=["upgrade", "status", "migrate"])
    parser.add_argument("source", nargs="?", type=Path, help="migrate: the file workspace folder to copy (only read, never changed)")
    parser.add_argument("--to", dest="target", type=Path, help="migrate: the folder of the database workspace it becomes")
    args = parser.parse_args(argv)
    if args.command == "migrate" and (args.source is None or args.target is None):
        parser.error("migrate needs SOURCE and --to TARGET")
    try:
        db.database_url()
    except RuntimeError as error:
        print(error, file=sys.stderr)
        return 2
    problem = db.reachable()
    if problem:
        print(problem + ". Is PostgreSQL running, and does the database exist (createdb dclab_dev)?", file=sys.stderr)
        return 2
    if args.command == "upgrade":
        db.upgrade()
    if args.command == "migrate":
        from .migrate import migrate, print_report

        db.upgrade()  # the schema first: a migration into an old schema would fail half way
        try:
            report = migrate(args.source, args.target)
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001 — the type only: a database error's text can carry rows and connection details
            print(f"The migration stopped: {type(exc).__name__}. Nothing was copied over what the target holds; run it again after fixing the cause.",
                  file=sys.stderr)
            return 1
        print_report(report)
        return 0 if report["ok"] else 1
    print(f"schema revision: {db.current() or 'none (run: python -m dclab_rnd.storage upgrade)'}; tables: {', '.join(TABLES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
