"""python -m dclab_rnd.storage upgrade | status: create or update the product's database schema."""

import argparse
import sys

from . import db
from .models import TABLES


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.storage", description=__doc__.strip().splitlines()[0])
    parser.add_argument("command", choices=["upgrade", "status"])
    args = parser.parse_args(argv)
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
    print(f"schema revision: {db.current() or 'none (run: python -m dclab_rnd.storage upgrade)'}; tables: {', '.join(TABLES)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
