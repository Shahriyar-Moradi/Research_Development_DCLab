"""``python -m dclab_rnd.kinds [KIND ...]``: run the kinds' acceptance (16.2) and write the capability file."""

from __future__ import annotations

import argparse
import sys

from . import acceptance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kinds", nargs="*", help="only these kinds (the file then keeps the others' last results)")
    args = parser.parse_args(argv)
    results = acceptance.run_all(args.kinds or None)
    if args.kinds:
        ran = {r["kind"] for r in results}
        results = [r for r in acceptance.load() if r["kind"] not in ran] + results
    acceptance.write(results)
    for row in acceptance.matrix(results):
        print(f"{row['kind']:<14} {row['status']:<10} {row['passed']}/{row['decisions']}"
              + (f"  missing: {'; '.join(row['missing'])}" if row["missing"] else ""))
    return 0 if all(r["status"] == "supported" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
