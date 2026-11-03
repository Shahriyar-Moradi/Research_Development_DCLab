"""``python -m dclab_rnd.agent_eval.system_tasks --split dev|sealed [--yes]`` (see the package)."""

from __future__ import annotations

import argparse
import json
import sys

from . import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.system_tasks")
    parser.add_argument("--split", default="dev", choices=("dev", "sealed"))
    parser.add_argument("--yes", action="store_true", help="sealed: read the sealed seeds, once")
    args = parser.parse_args(argv)
    path = run(args.split, args.yes)
    doc = json.loads(path.read_text())
    for r in doc["rows"]:
        print(f"{'ok  ' if r['passed'] else 'FAIL'} {r['kind']:15} seed {r['seed']:4}  checks {r['checks'][0]}/{r['checks'][1]}  {r['seconds']} s"
              + (f"  missing: {'; '.join(r['missing'][:3])}" if r["missing"] else ""))
    print(doc["verdict"])
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
