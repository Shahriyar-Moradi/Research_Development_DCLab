"""``python -m dclab_rnd.learning baseline --family groups --split check --n 20 [--write]``: the scoreboard's baseline (H0)."""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.learning", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    base = sub.add_parser("baseline", help="today's DCLab on a family's variants, nothing learned: the zero of every learning curve")
    base.add_argument("--family", default="groups")
    base.add_argument("--split", default="check", choices=("learning", "check", "sealed"))
    base.add_argument("--n", type=int, default=20)
    base.add_argument("--write", action="store_true", help="store the result in evidence/campaigns/learning_v1/results/")
    args = parser.parse_args(argv)
    from .. import experience
    from ..settings import Settings
    from .scoreboard import SealedSpent, baseline

    before = experience.current_log()  # every attempt is a run of the learning agent in the workspace's record
    if before is None:
        experience.install(experience.open_log(Settings.load().agent_home))
    try:
        result = baseline(args.family, args.split, args.n, write=args.write)
    except SealedSpent as error:
        print(str(error), file=sys.stderr)
        return 2
    finally:
        if before is None:
            experience.install(None)
    print(json.dumps({"id": result["id"], "path": result.get("path"), "verdict": result["verdict"], "summary": result["summary"]}, indent=1))
    return 1 if result["verdict"].startswith("fail") else 0


if __name__ == "__main__":
    sys.exit(main())
