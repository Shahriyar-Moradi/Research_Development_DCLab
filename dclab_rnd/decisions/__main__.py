"""Decision samples from the command line.

    python -m dclab_rnd.decisions stats  [--home agent_runs]
    python -m dclab_rnd.decisions export [--home agent_runs] [--out out/decisions]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from . import export, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.decisions", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    home = os.environ.get("DCLAB_AGENT_HOME", "agent_runs")
    p = sub.add_parser("stats", help="counts per decision point and how many carry a human label")
    p.add_argument("--home", type=Path, default=Path(home))
    p = sub.add_parser("export", help="samples.jsonl and manifest.json, split by dataset, ids masked")
    p.add_argument("--home", type=Path, default=Path(home))
    p.add_argument("--out", type=Path, default=Path("out/decisions"))
    args = parser.parse_args(argv)
    if args.command == "stats":
        print(json.dumps(stats(args.home), indent=2))
    else:
        manifest = export(args.home, args.out)
        print(json.dumps(manifest, indent=2))
        print(f"written: {args.out / 'samples.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
