"""CLI runner for the 3-arm No-Data Understanding & 5-Gate Harness evaluation (`python -m dclab_rnd.agent_eval.no_data`)."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from . import ARMS, evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate No-Data Problem Understanding & 5-Gate Scientific Harness.")
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument("--cap-eur", type=float, default=0.60)
    parser.add_argument("--live", action="store_true", help="Use configured live Gateway models within --cap-eur.")
    parser.add_argument("--limit", type=int, default=0, help="Optional subset of cases for fast live smoke check.")
    args = parser.parse_args(argv)

    gateway = None
    if args.live:
        from ...models import for_workspace

        ws_dir = Path("data/workspace")
        gateway = for_workspace(ws_dir if ws_dir.exists() else Path(tempfile.mkdtemp()))

    res = evaluate(split=args.split, arms=ARMS, gateway=gateway, cap_eur=args.cap_eur, limit=args.limit)
    print(json.dumps({"gate_id": res["gate_id"], "passed": res["passed"], "checks": res["checks"], "summaries": res["summaries"]}, indent=2))
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
