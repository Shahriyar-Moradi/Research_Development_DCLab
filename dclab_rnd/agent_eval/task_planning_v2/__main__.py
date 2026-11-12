"""CLI entrypoint for versioned task_planning_v2 evaluation."""
from __future__ import annotations

import argparse
import json

from . import CAMPAIGN, freeze, load_dotenv, run_live_campaign, run_offline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Versioned TP-06 v2 evaluation campaign")
    parser.add_argument("--mode", choices=("offline", "freeze", "run-live"), required=True)
    parser.add_argument("--cap-usd", default="3.80")
    parser.add_argument("--model", default="")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)

    if args.mode == "offline":
        res = run_offline(limit=args.limit)
        print(json.dumps({"cases_run": res["cases_run"], "verdict": (res["judgment"] or {}).get("verdict"), "reasons": (res["judgment"] or {}).get("reasons")}, indent=2))
        return 0
    if args.mode == "freeze":
        load_dotenv()
        from dclab_rnd.models import settings

        model = args.model.strip() or settings.served_model("task_planner")
        doc = freeze(CAMPAIGN, live_model=model, cap_usd=args.cap_usd)
        print(json.dumps({"frozen": True, "model": doc["live_model"], "cap_usd": doc["cap_usd"], "cases": len(doc["cases"])}, indent=2))
        return 0
    doc = run_live_campaign(CAMPAIGN, workers=args.workers)
    print(json.dumps({"summary": doc["summary"], "verdict": doc["judgment"]["verdict"], "reasons": doc["judgment"]["reasons"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
