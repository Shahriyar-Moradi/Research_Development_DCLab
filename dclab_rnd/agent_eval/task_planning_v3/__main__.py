"""CLI entrypoint for task_planning_v3 freeze, $0 offline smoke, and live release run."""
from __future__ import annotations

import argparse
import json

from . import CAMPAIGN, freeze, run_live_campaign, run_offline


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run task_planning_v3 release evaluation.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_freeze = sub.add_parser("freeze", help="Freeze cases, code hashes, thresholds, and OpenRouter prices.")
    p_freeze.add_argument("--live-model", required=True)
    p_freeze.add_argument("--cap-usd", default="3.50")

    p_off = sub.add_parser("run-offline", help="Run $0 scripted offline validation across Arms A/B/C.")
    p_off.add_argument("--limit", type=int, default=None)

    p_live = sub.add_parser("run-live", help="Run frozen Arms A, B, and live Arm C.")
    p_live.add_argument("--workers", type=int, default=8)

    args = parser.parse_args(argv)
    if args.cmd == "freeze":
        doc = freeze(CAMPAIGN, live_model=args.live_model, cap_usd=args.cap_usd)
        print(json.dumps({"frozen": doc["campaign_id"], "cap_usd": doc["cap_usd"], "model": doc["live_model"]}))
        return 0
    if args.cmd == "run-offline":
        res = run_offline(limit=args.limit)
        j = res["judgment"] or {}
        print(json.dumps({"cases_run": res["cases_run"], "v1": j.get("verdict"), "v2": j.get("verdict_v2")}))
        return 0
    doc = run_live_campaign(CAMPAIGN, workers=args.workers)
    j = doc["judgment"]
    print(json.dumps({"v1": j["verdict"], "v2": j["verdict_v2"], "summary": doc["summary"]}, indent=2))
    return 0 if j["verdict_v2"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
