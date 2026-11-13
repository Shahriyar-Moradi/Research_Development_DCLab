"""CLI for task_planning_v4 benchmark (headroom-check, verify-sealed, run-offline, run-live)."""
from __future__ import annotations

import argparse
import json

from . import run_headroom_check, run_live, run_offline, verify_sealed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="TP-08 task planning quality benchmark (v4).")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_hr = sub.add_parser("headroom-check", help="Prove baseline leaves real planning headroom.")
    p_hr.add_argument("--split", choices=("dev", "sealed"), default="dev")
    sub.add_parser("verify-sealed", help="Verify cases_sealed.json matches sealed.sha256.")

    p_off = sub.add_parser("run-offline", help="Run $0 scripted offline validation across B0, B1, and C.")
    p_off.add_argument("--split", choices=("dev", "sealed"), default="dev")
    p_off.add_argument("--limit", type=int, default=0)

    p_live = sub.add_parser("run-live", help="Run live model evaluation on dev or sealed (once!).")
    p_live.add_argument("--split", choices=("dev", "sealed"), default="dev")
    p_live.add_argument("--workers", type=int, default=4)
    p_live.add_argument("--cap-usd", default="1.20")
    p_live.add_argument("--tag", default="20261113")
    p_live.add_argument("--limit", type=int, default=0)
    p_live.add_argument("--result-name", default="")

    p_studio = sub.add_parser("run-live-studio", help="Run live end-to-end SolutionTeam scenarios with OpenRouter.")
    p_studio.add_argument("--cap-usd", default="1.20")
    p_studio.add_argument("--tag", default="adaptive_studio_live_20261113")

    args = parser.parse_args(argv)
    if args.cmd == "verify-sealed":
        sha = verify_sealed()
        print(json.dumps({"sealed_sha256": sha, "ok": True}))
        return 0
    if args.cmd == "run-studio" or args.cmd == "run-live-studio":
        from .live_studio import run_live_studio

        summary = run_live_studio(cap_usd=args.cap_usd, tag=args.tag)
        print(json.dumps(summary, indent=2))
        return 0 if summary["scenarios_passed"] == summary["scenarios_total"] else 1
    if args.cmd == "run-offline":
        doc = run_offline(split=args.split, limit=(args.limit or None))
        print(json.dumps({"split": doc["split"], "cases_run": doc["cases_run"], "judgment": doc["judgment"]}, indent=2))
        return 0
    if args.cmd == "run-live":
        doc = run_live(
            split=args.split,
            workers=args.workers,
            cap_usd=args.cap_usd,
            tag=args.tag,
            limit=(args.limit or None),
            result_name=(args.result_name or None),
        )
        print(json.dumps({
            "split": doc["split"],
            "verdict": doc["verdict"],
            "spent_eur": doc["spent_eur"],
            "exchange_capture": doc["exchange_capture"],
            "judgment": doc["judgment"],
            "vs_deterministic_catalog": doc["vs_deterministic_catalog"],
        }, indent=2))
        return 0 if doc["verdict"] == "PASS" else 1
    rep = run_headroom_check(args.split)
    print(json.dumps(rep, indent=2))
    return 0 if rep["has_headroom"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
