"""``python -m dclab_rnd.agent_eval.no_data``: the no-data blueprint's measurement. Free by default (nothing is sent); a live
run needs ``--live --yes --cap-eur`` and keeps every model call, with its thinking, under ``agent_runs/live_tests/``."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

from . import ARMS, KEEP_ROOT, RESULTS, ROOT, evaluate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure the no-data blueprint: the rules, the harness and, live, a model alone.")
    parser.add_argument("--split", default="dev", choices=("dev",))
    parser.add_argument("--cap-eur", type=float, default=None)
    parser.add_argument("--live", action="store_true", help="send the cases to the configured models (costs money)")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="the first N cases only")
    args = parser.parse_args(argv)
    if not args.live:
        res = evaluate(split=args.split, arms=ARMS, limit=args.limit)
        print(json.dumps({k: res[k] for k in ("gate_id", "passed", "checks", "summaries", "compare")}, indent=1))
        print("free run: the rules and the harness's floor; the plain arm is not measured without a model")
        return 0 if res["passed"] else 1
    if not args.yes or args.cap_eur is None or args.cap_eur <= 0:
        print("a live run sends the cases to a model: add --yes and --cap-eur above zero", file=sys.stderr)
        return 2
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
    from dclab_rnd import control
    from dclab_rnd.models.gateway import Gateway, mark_invented_tables
    from dclab_rnd.models.usage import FileUsage

    if not control.live_allowed():
        print("refused: the owner has switched live model calls off", file=sys.stderr)
        return 2
    used = [p.name for p in RESULTS.glob("NDE-*.json")] + [p.name for p in KEEP_ROOT.glob("*-no-data-eval-NDE-*")]
    numbers = [int(m.group(1)) for n in used if (m := re.search(r"NDE-(\d+)", n))]
    run = f"NDE-{1 + max(numbers or [0]):03d}"
    run_dir = KEEP_ROOT / f"{datetime.now(timezone.utc):%Y-%m-%d}-no-data-eval-{run}"
    # the run's own gateway: every exchange, with its thinking, under the run's folder (the cases are written, no table exists)
    gateway = mark_invented_tables(Gateway(FileUsage(run_dir / "usage.jsonl")), "no-data evaluation: written cases, no table")
    gateway.home, gateway.shadows = run_dir, None
    if not gateway.available("exam_plain"):
        print("refused: no model serves the arms", file=sys.stderr)
        return 2
    try:
        res = evaluate(split=args.split, arms=ARMS, gateway=gateway, cap_eur=args.cap_eur, limit=args.limit)
    except Exception as error:  # noqa: BLE001: the provider refused the key or the account ran dry: no result is stored for a run cut short
        print(f"stopped: {type(error).__name__}: {str(error)[:200]}; what was sent is kept in {run_dir.relative_to(ROOT)}; no result stored", file=sys.stderr)
        return 2
    record = {"id": run, "split": args.split, "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "arms": list(ARMS),
              "cap_eur": args.cap_eur, "kept": str(run_dir.relative_to(ROOT)), **res}
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{run}_{args.split}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: res[k] for k in ("gate_id", "passed", "checks", "summaries", "compare", "spent_eur")}, indent=1))
    print(f"stored {path.relative_to(ROOT)}; kept {run_dir.relative_to(ROOT)}; next: make archive")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
