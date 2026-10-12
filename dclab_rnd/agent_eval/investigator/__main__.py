"""Measure the data investigator (package 18.3).

    python -m dclab_rnd.agent_eval.investigator --split dev                  # the rules alone, and the plan of a live run
    python -m dclab_rnd.agent_eval.investigator --split dev --yes --cap-eur 1   # sends: the investigator and the plain model
    python -m dclab_rnd.agent_eval.investigator --split sealed --yes --cap-eur 1  # once: a second sealed run is refused

A live run needs the owner's yes and a euro cap; it is stored as evidence/campaigns/investigator_v1/results/INV-NNN_<split>.json
with its plan, its cost and the gate's verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from . import GATE, RESULTS, load, measure, next_id, plan, sealed_runs, verdict

MAX_TOKENS = 2000  # the model reasons first: a reply cut at the limit is a failed request, never an empty answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.investigator")
    parser.add_argument("--split", choices=("dev", "sealed"), default="dev")
    parser.add_argument("--yes", action="store_true", help="send the requests (the plan is printed either way)")
    parser.add_argument("--cap-eur", type=float, default=None, help="required with --yes for a priced model")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv  # the configured model is in .env, as for the server

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

    cases = load(args.split)
    if args.split == "dev":  # the sealed set's scores are read only in its one run
        rules = measure(cases, None)["summary"]["rules"]
        print(f"rules alone on dev: {rules['found']} of {rules['traps']} planted traps found, {rules['false_alarms']} clean columns accused")
    if not args.yes:
        from dclab_rnd.models.gateway import for_workspace

        print(json.dumps(plan(for_workspace(), cases, MAX_TOKENS), indent=1))
        print("nothing was sent: add --yes and --cap-eur to run it")
        return 0
    if args.split == "sealed" and sealed_runs():
        print(f"refused: the sealed set was already run ({sealed_runs()[0].name}); it is run once", file=sys.stderr)
        return 2

    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()  # the workspace usage log, its monthly cap and its routing
    gateway.shadows = None  # measurement traffic stays out of the shadow log
    the_plan = plan(gateway, cases, MAX_TOKENS)
    if not the_plan["available"]:
        print("refused: no model serves data_investigator here", file=sys.stderr)
        return 2
    if not the_plan["local"] and the_plan["max_eur"] is None:
        print("refused: the model has no price, so no euro cap can stop it (add it to the price table)", file=sys.stderr)
        return 2
    if not the_plan["local"] and (args.cap_eur is None or args.cap_eur <= 0):
        print("refused: a remote model needs --cap-eur above zero", file=sys.stderr)
        return 2
    if the_plan["max_eur"] is not None and args.cap_eur is not None and the_plan["max_eur"] > args.cap_eur:
        print(f"refused: the run could cost up to €{the_plan['max_eur']}, above the cap of €{args.cap_eur}", file=sys.stderr)
        return 2
    advisor = GatewayAdvisor(gateway, max_tokens=MAX_TOKENS, cap_eur=args.cap_eur)
    result = measure(cases, advisor)
    scored = not result["failures"]
    if not scored:  # a failed request: kept as a record of what it cost, without values or scores, and never a sealed run
        result = {"requests": result["requests"], "failures": result["failures"]}
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record = {"id": next_id(), "campaign_id": "investigator_v1", "kind": "data_investigator_measured", "split": args.split,
              "at": at, "gate": GATE, "plan": the_plan,
              "spent_eur": round(advisor.spent_eur(), 4), **result,
              "verdict": verdict(result, args.split) if scored else {"gate": GATE["version"], "status": "not scored",
                                                                      "why": f"{len(result['failures'])} request(s) failed"},
              "rule": "Scores on these seeded tables say how the investigator reads these column names; they are not a claim about "
                      "any other table or about production readiness."}
    # the campaign record's own keys, as every campaign's results carry them (the lab page reads them)
    record.update(experiment_id=record["id"], status="completed" if scored else "failed", completed_at=at)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{record['id']}_{args.split}{'' if scored else '_unscored'}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for arm, s in record.get("summary", {}).items():
        print(f"{arm:12s} found {s['found']:3d} of {s['traps']:3d} traps  false alarms {s['false_alarms']:3d}")
    print(f"spent €{record['spent_eur']}; verdict: {record['verdict']['status']}; stored {path.relative_to(RESULTS.parents[2])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
