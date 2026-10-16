"""Measure the map roles (package 24.2).

    python -m dclab_rnd.agent_eval.complex --split dev                   # the rules alone, and the plan of a live run
    python -m dclab_rnd.agent_eval.complex --split dev --yes --cap-eur 0.5   # sends one request a case: the role and the plain arm read the same reply
    python -m dclab_rnd.agent_eval.complex --split sealed --yes --cap-eur 0.5  # once: a second sealed run is refused

A live run needs the owner's yes and a euro cap; it is stored as evidence/campaigns/map_roles_v1/results/MAP-NNN_<split>.json
with its plan, its cost and the gate's verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import GATE, RESULTS, load, measure, next_id, plan, sealed_runs, verdict

MAX_TOKENS = 2500  # the model reasons before its JSON and how long is not in our hands: a reply cut at the limit is a failed request (the run is unscored), never a short answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.complex")
    parser.add_argument("--split", choices=("dev", "sealed"), default="dev")
    parser.add_argument("--yes", action="store_true", help="send the requests (the plan is printed either way)")
    parser.add_argument("--cap-eur", type=float, default=None, help="required with --yes for a priced model")
    args = parser.parse_args(argv)
    from dotenv import load_dotenv  # the configured model is in .env, as for the server

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

    cases = load(args.split)
    if args.split == "dev":  # the sealed set's scores are read only in its one run
        rules = measure(cases, None)["summary"]["rules"]
        print(f"rules alone on dev: mean F1 {rules['mean_f1']}, size right {rules['size_right']} of {rules['cases']}, exact {rules['exact']}")
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
        print("refused: no model serves decomposer here", file=sys.stderr)
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
    record = {"id": next_id(), "campaign_id": "map_roles_v1", "kind": "map_roles_measured", "split": args.split, "at": at, "gate": GATE, "plan": the_plan,
              "spent_eur": round(advisor.spent_eur(), 4), **result,
              "verdict": verdict(result, args.split) if scored else {"gate": GATE["version"], "status": "not scored", "why": "a request failed"}}
    record.update(experiment_id=record["id"], status="completed" if scored else "failed", completed_at=at)  # as the other suites' records, so the Lab page lists it
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / f"{record['id']}_{args.split}{'' if scored else '_unscored'}.json"
    out.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{out.name}: {record['verdict']['status']}; spent €{record['spent_eur']}")
    if scored:
        print(json.dumps(result["summary"], indent=1))
    return 0 if scored else 1


if __name__ == "__main__":
    sys.exit(main())
