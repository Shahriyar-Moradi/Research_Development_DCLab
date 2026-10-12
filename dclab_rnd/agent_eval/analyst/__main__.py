"""Measure the problem analyst (package 18.2).

    python -m dclab_rnd.agent_eval.analyst --split dev                  # the rules alone, and the plan of a live run
    python -m dclab_rnd.agent_eval.analyst --split dev --yes --cap-eur 1   # sends: the analyst and the plain model
    python -m dclab_rnd.agent_eval.analyst --split sealed --yes --cap-eur 1  # once: a second sealed run is refused

A live run needs the owner's yes and a euro cap; it is stored as evidence/campaigns/analyst_v1/results/ANA-NNN_<split>.json
with its plan, its cost and the gate's verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

from . import GATE, RESULTS, load, measure, next_id, plan, sealed_runs, verdict

MAX_TOKENS = 1500  # the model reasons first: a reply cut at the limit is a failed request, never an empty answer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.analyst")
    parser.add_argument("--split", choices=("dev", "sealed"), default="dev")
    parser.add_argument("--yes", action="store_true", help="send the requests (the plan is printed either way)")
    parser.add_argument("--cap-eur", type=float, default=None, help="required with --yes for a priced model")
    parser.add_argument("--role", choices=("analyst", "checker"), default="analyst",
                        help="checker: the brief checker on the dev and sealed briefs together (one run, its own gate)")
    args = parser.parse_args(argv)
    if args.role == "checker":
        return _checker(args)
    from dotenv import load_dotenv  # the configured model is in .env, as for the server

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)

    cases = load(args.split)
    if args.split == "dev":  # the sealed set's scores are read only in its one run
        rules = measure(cases, None)["summary"]["rules"]
        print(f"rules alone on dev: {rules['right']} right, {rules['wrong']} wrong, {rules['missed']} missed, "
              f"{rules['abstained']} abstained of {rules['fields']}")
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
        print("refused: no model serves problem_analyst here", file=sys.stderr)
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
    record = {"id": next_id(), "campaign_id": "analyst_v1", "kind": "problem_analyst_measured", "split": args.split,
              "at": at, "gate": GATE, "plan": the_plan,
              "spent_eur": round(advisor.spent_eur(), 4), **result,
              "verdict": verdict(result, args.split) if scored else {"gate": GATE["version"], "status": "not scored",
                                                                      "why": f"{len(result['failures'])} request(s) failed"},
              "rule": "Scores on 12 written briefs say how the analyst reads these descriptions; they are not a claim about any "
                      "other text or about production readiness."}
    # the campaign record's own keys, as every campaign's results carry them (the lab page reads them)
    record.update(experiment_id=record["id"], status="completed" if scored else "failed", completed_at=at)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{record['id']}_{args.split}{'' if scored else '_unscored'}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for arm, s in record.get("summary", {}).items():
        print(f"{arm:8s} right {s['right']:3d}  wrong {s['wrong']:3d}  missed {s['missed']:3d}  abstained {s['abstained']:3d}"
              + (f"  new wrong {len(s['new_wrong'])}" if "new_wrong" in s else ""))
    print(f"spent €{record['spent_eur']}; verdict: {record['verdict']['status']}; stored {path.relative_to(RESULTS.parents[2])}")
    return 0


def _checker(args: argparse.Namespace) -> int:
    """The brief checker on all 24 briefs, once: it is gated on the dev and sealed briefs together (CHECKER_GATE)."""
    from dotenv import load_dotenv

    from . import CHECKER_GATE, CHECKER_RESULTS, checker_verdict, measure_checker

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    cases = load("dev") + load("sealed")
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    the_plan = plan(gateway, cases, MAX_TOKENS, purpose="brief_checker")  # its own route, prompt and price
    if not args.yes:
        print(json.dumps(the_plan, indent=1))
        print("nothing was sent: add --yes and --cap-eur to run it")
        return 0
    if sorted(CHECKER_RESULTS.glob("CHK-*_all.json")):
        print("refused: the brief checker was already measured; it is run once", file=sys.stderr)
        return 2
    if not the_plan["available"] or (not the_plan["local"] and (the_plan["max_eur"] is None or args.cap_eur is None or args.cap_eur <= 0)):
        print("refused: no priced model with a cap above zero serves brief_checker here", file=sys.stderr)
        return 2
    if the_plan["max_eur"] is not None and the_plan["max_eur"] > args.cap_eur:
        print(f"refused: the run could cost up to €{the_plan['max_eur']}, above the cap of €{args.cap_eur}", file=sys.stderr)
        return 2
    from dclab_rnd.core_advisor import GatewayAdvisor

    gateway.shadows = None
    advisor = GatewayAdvisor(gateway, max_tokens=MAX_TOKENS, cap_eur=args.cap_eur)
    result = measure_checker(cases, advisor)
    scored = not result["failures"]
    if not scored:
        result = {"requests": result["requests"], "failures": result["failures"]}
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    numbers = [int(p.name[4:7]) for p in CHECKER_RESULTS.glob("CHK-*.json")]
    rid = f"CHK-{max(numbers, default=0) + 1:03d}"
    record = {"id": rid, "experiment_id": rid, "campaign_id": "brief_checker_v1", "kind": "brief_checker_measured", "split": "all",
              "at": at, "completed_at": at, "status": "completed" if scored else "failed", "gate": CHECKER_GATE, "plan": the_plan,
              "spent_eur": round(advisor.spent_eur(), 4), **result,
              "verdict": checker_verdict(result) if scored else {"gate": CHECKER_GATE["version"], "status": "not scored"},
              "rule": "Questions on 24 written briefs; the sealed half is not blind for this role. Not a claim about other text."}
    CHECKER_RESULTS.mkdir(parents=True, exist_ok=True)
    path = CHECKER_RESULTS / f"{rid}_all{'' if scored else '_unscored'}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record.get("summary"), indent=1))
    print(f"spent €{record['spent_eur']}; verdict: {record['verdict']['status']}; stored {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
