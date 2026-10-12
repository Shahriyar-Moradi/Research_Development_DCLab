"""Measure Step 3's roles on the replicas (packages 18.4, 18.5, 18.6).

    python -m dclab_rnd.agent_eval.roles                          # the rules alone (planner) and the plan of a live run
    python -m dclab_rnd.agent_eval.roles --yes --cap-eur 0.10     # sends: the planner, the engineer and the critic, once each

Each role's run is stored as evidence/campaigns/roles_v1/results/ROL-NNN_<role>.json with its cost and its gate's verdict;
a run in which a request failed is stored unscored.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import (GATES, RESULTS, critic_verdict, measure_reporter, reporter_verdict, engineer_v2_verdict, engineer_verdict, measure_critic, measure_engineer,
               measure_engineer_v2, measure_planner, planner_verdict)

MAX_TOKENS = 3000  # the planner's and the engineer's answers are longer than a field's
ROLES = {"planner": ("solution_planner", 3, measure_planner, planner_verdict),
         "engineer": ("feature_engineer", 1, measure_engineer, engineer_verdict),
         "engineer_v2": ("feature_engineer", 2, measure_engineer_v2, engineer_v2_verdict),
         "critic": ("critic", 4, measure_critic, critic_verdict),
         "reporter": ("reporter", 3, measure_reporter, reporter_verdict)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.roles")
    parser.add_argument("--role", choices=(*ROLES, "all"), default="all")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    from dclab_core import prompts
    from dclab_rnd.models import prices, settings
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    gateway.shadows = None
    chosen = [r for r in ROLES if r != "engineer_v2"] if args.role == "all" else [args.role]  # v2 is run on its own
    plan = {}
    for role in chosen:
        purpose, requests, _, _ = ROLES[role]
        tier = gateway.route(purpose)[0]
        public = settings.public(tier)
        each = prices.estimate(public.get("model"), bool(public.get("local")), len(prompts.text(purpose)) + 12_000, MAX_TOKENS)
        plan[role] = {"purpose": purpose, "model": public.get("model"), "local": bool(public.get("local")),
                      "available": gateway.available(purpose), "max_requests": requests,
                      "max_eur": round(each * requests, 3) if each is not None else None}
    total = sum(p["max_eur"] or 0 for p in plan.values())
    print(json.dumps({"roles": plan, "max_eur_total": round(total, 3)}, indent=1))
    if not args.yes:
        if "planner" in chosen:
            print("rules alone:", json.dumps(measure_planner(None)["summary"]))
        print("nothing was sent: add --yes and --cap-eur to run it")
        return 0
    if any(not p["available"] or (not p["local"] and p["max_eur"] is None) for p in plan.values()):
        print("refused: a role has no model, or a remote model without a price", file=sys.stderr)
        return 2
    if args.cap_eur is None or args.cap_eur <= 0 or total > args.cap_eur:
        print(f"refused: the runs could cost up to €{total:.3f}; give --cap-eur at least that", file=sys.stderr)
        return 2
    from dclab_rnd.agent_eval.analyst import Counting
    from dclab_rnd.core_advisor import GatewayAdvisor

    RESULTS.mkdir(parents=True, exist_ok=True)
    left = args.cap_eur
    for role in chosen:
        purpose, _, measure, verdict = ROLES[role]
        advisor = GatewayAdvisor(gateway, max_tokens=MAX_TOKENS, cap_eur=left)
        counting = Counting(advisor)
        result = measure(counting)
        spent = advisor.spent_eur()
        left -= spent
        scored = not counting.failures
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        numbers = [int(p.name[4:7]) for p in RESULTS.glob("ROL-*.json")]
        rid = f"ROL-{max(numbers, default=0) + 1:03d}"
        record = {"id": rid, "experiment_id": rid, "campaign_id": "roles_v1", "kind": f"{role}_measured", "role": role, "at": at,
                  "completed_at": at, "status": "completed" if scored else "failed", "gate": GATES[role], "plan": plan[role],
                  "spent_eur": round(spent, 4), "requests": counting.requests, "failures": counting.failures,
                  **({"result": result} if scored else {}),
                  "verdict": verdict(result) if scored else {"gate": GATES[role]["version"], "status": "not scored"},
                  "rule": "Measured on invented replicas; not a claim about other data or about production readiness."}
        path = RESULTS / f"{rid}_{role}{'' if scored else '_unscored'}.json"
        path.write_text(json.dumps(record, indent=1, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
        print(f"{role}: spent €{record['spent_eur']}, verdict {record['verdict']['status']}, stored {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
