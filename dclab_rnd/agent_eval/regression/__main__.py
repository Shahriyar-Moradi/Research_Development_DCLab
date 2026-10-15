"""``python -m dclab_rnd.agent_eval.regression [--home WORKSPACE] [--workers N] [--only ID ...] [--no-store]``; with a served
model: ``--live --cap-eur N`` (the gateway's configured models, never without a cap). ``--stale`` says what changed since
the last run."""

from __future__ import annotations

import argparse

from . import exercised, run, stale, store, verdict


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", default=None, help="a workspace folder: its accepted session cases join the suite")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--only", nargs="*")
    parser.add_argument("--no-store", action="store_true")
    parser.add_argument("--live", action="store_true", help="the gateway's served models play the agents and roles")
    parser.add_argument("--cap-eur", type=float, default=None, help="the most the live run may spend, split between the Home agent and the roles")
    parser.add_argument("--serve", nargs="*", default=[], help="purpose=tier: a candidate model served for this run only, as an approved routing would")
    parser.add_argument("--stale", action="store_true")
    args = parser.parse_args()
    if args.stale:
        changed = stale()
        print("the last run is current" if not changed else "changed since the last run: " + ", ".join(changed))
        return 0 if not changed else 1
    live, gateway = None, None
    if args.live:
        if args.cap_eur is None:
            print("a live run spends: give --cap-eur")
            return 2
        from types import SimpleNamespace

        from dclab_rnd.core_advisor import GatewayAdvisor
        from dclab_rnd.models import settings
        from dclab_rnd.models.gateway import Gateway

        gateway = Gateway()
        if args.serve:  # the candidate served for this run only, never a fallback to another tier
            doc = {"purposes": {}}
            for item in args.serve:
                purpose, tier = item.split("=", 1)
                current = settings.public(tier)
                doc["purposes"][purpose] = {"serve": tier, "serve_model": current["model"], "serve_endpoint": current["endpoint"]}
            gateway.routing = SimpleNamespace(get=lambda: doc)
            for purpose, entry in doc["purposes"].items():
                if gateway.route(purpose)[0] != entry["serve"]:
                    print(f"{purpose} would not be served by {entry['serve']}: refused")
                    return 2
        client = gateway.client("home_agent")
        if client is None:
            print("no model can serve the Home agent here (DCLAB_NO_LIVE_MODELS, or no key): nothing live to run")
            return 2
        client.run_limit_eur = args.cap_eur / 2  # one budget: half for the Home agent, half for the roles
        live = {"client": client, "advisor": GatewayAdvisor(gateway, cap_eur=args.cap_eur / 2)}
    rows = run(args.home, workers=args.workers, live=live, only=args.only)
    for r in rows:
        bad = [c["step"] for c in r.get("checks") or [] if not c["ok"]]
        print(f"{'ok  ' if r['passed'] else 'MISS'} {r['id']:7} {r['source']:8} {'; '.join(r.get('why') or [])[:200]} {('steps wrong: ' + ', '.join(bad)) if bad else ''}")
    result = verdict(rows)
    print(result)
    spent = sum(float(r.get("spent_eur") or 0.0) for r in rows)
    print(f"spent €{spent:.4f}" + (f" of €{args.cap_eur:.2f}" if args.cap_eur is not None else ""))
    if not args.no_store and not args.only:
        print(store(rows, result, exercised(rows), spent_eur=spent, live=live is not None))
    return 0 if result.startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
