"""The product's first conversation, live, on the new written people (S8 B3 of the 2026-10-25 plan: "$15 to test and
fix"): ``python -m dclab_rnd.live_people run --yes --cap-eur 6``.

Dev people of the new homework (``journeys_homework_v1.json``, a person's split ``understand_students.person_split``),
about three of each kind, each played twice: on the rules alone, and with the configured model (Claude only) as the
Home agent and the roles, exactly as ``agent_eval.journeys --live-arms`` plays them. Every live play is kept whole
(``keep``: its drafts with every turn, its project, its events) under ``agent_runs/live_tests/<day>-live-people/``, a
workspace the store reads (Claude's answers are silver there). The report lists, per person, whether each arm reached
the right problem and what went wrong, so the failures can be fixed and the same people played again once.

Nothing here trains on the people: they are dev, and the sealed people are never played here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
HOME = ROOT / "agent_runs" / "live_tests"
PER_KIND = 3
SEED = 0


def people(per_kind: int = PER_KIND, only: Optional[list[str]] = None) -> list[dict[str, Any]]:
    """Dev people of the new homework, the first ``per_kind`` of each kind by id (a fixed choice), or the ids named."""
    from .understand_students import homework, person_split

    chosen, by_kind = [], defaultdict(int)
    for source, j in sorted(homework(), key=lambda sj: sj[1]["id"]):
        if person_split(f"{source}:{j['id']}") != "dev":
            continue
        if only is not None:
            if j["id"] in only:
                chosen.append(j)
            continue
        if by_kind[j["kind"]] < per_kind:
            by_kind[j["kind"]] += 1
            chosen.append(j)
    return chosen


def _row(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in ("id", "kind", "family", "right", "parts", "why", "got", "flags", "models", "spent_eur", "decisions")}


def run(cap_eur: float, label: str = "first", only: Optional[list[str]] = None, home: Optional[Path] = None) -> dict[str, Any]:
    from .agent_eval.journeys import play
    from .core_advisor import GatewayAdvisor
    from .models.gateway import Gateway
    from .models.prices import price_of

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = Path(home or HOME / f"{day}-live-people") / label
    out.mkdir(parents=True, exist_ok=True)
    gateway = Gateway()
    client = gateway.client("home_agent")
    model = str(getattr(client, "model", "") or "")
    if client is None or not model.startswith("claude") or price_of(model) is None:
        raise SystemExit(f"the Home agent must be served by a priced Claude model here (it is {model or 'no model'}): nothing sent")
    client.run_limit_eur = cap_eur
    live = {"client": client, "advisor": GatewayAdvisor(gateway, cap_eur=cap_eur)}
    rows, most, stopped = [], 0.0, None
    for j in people(only=only):
        spent = client.spent_eur + live["advisor"].spent_eur()
        if rows and cap_eur - spent < 1.5 * most:  # the next person could run out mid-way and fall back to the rules
            stopped = f"stopped before {j['id']}: EUR {cap_eur - spent:.3f} left, a person took up to EUR {most:.3f}"
            break
        rules = play(j, SEED, None, None)
        try:
            claude = play(j, SEED, None, live, keep=out / "claude" / j["id"])
        except Exception as error:  # noqa: BLE001 - a person that breaks the live run is a finding, with its reason
            claude = {"id": j["id"], "kind": j["kind"], "right": False, "parts": {}, "why": [f"broke: {type(error).__name__}: {str(error)[:300]}"]}
        most = max(most, client.spent_eur + live["advisor"].spent_eur() - spent)
        row = {"id": j["id"], "kind": j["kind"], "words": j["words"], "rules": _row(rules), "claude": _row(claude)}
        rows.append(row)
        with (out / "rows.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
    spent = round(client.spent_eur + live["advisor"].spent_eur(), 4)
    summary = {"label": label, "people": len(rows), "model": model, "spent_eur": spent, "cap_eur": cap_eur, "stopped": stopped,
               "rules_right": sum(bool(r["rules"]["right"]) for r in rows), "claude_right": sum(bool(r["claude"]["right"]) for r in rows),
               "only_rules_right": [r["id"] for r in rows if r["rules"]["right"] and not r["claude"]["right"]],
               "only_claude_right": [r["id"] for r in rows if r["claude"]["right"] and not r["rules"]["right"]],
               "both_wrong": [r["id"] for r in rows if not r["rules"]["right"] and not r["claude"]["right"]],
               "roles": live["advisor"].report()}
    (out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str) + "\n", encoding="utf-8")
    return summary


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dclab_rnd.live_people", description=__doc__.split("\n")[0])
    ap.add_argument("command", choices=("plan", "run"))
    ap.add_argument("--yes", action="store_true")
    ap.add_argument("--cap-eur", type=float)
    ap.add_argument("--label", default="first")
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args(argv)
    chosen = people(only=args.only)
    if args.command == "plan" or not args.yes or args.cap_eur is None:
        print(f"Plan: {len(chosen)} dev people ({PER_KIND} a kind), each on the rules and with Claude as the Home agent and the roles, "
              "capped; add run --yes --cap-eur N.")
        return 0
    from . import real_shadow

    real_shadow.load_env()
    os.environ.pop("DCLAB_NO_LIVE_MODELS", None)
    print(json.dumps(run(args.cap_eur, args.label, args.only), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
