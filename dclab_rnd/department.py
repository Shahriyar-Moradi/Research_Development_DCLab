"""A department agent on a pricing replica (package 30.22): ``python -m dclab_rnd.department [--write]``.

Invented products (2,000: a price, a cost, weeks in stock, units a week). A rule agent proposes discounts by how long a product has sat
(10%, 30%, 40%); the policy allows one change of at most 25%, a run's impact up to a budget, and a person approves an impact above an
amount. The agent starts in shadow; a written person judges each proposal (right when the discount keeps a margin and stays within the
limit) and the track record moves the agent up a level after its run of right calls, down on a wrong one. ``--write`` stores the record in
``evidence/campaigns/department_v1/results/DPA-<n>_pricing.json`` and every step whole beside it (``runs/DPA-<n>_steps.jsonl.gz``).
Invented data: it says the guardrails hold, not that a discount is wise.
"""

from __future__ import annotations

import gzip
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "evidence" / "campaigns" / "department_v1"


def products(seed: int = 0, n: int = 2_000) -> list[dict[str, Any]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    cost = np.round(rng.uniform(5, 200, n), 2)
    price = np.round(cost * rng.uniform(1.1, 2.2, n), 2)
    weeks = rng.integers(1, 60, n)
    units = np.round(rng.gamma(2.0, 6.0, n), 1)
    return [{"sku": f"P{i:05d}", "cost": float(cost[i]), "price": float(price[i]), "weeks": int(weeks[i]), "units": float(units[i])} for i in range(n)]


STYLES = {"careful": ((50, 0.2), (35, 0.15), (20, 0.1)), "aggressive": ((50, 0.4), (35, 0.3), (20, 0.1))}  # weeks in stock -> discount


def propose(item: dict[str, Any], style: str = "aggressive") -> Optional[dict[str, Any]]:
    """A rule agent: a longer time in stock, a deeper discount (the careful one stays within what a person allows)."""
    discount = next((d for weeks, d in STYLES[style] if item["weeks"] > weeks), 0.0)
    if not discount:
        return None
    after = round(item["price"] * (1 - discount), 2)
    if style == "careful" and after <= item["cost"] * 1.05:
        return None  # the careful agent keeps a margin of its own
    return {"action": "set_price", "target": item["sku"], "before": item["price"], "after": after,
            "impact": round((item["price"] - after) * item["units"], 2), "reason": f"{item['weeks']} weeks in stock: a {discount:.0%} discount"}


def person_says(item: dict[str, Any], proposal: dict[str, Any], max_change: float) -> bool:
    """A written person's judgment: right when the price still covers the cost and the change is within what they would allow."""
    return proposal["after"] > item["cost"] and (proposal["before"] - proposal["after"]) / proposal["before"] <= max_change + 1e-9


def run(seed: int = 0, promote_after: int = 50, audit_log: Any = None) -> dict[str, Any]:
    """Both agents on the same products: what each proposed, what the policy let through, and the levels each earned; every step in
    the installed ledger and in ``audit_log`` when given."""
    agents = {style: one(seed, style, promote_after, audit_log) for style in STYLES}
    return {"products": agents["careful"]["products"], "agents": {k: {x: v for x, v in a.items() if x != "steps"} for k, a in agents.items()},
            "steps": [{"agent": k, **s} for k, a in agents.items() for s in a["steps"]],
            "limits": ["Invented products and a written person: the record says the guardrails hold (nothing past a limit or above the "
                       "approval amount applied, every step undoable), not that any discount is wise."]}


def one(seed: int, style: str, promote_after: int, audit_log: Any = None) -> dict[str, Any]:
    from dclab_core import department_agent as da

    items = products(seed)
    policy = da.Policy(name="pricing", actions=["set_price"], field="price", max_change=0.25, budget=60_000.0, approval_above=1_500.0,
                       level="shadow", promote_after=promote_after)
    state = {p["sku"]: p["price"] for p in items}
    units = {p["sku"]: p["units"] for p in items}
    agent = da.Agent(policy, state, impact_of=lambda sku, before, after: round((before - after) * units[sku], 2))  # the host reckons it
    record, moves = da.TrackRecord(), []
    by_sku = {p["sku"]: p for p in items}
    for item in items:
        raw = propose(item, style)
        if raw is None:
            continue
        step = agent.consider(da.Proposal(**raw))
        right = person_says(by_sku[raw["target"]], raw, 0.25)
        record = da.judge(record, right)
        agent.steps[-1] = step.model_copy(update={"judged": right})
        _ledger(agent.steps[-1], right, f"pricing_{style}", audit_log)
        level, kept = da.level_for(agent.policy, record, right)
        if level != agent.policy.level:
            moves.append({"step": step.id, "from": agent.policy.level, "to": level, "right_in_a_row": record.right_in_a_row})
            agent.policy = agent.policy.model_copy(update={"level": level})
        record = kept
    steps = agent.steps
    past_limit = [s for s in steps if not s.verdict.allowed]
    return {"policy": policy.model_dump(mode="json"), "products": len(items), "proposals": len(steps),
            "applied": sum(s.applied for s in steps), "flagged": sum(s.flagged for s in steps), "refused": len(past_limit),
            "refused_applied": sum(s.applied for s in past_limit), "needs_approval": sum(s.verdict.needs_approval for s in steps),
            "approval_applied": sum(s.applied for s in steps if s.verdict.needs_approval), "spent": round(agent.spent, 2),
            "budget": policy.budget, "levels": moves, "final_level": agent.policy.level,
            "by_level": dict(Counter(s.level for s in steps)), "steps": [s.model_dump(mode="json") for s in steps]}


def _ledger(step: Any, right: bool, agent: str, audit_log: Any = None) -> None:
    """Each step in the workspace's decision ledger (the person's judgment its label, when one is installed) and the audit trail."""
    from . import audit
    from .ledger import capture

    capture.department_proposal(agent, step, right, person="written_person")
    if audit_log is not None:
        audit.record(audit_log, "department", "agent", agent, move=step.proposal.action, status="applied" if step.applied else
                     "flagged" if step.flagged else "refused" if not step.verdict.allowed else "shadow",
                     args={"step": step.id, "level": step.level, "needs_approval": step.verdict.needs_approval, "judged_right": right},
                     message="; ".join(step.verdict.why)[:600] or "allowed")


def write(doc: dict[str, Any], campaign: Path = CAMPAIGN) -> Path:
    results, runs = campaign / "results", campaign / "runs"
    results.mkdir(parents=True, exist_ok=True)
    runs.mkdir(parents=True, exist_ok=True)
    n = 1 + max([int(p.name.split("_")[0].split("-")[1]) for p in results.glob("DPA-*.json")] or [0])
    ident = f"DPA-{n:03d}"
    path, kept = results / f"{ident}_pricing.json", runs / f"{ident}_steps.jsonl.gz"
    if path.exists() or kept.exists():
        raise SystemExit(f"{ident} exists: results are never overwritten")
    with gzip.open(kept, "wt", encoding="utf-8") as fh:
        for step in doc["steps"]:
            fh.write(json.dumps(step, ensure_ascii=False) + "\n")
    path.write_text(json.dumps({"id": ident, "experiment_id": ident, "campaign_id": "department_v1", "kind": "department_agent", "status": "completed",
                                "at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **{k: v for k, v in doc.items() if k != "steps"},
                                "detail": f"runs/{kept.name}"}, indent=1) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.department", description=__doc__.split("\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    doc = run(args.seed)
    print(json.dumps({k: {x: v for x, v in a.items() if x != "policy"} for k, a in doc["agents"].items()}, indent=1))
    if args.write:
        print(write(doc))
    return 0


if __name__ == "__main__":
    sys.exit(main())
