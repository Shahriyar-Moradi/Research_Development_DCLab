"""The decision core through the engine on the delivery-fraud replica (campaign ``fraud_decision_v1``; 34.1 and 34.4).

34.1 measured its intervals on simulated scores (the prediction interval held the holdout 54 of 60 times); the fraud
replica was never run through the engine with an ``act`` decision. Here each seed's replica (its rides, the partial
sample of known frauds joined by the engine, rows grouped by biker) runs the five stages with a decision stated as a
person would: review a ride, a fraud caught saves the ride's fare (``biker_fare``, known at the daily audit), a review
costs 25,000 (a tenth of a typical fare, so the cost can move the choice), at most 0.5% of the rides reviewed (the
team's budget). The final stage chooses on the out-of-fold scores of
the training rows and reads the holdout once; the record says what it realised there and the interval it expected.

The gate, fixed in ``manifest.json`` before the first run (a changed gate, budget or code is a new campaign version):

* the decision is chosen on every seed;
* it never acts on more of the holdout than its capacity allows;
* the holdout's realised value per ride lies inside the 90% prediction interval the choice gave, on at least 6 of 8
  seeds (a 90% interval holds about 7 of 8; 6 of 8 or more happens 96 times in 100 when it is honest).

The labels are partial (a sample of the frauds), so a fraud nobody found counts as a clean ride in both the choice and
the holdout: the realised value is a lower bound, said in the record.

    python -m dclab_rnd.decision_measure plan
    python -m dclab_rnd.decision_measure run --yes     # about 2 minutes a seed on the 8 GB Air; resumable on the same code
    python -m dclab_rnd.decision_measure report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "fraud_decision_v1"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
WORK = ROOT / "agent_runs" / CAMPAIGN_ID  # each seed's workspace kept whole: the store reads its sessions, ledger and runs
SEEDS = tuple(range(8))
DECISION = {"action": "act", "what": "review the ride", "value_column": "biker_fare", "cost_per_action": 25000.0, "capacity_share": 0.005,
            "notes": "a fraud caught saves the ride's fare; a review costs 25,000 (a tenth of a typical fare); the team reviews at most 0.5% of the rides"}
GATE = {
    "id": "fraud_decision_gate_v1",
    "chosen": "the decision is chosen on every seed (an operating point with a share, no refusal)",
    "capacity": "the holdout's acted share never exceeds the capacity share (with one row's rounding)",
    "coverage": "the holdout's realised value per ride lies inside the choice's 90% prediction interval on at least 6 of 8 seeds",
}
CODE_FILES = ("dclab_rnd/decision_measure.py", "dclab_rnd/campaign_capture.py", "dclab_rnd/ledger", "dclab_rnd/experience")  # with the engine's folders


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256() -> str:
    from .campaign_capture import ENGINE_CODE, code_sha256 as whole

    return whole((*ENGINE_CODE, *CODE_FILES))  # every Python file that makes the numbers (the review: not five of them)


CODE_AT_START = code_sha256()


def manifest() -> dict[str, Any]:
    body = {"campaign": CAMPAIGN_ID, "package": "34.1 and 34.4", "seeds": list(SEEDS), "decision": DECISION, "gate": GATE,
            "objective": ("Does the decision core keep its word through the engine on the delivery-fraud replica? A stated review "
                          "decision, chosen on out-of-fold scores, against what it realises on the holdout read once, over eight seeds.")}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def freeze() -> dict[str, Any]:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / "manifest.json"
    want = manifest()
    if path.exists():
        held = json.loads(path.read_text())
        if held.get("sha256") != want["sha256"]:
            raise SystemExit("The gate, the seeds or the decision changed since the campaign was frozen: start a new campaign version")
        if held.get("code_sha256") != CODE_AT_START:
            raise SystemExit("The code that makes the numbers changed since the campaign was frozen: start a new campaign version")
        return held
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, timeout=10).stdout.strip() or None
    except Exception:  # noqa: BLE001
        commit = None
    held = {**want, "code_sha256": CODE_AT_START, "code_files": list(CODE_FILES), "git_commit": commit, "frozen_at": _now()}
    path.write_text(json.dumps(held, indent=2) + "\n")
    return held


def run_one(seed: int) -> dict[str, Any]:
    from unittest import mock

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    case = kinds.get("rare_event").case(seed)
    if case is None:
        raise ValueError("the delivery-fraud replica is not part of this edition")
    case.solution = {**case.solution, "decision": DECISION}
    started = time.perf_counter()
    work = WORK / f"seed{seed}"
    if work.exists():
        import shutil

        shutil.rmtree(work)  # this campaign's own workspace for the seed, made again
    work.mkdir(parents=True)
    from .campaign_capture import recorded
    from .ledger import labels

    with recorded(work), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None):
        store, pid = acceptance.build(case, work)  # inside: the build's own decisions (target, roles, split) reach the ledger
        engine.run_all(store, pid)  # its ledger and session log kept in the seed's folder; the rules decide
        labels.from_replica(pid, case.facts, "delivery_fraud")  # 19.8: the planted traps label the decisions
    final = store.read_stage(pid, "final") or {}
    uses = int(store.get(pid).get("holdout_uses") or 0)
    d = (final.get("evidence") or {}).get("decision") or {}
    h = d.get("holdout") or {}
    interval = h.get("prediction_interval") or [None, None]
    realised = h.get("expected_value_per_row")
    rows = h.get("rows") or (final.get("evidence") or {}).get("holdout_rows")
    acted = h.get("actions")
    inside = None if realised is None or None in interval else bool(interval[0] <= realised <= interval[1])
    within = None if acted is None or not rows else bool(acted <= DECISION["capacity_share"] * rows + 1)
    number = SEEDS.index(seed) + 1
    result = {"id": f"FDC-{number:03d}", "experiment_id": f"FDC-{number:03d}", "status": "completed", "completed_at": _now(),
              "campaign_id": CAMPAIGN_ID, "dataset": "delivery_fraud", "seed": seed, "decision_status": d.get("decision"), "warning": d.get("warning"),
              "chosen_share": d.get("share"), "ranked_by": d.get("ranked_by"), "expected_value_per_row": d.get("expected_value_per_row"),
              "expected_interval": d.get("expected_value_per_row_interval"), "holdout_rows": rows, "holdout_actions": acted,
              "holdout_cases_found": h.get("cases_found"), "holdout_value_per_row": realised, "prediction_interval": interval,
              "inside_prediction_interval": inside, "within_capacity": within, "holdout_uses": uses,
              "model": (final.get("evidence") or {}).get("model"), "seconds": round(time.perf_counter() - started, 1),
              "code_sha256": CODE_AT_START, "applied_as": d.get("applied_as"), "capacity_binds": d.get("capacity_binds"),
              "limits": ["labels partial: a fraud nobody found counts as a clean ride, so the realised value is a lower bound",
                         "where the capacity binds, the choice is the capacity itself: the 'chosen' gate then tests that it was read, not the value"]}
    out = CAMPAIGN / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{result['id']}_seed{seed}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    chosen = all(r.get("decision_status") == "operating_point" and r.get("chosen_share") is not None and not r.get("warning") for r in rows)
    capacity = all(r.get("within_capacity") is True for r in rows)
    inside = sum(1 for r in rows if r.get("inside_prediction_interval") is True)
    gates = {"chosen": chosen, "capacity": capacity, "coverage": inside >= 6}
    complete = len(rows) == len(SEEDS)
    return {"seeds": len(rows), "of": len(SEEDS), "gates": gates, "inside_prediction_interval": inside,
            "verdict": ("pass" if all(gates.values()) else "fail") if complete else f"not yet: {len(rows)} of {len(SEEDS)} seeds",
            "realised": [r.get("holdout_value_per_row") for r in rows], "intervals": [r.get("prediction_interval") for r in rows], "gate": GATE}


def report() -> dict[str, Any]:
    folder = CAMPAIGN / "results"
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("FDC-0*.json"))] if folder.exists() else []
    out = judge(rows)
    if len(rows) == len(SEEDS) and not (folder / "FDC-100_decision.json").exists():  # written once: a result is never rewritten
        (folder / "FDC-100_decision.json").write_text(json.dumps({**out, "id": "FDC-100", "experiment_id": "FDC-100", "status": "completed",
                                                                  "completed_at": _now(), "campaign_id": CAMPAIGN_ID}, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="34.1/34.4: the decision core through the engine on the delivery-fraud replica")
    parser.add_argument("command", choices=("plan", "run", "report"))
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "plan" or (args.command == "run" and not args.yes):
        print(json.dumps({**manifest(), "code_sha256": CODE_AT_START}, indent=2))
        if args.command == "run":
            print("Add --yes to run it.")
        return 0
    if args.command == "report":
        print(json.dumps(report(), indent=2))
        return 0
    freeze()
    done = {int(p.stem.split("seed")[1]) for p in (CAMPAIGN / "results").glob("FDC-0*.json")} if (CAMPAIGN / "results").exists() else set()
    for seed in SEEDS:
        if seed in done:
            continue
        print(f"[fraud-decision] seed {seed} {_now()}", flush=True)
        try:
            r = run_one(seed)
            print(f"[fraud-decision] seed {seed}: {r['decision_status']} share {r['chosen_share']} realised {r['holdout_value_per_row']} "
                  f"in {r['prediction_interval']}: {r['inside_prediction_interval']}; capacity kept: {r['within_capacity']}", flush=True)
        except Exception as exc:  # noqa: BLE001 - a seed that fails is said, the rest still run
            print(f"[fraud-decision] seed {seed} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(json.dumps(report(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
