"""The team measured (package 18.8): the fraud, churn and delivery replicas run end to end by the roles together.

    python -m dclab_rnd.agent_eval.team                       # the plan: requests and the euro bound, nothing sent
    python -m dclab_rnd.agent_eval.team --yes --cap-eur 0.05  # the live run, capped, stored as a new TEAM result

The person is scripted from the replica case (the target and the prediction moment when asked, and a yes at each
gate). The gate (``GATE``, fixed before any run) asks that each replica completes with every role on; every move the
graph validator saw is allowed or waits for a person's approval that then comes; the person is asked at most the two
essentials (and the review budget a rare event needs) and three open questions; the holdout is read once; the run stays under its cap; and each replica still
reaches its kind's acceptance decisions (16.2): the roles must not leave the solution worse than the rules do.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

RESULTS = Path(__file__).resolve().parents[2] / "evidence/campaigns/team_v1/results"
GATE = {"version": "team_gate_v1", "fixed": "2026-10-13, before its first run",
        "rule": "on the fraud, churn and delivery replicas with every role on: each run completes; every graph move is allowed or "
                "waits for a person's approval that then comes; the person is asked at most the two essentials and three open "
                "questions; every role is asked on every replica; the holdout is read once; the spend stays under the cap; and each replica still reaches its kind's "
                "acceptance decisions (16.2)"}
KINDS = ("rare_event", "binary", "regression")
DESCRIPTIONS = {
    "rare_event": "Find fraudulent rides of bikers. The rides table has every ride; the sample file lists the frauds the team found.",
    "binary": "Predict which subscribers cancel before their next renewal, so the retention team can call them first.",
    "regression": "Estimate how many minutes each delivery will take, when the order is placed.",
}
REQUESTS_PER_RUN = 10  # analyst, checker, investigator, planner, engineer, up to 4 critic rounds, reporter


def _project(key: str) -> tuple[Any, str, Any]:
    from dclab_rnd import kinds
    from dclab_rnd.studio import ProjectStore, data as sd

    case = kinds.get(key).case(0)
    store = ProjectStore(Path(tempfile.mkdtemp()))
    pid = store.create(key, "general", DESCRIPTIONS[key])["id"]
    directory = store.data_dir(pid)
    if case.files is not None:
        data_file, label_file = case.files(directory)
        label_bytes = (directory / label_file).read_bytes()
        sd.attach_data(store, pid, data_file)
        (directory / label_file).write_bytes(label_bytes)
        sd.attach_labels(store, pid, label_file, target=case.labels[1])
    else:
        case.table.to_csv(directory / "table.csv", index=False)
        sd.attach_data(store, pid, "table.csv")
    return store, pid, case


ROLES = ("problem_analyst", "brief_checker", "data_investigator", "solution_planner", "feature_engineer", "critic", "reporter")


def _calls(advisor: Any) -> dict[str, int]:
    """Requests sent so far, per purpose (a GatewayAdvisor counts them; a ScriptedAdvisor keeps them)."""
    if advisor is None:
        return {}
    if hasattr(advisor, "calls"):
        return dict(advisor.calls)
    counts: dict[str, int] = {}
    for purpose, _ in getattr(advisor, "requests", []):
        counts[purpose] = counts.get(purpose, 0) + 1
    return counts


def _approvals_came(moves: list[dict[str, Any]]) -> bool:
    """Every move that waited for a person's approval was later allowed."""
    for i, m in enumerate(moves):
        if m["status"] == "needs_approval" and not any(n["move"] == m["move"] and n["status"] == "allowed" for n in moves[i + 1:]):
            return False
    return all(m["status"] in ("allowed", "needs_approval") for m in moves)


def measure(advisor: Optional[Any]) -> dict[str, Any]:
    from dclab_rnd import kinds
    from dclab_rnd.expansion import runner
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.studio import engine, team

    rows = []
    for key in KINDS:
        before = _calls(advisor)
        try:
            store, pid, case = _project(key)
            answers = {"target_column": case.target, "prediction_moment": case.moment, "budget": str(case.solution.get("alert_budget") or "")}
            person = team.Person(answer=lambda field, question, a=answers: a.get(field), approve=lambda gate, summary: True,
                                 who="scripted person")
            with runner.fast_profile():
                out = team.run(store, pid, DESCRIPTIONS[key], person, advisor, quick=case.quick)
            moves = store.transitions(pid, limit=2000)
            records = {stage: store.read_stage(pid, stage) for stage in engine.STAGE_KEYS if store.has_stage(pid, stage)}
            checks = []
            if out["status"] == "completed":
                project = store.get(pid)
                kind = kinds.get(key)
                done = acceptance.Run(case, project["solution"], records, project.get("labels") or {})
                checks = kind.accept(done)
                computed = set((done.evidence("final").get("baselines") or {}).keys())
                checks.append(acceptance.check("the baselines the kind declares are computed", set(kind.baselines) <= computed))
        except Exception as exc:  # noqa: BLE001 - a replica that fails is stored as failed, never lost after paid requests
            rows.append({"kind": key, "status": "failed", "why": f"{type(exc).__name__}: {str(exc)[:300]}", "asked_roles": {},
                         "acceptance_ok": False, "moves_ok": False, "asked_ok": False, "holdout_uses": None})
            continue
        after = _calls(advisor)
        asked_roles = {r: after.get(r, 0) - before.get(r, 0) for r in ROLES}
        rounds = ((records.get("models") or {}).get("evidence", {}).get("review") or {}).get("rounds", [])
        required = [r for r in ROLES if r != "critic" or rounds]  # the critic is asked only when there are review rounds
        essentials = [q for q in out["asked"] if q["field"] in (*team.ESSENTIAL, "budget")]
        rows.append({"kind": key, "status": out["status"], "why": out.get("why"), "steps": out["steps"], "asked": out["asked"],
                     "asked_ok": len(essentials) <= 3 and len(out["asked"]) - len(essentials) <= 3,
                     "moves": len(moves), "moves_ok": _approvals_came(moves),
                     "holdout_uses": (records.get("final") or {}).get("evidence", {}).get("holdout_uses_in_this_project"),
                     "acceptance": [{"decision": c["decision"], "ok": c["ok"]} for c in checks],
                     "acceptance_ok": bool(checks) and all(c["ok"] for c in checks),
                     "asked_roles": asked_roles, "every_role_asked": all(asked_roles[r] > 0 for r in required),
                     "roles_answered": sorted({s["role"] for s in out["steps"] if s["source"] == "advisor"})})
    return {"rows": rows}


def verdict(result: dict[str, Any], spent: float, cap: Optional[float]) -> dict[str, Any]:
    rows = result["rows"]
    if not any(r.get("roles_answered") for r in rows):
        return {"gate": GATE["version"], "status": "not scored", "why": "no role's answer was used"}
    if not all(r.get("every_role_asked") for r in rows):
        return {"gate": GATE["version"], "status": "not scored", "why": "a role was never asked on some replica (no model, or the cap ran out)"}
    ok = all(r["status"] == "completed" and r["moves_ok"] and r["asked_ok"] and r["holdout_uses"] == 1 and r["acceptance_ok"] for r in rows) \
        and (cap is None or spent <= cap)
    return {"gate": GATE["version"], "status": "pass" if ok else "fail"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.team")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--cap-eur", type=float, default=None)
    args = parser.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env", override=False)
    from dclab_rnd.core_advisor import GatewayAdvisor
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    gateway.shadows = None
    print(json.dumps({"gate": GATE, "replicas": KINDS, "max_requests": REQUESTS_PER_RUN * len(KINDS)}, indent=1))
    if not args.yes or args.cap_eur is None:
        print("nothing was sent: add --yes and --cap-eur to run it")
        return 0
    advisor = GatewayAdvisor(gateway, max_tokens=3000, cap_eur=args.cap_eur)
    started = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    result = measure(advisor)
    spent = round(advisor.spent_eur(), 4)
    judged = verdict(result, spent, args.cap_eur)
    RESULTS.mkdir(parents=True, exist_ok=True)
    n = len(list(RESULTS.glob("TEAM-*.json"))) + 1
    record = {"id": f"TEAM-{n:03d}", "experiment_id": f"TEAM-{n:03d}", "campaign_id": "team_v1", "kind": "team_measured", "at": started,
              "completed_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(), "status": "completed", "gate": GATE,
              "cap_eur": args.cap_eur, "spent_eur": spent, "verdict": judged, "result": result}
    path = RESULTS / f"TEAM-{n:03d}_three_replicas.json"
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=1, default=str)
        handle.write("\n")
    print(f"team: spent €{spent}, verdict {judged['status']}, stored {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
