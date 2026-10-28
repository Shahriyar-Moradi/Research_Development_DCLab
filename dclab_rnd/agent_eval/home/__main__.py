"""Measure the Home agent (package H1).

    python -m dclab_rnd.agent_eval.home --split dev                   # the rules alone, printed, and the plan of a model run
    python -m dclab_rnd.agent_eval.home --split dev --save            # the same, stored as the baseline (no model, nothing sent)
    python -m dclab_rnd.agent_eval.home --split dev --yes --cap-eur 2    # sends: Home with the model, and the plain arm
    python -m dclab_rnd.agent_eval.home --split sealed_v2 --yes --cap-eur 2 # once: a second run of a sealed set is refused
                                                                         # ("sealed" was spent by HOME-011)

A stored run is evidence/campaigns/home_v1/results/HOME-NNN_<split>.json with its plan, its cost and the gate's verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import RESULTS, SEALED, SPLITS, gate_for, load, measure, next_id, plan, sealed_runs, verdict

MAX_TOKENS = 1800


class Budget:
    """One euro cap over every client the run makes (one per draft and one per plain request): each new client may spend
    only what the earlier ones left, so the run as a whole stops at the cap."""

    def __init__(self, gateway: Any, cap_eur: float | None):
        self.gateway, self.cap_eur, self.clients = gateway, cap_eur, []

    def spent_eur(self) -> float:
        return sum(float(getattr(c, "spent_eur", 0.0) or 0.0) for c in self.clients)

    def client(self, draft_id: str) -> Any:
        client = self.gateway.client("home_agent", draft_id=draft_id)
        if client is not None and self.cap_eur is not None:
            client.run_limit_eur = max(0.0, self.cap_eur - self.spent_eur())
        if client is not None:
            self.clients.append(client)
        return client


def _store(record: dict[str, Any], split: str, scored: bool) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{record['id']}_{split}{'' if scored else '_unscored'}.json"
    path.write_text(json.dumps(record, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _shown(path: Path) -> str:
    from . import ROOT

    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


KEEP_ROOT = Path(__file__).resolve().parents[3] / "agent_runs" / "live_tests"  # the store's "tests" keeper reads it


def _print(summary: dict[str, Any]) -> None:
    for arm, s in summary.items():
        print(f"{arm:6s} track {s['track_accuracy']:.3f} ({s['track']['right']}/{s['cases']}) by language {s['track_accuracy_by_lang']}  "
              f"fields {s['fields']}  exit {s['exit_right']}/{s['cases']}  median questions {s['median_questions']}  "
              f"traps passed {sum(bool(v) for v in s['traps'].values())}/{len(s['traps'])}  "
              f"uncited {s['uncited']} quotes {s['quote_invalid']} rows {s['rows_exposed']} phantom {s['phantom']} lost to not-sure {s['lost']}")


def _record(result: dict[str, Any], split: str, the_plan: dict[str, Any] | None, spent: float, scored: bool) -> dict[str, Any]:
    at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    gate = gate_for(split)
    record = {"id": next_id(), "campaign_id": "home_v1", "kind": "home_agent_measured", "split": split, "at": at, "gate": gate,
              "plan": the_plan, "spent_eur": round(spent, 4), **result,
              "verdict": verdict(result, split) if scored else {"gate": gate["version"], "status": "not scored",
                                                                "why": f"{len(result['failures'])} request(s) failed"},
              "rule": "Scores on 36 written requests say how Home handles these requests; they are not a claim about any other "
                      "text or about production readiness."}
    record.update(experiment_id=record["id"], status="completed" if scored else "failed", completed_at=at)
    return record


def main(argv: list[str] | None = None) -> int:
    from dclab_rnd.memory.reading import memory_off_for_evaluation

    memory_off_for_evaluation()  # 35.4: a graded run never reads a remembered line
    parser = argparse.ArgumentParser(prog="python -m dclab_rnd.agent_eval.home")
    parser.add_argument("--split", choices=tuple(SPLITS), default="dev")
    parser.add_argument("--lang", choices=("en", "fa", "mixed"), default=None,
                        help="dev only: run the requests of one language (the owner, 2026-10-28: live runs in English only)")
    parser.add_argument("--yes", action="store_true", help="send the requests (the plan is printed either way)")
    parser.add_argument("--cap-eur", type=float, default=None, help="required with --yes for a priced model")
    parser.add_argument("--save", action="store_true", help="dev only: store the rules-alone run as evidence (nothing is sent)")
    args = parser.parse_args(argv)
    from dclab_rnd import experience as _experience

    _experience.sealed_run(getattr(args, "split", None))  # an exam's model calls keep their shape only (the owner's rule, 2026-10-28)
    from dotenv import load_dotenv  # the configured model is in .env, as for the server

    load_dotenv(Path(__file__).resolve().parents[3] / ".env", override=False)
    cases = load(args.split)
    if args.lang:
        if args.split in SEALED:
            parser.error("--lang is for dev: the sealed set is run once, whole")
        cases = [c for c in cases if c.get("lang") == args.lang]
    from dclab_rnd.models.gateway import for_workspace

    gateway = for_workspace()
    the_plan = plan(gateway, cases, MAX_TOKENS)
    if not args.yes:
        if args.split == "dev":  # the sealed set's scores are read only in its one run
            result = measure(cases)
            _print(result["summary"])
            print(json.dumps(verdict(result, "dev"), indent=1))
            if args.save:
                path = _store(_record(result, "dev", None, 0.0, True), "dev", True)
                print(f"stored {_shown(path)}")
        print(json.dumps(the_plan, indent=1))
        print("nothing was sent: add --yes and --cap-eur to run Home with the model")
        return 0
    if args.split in SEALED and sealed_runs(args.split):
        print(f"refused: the sealed set was already run ({sealed_runs(args.split)[0].name}); it is run once", file=sys.stderr)
        return 2
    if not the_plan["available"]:
        print("refused: no model serves home_agent here", file=sys.stderr)
        return 2
    if not the_plan["local"] and (the_plan["max_eur"] is None or args.cap_eur is None or args.cap_eur <= 0):
        print("refused: a remote model needs a price and --cap-eur above zero", file=sys.stderr)
        return 2
    gateway.shadows = None  # measurement traffic stays out of the shadow log
    budget = Budget(gateway, args.cap_eur)
    keep = None if args.split in SEALED else \
        KEEP_ROOT / f"{datetime.now(timezone.utc):%Y-%m-%d}-home-eval" / f"{args.split}-{datetime.now(timezone.utc):%H%M%S}"  # a sealed run is an exam: never kept
    result = measure(cases, budget.client, lambda case_id: budget.client("home-eval-plain"), keep=keep)
    result["kept"] = _shown(keep) if keep else None  # the run's drafts and replies, whole (agent_runs is the store's, never committed)
    scored = not result["failures"]
    if not scored:  # a failed request: kept as a record of what it cost, without values or scores, and never a sealed run
        result = {"arms": result["arms"], "requests": result["requests"], "failures": result["failures"], "summary": {},
                  "kept": result.get("kept")}
    path = _store(_record(result, args.split, the_plan, budget.spent_eur(), scored), args.split, scored)
    if scored:
        _print(result["summary"])
    print(f"spent €{budget.spent_eur():.4f}; stored {_shown(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
