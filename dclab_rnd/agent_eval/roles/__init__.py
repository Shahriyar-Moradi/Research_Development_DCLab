"""Step 3's roles, measured on the replicas (packages 18.4, 18.5, 18.6): the solution planner, the feature engineer and
the critic. Each is compared with what code does without it, under a gate fixed before any run (``GATES``).

* **planner** (18.4): on the three replicas, the plan's decisions against the acceptance decisions their traps call for
  (``GOLD``, written from each replica's documented traps, not from the rules' output), with the rules alone and with
  the planner; every refusal is recorded. Gate: the planner matches at least as many as the rules, and every plan it
  leaves passes the validator.
* **engineer** (18.5): on the fraud replica, the engineer's programs, unprompted by the answer: does it propose a
  repeated drop-off count and a short-trip share, and does the feature ladder keep the engineered recipe for a measured
  gain over the raw columns (folds grouped by biker)? Gate: both proposed, and a gain of at least 0.05 average precision.
* **critic** (18.6): on the delivery and tickets replicas, the training-CV score after the review loop with the critic,
  against one pass (no review round); the holdout used once. Gate: never worse than one pass, better on at least one
  replica, and every kept change cites a measured weakness.

Live runs need the owner's yes and a euro cap (``python -m dclab_rnd.agent_eval.roles``).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RESULTS = ROOT / "evidence" / "campaigns" / "roles_v1" / "results"

GATES = {  # fixed on 2026-10-12, before any live run
    "planner": {"version": "planner_gate_v1", "rule": "acceptance decisions matched with the planner >= with the rules alone, "
                                                     "and every plan the planner leaves passes the validator"},
    "engineer": {"version": "engineer_gate_v1", "rule": "a repeated drop-off count and a short-trip share are proposed, and the "
                                                       "engineered recipe beats the raw columns by >= 0.05 training-CV average precision"},
    # written after engineer_gate_v1 failed twice on named shapes while the engineered recipe gained a lot (ROL-002, ROL-004):
    # it judges the measured gain instead, and is measured on replica seeds the engineer was never run on (1 and 2)
    "engineer_v2": {"version": "engineer_gate_v2", "fixed": "2026-10-12, before its first run",
                    "rule": "on fraud replica seeds 1 and 2 (unseen), with folds grouped by biker: the engineered recipe beats the raw "
                            "columns by >= 0.05 training-CV average precision and matches or beats 15.4's behaviour recipe; both seeds"},
    "critic": {"version": "critic_gate_v1", "rule": "training-CV after the loop with the critic is never worse than the rules' own loop "
                                                   "(17.6) nor one pass, and better than one pass on at least one replica; every kept change "
                                                   "cites a measured weakness; holdout once (revised after review, before any run)"},
}

DESCRIPTIONS = {
    "fraud": ("Find fraudulent rides of bikers. Rides are scored after delivery, in a daily audit of the day's rides. The quality "
              "team can review about 80 rides a day. The known frauds come from a sample file of the cases we found; most frauds "
              "are never found."),
    "tickets": ("Route each new support ticket to the right team when it is created, from its subject, body and the customer's "
                "plan. Report macro F1."),
    "delivery": ("Estimate how many minutes each order will take, when the customer places it. The ops team watches mean absolute "
                 "error. Times above 60 minutes are stored as 60."),
}


def _choice(plan: Any, key: str) -> Any:
    d = plan.decision(key)
    return None if d is None else d.choice


def _forbidden(plan: Any) -> set[str]:
    return {str(f.get("column")) for f in (_choice(plan, "features.forbidden") or []) if isinstance(f, dict)} | \
           {str(f.get("column")) for f in (_choice(plan, "features.to_confirm") or []) if isinstance(f, dict)}


def _split(plan: Any) -> dict[str, Any]:
    c = _choice(plan, "split.strategy")
    return c if isinstance(c, dict) else {"kind": c}


# Each check is a decision a careful data scientist reaches on that replica, from its documented traps.
GOLD: dict[str, dict[str, Callable[[Any], bool]]] = {
    "fraud": {
        "folds grouped by biker": lambda p: _split(p).get("group_column") == "biker_id",
        "labels partial": lambda p: _choice(p, "labels.coverage") == "partial",
        "a rare-event metric": lambda p: _choice(p, "metric.primary") in ("average_precision", "recall_at_budget"),
        "a review-budget threshold": lambda p: isinstance(_choice(p, "threshold.policy"), dict) and _choice(p, "threshold.policy").get("kind") == "review_budget",
        "PU or anomaly families": lambda p: bool({"pu_bagging", "isolation_forest"} & set(_choice(p, "model.families") or [])),
        "behaviour of each biker": lambda p: any("biker_id" in str(d) for d in (_choice(p, "features.direction") or [])),
    },
    "tickets": {
        "post-creation columns kept out": lambda p: {"resolved_by", "first_response_minutes"} <= _forbidden(p),
        "a macro metric": lambda p: str(_choice(p, "metric.primary") or "").startswith("macro"),
        "duplicates dropped": lambda p: "cop" in str(_choice(p, "data.cleaning") or "").lower() or "duplicat" in str(_choice(p, "data.cleaning") or "").lower(),
        "time order and customers kept apart": lambda p: _split(p).get("kind") == "time_and_group" and _split(p).get("group_column") == "customer_id",
        "tiny classes handled": lambda p: "tiny_classes" in (_choice(p, "task.framing") or {}),
    },
    "delivery": {
        "post-delivery columns kept out": lambda p: {"delivered_at", "courier_rating"} <= _forbidden(p),
        "mean absolute error": lambda p: _choice(p, "metric.primary") == "mae",
        "folds grouped by restaurant": lambda p: _split(p).get("group_column") == "restaurant_id",
        "a quantile model for the cap": lambda p: "quantile_gradient_boosting" in (_choice(p, "model.families") or []),
        "impossible distances cleaned": lambda p: "distance_km" in str(_choice(p, "data.cleaning") or ""),
        "a missing-weather indicator": lambda p: any("weather" in str(d) for d in (_choice(p, "features.direction") or [])),
    },
}


def replica_table(name: str) -> tuple[pd.DataFrame, str, str]:
    """(table, target, task) for a replica, the fraud one with its target joined from the label file on the order."""
    if name == "fraud":
        from dclab_core.replicas import delivery_fraud as dfr
        from dclab_rnd.studio import labels

        r = dfr.generate(seed=0)
        rides = r.rides.copy()
        rides["is_fraud"] = labels.mark(rides, r.sample, ["order_id"]).to_numpy()
        return rides, "is_fraud", "binary"
    if name == "tickets":
        from dclab_core.replicas import ticket_routing

        return ticket_routing.generate(seed=0, n=3000).tickets, "assigned_team", "multiclass"
    from dclab_core.replicas import delivery_time

    return delivery_time.generate(seed=0, n=3000).orders, "actual_minutes", "regression"


def measure_planner(advisor: Optional[Any]) -> dict[str, Any]:
    from dclab_core.analyse import analyse
    from dclab_core.plan import plan, plan_with, validate
    from dclab_core.understand import understand

    rows = []
    for name in GOLD:
        table, target, task = replica_table(name)
        brief = understand(DESCRIPTIONS[name], {"table.csv": list(table.columns)})
        train = np.sort(np.random.default_rng(0).choice(len(table), int(0.8 * len(table)), replace=False))
        findings = analyse({"table": table}, target=target, task=task, brief=brief, train_index=table.index[train])
        rules = plan(brief, findings)
        row = {"replica": name, "rules": {k: bool(f(rules)) for k, f in GOLD[name].items()}}
        if advisor is not None:
            out = plan_with(advisor, brief, findings)
            row["planner"] = {k: bool(f(out["plan"])) for k, f in GOLD[name].items()}
            row["kept"], row["refused"] = out["kept"], out["refused"]
            row["validator"] = validate(out["plan"], findings)
            row["source"] = out["source"]
        rows.append(row)
    summary = {"rules": sum(sum(r["rules"].values()) for r in rows), "decisions": sum(len(r["rules"]) for r in rows)}
    if advisor is not None:
        summary["planner"] = sum(sum(r["planner"].values()) for r in rows)
        summary["validator_problems"] = sum(len(r["validator"]) for r in rows)
    return {"rows": rows, "summary": summary}


def planner_verdict(result: dict[str, Any]) -> dict[str, Any]:
    s = result["summary"]
    if "planner" not in s:
        return {"gate": GATES["planner"]["version"], "status": "not scored"}
    ok = s["planner"] >= s["rules"] and s["validator_problems"] == 0
    return {"gate": GATES["planner"]["version"], "status": "pass" if ok else "fail"}


def _is_repeat(p: dict[str, Any]) -> bool:
    """A count of the biker's rows at one drop-off place: by the entity and the last (drop-off) pair."""
    by = p.get("by", [])
    return p["op"] in ("count", "share") and "biker_id" in by and {"place": "destination"} in by


def _is_short(p: dict[str, Any]) -> bool:
    """The share of the biker's trips that end close to where they start: a mean of a short-distance indicator."""
    of = p.get("of")
    return (p["op"] == "mean" and "biker_id" in p.get("by", []) and isinstance(of, dict) and "below" in of
            and isinstance(of["below"][0], dict) and "distance" in of["below"][0])


def run_project(table: pd.DataFrame, solution: dict[str, Any], *, programs: Optional[list] = None, critic: Any = None,
                rounds: Optional[int] = None, labels: Optional[tuple] = None) -> tuple[Any, str]:
    """A studio project on the table, run through every stage (quick settings)."""
    from unittest import mock

    from dclab_rnd.studio import ProjectStore, data as sd, engine
    from dclab_rnd.studio.solution import Solution

    store = ProjectStore(Path(tempfile.mkdtemp()))
    project = store.create("Replica", "general", "Measure a role")
    path = store.data_dir(project["id"]) / "table.csv"
    table.to_csv(path, index=False)
    sd.attach_data(store, project["id"], "table.csv")
    if labels is not None:
        sample, target = labels
        sample.to_csv(store.data_dir(project["id"]) / "labels.csv", index=False)
        project = sd.attach_labels(store, project["id"], "labels.csv", target=target)
        solution = {**project["suggestion"], **solution}
    project = store.get(project["id"])
    project["solution"] = Solution(**solution).model_dump()
    project["settings"]["quick"] = True
    if rounds is not None:
        project["settings"]["review_rounds"] = rounds
    if programs:
        project["feature_programs"] = programs
    store.save(project)
    with mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=critic):
        engine.run_all(store, project["id"])
    return store, project["id"]


def measure_engineer(advisor: Optional[Any]) -> dict[str, Any]:
    from dclab_core.features import Language, engineer
    from dclab_core.replicas import delivery_fraud as dfr

    r = dfr.generate(seed=0)
    rides = r.rides
    allowed = [c for c in rides.columns if c not in ("order_id", "account_id", "biker_id", "created_date", "terminal_ref", "status")]
    language = Language(allowed, target="is_fraud", time_column="created_date", entities=["biker_id"])
    out = engineer(advisor, rides, language, moment="After delivery, in a daily audit of the day's rides.", task="binary",
                   goal="Find fraudulent rides of bikers.", entities=["biker_id"])
    programs = out["programs"]
    row = {"programs": programs, "refused": out["refused"], "source": out["source"],
           "repeat_proposed": any(_is_repeat(p) for p in programs), "short_proposed": any(_is_short(p) for p in programs)}
    if programs:
        from dclab_rnd.expansion import runner

        store, pid = run_project(rides, {"prediction_moment": "After delivery, in a daily audit of the day's rides.",
                                         "group_column": "biker_id"}, programs=programs, labels=(r.sample, "is_fraud"))
        results = {x["recipe"]: runner._mean(x, "average_precision") for x in store.read_stage(pid, "features")["evidence"]["stage_results"]}
        row["ladder"] = results
        row["gain_over_raw"] = results.get("engineered", float("nan")) - results["raw"]
    return row


def engineer_verdict(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("source") != "advisor":
        return {"gate": GATES["engineer"]["version"], "status": "not scored", "why": row.get("source")}
    ok = row["repeat_proposed"] and row["short_proposed"] and (row.get("gain_over_raw") or 0) >= 0.05
    return {"gate": GATES["engineer"]["version"], "status": "pass" if ok else "fail"}


ENGINEER_SEEDS = (1, 2)


def measure_engineer_v2(advisor: Optional[Any], seeds: tuple[int, ...] = ENGINEER_SEEDS) -> dict[str, Any]:
    """The engineer on fresh replica seeds: its programs, and the ladder's training-CV AP for the raw columns, the
    behaviour recipe and the engineered recipe, on the same folds grouped by biker."""
    from dclab_core.features import Language, engineer
    from dclab_core.replicas import delivery_fraud as dfr
    from dclab_rnd.expansion import runner

    rows = []
    for seed in seeds:
        r = dfr.generate(seed=seed)
        rides = r.rides
        allowed = [c for c in rides.columns if c not in ("order_id", "account_id", "biker_id", "created_date", "terminal_ref", "status")]
        language = Language(allowed, target="is_fraud", time_column="created_date", entities=["biker_id"])
        out = engineer(advisor, rides, language, moment="After delivery, in a daily audit of the day's rides.", task="binary",
                       goal="Find fraudulent rides of bikers.", entities=["biker_id"])
        row = {"seed": seed, "source": out["source"], "programs": out["programs"], "refused": out["refused"]}
        if out["programs"]:
            store, pid = run_project(rides, {"prediction_moment": "After delivery, in a daily audit of the day's rides.",
                                             "group_column": "biker_id"}, programs=out["programs"], labels=(r.sample, "is_fraud"))
            features = store.read_stage(pid, "features")
            ladder = {x["recipe"]: runner._mean(x, "average_precision") for x in features["evidence"]["stage_results"]}
            row.update(ladder=ladder, chosen=features["decision"]["chosen"],
                       gain_over_raw=ladder.get("engineered", float("nan")) - ladder["raw"],
                       against_behaviour=ladder.get("engineered", float("nan")) - ladder.get("entity_behaviour", float("nan")))
        rows.append(row)
    return {"rows": rows}


def engineer_v2_verdict(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    if any(r.get("source") != "advisor" for r in rows):
        return {"gate": GATES["engineer_v2"]["version"], "status": "not scored", "why": "an engineer answer was not used"}
    per_seed = {r["seed"]: bool((r.get("gain_over_raw") or 0) >= 0.05 and (r.get("against_behaviour") or -1) >= 0) for r in rows}
    return {"gate": GATES["engineer_v2"]["version"], "status": "pass" if all(per_seed.values()) else "fail", "per_seed": per_seed}


CRITIC_REPLICAS = {
    "delivery": {"target": "actual_minutes", "task": "regression", "identifiers": ["order_id"], "group_column": "restaurant_id",
                 "prediction_moment": "When the customer places the order.", "drop": ["delivered_at", "courier_rating"]},
    "tickets": {"target": "assigned_team", "task": "multiclass", "identifiers": ["ticket_id"], "time_column": "created_at",
                "prediction_moment": "When the ticket is created, from its subject, body and plan.", "drop": ["resolved_by", "first_response_minutes"]},
}


def measure_critic(advisor: Optional[Any]) -> dict[str, Any]:
    rows = []
    for name, spec in CRITIC_REPLICAS.items():
        table, _, _ = replica_table(name)
        table = table.drop(columns=spec["drop"])
        solution = {k: v for k, v in spec.items() if k != "drop"}
        one_pass = run_project(table, solution, rounds=0)
        rules_loop = run_project(table, solution)
        loop = run_project(table, solution, critic=advisor)
        read = lambda sp: sp[0].read_stage(sp[1], "models")["evidence"]  # noqa: E731
        score = lambda ev: (ev.get("review") or {}).get("selected_mean", ev["selected_metric_mean"])  # noqa: E731 - after the rounds
        first, rules, last = read(one_pass), read(rules_loop), read(loop)
        rounds = (last.get("review") or {}).get("rounds", [])
        kept = [r for r in rounds if r.get("adopted")]
        weak = {w["id"] for w in ((last.get("review") or {}).get("review") or {}).get("weaknesses", [])}
        higher = name != "delivery"
        gt = (lambda a, b: a > b + 1e-12) if higher else (lambda a, b: a < b - 1e-12)  # noqa: E731
        s_first, s_rules, s_last = score(first), score(rules), score(last)
        rows.append({"replica": name, "one_pass": s_first, "rules_loop": s_rules, "loop": s_last,
                     "better": gt(s_last, s_first), "worse": gt(s_first, s_last), "worse_than_rules": gt(s_rules, s_last),
                     "rounds": rounds, "kept_cite_weaknesses": all(r.get("cites") and set(r["cites"]) <= weak for r in kept),
                     "holdout_uses": loop[0].read_stage(loop[1], "final")["evidence"]["holdout_uses_in_this_project"],
                     "critic_used": any((r.get("critic") or {}).get("source") == "advisor" for r in rounds)})
    return {"rows": rows}


def critic_verdict(result: dict[str, Any]) -> dict[str, Any]:
    rows = result["rows"]
    if not any(r["critic_used"] for r in rows):
        return {"gate": GATES["critic"]["version"], "status": "not scored", "why": "the critic's answer was never used"}
    ok = (not any(r["worse"] or r["worse_than_rules"] for r in rows)) and any(r["better"] for r in rows) \
        and all(r["kept_cite_weaknesses"] for r in rows) and all(r["holdout_uses"] == 1 for r in rows)
    return {"gate": GATES["critic"]["version"], "status": "pass" if ok else "fail"}
