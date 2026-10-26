"""The decision's holdout interval on a time split (campaign ``time_decision_v4``; 34.1, plan item 6's open end).

``time_decision_v3`` (seeds 160 to 189, the replica's drift off): the share capacity passed every gate; one call a day
kept its centre (mean z -0.07) and width (RMS z 1.19) but 23 of 30 landed inside, one short of 80%, with no pattern in
the misses (not established: 7 in 100 under a right interval; the daily RMS z was above 1 in v2 and v3). v4 runs the
daily arm alone on 60 fresh seeds (190 to 249), the drift off, the gate unchanged: is the day by day interval right, or
slightly narrow?

``time_decision_v2`` (seeds 130 to 159): with the interval drawn from the latest fold (0.118.0), a share capacity passed
every gate (mean z 0.12) and one call a day kept 26 of 30 inside with its centre above (mean z 0.40); the replica kept
drifting past the training period. v3 runs the same replica with its drift off (``drift=0``: the cancel rate the same
over the two years) on fresh seeds (160 to 189), the gate unchanged: does the latest fold's interval keep its word on a
time split when the world stays the same, the daily arm included?

``time_decision_v1`` (seeds 100 to 129, 2026-10-26) failed on both arms, every miss above the interval: the replica
drifts (positive rate 0.217 out of fold, 0.236 on the latest fold, 0.247 on the holdout) and each fold's value rose
with it, so an interval from every fold sat low. Its next step, fixed with its gate: draw the holdout-sized samples from
the latest fold's rows only (dclab-core 0.118.0, ``sample_rows``). v2 measures that on fresh seeds (130 to 159), the
gate unchanged.

``fraud_decision_v3`` passed on an entity split, where each holdout-sized sample comes from one fold's rows. A time split
passes no folds: its out-of-fold rows were scored by expanding-window models, the first trained on the fewest rows, while
the holdout is scored by a model fitted on every training row. The interval is then the row-sized samples' alone, never
measured. The churn replica's first day-by-day reading (2026-10-25, 0.115.0) realised 10.6 a customer against an interval
of 4.8 to 8.6: one reading, pointing the way a weaker set of fold models would.

Each seed of the churn replica (subscribers dated by sign-up, split by time: the holdout is the latest fifth) runs the
five stages once per arm measured (``RUN_ARMS``; v1 to v3 ran both, v4 the daily arm alone), one decision each, stated as
a person would: call a customer likely to leave, a customer kept
is worth 120, a call costs 10.

* ``pooled``: at most 10% of the customers called (a share: the holdout read as one pool).
* ``daily``: at most one call a day (0.115.0: a time split's holdout read day by day, each day's best customer).

The gate, fixed in ``manifest.json`` before the first run, is ``fraud_decision_v3``'s, judged for each arm alone:

* the decision is chosen on every seed (an operating point; no refusal);
* on every seed that acted, never more of the holdout than its capacity allows (a share's rounding, or a day's slot);
* among the seeds that chose to act (at least 20), the realised value per customer lies inside the 90% prediction interval
  on at least 80% of them;
* z = (realised - centre) / (half-width / 1.645): |mean z| within 1.645/sqrt(N), and the 90% chi-squared interval of the
  root mean square of z contains 1. Fewer than 20 seeds acting is inconclusive, not a failure.

v1's next step, fixed with v1's gate (v2's own steps are in ``GATE``): a pass kept the interval on time splits; a centre
failure with mean z above 0 reads the interval from the latest fold's rows only (two causes are possible and both are helped: the fold models are
weaker than the final one, and the replica drifts, later customers leaving more often; each record keeps the positive
rate of the out-of-fold rows, the latest fold and the holdout, and each fold's value, so the report can say which);
mean z below 0 corrects the optimism of choosing the model; too narrow with misses on both sides adds blocks of days to
the samples; too wide drops a part of the samples' noise. A run that fails is recorded as failed (the "chosen" gate
counts it), so the campaign always reaches a verdict.

    python -m dclab_rnd.time_decision_measure plan
    python -m dclab_rnd.time_decision_measure run --yes     # two engine runs a seed; resumable on the same code
    python -m dclab_rnd.time_decision_measure report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_ID = "time_decision_v4"
CAMPAIGN = ROOT / "evidence" / "campaigns" / CAMPAIGN_ID
WORK = ROOT / "agent_runs" / CAMPAIGN_ID  # each run's workspace kept whole: the store reads its sessions, ledger and runs
SEEDS = tuple(range(190, 250))  # fresh: v1's to v3's seeds (100 to 189) and seed 0 never judge it; 60 for the daily arm's width
DRIFT = 0.0  # the replica's drift switched off, as in v3
RUN_ARMS = ("daily",)  # v4 measures the daily arm alone (the share arm passed with and without drift in v2 and v3)
BASE = {"action": "act", "what": "call the customer", "value_per_case": 120.0, "cost_per_action": 10.0}
ARMS = {
    "pooled": {**BASE, "capacity_share": 0.1, "notes": "a customer kept is worth 120, a call costs 10; at most 10% of the customers called"},
    "daily": {**BASE, "capacity_per_day": 1.0, "notes": "a customer kept is worth 120, a call costs 10; one call a day"},
}
GATE = {
    "id": "time_decision_gate_v4 (v1's, unchanged; the daily arm alone)",
    "chosen": "the decision is chosen on every seed (an operating point; no refusal)",
    "time_split": "every run's data stage split by time (the strategy recorded per run)",
    "capacity": "on every seed that acted, the holdout's actions never exceed its capacity (a share of its rows with one row's "
                "rounding; or the day by day slots the record states), and the holdout was read once",
    "coverage": "among the seeds that chose to act (at least 20), the realised value per customer lies inside the choice's 90% "
                "prediction interval on at least 80% of them",
    "centre": "each seed's z = (realised - the interval's centre) / (its half-width / 1.645); |mean z| <= 1.645 / sqrt(N)",
    "width": "the 90% chi-squared interval of the root mean square of z contains 1 (neither too narrow nor too wide)",
    "inconclusive": "fewer than 20 seeds acting: no verdict on the interval",
    "next": "a pass: keep the day by day interval, v3's coverage miss read as chance; too narrow (the width's interval above 1) "
            "with misses on both sides: add the model's fit-to-fit spread to the samples (the holdout is scored by the final model, "
            "the samples by one fold model never refitted; one call a day hangs on its fine ranking), and if the paired every-fold "
            "RMS z is above 1 too, the latest fold's smaller pool is not the cause; coverage below 80% with the width passing: pool "
            "v3's and v4's daily seeds (90) and read the pooled coverage and RMS z beside the gate; a centre off at the spread "
            "measured (|mean z| above 1.645 RMS z / sqrt(N); for routing only, the gate unchanged): above 0, if the folds' values "
            "rise from the first fold to the last, score the out-of-fold rows with models of the final size, otherwise read the "
            "seeds; below 0, correct the optimism of choosing the model; too wide: sample the latest two folds",
    "paired": "each run also keeps the interval drawn from every fold (0.115.0's) on the same rows and seed, with its z: reported "
              "beside the gate, never gated",
}
MIN_ACTED = 20
COVERAGE = 0.8
CODE_FILES = ("dclab_rnd/time_decision_measure.py", "dclab_rnd/campaign_capture.py", "dclab_rnd/ledger", "dclab_rnd/experience")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def code_sha256() -> str:
    from .campaign_capture import ENGINE_CODE, code_sha256 as whole

    return whole((*ENGINE_CODE, *CODE_FILES))


CODE_AT_START = code_sha256()


def manifest() -> dict[str, Any]:
    body = {"campaign": CAMPAIGN_ID, "package": "34.1", "seeds": list(SEEDS), "arms": {a: ARMS[a] for a in RUN_ARMS}, "gate": GATE, "drift": DRIFT,
            "dataset": "subscription churn (kinds binary case's replica, drift off)",
            "objective": ("Is the day by day interval on a time split right or slightly narrow? Sixty fresh seeds of the churn replica "
                          "with its drift off, one call a day read day by day against its 90% interval drawn from the latest fold (v3: "
                          "23 of 30 inside, centre and width passing).")}
    body["sha256"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    return body


def freeze() -> dict[str, Any]:
    CAMPAIGN.mkdir(parents=True, exist_ok=True)
    path = CAMPAIGN / "manifest.json"
    want = manifest()
    if path.exists():
        held = json.loads(path.read_text())
        if held.get("sha256") != want["sha256"]:
            raise SystemExit("The gate, the seeds or the arms changed since the campaign was frozen: start a new campaign version")
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


def _number(seed: int, arm: str) -> str:
    return f"TDC-{(list(ARMS).index(arm)) * 100 + SEEDS.index(seed) + 1:03d}" if seed in SEEDS else f"TDC-{arm}-{seed}"


def run_one(seed: int, arm: str) -> dict[str, Any]:
    from unittest import mock

    from dclab_rnd import kinds
    from dclab_rnd.kinds import acceptance
    from dclab_rnd.models.gateway import private_scope
    from dclab_rnd.studio import engine

    from .campaign_capture import recorded

    decision = ARMS[arm]
    from dclab_core.replicas import subscription_churn

    template = kinds.get("binary").case(seed)  # the binary kind's churn case, rebuilt with the replica's drift set by DRIFT
    replica = subscription_churn.generate(seed, drift=DRIFT)
    case = acceptance.Case(replica.subscribers, template.target, template.task, template.moment, {**template.solution, "decision": decision},
                           facts=replica.facts)
    started = time.perf_counter()
    work = WORK / f"seed{seed}_{arm}"
    if work.exists():
        shutil.rmtree(work)  # this campaign's own workspace for the run, made again
    work.mkdir(parents=True)
    from .ledger import labels

    import numpy as np

    from dclab_core import decide

    original, kept = decide.choose_operating_point, {}

    def paired(*args, **kwargs):  # the same call with every fold's rows (0.115.0's interval), kept beside the measured one
        out = original(*args, **kwargs)
        if kwargs.get("sample_rows") is not None:
            kept["sample_rows"] = kwargs["sample_rows"]
            kept["all_folds"] = (original(*args, **{**kwargs, "sample_rows": None}).get("holdout") or {}).get("prediction_interval")
        return out

    with recorded(work), private_scope(True), mock.patch("dclab_rnd.studio.engine._critic_advisor", return_value=None), \
            mock.patch.object(decide, "choose_operating_point", paired):
        store, pid = acceptance.build(case, work)
        engine.run_all(store, pid)
        labels.from_replica(pid, case.facts, "subscription_churn")  # the planted traps label the decisions (the review)
    final = store.read_stage(pid, "final") or {}
    split = ((store.read_stage(pid, "data") or {}).get("evidence") or {}).get("split_strategy")
    evidence = final.get("evidence") or {}
    d = evidence.get("decision") or {}
    h = d.get("holdout") or {}
    interval = h.get("prediction_interval") or [None, None]
    realised = h.get("expected_value_per_row")
    rows = evidence.get("holdout_rows")
    acted = h.get("actions")
    days = h.get("days")
    inside = None if realised is None or None in interval else bool(interval[0] <= realised <= interval[1])
    z = side = None
    if inside is not None and interval[1] > interval[0]:
        z = round((realised - (interval[0] + interval[1]) / 2) / ((interval[1] - interval[0]) / 2 / 1.645), 4)
        side = "inside" if inside else "below" if realised < interval[0] else "above"
    all_folds = kept.get("all_folds") or [None, None]
    inside_all = z_all = None
    if realised is not None and None not in all_folds and all_folds[1] > all_folds[0]:
        inside_all = bool(all_folds[0] <= realised <= all_folds[1])
        z_all = round((realised - (all_folds[0] + all_folds[1]) / 2) / ((all_folds[1] - all_folds[0]) / 2 / 1.645), 4)
    if acted is None:
        within = None
    elif arm == "daily":
        within = bool(days) and acted <= days["slots"] and days["most_in_a_day"] <= decision["capacity_per_day"]
    else:
        within = bool(rows) and acted <= decision["capacity_share"] * rows + 1
    number = _number(seed, arm)
    result = {"id": number, "experiment_id": number, "status": "completed", "completed_at": _now(), "campaign_id": CAMPAIGN_ID,
              "dataset": "subscription_churn", "seed": seed, "arm": arm, "split": split, "rates": _rates(store, pid, d.get("share")),
              "decision_status": d.get("decision"), "warning": d.get("warning"), "chosen_share": d.get("share"),
              "capacity_share": d.get("capacity_share"), "expected_value_per_row": d.get("expected_value_per_row"),
              "expected_interval": d.get("expected_value_per_row_interval"), "holdout_rows": rows, "holdout_actions": acted,
              "holdout_cases_found": h.get("cases_found"), "holdout_value_per_row": realised, "prediction_interval": interval,
              "inside_prediction_interval": inside, "z": z, "miss_side": side, "within_capacity": within, "days": days,
              "days_note": d.get("days_note"), "holdout_uses": int(store.get(pid).get("holdout_uses") or 0),
              "prediction_interval_method": h.get("prediction_interval_method"), "applied_as": d.get("applied_as"),
              "all_folds_interval": all_folds, "inside_all_folds": inside_all, "z_all_folds": z_all,
              "latest_fold_rows": int(np.asarray(kept["sample_rows"]).sum()) if kept.get("sample_rows") is not None else None,
              "latest_fold_is_the_last_block": (bool(np.asarray(kept["sample_rows"])[-int(np.asarray(kept["sample_rows"]).sum()):].all())
                                                if kept.get("sample_rows") is not None else None),
              "model": evidence.get("model"), "seconds": round(time.perf_counter() - started, 1), "code_sha256": CODE_AT_START}
    out = CAMPAIGN / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{number}_seed{seed}_{arm}.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result


def _rates(store: Any, pid: str, share: Any) -> dict[str, Any] | None:
    """Counts behind a centre that misses (the review): the positive rate of the out-of-fold rows, of the latest fold and
    of the holdout, and each fold's value per row when its top ``share`` by score is called. Rates and values only."""
    import numpy as np

    from dclab_rnd.studio import engine

    try:
        final = store.read_stage(pid, "final") or {}
        rows = engine.load_decision_rows(store, pid, (final.get("run") or {}).get("run"))
        b = engine.prepare(store, store.get(pid)).bundle
        y_train, y_test = b.y_train.to_numpy().astype(int), b.y_test.to_numpy().astype(int)
        folds = [np.asarray(v) for _, v in b.cv_splits]
        out = {"oof_positive_rate": round(float(y_train[np.concatenate(folds)].mean()), 4),
               "latest_fold_positive_rate": round(float(y_train[folds[-1]].mean()), 4),
               "holdout_positive_rate": round(float(y_test.mean()), 4), "fold_values": None}
        oof = np.concatenate(folds)
        if rows is not None and share and len(oof) == len(rows["y"]) and np.array_equal(y_train[oof], np.asarray(rows["y"]).astype(int)):
            score, gain = np.asarray(rows["score"], dtype=float), 120.0 * np.asarray(rows["y"]) - 10.0
            values, start = [], 0
            for f in folds:  # the saved rows follow the folds' order
                s, g = score[start:start + len(f)], gain[start:start + len(f)]
                top = np.argsort(-s, kind="mergesort")[: int(np.floor(float(share) * len(f)))]
                values.append(round(float(g[top].sum()) / len(f), 4))
                start += len(f)
            out["fold_values"] = values
        return out
    except Exception:  # noqa: BLE001 - counts beside the measured run, which stands without them
        return None


def failed(seed: int, arm: str, error: BaseException) -> dict[str, Any]:
    """A run that raised, written as a failed result (the review: else the campaign never reaches a verdict)."""
    number = _number(seed, arm)
    result = {"id": number, "experiment_id": number, "status": "failed", "completed_at": _now(), "campaign_id": CAMPAIGN_ID,
              "dataset": "subscription_churn", "seed": seed, "arm": arm, "decision_status": None, "chosen_share": None,
              "error": f"{type(error).__name__}: {str(error)[:300]}", "code_sha256": CODE_AT_START}
    out = CAMPAIGN / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{number}_seed{seed}_{arm}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def judge(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One arm's verdict, as fraud_decision_v3's gate reads it."""
    chosen = all(r.get("decision_status") == "operating_point" and r.get("chosen_share") is not None and not r.get("warning") for r in rows)
    acted = [r for r in rows if r.get("chosen_share")]
    capacity = all(r.get("within_capacity") is True and r.get("holdout_uses") == 1 for r in acted)
    timed = bool(rows) and all(r.get("split") == "time" for r in rows if r.get("status") != "failed")
    read = [r for r in acted if r.get("inside_prediction_interval") is not None]
    inside = sum(1 for r in read if r["inside_prediction_interval"])
    zs = [float(r["z"]) for r in read if r.get("z") is not None]
    centre = width = None
    if len(zs) >= 2:
        from scipy.stats import chi2

        n = len(zs)
        mean, rms = sum(zs) / n, (sum(z * z for z in zs) / n) ** 0.5
        centre = {"mean_z": round(mean, 4), "bound": round(1.645 / n ** 0.5, 4)}
        width = {"rms_z": round(rms, 4), "interval": [round(float((n * rms * rms / chi2.ppf(0.95, n)) ** 0.5), 4),
                                                      round(float((n * rms * rms / chi2.ppf(0.05, n)) ** 0.5), 4)]}
    enough = len(read) >= MIN_ACTED
    gates = {"chosen": bool(chosen), "time_split": timed, "capacity": bool(capacity), "coverage": bool(enough and inside >= COVERAGE * len(read)),
             "centre": bool(enough and centre is not None and abs(centre["mean_z"]) <= centre["bound"]),
             "width": bool(enough and width is not None and width["interval"][0] <= 1 <= width["interval"][1])}
    complete = len(rows) == len(SEEDS)
    sides = {s: sum(1 for r in read if r.get("miss_side") == s) for s in ("below", "above")}
    narrow = width is not None and width["interval"][0] > 1
    wide = width is not None and width["interval"][1] < 1
    rises = [r["rates"]["fold_values"][-1] - r["rates"]["fold_values"][0] for r in rows
             if (r.get("rates") or {}).get("fold_values")]
    rising = bool(rises) and sum(rises) / len(rises) > 0
    # routing only (the gate is unchanged): a centre counts as off when it is off at the spread measured, so a narrow
    # interval's wider scatter of mean z is not read as a shifted centre (the review of v4)
    off = centre is not None and width is not None and abs(centre["mean_z"]) > 1.645 * width["rms_z"] / len(zs) ** 0.5
    step = ("keep the day by day interval" if all(gates.values()) else
            ("score the out-of-fold rows with models of the final size" if rising else "read the seeds")
            if not gates["centre"] and off and centre["mean_z"] > 0 else
            "correct the optimism of choosing the model" if not gates["centre"] and off and centre["mean_z"] < 0 else
            "add the model's fit-to-fit spread to the samples" if not gates["width"] and narrow and sides["below"] and sides["above"] else
            "pool v3's and v4's daily seeds" if not gates["coverage"] and gates["width"] else
            "sample the latest two folds" if not gates["width"] and wide else "read the seeds")
    paired = [float(r["z_all_folds"]) for r in read if r.get("z_all_folds") is not None]
    verdict = ((("pass" if all(gates.values()) else "fail") if enough else "inconclusive: fewer than 20 seeds acted")
               if complete else f"not yet: {len(rows)} of {len(SEEDS)} seeds")
    return {"seeds": len(rows), "of": len(SEEDS), "gates": gates, "acted": len(acted), "inside_prediction_interval": inside,
            "coverage": round(inside / len(read), 4) if read else None, "misses": sides, "centre": centre, "width": width,
            "next": step if complete and enough else None, "verdict": verdict,
            "paired_all_folds": {"mean_z": round(sum(paired) / len(paired), 4) if paired else None,
                                 "rms_z": round((sum(z * z for z in paired) / len(paired)) ** 0.5, 4) if paired else None,
                                 "inside": sum(1 for r in read if r.get("inside_all_folds")), "read": len(paired)}}


def pooled_daily(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """v3's and v4's daily seeds together (90), reported beside the gate, never gated (the review of v4)."""
    earlier = ROOT / "evidence" / "campaigns" / "time_decision_v3" / "results"
    old = [json.loads(p.read_text()) for p in sorted(earlier.glob("TDC-[0-9]*_seed*.json"))] if earlier.exists() else []
    both = [r for r in [*old, *rows] if r.get("arm") == "daily" and r.get("inside_prediction_interval") is not None and r.get("z") is not None]
    if not both:
        return None
    zs = [float(r["z"]) for r in both]
    return {"seeds": len(both), "inside": sum(1 for r in both if r["inside_prediction_interval"]),
            "coverage": round(sum(1 for r in both if r["inside_prediction_interval"]) / len(both), 4),
            "mean_z": round(sum(zs) / len(zs), 4), "rms_z": round((sum(z * z for z in zs) / len(zs)) ** 0.5, 4)}


def report() -> dict[str, Any]:
    folder = CAMPAIGN / "results"
    rows = [json.loads(p.read_text()) for p in sorted(folder.glob("TDC-[0-9]*_seed*.json"))] if folder.exists() else []
    out = {"arms": {arm: judge([r for r in rows if r.get("arm") == arm]) for arm in RUN_ARMS}, "gate": GATE,
           "pooled_with_v3_daily": pooled_daily(rows)}
    out["verdict"] = {arm: v["verdict"] for arm, v in out["arms"].items()}
    complete = all(v["seeds"] == len(SEEDS) for v in out["arms"].values())
    if complete and not (folder / "TDC-900_decision.json").exists():  # written once: a result is never rewritten
        (folder / "TDC-900_decision.json").write_text(json.dumps({**out, "id": "TDC-900", "experiment_id": "TDC-900", "status": "completed",
                                                                  "completed_at": _now(), "campaign_id": CAMPAIGN_ID}, indent=2) + "\n")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="34.1: the decision's holdout interval on a time split (the churn replica)")
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
    folder = CAMPAIGN / "results"
    done = {(int(r["seed"]), r["arm"]) for r in (json.loads(p.read_text()) for p in folder.glob("TDC-[0-9]*_seed*.json"))} if folder.exists() else set()
    for seed in SEEDS:
        for arm in RUN_ARMS:
            if (seed, arm) in done:
                continue
            print(f"[time-decision] seed {seed} {arm} {_now()}", flush=True)
            try:
                r = run_one(seed, arm)
                print(f"[time-decision] seed {seed} {arm}: share {r['chosen_share']} realised {r['holdout_value_per_row']} in "
                      f"{r['prediction_interval']}: {r['inside_prediction_interval']}; capacity kept: {r['within_capacity']}", flush=True)
            except Exception as exc:  # noqa: BLE001 - a run that fails is recorded as failed, the rest still run
                failed(seed, arm, exc)
                print(f"[time-decision] seed {seed} {arm} failed: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
    print(json.dumps(report(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
